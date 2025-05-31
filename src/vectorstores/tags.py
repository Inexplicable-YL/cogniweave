import hashlib
import pickle
import uuid
from collections import defaultdict
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, TypedDict, cast

import anyio
import numpy as np
from anyio import to_thread
from langchain_community.docstore.base import AddableMixin, Docstore
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.faiss import dependable_faiss_import
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from src.utils import sync_func_wrapper

from .base import LazyFAISS


class TagDucumentId(TypedDict):
    file_id: list[str]


class DocumentMeta(TypedDict):
    tag_hashs: list[str]
    metadata: dict[Any, Any] | None


class TagsVector:
    vector: LazyFAISS

    folder_path: str
    index_name: str

    metastore: Docstore

    def __init__(
        self,
        folder_path: str,
        index_name: str,
        embeddings: Embeddings,
        **kwargs: Any,
    ) -> None:
        path = Path(folder_path)
        metastore = faiss_index = faiss_docstore = faiss_index_to_docstore_id = None

        if (path / f"{index_name}.faiss").exists():
            # load index separately since it is not picklable
            faiss = dependable_faiss_import()
            faiss_index = faiss.read_index(str(path / f"{index_name}.faiss"))

        if (path / f"{index_name}.pkl").exists():
            # load docstore and index_to_docstore_id
            with (path / f"{index_name}.pkl").open("rb") as f:
                (
                    metastore,
                    faiss_docstore,
                    faiss_index_to_docstore_id,
                ) = pickle.load(  # ignore[pickle]: explicit-opt-in  # noqa: S301
                    f
                )
        self.vector = LazyFAISS(
            embeddings, faiss_index, faiss_docstore, faiss_index_to_docstore_id, **kwargs
        )
        self.folder_path = folder_path
        self.index_name = index_name
        self.metastore = metastore or InMemoryDocstore()

    @staticmethod
    def _build_document(
        tag_hashs: list[str], content: str, metadata: dict[str, Any] | None = None
    ) -> Document:
        return Document(
            page_content=content, metadata=DocumentMeta(tag_hashs=tag_hashs, metadata=metadata)
        )

    def add_tags(
        self,
        tags: Iterable[str],
        content: str,
        metadata: dict[Any, Any] | None = None,
        id_: str | None = None,
        **kwargs: Any,
    ) -> list[str]:
        if not isinstance(self.metastore, AddableMixin):
            raise TypeError(
                "If trying to add texts, the underlying docstore should support "
                f"adding items, which {self.metastore} does not"
            )

        # register a new document
        hashs = [hashlib.sha256(tag.encode("utf-8")).hexdigest() for tag in tags]
        doc_id = id_ or str(uuid.uuid4())
        self.metastore.add({doc_id: self._build_document(hashs, content, metadata)})

        # add documents id to tags store
        unload_tags = {}
        for _hash, _tag in zip(hashs, tags, strict=True):
            if isinstance(tag_doc := self.vector.docstore.search(_hash), Document):
                cast("TagDucumentId", tag_doc.metadata)["file_id"].append(doc_id)
            else:
                unload_tags[_hash] = _tag

        # add new tag documents
        if unload_tags:
            self.vector.add_texts(
                unload_tags.values(),
                metadatas=[dict(TagDucumentId(file_id=[doc_id])) for _ in unload_tags.values()],
                ids=list(unload_tags.keys()),
                **kwargs,
            )

        return hashs

    async def aadd_tags(
        self,
        tags: Iterable[str],
        content: str,
        metadata: dict[Any, Any] | None = None,
        id_: str | None = None,
        **kwargs: Any,
    ) -> list[str]:
        if not isinstance(self.metastore, AddableMixin):
            raise TypeError(
                "If trying to add texts, the underlying docstore should support "
                f"adding items, which {self.metastore} does not"
            )

        # register a new document
        hashs = [hashlib.sha256(tag.encode("utf-8")).hexdigest() for tag in tags]
        doc_id = id_ or str(uuid.uuid4())
        self.metastore.add({doc_id: self._build_document(hashs, content, metadata)})

        # add documents id to tags store
        unload_tags = {}
        for _hash, _tag in zip(hashs, tags, strict=True):
            if isinstance(tag_doc := self.vector.docstore.search(_hash), Document):
                cast("TagDucumentId", tag_doc.metadata)["file_id"].append(doc_id)
            else:
                unload_tags[_hash] = _tag

        # add new tag documents
        if unload_tags:
            await self.vector.aadd_texts(
                unload_tags.values(),
                metadatas=[dict(TagDucumentId(file_id=[doc_id])) for _ in unload_tags.values()],
                ids=list(unload_tags.keys()),
                **kwargs,
            )

        return hashs

    def similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: Callable | dict[str, Any] | None = None,
        fetch_k: int = 5,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """
        Perform similarity search over tag-based vector index and return top-k documents with scores.

        This method first performs similarity search on the tag vector index, aggregates scores
        by associated file IDs, and then retrieves the actual documents from the metastore.
        An optional metadata filter can be applied to refine the final results.

        Args:
            embedding (list[float]): The embedding vector used as the search query.
            k (int, optional): Number of top documents to return. Defaults to 4.
            filter (Optional[Union[Callable, dict[str, Any]]]): Filter function or metadata filter.
                - If a callable, it should take a document's metadata and return a bool.
                - If a dict, it will be converted to a filter function.
                Defaults to None.
            fetch_k (int, optional): Number of candidate tag documents to retrieve before filtering.
                Only applies when filter is provided. Defaults to 20.
            **kwargs: kwargs to be passed to similarity search. Can include:
                score_threshold: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs

        Returns:
            list[tuple[Document, float]]: A list of tuples containing:
                - Document: the retrieved document
                - float: similarity score (lower means more similar, usually L2 distance)
        """
        tag_docs_and_scores = self.vector.similarity_search_with_score_by_vector(
            embedding, k=fetch_k if filter is None else fetch_k * 2, **kwargs
        )
        relevance_score_fn = self.vector._select_relevance_score_fn()
        if relevance_score_fn is None:
            raise ValueError(
                "relevance_score_fn must be provided to FAISS constructor to normalize scores"
            )
        tag_docs = [(doc, relevance_score_fn(score)) for doc, score in tag_docs_and_scores]

        file_map = defaultdict(float)
        for _d, _s in tag_docs:
            metadata = cast("TagDucumentId", _d.metadata)
            for file_id in metadata["file_id"]:
                file_map[file_id] += _s

        doc_map: list[tuple[str, float]] = sorted(
            file_map.items(), key=lambda item: item[1], reverse=True
        )

        docs = []

        if filter is not None:
            filter_func = self.vector._create_filter_func(filter)

        for doc_id, doc_score in doc_map:
            _doc = self.metastore.search(doc_id)
            if not isinstance(_doc, Document):
                raise TypeError(f"Could not find document for id {doc_id}, got {_doc}")
            doc = Document(
                page_content=_doc.page_content,
                metadata=cast("DocumentMeta", _doc.metadata)["metadata"],
            )
            if filter is not None:
                if filter_func(doc.metadata):  # type: ignore
                    docs.append((doc, doc_score))
            else:
                docs.append((doc, doc_score))

        if kwargs.get("extract_high_score", False):
            return self.extract_high_score(docs, fallback_k=k, **kwargs)
        return docs[:k]

    async def asimilarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: Callable | dict[str, Any] | None = None,
        fetch_k: int = 5,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """
        Perform similarity search over tag-based vector index and return top-k documents with scores.

        This method first performs similarity search on the tag vector index, aggregates scores
        by associated file IDs, and then retrieves the actual documents from the metastore.
        An optional metadata filter can be applied to refine the final results.

        Args:
            embedding (list[float]): The embedding vector used as the search query.
            k (int, optional): Number of top documents to return. Defaults to 4.
            filter (Optional[Union[Callable, dict[str, Any]]]): Filter function or metadata filter.
                - If a callable, it should take a document's metadata and return a bool.
                - If a dict, it will be converted to a filter function.
                Defaults to None.
            fetch_k (int, optional): Number of candidate tag documents to retrieve before filtering.
                Only applies when filter is provided. Defaults to 20.
            **kwargs: kwargs to be passed to similarity search. Can include:
                score_threshold: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs

        Returns:
            list[tuple[Document, float]]: A list of tuples containing:
                - Document: the retrieved document
                - float: similarity score (lower means more similar, usually L2 distance)
        """
        async_func = sync_func_wrapper(self.similarity_search_with_score_by_vector, to_thread=True)
        return await async_func(
            embedding,
            k=k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs,
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Callable | dict[str, Any] | None = None,
        fetch_k: int = 5,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata.
                Defaults to None. If a callable, it must take as input the
                metadata dict of Document and return a bool.

            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.

        Returns:
            List of documents most similar to the query text with
            L2 distance in float. Lower score represents more similarity.
        """
        embedding = self.vector._embed_query(query)
        return self.similarity_search_with_score_by_vector(
            embedding,
            k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs,
        )

    async def asimilarity_search_with_score(
        self,
        query: str,
        k: int = 4,
        filter: Callable | dict[str, Any] | None = None,
        fetch_k: int = 5,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs most similar to query asynchronously.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata.
                Defaults to None. If a callable, it must take as input the
                metadata dict of Document and return a bool.

            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.

        Returns:
            List of documents most similar to the query text with
            L2 distance in float. Lower score represents more similarity.
        """
        embedding = await self.vector._aembed_query(query)
        return await self.asimilarity_search_with_score_by_vector(
            embedding,
            k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs,
        )

    def similarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: dict[str, Any] | None = None,
        fetch_k: int = 5,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to embedding vector.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata.
                Defaults to None. If a callable, it must take as input the
                metadata dict of Document and return a bool.

            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.

        Returns:
            List of Documents most similar to the embedding.
        """
        docs_and_scores = self.similarity_search_with_score_by_vector(
            embedding,
            k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs,
        )
        return [doc for doc, _ in docs_and_scores]

    async def asimilarity_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: Callable | dict[str, Any] | None = None,
        fetch_k: int = 5,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to embedding vector asynchronously.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Dict[str, str]]): Filter by metadata.
                Defaults to None. If a callable, it must take as input the
                metadata dict of Document and return a bool.

            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.

        Returns:
            List of Documents most similar to the embedding.
        """
        docs_and_scores = await self.asimilarity_search_with_score_by_vector(
            embedding,
            k,
            filter=filter,
            fetch_k=fetch_k,
            **kwargs,
        )
        return [doc for doc, _ in docs_and_scores]

    def similarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Callable | dict[str, Any] | None = None,
        fetch_k: int = 5,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to query.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.

        Returns:
            List of Documents most similar to the query.
        """
        docs_and_scores = self.similarity_search_with_score(
            query, k, filter=filter, fetch_k=fetch_k, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    async def asimilarity_search(
        self,
        query: str,
        k: int = 4,
        filter: Callable | dict[str, Any] | None = None,
        fetch_k: int = 5,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs most similar to query asynchronously.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: (Optional[Dict[str, str]]): Filter by metadata. Defaults to None.
            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.

        Returns:
            List of Documents most similar to the query.
        """
        docs_and_scores = await self.asimilarity_search_with_score(
            query, k, filter=filter, fetch_k=fetch_k, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: list[float],
        *,
        k: int = 4,
        fetch_k: int = 5,
        lambda_mult: float = 0.5,
        filter: Callable | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs and their similarity scores selected using the maximal marginal
            relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch before filtering to
                     pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents and similarity scores selected by maximal marginal
                relevance and score for each.
        """
        tag_docs_and_scores = self.vector.max_marginal_relevance_search_with_score_by_vector(
            embedding,
            k=fetch_k if filter is None else fetch_k * 2,
            fetch_k=fetch_k * 2 if filter is None else fetch_k * 4,
            lambda_mult=lambda_mult,
        )
        relevance_score_fn = self.vector._select_relevance_score_fn()
        if relevance_score_fn is None:
            raise ValueError(
                "relevance_score_fn must be provided to FAISS constructor to normalize scores"
            )
        tag_docs = [(doc, relevance_score_fn(score)) for doc, score in tag_docs_and_scores]

        file_map = defaultdict(float)
        for _d, _s in tag_docs:
            metadata = cast("TagDucumentId", _d.metadata)
            for file_id in metadata["file_id"]:
                file_map[file_id] += _s

        doc_map: list[tuple[str, float]] = sorted(
            file_map.items(), key=lambda item: item[1], reverse=True
        )

        docs = []

        if filter is not None:
            filter_func = self.vector._create_filter_func(filter)

        for doc_id, doc_score in doc_map:
            _doc = self.metastore.search(doc_id)
            if not isinstance(_doc, Document):
                raise TypeError(f"Could not find document for id {doc_id}, got {_doc}")
            doc = Document(
                page_content=_doc.page_content,
                metadata=cast("DocumentMeta", _doc.metadata)["metadata"],
            )
            if filter is not None:
                if filter_func(doc.metadata):  # type: ignore
                    docs.append((doc, doc_score))
            else:
                docs.append((doc, doc_score))

        if kwargs.get("extract_high_score", False):
            return self.extract_high_score(docs, fallback_k=k, **kwargs)
        return docs[:k]

    async def amax_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: list[float],
        *,
        k: int = 4,
        fetch_k: int = 5,
        lambda_mult: float = 0.5,
        filter: Callable | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs and their similarity scores selected using the maximal marginal
            relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch before filtering to
                     pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents and similarity scores selected by maximal marginal
                relevance and score for each.
        """
        async_func = sync_func_wrapper(
            self.max_marginal_relevance_search_with_score_by_vector, to_thread=True
        )
        return await async_func(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )

    def max_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 5,
        lambda_mult: float = 0.5,
        filter: Callable | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch before filtering to
                     pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        docs_and_scores = self.max_marginal_relevance_search_with_score_by_vector(
            embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    async def amax_marginal_relevance_search_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        fetch_k: int = 5,
        lambda_mult: float = 0.5,
        filter: Callable | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance asynchronously.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            embedding: Embedding to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch before filtering to
                     pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        docs_and_scores = await self.amax_marginal_relevance_search_with_score_by_vector(
            embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter, **kwargs
        )
        return [doc for doc, _ in docs_and_scores]

    def max_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 5,
        lambda_mult: float = 0.5,
        filter: Callable | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch before filtering (if needed) to
                     pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        embedding = self.vector._embed_query(query)
        return self.max_marginal_relevance_search_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )

    async def amax_marginal_relevance_search(
        self,
        query: str,
        k: int = 4,
        fetch_k: int = 5,
        lambda_mult: float = 0.5,
        filter: Callable | dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[Document]:
        """Return docs selected using the maximal marginal relevance asynchronously.

        Maximal marginal relevance optimizes for similarity to query AND diversity
        among selected documents.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            fetch_k: Number of Documents to fetch before filtering (if needed) to
                     pass to MMR algorithm.
            lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        Returns:
            List of Documents selected by maximal marginal relevance.
        """
        embedding = await self.vector._aembed_query(query)
        return await self.amax_marginal_relevance_search_by_vector(
            embedding,
            k=k,
            fetch_k=fetch_k,
            lambda_mult=lambda_mult,
            filter=filter,
            **kwargs,
        )

    def delete_docs(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:  # noqa: ARG002
        """Delete docs by ID. These are the IDs in the vectorstore.

        Args:
            ids: List of ids to delete.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        if ids is None:
            raise ValueError("No ids provided to delete.")

        missing_ids = set()
        id_doc_map: dict[str, Document] = {}

        for id_ in ids:
            doc = self.metastore.search(id_)
            if isinstance(doc, str):
                missing_ids.add(id_)
            else:
                id_doc_map[id_] = doc

        if missing_ids:
            raise ValueError(
                f"Some specified ids do not exist in the current store. Ids not found: "
                f"{missing_ids}"
            )

        self.metastore.delete(list(id_doc_map.keys()))

        tag_doc_cache = {}
        for doc_id, doc in id_doc_map.items():
            metadata = cast("DocumentMeta", doc.metadata)
            for tag in metadata["tag_hashs"]:
                if tag not in tag_doc_cache:
                    tag_doc_cache[tag] = self.vector.docstore.search(tag)
                tag_doc = tag_doc_cache[tag]
                if isinstance(tag_doc, Document):
                    cast("TagDucumentId", tag_doc.metadata)["file_id"].remove(doc_id)
                else:
                    return None

        return True

    def delete_tags(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Delete tags by ID. These are the IDs in the vectorstore.

        Args:
            ids: List of ids to delete.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        return self.vector.delete(ids, **kwargs)

    def save_local(self) -> None:
        """Save FAISS index, docstore, and index_to_docstore_id to disk.

        Args:
            folder_path: folder path to save index, docstore,
                and index_to_docstore_id to.
            index_name: for saving with a specific index file name
        """
        path = Path(self.folder_path)
        path.mkdir(exist_ok=True, parents=True)

        # save index separately since it is not picklable
        faiss = dependable_faiss_import()
        faiss.write_index(self.vector.index, str(path / f"{self.index_name}.faiss"))

        # save docstore and index_to_docstore_id
        with (path / f"{self.index_name}.pkl").open("wb") as f:
            pickle.dump((self.metastore, self.vector.docstore, self.vector.index_to_docstore_id), f)

    async def asave_local(self) -> None:
        """Async save FAISS index, docstore, and index_to_docstore_id to disk.

        Args:
            folder_path: folder path to save index, docstore,
                and index_to_docstore_id to.
            index_name: for saving with a specific index file name
        """
        path = Path(self.folder_path)
        path.mkdir(exist_ok=True, parents=True)

        faiss = dependable_faiss_import()

        def _write_pickle(pkl_path: Path) -> None:
            with pkl_path.open("wb") as f:
                pickle.dump(
                    (self.metastore, self.vector.docstore, self.vector.index_to_docstore_id), f
                )

        async with anyio.create_task_group() as tg:
            tg.start_soon(
                to_thread.run_sync,
                faiss.write_index,
                self.vector.index,
                str(path / f"{self.index_name}.faiss"),
            )
            tg.start_soon(to_thread.run_sync, _write_pickle, path / f"{self.index_name}.pkl")

    @staticmethod
    def extract_high_score(
        results: list[tuple[Document, float]],
        *,
        min_relative_jump: float = 0.7,
        soft_threshold: float = 1,
        fallback_k: int = 3,
        min_gap: float = 0.5,
        **kwargs: Any,  # noqa: ARG004
    ) -> list[tuple[Document, float]]:
        if not results:
            return []
        if len(results) == 1:
            return results if results[0][1] > soft_threshold else []
        if len(results) == 2:  # noqa: PLR2004
            return (
                [results[0]]
                if results[0][1] - results[1][1] > min_gap
                else (results if results[0][1] + results[1][1] > soft_threshold * 2 else [])
            )

        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        scores = np.array([s for _, s in sorted_results])
        n = len(scores)
        x = np.arange(n)
        y = scores

        line_start, line_end = np.array([0, y[0]]), np.array([n - 1, y[-1]])
        line_vec = line_end - line_start
        line_vec_norm = line_vec / np.linalg.norm(line_vec)
        vec_from_start = np.stack([x, y], axis=1) - line_start
        proj_lengths = np.dot(vec_from_start, line_vec_norm)
        proj_points = np.outer(proj_lengths, line_vec_norm) + line_start
        distances = np.linalg.norm(vec_from_start - proj_points, axis=1)
        knee_idx = int(np.argmax(distances))

        max_jump = distances[knee_idx]
        score_range = y[0] - y[-1]
        relative_jump = max_jump / (score_range + 1e-8)
        if relative_jump > min_relative_jump and knee_idx >= 1:
            return sorted_results[: min(knee_idx, fallback_k)]

        if scores.mean() >= soft_threshold:
            return sorted_results[:fallback_k]

        return []
