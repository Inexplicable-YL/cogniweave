from __future__ import annotations

import logging
import pickle
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypedDict
from typing_extensions import override

import anyio
from anyio import to_thread
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.faiss import FAISS, dependable_faiss_import
from langchain_community.vectorstores.utils import (
    DistanceStrategy,
)
from langchain_core.embeddings import Embeddings

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from langchain_community.docstore.base import Docstore
    from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class SavePath(TypedDict):
    folder_path: Path
    index_name: str


class UninitializedWarning(UserWarning):
    """Warning raised when accessing a method before initialization."""


class LazyFAISS(FAISS):
    _save_path: SavePath | None = None

    def __init__(
        self,
        embedding_function: Callable[[str], list[float]] | Embeddings,
        index: Any | None = None,
        docstore: Docstore | None = None,
        index_to_docstore_id: dict[int, str] | None = None,
        relevance_score_fn: Callable[[float], float] | None = None,
        normalize_L2: bool = False,  # noqa: N803
        distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
    ) -> None:
        """Initialize with necessary components."""
        if not isinstance(embedding_function, Embeddings):
            logger.warning(
                "`embedding_function` is expected to be an Embeddings object, support "
                "for passing in a function will soon be removed."
            )
        self.embedding_function = embedding_function
        self.index = index
        self.docstore = docstore or InMemoryDocstore()
        self.index_to_docstore_id = index_to_docstore_id or {}
        self.distance_strategy = distance_strategy
        self.override_relevance_score_fn = relevance_score_fn
        self._normalize_L2 = normalize_L2
        if self.distance_strategy != DistanceStrategy.EUCLIDEAN_DISTANCE and self._normalize_L2:
            warnings.warn(
                f"Normalizing L2 is not applicable for metric type: {self.distance_strategy}",
                stacklevel=2,
            )

    def __init(
        self,
        embeddings: list[list[float]],
        normalize_L2: bool = False,  # noqa: N803
        distance_strategy: DistanceStrategy = DistanceStrategy.EUCLIDEAN_DISTANCE,
        **kwargs: Any,
    ) -> None:
        faiss = dependable_faiss_import()
        if distance_strategy == DistanceStrategy.MAX_INNER_PRODUCT:
            index = faiss.IndexFlatIP(len(embeddings[0]))
        else:
            # Default to L2, currently other metric types not initialized.
            index = faiss.IndexFlatL2(len(embeddings[0]))
        docstore = self.docstore or kwargs.pop("docstore", InMemoryDocstore())
        index_to_docstore_id = self.index_to_docstore_id or kwargs.pop("index_to_docstore_id", {})

        super().__init__(
            embedding_function=self.embedding_function,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            normalize_L2=normalize_L2,
            distance_strategy=distance_strategy,
            **kwargs,
        )

    def _FAISS__add(
        self,
        texts: Iterable[str],
        embeddings: Iterable[list[float]],
        metadatas: Iterable[dict] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        if self.index is None:
            self.__init(list(embeddings))
        return super()._FAISS__add(texts, embeddings, metadatas, ids)  # type: ignore

    @override
    def similarity_search_with_score_by_vector(
        self,
        embedding: list[float],
        k: int = 4,
        filter: Callable | dict[str, Any] | None = None,
        fetch_k: int = 20,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Return docs most similar to query.

        Args:
            embedding: Embedding vector to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter (Optional[Union[Callable, Dict[str, Any]]]): Filter by metadata.
                Defaults to None. If a callable, it must take as input the
                metadata dict of Document and return a bool.
            fetch_k: (Optional[int]) Number of Documents to fetch before filtering.
                      Defaults to 20.
            **kwargs: kwargs to be passed to similarity search. Can include:
                score_threshold: Optional, a floating point value between 0 to 1 to
                    filter the resulting set of retrieved docs

        Returns:
            List of documents most similar to the query text and L2 distance
            in float for each. Lower score represents more similarity.
        """
        if self.index is None:
            warnings.warn(
                "Cannot similarity search before initialization. "
                "Only 'add' methods are allowed when FAISS is uninitialized.",
                UninitializedWarning,
                stacklevel=2,
            )
            return []
        return super().similarity_search_with_score_by_vector(
            embedding, k, filter, fetch_k, **kwargs
        )

    @override
    def max_marginal_relevance_search_with_score_by_vector(
        self,
        embedding: list[float],
        *,
        k: int = 4,
        fetch_k: int = 20,
        lambda_mult: float = 0.5,
        filter: Callable | dict[str, Any] | None = None,
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
        if self.index is None:
            warnings.warn(
                "Cannot max marginal relevance search before initialization. "
                "Only 'add' methods are allowed when FAISS is uninitialized.",
                UninitializedWarning,
                stacklevel=2,
            )
            return []
        return super().max_marginal_relevance_search_with_score_by_vector(
            embedding, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult, filter=filter
        )

    @override
    def delete(self, ids: list[str] | None = None, **kwargs: Any) -> bool | None:
        """Delete by ID. These are the IDs in the vectorstore.

        Args:
            ids: List of ids to delete.

        Returns:
            Optional[bool]: True if deletion is successful,
            False otherwise, None if not implemented.
        """
        if self.index is None:
            warnings.warn(
                "Cannot delete before initialization. "
                "Only 'add' methods are allowed when FAISS is uninitialized.",
                UninitializedWarning,
                stacklevel=2,
            )
            return None
        return super().delete(ids, **kwargs)

    @override
    def merge_from(self, target: FAISS) -> None:
        """Merge another FAISS object with the current one.

        Add the target FAISS to the current one.

        Args:
            target: FAISS object you wish to merge into the current one

        Returns:
            None.
        """
        if self.index is None:
            warnings.warn(
                "Merging into an uninitialized FAISS object. "
                "The current object has been replaced with the target.",
                UninitializedWarning,
                stacklevel=2,
            )
            self.index = target.index
            self.docstore = target.docstore
            self.index_to_docstore_id = target.index_to_docstore_id
            return
        super().merge_from(target)

    @classmethod
    @override
    def load_local(
        cls,
        folder_path: str,
        embeddings: Embeddings,
        index_name: str = "index",
        *,
        allow_dangerous_deserialization: bool = False,
        **kwargs: Any,
    ) -> LazyFAISS:
        """Load FAISS index, docstore, and index_to_docstore_id from disk.

        Args:
            folder_path: folder path to load index, docstore,
                and index_to_docstore_id from.
            embeddings: Embeddings to use when generating queries
            index_name: for saving with a specific index file name
            allow_dangerous_deserialization: whether to allow deserialization
                of the data which involves loading a pickle file.
                Pickle files can be modified by malicious actors to deliver a
                malicious payload that results in execution of
                arbitrary code on your machine.
        """
        if not allow_dangerous_deserialization:
            raise ValueError(
                "The de-serialization relies loading a pickle file. "
                "Pickle files can be modified to deliver a malicious payload that "
                "results in execution of arbitrary code on your machine."
                "You will need to set `allow_dangerous_deserialization` to `True` to "
                "enable deserialization. If you do this, make sure that you "
                "trust the source of the data. For example, if you are loading a "
                "file that you created, and know that no one else has modified the "
                "file, then this is safe to do. Do not set this to `True` if you are "
                "loading a file from an untrusted source (e.g., some random site on "
                "the internet.)."
            )
        path = Path(folder_path)
        index = docstore = index_to_docstore_id = None

        if (path / f"{index_name}.faiss").exists():
            # load index separately since it is not picklable
            faiss = dependable_faiss_import()
            index = faiss.read_index(str(path / f"{index_name}.faiss"))

        if (path / f"{index_name}.pkl").exists():
            # load docstore and index_to_docstore_id
            with (path / f"{index_name}.pkl").open("rb") as f:
                (
                    docstore,
                    index_to_docstore_id,
                ) = pickle.load(  # ignore[pickle]: explicit-opt-in  # noqa: S301
                    f
                )
        vector = cls(embeddings, index, docstore, index_to_docstore_id, **kwargs)
        vector._save_path = {"folder_path": path, "index_name": index_name}
        return vector

    @override
    def save_local(self, folder_path: str | None = None, index_name: str = "index") -> None:
        """Save FAISS index, docstore, and index_to_docstore_id to disk.

        Args:
            folder_path: folder path to save index, docstore,
                and index_to_docstore_id to.
            index_name: for saving with a specific index file name
        """
        if folder_path is not None:
            path = Path(folder_path)
        elif self._save_path is not None:
            path = Path(self._save_path["folder_path"])
        else:
            raise ValueError(
                "Missing save path: either `folder_path` must be provided explicitly "
                "or `self._save_path` must be set beforehand."
            )
        path.mkdir(exist_ok=True, parents=True)

        # save index separately since it is not picklable
        faiss = dependable_faiss_import()
        faiss.write_index(self.index, str(path / f"{index_name}.faiss"))

        # save docstore and index_to_docstore_id
        with (path / f"{index_name}.pkl").open("wb") as f:
            pickle.dump((self.docstore, self.index_to_docstore_id), f)

    async def asave_local(self, folder_path: str | None = None, index_name: str = "index") -> None:
        """Async save FAISS index, docstore, and index_to_docstore_id to disk.

        Args:
            folder_path: folder path to save index, docstore,
                and index_to_docstore_id to.
            index_name: for saving with a specific index file name
        """
        if folder_path is not None:
            path = Path(folder_path)
        elif self._save_path is not None:
            path = Path(self._save_path["folder_path"])
        else:
            raise ValueError(
                "Missing save path: either `folder_path` must be provided explicitly "
                "or `self._save_path` must be set beforehand."
            )
        path.mkdir(exist_ok=True, parents=True)

        faiss = dependable_faiss_import()

        def _write_pickle(pkl_path: Path) -> None:
            with pkl_path.open("wb") as f:
                pickle.dump((self.docstore, self.index_to_docstore_id), f)

        async with anyio.create_task_group() as tg:
            tg.start_soon(
                to_thread.run_sync, faiss.write_index, self.index, str(path / f"{index_name}.faiss")
            )
            tg.start_soon(to_thread.run_sync, _write_pickle, path / f"{index_name}.pkl")

    def similarity_search_with_score_by_threshold(
        self,
        query: str,
        k: int = 4,
        filter: Callable | dict[str, Any] | None = None,
        fetch_k: int = 20,
        min_score: float | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Search for similar documents with scores, filtered by minimum similarity score.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter by metadata. Defaults to None.
            fetch_k: Number of Documents to fetch before filtering. Defaults to 20.
            min_score: Minimum similarity score threshold.
                    Results with scores below this threshold will be filtered out.
            **kwargs: Additional keyword arguments.

        Returns:
            List of documents with similarity scores.
        """

        return super().similarity_search_with_score(
            query, k, filter=filter, fetch_k=fetch_k, score_threshold=min_score, **kwargs
        )

    async def asimilarity_search_with_score_by_threshold(
        self,
        query: str,
        k: int = 4,
        filter: Callable | dict[str, Any] | None = None,
        fetch_k: int = 20,
        min_score: float | None = None,
        **kwargs: Any,
    ) -> list[tuple[Document, float]]:
        """Asynchronously search for similar documents with scores, filtered by minimum similarity score.

        Args:
            query: Text to look up documents similar to.
            k: Number of Documents to return. Defaults to 4.
            filter: Filter by metadata. Defaults to None.
            fetch_k: Number of Documents to fetch before filtering. Defaults to 20.
            min_score: Minimum similarity score threshold.
                    Results with scores below this threshold will be filtered out.
            **kwargs: Additional keyword arguments.

        Returns:
            List of documents with similarity scores.
        """

        return await super().asimilarity_search_with_score(
            query, k, filter=filter, fetch_k=fetch_k, score_threshold=min_score, **kwargs
        )
