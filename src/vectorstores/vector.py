from __future__ import annotations

import inspect
import logging
import pickle
import warnings
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    TypedDict,
)
from typing_extensions import override

import anyio
from anyio import to_thread
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores.faiss import FAISS, dependable_faiss_import
from langchain_community.vectorstores.utils import (
    DistanceStrategy,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from langchain_core.embeddings import Embeddings

logger = logging.getLogger(__name__)


class SavePath(TypedDict):
    folder_path: Path
    index_name: str


class UninitializedWarning(UserWarning):
    """Warning raised when accessing a method before initialization."""


class LazyFAISS(FAISS):
    _save_path: SavePath | None = None
    _uninitialized: bool = False

    @override
    def __getattribute__(self, name: str) -> Any:
        obj = super().__getattribute__(name)
        if (
            super().__getattribute__("_uninitialized")
            and callable(obj)
            and not name.startswith("_")
            and "add" not in name
        ):
            if "search" in name:
                warnings.warn(
                    f"Cannot call '{name}' before initialization. "
                    f"Only 'add' methods are allowed when _uninitialized=True.",
                    UninitializedWarning,
                    stacklevel=2,
                )
                return (
                    lambda *_args, **_kwargs: to_thread.run_sync(list)
                    if inspect.iscoroutinefunction(obj)
                    else []
                )
            raise RuntimeError(
                f"Cannot access '{name}' before initialization. "
                f"Only 'add' methods are allowed when _uninitialized=True."
            )

        return obj

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
        docstore = kwargs.pop("docstore", InMemoryDocstore())
        index_to_docstore_id = kwargs.pop("index_to_docstore_id", {})

        self.index = index
        self.docstore = docstore
        self.index_to_docstore_id = index_to_docstore_id
        self.distance_strategy = distance_strategy
        self._normalize_L2 = normalize_L2
        if self.distance_strategy != DistanceStrategy.EUCLIDEAN_DISTANCE and self._normalize_L2:
            warnings.warn(
                f"Normalizing L2 is not applicable for metric type: {self.distance_strategy}",
                stacklevel=2,
            )

    def _FAISS__add(
        self,
        texts: Iterable[str],
        embeddings: Iterable[list[float]],
        metadatas: Iterable[dict] | None = None,
        ids: list[str] | None = None,
    ) -> list[str]:
        if self._uninitialized:
            self.__init(list(embeddings))
            self._uninitialized = False
        return super()._FAISS__add(texts, embeddings, metadatas, ids)  # type: ignore

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
        if not ((path / f"{index_name}.faiss").exists() and (path / f"{index_name}.pkl").exists()):
            vector = object.__new__(LazyFAISS)
            vector._uninitialized = True
            vector._save_path = {"folder_path": path, "index_name": index_name}
            vector.embedding_function = embeddings
            return vector

        # load index separately since it is not picklable
        faiss = dependable_faiss_import()
        index = faiss.read_index(str(path / f"{index_name}.faiss"))

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
                "or `self.save_path` must be set beforehand."
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
                "or `self.save_path` must be set beforehand."
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
