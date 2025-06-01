from typing import Any, Generic, TypeVar

from pydantic import BaseModel, Field

MetaType = TypeVar("MetaType")
MetaStoreDataType = TypeVar("MetaStoreDataType")


class MetaData(BaseModel, Generic[MetaType]):
    id: str | None = Field(default=None, coerce_numbers_to_str=True)

    content: MetaType
    metadata: dict[Any, Any] = Field(default_factory=dict)


class MetaStore(Generic[MetaStoreDataType]):
    def __init__(self, _dict: dict[str, MetaStoreDataType] | None = None) -> None:
        """Initialize with dict."""
        self._dict = _dict if _dict is not None else {}

    def add(self, texts: dict[str, MetaStoreDataType]) -> None:
        """Add texts to in memory dictionary.

        Args:
            texts: dictionary of id -> MetaType.

        Returns:
            None
        """
        overlapping = set(texts).intersection(self._dict)
        if overlapping:
            raise ValueError(f"Tried to add ids that already exist: {overlapping}")
        self._dict = {**self._dict, **texts}

    def delete(self, ids: list) -> None:
        """Deleting IDs from in memory dictionary."""
        overlapping = set(ids).intersection(self._dict)
        if not overlapping:
            raise ValueError(f"Tried to delete ids that does not  exist: {ids}")
        for _id in ids:
            self._dict.pop(_id)

    def search(self, search: str) -> str | MetaStoreDataType:
        """Search via direct lookup.

        Args:
            search: id of a MetaType to search for.

        Returns:
            MetaType if found, else error message.
        """
        if search not in self._dict:
            return f"ID {search} not found."
        return self._dict[search]
