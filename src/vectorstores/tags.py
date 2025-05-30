from collections.abc import Iterable
from typing import Any
from pydantic import BaseModel

from .base import LazyFAISS


class TagsVector(BaseModel):
    vector: LazyFAISS

    def add_tags(
        self,
        tags: Iterable[str],
        metadatas: dict[Any, Any] | None = None,
        ids: str | None = None,
        **kwargs: Any,
    ) -> list[str]: ...
