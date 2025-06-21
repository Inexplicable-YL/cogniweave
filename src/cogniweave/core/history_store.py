from __future__ import annotations

from typing import Any, Literal

from cogniweave.core.prompts import LongMemoryPromptTemplate, ShortMemoryPromptTemplate
from cogniweave.history_store import (
    BaseHistoryStoreWithCache,
    BlockAttributeData,
    UserAttributeData,
)

_SHORT_MEMORY_KEY: Literal["_short_memory"] = "_short_memory"


class HistoryStore(BaseHistoryStoreWithCache):
    """History store with short memory convenience methods."""

    def __init__(
        self,
        *,
        db_url: str | None = None,
        echo: bool = False,
        max_cache_blocks: int = 20,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the history store with caching capabilities.

        Args:
            db_url (str | None): Database URL for the history store.
            echo (bool): Whether to echo SQL statements.
            max_cache_blocks (int): Maximum number of blocks to cache per session.
            **kwargs: Additional keyword arguments for compatibility with subclasses.
        """
        super().__init__(db_url=db_url, echo=echo, max_cache_blocks=max_cache_blocks, **kwargs)

    def add_short_memory(
        self,
        short_memory: ShortMemoryPromptTemplate,
        *,
        block_id: str,
        block_ts: float,
        session_id: str | None = None,
    ) -> None:
        """Add short memory data for a specific block.

        This is a convenience method that combines message storage with
        short memory attribute storage.

        Args:
            short_memory: Short memory prompt template instance.
            block_id: Unique identifier for the message block.
            block_ts: Unix timestamp for the block start time.
            session_id: Optional session/user ID. Uses block_id if not provided.

        Return:
            None: Messages and attributes are persisted to the database.
        """
        short_memory_data = short_memory.to_template_dict()

        self.add_block_attributes(
            [BlockAttributeData(type=_SHORT_MEMORY_KEY, value=short_memory_data)],
            block_id=block_id,
            block_ts=block_ts,
            session_id=session_id,
        )

    async def aadd_short_memory(
        self,
        short_memory: ShortMemoryPromptTemplate,
        *,
        block_id: str,
        block_ts: float,
        session_id: str | None = None,
    ) -> None:
        """Async add short memory data for a specific block.

        This is a convenience method that combines message storage with
        short memory attribute storage.

        Args:
            short_memory: Short memory prompt template instance.
            block_id: Unique identifier for the message block.
            block_ts: Unix timestamp for the block start time.
            session_id: Optional session/user ID. Uses block_id if not provided.

        Return:
            None: Messages and attributes are persisted to the database."""
        short_memory_data = short_memory.to_template_dict()

        await self.aadd_block_attributes(
            [BlockAttributeData(type=_SHORT_MEMORY_KEY, value=short_memory_data)],
            block_id=block_id,
            block_ts=block_ts,
            session_id=session_id,
        )

    def get_short_memory(self, block_id: str) -> ShortMemoryPromptTemplate | None:
        """Get short memory data for a specific block.

        This is a convenience method that wraps get_block_attributes
        to specifically retrieve short memory data.

        Args:
            block_id: The ID of the block to query.

        Return:
            ShortMemoryPromptTemplate | None: Short memory data if found, None otherwise.
        """
        attributes = self.get_block_attributes(block_id, types=[_SHORT_MEMORY_KEY])
        if attributes:
            return ShortMemoryPromptTemplate.load(attributes[0].get("value"))
        return None

    async def aget_short_memory(self, block_id: str) -> ShortMemoryPromptTemplate | None:
        """Async get short memory data for a specific block.

        This is a convenience method that wraps get_block_attributes
        to specifically retrieve short memory data.

        Args:
            block_id: The ID of the block to query.

        Return:
            ShortMemoryPromptTemplate | None: Short memory data if found, None otherwise.
        """
        attributes = await self.aget_block_attributes(block_id, types=[_SHORT_MEMORY_KEY])
        if attributes:
            return ShortMemoryPromptTemplate.load(attributes[0].get("value"))
        return None

    def add_long_memory(self, long_memory: LongMemoryPromptTemplate, *, session_id: str) -> None:
        """Add long memory data for a specific block.

        Args:
            long_memory: Long memory prompt template instance.
            session_id: session/user ID.

        Return:
            None: Messages and attributes are persisted to the database.
        """
        long_memory_data = long_memory.to_template_dict()

        self.add_user_attributes(
            [UserAttributeData(type="long_memory", value=long_memory_data)],
            session_id=session_id,
        )

    async def aadd_long_memory(
        self, long_memory: LongMemoryPromptTemplate, *, session_id: str
    ) -> None:
        """Async add long memory data for a specific block.

        Args:
            long_memory: Long memory prompt template instance.
            session_id: session/user ID.

        Return:
            None: Messages and attributes are persisted to the database.
        """
        long_memory_data = long_memory.to_template_dict()

        await self.aadd_user_attributes(
            [UserAttributeData(type="long_memory", value=long_memory_data)],
            session_id=session_id,
        )
