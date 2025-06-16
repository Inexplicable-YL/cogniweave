from __future__ import annotations

from typing import Literal

from cogniweave.core.prompts import (
    ShortMemoryPromptTemplate,
)
from cogniweave.historystore import BaseHistoryStoreWithCache, BlockAttributeData

_SHORT_MEMORY_KEY: Literal["_short_memory"] = "_short_memory"


class HistoryStore(BaseHistoryStoreWithCache):
    """History store with short memory convenience methods."""

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

        self.add_attributes(
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

        await self.aadd_attributes(
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
