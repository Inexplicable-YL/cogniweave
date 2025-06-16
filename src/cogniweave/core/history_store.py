from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel
from cogniweave.historystore.cache import BaseHistoryStoreWithCache

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage

class ShortMemory(BaseModel):
    """Short-term memory data model for conversation blocks."""
    
    summary: str
    tags: list[str]
    sentiment: str | None = None
    entities: list[str] | None = None
    confidence: float | None = None
    metadata: dict[str, str] | None = None


class LongMemory(BaseModel):
    """Long-term memory data model for conversation blocks."""
    
    key_insights: list[str]
    patterns: list[str] | None = None
    relationships: dict[str, str] | None = None
    importance_score: float | None = None
    metadata: dict[str, str] | None = None


class BlockContext(BaseModel):
    """Complete context for a conversation block."""
    
    block_id: str
    messages: list[tuple[BaseMessage, float]]
    short_memory: ShortMemory | None = None
    long_memory: LongMemory | None = None

class HistoryStore(BaseHistoryStoreWithCache):
    """History store with short memory convenience methods."""

    def add_short_memory(
        self,
        messages: list[tuple[BaseMessage, float]],
        *,
        block_id: str,
        block_ts: float,
        session_id: str | None = None,
        short_memory: ShortMemory,
    ) -> None:
        """Add messages and short memory to the store.

        Args:
            messages: List of (message, timestamp) pairs to store.
            block_id: Unique identifier for the message block.
            block_ts: Unix timestamp for the block start time.
            session_id: Optional session/user ID. Uses block_id if not provided.
            short_memory: Short memory object to store.
        """
        # Don't create block if no messages are provided
        if not messages:
            return

        # Store messages first
        self.add_messages(
            messages,
            block_id=block_id,
            block_ts=block_ts,
            session_id=session_id,
        )

        # Store short memory (always provided, not optional)
        attributes = [
            {
                "type": "short_memory",
                "value": short_memory.model_dump(),
            }
        ]
        self.add_attributes(
            attributes,
            block_id=block_id,
            block_ts=block_ts,
            session_id=session_id,
        )

    def add_memories(
        self,
        messages: list[tuple[BaseMessage, float]],
        *,
        block_id: str,
        block_ts: float,
        session_id: str | None = None,
        short_memory: ShortMemory | None = None,
        long_memory: LongMemory | None = None,
    ) -> None:
        """Add messages and memories to the store (matches workflow diagram).

        Args:
            messages: List of (message, timestamp) pairs to store.
            block_id: Unique identifier for the message block.
            block_ts: Unix timestamp for the block start time.
            session_id: Optional session/user ID. Uses block_id if not provided.
            short_memory: Optional short memory object to store.
            long_memory: Optional long memory object to store.
        """
        # Don't create block if no messages are provided
        if not messages:
            return

        # Store messages first
        self.add_messages(
            messages,
            block_id=block_id,
            block_ts=block_ts,
            session_id=session_id,
        )

        # Store memories if provided
        attributes = []
        if short_memory:
            attributes.append({
                "type": "short_memory",
                "value": short_memory.model_dump(),
            })
        if long_memory:
            attributes.append({
                "type": "long_memory", 
                "value": long_memory.model_dump(),
            })
        
        if attributes:
            self.add_attributes(
                attributes,
                block_id=block_id,
                block_ts=block_ts,
                session_id=session_id,
            )

    async def aadd_short_memory(
        self,
        messages: list[tuple[BaseMessage, float]],
        *,
        block_id: str,
        block_ts: float,
        session_id: str | None = None,
        short_memory: ShortMemory,
    ) -> None:
        """Async version of add_short_memory."""
        # Don't create block if no messages are provided
        if not messages:
            return

        # Store messages first
        await self.aadd_messages(
            messages,
            block_id=block_id,
            block_ts=block_ts,
            session_id=session_id,
        )

        # Store short memory (always provided, not optional)
        attributes = [
            {
                "type": "short_memory",
                "value": short_memory.model_dump(),
            }
        ]
        await self.aadd_attributes(
            attributes,
            block_id=block_id,
            block_ts=block_ts,
            session_id=session_id,
        )

    def get_short_memory(self, block_id: str) -> ShortMemory | None:
        """Get short memory data for a block.

        Args:
            block_id: The ID of the block to query.

        Return:
            ShortMemory object if found, None otherwise.
        """
        attributes = self.get_block_attributes(block_id, types=["short_memory"])
        if not attributes:
            return None

        stored_value = attributes[0].get("value")
        if not stored_value:
            return None

        # Deserialize back to ShortMemory object
        return ShortMemory.model_validate(stored_value)

    def get_long_memory(self, block_id: str) -> LongMemory | None:
        """Get long memory data for a block.

        Args:
            block_id: The ID of the block to query.

        Return:
            LongMemory object if found, None otherwise.
        """
        attributes = self.get_block_attributes(block_id, types=["long_memory"])
        if not attributes:
            return None

        stored_value = attributes[0].get("value")
        if not stored_value:
            return None

        # Deserialize back to LongMemory object
        return LongMemory.model_validate(stored_value)

    def get_block_context(self, block_id: str) -> BlockContext | None:
        """Get complete context for a block (matches workflow diagram).
        
        This method extracts history, short_memory, long_memory as shown in the diagram.

        Args:
            block_id: The ID of the block to query.

        Return:
            BlockContext with messages and memories, or None if block not found.
        """
        # Get messages
        messages = self.get_block_history_with_timestamps(block_id)
        if not messages:
            return None

        # Get memories
        short_memory = self.get_short_memory(block_id)
        long_memory = self.get_long_memory(block_id)

        return BlockContext(
            block_id=block_id,
            messages=messages,
            short_memory=short_memory,
            long_memory=long_memory,
        )

    async def aget_short_memory(self, block_id: str) -> ShortMemory | None:
        """Async version of get_short_memory."""
        attributes = await self.aget_block_attributes(block_id, types=["short_memory"])
        if not attributes:
            return None

        stored_value = attributes[0].get("value")
        if not stored_value:
            return None

        # Deserialize back to ShortMemory object
        return ShortMemory.model_validate(stored_value)

    async def aget_long_memory(self, block_id: str) -> LongMemory | None:
        """Async version of get_long_memory."""
        attributes = await self.aget_block_attributes(block_id, types=["long_memory"])
        if not attributes:
            return None

        stored_value = attributes[0].get("value")
        if not stored_value:
            return None

        # Deserialize back to LongMemory object
        return LongMemory.model_validate(stored_value)

    async def aget_block_context(self, block_id: str) -> BlockContext | None:
        """Async version of get_block_context."""
        # Get messages
        messages = await self.aget_block_history_with_timestamps(block_id)
        if not messages:
            return None

        # Get memories
        short_memory = await self.aget_short_memory(block_id)
        long_memory = await self.aget_long_memory(block_id)

        return BlockContext(
            block_id=block_id,
            messages=messages,
            short_memory=short_memory,
            long_memory=long_memory,
        )
