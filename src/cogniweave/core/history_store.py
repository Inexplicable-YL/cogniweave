from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cogniweave.historystore.cache import BaseHistoryStoreWithCache

if TYPE_CHECKING:
    from langchain_core.messages import BaseMessage


class HistoryStore(BaseHistoryStoreWithCache):
    """History store with short memory convenience methods."""

    def add_short_memory(
        self,
        messages: list[tuple[BaseMessage, float]],
        *,
        block_id: str,
        block_ts: float,
        session_id: str | None = None,
        short_memory_data: dict[str, Any] | None = None,
    ) -> None:
        """Add messages and short memory attributes to the store.

        This is a convenience method that combines message storage with
        short memory attribute storage.

        Args:
            messages: List of (message, timestamp) pairs to store.
            block_id: Unique identifier for the message block.
            block_ts: Unix timestamp for the block start time.
            session_id: Optional session/user ID. Uses block_id if not provided.
            short_memory_data: Optional short memory data to store as attributes.

        Return:
            None: Messages and attributes are persisted to the database.
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

        # Store short memory attributes if provided
        if short_memory_data:
            attributes = [
                {
                    "type": "short_memory",
                    "value": short_memory_data,
                }
            ]
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
        short_memory_data: dict[str, Any] | None = None,
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

        # Store short memory attributes if provided
        if short_memory_data:
            attributes = [
                {
                    "type": "short_memory",
                    "value": short_memory_data,
                }
            ]
            await self.aadd_attributes(
                attributes,
                block_id=block_id,
                block_ts=block_ts,
                session_id=session_id,
            )

    def get_short_memory(self, block_id: str) -> dict[str, Any] | None:
        """Get short memory data for a specific block.

        This is a convenience method that wraps get_block_attributes
        to specifically retrieve short memory data.

        Args:
            block_id: The ID of the block to query.

        Return:
            dict[str, Any] | None: Short memory data if found, None otherwise.
        """
        attributes = self.get_block_attributes(block_id, types=["short_memory"])
        if attributes:
            return attributes[0].get("value")
        return None

    async def aget_short_memory(self, block_id: str) -> dict[str, Any] | None:
        """Async version of get_short_memory."""
        attributes = await self.aget_block_attributes(block_id, types=["short_memory"])
        if attributes:
            return attributes[0].get("value")
        return None
