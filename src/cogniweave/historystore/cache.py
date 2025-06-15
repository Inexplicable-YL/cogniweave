from __future__ import annotations

import bisect
from collections import defaultdict
from dataclasses import dataclass, field
from functools import partial
from itertools import groupby
from typing import TYPE_CHECKING, Any, cast
from typing_extensions import override

from pydantic import PrivateAttr
from sortedcontainers import SortedDict, SortedList

if TYPE_CHECKING:
    from collections.abc import Iterable

    from langchain_core.messages import BaseMessage

from .base import BaseHistoryStore


@dataclass
class SessionCache:
    """Cache for a single session."""

    max_blocks: int

    blocks: dict[tuple[str, float], list[tuple[BaseMessage, float]]] = field(
        default_factory=lambda: SortedDict(lambda x: x[1]), init=False
    )
    start_block_ts: float = field(default=float("inf"), init=False)
    end_block_ts: float = field(default=float("inf"), init=False)
    start_msg_ts: float = field(default=float("inf"), init=False)
    end_msg_ts: float = field(default=float("inf"), init=False)

    def _recompute_ranges(self) -> None:
        first_block = cast("SortedDict", self.blocks).keys()[0]
        last_block = cast("SortedDict", self.blocks).keys()[-1]

        self.start_block_ts = first_block[1]
        self.end_block_ts = last_block[1]

        self.start_msg_ts = self.blocks[first_block][0][1]
        self.end_msg_ts = self.blocks[last_block][-1][1]

    def add_messages(
        self, block_id: str, block_ts: float, messages: Iterable[tuple[BaseMessage, float]]
    ) -> None:
        key = (block_id, float(block_ts))
        if key in self.blocks:
            cast("SortedList", self.blocks[key]).update(messages)
        else:
            self.blocks[key] = SortedList(messages, key=lambda x: x[1])
        while len(self.blocks) > self.max_blocks:
            cast("SortedDict", self.blocks).popitem(index=0)
        self._recompute_ranges()

    def get_blocks(self, start_time: float, end_time: float) -> list[tuple[str, float]]:
        return list(
            cast("SortedDict", self.blocks).irange(("", start_time), (chr(0x10FFFF), end_time))
        )

    def get_block_timestamp(self, block_id: str) -> float | None:
        """Get the timestamp of a block by its ID."""
        for bid, ts in self.blocks:
            if bid == block_id:
                return ts
        return None

    def get_block_histories_with_timestamps(
        self, block_ids: list[str]
    ) -> list[tuple[BaseMessage, float]]:
        result: list[tuple[BaseMessage, float]] = []
        for block_id in block_ids:
            for (bid, _), messages in self.blocks.items():
                if bid == block_id:
                    result.extend(messages)
        return result

    def get_messages(self, start_time: float, end_time: float) -> list[tuple[BaseMessage, float]]:
        keys = list(self.blocks.keys())
        block_ids = [k[0] for k in keys]
        timestamps = [k[1] for k in keys]
        start, end = (
            bisect.bisect_left(timestamps, start_time),
            bisect.bisect_right(timestamps, end_time) - 1,
        )
        left, right = max(0, start - 1), min(len(keys) - 1, end + 1)
        selected_ids = block_ids[left : right + 1]
        if not selected_ids:
            return []

        history = self.get_block_histories_with_timestamps(selected_ids)
        return [(msg, ts) for msg, ts in history if start_time <= ts <= end_time]


class BaseHistoryStoreWithCache(BaseHistoryStore):
    """
    Base class for history stores with caching capabilities.

    This class extends BaseHistoryStore to include caching functionality.
    It can be used as a base class for specific history store implementations
    that require caching.
    """

    _session_caches: dict[str, SessionCache] = PrivateAttr()

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
        super().__init__(db_url=db_url, echo=echo, **kwargs)
        self._session_caches = defaultdict(lambda: SessionCache(max_blocks=max_cache_blocks))

    @override
    def add_messages(
        self,
        messages: list[tuple[BaseMessage, float]],
        *,
        block_id: str,
        block_ts: float,
        session_id: str | None = None,
    ) -> None:
        """Persist a list of messages with timestamps to the store.

        Args:
            messages: List of (message, timestamp) pairs to store.
            block_id: Unique identifier for the message block.
            block_ts: Unix timestamp for the block start time.
            session_id: Optional session/user ID. Uses block_id if not provided.

        Return:
            None: Messages are persisted to the database.
        """
        if not messages:
            return
        session_id = session_id or block_id
        super().add_messages(messages, block_id=block_id, block_ts=block_ts, session_id=session_id)
        self._session_caches[session_id].add_messages(block_id, block_ts, messages)

    @override
    async def aadd_messages(
        self,
        messages: list[tuple[BaseMessage, float]],
        *,
        block_id: str,
        block_ts: float,
        session_id: str | None = None,
    ) -> None:
        """Async version of :meth:`add_messages`.

        Persist a list of messages with timestamps to the store asynchronously.

        Args:
            messages: List of (message, timestamp) pairs to store.
            block_id: Unique identifier for the message block.
            block_ts: Unix timestamp for the block start time.
            session_id: Optional session/user ID. Uses block_id if not provided.

        Return:
            None: Messages are persisted to the database.
        """
        if not messages:
            return
        session_id = session_id or block_id
        await super().aadd_messages(
            messages, block_id=block_id, block_ts=block_ts, session_id=session_id
        )
        self._session_caches[session_id].add_messages(block_id, block_ts, messages)

    @override
    def get_session_block_ids_with_timestamps(
        self,
        session_id: str,
        *,
        limit: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        **kwargs: Any,
    ) -> list[tuple[str, float]]:
        """Get block IDs and their start timestamps for a session, optionally filtered by time range.

        Args:
            session_id: The session/user ID to query.
            limit: Maximum number of blocks to return (returns most recent if specified).
            start_time: Optional minimum timestamp (inclusive) to filter blocks.
            end_time: Optional maximum timestamp (inclusive) to filter blocks.

        Return:
            list[tuple[str, float]]: List of (block_id, start_timestamp) pairs in chronological order.
                Returns empty list if session not found or no matching blocks.
        """
        if limit is not None and limit <= 0:
            return []

        start_time = start_time or 0.0
        end_time = end_time or float("inf")
        get_blocks_from_db = partial(
            super().get_session_block_ids_with_timestamps, session_id=session_id
        )

        result: list[tuple[str, float]] = []
        if start_time >= self._session_caches[session_id].start_block_ts:
            result = self._session_caches[session_id].get_blocks(
                start_time=start_time, end_time=end_time
            )
        elif end_time >= self._session_caches[session_id].start_block_ts:
            result = self._session_caches[session_id].get_blocks(
                start_time=self._session_caches[session_id].start_block_ts,
                end_time=end_time,
            )

            if limit is None or len(result) < limit:
                result = (
                    get_blocks_from_db(
                        limit=limit - len(result) if limit else None,
                        start_time=start_time,
                        end_time=self._session_caches[session_id].start_block_ts,
                    )
                    + result
                )
        if result:
            result = list(dict.fromkeys(sorted(result, key=lambda x: x[1])))
            if limit is not None:
                result = result[:limit] if kwargs.get("from_first", False) else result[-limit:]
            return result

        return get_blocks_from_db(
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            **kwargs,
        )

    @override
    async def aget_session_block_ids_with_timestamps(
        self,
        session_id: str,
        *,
        limit: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        **kwargs: Any,
    ) -> list[tuple[str, float]]:
        """Async version of get_session_block_ids_with_timestamps.

        Args:
            session_id: The session/user ID to query.
            limit: Maximum number of blocks to return (returns most recent if specified).
            start_time: Optional minimum timestamp (inclusive) to filter blocks.
            end_time: Optional maximum timestamp (inclusive) to filter blocks.

        Return:
            list[tuple[str, float]]: List of (block_id, start_timestamp) pairs in chronological order.
                Returns empty list if session not found or no matching blocks.
        """
        if limit is not None and limit <= 0:
            return []

        start_time = start_time or 0.0
        end_time = end_time or float("inf")
        aget_blocks_from_db = partial(
            super().aget_session_block_ids_with_timestamps, session_id=session_id
        )

        result: list[tuple[str, float]] = []
        if start_time >= self._session_caches[session_id].start_block_ts:
            result = self._session_caches[session_id].get_blocks(
                start_time=start_time, end_time=end_time
            )
        elif end_time >= self._session_caches[session_id].start_block_ts:
            result = self._session_caches[session_id].get_blocks(
                start_time=self._session_caches[session_id].start_block_ts,
                end_time=end_time,
            )
            if limit is None or len(result) < limit:
                result = (
                    await aget_blocks_from_db(
                        limit=limit - len(result) if limit else None,
                        start_time=start_time,
                        end_time=self._session_caches[session_id].start_block_ts,
                    )
                    + result
                )
        if result:
            result = list(dict.fromkeys(sorted(result, key=lambda x: x[1])))
            if limit is not None:
                result = result[:limit] if kwargs.get("from_first", False) else result[-limit:]
            return result

        return await aget_blocks_from_db(
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            **kwargs,
        )

    @override
    def get_session_history_with_timestamps(
        self,
        session_id: str,
        *,
        limit: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        **kwargs: Any,
    ) -> list[tuple[BaseMessage, float]]:
        """Get all messages with timestamps for a session, optionally filtered by time range.

        Args:
            session_id: The session/user ID to query.
            limit: Maximum number of messages to return (returns most recent if specified).
            start_time: Optional minimum timestamp (inclusive) to filter messages.
            end_time: Optional maximum timestamp (inclusive) to filter messages.

        Return:
            list[tuple[BaseMessage, float]]: List of (message, timestamp) pairs in chronological order.
                Returns empty list if session not found or no matching messages.
        """
        if limit is not None and limit <= 0:
            return []

        start_time = start_time or 0.0
        end_time = end_time or float("inf")
        get_history_from_db = partial(
            super().get_session_history_with_timestamps, session_id=session_id
        )

        result: list[tuple[BaseMessage, float]] = []
        if start_time >= self._session_caches[session_id].start_msg_ts:
            result = self._session_caches[session_id].get_messages(
                start_time=start_time, end_time=end_time
            )
        elif end_time >= self._session_caches[session_id].start_msg_ts:
            result = self._session_caches[session_id].get_messages(
                start_time=self._session_caches[session_id].start_msg_ts,
                end_time=end_time,
            )
            if limit is None or len(result) < limit:
                result = (
                    get_history_from_db(
                        limit=limit - len(result) if limit else None,
                        start_time=start_time,
                        end_time=self._session_caches[session_id].start_msg_ts,
                    )
                    + result
                )
        if result:
            result = [next(group) for _, group in groupby(sorted(result, key=lambda x: x[1]))]
            if limit is not None:
                result = result[:limit] if kwargs.get("from_first", False) else result[-limit:]
            return result

        return get_history_from_db(
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            **kwargs,
        )

    @override
    async def aget_session_history_with_timestamps(
        self,
        session_id: str,
        *,
        limit: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        **kwargs: Any,
    ) -> list[tuple[BaseMessage, float]]:
        """Async version of get_session_history_with_timestamps.

        Args:
            session_id: The session/user ID to query.
            limit: Maximum number of messages to return (returns most recent if specified).
            start_time: Optional minimum timestamp (inclusive) to filter messages.
            end_time: Optional maximum timestamp (inclusive) to filter messages.

        Return:
            list[tuple[BaseMessage, float]]: List of (message, timestamp) pairs in chronological order.
                Returns empty list if session not found or no matching messages.
        """
        if limit is not None and limit <= 0:
            return []

        start_time = start_time or 0.0
        end_time = end_time or float("inf")
        aget_history_from_db = partial(
            super().aget_session_history_with_timestamps, session_id=session_id
        )

        result: list[tuple[BaseMessage, float]] = []
        if start_time >= self._session_caches[session_id].start_msg_ts:
            result = self._session_caches[session_id].get_messages(
                start_time=start_time, end_time=end_time
            )
        elif end_time >= self._session_caches[session_id].start_msg_ts:
            result = self._session_caches[session_id].get_messages(
                start_time=self._session_caches[session_id].start_msg_ts,
                end_time=end_time,
            )
            if limit is None or len(result) < limit:
                result = (
                    await aget_history_from_db(
                        limit=limit - len(result) if limit else None,
                        start_time=start_time,
                        end_time=self._session_caches[session_id].start_msg_ts,
                    )
                    + result
                )
        if result:
            result = [next(group) for _, group in groupby(sorted(result, key=lambda x: x[1]))]
            if limit is not None:
                result = result[:limit] if kwargs.get("from_first", False) else result[-limit:]
            return result

        return await aget_history_from_db(
            limit=limit,
            start_time=start_time,
            end_time=end_time,
            **kwargs,
        )
