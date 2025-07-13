from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal
from typing_extensions import override

from langchain_core.runnables import RunnableSerializable
from pydantic import ConfigDict, Field, model_validator

from cogniweave.core.memory_maker.long_memory import LongTermMemoryMaker
from cogniweave.core.memory_maker.short_memory import ShortTermMemoryMaker
from cogniweave.history_stores import BaseHistoryStore  # noqa: TC001
from cogniweave.vector_stores import TagsVectorStore  # noqa: TC001

if TYPE_CHECKING:
    from langchain_core.runnables.config import RunnableConfig


class SummaryMemoryMaker(RunnableSerializable[dict[str, Any], None]):
    """Manage generation and storage of short and long memories."""

    lang: Literal["zh", "en"] = Field(default="zh")

    history_store: BaseHistoryStore
    vector_store: TagsVectorStore[str]

    model_config = ConfigDict(arbitrary_types_allowed=True)

    short_maker: ShortTermMemoryMaker | None = None
    long_maker: LongTermMemoryMaker | None = None

    @model_validator(mode="after")
    def _build_makers(self) -> SummaryMemoryMaker:
        self.short_maker = self.short_maker or ShortTermMemoryMaker(lang=self.lang)
        self.long_maker = self.long_maker or LongTermMemoryMaker(lang=self.lang)
        return self

    def _get_recent_block_ids(self, session_id: str) -> list[str]:
        return self.history_store.get_session_block_ids(session_id, limit=4)

    @override
    def invoke(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> None:
        session_id = input.get("session_id")
        if not isinstance(session_id, str):
            raise TypeError(f"session_id should be str, got {type(session_id)}")

        block_ids = self._get_recent_block_ids(session_id)
        if not block_ids:
            return

        # short memory on the second last block
        if len(block_ids) >= 2:  # noqa: PLR2004
            short_block_id = block_ids[-2]
            if self.history_store.get_short_memory(short_block_id) is None:
                history = self.history_store.get_block_history(short_block_id)
                ts = self.history_store.get_block_timestamp(short_block_id) or 0.0
                user_name = self.history_store.get_user_name(session_id) or ""
                assert self.short_maker is not None
                short_mem = self.short_maker.invoke(
                    {"history": history, "timestamp": ts, "name": user_name},
                    config=config,
                    **kwargs,
                )
                self.history_store.add_short_memory(
                    short_mem,
                    block_id=short_block_id,
                    session_id=session_id,
                )
                self.vector_store.add_tags(
                    short_mem.topic_tags,
                    content=short_block_id,
                    metadata={"session_id": session_id},
                )

        # long memory update using first three block ids
        if len(block_ids) >= 3:  # noqa: PLR2004
            long_ids = block_ids[:-1]
            existing = self.history_store.get_long_memory(session_id)
            if existing is None or existing.updated_block_id not in long_ids:
                history = self.history_store.get_block_histories(long_ids)
                ts = self.history_store.get_block_timestamp(long_ids[-1]) or 0.0
                assert self.long_maker is not None
                long_mem = self.long_maker.invoke(
                    {
                        "history": history,
                        "current_memory_template": existing,
                        "current_block_id": long_ids[-1],
                        "timestamp": ts,
                    },
                    config=config,
                    **kwargs,
                )
                self.history_store.add_long_memory(long_mem, session_id=session_id)

        return

    async def _aget_recent_block_ids(self, session_id: str) -> list[str]:
        return await self.history_store.aget_session_block_ids(session_id, limit=4)

    @override
    async def ainvoke(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> None:
        session_id = input.get("session_id")
        if not isinstance(session_id, str):
            raise TypeError(f"session_id should be str, got {type(session_id)}")

        block_ids = await self._aget_recent_block_ids(session_id)
        if not block_ids:
            return

        if len(block_ids) >= 2:  # noqa: PLR2004
            short_block_id = block_ids[-2]
            if await self.history_store.aget_short_memory(short_block_id) is None:
                history = await self.history_store.aget_block_history(short_block_id)
                ts = await self.history_store.aget_block_timestamp(short_block_id) or 0.0
                user_name = await self.history_store.aget_user_name(session_id) or ""
                assert self.short_maker is not None
                short_mem = await self.short_maker.ainvoke(
                    {"history": history, "timestamp": ts, "name": user_name},
                    config=config,
                    **kwargs,
                )
                await self.history_store.aadd_short_memory(
                    short_mem,
                    block_id=short_block_id,
                    session_id=session_id,
                )
                await self.vector_store.aadd_tags(
                    short_mem.topic_tags,
                    content=short_block_id,
                    metadata={"session_id": session_id},
                )

        if len(block_ids) >= 3:  # noqa: PLR2004
            long_ids = block_ids[:-1]
            existing = await self.history_store.aget_long_memory(session_id)
            if existing is None or existing.updated_block_id not in long_ids:
                history = await self.history_store.aget_block_histories(long_ids)
                ts = await self.history_store.aget_block_timestamp(long_ids[-1]) or 0.0
                assert self.long_maker is not None
                long_mem = await self.long_maker.ainvoke(
                    {
                        "history": history,
                        "current_memory_template": existing,
                        "current_block_id": long_ids[-1],
                        "timestamp": ts,
                    },
                    config=config,
                    **kwargs,
                )
                await self.history_store.aadd_long_memory(long_mem, session_id=session_id)

        return
