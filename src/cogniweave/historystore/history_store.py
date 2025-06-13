from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypedDict, cast
from typing_extensions import override

from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from langchain_core.runnables import RunnableSerializable
from pydantic import PrivateAttr
from sqlalchemy import create_engine, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from cogniweave.historystore.models import (
    Base,
    ChatBlock,
    ChatBlockAttribute,
    ChatMessage,
    User,
)

if TYPE_CHECKING:
    from langchain_core.runnables.config import RunnableConfig


class MessageInput(TypedDict):
    """Input for storing a single message."""

    message: BaseMessage
    timestamp: float


class BlockAttributeData(TypedDict, total=False):
    """Structure of a block attribute."""

    type: str
    value: Any


class AttributeInput(TypedDict):
    """Input for storing a block attribute."""

    block_attribute: BlockAttributeData


class BlockAttributeOutput(TypedDict):
    """TypedDict representation of a stored block attribute."""

    id: int
    block_id: int
    type: str
    value: Any | None


class HistoryStore(RunnableSerializable[dict[str, Any] | MessageInput | AttributeInput, None]):
    """Persist chat messages grouped by session."""

    _session_local: sessionmaker[Session] = PrivateAttr()
    _async_session_local: async_sessionmaker[AsyncSession] = PrivateAttr()

    user_key: str = "user"
    message_key: str = "message"
    timestamp_key: str = "timestamp"
    attribute_key: str = "block_attribute"

    def __init__(self, *, db_url: str | None = None, echo: bool = False) -> None:
        """Create a new ``HistoryStore``.

        Args:
            db_url: Database connection string. Defaults to ``os.getenv("CHAT_DB_URL")`` or a
                local SQLite file if unset.
            echo: If ``True``, SQLAlchemy will log all statements.
        """
        url = db_url or os.getenv("CHAT_DB_URL", "sqlite:///optimized_chat_db.sqlite")
        engine = create_engine(url, echo=echo, future=True)
        session_local = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

        async_url = url.replace("sqlite://", "sqlite+aiosqlite://")
        async_engine = create_async_engine(async_url, echo=echo, future=True)
        async_session_local = async_sessionmaker(
            bind=async_engine,
            autoflush=False,
            autocommit=False,
            expire_on_commit=False,
        )

        Base.metadata.create_all(bind=engine)
        super().__init__()
        self._session_local = session_local
        self._async_session_local = async_session_local

    def _get_or_create_user(self, session: Session, name: str) -> User:
        user = session.query(User).filter_by(name=name).first()
        if user is None:
            user = User(name=name)
            session.add(user)
            session.commit()
            session.refresh(user)
        return user

    async def _a_get_or_create_user(self, session: AsyncSession, name: str) -> User:
        result = await session.execute(select(User).filter_by(name=name))
        user = result.scalar_one_or_none()
        if user is None:
            user = User(name=name)
            session.add(user)
            await session.commit()
            await session.refresh(user)
        return user

    def _get_or_create_block(
        self, session: Session, user: User, context_id: str, start_ts: float
    ) -> ChatBlock:
        block = session.query(ChatBlock).filter_by(context_id=context_id).first()
        if block is None:
            block = ChatBlock(
                context_id=context_id,
                session_id=user.id,
                start_time=datetime.fromtimestamp(start_ts, tz=UTC),
            )
            session.add(block)
            session.commit()
            session.refresh(block)
        return block

    async def _a_get_or_create_block(
        self, session: AsyncSession, user: User, context_id: str, start_ts: float
    ) -> ChatBlock:
        result = await session.execute(select(ChatBlock).filter_by(context_id=context_id))
        block = result.scalar_one_or_none()
        if block is None:
            block = ChatBlock(
                context_id=context_id,
                session_id=user.id,
                start_time=datetime.fromtimestamp(start_ts, tz=UTC),
            )
            session.add(block)
            await session.commit()
            await session.refresh(block)
        return block

    @override
    def invoke(
        self,
        input: dict[str, Any] | MessageInput | AttributeInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if not config:
            raise ValueError("config must be provided")
        block_id = config.get("configurable", {}).get("block_id", "")
        block_ts = config.get("configurable", {}).get("block_timestamp")
        session_id = config.get("configurable", {}).get("session_id", block_id)

        if not block_id:
            raise ValueError("block_id is required")
        if not isinstance(block_ts, (int, float)):
            raise TypeError("block_timestamp is required")

        message = input.get(self.message_key)
        timestamp = input.get(self.timestamp_key)
        attribute = input.get(self.attribute_key)

        if message is None and attribute is None:
            raise ValueError("nothing to store")

        if message is not None:
            if not isinstance(message, BaseMessage):
                raise TypeError("message must be a BaseMessage")
            if not isinstance(timestamp, (int, float)):
                raise TypeError("timestamp must be a number")

        if attribute is not None:
            if not isinstance(attribute, dict):
                raise TypeError("block_attribute must be a dict")
            if not isinstance(attribute.get("type"), str):
                raise TypeError("block_attribute.type must be a str")

        context_id = block_id
        start_ts = float(block_ts)

        with self._session_local() as session:
            db_user = self._get_or_create_user(session, session_id)
            block = self._get_or_create_block(session, db_user, context_id, start_ts)
            if message is not None:
                record = ChatMessage(
                    block_id=block.id,
                    timestamp=datetime.fromtimestamp(float(cast("float | int", timestamp)), tz=UTC),
                    content=message_to_dict(message),
                )
                session.add(record)
            if attribute is not None:
                attr = ChatBlockAttribute(
                    block_id=block.id,
                    type=attribute["type"],
                    value=attribute.get("value"),
                )
                session.add(attr)
            session.commit()

    @override
    async def ainvoke(
        self,
        input: dict[str, Any] | MessageInput | AttributeInput,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if not config:
            raise ValueError("config must be provided")
        block_id = config.get("configurable", {}).get("block_id", "")
        block_ts = config.get("configurable", {}).get("block_timestamp")
        session_id = config.get("configurable", {}).get("session_id", block_id)

        if not block_id:
            raise ValueError("block_id is required")
        if not isinstance(block_ts, (int, float)):
            raise TypeError("block_timestamp is required")

        message = input.get(self.message_key)
        timestamp = input.get(self.timestamp_key)
        attribute = input.get(self.attribute_key)

        if message is None and attribute is None:
            raise ValueError("nothing to store")

        if message is not None:
            if not isinstance(message, BaseMessage):
                raise TypeError("message must be a BaseMessage")
            if not isinstance(timestamp, (int, float)):
                raise TypeError("timestamp must be a number")

        if attribute is not None:
            if not isinstance(attribute, dict):
                raise TypeError("block_attribute must be a dict")
            if not isinstance(attribute.get("type"), str):
                raise TypeError("block_attribute.type must be a str")

        context_id = block_id
        start_ts = float(block_ts)

        async with self._async_session_local() as session:
            db_user = await self._a_get_or_create_user(session, session_id)
            block = await self._a_get_or_create_block(session, db_user, context_id, start_ts)
            if message is not None:
                record = ChatMessage(
                    block_id=block.id,
                    timestamp=datetime.fromtimestamp(float(cast("float | int", timestamp)), tz=UTC),
                    content=message_to_dict(message),
                )
                session.add(record)
            if attribute is not None:
                attr = ChatBlockAttribute(
                    block_id=block.id,
                    type=attribute["type"],
                    value=attribute.get("value"),
                )
                session.add(attr)
            await session.commit()

    def get_block_timestamp(self, block_id: str) -> float | None:
        """Return the start timestamp of a chat block."""
        with self._session_local() as session:
            block = session.query(ChatBlock).filter_by(context_id=block_id).first()
            if not block:
                return None
            return block.start_time.replace(tzinfo=UTC).timestamp()

    async def aget_block_timestamp(self, block_id: str) -> float | None:
        """Asynchronously return the start timestamp of a chat block."""
        async with self._async_session_local() as session:
            result = await session.execute(select(ChatBlock).filter_by(context_id=block_id))
            block = result.scalar_one_or_none()
            if not block:
                return None
            return block.start_time.replace(tzinfo=UTC).timestamp()

    def get_history_with_timestamps(self, block_id: str) -> list[tuple[BaseMessage, float]]:
        """Return ordered messages with timestamps for a single block."""
        with self._session_local() as session:
            block = session.query(ChatBlock).filter_by(context_id=block_id).first()
            if not block:
                return []
            return [
                (messages_from_dict([m.content])[0], m.timestamp.replace(tzinfo=UTC).timestamp())
                for m in block.messages
            ]

    async def aget_history_with_timestamps(self, block_id: str) -> list[tuple[BaseMessage, float]]:
        """Asynchronously return ordered messages with timestamps for a single block."""
        async with self._async_session_local() as session:
            result = await session.execute(select(ChatBlock).filter_by(context_id=block_id))
            block = result.scalar_one_or_none()
            if not block:
                return []
            return [
                (messages_from_dict([m.content])[0], m.timestamp.replace(tzinfo=UTC).timestamp())
                for m in block.messages
            ]

    def get_history(self, block_id: str) -> list[BaseMessage]:
        """Return ordered messages for a single block."""
        with self._session_local() as session:
            block = session.query(ChatBlock).filter_by(context_id=block_id).first()
            if not block:
                return []
            return messages_from_dict([m.content for m in block.messages])

    async def aget_history(self, block_id: str) -> list[BaseMessage]:
        """Asynchronously return ordered messages for a single block."""
        async with self._async_session_local() as session:
            result = await session.execute(select(ChatBlock).filter_by(context_id=block_id))
            block = result.scalar_one_or_none()
            if not block:
                return []
            return messages_from_dict([m.content for m in block.messages])

    def get_histories_with_timestamps(
        self, block_ids: list[str]
    ) -> list[tuple[BaseMessage, float]]:
        """Concatenate histories with timestamps for multiple blocks in order."""
        messages: list[tuple[BaseMessage, float]] = []
        for sid in sorted(block_ids):
            messages.extend(self.get_history_with_timestamps(sid))
        return messages

    async def aget_histories_with_timestamps(
        self, block_ids: list[str]
    ) -> list[tuple[BaseMessage, float]]:
        """Asynchronously concatenate histories with timestamps for multiple blocks."""
        messages: list[tuple[BaseMessage, float]] = []
        for sid in sorted(block_ids):
            messages.extend(await self.aget_history_with_timestamps(sid))
        return messages

    def get_histories(self, block_ids: list[str]) -> list[BaseMessage]:
        """Concatenate histories for multiple blocks in order."""
        messages: list[BaseMessage] = []
        for sid in sorted(block_ids):
            messages.extend(self.get_history(sid))
        return messages

    async def aget_histories(self, block_ids: list[str]) -> list[BaseMessage]:
        """Asynchronously concatenate histories for multiple blocks."""
        messages: list[BaseMessage] = []
        for sid in sorted(block_ids):
            messages.extend(await self.aget_history(sid))
        return messages

    def get_block_attributes(
        self, block_id: str, *, types: list[str] | None = None
    ) -> list[BlockAttributeOutput]:
        """Return ordered attributes for a chat block."""
        with self._session_local() as session:
            block = session.query(ChatBlock).filter_by(context_id=block_id).first()
            if not block:
                return []

            attrs = sorted(block.attributes, key=lambda a: a.id)
            if types is not None:
                attrs = [attr for attr in attrs if attr.type in types]
            return [
                BlockAttributeOutput(
                    id=attr.id,
                    block_id=attr.block_id,
                    type=attr.type,
                    value=attr.value,
                )
                for attr in attrs
            ]

    async def aget_block_attributes(
        self, block_id: str, *, types: list[str] | None = None
    ) -> list[BlockAttributeOutput]:
        """Asynchronously return ordered attributes for a chat block."""
        async with self._async_session_local() as session:
            result = await session.execute(select(ChatBlock).filter_by(context_id=block_id))
            block = result.scalar_one_or_none()
            if not block:
                return []

            attrs = sorted(block.attributes, key=lambda a: a.id)
            if types is not None:
                attrs = [attr for attr in attrs if attr.type in types]
            return [
                BlockAttributeOutput(
                    id=attr.id,
                    block_id=attr.block_id,
                    type=attr.type,
                    value=attr.value,
                )
                for attr in attrs
            ]

    def get_session_block_ids_with_timestamps(
        self,
        session_id: str,
        *,
        limit: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[tuple[str, float]]:
        """Return block ids with timestamps for a session within a time range."""
        if limit is not None and limit <= 0:
            return []

        with self._session_local() as session:
            user = session.query(User).filter_by(name=session_id).first()
            if not user:
                return []

            stmt = session.query(ChatBlock).filter_by(session_id=user.id)
            if start_time is not None:
                stmt = stmt.filter(
                    ChatBlock.start_time >= datetime.fromtimestamp(start_time, tz=UTC)
                )
            if end_time is not None:
                stmt = stmt.filter(ChatBlock.start_time <= datetime.fromtimestamp(end_time, tz=UTC))
            blocks = stmt.order_by(ChatBlock.start_time).all()
            result = [
                (
                    block.context_id,
                    block.start_time.replace(tzinfo=UTC).timestamp(),
                )
                for block in blocks
            ]
            if limit is not None:
                result = result[-limit:]
            return result

    async def aget_session_block_ids_with_timestamps(
        self,
        session_id: str,
        *,
        limit: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[tuple[str, float]]:
        """Asynchronously return block ids with timestamps for a session."""
        if limit is not None and limit <= 0:
            return []

        async with self._async_session_local() as session:
            result = await session.execute(select(User).filter_by(name=session_id))
            user = result.scalar_one_or_none()
            if not user:
                return []

            stmt = select(ChatBlock).filter_by(session_id=user.id)
            if start_time is not None:
                stmt = stmt.filter(
                    ChatBlock.start_time >= datetime.fromtimestamp(start_time, tz=UTC)
                )
            if end_time is not None:
                stmt = stmt.filter(ChatBlock.start_time <= datetime.fromtimestamp(end_time, tz=UTC))
            stmt = stmt.order_by(ChatBlock.start_time)
            res = await session.execute(stmt)
            blocks = res.scalars().all()
            result = [
                (
                    block.context_id,
                    block.start_time.replace(tzinfo=UTC).timestamp(),
                )
                for block in blocks
            ]
            if limit is not None:
                result = result[-limit:]
            return result

    def get_session_block_ids(
        self,
        session_id: str,
        *,
        limit: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[str]:
        """Return block ids for a session within a time range."""
        return [
            bid
            for bid, _ in self.get_session_block_ids_with_timestamps(
                session_id, start_time=start_time, end_time=end_time, limit=limit
            )
        ]

    async def aget_session_block_ids(
        self,
        session_id: str,
        *,
        limit: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[str]:
        """Asynchronously return block ids for a session within a time range."""
        pairs = await self.aget_session_block_ids_with_timestamps(
            session_id, start_time=start_time, end_time=end_time, limit=limit
        )
        return [bid for bid, _ in pairs]

    def get_session_history_with_timestamps(
        self,
        session_id: str,
        *,
        limit: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[tuple[BaseMessage, float]]:
        """Return concatenated messages with timestamps for a session."""
        if limit is not None and limit <= 0:
            return []

        all_blocks = self.get_session_block_ids_with_timestamps(session_id)
        if not all_blocks:
            return []

        start_idx = 0
        end_idx = len(all_blocks)

        if start_time is not None:
            for i, (_, ts) in enumerate(all_blocks):
                if ts >= start_time:
                    start_idx = max(i - 1, 0)
                    break
            else:
                start_idx = len(all_blocks) - 1

        if end_time is not None:
            for i, (_, ts) in enumerate(all_blocks):
                if ts > end_time:
                    end_idx = min(i + 1, len(all_blocks))
                    break

        block_ids = [bid for bid, _ in all_blocks[start_idx:end_idx]]
        history = self.get_histories_with_timestamps(block_ids)
        result = [
            (msg, ts)
            for msg, ts in history
            if (start_time is None or ts >= start_time) and (end_time is None or ts <= end_time)
        ]
        if limit is not None:
            result = result[-limit:]
        return result

    async def aget_session_history_with_timestamps(
        self,
        session_id: str,
        *,
        limit: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[tuple[BaseMessage, float]]:
        """Asynchronously return concatenated messages with timestamps."""
        if limit is not None and limit <= 0:
            return []

        all_blocks = await self.aget_session_block_ids_with_timestamps(session_id)
        if not all_blocks:
            return []

        start_idx = 0
        end_idx = len(all_blocks)

        if start_time is not None:
            for i, (_, ts) in enumerate(all_blocks):
                if ts >= start_time:
                    start_idx = max(i - 1, 0)
                    break
            else:
                start_idx = len(all_blocks) - 1

        if end_time is not None:
            for i, (_, ts) in enumerate(all_blocks):
                if ts > end_time:
                    end_idx = min(i + 1, len(all_blocks))
                    break

        block_ids = [bid for bid, _ in all_blocks[start_idx:end_idx]]
        history = await self.aget_histories_with_timestamps(block_ids)
        result = [
            (msg, ts)
            for msg, ts in history
            if (start_time is None or ts >= start_time) and (end_time is None or ts <= end_time)
        ]
        if limit is not None:
            result = result[-limit:]
        return result

    def get_session_history(
        self,
        session_id: str,
        *,
        limit: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[BaseMessage]:
        """Return concatenated messages for a session within a time range."""
        return [
            msg
            for msg, _ in self.get_session_history_with_timestamps(
                session_id, start_time=start_time, end_time=end_time, limit=limit
            )
        ]

    async def aget_session_history(
        self,
        session_id: str,
        *,
        limit: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
    ) -> list[BaseMessage]:
        """Asynchronously return concatenated messages for a session."""
        pairs = await self.aget_session_history_with_timestamps(
            session_id, start_time=start_time, end_time=end_time, limit=limit
        )
        return [msg for msg, _ in pairs]
