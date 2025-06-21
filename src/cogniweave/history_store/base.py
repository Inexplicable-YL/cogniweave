from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, TypedDict

from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from pydantic import BaseModel, PrivateAttr
from sqlalchemy import create_engine, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from cogniweave.history_store.models import (
    Base,
    ChatBlock,
    ChatBlockAttribute,
    ChatMessage,
    User,
    UserAttribute,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from sqlalchemy.sql._typing import _ColumnExpressionArgument


class BlockAttributeData(TypedDict):
    """Structure of a block attribute.

    Attributes:
        type: The type/name of the attribute.
        value: The attribute's value (optional).
    """

    type: str
    value: Any


class UserAttributeData(TypedDict):
    """Structure of a user attribute."""

    type: str
    value: Any


class BaseHistoryStore(BaseModel):
    """Persist chat messages grouped by session.

    This class provides both synchronous and asynchronous interfaces for storing and retrieving
    chat messages and their metadata. Messages are grouped into blocks which can have additional
    attributes.
    """

    # persist chat messages grouped by session
    _session_local: sessionmaker[Session] = PrivateAttr()
    _async_session_local: async_sessionmaker[AsyncSession] = PrivateAttr()

    def __init__(self, *, db_url: str | None = None, echo: bool = False, **kwargs: Any) -> None:
        """Initialize a new HistoryStore instance.

        Args:
            db_url: Database connection string. If None, uses CHAT_DB_URL environment variable
                or defaults to local SQLite file.
            echo: If True, enables SQLAlchemy statement logging.

        Raises:
            ValueError: If database connection fails.
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
        super().__init__(**kwargs)
        self._session_local = session_local
        self._async_session_local = async_session_local

    def _get_or_create_user(self, session: Session, name: str) -> User:
        """Get existing user or create new one if not found.

        Args:
            session: SQLAlchemy session.
            name: User/session name.

        Return:
            User: The existing or newly created User instance.
        """
        user = session.query(User).filter_by(name=name).first()
        if user is None:
            user = User(name=name)
            session.add(user)
            session.flush()
        return user

    async def _a_get_or_create_user(self, session: AsyncSession, name: str) -> User:
        """Async version of _get_or_create_user.

        Args:
            session: Async SQLAlchemy session.
            name: User/session name.

        Return:
            User: The existing or newly created User instance.
        """
        result = await session.execute(select(User).filter_by(name=name))
        user = result.scalar_one_or_none()
        if user is None:
            user = User(name=name)
            session.add(user)
            await session.flush()
        return user

    def _get_or_create_block(
        self, session: Session, user: User, context_id: str, start_ts: float
    ) -> ChatBlock:
        """Get existing chat block or create new one if not found.

        Args:
            session: SQLAlchemy session.
            user: Owner User instance.
            context_id: Unique block/context ID.
            start_ts: Unix timestamp for block start time.

        Return:
            ChatBlock: The existing or newly created ChatBlock instance.
        """
        block = session.query(ChatBlock).filter_by(context_id=context_id).first()
        if block is None:
            block = ChatBlock(
                context_id=context_id,
                session_id=user.id,
                timestamp=datetime.fromtimestamp(start_ts, tz=UTC),
            )
            session.add(block)
            session.flush()
        return block

    async def _a_get_or_create_block(
        self, session: AsyncSession, user: User, context_id: str, start_ts: float
    ) -> ChatBlock:
        """Async version of _get_or_create_block.

        Args:
            session: Async SQLAlchemy session.
            user: Owner User instance.
            context_id: Unique block/context ID.
            start_ts: Unix timestamp for block start time.

        Return:
            ChatBlock: The existing or newly created ChatBlock instance.
        """
        result = await session.execute(select(ChatBlock).filter_by(context_id=context_id))
        block = result.scalar_one_or_none()
        if block is None:
            block = ChatBlock(
                context_id=context_id,
                session_id=user.id,
                timestamp=datetime.fromtimestamp(start_ts, tz=UTC),
            )
            session.add(block)
            await session.flush()
        return block

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

        context_id = block_id
        start_ts = float(block_ts)
        sid = session_id or block_id

        with self._session_local() as session:
            try:
                db_user = self._get_or_create_user(session, sid)
                block = self._get_or_create_block(session, db_user, context_id, start_ts)

                records = [
                    ChatMessage(
                        block_id=block.id,
                        timestamp=datetime.fromtimestamp(float(ts), tz=UTC),
                        content=message_to_dict(msg),
                    )
                    for msg, ts in messages
                ]

                session.add_all(records)
                session.commit()
            except Exception:
                session.rollback()
                raise

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

        context_id = block_id
        start_ts = float(block_ts)
        sid = session_id or block_id

        async with self._async_session_local() as session:
            try:
                db_user = await self._a_get_or_create_user(session, sid)
                block = await self._a_get_or_create_block(session, db_user, context_id, start_ts)

                records = [
                    ChatMessage(
                        block_id=block.id,
                        timestamp=datetime.fromtimestamp(float(ts), tz=UTC),
                        content=message_to_dict(msg),
                    )
                    for msg, ts in messages
                ]

                session.add_all(records)
                await session.commit()

            except Exception:
                await session.rollback()
                raise

    def add_block_attributes(
        self,
        attributes: list[BlockAttributeData],
        *,
        block_id: str,
        block_ts: float,
        session_id: str | None = None,
    ) -> None:
        """Persist a list of block attributes to the store.

        Args:
            attributes: List of attribute dictionaries containing 'type' and optional 'value'.
            block_id: Unique identifier for the attribute block.
            block_ts: Unix timestamp for the block start time.
            session_id: Optional session/user ID. Uses block_id if not provided.

        Return:
            None: Attributes are persisted to the database.
        """

        if not attributes:
            return

        context_id = block_id
        start_ts = float(block_ts)
        sid = session_id or block_id

        with self._session_local() as session:
            try:
                db_user = self._get_or_create_user(session, sid)
                block = self._get_or_create_block(session, db_user, context_id, start_ts)

                attr_recs = [
                    ChatBlockAttribute(
                        block_id=block.id,
                        type=attr["type"],
                        value=attr.get("value"),
                    )
                    for attr in attributes
                ]

                session.add_all(attr_recs)
                session.commit()
            except Exception:
                session.rollback()
                raise

    async def aadd_block_attributes(
        self,
        attributes: list[BlockAttributeData],
        *,
        block_id: str,
        block_ts: float,
        session_id: str | None = None,
    ) -> None:
        """Async version of :meth:`add_attributes`.

        Persist a list of block attributes to the store asynchronously.

        Args:
            attributes: List of attribute dictionaries containing 'type' and optional 'value'.
            block_id: Unique identifier for the attribute block.
            block_ts: Unix timestamp for the block start time.
            session_id: Optional session/user ID. Uses block_id if not provided.

        Return:
            None: Attributes are persisted to the database.
        """

        if not attributes:
            return

        context_id = block_id
        start_ts = float(block_ts)
        sid = session_id or block_id

        async with self._async_session_local() as session:
            try:
                db_user = await self._a_get_or_create_user(session, sid)
                block = await self._a_get_or_create_block(session, db_user, context_id, start_ts)

                attr_recs = [
                    ChatBlockAttribute(
                        block_id=block.id,
                        type=attr["type"],
                        value=attr.get("value"),
                    )
                    for attr in attributes
                ]

                session.add_all(attr_recs)
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    def add_user_attributes(
        self,
        attributes: list[UserAttributeData],
        *,
        session_id: str,
    ) -> None:
        """Persist user-level attributes, replacing existing ones of the same type."""

        if not attributes:
            return

        with self._session_local() as session:
            try:
                user = self._get_or_create_user(session, session_id)
                for attr in attributes:
                    rec = (
                        session.query(UserAttribute)
                        .filter_by(user_id=user.id, type=attr["type"])
                        .first()
                    )
                    if rec is None:
                        rec = UserAttribute(
                            user_id=user.id,
                            type=attr["type"],
                            value=attr.get("value"),
                        )
                        session.add(rec)
                    else:
                        rec.value = attr.get("value")
                session.commit()
            except Exception:
                session.rollback()
                raise

    async def aadd_user_attributes(
        self,
        attributes: list[UserAttributeData],
        *,
        session_id: str,
    ) -> None:
        """Async version of :meth:`add_user_attributes`."""

        if not attributes:
            return

        async with self._async_session_local() as session:
            try:
                user = await self._a_get_or_create_user(session, session_id)
                for attr in attributes:
                    result = await session.execute(
                        select(UserAttribute).filter_by(user_id=user.id, type=attr["type"])
                    )
                    rec = result.scalar_one_or_none()
                    if rec is None:
                        rec = UserAttribute(
                            user_id=user.id,
                            type=attr["type"],
                            value=attr.get("value"),
                        )
                        session.add(rec)
                    else:
                        rec.value = attr.get("value")
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    def get_block_timestamp(self, block_id: str) -> float | None:
        """Get the start timestamp of a chat block.

        Args:
            block_id: The ID of the chat block to query.

        Return:
            float | None: Unix timestamp of block start time, or None if not found.
        """
        with self._session_local() as session:
            block = session.query(ChatBlock).filter_by(context_id=block_id).first()
            if not block:
                return None
            return block.timestamp.replace(tzinfo=UTC).timestamp()

    async def aget_block_timestamp(self, block_id: str) -> float | None:
        """Async version of get_block_timestamp.

        Args:
            block_id: The ID of the chat block to query.

        Return:
            float | None: Unix timestamp of block start time, or None if not found.
        """
        async with self._async_session_local() as session:
            result = await session.execute(select(ChatBlock).filter_by(context_id=block_id))
            block = result.scalar_one_or_none()
            if not block:
                return None
            return block.timestamp.replace(tzinfo=UTC).timestamp()

    def get_block_history_with_timestamps(self, block_id: str) -> list[tuple[BaseMessage, float]]:
        """Get all messages in a block with their timestamps.

        Args:
            block_id: The ID of the chat block to query.

        Return:
            list[tuple[BaseMessage, float]]: List of (message, timestamp) pairs in chronological order.
        """
        with self._session_local() as session:
            block = session.query(ChatBlock).filter_by(context_id=block_id).first()
            if not block:
                return []
            return [
                (messages_from_dict([m.content])[0], m.timestamp.replace(tzinfo=UTC).timestamp())
                for m in block.messages
            ]

    async def aget_block_history_with_timestamps(
        self, block_id: str
    ) -> list[tuple[BaseMessage, float]]:
        """Async version of get_history_with_timestamps.

        Args:
            block_id: The ID of the chat block to query.

        Return:
            list[tuple[BaseMessage, float]]: List of (message, timestamp) pairs in chronological order.
        """
        async with self._async_session_local() as session:
            result = await session.execute(select(ChatBlock).filter_by(context_id=block_id))
            block = result.scalar_one_or_none()
            if not block:
                return []
            return [
                (messages_from_dict([m.content])[0], m.timestamp.replace(tzinfo=UTC).timestamp())
                for m in block.messages
            ]

    def get_block_history(self, block_id: str) -> list[BaseMessage]:
        """Get all messages in a block without timestamps.

        Args:
            block_id: The ID of the chat block to query.

        Return:
            list[BaseMessage]: List of messages in chronological order.
        """
        return [m for m, _ in self.get_block_history_with_timestamps(block_id)]

    async def aget_block_history(self, block_id: str) -> list[BaseMessage]:
        """Async version of get_history.

        Args:
            block_id: The ID of the chat block to query.

        Return:
            list[BaseMessage]: List of messages in chronological order.
        """
        return [m for m, _ in await self.aget_block_history_with_timestamps(block_id)]

    def _query_messages(
        self,
        session: Session,
        block_ids: list[str],
        *,
        limit: int | None = None,
        criteria: Sequence[_ColumnExpressionArgument[bool]] | None = None,
        **kwargs: Any,
    ) -> list[tuple[BaseMessage, float]]:
        """Return messages for multiple blocks ordered by timestamp."""

        if not block_ids or (limit is not None and limit <= 0):
            return []
        criteria = criteria or []

        stmt = (
            select(ChatMessage)
            .join(ChatBlock, ChatMessage.block_id == ChatBlock.id)
            .filter(ChatBlock.context_id.in_(block_ids))
            .filter(*criteria)
        )
        if limit is not None:
            if kwargs.get("from_first", False):
                stmt = stmt.order_by(ChatMessage.timestamp).limit(limit)
                result = list(session.scalars(stmt).all())
            else:
                stmt = stmt.order_by(ChatMessage.timestamp.desc()).limit(limit)
                result = list(reversed(session.scalars(stmt).all()))
        else:
            stmt = stmt.order_by(ChatMessage.timestamp)
            result = list(session.scalars(stmt).all())
        return [
            (
                messages_from_dict([rec.content])[0],
                rec.timestamp.replace(tzinfo=UTC).timestamp(),
            )
            for rec in result
        ]

    async def _a_query_messages(
        self,
        session: AsyncSession,
        block_ids: list[str],
        *,
        limit: int | None = None,
        criteria: Sequence[_ColumnExpressionArgument[bool]] | None = None,
        **kwargs: Any,
    ) -> list[tuple[BaseMessage, float]]:
        """Async version of ``_query_messages``."""

        if not block_ids or (limit is not None and limit <= 0):
            return []
        criteria = criteria or []

        stmt = (
            select(ChatMessage)
            .join(ChatBlock, ChatMessage.block_id == ChatBlock.id)
            .filter(ChatBlock.context_id.in_(block_ids))
            .filter(*criteria)
        )
        if limit is not None:
            if kwargs.get("from_first", False):
                stmt = stmt.order_by(ChatMessage.timestamp).limit(limit)
                rec = await session.execute(stmt)
                result = list(rec.scalars().all())
            else:
                stmt = stmt.order_by(ChatMessage.timestamp.desc()).limit(limit)
                rec = await session.execute(stmt)
                result = list(reversed(rec.scalars().all()))
        else:
            stmt = stmt.order_by(ChatMessage.timestamp)
            rec = await session.execute(stmt)
            result = list(rec.scalars().all())
        return [
            (
                messages_from_dict([rec.content])[0],
                rec.timestamp.replace(tzinfo=UTC).timestamp(),
            )
            for rec in result
        ]

    def get_block_histories_with_timestamps(
        self, block_ids: list[str]
    ) -> list[tuple[BaseMessage, float]]:
        """Get messages with timestamps from multiple blocks, concatenated in order.

        Args:
            block_ids: List of block IDs to retrieve messages from.

        Return:
            list[tuple[BaseMessage, float]]: Combined list of (message, timestamp) pairs
                from all blocks, in chronological order.
        """
        with self._session_local() as session:
            return self._query_messages(session, block_ids)

    async def aget_block_histories_with_timestamps(
        self, block_ids: list[str]
    ) -> list[tuple[BaseMessage, float]]:
        """Async version of get_histories_with_timestamps.

        Args:
            block_ids: List of block IDs to retrieve messages from.

        Return:
            list[tuple[BaseMessage, float]]: Combined list of (message, timestamp) pairs
                from all blocks, in chronological order.
        """
        async with self._async_session_local() as session:
            return await self._a_query_messages(session, block_ids)

    def get_block_histories(self, block_ids: list[str]) -> list[BaseMessage]:
        """Get messages from multiple blocks, concatenated in order.

        Args:
            block_ids: List of block IDs to retrieve messages from.

        Return:
            list[BaseMessage]: Combined list of messages from all blocks,
                in chronological order.
        """
        return [msg for msg, _ in self.get_block_histories_with_timestamps(block_ids)]

    async def aget_block_histories(self, block_ids: list[str]) -> list[BaseMessage]:
        """Async version of get_histories.

        Args:
            block_ids: List of block IDs to retrieve messages from.

        Return:
            list[BaseMessage]: Combined list of messages from all blocks,
                in chronological order.
        """
        pairs = await self.aget_block_histories_with_timestamps(block_ids)
        return [msg for msg, _ in pairs]

    def get_block_attributes(
        self, block_id: str, *, types: list[str] | None = None
    ) -> list[BlockAttributeData]:
        """Get all attributes for a chat block, optionally filtered by type.

        Args:
            block_id: The ID of the chat block to query.
            types: Optional list of attribute types to filter by.

        Return:
            list[BlockAttributeData]: List of block attributes in insertion order,
                optionally filtered by type.
        """
        with self._session_local() as session:
            block = session.query(ChatBlock).filter_by(context_id=block_id).first()
            if not block:
                return []

            attrs = sorted(block.attributes, key=lambda a: a.id)
            if types is not None:
                attrs = [attr for attr in attrs if attr.type in types]
            return [
                BlockAttributeData(
                    type=attr.type,
                    value=attr.value,
                )
                for attr in attrs
            ]

    async def aget_block_attributes(
        self, block_id: str, *, types: list[str] | None = None
    ) -> list[BlockAttributeData]:
        """Async version of get_block_attributes.

        Args:
            block_id: The ID of the chat block to query.
            types: Optional list of attribute types to filter by.

        Return:
            list[BlockAttributeData]: List of block attributes in insertion order,
                optionally filtered by type.
        """
        async with self._async_session_local() as session:
            result = await session.execute(select(ChatBlock).filter_by(context_id=block_id))
            block = result.scalar_one_or_none()
            if not block:
                return []

            attrs = sorted(block.attributes, key=lambda a: a.id)
            if types is not None:
                attrs = [attr for attr in attrs if attr.type in types]
            return [
                BlockAttributeData(
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

        start_date_time = get_datetime_from_timestamp(start_time)
        end_date_time = get_datetime_from_timestamp(end_time)
        with self._session_local() as session:
            user = session.query(User).filter_by(name=session_id).first()
            if not user:
                return []

            stmt = session.query(ChatBlock).filter_by(session_id=user.id)
            if start_date_time is not None:
                stmt = stmt.filter(ChatBlock.timestamp >= start_date_time)
            if end_date_time is not None:
                stmt = stmt.filter(ChatBlock.timestamp <= end_date_time)
            if limit is not None:
                if kwargs.get("from_first", False):
                    # if from_first is True, order by ascending timestamp
                    stmt = stmt.order_by(ChatBlock.timestamp).limit(limit)
                    blocks = stmt.all()
                else:
                    # default behavior: order by descending timestamp
                    stmt = stmt.order_by(ChatBlock.timestamp.desc()).limit(limit)
                    blocks = list(reversed(stmt.all()))
            else:
                blocks = stmt.order_by(ChatBlock.timestamp).all()
            return [
                (
                    block.context_id,
                    block.timestamp.replace(tzinfo=UTC).timestamp(),
                )
                for block in blocks
            ]

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

        start_date_time = get_datetime_from_timestamp(start_time)
        end_date_time = get_datetime_from_timestamp(end_time)
        async with self._async_session_local() as session:
            result = await session.execute(select(User).filter_by(name=session_id))
            user = result.scalar_one_or_none()
            if not user:
                return []

            stmt = select(ChatBlock).filter_by(session_id=user.id)
            if start_date_time is not None:
                stmt = stmt.filter(ChatBlock.timestamp >= start_date_time)
            if end_date_time is not None:
                stmt = stmt.filter(ChatBlock.timestamp <= end_date_time)
            if limit is not None:
                if kwargs.get("from_first", False):
                    # if from_first is True, order by ascending timestamp
                    stmt = stmt.order_by(ChatBlock.timestamp).limit(limit)
                    res = await session.execute(stmt)
                    blocks = res.scalars().all()
                else:
                    # default behavior: order by descending timestamp
                    stmt = stmt.order_by(ChatBlock.timestamp.desc()).limit(limit)
                    res = await session.execute(stmt)
                    blocks = list(reversed(res.scalars().all()))
            else:
                stmt = stmt.order_by(ChatBlock.timestamp)
                res = await session.execute(stmt)
                blocks = res.scalars().all()
            return [
                (
                    block.context_id,
                    block.timestamp.replace(tzinfo=UTC).timestamp(),
                )
                for block in blocks
            ]

    def get_session_block_ids(
        self,
        session_id: str,
        *,
        limit: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Get block IDs for a session, optionally filtered by time range.

        Args:
            session_id: The session/user ID to query.
            limit: Maximum number of blocks to return (returns most recent if specified).
            start_time: Optional minimum timestamp (inclusive) to filter blocks.
            end_time: Optional maximum timestamp (inclusive) to filter blocks.

        Return:
            list[str]: List of block IDs in chronological order.
                Returns empty list if session not found or no matching blocks.
        """
        return [
            bid
            for bid, _ in self.get_session_block_ids_with_timestamps(
                session_id, start_time=start_time, end_time=end_time, limit=limit, **kwargs
            )
        ]

    async def aget_session_block_ids(
        self,
        session_id: str,
        *,
        limit: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Async version of get_session_block_ids.

        Args:
            session_id: The session/user ID to query.
            limit: Maximum number of blocks to return (returns most recent if specified).
            start_time: Optional minimum timestamp (inclusive) to filter blocks.
            end_time: Optional maximum timestamp (inclusive) to filter blocks.

        Return:
            list[str]: List of block IDs in chronological order.
                Returns empty list if session not found or no matching blocks.
        """
        pairs = await self.aget_session_block_ids_with_timestamps(
            session_id, start_time=start_time, end_time=end_time, limit=limit, **kwargs
        )
        return [bid for bid, _ in pairs]

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

        block_limit = limit if start_time is None and end_time is None else None
        all_blocks = self.get_session_block_ids_with_timestamps(
            session_id, limit=block_limit, start_time=start_time, end_time=end_time
        )
        if not all_blocks:
            return []

        if start_time is not None:
            all_blocks = (
                self.get_session_block_ids_with_timestamps(session_id, limit=2, end_time=start_time)
                + all_blocks
            )
        if end_time is not None:
            all_blocks = all_blocks + self.get_session_block_ids_with_timestamps(
                session_id, limit=2, start_time=end_time, from_first=True
            )

        block_ids = [bid for bid, _ in dict.fromkeys(all_blocks)]
        with self._session_local() as session:
            return self._query_messages(
                session,
                block_ids,
                limit=limit,
                criteria=(
                    [ChatMessage.timestamp >= get_datetime_from_timestamp(start_time)]
                    if start_time
                    else []
                )
                + (
                    [ChatMessage.timestamp <= get_datetime_from_timestamp(end_time)]
                    if end_time
                    else []
                ),
                from_first=kwargs.get("from_first", False),
            )

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

        block_limit = limit if start_time is None and end_time is None else None
        all_blocks = await self.aget_session_block_ids_with_timestamps(
            session_id, limit=block_limit, start_time=start_time, end_time=end_time
        )
        if not all_blocks:
            return []

        if start_time is not None:
            all_blocks = (
                await self.aget_session_block_ids_with_timestamps(
                    session_id, limit=2, end_time=start_time
                )
                + all_blocks
            )
        if end_time is not None:
            all_blocks = all_blocks + await self.aget_session_block_ids_with_timestamps(
                session_id, limit=2, start_time=end_time, from_first=True
            )

        block_ids = [bid for bid, _ in dict.fromkeys(all_blocks)]
        async with self._async_session_local() as session:
            return await self._a_query_messages(
                session,
                block_ids,
                limit=limit,
                criteria=(
                    [ChatMessage.timestamp >= get_datetime_from_timestamp(start_time)]
                    if start_time
                    else []
                )
                + (
                    [ChatMessage.timestamp <= get_datetime_from_timestamp(end_time)]
                    if end_time
                    else []
                ),
                from_first=kwargs.get("from_first", False),
            )

    def get_session_history(
        self,
        session_id: str,
        *,
        limit: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        **kwargs: Any,
    ) -> list[BaseMessage]:
        """Get all messages for a session, optionally filtered by time range.

        Args:
            session_id: The session/user ID to query.
            limit: Maximum number of messages to return (returns most recent if specified).
            start_time: Optional minimum timestamp (inclusive) to filter messages.
            end_time: Optional maximum timestamp (inclusive) to filter messages.

        Return:
            list[BaseMessage]: List of messages in chronological order.
                Returns empty list if session not found or no matching messages.
        """
        return [
            msg
            for msg, _ in self.get_session_history_with_timestamps(
                session_id, start_time=start_time, end_time=end_time, limit=limit, **kwargs
            )
        ]

    async def aget_session_history(
        self,
        session_id: str,
        *,
        limit: int | None = None,
        start_time: float | None = None,
        end_time: float | None = None,
        **kwargs: Any,
    ) -> list[BaseMessage]:
        """Async version of get_session_history.

        Args:
            session_id: The session/user ID to query.
            limit: Maximum number of messages to return (returns most recent if specified).
            start_time: Optional minimum timestamp (inclusive) to filter messages.
            end_time: Optional maximum timestamp (inclusive) to filter messages.

        Return:
            list[BaseMessage]: List of messages in chronological order.
                Returns empty list if session not found or no matching messages.
        """
        pairs = await self.aget_session_history_with_timestamps(
            session_id, start_time=start_time, end_time=end_time, limit=limit, **kwargs
        )
        return [msg for msg, _ in pairs]


def get_datetime_from_timestamp(timestamp: float | None) -> datetime | None:
    try:
        if timestamp is None:
            return None
        if not isinstance(timestamp, (int, float)):
            return None
        if timestamp >= float("inf"):
            return datetime.max.replace(tzinfo=UTC)
        if timestamp <= 0:
            return datetime.min.replace(tzinfo=UTC)
        return datetime.fromtimestamp(timestamp, tz=UTC)
    except Exception:
        return None
