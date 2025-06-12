from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING, Any, cast
from typing_extensions import override

from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from langchain_core.runnables import RunnableSerializable
from pydantic import PrivateAttr
from sqlalchemy import create_engine, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import Session, sessionmaker

from cogniweave.core.database import (
    Base,
    ChatBlock,
    ChatBlockAttribute,
    ChatMessage,
    User,
)

if TYPE_CHECKING:
    from langchain_core.runnables.config import RunnableConfig


class HistoryStore(RunnableSerializable[dict[str, Any], None]):
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
                user_id=user.id,
                start_time=datetime.fromtimestamp(start_ts),
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
                user_id=user.id,
                start_time=datetime.fromtimestamp(start_ts),
            )
            session.add(block)
            await session.commit()
            await session.refresh(block)
        return block

    @override
    def invoke(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> None:
        if not config:
            raise ValueError("config must be provided")
        session_id = config.get("configurable", {}).get("session_id", "")
        session_ts = config.get("configurable", {}).get("session_timestamp")
        user_name = config.get("configurable", {}).get("user_name", session_id)

        if not session_id:
            raise ValueError("session_id is required")
        if not isinstance(session_ts, (int, float)):
            raise TypeError("session_timestamp is required")

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

        context_id = session_id
        start_ts = float(session_ts)

        with self._session_local() as session:
            db_user = self._get_or_create_user(session, user_name)
            block = self._get_or_create_block(session, db_user, context_id, start_ts)
            if message is not None:
                record = ChatMessage(
                    block_id=block.id,
                    timestamp=datetime.fromtimestamp(float(cast("float | int", timestamp))),
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
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> None:
        if not config:
            raise ValueError("config must be provided")
        session_id = config.get("configurable", {}).get("session_id", "")
        session_ts = config.get("configurable", {}).get("session_timestamp")
        user_name = config.get("configurable", {}).get("user_name", session_id)

        if not session_id:
            raise ValueError("session_id is required")
        if not isinstance(session_ts, (int, float)):
            raise TypeError("session_timestamp is required")

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

        context_id = session_id
        start_ts = float(session_ts)

        async with self._async_session_local() as session:
            db_user = await self._a_get_or_create_user(session, user_name)
            block = await self._a_get_or_create_block(session, db_user, context_id, start_ts)
            if message is not None:
                record = ChatMessage(
                    block_id=block.id,
                    timestamp=datetime.fromtimestamp(float(cast("float | int", timestamp))),
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

    def get_history(self, session_id: str) -> list[BaseMessage]:
        """Return ordered messages for a single session."""
        with self._session_local() as session:
            block = session.query(ChatBlock).filter_by(context_id=session_id).first()
            if not block:
                return []
            return messages_from_dict([m.content for m in block.messages])

    async def aget_history(self, session_id: str) -> list[BaseMessage]:
        """Asynchronously return ordered messages for a single session."""
        async with self._async_session_local() as session:
            result = await session.execute(select(ChatBlock).filter_by(context_id=session_id))
            block = result.scalar_one_or_none()
            if not block:
                return []
            return messages_from_dict([m.content for m in block.messages])

    def get_histories(self, session_ids: list[str]) -> list[BaseMessage]:
        """Concatenate histories for multiple sessions in order."""
        messages: list[BaseMessage] = []
        for sid in sorted(session_ids):
            messages.extend(self.get_history(sid))
        return messages

    async def aget_histories(self, session_ids: list[str]) -> list[BaseMessage]:
        """Asynchronously concatenate histories for multiple sessions."""
        messages: list[BaseMessage] = []
        for sid in sorted(session_ids):
            messages.extend(await self.aget_history(sid))
        return messages

    def get_block_attributes(
        self, session_id: str, *, types: list[str] | None = None
    ) -> list[ChatBlockAttribute]:
        """Return ordered attributes for a chat block."""
        with self._session_local() as session:
            block = session.query(ChatBlock).filter_by(context_id=session_id).first()
            if not block:
                return []

            attrs = sorted(block.attributes, key=lambda a: a.id)
            if types is not None:
                attrs = [attr for attr in attrs if attr.type in types]
            return attrs

    async def aget_block_attributes(
        self, session_id: str, *, types: list[str] | None = None
    ) -> list[ChatBlockAttribute]:
        """Asynchronously return ordered attributes for a chat block."""
        async with self._async_session_local() as session:
            result = await session.execute(select(ChatBlock).filter_by(context_id=session_id))
            block = result.scalar_one_or_none()
            if not block:
                return []

            attrs = sorted(block.attributes, key=lambda a: a.id)
            if types is not None:
                attrs = [attr for attr in attrs if attr.type in types]
            return attrs
