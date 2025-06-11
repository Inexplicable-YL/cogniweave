from __future__ import annotations

import os
from datetime import datetime
from typing import TYPE_CHECKING, Any
from typing_extensions import override

from anyio import to_thread
from langchain_core.messages import (
    BaseMessage,
    message_to_dict,
    messages_from_dict,
)
from langchain_core.runnables import RunnableSerializable
from pydantic import PrivateAttr
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from cogniweave.core.database import Base, ChatBlock, ChatMessage, User

if TYPE_CHECKING:
    from langchain_core.runnables.config import RunnableConfig


class HistoryStore(RunnableSerializable[dict[str, Any], None]):
    """Persist chat messages grouped by session."""

    _session_local: sessionmaker[Session] = PrivateAttr()

    user_key: str = "user"
    message_key: str = "message"
    timestamp_key: str = "timestamp"

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
        Base.metadata.create_all(bind=engine)
        super().__init__()
        self._session_local = session_local

    def _get_or_create_user(self, session: Session, name: str) -> User:
        user = session.query(User).filter_by(name=name).first()
        if user is None:
            user = User(name=name)
            session.add(user)
            session.commit()
            session.refresh(user)
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

    def _store(
        self,
        session: Session,
        user: str,
        msg: BaseMessage,
        ts: float,
        context_id: str,
        start_ts: float,
    ) -> None:
        db_user = self._get_or_create_user(session, user)
        block = self._get_or_create_block(session, db_user, context_id, start_ts)
        record = ChatMessage(
            block_id=block.id,
            timestamp=datetime.fromtimestamp(ts),
            content=message_to_dict(msg),
        )
        session.add(record)
        session.commit()

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
        if not isinstance(message, BaseMessage):
            raise TypeError("message must be a BaseMessage")
        if not isinstance(timestamp, (int, float)):
            raise TypeError("timestamp must be a number")

        context_id = session_id
        start_ts = float(session_ts)

        with self._session_local() as session:
            self._store(session, user_name, message, float(timestamp), context_id, start_ts)

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
        if not isinstance(message, BaseMessage):
            raise TypeError("message must be a BaseMessage")
        if not isinstance(timestamp, (int, float)):
            raise TypeError("timestamp must be a number")

        context_id = session_id
        start_ts = float(session_ts)

        await to_thread.run_sync(
            self._store,
            self._session_local(),
            user_name,
            message,
            float(timestamp),
            context_id,
            start_ts,
        )

    def get_history(self, session_id: str) -> list[BaseMessage]:
        """Return ordered messages for a single session."""
        with self._session_local() as session:
            block = session.query(ChatBlock).filter_by(context_id=session_id).first()
            if not block:
                return []
            return messages_from_dict([m.content for m in block.messages])

    async def aget_history(self, session_id: str) -> list[BaseMessage]:
        """Asynchronously return ordered messages for a single session."""
        return await to_thread.run_sync(self.get_history, session_id)

    def get_histories(self, session_ids: list[str]) -> list[BaseMessage]:
        """Concatenate histories for multiple sessions in order."""
        messages: list[BaseMessage] = []
        for sid in sorted(session_ids):
            messages.extend(self.get_history(sid))
        return messages

    async def aget_histories(self, session_ids: list[str]) -> list[BaseMessage]:
        """Asynchronously concatenate histories for multiple sessions."""
        return await to_thread.run_sync(self.get_histories, session_ids)
