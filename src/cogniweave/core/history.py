from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any
from typing_extensions import override

import anyio
from langchain_core.messages import BaseMessage, message_to_dict
from langchain_core.runnables import RunnableSerializable
from cogniweave.core.database import ChatBlock, ChatMessage, SessionLocal, User, init_db
from cogniweave.core.timesplit import ContextTimeSplitter

if TYPE_CHECKING:
    from langchain_core.runnables.config import RunnableConfig
    from sqlalchemy.orm import Session


class HistoryStore(RunnableSerializable[dict[str, Any], None]):
    """Store chat history with time-based segmentation."""

    splitter: ContextTimeSplitter | None = None
    user_key: str = "user"
    message_key: str = "message"
    timestamp_key: str = "timestamp"

    def __init__(self, **kwargs: Any) -> None:
        self.splitter = ContextTimeSplitter(**kwargs)
        super().__init__()
        init_db()

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
        user_name = config.get("configurable", {}).get("user_name", session_id)
        if not session_id:
            raise ValueError("session_id is required")

        message = input.get(self.message_key)
        timestamp = input.get(self.timestamp_key)
        if not isinstance(message, BaseMessage):
            raise TypeError("message must be a BaseMessage")
        if not isinstance(timestamp, (int, float)):
            raise TypeError("timestamp must be a number")

        assert self.splitter is not None
        split = self.splitter.invoke(
            {"timestamp": float(timestamp)}, config={"configurable": {"session_id": session_id}}
        )
        context_id = split["context_id"]
        start_ts = split["timestamp"]

        with SessionLocal() as session:
            self._store(session, user_name, message, float(timestamp), context_id, start_ts)

    @override
    async def ainvoke(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> None:
        if not config:
            raise ValueError("config must be provided")
        session_id = config.get("configurable", {}).get("session_id", "")
        user_name = config.get("configurable", {}).get("user_name", session_id)
        if not session_id:
            raise ValueError("session_id is required")

        message = input.get(self.message_key)
        timestamp = input.get(self.timestamp_key)
        if not isinstance(message, BaseMessage):
            raise TypeError("message must be a BaseMessage")
        if not isinstance(timestamp, (int, float)):
            raise TypeError("timestamp must be a number")

        assert self.splitter is not None
        split = await self.splitter.ainvoke(
            {"timestamp": float(timestamp)}, config={"configurable": {"session_id": session_id}}
        )
        context_id = split["context_id"]
        start_ts = split["timestamp"]

        await anyio.to_thread.run_sync(
            self._store,
            SessionLocal(),
            user_name,
            message,
            float(timestamp),
            context_id,
            start_ts,
        )
