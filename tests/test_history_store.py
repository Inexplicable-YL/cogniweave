import os
from pathlib import Path

from langchain_core.messages import HumanMessage

from cogniweave.core.database import ChatBlock, ChatMessage, SessionLocal, init_db
from cogniweave.core.history import HistoryStore


def test_history_store_segments_and_persists(tmp_path: Path) -> None:
    os.environ["CHAT_DB_URL"] = f"sqlite:///{tmp_path}/test.sqlite"
    init_db()
    store = HistoryStore()

    cfg = {"configurable": {"session_id": "user1", "user_name": "user1"}}
    base = 1000.0
    store.invoke({"message": HumanMessage("hi"), "timestamp": base}, config=cfg)
    store.invoke({"message": HumanMessage("how"), "timestamp": base + 10}, config=cfg)
    store.invoke({"message": HumanMessage("are"), "timestamp": base + 20}, config=cfg)
    store.invoke({"message": HumanMessage("you"), "timestamp": base + 200}, config=cfg)

    with SessionLocal() as session:
        blocks = session.query(ChatBlock).all()
        messages = session.query(ChatMessage).all()

    assert len(blocks) == 2
    assert len(messages) == 4
