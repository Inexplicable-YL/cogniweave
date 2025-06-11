from pathlib import Path

from langchain_core.messages import HumanMessage

from cogniweave.core.database import ChatBlock, ChatMessage
from cogniweave.core.database.history import HistoryStore


def test_history_store_persistence_and_retrieval(tmp_path: Path) -> None:
    store = HistoryStore(db_url=f"sqlite:///{tmp_path}/test.sqlite")

    cfg = {"configurable": {"session_id": "s1", "session_timestamp": 1000.0}}
    msgs = ["hi", "how", "are", "you"]
    for i, text in enumerate(msgs):
        store.invoke({"message": HumanMessage(text), "timestamp": 1000.0 + i}, config=cfg)

    with store._session_local() as session:
        blocks = session.query(ChatBlock).all()
        messages = session.query(ChatMessage).all()

    assert len(blocks) == 1
    assert len(messages) == 4
    history = store.get_history("s1")
    assert len(history) == 4


async def test_history_store_async(tmp_path: Path) -> None:
    store = HistoryStore(db_url=f"sqlite:///{tmp_path}/async.sqlite")

    cfg = {"configurable": {"session_id": "s2", "session_timestamp": 50.0}}
    await store.ainvoke({"message": HumanMessage("hello"), "timestamp": 51.0}, config=cfg)
    await store.ainvoke({"message": HumanMessage("world"), "timestamp": 52.0}, config=cfg)

    history = await store.aget_history("s2")
    assert len(history) == 2
