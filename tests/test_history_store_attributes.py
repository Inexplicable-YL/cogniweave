from pathlib import Path

from langchain_core.messages import HumanMessage

from cogniweave.core.database import ChatBlock
from cogniweave.core.database.history import HistoryStore


def test_block_attributes(tmp_path: Path) -> None:
    store = HistoryStore(db_url=f"sqlite:///{tmp_path}/attrs.sqlite")
    cfg = {"configurable": {"session_id": "attr", "session_timestamp": 1.0}}
    store.invoke({"message": HumanMessage("hello"), "timestamp": 1.1}, config=cfg)

    with store._session_local() as session:
        block = session.query(ChatBlock).filter_by(context_id="attr").first()
        assert block is not None
        block_id = block.id

    store.invoke({"block_attribute": {"type": "summary", "value": {"text": "hello"}}}, config=cfg)
    attrs = store.get_block_attributes(block_id)
    assert len(attrs) == 1
    assert attrs[0].type == "summary"
    assert attrs[0].value == {"text": "hello"}
