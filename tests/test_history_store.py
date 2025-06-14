from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from langchain_core.messages import HumanMessage

from cogniweave.historystore import ChatBlock, ChatMessage
from cogniweave.historystore.history_store import HistoryStore

if TYPE_CHECKING:
    from langchain_core.runnables.config import RunnableConfig


def test_basic_operations(tmp_path: Path) -> None:
    """Test basic store operations including message persistence and retrieval."""
    store = HistoryStore(db_url=f"sqlite:///{tmp_path}/test.sqlite")

    # Test message storage
    cfg = cast("RunnableConfig", {"configurable": {"block_id": "s1", "block_timestamp": 1000.0}})
    test_messages = ["hi", "how", "are", "you"]
    for i, text in enumerate(test_messages):
        store.invoke({"block_messages": [(HumanMessage(text), 1000.0 + i)]}, config=cfg)

    # Verify database records
    with store._session_local() as session:
        blocks = session.query(ChatBlock).all()
        messages = session.query(ChatMessage).all()
        assert len(blocks) == 1
        assert len(messages) == len(test_messages)

    # Test retrieval methods
    assert len(store.get_history("s1")) == len(test_messages)
    assert len(store.get_history_with_timestamps("s1")) == len(test_messages)
    assert store.get_block_timestamp("s1") == 1000.0
    assert store.get_block_timestamp("nonexistent") is None


async def test_async_basic_operations(tmp_path: Path) -> None:
    """Test basic async store operations."""
    store = HistoryStore(db_url=f"sqlite:///{tmp_path}/async.sqlite")

    # Test async message storage
    cfg = cast("RunnableConfig", {"configurable": {"block_id": "s2", "block_timestamp": 50.0}})
    await store.ainvoke({"block_messages": [(HumanMessage("hello"), 51.0)]}, config=cfg)
    await store.ainvoke({"block_messages": [(HumanMessage("world"), 52.0)]}, config=cfg)

    # Test async retrieval
    history = await store.aget_history("s2")
    assert len(history) == 2
    assert [msg.content for msg in history] == ["hello", "world"]


def test_block_attributes_operations(tmp_path: Path) -> None:
    """Test block attribute storage and retrieval."""
    store = HistoryStore(db_url=f"sqlite:///{tmp_path}/attrs.sqlite")
    cfg = cast("RunnableConfig", {"configurable": {"block_id": "attr", "block_timestamp": 1.0}})

    # Store initial message
    store.invoke({"block_messages": [(HumanMessage("hello"), 1.1)]}, config=cfg)

    # Test attribute storage
    test_attrs = [
        {"type": "summary", "value": {"text": "hello"}},
        {"type": "tag", "value": ["important"]},
    ]
    for attr in test_attrs:
        store.invoke({"block_attributes": [attr]}, config=cfg)

    # Test attribute retrieval
    attrs = store.get_block_attributes("attr")
    assert len(attrs) == len(test_attrs)
    assert {a["type"] for a in attrs} == {a["type"] for a in test_attrs}

    # Test filtered retrieval
    filtered = store.get_block_attributes("attr", types=["summary"])
    assert len(filtered) == 1
    assert filtered[0]["type"] == "summary"


async def test_async_block_attributes_operations(tmp_path: Path) -> None:
    """Test async block attribute operations."""
    store = HistoryStore(db_url=f"sqlite:///{tmp_path}/attrs_async.sqlite")
    cfg = cast("RunnableConfig", {"configurable": {"block_id": "attr", "block_timestamp": 1.0}})

    # Store initial message
    await store.ainvoke({"block_messages": [(HumanMessage("hello"), 1.1)]}, config=cfg)

    # Test async attribute storage
    test_attrs = [
        {"type": "summary", "value": {"text": "hello"}},
        {"type": "tag", "value": ["important"]},
    ]
    for attr in test_attrs:
        await store.ainvoke({"block_attributes": [attr]}, config=cfg)

    # Test async retrieval
    attrs = await store.aget_block_attributes("attr")
    assert len(attrs) == len(test_attrs)
    assert {a["type"] for a in attrs} == {a["type"] for a in test_attrs}


def test_invoke_validation(tmp_path: Path) -> None:
    store = HistoryStore(db_url=f"sqlite:///{tmp_path}/val.sqlite")
    good_cfg = cast(
        "RunnableConfig",
        {"configurable": {"block_id": "b", "block_timestamp": 1.0}},
    )

    with pytest.raises(ValueError, match="config must be provided"):
        store.invoke({}, config=None)

    bad_cfg = cast("RunnableConfig", {"configurable": {"block_timestamp": 1.0}})
    with pytest.raises(ValueError, match="block_id is required"):
        store.invoke({"block_messages": [(HumanMessage("x"), 1.0)]}, config=bad_cfg)

    with pytest.raises(TypeError):
        store.invoke({"block_messages": [("not msg", 1.0)]}, config=good_cfg)

    with pytest.raises(TypeError):
        store.invoke({"block_messages": [(HumanMessage("hi"), "bad")]}, config=good_cfg)

    with pytest.raises(ValueError, match="nothing to store"):
        store.invoke({}, config=good_cfg)

    with pytest.raises(TypeError):
        store.invoke({"block_attributes": "bad"}, config=good_cfg)

    with pytest.raises(TypeError):
        store.invoke({"block_attributes": [{"type": 123}]}, config=good_cfg)


async def test_ainvoke_validation(tmp_path: Path) -> None:
    store = HistoryStore(db_url=f"sqlite:///{tmp_path}/aval.sqlite")
    good_cfg = cast(
        "RunnableConfig",
        {"configurable": {"block_id": "b", "block_timestamp": 1.0}},
    )

    with pytest.raises(ValueError, match="config must be provided"):
        await store.ainvoke({}, config=None)

    bad_cfg = cast("RunnableConfig", {"configurable": {"block_timestamp": 1.0}})
    with pytest.raises(ValueError, match="block_id is required"):
        await store.ainvoke({"block_messages": [(HumanMessage("x"), 1.0)]}, config=bad_cfg)

    with pytest.raises(TypeError):
        await store.ainvoke({"block_messages": [("not msg", 1.0)]}, config=good_cfg)

    with pytest.raises(TypeError):
        await store.ainvoke({"block_messages": [(HumanMessage("hi"), "bad")]}, config=good_cfg)

    with pytest.raises(ValueError, match="nothing to store"):
        await store.ainvoke({}, config=good_cfg)

    with pytest.raises(TypeError):
        await store.ainvoke({"block_attributes": "bad"}, config=good_cfg)

    with pytest.raises(TypeError):
        await store.ainvoke({"block_attributes": [{"type": 123}]}, config=good_cfg)


def test_history_utilities(tmp_path: Path) -> None:
    store = HistoryStore(db_url=f"sqlite:///{tmp_path}/util.sqlite")

    cfg1 = cast("RunnableConfig", {"configurable": {"block_id": "b1", "block_timestamp": 1.0}})
    cfg2 = cast("RunnableConfig", {"configurable": {"block_id": "b2", "block_timestamp": 2.0}})

    store.invoke({"block_messages": [(HumanMessage("m1"), 1.1)]}, config=cfg1)
    store.invoke({"block_messages": [(HumanMessage("m2"), 2.1)]}, config=cfg2)
    store.invoke({"block_messages": [(HumanMessage("m3"), 1.2)]}, config=cfg1)
    store.invoke({"block_attributes": [{"type": "summary", "value": "s"}]}, config=cfg1)
    store.invoke({"block_attributes": [{"type": "tag", "value": ["t"]}]}, config=cfg1)

    assert [m.content for m, _ in store.get_history_with_timestamps("b1")] == ["m1", "m3"]
    assert len(store.get_histories(["b2", "b1"])) == 3
    assert len(store.get_histories_with_timestamps(["b2", "b1"])) == 3

    attrs_all = store.get_block_attributes("b1")
    attrs_filtered = store.get_block_attributes("b1", types=["summary"])
    assert len(attrs_all) == 2
    assert len(attrs_filtered) == 1
    assert attrs_filtered[0]["type"] == "summary"
    assert store.get_block_timestamp("b1") == 1.0
    assert store.get_history("nope") == []
    assert store.get_block_timestamp("nope") is None


async def test_async_history_utilities(tmp_path: Path) -> None:
    store = HistoryStore(db_url=f"sqlite:///{tmp_path}/util_async.sqlite")

    cfg1 = cast("RunnableConfig", {"configurable": {"block_id": "b1", "block_timestamp": 1.0}})
    cfg2 = cast("RunnableConfig", {"configurable": {"block_id": "b2", "block_timestamp": 2.0}})

    await store.ainvoke({"block_messages": [(HumanMessage("m1"), 1.1)]}, config=cfg1)
    await store.ainvoke({"block_messages": [(HumanMessage("m2"), 2.1)]}, config=cfg2)
    await store.ainvoke({"block_messages": [(HumanMessage("m3"), 1.2)]}, config=cfg1)
    await store.ainvoke({"block_attributes": [{"type": "summary", "value": "s"}]}, config=cfg1)

    res = await store.aget_history_with_timestamps("b1")
    assert [m.content for m, _ in res] == ["m1", "m3"]
    assert len(await store.aget_histories(["b2", "b1"])) == 3
    assert len(await store.aget_histories_with_timestamps(["b2", "b1"])) == 3
    attrs_all = await store.aget_block_attributes("b1")
    assert len(attrs_all) == 1
    ts = await store.aget_block_timestamp("b1")
    assert ts == 1.0


def test_session_range_utilities(tmp_path: Path) -> None:
    store = HistoryStore(db_url=f"sqlite:///{tmp_path}/range.sqlite")

    cfg1 = cast(
        "RunnableConfig",
        {"configurable": {"block_id": "b1", "block_timestamp": 1.0, "session_id": "s"}},
    )
    cfg2 = cast(
        "RunnableConfig",
        {"configurable": {"block_id": "b2", "block_timestamp": 2.0, "session_id": "s"}},
    )
    cfg3 = cast(
        "RunnableConfig",
        {"configurable": {"block_id": "b3", "block_timestamp": 3.0, "session_id": "s"}},
    )

    store.invoke({"block_messages": [(HumanMessage("m1"), 1.1)]}, config=cfg1)
    store.invoke({"block_messages": [(HumanMessage("m2"), 2.1)]}, config=cfg2)
    store.invoke({"block_messages": [(HumanMessage("m3"), 3.1)]}, config=cfg3)

    ids_ts = store.get_session_block_ids_with_timestamps("s", start_time=1.5, end_time=2.5)
    assert ids_ts == [("b2", 2.0)]
    ids = store.get_session_block_ids("s", start_time=1.5, end_time=2.5)
    assert ids == ["b2"]

    hist_ts_1 = store.get_session_history_with_timestamps("s", start_time=1.05, end_time=2.8)
    assert [m.content for m, _ in hist_ts_1] == ["m1", "m2"]
    hist_1 = store.get_session_history("s", end_time=2.8)
    assert [m.content for m in hist_1] == ["m1", "m2"]
    hist_ts_2 = store.get_session_history_with_timestamps("s", start_time=1.5, end_time=3.5)
    assert [m.content for m, _ in hist_ts_2] == ["m2", "m3"]
    hist_2 = store.get_session_history("s", start_time=1.5)
    assert [m.content for m in hist_2] == ["m2", "m3"]

    # Test limit parameter
    assert store.get_session_block_ids_with_timestamps("s", limit=0) == []
    assert store.get_session_block_ids_with_timestamps("s", limit=1) == [("b3", 3.0)]
    assert store.get_session_block_ids_with_timestamps("s", limit=2) == [("b2", 2.0), ("b3", 3.0)]
    assert store.get_session_block_ids_with_timestamps("s", limit=10) == [
        ("b1", 1.0),
        ("b2", 2.0),
        ("b3", 3.0),
    ]

    assert store.get_session_block_ids("s", limit=0) == []
    assert store.get_session_block_ids("s", limit=1) == ["b3"]
    assert store.get_session_block_ids("s", limit=2) == ["b2", "b3"]
    assert store.get_session_block_ids("s", limit=10) == ["b1", "b2", "b3"]

    assert store.get_session_history_with_timestamps("s", limit=0) == []
    assert [m.content for m, _ in store.get_session_history_with_timestamps("s", limit=1)] == ["m3"]
    assert [m.content for m, _ in store.get_session_history_with_timestamps("s", limit=2)] == [
        "m2",
        "m3",
    ]
    assert [m.content for m, _ in store.get_session_history_with_timestamps("s", limit=10)] == [
        "m1",
        "m2",
        "m3",
    ]

    assert store.get_session_history("s", limit=0) == []
    assert [m.content for m in store.get_session_history("s", limit=1)] == ["m3"]
    assert [m.content for m in store.get_session_history("s", limit=2)] == ["m2", "m3"]
    assert [m.content for m in store.get_session_history("s", limit=10)] == ["m1", "m2", "m3"]


async def test_async_session_range_utilities(tmp_path: Path) -> None:
    store = HistoryStore(db_url=f"sqlite:///{tmp_path}/range_async.sqlite")

    cfg1 = cast(
        "RunnableConfig",
        {"configurable": {"block_id": "b1", "block_timestamp": 1.0, "session_id": "s"}},
    )
    cfg2 = cast(
        "RunnableConfig",
        {"configurable": {"block_id": "b2", "block_timestamp": 2.0, "session_id": "s"}},
    )
    cfg3 = cast(
        "RunnableConfig",
        {"configurable": {"block_id": "b3", "block_timestamp": 3.0, "session_id": "s"}},
    )

    await store.ainvoke({"block_messages": [(HumanMessage("m1"), 1.1)]}, config=cfg1)
    await store.ainvoke({"block_messages": [(HumanMessage("m2"), 2.1)]}, config=cfg2)
    await store.ainvoke({"block_messages": [(HumanMessage("m3"), 3.1)]}, config=cfg3)

    ids_ts = await store.aget_session_block_ids_with_timestamps("s", start_time=1.5, end_time=2.5)
    assert ids_ts == [("b2", 2.0)]
    ids = await store.aget_session_block_ids("s", start_time=1.5, end_time=2.5)
    assert ids == ["b2"]

    hist_ts_1 = await store.aget_session_history_with_timestamps("s", start_time=1.05, end_time=2.8)
    assert [m.content for m, _ in hist_ts_1] == ["m1", "m2"]
    hist_1 = await store.aget_session_history("s", end_time=2.8)
    assert [m.content for m in hist_1] == ["m1", "m2"]
    hist_ts_2 = await store.aget_session_history_with_timestamps("s", start_time=1.5, end_time=3.5)
    assert [m.content for m, _ in hist_ts_2] == ["m2", "m3"]
    hist_2 = await store.aget_session_history("s", start_time=1.5)
    assert [m.content for m in hist_2] == ["m2", "m3"]

    # Test limit parameter
    assert await store.aget_session_block_ids_with_timestamps("s", limit=0) == []
    assert await store.aget_session_block_ids_with_timestamps("s", limit=1) == [("b3", 3.0)]
    assert await store.aget_session_block_ids_with_timestamps("s", limit=2) == [
        ("b2", 2.0),
        ("b3", 3.0),
    ]
    assert await store.aget_session_block_ids_with_timestamps("s", limit=10) == [
        ("b1", 1.0),
        ("b2", 2.0),
        ("b3", 3.0),
    ]

    assert await store.aget_session_block_ids("s", limit=0) == []
    assert await store.aget_session_block_ids("s", limit=1) == ["b3"]
    assert await store.aget_session_block_ids("s", limit=2) == ["b2", "b3"]
    assert await store.aget_session_block_ids("s", limit=10) == ["b1", "b2", "b3"]

    assert await store.aget_session_history_with_timestamps("s", limit=0) == []
    assert [
        m.content for m, _ in await store.aget_session_history_with_timestamps("s", limit=1)
    ] == ["m3"]
    assert [
        m.content for m, _ in await store.aget_session_history_with_timestamps("s", limit=2)
    ] == ["m2", "m3"]
    assert [
        m.content for m, _ in await store.aget_session_history_with_timestamps("s", limit=10)
    ] == ["m1", "m2", "m3"]

    assert await store.aget_session_history("s", limit=0) == []
    assert [m.content for m in await store.aget_session_history("s", limit=1)] == ["m3"]
    assert [m.content for m in await store.aget_session_history("s", limit=2)] == ["m2", "m3"]
    assert [m.content for m in await store.aget_session_history("s", limit=10)] == [
        "m1",
        "m2",
        "m3",
    ]
