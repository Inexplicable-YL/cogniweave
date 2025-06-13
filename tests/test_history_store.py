from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
from langchain_core.messages import HumanMessage

from cogniweave.historystore import ChatBlock, ChatMessage
from cogniweave.historystore.history_store import HistoryStore

if TYPE_CHECKING:
    from langchain_core.runnables.config import RunnableConfig


def test_history_store_persistence_and_retrieval(tmp_path: Path) -> None:
    store = HistoryStore(db_url=f"sqlite:///{tmp_path}/test.sqlite")

    cfg = cast("RunnableConfig", {"configurable": {"block_id": "s1", "block_timestamp": 1000.0}})
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

    cfg = cast("RunnableConfig", {"configurable": {"block_id": "s2", "block_timestamp": 50.0}})
    await store.ainvoke({"message": HumanMessage("hello"), "timestamp": 51.0}, config=cfg)
    await store.ainvoke({"message": HumanMessage("world"), "timestamp": 52.0}, config=cfg)

    history = await store.aget_history("s2")
    assert len(history) == 2


def test_block_attributes(tmp_path: Path) -> None:
    store = HistoryStore(db_url=f"sqlite:///{tmp_path}/attrs.sqlite")
    cfg = cast("RunnableConfig", {"configurable": {"block_id": "attr", "block_timestamp": 1.0}})
    store.invoke({"message": HumanMessage("hello"), "timestamp": 1.1}, config=cfg)

    store.invoke({"block_attribute": {"type": "summary", "value": {"text": "hello"}}}, config=cfg)
    attrs = store.get_block_attributes("attr")
    assert len(attrs) == 1
    assert attrs[0]["type"] == "summary"
    assert attrs[0]["value"] == {"text": "hello"}


async def test_block_attributes_async(tmp_path: Path) -> None:
    store = HistoryStore(db_url=f"sqlite:///{tmp_path}/attrs.sqlite")
    cfg = cast("RunnableConfig", {"configurable": {"block_id": "attr", "block_timestamp": 1.0}})
    store.invoke({"message": HumanMessage("hello"), "timestamp": 1.1}, config=cfg)

    await store.ainvoke(
        {"block_attribute": {"type": "summary", "value": {"text": "hello"}}}, config=cfg
    )
    attrs = await store.aget_block_attributes("attr")
    assert len(attrs) == 1
    assert attrs[0]["type"] == "summary"
    assert attrs[0]["value"] == {"text": "hello"}


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
        store.invoke({"message": HumanMessage("x"), "timestamp": 1.0}, config=bad_cfg)

    with pytest.raises(TypeError):
        store.invoke({"message": "not msg", "timestamp": 1.0}, config=good_cfg)

    with pytest.raises(TypeError):
        store.invoke({"message": HumanMessage("hi"), "timestamp": "bad"}, config=good_cfg)

    with pytest.raises(ValueError, match="nothing to store"):
        store.invoke({}, config=good_cfg)

    with pytest.raises(TypeError):
        store.invoke({"block_attribute": "bad"}, config=good_cfg)

    with pytest.raises(TypeError):
        store.invoke({"block_attribute": {"type": 123}}, config=good_cfg)


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
        await store.ainvoke({"message": HumanMessage("x"), "timestamp": 1.0}, config=bad_cfg)

    with pytest.raises(TypeError):
        await store.ainvoke({"message": "not msg", "timestamp": 1.0}, config=good_cfg)

    with pytest.raises(TypeError):
        await store.ainvoke({"message": HumanMessage("hi"), "timestamp": "bad"}, config=good_cfg)

    with pytest.raises(ValueError, match="nothing to store"):
        await store.ainvoke({}, config=good_cfg)

    with pytest.raises(TypeError):
        await store.ainvoke({"block_attribute": "bad"}, config=good_cfg)

    with pytest.raises(TypeError):
        await store.ainvoke({"block_attribute": {"type": 123}}, config=good_cfg)


def test_history_utilities(tmp_path: Path) -> None:
    store = HistoryStore(db_url=f"sqlite:///{tmp_path}/util.sqlite")

    cfg1 = cast("RunnableConfig", {"configurable": {"block_id": "b1", "block_timestamp": 1.0}})
    cfg2 = cast("RunnableConfig", {"configurable": {"block_id": "b2", "block_timestamp": 2.0}})

    store.invoke({"message": HumanMessage("m1"), "timestamp": 1.1}, config=cfg1)
    store.invoke({"message": HumanMessage("m2"), "timestamp": 2.1}, config=cfg2)
    store.invoke({"message": HumanMessage("m3"), "timestamp": 1.2}, config=cfg1)
    store.invoke({"block_attribute": {"type": "summary", "value": "s"}}, config=cfg1)
    store.invoke({"block_attribute": {"type": "tag", "value": ["t"]}}, config=cfg1)

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

    await store.ainvoke({"message": HumanMessage("m1"), "timestamp": 1.1}, config=cfg1)
    await store.ainvoke({"message": HumanMessage("m2"), "timestamp": 2.1}, config=cfg2)
    await store.ainvoke({"message": HumanMessage("m3"), "timestamp": 1.2}, config=cfg1)
    await store.ainvoke({"block_attribute": {"type": "summary", "value": "s"}}, config=cfg1)

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

    cfg1 = cast("RunnableConfig", {
        "configurable": {"block_id": "b1", "block_timestamp": 1.0, "session_id": "s"}
    })
    cfg2 = cast("RunnableConfig", {
        "configurable": {"block_id": "b2", "block_timestamp": 2.0, "session_id": "s"}
    })
    cfg3 = cast("RunnableConfig", {
        "configurable": {"block_id": "b3", "block_timestamp": 3.0, "session_id": "s"}
    })

    store.invoke({"message": HumanMessage("m1"), "timestamp": 1.1}, config=cfg1)
    store.invoke({"message": HumanMessage("m2"), "timestamp": 2.1}, config=cfg2)
    store.invoke({"message": HumanMessage("m3"), "timestamp": 3.1}, config=cfg3)

    ids_ts = store.get_session_block_ids_with_timestamps("s", start=1.5, end=2.5)
    assert ids_ts == [("b2", 2.0)]
    ids = store.get_session_block_ids("s", start=1.5, end=2.5)
    assert ids == ["b2"]

    hist_ts = store.get_session_history_with_timestamps("s", start=1.05, end=2.8)
    assert [m.content for m, _ in hist_ts] == ["m1", "m2"]
    hist = store.get_session_history("s", start=1.05, end=2.8)
    assert [m.content for m in hist] == ["m1", "m2"]


async def test_async_session_range_utilities(tmp_path: Path) -> None:
    store = HistoryStore(db_url=f"sqlite:///{tmp_path}/range_async.sqlite")

    cfg1 = cast("RunnableConfig", {
        "configurable": {"block_id": "b1", "block_timestamp": 1.0, "session_id": "s"}
    })
    cfg2 = cast("RunnableConfig", {
        "configurable": {"block_id": "b2", "block_timestamp": 2.0, "session_id": "s"}
    })
    cfg3 = cast("RunnableConfig", {
        "configurable": {"block_id": "b3", "block_timestamp": 3.0, "session_id": "s"}
    })

    await store.ainvoke({"message": HumanMessage("m1"), "timestamp": 1.1}, config=cfg1)
    await store.ainvoke({"message": HumanMessage("m2"), "timestamp": 2.1}, config=cfg2)
    await store.ainvoke({"message": HumanMessage("m3"), "timestamp": 3.1}, config=cfg3)

    ids_ts = await store.aget_session_block_ids_with_timestamps("s", start=1.5, end=2.5)
    assert ids_ts == [("b2", 2.0)]
    ids = await store.aget_session_block_ids("s", start=1.5, end=2.5)
    assert ids == ["b2"]

    hist_ts = await store.aget_session_history_with_timestamps("s", start=1.05, end=2.8)
    assert [m.content for m, _ in hist_ts] == ["m1", "m2"]
    hist = await store.aget_session_history("s", start=1.05, end=2.8)
    assert [m.content for m in hist] == ["m1", "m2"]


