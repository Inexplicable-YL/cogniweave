from pathlib import Path

from langchain_core.messages import HumanMessage
from sqlalchemy import select

from cogniweave.history_stores import (
    BaseHistoryStoreWithCache as BaseHistoryStore,
)
from cogniweave.history_stores import (
    ChatBlock,
    ChatBlockAttribute,
    ChatMessage,
    User,
    UserAttribute,
)


def test_basic_operations(tmp_path: Path) -> None:
    """Test basic store operations including message persistence and retrieval."""
    store = BaseHistoryStore(db_url=f"sqlite:///{tmp_path}/test.sqlite")

    # Test message storage
    test_messages = ["hi", "how", "are", "you"]

    store.add_messages(
        [(HumanMessage(text), 1000.0 + i) for i, text in enumerate(test_messages)],
        block_id="s1",
        block_ts=1000.0,
    )

    # Verify database records
    with store._session_local() as session:
        users = session.query(User).all()
        blocks = session.query(ChatBlock).all()
        messages = session.query(ChatMessage).all()
        assert len(users) == 1
        assert users[0].id is not None
        assert len(blocks) == 1
        assert blocks[0].session_id == users[0].id
        assert len(messages) == len(test_messages)

    # Test retrieval methods
    assert len(store.get_block_history("s1")) == len(test_messages)
    assert len(store.get_block_history_with_timestamps("s1")) == len(test_messages)
    assert store.get_block_timestamp("s1") == 1000.0
    assert store.get_block_timestamp("nonexistent") is None


async def test_async_basic_operations(tmp_path: Path) -> None:
    """Test basic async store operations."""
    store = BaseHistoryStore(db_url=f"sqlite:///{tmp_path}/async.sqlite")

    # Test async message storage
    await store.aadd_messages(
        [(HumanMessage("hello"), 51.0)],
        block_id="s2",
        block_ts=50.0,
    )
    await store.aadd_messages(
        [(HumanMessage("world"), 52.0)],
        block_id="s2",
        block_ts=50.0,
    )

    # Verify database records
    async with store._async_session_local() as session:
        result_users = await session.execute(select(User))
        users = result_users.scalars().all()
        result_blocks = await session.execute(select(ChatBlock))
        blocks = result_blocks.scalars().all()
        result_messages = await session.execute(select(ChatMessage))
        messages = result_messages.scalars().all()
        assert len(users) == 1
        assert users[0].id is not None
        assert len(blocks) == 1
        assert blocks[0].session_id == users[0].id
        assert len(messages) == 2

    # Test async retrieval
    history = await store.aget_block_history("s2")
    assert len(history) == 2
    assert [msg.content for msg in history] == ["hello", "world"]


def test_block_attributes_operations(tmp_path: Path) -> None:
    """Test block attribute storage and retrieval."""
    store = BaseHistoryStore(db_url=f"sqlite:///{tmp_path}/attrs.sqlite")

    # Store initial message
    store.add_messages(
        [(HumanMessage("hello"), 1.1)],
        block_id="attr",
        block_ts=1.0,
    )

    # Test attribute storage
    test_attrs = [
        {"type": "summary", "value": {"text": "hello"}},
        {"type": "tag", "value": ["important"]},
    ]
    for attr in test_attrs:
        store.add_block_attributes(
            [attr],  # type: ignore
            block_id="attr",
            block_ts=1.0,
        )

    # Verify database records
    with store._session_local() as session:
        users = session.query(User).all()
        blocks = session.query(ChatBlock).all()
        attrs = session.query(ChatBlockAttribute).all()
        assert len(users) == 1
        assert users[0].id is not None
        assert len(blocks) == 1
        assert blocks[0].session_id == users[0].id
        assert len(attrs) == len(test_attrs)

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
    store = BaseHistoryStore(db_url=f"sqlite:///{tmp_path}/attrs_async.sqlite")

    # Store initial message
    await store.aadd_messages(
        [(HumanMessage("hello"), 1.1)],
        block_id="attr",
        block_ts=1.0,
    )

    # Test async attribute storage
    test_attrs = [
        {"type": "summary", "value": {"text": "hello"}},
        {"type": "tag", "value": ["important"]},
    ]
    for attr in test_attrs:
        await store.aadd_block_attributes(
            [attr],  # type: ignore
            block_id="attr",
            block_ts=1.0,
        )

    # Verify database records
    async with store._async_session_local() as session:
        result_users = await session.execute(select(User))
        users = result_users.scalars().all()
        result_blocks = await session.execute(select(ChatBlock))
        blocks = result_blocks.scalars().all()
        result_attrs = await session.execute(select(ChatBlockAttribute))
        attrs_db = result_attrs.scalars().all()
        assert len(users) == 1
        assert users[0].id is not None
        assert len(blocks) == 1
        assert blocks[0].session_id == users[0].id
        assert len(attrs_db) == len(test_attrs)

    # Test async retrieval
    attrs = await store.aget_block_attributes("attr")
    assert len(attrs) == len(test_attrs)
    assert {a["type"] for a in attrs} == {a["type"] for a in test_attrs}


def test_user_attributes_operations(tmp_path: Path) -> None:
    store = BaseHistoryStore(db_url=f"sqlite:///{tmp_path}/user_attrs.sqlite")

    store.add_session_attributes(
        [{"type": "color", "value": "blue"}],
        session_id="u",
    )
    store.add_session_attributes(
        [{"type": "color", "value": "red"}, {"type": "role", "value": "admin"}],
        session_id="u",
    )

    with store._session_local() as session:
        user = session.query(User).filter_by(session_id="u").first()
        assert user is not None
        attrs = {a.type: a.value for a in user.attributes}
        assert attrs == {"color": "red", "role": "admin"}

    # Test retrieval helpers
    result = store.get_session_attributes("u")
    assert {a["type"]: a["value"] for a in result} == {
        "color": "red",
        "role": "admin",
    }
    filtered = store.get_session_attributes("u", types=["role"])
    assert len(filtered) == 1
    assert filtered[0]["type"] == "role"


async def test_async_user_attributes_operations(tmp_path: Path) -> None:
    store = BaseHistoryStore(db_url=f"sqlite:///{tmp_path}/user_attrs_async.sqlite")

    await store.aadd_session_attributes(
        [{"type": "color", "value": "blue"}],
        session_id="u",
    )
    await store.aadd_session_attributes(
        [{"type": "color", "value": "red"}, {"type": "role", "value": "admin"}],
        session_id="u",
    )

    async with store._async_session_local() as session:
        await session.execute(select(User))  # ensure user exists
        result_attrs = await session.execute(select(UserAttribute))
        attrs_db = result_attrs.scalars().all()
        attrs = {a.type: a.value for a in attrs_db}
        assert attrs == {"color": "red", "role": "admin"}

    result = await store.aget_session_attributes("u")
    assert {a["type"]: a["value"] for a in result} == {
        "color": "red",
        "role": "admin",
    }
    filtered = await store.aget_session_attributes("u", types=["color"])
    assert len(filtered) == 1
    assert filtered[0]["type"] == "color"


def test_history_utilities(tmp_path: Path) -> None:
    store = BaseHistoryStore(db_url=f"sqlite:///{tmp_path}/util.sqlite")

    store.add_messages(
        [(HumanMessage("m1"), 1.1)],
        block_id="b1",
        block_ts=1.0,
    )
    store.add_messages(
        [(HumanMessage("m2"), 2.1)],
        block_id="b2",
        block_ts=2.0,
    )
    store.add_messages(
        [(HumanMessage("m3"), 1.2)],
        block_id="b1",
        block_ts=1.0,
    )
    store.add_block_attributes(
        [{"type": "summary", "value": "s"}],
        block_id="b1",
        block_ts=1.0,
    )
    store.add_block_attributes(
        [{"type": "tag", "value": ["t"]}],
        block_id="b1",
        block_ts=1.0,
    )

    assert [m.content for m, _ in store.get_block_history_with_timestamps("b1")] == ["m1", "m3"]
    assert len(store.get_block_histories(["b2", "b1"])) == 3
    assert len(store.get_block_histories_with_timestamps(["b2", "b1"])) == 3

    attrs_all = store.get_block_attributes("b1")
    attrs_filtered = store.get_block_attributes("b1", types=["summary"])
    assert len(attrs_all) == 2
    assert len(attrs_filtered) == 1
    assert attrs_filtered[0]["type"] == "summary"
    assert store.get_block_timestamp("b1") == 1.0
    assert store.get_block_history("nope") == []
    assert store.get_block_timestamp("nope") is None


async def test_async_history_utilities(tmp_path: Path) -> None:
    store = BaseHistoryStore(db_url=f"sqlite:///{tmp_path}/util_async.sqlite")

    await store.aadd_messages(
        [(HumanMessage("m1"), 1.1)],
        block_id="b1",
        block_ts=1.0,
    )
    await store.aadd_messages(
        [(HumanMessage("m2"), 2.1)],
        block_id="b2",
        block_ts=2.0,
    )
    await store.aadd_messages(
        [(HumanMessage("m3"), 1.2)],
        block_id="b1",
        block_ts=1.0,
    )
    await store.aadd_block_attributes(
        [{"type": "summary", "value": "s"}],
        block_id="b1",
        block_ts=1.0,
    )

    res = await store.aget_block_history_with_timestamps("b1")
    assert [m.content for m, _ in res] == ["m1", "m3"]
    assert len(await store.aget_block_histories(["b2", "b1"])) == 3
    assert len(await store.aget_block_histories_with_timestamps(["b2", "b1"])) == 3
    attrs_all = await store.aget_block_attributes("b1")
    assert len(attrs_all) == 1
    ts = await store.aget_block_timestamp("b1")
    assert ts == 1.0


def test_session_range_utilities(tmp_path: Path) -> None:
    store = BaseHistoryStore(db_url=f"sqlite:///{tmp_path}/range.sqlite")

    store.add_messages(
        [(HumanMessage("m1"), 1.1)],
        block_id="b1",
        block_ts=1.0,
        session_id="s",
    )
    store.add_messages(
        [(HumanMessage("m2"), 2.1)],
        block_id="b2",
        block_ts=2.0,
        session_id="s",
    )
    store.add_messages(
        [(HumanMessage("m3"), 3.1)],
        block_id="b3",
        block_ts=3.0,
        session_id="s",
    )

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
    store = BaseHistoryStore(db_url=f"sqlite:///{tmp_path}/range_async.sqlite")

    await store.aadd_messages(
        [(HumanMessage("m1"), 1.1)],
        block_id="b1",
        block_ts=1.0,
        session_id="s",
    )
    await store.aadd_messages(
        [(HumanMessage("m2"), 2.1)],
        block_id="b2",
        block_ts=2.0,
        session_id="s",
    )
    await store.aadd_messages(
        [(HumanMessage("m3"), 3.1)],
        block_id="b3",
        block_ts=3.0,
        session_id="s",
    )

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


def test_histories_with_multiple_blocks_order(tmp_path: Path) -> None:
    """Messages from multiple blocks are returned chronologically."""

    store = BaseHistoryStore(db_url=f"sqlite:///{tmp_path}/multi.sqlite")

    store.add_messages([(HumanMessage("b1-1"), 1.1)], block_id="b1", block_ts=1.0)
    store.add_messages([(HumanMessage("b1-2"), 1.3)], block_id="b1", block_ts=1.0)
    store.add_messages([(HumanMessage("b2-1"), 0.6)], block_id="b2", block_ts=0.5)
    store.add_messages([(HumanMessage("b2-2"), 1.2)], block_id="b2", block_ts=0.5)
    store.add_messages([(HumanMessage("b2-3"), 1.4)], block_id="b2", block_ts=0.5)

    result = store.get_block_histories_with_timestamps(["b2", "b1"])
    assert [m.content for m, _ in result] == [
        "b2-1",
        "b1-1",
        "b2-2",
        "b1-2",
        "b2-3",
    ]


def test_histories_with_multiple_blocks_order_no_ts(tmp_path: Path) -> None:
    """`get_histories` preserves chronological ordering across blocks."""

    store = BaseHistoryStore(db_url=f"sqlite:///{tmp_path}/multi_no_ts.sqlite")

    store.add_messages([(HumanMessage("b1-1"), 1.1)], block_id="b1", block_ts=1.0)
    store.add_messages([(HumanMessage("b1-2"), 1.3)], block_id="b1", block_ts=1.0)
    store.add_messages([(HumanMessage("b2-1"), 0.6)], block_id="b2", block_ts=0.5)
    store.add_messages([(HumanMessage("b2-2"), 1.2)], block_id="b2", block_ts=0.5)
    store.add_messages([(HumanMessage("b2-3"), 1.4)], block_id="b2", block_ts=0.5)

    result = store.get_block_histories(["b2", "b1"])
    assert [m.content for m in result] == [
        "b2-1",
        "b1-1",
        "b2-2",
        "b1-2",
        "b2-3",
    ]


async def test_async_histories_with_multiple_blocks_order(tmp_path: Path) -> None:
    """Async variant of multi-block history ordering."""

    store = BaseHistoryStore(db_url=f"sqlite:///{tmp_path}/multi_async.sqlite")

    await store.aadd_messages([(HumanMessage("b1-1"), 1.1)], block_id="b1", block_ts=1.0)
    await store.aadd_messages([(HumanMessage("b1-2"), 1.3)], block_id="b1", block_ts=1.0)
    await store.aadd_messages([(HumanMessage("b2-1"), 0.6)], block_id="b2", block_ts=0.5)
    await store.aadd_messages([(HumanMessage("b2-2"), 1.2)], block_id="b2", block_ts=0.5)
    await store.aadd_messages([(HumanMessage("b2-3"), 1.4)], block_id="b2", block_ts=0.5)

    result = await store.aget_block_histories_with_timestamps(["b2", "b1"])
    assert [m.content for m, _ in result] == [
        "b2-1",
        "b1-1",
        "b2-2",
        "b1-2",
        "b2-3",
    ]


async def test_async_histories_with_multiple_blocks_order_no_ts(tmp_path: Path) -> None:
    """Async version of `get_histories` ordering test."""

    store = BaseHistoryStore(db_url=f"sqlite:///{tmp_path}/multi_async_no_ts.sqlite")

    await store.aadd_messages([(HumanMessage("b1-1"), 1.1)], block_id="b1", block_ts=1.0)
    await store.aadd_messages([(HumanMessage("b1-2"), 1.3)], block_id="b1", block_ts=1.0)
    await store.aadd_messages([(HumanMessage("b2-1"), 0.6)], block_id="b2", block_ts=0.5)
    await store.aadd_messages([(HumanMessage("b2-2"), 1.2)], block_id="b2", block_ts=0.5)
    await store.aadd_messages([(HumanMessage("b2-3"), 1.4)], block_id="b2", block_ts=0.5)

    result = await store.aget_block_histories(["b2", "b1"])
    assert [m.content for m in result] == [
        "b2-1",
        "b1-1",
        "b2-2",
        "b1-2",
        "b2-3",
    ]


def test_delete_session(tmp_path: Path) -> None:
    store = BaseHistoryStore(db_url=f"sqlite:///{tmp_path}/del_session.sqlite")
    store.add_messages([(HumanMessage("hi"), 1.0)], block_id="b1", block_ts=1.0, session_id="s")
    store.add_block_attributes(
        [{"type": "tag", "value": "v"}], block_id="b1", block_ts=1.0, session_id="s"
    )
    store.add_session_attributes([{"type": "role", "value": "admin"}], session_id="s")
    store.add_messages([(HumanMessage("other"), 2.0)], block_id="b2", block_ts=2.0, session_id="o")
    store.delete_session("s")

    with store._session_local() as session:
        assert session.query(User).filter_by(session_id="s").first() is None
        assert session.query(ChatBlock).filter_by(block_id="b1").first() is None
        assert session.query(ChatMessage).count() == 1
        assert session.query(ChatBlockAttribute).count() == 0
        assert session.query(User).filter_by(session_id="o").count() == 1


async def test_adelete_session(tmp_path: Path) -> None:
    store = BaseHistoryStore(db_url=f"sqlite:///{tmp_path}/adel_session.sqlite")
    await store.aadd_messages(
        [(HumanMessage("hi"), 1.0)], block_id="b1", block_ts=1.0, session_id="s"
    )
    await store.aadd_block_attributes(
        [{"type": "tag", "value": "v"}], block_id="b1", block_ts=1.0, session_id="s"
    )
    await store.aadd_session_attributes([{"type": "role", "value": "admin"}], session_id="s")
    await store.aadd_messages(
        [(HumanMessage("other"), 2.0)], block_id="b2", block_ts=2.0, session_id="o"
    )
    await store.adelete_session("s")

    async with store._async_session_local() as session:
        result = await session.execute(select(User).filter_by(session_id="s"))
        assert result.scalar_one_or_none() is None
        result_blocks = await session.execute(select(ChatBlock))
        assert len(result_blocks.scalars().all()) == 1
        result_attrs = await session.execute(select(ChatBlockAttribute))
        assert result_attrs.scalars().all() == []


def test_delete_block(tmp_path: Path) -> None:
    store = BaseHistoryStore(db_url=f"sqlite:///{tmp_path}/del_block.sqlite")
    store.add_messages([(HumanMessage("a"), 1.0)], block_id="b1", block_ts=1.0)
    store.add_block_attributes([{"type": "tag", "value": "v"}], block_id="b1", block_ts=1.0)
    store.add_messages([(HumanMessage("b"), 2.0)], block_id="b2", block_ts=2.0)
    store.delete_block("b1")

    with store._session_local() as session:
        assert session.query(ChatBlock).filter_by(block_id="b1").first() is None
        assert session.query(ChatBlock).filter_by(block_id="b2").count() == 1
        assert session.query(ChatMessage).count() == 1
        assert session.query(ChatBlockAttribute).count() == 0


async def test_adelete_block(tmp_path: Path) -> None:
    store = BaseHistoryStore(db_url=f"sqlite:///{tmp_path}/adel_block.sqlite")
    await store.aadd_messages([(HumanMessage("a"), 1.0)], block_id="b1", block_ts=1.0)
    await store.aadd_block_attributes([{"type": "tag", "value": "v"}], block_id="b1", block_ts=1.0)
    await store.aadd_messages([(HumanMessage("b"), 2.0)], block_id="b2", block_ts=2.0)
    await store.adelete_block("b1")

    async with store._async_session_local() as session:
        result_block = await session.execute(select(ChatBlock).filter_by(block_id="b1"))
        assert result_block.scalar_one_or_none() is None
        result_msgs = await session.execute(select(ChatMessage))
        assert len(result_msgs.scalars().all()) == 1
        result_attrs = await session.execute(select(ChatBlockAttribute))
        assert result_attrs.scalars().all() == []


def test_delete_session_attributes_filter(tmp_path: Path) -> None:
    store = BaseHistoryStore(db_url=f"sqlite:///{tmp_path}/del_sattr.sqlite")
    store.add_session_attributes(
        [{"type": "color", "value": "blue"}, {"type": "role", "value": "admin"}],
        session_id="s",
    )
    store.delete_session_attributes("s", types=["color"])
    with store._session_local() as session:
        user = session.query(User).filter_by(session_id="s").first()
        assert user is not None
        assert {a.type for a in user.attributes} == {"role"}
    store.delete_session_attributes("s")
    with store._session_local() as session:
        user = session.query(User).filter_by(session_id="s").first()
        assert user is not None
        assert user.attributes == []


def test_delete_block_attributes_filter(tmp_path: Path) -> None:
    store = BaseHistoryStore(db_url=f"sqlite:///{tmp_path}/del_battr.sqlite")
    store.add_messages([(HumanMessage("a"), 1.0)], block_id="b1", block_ts=1.0)
    store.add_block_attributes(
        [{"type": "tag", "value": "v"}, {"type": "summary", "value": "s"}],
        block_id="b1",
        block_ts=1.0,
    )
    store.delete_block_attributes("b1", types=["tag"])
    with store._session_local() as session:
        block = session.query(ChatBlock).filter_by(block_id="b1").first()
        assert block is not None
        assert {a.type for a in block.attributes} == {"summary"}
    store.delete_block_attributes("b1")
    with store._session_local() as session:
        block = session.query(ChatBlock).filter_by(block_id="b1").first()
        assert block is not None
        assert block.attributes == []


async def test_adelete_session_attributes_filter(tmp_path: Path) -> None:
    store = BaseHistoryStore(db_url=f"sqlite:///{tmp_path}/adel_sattr.sqlite")
    await store.aadd_session_attributes(
        [
            {"type": "color", "value": "blue"},
            {"type": "role", "value": "admin"},
        ],
        session_id="s",
    )
    await store.adelete_session_attributes("s", types=["role"])
    async with store._async_session_local() as session:
        result = await session.execute(select(User).filter_by(session_id="s"))
        user = result.scalar_one()
        assert {a.type for a in user.attributes} == {"color"}
    await store.adelete_session_attributes("s")
    async with store._async_session_local() as session:
        result = await session.execute(select(User).filter_by(session_id="s"))
        user = result.scalar_one()
        assert user.attributes == []


async def test_adelete_block_attributes_filter(tmp_path: Path) -> None:
    store = BaseHistoryStore(db_url=f"sqlite:///{tmp_path}/adel_battr.sqlite")
    await store.aadd_messages([(HumanMessage("a"), 1.0)], block_id="b1", block_ts=1.0)
    await store.aadd_block_attributes(
        [
            {"type": "tag", "value": "v"},
            {"type": "summary", "value": "s"},
        ],
        block_id="b1",
        block_ts=1.0,
    )
    await store.adelete_block_attributes("b1", types=["summary"])
    async with store._async_session_local() as session:
        result = await session.execute(select(ChatBlock).filter_by(block_id="b1"))
        block = result.scalar_one()
        assert {a.type for a in block.attributes} == {"tag"}
    await store.adelete_block_attributes("b1")
    async with store._async_session_local() as session:
        result = await session.execute(select(ChatBlock).filter_by(block_id="b1"))
        block = result.scalar_one()
        assert block.attributes == []


def test_delete_session_blocks(tmp_path: Path) -> None:
    store = BaseHistoryStore(db_url=f"sqlite:///{tmp_path}/del_sblocks.sqlite")
    store.add_messages([(HumanMessage("hi"), 1.0)], block_id="b1", block_ts=1.0, session_id="s")
    store.add_block_attributes(
        [{"type": "tag", "value": "v"}], block_id="b1", block_ts=1.0, session_id="s"
    )
    store.add_messages([(HumanMessage("other"), 2.0)], block_id="b2", block_ts=2.0, session_id="o")
    store.delete_session_blocks("s")

    with store._session_local() as session:
        user_s = session.query(User).filter_by(session_id="s").first()
        user_o = session.query(User).filter_by(session_id="o").first()
        assert user_s is not None
        assert session.query(ChatBlock).filter_by(session_id=user_s.id).count() == 0
        assert session.query(ChatMessage).filter_by(session_id=user_s.id).count() == 0
        assert session.query(ChatBlockAttribute).count() == 0
        assert user_o is not None
        assert session.query(ChatBlock).filter_by(session_id=user_o.id).count() == 1


async def test_adelete_session_blocks(tmp_path: Path) -> None:
    store = BaseHistoryStore(db_url=f"sqlite:///{tmp_path}/adel_sblocks.sqlite")
    await store.aadd_messages(
        [(HumanMessage("hi"), 1.0)], block_id="b1", block_ts=1.0, session_id="s"
    )
    await store.aadd_block_attributes(
        [{"type": "tag", "value": "v"}], block_id="b1", block_ts=1.0, session_id="s"
    )
    await store.aadd_messages(
        [(HumanMessage("other"), 2.0)], block_id="b2", block_ts=2.0, session_id="o"
    )
    await store.adelete_session_blocks("s")

    async with store._async_session_local() as session:
        result_user_s = await session.execute(select(User).filter_by(session_id="s"))
        user_s = result_user_s.scalar_one()
        result_user_o = await session.execute(select(User).filter_by(session_id="o"))
        user_o = result_user_o.scalar_one()
        result_blocks_s = await session.execute(select(ChatBlock).filter_by(session_id=user_s.id))
        assert result_blocks_s.scalars().all() == []
        result_msgs_s = await session.execute(select(ChatMessage).filter_by(session_id=user_s.id))
        assert result_msgs_s.scalars().all() == []
        result_attrs = await session.execute(select(ChatBlockAttribute))
        assert result_attrs.scalars().all() == []
        result_blocks_o = await session.execute(select(ChatBlock).filter_by(session_id=user_o.id))
        assert len(result_blocks_o.scalars().all()) == 1


def test_delete_session_histories(tmp_path: Path) -> None:
    store = BaseHistoryStore(db_url=f"sqlite:///{tmp_path}/del_shist.sqlite")
    store.add_messages([(HumanMessage("a"), 1.0)], block_id="b1", block_ts=1.0, session_id="s")
    store.add_messages([(HumanMessage("b"), 2.0)], block_id="b2", block_ts=2.0, session_id="s")
    store.add_messages([(HumanMessage("c"), 3.0)], block_id="b3", block_ts=3.0, session_id="o")
    store.delete_session_histories("s")

    with store._session_local() as session:
        user_s = session.query(User).filter_by(session_id="s").first()
        user_o = session.query(User).filter_by(session_id="o").first()
        assert user_s is not None
        assert session.query(ChatMessage).filter_by(session_id=user_s.id).count() == 0
        assert session.query(ChatBlock).filter_by(session_id=user_s.id).count() == 2
        assert user_o is not None
        assert session.query(ChatMessage).filter_by(session_id=user_o.id).count() == 1


async def test_adelete_session_histories(tmp_path: Path) -> None:
    store = BaseHistoryStore(db_url=f"sqlite:///{tmp_path}/adel_shist.sqlite")
    await store.aadd_messages(
        [(HumanMessage("a"), 1.0)], block_id="b1", block_ts=1.0, session_id="s"
    )
    await store.aadd_messages(
        [(HumanMessage("b"), 2.0)], block_id="b2", block_ts=2.0, session_id="s"
    )
    await store.aadd_messages(
        [(HumanMessage("c"), 3.0)], block_id="b3", block_ts=3.0, session_id="o"
    )
    await store.adelete_session_histories("s")

    async with store._async_session_local() as session:
        result_user_s = await session.execute(select(User).filter_by(session_id="s"))
        user_s = result_user_s.scalar_one()
        result_user_o = await session.execute(select(User).filter_by(session_id="o"))
        user_o = result_user_o.scalar_one()
        result_msgs_s = await session.execute(select(ChatMessage).filter_by(session_id=user_s.id))
        assert result_msgs_s.scalars().all() == []
        result_blocks_s = await session.execute(select(ChatBlock).filter_by(session_id=user_s.id))
        assert len(result_blocks_s.scalars().all()) == 2
        result_msgs_o = await session.execute(select(ChatMessage).filter_by(session_id=user_o.id))
        assert len(result_msgs_o.scalars().all()) == 1
