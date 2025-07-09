import asyncio
import time
from pathlib import Path
from typing import Any

from test_runnable import RunnableForTest

from cogniweave.core.time_splitter import TimeSplitter
from cogniweave.history_stores import BaseHistoryStore as HistoryStore
from cogniweave.core.runnables.history_store import RunnableWithHistoryStore

sleep_time = 0.1


def _get_input(i: int) -> Any:
    return (f"input_{i}", time.time())


def test_runnable_history(tmp_path: Path) -> None:
    test_runnable = RunnableForTest(sleep_time=sleep_time)
    test_runnable_with_history = RunnableWithHistoryStore(
        test_runnable,
        history_store=HistoryStore(db_url=f"sqlite:///{tmp_path}/test.sqlite"),
        time_splitter=TimeSplitter(),
        input_messages_key="input",
        auto_package=True,
    )
    for i in range(3):
        input = _get_input(i)
        result = test_runnable_with_history.invoke(
            {"input": input, "answer": f"answer_{i}"},
            config={"configurable": {"session_id": "bar"}},
        )
        assert result.content == f"answer_{i}"
        assert test_runnable._invoke_count == i + 1
        assert len(test_runnable._invoke_data["input"]) == i * 2 + 1
        time.sleep(sleep_time)


async def test_runnable_history_async(tmp_path: Path) -> None:
    test_runnable = RunnableForTest(sleep_time=sleep_time)
    test_runnable_with_history = RunnableWithHistoryStore(
        test_runnable,
        history_store=HistoryStore(db_url=f"sqlite:///{tmp_path}/test.sqlite"),
        time_splitter=TimeSplitter(),
        input_messages_key="input",
        auto_package=True,
    )
    for i in range(3):
        input = _get_input(i)
        result = await test_runnable_with_history.ainvoke(
            {"input": input, "answer": f"answer_{i}"},
            config={"configurable": {"session_id": "bar"}},
        )
        assert result.content == f"answer_{i}"
        assert test_runnable._async_invoke_count == i + 1
        assert len(test_runnable._async_invoke_data["input"]) == i * 2 + 1
        await asyncio.sleep(sleep_time)


def test_runnable_history_with_history_key(tmp_path: Path) -> None:
    test_runnable = RunnableForTest(sleep_time=sleep_time)
    test_runnable_with_history = RunnableWithHistoryStore(
        test_runnable,
        history_store=HistoryStore(db_url=f"sqlite:///{tmp_path}/test.sqlite"),
        time_splitter=TimeSplitter(),
        input_messages_key="input",
        history_messages_key="history",
        auto_package=True,
    )
    for i in range(3):
        input = _get_input(i)
        result = test_runnable_with_history.invoke(
            {"input": input, "answer": f"answer_{i}"},
            config={"configurable": {"session_id": "bar"}},
        )
        assert result.content == f"answer_{i}"
        assert test_runnable._invoke_count == i + 1
        assert test_runnable._invoke_data["input"] == input
        assert len(test_runnable._invoke_data["history"]) == i * 2
        time.sleep(sleep_time)


async def test_runnable_history_with_history_key_async(tmp_path: Path) -> None:
    test_runnable = RunnableForTest(sleep_time=sleep_time)
    test_runnable_with_history = RunnableWithHistoryStore(
        test_runnable,
        history_store=HistoryStore(db_url=f"sqlite:///{tmp_path}/test.sqlite"),
        time_splitter=TimeSplitter(),
        input_messages_key="input",
        history_messages_key="history",
        auto_package=True,
    )
    for i in range(3):
        input = _get_input(i)
        result = await test_runnable_with_history.ainvoke(
            {"input": input, "answer": f"answer_{i}"},
            config={"configurable": {"session_id": "bar"}},
        )
        assert result.content == f"answer_{i}"
        assert test_runnable._async_invoke_count == i + 1
        assert test_runnable._async_invoke_data["input"] == input
        assert len(test_runnable._async_invoke_data["history"]) == i * 2
        await asyncio.sleep(sleep_time)
