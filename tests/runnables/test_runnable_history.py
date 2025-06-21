import asyncio
import time
from pathlib import Path
from typing import Any
from typing_extensions import override

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig
from pydantic import PrivateAttr

from cogniweave.core.time_splitter import TimeSplitter
from cogniweave.history_store import BaseHistoryStore as HistoryStore
from cogniweave.runnables.history import RunnableWithHistory

sleep_time = 0.1


class TestRunnable(RunnableSerializable[dict[str, Any], BaseMessage]):
    """Test Runnable for RunnableWithHistory tests."""

    _invoke_data: dict[str, Any] = PrivateAttr(default_factory=dict)
    _invoke_count: int = PrivateAttr(default=0)
    _async_invoke_data: dict[str, Any] = PrivateAttr(default_factory=dict)
    _async_invoke_count: int = PrivateAttr(default=0)

    @override
    def invoke(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> BaseMessage:
        self._invoke_data = {**input, **(config or {})}
        self._invoke_count += 1
        time.sleep(sleep_time)
        return AIMessage(content=str(input["answer"]))

    @override
    async def ainvoke(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> BaseMessage:
        self._async_invoke_data = {**input, **(config or {})}
        self._async_invoke_count += 1
        await asyncio.sleep(sleep_time)
        return AIMessage(content=str(input["answer"]))


def test_runnable_history(tmp_path: Path) -> None:
    test_runnable = TestRunnable()
    test_runnable_with_history = RunnableWithHistory(
        test_runnable,
        history_store=HistoryStore(db_url=f"sqlite:///{tmp_path}/test.sqlite"),
        time_splitter=TimeSplitter(),
        input_messages_key="input",
        auto_package=True,
    )
    for i in range(3):
        now = time.time()
        result = test_runnable_with_history.invoke(
            {"input": (f"input_{i}", now), "answer": f"answer_{i}"},
            config={"configurable": {"session_id": "bar"}},
        )
        assert result.content == f"answer_{i}"
        assert test_runnable._invoke_count == i + 1
        assert len(test_runnable._invoke_data["input"]) == i * 2 + 1
        time.sleep(sleep_time)


async def test_runnable_history_async(tmp_path: Path) -> None:
    test_runnable = TestRunnable()
    test_runnable_with_history = RunnableWithHistory(
        test_runnable,
        history_store=HistoryStore(db_url=f"sqlite:///{tmp_path}/test.sqlite"),
        time_splitter=TimeSplitter(),
        input_messages_key="input",
        auto_package=True,
    )
    for i in range(3):
        now = time.time()
        result = await test_runnable_with_history.ainvoke(
            {"input": (f"input_{i}", now), "answer": f"answer_{i}"},
            config={"configurable": {"session_id": "bar"}},
        )
        assert result.content == f"answer_{i}"
        assert test_runnable._async_invoke_count == i + 1
        assert len(test_runnable._async_invoke_data["input"]) == i * 2 + 1
        await asyncio.sleep(sleep_time)


def test_runnable_history_with_history_key(tmp_path: Path) -> None:
    test_runnable = TestRunnable()
    test_runnable_with_history = RunnableWithHistory(
        test_runnable,
        history_store=HistoryStore(db_url=f"sqlite:///{tmp_path}/test.sqlite"),
        time_splitter=TimeSplitter(),
        input_messages_key="input",
        history_messages_key="history",
        auto_package=True,
    )
    for i in range(3):
        now = time.time()
        result = test_runnable_with_history.invoke(
            {"input": (f"input_{i}", now), "answer": f"answer_{i}"},
            config={"configurable": {"session_id": "bar"}},
        )
        assert result.content == f"answer_{i}"
        assert test_runnable._invoke_count == i + 1
        assert test_runnable._invoke_data["input"] == (f"input_{i}", now)
        assert len(test_runnable._invoke_data["history"]) == i * 2
        time.sleep(sleep_time)


async def test_runnable_history_with_history_key_async(tmp_path: Path) -> None:
    test_runnable = TestRunnable()
    test_runnable_with_history = RunnableWithHistory(
        test_runnable,
        history_store=HistoryStore(db_url=f"sqlite:///{tmp_path}/test.sqlite"),
        time_splitter=TimeSplitter(),
        input_messages_key="input",
        history_messages_key="history",
        auto_package=True,
    )
    for i in range(3):
        now = time.time()
        result = await test_runnable_with_history.ainvoke(
            {"input": (f"input_{i}", now), "answer": f"answer_{i}"},
            config={"configurable": {"session_id": "bar"}},
        )
        assert result.content == f"answer_{i}"
        assert test_runnable._async_invoke_count == i + 1
        assert test_runnable._async_invoke_data["input"] == (f"input_{i}", now)
        assert len(test_runnable._async_invoke_data["history"]) == i * 2
        await asyncio.sleep(sleep_time)
