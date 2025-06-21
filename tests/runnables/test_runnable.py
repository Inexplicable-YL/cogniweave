import asyncio
import time
from typing import Any
from typing_extensions import override

from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig
from pydantic import PrivateAttr


class RunnableForTest(RunnableSerializable[dict[str, Any], BaseMessage]):
    """Test Runnable for RunnableWithHistory tests."""

    sleep_time: float = 0.1

    _invoke_data: dict[str, Any] = PrivateAttr(default_factory=dict)
    _invoke_count: int = PrivateAttr(default=0)
    _async_invoke_data: dict[str, Any] = PrivateAttr(default_factory=dict)
    _async_invoke_count: int = PrivateAttr(default=0)

    def __init__(
        self,
        sleep_time: float = 0.1,
        **kwargs: Any,
    ) -> None:
        super().__init__(sleep_time=sleep_time, **kwargs)  # type: ignore

    @override
    def invoke(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> BaseMessage:
        self._invoke_data = {**input, **(config or {})}
        self._invoke_count += 1
        time.sleep(self.sleep_time)
        return AIMessage(content=str(input["answer"]))

    @override
    async def ainvoke(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> BaseMessage:
        self._async_invoke_data = {**input, **(config or {})}
        self._async_invoke_count += 1
        await asyncio.sleep(self.sleep_time)
        return AIMessage(content=str(input["answer"]))
