from abc import ABC, abstractmethod
from typing import Any
from typing_extensions import override

from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig


class BaseTimeSplitter(RunnableSerializable[dict[str, Any], tuple[str, float]], ABC):
    @abstractmethod
    @override
    def invoke(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> tuple[str, float]:
        """Get the context id and timestamp.

        Args:
            input (dict[str, Any]): Input data. Must contain a "timestamp" key.
            config (RunnableConfig | None, optional): Config data. Must contain a "configurable" key, with a "session_id" key.

        Returns:
            SplitterOutput: Output data.
        """
        raise NotImplementedError

    @abstractmethod
    @override
    async def ainvoke(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> tuple[str, float]:
        """Get the context id and timestamp.

        Args:
            input (dict[str, Any]): Input data. Must contain a "timestamp" key.
            config (RunnableConfig | None, optional): Config data. Must contain a "configurable" key, with a "session_id" key.

        Returns:
            SplitterOutput: Output data.
        """
        raise NotImplementedError
