from typing import Any
from typing_extensions import override

from langchain_core.runnables.config import RunnableConfig
from pydantic import Field

from cogniweave.core.time_splitter.base import BaseTimeSplitter

from .manager import ConditionDensityManager, DensityStrategy


class TimeSplitter(BaseTimeSplitter):
    manager: ConditionDensityManager
    id_timestamps: dict[str, tuple[str, float]] = Field(default_factory=dict)

    def __init__(
        self,
        *,
        time_window: float | None = None,
        density_strategy: DensityStrategy = DensityStrategy.AUTO,
        segment_factor: float = 0.5,
        segment_min: float = 60.0,
        segment_max: float = 3600.0,
        std_multiplier: float = 0.5,
        **kwargs: Any,
    ) -> None:
        manager = ConditionDensityManager(
            time_window=time_window,
            density_strategy=density_strategy,
            segment_factor=segment_factor,
            segment_min=segment_min,
            segment_max=segment_max,
            std_multiplier=std_multiplier,
            **kwargs,
        )
        super().__init__(manager=manager)  # type: ignore

    @override
    def invoke(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> tuple[str, float]:
        """Get the context id and timestamp.

        Args:
            input (dict[str, Any]): Input data. Must contain a "timestamp" key.
            config (RunnableConfig | None, optional): Config data. Must contain a "configurable" key, with a "session_id" key.

        Returns:
            tuple[str, float]: Output data.
        """
        if "timestamp" not in input or not isinstance(input["timestamp"], float | int):
            raise ValueError("timestamp is required and must be a float or int object.")
        if not config:
            raise ValueError("config is required and must be a dict object.")
        session_id = config.get("configurable", {}).get("session_id", "")
        if not session_id:
            raise ValueError("session_id is required and must be a string object.")

        context_id = self.manager.update_condition_density(session_id, input["timestamp"])

        if (
            session_id in self.id_timestamps and self.id_timestamps[session_id][0] != context_id
        ) or session_id not in self.id_timestamps:
            self.id_timestamps[session_id] = (context_id, input["timestamp"])

        return (context_id, self.id_timestamps[session_id][1])

    @override
    async def ainvoke(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> tuple[str, float]:
        """Get the context id and timestamp.

        Args:
            input (dict[str, Any]): Input data. Must contain a "timestamp" key.
            config (RunnableConfig | None, optional): Config data. Must contain a "configurable" key, with a "session_id" key.

        Returns:
            tuple[str, float]: Output data.
        """
        if "timestamp" not in input or not isinstance(input["timestamp"], float | int):
            raise ValueError("timestamp is required and must be a float or int object.")
        if not config:
            raise ValueError("config is required and must be a dict object.")
        session_id = config.get("configurable", {}).get("session_id", "")
        if not session_id:
            raise ValueError("session_id is required and must be a string object.")

        context_id = await self.manager.aupdate_condition_density(session_id, input["timestamp"])

        if (
            session_id in self.id_timestamps and self.id_timestamps[session_id][0] != context_id
        ) or session_id not in self.id_timestamps:
            self.id_timestamps[session_id] = (context_id, input["timestamp"])

        return (context_id, self.id_timestamps[session_id][1])
