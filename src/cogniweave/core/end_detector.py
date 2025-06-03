from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, cast
from typing_extensions import override

from langchain_core.messages import get_buffer_string
from pydantic import BaseModel, Field

from cogniweave.core.prompt_values.end_detector import EndDetectorPromptValue
from cogniweave.llms import PydanticSingleTurnChat
from cogniweave.utils import get_model_from_env, get_provider_from_env

if TYPE_CHECKING:
    from langchain_core.runnables.config import RunnableConfig


class ConversationEndResult(BaseModel):
    """Pydantic model for conversation end detection result."""

    end: bool = Field(..., description="Whether the user wants to end the conversation.")


class ConversationEndDetector(
    PydanticSingleTurnChat[Literal["zh", "en"], ConversationEndResult]
):
    """Detect whether the user wants to end the conversation.

    The model examines the latest user messages and predicts if the user is
    ending or intending to end the conversation. The synchronous / asynchronous
    `invoke` methods return a ``bool`` directly for convenience.
    """

    # Model configuration
    provider: str = Field(
        default_factory=get_provider_from_env("END_DETECTOR_MODEL", default="openai")
    )
    model_name: str = Field(
        default_factory=get_model_from_env(
            "END_DETECTOR_MODEL", default="gpt-4.1-nano"
        )
    )
    temperature: float = 0.0

    # Default language
    lang: Literal["zh", "en"] = Field(default="zh")

    prompt: EndDetectorPromptValue | None = Field(default=EndDetectorPromptValue())

    messages_variable_key: str = Field(default="messages")

    @staticmethod
    def _serialize_messages(messages: list[Any]) -> str:
        """Convert a list of messages to a single string for the LLM."""
        return get_buffer_string(messages, human_prefix="[User]", ai_prefix="[Assistant]")

    # ---------------------------------------------------------------------
    # Override invoke / ainvoke to return boolean instead of Pydantic model
    # ---------------------------------------------------------------------

    @override  # type: ignore[override]
    def invoke(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> bool:
        messages = cast(list[Any], input.get(self.messages_variable_key, []))
        if not isinstance(messages, list):
            raise TypeError(
                f"Expected list for '{self.messages_variable_key}', got {type(messages)}"
            )

        serialized = self._serialize_messages(messages)
        result = super().invoke({"input": serialized}, config=config, **kwargs)
        return cast(ConversationEndResult, result).end

    @override  # type: ignore[override]
    async def ainvoke(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
        input: dict[str, Any],
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> bool:
        messages = cast(list[Any], input.get(self.messages_variable_key, []))
        if not isinstance(messages, list):
            raise TypeError(
                f"Expected list for '{self.messages_variable_key}', got {type(messages)}"
            )

        serialized = self._serialize_messages(messages)
        result = await super().ainvoke({"input": serialized}, config=config, **kwargs)
        return cast(ConversationEndResult, result).end
