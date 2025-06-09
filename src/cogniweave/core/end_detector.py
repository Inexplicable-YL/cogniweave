from typing import TYPE_CHECKING, Any, Literal, Self, cast
from typing_extensions import override

from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnableSerializable
from pydantic import BaseModel, Field, model_validator

from cogniweave.core.prompt_values.end_detector import EndDetectorPromptValue
from cogniweave.llms import PydanticSingleTurnChat
from cogniweave.prompt_values import MultilingualSystemPromptValue
from cogniweave.utils import get_model_from_env, get_provider_from_env

if TYPE_CHECKING:
    from langchain_core.runnables.config import RunnableConfig


class ConversationEndResult(BaseModel):
    """Pydantic model for conversation end detection result."""

    end: bool = Field(..., description="Whether the user wants to end the conversation.")


class ConversationEndClassifier(PydanticSingleTurnChat[Literal["zh", "en"], ConversationEndResult]):
    """Conversation end detector."""

    provider: str = Field(
        default_factory=get_provider_from_env("END_DETECTOR_MODEL", default="openai")
    )
    model_name: str = Field(
        default_factory=get_model_from_env("END_DETECTOR_MODEL", default="gpt-4.1-mini")
    )
    temperature: float = 1.0  # add some randomness, does not affect that much tbh.

    prompt: MultilingualSystemPromptValue[Literal["zh", "en"]] | None = Field(
        default=EndDetectorPromptValue()
    )


class ConversationEndDetector(RunnableSerializable[dict[str, Any], bool]):
    """Conversation end detector."""

    lang: Literal["zh", "en"] = Field(default="zh")

    classifier: ConversationEndClassifier | None = None
    messages_variable_key: str = Field(default="messages")

    @staticmethod
    def _serialize_messages(messages: list[Any]) -> str:
        return get_buffer_string(messages, human_prefix="[User]", ai_prefix="[Assistant]")

    @model_validator(mode="after")
    def _build_chain_if_needed(self) -> "Self":
        self.classifier = self.classifier or ConversationEndClassifier(lang=self.lang)
        return self

    @override
    def invoke(
        self,
        input: dict[str, Any],
        config: "RunnableConfig | None" = None,
        **kwargs: Any,
    ) -> bool:
        messages = cast("list[Any]", input.get(self.messages_variable_key, []))
        if not isinstance(messages, list):
            raise TypeError(
                f"Expected list for '{self.messages_variable_key}', got {type(messages)}",
            )

        serialized = self._serialize_messages(messages)
        assert self.classifier is not None, "Inner classifier not initialized"
        return self.classifier.invoke({"input": serialized}, config=config, **kwargs).end

    @override
    async def ainvoke(
        self,
        input: dict[str, Any],
        config: "RunnableConfig | None" = None,
        **kwargs: Any,
    ) -> bool:
        messages = cast("list[Any]", input.get(self.messages_variable_key, []))
        if not isinstance(messages, list):
            raise TypeError(
                f"Expected list for '{self.messages_variable_key}', got {type(messages)}",
            )

        serialized = self._serialize_messages(messages)
        assert self.classifier is not None, "Inner classifier not initialized"
        return (await self.classifier.ainvoke({"input": serialized}, config=config, **kwargs)).end
