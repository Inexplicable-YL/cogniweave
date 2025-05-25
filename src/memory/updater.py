from datetime import datetime
from typing import Any, Literal, Self, cast
from typing_extensions import override

from langchain.schema import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field, model_validator

from src.llms import PydanticSingleTurnChat, StringSingleTurnChat
from src.prompt_values.summary import (
    SHORT_TERM_MEMORY_SUMMARY_EN,
    SHORT_TERM_MEMORY_SUMMARY_ZH,
)
from src.prompts.generator import ShortMemoryPromptTemplate
from src.utils import get_model_from_env, get_provider_from_env


class ShortTermMemoryChat(StringSingleTurnChat):
    """Short-term memory updater for chat models."""

    lang: Literal["en", "zh"] = Field(default="zh")

    provider: str = Field(
        default_factory=get_provider_from_env("SHORT_MEMORY_MODEL", default="openai")
    )
    model_name: str = Field(
        default_factory=get_model_from_env("SHORT_MEMORY_MODEL", default="gpt-4.1-mini")
    )
    temperature: float = 0.7

    @model_validator(mode="before")
    @classmethod
    def process_prompt_with_lang(
        cls,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        """Set the prompt based on the language."""
        if values.get("prompt") is None:
            if values["lang"] == "zh":
                values["prompt"] = SystemMessagePromptTemplate.from_template(
                    SHORT_TERM_MEMORY_SUMMARY_ZH
                )
            else:
                values["prompt"] = SystemMessagePromptTemplate.from_template(
                    SHORT_TERM_MEMORY_SUMMARY_EN
                )
        return values


class ContextTags(BaseModel):
    """Context tags for the chat model."""

    tags: list[str] = Field(default_factory=list)


class ShortTermTagsChat(PydanticSingleTurnChat[ContextTags]):
    """Short-term memory updater for chat models."""

    lang: Literal["en", "zh"] = Field(default="zh")

    provider: str = Field(
        default_factory=get_provider_from_env("SHORT_MEMORY_MODEL", default="openai")
    )
    model_name: str = Field(
        default_factory=get_model_from_env("SHORT_MEMORY_MODEL", default="gpt-4.1-mini")
    )
    temperature: float = 0.7

    @model_validator(mode="before")
    @classmethod
    def process_prompt_with_lang(
        cls,
        values: dict[str, Any],
    ) -> dict[str, Any]:
        """Set the prompt based on the language."""
        if values.get("prompt") is None:
            if values["lang"] == "zh":
                values["prompt"] = SystemMessagePromptTemplate.from_template(
                    ""  # 须补齐 prompt
                )
            else:
                values["prompt"] = SystemMessagePromptTemplate.from_template(
                    ""  # 须补齐 prompt
                )
        return values

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        """Initialize the chat model."""
        super().__init__(
            template=ContextTags,
            **kwargs,
        )


class ShortTermMemoryChatUpdater(RunnableSerializable[dict[str, Any], ShortMemoryPromptTemplate]):
    """Short-term memory updater for chat models."""

    lang: Literal["en", "zh"] = Field(default="zh")

    memory_chain: ShortTermMemoryChat | None = None
    tags_chain: ShortTermTagsChat | None = None

    name_variable_key: str = Field(default="name")
    history_variable_key: str = Field(default="history")

    @model_validator(mode="after")
    def build_chain_if_needed(self) -> Self:
        """Automatically build the chain if it is not provided."""
        self.memory_chain = self.memory_chain or ShortTermMemoryChat(
            lang=self.lang,
        )
        self.tags_chain = self.tags_chain or ShortTermTagsChat(
            lang=self.lang,
        )
        return self

    def _format_history(
        self,
        history: list[BaseMessage],
    ) -> str:
        """Format the chat history for the model."""
        chat_history: list[HumanMessage | AIMessage] = [
            msg for msg in history if isinstance(msg, (HumanMessage, AIMessage))
        ]
        return "\n".join(
            [
                f"[User]: {msg.content}"
                if isinstance(msg, HumanMessage)
                else f"[Assistant]: {msg.content}"
                for msg in chat_history
            ]
        )

    @override
    def invoke(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> ShortMemoryPromptTemplate:
        """Get the short-term memory from the model."""
        assert self.memory_chain is not None
        assert self.tags_chain is not None

        name = input.get(self.name_variable_key)
        history = input.get(self.history_variable_key)
        if not isinstance(history, list):
            raise TypeError(f"Expected a list for {self.history_variable_key}, got {type(history)}")
        message = f"UserName: {name}\nChatHistory: \n" + self._format_history(history)
        return ShortMemoryPromptTemplate.from_template(
            timestamp=datetime.fromtimestamp(cast("int | float", input.get("timestamp"))),
            chat_summary=self.memory_chain.invoke({"input": message}, config=config, **kwargs),
            topic_tags=self.tags_chain.invoke({"input": message}, config=config, **kwargs).tags,
        )

    @override
    async def ainvoke(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> ShortMemoryPromptTemplate:
        """Asynchronously get the short-term memory from the model."""
        assert self.memory_chain is not None
        assert self.tags_chain is not None

        name = input.get(self.name_variable_key)
        history = input.get(self.history_variable_key)
        if not isinstance(history, list):
            raise TypeError(f"Expected a list for {self.history_variable_key}, got {type(history)}")
        message = f"UserName: {name}\nChatHistory: \n" + self._format_history(history)
        return ShortMemoryPromptTemplate.from_template(
            timestamp=datetime.fromtimestamp(cast("int | float", input.get("timestamp"))),
            chat_summary=await self.memory_chain.ainvoke(
                {"input": message}, config=config, **kwargs
            ),
            topic_tags=[],
        )
