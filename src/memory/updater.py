from datetime import datetime
from typing import Any, Literal, Self, cast
from typing_extensions import override

import anyio
from langchain_core.messages import get_buffer_string
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
from src.prompt_values.tagger import (
    SHORT_TERM_MEMORY_TAGS_EN,
    SHORT_TERM_MEMORY_TAGS_ZH,
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
                    SHORT_TERM_MEMORY_TAGS_ZH
                )
            else:
                values["prompt"] = SystemMessagePromptTemplate.from_template(
                    SHORT_TERM_MEMORY_TAGS_EN
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

    def _format_message(
        self,
        **kwargs: Any,
    ) -> str:
        """Format the message for the model."""
        name = kwargs.get(self.name_variable_key)
        if not isinstance(name, str):
            raise TypeError(f"Expected a string for {self.name_variable_key}, got {type(name)}")
        history = kwargs.get(self.history_variable_key)
        if not isinstance(history, list):
            raise TypeError(f"Expected a list for {self.history_variable_key}, got {type(history)}")
        return f"UserName: {name}\nChatHistory: \n" + get_buffer_string(
            history, human_prefix="[User]", ai_prefix="[Assistant]"
        )

    @override
    def invoke(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> ShortMemoryPromptTemplate:
        """Get the short-term memory from the model."""
        assert self.memory_chain is not None
        assert self.tags_chain is not None

        message = self._format_message(**input)
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
        message = self._format_message(**input)

        chat_summary_result: str | None = None
        topic_tags_result: list[str] | None = None

        async def _get_chat_summary() -> None:
            nonlocal chat_summary_result
            assert self.memory_chain is not None
            chat_summary_result = await self.memory_chain.ainvoke(
                {"input": message}, config=config, **kwargs
            )

        async def _get_topic_tags() -> None:
            nonlocal topic_tags_result
            assert self.tags_chain is not None
            topic_tags_result = (
                await self.tags_chain.ainvoke({"input": message}, config=config, **kwargs)
            ).tags

        async with anyio.create_task_group() as tg:
            tg.start_soon(_get_chat_summary)
            tg.start_soon(_get_topic_tags)

        if not chat_summary_result:
            raise ValueError("Chat summary result is None, please check the model configuration.")
        if not topic_tags_result:
            raise ValueError("Topic tags result is None, please check the model configuration.")

        return ShortMemoryPromptTemplate.from_template(
            timestamp=datetime.fromtimestamp(cast("int | float", input.get("timestamp"))),
            chat_summary=chat_summary_result,
            topic_tags=topic_tags_result,
        )
