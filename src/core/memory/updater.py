import json
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Self, cast
from typing_extensions import override

import anyio
from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field, model_validator

from src.llms import PydanticSingleTurnChat, StringSingleTurnChat
from src.prompt_values.base import MultilingualSystemPromptValue
from src.prompt_values.long_memory import (
    LongTermMemoryExtractPromptValue,
    LongTermMemoryPromptValue,
)
from src.prompt_values.summary import ShortTermMemoryPromptValue
from src.prompt_values.tagger import ShortTermTagsPromptValue
from src.prompts.generator import ShortMemoryPromptTemplate
from src.utils import get_model_from_env, get_provider_from_env


class SummaryMaker(StringSingleTurnChat[Literal["zh", "en"]]):
    """Short-term memory updater for chat models."""

    provider: str = Field(
        default_factory=get_provider_from_env("SHORT_MEMORY_MODEL", default="openai")
    )
    model_name: str = Field(
        default_factory=get_model_from_env("SHORT_MEMORY_MODEL", default="gpt-4.1-mini")
    )
    temperature: float = 0.7

    prompt: MultilingualSystemPromptValue[Literal["zh", "en"]] | None = Field(
        default=ShortTermMemoryPromptValue()
    )


class ContextTags(BaseModel):
    """Context tags for the chat model."""

    tags: list[str] = Field(default_factory=list)


class TagsMaker(PydanticSingleTurnChat[Literal["zh", "en"], ContextTags]):
    """Short-term memory updater for chat models."""

    provider: str = Field(
        default_factory=get_provider_from_env("SHORT_MEMORY_MODEL", default="openai")
    )
    model_name: str = Field(
        default_factory=get_model_from_env("SHORT_MEMORY_MODEL", default="gpt-4.1-mini")
    )
    temperature: float = 0.7

    prompt: MultilingualSystemPromptValue[Literal["zh", "en"]] | None = Field(
        default=ShortTermTagsPromptValue()
    )


class ShortTermMemoryMaker(RunnableSerializable[dict[str, Any], ShortMemoryPromptTemplate]):
    """Short-term memory updater for chat models."""

    lang: Literal["zh", "en"] = Field(default="zh")

    memory_chain: SummaryMaker | None = None
    tags_chain: TagsMaker | None = None

    name_variable_key: str = Field(default="name")
    history_variable_key: str = Field(default="history")

    @model_validator(mode="after")
    def build_chain_if_needed(self) -> Self:
        """Automatically build the chain if it is not provided."""
        self.memory_chain = self.memory_chain or SummaryMaker(
            lang=self.lang,
        )
        self.tags_chain = self.tags_chain or TagsMaker(
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


class LongTermMemoryChat(StringSingleTurnChat[Literal["zh", "en"]]):
    """Long-term memory summarize"""

    provider: str = Field(
        default_factory=get_provider_from_env("LONG_MEMORY_MODEL", default="openai")
    )
    model_name: str = Field(
        default_factory=get_model_from_env("LONG_MEMORY_MODEL", default="o4-mini")
    )
    temperature: float = 1

    prompt: MultilingualSystemPromptValue[Literal["zh", "en"]] | None = Field(
        default=LongTermMemoryPromptValue()
    )


class LongTermMemoryUpdater(RunnableSerializable[dict[str, Any], str]):
    """Update and persist the long-term memory prompt."""

    lang: Literal["zh", "en"] = Field(default="zh")
    chat_chain: LongTermMemoryChat | None = None
    extract_chain: LongTermMemoryChat | None = None

    store_path: str = Field(default="./data/long_memory.json")

    history_variable_key: str = Field(default="history")

    @model_validator(mode="after")
    def build_chain_if_needed(self) -> Self:
        self.extract_chain = self.extract_chain or LongTermMemoryChat(
            lang=self.lang,
            prompt=LongTermMemoryExtractPromptValue(),
            model_name="o3",
            temperature=1.0,
        )

        self.chat_chain = self.chat_chain or LongTermMemoryChat(
            lang=self.lang,
            prompt=LongTermMemoryPromptValue(),
            model_name="gpt-4.1-mini",
            temperature=0.5,
        )
        return self

    def _load_long_memory(self) -> str:
        path = Path(self.store_path)
        if not path.exists():
            raise ValueError(f"Long-term memory file not found: {self.store_path}")
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            raw = data.get("prompt", "") if isinstance(data, dict) else ""
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to decode long-term memory JSON: {e}") from e
        if isinstance(raw, list):
            return json.dumps(raw, ensure_ascii=False, indent=2)
        # Empty string is considered an error
        if not raw:
            raise ValueError("Loaded long-term memory is empty")
        return raw

    def _save_long_memory(self, content: str) -> None:
        path = Path(self.store_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            prompt_value = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            prompt_value = content
        with path.open("w", encoding="utf-8") as f:
            json.dump({"prompt": prompt_value}, f, ensure_ascii=False, indent=2)

    def _format_message(self, history: list[Any]) -> str:
        return get_buffer_string(history, human_prefix="[User]", ai_prefix="[Assistant]")

    @override
    def invoke(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> str:
        assert self.chat_chain is not None
        # Retreive new long-term memory items from ChatHistory only
        new_items_str = self.extract(input, config=config, **kwargs)
        # Read old long-term memory
        old_ltm = self._load_long_memory()
        # Merge new items with old items together
        prompt_input = {
            "input": (
                f"<NewLongTermMemory>\n{new_items_str}\n</NewLongTermMemory>\n"
                f"<CurrentLongTermMemory>\n{old_ltm}\n</CurrentLongTermMemory>"
            )
        }
        merged_ltm = self.chat_chain.invoke(prompt_input, config=config, **kwargs)
        # Save and return the result
        self._save_long_memory(merged_ltm)
        return merged_ltm

    def extract(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> str:
        """Extract new long-term memory items from ChatHistory only, without merging."""
        assert self.extract_chain is not None
        history = input.get(self.history_variable_key, [])
        if not isinstance(history, list):
            raise TypeError(f"Expected a list for {self.history_variable_key}, got {type(history)}")
        extract_input = {"input": f"<ChatHistory>\n{self._format_message(history)}\n</ChatHistory>"}
        return self.extract_chain.invoke(extract_input, config=config, **kwargs)

    @override
    async def ainvoke(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> str:
        assert self.chat_chain is not None
        new_items_str = await self.aextract(input, config=config, **kwargs)
        old_ltm = self._load_long_memory()
        prompt_input = {
            "input": (
                f"<NewLTM>\n{new_items_str}\n</NewLTM>\n<CurrentLTM>\n{old_ltm}\n</CurrentLTM>"
            )
        }
        merged_ltm = await self.chat_chain.ainvoke(prompt_input, config=config, **kwargs)
        self._save_long_memory(merged_ltm)
        return merged_ltm

    async def aextract(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> str:
        assert self.extract_chain is not None
        history = input.get(self.history_variable_key, [])
        if not isinstance(history, list):
            raise TypeError(f"Expected a list for {self.history_variable_key}, got {type(history)}")
        extract_input = {"input": f"<ChatHistory>\n{self._format_message(history)}\n</ChatHistory>"}
        return await self.extract_chain.ainvoke(extract_input, config=config, **kwargs)


class ShortTermMemoryUpdater(BaseModel):
    """Short-term memory updater for chat models."""

    lang: Literal["zh", "en"] = Field(default="zh")

    memory_maker: ShortTermMemoryMaker | None = None

    @model_validator(mode="after")
    def build_chain_if_needed(self) -> Self:
        """Automatically build the chain if it is not provided."""
        self.memory_maker = self.memory_maker or ShortTermMemoryMaker(lang=self.lang)
        return self
