from datetime import datetime
from typing import Any, Literal, Self, cast
from typing_extensions import override

from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field, model_validator

from cogniweave.core.prompt_values.long_memory import (
    LongTermMemoryExtractPromptValue,
    LongTermMemoryPromptValue,
)
from cogniweave.core.prompts.long_memory import (
    LongMemoryExtractPromptTemplate,
    LongMemoryMergePromptTemplate,
    LongMemoryPromptTemplate,
)
from cogniweave.llms import PydanticSingleTurnChat
from cogniweave.prompt_values import MultilingualSystemPromptValue
from cogniweave.utils import get_model_from_env, get_provider_from_env


class LongTermOutput(BaseModel):
    """Output structure for long-term memory summary."""

    updated_memory: list[str]


class LongTermPydanticSummary(PydanticSingleTurnChat[Literal["zh", "en"], LongTermOutput]):
    """Long-term memory chain that outputs a Pydantic model."""

    provider: str = Field(
        default_factory=get_provider_from_env("LONG_MEMORY_MODEL", default="openai")
    )
    model_name: str = Field(
        default_factory=get_model_from_env("LONG_MEMORY_MODEL", default="gpt-4.1-mini")
    )
    temperature: float = Field(default=1.0)
    prompt: MultilingualSystemPromptValue[Literal["zh", "en"]] | None = Field(
        default=LongTermMemoryPromptValue()
    )


# JSON chat chain for extraction
class LongTermJsonExtract(PydanticSingleTurnChat[Literal["zh", "en"], LongTermOutput]):
    """Long-term memory extraction chain that outputs JSON."""

    provider: str = Field(
        default_factory=get_provider_from_env("LONG_MEMORY_MODEL", default="openai")
    )
    model_name: str = Field(default_factory=get_model_from_env("LONG_MEMORY_MODEL", default="o3"))
    temperature: float = Field(default=1.0)
    prompt: MultilingualSystemPromptValue[Literal["zh", "en"]] | None = Field(
        default=LongTermMemoryExtractPromptValue()
    )


class LongTermMemoryMaker(RunnableSerializable[dict[str, Any], LongMemoryPromptTemplate]):
    """Generate updated long-term memory without persistence."""

    lang: Literal["zh", "en"] = Field(default="zh")
    chat_chain: LongTermPydanticSummary | None = None
    extract_chain: LongTermJsonExtract | None = None

    history_variable_key: str = Field(default="history")
    current_memory_template_variable_key: str = Field(default="current_memory_template")
    current_block_id_variable_key: str = Field(default="current_block_id")
    timestamp_variable_key: str = Field(default="timestamp")

    @model_validator(mode="after")
    def build_chain_if_needed(self) -> Self:
        # Use JSON-based extraction chain
        self.extract_chain = self.extract_chain or LongTermJsonExtract(lang=self.lang)
        self.chat_chain = self.chat_chain or LongTermPydanticSummary(lang=self.lang)

        return self

    def _get_current_timestamp(self, input: dict[str, Any]) -> str:
        """Get the current timestamp in the format: YYYY-MM-DD HH:MM"""
        if timestamp := input.get(self.timestamp_variable_key, None):
            return datetime.fromtimestamp(cast("float", timestamp)).strftime("%Y-%m-%d %H:%M")

        return datetime.now().strftime("%Y-%m-%d %H:%M")

    def _get_current_date(self, input: dict[str, Any]) -> str:
        """Get the current date in the format: YYYY-MM-DD"""
        if timestamp := input.get(self.timestamp_variable_key, None):
            return datetime.fromtimestamp(cast("float", timestamp)).strftime("%Y-%m-%d")

        return datetime.now().strftime("%Y-%m-%d")

    def _get_current_memory_template(self, input: dict[str, Any]) -> LongMemoryPromptTemplate:
        """Get the current memory from the prompt template."""
        input_val = input.get(self.current_memory_template_variable_key)
        if isinstance(input_val, LongMemoryPromptTemplate):
            return input_val
        if isinstance(input_val, dict):
            return LongMemoryPromptTemplate.load(input_val)
        msg = f"Expected dict or LongMemoryPromptTemplate. Got {input_val}."
        raise ValueError(msg)

    def _extract(
        self,
        input: dict[str, Any],
        current_time: str,
        current_date: str,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Extract new long-term memory items from chat history only, without merging."""
        assert self.extract_chain is not None

        history = input.get(self.history_variable_key, [])
        if not isinstance(history, list):
            raise TypeError(f"Expected a list for {self.history_variable_key}, got {type(history)}")

        history_text = get_buffer_string(history, human_prefix="[User]", ai_prefix="[Assistant]")

        time_kwargs: dict[str, Any] = {
            "current_time": current_time,
            "current_date": current_date,
        }
        extract_template = LongMemoryExtractPromptTemplate.from_template(
            history=history_text, **time_kwargs
        )
        result = self.extract_chain.invoke(
            {"input": extract_template.format(), **time_kwargs},
            config=config,
            **kwargs,
        )
        return result.updated_memory

    async def _a_extract(
        self,
        input: dict[str, Any],
        current_time: str,
        current_date: str,
        config: RunnableConfig | None = None,
        **kwargs: Any,
    ) -> list[str]:
        """Asynchronously extract new long-term memory items from chat history only, without merging."""
        assert self.extract_chain is not None

        history = input.get(self.history_variable_key, [])
        if not isinstance(history, list):
            raise TypeError(f"Expected a list for {self.history_variable_key}, got {type(history)}")

        history_text = get_buffer_string(history, human_prefix="[User]", ai_prefix="[Assistant]")

        time_kwargs: dict[str, Any] = {
            "current_time": current_time,
            "current_date": current_date,
        }
        extract_template = LongMemoryExtractPromptTemplate.from_template(
            history=history_text, **time_kwargs
        )
        result = await self.extract_chain.ainvoke(
            {"input": extract_template.format(), **time_kwargs},
            config=config,
            **kwargs,
        )
        return result.updated_memory

    @override
    def invoke(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> LongMemoryPromptTemplate:
        """Extract and merge long-term memory, returning a Pydantic model.

        Args:
            input: Dictionary containing:
                - history: (list[BaseMessage]) List of chat message history
                - current_memory_template: (LongMemoryPromptTemplate | dict) Current memory state
                - current_block_id: (str) ID of the current memory block
                - timestamp: (float) Unix timestamp
            config: Optional RunnableConfig for the chain execution
            **kwargs: Additional arguments to pass to the chain

        Returns:
            LongMemoryPromptTemplate containing:
                - current_memory: Updated memory content
                - updated_block_id: ID of the updated memory block
                - updated_time: Timestamp of the update

        Raises:
            TypeError: If input[history] is not list
            ValueError: If current_memory_template is invalid
        """
        assert self.chat_chain is not None

        current_time = self._get_current_timestamp(input)
        current_date = self._get_current_date(input)

        new_items_str = self._extract(input, current_time, current_date, config=config, **kwargs)

        current_memory_template = self._get_current_memory_template(input)
        current_block_id = input.get(self.current_block_id_variable_key, "")
        current_memory = current_memory_template.current_memory
        updated_time = current_memory_template.updated_time

        time_kwargs: dict[str, Any] = {
            "current_time": current_time,
            "current_date": current_date,
            "last_update_time": updated_time,
        }
        merge_template = LongMemoryMergePromptTemplate.from_template(
            new_memory=new_items_str, current_memory=current_memory, **time_kwargs
        )
        result = self.chat_chain.invoke(
            {"input": merge_template.format(), **time_kwargs}, config=config, **kwargs
        )

        return LongMemoryPromptTemplate.from_template(
            current_memory=result.updated_memory,
            updated_block_id=current_block_id,
            updated_time=current_time,
        )

    @override
    async def ainvoke(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> LongMemoryPromptTemplate:
        """Asynchronously extract and merge long-term memory.

        Args:
            input: Dictionary containing:
                - history: (list[BaseMessage]) List of chat message history
                - current_memory_template: (LongMemoryPromptTemplate | dict) Current memory state
                - current_block_id: (str) ID of the current memory block
                - timestamp: (float) Unix timestamp
            config: Optional RunnableConfig for the chain execution
            **kwargs: Additional arguments to pass to the chain

        Returns:
            LongMemoryPromptTemplate containing:
                - current_memory: Updated memory content
                - updated_block_id: ID of the updated memory block
                - updated_time: Timestamp of the update

        Raises:
            TypeError: If input[history] is not list
            ValueError: If current_memory_template is invalid
        """
        assert self.chat_chain is not None

        current_time = self._get_current_timestamp(input)
        current_date = self._get_current_date(input)

        new_items_str = await self._a_extract(
            input, current_time, current_date, config=config, **kwargs
        )

        current_memory_template = self._get_current_memory_template(input)
        current_block_id = input.get(self.current_block_id_variable_key, "")
        current_memory = current_memory_template.current_memory
        updated_time = current_memory_template.updated_time

        time_kwargs: dict[str, Any] = {
            "current_time": current_time,
            "current_date": current_date,
            "last_update_time": updated_time,
        }
        merge_template = LongMemoryMergePromptTemplate.from_template(
            new_memory=new_items_str, current_memory=current_memory, **time_kwargs
        )
        result = await self.chat_chain.ainvoke(
            {"input": merge_template.format(), **time_kwargs}, config=config, **kwargs
        )

        return LongMemoryPromptTemplate.from_template(
            current_memory=result.updated_memory,
            updated_block_id=current_block_id,
            updated_time=current_time,
        )
