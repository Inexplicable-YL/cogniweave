import json
from datetime import datetime
from typing import Any, Literal, Self
from typing_extensions import override

from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field, model_validator

from cogniweave.core.prompt_values.long_memory import (
    LongTermMemoryExtractPromptValue,
    LongTermMemoryPromptValue,
)
from cogniweave.llms import StringSingleTurnChat
from cogniweave.prompt_values import MultilingualSystemPromptValue
from cogniweave.utils import get_model_from_env, get_provider_from_env


class LongTermSummary(StringSingleTurnChat[Literal["zh", "en"]]):
    """Long-term memory summarize"""

    provider: str = Field(
        default_factory=get_provider_from_env("LONG_MEMORY_MODEL", default="openai")
    )
    model_name: str = Field(
        default_factory=get_model_from_env("LONG_MEMORY_MODEL", default="gpt-4.1-mini")
    )
    temperature: float = 1

    prompt: MultilingualSystemPromptValue[Literal["zh", "en"]] | None = Field(
        default=LongTermMemoryPromptValue()
    )


class LongTermMemoryResult(BaseModel):
    """Long-term memory processing result"""

    updated_memory: list[str] = Field(description="Updated memory items")
    timestamp: str = Field(description="Processing timestamp")
    extracted_items: list[str] = Field(description="Newly extracted memory items")

    @classmethod
    def from_strings(
        cls,
        updated_memory_str: str,
        extracted_items_str: str,
        timestamp: str
    ) -> "LongTermMemoryResult":
        """Create result from string outputs"""
        try:
            updated_memory = json.loads(updated_memory_str) if updated_memory_str else []
        except json.JSONDecodeError:
            # If not JSON, treat as single item
            updated_memory = [updated_memory_str] if updated_memory_str else []

        try:
            extracted_items = json.loads(extracted_items_str) if extracted_items_str else []
        except json.JSONDecodeError:
            # If not JSON, treat as single item
            extracted_items = [extracted_items_str] if extracted_items_str else []

        # Ensure list and convert elements to string
        if not isinstance(updated_memory, list):
            updated_memory = [updated_memory]
        updated_memory = [e if isinstance(e, str) else json.dumps(e, ensure_ascii=False) for e in updated_memory]
        if not isinstance(extracted_items, list):
            extracted_items = [extracted_items]
        extracted_items = [e if isinstance(e, str) else json.dumps(e, ensure_ascii=False) for e in extracted_items]

        return cls(
            updated_memory=updated_memory,
            extracted_items=extracted_items,
            timestamp=timestamp
        )


class LongTermMemoryMaker(RunnableSerializable[dict[str, Any], LongTermMemoryResult]):
    """Generate updated long-term memory without persistence."""

    lang: Literal["zh", "en"] = Field(default="zh")
    chat_chain: LongTermSummary | None = None
    extract_chain: LongTermSummary | None = None

    history_variable_key: str = Field(default="history")
    current_memory_variable_key: str = Field(default="current_long_term_memory")
    last_update_time_variable_key: str = Field(default="last_update_time")

    @model_validator(mode="after")
    def build_chain_if_needed(self) -> Self:
        self.extract_chain = self.extract_chain or LongTermSummary(
            lang=self.lang,
            prompt=LongTermMemoryExtractPromptValue(),
            model_name="o3",
            temperature=1.0,
        )

        self.chat_chain = self.chat_chain or LongTermSummary(
            lang=self.lang,
            prompt=LongTermMemoryPromptValue(),
            model_name="gpt-4.1-mini",
            temperature=0.5,
        )
        return self

    def _get_current_timestamp(self) -> str:
        """Get the current timestamp in the format: YYYY-MM-DD HH:MM"""
        return datetime.now().strftime("%Y-%m-%d %H:%M")

    def _get_current_date(self) -> str:
        """Get the current date in the format: YYYY-MM-DD"""
        return datetime.now().strftime("%Y-%m-%d")

    def _parse_current_memory(self, current_memory: Any) -> str:
        """Parse the current memory content to string format"""
        if isinstance(current_memory, list):
            return json.dumps(current_memory, ensure_ascii=False, indent=2)
        if isinstance(current_memory, str):
            return current_memory
        if current_memory is None:
            return "[]"
        return str(current_memory)

    def _format_message(self, history: list[Any]) -> str:
        return get_buffer_string(history, human_prefix="[User]", ai_prefix="[Assistant]")

    def _format_time_params(self, current_time: str, current_date: str, last_update: str | None) -> dict[str, str]:
        """Format time parameters for prompt"""
        params = {
            "current_time": current_time,
            "current_date": current_date,
        }
        if last_update:
            params["last_update_time"] = last_update
        else:
            params["last_update_time"] = "No record" if self.lang == "zh" else "No record"
        return params

    def _extract(
        self, input: dict[str, Any], current_time: str, current_date: str,
        config: RunnableConfig | None = None, **kwargs: Any
    ) -> str:
        """Extract new long-term memory items from ChatHistory only, without merging."""
        assert self.extract_chain is not None

        history = input.get(self.history_variable_key, [])
        if not isinstance(history, list):
            raise TypeError(f"Expected a list for {self.history_variable_key}, got {type(history)}")

        extract_input = {
            "input": f"<ChatHistory>\n{self._format_message(history)}\n</ChatHistory>",
            "current_time": current_time,
            "current_date": current_date
        }

        return self.extract_chain.invoke(extract_input, config=config, **kwargs)

    async def _a_extract(
        self, input: dict[str, Any], current_time: str, current_date: str,
        config: RunnableConfig | None = None, **kwargs: Any
    ) -> str:
        assert self.extract_chain is not None

        history = input.get(self.history_variable_key, [])
        if not isinstance(history, list):
            raise TypeError(f"Expected a list for {self.history_variable_key}, got {type(history)}")

        extract_input = {
            "input": f"<ChatHistory>\n{self._format_message(history)}\n</ChatHistory>",
            "current_time": current_time,
            "current_date": current_date
        }

        return await self.extract_chain.ainvoke(extract_input, config=config, **kwargs)

    @override
    def invoke(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> LongTermMemoryResult:
        assert self.chat_chain is not None

        # Get current time information
        current_time = self._get_current_timestamp()
        current_date = self._get_current_date()

        # Retrieve new long-term memory items from ChatHistory only
        new_items_str = self._extract(input, current_time, current_date, config=config, **kwargs)

        # Get current long-term memory from input parameters
        current_memory = input.get(self.current_memory_variable_key, [])
        last_update = input.get(self.last_update_time_variable_key)

        # Parse current memory to string format
        old_ltm = self._parse_current_memory(current_memory)

        # Prepare time parameters
        time_params = self._format_time_params(current_time, current_date, last_update)

        # Merge new items with old items together, and update the long-term memory
        prompt_input = {
            "input": (
                f"<NewLongTermMemory>\n{new_items_str}\n</NewLongTermMemory>\n"
                f"<CurrentLongTermMemory>\n{old_ltm}\n</CurrentLongTermMemory>"
            ),
            **time_params
        }
        merged_ltm = self.chat_chain.invoke(prompt_input, config=config, **kwargs)

        # Return structured result
        return LongTermMemoryResult.from_strings(
            updated_memory_str=merged_ltm,
            extracted_items_str=new_items_str,
            timestamp=current_time
        )

    @override
    async def ainvoke(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> LongTermMemoryResult:
        assert self.chat_chain is not None

        # Get current time information
        current_time = self._get_current_timestamp()
        current_date = self._get_current_date()

        # Get current long-term memory from input parameters
        current_memory = input.get(self.current_memory_variable_key, [])
        last_update = input.get(self.last_update_time_variable_key)

        # Parse current memory to string format
        old_ltm = self._parse_current_memory(current_memory)

        # Prepare time parameters
        time_params = self._format_time_params(current_time, current_date, last_update)

        # Concurrent execution of extraction and merging
        new_items_str: str | None = None
        merged_ltm: str | None = None

        async def _extract_items() -> None:
            nonlocal new_items_str
            new_items_str = await self._a_extract(input, current_time, current_date, config=config, **kwargs)

        async def _merge_memories() -> None:
            nonlocal merged_ltm
            # Wait for extraction to complete first
            await _extract_items()

            prompt_input = {
                "input": (
                    f"<NewLongTermMemory>\n{new_items_str}\n</NewLongTermMemory>\n"
                    f"<CurrentLongTermMemory>\n{old_ltm}\n</CurrentLongTermMemory>"
                ),
                **time_params
            }
            merged_ltm = await self.chat_chain.ainvoke(prompt_input, config=config, **kwargs)

        # Execute merge (which includes extract)
        await _merge_memories()

        if not new_items_str:
            raise ValueError("Extracted items result is None, please check the model configuration.")
        if not merged_ltm:
            raise ValueError("Merged memory result is None, please check the model configuration.")

        # Return structured result
        return LongTermMemoryResult.from_strings(
            updated_memory_str=merged_ltm,
            extracted_items_str=new_items_str,
            timestamp=current_time
        )
