import json
from pathlib import Path
from typing import Any, Literal, Self
from typing_extensions import override

from langchain_core.messages import get_buffer_string
from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig
from pydantic import Field, model_validator

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
        default_factory=get_model_from_env("LONG_MEMORY_MODEL", default="o4-mini")
    )
    temperature: float = 1

    prompt: MultilingualSystemPromptValue[Literal["zh", "en"]] | None = Field(
        default=LongTermMemoryPromptValue()
    )


class LongTermMemoryMaker(RunnableSerializable[dict[str, Any], str]):
    """Update and persist the long-term memory prompt."""

    lang: Literal["zh", "en"] = Field(default="zh")
    chat_chain: LongTermSummary | None = None
    extract_chain: LongTermSummary | None = None

    store_path: str = Field(default="./data/long_memory.json")

    history_variable_key: str = Field(default="history")
    timestamp_variable_key: str = Field(default="timestamp")

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
        # Retrieve new long-term memory items from ChatHistory only
        new_items_str = self._extract(input, config=config, **kwargs)
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

    def _extract(
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
        new_items_str = await self._a_extract(input, config=config, **kwargs)
        old_ltm = self._load_long_memory()
        prompt_input = {
            "input": (
                f"<NewLTM>\n{new_items_str}\n</NewLTM>\n<CurrentLTM>\n{old_ltm}\n</CurrentLTM>"
            )
        }
        merged_ltm = await self.chat_chain.ainvoke(prompt_input, config=config, **kwargs)
        self._save_long_memory(merged_ltm)
        return merged_ltm

    async def _a_extract(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> str:
        assert self.extract_chain is not None

        history = input.get(self.history_variable_key, [])
        if not isinstance(history, list):
            raise TypeError(f"Expected a list for {self.history_variable_key}, got {type(history)}")
        extract_input = {"input": f"<ChatHistory>\n{self._format_message(history)}\n</ChatHistory>"}

        return await self.extract_chain.ainvoke(extract_input, config=config, **kwargs)
