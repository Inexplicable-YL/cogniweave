from typing import Any, Literal

from cogniweave.core.prompt_values.values.short_memory import (
    SHORT_TERM_MEMORY_PROMPT_EN,
    SHORT_TERM_MEMORY_PROMPT_ZH,
    SHORT_TERM_MEMORY_SUMMARY_EN,
    SHORT_TERM_MEMORY_SUMMARY_ZH,
    SHORT_TERM_MEMORY_TAGS_EN,
    SHORT_TERM_MEMORY_TAGS_ZH,
)
from cogniweave.prompt_values import MultilingualSystemPromptValue


class ShortMemorySummaryPromptValue(MultilingualSystemPromptValue[Literal["zh", "en"]]):
    """Short-term memory prompt template for chat models."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the prompt template with the specified language."""
        super().__init__(en=SHORT_TERM_MEMORY_SUMMARY_EN, zh=SHORT_TERM_MEMORY_SUMMARY_ZH, **kwargs)


class ShortMemoryTagsPromptValue(MultilingualSystemPromptValue[Literal["zh", "en"]]):
    """Short-term tags prompt template for chat models."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the prompt template with the specified language."""
        super().__init__(en=SHORT_TERM_MEMORY_TAGS_EN, zh=SHORT_TERM_MEMORY_TAGS_ZH, **kwargs)


class ShortTermMemoryPromptValue(MultilingualSystemPromptValue[Literal["zh", "en"]]):
    """Short-term memory system prompt wrapper."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(zh=SHORT_TERM_MEMORY_PROMPT_ZH, en=SHORT_TERM_MEMORY_PROMPT_EN, **kwargs)
