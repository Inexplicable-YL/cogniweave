from typing import Any, Literal

from cogniweave.prompt_values import MultilingualSystemPromptValue

from .values.summary import (
    SHORT_TERM_MEMORY_SUMMARY_EN,
    SHORT_TERM_MEMORY_SUMMARY_ZH,
)


class ShortTermMemoryPromptValue(MultilingualSystemPromptValue[Literal["zh", "en"]]):
    """Short-term memory prompt template for chat models."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the prompt template with the specified language."""
        super().__init__(en=SHORT_TERM_MEMORY_SUMMARY_EN, zh=SHORT_TERM_MEMORY_SUMMARY_ZH, **kwargs)
