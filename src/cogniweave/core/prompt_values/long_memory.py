from typing import Any, Literal

from cogniweave.core.prompt_values.values.long_memory import (
    LONG_TERM_MEMORY_EXTRACT_EN,
    LONG_TERM_MEMORY_EXTRACT_ZH,
    LONG_TERM_MEMORY_UPDATE_EN,
    LONG_TERM_MEMORY_UPDATE_ZH,
)
from cogniweave.prompt_values.base import MultilingualSystemPromptValue


class LongTermMemoryExtractPromptValue(MultilingualSystemPromptValue[Literal["zh", "en"]]):
    """Long-term memory extraction system prompt wrapper."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(zh=LONG_TERM_MEMORY_EXTRACT_ZH, en=LONG_TERM_MEMORY_EXTRACT_EN, **kwargs)


class LongTermMemoryPromptValue(MultilingualSystemPromptValue[Literal["zh", "en"]]):
    """Long-term memory system prompt wrapper."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(zh=LONG_TERM_MEMORY_UPDATE_ZH, en=LONG_TERM_MEMORY_UPDATE_EN, **kwargs)
