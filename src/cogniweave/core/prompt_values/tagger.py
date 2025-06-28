from typing import Any, Literal

from cogniweave.core.prompt_values.values.tagger import (
    SHORT_TERM_MEMORY_TAGS_EN,
    SHORT_TERM_MEMORY_TAGS_ZH,
)
from cogniweave.prompt_values import MultilingualSystemPromptValue


class ShortTermTagsPromptValue(MultilingualSystemPromptValue[Literal["zh", "en"]]):
    """Short-term tags prompt template for chat models."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the prompt template with the specified language."""
        super().__init__(en=SHORT_TERM_MEMORY_TAGS_EN, zh=SHORT_TERM_MEMORY_TAGS_ZH, **kwargs)
