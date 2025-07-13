from typing import Any, Literal

from cogniweave.core.prompt_values.values.short_memory import (
    SHORT_TERM_MEMORY_PROMPT_EN,
    SHORT_TERM_MEMORY_PROMPT_ZH,
    SHORT_TERM_MEMORY_SUMMARY_EN,
    SHORT_TERM_MEMORY_SUMMARY_ZH,
    SHORT_TERM_MEMORY_TAGS_EN,
    SHORT_TERM_MEMORY_TAGS_ZH,
)
from cogniweave.prompt_values import MultilingualStringPromptValue, MultilingualSystemPromptValue


class ShortMemorySummaryPromptValue(MultilingualSystemPromptValue[Literal["zh", "en"]]):
    """Short-term memory prompt template for chat models."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the prompt template with the specified language."""
        from cogniweave.config import get_config

        _config = get_config()
        defaults = {
            "en": SHORT_TERM_MEMORY_SUMMARY_EN,
            "zh": SHORT_TERM_MEMORY_SUMMARY_ZH,
        }
        prompt_values = defaults.copy()
        if _config:
            prompt_values.update(
                _config.prompt_values.short_memory.summary.model_dump(exclude_none=True)
            )
            prompt_values = {
                k: v.format(default=defaults[k]) if k in defaults else v
                for k, v in prompt_values.items()
                if isinstance(v, str)
            }
        super().__init__(
            **prompt_values,
            **kwargs,
        )


class ShortMemoryTagsPromptValue(MultilingualSystemPromptValue[Literal["zh", "en"]]):
    """Short-term tags prompt template for chat models."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the prompt template with the specified language."""
        from cogniweave.config import get_config

        _config = get_config()
        defaults = {
            "en": SHORT_TERM_MEMORY_TAGS_EN,
            "zh": SHORT_TERM_MEMORY_TAGS_ZH,
        }
        prompt_values = defaults.copy()
        if _config:
            prompt_values.update(
                _config.prompt_values.short_memory.tags.model_dump(exclude_none=True)
            )
            prompt_values = {
                k: v.format(default=defaults[k]) if k in defaults else v
                for k, v in prompt_values.items()
                if isinstance(v, str)
            }
        super().__init__(
            **prompt_values,
            **kwargs,
        )


class ShortTermMemoryPromptValue(MultilingualStringPromptValue[Literal["zh", "en"]]):
    """Short-term memory system prompt wrapper."""

    def __init__(self, **kwargs: Any) -> None:
        from cogniweave.config import get_config

        _config = get_config()
        defaults = {
            "en": SHORT_TERM_MEMORY_PROMPT_EN,
            "zh": SHORT_TERM_MEMORY_PROMPT_ZH,
        }
        prompt_values = defaults.copy()
        if _config:
            prompt_values.update(
                _config.prompt_values.short_memory.prompt.model_dump(exclude_none=True)
            )
            prompt_values = {
                k: v.format(default=defaults[k]) if k in defaults else v
                for k, v in prompt_values.items()
                if isinstance(v, str)
            }
        super().__init__(
            **prompt_values,
            **kwargs,
        )
