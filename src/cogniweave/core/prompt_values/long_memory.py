from typing import Any, Literal

from cogniweave.core.prompt_values.values.long_memory import (
    LONG_TERM_MEMORY_EXTRACT_EN,
    LONG_TERM_MEMORY_EXTRACT_ZH,
    LONG_TERM_MEMORY_PROMPT_EN,
    LONG_TERM_MEMORY_PROMPT_ZH,
    LONG_TERM_MEMORY_UPDATE_EN,
    LONG_TERM_MEMORY_UPDATE_ZH,
)
from cogniweave.prompt_values import MultilingualStringPromptValue, MultilingualSystemPromptValue


class LongMemoryExtractPromptValue(MultilingualSystemPromptValue[Literal["zh", "en"]]):
    """Long-term memory extraction system prompt wrapper."""

    def __init__(self, **kwargs: Any) -> None:
        from cogniweave.config import get_config

        _config = get_config()
        defaults = {
            "en": LONG_TERM_MEMORY_EXTRACT_EN,
            "zh": LONG_TERM_MEMORY_EXTRACT_ZH,
        }
        prompt_values = defaults.copy()
        if _config:
            prompt_values.update(
                _config.prompt_values.long_memory.extract.model_dump(exclude_none=True)
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


class LongMemoryUpdatePromptValue(MultilingualSystemPromptValue[Literal["zh", "en"]]):
    """Long-term memory system prompt wrapper."""

    def __init__(self, **kwargs: Any) -> None:
        from cogniweave.config import get_config

        _config = get_config()
        defaults = {
            "en": LONG_TERM_MEMORY_UPDATE_EN,
            "zh": LONG_TERM_MEMORY_UPDATE_ZH,
        }
        prompt_values = defaults.copy()
        if _config:
            prompt_values.update(
                _config.prompt_values.long_memory.extract.model_dump(exclude_none=True)
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


class LongTermMemoryPromptValue(MultilingualStringPromptValue[Literal["zh", "en"]]):
    """Long-term memory system prompt wrapper."""

    def __init__(self, **kwargs: Any) -> None:
        from cogniweave.config import get_config

        _config = get_config()
        defaults = {
            "en": LONG_TERM_MEMORY_PROMPT_EN,
            "zh": LONG_TERM_MEMORY_PROMPT_ZH,
        }
        prompt_values = defaults.copy()
        if _config:
            prompt_values.update(
                _config.prompt_values.long_memory.extract.model_dump(exclude_none=True)
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
