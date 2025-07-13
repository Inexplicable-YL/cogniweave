from typing import Any, Literal

from cogniweave.core.prompt_values.values.end_detector import (
    END_DETECTOR_PROMPT_EN,
    END_DETECTOR_PROMPT_ZH,
)
from cogniweave.prompt_values import MultilingualSystemPromptValue


class EndDetectorPromptValue(MultilingualSystemPromptValue[Literal["zh", "en"]]):
    """Prompt template for conversation end detection."""

    def __init__(self, **kwargs: Any) -> None:
        from cogniweave.config import get_config

        _config = get_config()
        defaults = {
            "en": END_DETECTOR_PROMPT_EN,
            "zh": END_DETECTOR_PROMPT_ZH,
        }
        prompt_values = defaults.copy()
        if _config:
            prompt_values.update(_config.prompt_values.end_detector.model_dump(exclude_none=True))
            prompt_values = {
                k: v.format(default=defaults[k]) if k in defaults else v
                for k, v in prompt_values.items()
                if isinstance(v, str)
            }
        super().__init__(
            **prompt_values,
            **kwargs,
        )
