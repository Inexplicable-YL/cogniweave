from typing import Any, Literal

from cogniweave.prompt_values import MultilingualSystemPromptValue

from .values.end_detector import (
    END_DETECTOR_PROMPT_EN,
    END_DETECTOR_PROMPT_ZH,
)


class EndDetectorPromptValue(MultilingualSystemPromptValue[Literal["zh", "en"]]):
    """Prompt template for conversation end detection."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            en=END_DETECTOR_PROMPT_EN,
            zh=END_DETECTOR_PROMPT_ZH,
            **kwargs,
        )
