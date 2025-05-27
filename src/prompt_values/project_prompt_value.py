"""專案共用的多語提示值實作"""
from typing import Any, ClassVar, Literal

from src.prompt_values.base import MultilingualSystemPromptValue

SupportLang = Literal["zh", "en"]

class ProjectPromptValue(MultilingualSystemPromptValue[SupportLang]):
    SupportLangType: ClassVar = SupportLang

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
