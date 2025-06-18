from __future__ import annotations

import json
from datetime import datetime
from typing import Any, ClassVar, TypedDict, overload
from typing_extensions import override
from pydantic import Field, model_validator
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.string import PromptTemplateFormat
from langchain_core.messages import get_buffer_string


class LongMemoryExtractTemplateDict(TypedDict):
    template: str
    history: str
    current_time: str
    current_date: str
    template_format: PromptTemplateFormat


class LongMemoryExtractPromptTemplate(PromptTemplate):
    """Template for long-term memory extraction."""
    _template: ClassVar[str] = (
        "<ChatHistory>\n"
        "{history}\n"
        "</ChatHistory>\n"
        "Current time: {current_time}\n"
        "Current date: {current_date}"
    )
    template: str = Field(default=_template)
    history: str
    current_time: str
    current_date: str
    template_format: PromptTemplateFormat = "f-string"

    @model_validator(mode="before")
    @classmethod
    def preprocess_input(cls, data: dict[str, Any]) -> dict[str, Any]:
        # Format history list into a buffer string
        hist = data.get("history")
        if not isinstance(hist, str):
            data["history"] = get_buffer_string(hist, human_prefix='[User]', ai_prefix='[Assistant]')
        return data

    @override
    @classmethod
    def from_template(
        cls,
        template: str | None = None,
        *,
        history: str,
        current_time: datetime | str,
        current_date: datetime | str,
        template_format: PromptTemplateFormat = "f-string",
    ) -> LongMemoryExtractPromptTemplate:
        """Create extraction prompt from variables."""
        # Convert datetime to formatted string
        if isinstance(current_time, datetime):
            current_time = current_time.strftime("%Y-%m-%d %H:%M")
        if isinstance(current_date, datetime):
            current_date = current_date.strftime("%Y-%m-%d")
        return cls(
            template=template or cls._template,
            history=history,
            current_time=current_time,
            current_date=current_date,
            template_format=template_format,
        )


class LongMemoryMergeTemplateDict(TypedDict):
    template: str
    new_memory: str
    current_memory: Any
    current_time: str
    current_date: str
    last_update_time: str
    template_format: PromptTemplateFormat


class LongMemoryMergePromptTemplate(PromptTemplate):
    """Template for long-term memory update/merge."""
    _template: ClassVar[str] = (
        "<NewLongTermMemory>\n"
        "{new_memory}\n"
        "</NewLongTermMemory>\n"
        "<CurrentLongTermMemory>\n"
        "{current_memory}\n"
        "</CurrentLongTermMemory>\n"
        "Current time: {current_time}\n"
        "Current date: {current_date}\n"
        "Last update time: {last_update_time}"
    )
    template: str = Field(default=_template)
    new_memory: str
    current_memory: Any
    current_time: str
    current_date: str
    last_update_time: str
    template_format: PromptTemplateFormat = "f-string"

    @model_validator(mode="before")
    @classmethod
    def preprocess_input(cls, data: dict[str, Any]) -> dict[str, Any]:
        # Format current_memory list or other types into a JSON string if needed
        cm = data.get("current_memory")
        if isinstance(cm, list):
            data["current_memory"] = json.dumps(cm, ensure_ascii=False, indent=2)
        elif not isinstance(cm, str):
            data["current_memory"] = str(cm)
        return data

    @override
    @classmethod
    def from_template(
        cls,
        template: str | None = None,
        *,
        new_memory: str,
        current_memory: Any,
        current_time: datetime | str,
        current_date: datetime | str,
        last_update_time: datetime | str,
        template_format: PromptTemplateFormat = "f-string",
    ) -> LongMemoryMergePromptTemplate:
        """Create merge prompt from variables."""
        # Convert datetime fields to formatted strings
        if isinstance(current_time, datetime):
            current_time = current_time.strftime("%Y-%m-%d %H:%M")
        if isinstance(last_update_time, datetime):
            last_update_time = last_update_time.strftime("%Y-%m-%d %H:%M")
        if isinstance(current_date, datetime):
            current_date = current_date.strftime("%Y-%m-%d")
        return cls(
            template=template or cls._template,
            new_memory=new_memory,
            current_memory=current_memory,
            current_time=current_time,
            current_date=current_date,
            last_update_time=last_update_time,
            template_format=template_format,
        )

class LongMemoryTemplateDict(TypedDict):
    template: str
    updated_memory: list[str]
    template_format: PromptTemplateFormat

class LongMemoryPromptTemplate(PromptTemplate):
    """Generative prompt template for long-term memory output."""
    _template: ClassVar[str] = ""
    template: str = Field(default=_template)
    updated_memory: list[str]
    template_format: PromptTemplateFormat = "f-string"

    @override
    @classmethod
    def from_template(
        cls,
        *,
        updated_memory: list[str],
        template: str | None = None,
        template_format: PromptTemplateFormat = "f-string",
    ) -> LongMemoryPromptTemplate:
        # 以 updated_memory 的 JSON 陣列字串作為 template
        memory_str = template or json.dumps(updated_memory, ensure_ascii=False)
        return cls(
            template=memory_str,
            updated_memory=updated_memory,
            template_format=template_format,
        )

    def to_template_dict(self) -> LongMemoryTemplateDict:
        return LongMemoryTemplateDict(
            template=self.template,
            updated_memory=self.updated_memory,
            template_format=self.template_format,
        )

    @overload
    @classmethod
    def load(
        cls, obj: LongMemoryTemplateDict | dict[Any, Any]
    ) -> LongMemoryPromptTemplate: ...
    @overload
    @classmethod
    def load(
        cls, obj: list[LongMemoryTemplateDict | dict[Any, Any]]
    ) -> list[LongMemoryPromptTemplate]: ...
    @classmethod
    def load(cls, obj: Any) -> LongMemoryPromptTemplate | list[LongMemoryPromptTemplate]:
        """Load a prompt template from dict(s)."""
        def _load(o: Any) -> Any:
            if isinstance(o, dict):
                data = LongMemoryTemplateDict(**o)
                return cls.from_template(**data)
            if isinstance(o, list):
                return [_load(item) for item in o]
            return o
        return _load(obj)