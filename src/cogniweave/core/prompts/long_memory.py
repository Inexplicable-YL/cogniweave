from __future__ import annotations

from datetime import datetime
from typing import Any, ClassVar, TypedDict, overload
from typing_extensions import override

from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.string import PromptTemplateFormat  # noqa: TC002
from pydantic import Field, model_validator


def _format_memory(updated_memory: list[str]) -> str:
    """Format the updated memory into a JSON string."""
    return (
        "\n".join(f"  {i}. {memory_item}" for i, memory_item in enumerate(updated_memory, 1)) + "\n"
    )


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
    def add_partial_variables(cls, values: dict[str, Any]) -> dict[str, Any]:
        values["partial_variables"] = values.get("partial_variables", {}) | {
            "history": values["history"],
            "current_time": values["current_time"],
            "current_date": values["current_date"],
        }
        return values

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
        **kwargs: Any,
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
            **kwargs,
        )


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

    new_memory: list[str]
    current_memory: list[str]
    current_time: str
    current_date: str
    last_update_time: str
    template_format: PromptTemplateFormat = "f-string"

    @model_validator(mode="before")
    @classmethod
    def preprocess_input(cls, values: dict[str, Any]) -> dict[str, Any]:
        # Format current_memory list or other types into a JSON string if needed
        values["partial_variables"] = values.get("partial_variables", {}) | {
            "new_memory": _format_memory(values["new_memory"]),
            "current_memory": _format_memory(values["current_memory"]),
            "current_time": values["current_time"],
            "current_date": values["current_date"],
            "last_update_time": values["last_update_time"],
        }
        return values

    @override
    @classmethod
    def from_template(
        cls,
        template: str | None = None,
        *,
        new_memory: list[str],
        current_memory: list[str],
        current_time: datetime | str,
        current_date: datetime | str,
        last_update_time: datetime | str,
        template_format: PromptTemplateFormat = "f-string",
        **kwargs: Any,
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
            **kwargs,
        )


class LongMemoryTemplateDict(TypedDict):
    template: str
    updated_memory: list[str]
    template_format: PromptTemplateFormat


class LongMemoryPromptTemplate(PromptTemplate):
    """Generative prompt template for long-term memory output."""

    _template: ClassVar[str] = "{updated_memory_json}"
    template: str = Field(default=_template)

    updated_memory: list[str]

    @model_validator(mode="before")
    @classmethod
    def setup_partial_variables(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Ensure partial_variables is correctly set"""
        if "updated_memory" in values:
            memory_json = _format_memory(values["updated_memory"])
            values["partial_variables"] = values.get("partial_variables", {}) | {
                "updated_memory_json": memory_json
            }
        return values

    @override
    @classmethod
    def from_template(
        cls,
        template: str | None = None,
        *,
        updated_memory: list[str],
        template_format: PromptTemplateFormat = "f-string",
        **kwargs: Any,
    ) -> LongMemoryPromptTemplate:
        """Create prompt template from variables."""
        return cls(
            template=template or cls._template,
            updated_memory=updated_memory,
            template_format=template_format,
            **kwargs,
        )

    def to_template_dict(self) -> LongMemoryTemplateDict:
        return LongMemoryTemplateDict(
            template=self.template,
            updated_memory=self.updated_memory,
            template_format=self.template_format,
        )

    @overload
    @classmethod
    def load(cls, obj: LongMemoryTemplateDict | dict[Any, Any]) -> LongMemoryPromptTemplate: ...

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
