from __future__ import annotations

from datetime import datetime, timedelta
from functools import partial
from typing import Any, ClassVar, TypedDict, overload
from typing_extensions import override

from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.string import PromptTemplateFormat  # noqa: TC002
from pydantic import Field, model_validator


class ShortMemoryTemplateDict(TypedDict):
    template: str
    timestamp: float | int
    chat_summary: str
    topic_tags: list[str]
    template_format: PromptTemplateFormat


def format_datetime_relative(old_time: datetime, now: datetime | None = None) -> str:
    """Format a datetime object to a relative string.

    Args:
        old_time: The datetime object to format.
        now: The current datetime object. If not provided, the current time will be used.
    """
    now = now or datetime.now()
    today = now.date()
    yesterday = today - timedelta(days=1)
    old_date = old_time.date()

    time_part = old_time.strftime("%H:%M")

    if old_date == today:
        return time_part
    if old_date == yesterday:
        return f"Yesterday {time_part}"
    date_part = old_time.strftime("%Y/%m/%d")
    return f"{date_part} {time_part}"


class ShortMemoryPromptTemplate(PromptTemplate):
    """Generative prompt template."""

    _template: ClassVar[str] = "[{time_str}]\n{chat_summary}\n"
    template: str = Field(default=_template)
    """The template to use for the prompt."""

    timestamp: datetime
    chat_summary: str
    topic_tags: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def preprocess_input(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Build the partial variables for the prompt."""
        data["partial_variables"] = {
            "chat_summary": data["chat_summary"],
            "time_str": partial(format_datetime_relative, old_time=data["timestamp"]),
        }
        return data

    @override
    @classmethod
    def from_template(
        cls,
        template: str | None = None,
        *,
        timestamp: datetime | float,
        chat_summary: str,
        topic_tags: list[str],
        template_format: PromptTemplateFormat = "f-string",
        **kwargs: Any,
    ) -> ShortMemoryPromptTemplate:
        """Create a new instance of the prompt template from a template string.

        Args:
            template: The template string to use.
            timestamp: The timestamp to use in the prompt.
            chat_summary: The chat summary to use in the prompt.
            topic_tags: The topic tags to use in the prompt.
            template_format: The format of the template string.
            partial_variables: Any additional variables to use in the prompt.
            **kwargs: Additional keyword arguments to pass to the parent class.
        """
        if isinstance(timestamp, float | int):
            timestamp = datetime.fromtimestamp(timestamp)
        return cls(
            template=template or cls._template,
            timestamp=timestamp,
            chat_summary=chat_summary,
            topic_tags=topic_tags,
            template_format=template_format,
            **kwargs,
        )

    @override
    def format(self, **kwargs: Any) -> str:
        kwargs.setdefault("chat_summary", self.chat_summary)
        kwargs.setdefault(
            "time_str",
            format_datetime_relative(old_time=self.timestamp, now=kwargs.get("timestamp")),
        )
        return super().format(**kwargs)

    def to_template_dict(self) -> ShortMemoryTemplateDict:
        """Convert the prompt template to a dictionary."""
        return ShortMemoryTemplateDict(
            template=self.template,
            timestamp=self.timestamp.timestamp(),
            chat_summary=self.chat_summary,
            topic_tags=self.topic_tags,
            template_format=self.template_format,
        )

    @overload
    @classmethod
    def load(cls, obj: ShortMemoryTemplateDict | dict[Any, Any]) -> ShortMemoryPromptTemplate: ...

    @overload
    @classmethod
    def load(
        cls, obj: list[ShortMemoryTemplateDict | dict[Any, Any]]
    ) -> list[ShortMemoryPromptTemplate]: ...

    @classmethod
    def load(
        cls,
        obj: Any,
    ) -> ShortMemoryPromptTemplate | list[ShortMemoryPromptTemplate]:
        """Load a prompt template from a dictionary."""

        def _load(
            obj: dict[Any, Any] | list[dict[Any, Any]],
        ) -> Any:
            if isinstance(obj, dict):
                template_obj = ShortMemoryTemplateDict(**obj)
                return ShortMemoryPromptTemplate.from_template(**template_obj)
            if isinstance(obj, list):
                return [_load(o) for o in obj]
            return obj

        return _load(obj)
