from __future__ import annotations

from datetime import datetime, timedelta
from functools import partial
from typing import TYPE_CHECKING, Any, Self, cast
from typing_extensions import override

from langchain_core.prompts.prompt import PromptTemplate
from pydantic import Field, model_validator

if TYPE_CHECKING:
    from langchain_core.prompts.string import PromptTemplateFormat


def format_datetime_relative(old_time: datetime, now: datetime | None = None) -> str:
    """Format a datetime object to a relative string.

    Args:
        old_time: The datetime object to format.
        now: The current datetime object. If not provided, the current time will be used.
    """
    now = now or datetime.now()  # noqa: DTZ005
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

    template: str = "[{time_str}]\n{chat_summary}"
    """The template to use for the prompt."""
    timestamp: datetime
    chat_summary: str
    topic_tags: list[str] = Field(default_factory=list)

    def get_chat_summary(self) -> str:
        """Get the chat summary."""
        return self.chat_summary

    @model_validator(mode="after")
    def build_partial_variables(self) -> Self:
        """Build the partial variables for the prompt."""
        self.partial_variables = {
            "chat_summary": self.get_chat_summary,
            "time_str": partial(format_datetime_relative, old_time=self.timestamp),
        }
        return self

    @override
    @classmethod
    def from_template(
        cls,
        template: str | None = None,
        *,
        timestamp: datetime,
        chat_summary: str,
        topic_tags: list[str],
        template_format: PromptTemplateFormat = "f-string",
        partial_variables: dict[str, Any] | None = None,
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
        return cast(
            "ShortMemoryPromptTemplate",
            super().from_template(
                template or cls.template,
                timestamp=timestamp,
                chat_summary=chat_summary,
                topic_tags=topic_tags,
                template_format=template_format or cls.template_format,
                partial_variables=partial_variables,
                **kwargs,
            ),
        )
