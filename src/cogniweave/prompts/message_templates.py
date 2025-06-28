from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Self, cast
from typing_extensions import override

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.messages.base import get_msg_title_repr
from langchain_core.prompts.chat import (
    _ImageTemplateParam,
    _StringImageMessagePromptTemplate,
    _TextTemplateParam,
)
from langchain_core.prompts.dict import DictPromptTemplate
from langchain_core.prompts.image import ImagePromptTemplate
from langchain_core.prompts.prompt import PromptTemplate
from langchain_core.prompts.string import (
    PromptTemplateFormat,
    StringPromptTemplate,
    get_template_variables,
)
from langchain_core.utils import get_colored_text
from pydantic import PositiveInt

if TYPE_CHECKING:
    from langchain_core.prompt_values import ImageURL

PromptTemplateType = StringPromptTemplate | ImagePromptTemplate | DictPromptTemplate


class MessageSegmentsPlaceholder(StringPromptTemplate):
    """Message segment prompt template.

    This is a message segment that can be used in a message prompt template.
    """

    variable_name: str
    """Name of variable to use as messages."""

    optional: bool = False
    """If True format_messages can be called with no arguments and will return an empty
        list. If False then a named argument with name `variable_name` must be passed
        in, even if the value is an empty list."""

    n_messages: PositiveInt | None = None
    """Maximum number of messages to include. If None, then will include all.
    Defaults to None."""

    def __init__(self, variable_name: str, *, optional: bool = False, **kwargs: Any) -> None:
        """Create a messages placeholder.

        Args:
            variable_name: Name of variable to use as messages.
            optional: If True format_messages can be called with no arguments and will
                return an empty list. If False then a named argument with name
                `variable_name` must be passed in, even if the value is an empty list.
                Defaults to False.]
        """
        datas = {
            "variable_name": variable_name,
            "optional": optional,
            "input_variables": [variable_name] if not optional else [],
        }
        super().__init__(
            **datas,
            **kwargs,
        )

    @override
    def format(self, **kwargs: Any) -> str:
        """Format messages from kwargs.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            List of BaseMessage.

        Raises:
            ValueError: If variable is not a list of messages.
        """
        value = kwargs.get(self.variable_name, []) if self.optional else kwargs[self.variable_name]
        if not isinstance(value, list):
            msg = (
                f"variable {self.variable_name} should be a list of str or StringPromptTemplate, "
                f"got {value} of type {type(value)}"
            )
            raise ValueError(msg)  # noqa: TRY004
        if self.n_messages:
            value = value[-self.n_messages :]
        content: str = ""
        for prompt in value:
            if isinstance(prompt, MessageSegmentsPlaceholder):
                raise TypeError(
                    "MessageSegmentsPlaceholder can't be placed inside itself, it's a cycle"
                )
            if isinstance(prompt, str):
                content += prompt
            elif isinstance(prompt, StringPromptTemplate):
                formatted: str = prompt.format(**kwargs)
                content += formatted
        return content

    @override
    async def aformat(self, **kwargs: Any) -> str:
        """Async format messages from kwargs.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            List of BaseMessage.

        Raises:
            ValueError: If variable is not a list of messages.
        """
        value = kwargs.get(self.variable_name, []) if self.optional else kwargs[self.variable_name]
        if not isinstance(value, list):
            msg = (
                f"variable {self.variable_name} should be a list of str or StringPromptTemplate, "
                f"got {value} of type {type(value)}"
            )
            raise ValueError(msg)  # noqa: TRY004
        if self.n_messages:
            value = value[-self.n_messages :]
        content: str = ""
        for prompt in value:
            if isinstance(prompt, MessageSegmentsPlaceholder):
                raise TypeError(
                    "MessageSegmentsPlaceholder can't be placed inside itself, it's a cycle"
                )
            if isinstance(prompt, str):
                content += prompt
            elif isinstance(prompt, StringPromptTemplate):
                formatted: str = prompt.format(**kwargs)
                content += formatted
        return content

    @override
    def pretty_repr(self, html: bool = False) -> str:
        """Human-readable representation.

        Args:
            html: Whether to format as HTML. Defaults to False.

        Returns:
            Human-readable representation.
        """
        var = "{" + self.variable_name + "}"
        if html:
            title = get_msg_title_repr("Message Segments Placeholder", bold=True)
            var = get_colored_text(var, "yellow")
        else:
            title = get_msg_title_repr("Message Segments Placeholder")
        return f"{title}\n\n{var}"


class _RichStringImageMessagePromptTemplate(_StringImageMessagePromptTemplate):
    """Rich String Image Message Prompt Template."""

    auto_merge_strings: bool = True

    @override
    @classmethod
    def from_template(
        cls: type[Self],
        template: str
        | Sequence[
            str | _TextTemplateParam | _ImageTemplateParam | dict[str, Any] | PromptTemplateType
        ],
        template_format: PromptTemplateFormat = "f-string",
        *,
        partial_variables: dict[str, Any] | None = None,
        auto_merge_strings: bool = True,
        **kwargs: Any,
    ) -> Self:
        if isinstance(template, str):
            prompt: StringPromptTemplate | list = PromptTemplate.from_template(
                template,
                template_format=template_format,
                partial_variables=partial_variables,
            )
            return cls(prompt=prompt, auto_merge_strings=auto_merge_strings, **kwargs)
        if isinstance(template, list):
            if (partial_variables is not None) and len(partial_variables) > 0:
                msg = "Partial variables are not supported for list of templates."
                raise ValueError(msg)
            prompt = []
            for tmpl in template:
                if isinstance(tmpl, PromptTemplateType):
                    prompt.append(tmpl)
                elif isinstance(tmpl, str) or (
                    isinstance(tmpl, dict)
                    and "text" in tmpl
                    and set(tmpl.keys()) <= {"type", "text"}
                ):
                    if isinstance(tmpl, str):
                        text: str = tmpl
                    else:
                        text = cast("_TextTemplateParam", tmpl)["text"]  # type: ignore[assignment]
                    prompt.append(
                        PromptTemplate.from_template(text, template_format=template_format)
                    )
                elif (
                    isinstance(tmpl, dict)
                    and "image_url" in tmpl
                    and set(tmpl.keys())
                    <= {
                        "type",
                        "image_url",
                    }
                ):
                    img_template = cast("_ImageTemplateParam", tmpl)["image_url"]  # type: ignore
                    input_variables = []
                    if isinstance(img_template, str):
                        vars = get_template_variables(img_template, template_format)
                        if vars:
                            if len(vars) > 1:
                                msg = (
                                    "Only one format variable allowed per image"
                                    f" template.\nGot: {vars}"
                                    f"\nFrom: {tmpl}"
                                )
                                raise ValueError(msg)
                            input_variables = [vars[0]]
                        img_template = {"url": img_template}
                        img_template_obj = ImagePromptTemplate(
                            input_variables=input_variables,
                            template=img_template,
                            template_format=template_format,
                        )
                    elif isinstance(img_template, dict):
                        img_template = dict(img_template)
                        for key in ["url", "path", "detail"]:
                            if key in img_template:
                                input_variables.extend(
                                    get_template_variables(img_template[key], template_format)
                                )
                        img_template_obj = ImagePromptTemplate(
                            input_variables=input_variables,
                            template=img_template,
                            template_format=template_format,
                        )
                    else:
                        msg = f"Invalid image template: {tmpl}"
                        raise ValueError(msg)  # noqa: TRY004
                    prompt.append(img_template_obj)
                elif isinstance(tmpl, dict):
                    if template_format == "jinja2":
                        msg = (
                            "jinja2 is unsafe and is not supported for templates "
                            "expressed as dicts. Please use 'f-string' or 'mustache' "
                            "format."
                        )
                        raise ValueError(msg)
                    data_template_obj = DictPromptTemplate(
                        template=cast("dict[str, Any]", tmpl),
                        template_format=template_format,
                    )
                    prompt.append(data_template_obj)
                else:
                    msg = f"Invalid template: {tmpl}"
                    raise ValueError(msg)
            return cls(prompt=prompt, auto_merge_strings=auto_merge_strings, **kwargs)
        msg = f"Invalid template: {template}"
        raise ValueError(msg)

    @override
    def format(self, **kwargs: Any) -> BaseMessage:
        """Format the prompt template.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            Formatted message.
        """
        if isinstance(self.prompt, StringPromptTemplate):
            text = self.prompt.format(**kwargs)
            return self._msg_class(content=text, additional_kwargs=self.additional_kwargs)
        content: list = []
        for prompt in self.prompt:
            formatted: str | ImageURL | dict[str, Any] = prompt.format(**kwargs)
            if isinstance(formatted, str):
                if (
                    self.auto_merge_strings
                    and content
                    and content[-1]["type"] == "text"
                    and isinstance(content[-1]["text"], str)
                ):
                    content[-1]["text"] += formatted
                else:
                    content.append({"type": "text", "text": formatted})
            elif isinstance(formatted, dict) and set(formatted.keys()) <= {"detail", "url"}:
                content.append({"type": "image_url", "image_url": formatted})
            elif isinstance(formatted, dict):
                formatted = cast("dict[str, Any]", formatted)
                if (
                    self.auto_merge_strings
                    and content
                    and content[-1]["type"] == "text"
                    and isinstance(content[-1]["text"], str)
                    and formatted["type"] == "text"
                    and isinstance(formatted["text"], str)
                ):
                    content[-1]["text"] += formatted["text"]
                else:
                    content.append(formatted)
        return self._msg_class(content=content, additional_kwargs=self.additional_kwargs)

    @override
    async def aformat(self, **kwargs: Any) -> BaseMessage:
        """Async format the prompt template.

        Args:
            **kwargs: Keyword arguments to use for formatting.

        Returns:
            Formatted message.
        """
        if isinstance(self.prompt, StringPromptTemplate):
            text = await self.prompt.aformat(**kwargs)
            return self._msg_class(content=text, additional_kwargs=self.additional_kwargs)
        content: list = []
        for prompt in self.prompt:
            formatted: str | ImageURL | dict[str, Any] = await prompt.aformat(**kwargs)
            if isinstance(formatted, str):
                if (
                    self.auto_merge_strings
                    and content
                    and content[-1]["type"] == "text"
                    and isinstance(content[-1]["text"], str)
                ):
                    content[-1]["text"] += formatted
                else:
                    content.append({"type": "text", "text": formatted})
            elif isinstance(formatted, dict) and set(formatted.keys()) <= {"detail", "url"}:
                content.append({"type": "image_url", "image_url": formatted})
            elif isinstance(formatted, dict):
                formatted = cast("dict[str, Any]", formatted)
                if (
                    self.auto_merge_strings
                    and content
                    and content[-1]["type"] == "text"
                    and isinstance(content[-1]["text"], str)
                    and formatted["type"] == "text"
                    and isinstance(formatted["text"], str)
                ):
                    content[-1]["text"] += formatted["text"]
                else:
                    content.append(formatted)
        return self._msg_class(content=content, additional_kwargs=self.additional_kwargs)


class RichHumanMessagePromptTemplate(_RichStringImageMessagePromptTemplate):
    """Human message prompt template. This is a message sent from the user."""

    _msg_class: type[BaseMessage] = HumanMessage


class RichAIMessagePromptTemplate(_RichStringImageMessagePromptTemplate):
    """AI message prompt template. This is a message sent from the AI."""

    _msg_class: type[BaseMessage] = AIMessage


class RichSystemMessagePromptTemplate(_RichStringImageMessagePromptTemplate):
    """System message prompt template.

    This is a message that is not sent to the user.
    """

    _msg_class: type[BaseMessage] = SystemMessage
