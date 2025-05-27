from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, Generic, TypeVar
from typing_extensions import override

from langchain_core.load.serializable import Serializable
from langchain_core.messages import (
    BaseMessage,
    SystemMessage,
)
from langchain_core.prompts.chat import (
    BaseChatPromptTemplate,
    SystemMessagePromptTemplate,
    _create_template_from_message_type,
)
from langchain_core.prompts.message import (
    BaseMessagePromptTemplate,
)
from langchain_core.prompts.string import PromptTemplateFormat
from pydantic import Field

MessageLike = BaseMessagePromptTemplate | BaseMessage | BaseChatPromptTemplate

MessageLikeRepresentation = (
    MessageLike
    | tuple[
        str | type,
        str | list[dict[str, Any]] | list[object],
    ]
    | str
    | dict[str, Any]
)

SystemMessageLike = SystemMessage | SystemMessagePromptTemplate
SystemMessageLikeRepresentation = SystemMessageLike | str

DEFAULT_SINGLE_TURN_PROMPT_ZH = (
    """你是一个有帮助的中文助手。请根据以下用户的问题进行简洁明了的回复。"""
)

DEFAULT_SINGLE_TURN_PROMPT_EN = """You are a helpful assistant. Please provide concise and clear responses to the user's questions."""

SupportLangType = TypeVar("SupportLangType", bound=str)


class BasePromptValue(Serializable, ABC):
    """Base abstract class for inputs to any language model.

    PromptValues can be converted to both LLM (pure text-generation) inputs and
    ChatModel inputs.
    """

    @override
    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable. Defaults to True."""
        return True

    @override
    @classmethod
    def get_lc_namespace(cls) -> list[str]:
        """Get the namespace of the langchain object.

        This is used to determine the namespace of the object when serializing.
        Defaults to [].
        """
        return []

    @abstractmethod
    def to_messages(self, **kwargs: Any) -> Generator[MessageLike]:
        """Return prompt as a list of Messages."""


class MultilingualSystemPromptValue(BasePromptValue, Generic[SupportLangType]):
    """Base class for prompt values."""

    prompts: dict[str, SystemMessageLikeRepresentation | list[SystemMessageLikeRepresentation]] = (
        Field(default_factory=dict)
    )

    def __init__(
        self,
        zh: SystemMessageLikeRepresentation
        | list[SystemMessageLikeRepresentation] = DEFAULT_SINGLE_TURN_PROMPT_ZH,
        en: SystemMessageLikeRepresentation
        | list[SystemMessageLikeRepresentation] = DEFAULT_SINGLE_TURN_PROMPT_EN,
        **kwargs: SystemMessageLikeRepresentation | list[SystemMessageLikeRepresentation],
    ) -> None:
        """Initialize the LangPromptValue with prompts for different languages."""
        prompts = {
            "zh": zh,
            "en": en,
        } | kwargs
        super().__init__(prompts=prompts)  # type: ignore[arg-type]

    @override
    def to_messages(self, lang: SupportLangType = "zh", **kwargs: Any) -> Generator[MessageLike]:
        if prompt := self.prompts.get(lang, None):
            prompt = prompt if isinstance(prompt, list) else [prompt]
            prompt = [_convert_to_system_message_template(p) for p in prompt]
            yield from prompt
        else:
            raise ValueError(
                f"Language '{lang}' not supported. Supported languages: {', '.join(self.prompts.keys())}"
            )


def _convert_to_system_message_template(
    message: SystemMessageLikeRepresentation,
    template_format: PromptTemplateFormat = "f-string",
) -> BaseMessage | BaseMessagePromptTemplate:
    """Instantiate a message from a variety of message formats.

    The message format can be one of the following:

    - BaseMessagePromptTemplate
    - BaseMessage
    - string: shorthand for ("human", template); e.g., "{user_input}"

    Args:
        message: a representation of a message in one of the supported formats.
        template_format: format of the template. Defaults to "f-string".

    Returns:
        an instance of a message or a message template.

    Raises:
        ValueError: If unexpected message type.
        ValueError: If 2-tuple does not have 2 elements.
    """
    if isinstance(message, SystemMessagePromptTemplate):
        _message: BaseMessage | BaseMessagePromptTemplate = message
    elif isinstance(message, SystemMessage):
        _message = message
    elif isinstance(message, str):
        _message = _create_template_from_message_type(
            "system", message, template_format=template_format
        )
    else:
        msg = f"Unsupported message type: {type(message)}"
        raise NotImplementedError(msg)

    return _message
