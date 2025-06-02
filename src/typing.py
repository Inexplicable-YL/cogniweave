from collections.abc import Callable
from typing import Any, Literal, TypeVar
from typing_extensions import override

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.prompts.chat import BaseChatPromptTemplate
from langchain_core.prompts.message import BaseMessagePromptTemplate
from pydantic import BaseModel

__all__ = []

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

GetSessionHistoryCallable = Callable[..., BaseChatMessageHistory]

Output = TypeVar("Output", covariant=True)  # noqa: PLC0105
PydanticOutput = TypeVar("PydanticOutput", bound=BaseModel, covariant=True)  # noqa: PLC0105
SupportLangType = TypeVar("SupportLangType", bound=str)

_T = TypeVar("_T")


class NotGiven:
    """
    A sentinel singleton class used to distinguish omitted keyword arguments
    from those passed in with the value None (which may have different behavior).

    For example:

    ```py
    def get(timeout: Union[int, NotGiven, None] = NotGiven()) -> Response: ...


    get(timeout=1)  # 1s timeout
    get(timeout=None)  # No timeout
    get()  # Default timeout behavior, which may not be statically known at the method definition.
    ```
    """

    def __bool__(self) -> Literal[False]:
        return False

    @override
    def __repr__(self) -> str:
        return "NOT_GIVEN"


NotGivenOr = _T | NotGiven
NOT_GIVEN = NotGiven()
