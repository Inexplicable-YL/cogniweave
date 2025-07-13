import dataclasses
import types
from collections import deque
from collections.abc import (
    Callable,
    Mapping,
    Sequence,
)
from typing import (
    Annotated,
    Any,
    Literal,
    TypeVar,
    Union,
    get_origin,
)
from typing_extensions import override

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.prompts.chat import BaseChatPromptTemplate
from langchain_core.prompts.message import BaseMessagePromptTemplate
from pydantic import (
    BaseModel,
)

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


def lenient_issubclass(cls: Any, class_or_tuple: type[Any] | tuple[type[Any], ...]) -> bool:
    """Check if `cls` is a subclass of `class_or_tuple`, allowing for lenient checks."""
    try:
        return isinstance(cls, type) and issubclass(cls, class_or_tuple)
    except TypeError:
        return False


def _type_is_complex_inner(type_: type[Any] | None) -> bool:
    if lenient_issubclass(type_, (str, bytes)):
        return False

    return lenient_issubclass(
        type_, (BaseModel, Mapping, Sequence, tuple, set, frozenset, deque)
    ) or dataclasses.is_dataclass(type_)


def type_is_complex(type_: type[Any]) -> bool:
    """Check if the type is complex, meaning it is not a simple type like str or bytes."""
    origin = get_origin(type_)
    return _type_is_complex_inner(type_) or _type_is_complex_inner(origin)


def origin_is_union(origin: type[Any] | None) -> bool:
    """Check if the origin is Union or types.UnionType."""
    return origin is Union or origin is types.UnionType


def origin_is_annotated(origin: type[Any] | None) -> bool:
    """Check if the origin is Annotated."""
    return origin is Annotated
