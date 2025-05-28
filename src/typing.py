from typing import (  # noqa: A005
    Literal,
    TypeVar,
)
from typing_extensions import override

from pydantic import BaseModel

__all__ = []
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
