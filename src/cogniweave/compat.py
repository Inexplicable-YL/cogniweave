"""本模块为 Pydantic 版本兼容层模块

为兼容 Pydantic V1 与 V2 版本，定义了一系列兼容函数与类供使用。

FrontMatter:
    mdx:
        format: md
    sidebar_position: 16
    description: nonebot.compat 模块
"""

from collections.abc import Callable, Generator  # noqa: TC003
from dataclasses import dataclass, is_dataclass
from functools import cached_property
from typing import (
    TYPE_CHECKING,
    Annotated,
    Any,
    Protocol,
    Self,
    TypeVar,
    get_args,
    get_origin,
)
from typing_extensions import is_typeddict, override

from pydantic import (
    BaseModel,
    ConfigDict,
    TypeAdapter,
    field_validator,
    model_validator,
)
from pydantic._internal._repr import display_as_type
from pydantic.fields import FieldInfo as BaseFieldInfo
from pydantic_core import PydanticUndefined, PydanticUndefinedType

from cogniweave.typing import origin_is_annotated

Required = Ellipsis
"""Alias of Ellipsis for compatibility with pydantic v1"""


T = TypeVar("T")


if TYPE_CHECKING:

    class _CustomValidationClass(Protocol):
        @classmethod
        def __get_validators__(cls) -> Generator[Callable[..., Any], None, None]: ...

    CVC = TypeVar("CVC", bound=_CustomValidationClass)


__all__ = (
    "DEFAULT_CONFIG",
    "ConfigDict",
    "FieldInfo",
    "ModelField",
    "PydanticUndefined",
    "PydanticUndefinedType",
    "Required",
    "TypeAdapter",
    "extract_field_info",
    "field_validator",
    "model_fields",
    "model_validator",
)

__autodoc__ = {
    "PydanticUndefined": "Pydantic Undefined object",
    "PydanticUndefinedType": "Pydantic Undefined type",
}


DEFAULT_CONFIG = ConfigDict(extra="allow", arbitrary_types_allowed=True)
"""Default config for validations"""


class FieldInfo(BaseFieldInfo):
    """FieldInfo class with extra property for compatibility with pydantic v1"""

    # make default can be positional argument
    def __init__(self, default: Any = PydanticUndefined, **kwargs: Any) -> None:
        super().__init__(default=default, **kwargs)

    @property
    def extra(self) -> dict[str, Any]:
        """Extra data that is not part of the standard pydantic fields.

        For compatibility with pydantic v1.
        """
        # extract extra data from attributes set except used slots
        # we need to call super in advance due to
        # comprehension not inlined in cpython < 3.12
        # https://peps.python.org/pep-0709/
        slots = super().__slots__
        return {k: v for k, v in self._attributes_set.items() if k not in slots}


@dataclass
class ModelField:
    """ModelField class for compatibility with pydantic v1"""

    name: str
    """The name of the field."""
    annotation: Any
    """The annotation of the field."""
    field_info: FieldInfo
    """The FieldInfo of the field."""

    @classmethod
    def _construct(cls, name: str, annotation: Any, field_info: FieldInfo) -> Self:
        return cls(name, annotation, field_info)

    @classmethod
    def construct(cls, name: str, annotation: Any, field_info: FieldInfo | None = None) -> Self:
        """Construct a ModelField from given infos."""
        return cls._construct(name, annotation, field_info or FieldInfo())

    @override
    def __hash__(self) -> int:
        # Each ModelField is unique for our purposes,
        # to allow store them in a set.
        return id(self)

    @cached_property
    def type_adapter(self) -> TypeAdapter:
        """TypeAdapter of the field.

        Cache the TypeAdapter to avoid creating it multiple times.
        Pydantic v2 uses too much cpu time to create TypeAdapter.

        See: https://github.com/pydantic/pydantic/issues/9834
        """
        return TypeAdapter(
            Annotated[self.annotation, self.field_info],
            config=None if self._annotation_has_config() else DEFAULT_CONFIG,
        )

    def _annotation_has_config(self) -> bool:
        """Check if the annotation has config.

        TypeAdapter raise error when annotation has config
        and given config is not None.
        """
        type_is_annotated = origin_is_annotated(get_origin(self.annotation))
        inner_type = get_args(self.annotation)[0] if type_is_annotated else self.annotation
        try:
            return (
                issubclass(inner_type, BaseModel)
                or is_dataclass(inner_type)
                or is_typeddict(inner_type)
            )
        except TypeError:
            return False

    def get_default(self) -> Any:
        """Get the default value of the field."""
        return self.field_info.get_default(call_default_factory=True)

    def _type_display(self) -> str:
        """Get the display of the type of the field."""
        return display_as_type(self.annotation)

    def validate_value(self, value: Any) -> Any:
        """Validate the value pass to the field."""
        return self.type_adapter.validate_python(value)


def extract_field_info(field_info: BaseFieldInfo) -> dict[str, Any]:
    """Get FieldInfo init kwargs from a FieldInfo instance."""

    kwargs = field_info._attributes_set.copy()
    kwargs["annotation"] = field_info.rebuild_annotation()
    return kwargs


def model_fields(model: type[BaseModel]) -> list[ModelField]:
    """Get field list of a model."""

    return [
        ModelField._construct(
            name=name,
            annotation=field_info.rebuild_annotation(),
            field_info=FieldInfo(**extract_field_info(field_info)),
        )
        for name, field_info in model.model_fields.items()
    ]
