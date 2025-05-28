import os
import re
from collections.abc import Callable, Generator
from datetime import datetime, timedelta
from typing import Any, TypeVar, cast, overload

from src.typing import NOT_GIVEN, NotGiven

_E = TypeVar("_E", bound=BaseException)


@overload
def get_provider_from_env(key: str, /) -> Callable[[], str]: ...


@overload
def get_provider_from_env(key: str, /, *, default: str) -> Callable[[], str]: ...


@overload
def get_provider_from_env(key: str, /, *, error_message: str) -> Callable[[], str]: ...


@overload
def get_provider_from_env(
    key: str, /, *, default: None, error_message: str | None
) -> Callable[[], str | None]: ...


def get_provider_from_env(
    key: str,
    /,
    *,
    default: str | NotGiven | None = NOT_GIVEN,
    error_message: str | None = None,
) -> Callable[[], str] | Callable[[], str | None]:
    """Create a factory method that gets a value from an environment variable.

    Args:
        key: The environment variable to look up. If a list of keys is provided,
            the first key found in the environment will be used.
            If no key is found, the default value will be used if set,
            otherwise an error will be raised.
        default: The default value to return if the environment variable is not set.
        error_message: the error message which will be raised if the key is not found
            and no default value is provided.
            This will be raised as a ValueError.
    """

    def get_from_env_fn() -> str | None:
        """Get a value from an environment variable."""
        if (
            isinstance(key, str)
            and key in os.environ
            and (match := re.fullmatch(r"([^/]+)/([^/]+)", os.environ[key]))
        ):
            return match.group(1).lower()

        if isinstance(default, (str, type(None))):
            return default
        if error_message:
            raise ValueError(error_message)
        msg = (
            f"Did not find {key}, please add an environment variable"
            f" `{key}` which contains it, or pass"
            f" `{key}` as a named parameter."
        )
        raise ValueError(msg)

    return get_from_env_fn


@overload
def get_model_from_env(key: str, /) -> Callable[[], str]: ...


@overload
def get_model_from_env(key: str, /, *, default: str) -> Callable[[], str]: ...


@overload
def get_model_from_env(key: str, /, *, error_message: str) -> Callable[[], str]: ...


@overload
def get_model_from_env(
    key: str, /, *, default: None, error_message: str | None
) -> Callable[[], str | None]: ...


def get_model_from_env(
    key: str,
    /,
    *,
    default: str | NotGiven | None = NOT_GIVEN,
    error_message: str | None = None,
) -> Callable[[], str] | Callable[[], str | None]:
    """Create a factory method that gets a value from an environment variable.

    Args:
        key: The environment variable to look up. If a list of keys is provided,
            the first key found in the environment will be used.
            If no key is found, the default value will be used if set,
            otherwise an error will be raised.
        default: The default value to return if the environment variable is not set.
        error_message: the error message which will be raised if the key is not found
            and no default value is provided.
            This will be raised as a ValueError.
    """

    def get_from_env_fn() -> str | None:
        """Get a value from an environment variable."""
        if (
            isinstance(key, str)
            and key in os.environ
            and (match := re.fullmatch(r"([^/]+)/([^/]+)", os.environ[key]))
        ):
            return match.group(2).lower()

        if isinstance(default, (str, type(None))):
            return default
        if error_message:
            raise ValueError(error_message)
        msg = (
            f"Did not find {key}, please add an environment variable"
            f" `{key}` which contains it, or pass"
            f" `{key}` as a named parameter."
        )
        raise ValueError(msg)

    return get_from_env_fn


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


def flatten_exception_group(
    exc_group: BaseExceptionGroup[_E],
) -> Generator[_E, None, None]:
    """递归遍历 BaseExceptionGroup ，并返回一个生成器"""
    for exc in exc_group.exceptions:
        if isinstance(exc, BaseExceptionGroup):
            yield from flatten_exception_group(cast("BaseExceptionGroup[_E]", exc))
        else:
            yield exc


def handle_exception(
    # msg: str,
    # level: Literal["debug", "info", "warning", "error", "critical"] = "error",
    # **kwargs: Any,
) -> Callable[[BaseExceptionGroup[Exception]], None]:
    """递归遍历 BaseExceptionGroup ，并输出日志"""

    def _handle(exc_group: BaseExceptionGroup[Exception]) -> None:
        for exc in flatten_exception_group(exc_group):
            pass
            # 干啥我也不知道，再说吧

    return _handle


def remove_not_given_params(**kwages: Any) -> dict[str, Any]:
    return {key: value for key, value in kwages.items() if not isinstance(value, NotGiven)}
