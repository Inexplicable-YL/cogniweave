import asyncio
import os
import re
from collections.abc import (
    AsyncGenerator,
    Callable,
    Coroutine,
    Generator,
)
from contextlib import AbstractContextManager as ContextManager
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from functools import partial
from typing import (
    Any,
    TypeVar,
    cast,
    overload,
)
from typing_extensions import ParamSpec

from cogniweave.typing import NOT_GIVEN, NotGiven

_T = TypeVar("_T")
_P = ParamSpec("_P")
_R = TypeVar("_R")
_E = TypeVar("_E", bound=BaseException)
_K = TypeVar("_K")


def deep_update(mapping: dict[_K, Any], *updating_mappings: dict[_K, Any]) -> dict[_K, Any]:
    """深度更新合并字典"""
    updated_mapping = mapping.copy()
    for updating_mapping in updating_mappings:
        for k, v in updating_mapping.items():
            if (
                k in updated_mapping
                and isinstance(updated_mapping[k], dict)
                and isinstance(v, dict)
            ):
                updated_mapping[k] = deep_update(updated_mapping[k], v)
            else:
                updated_mapping[k] = v
    return updated_mapping


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


def flatten_exception_group(
    exc_group: BaseExceptionGroup[_E],
) -> Generator[_E, None, None]:
    """递归遍历 BaseExceptionGroup ，并返回一个生成器"""
    for exc in exc_group.exceptions:
        if isinstance(exc, BaseExceptionGroup):
            yield from flatten_exception_group(cast("BaseExceptionGroup[_E]", exc))
        else:
            yield exc


def remove_not_given_params(**kwargs: Any) -> dict[str, Any]:
    """Remove ``NotGiven`` parameters from ``kwargs``."""

    return {key: value for key, value in kwargs.items() if not isinstance(value, NotGiven)}


def sync_func_wrapper(
    func: Callable[_P, _R], *, to_thread: bool = False
) -> Callable[_P, Coroutine[None, None, _R]]:
    """包装一个同步函数为异步函数。

    Args:
        func: 待包装的同步函数。
        to_thread: 是否在独立的线程中运行同步函数。默认为 `False`。

    Returns:
        异步函数。
    """
    if to_thread:

        async def _wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            loop = asyncio.get_running_loop()
            func_call = partial(func, *args, **kwargs)
            return await loop.run_in_executor(None, func_call)

    else:

        async def _wrapper(*args: _P.args, **kwargs: _P.kwargs) -> _R:
            return func(*args, **kwargs)

    return _wrapper


@asynccontextmanager
async def sync_ctx_manager_wrapper(
    cm: ContextManager[_T], *, to_thread: bool = False
) -> AsyncGenerator[_T, None]:
    """将同步上下文管理器包装为异步上下文管理器。

    Args:
        cm: 待包装的同步上下文管理器。
        to_thread: 是否在独立的线程中运行同步函数。默认为 `False`。

    Returns:
        异步上下文管理器。
    """
    try:
        yield await sync_func_wrapper(cm.__enter__, to_thread=to_thread)()
    except Exception as e:
        if not await sync_func_wrapper(cm.__exit__, to_thread=to_thread)(
            type(e), e, e.__traceback__
        ):
            raise
    else:
        await sync_func_wrapper(cm.__exit__, to_thread=to_thread)(None, None, None)
