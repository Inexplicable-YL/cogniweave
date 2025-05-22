import os
import re
from collections.abc import Callable
from typing import overload


class _NoDefaultType:
    """Type to indicate no default value is provided."""


_NoDefault = _NoDefaultType()


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
    default: str | _NoDefaultType | None = _NoDefault,
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
    default: str | _NoDefaultType | None = _NoDefault,
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
