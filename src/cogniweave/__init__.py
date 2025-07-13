import os
from pathlib import Path
from typing import Any

from cogniweave.config import DOTENV_TYPE, Config, Env

__all__ = [
    "get_config",
    "init_config",
]


_config: Config | None = None


def get_config() -> Config | None:
    """Get the global configuration object."""
    return _config


def init_config(
    *, _env_file: DOTENV_TYPE | None = None, _config_file: str | Path | None = None, **kwargs: Any
) -> None:
    """Initialize the global configuration object."""
    global _config  # noqa: PLW0603
    if not _config:
        env = Env()
        _env_file = _env_file or f".env.{env.environment}"
        _config = Config(
            **kwargs,
            _env_file=(
                (".env", _env_file) if isinstance(_env_file, (str, os.PathLike)) else _env_file
            ),
            _config_file=_config_file,
        )
