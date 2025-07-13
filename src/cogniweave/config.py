import abc
import json
import os
import tomllib
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    TypeAlias,
    cast,
    get_args,
    get_origin,
)
from typing_extensions import override

import yaml
from dotenv import dotenv_values
from pydantic import (
    BaseModel,
    ConfigDict,
)
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined, PydanticUndefinedType

from cogniweave.typing import (
    lenient_issubclass,
    origin_is_union,
    type_is_complex,
)
from cogniweave.utils import deep_update

DOTENV_TYPE: TypeAlias = Path | str | list[Path | str] | tuple[Path | str, ...]

ENV_FILE_SENTINEL = Path()

_config: "Config | None" = None


def get_config() -> "Config | None":
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


@dataclass
class ModelField:
    """Simplified model field information used for settings parsing."""

    name: str
    annotation: Any
    field_info: FieldInfo


def model_fields(model: type[BaseModel]) -> list[ModelField]:
    """Return model fields as :class:`ModelField` list compatible with pydantic v1 and v2."""

    # pydantic v2 provides ``model_fields`` containing ``FieldInfo`` objects
    fields = getattr(model, "model_fields", None)
    if fields is not None:
        return [
            ModelField(name=name, annotation=field.annotation, field_info=field)
            for name, field in fields.items()
        ]

    # fallback to pydantic v1 ``__fields__`` structure
    return [
        ModelField(
            name=name,
            annotation=field.annotation,
            field_info=field.field_info,
        )
        for name, field in getattr(model, "__fields__", {}).items()
    ]


class SettingsError(ValueError): ...


class BaseSettingsSource(abc.ABC):
    def __init__(self, settings_cls: type["BaseSettings"]) -> None:
        self.settings_cls = settings_cls

    @property
    def config(self) -> "SettingsConfig":
        return cast("SettingsConfig", self.settings_cls.model_config)

    @abc.abstractmethod
    def __call__(self) -> dict[str, Any]:
        raise NotImplementedError


class InitSettingsSource(BaseSettingsSource):
    __slots__ = ("init_kwargs",)

    def __init__(self, settings_cls: type["BaseSettings"], init_kwargs: dict[str, Any]) -> None:
        self.init_kwargs = init_kwargs
        super().__init__(settings_cls)

    @override
    def __call__(self) -> dict[str, Any]:
        return self.init_kwargs

    @override
    def __repr__(self) -> str:
        return f"InitSettingsSource(init_kwargs={self.init_kwargs!r})"


class DotEnvSettingsSource(BaseSettingsSource):
    def __init__(
        self,
        settings_cls: type["BaseSettings"],
        env_file: DOTENV_TYPE | None = ENV_FILE_SENTINEL,
        env_file_encoding: str | None = None,
        case_sensitive: bool | None = None,
        env_nested_delimiter: str | None = None,
    ) -> None:
        super().__init__(settings_cls)
        self.env_file = (
            env_file
            if env_file is not ENV_FILE_SENTINEL
            else self.config.get("env_file", (".env",))
        )
        self.env_file_encoding = (
            env_file_encoding
            if env_file_encoding is not None
            else self.config.get("env_file_encoding", "utf-8")
        )
        self.case_sensitive = (
            case_sensitive
            if case_sensitive is not None
            else self.config.get("case_sensitive", False)
        )
        self.env_nested_delimiter = (
            env_nested_delimiter
            if env_nested_delimiter is not None
            else self.config.get("env_nested_delimiter", None)
        )

    def _apply_case_sensitive(self, var_name: str) -> str:
        return var_name if self.case_sensitive else var_name.lower()

    def _field_is_complex(self, field: ModelField) -> tuple[bool, bool]:
        if type_is_complex(field.annotation):
            return True, False
        if origin_is_union(get_origin(field.annotation)) and any(
            type_is_complex(arg) for arg in get_args(field.annotation)
        ):
            return True, True
        return False, False

    def _parse_env_vars(self, env_vars: Mapping[str, str | None]) -> dict[str, str | None]:
        return {self._apply_case_sensitive(key): value for key, value in env_vars.items()}

    def _read_env_file(self, file_path: Path) -> dict[str, str | None]:
        file_vars = dotenv_values(file_path, encoding=self.env_file_encoding)
        return self._parse_env_vars(file_vars)

    def _read_env_files(self) -> dict[str, str | None]:
        env_files = self.env_file
        if env_files is None:
            return {}

        if isinstance(env_files, (str, os.PathLike)):
            env_files = [env_files]

        dotenv_vars: dict[str, str | None] = {}
        for env_file in env_files:
            env_path = Path(env_file).expanduser()
            if env_path.is_file():
                dotenv_vars.update(self._read_env_file(env_path))
        return dotenv_vars

    def _next_field(self, field: ModelField | None, key: str) -> ModelField | None:
        if not field or origin_is_union(get_origin(field.annotation)):
            return None
        if field.annotation and lenient_issubclass(field.annotation, BaseModel):
            for field in model_fields(field.annotation):  # noqa: B020, PLR1704
                if field.name == key:
                    return field
        return None

    def _explode_env_vars(
        self,
        field: ModelField,
        env_vars: dict[str, str | None],
        env_file_vars: dict[str, str | None],
    ) -> dict[str, Any]:
        if self.env_nested_delimiter is None:
            return {}

        prefix = f"{field.name}{self.env_nested_delimiter}"
        result: dict[str, Any] = {}
        for env_name, env_val in env_vars.items():
            if not env_name.startswith(prefix):
                continue

            # delete from file vars when used
            env_file_vars.pop(env_name, None)

            _, *keys, last_key = env_name.split(self.env_nested_delimiter)
            env_var = result
            target_field: ModelField | None = field
            for key in keys:
                target_field = self._next_field(target_field, key)
                env_var = env_var.setdefault(key, {})

            target_field = self._next_field(target_field, last_key)
            if target_field and env_val:
                is_complex, allow_parse_failure = self._field_is_complex(target_field)
                if is_complex:
                    try:
                        env_val = json.loads(env_val)
                    except ValueError as e:
                        if not allow_parse_failure:
                            raise SettingsError(f'error parsing env var "{env_name}"') from e

            env_var[last_key] = env_val

        return result

    @override
    def __call__(self) -> dict[str, Any]:
        """从环境变量和 dotenv 配置文件中读取配置项。"""

        d: dict[str, Any] = {}

        env_vars = self._parse_env_vars(os.environ)
        env_file_vars = self._read_env_files()
        env_vars = {**env_file_vars, **env_vars}

        for field in model_fields(self.settings_cls):
            field_name = field.name
            env_name = self._apply_case_sensitive(field_name)

            # try get values from env vars
            env_val = env_vars.get(env_name, PydanticUndefined)
            # delete from file vars when used
            if env_name in env_file_vars:
                del env_file_vars[env_name]

            is_complex, allow_parse_failure = self._field_is_complex(field)
            if is_complex:
                if isinstance(env_val, PydanticUndefinedType):
                    # field is complex but no value found so far, try explode_env_vars
                    if env_val_built := self._explode_env_vars(field, env_vars, env_file_vars):
                        d[field_name] = env_val_built
                elif env_val is None:
                    d[field_name] = env_val
                else:
                    # field is complex and there's a value
                    # decode that as JSON, then add explode_env_vars
                    try:
                        env_val = json.loads(env_val)
                    except ValueError as e:
                        if not allow_parse_failure:
                            raise SettingsError(f'error parsing env var "{env_name}"') from e

                    if isinstance(env_val, dict):
                        # field value is a dict
                        # try explode_env_vars to find more sub-values
                        d[field_name] = deep_update(
                            env_val,
                            self._explode_env_vars(field, env_vars, env_file_vars),
                        )
                    else:
                        d[field_name] = env_val
            elif env_val is not PydanticUndefined:
                # simplest case, field is not complex
                # we only need to add the value if it was found
                d[field_name] = env_val

        # remain user custom config
        for env_name in env_file_vars:
            env_val = env_vars[env_name]
            if env_val and (val_striped := env_val.strip()):
                # there's a value, decode that as JSON
                with suppress(ValueError):
                    env_val = json.loads(val_striped)

            # explode value when it's a nested dict
            env_name, *nested_keys = env_name.split(self.env_nested_delimiter)
            if nested_keys and (env_name not in d or isinstance(d[env_name], dict)):
                result = {}
                *keys, last_key = nested_keys
                _tmp = result
                for key in keys:
                    _tmp = _tmp.setdefault(key, {})
                _tmp[last_key] = env_val
                d[env_name] = deep_update(d.get(env_name, {}), result)
            elif not nested_keys:
                d[env_name] = env_val

        return d


class DotFileSettingsSource(BaseSettingsSource):
    def __init__(
        self,
        settings_cls: type["BaseSettings"],
        config_file: str | Path | None = None,
        config_file_encoding: str | None = None,
        case_sensitive: bool | None = None,
    ) -> None:
        super().__init__(settings_cls)
        self.config_file = str(
            config_file
            if config_file is not None
            else self.config.get("config_file", "config.toml")
        )
        self.config_file_encoding = (
            config_file_encoding
            if config_file_encoding is not None
            else self.config.get("config_file_encoding", "utf-8")
        )
        self.case_sensitive = (
            case_sensitive
            if case_sensitive is not None
            else self.config.get("case_sensitive", False)
        )

    def _apply_case_sensitive(self, key: str) -> str:
        """Return *key* untouched when *case_sensitive* is True, else lower case it."""
        return key if self.case_sensitive else key.lower()

    def _convert_keys(self, obj: Any) -> Any:
        """Recursively apply :pymeth:`_apply_case_sensitive` to every dict key."""
        if isinstance(obj, dict):
            return {
                (self._apply_case_sensitive(k) if isinstance(k, str) else k): self._convert_keys(v)
                for k, v in obj.items()
            }
        if isinstance(obj, list):
            return [self._convert_keys(v) for v in obj]
        return obj

    @override
    def __call__(self) -> dict[str, Any]:
        """从配置文件中读取配置项。"""
        d: dict[str, Any] = {}

        if self.config_file is not None and (config_path := Path(self.config_file)).is_file():
            try:
                with config_path.open("rb") as f:
                    if self.config_file.endswith(".json"):
                        d = json.load(f)
                    elif self.config_file.endswith(".toml"):
                        d = tomllib.load(f)
                    elif self.config_file.endswith((".yml", ".yaml")):
                        d = yaml.safe_load(f)
                    else:
                        raise ValueError(
                            "Read config file failed: Unable to determine config file type"
                        )
            except OSError as e:
                raise SettingsError(f"Can not open config file: {self.config_file!r}") from e
            except (
                ValueError,
                json.JSONDecodeError,
                tomllib.TOMLDecodeError,
                yaml.YAMLError,
            ) as e:
                raise SettingsError(f"Read config file failed: {self.config_file!r}") from e

        return self._convert_keys(d or {})


class SettingsConfig(ConfigDict, total=False):
    env_file: DOTENV_TYPE | None
    env_file_encoding: str
    env_nested_delimiter: str | None

    config_file: str | Path | None
    config_file_encoding: str

    case_sensitive: bool


class BaseSettings(BaseModel):
    if TYPE_CHECKING:
        # dummy getattr for pylance checking, actually not used
        def __getattr__(self, name: str) -> Any:  # pragma: no cover
            return self.__dict__.get(name)

    model_config = SettingsConfig(
        extra="allow",
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        config_file="config.toml",
        config_file_encoding="utf-8",
        case_sensitive=False,
    )

    def __init__(
        __settings_self__,  # pyright: ignore[reportSelfClsParameterName]  # noqa: N805
        _env_file: DOTENV_TYPE | None = ENV_FILE_SENTINEL,
        _env_file_encoding: str | None = None,
        _env_nested_delimiter: str | None = None,
        _config_file: str | Path | None = None,
        _config_file_encoding: str | None = None,
        **values: Any,
    ) -> None:
        super().__init__(
            **__settings_self__._settings_build_values(
                values,
                env_file=_env_file,
                env_file_encoding=_env_file_encoding,
                env_nested_delimiter=_env_nested_delimiter,
                config_file=_config_file,
                config_file_encoding=_config_file_encoding,
            )
        )

    def _settings_build_values(
        self,
        init_kwargs: dict[str, Any],
        env_file: DOTENV_TYPE | None = None,
        env_file_encoding: str | None = None,
        env_nested_delimiter: str | None = None,
        config_file: str | Path | None = None,
        config_file_encoding: str | None = None,
    ) -> dict[str, Any]:
        init_settings = InitSettingsSource(self.__class__, init_kwargs=init_kwargs)
        env_settings = DotEnvSettingsSource(
            self.__class__,
            env_file=env_file,
            env_file_encoding=env_file_encoding,
            env_nested_delimiter=env_nested_delimiter,
        )
        config_settings = DotFileSettingsSource(
            self.__class__,
            config_file=config_file,
            config_file_encoding=config_file_encoding,
        )
        return deep_update(config_settings(), env_settings(), init_settings())


class Env(BaseSettings):
    environment: str = "prod"


class PromptValues(BaseModel):
    """Base class for prompt values."""

    en: str | None = None
    zh: str | None = None


class ShortMemoryConfig(BaseModel):
    """Configuration for short-term memory prompt values."""

    summary: PromptValues = PromptValues()
    tags: PromptValues = PromptValues()
    prompt: PromptValues = PromptValues()


class LongMemoryConfig(BaseModel):
    """Configuration for long-term memory prompt values."""

    extract: PromptValues = PromptValues()
    update: PromptValues = PromptValues()
    prompt: PromptValues = PromptValues()


class PromptValuesConfig(BaseModel):
    """Configuration for prompt values."""

    chat: PromptValues = PromptValues()
    agent: PromptValues = PromptValues()

    end_detector: PromptValues = PromptValues()
    short_memory: ShortMemoryConfig = ShortMemoryConfig()
    long_memory: LongMemoryConfig = LongMemoryConfig()


class Config(BaseSettings):
    if TYPE_CHECKING:
        _env_file: DOTENV_TYPE | None = ".env", ".env.prod"
        _config_file: str | Path | None = "config.toml"

    index_name: str = "demo"
    folder_path: str | Path = Path("./.cache/")

    language: str = "zh"

    embeddings_model: str = "openai/text-embedding-ada-002"

    chat_model: str = "openai/gpt-4.1"
    chat_temperature: float = 1.0

    agent_model: str = "openai/gpt-4.1"
    agent_temperature: float = 1.0

    short_memory_model: str = "openai/gpt-4.1-mini"
    long_memory_model: str = "openai/gpt-o3"
    end_detector_model: str = "openai/gpt-4.1-mini"

    prompt_values: PromptValuesConfig = PromptValuesConfig()

    model_config = SettingsConfig(env_file=(".env", ".env.prod"))
