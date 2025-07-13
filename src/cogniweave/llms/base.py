from typing import Any, Self
from typing_extensions import override

import openai
from langchain_openai import ChatOpenAI as BaseChatOpenAI
from langchain_openai import OpenAIEmbeddings as BaseOpenAIEmbeddings
from langchain_openai.chat_models.base import global_ssl_context
from pydantic import Field, SecretStr, model_validator

from cogniweave.utils import get_from_config_or_env


class ChatOpenAI(BaseChatOpenAI):
    """Wrapper around OpenAI's Chat API, with dynamic env key loading based on provider."""

    provider: str = Field(default="openai")

    # Defaults are placeholders; they will be overridden in post-init
    openai_api_key: SecretStr | None = Field(alias="api_key", default=None)
    openai_api_base: str | None = Field(alias="base_url", default=None)
    openai_proxy: str | None = None

    @model_validator(mode="after")
    @override
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        if self.n is not None and self.n < 1:
            raise ValueError("n must be at least 1.")
        if self.n is not None and self.n > 1 and self.streaming:
            raise ValueError("n must be 1 when streaming.")

        provider = self.provider.upper()

        # Check OPENAI_ORGANIZATION for backwards compatibility.
        self.openai_organization = (
            self.openai_organization
            or get_from_config_or_env(f"{provider}_ORG_ID", default=None)()
            or get_from_config_or_env(f"{provider}_ORGANIZATION", default=None)()
        )
        self.openai_api_key = self.openai_api_key or SecretStr(
            get_from_config_or_env(f"{provider}_API_KEY", default=None)() or ""
        )
        self.openai_api_base = (
            self.openai_api_base or get_from_config_or_env(f"{provider}_API_BASE", default=None)()
        )
        self.openai_proxy = (
            self.openai_proxy or get_from_config_or_env(f"{provider}_PROXY", default=None)()
        )
        client_params: dict[str, Any] = {
            "api_key": (self.openai_api_key.get_secret_value() if self.openai_api_key else None),
            "organization": self.openai_organization,
            "base_url": self.openai_api_base,
            "timeout": self.request_timeout,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }
        if self.max_retries is not None:
            client_params["max_retries"] = self.max_retries

        if self.openai_proxy and (self.http_client or self.http_async_client):
            openai_proxy = self.openai_proxy
            http_client = self.http_client
            http_async_client = self.http_async_client
            raise ValueError(
                "Cannot specify 'openai_proxy' if one of "
                "'http_client'/'http_async_client' is already specified. Received:\n"
                f"{openai_proxy=}\n{http_client=}\n{http_async_client=}"
            )
        if not self.client:
            if self.openai_proxy and not self.http_client:
                try:
                    import httpx
                except ImportError as e:
                    raise ImportError(
                        "Could not import httpx python package. "
                        "Please install it with `pip install httpx`."
                    ) from e
                self.http_client = httpx.Client(proxy=self.openai_proxy, verify=global_ssl_context)
            sync_specific = {"http_client": self.http_client}
            self.root_client = openai.OpenAI(**client_params, **sync_specific)  # type: ignore[arg-type]
            self.client = self.root_client.chat.completions
        if not self.async_client:
            if self.openai_proxy and not self.http_async_client:
                try:
                    import httpx
                except ImportError as e:
                    raise ImportError(
                        "Could not import httpx python package. "
                        "Please install it with `pip install httpx`."
                    ) from e
                self.http_async_client = httpx.AsyncClient(
                    proxy=self.openai_proxy, verify=global_ssl_context
                )
            async_specific = {"http_client": self.http_async_client}
            self.root_async_client = openai.AsyncOpenAI(
                **client_params,  # type: ignore
                **async_specific,  # type: ignore[arg-type]
            )
            self.async_client = self.root_async_client.chat.completions
        return self


class OpenAIEmbeddings(BaseOpenAIEmbeddings):
    """Wrapper around OpenAI's Embedding API, with dynamic env key loading based on provider."""

    provider: str = Field(default="openai")

    # Defaults are placeholders; they will be overridden in post-init
    openai_api_key: SecretStr | None = Field(alias="api_key", default=None)
    openai_api_base: str | None = Field(alias="base_url", default=None)
    # to support explicit proxy for OpenAI
    openai_proxy: str | None = None
    openai_api_version: str | None = Field(alias="api_version", default=None)
    """Automatically inferred from env var `OPENAI_API_VERSION` if not provided."""
    openai_api_type: str | None = None
    openai_organization: str | None = Field(alias="organization", default=None)

    @model_validator(mode="after")
    @override
    def validate_environment(self) -> Self:
        """Validate that api key and python package exists in environment."""
        if self.openai_api_type in ("azure", "azure_ad", "azuread"):
            raise ValueError(
                "If you are using Azure, please use the `AzureOpenAIEmbeddings` class."
            )

        provider = self.provider.upper()

        self.openai_organization = (
            self.openai_organization
            or get_from_config_or_env(f"{provider}_ORG_ID", default=None)()
            or get_from_config_or_env(f"{provider}_ORGANIZATION", default=None)()
        )
        self.openai_api_key = self.openai_api_key or SecretStr(
            get_from_config_or_env(f"{provider}_API_KEY", default=None)() or ""
        )
        self.openai_api_base = (
            self.openai_api_base or get_from_config_or_env(f"{provider}_API_BASE", default=None)()
        )
        self.openai_proxy = (
            self.openai_proxy or get_from_config_or_env(f"{provider}_PROXY", default=None)()
        )
        self.openai_api_version = (
            self.openai_api_version
            or get_from_config_or_env(f"{provider}_API_VERSION", default=None)()
        )
        self.openai_api_type = (
            self.openai_api_type or get_from_config_or_env(f"{provider}_API_TYPE", default=None)()
        )
        client_params: dict[str, Any] = {
            "api_key": (self.openai_api_key.get_secret_value() if self.openai_api_key else None),
            "organization": self.openai_organization,
            "base_url": self.openai_api_base,
            "timeout": self.request_timeout,
            "max_retries": self.max_retries,
            "default_headers": self.default_headers,
            "default_query": self.default_query,
        }

        if self.openai_proxy and (self.http_client or self.http_async_client):
            openai_proxy = self.openai_proxy
            http_client = self.http_client
            http_async_client = self.http_async_client
            raise ValueError(
                "Cannot specify 'openai_proxy' if one of "
                "'http_client'/'http_async_client' is already specified. Received:\n"
                f"{openai_proxy=}\n{http_client=}\n{http_async_client=}"
            )
        if not self.client:
            if self.openai_proxy and not self.http_client:
                try:
                    import httpx
                except ImportError as e:
                    raise ImportError(
                        "Could not import httpx python package. "
                        "Please install it with `pip install httpx`."
                    ) from e
                self.http_client = httpx.Client(proxy=self.openai_proxy)
            sync_specific = {"http_client": self.http_client}
            self.client = openai.OpenAI(**client_params, **sync_specific).embeddings  # type: ignore[arg-type]
        if not self.async_client:
            if self.openai_proxy and not self.http_async_client:
                try:
                    import httpx
                except ImportError as e:
                    raise ImportError(
                        "Could not import httpx python package. "
                        "Please install it with `pip install httpx`."
                    ) from e
                self.http_async_client = httpx.AsyncClient(proxy=self.openai_proxy)
            async_specific = {"http_client": self.http_async_client}
            self.async_client = openai.AsyncOpenAI(
                **client_params,  # type: ignore[arg-type]
                **async_specific,  # type: ignore[arg-type]
            ).embeddings
        return self
