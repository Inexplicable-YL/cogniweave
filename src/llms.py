import os
from typing import Any, Generic, Literal, Self, TypeVar, cast
from typing_extensions import override

import openai
from langchain_core.messages import (
    BaseMessage,
)
from langchain_core.output_parsers import (
    BaseOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
    StrOutputParser,
)
from langchain_core.prompts.chat import (
    BaseChatPromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.prompts.message import (
    BaseMessagePromptTemplate,
)
from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI as BaseChatOpenAI
from langchain_openai import OpenAIEmbeddings as BaseOpenAIEmbeddings
from langchain_openai.chat_models.base import global_ssl_context
from pydantic import BaseModel, Field, SecretStr, model_validator

from src.prompt_values.summary import DEFAULT_SINGLE_TURN_PROMPT_ZH, DEFAULT_SINGLE_TURN_PROMPT_EN

Output = TypeVar("Output", covariant=True)  # noqa: PLC0105
PydanticOutput = TypeVar("PydanticOutput", bound=BaseModel, covariant=True)  # noqa: PLC0105

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
            or os.getenv(f"{provider}_ORG_ID")
            or os.getenv(f"{provider}_ORGANIZATION")
        )
        self.openai_api_key = self.openai_api_key or SecretStr(
            os.getenv(f"{provider}_API_KEY") or ""
        )
        self.openai_api_base = self.openai_api_base or os.getenv(f"{provider}_API_BASE")
        self.openai_proxy = self.openai_proxy or os.getenv(f"{provider}_PROXY")
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
            or os.getenv(f"{provider}_ORG_ID")
            or os.getenv(f"{provider}_ORGANIZATION")
        )
        self.openai_api_key = self.openai_api_key or SecretStr(
            os.getenv(f"{provider}_API_KEY") or ""
        )
        self.openai_api_base = self.openai_api_base or os.getenv(f"{provider}_API_BASE")
        self.openai_proxy = self.openai_proxy or os.getenv(f"{provider}_PROXY")
        self.openai_api_version = self.openai_api_version or os.getenv(f"{provider}_API_VERSION")
        self.openai_api_type = self.openai_api_type or os.getenv(f"{provider}_API_TYPE")
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


class SingleTurnChatBase(RunnableSerializable[dict[str, Any], Output], Generic[Output]):
    """A base class for single-turn chat models."""

    lang: Literal["en", "zh"] = Field(default="zh")
    provider: str = Field(default="openai")
    client: BaseChatOpenAI | ChatOpenAI | None = Field(alias="llm", default=None)
    prompt: MessageLikeRepresentation | None = None
    parser: BaseOutputParser[Any] | None = None
    chain: RunnableSerializable[dict[str, Any], Output] | None = None

    response_format: dict[str, Any] | type[BaseModel] | None = None

    model_name: str = Field(default="gpt-3.5-turbo", alias="model")
    temperature: float = Field(default=0.7)

    client_params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def build_chain_if_needed(self) -> Self:
        """Automatically build the chain if it is not provided."""
        if self.chain is None:
            self.client = self.client or ChatOpenAI(
                provider=self.provider,
                model=self.model_name,
                temperature=self.temperature,
                **self.client_params,
            )
            if self.response_format:
                self.client = cast(
                    "BaseChatOpenAI", self.client.bind(response_format=self.response_format)
                )
            if self.prompt is None:
                if self.lang == "zh":
                    self.prompt = SystemMessagePromptTemplate.from_template(
                        DEFAULT_SINGLE_TURN_PROMPT_ZH
                    )
                else:
                    self.prompt = SystemMessagePromptTemplate.from_template(
                        DEFAULT_SINGLE_TURN_PROMPT_EN
                    )
            
            prompt_template = ChatPromptTemplate.from_messages(
                [self.prompt, HumanMessagePromptTemplate.from_template("{input}")]
            )
            self.parser = self.parser or StrOutputParser()
            self.chain = cast(
                "RunnableSerializable[dict[str, Any], Output]",
                prompt_template | self.client | self.parser,
            )
        return self

    @override
    def invoke(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        """Synchronous call to the single-turn chat model."""
        assert self.chain is not None
        return self.chain.invoke(input, config=config, **kwargs)

    @override
    async def ainvoke(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> Output:
        """Asynchronous call to the single-turn chat model."""
        assert self.chain is not None
        return await self.chain.ainvoke(input, config=config, **kwargs)


class StringSingleTurnChat(SingleTurnChatBase[str]):
    """A single-turn chat model that returns a string response."""

    response_format: dict[str, Any] | type[BaseModel] | None = None
    parser: BaseOutputParser[Any] | None = StrOutputParser()

    def __init__(
        self,
        provider: str = "openai",
        *,
        lang: Literal["en", "zh"] = "zh",
        llm: BaseChatOpenAI | ChatOpenAI | None = None,
        prompt: MessageLikeRepresentation | None = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        llm_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        params = {
            "lang": lang,
            "provider": provider,
            "llm": llm,
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            "client_params": llm_params or {},
        }
        super().__init__(**params, **kwargs)  # type: ignore[arg-type]


class JsonSingleTurnChat(SingleTurnChatBase[dict[Any, Any]]):
    """A single-turn chat model that returns a JSON response."""

    response_format: dict[str, Any] | type[BaseModel] | None = Field(
        default={"type": "json_object"}
    )
    parser: BaseOutputParser[Any] | None = JsonOutputParser()

    def __init__(
        self,
        provider: str = "openai",
        *,
        lang: Literal["en", "zh"] = "zh",
        llm: BaseChatOpenAI | ChatOpenAI | None = None,
        prompt: MessageLikeRepresentation | None = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        llm_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        params = {
            "lang": lang,
            "provider": provider,
            "llm": llm,
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            "client_params": llm_params or {},
        }
        super().__init__(**params, **kwargs)  # type: ignore[arg-type]


class PydanticSingleTurnChat(SingleTurnChatBase[PydanticOutput], Generic[PydanticOutput]):
    """A single-turn chat model that returns a Pydantic model response."""

    response_format: dict[str, Any] | type[BaseModel] | None = None

    structured_output: bool = True
    """Whether to use structured output."""
    parser: BaseOutputParser[Any] | None = None

    def __init__(
        self,
        template: type[PydanticOutput],
        provider: str = "openai",
        *,
        lang: Literal["en", "zh"] = "zh",
        structured_output: bool = True,
        llm: BaseChatOpenAI | ChatOpenAI | None = None,
        prompt: MessageLikeRepresentation | None = None,
        model: str = "gpt-4.1-mini",
        temperature: float = 0.7,
        llm_params: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        params = {
            "lang": lang,
            "provider": provider,
            "response_format": template,
            "structured_output": structured_output,
            "llm": llm,
            "prompt": prompt,
            "model": model,
            "temperature": temperature,
            "client_params": llm_params or {},
        }
        super().__init__(**params, **kwargs)  # type: ignore[arg-type]

    @model_validator(mode="after")
    def build_chain_if_needed(self) -> Self:
        """Automatically build the chain if it is not provided."""
        if self.chain is None:
            self.client = self.client or ChatOpenAI(
                provider=self.provider,
                model=self.model_name,
                temperature=self.temperature,
                **self.client_params,
            )
            self.client = cast(
                "BaseChatOpenAI",
                self.client.bind(
                    response_format=self.response_format
                    if self.structured_output
                    else {"type": "json_object"}
                ),
            )
            if self.prompt is None:
                if self.lang == "zh":
                    self.prompt = SystemMessagePromptTemplate.from_template(
                        DEFAULT_SINGLE_TURN_PROMPT_ZH
                    )
                else:
                    self.prompt = SystemMessagePromptTemplate.from_template(
                        DEFAULT_SINGLE_TURN_PROMPT_EN
                    )
            
            self.parser = PydanticOutputParser(
                pydantic_object=cast("type[PydanticOutput]", self.response_format)
            )
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    self.prompt,
                    *(
                        [SystemMessagePromptTemplate.from_template("{format_instructions}")]
                        if not self.structured_output
                        else []
                    ),
                    HumanMessagePromptTemplate.from_template("{input}"),
                ]
            )
            if not self.structured_output:
                prompt_template = prompt_template.partial(
                    format_instructions=self.parser.get_format_instructions()
                )
            self.chain = prompt_template | self.client | self.parser
        return self
