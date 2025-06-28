from collections.abc import AsyncIterator, Iterator
from typing import (
    Any,
    Generic,
    Self,
    cast,
)
from typing_extensions import override

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.output_parsers import (
    BaseOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
    StrOutputParser,
)
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI as BaseChatOpenAI
from pydantic import BaseModel, Field, model_validator

from cogniweave.prompt_values.base import MultilingualSystemPromptValue
from cogniweave.typing import (
    MessageLikeRepresentation,
    Output,
    PydanticOutput,
    SupportLangType,
)

from .base import ChatOpenAI


def _get_verbosity() -> bool:
    from langchain.globals import get_verbose

    return get_verbose()


class SingleTurnChatBase(
    RunnableSerializable[dict[str, Any], Output], Generic[SupportLangType, Output]
):
    """A base class for single-turn chat models."""

    # Language code for multilingual prompt handling
    lang: SupportLangType

    # Model configuration
    provider: str = Field(default="openai")
    model_name: str = Field(default="gpt-4", alias="model")
    temperature: float = Field(default=0.0)
    client_params: dict[str, Any] = Field(default_factory=dict)

    # Optional LLM client override
    client: BaseChatOpenAI | None = Field(alias="llm", default=None)

    # System prompt handler (multilingual support)
    prompt: MultilingualSystemPromptValue[SupportLangType] | None = None

    # Custom contexts used by the agent
    contexts: list[MessageLikeRepresentation] = Field(default_factory=list)

    # Output parser
    parser: BaseOutputParser[Any] | None = None

    # Internally built chain (AgentExecutor)
    chain: RunnableSerializable[dict[str, Any], Output] | None = None

    # Response format
    response_format: dict[str, Any] | type[BaseModel] | None = None

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
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    *(self.prompt.to_messages(lang=self.lang) if self.prompt else []),
                    *self.contexts,
                    HumanMessagePromptTemplate.from_template("{input}"),
                ]
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

    @override
    def stream(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> Iterator[Output]:
        assert self.chain is not None
        yield from self.chain.stream(input, config=config, **kwargs)

    @override
    async def astream(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> AsyncIterator[Output]:
        assert self.chain is not None
        async for chunk in self.chain.astream(input, config=config, **kwargs):
            yield chunk


class StringSingleTurnChat(SingleTurnChatBase[SupportLangType, str], Generic[SupportLangType]):
    """A single-turn chat model that returns a string response."""

    response_format: dict[str, Any] | type[BaseModel] | None = None
    parser: BaseOutputParser[Any] | None = StrOutputParser()


class JsonSingleTurnChat(
    SingleTurnChatBase[SupportLangType, dict[Any, Any]], Generic[SupportLangType]
):
    """A single-turn chat model that returns a JSON response."""

    response_format: dict[str, Any] | type[BaseModel] | None = Field(
        default={"type": "json_object"}
    )
    parser: BaseOutputParser[Any] | None = JsonOutputParser()


class PydanticSingleTurnChat(
    SingleTurnChatBase[SupportLangType, PydanticOutput], Generic[SupportLangType, PydanticOutput]
):
    """A single-turn chat model that returns a Pydantic model response."""

    structured_output: bool = True
    """Whether to use structured output."""
    parser: BaseOutputParser[Any] | None = None

    @model_validator(mode="after")
    def build_chain_if_needed(self) -> Self:
        """Automatically build the chain if it is not provided."""
        self.response_format = self.response_format or self.OutputType
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

            self.parser = PydanticOutputParser(
                pydantic_object=cast("type[PydanticOutput]", self.response_format)
            )

            prompt_template = ChatPromptTemplate.from_messages(
                [
                    *(self.prompt.to_messages(lang=self.lang) if self.prompt else []),
                    *(
                        [SystemMessagePromptTemplate.from_template("{format_instructions}")]
                        if not self.structured_output
                        else []
                    ),
                    *self.contexts,
                    HumanMessagePromptTemplate.from_template("{input}"),
                ]
            )

            if not self.structured_output:
                prompt_template = prompt_template.partial(
                    format_instructions=self.parser.get_format_instructions()
                )

            self.chain = prompt_template | self.client | self.parser
        return self


class AgentBase(RunnableSerializable[dict[str, Any], dict[str, Any]], Generic[SupportLangType]):
    """
    Base class for creating a Function Calling Agent using LangChain.
    Automatically builds a chain from a prompt template, OpenAI-compatible model,
    and list of tools.
    """

    # Language code for multilingual prompt handling
    lang: SupportLangType

    # Model configuration
    provider: str = Field(default="openai")
    model_name: str = Field(default="gpt-4", alias="model")
    temperature: float = Field(default=0.0)
    client_params: dict[str, Any] = Field(default_factory=dict)

    # Optional LLM client override
    client: BaseChatOpenAI | None = Field(alias="llm", default=None)

    # System prompt handler (multilingual support)
    prompt: MultilingualSystemPromptValue[SupportLangType] | None = None

    # Custom contexts used by the agent
    contexts: list[MessageLikeRepresentation] = Field(default_factory=list)

    # External tools used by the agent
    tools: list[BaseTool] = Field(default_factory=list)

    # Internally built chain (AgentExecutor)
    chain: RunnableSerializable[dict[str, Any], dict[str, Any]] | None = None

    # Verbosity
    verbose: bool = Field(default_factory=_get_verbosity)

    @model_validator(mode="after")
    def build_chain_if_needed(self) -> Self:
        """
        Automatically build the Agent chain if not already provided.
        Combines prompt, tools, and OpenAI-compatible model.
        """
        if self.chain is None:
            # create or reuse model
            self.client = self.client or ChatOpenAI(
                provider=self.provider,
                model=self.model_name,
                temperature=self.temperature,
                streaming=True,
                **self.client_params,
            )
            # build prompt template (with multilingual system prompt)
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    *(self.prompt.to_messages(lang=self.lang) if self.prompt else []),
                    *self.contexts,
                    HumanMessagePromptTemplate.from_template("{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )
            # build function-calling agent and executor
            agent = create_openai_functions_agent(
                llm=self.client,
                tools=self.tools,
                prompt=prompt_template,
            )
            # build executor
            self.chain = AgentExecutor(agent=agent, tools=self.tools, verbose=self.verbose)

        return self

    @override
    def invoke(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Synchronously invoke the agent."""
        assert self.chain is not None
        return self.chain.invoke(input, config=config, **kwargs)

    @override
    async def ainvoke(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Asynchronously invoke the agent."""
        assert self.chain is not None
        return await self.chain.ainvoke(input, config=config, **kwargs)

    @override
    def stream(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> Iterator[dict[str, Any]]:
        assert self.chain is not None
        yield from self.chain.stream(input, config=config, **kwargs)

    @override
    async def astream(
        self, input: dict[str, Any], config: RunnableConfig | None = None, **kwargs: Any
    ) -> AsyncIterator[dict[str, Any]]:
        assert self.chain is not None
        async for chunk in self.chain.astream(input, config=config, **kwargs):
            yield chunk
