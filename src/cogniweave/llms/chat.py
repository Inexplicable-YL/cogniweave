from collections.abc import Sequence
from typing import (
    Any,
    Generic,
    cast,
)

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_core.messages.base import BaseMessage
from langchain_core.output_parsers import (
    BaseOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
    StrOutputParser,
)
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.base import Runnable, RunnableBindingBase
from langchain_core.runnables.passthrough import RunnablePassthrough
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI as BaseChatOpenAI
from pydantic import BaseModel, Field

from cogniweave.prompt_values import MultilingualSystemPromptValue
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
    RunnableBindingBase[dict[str, Any], Output], Generic[SupportLangType, Output]
):
    """A base class for single-turn chat models."""

    # Language code for multilingual prompt handling
    lang: SupportLangType

    # Model configuration
    provider: str = Field(default="openai")
    model_name: str = Field(default="gpt-4", alias="model")
    temperature: float = Field(default=0.0)
    client_params: dict[str, Any] = Field(default_factory=dict)

    # System prompt handler (multilingual support)
    prompt: MultilingualSystemPromptValue[SupportLangType] | None = None

    # Custom contexts used by the agent
    contexts: list[MessageLikeRepresentation] = Field(default_factory=list)

    # Output parser
    parser: BaseOutputParser[Any] = Field(default_factory=StrOutputParser)

    # Response format
    response_format: dict[str, Any] | type[BaseModel] | None = None

    def __init__(
        self,
        *,
        lang: SupportLangType,
        provider: str = "openai",
        model: str = "gpt-4",
        temperature: float = 0.0,
        client_params: dict[str, Any] | None = None,
        client: BaseChatOpenAI | None = None,
        prompt: MultilingualSystemPromptValue[SupportLangType] | None = None,
        contexts: list[MessageLikeRepresentation] | None = None,
        parser: BaseOutputParser[Any] | None = None,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        **kwargs: Any,
    ) -> None:
        client = client or ChatOpenAI(
            provider=provider,
            model=model,
            temperature=temperature,
            **client_params or {},
        )
        if response_format:
            client = cast(
                "BaseChatOpenAI",
                client.bind(response_format=self.response_format).with_config(
                    run_name="run_with_response_format"
                ),
            )
        prompt_template = kwargs.pop("prompt_template", None) or ChatPromptTemplate.from_messages(
            [
                *(prompt.to_messages(lang=lang) if prompt else []),
                *(contexts or []),
                MessagesPlaceholder(variable_name="input"),
            ]
        )
        parser = parser or StrOutputParser()
        runnable = cast(
            "RunnableSerializable[dict[str, Any], Output]",
            prompt_template | client | parser,
        ).with_config(run_name="runnable")
        format_input_chain = RunnablePassthrough.assign(input=self._get_input_messages).with_config(
            run_name="format_input"
        )
        bound: Runnable = (format_input_chain | runnable).with_config(run_name="SingleTurnChatBase")

        super().__init__(
            lang=lang,
            bound=bound,
            provider=provider,
            model=model,
            temperature=temperature,
            client_params=client_params,
            client=client,
            prompt=prompt,
            contexts=contexts,
            parser=parser,
            response_format=response_format,
            **kwargs,
        )

    def _get_input_messages(
        self, value: str | BaseMessage | Sequence[BaseMessage] | dict
    ) -> list[BaseMessage]:
        from langchain_core.messages import BaseMessage

        # If dictionary, try to pluck the single key representing messages
        if isinstance(value, dict):
            key = next(iter(value.keys())) if len(value) == 1 else "input"
            value = value[key]

        # If value is a string, convert to a human message
        if isinstance(value, str):
            from langchain_core.messages import HumanMessage

            return [HumanMessage(content=value)]
        # If value is a single message, convert to a list
        if isinstance(value, BaseMessage):
            return [value]
        # If value is a list or tuple...
        if isinstance(value, (list, tuple)):
            # Handle empty case
            if len(value) == 0:
                return list(value)
            # If is a list of list, then return the first value
            # This occurs for chat models - since we batch inputs
            if isinstance(value[0], list):
                if len(value) != 1:
                    msg = f"Expected a single list of messages. Got {value}."
                    raise ValueError(msg)
                return value[0]
            return list(value)
        msg = f"Expected str, BaseMessage, list[BaseMessage], or tuple[BaseMessage]. Got {value}."
        raise ValueError(msg)


class StringSingleTurnChat(SingleTurnChatBase[SupportLangType, str], Generic[SupportLangType]):
    """A single-turn chat model that returns a string response."""

    def __init__(
        self,
        *,
        lang: SupportLangType,
        provider: str = "openai",
        model: str = "gpt-4",
        temperature: float = 0.0,
        client_params: dict[str, Any] | None = None,
        client: BaseChatOpenAI | None = None,
        prompt: MultilingualSystemPromptValue[SupportLangType] | None = None,
        contexts: list[MessageLikeRepresentation] | None = None,
        **kwargs: Any,
    ) -> None:
        parser = StrOutputParser()
        super().__init__(
            lang=lang,
            provider=provider,
            model=model,
            temperature=temperature,
            client_params=client_params,
            client=client,
            prompt=prompt,
            contexts=contexts,
            parser=parser,
            **kwargs,
        )


class JsonSingleTurnChat(
    SingleTurnChatBase[SupportLangType, dict[Any, Any]], Generic[SupportLangType]
):
    """A single-turn chat model that returns a JSON response."""

    def __init__(
        self,
        *,
        lang: SupportLangType,
        provider: str = "openai",
        model: str = "gpt-4",
        temperature: float = 0.0,
        client_params: dict[str, Any] | None = None,
        client: BaseChatOpenAI | None = None,
        prompt: MultilingualSystemPromptValue[SupportLangType] | None = None,
        contexts: list[MessageLikeRepresentation] | None = None,
        **kwargs: Any,
    ) -> None:
        response_format = {"type": "json_object"}
        parser = JsonOutputParser()
        super().__init__(
            lang=lang,
            provider=provider,
            model=model,
            temperature=temperature,
            client_params=client_params,
            client=client,
            prompt=prompt,
            contexts=contexts,
            parser=parser,
            response_format=response_format,
            **kwargs,
        )


class PydanticSingleTurnChat(
    SingleTurnChatBase[SupportLangType, PydanticOutput], Generic[SupportLangType, PydanticOutput]
):
    """A single-turn chat model that returns a Pydantic model response."""

    structured_output: bool = True

    def __init__(
        self,
        response_format: dict[str, Any] | type[BaseModel] | None = None,
        structured_output: bool = True,
        *,
        lang: SupportLangType,
        provider: str = "openai",
        model: str = "gpt-4",
        temperature: float = 0.0,
        client_params: dict[str, Any] | None = None,
        client: BaseChatOpenAI | None = None,
        prompt: MultilingualSystemPromptValue[SupportLangType] | None = None,
        contexts: list[MessageLikeRepresentation] | None = None,
        **kwargs: Any,
    ) -> None:
        response_format = (
            (response_format or self.OutputType) if structured_output else {"type": "json_object"}
        )
        parser = PydanticOutputParser(pydantic_object=cast("type[PydanticOutput]", response_format))
        prompt_template = ChatPromptTemplate.from_messages(
            [
                *(prompt.to_messages(lang=lang) if prompt else []),
                *(
                    [SystemMessagePromptTemplate.from_template(parser.get_format_instructions())]
                    if not structured_output
                    else []
                ),
                *(contexts or []),
                MessagesPlaceholder(variable_name="input"),
            ]
        )
        super().__init__(
            lang=lang,
            provider=provider,
            model=model,
            temperature=temperature,
            client_params=client_params,
            client=client,
            prompt=prompt,
            contexts=contexts,
            parser=parser,
            response_format=response_format,
            structured_output=structured_output,
            prompt_template=prompt_template,
            **kwargs,
        )


class AgentBase(RunnableBindingBase[dict[str, Any], dict[str, Any]], Generic[SupportLangType]):
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

    def __init__(
        self,
        *,
        lang: SupportLangType,
        provider: str = "openai",
        model: str = "gpt-4",
        temperature: float = 0.0,
        client_params: dict[str, Any] | None = None,
        client: BaseChatOpenAI | None = None,
        prompt: MultilingualSystemPromptValue[SupportLangType] | None = None,
        contexts: list[MessageLikeRepresentation] | None = None,
        tools: list[BaseTool] | None = None,
        **kwargs: Any,
    ) -> None:
        client = client or ChatOpenAI(
            provider=provider,
            model=model,
            temperature=temperature,
            **client_params or {},
        )
        prompt_template = kwargs.pop("prompt_template", None) or ChatPromptTemplate.from_messages(
            [
                *(prompt.to_messages(lang=lang) if prompt else []),
                *(contexts or []),
                MessagesPlaceholder(variable_name="input"),
                MessagesPlaceholder(variable_name="agent_scratchpad"),
            ]
        )
        tools = tools or []
        agent = create_openai_functions_agent(
            llm=client,
            tools=tools,
            prompt=prompt_template,
        )
        format_input_chain = RunnablePassthrough.assign(input=self._get_input_messages).with_config(
            run_name="format_input"
        )
        runnable = AgentExecutor(
            agent=agent, tools=tools, verbose=kwargs.get("verbose", _get_verbosity())
        )
        bound: Runnable = (format_input_chain | runnable).with_config(run_name="SingleTurnChatBase")

        super().__init__(
            lang=lang,
            bound=bound,
            provider=provider,
            model=model,
            temperature=temperature,
            client_params=client_params,
            client=client,
            prompt=prompt,
            contexts=contexts,
            **kwargs,
        )

    def _get_input_messages(
        self, value: str | BaseMessage | Sequence[BaseMessage] | dict
    ) -> list[BaseMessage]:
        from langchain_core.messages import BaseMessage

        # If dictionary, try to pluck the single key representing messages
        if isinstance(value, dict):
            key = next(iter(value.keys())) if len(value) == 1 else "input"
            value = value[key]

        # If value is a string, convert to a human message
        if isinstance(value, str):
            from langchain_core.messages import HumanMessage

            return [HumanMessage(content=value)]
        # If value is a single message, convert to a list
        if isinstance(value, BaseMessage):
            return [value]
        # If value is a list or tuple...
        if isinstance(value, (list, tuple)):
            # Handle empty case
            if len(value) == 0:
                return list(value)
            # If is a list of list, then return the first value
            # This occurs for chat models - since we batch inputs
            if isinstance(value[0], list):
                if len(value) != 1:
                    msg = f"Expected a single list of messages. Got {value}."
                    raise ValueError(msg)
                return value[0]
            return list(value)
        msg = f"Expected str, BaseMessage, list[BaseMessage], or tuple[BaseMessage]. Got {value}."
        raise ValueError(msg)
