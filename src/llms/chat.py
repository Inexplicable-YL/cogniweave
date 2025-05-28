from typing import Any, Generic, Self, cast
from typing_extensions import override

from langchain_core.output_parsers import (
    BaseOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
    StrOutputParser,
)
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_core.runnables import RunnableSerializable
from langchain_core.runnables.config import RunnableConfig
from langchain_openai import ChatOpenAI as BaseChatOpenAI
from pydantic import BaseModel, Field, model_validator

from src.prompt_values.base import MultilingualSystemPromptValue
from src.typing import (
    Output,
    PydanticOutput,
    SupportLangType,
)

from .base import ChatOpenAI


class SingleTurnChatBase(
    RunnableSerializable[dict[str, Any], Output], Generic[SupportLangType, Output]
):
    """A base class for single-turn chat models."""

    lang: SupportLangType

    provider: str = Field(default="openai")
    client: BaseChatOpenAI | ChatOpenAI | None = Field(alias="llm", default=None)

    prompt: MultilingualSystemPromptValue[SupportLangType] | None = None
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
            prompt_template = ChatPromptTemplate.from_messages(
                [
                    *(self.prompt.to_messages(lang=self.lang) if self.prompt else []),
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
                    HumanMessagePromptTemplate.from_template("{input}"),
                ]
            )

            if not self.structured_output:
                prompt_template = prompt_template.partial(
                    format_instructions=self.parser.get_format_instructions()
                )

            self.chain = prompt_template | self.client | self.parser
        return self
