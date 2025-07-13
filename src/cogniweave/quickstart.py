from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_core.prompts import MessagesPlaceholder

from cogniweave.core.end_detector import EndDetector
from cogniweave.core.runnables.end_detector import RunnableWithEndDetector
from cogniweave.core.runnables.history_store import RunnableWithHistoryStore
from cogniweave.core.runnables.memory_maker import RunnableWithMemoryMaker
from cogniweave.core.time_splitter import TimeSplitter
from cogniweave.core.vector_stores import TagsVectorStore
from cogniweave.history_stores import BaseHistoryStore as HistoryStore
from cogniweave.llms import AgentBase, OpenAIEmbeddings, StringSingleTurnChat
from cogniweave.prompt_values import MultilingualStringPromptValue
from cogniweave.prompts import MessageSegmentsPlaceholder, RichSystemMessagePromptTemplate
from cogniweave.utils import (
    get_from_config_or_env,
    get_model_from_config_or_env,
    get_provider_from_config_or_env,
)

if TYPE_CHECKING:
    from langchain_core.tools import BaseTool

DEF_FOLDER_PATH = Path("./.cache/")


def create_embeddings(
    provider: str | None = None,
    model: str | None = None,
) -> OpenAIEmbeddings:
    """Create default embeddings instance."""
    return OpenAIEmbeddings(
        provider=get_provider_from_config_or_env(
            "EMBEDDINGS_MODEL", default=provider or "openai"
        )(),
        model=get_model_from_config_or_env(
            "EMBEDDINGS_MODEL", default=model or "text-embedding-ada-002"
        )(),
    )


def create_history_store(
    *, index_name: str = "demo", folder_path: str | Path = DEF_FOLDER_PATH
) -> HistoryStore:
    """Create a history store backed by a SQLite database."""
    return HistoryStore(db_url=f"sqlite:///{folder_path}/{index_name}.sqlite")


def create_vector_store(
    embeddings: OpenAIEmbeddings,
    *,
    index_name: str = "demo",
    folder_path: str | Path = DEF_FOLDER_PATH,
) -> TagsVectorStore:
    """Create a vector store for long term memory."""
    return TagsVectorStore(
        folder_path=str(folder_path),
        index_name=index_name,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
        auto_save=True,
    )


def create_chat(
    lang: str | None = None,
    *,
    prompt: str | None = None,
    temperature: float | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> StringSingleTurnChat:
    """Create the base chat agent."""
    from cogniweave.config import get_config

    lang = lang or get_from_config_or_env("LANGUAGE", default="zh")()
    _config = get_config()
    prompt_values = _config.prompt_values.chat.model_dump(exclude_none=True) if _config else {}
    prompt_list = [
        *(
            [prompt]
            if prompt
            else MultilingualStringPromptValue(**prompt_values).to_messages(lang=lang)
        ),
        "\n",
    ]
    return StringSingleTurnChat(
        lang=lang,
        provider=get_provider_from_config_or_env("CHAT_MODEL", default=provider or "openai")(),
        model=get_model_from_config_or_env("CHAT_MODEL", default=model or "gpt-4.1")(),
        temperature=(
            temperature
            if temperature is not None
            else float(get_from_config_or_env("CHAT_TEMPERATURE", default="1.0")())
        ),
        contexts=[
            RichSystemMessagePromptTemplate.from_template(
                [
                    *prompt_list,
                    MessageSegmentsPlaceholder(variable_name="long_memory"),
                ]
            ),
            MessagesPlaceholder(variable_name="history", optional=True),
        ],
    )


def create_agent(
    lang: str | None = None,
    *,
    prompt: str | None = None,
    temperature: float = 1.0,
    tools: list[BaseTool] | None = None,
    provider: str | None = None,
    model: str | None = None,
) -> AgentBase:
    """Create the base chat agent."""
    from cogniweave.config import get_config

    lang = lang or get_from_config_or_env("LANGUAGE", default="zh")()
    _config = get_config()
    prompt_values = _config.prompt_values.agent.model_dump(exclude_none=True) if _config else {}
    prompt_list = [
        *(
            [prompt]
            if prompt
            else MultilingualStringPromptValue(**prompt_values).to_messages(lang=lang)
        ),
        "\n",
    ]
    return AgentBase(
        lang=lang,
        provider=get_provider_from_config_or_env("AGENT_MODEL", default=provider or "openai")(),
        model=get_model_from_config_or_env("AGENT_MODEL", default=model or "gpt-4.1")(),
        temperature=(
            temperature
            if temperature is not None
            else float(get_from_config_or_env("AGENT_TEMPERATURE", default="1.0")())
        ),
        contexts=[
            RichSystemMessagePromptTemplate.from_template(
                [
                    *prompt_list,
                    MessageSegmentsPlaceholder(variable_name="long_memory"),
                ]
            ),
            MessagesPlaceholder(variable_name="history", optional=True),
        ],
        tools=tools or [],
    )


def build_pipeline(
    lang: str | None = None,
    prompt: str | None = None,
    *,
    temperature: float | None = None,
    index_name: str | None = None,
    folder_path: str | Path | None = None,
) -> RunnableWithHistoryStore:
    """Assemble the runnable pipeline used in the demos."""
    from cogniweave.config import get_config

    _config = get_config()
    index_name = index_name or (_config.index_name if _config else "demo")
    folder_path = folder_path or (_config.folder_path if _config else DEF_FOLDER_PATH)
    embeddings = create_embeddings()
    history_store = create_history_store(index_name=index_name, folder_path=folder_path)
    vector_store = create_vector_store(embeddings, index_name=index_name, folder_path=folder_path)
    chat = create_chat(lang=lang, prompt=prompt, temperature=temperature)

    pipeline = RunnableWithMemoryMaker(
        chat,
        history_store=history_store,
        vector_store=vector_store,
        input_messages_key="input",
        history_messages_key="history",
        short_memory_key="short_memory",
        long_memory_key="long_memory",
    )
    pipeline = RunnableWithEndDetector(
        pipeline,
        end_detector=EndDetector(),
        default={"output": []},
        history_messages_key="history",
    )
    return RunnableWithHistoryStore(
        pipeline,
        history_store=history_store,
        time_splitter=TimeSplitter(),
        input_messages_key="input",
        history_messages_key="history",
    )
