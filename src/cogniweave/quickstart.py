from __future__ import annotations

from pathlib import Path

from langchain_core.prompts import MessagesPlaceholder

from cogniweave.core.end_detector import EndDetector
from cogniweave.core.history_stores import BaseHistoryStore as HistoryStore
from cogniweave.core.time_splitter import TimeSplitter
from cogniweave.core.vector_stores import TagsVectorStore
from cogniweave.llms import OpenAIEmbeddings, StringSingleTurnChat
from cogniweave.prompts import MessageSegmentsPlaceholder, RichSystemMessagePromptTemplate
from cogniweave.runnables.end_detector import RunnableWithEndDetector
from cogniweave.runnables.history_store import RunnableWithHistoryStore
from cogniweave.runnables.memory_maker import RunnableWithMemoryMaker
from cogniweave.utils import get_model_from_env, get_provider_from_env

DEF_FOLDER_PATH = Path("./.cache/")


def create_embeddings(
    provider: str | None = None,
    model: str | None = None,
) -> OpenAIEmbeddings:
    """Create default embeddings instance."""
    return OpenAIEmbeddings(
        provider=get_provider_from_env("EMBEDDINGS_MODEL", default=provider or "openai")(),
        model=get_model_from_env("EMBEDDINGS_MODEL", default=model or "text-embedding-ada-002")(),
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


def create_agent(
    provider: str | None = None,
    model: str | None = None,
) -> StringSingleTurnChat:
    """Create the base chat agent."""
    return StringSingleTurnChat(
        lang="zh",
        provider=get_provider_from_env("AGENT_MODEL", default=provider or "openai")(),
        model=get_model_from_env("AGENT_MODEL", default=model or "gpt-4.1-mini")(),
        contexts=[
            RichSystemMessagePromptTemplate.from_template(
                [
                    "你是一个AI助手，你叫CogniWeave，你的任务是回答用户的问题。\n",
                    MessageSegmentsPlaceholder(variable_name="long_memory"),
                ]
            ),
            MessagesPlaceholder(variable_name="history", optional=True),
        ],
    )


def build_pipeline(
    index_name: str = "demo",
    folder_path: str | Path = DEF_FOLDER_PATH,
) -> RunnableWithHistoryStore:
    """Assemble the runnable pipeline used in the demos."""
    embeddings = create_embeddings()
    history_store = create_history_store(index_name=index_name, folder_path=folder_path)
    vector_store = create_vector_store(embeddings, index_name=index_name, folder_path=folder_path)
    agent = create_agent()

    pipeline = RunnableWithMemoryMaker(
        agent,
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
