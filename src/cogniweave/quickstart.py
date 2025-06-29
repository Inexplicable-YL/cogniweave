from __future__ import annotations

from pathlib import Path

from langchain_core.messages import MessagesPlaceholder

from cogniweave.core.end_detector import EndDetector
from cogniweave.core.history_stores import BaseHistoryStore as HistoryStore
from cogniweave.core.time_splitter import TimeSplitter
from cogniweave.core.vector_stores import TagsVectorStore
from cogniweave.llms import OpenAIEmbeddings, StringSingleTurnChat
from cogniweave.prompts import MessageSegmentsPlaceholder, RichSystemMessagePromptTemplate
from cogniweave.runnables.end_detector import RunnableWithEndDetector
from cogniweave.runnables.history_store import RunnableWithHistoryStore
from cogniweave.runnables.memory_maker import RunnableWithMemoryMaker

DEF_DB = Path("./.cache/history_cache/demo.sqlite")
DEF_MODEL_CACHE = Path("./.cache/model_cache")


def create_embeddings() -> OpenAIEmbeddings:
    """Create default embeddings instance."""
    return OpenAIEmbeddings()


def create_history_store(db_path: str | Path = DEF_DB) -> HistoryStore:
    """Create a history store backed by a SQLite database."""
    return HistoryStore(db_url=f"sqlite:///{Path(db_path)}")


def create_vector_store(
    session_id: str,
    embeddings: OpenAIEmbeddings,
    cache_path: str | Path = DEF_MODEL_CACHE,
) -> TagsVectorStore:
    """Create a vector store for long term memory."""
    return TagsVectorStore(
        folder_path=str(cache_path),
        index_name=session_id,
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
        auto_save=True,
    )


def create_agent() -> StringSingleTurnChat:
    """Create the base chat agent."""
    return StringSingleTurnChat(
        lang="zh",
        provider="deepseek",
        model="deepseek-chat",
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
    session_id: str = "demo",
    db_path: str | Path = DEF_DB,
    model_cache: str | Path = DEF_MODEL_CACHE,
) -> RunnableWithHistoryStore:
    """Assemble the runnable pipeline used in the demos."""
    embeddings = create_embeddings()
    history_store = create_history_store(db_path)
    vector_store = create_vector_store(session_id, embeddings, model_cache)
    agent = create_agent()

    pipeline: RunnableWithHistoryStore = RunnableWithMemoryMaker(
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
    pipeline = RunnableWithHistoryStore(
        pipeline,
        history_store=history_store,
        time_splitter=TimeSplitter(),
        input_messages_key="input",
        history_messages_key="history",
    )
    return pipeline
