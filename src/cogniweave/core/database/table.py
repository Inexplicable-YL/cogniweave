from __future__ import annotations

from datetime import datetime
from typing import Any
from typing_extensions import override

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker

# ------------------------------------------------------------------------------
# Base and Engine Configuration
# ------------------------------------------------------------------------------

# Base class for declarative models
Base = declarative_base()

# Create engine (SQLite for example; swap with PostgreSQL/MySQL URI in production)
ENGINE = create_engine(
    "sqlite:///optimized_chat_db.sqlite",
    echo=False,
    future=True,
)

# Session factory for creating new Session objects
SessionLocal = sessionmaker(bind=ENGINE, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    """Initialize all tables in the database.

    Must be called once at application startup.
    """
    Base.metadata.create_all(bind=ENGINE)


# ------------------------------------------------------------------------------
# ORM Models
# ------------------------------------------------------------------------------


class User(Base):
    """Represents a chat user.

    Attributes:
        id: Primary key.
        name: Display name of the user.
        chat_blocks: All chat blocks belonging to this user, ordered by start_time.
    """

    __tablename__ = "users"

    id: Column[int] = Column(Integer, primary_key=True)
    name: Column[str] = Column(String, nullable=False, unique=True)

    chat_blocks = relationship(
        "ChatBlock",
        back_populates="user",
        cascade="all, delete-orphan",
        order_by="ChatBlock.start_time",
        lazy="selectin",
    )

    @override
    def __repr__(self) -> str:
        return f"<User(id={self.id}, name={self.name!r})>"


class ChatBlock(Base):
    """Represents a contiguous block of chat messages for a user.

    Attributes:
        id: Primary key.
        user_id: Foreign key to users.id.
        start_time: Timestamp of the earliest message in this block.
        user: Back-reference to User.
        messages: All messages in this block, ordered by timestamp.
        attributes: Any auxiliary data (short-term memory, topics, embeddings).
    """

    __tablename__ = "chat_blocks"

    id: Column[int] = Column(Integer, primary_key=True)
    user_id: Column[int] = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    start_time: Column[datetime] = Column(DateTime, nullable=False, index=True)

    user = relationship("User", back_populates="chat_blocks", lazy="joined")
    messages = relationship(
        "ChatMessage",
        back_populates="block",
        cascade="all, delete-orphan",
        order_by="ChatMessage.timestamp",
        lazy="selectin",
    )
    attributes = relationship(
        "ChatBlockAttribute",
        back_populates="block",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (Index("idx_chat_blocks_user_start", "user_id", "start_time"),)

    @override
    def __repr__(self) -> str:
        return f"<ChatBlock(id={self.id}, user_id={self.user_id}, start_time={self.start_time})>"


class ChatMessage(Base):
    """Represents a single chat message within a ChatBlock.

    Attributes:
        id: Primary key.
        block_id: Foreign key to chat_blocks.id.
        timestamp: Timestamp of the message (used for ordering within block).
        content: JSON blob containing the message content.
        block: Back-reference to ChatBlock.
    """

    __tablename__ = "chat_messages"

    id: Column[int] = Column(Integer, primary_key=True)
    block_id: Column[int] = Column(
        Integer,
        ForeignKey("chat_blocks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    timestamp: Column[datetime] = Column(DateTime, nullable=False, index=True)
    content: Column[dict[str, Any]] = Column(JSON, nullable=False)

    block = relationship("ChatBlock", back_populates="messages", lazy="joined")

    __table_args__ = (Index("idx_messages_block_timestamp", "block_id", "timestamp"),)

    @override
    def __repr__(self) -> str:
        return (
            f"<ChatMessage(id={self.id}, block_id={self.block_id}, "
            f"timestamp={self.timestamp}, content={self.content})>"
        )


class ChatBlockAttribute(Base):
    """Auxiliary data for a ChatBlock, such as short-term memory or embedding.

    Attributes:
        id: Primary key.
        block_id: Foreign key to chat_blocks.id.
        type: Type of attribute (e.g., 'short_term_memory', 'topic', 'embedding').
        value: Arbitrary JSON value storing the attribute data.
        block: Back-reference to ChatBlock.
    """

    __tablename__ = "chat_block_attributes"

    id: Column[int] = Column(Integer, primary_key=True)
    block_id: Column[int] = Column(
        Integer,
        ForeignKey("chat_blocks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    type: Column[str] = Column(String, nullable=False, index=True)
    value: Column[Any] = Column(JSON, nullable=False)

    block = relationship("ChatBlock", back_populates="attributes", lazy="joined")

    __table_args__ = (Index("idx_block_attributes_block_type", "block_id", "type"),)

    @override
    def __repr__(self) -> str:
        return f"<ChatBlockAttribute(id={self.id}, block_id={self.block_id}, type={self.type!r})>"
