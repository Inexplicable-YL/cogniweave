from __future__ import annotations

from typing import TYPE_CHECKING, Any
from typing_extensions import override

from sqlalchemy import JSON, Column, DateTime, ForeignKey, Index, Integer, String
from sqlalchemy.orm import relationship

from .base import Base

if TYPE_CHECKING:
    from datetime import datetime


class User(Base):
    """Represents a chat user."""

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
    """Represents a contiguous block of chat messages for a user."""

    __tablename__ = "chat_blocks"

    id: Column[int] = Column(Integer, primary_key=True)
    context_id: Column[str] = Column(String, nullable=False, unique=True, index=True)
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
        return (
            f"<ChatBlock(id={self.id}, context_id={self.context_id!r}, "
            f"user_id={self.user_id}, start_time={self.start_time})>"
        )


class ChatMessage(Base):
    """Represents a single chat message within a ChatBlock."""

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
    """Auxiliary data for a ChatBlock, such as short-term memory or embedding."""

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
