from __future__ import annotations

from datetime import datetime  # noqa: TC003
from typing import Any
from typing_extensions import override

from sqlalchemy import JSON, DateTime, ForeignKey, Index, String, Integer
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    id: Mapped[int] = mapped_column(primary_key=True)


class User(Base):
    """Represents a chat user."""

    __tablename__ = "users"

    session_id: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    name: Mapped[str | None] = mapped_column(String, nullable=True)

    chat_blocks: Mapped[list[ChatBlock]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        order_by="ChatBlock.timestamp",
        lazy="selectin",
    )
    messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        order_by="ChatMessage.timestamp",
        lazy="selectin",
    )
    attributes: Mapped[list[UserAttribute]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (
        Index("idx_user_session", "session_id"),
        Index("idx_user_name", "name"),
    )

    @override
    def __repr__(self) -> str:
        return (
            f"<User(id={self.id}, session_id={self.session_id!r}, "
            f"name={self.name!r})>"
        )


class UserAttribute(Base):
    """Auxiliary data for a User."""

    __tablename__ = "user_attributes"

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    value: Mapped[Any] = mapped_column(JSON, nullable=False)

    user: Mapped[User] = relationship("User", back_populates="attributes", lazy="joined")

    __table_args__ = (Index("idx_user_attributes_user_type", "user_id", "type"),)

    @override
    def __repr__(self) -> str:
        return f"<UserAttribute(id={self.id}, user_id={self.user_id}, type={self.type!r})>"


class ChatBlock(Base):
    """Represents a contiguous block of chat messages for a user."""

    __tablename__ = "chat_blocks"

    context_id: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)

    user: Mapped[User] = relationship("User", back_populates="chat_blocks", lazy="joined")
    messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="block",
        cascade="all, delete-orphan",
        order_by="ChatMessage.timestamp",
        lazy="selectin",
    )
    attributes: Mapped[list[ChatBlockAttribute]] = relationship(
        back_populates="block",
        cascade="all, delete-orphan",
        lazy="selectin",
    )

    __table_args__ = (Index("idx_chat_blocks_session_start", "session_id", "timestamp"),)

    @override
    def __repr__(self) -> str:
        return (
            f"<ChatBlock(id={self.id}, context_id={self.context_id!r}, "
            f"session_id={self.session_id}, timestamp={self.timestamp})>"
        )


class ChatMessage(Base):
    """Represents a single chat message within a ChatBlock."""

    __tablename__ = "chat_messages"

    block_id: Mapped[int] = mapped_column(
        ForeignKey("chat_blocks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    session_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    uid: Mapped[int] = mapped_column(Integer, autoincrement=True, index=True, nullable=False)
    timestamp: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    content: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)

    block: Mapped[ChatBlock] = relationship(
        "ChatBlock", back_populates="messages", lazy="joined"
    )
    user: Mapped[User] = relationship(
        "User", back_populates="messages", lazy="joined"
    )

    __table_args__ = (
        Index("idx_messages_block_timestamp", "block_id", "timestamp"),
        Index("idx_messages_timestamp", "timestamp"),
        Index("idx_messages_session_timestamp", "session_id", "timestamp"),
    )

    @override
    def __repr__(self) -> str:
        return (
            f"<ChatMessage(id={self.id}, uid={self.uid}, block_id={self.block_id}, "
            f"timestamp={self.timestamp}, content={self.content})>"
        )


class ChatBlockAttribute(Base):
    """Auxiliary data for a ChatBlock, such as short-term memory or embedding."""

    __tablename__ = "chat_block_attributes"

    id: Mapped[int] = mapped_column(primary_key=True)
    block_id: Mapped[int] = mapped_column(
        ForeignKey("chat_blocks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    value: Mapped[Any] = mapped_column(JSON, nullable=False)

    block: Mapped[ChatBlock] = relationship("ChatBlock", back_populates="attributes", lazy="joined")

    __table_args__ = (Index("idx_block_attributes_block_type", "block_id", "type"),)

    @override
    def __repr__(self) -> str:
        return f"<ChatBlockAttribute(id={self.id}, block_id={self.block_id}, type={self.type!r})>"
