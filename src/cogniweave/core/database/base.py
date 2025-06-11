from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()

DB_URL = os.getenv("CHAT_DB_URL", "sqlite:///optimized_chat_db.sqlite")
ENGINE = create_engine(DB_URL, echo=False, future=True)

SessionLocal = sessionmaker(bind=ENGINE, autoflush=False, autocommit=False, future=True)


def init_db() -> None:
    """Initialize all tables in the database."""
    Base.metadata.create_all(bind=ENGINE)
