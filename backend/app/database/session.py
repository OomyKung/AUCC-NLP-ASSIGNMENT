"""Database engine, session factory and the FastAPI session dependency."""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.database.base import Base

_is_sqlite = settings.database_url.startswith("sqlite")

engine: Engine = create_engine(
    settings.database_url,
    echo=settings.sql_echo,
    future=True,
    # SQLite alone needs this: FastAPI may touch a session from another thread.
    connect_args={"check_same_thread": False} if _is_sqlite else {},
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


if _is_sqlite:

    @event.listens_for(engine, "connect")
    def _configure_sqlite(dbapi_connection, _connection_record) -> None:
        """Enable foreign keys and WAL, which SQLite leaves off by default."""
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()


def init_db() -> None:
    """Create every table. Safe to call repeatedly."""
    # Importing the models registers them on Base.metadata.
    from app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a session that is always closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
