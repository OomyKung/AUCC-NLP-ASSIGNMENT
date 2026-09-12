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
    """Create every table, and add columns a previous version did not have.

    Safe to call repeatedly.
    """
    # Importing the models registers them on Base.metadata.
    from app import models  # noqa: F401

    Base.metadata.create_all(bind=engine)
    _add_missing_columns()


def _add_missing_columns() -> None:
    """Add columns and indexes that the models have and the database does not.

    ``create_all`` creates missing *tables* and silently ignores missing
    *columns*, so a database built by an earlier version keeps starting and then
    fails on the first query with "no such column". Rebuilding is cheap here
    (``python seed.py`` takes 80 seconds), but a fresh clone is not the case that
    matters -- the person who already has data is, and they should not have to
    discover this from a stack trace.

    Deliberately not Alembic: one project, one database file, and additive
    columns only. Anything that needs a real migration -- a rename, a type
    change, a backfill -- needs a real migration tool, and this will not pretend
    to be one. It only adds columns with a default, and only on SQLite.
    """
    if not _is_sqlite:
        return
    from sqlalchemy import text

    with engine.begin() as connection:
        for table in Base.metadata.sorted_tables:
            existing = {
                row[1]
                for row in connection.execute(
                    text(f"PRAGMA table_info('{table.name}')")
                )
            }
            if not existing:
                continue  # the table itself is new; create_all just made it
            for column in table.columns:
                if column.name in existing or column.primary_key:
                    continue
                default = column.default.arg if column.default is not None else None
                if default is None or callable(default):
                    # No safe value to backfill with; leave it to a real
                    # migration rather than guessing.
                    continue
                literal = repr(default) if isinstance(default, str) else str(default)
                connection.execute(
                    text(
                        f"ALTER TABLE {table.name} "
                        f"ADD COLUMN {column.name} {column.type.compile(engine.dialect)} "
                        f"DEFAULT {literal}"
                    )
                )
            # Indexes are the same story as columns: create_all adds them only
            # for tables it creates, so a new index never reaches an existing
            # database and the query it was added for stays slow.
            for index in table.indexes:
                index.create(bind=connection, checkfirst=True)


def get_db() -> Generator[Session, None, None]:
    """FastAPI dependency yielding a session that is always closed."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
