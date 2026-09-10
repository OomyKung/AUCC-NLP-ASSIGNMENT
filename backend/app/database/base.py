"""Declarative base and shared column types."""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, TypeDecorator
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Base class for every ORM model."""


def utcnow() -> datetime:
    """Timezone-aware UTC now (``datetime.utcnow`` is deprecated)."""
    return datetime.now(UTC)


class UtcDateTime(TypeDecorator):
    """A datetime column that is always timezone-aware UTC in Python.

    SQLite has no timezone type: it stores whatever wall-clock value it is
    given and hands back a *naive* datetime. Mixing those with the aware
    datetimes the collectors produce raises ``TypeError`` on subtraction, which
    would break trend aggregation and window arithmetic in subtle ways.

    This decorator normalises both directions -- values are converted to UTC on
    write and re-tagged as UTC on read -- so the rest of the codebase can treat
    every datetime as aware without caring which database is behind it.
    """

    impl = DateTime(timezone=True)
    cache_ok = True

    def process_bind_param(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            # Naive input is assumed to already be UTC.
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)

    def process_result_value(self, value: datetime | None, dialect) -> datetime | None:
        if value is None:
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
