"""Shared pytest fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.base import Base
from app.main import create_app
from app.services.collectors.base import CollectResult, RawChatMessage, StreamInfo


@pytest.fixture(scope="session")
def client() -> TestClient:
    """A TestClient bound to a freshly built app instance."""
    return TestClient(create_app())


@pytest.fixture
def db_session() -> Iterator[Session]:
    """An isolated in-memory database, torn down after each test.

    StaticPool keeps every connection pointed at the same in-memory database,
    which SQLite otherwise scopes per connection.
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    session = factory()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture
def make_messages():
    """Build a list of synthetic chat messages at a fixed cadence."""

    def _make(
        count: int,
        *,
        seconds_apart: int = 10,
        start: datetime | None = None,
        text: str = "ข้อความทดสอบ",
    ) -> list[RawChatMessage]:
        origin = start or datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
        return [
            RawChatMessage(
                text=f"{text} {index}",
                published_at=origin + timedelta(seconds=index * seconds_apart),
                message_id=f"msg-{index}",
                author=f"user{index % 7}",
                offset_ms=index * seconds_apart * 1000,
            )
            for index in range(count)
        ]

    return _make


@pytest.fixture
def make_result(make_messages):
    """Build a CollectResult around synthetic messages."""

    def _make(count: int = 10, video_id: str = "testVideo01", **kwargs) -> CollectResult:
        return CollectResult(
            stream=StreamInfo(
                video_id=video_id,
                url=f"https://www.youtube.com/watch?v={video_id}",
                title="ทดสอบ ถ่ายทอดสด",
                channel="ช่องทดสอบ",
                collector="file",
            ),
            messages=make_messages(count, **kwargs),
        )

    return _make
