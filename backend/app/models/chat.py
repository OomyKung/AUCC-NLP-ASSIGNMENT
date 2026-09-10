"""YouTube live-chat models.

``ChatStream`` is one YouTube video/broadcast that chat was collected from.
``ChatMessage`` is a single raw chat message, kept individually so per-message
sentiment and the message-level drill-down stay available after messages have
been grouped into analysable windows.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, UtcDateTime, utcnow

if TYPE_CHECKING:
    from app.models.news import NewsArticle


class ChatStream(Base):
    """A YouTube video whose live chat has been collected."""

    __tablename__ = "chat_streams"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # YouTube's 11-character video id; the natural key for deduplication.
    video_id: Mapped[str] = mapped_column(
        String(32), unique=True, nullable=False, index=True
    )
    url: Mapped[str] = mapped_column(String(500), nullable=False)
    title: Mapped[str | None] = mapped_column(String(500))
    channel: Mapped[str | None] = mapped_column(String(200), index=True)

    # Which collector produced this: ytdlp | youtube_api | file
    collector: Mapped[str] = mapped_column(String(20), default="ytdlp")
    is_live: Mapped[bool] = mapped_column(Boolean, default=False)

    message_count: Mapped[int] = mapped_column(Integer, default=0)
    window_count: Mapped[int] = mapped_column(Integer, default=0)
    first_message_at: Mapped[datetime | None] = mapped_column(UtcDateTime)
    last_message_at: Mapped[datetime | None] = mapped_column(UtcDateTime)

    collected_at: Mapped[datetime] = mapped_column(
        UtcDateTime, default=utcnow, index=True
    )

    messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="stream", cascade="all, delete-orphan", lazy="select"
    )
    windows: Mapped[list[NewsArticle]] = relationship(
        back_populates="stream", lazy="select"
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<ChatStream {self.video_id} messages={self.message_count}>"


class ChatMessage(Base):
    """One raw live-chat message."""

    __tablename__ = "chat_messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    stream_id: Mapped[int] = mapped_column(
        ForeignKey("chat_streams.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # YouTube's own message id, used to make re-imports idempotent.
    message_id: Mapped[str | None] = mapped_column(String(120))
    author: Mapped[str | None] = mapped_column(String(200))
    text: Mapped[str] = mapped_column(Text, nullable=False)

    published_at: Mapped[datetime] = mapped_column(
        UtcDateTime, default=utcnow, index=True
    )
    # Milliseconds from the start of the stream (present on replays).
    offset_ms: Mapped[int | None] = mapped_column(Integer)

    # --------------------------------------------- per-message NLP results
    sentiment: Mapped[str | None] = mapped_column(String(20), index=True)
    sentiment_confidence: Mapped[float | None] = mapped_column(Float)

    # Messages that are pure emote/repetition spam are excluded from analysis.
    is_spam: Mapped[bool] = mapped_column(Boolean, default=False, index=True)

    # Which window this message was grouped into.
    window_index: Mapped[int | None] = mapped_column(Integer, index=True)
    news_id: Mapped[int | None] = mapped_column(
        ForeignKey("news.id", ondelete="SET NULL"), index=True
    )

    stream: Mapped[ChatStream] = relationship(back_populates="messages")
    article: Mapped[NewsArticle | None] = relationship(back_populates="messages")

    __table_args__ = (
        # A YouTube message id is unique within its stream.
        UniqueConstraint("stream_id", "message_id", name="uq_chat_stream_message"),
        Index("ix_chat_stream_published", "stream_id", "published_at"),
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<ChatMessage {self.author}: {self.text[:30]!r}>"
