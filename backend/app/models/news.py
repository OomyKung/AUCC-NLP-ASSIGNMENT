"""The analysed-document model.

A row here is one thing the NLP pipeline has analysed. It may originate from a
pasted article, a seeded dataset row, or a window of YouTube live-chat messages
(see ``app/services/windowing.py``) -- ``source_type`` records which.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import JSON, Float, ForeignKey, Index, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, UtcDateTime, utcnow

if TYPE_CHECKING:
    from app.models.analysis import NLPAnalysis
    from app.models.chat import ChatMessage, ChatStream


class SourceType(str, enum.Enum):
    """Where a document came from."""

    ARTICLE = "article"  # pasted or seeded news article
    CHAT_WINDOW = "chat_window"  # a window of YouTube live-chat messages
    DATASET = "dataset"  # a labelled training-set row loaded for demo


class NewsArticle(Base):
    """One analysed news document."""

    __tablename__ = "news"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # ------------------------------------------------------------- content
    title: Mapped[str] = mapped_column(String(500), nullable=False, index=True)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str | None] = mapped_column(Text)

    # -------------------------------------------------------- NLP results
    topic: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    topic_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    sentiment: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    sentiment_confidence: Mapped[float] = mapped_column(Float, default=0.0)

    # ``[{"word": "รถยนต์", "score": 0.41}, ...]`` -- scores kept for the UI.
    keywords: Mapped[list] = mapped_column(JSON, default=list)

    # ------------------------------------------------------- provenance
    source: Mapped[str] = mapped_column(String(200), default="unknown", index=True)
    url: Mapped[str | None] = mapped_column(String(1000))
    published_at: Mapped[datetime] = mapped_column(
        UtcDateTime, default=utcnow, index=True
    )
    created_at: Mapped[datetime] = mapped_column(
        UtcDateTime, default=utcnow, index=True
    )

    source_type: Mapped[str] = mapped_column(
        String(20), default=SourceType.ARTICLE.value, index=True
    )
    # Deduplication: sha256 of the normalised content.
    content_hash: Mapped[str | None] = mapped_column(String(64), unique=True, index=True)

    # ------------------------------- chat-window provenance (nullable for articles)
    stream_id: Mapped[int | None] = mapped_column(
        ForeignKey("chat_streams.id", ondelete="SET NULL"), index=True
    )
    message_count: Mapped[int | None] = mapped_column(Integer)
    window_start: Mapped[datetime | None] = mapped_column(UtcDateTime)
    window_end: Mapped[datetime | None] = mapped_column(UtcDateTime)

    # ----------------------------------------------------------- relations
    analysis: Mapped[NLPAnalysis | None] = relationship(
        back_populates="article",
        cascade="all, delete-orphan",
        uselist=False,
        lazy="selectin",
    )
    stream: Mapped[ChatStream | None] = relationship(back_populates="windows")
    messages: Mapped[list[ChatMessage]] = relationship(
        back_populates="article", lazy="select"
    )

    # Composite indexes for the Explorer's common filter combinations.
    __table_args__ = (
        Index("ix_news_topic_sentiment", "topic", "sentiment"),
        Index("ix_news_published_topic", "published_at", "topic"),
    )

    @property
    def keyword_words(self) -> list[str]:
        """Keywords as plain strings, tolerating either stored shape."""
        result: list[str] = []
        for item in self.keywords or []:
            if isinstance(item, dict):
                word = item.get("word")
                if word:
                    result.append(str(word))
            elif item:
                result.append(str(item))
        return result

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<NewsArticle id={self.id} topic={self.topic} title={self.title[:30]!r}>"
