"""Spoken-content models: what was said in the video, and the stories in it.

These sit alongside the chat models rather than replacing them. A news video now
has two independent streams of evidence:

* **chat** -- what the audience said, already modelled by ``ChatStream`` /
  ``ChatMessage`` and grouped into windows
* **transcript** -- what the *newsreader* said, modelled here and segmented into
  individual stories

Both hang off the same ``ChatStream`` row, because the video is the thing they
have in common.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import (
    JSON,
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
    from app.models.chat import ChatStream


class VideoTranscript(Base):
    """The timed transcript of one video's audio."""

    __tablename__ = "video_transcripts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    stream_id: Mapped[int] = mapped_column(
        ForeignKey("chat_streams.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )

    # Which provider produced it, e.g. "youtube-asr:th-orig" or "whisper:small".
    # Stored rather than assumed, so a result stays traceable after a backend
    # swap -- the same reason NLPAnalysis records its model versions.
    source: Mapped[str] = mapped_column(String(60), nullable=False)
    language: Mapped[str] = mapped_column(String(12), default="th")

    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    cue_count: Mapped[int] = mapped_column(Integer, default=0)
    character_count: Mapped[int] = mapped_column(Integer, default=0)

    # Cues as compact JSON. A four-hour programme is ~5,600 cues; a separate
    # table would mean 5,600 rows per video for data that is always read whole
    # and never queried by itself.
    cues: Mapped[list] = mapped_column(JSON, default=list)

    fetched_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)

    stream: Mapped[ChatStream] = relationship(back_populates="transcript")
    segments: Mapped[list[NewsSegment]] = relationship(
        back_populates="transcript",
        cascade="all, delete-orphan",
        lazy="select",
        order_by="NewsSegment.start_ms",
    )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<VideoTranscript stream={self.stream_id} cues={self.cue_count}>"


class NewsSegment(Base):
    """One story inside a programme: a time range with its own analysis."""

    __tablename__ = "news_segments"
    __table_args__ = (
        # One segment per (transcript, start): re-analysing a programme must
        # update rows rather than accumulate duplicates.
        UniqueConstraint("transcript_id", "start_ms", name="uq_segment_start"),
        Index("ix_segment_topic_start", "topic", "start_ms"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    transcript_id: Mapped[int] = mapped_column(
        ForeignKey("video_transcripts.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    stream_id: Mapped[int] = mapped_column(
        ForeignKey("chat_streams.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Position in the programme, so the timeline can be ordered without sorting
    # on time when the caller already has the rows.
    position: Mapped[int] = mapped_column(Integer, default=0)

    start_ms: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    end_ms: Mapped[int] = mapped_column(Integer, nullable=False)

    headline: Mapped[str] = mapped_column(String(300), default="")
    summary: Mapped[str] = mapped_column(Text, default="")
    transcript_text: Mapped[str] = mapped_column(Text, default="")

    topic: Mapped[str] = mapped_column(String(40), default="other", index=True)
    topic_confidence: Mapped[float] = mapped_column(Float, default=0.0)
    sentiment: Mapped[str] = mapped_column(String(20), default="neutral", index=True)
    sentiment_confidence: Mapped[float] = mapped_column(Float, default=0.0)

    keywords: Mapped[list] = mapped_column(JSON, default=list)
    entities: Mapped[list] = mapped_column(JSON, default=list)

    # Why the segmenter put a boundary here. Kept so the UI can explain a split
    # instead of presenting it as an oracle: "lexical-cohesion+topic-change".
    boundary_reasons: Mapped[list] = mapped_column(JSON, default=list)
    boundary_confidence: Mapped[float] = mapped_column(Float, default=0.0)

    # Name spellings the LLM corrected, as [[asr_form, corrected], ...].
    # Empty unless LLM_CORRECT_NAMES is on, which it is not by default: a small
    # local model invents Thai names rather than admitting it cannot tell.
    # Stored rather than applied silently -- a correction is a claim about what
    # was said, and a reader should be able to see the ASR form it replaced.
    name_corrections: Mapped[list] = mapped_column(JSON, default=list)
    # Which component produced the headline and entities: "" for the
    # extractive default, "llm" when the model was used.
    enriched_by: Mapped[str] = mapped_column(String(20), default="")

    # Deep link to the exact moment, and the cached frame captured there.
    youtube_url: Mapped[str] = mapped_column(String(200), default="")
    frame_path: Mapped[str | None] = mapped_column(String(300))

    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)

    transcript: Mapped[VideoTranscript] = relationship(back_populates="segments")

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    @property
    def timecode(self) -> str:
        """``H:MM:SS`` label, as a viewer reads it."""
        total = self.start_ms // 1000
        hours, remainder = divmod(total, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<NewsSegment {self.timecode} {self.topic}>"
