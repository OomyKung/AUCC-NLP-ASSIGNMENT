"""Response models for dashboard statistics and charts."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class LabelCount(BaseModel):
    """One bar in a categorical chart."""

    key: str
    label: str  # Thai display label
    label_en: str
    color: str
    count: int


class SentimentSlice(BaseModel):
    """One slice of the sentiment donut."""

    key: str
    label: str
    label_en: str
    color: str
    count: int
    percentage: float


class TrendPoint(BaseModel):
    """One point on the volume-over-time chart."""

    period: str  # ISO date or year-week / year-month bucket
    count: int
    positive: int = 0
    neutral: int = 0
    negative: int = 0


class TopicSentimentRow(BaseModel):
    """One stacked bar: a topic broken down by sentiment.

    This is the chart that demonstrates the relationship between the two
    classifiers, so it carries the totals needed to render labels too.
    """

    topic: str
    label: str
    label_en: str
    color: str
    positive: int = 0
    neutral: int = 0
    negative: int = 0
    total: int = 0


class KeywordCount(BaseModel):
    """One entry in the keyword cloud."""

    word: str
    count: int
    score: float


class StatisticsResponse(BaseModel):
    """Everything the dashboard's cards and charts need, in one call."""

    total_news: int
    positive: int
    neutral: int
    negative: int
    most_common_topic: str | None = None
    most_common_topic_label: str | None = None
    most_common_topic_count: int = 0

    # Corpus-level context, shown as secondary figures.
    total_messages: int = 0
    scored_messages: int = 0
    noise_messages: int = 0
    total_streams: int = 0
    chat_windows: int = 0
    articles: int = 0

    by_topic: list[LabelCount] = Field(default_factory=list)
    by_sentiment: list[SentimentSlice] = Field(default_factory=list)
    topic_sentiment: list[TopicSentimentRow] = Field(default_factory=list)
    top_keywords: list[KeywordCount] = Field(default_factory=list)
    avg_topic_confidence: float = 0.0
    avg_sentiment_confidence: float = 0.0


TrendGranularity = Literal["daily", "weekly", "monthly"]


class TrendResponse(BaseModel):
    """Volume over time at a chosen granularity."""

    granularity: TrendGranularity
    points: list[TrendPoint]


class StreamOut(BaseModel):
    """A collected YouTube stream."""

    id: int
    video_id: str
    url: str
    title: str | None = None
    channel: str | None = None
    collector: str
    is_live: bool
    message_count: int
    window_count: int
    first_message_at: str | None = None
    last_message_at: str | None = None


class BackendStatusOut(BaseModel):
    """Which backend is serving one pipeline stage."""

    stage: str
    requested: str
    active: str
    trained: bool
    note: str = ""
