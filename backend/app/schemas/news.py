"""Request and response models for news documents."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.taxonomy import SENTIMENT_SLUGS, TOPIC_SLUGS


class KeywordOut(BaseModel):
    """One extracted keyword."""

    word: str
    score: float


class EntityOut(BaseModel):
    """One named entity."""

    text: str
    label: str


class NewsListItem(BaseModel):
    """A document as shown in list views and cards.

    Deliberately excludes the full body and token lists so list endpoints stay
    small; the detail endpoint carries those.
    """

    model_config = ConfigDict(from_attributes=True)

    id: int
    title: str
    summary: str | None = None
    topic: str
    topic_confidence: float
    sentiment: str
    sentiment_confidence: float
    keywords: list[KeywordOut] = Field(default_factory=list)
    source: str
    url: str | None = None
    published_at: datetime
    source_type: str
    message_count: int | None = None

    @field_validator("keywords", mode="before")
    @classmethod
    def _coerce_keywords(cls, value):
        """Accept either the stored dict shape or a bare list of strings."""
        if not value:
            return []
        coerced = []
        for item in value:
            if isinstance(item, dict):
                coerced.append(
                    {"word": item.get("word", ""), "score": item.get("score", 0.0)}
                )
            else:
                coerced.append({"word": str(item), "score": 0.0})
        return coerced


class AnalysisOut(BaseModel):
    """The explainable NLP detail for one document."""

    model_config = ConfigDict(from_attributes=True)

    cleaned_text: str | None = None
    tokens: list[str] = Field(default_factory=list)
    filtered_tokens: list[str] = Field(default_factory=list)
    token_count: int = 0
    unique_token_count: int = 0
    stopword_removed_count: int = 0
    topic_probabilities: dict[str, float] = Field(default_factory=dict)
    sentiment_probabilities: dict[str, float] = Field(default_factory=dict)
    entities: list[EntityOut] = Field(default_factory=list)
    keyword_scores: list[KeywordOut] = Field(default_factory=list)
    sentences: list[str] = Field(default_factory=list)
    model_versions: dict[str, str] = Field(default_factory=dict)
    pipeline_version: str = "1.0.0"
    processing_ms: float = 0.0


class ChatMessageOut(BaseModel):
    """One raw chat message, for the window drill-down."""

    model_config = ConfigDict(from_attributes=True)

    id: int
    author: str | None = None
    text: str
    published_at: datetime
    sentiment: str | None = None
    sentiment_confidence: float | None = None


class NewsDetail(NewsListItem):
    """A document with its full body and NLP analysis."""

    content: str
    created_at: datetime
    window_start: datetime | None = None
    window_end: datetime | None = None
    analysis: AnalysisOut | None = None


class NewsPage(BaseModel):
    """One page of documents."""

    items: list[NewsListItem]
    total: int
    page: int
    page_size: int
    pages: int


class AnalyzeRequest(BaseModel):
    """Body for POST /api/analyze and POST /api/news."""

    title: str = Field(min_length=1, max_length=500)
    content: str = Field(min_length=1)
    source: str | None = Field(default=None, max_length=200)
    url: str | None = Field(default=None, max_length=1000)
    published_at: datetime | None = None

    @field_validator("title", "content")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("must not be blank")
        return cleaned


class AnalyzeResponse(BaseModel):
    """Result of analysing text, with the stored document when persisted."""

    topic: str
    topic_confidence: float
    sentiment: str
    sentiment_confidence: float
    summary: str
    keywords: list[KeywordOut]
    entities: list[EntityOut]
    topic_probabilities: dict[str, float]
    sentiment_probabilities: dict[str, float]
    cleaned_text: str
    tokens: list[str]
    filtered_tokens: list[str]
    token_count: int
    unique_token_count: int
    stopword_removed_count: int
    sentences: list[str]
    processing_ms: float
    model_versions: dict[str, str]
    news_id: int | None = None


SortOption = Literal["newest", "oldest", "confidence", "messages"]

# Exported for the API layer's query validation.
TOPIC_VALUES = set(TOPIC_SLUGS)
SENTIMENT_VALUES = set(SENTIMENT_SLUGS)
