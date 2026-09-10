"""News document endpoints: listing, filtering, detail and creation."""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.chat import ChatMessage
from app.models.news import NewsArticle
from app.schemas.news import (
    AnalyzeRequest,
    ChatMessageOut,
    NewsDetail,
    NewsListItem,
    NewsPage,
)
from app.services.ingest import analyse_text
from app.taxonomy import SENTIMENT_SLUGS, TOPIC_SLUGS

router = APIRouter(prefix="/news", tags=["news"])

MAX_PAGE_SIZE = 100


def _apply_filters(
    query,
    *,
    search: str | None,
    topic: str | None,
    sentiment: str | None,
    source_type: str | None,
    date_from: datetime | None,
    date_to: datetime | None,
    min_confidence: float | None,
):
    """Apply the Explorer's filters to a select statement."""
    if search:
        # `%` and `_` are LIKE wildcards. Without escaping, searching for a
        # literal "%" matches every row, and "_" matches any single character --
        # quietly wrong results rather than an error.
        escaped = (
            search.strip()
            .replace("\\", "\\\\")
            .replace("%", "\\%")
            .replace("_", "\\_")
        )
        needle = f"%{escaped}%"
        # Search headline, body and summary so a keyword in the text is findable.
        query = query.where(
            or_(
                NewsArticle.title.like(needle, escape="\\"),
                NewsArticle.content.like(needle, escape="\\"),
                NewsArticle.summary.like(needle, escape="\\"),
            )
        )
    if topic:
        query = query.where(NewsArticle.topic == topic)
    if sentiment:
        query = query.where(NewsArticle.sentiment == sentiment)
    if source_type:
        query = query.where(NewsArticle.source_type == source_type)
    if date_from:
        query = query.where(NewsArticle.published_at >= date_from)
    if date_to:
        query = query.where(NewsArticle.published_at <= date_to)
    if min_confidence is not None:
        query = query.where(NewsArticle.sentiment_confidence >= min_confidence)
    return query


_SORTS = {
    "newest": NewsArticle.published_at.desc(),
    "oldest": NewsArticle.published_at.asc(),
    "confidence": NewsArticle.sentiment_confidence.desc(),
    "messages": NewsArticle.message_count.desc().nulls_last(),
}


@router.get("", response_model=NewsPage, summary="List and filter documents")
def list_news(
    db: Session = Depends(get_db),
    search: str | None = Query(None, max_length=200, description="Free-text search"),
    topic: str | None = Query(None, description="Filter by topic slug"),
    sentiment: str | None = Query(None, description="Filter by sentiment slug"),
    source_type: str | None = Query(None, description="article | chat_window | dataset"),
    date_from: datetime | None = Query(None),
    date_to: datetime | None = Query(None),
    min_confidence: float | None = Query(None, ge=0.0, le=1.0),
    sort: str = Query("newest", description="newest | oldest | confidence | messages"),
    page: int = Query(1, ge=1),
    page_size: int = Query(12, ge=1, le=MAX_PAGE_SIZE),
) -> NewsPage:
    """Return one page of documents matching the filters."""
    if topic and topic not in TOPIC_SLUGS:
        raise HTTPException(422, f"Unknown topic {topic!r}")
    if sentiment and sentiment not in SENTIMENT_SLUGS:
        raise HTTPException(422, f"Unknown sentiment {sentiment!r}")
    if sort not in _SORTS:
        raise HTTPException(422, f"Unknown sort {sort!r}")

    filters = {
        "search": search,
        "topic": topic,
        "sentiment": sentiment,
        "source_type": source_type,
        "date_from": date_from,
        "date_to": date_to,
        "min_confidence": min_confidence,
    }

    total = (
        db.scalar(
            _apply_filters(select(func.count()).select_from(NewsArticle), **filters)
        )
        or 0
    )

    rows = db.scalars(
        _apply_filters(select(NewsArticle), **filters)
        .order_by(_SORTS[sort])
        .offset((page - 1) * page_size)
        .limit(page_size)
    ).all()

    return NewsPage(
        items=[NewsListItem.model_validate(row) for row in rows],
        total=total,
        page=page,
        page_size=page_size,
        pages=max(1, (total + page_size - 1) // page_size),
    )


@router.get("/{news_id}", response_model=NewsDetail, summary="One document in full")
def get_news(news_id: int, db: Session = Depends(get_db)) -> NewsDetail:
    """Return a document with its full text and NLP analysis."""
    article = db.get(NewsArticle, news_id)
    if article is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"News {news_id} not found")
    return NewsDetail.model_validate(article)


@router.get(
    "/{news_id}/messages",
    response_model=list[ChatMessageOut],
    summary="The chat messages behind a window",
)
def get_news_messages(
    news_id: int,
    db: Session = Depends(get_db),
    limit: int = Query(200, ge=1, le=500),
) -> list[ChatMessageOut]:
    """Return the individual messages that make up a chat window."""
    if db.get(NewsArticle, news_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"News {news_id} not found")

    rows = db.scalars(
        select(ChatMessage)
        .where(ChatMessage.news_id == news_id)
        .order_by(ChatMessage.published_at)
        .limit(limit)
    ).all()
    return [ChatMessageOut.model_validate(row) for row in rows]


@router.post(
    "",
    response_model=NewsDetail,
    status_code=status.HTTP_201_CREATED,
    summary="Analyse and store a document",
)
def create_news(payload: AnalyzeRequest, db: Session = Depends(get_db)) -> NewsDetail:
    """Run the pipeline over submitted text and store the result."""
    article = analyse_text(
        db,
        title=payload.title,
        content=payload.content,
        source=payload.source,
        url=payload.url,
        published_at=payload.published_at,
    )
    return NewsDetail.model_validate(article)


@router.delete(
    "/{news_id}", status_code=status.HTTP_204_NO_CONTENT, summary="Delete a document"
)
def delete_news(news_id: int, db: Session = Depends(get_db)) -> None:
    """Delete a document, detaching any chat messages that referenced it."""
    article = db.get(NewsArticle, news_id)
    if article is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"News {news_id} not found")

    for message in db.scalars(
        select(ChatMessage).where(ChatMessage.news_id == news_id)
    ):
        message.news_id = None
        message.window_index = None

    db.delete(article)
    db.commit()
