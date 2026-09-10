"""Dashboard statistics, charts and pipeline-status endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.chat import ChatStream
from app.nlp.registry import get_components
from app.schemas.statistics import (
    BackendStatusOut,
    KeywordCount,
    StatisticsResponse,
    StreamOut,
    TrendResponse,
)
from app.services.statistics import build_statistics, build_trend, top_keywords
from app.taxonomy import TOPIC_SLUGS

router = APIRouter(tags=["statistics"])


@router.get(
    "/statistics", response_model=StatisticsResponse, summary="Dashboard figures"
)
def statistics(
    db: Session = Depends(get_db),
    keyword_limit: int = Query(40, ge=1, le=200),
) -> StatisticsResponse:
    """Every stat card and chart series in one call."""
    return build_statistics(db, keyword_limit=keyword_limit)


@router.get("/statistics/trend", response_model=TrendResponse, summary="Volume trend")
def trend(
    db: Session = Depends(get_db),
    granularity: str = Query("daily", description="daily | weekly | monthly"),
) -> TrendResponse:
    """Document volume over time, split by sentiment."""
    if granularity not in {"daily", "weekly", "monthly"}:
        raise HTTPException(422, f"Unknown granularity {granularity!r}")
    return build_trend(db, granularity)


@router.get(
    "/statistics/keywords",
    response_model=list[KeywordCount],
    summary="Keyword cloud data",
)
def keywords(
    db: Session = Depends(get_db),
    limit: int = Query(60, ge=1, le=300),
    topic: str | None = Query(None, description="Restrict to one topic"),
) -> list[KeywordCount]:
    """Aggregate keywords across documents."""
    if topic and topic not in TOPIC_SLUGS:
        raise HTTPException(422, f"Unknown topic {topic!r}")
    return top_keywords(db, limit=limit, topic=topic)


@router.get("/streams", response_model=list[StreamOut], summary="Collected streams")
def streams(db: Session = Depends(get_db)) -> list[StreamOut]:
    """The YouTube streams chat has been collected from."""
    rows = db.scalars(
        select(ChatStream).order_by(ChatStream.message_count.desc())
    ).all()
    return [
        StreamOut(
            id=row.id,
            video_id=row.video_id,
            url=row.url,
            title=row.title,
            channel=row.channel,
            collector=row.collector,
            is_live=row.is_live,
            message_count=row.message_count,
            window_count=row.window_count,
            first_message_at=(
                row.first_message_at.isoformat() if row.first_message_at else None
            ),
            last_message_at=(
                row.last_message_at.isoformat() if row.last_message_at else None
            ),
        )
        for row in rows
    ]


@router.get(
    "/pipeline",
    response_model=list[BackendStatusOut],
    summary="Which NLP backend serves each stage",
)
def pipeline_status() -> list[BackendStatusOut]:
    """Report the active backend per stage.

    Surfaces honestly when a configured backend fell back to a baseline, so the
    UI can label results as untrained rather than implying a fitted model.
    """
    components = get_components()
    return [
        BackendStatusOut(
            stage=stage,
            requested=status.requested,
            active=status.active,
            trained=status.trained,
            note=status.note,
        )
        for stage, status in components.status.items()
    ]
