"""Ad-hoc analysis and YouTube ingestion endpoints."""

from __future__ import annotations

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.chat import ChatStream
from app.schemas.news import AnalyzeRequest, AnalyzeResponse
from app.services.chat_store import store_messages
from app.services.collectors import AVAILABLE_COLLECTORS, CollectorError, get_collector
from app.services.ingest import analyse_all_streams, analyse_stream
from app.services.pipeline import get_pipeline
from app.services.snapshot import SnapshotWouldShrink, list_snapshots, write_snapshot

router = APIRouter(tags=["analysis"])


@router.post("/analyze", response_model=AnalyzeResponse, summary="Analyse text")
def analyze(
    payload: AnalyzeRequest,
    store: bool = Query(False, description="Also persist the result as a document"),
    db: Session = Depends(get_db),
) -> AnalyzeResponse:
    """Run the full NLP pipeline over submitted Thai text.

    Returns every intermediate value as well as the predictions, so the NLP
    Analysis page can show what the models actually consumed.
    """
    pipeline = get_pipeline()
    result = pipeline.analyse(payload.content, title=payload.title)
    body = result.as_dict()

    news_id = None
    if store:
        from app.services.ingest import analyse_text

        article = analyse_text(
            db,
            title=payload.title,
            content=payload.content,
            source=payload.source,
            url=payload.url,
            published_at=payload.published_at,
            pipeline=pipeline,
        )
        news_id = article.id

    return AnalyzeResponse(
        topic=body["topic"],
        topic_confidence=body["topic_confidence"],
        sentiment=body["sentiment"],
        sentiment_confidence=body["sentiment_confidence"],
        summary=body["summary"],
        keywords=body["keywords"],
        entities=body["entities"],
        topic_probabilities=body["topic_probabilities"],
        sentiment_probabilities=body["sentiment_probabilities"],
        cleaned_text=body["cleaned_text"],
        tokens=body["tokens"],
        filtered_tokens=body["filtered_tokens"],
        token_count=body["token_count"],
        unique_token_count=body["unique_token_count"],
        stopword_removed_count=body["stopword_removed_count"],
        sentences=body["sentences"],
        processing_ms=body["processing_ms"],
        model_versions=body["model_versions"],
        news_id=news_id,
    )


class IngestRequest(BaseModel):
    """Body for POST /api/ingest/youtube."""

    source: str = Field(
        min_length=1,
        max_length=500,
        description="YouTube URL, video id, or snapshot filename",
    )
    collector: str | None = Field(
        default=None, description="ytdlp | youtube_api | file"
    )
    limit: int | None = Field(default=None, ge=1, le=50_000)
    save_snapshot: bool = Field(
        default=False, description="Write an offline snapshot of the collected chat"
    )
    analyse: bool = Field(default=True, description="Run the NLP pipeline afterwards")


class IngestResponse(BaseModel):
    """Result of an ingestion run."""

    video_id: str
    title: str | None = None
    channel: str | None = None
    is_live: bool = False
    collected: int
    stored: int
    duplicates: int
    windows_created: int = 0
    messages_scored: int = 0
    messages_skipped_noise: int = 0
    snapshot: str | None = None


@router.post(
    "/ingest/youtube", response_model=IngestResponse, summary="Collect and analyse chat"
)
def ingest_youtube(
    payload: IngestRequest, db: Session = Depends(get_db)
) -> IngestResponse:
    """Collect a YouTube stream's chat, store it, and analyse it into documents."""
    if payload.collector and payload.collector not in AVAILABLE_COLLECTORS:
        raise HTTPException(
            422,
            f"Unknown collector {payload.collector!r}. "
            f"Available: {', '.join(AVAILABLE_COLLECTORS)}.",
        )

    try:
        collector = get_collector(payload.collector)
        result = collector.collect(payload.source, limit=payload.limit)
    except CollectorError as exc:
        # A collection failure is the user's problem to act on (private video,
        # chat disabled, no network), so report it as a 400 with the real reason.
        raise HTTPException(400, str(exc)) from exc

    snapshot_name = None
    if payload.save_snapshot:
        try:
            snapshot_name = write_snapshot(result).name
        except SnapshotWouldShrink as exc:
            # The messages were still collected and stored; only the snapshot
            # write was refused, so report it as a conflict the caller can fix.
            raise HTTPException(409, str(exc)) from exc

    stored = store_messages(db, result)

    windows = scored = noise = 0
    if payload.analyse:
        report = analyse_stream(db, stored.stream, get_pipeline())
        windows = report.windows_created
        scored = report.messages_scored
        noise = report.messages_skipped_noise

    return IngestResponse(
        video_id=result.stream.video_id,
        title=result.stream.title,
        channel=result.stream.channel,
        is_live=result.stream.is_live,
        collected=result.count,
        stored=stored.stored,
        duplicates=stored.skipped_duplicates,
        windows_created=windows,
        messages_scored=scored,
        messages_skipped_noise=noise,
        snapshot=snapshot_name,
    )


@router.get("/ingest/snapshots", summary="Offline snapshots available for replay")
def snapshots() -> list[dict]:
    """List committed chat snapshots, which replay with no network access."""
    return list_snapshots()


@router.post("/ingest/reanalyse", summary="Re-run the pipeline over stored chat")
def reanalyse(
    db: Session = Depends(get_db),
    stream_id: int | None = Body(
        default=None, embed=True, description="Limit to one stream"
    ),
) -> dict:
    """Re-analyse stored chat, for use after swapping a model.

    Existing documents for the affected streams are replaced rather than
    duplicated.
    """
    pipeline = get_pipeline()

    if stream_id is not None:
        stream = db.get(ChatStream, stream_id)
        if stream is None:
            raise HTTPException(404, f"Stream {stream_id} not found")
        return analyse_stream(db, stream, pipeline).as_dict()

    if not db.scalar(select(ChatStream).limit(1)):
        raise HTTPException(
            400,
            "No chat has been collected yet. Import a stream first via "
            "POST /api/ingest/youtube.",
        )
    return analyse_all_streams(db, pipeline).as_dict()
