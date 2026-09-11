"""Ad-hoc analysis and YouTube ingestion endpoints."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.chat import ChatStream
from app.schemas.news import AnalyzeRequest, AnalyzeResponse
from app.services.chat_store import store_messages
from app.services.broadcast import analyse_video
from app.services.collectors import (
    AVAILABLE_COLLECTORS,
    ChatUnavailable,
    CollectorError,
    get_collector,
)
from app.services.ingest import analyse_all_streams, analyse_stream
from app.services.pipeline import get_pipeline
from app.services.snapshot import SnapshotWouldShrink, list_snapshots, write_snapshot
from app.services.transcripts import TranscriptUnavailable

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
    transcript: Literal["auto", "always", "never"] = Field(
        default="auto",
        description=(
            "Whether to also analyse what was *said* in the video, which needs "
            "no chat at all. 'auto' runs it only when the video has no chat -- "
            "news channels routinely switch chat replay off once a broadcast "
            "ends, and the programme is still fully analysable. 'always' builds "
            "the story timeline even when chat was collected; 'never' skips it."
        ),
    )
    with_frames: bool = Field(
        default=True,
        description=(
            "Capture a still frame per story. Costs a second or two each, so "
            "turn it off for a long programme you only want the text of."
        ),
    )


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

    # The two halves are reported separately because either can succeed alone:
    # a video with chat disabled still has a transcript, and a video with no
    # captions still has chat.
    chat_available: bool = True
    chat_note: str = ""
    segments_created: int = 0
    frames_captured: int = 0
    transcript_source: str | None = None
    transcript_note: str = ""
    # How many stories got a written headline. Zero on a video nobody has run
    # analyse_video.py over, because writing them takes ~20s each -- longer than
    # a request should hold. Reported so the UI can say so rather than leaving
    # the user wondering why these headlines read worse than the shipped ones.
    headlines_cached: int = 0


@router.post(
    "/ingest/youtube", response_model=IngestResponse, summary="Collect and analyse chat"
)
def ingest_youtube(
    payload: IngestRequest, db: Session = Depends(get_db)
) -> IngestResponse:
    """Analyse a YouTube video: its chat, what was said in it, or both.

    The two are independent sources of evidence and either can be missing. News
    channels routinely switch chat replay off once a broadcast ends, and that
    used to fail the whole import -- even though the programme's own audio was
    still there to transcribe, segment into stories and put on the timeline. So
    a missing chat is now a note, not an error, and only a video with *neither*
    is a failure.
    """
    if payload.collector and payload.collector not in AVAILABLE_COLLECTORS:
        raise HTTPException(
            422,
            f"Unknown collector {payload.collector!r}. "
            f"Available: {', '.join(AVAILABLE_COLLECTORS)}.",
        )

    result = None
    chat_note = ""
    stream_info = None
    try:
        collector = get_collector(payload.collector)
        result = collector.collect(payload.source, limit=payload.limit)
    except ChatUnavailable as exc:
        # The video is fine, it simply has no chat -- and the metadata request
        # already succeeded, so the title and channel survive for the fallback.
        chat_note = str(exc)
        stream_info = exc.stream
    except CollectorError as exc:
        # Anything else is a real failure of the *video* (private, removed, no
        # network), and no amount of transcript work will fix it.
        raise HTTPException(400, str(exc)) from exc

    snapshot_name = None
    stored = None
    windows = scored = noise = 0
    if result is not None:
        if payload.save_snapshot:
            try:
                snapshot_name = write_snapshot(result).name
            except SnapshotWouldShrink as exc:
                # The messages were still collected and stored; only the snapshot
                # write was refused, so report it as a conflict the caller can fix.
                raise HTTPException(409, str(exc)) from exc

        stored = store_messages(db, result)
        if payload.analyse:
            report = analyse_stream(db, stored.stream, get_pipeline())
            windows = report.windows_created
            scored = report.messages_scored
            noise = report.messages_skipped_noise

    # ------------------------------------------------- what was said in it
    wants_transcript = payload.transcript == "always" or (
        payload.transcript == "auto" and result is None
    )
    broadcast = None
    transcript_note = ""
    if wants_transcript:
        try:
            broadcast = analyse_video(
                db,
                payload.source,
                with_frames=payload.with_frames,
                title=(stream_info.title if stream_info else None),
                # Headlines already written are reused; new ones are not written
                # here, because 20 seconds a story is longer than a request
                # should take. analyse_video.py is the place for that.
                allow_model=False,
            )
        except (TranscriptUnavailable, CollectorError) as exc:
            transcript_note = str(exc)

    if result is None and broadcast is None:
        # Neither source produced anything, which is the only real failure.
        detail = chat_note
        if transcript_note:
            detail = f"{chat_note} {transcript_note}".strip()
        raise HTTPException(400, detail or "Nothing could be collected for this video.")

    info = result.stream if result is not None else stream_info
    return IngestResponse(
        video_id=(
            info.video_id
            if info is not None
            else (broadcast.video_id if broadcast else payload.source)
        ),
        title=info.title if info is not None else None,
        channel=info.channel if info is not None else None,
        is_live=bool(info.is_live) if info is not None else False,
        collected=result.count if result is not None else 0,
        stored=stored.stored if stored is not None else 0,
        duplicates=stored.skipped_duplicates if stored is not None else 0,
        windows_created=windows,
        messages_scored=scored,
        messages_skipped_noise=noise,
        snapshot=snapshot_name,
        chat_available=result is not None,
        chat_note=chat_note,
        segments_created=broadcast.segment_count if broadcast else 0,
        frames_captured=broadcast.frames_captured if broadcast else 0,
        transcript_source=broadcast.source if broadcast else None,
        transcript_note=transcript_note,
        headlines_cached=broadcast.headlines_cached if broadcast else 0,
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
