"""Endpoints for the spoken-content timeline.

Serves what the newsreader said -- split into stories, each with a topic, a
captured frame and a deep link to the moment it starts -- as opposed to the chat
endpoints, which serve what the audience said.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.session import get_db
from app.models.broadcast import NewsSegment, VideoTranscript
from app.models.chat import ChatStream
from app.schemas.broadcast import (
    AnalyseVideoRequest,
    AnalyseVideoResponse,
    ProgrammeOut,
    SegmentListResponse,
    SegmentOut,
    TranscriptOut,
)
from app.services.broadcast import analyse_video, list_segments
from app.services.collectors.base import CollectorError
from app.services.transcripts import TranscriptUnavailable
from app.taxonomy import SENTIMENT_BY_SLUG, TOPIC_BY_SLUG

router = APIRouter(tags=["broadcast"])

# How much spoken text a segment card shows before "read more".
PREVIEW_CHARS = 600


def _preview(text: str) -> str:
    """Bounded excerpt of a story's transcript."""
    body = (text or "").strip()
    if len(body) <= PREVIEW_CHARS:
        return body
    return body[:PREVIEW_CHARS].rstrip() + "…"


def _to_out(segment: NewsSegment) -> SegmentOut:
    """Map a row to its response shape, resolving display names once here."""
    topic = TOPIC_BY_SLUG.get(segment.topic)
    sentiment = SENTIMENT_BY_SLUG.get(segment.sentiment)
    return SegmentOut(
        id=segment.id,
        position=segment.position,
        start_ms=segment.start_ms,
        end_ms=segment.end_ms,
        duration_ms=segment.duration_ms,
        timecode=segment.timecode,
        headline=segment.headline,
        summary=segment.summary,
        topic=segment.topic,
        topic_label=topic.thai if topic else segment.topic,
        topic_color=topic.color if topic else "#64748B",
        topic_confidence=segment.topic_confidence,
        sentiment=segment.sentiment,
        sentiment_label=sentiment.thai if sentiment else segment.sentiment,
        sentiment_confidence=segment.sentiment_confidence,
        keywords=list(segment.keywords or []),
        entities=list(segment.entities or []),
        transcript_text_preview=_preview(segment.transcript_text),
        boundary_reasons=list(segment.boundary_reasons or []),
        boundary_confidence=segment.boundary_confidence,
        youtube_url=segment.youtube_url,
        # Frames are served by the static mount in main.py; the stored value is
        # relative to the data directory so it survives a move or a fresh clone.
        frame_url=f"/media/{segment.frame_path}" if segment.frame_path else None,
    )


@router.get(
    "/broadcast/segments",
    response_model=SegmentListResponse,
    summary="News stories detected in video audio",
)
def segments(
    db: Session = Depends(get_db),
    video_id: str | None = Query(None, description="Restrict to one video"),
    topic: str | None = Query(None, description="Restrict to one topic slug"),
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> SegmentListResponse:
    """Every detected story, newest programme first."""
    rows, total = list_segments(
        db, video_id=video_id, topic=topic, limit=limit, offset=offset
    )
    return SegmentListResponse(
        total=total, limit=limit, offset=offset, items=[_to_out(r) for r in rows]
    )


@router.get(
    "/broadcast/programmes",
    response_model=list[ProgrammeOut],
    summary="Videos that have a transcript",
)
def programmes(db: Session = Depends(get_db)) -> list[ProgrammeOut]:
    """One entry per analysed video, with its story count."""
    results: list[ProgrammeOut] = []
    transcripts = list(db.scalars(select(VideoTranscript)))
    for transcript in transcripts:
        stream = db.get(ChatStream, transcript.stream_id)
        if stream is None:
            continue
        results.append(
            ProgrammeOut(
                video_id=stream.video_id,
                title=stream.title,
                channel=stream.channel,
                url=stream.url,
                transcript=TranscriptOut(
                    source=transcript.source,
                    language=transcript.language,
                    duration_ms=transcript.duration_ms,
                    cue_count=transcript.cue_count,
                    character_count=transcript.character_count,
                ),
                segment_count=len(transcript.segments),
            )
        )
    return results


@router.get(
    "/broadcast/programmes/{video_id}",
    response_model=ProgrammeOut,
    summary="One programme with all its stories",
)
def programme(video_id: str, db: Session = Depends(get_db)) -> ProgrammeOut:
    """The full timeline for one video."""
    stream = db.scalar(select(ChatStream).where(ChatStream.video_id == video_id))
    if stream is None or stream.transcript is None:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No transcript stored for {video_id}. "
                "Analyse it first: POST /api/broadcast/analyse"
            ),
        )

    transcript = stream.transcript
    rows, _total = list_segments(db, video_id=video_id, limit=500)
    return ProgrammeOut(
        video_id=stream.video_id,
        title=stream.title,
        channel=stream.channel,
        url=stream.url,
        transcript=TranscriptOut(
            source=transcript.source,
            language=transcript.language,
            duration_ms=transcript.duration_ms,
            cue_count=transcript.cue_count,
            character_count=transcript.character_count,
        ),
        segment_count=len(rows),
        segments=[_to_out(r) for r in rows],
    )


@router.post(
    "/broadcast/analyse",
    response_model=AnalyseVideoResponse,
    summary="Transcribe a video and split it into stories",
)
def analyse(
    payload: AnalyseVideoRequest, db: Session = Depends(get_db)
) -> AnalyseVideoResponse:
    """Run the full spoken-content pipeline over one video.

    Slow by nature -- a four-hour programme takes roughly a minute -- because it
    fetches the transcript, classifies every 30-second block, segments, and
    captures a frame per story.
    """
    try:
        result = analyse_video(db, payload.url, with_frames=payload.with_frames)
    except TranscriptUnavailable as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except CollectorError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AnalyseVideoResponse(**result.as_dict())
