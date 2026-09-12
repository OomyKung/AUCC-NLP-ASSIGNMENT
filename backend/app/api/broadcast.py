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
    EnrichmentJobRequest,
    EnrichmentJobStatus,
    ProgrammeOut,
    ReactionOut,
    SegmentReaction,
    SegmentListResponse,
    SegmentOut,
    TranscriptOut,
)
from app.services import enrichment_jobs, reactions
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
        name_corrections=[list(pair) for pair in (segment.name_corrections or [])],
        enriched_by=segment.enriched_by or "",
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
        # One query for the whole programme, not one per story.
        chat_counts = reactions.count_by_segment(db, list(transcript.segments))
        chat_segments = sum(1 for count in chat_counts.values() if count)
        pending_reactions = sum(
            1
            for segment in transcript.segments
            if chat_counts.get(segment.id, 0) >= reactions.MIN_MESSAGES_TO_SUMMARISE
            and not segment.chat_summary
        )
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
                pending_headlines=sum(
                    1
                    for segment in transcript.segments
                    if segment.enriched_by != "llm"
                ),
                pending_reactions=pending_reactions,
                chat_segments=chat_segments,
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

    Headlines already written for this video are reused from the committed cache,
    but new ones are *not* written unless ``write_headlines`` asks: a model spends
    about 20 seconds a story, so a 76-story programme would hold the request open
    for 25 minutes. ``analyse_video.py`` is the place for that, and it reports
    progress while it works.
    """
    try:
        result = analyse_video(
            db,
            payload.url,
            with_frames=payload.with_frames,
            allow_model=payload.write_headlines,
        )
    except TranscriptUnavailable as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except CollectorError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return AnalyseVideoResponse(**result.as_dict())


@router.post(
    "/broadcast/enrich",
    response_model=EnrichmentJobStatus,
    summary="Write LLM headlines and chat summaries, in the background",
)
def start_enrichment(payload: EnrichmentJobRequest) -> EnrichmentJobStatus:
    """Fill in the model work an import could not wait for.

    Returns immediately with a job to poll. The work runs in a worker thread and
    commits each unit as it lands, so the timeline fills in while the caller
    watches and nothing is lost if the poll stops or the process restarts.

    Calling this again for a programme already being written returns the same
    job rather than starting a second one.
    """
    try:
        job = enrichment_jobs.start(payload.video_id)
    except enrichment_jobs.JobRejected as exc:
        # Every rejection is something the caller can act on: another programme
        # is running, the video was never imported, or there is nothing to do.
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return EnrichmentJobStatus(**job.as_dict())


@router.get(
    "/broadcast/enrich/{video_id}",
    response_model=EnrichmentJobStatus,
    summary="Progress of an enrichment job",
)
def enrichment_status(video_id: str) -> EnrichmentJobStatus:
    job = enrichment_jobs.get(video_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail=f"No enrichment job has been started for {video_id}."
        )
    return EnrichmentJobStatus(**job.as_dict())


@router.delete(
    "/broadcast/enrich/{video_id}",
    response_model=EnrichmentJobStatus,
    summary="Stop an enrichment job",
)
def stop_enrichment(video_id: str) -> EnrichmentJobStatus:
    """Stop after the unit in flight. Everything written so far is kept."""
    job = enrichment_jobs.cancel(video_id)
    if job is None:
        raise HTTPException(
            status_code=404, detail=f"No enrichment job has been started for {video_id}."
        )
    return EnrichmentJobStatus(**job.as_dict())


@router.get(
    "/broadcast/programmes/{video_id}/reactions",
    response_model=list[SegmentReaction],
    summary="Every story's audience reaction, without the messages",
)
def programme_reactions(
    video_id: str, db: Session = Depends(get_db)
) -> list[SegmentReaction]:
    """One request for the whole timeline's viewer view.

    Rendering 76 cards must not be 76 requests, and shipping every message to
    draw a mood bar would be most of a megabyte nobody asked to read.
    """
    segments = enrichment_jobs.segments_for(db, video_id)
    if not segments:
        raise HTTPException(
            status_code=404, detail=f"No stories are stored for {video_id}."
        )
    overview = reactions.overview_by_segment(db, segments)
    return [
        SegmentReaction(
            segment_id=segment_id,
            total=reaction.total,
            sentiment_counts=reaction.sentiment_counts,
            mood=reaction.mood,
            summary=reaction.summary,
        )
        for segment_id, reaction in overview.items()
    ]


@router.get(
    "/broadcast/segments/{segment_id}/chat",
    response_model=ReactionOut,
    summary="What viewers said while this story was on air",
)
def segment_reaction(
    segment_id: int, db: Session = Depends(get_db)
) -> ReactionOut:
    """The other half of the broadcast: the audience, aligned to the same clock.

    Chat and transcript are timestamped against the same stream, so a story's
    time range selects the messages that reacted to it. Everything but the
    one-line summary is computed from stored data, so this stays fast and works
    with no model at all.
    """
    segment = db.get(NewsSegment, segment_id)
    if segment is None:
        raise HTTPException(status_code=404, detail=f"No story with id {segment_id}.")
    return ReactionOut(**reactions.build(db, segment).as_dict())
