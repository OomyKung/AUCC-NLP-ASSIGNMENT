"""Turn a news video into a timeline of stories, and store it.

This is the orchestration layer for the three capabilities the dashboard needs:

1. fetch what was *said* (``app.services.transcripts``)
2. split it into stories and analyse each (``app.services.segmentation``)
3. capture a frame at each story's first moment (``app.services.frames``)

and persist the result so the timeline is served from the database rather than
recomputed per request -- segmenting a four-hour programme takes about ten
seconds, which is fine for ingest and far too slow for a page load.

Frame capture is deliberately best-effort. A missing thumbnail makes a timeline
card plainer; it does not make the analysis wrong, so a storyboard failure is
recorded and skipped rather than aborting an ingest that otherwise succeeded.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.broadcast import NewsSegment, VideoTranscript
from app.models.chat import ChatStream
from app.services import enrichment_cache
from app.services.collectors.base import extract_video_id, watch_url
from app.services.frames import FrameUnavailable, capture_frames
from app.services.segmentation import Segment, segment_transcript
from app.services.transcripts import (
    Transcript,
    TranscriptUnavailable,
    get_transcript_provider,
)


@dataclass(slots=True)
class BroadcastResult:
    """What one ingest produced, for the API and the CLI to report."""

    video_id: str
    source: str
    duration_ms: int
    cue_count: int
    segment_count: int
    frames_captured: int
    frame_note: str = ""
    # Where the headlines came from. Reported rather than inferred, because
    # "cached" and "just written by a model" cost 0 and 20 seconds a story and a
    # caller watching a long run deserves to know which it is getting.
    headlines_cached: int = 0
    headlines_written: int = 0

    def as_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "transcript_source": self.source,
            "duration_ms": self.duration_ms,
            "cue_count": self.cue_count,
            "segment_count": self.segment_count,
            "frames_captured": self.frames_captured,
            "frame_note": self.frame_note,
            "headlines_cached": self.headlines_cached,
            "headlines_written": self.headlines_written,
        }


def transcript_snapshot_path(video_id: str) -> Path:
    """Where a fetched transcript is cached for offline rebuilds."""
    return settings.data_dir / "transcripts" / f"{video_id}.json"


def save_transcript_snapshot(transcript: Transcript) -> Path:
    """Write a transcript to disk so seed.py can rebuild without a network.

    The same reasoning as the committed chat snapshots: an analysis that only
    works when YouTube is reachable is not demonstrable, and a presentation is
    exactly when the network fails.
    """
    import json

    target = transcript_snapshot_path(transcript.video_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(transcript.as_dict(), ensure_ascii=False), encoding="utf-8"
    )
    return target


def load_transcript_snapshot(video_id: str) -> Transcript | None:
    """Read a cached transcript, or None when there is not one."""
    import json

    source = transcript_snapshot_path(video_id)
    if not source.is_file():
        return None
    try:
        return Transcript.from_dict(json.loads(source.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValueError):
        return None


def _relative_frame_path(path: Path) -> str:
    """Store frames relative to the data directory.

    An absolute path would break the moment the project moves, and these rows
    are committed-adjacent data that has to survive a clone into a different
    directory.
    """
    try:
        return path.relative_to(settings.data_dir).as_posix()
    except ValueError:
        return path.name


def ensure_stream(db: Session, video_id: str, *, title: str | None = None) -> ChatStream:
    """Find or create the stream row a transcript hangs off.

    A video can be analysed for its spoken content without its chat ever having
    been collected, so the row may not exist yet.
    """
    stream = db.scalar(select(ChatStream).where(ChatStream.video_id == video_id))
    if stream is not None:
        return stream

    stream = ChatStream(
        video_id=video_id,
        url=watch_url(video_id),
        title=title,
        collector="transcript",
    )
    db.add(stream)
    db.flush()
    return stream


def store_transcript(
    db: Session, stream: ChatStream, transcript: Transcript
) -> VideoTranscript:
    """Persist a transcript, replacing any previous one for this stream."""
    existing = db.scalar(
        select(VideoTranscript).where(VideoTranscript.stream_id == stream.id)
    )
    if existing is not None:
        # Replacing rather than appending: a re-fetch supersedes, and the
        # cascade removes the segments derived from the old text so they can
        # never outlive the transcript they were computed from.
        db.delete(existing)
        db.flush()

    row = VideoTranscript(
        stream_id=stream.id,
        source=transcript.source,
        language=transcript.language,
        duration_ms=transcript.duration_ms,
        cue_count=len(transcript.cues),
        character_count=transcript.character_count,
        cues=[cue.as_dict() for cue in transcript.cues],
    )
    db.add(row)
    db.flush()
    return row


def store_segments(
    db: Session,
    transcript_row: VideoTranscript,
    stream: ChatStream,
    segments: list[Segment],
    frames: dict[int, Path],
) -> int:
    """Persist the story segments for a transcript."""
    for segment in segments:
        frame = frames.get(segment.start_seconds)
        db.add(
            NewsSegment(
                transcript_id=transcript_row.id,
                stream_id=stream.id,
                position=segment.index,
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                headline=segment.headline[:300],
                summary=segment.summary,
                transcript_text=segment.text,
                topic=segment.topic or "other",
                topic_confidence=segment.topic_confidence,
                sentiment=segment.sentiment or "neutral",
                sentiment_confidence=segment.sentiment_confidence,
                keywords=list(segment.keywords),
                entities=list(segment.entities),
                boundary_reasons=list(segment.boundary_reasons),
                name_corrections=[list(pair) for pair in segment.name_corrections],
                enriched_by=segment.enriched_by,
                boundary_confidence=segment.boundary_confidence,
                youtube_url=segment.youtube_url(stream.video_id),
                frame_path=_relative_frame_path(frame) if frame else None,
            )
        )
    db.flush()
    return len(segments)


def analyse_video(
    db: Session,
    source: str,
    *,
    with_frames: bool = True,
    title: str | None = None,
    refresh: bool = False,
    progress: Callable[[int, int, Segment], None] | None = None,
    allow_model: bool = True,
) -> BroadcastResult:
    """Fetch, segment, capture and store one video's spoken content.

    Raises:
        TranscriptUnavailable: when no transcript can be produced, which is the
            only failure that makes the whole operation pointless.

    Args:
        refresh: Ignore any cached transcript snapshot and re-fetch.
        progress: Called after each story is analysed, as
            ``(done, total, segment)``. With LLM headlines enabled a long
            programme takes minutes, so a CLI caller needs to report movement.
        allow_model: Whether a headline may be *written*. Cached headlines are
            used either way. False keeps an HTTP request short -- writing 76 of
            them takes 25 minutes.
    """
    video_id = extract_video_id(source)

    # An existing snapshot is used unless a refresh is asked for, so re-running
    # the analysis (or seeding a fresh clone) needs no network.
    transcript = None if refresh else load_transcript_snapshot(video_id)
    if transcript is None:
        transcript = get_transcript_provider().fetch(video_id)
        save_transcript_snapshot(transcript)

    # Headlines already written for this video are reused rather than paid for
    # again: 20 seconds a story adds up to 40 minutes over the stored
    # programmes, and the result is committed so every clone gets the same
    # timeline without running a model at all.
    # autosave, so a 25-minute run interrupted at minute 24 keeps what it wrote.
    cache = enrichment_cache.load(video_id, autosave=True)
    segments, _boundaries, _blocks = segment_transcript(
        transcript, progress=progress, cache=cache, allow_model=allow_model
    )

    frames: dict[int, Path] = {}
    note = ""
    if with_frames and segments:
        try:
            frames = capture_frames(
                video_id, [segment.start_seconds for segment in segments]
            )
        except FrameUnavailable as exc:
            # Cosmetic loss only: the timeline still works without thumbnails.
            note = str(exc)

    stream = ensure_stream(db, video_id, title=title)
    transcript_row = store_transcript(db, stream, transcript)
    store_segments(db, transcript_row, stream, segments, frames)
    db.commit()

    return BroadcastResult(
        video_id=video_id,
        source=transcript.source,
        duration_ms=transcript.duration_ms,
        cue_count=len(transcript.cues),
        segment_count=len(segments),
        frames_captured=len(frames),
        frame_note=note,
        headlines_cached=cache.hits,
        headlines_written=cache.writes,
    )


def list_segments(
    db: Session,
    *,
    video_id: str | None = None,
    topic: str | None = None,
    limit: int = 200,
    offset: int = 0,
) -> tuple[list[NewsSegment], int]:
    """Segments for the timeline, newest programme first within a video."""
    from sqlalchemy import func

    statement = select(NewsSegment)
    counter = select(func.count()).select_from(NewsSegment)

    if video_id:
        stream_ids = select(ChatStream.id).where(ChatStream.video_id == video_id)
        statement = statement.where(NewsSegment.stream_id.in_(stream_ids))
        counter = counter.where(NewsSegment.stream_id.in_(stream_ids))
    if topic:
        statement = statement.where(NewsSegment.topic == topic)
        counter = counter.where(NewsSegment.topic == topic)

    total = int(db.scalar(counter) or 0)
    rows = list(
        db.scalars(
            statement.order_by(NewsSegment.stream_id, NewsSegment.start_ms)
            .offset(offset)
            .limit(limit)
        )
    )
    return rows, total


__all__ = [
    "BroadcastResult",
    "TranscriptUnavailable",
    "analyse_video",
    "list_segments",
]
