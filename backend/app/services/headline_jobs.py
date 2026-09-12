"""Write LLM headlines for an already-analysed programme, in the background.

Why this exists
---------------
Importing a video through the UI produces the extractive headline for every
story, because writing one takes about 20 seconds and a 56-story programme is
nineteen minutes -- far longer than an HTTP request may hold. Telling the user to
go and run ``analyse_video.py`` instead is not a feature, it is a limitation with
instructions attached: the imported programme sits there reading visibly worse
than the two that ship with the repository.

So the work moves to a worker thread and the UI watches it. The import stays
fast, the timeline fills in while the user looks at it, and every headline is
committed the moment it is written -- a refresh, a reload, or a killed process
keeps everything finished so far.

Design notes
------------
**One job at a time, globally.** Not a queueing nicety: the local model holds
about 6 GB, and two of these at once is how this machine runs out of memory.

**Rows are updated, not re-analysed.** The segments already exist with their
text, topic, keywords and frames; only the headline is missing. Re-running the
whole pipeline would re-fetch, re-segment and re-capture for nothing, and risk
moving a boundary under a timeline the user is reading.

**In-process and deliberately not durable.** A restart loses the job, not the
work: every headline reaches the database and the enrichment cache before the
next one starts. A task queue would add a broker to a project that has to run
with two commands.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database.session import SessionLocal
from app.models.broadcast import NewsSegment, VideoTranscript
from app.models.chat import ChatStream
from app.services import enrichment_cache
from app.services.segmentation import LLM_FAILURE_LIMIT, Segment, enrich_with_llm

# States a job moves through. "cancelled" is reachable only by an explicit
# request; "failed" means the provider stopped answering.
QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"


@dataclass
class HeadlineJob:
    """Progress of one programme's headline writing."""

    video_id: str
    total: int = 0
    done: int = 0
    written: int = 0
    reused: int = 0
    state: str = QUEUED
    note: str = ""
    # The most recent headline, so the UI can show what is happening rather than
    # only how far along it is.
    latest: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    cancel_requested: threading.Event = field(
        default_factory=threading.Event, repr=False
    )

    @property
    def running(self) -> bool:
        return self.state in {QUEUED, RUNNING}

    @property
    def elapsed_seconds(self) -> float:
        return (self.finished_at or time.time()) - self.started_at

    @property
    def eta_seconds(self) -> float | None:
        """Estimated from the rate so far, or None until the first one lands."""
        if not self.running or self.done == 0:
            return None
        remaining = max(0, self.total - self.done)
        return remaining * (self.elapsed_seconds / self.done)

    def as_dict(self) -> dict:
        eta = self.eta_seconds
        return {
            "video_id": self.video_id,
            "state": self.state,
            "total": self.total,
            "done": self.done,
            "written": self.written,
            "reused": self.reused,
            "latest": self.latest,
            "note": self.note,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "eta_seconds": round(eta, 1) if eta is not None else None,
        }


_lock = threading.Lock()
_jobs: dict[str, HeadlineJob] = {}


class JobRejected(RuntimeError):
    """Raised when a job cannot start, with a reason meant for the user."""


def get(video_id: str) -> HeadlineJob | None:
    with _lock:
        return _jobs.get(video_id)


def active() -> HeadlineJob | None:
    """The job currently running, if any. Only one runs at a time."""
    with _lock:
        return next((job for job in _jobs.values() if job.running), None)


def cancel(video_id: str) -> HeadlineJob | None:
    """Ask a job to stop after the story it is on.

    Headlines already written stay written -- each one was committed as it
    landed, which is the whole reason cancelling is safe.
    """
    job = get(video_id)
    if job is not None and job.running:
        job.cancel_requested.set()
    return job


def segments_for(db: Session, video_id: str) -> list[NewsSegment]:
    """Every stored story for one video, in broadcast order."""
    return list(
        db.scalars(
            select(NewsSegment)
            .join(VideoTranscript, NewsSegment.transcript_id == VideoTranscript.id)
            .join(ChatStream, VideoTranscript.stream_id == ChatStream.id)
            .where(ChatStream.video_id == video_id)
            .order_by(NewsSegment.start_ms)
        )
    )


def pending_count(db: Session, video_id: str) -> int:
    """Stories in this programme with no written headline yet."""
    return sum(1 for row in segments_for(db, video_id) if row.enriched_by != "llm")


def start(video_id: str) -> HeadlineJob:
    """Begin writing headlines for ``video_id``.

    Returns the job already running for this video rather than starting a
    second one, so a double-clicked button is harmless.

    Raises:
        JobRejected: when another programme is being written, when the video has
            no stored stories, or when it has nothing left to write.
    """
    with _lock:
        busy = next((job for job in _jobs.values() if job.running), None)
        if busy is not None:
            if busy.video_id == video_id:
                return busy
            raise JobRejected(
                f"Already writing headlines for {busy.video_id} "
                f"({busy.done}/{busy.total} done). The local model holds several "
                "gigabytes of memory, so one programme at a time."
            )

        with SessionLocal() as db:
            rows = segments_for(db, video_id)
            if not rows:
                raise JobRejected(
                    f"No stories are stored for {video_id}. Import the video first."
                )
            outstanding = sum(1 for row in rows if row.enriched_by != "llm")
            if outstanding == 0:
                raise JobRejected(
                    f"Every story in {video_id} already has a written headline."
                )

        job = HeadlineJob(video_id=video_id, total=outstanding)
        _jobs[video_id] = job

    threading.Thread(
        target=_run, args=(job,), name=f"headlines-{video_id}", daemon=True
    ).start()
    return job


def _run(job: HeadlineJob) -> None:
    """Worker body. Commits after every headline, so nothing is ever lost."""
    consecutive_failures = 0
    try:
        job.state = RUNNING
        cache = enrichment_cache.load(job.video_id, autosave=True)

        with SessionLocal() as db:
            for row in segments_for(db, job.video_id):
                if job.cancel_requested.is_set():
                    job.state = CANCELLED
                    job.note = "Cancelled. Headlines already written were kept."
                    return
                if row.enriched_by == "llm":
                    continue

                segment = Segment(
                    index=row.position,
                    start_ms=row.start_ms,
                    end_ms=row.end_ms,
                    text=row.transcript_text or "",
                )
                # Carried over so a rejected completion leaves the extractive
                # headline in place rather than blanking the story.
                segment.headline = row.headline
                segment.summary = row.summary or ""

                status = enrich_with_llm(segment, cache=cache, allow_model=True)

                if status == "unavailable":
                    consecutive_failures += 1
                    job.done += 1
                    if consecutive_failures >= LLM_FAILURE_LIMIT:
                        job.state = FAILED
                        job.note = (
                            "The model stopped answering -- is Ollama running? "
                            "Headlines already written were kept."
                        )
                        return
                    continue

                consecutive_failures = 0
                row.headline = segment.headline
                # Only what the model actually produced. In headline-only mode it
                # returns no entities, and overwriting the extractor's with an
                # empty list would throw them away.
                if segment.entities:
                    row.entities = segment.entities
                if segment.name_corrections:
                    row.name_corrections = [
                        list(pair) for pair in segment.name_corrections
                    ]
                    row.summary = segment.summary
                row.enriched_by = "llm"
                # Per story, so the timeline fills in while the user watches and
                # a restart keeps everything finished so far.
                db.commit()

                job.done += 1
                job.latest = segment.headline
                if status == "model":
                    job.written += 1
                else:
                    job.reused += 1

        job.state = DONE
    except Exception as exc:  # noqa: BLE001 - a worker thread must not die silently
        job.state = FAILED
        job.note = f"{type(exc).__name__}: {exc}"
    finally:
        job.finished_at = time.time()
