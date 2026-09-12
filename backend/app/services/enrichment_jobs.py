"""Fill in the LLM work an import could not wait for, in the background.

There are two jobs a model does well here, and neither fits in a request:

* **A headline** for each story, because the extractive one is a phrase lifted
  out of ASR text and often reads like it.
* **One line of what the audience said** while that story was on air, because a
  hundred short Thai messages full of ``5555`` do not read as an opinion until
  something compresses them.

Both cost 10-40 seconds each. A 56-story programme is half an hour of headlines
and another half hour of reactions -- far too long to hold an HTTP request, and
"go and run a CLI" is a limitation with instructions attached rather than a
feature. So the work moves to a worker thread and the page watches it.

Design notes
------------
**Work is counted in units, not stories.** A story needing both a headline and a
reaction summary is two units, so the progress bar means what it says.

**One job at a time, globally.** Not a queueing nicety: the local model holds
about 6 GB, and two at once is how a 16 GB machine runs out of memory.

**Rows are updated, not re-analysed.** The stories already have their text,
topic, keywords and frames; re-running the pipeline would re-fetch, re-segment
and re-capture for nothing, and could move a boundary under a timeline someone
is reading.

**In-process and deliberately not durable.** A restart loses the job, not the
work: every unit is committed before the next one starts. A task queue would add
a broker to a project that has to run with two commands.
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
from app.services import enrichment_cache, reactions
from app.services.segmentation import LLM_FAILURE_LIMIT, Segment, enrich_with_llm

# States a job moves through. "cancelled" is reachable only by an explicit
# request; "failed" means the provider stopped answering.
QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
CANCELLED = "cancelled"

# The two kinds of unit a job performs.
HEADLINE = "headline"
REACTION = "reaction"


@dataclass
class EnrichmentJob:
    """Progress of one programme's LLM work."""

    video_id: str
    total: int = 0
    done: int = 0
    headlines: int = 0
    reactions: int = 0
    state: str = QUEUED
    note: str = ""
    # The most recent thing written, so the UI can show what is happening rather
    # than only how far along it is.
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
        """Estimated from the rate so far, or None until the first unit lands."""
        if not self.running or self.done == 0:
            return None
        return max(0, self.total - self.done) * (self.elapsed_seconds / self.done)

    def as_dict(self) -> dict:
        eta = self.eta_seconds
        return {
            "video_id": self.video_id,
            "state": self.state,
            "total": self.total,
            "done": self.done,
            "headlines": self.headlines,
            "reactions": self.reactions,
            "latest": self.latest,
            "note": self.note,
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "eta_seconds": round(eta, 1) if eta is not None else None,
        }


_lock = threading.Lock()
_jobs: dict[str, EnrichmentJob] = {}


class JobRejected(RuntimeError):
    """Raised when a job cannot start, with a reason meant for the user."""


def get(video_id: str) -> EnrichmentJob | None:
    with _lock:
        return _jobs.get(video_id)


def active() -> EnrichmentJob | None:
    """The job currently running, if any. Only one runs at a time."""
    with _lock:
        return next((job for job in _jobs.values() if job.running), None)


def cancel(video_id: str) -> EnrichmentJob | None:
    """Ask a job to stop after the unit it is on.

    Everything already written stays written -- each unit was committed as it
    landed, which is what makes cancelling safe rather than wasteful.
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


def plan(db: Session, video_id: str) -> list[tuple[int, str]]:
    """The units of work outstanding, as ``(segment id, kind)``.

    A reaction is only planned where there is actually chat to summarise: most
    videos have none at all, and a handful of messages is two people talking
    rather than an audience.
    """
    rows = segments_for(db, video_id)
    chat_counts = reactions.count_by_segment(db, rows)
    tasks: list[tuple[int, str]] = []
    for row in rows:
        if row.enriched_by != "llm":
            tasks.append((row.id, HEADLINE))
        if (
            not row.chat_summary
            and chat_counts.get(row.id, 0) >= reactions.MIN_MESSAGES_TO_SUMMARISE
        ):
            tasks.append((row.id, REACTION))
    return tasks


def pending(db: Session, video_id: str) -> tuple[int, int]:
    """``(headlines, reactions)`` still to write for this programme."""
    tasks = plan(db, video_id)
    return (
        sum(1 for _, kind in tasks if kind == HEADLINE),
        sum(1 for _, kind in tasks if kind == REACTION),
    )


def start(video_id: str) -> EnrichmentJob:
    """Begin writing headlines and chat summaries for ``video_id``.

    Returns the job already running for this video rather than starting a
    second one, so a double-clicked button is harmless.

    Raises:
        JobRejected: when another programme is being written, when the video has
            no stored stories, or when there is nothing left to write.
    """
    with _lock:
        busy = next((job for job in _jobs.values() if job.running), None)
        if busy is not None:
            if busy.video_id == video_id:
                return busy
            raise JobRejected(
                f"Already enriching {busy.video_id} ({busy.done}/{busy.total} done). "
                "The local model holds several gigabytes of memory, so one "
                "programme at a time."
            )

        with SessionLocal() as db:
            if not segments_for(db, video_id):
                raise JobRejected(
                    f"No stories are stored for {video_id}. Import the video first."
                )
            tasks = plan(db, video_id)
            if not tasks:
                raise JobRejected(
                    f"Everything the model can write for {video_id} is already written."
                )

        job = EnrichmentJob(video_id=video_id, total=len(tasks))
        _jobs[video_id] = job

    threading.Thread(
        target=_run, args=(job,), name=f"enrich-{video_id}", daemon=True
    ).start()
    return job


def _write_headline(db: Session, row: NewsSegment, cache) -> str:
    """Returns the enrichment status, and updates ``row`` when it worked."""
    segment = Segment(
        index=row.position,
        start_ms=row.start_ms,
        end_ms=row.end_ms,
        text=row.transcript_text or "",
    )
    # Carried over so a rejected completion leaves the extractive headline in
    # place rather than blanking the story.
    segment.headline = row.headline
    segment.summary = row.summary or ""

    status = enrich_with_llm(segment, cache=cache, allow_model=True)
    if status == "unavailable":
        return status

    row.headline = segment.headline
    # Only what the model actually produced. In headline-only mode it returns no
    # entities, and overwriting the extractor's with an empty list would throw
    # them away.
    if segment.entities:
        row.entities = segment.entities
    if segment.name_corrections:
        row.name_corrections = [list(pair) for pair in segment.name_corrections]
        row.summary = segment.summary
    row.enriched_by = "llm"
    return status


def _write_reaction(db: Session, row: NewsSegment) -> str:
    """Summarise the chat that arrived while this story was on air."""
    from app.nlp.llm_enrich import LLMUnavailable

    messages = reactions.messages_in(db, row)
    row.chat_message_count = len(messages)
    if len(messages) < reactions.MIN_MESSAGES_TO_SUMMARISE:
        # Not a failure of the model: there was nothing to summarise. Recorded
        # so the job does not come back to it.
        return "skipped"
    try:
        row.chat_summary = reactions.summarise_texts([m.text for m in messages])
    except LLMUnavailable:
        return "unavailable"
    return "model"


def _run(job: EnrichmentJob) -> None:
    """Worker body. Commits after every unit, so nothing is ever lost."""
    consecutive_failures = 0
    try:
        job.state = RUNNING
        cache = enrichment_cache.load(job.video_id, autosave=True)

        with SessionLocal() as db:
            for segment_id, kind in plan(db, job.video_id):
                if job.cancel_requested.is_set():
                    job.state = CANCELLED
                    job.note = "Cancelled. Everything already written was kept."
                    return

                row = db.get(NewsSegment, segment_id)
                if row is None:  # deleted under us by a re-analysis
                    job.done += 1
                    continue

                if kind == HEADLINE:
                    status = _write_headline(db, row, cache)
                else:
                    status = _write_reaction(db, row)

                if status == "unavailable":
                    consecutive_failures += 1
                    job.done += 1
                    if consecutive_failures >= LLM_FAILURE_LIMIT:
                        job.state = FAILED
                        job.note = (
                            "The model stopped answering -- is Ollama running? "
                            "Everything already written was kept."
                        )
                        return
                    continue

                consecutive_failures = 0
                # Per unit, so the timeline fills in while the user watches and
                # a restart keeps everything finished so far.
                db.commit()
                job.done += 1
                if kind == HEADLINE:
                    job.headlines += 1
                    job.latest = row.headline
                elif status == "model":
                    job.reactions += 1
                    job.latest = row.chat_summary

        job.state = DONE
    except Exception as exc:  # noqa: BLE001 - a worker thread must not die silently
        job.state = FAILED
        job.note = f"{type(exc).__name__}: {exc}"
    finally:
        job.finished_at = time.time()
