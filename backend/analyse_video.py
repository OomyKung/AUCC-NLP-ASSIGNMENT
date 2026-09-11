"""Transcribe a news video, split it into stories, and store the timeline.

This is the entry point for the spoken-content half of the project: it listens to
what the newsreader actually said (via the configured transcript backend), finds
where one story ends and the next begins, classifies each, captures a still frame
at each story's first moment, and writes a deep link back to that second.

Usage::

    python analyse_video.py https://www.youtube.com/watch?v=VIDEO_ID
    python analyse_video.py VIDEO_ID --no-frames      # skip thumbnails
    python analyse_video.py VIDEO_ID --show 20        # print more of the timeline
    python analyse_video.py --list                    # what is already stored

A four-hour programme takes roughly a minute: a second or two to fetch the
transcript, ten seconds to classify every 30-second block and segment, and the
rest capturing one frame per story.
"""

from __future__ import annotations

import argparse
import sys
import time

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from sqlalchemy import select  # noqa: E402

from app.database.base import Base  # noqa: E402
from app.database.session import SessionLocal, engine  # noqa: E402
from app.models.broadcast import VideoTranscript  # noqa: E402
from app.models.chat import ChatStream  # noqa: E402
from app.services.broadcast import analyse_video, list_segments  # noqa: E402
from app.services.collectors.base import CollectorError  # noqa: E402
from app.services.transcripts import (  # noqa: E402
    TranscriptUnavailable,
    provider_status,
)
from app.config import settings  # noqa: E402
from app.services.segmentation import Segment  # noqa: E402


def llm_status() -> str:
    """One line describing who will write the headlines, and what it costs."""
    if not settings.llm_enrich_segments:
        return "off (extractive headlines)"
    if settings.llm_provider == "ollama":
        return (
            f"ollama {settings.ollama_model} at {settings.ollama_base_url}"
            " (local, free, ~20s per story)"
        )
    if settings.has_llm:
        return f"anthropic {settings.llm_model} (billed per story)"
    return "unavailable: LLM_PROVIDER=anthropic but LLM_API_KEY is unset"


def report_progress(done: int, total: int, segment: Segment) -> None:
    """Overwrite one line per story.

    A local model spends about 20 seconds per story, so a long programme is a
    quarter of an hour of apparent silence without this.
    """
    headline = segment.headline[:48]
    sys.stdout.write(f"\r  story {done:>3}/{total}  {segment.timecode:>8}  {headline}")
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")


def show_stored() -> int:
    """Print every programme that already has a timeline."""
    with SessionLocal() as db:
        rows = list(db.scalars(select(VideoTranscript)))
        if not rows:
            print("No programmes analysed yet.")
            print("Try:  python analyse_video.py <YouTube URL>")
            return 0
        print(f"{len(rows)} programme(s) stored:\n")
        for transcript in rows:
            stream = db.get(ChatStream, transcript.stream_id)
            title = (stream.title if stream else None) or "(untitled)"
            print(f"  {stream.video_id if stream else '?':14} {title[:52]}")
            print(
                f"  {'':14} {transcript.duration_ms / 60000:.0f} min, "
                f"{len(transcript.segments)} stories, "
                f"{transcript.cue_count:,} cues, source {transcript.source}"
            )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "source", nargs="?", help="YouTube URL or 11-character video id"
    )
    parser.add_argument(
        "--no-frames",
        action="store_true",
        help="Skip capturing a still frame per story.",
    )
    parser.add_argument(
        "--show", type=int, default=12, help="How many stories to print (default 12)."
    )
    parser.add_argument(
        "--list", action="store_true", help="List programmes already analysed."
    )
    args = parser.parse_args(argv)

    Base.metadata.create_all(engine)

    if args.list:
        return show_stored()
    if not args.source:
        parser.print_help()
        return 2

    status = provider_status()
    print(f"transcript backend : {status['active']}", end="")
    print(f"  ({status['note']})" if status.get("note") else "")
    print(f"headline writer    : {llm_status()}")

    began = time.perf_counter()
    try:
        with SessionLocal() as db:
            result = analyse_video(
                db,
                args.source,
                with_frames=not args.no_frames,
                progress=report_progress,
            )
    except TranscriptUnavailable as exc:
        print(f"\nNo transcript available: {exc}", file=sys.stderr)
        return 1
    except CollectorError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    elapsed = time.perf_counter() - began
    print(f"\nanalysed {result.video_id} in {elapsed:.0f}s")
    print(f"  transcript : {result.source}")
    print(f"  length     : {result.duration_ms / 60000:.0f} min, {result.cue_count:,} cues")
    print(f"  stories    : {result.segment_count}")
    if result.headlines_cached or result.headlines_written:
        print(
            f"  headlines  : {result.headlines_written} written by the model, "
            f"{result.headlines_cached} reused from data/enrichments"
        )
    print(f"  frames     : {result.frames_captured}")
    if result.frame_note:
        print(f"  frame note : {result.frame_note}")

    with SessionLocal() as db:
        rows, _total = list_segments(db, video_id=result.video_id, limit=args.show)
        if rows:
            print(f"\nfirst {len(rows)} stories:")
            for row in rows:
                reasons = "+".join(row.boundary_reasons) or "-"
                print(f"  {row.timecode:>8}  [{row.topic:13}] {row.headline[:52]}")
                print(f"  {'':8}  {row.youtube_url}")
                print(f"  {'':8}  boundary: {reasons}")

    print("\nOpen the timeline at:  http://localhost:5173/timeline")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
