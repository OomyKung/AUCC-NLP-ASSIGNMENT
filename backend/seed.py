"""Populate the database so a fresh clone has something to show.

Usage::

    python seed.py                 # sample articles + every committed snapshot
    python seed.py --articles-only # skip the chat snapshots (much faster)
    python seed.py --reset         # drop and recreate the tables first
    python seed.py --limit 2000    # cap messages per snapshot

Everything it loads is committed to the repository, so this needs no network
access. Re-running is safe: articles deduplicate on a content hash and chat
imports deduplicate on message id.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from sqlalchemy import func, select  # noqa: E402

from app.config import settings  # noqa: E402
from app.database.base import Base  # noqa: E402
from app.database.session import SessionLocal, engine, init_db  # noqa: E402
from app.models.chat import ChatMessage, ChatStream  # noqa: E402
from app.models.news import NewsArticle  # noqa: E402
from app.nlp.registry import get_components, reset_components  # noqa: E402
from app.services.chat_store import store_messages  # noqa: E402
from app.services.collectors import CollectorError, get_collector  # noqa: E402
from app.services.ingest import analyse_stream, analyse_text, fit_keyword_idf  # noqa: E402
from app.services.pipeline import get_pipeline  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_CSV = ROOT / "data" / "sample_news.csv"


def seed_articles(pipeline) -> int:
    """Analyse and store the sample articles from ``data/sample_news.csv``."""
    if not SAMPLE_CSV.is_file():
        print(f"  skipped: {SAMPLE_CSV.name} not found (run build_dataset.py)")
        return 0

    with open(SAMPLE_CSV, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    stored = 0
    with SessionLocal() as db:
        for row in rows:
            published = None
            raw = (row.get("published_at") or "").strip()
            if raw:
                try:
                    published = datetime.fromisoformat(raw)
                    if published.tzinfo is None:
                        published = published.replace(tzinfo=UTC)
                except ValueError:
                    published = None

            analyse_text(
                db,
                title=row["title"],
                content=row["content"],
                source=row.get("source") or "ชุดข้อมูลตัวอย่าง",
                url=(row.get("url") or "").strip() or None,
                published_at=published,
                pipeline=pipeline,
            )
            stored += 1
    return stored


def seed_snapshots(pipeline, *, limit: int | None, analyse: bool) -> tuple[int, int, int]:
    """Import every committed chat snapshot and analyse it into windows."""
    snapshots = sorted(settings.chat_snapshot_dir.glob("*.jsonl"))
    if not snapshots:
        print(f"  no snapshots in {settings.chat_snapshot_dir}")
        return 0, 0, 0

    collector = get_collector("file")
    streams = messages = windows = 0

    for path in snapshots:
        try:
            result = collector.collect(path.name, limit=limit)
        except CollectorError as exc:
            print(f"  {path.name}: FAILED - {exc}")
            continue

        with SessionLocal() as db:
            stored = store_messages(db, result)
            streams += 1
            messages += stored.stored
            print(
                f"  {path.name:24} {stored.stored:>6} new, "
                f"{stored.skipped_duplicates:>6} duplicate"
            )

            if analyse:
                report = analyse_stream(db, stored.stream, pipeline)
                windows += report.windows_created

    return streams, messages, windows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Drop and recreate every table before seeding.",
    )
    parser.add_argument(
        "--articles-only",
        action="store_true",
        help="Load only the sample articles, skipping the chat snapshots.",
    )
    parser.add_argument(
        "--no-analyse",
        action="store_true",
        help="Import chat without running the NLP pipeline over it.",
    )
    parser.add_argument(
        "--no-broadcast",
        action="store_true",
        help="Skip rebuilding story timelines from transcript snapshots.",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Max messages per snapshot."
    )
    args = parser.parse_args(argv)

    started = time.perf_counter()

    if args.reset:
        print("Dropping existing tables...")
        Base.metadata.drop_all(bind=engine)

    init_db()
    reset_components()

    components = get_components()
    print("Active NLP backends:")
    for stage, status in components.status.items():
        suffix = "" if status.trained else "  (baseline)"
        print(f"  {stage:15} {status.active}{suffix}")
        if status.note:
            print(f"  {'':15} note: {status.note}")

    pipeline = get_pipeline()

    print("\nSeeding sample articles...")
    articles = seed_articles(pipeline)
    print(f"  {articles} article(s) analysed and stored")

    streams = messages = windows = 0
    if not args.articles_only:
        print("\nImporting committed chat snapshots (offline)...")
        # Corpus IDF first, so keyword scores use real statistics.
        with SessionLocal() as db:
            if db.scalar(select(func.count()).select_from(ChatMessage)):
                fit_keyword_idf(db, pipeline)

        streams, messages, windows = seed_snapshots(
            pipeline, limit=args.limit, analyse=not args.no_analyse
        )

        # Refit IDF now that the full corpus is present, then re-analyse so the
        # keyword scores reflect the whole corpus rather than a partial one.
        if messages and not args.no_analyse:
            print("\nRefitting keyword IDF over the full corpus...")
            with SessionLocal() as db:
                documents = fit_keyword_idf(db, pipeline)
                print(f"  IDF fitted on {documents} windows")
                for stream in db.scalars(select(ChatStream).order_by(ChatStream.id)):
                    analyse_stream(db, stream, pipeline)

    # ------------------------------------------------------- spoken content
    # Rebuild the story timelines from committed transcript snapshots. No
    # network: the transcripts are on disk and the frames are already cached,
    # which is the whole point -- a demo must not depend on YouTube being up.
    if not args.no_broadcast:
        snapshots = sorted((settings.data_dir / "transcripts").glob("*.json"))
        if snapshots:
            print()
            print(f"Rebuilding story timelines from {len(snapshots)} transcript(s)...")
            from app.services.broadcast import analyse_video

            for snapshot in snapshots:
                video_id = snapshot.stem
                try:
                    with SessionLocal() as db:
                        result = analyse_video(db, video_id, with_frames=True)
                    written = result.headlines_written
                    note = ""
                    if result.headlines_cached:
                        note = f", {result.headlines_cached} cached headlines"
                    if written:
                        # A model writes one in ~20 seconds, so say so rather
                        # than letting the seed look hung.
                        note += f", {written} headlines written by the LLM"
                    print(
                        f"  {video_id:14} {result.segment_count:>3} stories, "
                        f"{result.frames_captured:>3} frames{note}"
                    )
                except Exception as exc:  # noqa: BLE001 - one bad file must not stop the seed
                    print(f"  {video_id:14} skipped: {type(exc).__name__}: {exc}")

    with SessionLocal() as db:
        totals = {
            "documents": db.scalar(select(func.count()).select_from(NewsArticle)) or 0,
            "chat messages": db.scalar(select(func.count()).select_from(ChatMessage))
            or 0,
            "streams": db.scalar(select(func.count()).select_from(ChatStream)) or 0,
        }

    print(f"\n{'=' * 60}\nDatabase ready in {time.perf_counter() - started:.0f}s")
    for label, value in totals.items():
        print(f"  {label:16} {value:>8,}")
    print(f"{'=' * 60}")
    print("\nStart the API with:  uvicorn app.main:app --reload")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
