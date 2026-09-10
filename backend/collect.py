"""Collect Thai YouTube live chat into the database and/or a snapshot file.

Examples::

    # Discover Thai news streams that have a chat replay
    python collect.py --search "ข่าว ไทยรัฐ live"

    # Collect one stream, store it, and save an offline snapshot
    python collect.py https://www.youtube.com/watch?v=VIDEO_ID --snapshot

    # Replay a committed snapshot with no network access
    python collect.py VIDEO_ID.jsonl --collector file

    # List the snapshots available offline
    python collect.py --list-snapshots
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

# Thai output must not be mangled by the Windows console codepage.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from app.config import settings  # noqa: E402
from app.database.session import SessionLocal, init_db  # noqa: E402
from app.services.chat_store import store_messages  # noqa: E402
from app.services.collectors import CollectorError, get_collector  # noqa: E402
from app.services.snapshot import (  # noqa: E402
    SnapshotWouldShrink,
    list_snapshots,
    write_snapshot,
)
from app.services.windowing import build_windows  # noqa: E402


def search_streams(query: str, limit: int) -> list[dict]:
    """Find candidate videos with yt-dlp's search, newest first.

    Only ``was_live``/``is_live`` results are useful, because a video that was
    never a livestream has no chat to collect.
    """
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--no-warnings",
        "--ignore-config",
        "--flat-playlist",
        "--dump-json",
        "--skip-download",
        f"ytsearch{limit}:{query}",
    ]
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    results: list[dict] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        results.append(
            {
                "id": payload.get("id"),
                "title": payload.get("title"),
                "channel": payload.get("channel") or payload.get("uploader"),
                "live_status": payload.get("live_status"),
            }
        )
    return results


def _print_search(results: list[dict]) -> None:
    if not results:
        print("No results. Try a different query.")
        return

    print(f"{'video id':<13} {'status':<10} {'channel':<24} title")
    print("-" * 96)
    for item in results:
        status = str(item["live_status"] or "-")
        marker = "*" if status in {"was_live", "is_live"} else " "
        channel = (item["channel"] or "")[:23]
        title = (item["title"] or "")[:44]
        print(f"{marker}{item['id'] or '?':<12} {status:<10} {channel:<24} {title}")
    print("\n* = livestream, so chat may be available.")


def _print_snapshots() -> None:
    snapshots = list_snapshots()
    if not snapshots:
        print(f"No snapshots in {settings.chat_snapshot_dir}")
        return

    print(f"{'file':<26} {'messages':>8}  {'size':>8}  title")
    print("-" * 92)
    for snap in snapshots:
        count = snap["message_count"] if snap["message_count"] is not None else "?"
        print(
            f"{snap['file']:<26} {str(count):>8}  {snap['size_kb']:>6.1f}kB  "
            f"{(snap['title'] or '')[:40]}"
        )


def collect_one(
    source: str,
    *,
    collector_name: str | None,
    limit: int | None,
    save_snapshot: bool,
    store: bool,
) -> int:
    """Collect a single source. Returns a process exit code."""
    collector = get_collector(collector_name)
    print(f"Collecting {source!r} with the {collector.name!r} collector...")

    try:
        result = collector.collect(source, limit=limit)
    except CollectorError as exc:
        print(f"FAILED: {exc}", file=sys.stderr)
        return 1

    stream = result.stream
    print(f"  channel : {stream.channel or '-'}")
    print(f"  title   : {(stream.title or '-')[:70]}")
    print(f"  live    : {stream.is_live}")
    print(f"  messages: {result.count}")

    if result.messages:
        span = (
            result.messages[-1].published_at - result.messages[0].published_at
        ).total_seconds() / 60
        rate = result.count / span if span > 0 else 0.0
        print(f"  span    : {span:.0f} min ({rate:.1f} messages/min)")

    windows = build_windows(result.messages)
    if windows:
        avg = sum(w.count for w in windows) / len(windows)
        print(f"  windows : {len(windows)} (avg {avg:.0f} messages each)")
    else:
        print(
            f"  windows : 0 -- fewer than "
            f"{settings.chat_window_min_messages} messages per window"
        )

    if save_snapshot:
        try:
            path = write_snapshot(result)
            print(f"  snapshot: {path.name} ({path.stat().st_size / 1024:.1f} kB)")
        except SnapshotWouldShrink as exc:
            print(f"  snapshot: SKIPPED - {exc}", file=sys.stderr)

    if store:
        init_db()
        with SessionLocal() as db:
            stored = store_messages(db, result)
        print(
            f"  stored  : {stored.stored} new, {stored.skipped_duplicates} duplicates "
            f"skipped ({stored.total_in_stream} total in stream)"
        )

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect Thai YouTube live chat.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "sources",
        nargs="*",
        help="YouTube URLs, video ids, or snapshot filenames.",
    )
    parser.add_argument(
        "--collector",
        choices=["ytdlp", "youtube_api", "file"],
        help=f"Data source (default from config: {settings.collector}).",
    )
    parser.add_argument("--limit", type=int, help="Maximum messages per source.")
    parser.add_argument(
        "--snapshot",
        action="store_true",
        help="Write an offline snapshot to data/chat_snapshots/.",
    )
    parser.add_argument(
        "--no-store",
        action="store_true",
        help="Collect without writing to the database.",
    )
    parser.add_argument("--search", metavar="QUERY", help="Search for streams and exit.")
    parser.add_argument(
        "--max", type=int, default=10, help="Number of search results (default 10)."
    )
    parser.add_argument(
        "--list-snapshots", action="store_true", help="List offline snapshots and exit."
    )

    args = parser.parse_args(argv)

    if args.list_snapshots:
        _print_snapshots()
        return 0

    if args.search:
        _print_search(search_streams(args.search, args.max))
        return 0

    if not args.sources:
        parser.error("Give at least one source, or use --search / --list-snapshots.")

    exit_code = 0
    for index, source in enumerate(args.sources):
        if index:
            print()
        exit_code |= collect_one(
            source,
            collector_name=args.collector,
            limit=args.limit,
            save_snapshot=args.snapshot,
            store=not args.no_store,
        )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
