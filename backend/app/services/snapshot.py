"""Read/write normalised chat snapshots.

A snapshot is JSON Lines: a ``_stream`` header record followed by one message
per line. Committing snapshots under ``data/chat_snapshots/`` is what lets the
dashboard be demonstrated, and its dataset rebuilt, with no network access.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import settings
from app.services.collectors.base import CollectResult


def snapshot_path(video_id: str, directory: Path | None = None) -> Path:
    """Conventional snapshot path for a video id."""
    base = directory or settings.chat_snapshot_dir
    return base / f"{video_id}.jsonl"


class SnapshotWouldShrink(RuntimeError):
    """Raised when writing a snapshot would discard already-collected messages."""


def snapshot_message_count(path: Path) -> int | None:
    """Messages recorded in an existing snapshot's header, if readable."""
    if not path.is_file():
        return None
    try:
        with open(path, encoding="utf-8") as handle:
            header = json.loads(handle.readline() or "{}")
    except (OSError, json.JSONDecodeError):
        return None
    stream = header.get("_stream")
    if isinstance(stream, dict):
        count = stream.get("message_count")
        if isinstance(count, int):
            return count
    return None


def write_snapshot(
    result: CollectResult, path: Path | None = None, *, allow_shrink: bool = False
) -> Path:
    """Write a collect result to a snapshot file and return its path.

    Refuses to replace an existing snapshot with a smaller one unless
    ``allow_shrink`` is set. This guards against real data loss: re-importing the
    same video with a lower ``--limit`` would otherwise silently overwrite a
    large committed snapshot with a truncated one, discarding messages that can
    no longer be re-collected once the stream's chat replay is gone. This
    actually happened during development, costing 4,000 collected messages.

    Raises:
        SnapshotWouldShrink: when the new snapshot has fewer messages.
    """
    target = path or snapshot_path(result.stream.video_id)
    target.parent.mkdir(parents=True, exist_ok=True)

    if not allow_shrink:
        existing = snapshot_message_count(target)
        if existing is not None and result.count < existing:
            raise SnapshotWouldShrink(
                f"Refusing to overwrite {target.name}: it already holds {existing} "
                f"messages and this import collected only {result.count}. "
                "Collect with a higher limit, or pass allow_shrink=True to replace it."
            )

    stream = result.stream
    header = {
        "_stream": {
            "video_id": stream.video_id,
            "url": stream.url,
            "title": stream.title,
            "channel": stream.channel,
            "is_live": stream.is_live,
            "collector": stream.collector,
            "message_count": result.count,
        }
    }

    with open(target, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(header, ensure_ascii=False) + "\n")
        for message in result.messages:
            record = {
                "message_id": message.message_id,
                "author": message.author,
                "text": message.text,
                "published_at": message.published_at.isoformat(),
                "offset_ms": message.offset_ms,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    return target


def list_snapshots(directory: Path | None = None) -> list[dict]:
    """Summarise available snapshots for the API and the import UI."""
    base = directory or settings.chat_snapshot_dir
    if not base.is_dir():
        return []

    summaries: list[dict] = []
    for path in sorted(base.glob("*.jsonl")):
        summary = {
            "file": path.name,
            "video_id": path.stem,
            "title": None,
            "channel": None,
            "message_count": None,
            "size_kb": round(path.stat().st_size / 1024, 1),
        }
        try:
            with open(path, encoding="utf-8") as handle:
                header = json.loads(handle.readline() or "{}")
            stream = header.get("_stream")
            if isinstance(stream, dict):
                summary["title"] = stream.get("title")
                summary["channel"] = stream.get("channel")
                summary["message_count"] = stream.get("message_count")
        except (OSError, json.JSONDecodeError):
            # A malformed snapshot should not hide the healthy ones.
            pass
        summaries.append(summary)
    return summaries
