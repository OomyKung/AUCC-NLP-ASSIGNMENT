"""Chat collector that replays a snapshot from disk.

This is what guarantees the project demonstrates with no network at all: a
snapshot of real collected chat is committed under ``data/chat_snapshots/`` and
replayed from there.

Two formats are accepted:

* **Normalised snapshot** (what :mod:`app.services.snapshot` writes) -- JSON
  Lines whose first record is a ``_stream`` header followed by one message per
  line.
* **Raw yt-dlp** ``*.live_chat.json`` -- passed straight to the InnerTube parser,
  so a file saved by ``yt-dlp`` by hand also works.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from app.config import settings
from app.services.collectors.base import (
    CollectorError,
    CollectResult,
    RawChatMessage,
    StreamInfo,
)
from app.services.collectors.ytchat_parser import parse_chat_file

# Keys that identify a raw YouTube chat record rather than our own format.
_RAW_MARKERS = ("replayChatItemAction", "addChatItemAction")


class FileCollector:
    """Replay chat from a local snapshot file."""

    name = "file"

    def _resolve(self, source: str) -> Path:
        """Locate the snapshot, accepting an absolute path or a bare filename."""
        if not source or not source.strip():
            raise CollectorError("No snapshot file given.")

        candidate = Path(source.strip()).expanduser()
        searched = [candidate]
        if not candidate.is_absolute():
            for base in (settings.chat_snapshot_dir, settings.data_dir):
                for name in (candidate.name, f"{candidate.name}.jsonl"):
                    searched.append(base / name)

        for path in searched:
            if path.is_file():
                return path

        available = sorted(p.name for p in settings.chat_snapshot_dir.glob("*.jsonl"))
        hint = f" Available snapshots: {', '.join(available)}" if available else ""
        raise CollectorError(f"Snapshot {source!r} not found.{hint}")

    def collect(self, source: str, *, limit: int | None = None) -> CollectResult:
        """Read messages from a snapshot file."""
        path = self._resolve(source)
        limit = limit or settings.max_messages_per_import

        with open(path, encoding="utf-8") as handle:
            first_line = handle.readline()

        if any(marker in first_line for marker in _RAW_MARKERS):
            return self._collect_raw(path, limit=limit)
        return self._collect_normalised(path, limit=limit)

    # ------------------------------------------------------------------ formats
    def _collect_raw(self, path: Path, *, limit: int) -> CollectResult:
        """Parse a raw yt-dlp live_chat.json file."""
        messages = parse_chat_file(path, limit=limit)
        if not messages:
            raise CollectorError(f"No readable chat messages in {path.name}.")

        # A raw file carries no metadata, so derive an id from the filename
        # (yt-dlp names files "<video_id>.live_chat.json").
        video_id = path.name.split(".")[0][:32] or path.stem
        messages.sort(key=lambda m: m.published_at)
        return CollectResult(
            stream=StreamInfo(
                video_id=video_id,
                url=f"https://www.youtube.com/watch?v={video_id}",
                title=path.stem,
                collector=self.name,
            ),
            messages=messages,
        )

    def _collect_normalised(self, path: Path, *, limit: int) -> CollectResult:
        """Parse a snapshot written by :mod:`app.services.snapshot`."""
        stream: StreamInfo | None = None
        messages: list[RawChatMessage] = []

        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(record, dict):
                    continue

                header = record.get("_stream")
                if isinstance(header, dict):
                    stream = StreamInfo(
                        video_id=str(header.get("video_id") or path.stem)[:32],
                        url=str(header.get("url") or ""),
                        title=header.get("title"),
                        channel=header.get("channel"),
                        is_live=bool(header.get("is_live")),
                        collector=self.name,
                    )
                    continue

                message = _message_from_record(record)
                if message is not None:
                    messages.append(message)
                    if len(messages) >= limit:
                        break

        if not messages:
            raise CollectorError(f"No readable chat messages in {path.name}.")

        if stream is None:
            stream = StreamInfo(
                video_id=path.stem[:32],
                url="",
                title=path.stem,
                collector=self.name,
            )

        messages.sort(key=lambda m: m.published_at)
        return CollectResult(stream=stream, messages=messages)


def _message_from_record(record: dict) -> RawChatMessage | None:
    """Build a message from a normalised snapshot record."""
    text = str(record.get("text") or "").strip()
    if not text:
        return None

    raw_timestamp = record.get("published_at")
    try:
        published_at = datetime.fromisoformat(str(raw_timestamp))
        if published_at.tzinfo is None:
            published_at = published_at.replace(tzinfo=UTC)
    except (TypeError, ValueError):
        published_at = datetime.now(UTC)

    offset = record.get("offset_ms")
    return RawChatMessage(
        text=text,
        published_at=published_at,
        message_id=record.get("message_id"),
        author=record.get("author"),
        offset_ms=int(offset) if isinstance(offset, (int, float)) else None,
    )
