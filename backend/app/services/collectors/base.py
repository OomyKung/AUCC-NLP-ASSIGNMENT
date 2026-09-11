"""Collector contract shared by every chat data source.

A collector turns a *source* (a YouTube URL, a video id, or a local snapshot
path) into a :class:`CollectResult`. Everything downstream -- windowing, the
NLP pipeline, the API -- depends only on this contract, so a new data source
means writing one class and registering it in ``__init__.py``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable


class CollectorError(RuntimeError):
    """Raised when a collector cannot retrieve chat.

    Carries a message intended to be shown to the user, so API handlers can
    surface the real reason (private video, chat disabled, no network) instead
    of a generic 500.
    """



@dataclass(slots=True)
class RawChatMessage:
    """One chat message, normalised across collectors."""

    text: str
    published_at: datetime
    message_id: str | None = None
    author: str | None = None
    # Milliseconds from stream start; present on replays, absent when live.
    offset_ms: int | None = None


@dataclass(slots=True)
class StreamInfo:
    """Metadata about the video the chat belongs to."""

    video_id: str
    url: str
    title: str | None = None
    channel: str | None = None
    is_live: bool = False
    collector: str = "unknown"


class ChatUnavailable(CollectorError):
    """The video is fine; it just has no chat to collect.

    Distinguished from every other collection failure because it is not really a
    failure of the *video*: news channels routinely turn chat replay off once a
    broadcast ends, and the programme is still fully analysable from what the
    newsreader said. The caller can fall back to the transcript.

    Carries the :class:`StreamInfo` already fetched, because the metadata
    request succeeded -- the title and channel are known, and throwing them away
    would leave the fallback analysing a nameless video.
    """

    def __init__(self, message: str, stream: StreamInfo | None = None) -> None:
        super().__init__(message)
        self.stream = stream


@dataclass(slots=True)
class CollectResult:
    """What a collector returns."""

    stream: StreamInfo
    messages: list[RawChatMessage] = field(default_factory=list)

    @property
    def count(self) -> int:
        return len(self.messages)


@runtime_checkable
class ChatCollector(Protocol):
    """Any chat data source."""

    name: str

    def collect(self, source: str, *, limit: int | None = None) -> CollectResult:
        """Fetch chat messages for ``source``, newest-last.

        Raises:
            CollectorError: when chat cannot be retrieved.
        """
        ...


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------

# Matches the id in watch?v=, youtu.be/, /live/, /embed/, /shorts/ URLs.
_VIDEO_ID_PATTERNS = (
    re.compile(r"(?:v=|/live/|/embed/|/shorts/|youtu\.be/)([0-9A-Za-z_-]{11})"),
    re.compile(r"^([0-9A-Za-z_-]{11})$"),
)


def extract_video_id(source: str) -> str:
    """Pull the 11-character video id out of a URL, or accept a bare id.

    Raises:
        CollectorError: when ``source`` contains no recognisable video id.
    """
    candidate = (source or "").strip()
    for pattern in _VIDEO_ID_PATTERNS:
        match = pattern.search(candidate)
        if match:
            return match.group(1)
    raise CollectorError(
        f"Could not find a YouTube video id in {source!r}. "
        "Expected a watch/live/youtu.be URL or an 11-character video id."
    )


def watch_url(video_id: str) -> str:
    """Canonical watch URL for a video id."""
    return f"https://www.youtube.com/watch?v={video_id}"


def usec_to_datetime(value: object) -> datetime:
    """Convert YouTube's microsecond epoch timestamp to an aware datetime.

    Falls back to *now* when the field is missing or malformed, so one bad
    record cannot abort an import of thousands of messages.
    """
    try:
        return datetime.fromtimestamp(int(value) / 1_000_000, tz=UTC)  # type: ignore[arg-type]
    except (TypeError, ValueError, OSError, OverflowError):
        return datetime.now(UTC)
