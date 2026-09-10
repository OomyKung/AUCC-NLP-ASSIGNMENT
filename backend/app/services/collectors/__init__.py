"""Chat collector registry.

Adding a data source means writing a class that satisfies
:class:`~app.services.collectors.base.ChatCollector` and registering it here.
Nothing else in the codebase needs to change.
"""

from __future__ import annotations

from app.config import settings
from app.services.collectors.base import (
    ChatCollector,
    CollectorError,
    CollectResult,
    RawChatMessage,
    StreamInfo,
    extract_video_id,
)
from app.services.collectors.file import FileCollector
from app.services.collectors.youtube_api import YouTubeApiCollector
from app.services.collectors.ytdlp import YtDlpCollector

_COLLECTORS: dict[str, type] = {
    YtDlpCollector.name: YtDlpCollector,
    YouTubeApiCollector.name: YouTubeApiCollector,
    FileCollector.name: FileCollector,
}

AVAILABLE_COLLECTORS: tuple[str, ...] = tuple(_COLLECTORS)


def get_collector(name: str | None = None) -> ChatCollector:
    """Build a collector by name, defaulting to the configured one.

    Raises:
        CollectorError: when ``name`` is not a known collector.
    """
    key = (name or settings.collector).strip().lower()
    collector_class = _COLLECTORS.get(key)
    if collector_class is None:
        raise CollectorError(
            f"Unknown collector {name!r}. Available: {', '.join(AVAILABLE_COLLECTORS)}."
        )
    return collector_class()  # type: ignore[return-value]


__all__ = [
    "AVAILABLE_COLLECTORS",
    "ChatCollector",
    "CollectResult",
    "CollectorError",
    "FileCollector",
    "RawChatMessage",
    "StreamInfo",
    "YouTubeApiCollector",
    "YtDlpCollector",
    "extract_video_id",
    "get_collector",
]
