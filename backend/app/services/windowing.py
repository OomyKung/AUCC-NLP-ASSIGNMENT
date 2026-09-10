"""Group chat messages into analysable windows.

A single live-chat message is a handful of words -- far too little to classify a
topic or summarise. Consecutive messages are therefore grouped into *windows*,
and a window is what the NLP pipeline treats as one document.

A window closes at whichever bound is reached first:

* ``CHAT_WINDOW_SIZE`` messages (default 200), or
* ``CHAT_WINDOW_MINUTES`` minutes of stream time (default 60, a safety valve
  so a quiet stream cannot produce one enormous window).

Windows below ``CHAT_WINDOW_MIN_MESSAGES`` are dropped as too thin to analyse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from app.config import settings
from app.services.collectors.base import RawChatMessage


@dataclass(slots=True)
class ChatWindow:
    """One group of consecutive chat messages."""

    index: int
    messages: list[RawChatMessage] = field(default_factory=list)

    @property
    def start(self) -> datetime:
        return self.messages[0].published_at

    @property
    def end(self) -> datetime:
        return self.messages[-1].published_at

    @property
    def count(self) -> int:
        return len(self.messages)

    def as_document(self, separator: str = "\n") -> str:
        """Join the messages into the text the NLP pipeline analyses."""
        return separator.join(message.text for message in self.messages)

    def duration_seconds(self) -> float:
        return max((self.end - self.start).total_seconds(), 0.0)


def build_windows(
    messages: list[RawChatMessage],
    *,
    size: int | None = None,
    minutes: int | None = None,
    min_messages: int | None = None,
) -> list[ChatWindow]:
    """Split messages into windows, oldest first.

    Args:
        messages: Messages to group. Sorted by time internally, so callers need
            not pre-sort.
        size: Maximum messages per window. Defaults to ``CHAT_WINDOW_SIZE``.
        minutes: Maximum window span in minutes. Defaults to
            ``CHAT_WINDOW_MINUTES``. Pass ``0`` to disable the time bound.
        min_messages: Windows smaller than this are discarded. The final window
            is subject to the same rule, so a thin tail is not analysed.

    Returns:
        Windows in chronological order, re-indexed from 0 after filtering.
    """
    max_size = size if size is not None else settings.chat_window_size
    max_minutes = minutes if minutes is not None else settings.chat_window_minutes
    floor = (
        min_messages if min_messages is not None else settings.chat_window_min_messages
    )

    if max_size < 1:
        raise ValueError("Window size must be at least 1 message.")

    if not messages:
        return []

    ordered = sorted(messages, key=lambda m: m.published_at)
    span = timedelta(minutes=max_minutes) if max_minutes and max_minutes > 0 else None

    windows: list[ChatWindow] = []
    current: list[RawChatMessage] = []
    window_started_at: datetime | None = None

    for message in ordered:
        if not current:
            current = [message]
            window_started_at = message.published_at
            continue

        # Time bound: does adding this message stretch the window too far?
        too_long = (
            span is not None
            and window_started_at is not None
            and message.published_at - window_started_at >= span
        )
        if too_long or len(current) >= max_size:
            windows.append(ChatWindow(index=len(windows), messages=current))
            current = [message]
            window_started_at = message.published_at
            continue

        current.append(message)

    if current:
        windows.append(ChatWindow(index=len(windows), messages=current))

    # Drop thin windows, then re-index so indexes stay contiguous.
    kept = [w for w in windows if w.count >= floor]
    for new_index, window in enumerate(kept):
        window.index = new_index
    return kept


# Livestream titles are full of decoration that carries no information and,
# repeated on every window, makes them all look identical in the dashboard.
_TITLE_NOISE = re.compile(
    r"(?:^|\s)(?:🔴|⭕|🟢|▶️?|LIVE\s*!*\s*:?|สด\s*:|ถ่ายทอดสด\s*:)+",
    re.IGNORECASE,
)


def _tidy_stream_title(title: str | None, max_length: int = 48) -> str:
    """Strip livestream decoration and shorten to a usable prefix."""
    if not title:
        return "แชทสด"

    cleaned = _TITLE_NOISE.sub(" ", title)
    # Quotes routinely end up unbalanced once the title is truncated.
    cleaned = cleaned.replace('"', "").replace("“", "").replace("”", "")
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" :|-–—#'")
    if not cleaned:
        return "แชทสด"
    if len(cleaned) > max_length:
        cleaned = cleaned[:max_length].rstrip() + "…"
    return cleaned


def synthesise_title(
    window: ChatWindow,
    *,
    stream_title: str | None = None,
    keywords: list[str] | None = None,
    max_length: int = 160,
) -> str:
    """Build a readable headline for a chat window.

    Leads with the window's top keywords once the pipeline has produced them,
    because that is what actually distinguishes one window from another. Falls
    back to the tidied stream title plus the window's clock time, so a window is
    never nameless in the dashboard.
    """
    stamp = f"{window.start:%d %b %H:%M}"
    base = _tidy_stream_title(stream_title)

    if keywords:
        topic_hint = " · ".join(keywords[:4])
        title = f"{topic_hint} — {base} ({stamp})"
    else:
        title = f"{base} — ช่วง {stamp} ({window.count} ข้อความ)"

    if len(title) > max_length:
        title = title[: max_length - 1].rstrip() + "…"
    return title
