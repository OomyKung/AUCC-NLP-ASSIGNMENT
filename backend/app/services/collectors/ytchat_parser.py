"""Parser for YouTube's live-chat JSON, as written by yt-dlp.

yt-dlp saves chat as JSON Lines, one YouTube "action" per line, using
YouTube's internal (InnerTube) renderer shapes. Those shapes are deeply
nested and vary between live and replay captures, so all of that knowledge is
isolated here: the collectors just call :func:`parse_chat_line`.

Reference shape of a replay line::

    {"replayChatItemAction": {
        "actions": [{"addChatItemAction": {"item": {
            "liveChatTextMessageRenderer": {
                "message": {"runs": [{"text": "สวัสดีครับ"}]},
                "authorName": {"simpleText": "somebody"},
                "timestampUsec": "1725900000000000",
                "id": "ChwKGkNK..."}}}}],
        "videoOffsetTimeMsec": "132000"}}
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

from app.services.collectors.base import RawChatMessage, usec_to_datetime

# Renderers that carry user-authored text we want to analyse.
_TEXT_RENDERERS = (
    "liveChatTextMessageRenderer",
    "liveChatPaidMessageRenderer",  # Super Chat: has a message plus an amount
)

# Renderers whose text lives in a different field.
_SUBTEXT_RENDERERS = (
    "liveChatMembershipItemRenderer",  # membership announcements
)


def _runs_to_text(message: dict[str, Any] | None) -> str:
    """Flatten a YouTube ``message`` object into plain text.

    A message is a list of "runs": plain text, unicode emoji, or custom channel
    emotes. Unicode emoji are kept as characters; custom emotes are rendered as
    their ``:shortcut:`` so that a message made only of emotes is still visible
    to the spam filter rather than silently becoming empty.
    """
    if not message:
        return ""

    if "simpleText" in message:
        return str(message["simpleText"])

    parts: list[str] = []
    for run in message.get("runs", []) or []:
        if not isinstance(run, dict):
            continue
        if "text" in run:
            parts.append(str(run["text"]))
            continue

        emoji = run.get("emoji")
        if not isinstance(emoji, dict):
            continue
        if emoji.get("isCustomEmoji"):
            shortcuts = emoji.get("shortcuts") or []
            if shortcuts:
                parts.append(str(shortcuts[0]))
        else:
            # For standard emoji, emojiId is the character itself.
            parts.append(str(emoji.get("emojiId") or ""))

    return "".join(parts)


def _renderer_to_message(
    renderer: dict[str, Any],
    *,
    text: str,
    offset_ms: int | None,
) -> RawChatMessage | None:
    """Build a :class:`RawChatMessage` from a renderer, or ``None`` if empty."""
    if not text.strip():
        return None

    author = None
    author_name = renderer.get("authorName")
    if isinstance(author_name, dict):
        author = author_name.get("simpleText")

    return RawChatMessage(
        text=text,
        published_at=usec_to_datetime(renderer.get("timestampUsec")),
        message_id=renderer.get("id"),
        author=str(author) if author else None,
        offset_ms=offset_ms,
    )


def _coerce_offset(value: object) -> int | None:
    """Parse ``videoOffsetTimeMsec``, tolerating missing or odd values."""
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _iter_chat_actions(payload: dict[str, Any]) -> Iterator[tuple[dict[str, Any], int | None]]:
    """Yield ``(action, offset_ms)`` pairs from one parsed JSON line.

    Handles both the replay wrapper and live captures where the action sits at
    the top level.
    """
    replay = payload.get("replayChatItemAction")
    if isinstance(replay, dict):
        offset = _coerce_offset(
            replay.get("videoOffsetTimeMsec") or payload.get("videoOffsetTimeMsec")
        )
        for action in replay.get("actions", []) or []:
            if isinstance(action, dict):
                yield action, offset
        return

    # Live capture: the action itself is the payload.
    offset = _coerce_offset(payload.get("videoOffsetTimeMsec"))
    yield payload, offset


def parse_chat_payload(payload: dict[str, Any]) -> list[RawChatMessage]:
    """Extract every analysable message from one parsed chat JSON object."""
    messages: list[RawChatMessage] = []

    for action, offset_ms in _iter_chat_actions(payload):
        add_item = action.get("addChatItemAction")
        if not isinstance(add_item, dict):
            # Ticker items, deletions, banners and mode changes carry no text.
            continue

        item = add_item.get("item")
        if not isinstance(item, dict):
            continue

        for key, renderer in item.items():
            if not isinstance(renderer, dict):
                continue

            if key in _TEXT_RENDERERS:
                text = _runs_to_text(renderer.get("message"))
            elif key in _SUBTEXT_RENDERERS:
                text = _runs_to_text(renderer.get("headerSubtext"))
            else:
                # Stickers, gift purchases, placeholders: nothing to analyse.
                continue

            message = _renderer_to_message(renderer, text=text, offset_ms=offset_ms)
            if message is not None:
                messages.append(message)

    return messages


def parse_chat_line(line: str) -> list[RawChatMessage]:
    """Parse a single JSON Lines record. Malformed lines yield nothing."""
    line = line.strip()
    if not line:
        return []
    try:
        payload = json.loads(line)
    except json.JSONDecodeError:
        return []
    if not isinstance(payload, dict):
        return []
    return parse_chat_payload(payload)


def parse_chat_file(path, *, limit: int | None = None) -> list[RawChatMessage]:
    """Parse a yt-dlp ``*.live_chat.json`` file (JSON Lines).

    Args:
        path: File to read.
        limit: Stop after this many messages, for bounded imports.
    """
    messages: list[RawChatMessage] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            messages.extend(parse_chat_line(line))
            if limit is not None and len(messages) >= limit:
                return messages[:limit]
    return messages
