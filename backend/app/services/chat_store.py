"""Persist collected chat to the database.

Kept separate from the NLP pipeline so that raw collection is useful (and
testable) on its own. Storage is idempotent: re-importing the same stream adds
only messages that are genuinely new, which matters because live monitoring
re-polls the same video repeatedly.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.chat import ChatMessage, ChatStream
from app.services.collectors.base import CollectResult, RawChatMessage


@dataclass(slots=True)
class StoreResult:
    """Outcome of a store operation."""

    stream: ChatStream
    stored: int
    skipped_duplicates: int

    @property
    def total_in_stream(self) -> int:
        return self.stream.message_count


def _fingerprint(message: RawChatMessage) -> str:
    """Stable identity for a message that YouTube gave no id.

    Uses author + text + timestamp so that genuinely repeated messages from the
    same person at the same instant collapse, while an identical phrase sent
    later is still kept.
    """
    payload = f"{message.author or ''}|{message.text}|{message.published_at.isoformat()}"
    return "fp_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def get_or_create_stream(db: Session, result: CollectResult) -> ChatStream:
    """Fetch the stream row for this video, creating it if new.

    Metadata is refreshed on re-import so a stream that has since ended stops
    being reported as live.
    """
    info = result.stream
    stream = db.scalar(select(ChatStream).where(ChatStream.video_id == info.video_id))

    if stream is None:
        stream = ChatStream(
            video_id=info.video_id,
            url=info.url,
            title=info.title,
            channel=info.channel,
            collector=info.collector,
            is_live=info.is_live,
        )
        db.add(stream)
        db.flush()  # assign the primary key
        return stream

    stream.url = info.url or stream.url
    stream.title = info.title or stream.title
    stream.channel = info.channel or stream.channel
    stream.collector = info.collector
    stream.is_live = info.is_live
    return stream


def store_messages(db: Session, result: CollectResult) -> StoreResult:
    """Store a collect result, skipping messages already present.

    The stream's counters and time bounds are recomputed from the database
    afterwards, so they stay correct across repeated partial imports.
    """
    stream = get_or_create_stream(db, result)

    existing_ids = set(
        db.scalars(
            select(ChatMessage.message_id).where(ChatMessage.stream_id == stream.id)
        ).all()
    )

    stored = 0
    duplicates = 0
    seen_in_batch: set[str] = set()

    for message in result.messages:
        identity = message.message_id or _fingerprint(message)
        if identity in existing_ids or identity in seen_in_batch:
            duplicates += 1
            continue
        seen_in_batch.add(identity)

        db.add(
            ChatMessage(
                stream_id=stream.id,
                message_id=identity,
                author=message.author,
                text=message.text,
                published_at=message.published_at,
                offset_ms=message.offset_ms,
            )
        )
        stored += 1

    db.flush()
    _refresh_stream_counters(db, stream)
    db.commit()

    return StoreResult(stream=stream, stored=stored, skipped_duplicates=duplicates)


def _refresh_stream_counters(db: Session, stream: ChatStream) -> None:
    """Recompute message count and time bounds from stored rows."""
    row = db.execute(
        select(
            func.count(ChatMessage.id),
            func.min(ChatMessage.published_at),
            func.max(ChatMessage.published_at),
        ).where(ChatMessage.stream_id == stream.id)
    ).one()

    stream.message_count = int(row[0] or 0)
    stream.first_message_at = row[1]
    stream.last_message_at = row[2]
