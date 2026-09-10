"""Tests for persisting collected chat.

Idempotency is the property that matters most here: live monitoring re-polls
the same video repeatedly, so a second import of overlapping messages must not
duplicate rows.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import func, select

from app.models.chat import ChatMessage, ChatStream
from app.services.chat_store import store_messages
from app.services.collectors.base import CollectResult, RawChatMessage, StreamInfo


def test_store_creates_stream_and_messages(db_session, make_result):
    result = store_messages(db_session, make_result(25, video_id="storeTest01"))

    assert result.stored == 25
    assert result.skipped_duplicates == 0
    assert result.stream.video_id == "storeTest01"
    assert result.stream.message_count == 25
    assert db_session.scalar(select(func.count(ChatMessage.id))) == 25


def test_reimport_is_idempotent(db_session, make_result):
    payload = make_result(30, video_id="storeTest02")
    store_messages(db_session, payload)

    second = store_messages(db_session, payload)

    assert second.stored == 0
    assert second.skipped_duplicates == 30
    # Crucially, no duplicate rows and no double-counting.
    assert db_session.scalar(select(func.count(ChatMessage.id))) == 30
    assert second.stream.message_count == 30
    assert db_session.scalar(select(func.count(ChatStream.id))) == 1


def test_overlapping_import_stores_only_new_messages(db_session, make_messages):
    stream = StreamInfo(video_id="overlap01", url="u", collector="file")
    everything = make_messages(20)

    store_messages(db_session, CollectResult(stream=stream, messages=everything[:12]))
    second = store_messages(
        db_session, CollectResult(stream=stream, messages=everything[8:])
    )

    assert second.stored == 8  # messages 12..19
    assert second.skipped_duplicates == 4  # messages 8..11 already present
    assert second.stream.message_count == 20


def test_duplicates_within_one_batch_are_collapsed(db_session):
    message = RawChatMessage(
        text="ซ้ำ",
        published_at=datetime(2026, 9, 1, tzinfo=UTC),
        message_id="same-id",
    )
    result = CollectResult(
        stream=StreamInfo(video_id="batchdup01", url="u", collector="file"),
        messages=[message, message, message],
    )

    stored = store_messages(db_session, result)

    assert stored.stored == 1
    assert stored.skipped_duplicates == 2


def test_messages_without_an_id_are_fingerprinted(db_session):
    """Messages lacking a YouTube id still deduplicate by content."""
    timestamp = datetime(2026, 9, 1, 10, 30, tzinfo=UTC)
    stream = StreamInfo(video_id="nofingerpr1", url="u", collector="file")
    messages = [
        RawChatMessage(text="ไม่มีไอดี", published_at=timestamp, author="somchai"),
        # Identical author + text + timestamp: the same message.
        RawChatMessage(text="ไม่มีไอดี", published_at=timestamp, author="somchai"),
        # Same words from a different person is a genuinely different message.
        RawChatMessage(text="ไม่มีไอดี", published_at=timestamp, author="somsri"),
    ]

    stored = store_messages(db_session, CollectResult(stream=stream, messages=messages))

    assert stored.stored == 2
    assert stored.skipped_duplicates == 1


def test_stream_metadata_and_time_bounds_are_maintained(db_session, make_result):
    result = make_result(40, video_id="bounds00001", seconds_apart=30)

    stored = store_messages(db_session, result)
    stream = stored.stream

    assert stream.first_message_at == result.messages[0].published_at
    assert stream.last_message_at == result.messages[-1].published_at
    assert stream.channel == "ช่องทดสอบ"


def test_stream_stops_being_reported_as_live_after_it_ends(db_session, make_messages):
    messages = make_messages(5)
    live = CollectResult(
        stream=StreamInfo(video_id="wentoffair", url="u", is_live=True, collector="ytdlp"),
        messages=messages,
    )
    store_messages(db_session, live)

    ended = CollectResult(
        stream=StreamInfo(
            video_id="wentoffair",
            url="u",
            title="ชื่อรายการ",
            is_live=False,
            collector="ytdlp",
        ),
        messages=messages,
    )
    stored = store_messages(db_session, ended)

    assert stored.stream.is_live is False
    assert stored.stream.title == "ชื่อรายการ"


def test_thai_text_survives_the_database_round_trip(db_session):
    original = "รถยนต์ชนกันกลางสี่แยก มีผู้ได้รับบาดเจ็บ 5 ราย 🚑"
    result = CollectResult(
        stream=StreamInfo(video_id="thairoundtr", url="u", collector="file"),
        messages=[
            RawChatMessage(
                text=original,
                published_at=datetime(2026, 9, 1, tzinfo=UTC),
                message_id="thai-1",
            )
        ],
    )

    store_messages(db_session, result)

    stored = db_session.scalar(select(ChatMessage))
    assert stored.text == original
