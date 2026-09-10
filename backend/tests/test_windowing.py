"""Tests for chat windowing.

Windowing is what gives the NLP pipeline enough text to work with, so the
bounds are worth pinning down precisely.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.services.collectors.base import RawChatMessage
from app.services.windowing import (
    _tidy_stream_title,
    build_windows,
    synthesise_title,
)


def test_message_count_is_the_governing_bound(make_messages):
    # 10s apart, so 450 messages span 75 minutes; disable the time bound to
    # isolate the size bound.
    windows = build_windows(make_messages(450), size=200, minutes=0)

    assert [w.count for w in windows] == [200, 200, 50]
    assert [w.index for w in windows] == [0, 1, 2]


def test_time_bound_closes_a_window_early(make_messages):
    # 60s apart with a 5 minute bound: a window can never reach 200 messages.
    windows = build_windows(make_messages(30, seconds_apart=60), size=200, minutes=5)

    assert len(windows) == 6
    assert all(w.count == 5 for w in windows)
    assert all(w.duration_seconds() <= 5 * 60 for w in windows)


def test_time_bound_can_be_disabled(make_messages):
    windows = build_windows(make_messages(50, seconds_apart=3600), size=200, minutes=0)

    # 50 messages an hour apart collapse into one window when time is ignored.
    assert len(windows) == 1
    assert windows[0].count == 50


def test_thin_windows_are_dropped_and_indexes_stay_contiguous(make_messages):
    # 205 messages at size 100 gives 100/100/5; the 5-message tail is too thin.
    windows = build_windows(make_messages(205), size=100, minutes=0, min_messages=10)

    assert [w.count for w in windows] == [100, 100]
    assert [w.index for w in windows] == [0, 1]


def test_final_partial_window_is_kept_when_large_enough(make_messages):
    windows = build_windows(make_messages(250), size=100, minutes=0, min_messages=10)

    assert [w.count for w in windows] == [100, 100, 50]


def test_empty_input_produces_no_windows():
    assert build_windows([]) == []


def test_messages_are_sorted_before_grouping():
    origin = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
    shuffled = [
        RawChatMessage(text="ที่สาม", published_at=origin + timedelta(minutes=2)),
        RawChatMessage(text="ที่หนึ่ง", published_at=origin),
        RawChatMessage(text="ที่สอง", published_at=origin + timedelta(minutes=1)),
    ]

    windows = build_windows(shuffled, size=200, minutes=0, min_messages=1)

    assert len(windows) == 1
    assert [m.text for m in windows[0].messages] == ["ที่หนึ่ง", "ที่สอง", "ที่สาม"]
    assert windows[0].start == origin


def test_invalid_window_size_is_rejected(make_messages):
    with pytest.raises(ValueError, match="at least 1 message"):
        build_windows(make_messages(5), size=0)


def test_as_document_joins_every_message(make_messages):
    windows = build_windows(make_messages(12), size=200, minutes=0, min_messages=1)

    document = windows[0].as_document()
    assert document.count("\n") == 11
    assert "ข้อความทดสอบ 0" in document
    assert "ข้อความทดสอบ 11" in document


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("🔴LIVE : ข่าวเที่ยงไทยรัฐ", "ข่าวเที่ยงไทยรัฐ"),
        ("LIVE!! รายการคุยข่าวเช้า", "รายการคุยข่าวเช้า"),
        ('🔴LIVE : "สนธิ" ประกาศลงถนน', "สนธิ ประกาศลงถนน"),
        ("🔴 LIVE :", "แชทสด"),  # nothing left after stripping decoration
        (None, "แชทสด"),
        ("", "แชทสด"),
    ],
)
def test_tidy_stream_title_strips_livestream_decoration(raw, expected):
    assert _tidy_stream_title(raw) == expected


def test_synthesised_title_leads_with_keywords_when_available(make_messages):
    window = build_windows(make_messages(20), size=200, minutes=0, min_messages=1)[0]

    with_keywords = synthesise_title(
        window,
        stream_title="🔴LIVE : ข่าวเที่ยง",
        keywords=["อุบัติเหตุ", "รถยนต์", "บาดเจ็บ"],
    )
    without = synthesise_title(window, stream_title="🔴LIVE : ข่าวเที่ยง")

    # Keywords are what distinguish one window from another, so they lead.
    assert with_keywords.startswith("อุบัติเหตุ · รถยนต์ · บาดเจ็บ")
    assert "ข่าวเที่ยง" in with_keywords
    assert "20 ข้อความ" in without


def test_synthesised_title_respects_max_length(make_messages):
    window = build_windows(make_messages(20), size=200, minutes=0, min_messages=1)[0]

    title = synthesise_title(
        window,
        stream_title="ก" * 300,
        keywords=["คำ" * 40],
        max_length=80,
    )

    assert len(title) <= 80
    assert title.endswith("…")
