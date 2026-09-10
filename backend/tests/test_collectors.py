"""Tests for the chat collectors and the YouTube chat parser."""

from __future__ import annotations

import json

import pytest

from app.services.collectors import (
    AVAILABLE_COLLECTORS,
    CollectorError,
    FileCollector,
    YouTubeApiCollector,
    YtDlpCollector,
    get_collector,
)
from app.services.collectors.base import extract_video_id, usec_to_datetime
from app.services.collectors.ytchat_parser import parse_chat_line, parse_chat_payload
from app.services.snapshot import write_snapshot

# --------------------------------------------------------------------------
# Video id extraction
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "https://www.youtube.com/watch?v=2VZxO1_S7rA",
        "https://youtu.be/2VZxO1_S7rA",
        "https://www.youtube.com/live/2VZxO1_S7rA",
        "https://www.youtube.com/embed/2VZxO1_S7rA",
        "https://www.youtube.com/watch?v=2VZxO1_S7rA&t=120s",
        "2VZxO1_S7rA",
    ],
)
def test_extract_video_id_handles_every_url_shape(source):
    assert extract_video_id(source) == "2VZxO1_S7rA"


@pytest.mark.parametrize("source", ["", "   ", "https://example.com/video", "abc"])
def test_extract_video_id_rejects_bad_input(source):
    with pytest.raises(CollectorError, match="video id"):
        extract_video_id(source)


def test_usec_to_datetime_falls_back_instead_of_raising():
    # A single malformed timestamp must not abort an import of thousands.
    assert usec_to_datetime("1725900000000000").year == 2024
    for bad in (None, "", "not-a-number", object()):
        assert usec_to_datetime(bad) is not None


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------


def test_registry_exposes_every_collector():
    assert set(AVAILABLE_COLLECTORS) == {"ytdlp", "youtube_api", "file"}
    assert isinstance(get_collector("ytdlp"), YtDlpCollector)
    assert isinstance(get_collector("file"), FileCollector)
    assert isinstance(get_collector("YouTube_API"), YouTubeApiCollector)


def test_registry_rejects_unknown_collector():
    with pytest.raises(CollectorError, match="Unknown collector"):
        get_collector("telepathy")


def test_youtube_api_collector_explains_the_missing_key():
    collector = YouTubeApiCollector(api_key=None)
    with pytest.raises(CollectorError, match="YOUTUBE_API_KEY"):
        collector.collect("2VZxO1_S7rA")


# --------------------------------------------------------------------------
# YouTube InnerTube chat parsing
# --------------------------------------------------------------------------


def _replay_line(text_runs: list[dict], **renderer_extra) -> str:
    """Build one yt-dlp replay JSON line around the given message runs."""
    renderer = {
        "message": {"runs": text_runs},
        "authorName": {"simpleText": "ผู้ชม"},
        "timestampUsec": "1725900000000000",
        "id": "ChwKGkNK",
        **renderer_extra,
    }
    payload = {
        "replayChatItemAction": {
            "actions": [
                {"addChatItemAction": {"item": {"liveChatTextMessageRenderer": renderer}}}
            ],
            "videoOffsetTimeMsec": "132000",
        }
    }
    return json.dumps(payload, ensure_ascii=False)


def test_parses_a_plain_thai_message():
    messages = parse_chat_line(_replay_line([{"text": "สู้ๆ นะครับ"}]))

    assert len(messages) == 1
    message = messages[0]
    assert message.text == "สู้ๆ นะครับ"
    assert message.author == "ผู้ชม"
    assert message.message_id == "ChwKGkNK"
    assert message.offset_ms == 132000


def test_unicode_emoji_are_kept_and_custom_emotes_become_shortcuts():
    messages = parse_chat_line(
        _replay_line(
            [
                {"text": "ไทยชนะ "},
                {"emoji": {"emojiId": "😀"}},
                {"emoji": {"isCustomEmoji": True, "shortcuts": [":_cheer:"]}},
            ]
        )
    )

    # A message made only of custom emotes must stay visible to the spam
    # filter rather than silently becoming empty.
    assert messages[0].text == "ไทยชนะ 😀:_cheer:"


def test_super_chat_messages_are_captured():
    payload = {
        "replayChatItemAction": {
            "actions": [
                {
                    "addChatItemAction": {
                        "item": {
                            "liveChatPaidMessageRenderer": {
                                "message": {"runs": [{"text": "เป็นกำลังใจให้"}]},
                                "authorName": {"simpleText": "แฟนคลับ"},
                                "timestampUsec": "1725900000000000",
                                "id": "paid-1",
                            }
                        }
                    }
                }
            ]
        }
    }

    messages = parse_chat_payload(payload)
    assert [m.text for m in messages] == ["เป็นกำลังใจให้"]


def test_non_text_actions_are_ignored():
    # Ticker items, stickers and deletions carry nothing to analyse.
    ignored = [
        {"replayChatItemAction": {"actions": [{"addLiveChatTickerItemAction": {}}]}},
        {
            "replayChatItemAction": {
                "actions": [
                    {"addChatItemAction": {"item": {"liveChatPaidStickerRenderer": {}}}}
                ]
            }
        },
        {"replayChatItemAction": {"actions": [{"markChatItemAsDeletedAction": {}}]}},
    ]
    for payload in ignored:
        assert parse_chat_payload(payload) == []


def test_empty_messages_are_dropped():
    assert parse_chat_line(_replay_line([{"text": "   "}])) == []


@pytest.mark.parametrize("line", ["", "   ", "not json", "[]", "null", '{"unexpected": 1}'])
def test_malformed_lines_never_raise(line):
    # One corrupt line must not take down an import.
    assert parse_chat_line(line) == []


def test_live_capture_shape_without_the_replay_wrapper():
    payload = {
        "addChatItemAction": {
            "item": {
                "liveChatTextMessageRenderer": {
                    "message": {"runs": [{"text": "สวัสดีครับ"}]},
                    "authorName": {"simpleText": "คนดู"},
                    "timestampUsec": "1725900000000000",
                    "id": "live-1",
                }
            }
        }
    }

    assert [m.text for m in parse_chat_payload(payload)] == ["สวัสดีครับ"]


# --------------------------------------------------------------------------
# File collector round-trip
# --------------------------------------------------------------------------


def test_snapshot_round_trip_preserves_thai_and_metadata(tmp_path, make_result):
    result = make_result(12, video_id="roundTrip1")
    path = write_snapshot(result, tmp_path / "roundTrip1.jsonl")

    replayed = FileCollector().collect(str(path))

    assert replayed.count == 12
    assert replayed.stream.video_id == "roundTrip1"
    assert replayed.stream.channel == "ช่องทดสอบ"
    assert [m.text for m in replayed.messages] == [m.text for m in result.messages]
    assert replayed.messages[0].published_at == result.messages[0].published_at


def test_file_collector_honours_the_limit(tmp_path, make_result):
    path = write_snapshot(make_result(50), tmp_path / "limited.jsonl")

    assert FileCollector().collect(str(path), limit=7).count == 7


def test_file_collector_reads_raw_ytdlp_format(tmp_path):
    raw = tmp_path / "rawVideoId1.live_chat.json"
    raw.write_text(
        "\n".join(
            [
                _replay_line([{"text": "ข้อความแรก"}]),
                "corrupt line that must be skipped",
                _replay_line([{"text": "ข้อความที่สอง"}]),
            ]
        ),
        encoding="utf-8",
    )

    result = FileCollector().collect(str(raw))

    assert [m.text for m in result.messages] == ["ข้อความแรก", "ข้อความที่สอง"]
    assert result.stream.video_id == "rawVideoId1"


def test_file_collector_error_lists_available_snapshots(tmp_path):
    with pytest.raises(CollectorError, match="not found"):
        FileCollector().collect(str(tmp_path / "missing.jsonl"))


def test_file_collector_rejects_a_snapshot_with_no_messages(tmp_path):
    empty = tmp_path / "empty.jsonl"
    empty.write_text('{"_stream": {"video_id": "x"}}\n', encoding="utf-8")

    with pytest.raises(CollectorError, match="No readable chat messages"):
        FileCollector().collect(str(empty))
