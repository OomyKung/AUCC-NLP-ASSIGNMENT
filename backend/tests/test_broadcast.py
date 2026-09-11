"""Tests for the spoken-content pipeline: transcripts, segmentation, frames.

These cover the logic that is easy to get quietly wrong and hard to notice:
caption cleaning, cue end-time resolution, the boundary-signal combination, and
the storyboard tile arithmetic. Every case here is drawn from a real measured
property of the YouTube caption format rather than invented.
"""

from __future__ import annotations

import pytest

from app.services.frames import StoryboardTrack
from app.services.segmentation import (
    Block,
    Segment,
    _merge_short,
    cohesion_scores,
    depth_scores,
    label_change_indices,
    music_gap_indices,
    synthesise_headline,
)
from app.services.transcripts.base import (
    Transcript,
    TranscriptCue,
    clean_cue_text,
    resolve_cue_ends,
)


# --------------------------------------------------------------------------
# Caption cleaning
# --------------------------------------------------------------------------


def test_layout_newlines_are_dropped():
    """Roughly half of YouTube's json3 events are bare newlines for on-screen
    placement. Treating them as speech would double the cue count."""
    text, non_speech, _speaker = clean_cue_text("\n")

    assert text == ""
    assert not non_speech


def test_music_marker_is_flagged_not_deleted():
    """[เพลง] is not speech, but its position marks a programme transition."""
    text, non_speech, _speaker = clean_cue_text("[เพลง]")

    assert text == ""
    assert non_speech is True


def test_speaker_change_marker_is_recorded_and_stripped():
    text, _non_speech, speaker = clean_cue_text(">> ข้าชอบแม่ตะเพราแก้วมาก")

    assert speaker is True
    assert text.startswith("ข้าชอบ")
    assert ">>" not in text


def test_marker_plus_words_keeps_the_words():
    text, non_speech, _speaker = clean_cue_text("[เพลง] สวัสดีครับ")

    assert text == "สวัสดีครับ"
    assert non_speech is False


# --------------------------------------------------------------------------
# Cue end times
# --------------------------------------------------------------------------


def test_cue_ends_do_not_overlap():
    """YouTube sizes a caption for how long it stays on screen, not for how long
    its words are spoken, so reported durations overlap heavily -- 10,772 of
    11,148 consecutive pairs in one measured programme. A cue must end when the
    next begins or every time-based grouping is corrupted."""
    starts_and_durations = [(0, 7000), (3000, 7500), (6000, 5000)]

    ends = resolve_cue_ends(starts_and_durations)

    assert ends[0] <= 3000
    assert ends[1] <= 6000
    for (start, _duration), end in zip(starts_and_durations, ends, strict=True):
        assert end >= start


def test_long_silence_does_not_stretch_a_cue():
    """A gap before the next cue must not make this one cover minutes."""
    ends = resolve_cue_ends([(0, 0), (600_000, 1000)], max_gap_ms=10_000)

    assert ends[0] <= 10_000


def test_zero_duration_cue_is_handled():
    ends = resolve_cue_ends([(1000, 0), (4000, 1000)])

    assert ends[0] == 4000


# --------------------------------------------------------------------------
# Transcript slicing
# --------------------------------------------------------------------------


def _transcript() -> Transcript:
    return Transcript(
        video_id="abc",
        source="test",
        language="th",
        duration_ms=40_000,
        cues=[
            TranscriptCue(0, 10_000, "ตำรวจจับกุมผู้ต้องหา"),
            TranscriptCue(10_000, 20_000, "", non_speech=True),
            TranscriptCue(20_000, 30_000, "รถกระบะชนเสาไฟฟ้า"),
            TranscriptCue(30_000, 40_000, "ทีมชาติไทยชนะ"),
        ],
    )


def test_text_between_selects_overlapping_speech_only():
    transcript = _transcript()

    assert transcript.text_between(0, 10_000) == "ตำรวจจับกุมผู้ต้องหา"
    # The non-speech cue contributes nothing.
    assert transcript.text_between(10_000, 20_000) == ""


def test_text_between_joins_without_spaces():
    """Thai has no inter-word spaces; inserting one would create a boundary the
    tokeniser then has to undo, changing the tokenisation."""
    joined = _transcript().text_between(20_000, 40_000)

    assert joined == "รถกระบะชนเสาไฟฟ้าทีมชาติไทยชนะ"
    assert " " not in joined


def test_speech_cues_and_counts_exclude_non_speech():
    transcript = _transcript()

    assert len(transcript.speech_cues) == 3
    assert transcript.character_count == sum(
        len(c.text) for c in transcript.speech_cues
    )


def test_transcript_round_trips_through_json():
    original = _transcript()

    restored = Transcript.from_dict(original.as_dict())

    assert len(restored) == len(original)
    assert restored.cues[1].non_speech is True
    assert restored.text_between(0, 40_000) == original.text_between(0, 40_000)


# --------------------------------------------------------------------------
# Segmentation signals
# --------------------------------------------------------------------------


def _blocks(token_groups: list[list[str]]) -> list[Block]:
    return [
        Block(
            index=index,
            start_ms=index * 30_000,
            end_ms=(index + 1) * 30_000,
            text="ก" * 60,
            tokens=list(tokens),
        )
        for index, tokens in enumerate(token_groups)
    ]


def test_cohesion_dips_where_vocabulary_changes():
    """Three blocks about police, then three about football."""
    police = ["ตำรวจ", "จับกุม", "ผู้ต้องหา"]
    football = ["ฟุตบอล", "ทีมชาติ", "ประตู"]
    blocks = _blocks([police] * 3 + [football] * 3)

    scores = cohesion_scores(blocks, window=2)

    # The gap at index 3 separates the two subjects entirely.
    assert scores[3] == pytest.approx(0.0)
    # A gap inside one subject stays cohesive.
    assert scores[2] > scores[3]


def test_depth_score_finds_the_valley():
    depths = depth_scores([1.0, 0.9, 0.2, 0.85, 1.0])

    assert depths[2] == max(depths)
    assert depths[2] > 0


def test_label_change_needs_a_run_not_a_single_block():
    """One disagreeing block is classifier noise on 30 seconds of speech."""
    blocks = _blocks([["a"]] * 5)
    for block, topic in zip(blocks, ["crime", "crime", "sports", "crime", "crime"]):
        block.topic = topic

    assert label_change_indices(blocks, run=2) == set()


def test_label_change_fires_on_a_sustained_change():
    blocks = _blocks([["a"]] * 5)
    for block, topic in zip(blocks, ["crime", "crime", "sports", "sports", "sports"]):
        block.topic = topic

    assert 2 in label_change_indices(blocks, run=2)


def test_music_gap_needs_enough_silence():
    blocks = _blocks([["a"]] * 3)
    blocks[1].non_speech_ms = 500
    blocks[2].non_speech_ms = 8000

    hits = music_gap_indices(blocks, threshold_ms=4000)

    assert hits == {2}


def test_block_zero_is_never_a_boundary():
    """There is nothing before the first block to be a boundary from."""
    blocks = _blocks([["a"]] * 3)
    blocks[0].non_speech_ms = 99_000

    assert 0 not in music_gap_indices(blocks)


# --------------------------------------------------------------------------
# Short-segment merging
# --------------------------------------------------------------------------


def test_short_spans_are_absorbed():
    spans = [(0, 120_000, None), (120_000, 130_000, None), (130_000, 300_000, None)]

    merged = _merge_short(list(spans), minimum_ms=60_000)

    assert all(end - start >= 60_000 for start, end, _ in merged)
    # No time is lost or invented by merging.
    assert merged[0][0] == 0
    assert merged[-1][1] == 300_000


def test_leading_short_span_absorbs_forwards():
    """The first span has no predecessor, so it must merge into its successor."""
    spans = [(0, 5_000, None), (5_000, 200_000, None)]

    merged = _merge_short(list(spans), minimum_ms=60_000)

    assert len(merged) == 1
    assert merged[0][:2] == (0, 200_000)


# --------------------------------------------------------------------------
# Deep links and labels
# --------------------------------------------------------------------------


def test_deep_link_lands_just_before_the_first_word():
    segment = Segment(index=0, start_ms=448_000, end_ms=700_000, text="x")

    url = segment.youtube_url("Gq9FLCg1yFo")

    assert url == "https://www.youtube.com/watch?v=Gq9FLCg1yFo&t=446s"


def test_deep_link_never_goes_negative():
    segment = Segment(index=0, start_ms=0, end_ms=60_000, text="x")

    assert segment.youtube_url("abc").endswith("&t=0s")


@pytest.mark.parametrize(
    ("start_ms", "expected"),
    [(0, "0:00"), (65_000, "1:05"), (3_600_000, "1:00:00"), (4_265_000, "1:11:05")],
)
def test_timecode_reads_like_a_video_player(start_ms, expected):
    assert Segment(index=0, start_ms=start_ms, end_ms=start_ms, text="").timecode == expected


def test_headline_prefers_keywords():
    segment = Segment(index=0, start_ms=0, end_ms=1, text="ยาว" * 500)
    segment.keywords = ["ลิง", "ซาก", "ทนายความ", "ก่อเหตุ", "ตำรวจ"]

    headline = synthesise_headline(segment)

    assert headline == "ลิง · ซาก · ทนายความ · ก่อเหตุ"


def test_headline_is_never_empty():
    segment = Segment(index=0, start_ms=90_000, end_ms=120_000, text="")

    assert synthesise_headline(segment)


# --------------------------------------------------------------------------
# Storyboard arithmetic
# --------------------------------------------------------------------------


def _track() -> StoryboardTrack:
    """Matches a real measured track: sb0, 3x3 tiles, ~90s per sheet."""
    return StoryboardTrack(
        format_id="sb0",
        tile_width=320,
        tile_height=180,
        rows=3,
        columns=3,
        fragments=[("http://sheet0", 90.0), ("http://sheet1", 90.0)],
    )


def test_storyboard_resolution_is_one_frame_per_ten_seconds():
    assert _track().seconds_per_tile == pytest.approx(10.0)


@pytest.mark.parametrize(
    ("second", "sheet", "row", "column"),
    [
        (0, "http://sheet0", 0, 0),
        (25, "http://sheet0", 0, 2),
        (85, "http://sheet0", 2, 2),
        (95, "http://sheet1", 0, 0),
    ],
)
def test_storyboard_locates_the_right_tile(second, sheet, row, column):
    url, got_row, got_column = _track().locate(second)

    assert (url, got_row, got_column) == (sheet, row, column)


def test_storyboard_clamps_past_the_end():
    """A timestamp beyond the last sheet must still return a valid tile."""
    url, row, column = _track().locate(10_000)

    assert url == "http://sheet1"
    assert (row, column) == (2, 2)


def test_isolated_label_blip_does_not_split_a_story_at_the_return():
    """The failure this smoothing exists for.

    Forward hysteresis alone refuses to split at the blip, then splits at the
    *return*: from the blip's label, the following blocks look like a perfectly
    sustained change. One story ends up cut in two by a single bad block.
    """
    from app.services.segmentation import smooth_labels

    assert smooth_labels(["crime", "crime", "sports", "crime", "crime"]) == [
        "crime",
        "crime",
        "crime",
        "crime",
        "crime",
    ]


def test_smoothing_keeps_a_genuine_two_block_story():
    """A label held for two blocks is a short story, not a blip."""
    from app.services.segmentation import smooth_labels

    assert smooth_labels(["crime", "sports", "sports", "crime"]) == [
        "crime",
        "sports",
        "sports",
        "crime",
    ]


# --------------------------------------------------------------------------
# Headlines
# --------------------------------------------------------------------------


def test_token_offsets_survive_whitespace():
    """The tokeniser drops whitespace, so cumulative token lengths drift and
    every slice after the first space starts mid-word (``้วันศุกร์``)."""
    from app.nlp.headline import token_offsets

    text = "ตำรวจ จับกุม ผู้ต้องหา"
    tokens = ["ตำรวจ", "จับกุม", "ผู้ต้องหา"]

    offsets = token_offsets(text, tokens)

    for token, offset in zip(tokens, offsets, strict=True):
        assert text[offset : offset + len(token)] == token


def test_headline_is_a_phrase_not_a_keyword_list():
    """The whole point: ``ติดตาม · นิติ · ศุกร์ · เช้านี้`` described nothing."""
    from app.nlp.headline import extract_headline

    text = (
        "สวัสดีครับท่านผู้ชมนะครับ"
        "ตำรวจจับกุมผู้ต้องหาคดียาเสพติดรายใหญ่ยึดของกลางมูลค่ากว่าสิบล้านบาท"
        "ที่บ้านพักย่านชานเมืองนะครับ"
    )

    headline = extract_headline(text, keywords=["ตำรวจ", "จับกุม", "ยาเสพติด"])

    assert headline
    assert "·" not in headline
    assert "ตำรวจ" in headline or "จับกุม" in headline


def test_headline_does_not_start_or_end_on_a_particle():
    from app.nlp.headline import extract_headline

    text = (
        "นะครับรถกระบะเสียหลักพุ่งชนเสาไฟฟ้าริมถนนมีผู้ได้รับบาดเจ็บสามราย"
        "เจ้าหน้าที่กู้ภัยนำส่งโรงพยาบาลแล้วนะครับ"
    )

    headline = extract_headline(text, keywords=["รถกระบะ", "เสาไฟฟ้า", "บาดเจ็บ"])

    assert headline
    assert not headline.startswith(("นะ", "ครับ", "ค่ะ", "ก็", "แล้ว"))
    assert not headline.endswith(("นะ", "ก็", "ที่", "และ", "ของ"))


def test_headline_prefers_starting_on_a_cue_boundary():
    """A cue is a real unit of speech. Without this the span can open on a
    fragment of a split name, because Thai has no spaces and the tokeniser
    splits inside names as readily as between words."""
    from app.nlp.headline import extract_headline

    first = "ช่วงนี้อากาศร้อนมากนะครับ"
    second = "ตำรวจจับกุมผู้ต้องหาคดียาเสพติดรายใหญ่ยึดของกลางจำนวนมาก"
    text = first + second

    headline = extract_headline(
        text,
        keywords=["ตำรวจ", "จับกุม", "ยาเสพติด"],
        cue_starts={0, len(first)},
    )

    assert headline.startswith("ตำรวจ")


def test_headline_returns_empty_on_text_too_short_to_describe():
    from app.nlp.headline import extract_headline

    assert extract_headline("สวัสดี", keywords=["สวัสดี"]) == ""
    assert extract_headline("", keywords=[]) == ""


def test_segment_headline_falls_back_rather_than_being_blank():
    """A rambling or badly transcribed story may have no dense span. A weak
    title beats a nameless card."""
    segment = Segment(index=0, start_ms=185_000, end_ms=245_000, text="อ่า")
    segment.keywords = ["ตำรวจ", "จับกุม"]

    assert synthesise_headline(segment)
