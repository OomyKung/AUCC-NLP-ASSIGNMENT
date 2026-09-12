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


# --------------------------------------------------------------------------
# Boundary refinement
# --------------------------------------------------------------------------


def _timed_transcript() -> Transcript:
    """Two stories with a music sting between them at 60s."""
    cues = [
        TranscriptCue(0, 10_000, "ตำรวจจับกุมผู้ต้องหาคดียาเสพติด"),
        TranscriptCue(10_000, 20_000, "ยึดของกลางมูลค่ากว่าสิบล้านบาท"),
        TranscriptCue(20_000, 30_000, "เจ้าหน้าที่ขยายผลจับกุมเครือข่าย"),
        TranscriptCue(30_000, 45_000, "ผู้ต้องหาให้การรับสารภาพกับตำรวจ"),
        TranscriptCue(45_000, 58_000, "", non_speech=True),
        TranscriptCue(58_000, 70_000, "ทีมชาติไทยชนะการแข่งขันฟุตบอล", speaker_change=True),
        TranscriptCue(70_000, 80_000, "กองเชียร์ร่วมฉลองชัยชนะที่สนามกีฬา"),
        TranscriptCue(80_000, 95_000, "นักฟุตบอลทีมชาติขอบคุณแฟนบอล"),
    ]
    return Transcript(
        video_id="x", source="test", language="th", duration_ms=95_000, cues=cues
    )


def test_refinement_snaps_to_the_music_break():
    """Thai news puts a sting between items, so the next story starts when it
    ends. That is the most reliable cut available."""
    from app.services.segmentation import _cue_token_index, refine_boundary

    transcript = _timed_transcript()
    tokens = _cue_token_index(transcript)

    time_ms, reason = refine_boundary(transcript, tokens, 60_000)

    assert reason == "music-cut"
    assert time_ms == 58_000


def test_refinement_moves_off_the_block_grid():
    """The defect this fixes: every boundary landed on a 30s grid point, so the
    reported time was out by up to 15s and the block holding the switch was a
    mixture of two stories."""
    from app.services.segmentation import _cue_token_index, refine_boundary

    transcript = _timed_transcript()
    # Remove the music cue so refinement has to use lexical evidence.
    transcript.cues = [c for c in transcript.cues if not c.non_speech]
    tokens = _cue_token_index(transcript)

    time_ms, reason = refine_boundary(transcript, tokens, 60_000)

    assert reason in {"lexical-cut", "speaker-cut"}
    assert time_ms % 30_000 != 0


def test_refinement_falls_back_to_the_grid_when_there_is_nothing_to_use():
    from app.services.segmentation import refine_boundary

    empty = Transcript(video_id="x", source="test", language="th", cues=[])

    assert refine_boundary(empty, [], 60_000) == (60_000, "grid")


def test_mixed_span_is_split_where_the_vocabulary_changes():
    """Bottom-up detection misses gradual transitions, leaving one span across
    two stories -- measured at 54% of segments before this existed."""
    from app.services.segmentation import _cue_token_index, best_internal_cut

    transcript = _timed_transcript()
    tokens = _cue_token_index(transcript)

    cut, strength = best_internal_cut(
        transcript, tokens, 0, 95_000, minimum_ms=20_000
    )

    assert cut is not None
    # The two stories share no vocabulary, so the cut must be a strong one.
    assert strength > 0.7
    # The real change is at the football story; the cut must be near it, not
    # in the middle of either story.
    assert 45_000 <= cut <= 80_000


def test_internal_cut_respects_the_minimum_segment_length():
    """Both halves must survive the minimum, so the search is restricted rather
    than the result rejected afterwards."""
    from app.services.segmentation import _cue_token_index, best_internal_cut

    transcript = _timed_transcript()
    tokens = _cue_token_index(transcript)

    cut, _strength = best_internal_cut(
        transcript, tokens, 0, 95_000, minimum_ms=60_000
    )
    assert cut is None


def test_topic_disagreement_alone_does_not_split_a_coherent_story():
    """The regression this veto exists for.

    Splitting on classifier disagreement alone fragmented coherent stories,
    because the classifier is the unreliable part -- it sees forty seconds of
    out-of-domain speech. One five-minute report became four segments, two of
    them mislabelled, where it had been correct as a single story.
    """
    from app.services.segmentation import (
        MIXED_SPLIT_MIN_DISSIMILARITY,
        _cue_token_index,
        best_internal_cut,
    )

    # One story throughout: the vocabulary never changes.
    cues = [
        TranscriptCue(i * 10_000, (i + 1) * 10_000, "ตำรวจจับกุมผู้ต้องหายาเสพติดของกลาง")
        for i in range(10)
    ]
    transcript = Transcript(
        video_id="x", source="test", language="th", duration_ms=100_000, cues=cues
    )
    tokens = _cue_token_index(transcript)

    _cut, strength = best_internal_cut(
        transcript, tokens, 0, 100_000, minimum_ms=20_000
    )

    assert strength < MIXED_SPLIT_MIN_DISSIMILARITY


# --------------------------------------------------------------------------
# ASR name repair
# --------------------------------------------------------------------------


def test_no_provider_available_degrades_rather_than_breaking(monkeypatch):
    """Nothing here may be load-bearing. A fresh clone may have no API key and
    no Ollama running, and must still produce a full timeline.

    (This used to assert the absence of LLM_API_KEY alone made enrichment
    impossible. Ollama needs no key, so the condition is now "no provider
    reachable", not "no key".)
    """
    from app.config import settings
    from app.nlp import llm_enrich

    monkeypatch.setattr(settings, "llm_provider", "ollama")

    def unreachable(prompt, *, timeout):
        raise ConnectionError("connection refused")

    monkeypatch.setattr(llm_enrich, "_call_ollama", unreachable)

    with pytest.raises(llm_enrich.LLMUnavailable):
        llm_enrich.enrich_segment(
            "ตำรวจจับกุมผู้ต้องหาคดียาเสพติดรายใหญ่ยึดของกลางจำนวนมาก"
        )


def test_parses_a_fenced_json_completion():
    """Models wrap JSON in code fences given half a chance."""
    from app.nlp.llm_enrich import _parse

    result = _parse(
        '```json\n{"headline": "นักกีฬาเข้าพบนายกรัฐมนตรี", '
        '"entities": [{"text": "ศศิภาพร จันทวิสูตร", "label": "PERSON", '
        '"asr": "สักสิภาพรจันทวิสูตร"}]}\n```'
    )

    assert result.headline == "นักกีฬาเข้าพบนายกรัฐมนตรี"
    assert result.entities == [{"text": "ศศิภาพร จันทวิสูตร", "label": "PERSON"}]
    assert result.corrections == [("สักสิภาพรจันทวิสูตร", "ศศิภาพร จันทวิสูตร")]


def test_a_name_left_unchanged_is_not_recorded_as_a_correction():
    """The prompt tells the model to keep the ASR form when unsure; that is not
    a correction and must not be presented as one."""
    from app.nlp.llm_enrich import _parse

    result = _parse(
        '{"headline": "ข่าว", "entities": '
        '[{"text": "สมชาย", "label": "PERSON", "asr": "สมชาย"}]}'
    )

    assert result.entities
    assert result.corrections == []


def test_rejects_entities_that_are_not_thai_or_not_a_known_type():
    from app.nlp.llm_enrich import _parse

    result = _parse(
        '{"headline": "ข่าว", "entities": ['
        '{"text": "PERSON", "label": "PERSON"},'
        '{"text": "กรุงเทพ", "label": "CITY"},'
        '{"text": "ตำรวจ", "label": "ORGANIZATION"}]}'
    )

    assert result.entities == [{"text": "ตำรวจ", "label": "ORGANIZATION"}]


def test_malformed_completion_raises_rather_than_returning_junk():
    from app.nlp.llm_enrich import LLMUnavailable, _parse

    with pytest.raises(LLMUnavailable):
        _parse("I'm sorry, I can't help with that.")


def test_corrections_apply_longest_first():
    """A short form must not pre-empt the longer one that contains it."""
    from app.nlp.llm_enrich import apply_corrections

    text = "สักสิภาพรจันทวิสูตรลงแข่ง"
    fixed = apply_corrections(
        text, [("สักสิ", "ศศิ"), ("สักสิภาพรจันทวิสูตร", "ศศิภาพร จันทวิสูตร")]
    )

    assert fixed == "ศศิภาพร จันทวิสูตรลงแข่ง"


# --------------------------------------------------------------------------
# Rule-based NER on broadcast speech
# --------------------------------------------------------------------------


def test_presenter_addressing_the_audience_is_not_a_person():
    """"คุณผู้ชม" produced 146 spurious PERSON entities across one programme --
    more than any real name -- because "คุณ" is a personal title."""
    from app.nlp.entities import RuleEntityRecognizer

    found = RuleEntityRecognizer().extract(
        "สวัสดีครับคุณผู้ชมวันนี้นายสมชายแถลงข่าว"
    )
    people = [e.text for e in found if e.label == "PERSON"]

    assert not any("ผู้ชม" in name for name in people)
    assert any("สมชาย" in name for name in people)


def test_a_verb_is_not_part_of_a_name():
    from app.nlp.entities import RuleEntityRecognizer

    found = RuleEntityRecognizer().extract("นายทรงพล ขับรถออกจากบ้าน")
    people = [e.text for e in found if e.label == "PERSON"]

    assert people
    assert all("ขับ" not in name for name in people)


def test_ambiguous_province_needs_a_location_prefix():
    """เลย is the province Loei and also the everyday word "at all". Without a
    prefix requirement it produced 102 false LOCATION hits -- more than every
    real province combined."""
    from app.nlp.entities import RuleEntityRecognizer

    recogniser = RuleEntityRecognizer()

    particle = recogniser.extract("เขาไม่มาเลยนะครับวันนี้")
    assert not any(e.text == "เลย" for e in particle if e.label == "LOCATION")

    place = recogniser.extract("เกิดเหตุที่จังหวัดเลยเมื่อคืนนี้")
    assert any(e.label == "LOCATION" for e in place)


# --------------------------------------------------------------------------
# Name repair from the project's own evidence
# --------------------------------------------------------------------------


def _lexicon(**weights):
    from collections import Counter

    return Counter(weights)


def test_welded_suffix_is_trimmed_against_the_lexicon():
    """The ASR welds the next word onto a name, so one person becomes many:
    ไอซ์ appeared as ไอซ์ดำ, ไอซ์รัก, ไอซ์เนี่ย and four more."""
    from app.nlp.name_repair import correct_name

    lexicon = _lexicon(**{"ไอซ์": 72})

    assert correct_name("ไอซ์เนี่ย", lexicon) == "ไอซ์"
    assert correct_name("ไอซ์ไง", lexicon) == "ไอซ์"


def test_a_real_name_is_not_truncated_to_a_shorter_one():
    """The failure this guard exists for. Thai given names are built from
    components that are themselves names, so "the stem is attested" is not on
    its own evidence that the rest is not part of the name."""
    from app.nlp.name_repair import correct_name

    lexicon = _lexicon(**{"สุภา": 5, "เลิศ": 6, "มัลิ": 14})

    assert correct_name("สุภาพร", lexicon) == "สุภาพร"
    assert correct_name("เลิศศักดิ์", lexicon) == "เลิศศักดิ์"
    assert correct_name("มัลิกา", lexicon) == "มัลิกา"


def test_an_attested_spelling_is_never_overruled():
    from app.nlp.name_repair import correct_name

    lexicon = _lexicon(**{"ชัยชนก": 4, "ชัย": 90})

    assert correct_name("ชัยชนก", lexicon) == "ชัยชนก"


def test_variants_collapse_onto_a_stem_that_was_seen_alone():
    from app.nlp.name_repair import collapse_variants

    bodies = ["กัน", "กันจอม", "กันติด", "กันพูด", "กันนะ"]

    mapping = collapse_variants(bodies)

    assert {mapping[b] for b in bodies} == {"กัน"}


def test_different_people_sharing_a_prefix_do_not_collapse():
    """สุพนัส, สุภาพร and สุริยัน are three people. Collapsing them onto "สุ"
    would merge them, so a stem is only accepted when it was itself observed
    standing alone -- and "สุ" never is."""
    from app.nlp.name_repair import collapse_variants

    bodies = ["สุพนัส", "สุภาพร", "สุริยัน", "สุรินทร์"]

    mapping = collapse_variants(bodies)

    assert mapping == {b: b for b in bodies}


def test_lexicon_harvest_tokenises_instead_of_pattern_matching():
    """A regex character class runs straight through the following words,
    because Thai has no boundaries: น้องออมสินสวยมาก yields ออมสินสวยมาก."""
    from app.nlp.name_repair import build_lexicon

    lexicon = build_lexicon(["น้องออมสินสวยมาก", "แฟนออมสินหล่อ"], [])

    assert lexicon.get("ออมสิน", 0) > 0
    assert not any("สวยมาก" in term for term in lexicon)


def test_an_asr_misspelling_is_left_alone_rather_than_guessed():
    """สักสิภาพร is never written correctly anywhere in this project's data, so
    there is nothing to correct it from. Inventing a plausible Thai name would
    look better and be worse, because a reader cannot tell the two apart."""
    from app.nlp.name_repair import correct_name

    lexicon = _lexicon(**{"ศศิ": 9, "ภาพร": 5})

    assert correct_name("สักสิภาพร", lexicon) == "สักสิภาพร"


# --------------------------------------------------------------------------
# LLM provider routing
# --------------------------------------------------------------------------


def test_name_repair_is_off_by_default():
    """A 7B model told not to guess produced อนุทินชื่นกล่าว for
    อนุทินชาวรกูล and อำพันสิทธิ์จันทวิสูตร for อำสินสักสิภาพร. A
    plausible-looking wrong name is worse than a garbled one."""
    from app.config import settings

    assert settings.llm_correct_names is False


def test_headline_only_mode_never_sends_the_correction_prompt(monkeypatch):
    """The guard is on the *request*, not on the response. A prompt that invites
    a guess gets one, so it must not be sent at all."""
    from app.config import settings
    from app.nlp import llm_enrich

    monkeypatch.setattr(settings, "llm_correct_names", False)
    sent: list[str] = []

    def fake(prompt, *, timeout):
        sent.append(prompt)
        return "จับกุมผู้ต้องหาคดียาเสพติด"

    monkeypatch.setattr(llm_enrich, "_complete", fake)
    result = llm_enrich.enrich_segment("ตำรวจจับกุมผู้ต้องหาคดียาเสพติดรายใหญ่ยึดของกลาง")

    assert result.headline == "จับกุมผู้ต้องหาคดียาเสพติด"
    assert result.entities == []
    assert len(sent) == 1
    # The word the correction prompt uses for "correct the spelling".
    assert "แก้การสะกด" not in sent[0]


def test_correction_prompt_is_used_when_explicitly_enabled(monkeypatch):
    from app.config import settings
    from app.nlp import llm_enrich

    monkeypatch.setattr(settings, "llm_correct_names", True)
    sent: list[str] = []

    def fake(prompt, *, timeout):
        sent.append(prompt)
        return '{"headline": "ข่าว", "entities": []}'

    monkeypatch.setattr(llm_enrich, "_complete", fake)
    llm_enrich.enrich_segment("ตำรวจจับกุมผู้ต้องหาคดียาเสพติดรายใหญ่ยึดของกลาง")

    assert "แก้การสะกด" in sent[0]


def test_ollama_needs_no_api_key(monkeypatch):
    """The whole point of the local provider: free, and no account."""
    from app.config import settings
    from app.nlp import llm_enrich

    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "llm_api_key", None)
    monkeypatch.setattr(
        llm_enrich, "_call_ollama", lambda prompt, *, timeout: "พาดหัวข่าว"
    )

    assert llm_enrich._complete("x", timeout=5) == "พาดหัวข่าว"


def test_anthropic_without_a_key_is_reported_not_attempted(monkeypatch):
    from app.config import settings
    from app.nlp import llm_enrich

    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "llm_api_key", None)

    with pytest.raises(llm_enrich.LLMUnavailable, match="LLM_API_KEY"):
        llm_enrich._complete("x", timeout=5)


def test_a_failing_model_leaves_the_extractive_headline_alone(monkeypatch):
    """A worse headline is a far better outcome than a missing story."""
    from app.nlp import llm_enrich
    from app.services.segmentation import enrich_with_llm

    segment = Segment(index=0, start_ms=0, end_ms=60_000, text="ตำรวจจับกุมผู้ต้องหา")
    segment.headline = "extractive headline"

    def boom(prompt, *, timeout):
        raise llm_enrich.LLMUnavailable("ollama is not running")

    monkeypatch.setattr(llm_enrich, "_complete", boom)
    assert enrich_with_llm(segment) == "unavailable"

    assert segment.headline == "extractive headline"
    assert segment.enriched_by == ""
    # Reported as unavailable, so the caller can stop asking a dead provider.
    assert enrich_with_llm(segment) == "unavailable"


def test_a_dead_provider_stops_being_asked(monkeypatch):
    """The circuit breaker, and the reason it exists.

    The enrichment timeout is 120 seconds. Without this, a 76-story programme
    run against an Ollama that is not running would sit for two and a half hours
    to produce exactly the extractive result it could have produced at once.
    """
    from app.config import settings
    from app.nlp import llm_enrich
    from app.services import segmentation

    monkeypatch.setattr(settings, "llm_enrich_segments", True)
    monkeypatch.setattr(settings, "llm_provider", "ollama")

    calls: list[str] = []

    def unreachable(prompt, *, timeout):
        calls.append(prompt)
        raise llm_enrich.LLMUnavailable("connection refused")

    monkeypatch.setattr(llm_enrich, "_complete", unreachable)

    segments = [
        Segment(
            index=index,
            start_ms=index * 60_000,
            end_ms=(index + 1) * 60_000,
            text=f"ข่าวเรื่องที่ {index} ตำรวจจับกุมผู้ต้องหาคดียาเสพติดรายใหญ่ยึดของกลาง",
        )
        for index in range(10)
    ]
    segmentation.analyse_segments(segments)

    assert len(calls) == segmentation.LLM_FAILURE_LIMIT
    # Every story still got its extractive analysis.
    assert all(segment.headline for segment in segments)
    assert all(segment.enriched_by == "" for segment in segments)


def test_a_tripped_breaker_still_reads_the_cache(monkeypatch):
    """The headlines already written are on disk and have nothing to do with the
    provider being down. Disabling enrichment outright would throw them away."""
    from app.config import settings
    from app.nlp import llm_enrich
    from app.services import segmentation
    from app.services.enrichment_cache import EnrichmentCache

    monkeypatch.setattr(settings, "llm_enrich_segments", True)
    monkeypatch.setattr(settings, "llm_provider", "ollama")

    calls: list[str] = []

    def unreachable(prompt, *, timeout):
        calls.append(prompt)
        raise llm_enrich.LLMUnavailable("connection refused")

    monkeypatch.setattr(llm_enrich, "_complete", unreachable)

    texts = [
        f"ข่าวเรื่องที่ {index} ตำรวจจับกุมผู้ต้องหาคดียาเสพติดรายใหญ่ยึดของกลาง"
        for index in range(8)
    ]
    cache = EnrichmentCache("video")
    # The last story was written on an earlier run, before the provider died.
    cache.put(texts[-1], headline="พาดหัวจากแคช", entities=[], corrections=[])

    segments = [_segment(text, index=index) for index, text in enumerate(texts)]
    segmentation.analyse_segments(segments, cache=cache)

    assert len(calls) == segmentation.LLM_FAILURE_LIMIT
    assert segments[-1].headline == "พาดหัวจากแคช"
    assert segments[-1].enriched_by == "llm"


def test_a_recovering_provider_keeps_being_asked(monkeypatch):
    """One timeout must not disable the model for the rest of the programme --
    the breaker counts *consecutive* failures, not total ones."""
    from app.config import settings
    from app.services import segmentation

    monkeypatch.setattr(settings, "llm_enrich_segments", True)
    monkeypatch.setattr(settings, "llm_provider", "ollama")

    attempts: list[int] = []

    def flaky(segment, *, cache=None, allow_model=True):
        attempts.append(segment.index)
        # Fails on every third story, so failures never run consecutively.
        worked = segment.index % 3 != 0
        if worked:
            segment.enriched_by = "llm"
        return "model" if worked else "unavailable"

    monkeypatch.setattr(segmentation, "enrich_with_llm", flaky)

    segments = [
        Segment(
            index=index,
            start_ms=index * 60_000,
            end_ms=(index + 1) * 60_000,
            text="ตำรวจจับกุมผู้ต้องหาคดียาเสพติดรายใหญ่ยึดของกลางจำนวนมาก",
        )
        for index in range(9)
    ]
    segmentation.analyse_segments(segments)

    assert len(attempts) == 9


def test_progress_is_reported_per_story(monkeypatch):
    """A local model spends ~20s per story, so a caller must be able to say so
    rather than print nothing for a quarter of an hour."""
    from app.config import settings
    from app.services import segmentation

    monkeypatch.setattr(settings, "llm_enrich_segments", False)

    seen: list[tuple[int, int]] = []
    segments = [
        Segment(
            index=index,
            start_ms=index * 60_000,
            end_ms=(index + 1) * 60_000,
            text="ตำรวจจับกุมผู้ต้องหาคดียาเสพติดรายใหญ่ยึดของกลางจำนวนมาก",
        )
        for index in range(3)
    ]
    segmentation.analyse_segments(
        segments, progress=lambda done, total, segment: seen.append((done, total))
    )

    assert seen == [(1, 3), (2, 3), (3, 3)]


# --------------------------------------------------------------------------
# Enrichment cache: the 40 minutes of model time is paid once, then committed
# --------------------------------------------------------------------------


NEWLINE = chr(10)


def _segment(text: str, index: int = 0) -> Segment:
    return Segment(
        index=index,
        start_ms=index * 60_000,
        end_ms=(index + 1) * 60_000,
        text=text,
    )


def test_a_cached_headline_is_reused_without_calling_the_model(monkeypatch):
    """The reason the cache exists: 20 seconds a story, 112 stories."""
    from app.nlp import llm_enrich
    from app.services.enrichment_cache import EnrichmentCache
    from app.services.segmentation import enrich_with_llm

    text = "ตำรวจจับกุมผู้ต้องหาคดียาเสพติดรายใหญ่ยึดยาบ้าหนึ่งล้านเม็ด"
    cache = EnrichmentCache("video")
    cache.put(text, headline="จับกุมคดียาเสพติดรายใหญ่", entities=[], corrections=[])

    def must_not_be_called(prompt, *, timeout):
        raise AssertionError("the model was called despite a cache hit")

    monkeypatch.setattr(llm_enrich, "_complete", must_not_be_called)

    segment = _segment(text)
    segment.headline = "extractive"
    assert enrich_with_llm(segment, cache=cache) == "cache"
    assert segment.headline == "จับกุมคดียาเสพติดรายใหญ่"
    assert segment.enriched_by == "llm"


def test_a_changed_segment_misses_rather_than_reusing_the_old_headline(monkeypatch):
    """If segmentation moves a boundary the story is not the same story, so the
    previous headline must not be stapled onto new content."""
    from app.services.enrichment_cache import EnrichmentCache

    cache = EnrichmentCache("video")
    cache.put("ข่าวแรกเรื่องอุบัติเหตุบนถนน", headline="อุบัติเหตุ", entities=[], corrections=[])

    assert cache.get("ข่าวแรกเรื่องอุบัติเหตุบนถนน") is not None
    assert cache.get("ข่าวแรกเรื่องอุบัติเหตุบนถนนและข่าวที่สอง") is None


def test_whitespace_alone_does_not_cause_a_miss():
    """The ASR's spacing is not meaningful; anything else is."""
    from app.services.enrichment_cache import EnrichmentCache

    cache = EnrichmentCache("video")
    cache.put("ข่าว  เรื่อง\nอุบัติเหตุ", headline="อุบัติเหตุ", entities=[], corrections=[])
    assert cache.get("ข่าว เรื่อง อุบัติเหตุ") is not None


def test_switching_models_does_not_serve_the_previous_one(monkeypatch):
    from app.config import settings
    from app.services.enrichment_cache import EnrichmentCache

    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(settings, "ollama_model", "qwen2.5:7b")
    cache = EnrichmentCache("video")
    cache.put("ข่าวเรื่องอุบัติเหตุบนถนนสายหลัก", headline="เขียนโดย qwen", entities=[], corrections=[])

    monkeypatch.setattr(settings, "ollama_model", "llama3.1:8b")
    assert cache.get("ข่าวเรื่องอุบัติเหตุบนถนนสายหลัก") is None


def test_no_cache_means_no_file_is_written(tmp_path, monkeypatch):
    """Passing no cache disables it outright, which is what keeps a test run
    from writing into committed data."""
    from app.config import settings
    from app.nlp import llm_enrich
    from app.services.segmentation import enrich_with_llm

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(
        llm_enrich, "_complete", lambda prompt, *, timeout: "พาดหัวจากโมเดล"
    )

    segment = _segment("ตำรวจจับกุมผู้ต้องหาคดียาเสพติดรายใหญ่ยึดของกลางจำนวนมาก")
    assert enrich_with_llm(segment) == "model"
    assert segment.headline == "พาดหัวจากโมเดล"
    assert not (tmp_path / "enrichments").exists()


def test_the_cache_round_trips_through_disk(tmp_path, monkeypatch):
    from app.config import settings
    from app.services import enrichment_cache

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    text = "ตำรวจจับกุมผู้ต้องหาคดียาเสพติดรายใหญ่ยึดของกลางจำนวนมาก"

    cache = enrichment_cache.load("abc123")
    assert cache.get(text) is None
    cache.put(text, headline="จับกุมคดียาเสพติด", entities=[{"text": "ตำรวจ", "label": "ORGANIZATION"}], corrections=[("อำสิน", "ออมสิน")])
    enrichment_cache.save(cache)

    reloaded = enrichment_cache.load("abc123")
    entry = reloaded.get(text)
    assert entry is not None
    assert entry["headline"] == "จับกุมคดียาเสพติด"
    assert entry["corrections"] == [["อำสิน", "ออมสิน"]]
    # Thai is stored readable, so the committed file can be reviewed in a diff.
    assert "จับกุมคดียาเสพติด" in enrichment_cache.cache_path("abc123").read_text(encoding="utf-8")


def test_an_unsaved_cache_leaves_no_file(tmp_path, monkeypatch):
    """A read-only pass must not rewrite committed data."""
    from app.config import settings
    from app.services import enrichment_cache

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    cache = enrichment_cache.load("abc123")
    cache.get("ข่าวที่ไม่เคยถูกเขียนพาดหัว")
    enrichment_cache.save(cache)
    assert not enrichment_cache.cache_path("abc123").exists()


def test_a_corrupt_cache_file_is_ignored_not_fatal(tmp_path, monkeypatch):
    from app.config import settings
    from app.services import enrichment_cache

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    path = enrichment_cache.cache_path("abc123")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")

    cache = enrichment_cache.load("abc123")
    assert cache.entries == {}


def test_autosave_keeps_the_work_of_an_interrupted_run(tmp_path, monkeypatch):
    """A 76-story programme is 25 minutes of model time. Saving only at the end
    means a Ctrl+C at minute 24 throws all of it away."""
    from app.config import settings
    from app.services import enrichment_cache

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    cache = enrichment_cache.load("abc123", autosave=True)
    cache.put("ข่าวแรกเรื่องอุบัติเหตุบนถนน", headline="อุบัติเหตุบนถนน", entities=[], corrections=[])

    # On disk already, with no save() call of our own.
    reloaded = enrichment_cache.load("abc123")
    assert reloaded.get("ข่าวแรกเรื่องอุบัติเหตุบนถนน") is not None
    # And nothing left to write, so the final save is a no-op rather than a
    # redundant rewrite of the committed file.
    assert cache.dirty is False


def test_a_model_labelling_its_own_headline_is_stripped():
    """The prompt says ไม่ต้องขึ้นต้นว่า "พาดหัว" and 2 of the first 36 came back
    with exactly that. Instructions reduce this; they do not remove it."""
    from app.nlp.llm_enrich import _clean_headline

    assert _clean_headline("พาดหัว: โค้ชวอลเลย์บอลคว้าแชมป์") == "โค้ชวอลเลย์บอลคว้าแชมป์"
    assert (
        _clean_headline('พาดหัวข่าว : "เหตุยิงลิงแสมในสงขลา"') == "เหตุยิงลิงแสมในสงขลา"
    )
    # A bare word is not a label: a headline may genuinely open with สรุป.
    assert _clean_headline("สรุปสถานการณ์น้ำท่วมภาคเหนือ") == "สรุปสถานการณ์น้ำท่วมภาคเหนือ"
    # Nothing but a label is a failure, reported as one.
    assert _clean_headline("พาดหัว:") == ""


def test_a_label_only_completion_falls_back_to_the_extractive_headline(monkeypatch):
    from app.nlp import llm_enrich
    from app.services.segmentation import enrich_with_llm

    monkeypatch.setattr(llm_enrich, "_complete", lambda prompt, *, timeout: "พาดหัว:")
    segment = _segment("ตำรวจจับกุมผู้ต้องหาคดียาเสพติดรายใหญ่ยึดของกลางจำนวนมาก")
    segment.headline = "extractive headline"

    assert enrich_with_llm(segment) == "unavailable"
    assert segment.headline == "extractive headline"


def test_tightening_the_cleaner_reaches_already_cached_headlines(monkeypatch):
    """Cached work should improve with the rule, not stay frozen at the version
    that was current when it was written."""
    from app.nlp import llm_enrich
    from app.services.enrichment_cache import EnrichmentCache
    from app.services.segmentation import enrich_with_llm

    text = "ตำรวจจับกุมผู้ต้องหาคดียาเสพติดรายใหญ่ยึดของกลางจำนวนมาก"
    cache = EnrichmentCache("video")
    # Written before the label rule existed.
    cache.put(text, headline="พาดหัว: จับกุมคดียาเสพติด", entities=[], corrections=[])

    monkeypatch.setattr(
        llm_enrich,
        "_complete",
        lambda prompt, *, timeout: (_ for _ in ()).throw(
            AssertionError("model called on a cache hit")
        ),
    )
    segment = _segment(text)
    assert enrich_with_llm(segment, cache=cache) == "cache"
    assert segment.headline == "จับกุมคดียาเสพติด"


def test_disallowing_the_model_still_reads_the_cache(monkeypatch):
    """What keeps POST /broadcast/analyse a request rather than a 25-minute wait:
    a shipped programme still gets its written headlines, a new one does not stall
    the client waiting for 76 of them."""
    from app.nlp import llm_enrich
    from app.services.enrichment_cache import EnrichmentCache
    from app.services.segmentation import enrich_with_llm

    cached_text = "ตำรวจจับกุมผู้ต้องหาคดียาเสพติดรายใหญ่ยึดของกลางจำนวนมาก"
    fresh_text = "ฝนตกหนักต่อเนื่องทำให้เกิดน้ำท่วมในหลายจังหวัดภาคเหนือ"
    cache = EnrichmentCache("video")
    cache.put(cached_text, headline="จับกุมคดียาเสพติด", entities=[], corrections=[])

    def must_not_be_called(prompt, *, timeout):
        raise AssertionError("the model was called with allow_model=False")

    monkeypatch.setattr(llm_enrich, "_complete", must_not_be_called)

    hit = _segment(cached_text)
    assert enrich_with_llm(hit, cache=cache, allow_model=False) == "cache"
    assert hit.headline == "จับกุมคดียาเสพติด"

    miss = _segment(fresh_text, index=1)
    miss.headline = "extractive headline"
    assert enrich_with_llm(miss, cache=cache, allow_model=False) == "skipped"
    assert miss.headline == "extractive headline"


def test_a_skip_does_not_trip_the_circuit_breaker(monkeypatch):
    """A skip is a decision, not a failure: it must not stop the remaining
    stories from reading their cached headlines."""
    from app.config import settings
    from app.nlp import llm_enrich
    from app.services import segmentation
    from app.services.enrichment_cache import EnrichmentCache

    monkeypatch.setattr(settings, "llm_enrich_segments", True)
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(
        llm_enrich,
        "_complete",
        lambda prompt, *, timeout: (_ for _ in ()).throw(
            AssertionError("model called with allow_model=False")
        ),
    )

    texts = [
        f"ข่าวเรื่องที่ {index} ตำรวจจับกุมผู้ต้องหาคดียาเสพติดรายใหญ่ยึดของกลาง"
        for index in range(8)
    ]
    cache = EnrichmentCache("video")
    # Only the last story has a cached headline; the seven before it are skips.
    cache.put(texts[-1], headline="พาดหัวจากแคช", entities=[], corrections=[])

    segments = [_segment(text, index=index) for index, text in enumerate(texts)]
    segmentation.analyse_segments(segments, cache=cache, allow_model=False)

    assert segments[-1].headline == "พาดหัวจากแคช"
    assert segments[-1].enriched_by == "llm"
    assert all(segment.enriched_by == "" for segment in segments[:-1])


def test_a_restated_question_is_trimmed_to_the_headline():
    """The prompt asks สรุปว่าข้อความนี้พูดถึงเรื่องอะไร, and the model answers
    literally: "ข้อความนี้พูดถึงเรื่อง..." -- a sentence about the text instead of
    a headline for it. The part after the preamble is the headline."""
    from app.nlp.llm_enrich import _clean_headline

    assert (
        _clean_headline("ข้อความนี้พูดถึงเรื่องการลุยธุรกิจโรงแรม")
        == "การลุยธุรกิจโรงแรม"
    )
    assert _clean_headline("ข่าวนี้เกี่ยวกับเหตุยิงลิงแสมในสงขลา") == "เหตุยิงลิงแสมในสงขลา"
    # Label and preamble together, as they sometimes arrive.
    assert _clean_headline("พาดหัว: ข้อความนี้กล่าวถึงน้ำท่วมภาคเหนือ") == "น้ำท่วมภาคเหนือ"
    # A bare noun phrase that merely starts with ข่าว is not a preamble.
    assert _clean_headline("ข่าวเช้าช่อง8") == "ข่าวเช้าช่อง8"


def test_an_unusable_cached_headline_is_asked_again(monkeypatch):
    """A cached entry that cleans down to nothing is not a result. Marking the
    story as LLM-enriched with an empty headline would claim work that is not
    there."""
    from app.nlp import llm_enrich
    from app.services.enrichment_cache import EnrichmentCache
    from app.services.segmentation import enrich_with_llm

    text = "ตำรวจจับกุมผู้ต้องหาคดียาเสพติดรายใหญ่ยึดของกลางจำนวนมาก"
    cache = EnrichmentCache("video")
    cache.put(text, headline="พาดหัว:", entities=[], corrections=[])

    monkeypatch.setattr(
        llm_enrich, "_complete", lambda prompt, *, timeout: "จับกุมคดียาเสพติดรายใหญ่"
    )
    segment = _segment(text)
    assert enrich_with_llm(segment, cache=cache) == "model"
    assert segment.headline == "จับกุมคดียาเสพติดรายใหญ่"

    # And with no model to ask, the extractive headline stands rather than an
    # empty one.
    other = _segment(text, index=1)
    other.headline = "extractive headline"
    cache_only = EnrichmentCache("video")
    cache_only.put(text, headline="พาดหัว:", entities=[], corrections=[])
    assert enrich_with_llm(other, cache=cache_only, allow_model=False) == "skipped"
    assert other.headline == "extractive headline"
    assert other.enriched_by == ""


def test_written_headlines_are_counted_not_inferred():
    """The count is what the model was asked for. Subtracting cache hits from the
    enriched total could not tell a stale entry from a fresh call."""
    from app.services.enrichment_cache import EnrichmentCache

    cache = EnrichmentCache("video")
    assert cache.writes == 0
    cache.put("ข่าวเรื่องอุบัติเหตุบนถนนสายหลัก", headline="อุบัติเหตุ", entities=[], corrections=[])
    cache.put("ข่าวเรื่องน้ำท่วมภาคเหนือหลายจังหวัด", headline="น้ำท่วม", entities=[], corrections=[])
    assert cache.writes == 2
    assert cache.hits == 0


def test_a_headline_in_the_wrong_language_is_rejected():
    """Asked for a Thai headline, qwen2.5:7b answered story 57 of the second
    programme in Chinese. A headline in the wrong language is worse than a clumsy
    Thai one, because the timeline stops being readable."""
    from app.nlp.llm_enrich import _clean_headline

    assert _clean_headline("政坛对峙：反击与回应") == ""
    assert _clean_headline("Political standoff in parliament") == ""
    # And the mixed case, which is the one that actually reaches the screen: the
    # model starts in Thai and switches mid-sentence. Half a headline in the
    # wrong script is worse than none -- the extractive fallback was readable.
    assert _clean_headline("พาดหัวข่าว: ตำรวจยึด成果如下：新闻标题") == ""
    # Accented Latin is the same defect in a quieter costume: "démarchงบฯ กกต."
    # came back from a real programme.
    assert _clean_headline("démarchงบฯ กกต. ผู้ทรงเกียรติยืนยัน") == ""
    # Plain ASCII inside a Thai headline is ordinary and stays.
    assert _clean_headline("จัดระเบียบ Data Center ตามมติครม.") == "จัดระเบียบ Data Center ตามมติครม."


def test_only_the_first_line_of_a_completion_is_the_headline():
    """A model that explains its own answer would otherwise have the explanation
    folded in by the whitespace collapse."""
    from app.nlp.llm_enrich import _clean_headline

    completion = "ตำรวจจับผู้ต้องหาคดียาเสพติด" + NEWLINE * 2 + "นี่คือพาดหัวที่กระชับและตรงประเด็น"
    assert _clean_headline(completion) == "ตำรวจจับผู้ต้องหาคดียาเสพติด"
    # Leading blank lines are skipped rather than read as an empty headline.
    assert _clean_headline(NEWLINE + "  " + NEWLINE + "น้ำท่วมภาคเหนือ") == "น้ำท่วมภาคเหนือ"


def test_a_cached_headline_does_not_freeze_the_entities(monkeypatch):
    """The entities come from the rule-based extractor on every run. A cache
    entry that carried them would mean a later improvement to entity extraction
    silently never reached a cached story."""
    from app.nlp import llm_enrich
    from app.services.enrichment_cache import EnrichmentCache
    from app.services.segmentation import enrich_with_llm

    text = "ตำรวจจับกุมผู้ต้องหาคดียาเสพติดรายใหญ่ยึดของกลางจำนวนมาก"
    cache = EnrichmentCache("video")
    cache.put(text, headline="จับกุมคดียาเสพติด", entities=[], corrections=[])

    monkeypatch.setattr(
        llm_enrich,
        "_complete",
        lambda prompt, *, timeout: (_ for _ in ()).throw(
            AssertionError("model called on a cache hit")
        ),
    )
    segment = _segment(text)
    fresh = [{"text": "ตำรวจ", "label": "ORGANIZATION"}]
    segment.entities = fresh

    assert enrich_with_llm(segment, cache=cache) == "cache"
    assert segment.headline == "จับกุมคดียาเสพติด"
    assert segment.entities == fresh


def test_the_reported_totals_add_up_to_the_stories(monkeypatch):
    """A report that does not add up undermines every other number in it. A
    rejected cache entry is a write, not a reuse -- counting the read as both
    made a 36-story programme report "1 written, 36 reused"."""
    from app.nlp import llm_enrich
    from app.services.enrichment_cache import EnrichmentCache
    from app.services.segmentation import enrich_with_llm

    good = "ตำรวจจับกุมผู้ต้องหาคดียาเสพติดรายใหญ่ยึดของกลางจำนวนมาก"
    stale = "ฝนตกหนักต่อเนื่องทำให้เกิดน้ำท่วมในหลายจังหวัดภาคเหนือ"
    cache = EnrichmentCache("video")
    cache.put(good, headline="จับกุมคดียาเสพติด", entities=[], corrections=[])
    cache.put(stale, headline="พาดหัว:", entities=[], corrections=[])
    cache.writes = 0  # the setup is not part of what we are counting

    monkeypatch.setattr(
        llm_enrich, "_complete", lambda prompt, *, timeout: "น้ำท่วมภาคเหนือ"
    )
    assert enrich_with_llm(_segment(good), cache=cache) == "cache"
    assert enrich_with_llm(_segment(stale, index=1), cache=cache) == "model"

    assert cache.hits == 1
    assert cache.writes == 1
    assert cache.hits + cache.writes == 2


# --------------------------------------------------------------------------
# Writing headlines after the fact
#
# An imported video gets extractive headlines, because 20-40 seconds a story is
# far longer than a request may hold. Telling the user to go and run a CLI is a
# limitation with instructions attached, so the work runs as a job instead.
# --------------------------------------------------------------------------


@pytest.fixture
def broadcast_db(db_session):
    """A stored programme with three stories and no written headlines."""
    from app.models.broadcast import NewsSegment, VideoTranscript
    from app.models.chat import ChatStream

    stream = ChatStream(
        video_id="jobVideo001",
        url="https://www.youtube.com/watch?v=jobVideo001",
        title="ข่าวเช้า",
        collector="transcript",
    )
    db_session.add(stream)
    db_session.flush()
    transcript = VideoTranscript(
        stream_id=stream.id, source="youtube-asr:th-orig", duration_ms=600_000
    )
    db_session.add(transcript)
    db_session.flush()
    for index in range(3):
        db_session.add(
            NewsSegment(
                transcript_id=transcript.id,
                stream_id=stream.id,
                position=index,
                start_ms=index * 60_000,
                end_ms=(index + 1) * 60_000,
                headline=f"extractive {index}",
                transcript_text=(
                    f"ข่าวเรื่องที่ {index} "
                    "ตำรวจจับกุมผู้ต้องหาคดียาเสพติดรายใหญ่ยึดของกลางจำนวนมาก"
                ),
                entities=[{"text": "ตำรวจ", "label": "ORGANIZATION"}],
            )
        )
    db_session.commit()
    return db_session


@pytest.fixture
def job_registry(monkeypatch, tmp_path, broadcast_db):
    """Point the job service at the test database and clear its registry.

    ``data_dir`` is redirected too, and that is not belt and braces: the job
    always passes an enrichment cache, so without this the tests write headlines
    into the committed ``data/enrichments/`` -- which they did, and the next test
    then read them back as cache hits and quietly stopped exercising the model
    path at all.
    """
    from app.config import settings
    from app.services import headline_jobs

    monkeypatch.setattr(settings, "data_dir", tmp_path)
    monkeypatch.setattr(
        headline_jobs, "SessionLocal", lambda: _NonClosingSession(broadcast_db)
    )
    headline_jobs._jobs.clear()
    yield headline_jobs
    headline_jobs._jobs.clear()


class _NonClosingSession:
    """Hands the worker the test session without letting it close it."""

    def __init__(self, session):
        self._session = session

    def __enter__(self):
        return self._session

    def __exit__(self, *exc):
        return False


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    import time

    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_a_job_writes_every_missing_headline(job_registry, broadcast_db, monkeypatch):
    from app.models.broadcast import NewsSegment
    from app.nlp import llm_enrich

    monkeypatch.setattr(
        llm_enrich, "_complete", lambda prompt, *, timeout: "จับกุมคดียาเสพติดรายใหญ่"
    )

    job = job_registry.start("jobVideo001")
    assert job.total == 3
    assert _wait_for(lambda: not job.running), job.state
    assert job.state == "done"
    assert job.written == 3

    rows = broadcast_db.query(NewsSegment).all()
    assert all(row.enriched_by == "llm" for row in rows)
    assert all(row.headline == "จับกุมคดียาเสพติดรายใหญ่" for row in rows)
    # The extractor's entities are untouched: in headline-only mode the model
    # returns none, and an empty list must not overwrite them.
    assert all(row.entities == [{"text": "ตำรวจ", "label": "ORGANIZATION"}] for row in rows)


def test_only_one_programme_is_written_at_a_time(job_registry, monkeypatch):
    """Not a queueing nicety: the local model holds about 6 GB, and two at once
    is how this machine runs out of memory."""
    import threading

    from app.nlp import llm_enrich

    release = threading.Event()
    monkeypatch.setattr(
        llm_enrich,
        "_complete",
        lambda prompt, *, timeout: (release.wait(3), "พาดหัวข่าว")[1],
    )

    job = job_registry.start("jobVideo001")
    assert _wait_for(lambda: job.state == "running")
    try:
        with pytest.raises(job_registry.JobRejected, match="Already writing"):
            job_registry.start("someOtherVid")
        # The same video is not a second job -- a double-clicked button is safe.
        assert job_registry.start("jobVideo001") is job
    finally:
        release.set()
        _wait_for(lambda: not job.running)


def test_a_job_for_an_unknown_video_is_refused(job_registry):
    with pytest.raises(job_registry.JobRejected, match="No stories"):
        job_registry.start("neverImported")


def test_a_finished_programme_is_refused(job_registry, broadcast_db):
    from app.models.broadcast import NewsSegment

    for row in broadcast_db.query(NewsSegment).all():
        row.enriched_by = "llm"
    broadcast_db.commit()

    with pytest.raises(job_registry.JobRejected, match="already has a written"):
        job_registry.start("jobVideo001")


def test_a_dead_provider_fails_the_job_rather_than_grinding_through_it(
    job_registry, monkeypatch
):
    from app.nlp import llm_enrich

    def unreachable(prompt, *, timeout):
        raise llm_enrich.LLMUnavailable("connection refused")

    monkeypatch.setattr(llm_enrich, "_complete", unreachable)

    job = job_registry.start("jobVideo001")
    assert _wait_for(lambda: not job.running), job.state
    assert job.state == "failed"
    assert "Ollama" in job.note


def test_cancelling_keeps_what_was_already_written(
    job_registry, broadcast_db, monkeypatch
):
    """Every headline is committed as it lands, which is what makes stopping
    safe rather than wasteful."""
    import threading

    from app.models.broadcast import NewsSegment
    from app.nlp import llm_enrich

    first_done = threading.Event()
    hold = threading.Event()
    calls = []

    def slow(prompt, *, timeout):
        calls.append(prompt)
        if len(calls) == 1:
            return "พาดหัวแรก"
        first_done.set()
        hold.wait(3)
        return "พาดหัวที่สอง"

    monkeypatch.setattr(llm_enrich, "_complete", slow)
    try:
        job = job_registry.start("jobVideo001")
        assert _wait_for(first_done.is_set)
        job_registry.cancel("jobVideo001")
    finally:
        hold.set()

    assert _wait_for(lambda: not job.running), job.state
    assert job.state == "cancelled"

    written = [
        row for row in broadcast_db.query(NewsSegment).all() if row.enriched_by == "llm"
    ]
    assert written, "the headline finished before cancelling should be kept"
    assert any(row.headline == "พาดหัวแรก" for row in written)


def test_the_headline_endpoints_report_what_the_caller_can_act_on(job_registry):
    """404 for a programme nobody started, 409 for one with nothing to do --
    both are conditions the caller can do something about."""
    from fastapi.testclient import TestClient

    from app.main import create_app

    with TestClient(create_app()) as client:
        assert client.get("/api/broadcast/headlines/neverStarted").status_code == 404
        assert client.delete("/api/broadcast/headlines/neverStarted").status_code == 404

        response = client.post(
            "/api/broadcast/headlines", json={"video_id": "neverImported"}
        )
        assert response.status_code == 409
        assert "Import the video first" in response.json()["detail"]
