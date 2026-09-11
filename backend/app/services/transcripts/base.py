"""Transcript contract shared by every speech-to-text source.

A transcript provider turns a video into timed Thai text. Everything downstream
-- segmentation, topic classification, the timeline UI -- depends only on this
contract, so swapping YouTube's ASR for a locally-run Whisper model means writing
one class and registering it.

The unit is a :class:`TranscriptCue`: a short run of speech with a start time and
an end time in milliseconds from the beginning of the video. Timestamps are the
whole point of this layer -- they are what lets a news story link back to the
moment it was spoken.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


class TranscriptUnavailable(RuntimeError):
    """Raised when a transcript cannot be produced.

    Carries a message meant for the user, so an API handler can say *why*
    (no captions published, no network, ASR not installed) rather than 500.
    """


# Non-speech events YouTube's ASR emits in brackets: [เพลง] music,
# [เสียงหัวเราะ] laughter, [เสียงปรบมือ] applause. They are not speech, but
# their *position* is useful -- in a Thai news programme a music sting almost
# always marks a transition between stories, so they are kept as flags rather
# than silently dropped.
NON_SPEECH = re.compile(r"\[[^\]]{1,24}\]")

# YouTube's ASR marks a change of speaker with ">>" at the start of a line.
# Also useful as a boundary hint: a new anchor usually means a new story.
SPEAKER_CHANGE = re.compile(r"^\s*>>\s*")


@dataclass(slots=True)
class TranscriptCue:
    """One run of transcribed speech, timed against the video."""

    start_ms: int
    end_ms: int
    text: str
    # True when this cue was a bracketed non-speech marker such as [เพลง].
    non_speech: bool = False
    # True when the source marked a change of speaker at this cue.
    speaker_change: bool = False

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    @property
    def start_seconds(self) -> int:
        return self.start_ms // 1000

    def as_dict(self) -> dict:
        """Compact form for JSON storage. Keys are short because a long
        programme has tens of thousands of these."""
        payload: dict = {"t": self.start_ms, "e": self.end_ms, "x": self.text}
        if self.non_speech:
            payload["n"] = 1
        if self.speaker_change:
            payload["s"] = 1
        return payload

    @classmethod
    def from_dict(cls, payload: dict) -> TranscriptCue:
        return cls(
            start_ms=int(payload.get("t", 0)),
            end_ms=int(payload.get("e", 0)),
            text=str(payload.get("x", "")),
            non_speech=bool(payload.get("n")),
            speaker_change=bool(payload.get("s")),
        )


@dataclass(slots=True)
class Transcript:
    """A whole video's timed transcript."""

    video_id: str
    source: str  # e.g. "youtube-asr:th-orig" or "whisper:small"
    language: str
    cues: list[TranscriptCue] = field(default_factory=list)
    # Video length when the provider knows it; otherwise the last cue's end.
    duration_ms: int = 0

    def __len__(self) -> int:
        return len(self.cues)

    @property
    def speech_cues(self) -> list[TranscriptCue]:
        """Only cues that carry actual words."""
        return [cue for cue in self.cues if cue.text and not cue.non_speech]

    @property
    def character_count(self) -> int:
        return sum(len(cue.text) for cue in self.speech_cues)

    @property
    def covered_ms(self) -> int:
        """Total time spanned by speech, which is less than the video length
        whenever there is music or silence."""
        return sum(cue.duration_ms for cue in self.speech_cues)

    def text_between(self, start_ms: int, end_ms: int) -> str:
        """Joined speech text overlapping ``[start_ms, end_ms)``.

        Thai is written without spaces, so cues are concatenated directly --
        inserting spaces would create word boundaries the tokeniser then has to
        undo, and would change the tokenisation of the joined text.
        """
        parts = [
            cue.text
            for cue in self.cues
            if cue.text
            and not cue.non_speech
            and cue.start_ms < end_ms
            and cue.end_ms > start_ms
        ]
        return "".join(parts)

    def as_dict(self) -> dict:
        return {
            "video_id": self.video_id,
            "source": self.source,
            "language": self.language,
            "duration_ms": self.duration_ms,
            "cues": [cue.as_dict() for cue in self.cues],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> Transcript:
        return cls(
            video_id=str(payload.get("video_id", "")),
            source=str(payload.get("source", "unknown")),
            language=str(payload.get("language", "th")),
            duration_ms=int(payload.get("duration_ms", 0)),
            cues=[TranscriptCue.from_dict(c) for c in payload.get("cues", [])],
        )


@runtime_checkable
class TranscriptProvider(Protocol):
    """Any source of timed speech text for a video."""

    name: str

    def fetch(self, source: str) -> Transcript:
        """Produce a transcript for ``source`` (a URL or a video id).

        Raises:
            TranscriptUnavailable: when no transcript can be produced.
        """
        ...


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------


def clean_cue_text(raw: str) -> tuple[str, bool, bool]:
    """Normalise one raw caption string.

    Returns ``(text, is_non_speech, had_speaker_change)``.

    YouTube's json3 output is noisier than it looks: roughly half of its events
    are bare newlines used for on-screen layout, and the real text events carry
    a ">>" prefix whenever the speaker changes. Both are handled here so every
    provider produces the same shape.
    """
    text = (raw or "").replace(" ", " ")

    speaker_change = bool(SPEAKER_CHANGE.search(text))
    text = SPEAKER_CHANGE.sub("", text)

    markers = NON_SPEECH.findall(text)
    stripped = NON_SPEECH.sub("", text).strip()

    # Collapse the newlines and runs of spaces YouTube uses for layout.
    stripped = re.sub(r"\s+", " ", stripped).strip()

    # A cue that was *only* a marker is non-speech; one that had a marker plus
    # words keeps the words.
    if markers and not stripped:
        return "", True, speaker_change
    return stripped, False, speaker_change


def resolve_cue_ends(
    starts_and_durations: list[tuple[int, int]],
    *,
    max_gap_ms: int = 10_000,
) -> list[int]:
    """Work out an end time for each cue.

    Needed because the reported durations overlap heavily -- in a measured
    sample, 10,772 of 11,148 consecutive pairs overlapped, because YouTube sizes
    each caption for how long it stays *on screen*, not for how long its words
    are spoken. Using those durations directly would make every cue overlap its
    neighbours and corrupt any time-based grouping.

    So a cue ends when the next one starts, capped by its own reported duration
    and by ``max_gap_ms`` so a long silence before the next cue does not stretch
    it to cover minutes of nothing.
    """
    ends: list[int] = []
    for index, (start, duration) in enumerate(starts_and_durations):
        if index + 1 < len(starts_and_durations):
            next_start = starts_and_durations[index + 1][0]
        else:
            next_start = start + max(duration, 1000)

        limit = start + duration if duration > 0 else next_start
        end = min(next_start, limit) if limit > start else next_start
        # Never negative, never longer than max_gap_ms.
        end = max(start, min(end, start + max_gap_ms))
        ends.append(end)
    return ends
