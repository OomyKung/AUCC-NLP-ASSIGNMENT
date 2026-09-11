"""Transcript from YouTube's own speech recognition of the video audio.

This is the default provider, and it is worth being precise about what it is:
YouTube runs ASR over the published audio and exposes the result as
"automatic captions". Requesting ``th-orig`` asks for the track transcribed from
the **original Thai audio** rather than a machine translation of it, so the text
really is derived from what the newsreader said -- not from chat, not from the
description.

Why this rather than running our own ASR by default:

* It is already computed, so a four-hour programme downloads in about a second
  instead of taking hours of CPU.
* It arrives with millisecond timings, which is the entire point of this layer.
* Quality on Thai news audio is good enough to classify and segment. Spot
  checks read cleanly ("หลังจากที่ลิงถูกยิงจนตกลงมาตาย"), with occasional
  name errors.

Its limits are real and stated in the README: it is not our model, it does not
exist for every video, and it has no word-level confidence we can trust. The
Whisper provider alongside this one covers the case where an own-ASR result is
needed.

yt-dlp is invoked as a subprocess via ``python -m yt_dlp``, matching the chat
collector, so both share the same executable resolution and timeout handling.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from app.config import settings
from app.services.collectors.base import extract_video_id, watch_url
from app.services.transcripts.base import (
    Transcript,
    TranscriptCue,
    TranscriptUnavailable,
    clean_cue_text,
    resolve_cue_ends,
)

# Preference order. "th-orig" is the track transcribed from the original Thai
# audio; plain "th" can be a translation of another language's track, which
# would put us a translation away from what was actually said.
LANGUAGE_PREFERENCE = ("th-orig", "th")


class YouTubeCaptionProvider:
    """Timed Thai text from YouTube's automatic captions."""

    name = "youtube-asr"

    def __init__(self, timeout: int | None = None) -> None:
        self._timeout = timeout or settings.ytdlp_timeout_seconds

    @staticmethod
    def _base_command() -> list[str]:
        return [sys.executable, "-m", "yt_dlp"]

    def _download(self, url: str, language: str, target: Path) -> Path | None:
        """Ask yt-dlp for one caption track. Returns the file, or None."""
        command = [
            *self._base_command(),
            "--skip-download",
            "--write-auto-subs",
            "--sub-langs",
            language,
            "--sub-format",
            "json3",
            "--no-warnings",
            "-o",
            str(target / "%(id)s.%(ext)s"),
            url,
        ]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self._timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TranscriptUnavailable(
                f"Timed out after {self._timeout}s fetching captions for {url}."
            ) from exc
        except FileNotFoundError as exc:  # pragma: no cover - environment issue
            raise TranscriptUnavailable(
                "yt-dlp is not installed. Install it with: "
                "pip install -r requirements.txt"
            ) from exc

        files = sorted(target.glob("*.json3"))
        if files:
            return files[0]

        # No file: decide whether this is "no captions" or a real failure, so
        # the caller can try the next language instead of aborting.
        stderr = (completed.stderr or "").strip()
        if stderr and "no subtitles" not in stderr.lower():
            lowered = stderr.lower()
            for fragment, message in (
                ("private video", "This video is private."),
                ("unavailable", "This video is unavailable."),
                ("sign in", "This video requires sign-in."),
                ("getaddrinfo", "No network connection."),
                ("failed to resolve", "No network connection."),
            ):
                if fragment in lowered:
                    raise TranscriptUnavailable(message)
        return None

    @staticmethod
    def _parse(path: Path, video_id: str, source: str) -> Transcript:
        """Turn a json3 caption file into cues."""
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TranscriptUnavailable(
                f"Caption file for {video_id} could not be read: {exc}"
            ) from exc

        events = [event for event in payload.get("events") or [] if event.get("segs")]
        if not events:
            raise TranscriptUnavailable(f"Caption track for {video_id} is empty.")

        # Pass 1: clean the text and drop the layout-only events. Roughly half
        # of what YouTube emits is a bare newline used for on-screen placement.
        cleaned: list[tuple[int, int, str, bool, bool]] = []
        for event in events:
            raw = "".join(seg.get("utf8", "") for seg in event.get("segs") or [])
            text, non_speech, speaker_change = clean_cue_text(raw)
            if not text and not non_speech:
                continue
            cleaned.append(
                (
                    int(event.get("tStartMs") or 0),
                    int(event.get("dDurationMs") or 0),
                    text,
                    non_speech,
                    speaker_change,
                )
            )

        if not cleaned:
            raise TranscriptUnavailable(
                f"Caption track for {video_id} contained no speech."
            )

        cleaned.sort(key=lambda row: row[0])
        ends = resolve_cue_ends([(row[0], row[1]) for row in cleaned])

        cues = [
            TranscriptCue(
                start_ms=row[0],
                end_ms=end,
                text=row[2],
                non_speech=row[3],
                speaker_change=row[4],
            )
            for row, end in zip(cleaned, ends, strict=True)
        ]
        return Transcript(
            video_id=video_id,
            source=source,
            language="th",
            cues=cues,
            duration_ms=cues[-1].end_ms if cues else 0,
        )

    def fetch(self, source: str) -> Transcript:
        """Download and parse the best available Thai caption track."""
        video_id = extract_video_id(source)
        url = watch_url(video_id)

        with tempfile.TemporaryDirectory(prefix="thai-news-subs-") as directory:
            target = Path(directory)
            for language in LANGUAGE_PREFERENCE:
                # Each attempt gets a clean directory so a previous language's
                # file cannot be mistaken for this one's.
                attempt = target / language
                attempt.mkdir(parents=True, exist_ok=True)
                path = self._download(url, language, attempt)
                if path is not None:
                    return self._parse(path, video_id, f"{self.name}:{language}")

        raise TranscriptUnavailable(
            f"No Thai automatic captions are published for {video_id}. "
            "Tried: " + ", ".join(LANGUAGE_PREFERENCE) + ". "
            "Set TRANSCRIPT_BACKEND=whisper to transcribe the audio locally instead."
        )
