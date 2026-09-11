"""Transcript providers -- the single place a speech-to-text source is chosen.

Mirrors ``app/nlp/registry.py``: nothing outside this module constructs a
provider, so replacing YouTube's ASR with a local Whisper model means writing one
adapter and changing one ``.env`` value.

Selection is capability aware. Whisper needs ``openai-whisper`` and a lot of CPU
time; when it is requested but not installed, the registry falls back to YouTube
captions and records why, rather than failing a request.
"""

from __future__ import annotations

from app.config import settings
from app.services.transcripts.base import (
    Transcript,
    TranscriptCue,
    TranscriptProvider,
    TranscriptUnavailable,
)
from app.services.transcripts.youtube import YouTubeCaptionProvider

__all__ = [
    "Transcript",
    "TranscriptCue",
    "TranscriptProvider",
    "TranscriptUnavailable",
    "YouTubeCaptionProvider",
    "get_transcript_provider",
    "provider_status",
]

# Why the active provider is what it is, for /api/pipeline to report honestly.
_STATUS: dict[str, str] = {}


def get_transcript_provider(requested: str | None = None) -> TranscriptProvider:
    """Build the configured transcript provider.

    Falls back to YouTube captions when a heavier backend is unavailable, and
    records the reason in :func:`provider_status`.
    """
    global _STATUS
    choice = requested or settings.transcript_backend

    if choice == "whisper":
        try:
            from app.services.transcripts.whisper import WhisperProvider

            provider = WhisperProvider()
            _STATUS = {"requested": choice, "active": provider.name, "note": ""}
            return provider
        except (ImportError, TranscriptUnavailable) as exc:
            note = f"whisper unavailable: {exc}"
    elif choice == "youtube":
        note = ""
    else:
        note = f"unknown transcript backend {choice!r}"

    fallback = YouTubeCaptionProvider()
    _STATUS = {"requested": choice, "active": fallback.name, "note": note}
    return fallback


def provider_status() -> dict[str, str]:
    """What the last :func:`get_transcript_provider` call resolved to."""
    if not _STATUS:
        get_transcript_provider()
    return dict(_STATUS)
