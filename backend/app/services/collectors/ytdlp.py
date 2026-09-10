"""Chat collector backed by yt-dlp.

This is the default collector because it needs no API key and, unlike the
official YouTube Data API, it can read the chat *replay* of an already-finished
stream -- which is what makes seeding a reproducible dataset possible.

yt-dlp is invoked as a subprocess via ``python -m yt_dlp`` (rather than the
``yt-dlp`` executable) so it always resolves inside the active virtualenv.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from app.config import settings
from app.services.collectors.base import (
    CollectorError,
    CollectResult,
    StreamInfo,
    extract_video_id,
    watch_url,
)
from app.services.collectors.ytchat_parser import parse_chat_file


class YtDlpCollector:
    """Collect YouTube live chat (live or replay) using yt-dlp."""

    name = "ytdlp"

    def __init__(self, timeout: int | None = None) -> None:
        self.timeout = timeout or settings.ytdlp_timeout_seconds

    # ------------------------------------------------------------------ internals
    @staticmethod
    def _base_command() -> list[str]:
        return [sys.executable, "-m", "yt_dlp", "--no-warnings", "--ignore-config"]

    def _fetch_metadata(self, url: str) -> dict:
        """Read video metadata as JSON.

        Raises:
            CollectorError: when the video is unavailable or yt-dlp is broken.
        """
        command = [*self._base_command(), "--dump-single-json", "--skip-download", url]
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise CollectorError(
                f"Timed out after {self.timeout}s reading video metadata."
            ) from exc
        except FileNotFoundError as exc:  # pragma: no cover - env misconfiguration
            raise CollectorError(
                "Could not run yt-dlp. Reinstall it with: pip install -r requirements.txt"
            ) from exc

        if completed.returncode != 0 or not completed.stdout.strip():
            raise CollectorError(_clean_ytdlp_error(completed.stderr))

        try:
            return json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise CollectorError("yt-dlp returned metadata that could not be parsed.") from exc

    def _download_chat(self, url: str, target_dir: Path) -> Path | None:
        """Download the live_chat subtitle track into ``target_dir``.

        A live stream never "finishes", so a timeout is expected rather than
        exceptional: whatever chat has been written by then is still usable.
        """
        command = [
            *self._base_command(),
            "--skip-download",
            "--write-subs",
            "--sub-langs",
            "live_chat",
            "-o",
            str(target_dir / "%(id)s.%(ext)s"),
            url,
        ]
        try:
            subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=self.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            # Expected for live streams: fall through and use partial output.
            pass

        files = sorted(target_dir.glob("*.live_chat.json"))
        return files[0] if files else None

    # -------------------------------------------------------------------- public
    def collect(self, source: str, *, limit: int | None = None) -> CollectResult:
        """Collect chat for a YouTube URL or bare video id."""
        video_id = extract_video_id(source)
        url = watch_url(video_id)
        limit = limit or settings.max_messages_per_import

        metadata = self._fetch_metadata(url)
        stream = StreamInfo(
            video_id=metadata.get("id") or video_id,
            url=metadata.get("webpage_url") or url,
            title=metadata.get("title"),
            channel=metadata.get("channel") or metadata.get("uploader"),
            is_live=bool(metadata.get("is_live")),
            collector=self.name,
        )

        with tempfile.TemporaryDirectory(prefix="thainews-chat-") as tmp:
            chat_path = self._download_chat(url, Path(tmp))
            if chat_path is None:
                raise CollectorError(
                    "No live chat available for this video. Live chat must be enabled, "
                    "and finished streams only keep a chat replay if the uploader left "
                    "it on. Try a different stream, or import a saved snapshot instead."
                )
            messages = parse_chat_file(chat_path, limit=limit)

        if not messages:
            raise CollectorError(
                "The chat track downloaded but contained no readable messages."
            )

        messages.sort(key=lambda m: m.published_at)
        return CollectResult(stream=stream, messages=messages)


def _clean_ytdlp_error(stderr: str) -> str:
    """Turn yt-dlp's stderr into one actionable sentence."""
    lines = [line.strip() for line in (stderr or "").splitlines() if line.strip()]
    for line in lines:
        if "ERROR:" in line:
            detail = line.split("ERROR:", 1)[1].strip()
            return f"yt-dlp could not read this video: {detail}"
    if lines:
        return f"yt-dlp could not read this video: {lines[-1]}"
    return "yt-dlp could not read this video (no error detail returned)."
