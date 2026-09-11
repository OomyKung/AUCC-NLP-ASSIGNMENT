r"""Capture a still frame from the video at a given moment.

Why storyboards rather than ffmpeg
----------------------------------
The obvious way to grab a frame is ``ffmpeg -ss <t> -i <stream> -frames:v 1``.
That needs the ffmpeg binary, which is not installed here and is not a Python
dependency anyone can ``pip install`` reliably on Windows, and it also means
downloading part of a multi-gigabyte video stream for one thumbnail.

YouTube already publishes what is needed. Every video has **storyboards**: sprite
sheets of evenly-spaced frames, which is what the player shows when you scrub the
timeline. For a 249-minute programme the ``sb0`` track is a grid of 3x3 tiles at
320x180 per sheet, 167 sheets, so there is a real frame from the video roughly
every **10 seconds** -- finer than the story boundaries this project detects.

So a "screen capture at 12:30" is: work out which sheet and which tile covers
12:30, fetch that one sheet (a few kilobytes), crop the tile, save it. No video
download, no ffmpeg, and the frame is genuinely from the video at that moment.

The trade-off, stated plainly: 320x180 is thumbnail resolution. It is the right
size for a timeline card and too small to read on-screen text. If ffmpeg is
available, :func:`capture_frame` is the one place to swap in a full-resolution
implementation -- everything else depends only on the returned path.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

from app.config import settings
from app.services.collectors.base import extract_video_id, watch_url


class FrameUnavailable(RuntimeError):
    """Raised when no frame can be produced for a timestamp."""


@dataclass(slots=True)
class StoryboardTrack:
    """One storyboard level: a grid of frames spread over the whole video."""

    format_id: str
    tile_width: int
    tile_height: int
    rows: int
    columns: int
    # (sheet url, seconds this sheet covers) in play order.
    fragments: list[tuple[str, float]]

    @property
    def tiles_per_sheet(self) -> int:
        return max(1, self.rows * self.columns)

    @property
    def seconds_per_tile(self) -> float:
        """Time between consecutive frames, i.e. the capture resolution."""
        if not self.fragments:
            return 0.0
        return self.fragments[0][1] / self.tiles_per_sheet

    def locate(self, second: float) -> tuple[str, int, int]:
        """Which sheet and tile covers ``second``.

        Returns ``(sheet_url, row, column)``.
        """
        if not self.fragments:
            raise FrameUnavailable("storyboard track has no sheets")

        remaining = max(0.0, second)
        for url, duration in self.fragments:
            if remaining < duration or duration <= 0:
                index = int(remaining // max(self.seconds_per_tile, 1e-6))
                index = min(index, self.tiles_per_sheet - 1)
                return url, index // self.columns, index % self.columns
            remaining -= duration

        # Past the end: clamp to the final tile of the final sheet.
        url, _duration = self.fragments[-1]
        last = self.tiles_per_sheet - 1
        return url, last // self.columns, last % self.columns


def _frames_dir(video_id: str) -> Path:
    return settings.data_dir / "frames" / video_id


def frame_path(video_id: str, second: int) -> Path:
    """Where a captured frame is cached."""
    return _frames_dir(video_id) / f"{second:06d}.jpg"


def fetch_storyboards(source: str, *, timeout: int | None = None) -> list[StoryboardTrack]:
    """List the video's storyboard tracks, finest resolution first."""
    video_id = extract_video_id(source)
    command = [
        sys.executable,
        "-m",
        "yt_dlp",
        "--skip-download",
        "--dump-single-json",
        "--no-warnings",
        watch_url(video_id),
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout or settings.ytdlp_timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise FrameUnavailable(f"Timed out reading video metadata for {video_id}.") from exc
    except FileNotFoundError as exc:  # pragma: no cover - environment issue
        raise FrameUnavailable("yt-dlp is not installed.") from exc

    if not completed.stdout.strip():
        raise FrameUnavailable(
            f"Could not read video metadata for {video_id}: "
            f"{(completed.stderr or '').strip()[:200]}"
        )

    try:
        info = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise FrameUnavailable(f"Video metadata for {video_id} was not JSON.") from exc

    tracks: list[StoryboardTrack] = []
    for fmt in info.get("formats") or []:
        if not str(fmt.get("format_id", "")).startswith("sb"):
            continue
        fragments = [
            (str(fragment.get("url")), float(fragment.get("duration") or 0.0))
            for fragment in fmt.get("fragments") or []
            if fragment.get("url")
        ]
        if not fragments:
            continue
        tracks.append(
            StoryboardTrack(
                format_id=str(fmt.get("format_id")),
                tile_width=int(fmt.get("width") or 0),
                tile_height=int(fmt.get("height") or 0),
                rows=int(fmt.get("rows") or 1),
                columns=int(fmt.get("columns") or 1),
                fragments=fragments,
            )
        )

    if not tracks:
        raise FrameUnavailable(f"{video_id} publishes no storyboards.")

    # Finest time resolution first, then largest tile.
    tracks.sort(key=lambda t: (t.seconds_per_tile, -t.tile_width))
    return tracks


def capture_frame(
    source: str,
    second: int,
    *,
    track: StoryboardTrack | None = None,
    overwrite: bool = False,
) -> Path:
    """Save the video's frame at ``second`` and return its path.

    Cached: a frame already on disk is returned untouched, so re-analysing a
    programme does not re-download sheets.
    """
    video_id = extract_video_id(source)
    target = frame_path(video_id, second)
    if target.is_file() and not overwrite:
        return target

    chosen = track or fetch_storyboards(source)[0]
    url, row, column = chosen.locate(float(second))

    try:
        response = httpx.get(url, timeout=30.0, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise FrameUnavailable(f"Could not download storyboard sheet: {exc}") from exc

    try:
        import io

        from PIL import Image

        sheet = Image.open(io.BytesIO(response.content))
        # Trust the sheet's real size over the reported tile size: some tracks
        # report a nominal tile size that does not divide the sheet exactly.
        tile_width = sheet.width // chosen.columns
        tile_height = sheet.height // chosen.rows
        box = (
            column * tile_width,
            row * tile_height,
            (column + 1) * tile_width,
            (row + 1) * tile_height,
        )
        tile = sheet.crop(box).convert("RGB")
        target.parent.mkdir(parents=True, exist_ok=True)
        tile.save(target, format="JPEG", quality=82, optimize=True)
    except ImportError as exc:
        raise FrameUnavailable(
            "Pillow is required to crop video frames: "
            "pip install -r requirements.txt"
        ) from exc
    except (OSError, ValueError) as exc:
        raise FrameUnavailable(f"Could not crop storyboard tile: {exc}") from exc

    return target


def capture_frames(source: str, seconds: list[int], *, overwrite: bool = False) -> dict[int, Path]:
    """Capture several frames, reading the storyboard metadata once.

    Sheets are fetched at most once each even when several requested timestamps
    fall inside the same sheet, which they routinely do: sheets cover ~90s and
    news stories run 1-5 minutes.
    """
    if not seconds:
        return {}

    video_id = extract_video_id(source)
    wanted = sorted(set(seconds))

    # Serve everything already on disk first, and only reach for the network if
    # something is genuinely missing. Fetching storyboard metadata up front
    # would make a fully-cached rebuild require a network connection, which
    # defeats the point of caching the frames at all -- `seed.py` rebuilds the
    # timelines from committed snapshots and must work with no connectivity.
    captured: dict[int, Path] = {}
    missing: list[int] = []
    for second in wanted:
        path = frame_path(video_id, second)
        if path.is_file() and not overwrite:
            captured[second] = path
        else:
            missing.append(second)

    if not missing:
        return captured

    try:
        track = fetch_storyboards(source)[0]
    except FrameUnavailable:
        # Offline, or no storyboards published. Whatever was cached still
        # stands; the rest of the timeline simply has no thumbnails.
        return captured

    for second in missing:
        try:
            captured[second] = capture_frame(
                source, second, track=track, overwrite=overwrite
            )
        except FrameUnavailable:
            # One unavailable frame must not abandon the rest; a missing
            # thumbnail is a cosmetic loss, not a failed analysis.
            continue
    return captured


def describe_resolution(track: StoryboardTrack) -> str:
    """Human-readable summary, for the README and /api/pipeline."""
    return (
        f"{track.format_id}: {track.tile_width}x{track.tile_height} "
        f"every {track.seconds_per_tile:.0f}s "
        f"({len(track.fragments)} sheets, {track.rows}x{track.columns} tiles)"
    )
