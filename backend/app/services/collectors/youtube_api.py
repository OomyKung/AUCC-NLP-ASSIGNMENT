"""Chat collector using the official YouTube Data API v3.

Provided as a licence-clean alternative to the yt-dlp collector, selectable
with ``COLLECTOR=youtube_api`` once ``YOUTUBE_API_KEY`` is set.

Two limitations are inherent to the official API and worth stating plainly in
a project report:

1. ``liveChatMessages.list`` only serves **currently active** broadcasts. The
   chat of a finished stream is not retrievable, so this collector cannot seed
   a reproducible dataset the way the yt-dlp collector can.
2. Every call costs quota (10,000 units/day by default).

Implemented with ``httpx`` directly rather than ``google-api-python-client`` to
avoid pulling a large dependency tree for two endpoints.
"""

from __future__ import annotations

from datetime import UTC, datetime

import httpx

from app.config import settings
from app.services.collectors.base import (
    CollectorError,
    CollectResult,
    RawChatMessage,
    StreamInfo,
    extract_video_id,
    watch_url,
)

_API_ROOT = "https://www.googleapis.com/youtube/v3"
# The API caps page size at 2000; stay well under it to keep responses small.
_PAGE_SIZE = 500


class YouTubeApiCollector:
    """Collect live chat from an active broadcast via the official API."""

    name = "youtube_api"

    def __init__(self, api_key: str | None = None, timeout: float = 30.0) -> None:
        self.api_key = api_key or settings.youtube_api_key
        self.timeout = timeout

    def _get(self, client: httpx.Client, path: str, params: dict) -> dict:
        """Issue one API call, translating failures into CollectorError."""
        try:
            response = client.get(
                f"{_API_ROOT}/{path}",
                params={**params, "key": self.api_key},
                timeout=self.timeout,
            )
        except httpx.HTTPError as exc:
            raise CollectorError(f"Could not reach the YouTube API: {exc}") from exc

        if response.status_code == 403:
            raise CollectorError(
                "The YouTube API rejected the request (403). The key may be invalid, "
                "restricted, or out of daily quota."
            )
        if response.status_code != 200:
            raise CollectorError(
                f"The YouTube API returned {response.status_code}: {response.text[:200]}"
            )
        return response.json()

    def collect(self, source: str, *, limit: int | None = None) -> CollectResult:
        """Collect chat for an active broadcast."""
        if not self.api_key:
            raise CollectorError(
                "COLLECTOR=youtube_api requires YOUTUBE_API_KEY in .env. "
                "Use COLLECTOR=ytdlp to collect without an API key."
            )

        video_id = extract_video_id(source)
        limit = limit or settings.max_messages_per_import

        with httpx.Client() as client:
            video = self._get(
                client,
                "videos",
                {"part": "snippet,liveStreamingDetails", "id": video_id},
            )
            items = video.get("items") or []
            if not items:
                raise CollectorError(f"Video {video_id} not found or not public.")

            snippet = items[0].get("snippet") or {}
            details = items[0].get("liveStreamingDetails") or {}
            live_chat_id = details.get("activeLiveChatId")
            if not live_chat_id:
                raise CollectorError(
                    "This video has no active live chat. The official API cannot read "
                    "the chat replay of a finished stream -- use COLLECTOR=ytdlp for that."
                )

            stream = StreamInfo(
                video_id=video_id,
                url=watch_url(video_id),
                title=snippet.get("title"),
                channel=snippet.get("channelTitle"),
                is_live=True,
                collector=self.name,
            )

            messages: list[RawChatMessage] = []
            page_token: str | None = None
            while len(messages) < limit:
                params = {
                    "part": "snippet,authorDetails",
                    "liveChatId": live_chat_id,
                    "maxResults": min(_PAGE_SIZE, limit - len(messages)),
                }
                if page_token:
                    params["pageToken"] = page_token

                payload = self._get(client, "liveChat/messages", params)
                for item in payload.get("items") or []:
                    message = _message_from_item(item)
                    if message is not None:
                        messages.append(message)

                page_token = payload.get("nextPageToken")
                # The API returns the same token when it has served everything
                # currently available; stop rather than polling in a tight loop.
                if not page_token or not payload.get("items"):
                    break

        if not messages:
            raise CollectorError(
                "The live chat is active but has no messages available yet."
            )

        messages.sort(key=lambda m: m.published_at)
        return CollectResult(stream=stream, messages=messages)


def _message_from_item(item: dict) -> RawChatMessage | None:
    """Convert one API item into a normalised message."""
    snippet = item.get("snippet") or {}
    text = str(snippet.get("displayMessage") or "").strip()
    if not text:
        return None

    try:
        published_at = datetime.fromisoformat(
            str(snippet.get("publishedAt", "")).replace("Z", "+00:00")
        )
    except ValueError:
        published_at = datetime.now(UTC)

    author = (item.get("authorDetails") or {}).get("displayName")
    return RawChatMessage(
        text=text,
        published_at=published_at,
        message_id=item.get("id"),
        author=author,
    )
