"""Response shapes for the spoken-content timeline."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SegmentOut(BaseModel):
    """One story in a programme."""

    id: int
    position: int
    start_ms: int
    end_ms: int
    duration_ms: int
    timecode: str = Field(description="H:MM:SS label as a viewer reads it")

    headline: str
    summary: str
    topic: str
    topic_label: str = Field(description="Thai display name for the topic")
    topic_color: str
    topic_confidence: float
    sentiment: str
    sentiment_label: str
    sentiment_confidence: float

    keywords: list[str] = []
    entities: list[dict] = []

    # A bounded slice of what was said. The full text of a five-minute story is
    # thousands of characters and the list endpoint returns a hundred of them,
    # so sending it whole would make the timeline payload megabytes.
    transcript_text_preview: str = ""

    # Why the segmenter cut here, so the UI can explain a split rather than
    # presenting it as an oracle.
    boundary_reasons: list[str] = []
    boundary_confidence: float = 0.0

    youtube_url: str = Field(description="Deep link to the moment this story starts")
    frame_url: str | None = Field(
        default=None, description="Captured video frame at that moment"
    )


class TranscriptOut(BaseModel):
    """Metadata about a programme's transcript, without the cues."""

    source: str
    language: str
    duration_ms: int
    cue_count: int
    character_count: int


class ProgrammeOut(BaseModel):
    """A video and the stories found inside it."""

    video_id: str
    title: str | None = None
    channel: str | None = None
    url: str
    transcript: TranscriptOut | None = None
    segment_count: int = 0
    segments: list[SegmentOut] = []


class SegmentListResponse(BaseModel):
    """Paginated segments across programmes."""

    total: int
    limit: int
    offset: int
    items: list[SegmentOut] = []


class AnalyseVideoRequest(BaseModel):
    """Request to transcribe and segment a video."""

    url: str = Field(description="YouTube URL or 11-character video id")
    with_frames: bool = Field(
        default=True, description="Capture a still frame at each story start"
    )


class AnalyseVideoResponse(BaseModel):
    """What one analysis produced."""

    video_id: str
    transcript_source: str
    duration_ms: int
    cue_count: int
    segment_count: int
    frames_captured: int
    frame_note: str = ""
