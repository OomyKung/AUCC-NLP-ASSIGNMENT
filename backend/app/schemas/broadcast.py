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

    # ASR name spellings that were corrected, as [asr_form, corrected] pairs,
    # so the page can show the correction instead of applying it invisibly.
    name_corrections: list[list[str]] = []
    enriched_by: str = ""

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
    # Work the LLM could still do here. Surfaced so the timeline can offer it
    # rather than leaving an imported programme quietly reading worse than the
    # ones that ship with the repository.
    pending_headlines: int = 0
    pending_reactions: int = 0
    # Stories that have any chat at all, which is what makes the viewers' view
    # worth offering on this programme.
    chat_segments: int = 0
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
    write_headlines: bool = Field(
        default=False,
        description=(
            "Let the LLM write a headline for any story that has none cached. "
            "Off by default because it takes about 20 seconds per story, which "
            "is longer than an HTTP client will wait on a long programme; run "
            "analyse_video.py for that. Cached headlines are used either way."
        ),
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
    headlines_cached: int = 0
    headlines_written: int = 0


class EnrichmentJobRequest(BaseModel):
    """Ask the LLM to write what an import could not wait for."""

    video_id: str = Field(min_length=5, max_length=40)


class EnrichmentJobStatus(BaseModel):
    """Progress of one programme's LLM work.

    Counted in units rather than stories: a story needing both a headline and a
    chat summary is two, so the progress bar means what it says. Reported rather
    than guessed at, because at 10-40 seconds a unit a long programme is most of
    an hour and the UI needs something honest to show while it happens.
    """

    video_id: str
    # queued | running | done | failed | cancelled
    state: str
    total: int
    done: int
    headlines: int
    reactions: int
    latest: str = ""
    note: str = ""
    elapsed_seconds: float = 0.0
    eta_seconds: float | None = None


class ChatSample(BaseModel):
    """One viewer message, as shown beside the story it reacted to."""

    text: str
    sentiment: str
    confidence: float = 0.0
    offset_ms: int


class SegmentReaction(BaseModel):
    """One story's chat, without the messages themselves.

    What the timeline needs to render every card at once: how many viewers
    spoke, how they felt, and the one line describing it. The messages and the
    keywords cost tokenising thousands of rows, so they come per story from
    ``/broadcast/segments/{id}/chat`` when a card is opened.
    """

    segment_id: int
    total: int = 0
    sentiment_counts: dict[str, int] = {}
    mood: str = ""
    summary: str = ""


class ReactionOut(BaseModel):
    """What the audience said while one story was on air."""

    total: int = 0
    sentiment_counts: dict[str, int] = {}
    mood: str = ""
    keywords: list[str] = []
    samples: list[ChatSample] = []
    # One line written by the LLM. Empty until someone asks for it, and on the
    # many videos that have no chat at all.
    summary: str = ""
