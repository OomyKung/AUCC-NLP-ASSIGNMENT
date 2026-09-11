"""Application configuration.

All tunables live here and are overridable from ``.env`` so that NLP backends,
chat-windowing behaviour and data sources can be changed without touching code.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/config.py -> backend/app -> backend -> <project root>
APP_DIR = Path(__file__).resolve().parent
BACKEND_DIR = APP_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    """Typed application settings, populated from environment / ``.env``."""

    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ------------------------------------------------------------------ app
    app_name: str = "Thai News Intelligence"
    app_version: str = "1.0.0"
    debug: bool = True
    api_prefix: str = "/api"

    # CORS origins for the Vite dev server.
    cors_origins: list[str] = [
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:4173",
        "http://127.0.0.1:4173",
    ]

    # ------------------------------------------------------------- database
    database_url: str = f"sqlite:///{(BACKEND_DIR / 'thai_news.db').as_posix()}"
    sql_echo: bool = False

    # ------------------------------------------------------------ file paths
    model_dir: Path = BACKEND_DIR / "models"
    data_dir: Path = PROJECT_ROOT / "data"
    chat_snapshot_dir: Path = PROJECT_ROOT / "data" / "chat_snapshots"

    # ----------------------------------------------------- NLP backend wiring
    # Swapping any model = change one value here (see app/nlp/registry.py).
    nlp_tokenizer: Literal["newmm", "newmm-safe", "longest", "mm"] = "newmm"
    # Both tasks blend the trained model with their rule-based baseline, because
    # the two make different mistakes. Weights are fitted by ensemble.py on
    # out-of-fold training predictions; see Results in the README.
    #   topic     : CV macro-F1 0.764 -> 0.780
    #   sentiment : CV macro-F1 0.716 -> 0.725
    #
    # Sentiment's weight fitted to exactly 1.0 on the smaller 747-row dataset,
    # meaning the lexicon added nothing and the blend was pure overhead. With
    # more data and a calibrated LinearSVC base it now fits to 0.85, so the
    # lexicon is contributing again. The value is measured each time rather
    # than assumed, which is why it is a config entry and not a constant.
    nlp_topic_backend: Literal["sklearn", "blend", "transformer"] = "blend"
    nlp_sentiment_backend: Literal[
        "sklearn", "blend", "lexicon", "transformer"
    ] = "blend"
    # Per-message chat sentiment uses a SEPARATE backend from news sentiment,
    # because the domain gap between news prose and live-chat register is large
    # and was measured in both directions:
    #
    #   * The news-trained model on chat: labelled 59% of real chat messages
    #     positive and called "แย่ที่สุด" ("the worst") positive.
    #   * A social-media-trained model on news: 0.287 accuracy.
    #
    # So each domain gets a model trained on its own register. "wisesight" is
    # trained by train_chat_sentiment.py on the Wisesight corpus (~23.5k real
    # Thai social-media messages) and scores 0.710 accuracy / 0.673 macro-F1 on
    # that corpus's official test split. On those same 2,614 rows the lexicon
    # it replaces scores 0.436 macro-F1 and the news-trained model 0.319, so
    # the gain is measured against both alternatives on one test set. The
    # lexicon remains the fallback when the artefact is absent, so a fresh
    # clone still works.
    nlp_chat_sentiment_backend: Literal[
        "wisesight", "sklearn", "lexicon", "transformer"
    ] = "wisesight"
    nlp_summarizer_backend: Literal["extractive", "llm"] = "extractive"

    # Blend weights: P = alpha * P_model + (1 - alpha) * P_rules. Fitted by
    # ensemble.py on out-of-fold training predictions -- never on the held-out
    # set. alpha=1.0 is the trained model alone, 0.0 the rule baseline alone.
    # Re-fit these with `python ensemble.py` after retraining.
    nlp_topic_blend_alpha: float = 0.90
    nlp_sentiment_blend_alpha: float = 0.85
    nlp_ner_backend: Literal["rules", "pythainlp"] = "rules"

    # ------------------------------------------------- transcripts (video audio)
    # Where the spoken news content comes from.
    #   youtube : YouTube's own ASR of the original Thai audio ("automatic
    #             captions", th-orig track). Already computed, arrives with
    #             millisecond timings, and a four-hour programme downloads in
    #             about a second.
    #   whisper : transcribe the audio ourselves with openai-whisper. Better
    #             provenance for a paper, but hours of CPU per programme and an
    #             extra ~2 GB of dependencies, so it is opt-in.
    transcript_backend: Literal["youtube", "whisper"] = "youtube"
    whisper_model: str = "small"

    # Story segmentation. A news item is rarely shorter than a minute, and
    # without a floor the depth score produces runs of boundaries around one
    # transition and the timeline fills with eight-second "stories".
    segment_min_seconds: int = 60
    segment_block_seconds: int = 30

    # Where a story's still frame comes from.
    #   ffmpeg     : a real 1280x720 frame from the video stream, on which the
    #                burnt-in story banner and clock are readable. ~5s each.
    #                The binary ships with imageio-ffmpeg, so no manual install.
    #   storyboard : a 320x180 tile from YouTube's scrubbing sprite sheets.
    #                ~0.1s each, and the automatic fallback when ffmpeg or the
    #                stream is unavailable.
    frame_backend: Literal["ffmpeg", "storyboard"] = "ffmpeg"
    frame_height: int = 720

    # Top-down splitting of spans whose halves classify as different topics.
    # OFF because it was measured and it made things worse. It uses the topic
    # classifier as its criterion, and the classifier is the unreliable part
    # here -- on 40 seconds of out-of-domain broadcast speech it wobbles, so
    # splitting on its disagreement fragments coherent stories. One five-minute
    # report on a shot monkey became four segments with two wrong labels, where
    # refinement alone kept it as a single, correctly-labelled `crime` story.
    #
    # It does improve the "halves disagree" rate (43% vs 54%), but that metric
    # rewards fragmentation, which is precisely the failure -- a reminder that a
    # proxy measured with a noisy instrument can move the wrong way.
    segment_split_mixed: bool = False

    # Optional Hugging Face models (only loaded when a transformer backend is on).
    hf_topic_model: str = "airesearch/wangchanberta-base-att-spm-uncased"
    hf_sentiment_model: str = "phoner45/wangchan-sentiment-thai-text-model"

    # ---------------------------------------------------------- summarizer
    summary_min_sentences: int = 2
    summary_max_sentences: int = 4

    # Optional external LLM summarizer. Disabled unless a key is present.
    llm_api_key: str | None = None
    llm_model: str = "claude-sonnet-5"
    llm_base_url: str = "https://api.anthropic.com/v1/messages"

    # Which model serves the enrichment.
    #   ollama    : a model running locally. Free, no key, no account, and no
    #               data leaves the machine. Needs `ollama serve` and a pulled
    #               model. ~20s per story on CPU.
    #   anthropic : the hosted API. Needs LLM_API_KEY and costs money.
    llm_provider: Literal["ollama", "anthropic"] = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen2.5:7b"

    # Let the model rewrite Thai names it thinks the ASR got wrong.
    #
    # OFF, and it must stay off for local models. Measured on qwen2.5:7b, which
    # was explicitly instructed not to guess:
    #     อนุทินชาวรกูล      -> อนุทินชื่นกล่าว      (invented)
    #     อัถสิทธิ์เวชชาชีวะ -> อัชสิทธิ์เวชชาชีวะ  (still wrong)
    #     อำสินสักสิภาพร...  -> อำพันสิทธิ์จันทวิสูตร (invented)
    # A 7B model does not know individual Thai politicians or athletes, so it
    # produces plausible-looking wrong names -- strictly worse than a visibly
    # garbled one, because a reader cannot tell it happened. Headlines need no
    # such knowledge, which is why those are on by default and this is not.
    llm_correct_names: bool = False

    # Let the LLM write the story headlines (and, with llm_correct_names,
    # repair names). Inert when no provider answers -- three failures and the
    # rest of the programme falls back to the extractive headline. It exists
    # because every local approach to the name half was
    # measured and could not do it: Thai soundex does not match the manglings
    # (the ASR inserts syllables), PyThaiNLP's 22k name corpora do not contain
    # the names, and cross-referencing the chat and transcript only works when
    # the ASR happened to get it right somewhere else -- which for
    # "ศศิภาพร จันทวิสูตร" it never does. See app/nlp/llm_enrich.py.
    llm_enrich_segments: bool = True

    # -------------------------------------------------- keyword extraction
    keyword_min_count: int = 5
    keyword_max_count: int = 10

    # ------------------------------------------------- chat ingestion (YouTube)
    collector: Literal["ytdlp", "youtube_api", "file"] = "ytdlp"
    youtube_api_key: str | None = None

    # A chat window closes at whichever bound is hit first. The message count
    # is the bound that normally governs; the time bound is a safety valve that
    # stops a quiet stream producing one enormous window. Real Thai news chat
    # runs ~2 messages/minute, so a tight time bound would fire first and keep
    # windows far below the intended size. Set minutes to 0 to disable it.
    chat_window_size: int = 200
    chat_window_minutes: int = 60
    # Windows smaller than this are discarded as too thin to analyse.
    chat_window_min_messages: int = 5

    # Live monitoring poll interval, seconds.
    live_poll_seconds: int = 10
    # Safety cap on a single import so a huge VOD cannot hang the request.
    max_messages_per_import: int = 20_000
    ytdlp_timeout_seconds: int = 300

    @property
    def has_llm(self) -> bool:
        """True when an external LLM summarizer is actually configured."""
        return bool(self.llm_api_key and self.llm_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    """Cached settings accessor (import this, not the class)."""
    return Settings()


settings = get_settings()
