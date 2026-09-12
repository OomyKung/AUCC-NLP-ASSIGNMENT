"""NLP component registry -- the single place models are chosen.

This is the seam that makes the NLP layer swappable (spec section 9). Nothing
outside this module constructs a backend, so replacing a model means writing one
adapter class, registering it here, and changing one ``.env`` value.

Selection is also *capability aware*: when a backend is configured but its
trained artefact is missing, the registry falls back to the rule-based baseline
and records why. That is what lets a fresh clone produce real results instead of
crashing or showing placeholders, while ``/api/health`` still reports honestly
which backend is actually serving.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

from app.config import settings
from app.nlp.base import (
    EntityRecognizer,
    KeywordExtractor,
    SentimentBackend,
    Summarizer,
    Tokenizer,
    TopicBackend,
)
from app.nlp.backends.gazetteer_topic import GazetteerTopicBackend
from app.nlp.backends.lexicon_sentiment import LexiconSentimentBackend
from app.nlp.entities import PyThaiNLPEntityRecognizer, RuleEntityRecognizer
from app.nlp.keyword_extractor import TfidfKeywordExtractor
from app.nlp.summarizer import ExtractiveSummarizer, LLMSummarizer
from app.nlp.tokenizer import get_tokenizer

PIPELINE_VERSION = "1.0.0"


@dataclass(slots=True)
class BackendStatus:
    """Which backend is actually serving each stage, and why."""

    requested: str
    active: str
    trained: bool
    note: str = ""


@dataclass(slots=True)
class NLPComponents:
    """Every component the pipeline needs, resolved once."""

    tokenizer: Tokenizer
    topic: TopicBackend
    sentiment: SentimentBackend
    # Separate backend for short chat messages; see config for why.
    chat_sentiment: SentimentBackend
    keywords: KeywordExtractor
    summarizer: Summarizer
    entities: EntityRecognizer
    status: dict[str, BackendStatus] = field(default_factory=dict)

    def model_versions(self) -> dict[str, str]:
        """Names of the active backends, stored with each analysis so results
        stay traceable after a model swap."""
        return {
            "tokenizer": self.tokenizer.name,
            "topic": self.topic.name,
            "sentiment": self.sentiment.name,
            "chat_sentiment": self.chat_sentiment.name,
            "keywords": self.keywords.name,
            "summarizer": self.summarizer.name,
            "entities": self.entities.name,
            "pipeline": PIPELINE_VERSION,
        }


# --------------------------------------------------------------------------
# Individual builders
# --------------------------------------------------------------------------


def _build_topic() -> tuple[TopicBackend, BackendStatus]:
    """Resolve the topic backend, preferring a trained model when present."""
    requested = settings.nlp_topic_backend

    if requested == "sklearn":
        try:
            from app.nlp.backends.sklearn_topic import SklearnTopicBackend

            backend = SklearnTopicBackend.load()
            if backend is not None:
                return backend, BackendStatus(requested, backend.name, True)
            note = "no trained model found (run: python train.py)"
        except ImportError as exc:
            note = f"sklearn backend unavailable: {exc}"
    elif requested == "blend":
        try:
            from app.nlp.backends.blended import BlendedTopicBackend

            backend = BlendedTopicBackend.load(settings.nlp_topic_blend_alpha)
            if backend is not None:
                return backend, BackendStatus(requested, backend.name, True)
            note = "no trained model to blend (run: python train.py)"
        except (ImportError, ValueError) as exc:
            note = f"blend backend unavailable: {exc}"
    elif requested == "transformer":
        try:
            from app.nlp.backends.hf_transformer import HFTopicBackend

            backend = HFTopicBackend()
            return backend, BackendStatus(requested, backend.name, True)
        except Exception as exc:
            note = f"transformer backend unavailable: {exc}"
    else:
        note = f"unknown backend {requested!r}"

    fallback = GazetteerTopicBackend()
    return fallback, BackendStatus(requested, fallback.name, False, note)


def _build_sentiment(
    requested: str | None = None,
) -> tuple[SentimentBackend, BackendStatus]:
    """Resolve a sentiment backend, preferring a trained model when present."""
    requested = requested or settings.nlp_sentiment_backend

    if requested == "lexicon":
        backend = LexiconSentimentBackend()
        return backend, BackendStatus(requested, backend.name, False)

    if requested in {"sklearn", "wisesight"}:
        # "wisesight" is the same adapter pointed at the chat-domain artefact
        # that train_chat_sentiment.py writes, rather than the news one.
        kind = "chat_sentiment" if requested == "wisesight" else "sentiment"
        script = (
            "python train_chat_sentiment.py"
            if requested == "wisesight"
            else "python train.py"
        )
        try:
            from app.nlp.backends.sklearn_sentiment import SklearnSentimentBackend

            backend = SklearnSentimentBackend.load(kind=kind)
            if backend is not None:
                return backend, BackendStatus(requested, backend.name, True)
            note = f"no trained model found (run: {script})"
        except ImportError as exc:
            note = f"sklearn backend unavailable: {exc}"
    elif requested in {"blend", "chat-blend"}:
        # "chat-blend" blends the *chat* artefact with the lexicon, using its
        # own weight: the two domains needed different ones (news 0.85, chat
        # 0.75), each fitted out-of-fold on its own corpus.
        chat = requested == "chat-blend"
        try:
            from app.nlp.backends.blended import BlendedSentimentBackend

            backend = BlendedSentimentBackend.load(
                settings.nlp_chat_sentiment_blend_alpha
                if chat
                else settings.nlp_sentiment_blend_alpha,
                kind="chat_sentiment" if chat else "sentiment",
            )
            if backend is not None:
                return backend, BackendStatus(requested, backend.name, True)
            script = "train_chat_sentiment.py" if chat else "train.py"
            note = f"no trained model to blend (run: python {script})"
        except (ImportError, ValueError) as exc:
            note = f"blend backend unavailable: {exc}"
    elif requested == "transformer":
        try:
            from app.nlp.backends.hf_transformer import HFSentimentBackend

            backend = HFSentimentBackend()
            return backend, BackendStatus(requested, backend.name, True)
        except Exception as exc:
            note = f"transformer backend unavailable: {exc}"
    else:
        note = f"unknown backend {requested!r}"

    fallback = LexiconSentimentBackend()
    return fallback, BackendStatus(requested, fallback.name, False, note)


def _build_summarizer() -> tuple[Summarizer, BackendStatus]:
    """Resolve the summarizer; the LLM path needs a configured key."""
    requested = settings.nlp_summarizer_backend

    if requested == "llm":
        if settings.has_llm:
            backend = LLMSummarizer()
            return backend, BackendStatus(requested, backend.name, True)
        fallback = ExtractiveSummarizer()
        return fallback, BackendStatus(
            requested, fallback.name, False, "LLM_API_KEY is not set"
        )

    backend = ExtractiveSummarizer()
    return backend, BackendStatus(requested, backend.name, False)


def _build_entities() -> tuple[EntityRecognizer, BackendStatus]:
    """Resolve the entity recognizer."""
    requested = settings.nlp_ner_backend
    if requested == "pythainlp":
        backend = PyThaiNLPEntityRecognizer()
        return backend, BackendStatus(requested, backend.name, True)
    rules = RuleEntityRecognizer()
    return rules, BackendStatus(requested, rules.name, False)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def get_components() -> NLPComponents:
    """Build and cache every NLP component.

    Cached because loading models and building the tokenizer trie is expensive
    relative to a request. Call :func:`reset_components` after training so the
    freshly written artefacts are picked up.
    """
    topic, topic_status = _build_topic()
    sentiment, sentiment_status = _build_sentiment()
    chat_sentiment, chat_sentiment_status = _build_sentiment(
        settings.nlp_chat_sentiment_backend
    )
    summarizer, summarizer_status = _build_summarizer()
    entities, entities_status = _build_entities()

    keywords = TfidfKeywordExtractor.load()
    keyword_status = BackendStatus(
        requested="tfidf",
        active=keywords.name,
        trained=keywords.is_fitted,
        note=""
        if keywords.is_fitted
        else "no corpus IDF cached; using length-weighted term frequency",
    )

    tokenizer = get_tokenizer()

    return NLPComponents(
        tokenizer=tokenizer,
        topic=topic,
        sentiment=sentiment,
        chat_sentiment=chat_sentiment,
        keywords=keywords,
        summarizer=summarizer,
        entities=entities,
        status={
            "tokenizer": BackendStatus(
                settings.nlp_tokenizer, tokenizer.name, True
            ),
            "topic": topic_status,
            "sentiment": sentiment_status,
            "chat_sentiment": chat_sentiment_status,
            "keywords": keyword_status,
            "summarizer": summarizer_status,
            "entities": entities_status,
        },
    )


def reset_components() -> None:
    """Clear the cache so newly trained models are loaded on next use."""
    get_components.cache_clear()
