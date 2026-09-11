"""The NLP pipeline.

Runs one document through every stage and returns both the results and the
intermediate values, so the NLP Analysis page can show exactly what the models
consumed rather than a re-derived approximation.

Stages, matching the diagram in the UI::

    raw text -> cleaning -> tokenisation -> stopword removal -> features
             -> topic -> sentiment -> keywords -> summary -> entities
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

from app.nlp.base import Entity, Keyword, Prediction, Summary
from app.nlp.preprocessing import PreprocessResult
from app.nlp.registry import PIPELINE_VERSION, NLPComponents, get_components


@dataclass(slots=True)
class AnalysisResult:
    """Everything one pipeline run produced."""

    title: str
    text: str
    preprocessing: PreprocessResult
    topic: Prediction
    sentiment: Prediction
    keywords: list[Keyword] = field(default_factory=list)
    summary: Summary | None = None
    entities: list[Entity] = field(default_factory=list)
    processing_ms: float = 0.0
    model_versions: dict[str, str] = field(default_factory=dict)

    @property
    def keyword_words(self) -> list[str]:
        return [keyword.word for keyword in self.keywords]

    def content_hash(self) -> str:
        """Stable hash of the analysed text, used to deduplicate documents."""
        return hashlib.sha256(
            self.preprocessing.cleaned_text.encode("utf-8")
        ).hexdigest()

    def as_dict(self) -> dict:
        """Serialise for the API and for storage."""
        pre = self.preprocessing
        return {
            "title": self.title,
            "summary": self.summary.text if self.summary else "",
            "topic": self.topic.label,
            "topic_confidence": round(self.topic.confidence, 4),
            "sentiment": self.sentiment.label,
            "sentiment_confidence": round(self.sentiment.confidence, 4),
            "keywords": [keyword.as_dict() for keyword in self.keywords],
            "entities": [entity.as_dict() for entity in self.entities],
            "topic_probabilities": {
                key: round(value, 4) for key, value in self.topic.probabilities.items()
            },
            "sentiment_probabilities": {
                key: round(value, 4)
                for key, value in self.sentiment.probabilities.items()
            },
            "cleaned_text": pre.cleaned_text,
            "tokens": pre.tokens,
            "filtered_tokens": pre.filtered_tokens,
            "sentences": pre.sentences,
            "token_count": pre.token_count,
            "unique_token_count": pre.unique_token_count,
            "stopword_removed_count": pre.stopword_removed_count,
            "processing_ms": round(self.processing_ms, 2),
            "model_versions": self.model_versions,
            "pipeline_version": PIPELINE_VERSION,
        }


class NLPPipeline:
    """Orchestrates the NLP components for one document."""

    def __init__(self, components: NLPComponents | None = None) -> None:
        self.components = components or get_components()

    def analyse(
        self,
        text: str,
        *,
        title: str | None = None,
        with_summary: bool = True,
        with_entities: bool = True,
        keyword_count: int | None = None,
        keyword_exclude: frozenset[str] | None = None,
    ) -> AnalysisResult:
        """Run the full pipeline over one document.

        Args:
            text: The document body.
            title: Optional headline. Included in the classified text and used
                to boost keywords, because a headline word is usually what the
                document is about.
            with_summary: Skip summarisation for short inputs where it adds
                nothing (per-message chat sentiment, for instance).
            with_entities: Skip entity extraction when not needed.
            keyword_count: Override how many keywords to return.
            keyword_exclude: Extra terms barred from becoming keywords. Used for
                broadcast transcripts, where spoken filler would otherwise
                dominate -- see app.nlp.stopwords.BROADCAST_FILLER.
        """
        started = time.perf_counter()
        parts = self.components

        # Title carries strong signal, so it is analysed together with the body.
        combined = f"{title}\n{text}" if title else (text or "")

        preprocessing = parts.tokenizer.preprocess(
            combined, with_sentences=with_summary
        )

        # Topic and keywords use the filtered stream (filler removed); sentiment
        # uses the full stream so negation survives.
        topic = parts.topic.predict(
            preprocessing.filtered_text, preprocessing.filtered_tokens
        )
        sentiment = parts.sentiment.predict(
            preprocessing.cleaned_text, preprocessing.tokens
        )

        keywords = parts.keywords.extract(
            preprocessing.cleaned_text,
            preprocessing.tokens,
            keyword_count,
            title=title,
            exclude=keyword_exclude,
        )

        summary: Summary | None = None
        if with_summary:
            # Summarise the body only: repeating the headline is not a summary.
            summary = parts.summarizer.summarize(text or combined)

        entities: list[Entity] = []
        if with_entities:
            entities = parts.entities.extract(
                preprocessing.cleaned_text, preprocessing.tokens
            )

        return AnalysisResult(
            title=title or "",
            text=text or "",
            preprocessing=preprocessing,
            topic=topic,
            sentiment=sentiment,
            keywords=keywords,
            summary=summary,
            entities=entities,
            processing_ms=(time.perf_counter() - started) * 1000,
            model_versions=parts.model_versions(),
        )

    def sentiment_only(self, texts: list[str]) -> list[Prediction]:
        """Per-message sentiment for a batch of short chat messages.

        Uses the *chat* sentiment backend rather than the document one. The
        difference matters: the news-trained model mislabels short informal chat
        (see ``nlp_chat_sentiment_backend`` in config for the measurements),
        while the lexicon was built for exactly this register.
        """
        return self.components.chat_sentiment.predict_many(texts)


def get_pipeline() -> NLPPipeline:
    """Build a pipeline over the cached components."""
    return NLPPipeline()
