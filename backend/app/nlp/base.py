"""Protocols every NLP component implements.

These are the seams that make the NLP layer swappable (spec section 9). API
handlers, services and the UI depend only on these shapes, never on a concrete
model, so replacing TF-IDF with a fine-tuned transformer means writing one
adapter class and changing one ``.env`` value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable


# --------------------------------------------------------------------------
# Result types
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Prediction:
    """A single-label prediction with its full probability distribution.

    ``probabilities`` is what makes a result explainable in the UI, so a backend
    that cannot produce calibrated probabilities should still return a
    normalised distribution over its label set.
    """

    label: str
    confidence: float
    probabilities: dict[str, float] = field(default_factory=dict)
    model_name: str = "unknown"

    def top(self, count: int = 4) -> list[tuple[str, float]]:
        """The highest-scoring labels, for the detail page's probability table."""
        return sorted(self.probabilities.items(), key=lambda kv: -kv[1])[:count]


@dataclass(slots=True)
class Keyword:
    """One extracted keyword and its score."""

    word: str
    score: float

    def as_dict(self) -> dict:
        return {"word": self.word, "score": round(self.score, 4)}


@dataclass(slots=True)
class Entity:
    """One named entity."""

    text: str
    label: str

    def as_dict(self) -> dict:
        return {"text": self.text, "label": self.label}


@dataclass(slots=True)
class Summary:
    """A generated summary."""

    text: str
    sentences: list[str] = field(default_factory=list)
    method: str = "extractive"


# --------------------------------------------------------------------------
# Component protocols
# --------------------------------------------------------------------------


@runtime_checkable
class Tokenizer(Protocol):
    """Splits Thai text into tokens."""

    name: str

    def tokenize(self, text: str) -> list[str]:
        """Return the tokens of ``text``, excluding whitespace."""
        ...

    def sentences(self, text: str) -> list[str]:
        """Split ``text`` into sentences."""
        ...


@runtime_checkable
class TopicBackend(Protocol):
    """Assigns one of the 15 news categories."""

    name: str

    def predict(self, text: str, tokens: list[str] | None = None) -> Prediction:
        """Classify ``text``.

        Args:
            text: Cleaned document text.
            tokens: Pre-computed tokens, so callers that already tokenised do
                not pay for it twice. Backends may ignore this.
        """
        ...

    @property
    def is_trained(self) -> bool:
        """False when no model artefact is loaded, so callers can warn."""
        ...


@runtime_checkable
class SentimentBackend(Protocol):
    """Assigns positive / neutral / negative."""

    name: str

    def predict(self, text: str, tokens: list[str] | None = None) -> Prediction:
        """Classify the sentiment of ``text``."""
        ...

    def predict_many(self, texts: list[str]) -> list[Prediction]:
        """Classify a batch. Exists because per-message chat sentiment runs
        over tens of thousands of rows, where per-call overhead dominates."""
        ...

    @property
    def is_trained(self) -> bool: ...


@runtime_checkable
class KeywordExtractor(Protocol):
    """Extracts the most informative words from a document."""

    name: str

    def extract(
        self, text: str, tokens: list[str] | None = None, count: int = 10
    ) -> list[Keyword]:
        """Return up to ``count`` keywords, highest score first."""
        ...


@runtime_checkable
class Summarizer(Protocol):
    """Produces a short summary of a document."""

    name: str

    def summarize(
        self, text: str, *, min_sentences: int = 2, max_sentences: int = 4
    ) -> Summary:
        """Summarise ``text`` in roughly ``min``..``max`` sentences."""
        ...


@runtime_checkable
class EntityRecognizer(Protocol):
    """Finds named entities."""

    name: str

    def extract(self, text: str, tokens: list[str] | None = None) -> list[Entity]:
        """Return the entities found in ``text``."""
        ...
