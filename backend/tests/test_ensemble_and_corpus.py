"""Tests for the blended backends and the Wisesight corpus loader."""

from __future__ import annotations

import pytest

from app.nlp import wisesight
from app.nlp.backends.blended import (
    BlendedSentimentBackend,
    BlendedTopicBackend,
    _BlendedBackend,
)
from app.nlp.backends.sklearn_model import load_model
from app.nlp.base import Prediction
from app.taxonomy import SENTIMENT_SLUGS, TOPIC_SLUGS

TRAINED_TOPIC = load_model("topic") is not None
TRAINED_SENTIMENT = load_model("sentiment") is not None

needs_topic = pytest.mark.skipif(
    not TRAINED_TOPIC, reason="no trained topic artefact; run python train.py"
)
needs_sentiment = pytest.mark.skipif(
    not TRAINED_SENTIMENT, reason="no trained sentiment artefact; run python train.py"
)


# --------------------------------------------------------------------------
# Blending maths, tested against stub members so the arithmetic is exact
# --------------------------------------------------------------------------


class _Stub:
    """A fixed backend, so the blend's arithmetic can be asserted exactly."""

    def __init__(self, name: str, probabilities: dict[str, float]) -> None:
        self.name = name
        self._probabilities = probabilities

    def predict(self, text: str, tokens: list[str] | None = None) -> Prediction:
        best = max(self._probabilities, key=lambda k: self._probabilities[k])
        return Prediction(
            label=best,
            confidence=self._probabilities[best],
            probabilities=dict(self._probabilities),
            model_name=self.name,
        )


def _blend(alpha: float) -> _BlendedBackend:
    model = _Stub("model", {"a": 0.8, "b": 0.2, "c": 0.0})
    rules = _Stub("rules", {"a": 0.0, "b": 0.4, "c": 0.6})
    return _BlendedBackend(model, rules, alpha)


def test_blend_is_the_weighted_average():
    prediction = _blend(0.5).predict("x")

    assert prediction.probabilities["a"] == pytest.approx(0.40)
    assert prediction.probabilities["b"] == pytest.approx(0.30)
    assert prediction.probabilities["c"] == pytest.approx(0.30)
    assert prediction.label == "a"


def test_alpha_one_is_the_model_alone():
    prediction = _blend(1.0).predict("x")

    assert prediction.probabilities["a"] == pytest.approx(0.8)
    assert prediction.label == "a"


def test_alpha_zero_is_the_rules_alone():
    prediction = _blend(0.0).predict("x")

    assert prediction.probabilities["c"] == pytest.approx(0.6)
    assert prediction.label == "c"


def test_blend_can_overturn_both_members():
    """The blend's label comes from the blended distribution.

    Here the model says "a" and the rules say "c", but averaging makes "b" the
    winner. This is the whole point of blending, and it is also why the label is
    re-derived rather than taken from either member.
    """
    model = _Stub("model", {"a": 0.5, "b": 0.45, "c": 0.05})
    rules = _Stub("rules", {"a": 0.0, "b": 0.45, "c": 0.55})
    prediction = _BlendedBackend(model, rules, 0.5).predict("x")

    assert prediction.label == "b"
    assert prediction.confidence == pytest.approx(0.45)


def test_blend_normalises_to_one():
    model = _Stub("model", {"a": 0.5, "b": 0.1})  # deliberately not normalised
    rules = _Stub("rules", {"a": 0.2, "b": 0.2})
    prediction = _BlendedBackend(model, rules, 0.5).predict("x")

    assert sum(prediction.probabilities.values()) == pytest.approx(1.0)


def test_confidence_always_matches_the_reported_label():
    for alpha in (0.0, 0.25, 0.5, 0.75, 1.0):
        prediction = _blend(alpha).predict("x")
        assert prediction.confidence == pytest.approx(
            prediction.probabilities[prediction.label]
        )


def test_blend_names_both_members():
    """An analysis row must record what produced it, not claim a single model."""
    name = _blend(0.85).name

    assert "model" in name and "rules" in name and "0.85" in name


@pytest.mark.parametrize("alpha", [-0.1, 1.1, 2.0])
def test_invalid_alpha_is_rejected(alpha):
    with pytest.raises(ValueError, match="alpha"):
        _blend(alpha)


def test_predict_many_matches_predict():
    blend = _blend(0.6)
    texts = ["one", "two", ""]

    batch = blend.predict_many(texts)
    singles = [blend.predict(text) for text in texts]

    assert [item.label for item in batch] == [item.label for item in singles]
    assert blend.predict_many([]) == []


# --------------------------------------------------------------------------
# The real backends
# --------------------------------------------------------------------------


@needs_topic
def test_blended_topic_covers_the_whole_taxonomy():
    backend = BlendedTopicBackend.load(0.85)
    assert backend is not None
    assert backend.is_trained is True

    prediction = backend.predict("ทีมชาติไทยคว้าชัยในการแข่งขันฟุตบอลนัดชิงชนะเลิศ")

    assert prediction.label == "sports"
    assert set(prediction.probabilities) == set(TOPIC_SLUGS)
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0)


@needs_sentiment
def test_blended_sentiment_covers_the_whole_taxonomy():
    backend = BlendedSentimentBackend.load(0.80)
    assert backend is not None

    prediction = backend.predict("ทีมชาติไทยเล่นได้ยอดเยี่ยมมาก ประทับใจสุดๆ")

    assert set(prediction.probabilities) == set(SENTIMENT_SLUGS)
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0)


def test_blend_load_returns_none_without_a_model(monkeypatch):
    """A missing artefact must be a clean ``None``, so the registry can fall back."""
    monkeypatch.setattr(
        "app.nlp.backends.sklearn_topic.SklearnTopicBackend.load",
        classmethod(lambda cls: None),
    )
    assert BlendedTopicBackend.load(0.85) is None


# --------------------------------------------------------------------------
# Corpus loader
# --------------------------------------------------------------------------


needs_corpus = pytest.mark.skipif(
    not wisesight.is_cached(),
    reason="Wisesight corpus not downloaded; run python train_chat_sentiment.py",
)


def test_category_mapping_is_complete():
    """All four corpus classes map somewhere, and the three real ones are ours."""
    assert set(wisesight.CATEGORY_TO_LABEL) == {0, 1, 2, 3}
    mapped = set(wisesight.CATEGORY_TO_LABEL.values())
    assert {"positive", "neutral", "negative"} <= mapped
    assert "question" in mapped


def test_citation_names_source_and_licence():
    citation = wisesight.citation()
    assert wisesight.REPO_ID in citation
    assert wisesight.LICENSE in citation


def test_unknown_split_is_rejected():
    with pytest.raises(ValueError, match="unknown split"):
        wisesight.load_split("holdout")


@needs_corpus
def test_questions_are_dropped_by_default():
    """The ``q`` class is not a sentiment, so it must not pollute neutral."""
    dropped = wisesight.load_split("test")
    folded = wisesight.load_split("test", keep_questions=True)

    assert set(dropped.labels) == {"positive", "neutral", "negative"}
    assert set(folded.labels) == {"positive", "neutral", "negative"}
    # Folding adds rows, and every added row lands in neutral.
    assert len(folded) > len(dropped)
    assert folded.distribution()["neutral"] > dropped.distribution()["neutral"]
    assert folded.distribution()["positive"] == dropped.distribution()["positive"]


@needs_corpus
def test_blank_rows_are_filtered():
    for name in ("train", "validation", "test"):
        split = wisesight.load_split(name)
        assert all(text.strip() for text in split.texts)
        assert len(split.texts) == len(split.labels)


@needs_corpus
def test_splits_are_disjoint():
    """Official splits must not overlap, or the benchmark is meaningless."""
    train = set(wisesight.load_split("train").texts)
    test = set(wisesight.load_split("test").texts)

    overlap = train & test
    # A handful of identical short messages ("ครับ") legitimately recur; the
    # splits are only broken if a large share is shared.
    assert len(overlap) / len(test) < 0.02
