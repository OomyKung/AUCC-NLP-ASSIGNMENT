"""Gazetteer-based Thai topic classification.

The baseline topic backend: it scores a document against per-category term
lists (see :mod:`app.nlp.topic_terms`) and normalises the result into a
probability distribution over all 15 categories.

This is genuine inference with real, inspectable evidence -- not a placeholder --
so the dashboard shows meaningful topics before any model is trained. Once
``train.py`` has produced ``topic_model.pkl``, the registry prefers the fitted
scikit-learn backend instead; ``is_trained`` reports False here so the UI can
label results as a baseline honestly.
"""

from __future__ import annotations


from app.nlp.base import Prediction
from app.nlp.matching import TermMatcher
from app.nlp.topic_terms import TOPIC_TERMS, all_terms, gazetteer_size, topics_for_term
from app.taxonomy import FALLBACK_TOPIC, TOPIC_SLUGS

# Score below which a document is called "other" rather than forced into a
# category on thin evidence.
MIN_EVIDENCE = 0.9

# Additive smoothing, so every category keeps a small non-zero probability
# without flattening the distribution. Deliberately not a softmax: with 15
# categories and most scores at zero, a softmax gives the winning topic ~0.13,
# which reads as broken in the UI even when the prediction is correct. A
# normalised share of matched evidence is both higher-contrast and easier to
# explain to a reader ("78% of the topic evidence pointed at Accident").
_SMOOTHING = 0.02


class GazetteerTopicBackend:
    """Score topics by matching category-indicative terms."""

    name = "gazetteer"

    def __init__(self) -> None:
        self._matcher = TermMatcher(all_terms())

    @property
    def is_trained(self) -> bool:
        """False: rule-based baseline, not a fitted model."""
        return False

    # ------------------------------------------------------------------ scoring
    def score(self, tokens: list[str]) -> tuple[dict[str, float], list[dict]]:
        """Accumulate per-topic scores and the evidence behind them."""
        scores: dict[str, float] = dict.fromkeys(TOPIC_SLUGS, 0.0)
        evidence: list[dict] = []

        for match in self._matcher.find(tokens):
            attributions = topics_for_term(match.term)
            if not attributions:
                continue
            for topic, weight in attributions:
                scores[topic] += weight
            evidence.append(
                {
                    "term": match.term,
                    "via": match.via,
                    "topics": [
                        {"topic": topic, "weight": weight}
                        for topic, weight in attributions
                    ],
                }
            )

        return scores, evidence

    def _distribution(self, scores: dict[str, float]) -> dict[str, float]:
        """Normalise raw scores into a probability distribution.

        Each category's probability is its share of the total matched evidence,
        with light additive smoothing so all 15 categories stay present for the
        detail page's probability table.
        """
        best = max(scores.values())
        if best <= 0:
            # No evidence at all: all mass on the fallback category.
            empty = {slug: 0.0 for slug in TOPIC_SLUGS}
            empty[FALLBACK_TOPIC] = 1.0
            return empty

        total = sum(scores.values()) + _SMOOTHING * len(TOPIC_SLUGS)
        return {
            slug: (value + _SMOOTHING) / total for slug, value in scores.items()
        }

    # ------------------------------------------------------------------ predict
    def predict(self, text: str, tokens: list[str] | None = None) -> Prediction:
        """Classify ``text`` into one of the 15 categories."""
        if tokens is None:
            from app.nlp.tokenizer import get_tokenizer

            tokens = get_tokenizer().tokenize(text or "")

        scores, _evidence = self.score(tokens)
        probabilities = self._distribution(scores)

        best_topic = max(scores, key=lambda slug: scores[slug])
        if scores[best_topic] < MIN_EVIDENCE:
            # Too little evidence to claim a category.
            best_topic = FALLBACK_TOPIC

        return Prediction(
            label=best_topic,
            confidence=probabilities[best_topic],
            probabilities=probabilities,
            model_name=self.name,
        )

    def explain(self, text: str, tokens: list[str] | None = None) -> dict:
        """Full explanation of one prediction, for the NLP Analysis page."""
        if tokens is None:
            from app.nlp.tokenizer import get_tokenizer

            tokens = get_tokenizer().tokenize(text or "")

        scores, evidence = self.score(tokens)
        prediction = self.predict(text, tokens)
        return {
            "label": prediction.label,
            "confidence": prediction.confidence,
            "probabilities": prediction.probabilities,
            "raw_scores": {k: round(v, 3) for k, v in scores.items() if v > 0},
            "min_evidence": MIN_EVIDENCE,
            "evidence": evidence,
            "gazetteer": gazetteer_size(),
            "term_count": sum(len(t) for t in TOPIC_TERMS.values()),
        }
