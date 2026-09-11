"""Blended rule + statistical backends.

Measured motivation, not a guess. On the held-out news split the trained model
and the rule-based baseline make *different* mistakes:

======  ==================  ==================  ==========  ===============
task    only rules correct  only model correct  both wrong  oracle ceiling
======  ==================  ==================  ==========  ===============
topic   18 / 150            24 / 150            23 / 150    0.847 accuracy
======  ==================  ==================  ==========  ===============

Two models that fail on different rows can be combined for a real gain, and this
is the simplest combination that captures part of it::

    P = alpha * P_model + (1 - alpha) * P_rules

``alpha`` is fitted by :mod:`ensemble` on **out-of-fold** training predictions --
never on the held-out set, and never on in-fold predictions, where a fitted model
looks near-perfect on its own training rows and the search would simply return
``alpha = 1``.

Why a single weight and not a meta-learner: stacking a logistic-regression
meta-learner over both probability vectors was measured too and came out *worse*
than the plain model (topic CV macro-F1 0.707 vs 0.720). With 597 training rows
and 30 meta-features it overfits. Sharpening the gazetteer distribution with a
temperature parameter was also tried; it moved cross-validation and held-out
scores in opposite directions, i.e. it was noise, so it is not used here. One
parameter, fitted honestly, is what survived.

The blend is opt-in per task via ``NLP_TOPIC_BACKEND=blend`` /
``NLP_SENTIMENT_BACKEND=blend``.
"""

from __future__ import annotations

from app.nlp.base import Prediction


class _BlendedBackend:
    """Weighted average of a trained model's and a rule backend's distributions."""

    task: str = ""

    def __init__(self, model, rules, alpha: float) -> None:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        self._model = model
        self._rules = rules
        self._alpha = alpha
        # Names both members, so an analysis row records exactly what produced
        # it and the UI can say so rather than claiming a single model.
        self.name = f"blend({model.name}+{rules.name}@{alpha:g})"

    @property
    def is_trained(self) -> bool:
        """True: one member is a fitted model, so this is not a cold-start path."""
        return True

    @property
    def alpha(self) -> float:
        return self._alpha

    def _combine(self, model: Prediction, rules: Prediction) -> Prediction:
        """Average the two distributions and re-derive the label from the result.

        The label comes from the blended distribution rather than from either
        member, so the reported confidence and the reported label can never
        disagree -- which is what would happen if one model's label were kept
        alongside the other's probabilities.
        """
        labels = set(model.probabilities) | set(rules.probabilities)
        blended = {
            label: self._alpha * model.probabilities.get(label, 0.0)
            + (1.0 - self._alpha) * rules.probabilities.get(label, 0.0)
            for label in labels
        }

        total = sum(blended.values())
        if total > 0:
            blended = {label: value / total for label, value in blended.items()}

        best = max(blended, key=lambda label: blended[label])
        return Prediction(
            label=best,
            confidence=blended[best],
            probabilities=blended,
            model_name=self.name,
        )

    def predict(self, text: str, tokens: list[str] | None = None) -> Prediction:
        """Classify ``text`` with both members and blend the result."""
        return self._combine(
            self._model.predict(text, tokens=tokens),
            self._rules.predict(text, tokens=tokens),
        )

    def predict_many(self, texts: list[str]) -> list[Prediction]:
        """Batch form, used by per-message chat scoring.

        The trained model vectorises the whole batch in one pass; the rule
        backend has no batch advantage, so it is called per text.
        """
        if not texts:
            return []

        if hasattr(self._model, "predict_many"):
            model_predictions = self._model.predict_many(texts)
        else:
            model_predictions = [self._model.predict(text) for text in texts]

        return [
            self._combine(model, self._rules.predict(text))
            for text, model in zip(texts, model_predictions, strict=True)
        ]


class BlendedTopicBackend(_BlendedBackend):
    """Trained topic model blended with the gazetteer baseline."""

    task = "topic"

    @classmethod
    def load(cls, alpha: float) -> BlendedTopicBackend | None:
        """Build the blend, or ``None`` when there is no trained model to blend."""
        from app.nlp.backends.gazetteer_topic import GazetteerTopicBackend
        from app.nlp.backends.sklearn_topic import SklearnTopicBackend

        model = SklearnTopicBackend.load()
        if model is None:
            return None
        return cls(model, GazetteerTopicBackend(), alpha)


class BlendedSentimentBackend(_BlendedBackend):
    """Trained sentiment model blended with the polarity lexicon."""

    task = "sentiment"

    @classmethod
    def load(cls, alpha: float, kind: str = "sentiment") -> BlendedSentimentBackend | None:
        """Build the blend, or ``None`` when there is no trained model to blend."""
        from app.nlp.backends.lexicon_sentiment import LexiconSentimentBackend
        from app.nlp.backends.sklearn_sentiment import SklearnSentimentBackend

        model = SklearnSentimentBackend.load(kind=kind)
        if model is None:
            return None
        return cls(model, LexiconSentimentBackend(), alpha)
