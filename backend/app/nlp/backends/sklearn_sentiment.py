"""Trained scikit-learn sentiment classifier."""

from __future__ import annotations

from app.nlp.backends.sklearn_model import TrainedModel, load_model
from app.nlp.base import Prediction
from app.taxonomy import FALLBACK_SENTIMENT, SENTIMENT_SLUGS


class SklearnSentimentBackend:
    """Sentiment classification with the fitted TF-IDF pipeline."""

    def __init__(self, model: TrainedModel) -> None:
        self.model = model
        self.name = model.name

    @classmethod
    def load(cls, kind: str = "sentiment") -> SklearnSentimentBackend | None:
        """Build from the saved artefact, or ``None`` if there is not one.

        ``kind`` selects which artefact to load. ``"sentiment"`` is the
        news-trained model; ``"chat_sentiment"`` is the one trained on the
        Wisesight social-media corpus by ``train_chat_sentiment.py``. They are
        separate files because the measured cross-domain transfer between news
        prose and chat register is poor in both directions.
        """
        model = load_model(kind)
        return cls(model) if model is not None else None

    @property
    def is_trained(self) -> bool:
        return True

    def _empty(self) -> Prediction:
        return Prediction(
            label=FALLBACK_SENTIMENT,
            confidence=1.0,
            probabilities={
                slug: 1.0 if slug == FALLBACK_SENTIMENT else 0.0
                for slug in SENTIMENT_SLUGS
            },
            model_name=self.name,
        )

    def _to_prediction(self, distribution: dict[str, float]) -> Prediction:
        probabilities = {
            slug: float(distribution.get(slug, 0.0)) for slug in SENTIMENT_SLUGS
        }
        label = max(probabilities, key=lambda slug: probabilities[slug])
        return Prediction(
            label=label,
            confidence=probabilities[label],
            probabilities=probabilities,
            model_name=self.name,
        )

    def predict(self, text: str, tokens: list[str] | None = None) -> Prediction:
        """Classify the sentiment of ``text``.

        ``tokens`` is ignored so the fitted vectoriser applies the same analyser
        it was trained with.
        """
        if not text or not text.strip():
            return self._empty()
        return self._to_prediction(self.model.predict_proba([text])[0])

    def predict_many(self, texts: list[str]) -> list[Prediction]:
        """Classify a batch in one vectorisation pass.

        Per-message chat sentiment runs over tens of thousands of rows, where
        calling predict() per row would dominate the runtime.
        """
        if not texts:
            return []

        usable = [
            (index, text) for index, text in enumerate(texts) if text and text.strip()
        ]
        results: list[Prediction] = [self._empty() for _ in texts]
        if not usable:
            return results

        distributions = self.model.predict_proba([text for _, text in usable])
        for (index, _), distribution in zip(usable, distributions, strict=True):
            results[index] = self._to_prediction(distribution)
        return results
