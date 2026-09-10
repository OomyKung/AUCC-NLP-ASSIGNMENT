"""Trained scikit-learn topic classifier."""

from __future__ import annotations

from app.nlp.backends.sklearn_model import TrainedModel, load_model
from app.nlp.base import Prediction
from app.taxonomy import FALLBACK_TOPIC, TOPIC_SLUGS


class SklearnTopicBackend:
    """Topic classification with the fitted TF-IDF pipeline."""

    def __init__(self, model: TrainedModel) -> None:
        self.model = model
        self.name = model.name

    @classmethod
    def load(cls) -> SklearnTopicBackend | None:
        """Build from the saved artefact, or ``None`` if there is not one."""
        model = load_model("topic")
        return cls(model) if model is not None else None

    @property
    def is_trained(self) -> bool:
        return True

    def predict(self, text: str, tokens: list[str] | None = None) -> Prediction:
        """Classify ``text`` into one of the 15 categories.

        ``tokens`` is ignored: the fitted vectoriser must see raw text so it
        applies exactly the same analyser it was trained with. Re-using
        externally computed tokens here would silently change the features.
        """
        if not text or not text.strip():
            return Prediction(
                label=FALLBACK_TOPIC,
                confidence=0.0,
                probabilities={slug: 0.0 for slug in TOPIC_SLUGS},
                model_name=self.name,
            )

        distribution = self.model.predict_proba([text])[0]

        # Present all 15 categories, so the detail page's table is complete
        # even for labels the training set under-represented.
        probabilities = {slug: float(distribution.get(slug, 0.0)) for slug in TOPIC_SLUGS}
        label = max(probabilities, key=lambda slug: probabilities[slug])

        return Prediction(
            label=label,
            confidence=probabilities[label],
            probabilities=probabilities,
            model_name=self.name,
        )
