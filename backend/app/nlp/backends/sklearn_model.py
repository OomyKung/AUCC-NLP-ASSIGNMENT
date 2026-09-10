"""Shared loader for the trained scikit-learn classifiers.

``train.py`` writes an artefact holding the fitted pipeline plus the metadata
needed to interpret it. Both the topic and sentiment backends load through here,
so the on-disk format is defined in exactly one place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib

from app.config import settings

# Bumped when the artefact layout changes, so an old file is rejected rather
# than loaded into a mismatched reader.
ARTEFACT_VERSION = 1


@dataclass(slots=True)
class TrainedModel:
    """A fitted pipeline and everything needed to use it."""

    pipeline: Any  # sklearn Pipeline: features -> classifier
    labels: list[str]
    kind: str  # "topic" or "sentiment"
    algorithm: str
    trained_at: str
    metrics: dict = field(default_factory=dict)
    version: int = ARTEFACT_VERSION

    @property
    def name(self) -> str:
        return f"sklearn:{self.algorithm}"

    def predict_proba(self, texts: list[str]) -> list[dict[str, float]]:
        """Probability distribution over labels, for each input text.

        ``LinearSVC`` has no ``predict_proba``, so its signed decision values
        are converted with a softmax. That is a monotone transform of the
        margins, which keeps the ranking honest while still giving the UI a
        normalised distribution to display.
        """
        import numpy as np

        classifier = self.pipeline[-1]

        if hasattr(classifier, "predict_proba"):
            matrix = self.pipeline.predict_proba(texts)
        else:
            scores = self.pipeline.decision_function(texts)
            scores = np.atleast_2d(scores)
            if scores.shape[1] == 1:  # binary case
                scores = np.hstack([-scores, scores])
            shifted = scores - scores.max(axis=1, keepdims=True)
            exponentials = np.exp(shifted)
            matrix = exponentials / exponentials.sum(axis=1, keepdims=True)

        classes = list(getattr(classifier, "classes_", self.labels))
        return [
            {str(label): float(row[index]) for index, label in enumerate(classes)}
            for row in matrix
        ]


def artefact_path(kind: str, directory: Path | None = None) -> Path:
    """Where the artefact for ``kind`` lives."""
    base = directory or settings.model_dir
    return base / f"{kind}_model.pkl"


def save_model(model: TrainedModel, directory: Path | None = None) -> Path:
    """Persist a trained model."""
    target = artefact_path(model.kind, directory)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "version": ARTEFACT_VERSION,
            "pipeline": model.pipeline,
            "labels": model.labels,
            "kind": model.kind,
            "algorithm": model.algorithm,
            "trained_at": model.trained_at,
            "metrics": model.metrics,
        },
        target,
    )
    return target


def load_model(kind: str, directory: Path | None = None) -> TrainedModel | None:
    """Load a trained model, or ``None`` when there is nothing usable.

    Returning ``None`` rather than raising is deliberate: the registry treats a
    missing or unreadable artefact as "fall back to the baseline and say so",
    which is what keeps a fresh clone working.
    """
    source = artefact_path(kind, directory)
    if not source.is_file():
        return None

    try:
        payload = joblib.load(source)
    except Exception:
        # A corrupt or version-mismatched artefact must not break startup.
        return None

    if not isinstance(payload, dict) or payload.get("version") != ARTEFACT_VERSION:
        return None
    if "pipeline" not in payload:
        return None

    return TrainedModel(
        pipeline=payload["pipeline"],
        labels=list(payload.get("labels") or []),
        kind=payload.get("kind", kind),
        algorithm=payload.get("algorithm", "unknown"),
        trained_at=payload.get("trained_at", ""),
        metrics=payload.get("metrics") or {},
    )
