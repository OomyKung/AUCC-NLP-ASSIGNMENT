"""WangchanBERTa backends for topic and sentiment classification.

Selected with ``NLP_TOPIC_BACKEND=transformer`` / ``NLP_SENTIMENT_BACKEND=transformer``
once ``python train_transformer.py`` has produced a fine-tuned model.

Everything heavy is imported lazily inside the constructor, so a project without
``torch`` installed still starts: the registry catches the ImportError, falls
back to the classical backend, and reports why through ``/api/pipeline``.
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import settings
from app.nlp.base import Prediction
from app.taxonomy import (
    FALLBACK_SENTIMENT,
    FALLBACK_TOPIC,
    SENTIMENT_SLUGS,
    TOPIC_SLUGS,
)

# Inference batch size. Chat scoring runs over tens of thousands of very short
# texts, where per-call overhead would otherwise dominate.
BATCH_SIZE = 16
MAX_LENGTH = 256


class _TransformerBackend:
    """Shared loading and inference for the fine-tuned classifiers."""

    task: str = ""
    label_order: tuple[str, ...] = ()
    fallback_label: str = ""

    def __init__(self, model_dir: Path | None = None) -> None:
        # Imported here, not at module level, so the absence of torch is a
        # recoverable condition rather than an import-time crash.
        import torch
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
        )

        self._torch = torch
        directory = model_dir or (settings.model_dir / f"wangchanberta_{self.task}")

        if not (directory / "config.json").is_file():
            raise FileNotFoundError(
                f"No fine-tuned model at {directory}. "
                "Train one with: python train_transformer.py"
            )

        self._tokenizer = AutoTokenizer.from_pretrained(directory)
        self._model = AutoModelForSequenceClassification.from_pretrained(directory)
        self._model.eval()

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

        # id2label comes from the saved config, so label order cannot drift
        # from what the model was trained with.
        config = self._model.config
        self._id_to_label = {
            int(key): str(value) for key, value in (config.id2label or {}).items()
        }

        metrics_path = directory / "metrics.json"
        self.metrics: dict = {}
        if metrics_path.is_file():
            try:
                self.metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.metrics = {}

        self.name = "wangchanberta"

    @property
    def is_trained(self) -> bool:
        return True

    # ------------------------------------------------------------------ inference
    def _distributions(self, texts: list[str]) -> list[dict[str, float]]:
        """Softmax distributions over the model's own label set."""
        torch = self._torch
        results: list[dict[str, float]] = []

        with torch.no_grad():
            for start in range(0, len(texts), BATCH_SIZE):
                chunk = texts[start : start + BATCH_SIZE]
                encoded = self._tokenizer(
                    chunk,
                    truncation=True,
                    padding=True,
                    max_length=MAX_LENGTH,
                    return_tensors="pt",
                ).to(self._device)

                logits = self._model(**encoded).logits
                probabilities = torch.softmax(logits, dim=-1).cpu().numpy()

                for row in probabilities:
                    results.append(
                        {
                            self._id_to_label.get(index, str(index)): float(value)
                            for index, value in enumerate(row)
                        }
                    )
        return results

    def _to_prediction(self, distribution: dict[str, float]) -> Prediction:
        # Present every label in the taxonomy, so the detail page's probability
        # table is complete even for classes the training set under-represented.
        probabilities = {
            label: float(distribution.get(label, 0.0)) for label in self.label_order
        }
        total = sum(probabilities.values())
        if total > 0:
            probabilities = {k: v / total for k, v in probabilities.items()}
        label = max(probabilities, key=lambda key: probabilities[key])
        return Prediction(
            label=label,
            confidence=probabilities[label],
            probabilities=probabilities,
            model_name=self.name,
        )

    def _empty(self) -> Prediction:
        return Prediction(
            label=self.fallback_label,
            confidence=0.0,
            probabilities={label: 0.0 for label in self.label_order},
            model_name=self.name,
        )

    def predict(self, text: str, tokens: list[str] | None = None) -> Prediction:
        """Classify ``text``.

        ``tokens`` is ignored: the model has its own SentencePiece tokeniser and
        must see raw text, or the features would differ from training.
        """
        if not text or not text.strip():
            return self._empty()
        return self._to_prediction(self._distributions([text])[0])

    def predict_many(self, texts: list[str]) -> list[Prediction]:
        """Classify a batch in as few forward passes as possible."""
        if not texts:
            return []

        usable = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        results = [self._empty() for _ in texts]
        if not usable:
            return results

        distributions = self._distributions([t for _, t in usable])
        for (index, _), distribution in zip(usable, distributions, strict=True):
            results[index] = self._to_prediction(distribution)
        return results


class HFTopicBackend(_TransformerBackend):
    """Fine-tuned WangchanBERTa for the 15 news categories."""

    task = "topic"
    label_order = TOPIC_SLUGS
    fallback_label = FALLBACK_TOPIC


class HFSentimentBackend(_TransformerBackend):
    """Fine-tuned WangchanBERTa for 3-class sentiment."""

    task = "sentiment"
    label_order = SENTIMENT_SLUGS
    fallback_label = FALLBACK_SENTIMENT
