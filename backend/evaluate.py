"""Evaluate the trained models and write ``models/metrics.json``.

Usage::

    python evaluate.py

The trained classifiers and the rule-based baselines are evaluated on the **same
held-out split**, produced with the same seed as ``train.py``. That is the only
comparison that means anything: the gazetteer and lexicon were written by hand
against this domain, so scoring them on data they were designed around would
flatter them. Scoring both on identical unseen rows shows what training actually
bought.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402

from app.config import settings  # noqa: E402
from app.nlp.backends.gazetteer_topic import GazetteerTopicBackend  # noqa: E402
from app.nlp.backends.lexicon_sentiment import LexiconSentimentBackend  # noqa: E402
from app.nlp.backends.sklearn_model import load_model  # noqa: E402
from app.taxonomy import SENTIMENT_SLUGS, TOPIC_SLUGS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "news_dataset.csv"
METRICS_PATH = Path(__file__).resolve().parent / "models" / "metrics.json"

# Must match train.py, or the "held-out" rows would not be held out.
RANDOM_STATE = 42
TEST_SIZE = 0.2


def blend_labels(
    trained, rules, texts: list[str], labels: list[str], alpha: float
) -> list[str]:
    """Predicted labels from the weighted blend of a trained model and rules.

    Mirrors app/nlp/backends/blended.py, deliberately using the same arithmetic
    on the same inputs -- if the two drifted apart, this page would report a
    model other than the one serving requests.
    """
    model_probabilities = trained.predict_proba(texts)
    predictions: list[str] = []
    for text, model_row in zip(texts, model_probabilities, strict=True):
        rule_row = rules.predict(text).probabilities
        blended = {
            label: alpha * model_row.get(label, 0.0)
            + (1.0 - alpha) * rule_row.get(label, 0.0)
            for label in labels
        }
        predictions.append(max(blended, key=lambda label: blended[label]))
    return predictions


def score(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> dict:
    """Compute the full metric set for one model on one split."""
    report = classification_report(
        y_true, y_pred, labels=labels, zero_division=0, output_dict=True
    )
    per_class = {
        label: {
            "precision": float(report.get(label, {}).get("precision", 0.0)),
            "recall": float(report.get(label, {}).get("recall", 0.0)),
            "f1": float(report.get(label, {}).get("f1-score", 0.0)),
            "support": int(report.get(label, {}).get("support", 0)),
        }
        for label in labels
    }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_weighted": float(
            precision_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": labels,
            "matrix": [
                [int(value) for value in row]
                for row in confusion_matrix(y_true, y_pred, labels=labels)
            ],
        },
        "support": len(y_true),
    }


def print_summary(name: str, metrics: dict) -> None:
    print(
        f"  {name:22} acc {metrics['accuracy']:.4f}  "
        f"P {metrics['precision_macro']:.4f}  "
        f"R {metrics['recall_macro']:.4f}  "
        f"F1 {metrics['f1_macro']:.4f}"
    )


def main() -> int:
    if not DATASET.is_file():
        raise SystemExit(
            f"Dataset not found: {DATASET}\nBuild it with: python build_dataset.py"
        )

    with open(DATASET, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    texts = [f"{row['title']} {row['content']}".strip() for row in rows]
    topics = [row["topic"].strip() for row in rows]
    sentiments = [row["sentiment"].strip() for row in rows]

    payload: dict = {
        "generated_at": datetime.now(UTC).isoformat(),
        "dataset": {
            "path": str(DATASET.relative_to(ROOT)),
            "documents": len(rows),
            "topics": dict(Counter(topics)),
            "sentiments": dict(Counter(sentiments)),
        },
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratified": True,
            "note": (
                "Trained models and rule-based baselines are scored on the same "
                "held-out rows, which is the only fair comparison."
            ),
        },
        "tasks": {},
    }

    for task, labels_all, label_set in (
        ("topic", topics, list(TOPIC_SLUGS)),
        ("sentiment", sentiments, list(SENTIMENT_SLUGS)),
    ):
        print(f"\n{'=' * 74}\n{task.upper()}\n{'=' * 74}")

        _x_train, x_test, _y_train, y_test = train_test_split(
            texts,
            labels_all,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=labels_all,
        )
        present = [label for label in label_set if label in set(labels_all)]
        print(f"held-out rows: {len(x_test)}")

        entry: dict = {"labels": present, "models": {}}

        # ---------------------------------------------------------- trained
        trained = load_model(task)
        if trained is None:
            print("  trained model: NOT FOUND (run: python train.py)")
        else:
            predictions = list(trained.pipeline.predict(x_test))
            metrics = score(y_test, predictions, present)
            metrics["algorithm"] = trained.algorithm
            metrics["trained_at"] = trained.trained_at
            # Cross-validation was computed during training; carry it through.
            metrics["cross_validation"] = (trained.metrics or {}).get(
                "cross_validation"
            )
            entry["models"]["trained"] = metrics
            print_summary(f"trained ({trained.algorithm})", metrics)

        # --------------------------------------------------------- baseline
        if task == "topic":
            backend = GazetteerTopicBackend()
            baseline_name = "gazetteer"
        else:
            backend = LexiconSentimentBackend()
            baseline_name = "lexicon"

        baseline_predictions = [backend.predict(text).label for text in x_test]
        baseline_metrics = score(y_test, baseline_predictions, present)
        baseline_metrics["algorithm"] = baseline_name
        entry["models"]["baseline"] = baseline_metrics
        print_summary(f"baseline ({baseline_name})", baseline_metrics)

        # ------------------------------------------------------------ blend
        # The blend of the two above is what actually serves requests by
        # default, so it has to appear here: an Evaluation page that reported
        # only the plain trained model would be describing a model that is not
        # running. The weight comes from config, fitted by ensemble.py on
        # out-of-fold training predictions.
        if trained is not None:
            alpha = (
                settings.nlp_topic_blend_alpha
                if task == "topic"
                else settings.nlp_sentiment_blend_alpha
            )
            blend_predictions = blend_labels(
                trained, backend, x_test, present, alpha
            )
            blend_metrics = score(y_test, blend_predictions, present)
            blend_metrics["algorithm"] = f"blend({trained.algorithm}+{baseline_name})"
            blend_metrics["alpha"] = alpha
            entry["models"]["blend"] = blend_metrics
            print_summary(f"blend (alpha={alpha:g})", blend_metrics)

        # ----------------------------------------------------- random floor
        # The score a coin-flip would get, so the reader can judge the rest.
        entry["random_baseline_accuracy"] = 1.0 / len(present) if present else 0.0
        print(
            f"  {'random guess':22} acc {entry['random_baseline_accuracy']:.4f}"
            f"  ({len(present)} classes)"
        )

        # Which of the scored models is the one actually serving requests.
        # Without this the page could show a metric table for a model the API is
        # not using, which is exactly the kind of quiet mismatch that makes an
        # evaluation page untrustworthy.
        requested = (
            settings.nlp_topic_backend
            if task == "topic"
            else settings.nlp_sentiment_backend
        )
        active = {"blend": "blend", "sklearn": "trained", "transformer": "trained"}.get(
            requested, "baseline"
        )
        if active not in entry["models"]:
            active = "trained" if "trained" in entry["models"] else "baseline"
        # A blend at alpha=1.0 is arithmetically the trained model, so calling it
        # a blend on the page would overstate what is running.
        if active == "blend" and entry["models"]["blend"].get("alpha") == 1.0:
            active = "trained"
        entry["active_model"] = active
        entry["configured_backend"] = requested
        print(f"  {'serving':22} {active} (NLP backend: {requested})")

        if "trained" in entry["models"]:
            gain = (
                entry["models"]["trained"]["f1_macro"]
                - baseline_metrics["f1_macro"]
            )
            entry["trained_vs_baseline_f1_macro"] = float(gain)
            print(f"  => training changed macro-F1 by {gain:+.4f}")

        print("\n  per-class F1 (trained vs baseline):")
        for label in present:
            trained_f1 = (
                entry["models"]
                .get("trained", {})
                .get("per_class", {})
                .get(label, {})
                .get("f1")
            )
            base_f1 = baseline_metrics["per_class"][label]["f1"]
            trained_text = f"{trained_f1:.3f}" if trained_f1 is not None else "  -  "
            print(f"    {label:16} {trained_text}   {base_f1:.3f}")

        payload["tasks"][task] = entry

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {METRICS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
