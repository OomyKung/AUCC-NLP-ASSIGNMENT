"""Train the Thai topic and sentiment classifiers.

Usage::

    python train.py                      # train both from data/news_dataset.csv
    python train.py --task topic         # one task only
    python train.py --algorithm svm      # LinearSVC instead of logistic regression
    python train.py --no-cv              # skip cross-validation (faster)

Writes ``backend/models/topic_model.pkl`` and ``sentiment_model.pkl``.

Reported numbers come from a stratified hold-out split that the model never
saw, plus stratified k-fold cross-validation over the whole set. Macro-averaged
scores are reported alongside accuracy because the classes are not balanced --
accuracy alone would flatter a model that simply favours the largest class.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from app.nlp.backends.sklearn_model import TrainedModel, save_model  # noqa: E402
from app.nlp.features import build_vectorizer  # noqa: E402
from app.taxonomy import SENTIMENT_SLUGS, TOPIC_SLUGS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "news_dataset.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


def load_dataset(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    """Read the dataset into texts and per-task label lists."""
    if not path.is_file():
        raise SystemExit(
            f"Dataset not found: {path}\nBuild it first with: python build_dataset.py"
        )

    with open(path, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise SystemExit(f"Dataset is empty: {path}")

    # Title and body are concatenated: the headline carries strong signal and is
    # available at inference time too, so training without it would waste it.
    texts = [f"{row['title']} {row['content']}".strip() for row in rows]
    labels = {
        "topic": [row["topic"].strip() for row in rows],
        "sentiment": [row["sentiment"].strip() for row in rows],
    }
    return texts, labels


# Per-task hyperparameters, chosen by tune.py: a 144-point grid scored with
# 5-fold cross-validation on the TRAINING split only. Provenance is in
# models/tuning.json.
#
# Topic's entry is the configuration the project already used -- the search
# confirmed it rather than improving on it, which is worth recording as a
# result. Sentiment moved: a weaker penalty (C 4 -> 1), a word floor of 1 and
# longer character n-grams lifted cross-validated macro-F1 from 0.679 to 0.708.
# Sentiment has 3 classes over 597 rows, so it can afford rarer features than
# the 15-class topic task, where a min_df of 1 invites memorisation.
TUNED: dict[str, dict] = {
    "topic": {
        "classifier_c": 4.0,
        "features": {"word_min_df": 2, "char_min_df": 3, "char_ngram_range": (2, 4)},
    },
    "sentiment": {
        "classifier_c": 1.0,
        "features": {"word_min_df": 1, "char_min_df": 2, "char_ngram_range": (3, 5)},
    },
}


def build_classifier(algorithm: str, class_count: int, penalty: float = 4.0):  # noqa: ARG001
    """Construct the classifier stage.

    LinearSVC is often stronger on sparse text but gives no probabilities, so it
    is wrapped in calibration -- the UI shows probability tables, and an
    uncalibrated margin would be misleading there.
    """
    if algorithm == "svm":
        base = LinearSVC(C=1.0, class_weight="balanced", random_state=RANDOM_STATE)
        # 3 folds keeps calibration affordable on a small dataset.
        return CalibratedClassifierCV(base, cv=3, method="sigmoid")

    # Multinomial handling is automatic in scikit-learn 1.9; the `multi_class`
    # argument was removed, so it must not be passed.
    return LogisticRegression(
        C=penalty,
        max_iter=2000,
        # Compensates for the class imbalance rather than letting the model
        # drift toward whichever label is most common.
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def train_task(
    task: str,
    texts: list[str],
    labels: list[str],
    *,
    algorithm: str,
    expected_labels: tuple[str, ...],
    run_cv: bool,
) -> tuple[TrainedModel, dict]:
    """Train one task and return the model plus its metrics."""
    print(f"\n{'=' * 70}\n{task.upper()} CLASSIFICATION\n{'=' * 70}")

    distribution = Counter(labels)
    print(f"documents      : {len(texts)}")
    print(f"classes        : {len(distribution)}")
    print(
        "class balance  : "
        + ", ".join(f"{name} {count}" for name, count in distribution.most_common())
    )

    # Stratified split keeps every class represented in the test set.
    x_train, x_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    print(f"train / test   : {len(x_train)} / {len(x_test)}")

    tuned = TUNED.get(task, {})
    feature_options = tuned.get("features", {})
    penalty = tuned.get("classifier_c", 4.0)
    print(f"features       : {feature_options or 'defaults'}")
    print(f"C              : {penalty}")

    pipeline = Pipeline(
        [
            ("features", build_vectorizer(**feature_options)),
            (
                "classifier",
                build_classifier(algorithm, len(distribution), penalty),
            ),
        ]
    )

    started = time.perf_counter()
    pipeline.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - started
    print(f"fitted in      : {fit_seconds:.1f}s")

    predictions = pipeline.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    # zero_division=0 so a class absent from predictions scores 0 rather than
    # raising, which would hide the problem.
    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(
            precision_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "precision_weighted": float(
            precision_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(y_test, predictions, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(y_test, predictions, average="weighted", zero_division=0)
        ),
    }

    print(
        f"\naccuracy       : {metrics['accuracy']:.4f}\n"
        f"precision macro: {metrics['precision_macro']:.4f}\n"
        f"recall macro   : {metrics['recall_macro']:.4f}\n"
        f"f1 macro       : {metrics['f1_macro']:.4f}\n"
        f"f1 weighted    : {metrics['f1_weighted']:.4f}"
    )

    print("\nper-class report:")
    print(
        classification_report(
            y_test, predictions, zero_division=0, digits=3, labels=sorted(distribution)
        )
    )

    # Confusion matrix over the full label set, in taxonomy order, so the UI can
    # render it without guessing the axis ordering.
    present = [label for label in expected_labels if label in distribution]
    matrix = confusion_matrix(y_test, predictions, labels=present)

    per_class = {}
    class_report = classification_report(
        y_test, predictions, zero_division=0, output_dict=True, labels=present
    )
    for label in present:
        entry = class_report.get(label, {})
        per_class[label] = {
            "precision": float(entry.get("precision", 0.0)),
            "recall": float(entry.get("recall", 0.0)),
            "f1": float(entry.get("f1-score", 0.0)),
            "support": int(entry.get("support", 0)),
        }

    cross_validation = None
    if run_cv:
        folds = min(CV_FOLDS, min(distribution.values()))
        if folds < 2:
            print(
                "\ncross-validation skipped: a class has too few examples to fold."
            )
        else:
            print(f"\nrunning {folds}-fold cross-validation...")
            splitter = StratifiedKFold(
                n_splits=folds, shuffle=True, random_state=RANDOM_STATE
            )
            scores = cross_val_score(
                pipeline, texts, labels, cv=splitter, scoring="f1_macro"
            )
            cross_validation = {
                "folds": int(folds),
                "scoring": "f1_macro",
                "scores": [float(score) for score in scores],
                "mean": float(np.mean(scores)),
                "std": float(np.std(scores)),
            }
            print(
                f"cv f1_macro    : {cross_validation['mean']:.4f} "
                f"(+/- {cross_validation['std']:.4f})"
            )

    full_metrics = {
        **metrics,
        "per_class": per_class,
        "confusion_matrix": {
            "labels": present,
            "matrix": [[int(value) for value in row] for row in matrix],
        },
        "cross_validation": cross_validation,
        "dataset": {
            "documents": len(texts),
            "train": len(x_train),
            "test": len(x_test),
            "classes": len(distribution),
            "class_distribution": dict(distribution),
        },
        "algorithm": algorithm,
        "fit_seconds": round(fit_seconds, 2),
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
    }

    model = TrainedModel(
        pipeline=pipeline,
        labels=present,
        kind=task,
        algorithm=algorithm,
        trained_at=datetime.now(UTC).isoformat(),
        metrics=full_metrics,
    )
    return model, full_metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--task",
        choices=["topic", "sentiment", "both"],
        default="both",
        help="Which classifier to train (default: both).",
    )
    parser.add_argument(
        "--algorithm",
        choices=["logreg", "svm"],
        default="logreg",
        help="Classifier algorithm (default: logreg).",
    )
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument(
        "--no-cv", action="store_true", help="Skip cross-validation."
    )
    args = parser.parse_args(argv)

    print(f"dataset: {args.dataset}")
    texts, labels = load_dataset(args.dataset)

    tasks = ["topic", "sentiment"] if args.task == "both" else [args.task]
    expected = {"topic": TOPIC_SLUGS, "sentiment": SENTIMENT_SLUGS}

    for task in tasks:
        model, _metrics = train_task(
            task,
            texts,
            labels[task],
            algorithm=args.algorithm,
            expected_labels=expected[task],
            run_cv=not args.no_cv,
        )
        path = save_model(model)
        print(f"\nsaved: {path}")

    print(
        "\nDone. Run `python evaluate.py` to write models/metrics.json for the "
        "Evaluation page."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
