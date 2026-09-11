"""Train the Thai topic and sentiment classifiers.

Usage::

    python train.py                      # train both from data/news_dataset.csv
    python train.py --task topic         # one task only
    python train.py --algorithm svm      # override the per-task tuned algorithm
    python train.py --no-cv              # skip cross-validation (faster)

Writes ``backend/models/topic_model.pkl`` and ``sentiment_model.pkl``.

Reported numbers come from a stratified hold-out split that the model never
saw, plus stratified k-fold cross-validation **on the training split only** --
so the two are independent estimates rather than two views of overlapping rows.
Macro-averaged scores are reported alongside accuracy because the classes are
not balanced: accuracy alone would flatter a model that simply favours the
largest class.
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
# One caveat about that provenance, because it matters: tune.py pre-tokenises
# documents once so the grid is affordable, which is a slightly different feature
# pipeline from build_vectorizer. Its *ranking* therefore cannot be adopted on
# trust -- on sentiment it scored the shipping config 0.723 where build_vectorizer
# measures 0.705. Every candidate below was re-scored through the production
# pipeline before being adopted, and only configurations that won there are here.
#
# Two findings worth recording:
#
# * Topic now prefers class_weight=None. The dataset expansion left the topic
#   classes far more even (49-62 rows each), so re-weighting them mostly adds
#   variance. Sentiment is still imbalanced (366/203/267) and still wants
#   balanced weights.
# * Sentiment prefers a calibrated LinearSVC over logistic regression. The margin
#   is small (0.716 vs 0.705) but it holds through the real pipeline.
TUNED: dict[str, dict] = {
    "topic": {
        "algorithm": "logreg",
        "classifier_c": 4.0,
        "class_weight": None,
        "features": {"word_min_df": 1, "char_min_df": 2, "char_ngram_range": (2, 4)},
    },
    "sentiment": {
        "algorithm": "svm",
        "classifier_c": 0.5,
        "class_weight": "balanced",
        "features": {"word_min_df": 1, "char_min_df": 2, "char_ngram_range": (2, 4)},
    },
}


def build_classifier(
    algorithm: str,
    class_count: int,  # noqa: ARG001 - kept for call-site symmetry
    penalty: float = 4.0,
    class_weight: str | None = "balanced",
):
    """Construct the classifier stage.

    LinearSVC is often stronger on sparse text but gives no probabilities, so it
    is wrapped in calibration -- the UI shows probability tables, and an
    uncalibrated margin would be misleading there.
    """
    if algorithm == "svm":
        base = LinearSVC(
            C=penalty, class_weight=class_weight, random_state=RANDOM_STATE
        )
        # 3 folds keeps calibration affordable on a small dataset.
        return CalibratedClassifierCV(base, cv=3, method="sigmoid")

    # Multinomial handling is automatic in scikit-learn 1.9; the `multi_class`
    # argument was removed, so it must not be passed.
    return LogisticRegression(
        C=penalty,
        max_iter=2000,
        # None for topic, "balanced" for sentiment -- see TUNED.
        class_weight=class_weight,
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
    weight = tuned.get("class_weight", "balanced")
    # An explicit --algorithm overrides the tuned choice; otherwise each task
    # uses whichever algorithm won cross-validation for it.
    if algorithm is None:
        algorithm = tuned.get("algorithm", "logreg")
    print(f"algorithm      : {algorithm}")
    print(f"features       : {feature_options or 'defaults'}")
    print(f"C              : {penalty}")
    print(f"class_weight   : {weight}")

    pipeline = Pipeline(
        [
            ("features", build_vectorizer(**feature_options)),
            (
                "classifier",
                build_classifier(algorithm, len(distribution), penalty, weight),
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
        train_distribution = Counter(y_train)
        folds = min(CV_FOLDS, min(train_distribution.values()))
        if folds < 2:
            print(
                "\ncross-validation skipped: a class has too few examples to fold."
            )
        else:
            print(f"\nrunning {folds}-fold cross-validation...")
            splitter = StratifiedKFold(
                n_splits=folds, shuffle=True, random_state=RANDOM_STATE
            )
            # The TRAINING split only, not the whole dataset. Cross-validating
            # over everything would fold the held-out rows into training, which
            # makes the CV mean and the hold-out score two views of overlapping
            # data rather than independent estimates -- and it is the hold-out
            # number that is supposed to be untouched.
            scores = cross_val_score(
                pipeline, x_train, y_train, cv=splitter, scoring="f1_macro"
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
        default=None,
        help="Override the per-task tuned algorithm (default: whichever won CV).",
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
