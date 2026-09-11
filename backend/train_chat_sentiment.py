"""Train the chat-domain sentiment classifier on the Wisesight corpus.

This is the answer to a limit that hyperparameter search made obvious: sentiment
on the 747-row authored dataset plateaus near macro-F1 0.70, and a 144-point grid
moved it ~0.02. That is a data limit, not a modelling one.

Wisesight Sentiment gives ~21.6k human-labelled real Thai social-media messages
(see :mod:`app.nlp.wisesight` for provenance and licensing). Two things follow:

1. **Scale.** ~29x the sentiment training data, and ~50x the ``neutral``
   examples, which was the weakest class by a wide margin.
2. **Domain match.** Per-message chat sentiment was routed to the rule-based
   lexicon backend because a news-prose model transfers badly to live chat.
   Wisesight *is* social-media register, so the model this script produces is
   trained on the domain the chat pipeline actually runs on.

The artefact is saved as ``chat_sentiment_model.pkl`` -- deliberately separate
from ``sentiment_model.pkl``. The news pipeline keeps its news-trained model and
the chat pipeline gets a chat-trained one; conflating them is the domain mismatch
this project already measured.

Evaluation uses the corpus's **official test split**, unchanged, so the numbers
are comparable with published work on this benchmark. The authored news test set
is also scored as a cross-domain check -- it answers "does a social-media model
transfer to news prose?", which is the mirror of the original problem.

Usage::

    python train_chat_sentiment.py
    python train_chat_sentiment.py --keep-questions    # 4-class-comparable
    python train_chat_sentiment.py --algorithm svc
    python train_chat_sentiment.py --max-rows 5000     # quick smoke run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from app.config import settings  # noqa: E402
from app.nlp import wisesight  # noqa: E402
from app.nlp.backends.sklearn_model import TrainedModel, save_model  # noqa: E402
from app.nlp.features import build_vectorizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
NEWS_DATASET = ROOT / "data" / "news_dataset.csv"
METRICS = Path(__file__).resolve().parent / "models" / "chat_sentiment_metrics.json"

RANDOM_STATE = 42
KIND = "chat_sentiment"


def build_classifier(algorithm: str):
    """The classifier, chosen to keep calibrated probabilities available.

    The UI shows a confidence figure and a full distribution, so a backend that
    cannot produce probabilities is not acceptable here. ``LinearSVC`` is wrapped
    in ``CalibratedClassifierCV`` rather than having its margins softmaxed,
    because at this data size proper calibration is affordable.
    """
    if algorithm == "svc":
        return CalibratedClassifierCV(
            LinearSVC(C=0.5, class_weight="balanced", random_state=RANDOM_STATE),
            cv=3,
        )
    # No n_jobs: with the lbfgs solver sklearn fits the multinomial objective
    # in one problem, so the argument would advertise parallelism that never
    # happens.
    return LogisticRegression(
        C=4.0,
        max_iter=3000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def load_news_test() -> tuple[list[str], list[str]]:
    """The authored news test split, for the cross-domain check.

    Recreates exactly the split train.py uses, so this measures transfer onto
    rows no model in this project has ever trained on.
    """
    from sklearn.model_selection import train_test_split

    with open(NEWS_DATASET, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    texts = [f"{row['title']} {row['content']}".strip() for row in rows]
    labels = [row["sentiment"].strip() for row in rows]
    _x, x_test, _y, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=RANDOM_STATE, stratify=labels
    )
    return x_test, y_test


def report(name: str, y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    """Score one evaluation and print it."""
    accuracy = float(accuracy_score(y_true, y_pred))
    macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    print(f"\n{name}")
    print("-" * len(name))
    print(f"rows         : {len(y_true)}")
    print(f"accuracy     : {accuracy:.4f}")
    print(f"f1 macro     : {macro:.4f}")
    print(f"f1 weighted  : {weighted:.4f}")
    print()
    print(classification_report(y_true, y_pred, zero_division=0, digits=3))

    present = [label for label in labels if label in set(y_true) | set(y_pred)]
    matrix = confusion_matrix(y_true, y_pred, labels=present).tolist()
    return {
        "rows": len(y_true),
        "accuracy": accuracy,
        "f1_macro": macro,
        "f1_weighted": weighted,
        "labels": present,
        "confusion_matrix": matrix,
        "report": classification_report(
            y_true, y_pred, zero_division=0, digits=3, output_dict=True
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--algorithm", choices=["logreg", "svc"], default="logreg"
    )
    parser.add_argument(
        "--keep-questions",
        action="store_true",
        help="Fold the corpus 'q' class into neutral instead of dropping it.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Cap training rows (0 = all). For a quick smoke run.",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="Re-download the corpus."
    )
    args = parser.parse_args(argv)

    print("=" * 74)
    print("CHAT SENTIMENT - training on the Wisesight corpus")
    print("=" * 74)
    print(wisesight.citation())

    try:
        splits = wisesight.load_all(
            keep_questions=args.keep_questions, refresh=args.refresh
        )
    except wisesight.CorpusUnavailable as exc:
        print(f"\nCorpus unavailable: {exc}")
        print(
            "\nThe chat pipeline keeps working without this model -- it falls back\n"
            "to the rule-based lexicon backend. Re-run with a network connection\n"
            "to train the chat-domain model."
        )
        return 1

    train, validation, test = splits["train"], splits["validation"], splits["test"]
    x_train, y_train = list(train.texts), list(train.labels)

    # The official validation split is real labelled data from the same
    # distribution, so it is folded into training rather than wasted: model
    # selection here comes from the already-completed grid search, and the
    # official *test* split stays untouched as the reported benchmark.
    x_train += validation.texts
    y_train += validation.labels

    if args.max_rows and args.max_rows < len(x_train):
        x_train, y_train = x_train[: args.max_rows], y_train[: args.max_rows]
        print(f"\n(smoke run: capped at {args.max_rows} training rows)")

    print(f"\nquestions    : {'folded into neutral' if args.keep_questions else 'dropped'}")
    print(f"train        : {len(x_train)} rows (train + validation)")
    print(f"test         : {len(test)} rows (official split, untouched)")
    print(
        "train balance: "
        f"{ {label: y_train.count(label) for label in sorted(set(y_train))} }"
    )
    print(f"test balance : {test.distribution()}")

    labels = sorted(set(y_train))
    # Feature thresholds scaled for the corpus size. The news defaults
    # (min_df 2/3, 60k cap) were set for 597 documents; at ~24k documents many
    # more terms clear a low min_df, so the thresholds rise to keep one-off
    # spellings out while the cap rises to stop discarding genuine vocabulary.
    pipeline = Pipeline(
        [
            (
                "features",
                build_vectorizer(
                    word_min_df=3, char_min_df=5, max_features=200_000
                ),
            ),
            ("classifier", build_classifier(args.algorithm)),
        ]
    )

    print(f"\nalgorithm    : {args.algorithm}")
    print("fitting...", flush=True)
    began = time.perf_counter()
    pipeline.fit(x_train, y_train)
    fit_seconds = time.perf_counter() - began
    print(f"fitted in    : {fit_seconds:.0f}s")

    vectoriser = pipeline.named_steps["features"]
    feature_count = sum(
        len(step.vocabulary_)
        for _name, step in vectoriser.transformer_list
        if hasattr(step, "vocabulary_")
    )
    print(f"features     : {feature_count}")

    # ------------------------------------------------------- in-domain benchmark
    in_domain = report(
        "Wisesight official test split (in-domain benchmark)",
        list(test.labels),
        list(pipeline.predict(test.texts)),
        labels,
    )

    # ------------------------------------------------------- cross-domain check
    cross_domain = None
    if NEWS_DATASET.is_file():
        news_x, news_y = load_news_test()
        cross_domain = report(
            "Authored news test split (cross-domain transfer)",
            news_y,
            list(pipeline.predict(news_x)),
            labels,
        )

    # ------------------------------------------------------------------- save
    metrics = {
        "corpus": wisesight.REPO_ID,
        "license": wisesight.LICENSE,
        "citation": wisesight.citation(),
        "questions": "neutral" if args.keep_questions else "dropped",
        "algorithm": args.algorithm,
        "train_rows": len(x_train),
        "features": feature_count,
        "fit_seconds": round(fit_seconds, 1),
        "trained_at": datetime.now(UTC).isoformat(),
        "labels": labels,
        "in_domain": in_domain,
        "cross_domain": cross_domain,
    }

    model = TrainedModel(
        pipeline=pipeline,
        labels=labels,
        kind=KIND,
        algorithm=f"{args.algorithm}-wisesight",
        trained_at=metrics["trained_at"],
        metrics={
            "accuracy": in_domain["accuracy"],
            "f1_macro": in_domain["f1_macro"],
            "corpus": wisesight.REPO_ID,
            "train_rows": len(x_train),
        },
    )
    path = save_model(model, settings.model_dir)
    METRICS.parent.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\nsaved model  : {path}")
    print(f"saved metrics: {METRICS}")
    print(
        "\nEnable it for per-message chat sentiment with:\n"
        "  NLP_CHAT_SENTIMENT_BACKEND=wisesight"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
