"""Hyperparameter search for the classical models.

Methodology note, which matters more than the numbers:

Model selection uses **5-fold cross-validation on the training split only**. The
held-out test set is scored exactly once, at the end, with the winning
configuration. Searching against the test set would inflate the reported score
and would not survive review -- the whole point of a held-out set is that it is
touched once.

Speed: the Thai tokeniser is the expensive part of fitting, and a grid search
re-tokenises the same documents hundreds of times. Documents are therefore
tokenised once up front and the vectorisers run over the pre-tokenised text, so
a search that would take hours takes minutes.

Usage::

    python tune.py                     # both tasks
    python tune.py --task topic
    python tune.py --quick             # smaller grid
    python tune.py --jobs 4            # parallel folds
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import numpy as np  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import classification_report, f1_score  # noqa: E402
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split  # noqa: E402
from sklearn.naive_bayes import ComplementNB  # noqa: E402
from sklearn.pipeline import FeatureUnion, Pipeline  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from app.nlp.preprocessing import clean_text, filter_tokens  # noqa: E402
from app.nlp.tokenizer import get_tokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "news_dataset.csv"
OUTPUT = Path(__file__).resolve().parent / "models" / "tuning.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


def prepare(texts: list[str]) -> tuple[list[str], list[str]]:
    """Tokenise once. Returns (space-joined tokens, cleaned text).

    The grid search would otherwise re-run the Thai tokeniser for every
    candidate and every fold.
    """
    tokenizer = get_tokenizer()
    joined: list[str] = []
    cleaned: list[str] = []
    for text in texts:
        clean = clean_text(text)
        tokens = filter_tokens(tokenizer.tokenize(clean), protect_polarity=True)
        bigrams = [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
        joined.append(" ".join(tokens + bigrams))
        cleaned.append(clean)
    return joined, cleaned


class Combined:
    """Holds both views of a document so one array can feed both vectorisers."""

    def __init__(self, joined: list[str], cleaned: list[str]) -> None:
        self.rows = np.array(list(zip(joined, cleaned, strict=True)), dtype=object)


def word_column(matrix):
    return matrix[:, 0]


def char_column(matrix):
    return matrix[:, 1]


def build_pipeline(classifier) -> Pipeline:
    """Word + character TF-IDF over pre-tokenised input."""
    from sklearn.preprocessing import FunctionTransformer

    word = Pipeline(
        [
            ("select", FunctionTransformer(word_column, validate=False)),
            # Input is already tokenised, so split on whitespace only.
            ("tfidf", TfidfVectorizer(token_pattern=r"\S+", sublinear_tf=True)),
        ]
    )
    char = Pipeline(
        [
            ("select", FunctionTransformer(char_column, validate=False)),
            ("tfidf", TfidfVectorizer(analyzer="char_wb", sublinear_tf=True)),
        ]
    )
    return Pipeline(
        [
            ("features", FeatureUnion([("word", word), ("char", char)])),
            ("classifier", classifier),
        ]
    )


def grids(quick: bool) -> list[tuple[str, object, dict]]:
    """Candidate models and their search spaces."""
    word_df = [1, 2] if quick else [1, 2, 3]
    char_ngrams = [(2, 4), (2, 5)] if quick else [(2, 4), (2, 5), (3, 5)]
    char_df = [2] if quick else [2, 3]

    shared = {
        "features__word__tfidf__min_df": word_df,
        "features__char__tfidf__ngram_range": char_ngrams,
        "features__char__tfidf__min_df": char_df,
    }

    candidates: list[tuple[str, object, dict]] = [
        (
            "LogisticRegression",
            LogisticRegression(max_iter=3000, random_state=RANDOM_STATE),
            {
                **shared,
                "classifier__C": [4.0, 10.0] if quick else [1.0, 4.0, 10.0, 20.0],
                "classifier__class_weight": ["balanced", None],
            },
        ),
        (
            "LinearSVC",
            LinearSVC(random_state=RANDOM_STATE, max_iter=5000),
            {
                **shared,
                "classifier__C": [0.5, 1.0] if quick else [0.25, 0.5, 1.0, 2.0],
                "classifier__class_weight": ["balanced", None],
            },
        ),
        (
            "ComplementNB",
            ComplementNB(),
            {**shared, "classifier__alpha": [0.1, 0.3, 1.0]},
        ),
    ]
    return candidates


def tune_task(
    task: str, texts: list[str], labels: list[str], *, quick: bool, jobs: int
) -> dict:
    print(f"\n{'=' * 76}\n{task.upper()} - hyperparameter search\n{'=' * 76}")

    started = time.perf_counter()
    joined, cleaned = prepare(texts)
    print(f"tokenised {len(texts)} documents in {time.perf_counter() - started:.0f}s")

    features = Combined(joined, cleaned).rows

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    print(f"train / test : {len(x_train)} / {len(x_test)}")
    print("selection    : 5-fold CV on the TRAINING split only\n")

    splitter = StratifiedKFold(
        n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE
    )

    leaderboard: list[dict] = []
    best_overall = None

    for name, classifier, grid in grids(quick):
        combinations = int(np.prod([len(v) for v in grid.values()]))
        print(f"  {name:20} searching {combinations} combinations...", flush=True)

        search = GridSearchCV(
            build_pipeline(classifier),
            grid,
            scoring="f1_macro",
            cv=splitter,
            n_jobs=jobs,
            refit=True,
        )
        began = time.perf_counter()
        search.fit(x_train, y_train)
        took = time.perf_counter() - began

        entry = {
            "model": name,
            "cv_f1_macro": float(search.best_score_),
            "params": {k: str(v) for k, v in search.best_params_.items()},
            "search_seconds": round(took, 1),
        }
        leaderboard.append(entry)
        print(
            f"  {name:20} best CV macro-F1 {search.best_score_:.4f}  ({took:.0f}s)"
        )

        if best_overall is None or search.best_score_ > best_overall[1].best_score_:
            best_overall = (name, search)

    assert best_overall is not None
    name, search = best_overall

    # The held-out set is touched exactly once, here.
    predictions = search.best_estimator_.predict(x_test)
    held_out = float(f1_score(y_test, predictions, average="macro", zero_division=0))
    accuracy = float((predictions == np.array(y_test)).mean())

    print(f"\n  winner       : {name}")
    print(f"  CV macro-F1  : {search.best_score_:.4f}")
    print(f"  HELD-OUT     : macro-F1 {held_out:.4f}   accuracy {accuracy:.4f}")
    print("\n  best parameters:")
    for key, value in sorted(search.best_params_.items()):
        print(f"    {key:44} {value}")

    print("\n  per-class on held-out:")
    print(classification_report(y_test, predictions, zero_division=0, digits=3))

    return {
        "winner": name,
        "cv_f1_macro": float(search.best_score_),
        "held_out_f1_macro": held_out,
        "held_out_accuracy": accuracy,
        "best_params": {k: str(v) for k, v in search.best_params_.items()},
        "leaderboard": sorted(leaderboard, key=lambda e: -e["cv_f1_macro"]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--task", choices=["topic", "sentiment", "both"], default="both")
    parser.add_argument("--quick", action="store_true", help="Smaller grid.")
    parser.add_argument(
        "--jobs", type=int, default=4, help="Parallel workers for the search."
    )
    args = parser.parse_args(argv)

    rows = list(csv.DictReader(open(DATASET, encoding="utf-8")))
    texts = [f"{r['title']} {r['content']}" for r in rows]

    tasks = ["topic", "sentiment"] if args.task == "both" else [args.task]
    results = {}
    for task in tasks:
        results[task] = tune_task(
            task, texts, [r[task] for r in rows], quick=args.quick, jobs=args.jobs
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
