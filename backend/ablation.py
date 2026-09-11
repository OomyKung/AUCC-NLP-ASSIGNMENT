"""Ablation experiments for the project report.

Runs three controlled comparisons on the same stratified split used everywhere
else (seed 42, 20% held out), so the numbers line up with train.py and
evaluate.py:

1. **Negation protection** -- PyThaiNLP's stopword list contains ``ไม่``. Does
   protecting negators from stopword removal actually matter for sentiment?
2. **Text aggregation** -- classifying one short unit at a time versus
   classifying several aggregated together. This is the windowing question:
   chat messages are individually too short, so does grouping help?
3. **Feature representation** -- word n-grams, character n-grams, or both.

Writes ``models/ablation.json`` and prints tables ready to paste into a report.

Usage::

    python ablation.py                # all three
    python ablation.py --only negation
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
from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import FeatureUnion, Pipeline  # noqa: E402

from app.nlp.features import thai_char_preprocessor  # noqa: E402
from train import TUNED  # noqa: E402
from app.nlp.preprocessing import clean_text, filter_tokens  # noqa: E402
from app.nlp.tokenizer import get_tokenizer  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "news_dataset.csv"
OUTPUT = Path(__file__).resolve().parent / "models" / "ablation.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2


# --------------------------------------------------------------------------
# Analysers (module level so the pipelines stay picklable)
# --------------------------------------------------------------------------


def _tokens(text: str, *, protect: bool) -> list[str]:
    cleaned = clean_text(text or "")
    return filter_tokens(get_tokenizer().tokenize(cleaned), protect_polarity=protect)


def analyser_protected(text: str) -> list[str]:
    """Negators kept (the project's default)."""
    tokens = _tokens(text, protect=True)
    return tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]


def analyser_unprotected(text: str) -> list[str]:
    """Negators removed along with other stopwords."""
    tokens = _tokens(text, protect=False)
    return tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]


def _tuned(task: str) -> dict:
    """train.py's fitted settings for ``task``.

    An ablation is only meaningful against the configuration that actually
    ships. Using generic defaults here would measure the effect of removing a
    feature from a model nobody runs.
    """
    return TUNED.get(task, {})


def _word_vectoriser(analyser, task: str = "topic") -> TfidfVectorizer:
    features = _tuned(task).get("features", {})
    return TfidfVectorizer(
        analyzer=analyser,
        min_df=features.get("word_min_df", 2),
        sublinear_tf=True,
    )


def _char_vectoriser(task: str = "topic") -> TfidfVectorizer:
    features = _tuned(task).get("features", {})
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=features.get("char_ngram_range", (2, 4)),
        preprocessor=thai_char_preprocessor,
        min_df=features.get("char_min_df", 3),
        sublinear_tf=True,
    )


def _classifier(task: str = "topic") -> LogisticRegression:
    return LogisticRegression(
        C=_tuned(task).get("classifier_c", 4.0),
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )


def _score(pipeline, x_train, y_train, x_test, y_test) -> dict:
    started = time.perf_counter()
    pipeline.fit(x_train, y_train)
    predictions = pipeline.predict(x_test)
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "fit_seconds": round(time.perf_counter() - started, 1),
    }


def load() -> tuple[list[str], list[str], list[str]]:
    with open(DATASET, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    texts = [f"{r['title']} {r['content']}".strip() for r in rows]
    return texts, [r["topic"] for r in rows], [r["sentiment"] for r in rows]


def split(texts, labels):
    return train_test_split(
        texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )


def table(title: str, rows: list[tuple[str, dict]], baseline_key: str) -> None:
    """Print a comparison table with deltas against the named baseline."""
    print(f"\n{title}\n{'-' * len(title)}")
    print(f"{'configuration':38} {'accuracy':>9} {'macro-F1':>9} {'Δ F1':>8}")
    base = dict(rows)[baseline_key]["f1_macro"]
    for name, metrics in rows:
        delta = metrics["f1_macro"] - base
        marker = "  (baseline)" if name == baseline_key else ""
        print(
            f"{name:38} {metrics['accuracy']:>9.4f} {metrics['f1_macro']:>9.4f} "
            f"{delta:>+8.4f}{marker}"
        )


# --------------------------------------------------------------------------
# 1. Negation protection
# --------------------------------------------------------------------------


def ablation_negation(texts, sentiments) -> dict:
    print("\n" + "=" * 74)
    print("ABLATION 1 - Does protecting negators from stopword removal matter?")
    print("=" * 74)
    print(
        "PyThaiNLP's stopword list contains ไม่ ('not'), the most frequent token in\n"
        "the corpus. Removing it turns ไม่ดี ('not good') into ดี ('good')."
    )

    x_train, x_test, y_train, y_test = split(texts, sentiments)
    results = {}

    for name, analyser in (
        ("negators protected (default)", analyser_protected),
        ("negators removed as stopwords", analyser_unprotected),
    ):
        pipeline = Pipeline(
            [
                ("features", _word_vectoriser(analyser, "sentiment")),
                ("classifier", _classifier("sentiment")),
            ]
        )
        results[name] = _score(pipeline, x_train, y_train, x_test, y_test)

    table(
        "Sentiment classification (word features only)",
        list(results.items()),
        "negators protected (default)",
    )

    # A concrete demonstration on minimal pairs, which is what a reader remembers.
    print("\n  Minimal pairs after stopword removal:")
    for text in ["ไม่ดี", "ดี", "ไม่ชอบเลย", "ทีมไทยเล่นไม่ดีมาก"]:
        kept = filter_tokens(get_tokenizer().tokenize(text), protect_polarity=True)
        dropped = filter_tokens(get_tokenizer().tokenize(text), protect_polarity=False)
        print(f"    {text:22} protected={kept}  removed={dropped}")

    return results


# --------------------------------------------------------------------------
# 2. Aggregation (the windowing question)
# --------------------------------------------------------------------------


def ablation_aggregation(texts, topics) -> dict:
    print("\n" + "=" * 74)
    print("ABLATION 2 - Does aggregating short units beat classifying them alone?")
    print("=" * 74)
    print(
        "Chat messages are individually short (median 19 characters), which is why\n"
        "the system groups them into windows. To test that decision on labelled\n"
        "data, each news document is split into sentences: 'per-unit' classifies\n"
        "one sentence at a time and takes a majority vote, 'aggregated' classifies\n"
        "the whole document at once."
    )

    tokenizer = get_tokenizer()
    x_train, x_test, y_train, y_test = split(texts, topics)

    # Train once on whole documents; only the inference strategy differs.
    pipeline = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        ("word", _word_vectoriser(analyser_protected, "topic")),
                        ("char", _char_vectoriser("topic")),
                    ]
                ),
            ),
            ("classifier", _classifier("topic")),
        ]
    )
    pipeline.fit(x_train, y_train)

    # Aggregated: the document as one input.
    aggregated = pipeline.predict(x_test)

    # Per-unit: classify each sentence, then majority vote.
    per_unit = []
    unit_counts = []
    for document in x_test:
        sentences = [s for s in tokenizer.sentences(document) if len(s.strip()) > 10]
        if not sentences:
            per_unit.append(pipeline.predict([document])[0])
            unit_counts.append(1)
            continue
        votes = pipeline.predict(sentences)
        unit_counts.append(len(sentences))
        values, counts = np.unique(votes, return_counts=True)
        per_unit.append(values[counts.argmax()])

    results = {
        "per-unit (majority vote)": {
            "accuracy": float(accuracy_score(y_test, per_unit)),
            "f1_macro": float(f1_score(y_test, per_unit, average="macro", zero_division=0)),
        },
        "aggregated (whole document)": {
            "accuracy": float(accuracy_score(y_test, aggregated)),
            "f1_macro": float(
                f1_score(y_test, aggregated, average="macro", zero_division=0)
            ),
        },
    }

    table(
        "Topic classification",
        list(results.items()),
        "per-unit (majority vote)",
    )

    lengths = [len(d) for d in x_test]
    print(
        f"\n  Mean units per document : {np.mean(unit_counts):.1f}\n"
        f"  Mean document length    : {np.mean(lengths):.0f} characters\n"
        f"  Mean unit length        : {np.mean(lengths) / np.mean(unit_counts):.0f} characters"
    )
    results["units_per_document"] = float(np.mean(unit_counts))
    return results


# --------------------------------------------------------------------------
# 3. Feature representation
# --------------------------------------------------------------------------


def ablation_features(texts, topics, sentiments) -> dict:
    print("\n" + "=" * 74)
    print("ABLATION 3 - Word n-grams, character n-grams, or both?")
    print("=" * 74)
    print(
        "Thai is written without spaces, so tokenisation can fail on informal or\n"
        "unseen words. Character n-grams do not depend on correct segmentation."
    )

    def configurations(task: str) -> dict:
        return {
            "word only": lambda: _word_vectoriser(analyser_protected, task),
            "char only": lambda: _char_vectoriser(task),
            "word + char (default)": lambda: FeatureUnion(
                [
                    ("word", _word_vectoriser(analyser_protected, task)),
                    ("char", _char_vectoriser(task)),
                ]
            ),
        }

    results: dict[str, dict] = {}
    for task, labels in (("topic", topics), ("sentiment", sentiments)):
        x_train, x_test, y_train, y_test = split(texts, labels)
        task_results = {}
        for name, build in configurations(task).items():
            pipeline = Pipeline(
                [("features", build()), ("classifier", _classifier(task))]
            )
            task_results[name] = _score(pipeline, x_train, y_train, x_test, y_test)
        results[task] = task_results
        table(
            f"{task.capitalize()} classification",
            list(task_results.items()),
            "word + char (default)",
        )

    return results


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--only",
        choices=["negation", "aggregation", "features"],
        help="Run a single ablation.",
    )
    args = parser.parse_args(argv)

    if not DATASET.is_file():
        raise SystemExit(f"Dataset not found: {DATASET}")

    texts, topics, sentiments = load()
    print(f"dataset: {len(texts)} documents, split {int((1-TEST_SIZE)*100)}/"
          f"{int(TEST_SIZE*100)} stratified, seed {RANDOM_STATE}")

    payload: dict = {"split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE}}

    if args.only in (None, "negation"):
        payload["negation"] = ablation_negation(texts, sentiments)
    if args.only in (None, "aggregation"):
        payload["aggregation"] = ablation_aggregation(texts, topics)
    if args.only in (None, "features"):
        payload["features"] = ablation_features(texts, topics, sentiments)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
