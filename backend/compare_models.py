"""Compare every modelling approach on one held-out split, for both tasks.

All candidates are scored on the same 150 rows (seed 42, 20% held out, the same
split train.py and evaluate.py use), so the numbers are directly comparable.

Sentiment:

* the rule-based Thai lexicon (no training)
* TF-IDF + logistic regression
* WangchanBERTa fine-tuned on this dataset
* an off-the-shelf Thai sentiment model from the Hub (no fine-tuning)

Topic:

* the rule-based gazetteer (no training)
* TF-IDF + logistic regression
* WangchanBERTa fine-tuned on this dataset

Prints the predicted-label distribution alongside the headline metrics, because
that is what makes a failure mode legible -- a model that collapses onto one
class can still post a respectable-looking accuracy.

Writes ``models/model_comparison.json``.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from sklearn.metrics import accuracy_score, f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from app.config import settings  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "news_dataset.csv"
OUTPUT = Path(__file__).resolve().parent / "models" / "model_comparison.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Filled in from the actual split, so the label cannot drift from the data the
# transformer was fine-tuned on. It said "597 rows" for a while after the
# dataset grew to 836.
TRAIN_ROWS = 0

# Hub models use their own label vocabularies.
LABEL_ALIASES = {
    "pos": "positive",
    "neu": "neutral",
    "neg": "negative",
    "positive": "positive",
    "neutral": "neutral",
    "negative": "negative",
    "label_0": "positive",
    "label_1": "neutral",
    "label_2": "negative",
}


def normalise(label: str) -> str:
    return LABEL_ALIASES.get(str(label).strip().lower(), "neutral")


def score(name: str, y_true: list[str], y_pred: list[str]) -> dict:
    """Score one model and print its row."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "distinct_predicted": len(set(y_pred)),
        "predicted_distribution": {
            label: y_pred.count(label) for label in sorted(set(y_pred))
        },
    }
    print(
        f"  {name:44} acc {metrics['accuracy']:.4f}   "
        f"macro-F1 {metrics['f1_macro']:.4f}"
    )
    return metrics


def compare_sentiment(x_test: list[str], y_test: list[str]) -> dict:
    print("\nSENTIMENT - all approaches on identical rows")
    print("-" * 76)
    results: dict[str, dict] = {}

    from app.nlp.backends.lexicon_sentiment import LexiconSentimentBackend

    lexicon = LexiconSentimentBackend()
    results["lexicon (rules, no training)"] = score(
        "lexicon (rules, no training)", y_test, [lexicon.predict(t).label for t in x_test]
    )

    from app.nlp.backends.sklearn_sentiment import SklearnSentimentBackend

    classical = SklearnSentimentBackend.load()
    if classical is not None:
        results["TF-IDF + logistic regression"] = score(
            "TF-IDF + logistic regression",
            y_test,
            [p.label for p in classical.predict_many(x_test)],
        )

    try:
        from app.nlp.backends.hf_transformer import HFSentimentBackend

        finetuned = HFSentimentBackend()
        results[f"WangchanBERTa fine-tuned ({TRAIN_ROWS} rows)"] = score(
            f"WangchanBERTa fine-tuned ({TRAIN_ROWS} rows)",
            y_test,
            [p.label for p in finetuned.predict_many(x_test)],
        )
    except Exception as exc:
        print(f"  WangchanBERTa fine-tuned: unavailable ({type(exc).__name__})")

    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        name = settings.hf_sentiment_model
        tokenizer = AutoTokenizer.from_pretrained(name)
        model = AutoModelForSequenceClassification.from_pretrained(name)
        model.eval()

        predictions: list[str] = []
        with torch.no_grad():
            for start in range(0, len(x_test), 16):
                encoded = tokenizer(
                    x_test[start : start + 16],
                    truncation=True,
                    padding=True,
                    max_length=128,
                    return_tensors="pt",
                )
                indices = model(**encoded).logits.argmax(-1).tolist()
                predictions.extend(normalise(model.config.id2label[i]) for i in indices)

        short = name.split("/")[-1]
        results[f"off-the-shelf: {short}"] = score(
            f"off-the-shelf ({short[:28]})", y_test, predictions
        )
    except Exception as exc:
        print(f"  off-the-shelf model: unavailable ({type(exc).__name__})")

    return results


def compare_topic(x_test: list[str], y_test: list[str]) -> dict:
    print("\nTOPIC - all approaches on identical rows")
    print("-" * 76)
    results: dict[str, dict] = {}

    from app.nlp.backends.gazetteer_topic import GazetteerTopicBackend

    gazetteer = GazetteerTopicBackend()
    results["gazetteer (rules, no training)"] = score(
        "gazetteer (rules, no training)",
        y_test,
        [gazetteer.predict(t).label for t in x_test],
    )

    from app.nlp.backends.sklearn_topic import SklearnTopicBackend

    classical = SklearnTopicBackend.load()
    if classical is not None:
        results["TF-IDF + logistic regression"] = score(
            "TF-IDF + logistic regression",
            y_test,
            [classical.predict(t).label for t in x_test],
        )

    try:
        from app.nlp.backends.hf_transformer import HFTopicBackend

        finetuned = HFTopicBackend()
        results[f"WangchanBERTa fine-tuned ({TRAIN_ROWS} rows)"] = score(
            f"WangchanBERTa fine-tuned ({TRAIN_ROWS} rows)",
            y_test,
            [p.label for p in finetuned.predict_many(x_test)],
        )
    except Exception as exc:
        print(f"  WangchanBERTa fine-tuned: unavailable ({type(exc).__name__})")

    return results


def show_distributions(results: dict, y_test: list[str], classes: int) -> None:
    """Print predicted-label distributions, which reveal class collapse."""
    truth = ", ".join(
        f"{label} {y_test.count(label)}" for label in sorted(set(y_test))
    )
    print(f"\n  predicted distribution (true: {truth})")
    for name, metrics in sorted(results.items(), key=lambda kv: -kv[1]["f1_macro"]):
        used = metrics["distinct_predicted"]
        flag = "  <- collapsed" if used <= max(1, classes // 3) else ""
        if classes <= 3:
            print(f"    {name:44} {metrics['predicted_distribution']}{flag}")
        else:
            print(f"    {name:44} uses {used}/{classes} classes{flag}")


def main() -> int:
    rows = list(csv.DictReader(open(DATASET, encoding="utf-8")))
    texts = [f"{r['title']} {r['content']}" for r in rows]

    payload: dict = {
        "split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE},
        "tasks": {},
    }

    for task, comparator in (
        ("sentiment", compare_sentiment),
        ("topic", compare_topic),
    ):
        labels = [r[task] for r in rows]
        _x_train, x_test, _y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=labels,
        )
        global TRAIN_ROWS
        TRAIN_ROWS = len(_x_train)
        if task == "sentiment":
            print(
                f"held-out rows: {len(x_test)}  train rows: {TRAIN_ROWS}  "
                f"(seed {RANDOM_STATE}, stratified)"
            )

        results = comparator(x_test, y_test)
        classes = len(set(labels))
        show_distributions(results, y_test, classes)

        if results:
            ranked = sorted(results.items(), key=lambda kv: -kv[1]["f1_macro"])
            best_name, best = ranked[0]
            print(f"\n  best: {best_name} (macro-F1 {best['f1_macro']:.4f})")
            transformer = results.get(f"WangchanBERTa fine-tuned ({TRAIN_ROWS} rows)")
            classical = results.get("TF-IDF + logistic regression")
            if transformer and classical:
                gap = classical["f1_macro"] - transformer["f1_macro"]
                print(f"  TF-IDF minus transformer: {gap:+.4f} macro-F1")

        payload["tasks"][task] = {
            "held_out_rows": len(x_test),
            "classes": classes,
            "true_distribution": {
                label: y_test.count(label) for label in sorted(set(y_test))
            },
            "models": results,
        }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nWrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
