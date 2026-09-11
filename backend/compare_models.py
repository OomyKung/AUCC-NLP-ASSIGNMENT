"""Compare every sentiment approach on one held-out split.

Puts all candidates on the same 150 rows (seed 42, 20% held out) so the numbers
are directly comparable:

* the rule-based Thai lexicon (no training)
* TF-IDF + logistic regression (trained here)
* WangchanBERTa fine-tuned on this dataset
* an off-the-shelf Thai sentiment model from the Hub (no fine-tuning)

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

from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402

from app.config import settings  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "news_dataset.csv"
OUTPUT = Path(__file__).resolve().parent / "models" / "model_comparison.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2

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
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "predicted_distribution": {
            label: y_pred.count(label) for label in sorted(set(y_pred))
        },
    }
    print(
        f"  {name:44} acc {metrics['accuracy']:.4f}   macro-F1 {metrics['f1_macro']:.4f}"
    )
    return metrics


def main() -> int:
    rows = list(csv.DictReader(open(DATASET, encoding="utf-8")))
    texts = [f"{r['title']} {r['content']}" for r in rows]
    labels = [r["sentiment"] for r in rows]

    _x_train, x_test, _y_train, y_test = train_test_split(
        texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )
    print(f"held-out rows: {len(x_test)}  (seed {RANDOM_STATE}, stratified)\n")
    print("SENTIMENT - all approaches on identical rows")
    print("-" * 74)

    results: dict[str, dict] = {}

    # 1. Rule-based lexicon ------------------------------------------------
    from app.nlp.backends.lexicon_sentiment import LexiconSentimentBackend

    lexicon = LexiconSentimentBackend()
    results["lexicon (rules, no training)"] = score(
        "lexicon (rules, no training)",
        y_test,
        [lexicon.predict(t).label for t in x_test],
    )

    # 2. TF-IDF ------------------------------------------------------------
    from app.nlp.backends.sklearn_sentiment import SklearnSentimentBackend

    sklearn_backend = SklearnSentimentBackend.load()
    if sklearn_backend is not None:
        results["TF-IDF + logistic regression"] = score(
            "TF-IDF + logistic regression",
            y_test,
            [p.label for p in sklearn_backend.predict_many(x_test)],
        )

    # 3. Fine-tuned WangchanBERTa -----------------------------------------
    try:
        from app.nlp.backends.hf_transformer import HFSentimentBackend

        finetuned = HFSentimentBackend()
        results["WangchanBERTa fine-tuned (597 rows)"] = score(
            "WangchanBERTa fine-tuned (597 rows)",
            y_test,
            [p.label for p in finetuned.predict_many(x_test)],
        )
    except Exception as exc:
        print(f"  WangchanBERTa fine-tuned: unavailable ({type(exc).__name__})")

    # 4. Off-the-shelf Hub model ------------------------------------------
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
                predictions.extend(
                    normalise(model.config.id2label[i]) for i in indices
                )

        results[f"off-the-shelf: {name.split('/')[-1]}"] = score(
            f"off-the-shelf ({name.split('/')[-1][:28]})", y_test, predictions
        )
    except Exception as exc:
        print(f"  off-the-shelf model: unavailable ({type(exc).__name__}: {exc})")

    # Detail for the best and worst, which is what a report needs.
    print()
    ranked = sorted(results.items(), key=lambda kv: -kv[1]["f1_macro"])
    print(f"Best : {ranked[0][0]}  (macro-F1 {ranked[0][1]['f1_macro']:.4f})")
    print(f"Worst: {ranked[-1][0]}  (macro-F1 {ranked[-1][1]['f1_macro']:.4f})")

    print("\nPredicted-label distribution (true: positive 69, neutral 31, negative 50):")
    for name, metrics in ranked:
        print(f"  {name:44} {metrics['predicted_distribution']}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "split": {"test_size": TEST_SIZE, "random_state": RANDOM_STATE},
                "held_out_rows": len(x_test),
                "true_distribution": {
                    label: y_test.count(label) for label in sorted(set(y_test))
                },
                "models": results,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
