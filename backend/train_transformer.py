"""Fine-tune WangchanBERTa for Thai topic and sentiment classification.

WangchanBERTa is the standard Thai pre-trained language model. It uses
SentencePiece subword units, which is why it is worth comparing against the
TF-IDF models: the character n-gram ablation showed subword information carries
most of the signal for Thai sentiment, and a subword transformer models that
natively instead of approximating it.

Uses the **same** stratified split as train.py (seed 42, 20% held out), so the
numbers sit directly beside the classical models in evaluate.py.

A plain PyTorch loop is used rather than the Trainer API: it is a handful of
lines, has no version-churn surprises, and makes the training procedure legible
for a project report.

Usage::

    python train_transformer.py                    # both tasks
    python train_transformer.py --task sentiment   # one task
    python train_transformer.py --epochs 4 --batch-size 8
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

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import accuracy_score, classification_report, f1_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from app.config import settings  # noqa: E402
from app.taxonomy import SENTIMENT_SLUGS, TOPIC_SLUGS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "data" / "news_dataset.csv"

MODEL_NAME = "airesearch/wangchanberta-base-att-spm-uncased"
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_LENGTH = 256


class TextDataset(Dataset):
    """Tokenised texts with integer labels."""

    def __init__(self, texts: list[str], labels: list[int], tokenizer) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def load_dataset() -> tuple[list[str], dict[str, list[str]]]:
    with open(DATASET, encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    texts = [f"{row['title']} {row['content']}".strip() for row in rows]
    return texts, {
        "topic": [row["topic"].strip() for row in rows],
        "sentiment": [row["sentiment"].strip() for row in rows],
    }


def train_task(
    task: str,
    texts: list[str],
    labels: list[str],
    label_order: tuple[str, ...],
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> dict:
    """Fine-tune one classifier and return its metrics."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    print(f"\n{'=' * 74}\n{task.upper()} - fine-tuning {MODEL_NAME}\n{'=' * 74}")

    present = [label for label in label_order if label in set(labels)]
    to_index = {label: index for index, label in enumerate(present)}

    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels
    )
    print(f"classes      : {len(present)}")
    print(f"train / test : {len(x_train)} / {len(x_test)}")

    torch.manual_seed(RANDOM_STATE)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(present),
        id2label={index: label for label, index in to_index.items()},
        label2id=to_index,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"device       : {device}")

    train_loader = DataLoader(
        TextDataset(x_train, [to_index[y] for y in y_train], tokenizer),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = DataLoader(
        TextDataset(x_test, [to_index[y] for y in y_test], tokenizer),
        batch_size=batch_size,
    )

    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=learning_rate, total_steps=total_steps, pct_start=0.1
    )

    # Class weights: the sentiment set is imbalanced (345/250/152), and the
    # classical models use balanced weights, so this keeps the comparison fair.
    counts = np.bincount([to_index[y] for y in y_train], minlength=len(present))
    weights = torch.tensor(
        len(y_train) / (len(present) * np.maximum(counts, 1)), dtype=torch.float
    ).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    started = time.perf_counter()
    model.train()
    for epoch in range(1, epochs + 1):
        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = {key: value.to(device) for key, value in batch.items()}
            labels_batch = batch.pop("labels")

            optimiser.zero_grad()
            logits = model(**batch).logits
            loss = loss_fn(logits, labels_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()

            running += loss.item()
            if step % 10 == 0 or step == len(train_loader):
                elapsed = time.perf_counter() - started
                print(
                    f"  epoch {epoch}/{epochs}  step {step:>3}/{len(train_loader)}  "
                    f"loss {running / step:.4f}  ({elapsed:.0f}s)",
                    flush=True,
                )

    fit_seconds = time.perf_counter() - started
    print(f"trained in   : {fit_seconds:.0f}s")

    # ------------------------------------------------------------- evaluate
    model.eval()
    predictions: list[int] = []
    truths: list[int] = []
    with torch.no_grad():
        for batch in test_loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            labels_batch = batch.pop("labels")
            logits = model(**batch).logits
            predictions.extend(logits.argmax(dim=-1).cpu().tolist())
            truths.extend(labels_batch.cpu().tolist())

    index_to_label = {index: label for label, index in to_index.items()}
    y_pred = [index_to_label[i] for i in predictions]
    y_true = [index_to_label[i] for i in truths]

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "model": MODEL_NAME,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "fit_seconds": round(fit_seconds, 1),
        "trained_at": datetime.now(UTC).isoformat(),
        "labels": present,
    }

    print(
        f"\naccuracy     : {metrics['accuracy']:.4f}\n"
        f"f1 macro     : {metrics['f1_macro']:.4f}"
    )
    print("\nper-class report:")
    print(classification_report(y_true, y_pred, zero_division=0, digits=3))

    # --------------------------------------------------------------- save
    target = settings.model_dir / f"wangchanberta_{task}"
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target)
    tokenizer.save_pretrained(target)
    (target / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"saved        : {target}")

    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--task", choices=["topic", "sentiment", "both"], default="both"
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    args = parser.parse_args(argv)

    if not DATASET.is_file():
        raise SystemExit(f"Dataset not found: {DATASET}")

    texts, labels = load_dataset()
    tasks = ["topic", "sentiment"] if args.task == "both" else [args.task]
    order = {"topic": TOPIC_SLUGS, "sentiment": SENTIMENT_SLUGS}

    summary = {}
    for task in tasks:
        summary[task] = train_task(
            task,
            texts,
            labels[task],
            order[task],
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )

    path = settings.model_dir / "transformer_metrics.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nWrote {path}")
    print("\nEnable it with:  NLP_TOPIC_BACKEND=transformer  (and/or sentiment)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
