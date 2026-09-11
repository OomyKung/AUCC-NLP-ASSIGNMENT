"""Loader for the Wisesight Sentiment corpus.

Why this corpus is here
-----------------------
``data/news_dataset.csv`` is authored, 747 rows, and written in news-prose
register. Two consequences were measured rather than assumed:

* Sentiment plateaus around macro-F1 0.70, with ``neutral`` the weakest class
  (only 152 training examples). Hyperparameter search moved it by ~0.02, which
  is the signature of a data limit rather than a modelling one.
* Per-message chat sentiment had to be routed to the rule-based lexicon backend
  (``NLP_CHAT_SENTIMENT_BACKEND=lexicon``) because a model trained on news prose
  scores live-chat text badly -- different vocabulary, different length,
  different conventions.

Wisesight Sentiment addresses both. It is ~21.6k human-labelled **real Thai
social-media messages**: roughly 29x the training data, in the register the chat
pipeline actually sees, from a citable public source instead of an authored file.

Provenance
----------
Wisesight (Thailand) / PyThaiNLP, released CC0-1.0 (public domain). Distributed
via the Hugging Face Hub as ``pythainlp/wisesight_sentiment``. It ships an
official train/validation/test split, which this module preserves -- benchmark
numbers are only comparable against other published work if the official test
split is used unchanged.

Labels
------
The corpus is 4-class with label order ``["pos", "neu", "neg", "q"]``. The
``q`` class ("question") is not a sentiment: it marks interrogative posts. For a
3-class task it is **dropped** by default rather than folded into neutral, which
would pollute the class this project most needs to get right. ``--keep-questions``
maps it to neutral instead, for the 4-class-comparable run.

The files are not committed. They are downloaded on demand and cached under
``data/wisesight/``, so the repository carries no redistributed corpus and the
download is a single documented command.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.config import settings

REPO_ID = "pythainlp/wisesight_sentiment"
REPO_TYPE = "dataset"
LICENSE = "CC0-1.0"

# Official split filenames in the Hub repository.
SPLIT_FILES = {
    "train": "wisesight_sentiment/train-00000-of-00001.parquet",
    "validation": "wisesight_sentiment/validation-00000-of-00001.parquet",
    "test": "wisesight_sentiment/test-00000-of-00001.parquet",
}

# The corpus stores integer categories. This order is the corpus's own
# ``ClassLabel`` order -- not a guess: the resulting per-class counts reproduce
# the published distribution (train: neu 11795, neg 5491, pos 3866, q 476).
CATEGORY_TO_LABEL = {0: "positive", 1: "neutral", 2: "negative", 3: "question"}

# Where the mapped corpus is cached. Gitignored: see the module docstring.
CACHE_DIR = settings.data_dir / "wisesight"


@dataclass(slots=True)
class Split:
    """One split of the corpus, already mapped to this project's label names."""

    name: str
    texts: list[str]
    labels: list[str]

    def __len__(self) -> int:
        return len(self.texts)

    def distribution(self) -> dict[str, int]:
        return {label: self.labels.count(label) for label in sorted(set(self.labels))}


class CorpusUnavailable(RuntimeError):
    """Raised when the corpus can be neither read from cache nor downloaded.

    A distinct type so callers can tell "no corpus" apart from a genuine bug and
    print an actionable message instead of a traceback.
    """


def _cache_path(split: str) -> Path:
    return CACHE_DIR / f"{split}.csv"


def _download(split: str) -> Path:
    """Fetch one split's parquet from the Hub."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:  # pragma: no cover - dependency is declared
        raise CorpusUnavailable(
            "huggingface_hub is required to download the corpus: "
            "pip install -r requirements.txt"
        ) from exc

    try:
        return Path(
            hf_hub_download(REPO_ID, SPLIT_FILES[split], repo_type=REPO_TYPE)
        )
    except Exception as exc:
        raise CorpusUnavailable(
            f"could not download the {split} split of {REPO_ID}: {exc}"
        ) from exc


def _read_parquet(path: Path) -> list[tuple[str, int]]:
    """Read (text, category) pairs from a parquet file."""
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise CorpusUnavailable(
            "pyarrow is required to read the corpus: pip install -r requirements.txt"
        ) from exc

    table = parquet.read_table(path)
    columns = table.column_names
    if "texts" not in columns or "category" not in columns:
        raise CorpusUnavailable(
            f"unexpected corpus schema {columns!r}; expected 'texts' and 'category'"
        )
    texts = table.column("texts").to_pylist()
    categories = table.column("category").to_pylist()
    return list(zip(texts, categories, strict=True))


def _write_cache(split: str, rows: list[tuple[str, str]]) -> Path:
    """Cache a mapped split as CSV, so later runs need no network."""
    import csv

    target = _cache_path(split)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["text", "sentiment"])
        writer.writerows(rows)
    return target


def _read_cache(split: str) -> list[tuple[str, str]] | None:
    import csv

    source = _cache_path(split)
    if not source.is_file():
        return None
    with open(source, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [(row["text"], row["sentiment"]) for row in reader]
    return rows or None


def load_split(
    split: str,
    *,
    keep_questions: bool = False,
    refresh: bool = False,
) -> Split:
    """Load one official split, from cache when available.

    Args:
        split: ``train``, ``validation`` or ``test``.
        keep_questions: Map the ``q`` class to neutral instead of dropping it.
        refresh: Ignore the cache and re-download.
    """
    if split not in SPLIT_FILES:
        raise ValueError(f"unknown split {split!r}; expected one of {list(SPLIT_FILES)}")

    cached = None if refresh else _read_cache(split)
    if cached is None:
        pairs = _read_parquet(_download(split))
        cached = [
            (text, CATEGORY_TO_LABEL.get(category, "neutral"))
            for text, category in pairs
        ]
        _write_cache(split, cached)

    texts: list[str] = []
    labels: list[str] = []
    for text, label in cached:
        # Some rows are blank or whitespace-only; they carry no signal and would
        # only add noise to both training and the reported metrics.
        if not text or not text.strip():
            continue
        if label == "question":
            if not keep_questions:
                continue
            label = "neutral"
        texts.append(text)
        labels.append(label)

    return Split(name=split, texts=texts, labels=labels)


def load_all(
    *, keep_questions: bool = False, refresh: bool = False
) -> dict[str, Split]:
    """Load every official split."""
    return {
        name: load_split(name, keep_questions=keep_questions, refresh=refresh)
        for name in SPLIT_FILES
    }


def is_cached() -> bool:
    """True when every split is already on disk."""
    return all(_cache_path(split).is_file() for split in SPLIT_FILES)


def citation() -> str:
    """Attribution string, for the README and the Evaluation page."""
    return (
        "Wisesight Sentiment Corpus (Wisesight Thailand / PyThaiNLP), "
        f"{LICENSE}, via Hugging Face Hub: {REPO_ID}"
    )
