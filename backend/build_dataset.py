"""Merge the per-category dataset parts into the training and sample files.

Produces:

* ``data/news_dataset.csv``  -- the training set (spec section 21 columns:
  title, content, topic, sentiment)
* ``data/sample_news.csv``   -- a 50-row display-oriented seed (spec section 14)
  with source, url and published_at added

Validates as it goes: label validity, duplicate detection, per-class balance and
minimum content length. A dataset that silently contains duplicates or unknown
labels would quietly inflate the evaluation figures, so this fails loudly.
"""

from __future__ import annotations

import csv
import random
import sys
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from app.taxonomy import SENTIMENT_SLUGS, TOPIC_SLUGS  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
PARTS_DIR = ROOT / "data" / "dataset_parts"
DATASET_PATH = ROOT / "data" / "news_dataset.csv"
SAMPLE_PATH = ROOT / "data" / "sample_news.csv"

MIN_CONTENT_CHARS = 40
SAMPLE_SIZE = 50
SEED = 20260910

# Plausible Thai news outlets, assigned to sample rows for display only.
SOURCES = (
    "ไทยรัฐ",
    "เดลินิวส์",
    "มติชน",
    "ข่าวสด",
    "ประชาชาติธุรกิจ",
    "กรุงเทพธุรกิจ",
    "PPTV",
    "Thai PBS",
    "TNN",
    "อมรินทร์ทีวี",
)


def load_parts() -> list[dict]:
    """Read every part file, in filename order."""
    rows: list[dict] = []
    files = sorted(PARTS_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"No dataset parts found in {PARTS_DIR}")

    for path in files:
        with open(path, encoding="utf-8", newline="") as handle:
            part = list(csv.DictReader(handle))
        print(f"  {path.name:24} {len(part):3} rows")
        for row in part:
            row["_file"] = path.name
        rows.extend(part)
    return rows


def validate(rows: list[dict]) -> list[str]:
    """Return a list of problems found. Empty means the dataset is sound."""
    problems: list[str] = []

    seen_titles: dict[str, str] = {}
    seen_contents: dict[str, str] = {}

    for index, row in enumerate(rows, start=1):
        where = f"{row['_file']} row {index}"

        title = (row.get("title") or "").strip()
        content = (row.get("content") or "").strip()
        topic = (row.get("topic") or "").strip()
        sentiment = (row.get("sentiment") or "").strip()

        if not title:
            problems.append(f"{where}: empty title")
        if len(content) < MIN_CONTENT_CHARS:
            problems.append(
                f"{where}: content shorter than {MIN_CONTENT_CHARS} chars"
            )
        if topic not in TOPIC_SLUGS:
            problems.append(f"{where}: unknown topic {topic!r}")
        if sentiment not in SENTIMENT_SLUGS:
            problems.append(f"{where}: unknown sentiment {sentiment!r}")

        # Duplicates would inflate accuracy by leaking train rows into test.
        if title in seen_titles:
            problems.append(f"{where}: duplicate title, also in {seen_titles[title]}")
        else:
            seen_titles[title] = where

        if content in seen_contents:
            problems.append(
                f"{where}: duplicate content, also in {seen_contents[content]}"
            )
        else:
            seen_contents[content] = where

    return problems


def report(rows: list[dict]) -> None:
    """Print the class balance, which is what makes the metrics interpretable."""
    topics = Counter(row["topic"] for row in rows)
    sentiments = Counter(row["sentiment"] for row in rows)

    print(f"\n{'topic':16} {'rows':>5}  {'pos':>4} {'neu':>4} {'neg':>4}")
    print("-" * 44)
    for slug in TOPIC_SLUGS:
        subset = [row for row in rows if row["topic"] == slug]
        counts = Counter(row["sentiment"] for row in subset)
        print(
            f"{slug:16} {len(subset):>5}  "
            f"{counts['positive']:>4} {counts['neutral']:>4} {counts['negative']:>4}"
        )
    print("-" * 44)
    print(f"{'TOTAL':16} {len(rows):>5}  ", end="")
    print(
        f"{sentiments['positive']:>4} {sentiments['neutral']:>4} "
        f"{sentiments['negative']:>4}"
    )

    sizes = set(topics.values())
    if len(sizes) == 1:
        print(f"\nPerfectly balanced: {sizes.pop()} rows per topic.")
    else:
        print(f"\nTopic sizes range {min(topics.values())}-{max(topics.values())}.")


def write_dataset(rows: list[dict]) -> None:
    """Write the training CSV with exactly the four specified columns."""
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_PATH, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["title", "content", "topic", "sentiment"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "title": row["title"].strip(),
                    "content": row["content"].strip(),
                    "topic": row["topic"].strip(),
                    "sentiment": row["sentiment"].strip(),
                }
            )


def write_sample(rows: list[dict]) -> None:
    """Write a stratified 50-row sample with display metadata.

    Stratified so all 15 categories appear, rather than a random draw that
    might omit several.
    """
    rng = random.Random(SEED)

    by_topic: dict[str, list[dict]] = {}
    for row in rows:
        by_topic.setdefault(row["topic"], []).append(row)

    picked: list[dict] = []
    # At least three per topic (45), then fill to 50 at random.
    for slug in TOPIC_SLUGS:
        candidates = by_topic.get(slug, [])
        picked.extend(rng.sample(candidates, min(3, len(candidates))))

    remaining = [row for row in rows if row not in picked]
    rng.shuffle(remaining)
    picked.extend(remaining[: max(0, SAMPLE_SIZE - len(picked))])
    picked = picked[:SAMPLE_SIZE]
    rng.shuffle(picked)

    now = datetime.now(UTC)
    with open(SAMPLE_PATH, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "title",
                "content",
                "topic",
                "sentiment",
                "source",
                "url",
                "published_at",
            ],
        )
        writer.writeheader()
        for offset, row in enumerate(picked):
            published = now - timedelta(days=offset % 30, hours=rng.randint(0, 23))
            writer.writerow(
                {
                    "title": row["title"].strip(),
                    "content": row["content"].strip(),
                    "topic": row["topic"].strip(),
                    "sentiment": row["sentiment"].strip(),
                    "source": rng.choice(SOURCES),
                    "url": "",
                    "published_at": published.isoformat(),
                }
            )


def main() -> int:
    print("Reading dataset parts:")
    rows = load_parts()

    problems = validate(rows)
    if problems:
        print(f"\n{len(problems)} problem(s) found:", file=sys.stderr)
        for problem in problems[:40]:
            print(f"  - {problem}", file=sys.stderr)
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more", file=sys.stderr)
        return 1

    report(rows)

    write_dataset(rows)
    write_sample(rows)

    print(f"\nWrote {DATASET_PATH.relative_to(ROOT)}  ({len(rows)} rows)")
    print(f"Wrote {SAMPLE_PATH.relative_to(ROOT)}  ({SAMPLE_SIZE} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
