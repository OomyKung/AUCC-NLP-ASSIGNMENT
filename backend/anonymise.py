"""Replace YouTube author handles with stable pseudonyms.

The collected chat is real, and a public repository should not carry 5,706
identifiable accounts alongside their political opinions. Message text,
timestamps and every derived metric are untouched -- only the ``author`` field
changes.

The pseudonym is a keyed hash of the handle, so:

* the same person always maps to the same pseudonym, which is what the
  message drill-down needs in order to distinguish speakers, and what the
  deduplication fingerprint (author + text + timestamp) relies on
* the mapping is not reversible without the original handle

The author field never reaches the NLP pipeline, so no result changes.

Usage::

    python anonymise.py --check     # report what would change
    python anonymise.py             # rewrite snapshots and the database
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402
from app.database.session import SessionLocal  # noqa: E402
from app.models.chat import ChatMessage  # noqa: E402

# Fixed salt: the goal is to remove identifiability from the published data,
# not to defend against an attacker who has this file and a handle list.
# Keeping it in-repo makes the transformation reproducible for anyone verifying
# the dataset.
SALT = "thai-news-intelligence-pseudonym-v1"

PREFIX = "ผู้ชม"
ALREADY_DONE_MARKER = PREFIX + " #"


def pseudonym(handle: str) -> str:
    """Stable, non-reversible pseudonym for a YouTube handle.

    A 4-byte digest is used rather than a shorter one on purpose. The corpus has
    5,706 distinct handles; with a 2-byte digest (65,536 buckets) the birthday
    bound predicts roughly 250 collisions, which would merge distinct speakers
    and defeat the message drill-down. 4 bytes makes an expected collision count
    of about 0.004.
    """
    digest = hashlib.blake2s(
        f"{SALT}|{handle}".encode(), digest_size=4
    ).hexdigest()
    return f"{ALREADY_DONE_MARKER}{digest}"


def convert(handle: str | None) -> str | None:
    """Pseudonymise a handle, leaving already-converted values alone."""
    if not handle:
        return handle
    if handle.startswith(ALREADY_DONE_MARKER):
        return handle
    return pseudonym(handle)


def process_snapshots(*, dry_run: bool) -> tuple[int, int, int]:
    """Rewrite the committed snapshots in place.

    Returns ``(files, messages, distinct_handles)``.
    """
    files = 0
    messages = 0
    handles: set[str] = set()

    for path in sorted(settings.chat_snapshot_dir.glob("*.jsonl")):
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines:
            continue

        output: list[str] = []
        changed = False

        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                output.append(line)
                continue

            # The header record carries no author.
            if index == 0 and "_stream" in record:
                output.append(json.dumps(record, ensure_ascii=False))
                continue

            author = record.get("author")
            if author and not str(author).startswith(ALREADY_DONE_MARKER):
                handles.add(str(author))
                record["author"] = convert(str(author))
                changed = True
                messages += 1

            output.append(json.dumps(record, ensure_ascii=False))

        if changed:
            files += 1
            if not dry_run:
                path.write_text(
                    "\n".join(output) + "\n", encoding="utf-8", newline="\n"
                )

    return files, messages, len(handles)


def process_database(*, dry_run: bool) -> tuple[int, int]:
    """Rewrite stored chat messages. Returns ``(updated, distinct_handles)``."""
    updated = 0
    handles: set[str] = set()

    with SessionLocal() as db:
        rows = list(db.scalars(select(ChatMessage)))
        for row in rows:
            if row.author and not row.author.startswith(ALREADY_DONE_MARKER):
                handles.add(row.author)
                if not dry_run:
                    row.author = convert(row.author)
                updated += 1
        if not dry_run:
            db.commit()

    return updated, len(handles)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check",
        action="store_true",
        help="Report what would change without writing anything.",
    )
    args = parser.parse_args(argv)

    mode = "CHECK (no files written)" if args.check else "REWRITING"
    print(f"{mode}\n")

    print("Snapshots:")
    files, messages, handles = process_snapshots(dry_run=args.check)
    print(f"  files affected     {files}")
    print(f"  messages           {messages:,}")
    print(f"  distinct handles   {handles:,}")

    print("\nDatabase:")
    try:
        updated, db_handles = process_database(dry_run=args.check)
        print(f"  messages           {updated:,}")
        print(f"  distinct handles   {db_handles:,}")
    except Exception as exc:  # database may not exist yet
        print(f"  skipped: {exc}")

    print("\nExample mapping:")
    for handle in ["@I-nawa01", "@thairathnews", "@Padid-l9t"]:
        print(f"  {handle:20} -> {pseudonym(handle)}")

    if args.check:
        print("\nRun without --check to apply.")
    else:
        print("\nDone. Message text, timestamps and all metrics are unchanged.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
