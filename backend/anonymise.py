r"""Replace YouTube handles with stable pseudonyms.

The collected chat is real, and a public repository should not carry 5,706
identifiable accounts alongside their political opinions.

**Two** places carry handles, and missing the second is easy:

1. The ``author`` field -- who sent the message.
2. ``@mentions`` inside ``text`` -- viewers replying to each other by handle.
   These are just as identifying as the author field, and an earlier version of
   this script rewrote only the author, leaving 98 raw handles in the published
   snapshots.

The pseudonym is a keyed hash of the handle, so:

* the same person always maps to the same pseudonym, which is what the
  message drill-down needs in order to distinguish speakers, and what the
  deduplication fingerprint (author + text + timestamp) relies on
* a mention of someone who also posts maps to that poster's pseudonym, so the
  reply structure of the conversation survives
* the mapping is not reversible without the original handle

Why no analysis result changes
------------------------------
The author field never reaches the NLP pipeline. Mentions do reach it, but
``clean_text`` already strips ``@[\w.\-]+`` before tokenisation, so the raw
handle was never a feature. The mention pseudonym is deliberately **ASCII**
(``@viewer-<digest>``) so that the same regex still removes it completely --
verified to produce byte-identical cleaned text and identical tokens. A Thai
pseudonym would not: ``\w`` stops at Thai combining marks, which would leave
fragments like ``ชม`` behind and inject them into keyword extraction.

Timestamps and every derived metric are untouched.

Usage::

    python anonymise.py --check     # report what would change
    python anonymise.py             # rewrite snapshots and the database
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

from sqlalchemy import select  # noqa: E402

from app.config import settings  # noqa: E402
from app.database.session import SessionLocal  # noqa: E402
from app.models.chat import ChatMessage  # noqa: E402
from app.models.news import NewsArticle  # noqa: E402

# Fixed salt: the goal is to remove identifiability from the published data,
# not to defend against an attacker who has this file and a handle list.
# Keeping it in-repo makes the transformation reproducible for anyone verifying
# the dataset.
SALT = "thai-news-intelligence-pseudonym-v1"

PREFIX = "ผู้ชม"
ALREADY_DONE_MARKER = PREFIX + " #"

# In-message mentions. YouTube handles are ASCII: letters, digits, dot, hyphen,
# underscore. The pattern is anchored on "@" and requires at least three
# following characters, so a bare "@" or "@ " is left alone.
MENTION = re.compile(r"@([A-Za-z0-9_.\-]{3,})")
# The replacement stays ASCII on purpose -- see the module docstring.
MENTION_PREFIX = "@viewer-"


def mention_pseudonym(handle: str) -> str:
    """Pseudonym for an in-message mention, keyed identically to the author.

    The leading "@" is included in the hashed value so a mention of ``@alice``
    and an author field of ``@alice`` produce the same digest, which is what
    keeps the reply graph intact.
    """
    digest = hashlib.blake2s(
        f"{SALT}|@{handle}".encode(), digest_size=4
    ).hexdigest()
    return f"{MENTION_PREFIX}{digest}"


def convert_text(text: str | None) -> tuple[str | None, set[str]]:
    """Pseudonymise every mention in ``text``.

    Returns the rewritten text and the raw handles that were found, so the
    caller can report scale. Idempotent: an already-converted mention hashes to
    a value that no longer looks like a YouTube handle only by accident, so
    conversions are skipped explicitly instead.
    """
    if not text or "@" not in text:
        return text, set()

    found: set[str] = set()

    def replace(match: re.Match[str]) -> str:
        handle = match.group(1)
        if match.group(0).startswith(MENTION_PREFIX):
            return match.group(0)
        found.add(f"@{handle}")
        return mention_pseudonym(handle)

    return MENTION.sub(replace, text), found


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


def process_snapshots(*, dry_run: bool) -> tuple[int, int, int, int, int]:
    """Rewrite the committed snapshots in place.

    Returns ``(files, messages, distinct_handles, mentions, distinct_mentioned)``.
    """
    files = 0
    messages = 0
    handles: set[str] = set()
    mentions = 0
    mentioned: set[str] = set()

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

            # Mentions inside the message body are handled separately: a
            # message can carry one even when its author is already converted.
            text = record.get("text")
            if isinstance(text, str):
                rewritten, found = convert_text(text)
                if found:
                    record["text"] = rewritten
                    mentioned |= found
                    mentions += 1
                    changed = True

            output.append(json.dumps(record, ensure_ascii=False))

        if changed:
            files += 1
            if not dry_run:
                path.write_text(
                    "\n".join(output) + "\n", encoding="utf-8", newline="\n"
                )

    return files, messages, len(handles), mentions, len(mentioned)


def process_database(*, dry_run: bool) -> tuple[int, int, int, int]:
    """Rewrite stored chat messages and the window documents built from them.

    Returns ``(updated, distinct_handles, mentions, distinct_mentioned)``.

    The window documents matter as much as the messages: a window's ``content``
    is the concatenated text of its messages, so a handle left in a message
    reappears there -- and that field is what the API serves to the detail page.
    The stored *analysis* is unaffected either way, because ``clean_text``
    removes mentions before tokenisation.
    """
    updated = 0
    handles: set[str] = set()
    mentions = 0
    mentioned: set[str] = set()

    with SessionLocal() as db:
        for row in db.scalars(select(ChatMessage)):
            if row.author and not row.author.startswith(ALREADY_DONE_MARKER):
                handles.add(row.author)
                if not dry_run:
                    row.author = convert(row.author)
                updated += 1

            rewritten, found = convert_text(row.text)
            if found:
                mentioned |= found
                mentions += 1
                if not dry_run:
                    row.text = rewritten

        for article in db.scalars(select(NewsArticle)):
            for field in ("title", "content", "summary"):
                value = getattr(article, field, None)
                if not isinstance(value, str):
                    continue
                rewritten, found = convert_text(value)
                if found:
                    mentioned |= found
                    mentions += 1
                    if not dry_run:
                        setattr(article, field, rewritten)

        if not dry_run:
            db.commit()

    return updated, len(handles), mentions, len(mentioned)


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
    files, messages, handles, mentions, mentioned = process_snapshots(
        dry_run=args.check
    )
    print(f"  files affected        {files}")
    print(f"  author fields         {messages:,}  ({handles:,} distinct handles)")
    print(f"  messages w/ mentions  {mentions:,}  ({mentioned:,} distinct handles)")

    print("\nDatabase:")
    try:
        updated, db_handles, db_mentions, db_mentioned = process_database(
            dry_run=args.check
        )
        print(f"  author fields         {updated:,}  ({db_handles:,} distinct handles)")
        print(
            f"  messages w/ mentions  {db_mentions:,}  "
            f"({db_mentioned:,} distinct handles)"
        )
    except Exception as exc:  # database may not exist yet
        print(f"  skipped: {exc}")

    print("\nExample mapping:")
    for handle in ["@I-nawa01", "@thairathnews", "@Padid-l9t"]:
        print(f"  author  {handle:20} -> {pseudonym(handle)}")
        print(f"  mention {handle:20} -> {mention_pseudonym(handle[1:])}")

    if args.check:
        print("\nRun without --check to apply.")
    else:
        print(
            "\nDone. Timestamps and every metric are unchanged. Text changes only\n"
            "where a mention was replaced, and clean_text strips mentions before\n"
            "tokenisation either way, so no analysis result moves."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
