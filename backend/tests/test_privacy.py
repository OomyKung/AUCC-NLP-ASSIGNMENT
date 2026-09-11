"""Guards on the published data: no identifiable YouTube handles.

These tests exist because of a real leak. `anonymise.py` originally rewrote only
the ``author`` field, and 98 raw handles stayed in the committed snapshots inside
message text, where viewers reply to each other by handle. Nothing checked, so
nothing caught it.

The committed snapshots are the published artefact, so the check runs against
the actual files rather than a fixture.
"""

from __future__ import annotations

import json
import re

import pytest

from anonymise import (
    ALREADY_DONE_MARKER,
    MENTION,
    MENTION_PREFIX,
    convert,
    convert_text,
    mention_pseudonym,
    pseudonym,
)
from app.config import settings

SNAPSHOTS = sorted(settings.chat_snapshot_dir.glob("*.jsonl"))

needs_snapshots = pytest.mark.skipif(
    not SNAPSHOTS, reason="no chat snapshots present"
)


def _messages():
    """Every message record in every committed snapshot."""
    for path in SNAPSHOTS:
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                if "_stream" in record:  # header
                    continue
                yield path.name, record


# --------------------------------------------------------------------------
# The published files
# --------------------------------------------------------------------------


@needs_snapshots
def test_no_raw_handles_in_author_fields():
    offenders = [
        (name, record["author"])
        for name, record in _messages()
        if record.get("author")
        and not str(record["author"]).startswith(ALREADY_DONE_MARKER)
    ]
    assert not offenders, f"raw author handles in snapshots: {offenders[:5]}"


@needs_snapshots
def test_no_raw_handles_in_message_text():
    """The regression this file was written for."""
    offenders = []
    for name, record in _messages():
        text = record.get("text") or ""
        for match in MENTION.finditer(text):
            if not match.group(0).startswith(MENTION_PREFIX):
                offenders.append((name, match.group(0)))
    assert not offenders, f"raw @mentions in message text: {offenders[:5]}"


@needs_snapshots
def test_no_emails_or_phone_numbers_in_message_text():
    """Other shapes of identifier that chat can carry.

    A plain "9 or more digits" rule does not work on Thai chat: "5555555555" is
    laughter ("5" is pronounced *ha*), and long runs of one digit are also used
    for emphasis. A phone number has *varied* digits, so the run must contain at
    least four distinct ones to count -- that still catches 0812345678 while
    ignoring 55555555 and 00000000.
    """
    email = re.compile(r"[\w.+-]+@[\w-]+\.[A-Za-z]{2,}")
    digits = re.compile(r"\d{9,}")

    offenders = []
    for name, record in _messages():
        text = record.get("text") or ""
        if email.search(text):
            offenders.append((name, "email", text[:60]))
        for run in digits.findall(text):
            if len(set(run)) >= 4:
                offenders.append((name, "phone-like", run))
    assert not offenders, f"possible identifiers in message text: {offenders[:5]}"


# --------------------------------------------------------------------------
# The transformation itself
# --------------------------------------------------------------------------


def test_author_and_mention_share_one_identity():
    """A mention of a poster must resolve to that poster.

    Without this the reply structure of a conversation is lost: the same person
    would appear under two unrelated pseudonyms depending on whether they were
    speaking or being spoken to.
    """
    author = pseudonym("@somebody")
    mention = mention_pseudonym("somebody")

    assert author.removeprefix(ALREADY_DONE_MARKER) == mention.removeprefix(
        MENTION_PREFIX
    )


def test_mention_pseudonym_is_ascii():
    """It must stay ASCII so clean_text's mention regex still removes it.

    A Thai pseudonym breaks: ``\\w`` stops at Thai combining marks, so the
    stripper would leave fragments behind and feed them to keyword extraction.
    """
    value = mention_pseudonym("somebody")

    assert value.isascii()
    assert MENTION.fullmatch(value)


def test_mention_substitution_leaves_no_trace_after_cleaning():
    """The substitution provably cannot change an analysis result."""
    from app.nlp.preprocessing import clean_text, filter_tokens
    from app.nlp.tokenizer import get_tokenizer

    tokenizer = get_tokenizer()
    original = "@not_a_real_handle ฟรีครับ ไม่ต้องจ่าย"
    rewritten, found = convert_text(original)

    assert found == {"@not_a_real_handle"}
    assert clean_text(original) == clean_text(rewritten)
    assert filter_tokens(tokenizer.tokenize(clean_text(original))) == filter_tokens(
        tokenizer.tokenize(clean_text(rewritten))
    )


def test_conversion_is_idempotent():
    once, _ = convert_text("@not_a_real_handle สวัสดี")
    twice, found = convert_text(once)

    assert twice == once
    assert not found
    assert convert(convert("@someone")) == convert("@someone")


def test_pseudonyms_are_stable_and_distinct():
    assert pseudonym("@alice") == pseudonym("@alice")
    assert pseudonym("@alice") != pseudonym("@bob")
    assert mention_pseudonym("alice") != mention_pseudonym("bob")


def test_short_and_bare_at_signs_are_left_alone():
    """An "@" that is not a handle must not be mangled."""
    for text in ["ราคา @ 50 บาท", "@ab", "อีเมล @", "ส่ง@"]:
        rewritten, found = convert_text(text)
        assert rewritten == text, text
        assert not found


def test_multiple_mentions_in_one_message_all_convert():
    text = "@alice_01 กับ @bob-02 ทั้งคู่เลย"
    rewritten, found = convert_text(text)

    assert found == {"@alice_01", "@bob-02"}
    assert "@alice_01" not in rewritten
    assert "@bob-02" not in rewritten
    assert rewritten.count(MENTION_PREFIX) == 2


def test_text_without_an_at_sign_is_returned_unchanged():
    text = "สวัสดีครับ ไม่มีการกล่าวถึงใคร"
    rewritten, found = convert_text(text)

    assert rewritten is text
    assert not found


def test_none_and_empty_text_are_handled():
    assert convert_text(None) == (None, set())
    assert convert_text("") == ("", set())
