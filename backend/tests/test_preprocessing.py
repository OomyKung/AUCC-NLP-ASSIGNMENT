"""Tests for Thai text cleaning, noise detection and token filtering.

The cases here are drawn from patterns actually present in the collected chat
corpus, not invented ones.
"""

from __future__ import annotations

import pytest

from app.nlp.preprocessing import (
    clean_text,
    filter_tokens,
    is_noise,
    thai_ratio,
)
from app.nlp.stopwords import (
    NEGATION_WORDS,
    PROTECTED_WORDS,
    get_stopwords,
    is_stopword,
)

# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("<b>ไม่ดี</b>", "ไม่ดี"),  # HTML tags
        ("ดี &amp; เยี่ยม", "ดี เยี่ยม"),  # HTML entities
        ("ดูที่ https://youtu.be/abc นะ", "ดูที่ นะ"),  # URLs
        ("ดูที่ www.thairath.co.th ครับ", "ดูที่ ครับ"),
        ("@somebody ว่าไง", "ว่าไง"),  # mentions
        ("ไทย​ชนะ", "ไทยชนะ"),  # zero-width characters
        ("ไทย   ชนะ\n\nแล้ว", "ไทย ชนะ แล้ว"),  # whitespace collapse
    ],
)
def test_clean_text_strips_noise(raw, expected):
    assert clean_text(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Thai laughter: runs canonicalise to one token, never to a bare "5".
        ("5555555", "555"),
        ("ตลก 555555 มาก", "ตลก 555 มาก"),
        ("55555ตลก", "555 ตลก"),
    ],
)
def test_laughter_is_canonicalised(raw, expected):
    assert clean_text(raw) == expected


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Chat elongation collapses back to the real dictionary word.
        ("ดีมากกกกก", "ดีมาก"),
        ("สวยยยยย", "สวย"),
        ("เก่งงงง", "เก่ง"),
        ("สู้ๆๆๆๆๆ", "สู้ๆ"),
        # Thai's legitimate double consonants are always exactly two, so a
        # 3+ collapse must never damage them.
        ("ธรรมชาติ", "ธรรมชาติ"),
        ("กรรมการ", "กรรมการ"),
        ("วรรณกรรม", "วรรณกรรม"),
        # Digit spam and symbol runs.
        ("11111111", "1"),
        ("ดี!!!!!", "ดี!"),
    ],
)
def test_repetition_collapsing(raw, expected):
    assert clean_text(raw) == expected


def test_emoji_are_kept_by_default_but_can_be_stripped():
    # Emoji carry real sentiment in chat, so they survive by default.
    assert "🧡" in clean_text("รักเลย 🧡")
    assert "🧡" not in clean_text("รักเลย 🧡", keep_emoji=False)
    assert "รักเลย" in clean_text("รักเลย 🧡", keep_emoji=False)


def test_emoji_runs_collapse_to_one():
    assert clean_text("🧡🧡🧡🧡") == "🧡"


@pytest.mark.parametrize("raw", ["", "   ", "\n\t", None])
def test_clean_text_handles_empty_input(raw):
    assert clean_text(raw) == ""


def test_clean_text_normalises_fullwidth_forms():
    # NFKC folds full-width Latin/digits onto their ASCII equivalents.
    assert clean_text("ＴＨＡＩ １２３") == "THAI 123"


# --------------------------------------------------------------------------
# Noise detection
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw",
    [
        "11111111",  # digit spam, 15.6% of the corpus has repeats like this
        "🧡🧡🧡🧡",  # bare emoji
        "ค",  # single stray character
        "",
        "   ",
        "12",
        "!!!!",
        "ok",  # short Latin-only
    ],
)
def test_is_noise_flags_junk(raw):
    assert is_noise(raw) is True


@pytest.mark.parametrize(
    "raw",
    [
        "ไม่ดี",
        "สู้ๆ นะครับ",
        "รถยนต์ชนกันกลางสี่แยก",
        "ตลก 555 มาก",
        "รักเลย 🧡",
        "ทีมชาติไทยเล่นดีมาก",
    ],
)
def test_is_noise_keeps_real_thai_content(raw):
    assert is_noise(raw) is False


def test_thai_ratio():
    assert thai_ratio("ไทย") == 1.0
    assert thai_ratio("abc") == 0.0
    assert thai_ratio("") == 0.0
    assert 0.4 < thai_ratio("ไทย123") < 0.6


# --------------------------------------------------------------------------
# Stopwords and negation protection
# --------------------------------------------------------------------------


def test_negation_is_never_a_stopword():
    """The correctness property: dropping ไม่ inverts sentiment.

    PyThaiNLP's list contains ไม่, the most frequent token in the corpus, so
    this guard is what stops "ไม่ดี" (not good) becoming "ดี" (good).
    """
    stopwords = get_stopwords()
    assert not (NEGATION_WORDS & stopwords)
    assert "ไม่" not in stopwords
    assert not is_stopword("ไม่")


def test_chat_filler_is_a_stopword():
    stopwords = get_stopwords()
    assert "ครับ" in stopwords
    assert "ค่ะ" in stopwords
    # Laughter is not topical, so it is filtered for topic/keyword features
    # even though clean_text preserves it for sentiment.
    assert "555" in stopwords


def test_polarity_protection_can_be_disabled_explicitly():
    unprotected = get_stopwords(protect_polarity=False)
    # ไม่ comes back only when the caller explicitly opts out.
    assert "ไม่" in unprotected


# --------------------------------------------------------------------------
# Token filtering
# --------------------------------------------------------------------------


def test_filter_tokens_removes_stopwords_but_keeps_negation():
    tokens = ["ทีม", "ไทย", "เล่น", "ไม่", "ดี", "ครับ", "นะ"]

    filtered = filter_tokens(tokens)

    assert "ไม่" in filtered, "negation must survive stopword removal"
    assert "ครับ" not in filtered
    assert "นะ" not in filtered
    assert filtered == ["ทีม", "ไทย", "เล่น", "ไม่", "ดี"]


def test_filter_tokens_keeps_short_protected_words():
    # "มิ" is a 2-character negator; the length floor must not remove it.
    assert "มิ" in filter_tokens(["มิ", "ก"])


def test_filter_tokens_drops_punctuation_numbers_and_single_chars():
    tokens = ["รถยนต์", "!", "5", "123", "ก", "อุบัติเหตุ", "-"]

    assert filter_tokens(tokens) == ["รถยนต์", "อุบัติเหตุ"]


def test_filter_tokens_can_keep_numbers():
    assert "123" in filter_tokens(["123", "รถ"], drop_numbers=False)


def test_filter_tokens_handles_empty_and_whitespace_tokens():
    assert filter_tokens(["", "  ", "\n"]) == []
    assert filter_tokens([]) == []


def test_every_protected_word_survives_filtering():
    # A regression guard: the whole protected set must pass through intact.
    filtered = filter_tokens(sorted(PROTECTED_WORDS))
    assert set(filtered) == PROTECTED_WORDS
