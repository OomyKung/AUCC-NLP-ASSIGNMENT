"""Tests for Thai tokenisation.

Thai has no spaces between words, so tokenisation is a modelling step and its
output shape is worth pinning down.
"""

from __future__ import annotations

import pytest

from app.nlp.base import Tokenizer
from app.nlp.tokenizer import SUPPORTED_ENGINES, ThaiTokenizer, get_tokenizer


@pytest.fixture(scope="module")
def tokenizer() -> ThaiTokenizer:
    return get_tokenizer("newmm")


def test_satisfies_the_tokenizer_protocol(tokenizer):
    # The seam that makes the NLP layer swappable.
    assert isinstance(tokenizer, Tokenizer)


def test_tokenises_thai_without_spaces(tokenizer):
    tokens = tokenizer.tokenize("รถยนต์ชนกันกลางสี่แยก")

    assert tokens == ["รถยนต์", "ชน", "กัน", "กลาง", "สี่แยก"]


def test_tokenises_mixed_thai_and_digits(tokenizer):
    tokens = tokenizer.tokenize("มีผู้ได้รับบาดเจ็บ 5 ราย")

    assert "ผู้" in tokens
    assert "5" in tokens
    assert "ราย" in tokens


def test_whitespace_is_never_returned_as_a_token(tokenizer):
    tokens = tokenizer.tokenize("ไทย   ชนะ\n\nแล้ว")

    assert all(token.strip() for token in tokens)


@pytest.mark.parametrize("text", ["", "   ", "\n"])
def test_empty_input_yields_no_tokens(tokenizer, text):
    assert tokenizer.tokenize(text) == []
    assert tokenizer.sentences(text) == []


def test_sentence_segmentation(tokenizer):
    sentences = tokenizer.sentences(
        "ฝนตกหนักทำให้เกิดน้ำท่วมในหลายพื้นที่ "
        "ประชาชนได้รับความเดือดร้อนจำนวนมาก "
        "เจ้าหน้าที่เร่งเข้าช่วยเหลือผู้ประสบภัย"
    )

    assert len(sentences) == 3
    assert sentences[0].startswith("ฝนตกหนัก")
    # Segments must be stripped and non-empty; the summarizer relies on this.
    assert all(s == s.strip() and s for s in sentences)


def test_unsupported_engine_is_rejected_with_a_helpful_message():
    with pytest.raises(ValueError, match="Unsupported tokenizer engine"):
        ThaiTokenizer("neural-magic")


@pytest.mark.parametrize("engine", SUPPORTED_ENGINES)
def test_every_supported_engine_tokenises(engine):
    assert get_tokenizer(engine).tokenize("ทีมชาติไทยชนะ")


def test_get_tokenizer_is_cached():
    # newmm builds a dictionary trie on first use, so instances must be shared.
    assert get_tokenizer("newmm") is get_tokenizer("newmm")


# --------------------------------------------------------------------------
# The full preprocess() chain
# --------------------------------------------------------------------------


def test_preprocess_returns_both_token_streams(tokenizer):
    result = tokenizer.preprocess("ทีมไทยเล่นไม่ดีเลยครับ 5555555")

    # Full stream keeps negation for sentiment.
    assert "ไม่" in result.tokens
    # Filtered stream drops filler for topic/keyword features.
    assert "ครับ" in result.tokens
    assert "ครับ" not in result.filtered_tokens
    assert "ไม่" in result.filtered_tokens
    assert len(result.filtered_tokens) < len(result.tokens)


def test_preprocess_counts_are_consistent(tokenizer):
    result = tokenizer.preprocess("รถยนต์ชนกัน รถยนต์เสียหาย")

    assert result.token_count == len(result.tokens)
    assert result.unique_token_count == len(set(result.tokens))
    # "รถยนต์" appears twice, so unique must be lower than total.
    assert result.unique_token_count < result.token_count
    assert result.stopword_removed_count == len(result.tokens) - len(
        result.filtered_tokens
    )


def test_preprocess_exposes_cleaned_text_and_raw(tokenizer):
    raw = "<b>ดีมากกกกก</b> https://example.com"
    result = tokenizer.preprocess(raw)

    assert result.raw_text == raw
    assert result.cleaned_text == "ดีมาก"
    assert result.filtered_text == " ".join(result.filtered_tokens)


def test_preprocess_can_skip_sentence_segmentation(tokenizer):
    # Sentence segmentation is the slow step; per-message runs skip it.
    assert tokenizer.preprocess("ไทยชนะแล้ว", with_sentences=False).sentences == []
    assert tokenizer.preprocess("ไทยชนะแล้ว", with_sentences=True).sentences


def test_preprocess_survives_junk_input(tokenizer):
    for junk in ["", "   ", "11111", "🧡🧡🧡", None]:
        result = tokenizer.preprocess(junk or "")
        assert isinstance(result.tokens, list)
        assert isinstance(result.filtered_tokens, list)
