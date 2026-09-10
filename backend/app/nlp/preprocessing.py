"""Thai text cleaning and normalisation.

Tuned against the collected corpus of 25,928 real Thai live-chat messages
rather than against invented examples. What that corpus actually contains:

* median message length 19 characters -- very short, which is why messages are
  windowed before analysis
* 15.6% contain a character repeated 4+ times (``11111``, ``สู้ๆๆๆๆ``)
* 14.8% contain no Thai characters at all (digit spam, bare emoji)
* 9.8% contain emoji, which carry real sentiment in chat and are kept
* 4.2% contain ``555`` -- Thai laughter, equivalent to "hahaha". Since ``555``
  and ``5555`` tokenise as different tokens, runs are canonicalised to one form
  instead of being deleted.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from pythainlp.util import normalize as thai_normalize

from app.nlp.stopwords import PROTECTED_WORDS, PUNCTUATION_TOKENS, get_stopwords

# Thai Unicode block.
THAI_CHARS = re.compile(r"[฀-๿]")

_HTML_TAG = re.compile(r"<[^>]+>")
_HTML_ENTITY = re.compile(r"&(?:[a-zA-Z]+|#\d+);")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_MENTION = re.compile(r"@[\w.\-]+")
_ZERO_WIDTH = re.compile(r"[​-‏‪-‮﻿­]")
_WHITESPACE = re.compile(r"\s+")

# Thai laughter: any run of three or more "5" characters.
_LAUGH = re.compile(r"5{3,}")
_LAUGH_TOKEN = "555"
# A placeholder that survives the repetition-collapsing pass below. Using a
# control character guarantees it cannot appear in real chat text.
_LAUGH_PLACEHOLDER = "\x00L\x00"

# Repetition collapsing, applied in this order.
_REPEAT_DIGIT = re.compile(r"(\d)\1{2,}")  # 11111 -> 1
# Collapse a letter repeated 3+ times down to a single one. Thai's legitimate
# double consonants (ธรรม, กรรม, วรรณ) are always exactly two, so they are never
# touched, while chat elongation (มากกกกก, สวยยยยย) collapses to the real word.
_REPEAT_WORD_CHAR = re.compile(r"([^\W\d_])\1{2,}", re.UNICODE)
_REPEAT_SYMBOL = re.compile(r"([^\w\s])\1+", re.UNICODE)  # !!!! -> !, emoji runs -> one

# Thai repetition mark (ๆ) repeated: สู้ๆๆๆๆ -> สู้ๆ
_REPEAT_MAI_YAMOK = re.compile(r"ๆ{2,}")


@dataclass(slots=True)
class PreprocessResult:
    """Everything the pipeline and the NLP Analysis page need from one document.

    Two token streams are returned deliberately:

    * ``tokens`` -- every token, negation intact. Used for sentiment, where
      dropping ``ไม่`` would invert the result.
    * ``filtered_tokens`` -- stopwords, punctuation and noise removed. Used for
      topic classification and keyword extraction, where filler only dilutes
      the signal.
    """

    raw_text: str
    cleaned_text: str
    tokens: list[str] = field(default_factory=list)
    filtered_tokens: list[str] = field(default_factory=list)
    sentences: list[str] = field(default_factory=list)

    @property
    def token_count(self) -> int:
        return len(self.tokens)

    @property
    def unique_token_count(self) -> int:
        return len(set(self.tokens))

    @property
    def stopword_removed_count(self) -> int:
        return len(self.tokens) - len(self.filtered_tokens)

    @property
    def filtered_text(self) -> str:
        """Space-joined filtered tokens, the input a TF-IDF model sees."""
        return " ".join(self.filtered_tokens)


def clean_text(text: str, *, keep_emoji: bool = True) -> str:
    """Normalise raw Thai text for analysis.

    Args:
        text: Raw input.
        keep_emoji: Keep emoji characters. Default True because emoji are a
            strong sentiment signal in chat; set False for topic-only features.

    Returns:
        Cleaned text. Never ``None``; an unusable input yields ``""``.
    """
    if not text:
        return ""

    # Compatibility-normalise first so full-width and decomposed forms collapse.
    result = unicodedata.normalize("NFKC", str(text))

    result = _HTML_TAG.sub(" ", result)
    result = _HTML_ENTITY.sub(" ", result)
    result = _URL.sub(" ", result)
    result = _MENTION.sub(" ", result)
    result = _ZERO_WIDTH.sub("", result)

    # Protect laughter before the digit-repetition pass would shrink it to "5".
    result = _LAUGH.sub(_LAUGH_PLACEHOLDER, result)

    result = _REPEAT_MAI_YAMOK.sub("ๆ", result)
    result = _REPEAT_DIGIT.sub(r"\1", result)
    result = _REPEAT_WORD_CHAR.sub(r"\1", result)
    result = _REPEAT_SYMBOL.sub(r"\1", result)

    # Restore laughter, spaced so the tokenizer emits it as its own token.
    result = result.replace(_LAUGH_PLACEHOLDER, f" {_LAUGH_TOKEN} ")

    if not keep_emoji:
        result = _strip_symbols(result)

    # Thai-specific normalisation: fixes tone/vowel ordering and stray marks.
    try:
        result = thai_normalize(result)
    except Exception:
        # Never let normalisation failure lose the document.
        pass

    return _WHITESPACE.sub(" ", result).strip()


def _strip_symbols(text: str) -> str:
    """Drop emoji and pictographs, keeping letters, digits and punctuation."""
    return "".join(
        char
        for char in text
        # So/Sk = symbol-other / symbol-modifier, which is where emoji live.
        if unicodedata.category(char) not in {"So", "Sk"}
    )


def thai_ratio(text: str) -> float:
    """Fraction of non-space characters that are Thai."""
    stripped = [char for char in text if not char.isspace()]
    if not stripped:
        return 0.0
    return len(THAI_CHARS.findall(text)) / len(stripped)


def is_noise(
    text: str,
    *,
    min_thai_ratio: float = 0.2,
    min_length: int = 2,
) -> bool:
    """True when a message carries no analysable Thai content.

    Catches the dominant junk in real chat: digit spam (``11111``), bare emoji
    runs (``🧡🧡🧡🧡``), and single stray characters. Deliberately conservative --
    a message with any real Thai content is kept, because false positives here
    silently shrink the corpus.
    """
    cleaned = clean_text(text or "")
    if len(cleaned) < min_length:
        return True

    # Nothing but digits, punctuation and spaces.
    if not re.search(r"[^\W\d_]", cleaned, re.UNICODE):
        return True

    # Latin-only chat ("ok", "goal") is legitimate but carries no Thai signal;
    # keep it only when it is long enough to be a real sentence.
    if not THAI_CHARS.search(cleaned):
        return len(cleaned) < 15

    return thai_ratio(cleaned) < min_thai_ratio and len(cleaned) < 15


def filter_tokens(
    tokens: list[str],
    *,
    protect_polarity: bool = True,
    min_length: int = 2,
    drop_numbers: bool = True,
) -> list[str]:
    """Remove stopwords, punctuation and uninformative tokens.

    Args:
        tokens: Tokens to filter.
        protect_polarity: Keep negation/intensifier words (see
            :mod:`app.nlp.stopwords`).
        min_length: Drop tokens shorter than this, except protected ones.
        drop_numbers: Drop pure numbers, which are rarely topical in chat.
    """
    stopwords = get_stopwords(protect_polarity=protect_polarity)
    kept: list[str] = []

    protected = PROTECTED_WORDS if protect_polarity else frozenset()

    for token in tokens:
        word = token.strip()
        if not word:
            continue
        # A protected polarity word survives every other rule, including the
        # length floor -- short negators like "มิ" must not be filtered out.
        if word in protected:
            kept.append(word)
            continue
        if word in PUNCTUATION_TOKENS:
            continue
        if word in stopwords:
            continue
        if drop_numbers and word.isdigit():
            continue
        # Single Thai characters are almost always tokenizer artefacts.
        if len(word) < min_length:
            continue
        if not re.search(r"[^\W_]", word, re.UNICODE):
            continue
        kept.append(word)

    return kept
