"""Thai stopword sets.

Built from PyThaiNLP's list, extended with live-chat filler, and -- importantly
-- with negation words removed.

**Why negation is protected:** PyThaiNLP's stopword list contains ``ไม่``
("not"), which is the single most frequent token in the collected chat corpus.
Dropping it turns ``ไม่ดี`` ("not good") into ``ดี`` ("good") and inverts the
sentiment of the document. Stopword removal is useful for topic classification
and keyword extraction, but it must never destroy polarity, so every negator is
whitelisted here and kept.
"""

from __future__ import annotations

from functools import lru_cache

from pythainlp.corpus import thai_stopwords

# Words that flip or scope polarity. These are never removed.
NEGATION_WORDS: frozenset[str] = frozenset(
    {
        "ไม่",
        "ไม่ได้",
        "ไม่ใช่",
        "ไม่มี",
        "ไม่เคย",
        "มิ",
        "มิได้",
        "ไร้",
        "ขาด",
        "อย่า",
        "ห้าม",
        "ปฏิเสธ",
        "หยุด",
        "เลิก",
        "ผิด",
    }
)

# Intensifiers and degree words. Kept for the same reason as negation: they
# modulate sentiment strength ("ดีมาก" vs "ดี").
INTENSIFIER_WORDS: frozenset[str] = frozenset(
    {
        "มาก",
        "มากๆ",
        "สุด",
        "ที่สุด",
        "เกิน",
        "เหลือเกิน",
        "นิดหน่อย",
        "หน่อย",
    }
)

# Protected across the board: removing these changes meaning, not just noise.
PROTECTED_WORDS: frozenset[str] = NEGATION_WORDS | INTENSIFIER_WORDS

# Live-chat filler that PyThaiNLP's news-oriented list does not cover.
# Politeness particles, greetings and interjections carry no topical signal.
CHAT_STOPWORDS: frozenset[str] = frozenset(
    {
        # Politeness particles
        "ครับ",
        "ค่ะ",
        "คะ",
        "ค่า",
        "จ้า",
        "จ้ะ",
        "จ๊ะ",
        "ฮะ",
        "ฮ่ะ",
        "นะคะ",
        "นะครับ",
        "ครับผม",
        # Interjections / filler
        "อ่อ",
        "อ๋อ",
        "เอ่อ",
        "อืม",
        "อือ",
        "เออ",
        "โอ้",
        "โอ๊ะ",
        "เฮ้ย",
        "เห้ย",
        "ว้าย",
        "อุ๊ย",
        "เอ๊ะ",
        "หรอ",
        "เหรอ",
        "หรือเปล่า",
        # Chat shorthand with no topical content
        "555",
        "อ่ะ",
        "ป่ะ",
        "ปะ",
        "ละ",
        "ล่ะ",
        "สิ",
        "ดิ",
        "เนอะ",
        "นะ",
        "น่ะ",
        "งะ",
        "แหละ",
        "แล้วก็",
        "ๆ",
        # Latin filler seen in Thai chat
        "555555",
        "lol",
        "haha",
        "ok",
        "okay",
        "yes",
        "no",
    }
)

# Punctuation and symbols the tokenizer emits as standalone tokens.
# Filler specific to broadcast speech, found by measurement rather than guessed:
# these are the terms that surfaced as "keywords" in a large share of the stories
# in one 249-minute Thai news programme, and a term that appears in half the
# stories cannot be distinguishing one from another.
#
#   ผู้ชม   48% of stories -- "คุณผู้ชม", the presenter addressing the audience
#   อ่า     12%            -- hesitation noise the ASR transcribes literally
#   เงี้ย    9%            -- colloquial "like this"
#   ท่าน     9%            -- bare honorific, no referent of its own
#   ที่นี่    6%            -- "here"
#   บอ       7%            -- an ASR fragment, not a word
#
# Deliberately NOT included: ข่าว (news), จันทร์/อาทิตย์ (day names), เช้า
# (morning). They recur because of what the programme is, but they can still
# carry meaning in a specific story, and over-trimming a keyword list is worse
# than leaving a weak term in.
#
# This set is applied ONLY to keyword extraction on broadcast transcripts. It is
# kept out of the general stopword list on purpose: the trained classifiers'
# feature space is built from that list, so adding words to it would silently
# change every model metric in the project.
BROADCAST_FILLER: frozenset[str] = frozenset(
    {
        "ผู้ชม",
        "คุณผู้ชม",
        "อ่า",
        "เงี้ย",
        "ท่าน",
        "ที่นี่",
        "บอ",
        "ครับผม",
        "ค่ะ",
        "นะคะ",
        "อือ",
        "เอ่อ",
        "แบบนี้",
        "อย่างเงี้ย",
    }
)


PUNCTUATION_TOKENS: frozenset[str] = frozenset(
    set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~") | {"…", "ฯ", "“", "”", "‘", "’", "—", "–"}
)


@lru_cache(maxsize=4)
def get_stopwords(
    *,
    include_chat: bool = True,
    protect_polarity: bool = True,
) -> frozenset[str]:
    """Build the stopword set.

    Args:
        include_chat: Add live-chat filler on top of PyThaiNLP's list.
        protect_polarity: Keep negation and intensifier words out of the set.
            Leave this on unless you are certain polarity does not matter --
            switching it off will invert sentiment on negated text.
    """
    words = set(thai_stopwords())
    if include_chat:
        words |= CHAT_STOPWORDS
    if protect_polarity:
        words -= PROTECTED_WORDS
    return frozenset(words)


def is_stopword(token: str, *, protect_polarity: bool = True) -> bool:
    """True when ``token`` should be dropped before feature extraction."""
    return token in get_stopwords(protect_polarity=protect_polarity)
