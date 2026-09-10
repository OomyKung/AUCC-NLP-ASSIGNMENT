"""Thai sentiment lexicon.

Used by the lexicon sentiment backend, which is the cold-start path: it gives
real, explainable inference before any model has been trained, so a fresh clone
of the project produces genuine sentiment rather than placeholder values.

Weights are on a -1..+1 scale by intensity: a strong word like ``สุดยอด``
outweighs a mild one like ``โอเค``. Vocabulary is drawn from the domains this
project actually sees -- news reporting and live-chat reactions to it -- so
sports cheering (``สู้ๆ``), chat slang (``ห่วย``, ``กาก``) and news outcome words
(``เสียชีวิต``, ``ฟื้นตัว``) are all represented.
"""

from __future__ import annotations

from app.nlp.matching import TermMatcher

# --------------------------------------------------------------------------
# Positive vocabulary
# --------------------------------------------------------------------------

POSITIVE_WORDS: dict[str, float] = {
    # Strong praise
    "สุดยอด": 1.0,
    "ยอดเยี่ยม": 1.0,
    "เยี่ยม": 0.9,
    "ดีเยี่ยม": 1.0,
    "ประทับใจ": 0.9,
    "ปลื้ม": 0.9,
    "ภูมิใจ": 0.9,
    "ชื่นชม": 0.85,
    "น่าชื่นชม": 0.85,
    "เลิศ": 0.9,
    "ดีมาก": 0.9,
    # General positive
    "ดี": 0.6,
    "ดีขึ้น": 0.7,
    "โอเค": 0.35,
    "เก่ง": 0.8,
    "สวย": 0.7,
    "น่ารัก": 0.7,
    "หล่อ": 0.6,
    "สนุก": 0.75,
    "ตลก": 0.5,
    "มัน": 0.4,
    "ชอบ": 0.7,
    "รัก": 0.85,
    "หลงรัก": 0.85,
    "ดีใจ": 0.8,
    "มีความสุข": 0.9,
    "สุข": 0.7,
    "ยินดี": 0.75,
    "ขอบคุณ": 0.6,
    "ซึ้ง": 0.6,
    "อบอุ่น": 0.6,
    "เมตตา": 0.7,
    "ใจดี": 0.7,
    # Success / achievement
    "ชนะ": 0.85,
    "ชัยชนะ": 0.9,
    "สำเร็จ": 0.85,
    "ประสบความสำเร็จ": 0.9,
    "แชมป์": 0.85,
    "เหรียญทอง": 0.9,
    "คว้าชัย": 0.85,
    "ทำได้": 0.7,
    "ผ่าน": 0.4,
    "เข้ารอบ": 0.7,
    "ก้าวหน้า": 0.7,
    "พัฒนา": 0.6,
    "เติบโต": 0.65,
    "ฟื้นตัว": 0.7,
    "ปรับตัวขึ้น": 0.65,
    "กำไร": 0.7,
    "เพิ่มขึ้น": 0.35,
    "ลดลง": -0.1,
    # Safety / relief
    "ปลอดภัย": 0.8,
    "รอด": 0.75,
    "รอดชีวิต": 0.85,
    "ช่วยเหลือ": 0.6,
    "ช่วยได้": 0.7,
    "หายดี": 0.8,
    "แข็งแรง": 0.7,
    "สุขภาพดี": 0.8,
    "บริจาค": 0.6,
    "เยียวยา": 0.5,
    "แก้ไข": 0.45,
    "คืน": 0.4,
    "โปร่งใส": 0.7,
    "เรียบร้อย": 0.55,
    "ราบรื่น": 0.6,
    # Encouragement (very common in live chat)
    "สู้": 0.6,
    "สู้ๆ": 0.75,
    "เชียร์": 0.6,
    "กำลังใจ": 0.7,
    "เป็นกำลังใจ": 0.75,
    "หวัง": 0.35,
    "โชคดี": 0.7,
    "เอาใจช่วย": 0.7,
    "555": 0.35,  # Thai laughter
}

# --------------------------------------------------------------------------
# Negative vocabulary
# --------------------------------------------------------------------------

NEGATIVE_WORDS: dict[str, float] = {
    # Strong criticism
    "แย่": -0.8,
    "แย่มาก": -0.95,
    "เลว": -0.9,
    "ห่วย": -0.9,
    "ห่วยแตก": -1.0,
    "กาก": -0.85,
    "แย่ที่สุด": -1.0,
    "โง่": -0.85,
    "งี่เง่า": -0.85,
    "น่าเบื่อ": -0.6,
    "เสียดาย": -0.5,
    "ไม่ไหว": -0.65,
    "ไม่โอเค": -0.7,
    "รับไม่ได้": -0.8,
    # Emotions
    "เสียใจ": -0.8,
    "ผิดหวัง": -0.8,
    "โกรธ": -0.8,
    "โมโห": -0.8,
    "เกลียด": -0.9,
    "กลัว": -0.6,
    "กังวล": -0.5,
    "เครียด": -0.6,
    "ท้อ": -0.65,
    "หมดหวัง": -0.85,
    "น่าสงสาร": -0.5,
    "สงสาร": -0.4,
    "ร้องไห้": -0.6,
    "ทุกข์": -0.75,
    # Harm / loss
    "เสียชีวิต": -0.9,
    "ตาย": -0.9,
    "บาดเจ็บ": -0.8,
    "เจ็บ": -0.6,
    "สาหัส": -0.85,
    "วิกฤต": -0.85,
    "เสียหาย": -0.75,
    "พัง": -0.7,
    "ล้มเหลว": -0.85,
    "แพ้": -0.75,
    "ขาดทุน": -0.8,
    "หนี้": -0.6,
    "ยากจน": -0.7,
    "ลำบาก": -0.65,
    "เดือดร้อน": -0.7,
    "อันตราย": -0.75,
    "รุนแรง": -0.7,
    "ป่วย": -0.65,
    "ระบาด": -0.7,
    "ติดเชื้อ": -0.7,
    # Problems / conflict
    "ปัญหา": -0.55,
    "ผิดพลาด": -0.7,
    "ล่าช้า": -0.55,
    "ขัดแย้ง": -0.65,
    "ประท้วง": -0.5,
    "ทะเลาะ": -0.7,
    "วุ่นวาย": -0.6,
    "โกง": -0.9,
    "ทุจริต": -0.9,
    "หลอกลวง": -0.85,
    "หลอก": -0.75,
    "ฉ้อโกง": -0.85,
    "อาชญากรรม": -0.7,
    "ปล้น": -0.8,
    "ลักทรัพย์": -0.75,
    "ยาเสพติด": -0.75,
    "อุบัติเหตุ": -0.7,
    "ไฟไหม้": -0.8,
    "น้ำท่วม": -0.75,
    "ภัยพิบัติ": -0.85,
    "แพง": -0.6,
    "ค่าครองชีพ": -0.4,
    "มลพิษ": -0.7,
    "ฝุ่น": -0.5,
    "ขยะ": -0.4,
    # Complaint verbs common in chat
    "ด่า": -0.7,
    "บ่น": -0.5,
    "ตำหนิ": -0.6,
    "โทษ": -0.5,
    "ไล่": -0.6,
    "ออกไป": -0.5,
    "ลาออก": -0.4,
}

# --------------------------------------------------------------------------
# Emoji
# --------------------------------------------------------------------------

# Emoji are a strong sentiment signal in chat (9.8% of the collected corpus),
# which is why clean_text() preserves them.
EMOJI_SENTIMENT: dict[str, float] = {
    "😀": 0.7, "😃": 0.7, "😄": 0.8, "😁": 0.7, "😊": 0.8, "🙂": 0.5,
    "😍": 0.9, "🥰": 0.9, "😘": 0.8, "❤": 0.85, "❤️": 0.85, "🧡": 0.8,
    "💚": 0.8, "💙": 0.8, "💜": 0.8, "🩷": 0.8, "💖": 0.85, "💕": 0.8,
    "👍": 0.7, "👏": 0.75, "🙏": 0.5, "🎉": 0.8, "🥳": 0.85, "✨": 0.6,
    "💪": 0.7, "🔥": 0.5, "⚽": 0.2, "🏆": 0.85, "🥇": 0.85, "😂": 0.5,
    "🤣": 0.5, "😆": 0.6, "😅": 0.2, "🙌": 0.75, "💯": 0.7, "⭐": 0.6,
    "😢": -0.7, "😭": -0.8, "😞": -0.7, "😔": -0.6, "😟": -0.6,
    "😡": -0.9, "🤬": -1.0, "😠": -0.85, "👎": -0.8, "💔": -0.85,
    "😱": -0.6, "😰": -0.6, "🤮": -0.9, "🙄": -0.5, "😤": -0.6,
    "☹": -0.6, "☹️": -0.6, "😩": -0.65, "😫": -0.65, "🤦": -0.6,
}

# --------------------------------------------------------------------------
# Modifiers
# --------------------------------------------------------------------------

# Words that invert the polarity of what follows. Kept in sync with
# app.nlp.stopwords.NEGATION_WORDS, which protects them from being filtered out.
NEGATORS: frozenset[str] = frozenset(
    {
        "ไม่",
        "ไม่ได้",
        "ไม่ใช่",
        "ไม่มี",
        "ไม่เคย",
        "มิ",
        "มิได้",
        "ไร้",
        "อย่า",
        "ห้าม",
        "เลิก",
        "หยุด",
    }
)

# Multipliers applied to the next sentiment word.
INTENSIFIERS: dict[str, float] = {
    "มาก": 1.5,
    "มากๆ": 1.8,
    "สุด": 1.7,
    "ที่สุด": 1.8,
    "จริง": 1.3,
    "จริงๆ": 1.4,
    "เหลือเกิน": 1.6,
    "เกิน": 1.4,
    "สุดๆ": 1.8,
}

DIMINISHERS: dict[str, float] = {
    "หน่อย": 0.6,
    "นิดหน่อย": 0.5,
    "นิด": 0.5,
    "เล็กน้อย": 0.5,
    "ค่อนข้าง": 0.8,
    "พอ": 0.7,
}

# How many tokens after a negator remain within its scope.
NEGATION_WINDOW = 3


def word_score(token: str) -> float:
    """Base polarity of one token, 0.0 when it is not in the lexicon."""
    if token in POSITIVE_WORDS:
        return POSITIVE_WORDS[token]
    if token in NEGATIVE_WORDS:
        return NEGATIVE_WORDS[token]
    return EMOJI_SENTIMENT.get(token, 0.0)


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------
#
# Matching is delegated to the shared TermMatcher (see app/nlp/matching.py),
# which handles multi-token phrases and Thai compounds. Emoji need no special
# case: they are single tokens, so exact matching finds them, and they are
# shorter than the substring threshold so they cannot fire inside a word.

_ALL_TERMS: dict[str, float] = {**POSITIVE_WORDS, **NEGATIVE_WORDS, **EMOJI_SENTIMENT}

_MATCHER = TermMatcher(_ALL_TERMS)


def match_terms(tokens: list[str]) -> list[tuple[int, int, str, float]]:
    """Find sentiment terms in ``tokens``.

    Returns:
        Non-overlapping ``(start, span, term, weight)`` tuples in token order.
    """
    return [(m.start, m.span, m.term, m.weight) for m in _MATCHER.find(tokens)]


def lexicon_size() -> dict[str, int]:
    """Sizes of each vocabulary, reported by the API for transparency."""
    return {
        "positive": len(POSITIVE_WORDS),
        "negative": len(NEGATIVE_WORDS),
        "emoji": len(EMOJI_SENTIMENT),
        "negators": len(NEGATORS),
        "intensifiers": len(INTENSIFIERS),
        "diminishers": len(DIMINISHERS),
    }
