"""Named entity recognition for Thai text.

The default backend is rule- and gazetteer-based: Thai marks people, places and
organisations with highly regular lead-in words (``นาย``, ``จังหวัด``,
``กระทรวง``, ``บริษัท``), which makes precise extraction possible with no model
download and no training data.

``NLP_NER_BACKEND=pythainlp`` switches to PyThaiNLP's CRF tagger, which is more
general but needs a corpus download on first use.
"""

from __future__ import annotations

import re

from app.nlp.base import Entity

# --------------------------------------------------------------------------
# Gazetteers
# --------------------------------------------------------------------------

# All 77 Thai provinces.
PROVINCES: frozenset[str] = frozenset(
    """
    กระบี่ กาญจนบุรี กาฬสินธุ์ กำแพงเพชร ขอนแก่น จันทบุรี ฉะเชิงเทรา ชลบุรี ชัยนาท ชัยภูมิ
    ชุมพร เชียงราย เชียงใหม่ ตรัง ตราด ตาก นครนายก นครปฐม นครพนม นครราชสีมา นครศรีธรรมราช
    นครสวรรค์ นนทบุรี นราธิวาส น่าน บึงกาฬ บุรีรัมย์ ปทุมธานี ประจวบคีรีขันธ์ ปราจีนบุรี ปัตตานี
    พระนครศรีอยุธยา พะเยา พังงา พัทลุง พิจิตร พิษณุโลก เพชรบุรี เพชรบูรณ์ แพร่ ภูเก็ต มหาสารคาม
    มุกดาหาร แม่ฮ่องสอน ยโสธร ยะลา ร้อยเอ็ด ระนอง ระยอง ราชบุรี ลพบุรี ลำปาง ลำพูน เลย ศรีสะเกษ
    สกลนคร สงขลา สตูล สมุทรปราการ สมุทรสงคราม สมุทรสาคร สระแก้ว สระบุรี สิงห์บุรี สุโขทัย
    สุพรรณบุรี สุราษฎร์ธานี สุรินทร์ หนองคาย หนองบัวลำภู อ่างทอง อำนาจเจริญ อุดรธานี อุตรดิตถ์
    อุทัยธานี อุบลราชธานี กรุงเทพมหานคร
    """.split()
)

# Titles that introduce a person's name.
PERSON_TITLES: tuple[str, ...] = (
    "นายกรัฐมนตรี", "รัฐมนตรี", "พล.ต.อ.", "พล.ต.ท.", "พล.ต.ต.", "พล.อ.", "พล.ท.", "พล.ร.อ.",
    "พ.ต.อ.", "พ.ต.ท.", "พ.ต.ต.", "ร.ต.อ.", "ศ.ดร.", "รศ.ดร.", "ผศ.ดร.", "นพ.", "พญ.", "ดร.",
    "นางสาว", "ด.ช.", "ด.ญ.", "นาย", "นาง", "คุณ",
)

# Words that begin an organisation name.
ORG_PREFIXES: tuple[str, ...] = (
    "กระทรวง", "กรม", "สำนักงาน", "การไฟฟ้า", "การประปา", "มหาวิทยาลัย", "โรงพยาบาล",
    "โรงเรียน", "บริษัท", "ธนาคาร", "สถาบัน", "ศูนย์", "สมาคม", "มูลนิธิ", "องค์การ",
    "เทศบาล", "สภา", "คณะกรรมการ", "สถานี", "การท่า", "สหกรณ์",
)

# Common words that follow a prefix but are not part of the entity name.
# Without this, "จังหวัดขอนแก่น ทำให้..." yields "จังหวัดขอนแก่นทำให้".
_NAME_STOP_WORDS: frozenset[str] = frozenset(
    {
        "ทำให้", "ทันที", "และ", "หรือ", "แต่", "ที่", "ซึ่ง", "โดย", "จาก", "ใน",
        "ระบุ", "เผย", "กล่าว", "แจ้ง", "พร้อม", "หลัง", "เมื่อ", "ว่า", "เป็น",
        "มี", "ได้", "ให้", "ไป", "มา", "แล้ว", "ยัง", "จะ", "ก็", "ๆ",
        # Verbs that ran into names on broadcast transcripts, producing
        # "นายทรงพลขับ" (a name welded to "drive") and similar.
        "ขับ", "เข้า", "ออก", "ถูก", "พา", "นำ", "ขอ", "ทำ", "เดิน", "วิ่ง",
        "ขึ้น", "ลง", "อยู่", "ต้อง", "เคย", "กำลัง", "เลย", "ด้วย", "กับ",
        "เจอ", "พบ", "รับ", "ส่ง", "บอก", "ถาม", "ตอบ", "คิด", "รู้",
    }
)

# "คุณผู้ชม" is a presenter addressing the audience, not a person in the news.
# It produced 39 spurious PERSON entities in one programme -- more than any
# real name -- because "คุณ" is a personal title and the extractor took
# whatever followed it.
_NOT_A_PERSON: frozenset[str] = frozenset(
    {"ผู้ชม", "ผู้ฟัง", "ผู้อ่าน", "ท่าน", "ทุกท่าน", "ผู้ชมครับ", "ผู้ชมค่ะ"}
)

# A Thai given name is at least two characters; a single character after a
# title is a tokenisation artefact ("นางสาวน").
MIN_NAME_CHARS = 2

# Province names that are also ordinary Thai words, so they need a prefix
# before being read as places.
_AMBIGUOUS_PROVINCES: frozenset[str] = frozenset(
    {"เลย", "ตาก", "น่าน", "แพร่", "ตรัง", "ระนอง"}
)

LOCATION_PREFIXES: tuple[str, ...] = (
    "จังหวัด", "อำเภอ", "ตำบล", "เขต", "แขวง", "หมู่บ้าน", "ถนน", "ซอย", "แม่น้ำ", "ทะเล",
)

# --------------------------------------------------------------------------
# Patterns
# --------------------------------------------------------------------------

THAI_MONTHS = (
    "มกราคม|กุมภาพันธ์|มีนาคม|เมษายน|พฤษภาคม|มิถุนายน|กรกฎาคม|สิงหาคม|กันยายน|ตุลาคม|"
    "พฤศจิกายน|ธันวาคม|ม\\.ค\\.|ก\\.พ\\.|มี\\.ค\\.|เม\\.ย\\.|พ\\.ค\\.|มิ\\.ย\\.|ก\\.ค\\.|"
    "ส\\.ค\\.|ก\\.ย\\.|ต\\.ค\\.|พ\\.ย\\.|ธ\\.ค\\."
)

_DATE = re.compile(rf"\d{{1,2}}\s*(?:{THAI_MONTHS})\s*(?:\d{{2,4}})?")
_TIME = re.compile(r"\d{1,2}[.:]\d{2}\s*(?:น\.|นาฬิกา)?")
_MONEY = re.compile(r"\d[\d,.]*\s*(?:ล้าน|แสน|หมื่น|พัน|ร้อย)?\s*(?:บาท|ดอลลาร์|ยูโร|เยน)")
_PERCENT = re.compile(r"(?:ร้อยละ\s*\d[\d,.]*|\d[\d,.]*\s*(?:%|เปอร์เซ็นต์))")
_QUANTITY = re.compile(
    r"\d[\d,.]*\s*(?:ราย|คน|หลัง|คัน|แห่ง|กิโลเมตร|เมตร|ไร่|ตัน|กิโลกรัม|ชั่วโมง|นาที|วัน|ปี)"
)


class RuleEntityRecognizer:
    """Extract Thai entities using titles, gazetteers and patterns."""

    name = "rules"

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _person_names(tokens: list[str]) -> list[str]:
        """Names following a personal title.

        Thai names are not capitalised, so the title is the only reliable
        boundary marker; up to two following tokens are taken as the name.
        """
        found: list[str] = []
        for index, token in enumerate(tokens):
            if token not in PERSON_TITLES:
                continue
            # "คุณผู้ชม" is the presenter addressing the audience.
            following = tokens[index + 1] if index + 1 < len(tokens) else ""
            if following in _NOT_A_PERSON:
                continue

            parts: list[str] = []
            for candidate in tokens[index + 1 : index + 3]:
                # Stop at punctuation, digits or another title.
                if candidate in PERSON_TITLES or candidate.isdigit():
                    break
                if candidate in _NAME_STOP_WORDS or candidate in _NOT_A_PERSON:
                    break
                if not re.search(r"[^\W\d_]", candidate, re.UNICODE):
                    break
                parts.append(candidate)

            name = "".join(parts)
            # A single character after a title is a tokenisation artefact.
            if len(name) >= MIN_NAME_CHARS:
                found.append(token + name)
        return found

    @staticmethod
    def _prefixed(tokens: list[str], prefixes: tuple[str, ...]) -> list[str]:
        """Entities formed by a known prefix plus what follows.

        Handles both shapes the tokenizer produces: the prefix welded into one
        token (``กระทรวงสาธารณสุข``) and split across tokens (``กระทรวง`` +
        ``สาธารณสุข``).
        """
        found: list[str] = []
        for index, token in enumerate(tokens):
            for prefix in prefixes:
                if token == prefix:
                    # Only the immediately following token is taken. Taking two
                    # was greedy enough to swallow trailing words, producing
                    # entities like "โรงพยาบาลขอนแก่นทันที".
                    following = tokens[index + 1 : index + 2]
                    parts = [
                        part
                        for part in following
                        if re.search(r"[^\W\d_]", part, re.UNICODE)
                        and part not in _NAME_STOP_WORDS
                    ]
                    if parts:
                        found.append(prefix + parts[0])
                    break
                if token.startswith(prefix) and len(token) > len(prefix):
                    found.append(token)
                    break
        return found

    # ---------------------------------------------------------------- extract
    def extract(self, text: str, tokens: list[str] | None = None) -> list[Entity]:
        """Return the entities found in ``text``, de-duplicated."""
        if not text:
            return []

        if tokens is None:
            from app.nlp.tokenizer import get_tokenizer

            tokens = get_tokenizer().tokenize(text)

        collected: list[tuple[str, str]] = []

        for name in self._person_names(tokens):
            collected.append((name, "PERSON"))

        for org in self._prefixed(tokens, ORG_PREFIXES):
            collected.append((org, "ORGANIZATION"))

        for place in self._prefixed(tokens, LOCATION_PREFIXES):
            collected.append((place, "LOCATION"))

        # Most province names are unambiguous, so match them directly.
        for index, token in enumerate(tokens):
            if token not in PROVINCES:
                continue
            # A handful are also everyday words: เลย ("at all"), ตาก ("to
            # dry"), น่าน, แพร่ ("to spread"). Accept those only when a
            # location prefix precedes them, or the particle sense floods the
            # results -- "เลย" alone produced more LOCATION hits than every
            # real province combined.
            if token in _AMBIGUOUS_PROVINCES:
                previous = tokens[index - 1] if index else ""
                if previous not in LOCATION_PREFIXES:
                    continue
            collected.append((token, "LOCATION"))

        for pattern, label in (
            (_DATE, "DATE"),
            (_TIME, "TIME"),
            (_MONEY, "MONEY"),
            (_PERCENT, "PERCENT"),
            (_QUANTITY, "QUANTITY"),
        ):
            for match in pattern.finditer(text):
                value = match.group().strip()
                if value:
                    collected.append((value, label))

        # De-duplicate, keeping first-seen order.
        seen: set[tuple[str, str]] = set()
        entities: list[Entity] = []
        for value, label in collected:
            key = (value, label)
            if key in seen:
                continue
            seen.add(key)
            entities.append(Entity(text=value, label=label))
        return entities


class PyThaiNLPEntityRecognizer:
    """PyThaiNLP CRF named entity recognition.

    Opt-in via ``NLP_NER_BACKEND=pythainlp``. Downloads a corpus on first use,
    and falls back to the rule-based recognizer if that is not possible.
    """

    name = "pythainlp"

    def __init__(self) -> None:
        self._fallback = RuleEntityRecognizer()
        self._tagger = None

    def _ensure_tagger(self) -> object | None:
        if self._tagger is None:
            try:
                from pythainlp.tag.named_entity import NER

                self._tagger = NER(engine="thainer")
            except Exception:
                self._tagger = False  # remember the failure, do not retry
        return self._tagger or None

    def extract(self, text: str, tokens: list[str] | None = None) -> list[Entity]:
        """Tag entities with the CRF model, or fall back to rules."""
        tagger = self._ensure_tagger()
        if tagger is None or not text:
            return self._fallback.extract(text, tokens)

        try:
            tagged = tagger.tag(text, pos=False)  # type: ignore[attr-defined]
        except Exception:
            return self._fallback.extract(text, tokens)

        entities: list[Entity] = []
        current: list[str] = []
        current_label: str | None = None

        for word, tag in tagged:
            if tag.startswith("B-"):
                if current and current_label:
                    entities.append(Entity("".join(current), current_label))
                current = [word]
                current_label = tag[2:]
            elif tag.startswith("I-") and current_label == tag[2:]:
                current.append(word)
            else:
                if current and current_label:
                    entities.append(Entity("".join(current), current_label))
                current = []
                current_label = None

        if current and current_label:
            entities.append(Entity("".join(current), current_label))

        return entities or self._fallback.extract(text, tokens)
