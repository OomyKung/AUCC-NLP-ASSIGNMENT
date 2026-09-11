r"""Repair ASR-mangled Thai names, and write a headline, with an LLM.

Why an LLM, after everything else was tried
-------------------------------------------
YouTube's speech recognition transcribes Thai names phonetically and gets them
wrong: ``ศศิภาพร จันทวิสูตร`` comes out as ``สักสิภาพรจันทวิสูตร``. No amount of
NER fixes that -- NER locates spans, it does not correct spelling -- so a
correction needs an outside reference. Four were measured first, and each failed
for a different reason:

================================  =========================================
Approach                          Why it does not work here
================================  =========================================
PyThaiNLP Thai soundex            ``ศศิภาพร``/``สักสิภาพร`` do not match on
(lk82, udom83, metasound)         any of the three algorithms: the ASR
                                  *inserts a syllable*, which changes the
                                  consonant code the algorithms encode.
PyThaiNLP name corpora            22k given and family names, and neither
                                  ``ศศิภาพร`` nor ``จันทวิสูตร`` is in them.
                                  There is nothing to match against.
Cross-reference chat/transcript   Works sometimes and cannot be relied on:
                                  ``จันทวิสูตร`` and ``รัชตะเกียงไกร`` do
                                  appear correctly elsewhere in the same
                                  transcript, but ``ศศิภาพร`` never does.
Switch NER to the CRF model       Better *spans* -- it captures
                                  ``คุณสาธิตวงษ์หนองเตย`` whole rather than
                                  truncating -- but the characters are still
                                  the ASR's.
================================  =========================================

What is actually needed is knowledge of which Thai names exist and which one a
phonetic approximation was reaching for. That is world knowledge, which is what
an LLM has and a gazetteer of this size does not.

Opt-in, and honest about it
---------------------------
Requires ``LLM_API_KEY``. Without it the pipeline behaves exactly as before --
extractive headline, rule-based entities -- so nothing here is load-bearing for
a fresh clone or an offline demo. Every failure degrades to the existing result
rather than breaking a segment, because a worse headline is a far better outcome
than a missing story.

One call per segment does both jobs: correcting the names and writing the
headline need the same context, and sending it twice would double the cost for
nothing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from app.config import settings

# Transcript sent per segment. A five-minute story is a few thousand characters;
# this keeps the request small while leaving enough for the model to recognise
# who is being talked about.
MAX_CONTEXT_CHARS = 3_000

# Guards on what comes back, because a model can return anything.
MAX_HEADLINE_CHARS = 120
MAX_ENTITIES = 12

_THAI = re.compile(r"[฀-๿]")

PROMPT = """คุณเป็นบรรณาธิการข่าวไทย ข้อความด้านล่างเป็นคำถอดเสียงอัตโนมัติ (ASR) จากรายการข่าวโทรทัศน์ไทย ซึ่งมักสะกดชื่อคนผิดเพราะถอดตามเสียง เช่น "ศศิภาพร จันทวิสูตร" อาจกลายเป็น "สักสิภาพรจันทวิสูตร"

งานของคุณ:
1. เขียนพาดหัวข่าวภาษาไทยสั้น ๆ ไม่เกิน 100 ตัวอักษร บอกว่าข่าวนี้เกี่ยวกับอะไร
2. ระบุชื่อเฉพาะที่ปรากฏ (บุคคล องค์กร สถานที่) และแก้การสะกดให้ถูกต้องตามที่ควรจะเป็น

กติกา:
- ถ้าไม่แน่ใจว่าชื่อที่ถูกต้องคืออะไร ให้คงรูปเดิมไว้ อย่าเดา
- ห้ามเพิ่มข้อมูลที่ไม่มีในข้อความ
- ตอบเป็น JSON เท่านั้น ไม่ต้องมีคำอธิบายอื่น

รูปแบบคำตอบ:
{{"headline": "...", "entities": [{{"text": "ชื่อที่ถูกต้อง", "label": "PERSON|ORGANIZATION|LOCATION", "asr": "รูปที่ ASR ถอดมา"}}]}}

ข้อความ:
{text}"""


@dataclass(slots=True)
class Enrichment:
    """What the model returned for one segment."""

    headline: str = ""
    entities: list[dict] = field(default_factory=list)
    # Names it actually changed, as (asr_form, corrected) -- worth surfacing so
    # a reader can see the correction rather than trust it silently.
    corrections: list[tuple[str, str]] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        return bool(self.headline or self.entities)


class LLMUnavailable(RuntimeError):
    """Raised when the model cannot be reached or returns nothing usable."""


def _clean_headline(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip().strip('"')
    if len(text) > MAX_HEADLINE_CHARS:
        text = text[: MAX_HEADLINE_CHARS - 1].rstrip() + "…"
    return text


def _parse(payload: str) -> Enrichment:
    """Read the model's JSON, tolerating the fences models like to add."""
    body = payload.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)\s*```", body, re.S)
    if fenced:
        body = fenced.group(1)
    start, end = body.find("{"), body.rfind("}")
    if start < 0 or end <= start:
        raise LLMUnavailable("no JSON object in the completion")

    try:
        data = json.loads(body[start : end + 1])
    except json.JSONDecodeError as exc:
        raise LLMUnavailable(f"completion was not valid JSON: {exc}") from exc

    entities: list[dict] = []
    corrections: list[tuple[str, str]] = []
    for item in (data.get("entities") or [])[:MAX_ENTITIES]:
        if not isinstance(item, dict):
            continue
        text = re.sub(r"\s+", " ", str(item.get("text") or "")).strip()
        label = str(item.get("label") or "").upper()
        if not text or label not in {"PERSON", "ORGANIZATION", "LOCATION"}:
            continue
        # Reject anything without a Thai character: the model occasionally
        # echoes an instruction or a stray English word.
        if not _THAI.search(text):
            continue
        entities.append({"text": text, "label": label})
        asr = re.sub(r"\s+", "", str(item.get("asr") or ""))
        if asr and asr != re.sub(r"\s+", "", text):
            corrections.append((asr, text))

    return Enrichment(
        headline=_clean_headline(data.get("headline")),
        entities=entities,
        corrections=corrections,
    )


def enrich_segment(text: str, *, timeout: float = 40.0) -> Enrichment:
    """Headline and corrected entities for one story.

    Raises:
        LLMUnavailable: on any failure, so the caller can fall back rather than
            lose the segment.
    """
    if not settings.has_llm:
        raise LLMUnavailable("LLM_API_KEY is not set")

    body = (text or "").strip()[:MAX_CONTEXT_CHARS]
    if len(body) < 40:
        raise LLMUnavailable("segment too short to enrich")

    try:
        import httpx

        response = httpx.post(
            settings.llm_base_url,
            headers={
                "x-api-key": settings.llm_api_key or "",
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": settings.llm_model,
                "max_tokens": 1024,
                "messages": [{"role": "user", "content": PROMPT.format(text=body)}],
            },
            timeout=timeout,
        )
        response.raise_for_status()
        blocks = response.json().get("content") or []
        completion = "".join(
            block.get("text", "") for block in blocks if isinstance(block, dict)
        )
    except Exception as exc:  # network, auth, rate limit, malformed response
        raise LLMUnavailable(f"{type(exc).__name__}: {exc}") from exc

    enrichment = _parse(completion)
    if not enrichment.usable:
        raise LLMUnavailable("completion contained neither a headline nor entities")
    return enrichment


def apply_corrections(text: str, corrections: list[tuple[str, str]]) -> str:
    """Rewrite ASR name forms in ``text`` to their corrected spellings.

    Longest first, so correcting ``สักสิภาพรจันทวิสูตร`` does not get pre-empted
    by a shorter overlapping form.
    """
    result = text or ""
    for asr, fixed in sorted(corrections, key=lambda pair: -len(pair[0])):
        if asr and asr in result:
            result = result.replace(asr, fixed)
    return result
