r"""Write story headlines, and optionally repair ASR-mangled Thai names.

Two jobs, and they are not equally achievable
---------------------------------------------
**Headlines need no world knowledge**, only the transcript. A 7B model running
locally does them well -- measured against the extractive baseline on real
stories:

    ซากดิงที่ถูกยิงตายจมกองเลือดใส่ตะกร้าสีชมพู...
      -> เหตุยิงลิงแสมตกลงมาจากบ้านในสงขลา

    กรุงเทพนท์ 43เห็นผู้ก่อเหตุขับกระบะเข้าออกซอย...
      -> เหตุสลดหญิงถูกฆ่าหลังถูกตามง้อคืนดี

So this is **on by default via Ollama**: free, local, no key, no account, and
no data leaves the machine.

**Name repair needs knowing which Thai names exist**, and that is the part a
small local model cannot do. YouTube's ASR writes ``ศศิภาพร จันทวิสูตร`` as
``สักสิภาพรจันทวิสูตร``, and every local approach was measured and failed:
Thai soundex does not match (the ASR inserts a syllable), PyThaiNLP's 22k name
corpora do not contain the names, and cross-referencing the chat only works when
the ASR happened to get it right somewhere else.

Asking a 7B model instead makes it **worse**. Told explicitly not to guess,
qwen2.5:7b produced:

    อนุทินชาวรกูล      -> อนุทินชื่นกล่าว        (invented)
    อัถสิทธิ์เวชชาชีวะ  -> อัชสิทธิ์เวชชาชีวะ    (still wrong)
    อำสินสักสิภาพร...  -> อำพันสิทธิ์จันทวิสูตร  (invented)

A plausible-looking wrong name is strictly worse than a visibly garbled one,
because a reader cannot tell it happened. Name repair is therefore **off by
default** (``LLM_CORRECT_NAMES``), and the prompt that invites it is not even
sent unless it is switched on -- a prompt that asks for a guess gets one.

Providers
---------
``ollama`` (default) runs a model locally: free, ~20s per story on CPU.
``anthropic`` uses the hosted API, needs ``LLM_API_KEY``, and is the only
option worth enabling name repair on.

Either way every failure degrades to the extractive result rather than losing
the segment, and with neither available the pipeline behaves exactly as before.
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

# Headline only. No world knowledge required, so a small local model does it
# well -- measured against the extractive baseline on real stories:
#   "ซากดิงที่ถูกยิงตายจมกองเลือดใส่ตะกร้าสีชมพู..."
#     -> "เหตุยิงลิงแสมตกลงมาจากบ้านในสงขลา"
# The instruction deliberately avoids the phrase "ข้อความนี้พูดถึงเรื่องอะไร".
# An earlier wording asked exactly that and the model answered it literally --
# "ข้อความนี้พูดถึงเรื่องการลุยธุรกิจโรงแรม..." -- a sentence *about* the
# transcript instead of a headline *for* it. Naming the output and not the
# question reduces that; _clean_headline still trims it when it happens.
HEADLINE_PROMPT = """เขียนพาดหัวข่าวภาษาไทยสำหรับคำถอดเสียงข่าวด้านล่าง ความยาวไม่เกิน 80 ตัวอักษร
พาดหัวต้องบอกว่าใครทำอะไรที่ไหน เหมือนพาดหัวในหน้าหนึ่งของหนังสือพิมพ์
ตอบเป็นพาดหัวเพียงบรรทัดเดียว ไม่ต้องอธิบาย ไม่ต้องใส่เครื่องหมายคำพูด
ห้ามขึ้นต้นด้วยคำว่า "พาดหัว" หรือ "ข้อความนี้"

คำถอดเสียง: {text}"""

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


# Models label their own output despite being told not to: 2 of the first 36
# headlines came back as "พาดหัว: ..." even though the prompt says
# ไม่ต้องขึ้นต้นว่า "พาดหัว". Instructions reduce this; they do not remove it, so
# the label is also stripped on the way out. Only an explicit ``label:`` form is
# removed, never a bare word, or a headline that genuinely opens with สรุป would
# lose it.
_LABEL_PREFIX = re.compile(
    r"^(?:พาดหัวข่าว|พาดหัว|หัวข้อข่าว|หัวข้อ|สรุปข่าว|สรุป|headline)\s*[:：]\s*",
    re.IGNORECASE,
)

# The other thing a model does instead of answering: restate the question.
# "จงเขียนพาดหัข่าว...สรุปว่าข้อความนี้พูดถึงเรื่องอะไร" comes back as
# "ข้อความนี้พูดถึงเรื่องการลุยธุรกิจโรงแรม..." -- a sentence about the text
# rather than a headline for it. The remainder after the preamble is the
# headline, so it is trimmed rather than thrown away.
_PREAMBLE = re.compile(
    r"^(?:ข้อความ|ข่าว|เนื้อหา|คลิป|บทความ)(?:นี้)?\s*"
    r"(?:พูดถึง|กล่าวถึง|เกี่ยวกับ|นำเสนอ|รายงาน|เป็นเรื่อง)\s*(?:เรื่อง)?\s*"
)

# Scripts and letters a Thai news headline never uses. qwen2.5 drifts into
# Chinese --
# story 57 of the second programme came back entirely as 政坛对峙：反击与回应,
# and a short prompt produced Thai that switched mid-sentence. Either way the
# headline is unusable, and half a headline in the wrong script is worse than
# none because the extractive fallback would have been readable.
_FOREIGN_SCRIPT = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff"  # japanese, chinese
    r"\uac00-\ud7af\u0400-\u04ff"  # korean, cyrillic
    r"\u00c0-\u024f]"  # accented latin: "démarchงบฯ กกต." was a real completion
)

# Straight and typographic quotes, which models like to wrap a headline in.
_QUOTES = "\"'“”‘’「」"


def _clean_headline(value: object) -> str:
    """Normalise whatever the model returned into a bare headline.

    Applied to cached headlines too, so tightening this reaches work that was
    already paid for rather than only new calls.
    """
    # The first non-empty line only. A model that keeps talking after
    # answering -- a second paragraph explaining its own headline -- would
    # otherwise have that commentary folded in by the whitespace collapse.
    lines = [line for line in str(value or "").splitlines() if line.strip()]
    text = re.sub(r"\s+", " ", lines[0] if lines else "").strip()
    text = text.strip(_QUOTES).strip()
    # A completion of nothing but its own label is a failure, and returning it
    # empty is how the caller learns to keep the extractive headline instead of
    # printing "พาดหัว:" on the timeline.
    text = _LABEL_PREFIX.sub("", text).strip().strip(_QUOTES).strip()
    text = _PREAMBLE.sub("", text).strip().strip(_QUOTES).strip()
    # A Thai news headline with no Thai in it is not a headline. Asked for one,
    # qwen2.5:7b answered story 57 of the second programme in Chinese
    # ("政坛对峙：反击与回应"), and a headline in the wrong language is worse than a
    # clumsy Thai one because the timeline stops being readable. Rejected here
    # rather than checked at the call site, so a cached one is rejected too.
    if _FOREIGN_SCRIPT.search(text) or not _THAI.search(text):
        return ""
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


def _call_ollama(prompt: str, *, timeout: float) -> str:
    """Ask a locally-running model. Free, no key, nothing leaves the machine."""
    import httpx

    response = httpx.post(
        f"{settings.ollama_base_url.rstrip('/')}/api/generate",
        json={
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "num_predict": 400,
                # Bounded on purpose. qwen2.5's own context is 32k, and letting
                # the server size its KV cache for that costs gigabytes of RAM it
                # will never use: MAX_CONTEXT_CHARS caps the prompt at 3,000
                # characters. On a 16 GB machine the difference is whether the
                # run survives -- the 7B model already holds ~6 GB.
                "num_ctx": 4096,
            },
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return str(response.json().get("response") or "")


def _call_anthropic(prompt: str, *, timeout: float) -> str:
    """Ask the hosted API. Needs a key and costs money."""
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
            "messages": [{"role": "user", "content": prompt}],
        },
        timeout=timeout,
    )
    response.raise_for_status()
    blocks = response.json().get("content") or []
    return "".join(b.get("text", "") for b in blocks if isinstance(b, dict))


def _complete(prompt: str, *, timeout: float) -> str:
    """Route to the configured provider."""
    try:
        if settings.llm_provider == "ollama":
            return _call_ollama(prompt, timeout=timeout)
        if not settings.has_llm:
            raise LLMUnavailable("LLM_API_KEY is not set")
        return _call_anthropic(prompt, timeout=timeout)
    except LLMUnavailable:
        raise
    except Exception as exc:
        raise LLMUnavailable(f"{type(exc).__name__}: {exc}") from exc


def enrich_segment(text: str, *, timeout: float = 120.0) -> Enrichment:
    """Headline and corrected entities for one story.

    Raises:
        LLMUnavailable: on any failure, so the caller can fall back rather than
            lose the segment.
    """
    body = (text or "").strip()[:MAX_CONTEXT_CHARS]
    if len(body) < 40:
        raise LLMUnavailable("segment too short to enrich")

    # Name correction is a separate, riskier capability. A local model was
    # measured inventing Thai names it was told not to guess at, so asking for
    # them at all is gated rather than merely ignored afterwards -- a prompt
    # that invites a guess gets one.
    if settings.llm_correct_names:
        enrichment = _parse(_complete(PROMPT.format(text=body), timeout=timeout))
    else:
        headline = _clean_headline(
            _complete(HEADLINE_PROMPT.format(text=body), timeout=timeout)
        )
        enrichment = Enrichment(headline=headline)

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
