r"""Write a readable headline for a broadcast news story.

The problem this replaces
-------------------------
Joining a story's top keywords produced titles like
``ติดตาม · นิติ · ศุกร์ · เช้านี้`` -- four disconnected words that say nothing
about what happened. A reader scanning a timeline needs a description, not a
bag of terms.

Why this is harder than summarising an article
----------------------------------------------
There is no headline in the source to find. An ASR transcript of live Thai
broadcast is unpunctuated, spontaneous speech: presenters interrupt each other,
restate, and pad with ``นะครับ`` and ``ท่านผู้ชม``. Sentence segmentation does
not rescue it either -- on this material PyThaiNLP's ``crfcut`` returns run-on
blocks of 700+ characters next to fragments of 2, so "take the best sentence"
has nothing sensible to take.

So the headline is built by finding the most **information-dense span** of actual
speech: a short run of tokens that is about the story and reads as a phrase.

How the span is chosen
----------------------
Every candidate window of 7-20 tokens is scored on:

* **IDF mass** of its content words -- rare terms carry the story
* **overlap with the story's own keywords** -- proof it is on topic, not a digression
* **starting on a keyword**, which usually means starting on the subject
* **penalties** for broadcast filler, bare digits, and starting late in the story

and the winner is trimmed so it cannot begin or end on a dangling particle.

One subtlety that caused visibly broken output: the tokeniser drops whitespace,
so character offsets computed from cumulative token lengths drift and slices
start mid-word (``้วันศุกร์``). Offsets are therefore located in the source text
directly.

Honest limits
-------------
Quality is bounded by the transcript. ASR mangles names (``ชาญวีรกูล`` becomes
``ชาวรกูล``), and a presenter who rambles gives nothing dense to extract. Roughly
two thirds of stories get a genuinely descriptive headline; the rest get
something on-topic but clumsy. A language model writes a much better one -- and
needs no world knowledge to do it, only the transcript -- which is why the seam
exists; see :mod:`app.nlp.llm_enrich`. That is on by default through a local
model, so this extractive path is the fallback rather than the norm.
"""

from __future__ import annotations

import math
import re

from app.nlp.stopwords import BROADCAST_FILLER, get_stopwords

# Token count of a candidate span. Below 7 reads as a fragment; above 20 stops
# being a headline and becomes a sentence.
MIN_SPAN_TOKENS = 7
MAX_SPAN_TOKENS = 20
MAX_HEADLINE_CHARS = 85

# Content words needed before a span is considered at all.
MIN_CONTENT_TOKENS = 4

_THAI = re.compile(r"[฀-๿]")

# Particles, conjunctions and pronouns that must not open or close a headline.
# Thai allows most of these mid-phrase, but leading with one reads as a
# sentence cut in half.
_EDGE_TOKENS = frozenset(
    {
        "นะ", "ครับ", "ค่ะ", "คะ", "ก็", "ที่", "ซึ่ง", "แล้ว", "และ", "แต่", "ว่า",
        "เนี่ย", "อ่า", "เอ่อ", "เลย", "อยู่", "ของ", "ให้", "กับ", "จะ", "ได้",
        "มา", "ไป", "นี้", "นั้น", "ๆ", "เป็น", "คือ", "ใน", "จาก", "โดย", "ด้วย",
        "ทาง", "อัน", "ถึง", "ต่อ", "อะ", "ฮะ", "หรือ", "เมื่อ", "พอ", "ทั้ง",
        "อีก", "การ", "ความ", "มี", "ไม่", "เรา", "ผม", "เขา", "มัน", "อ้า",
        "เออ", "โอ้", "แหละ", "สิ", "ล่ะ", "หน่อย", "ไง",
    }
)

_LEAD_FILLER = BROADCAST_FILLER | frozenset(
    {"ท่านผู้ชม", "นะครับ", "นะคะ", "ครับผม", "สวัสดี", "ขอต้อนรับ", "เจอกัน"}
)


def token_offsets(text: str, tokens: list[str]) -> list[int]:
    """Character offset of each token within ``text``.

    Located by searching forward rather than by accumulating token lengths: the
    tokeniser drops whitespace, so cumulative lengths drift out of alignment and
    every slice after the first space starts mid-word.
    """
    offsets: list[int] = []
    position = 0
    for token in tokens:
        found = text.find(token, position)
        if found < 0:
            found = position
        offsets.append(found)
        position = found + len(token)
    return offsets


def _content_tokens(tokens: list[str], stopwords: frozenset[str]) -> list[str]:
    return [
        token
        for token in tokens
        if len(token) >= 2
        and _THAI.search(token)
        and token not in stopwords
        and token not in _LEAD_FILLER
    ]


def extract_headline(
    text: str,
    *,
    keywords: list[str] | None = None,
    idf: object | None = None,
    cue_starts: set[int] | None = None,
    max_chars: int = MAX_HEADLINE_CHARS,
) -> str:
    """Best descriptive span of ``text``, or ``""`` when there is none.

    Args:
        text: Cleaned transcript of one story.
        keywords: The story's extracted keywords; spans containing them score
            higher, because they are what the story is about.
        idf: Optional fitted keyword extractor, used for term rarity. Without
            it every content term counts equally, which still works but is
            blunter.
        cue_starts: Character offsets where a transcript cue begins. A cue is a
            natural unit of speech, so starting there is far more likely to open
            on a clean phrase. Without this the span can begin on a fragment of
            a split name -- ``พงษ์รัชตะเกียงไกร`` instead of
            ``เกียรติพงษ์ รัชตะเกียงไกร`` -- because Thai has no spaces and the
            tokeniser splits inside names.
    """
    from app.nlp.tokenizer import get_tokenizer

    body = (text or "").strip()
    if not body:
        return ""

    tokens = get_tokenizer().tokenize(body)
    if len(tokens) < MIN_SPAN_TOKENS:
        return ""

    offsets = token_offsets(body, tokens)
    offsets.append(len(body))

    stopwords = get_stopwords(protect_polarity=False)
    wanted = set(keywords or ())

    def weight(term: str) -> float:
        if idf is None:
            return 1.0
        try:
            return float(idf._idf(term))  # noqa: SLF001 - internal by design
        except Exception:
            return 1.0

    total = len(tokens)
    best: tuple[int, int] | None = None
    best_score = -math.inf

    for start in range(total):
        if tokens[start] in _EDGE_TOKENS or tokens[start] in _LEAD_FILLER:
            continue
        for length in range(MIN_SPAN_TOKENS, MAX_SPAN_TOKENS + 1):
            stop = start + length
            if stop > total:
                break
            if offsets[stop] - offsets[start] > max_chars:
                break

            window = tokens[start:stop]
            content = _content_tokens(window, stopwords)
            if len(content) < MIN_CONTENT_TOKENS:
                continue
            hits = wanted & set(content)
            if wanted and not hits:
                # A span with none of the story's keywords is a digression.
                continue

            score = sum(weight(term) for term in set(content))
            score += 2.5 * len(hits)
            if tokens[start] in wanted:
                score += 2.0
            score -= 1.2 * sum(1 for token in window if token in _LEAD_FILLER)
            score -= 0.6 * sum(1 for token in window if token.isdigit())
            if cue_starts and offsets[start] in cue_starts:
                # Opening on a unit of speech, not mid-phrase.
                score += 3.0
            score /= math.sqrt(len(window))
            # Mild preference for the opening: a news item leads with the story.
            score -= (start / total) * 1.2

            if score > best_score:
                best_score = score
                best = (start, stop)

    if best is None:
        return ""

    start, stop = best
    while start < stop and (
        tokens[start] in _EDGE_TOKENS or tokens[start] in _LEAD_FILLER
    ):
        start += 1
    while stop > start and (
        tokens[stop - 1] in _EDGE_TOKENS or tokens[stop - 1] in _LEAD_FILLER
    ):
        stop -= 1
    if stop <= start:
        return ""

    headline = body[offsets[start] : offsets[stop]].strip()
    return re.sub(r"\s+", " ", headline)
