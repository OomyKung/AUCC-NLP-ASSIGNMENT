"""What the audience said while a story was on air.

The project has two recordings of the same broadcast: the newsreader's words,
transcribed and cut into stories, and the viewers' words, timestamped by the
same clock. Nothing was doing anything with the fact that those two share a
timeline, so this pairs them -- for any story, the chat that arrived during it.

That is the whole idea: the reporter says a man was arrested; the chat says the
sentence is too light. Neither is the news on its own.

Two things are computed here:

* **The facts**, from stored data alone: how many messages, the sentiment mix
  the per-message classifier already produced, the words viewers used, and a
  sample to read. No model, no cost, always available.
* **One line of what they were on about**, from the LLM. A hundred short Thai
  messages full of ``5555`` and emotes do not read as an opinion until something
  compresses them, and that is exactly the job a language model is good at --
  no world knowledge required, only the text.

Alignment is by ``offset_ms``: milliseconds from the start of the stream, which
both the chat collector and the transcript provider record. A message with no
offset (a live tail rather than a replay) is skipped rather than guessed at.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from sqlalchemy import and_, func, select
from sqlalchemy.orm import Session

from app.models.broadcast import NewsSegment
from app.models.chat import ChatMessage

# Messages read back to the user per story. Enough to see the mood without
# turning a timeline card into a chat log.
SAMPLE_SIZE = 12

# Chat sent to the model. Beyond this the summary stops improving and the call
# gets slow; a busy story is a few hundred messages of "5555".
MAX_SUMMARY_MESSAGES = 120
MAX_SUMMARY_CHARS = 2_500

# Below this there is no "audience reaction" to summarise, only a couple of
# people talking.
MIN_MESSAGES_TO_SUMMARISE = 5

_WHITESPACE = re.compile(r"\s+")


@dataclass(slots=True)
class Reaction:
    """The chat that arrived while one story was on air."""

    total: int = 0
    sentiment_counts: dict[str, int] = field(default_factory=dict)
    keywords: list[str] = field(default_factory=list)
    samples: list[dict] = field(default_factory=list)
    summary: str = ""

    @property
    def mood(self) -> str:
        """The sentiment most messages carried.

        ``""`` when nothing was said, and ``"mixed"`` when the top two are level
        -- 4 negative against 4 neutral has no majority, and picking one is
        both untrue and unstable: ``max`` follows dictionary order, so the same
        story reported a different mood depending on which query built it.
        """
        if not self.sentiment_counts:
            return ""
        ranked = sorted(self.sentiment_counts.items(), key=lambda pair: (-pair[1], pair[0]))
        if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
            return "mixed"
        return ranked[0][0]

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "sentiment_counts": self.sentiment_counts,
            "mood": self.mood,
            "keywords": self.keywords,
            "samples": self.samples,
            "summary": self.summary,
        }


def messages_in(db: Session, segment: NewsSegment) -> list[ChatMessage]:
    """Every non-spam chat message sent while ``segment`` was on air.

    Spam is excluded for the same reason the pipeline excludes it from analysis:
    forty copies of one emote is not forty opinions.
    """
    return list(
        db.scalars(
            select(ChatMessage)
            .where(
                ChatMessage.stream_id == segment.stream_id,
                ChatMessage.is_spam.is_(False),
                ChatMessage.offset_ms.is_not(None),
                ChatMessage.offset_ms >= segment.start_ms,
                ChatMessage.offset_ms < segment.end_ms,
            )
            .order_by(ChatMessage.offset_ms)
        )
    )


def count_by_segment(db: Session, segments: list[NewsSegment]) -> dict[int, int]:
    """How many messages fell inside each story, as ``{segment id: count}``.

    One query for the whole programme rather than one per story: the timeline
    lists every story at once, and 168 correlated counts is a page that feels
    slow for no reason.
    """
    if not segments:
        return {}
    stream_ids = {segment.stream_id for segment in segments}
    rows = db.execute(
        select(NewsSegment.id, func.count(ChatMessage.id))
        .select_from(NewsSegment)
        .outerjoin(
            ChatMessage,
            and_(
                ChatMessage.stream_id == NewsSegment.stream_id,
                ChatMessage.is_spam.is_(False),
                ChatMessage.offset_ms.is_not(None),
                ChatMessage.offset_ms >= NewsSegment.start_ms,
                ChatMessage.offset_ms < NewsSegment.end_ms,
            ),
        )
        .where(NewsSegment.stream_id.in_(stream_ids))
        .group_by(NewsSegment.id)
    ).all()
    return {segment_id: count for segment_id, count in rows}


def overview_by_segment(
    db: Session, segments: list[NewsSegment]
) -> dict[int, Reaction]:
    """The headline facts of every story's chat, in one query.

    What the timeline needs to render 76 cards: how many messages and how they
    felt. Deliberately *not* the keywords or the sample messages -- those mean
    tokenising and shipping thousands of messages the reader has not asked to
    see, so they are fetched per story when a card is opened.
    """
    if not segments:
        return {}
    by_id = {segment.id: segment for segment in segments}
    rows = db.execute(
        select(NewsSegment.id, ChatMessage.sentiment, func.count(ChatMessage.id))
        .select_from(NewsSegment)
        .join(
            ChatMessage,
            and_(
                ChatMessage.stream_id == NewsSegment.stream_id,
                ChatMessage.is_spam.is_(False),
                ChatMessage.offset_ms.is_not(None),
                ChatMessage.offset_ms >= NewsSegment.start_ms,
                ChatMessage.offset_ms < NewsSegment.end_ms,
            ),
        )
        .where(NewsSegment.id.in_(by_id))
        .group_by(NewsSegment.id, ChatMessage.sentiment)
    ).all()

    result: dict[int, Reaction] = {
        segment.id: Reaction(summary=segment.chat_summary or "")
        for segment in segments
    }
    for segment_id, sentiment, count in rows:
        reaction = result[segment_id]
        reaction.total += count
        if sentiment:
            reaction.sentiment_counts[sentiment] = count
    return result


def summarise_texts(texts: list[str], *, timeout: float = 120.0) -> str:
    """One Thai line describing what these viewers were saying.

    Raises:
        LLMUnavailable: on any failure, so a caller can fall back to showing the
            messages themselves -- which is still useful, just less digestible.
    """
    from app.nlp.llm_enrich import CHAT_PROMPT, LLMUnavailable, _clean_headline, _complete

    cleaned = [_WHITESPACE.sub(" ", text).strip() for text in texts if text]
    cleaned = [text for text in cleaned if text][:MAX_SUMMARY_MESSAGES]
    if len(cleaned) < MIN_MESSAGES_TO_SUMMARISE:
        raise LLMUnavailable("too few messages to summarise")

    body = ""
    for text in cleaned:
        if len(body) + len(text) > MAX_SUMMARY_CHARS:
            break
        body += f"- {text}\n"

    summary = _clean_headline(_complete(CHAT_PROMPT.format(text=body), timeout=timeout))
    if not summary:
        raise LLMUnavailable("the model returned nothing usable for the chat")
    return summary


def build(db: Session, segment: NewsSegment) -> Reaction:
    """Everything the UI shows for one story's audience reaction.

    Reads the stored summary rather than writing one: producing it costs a model
    call, so it happens in a background job (see
    :mod:`app.services.enrichment_jobs`) and this only reports what is there.
    """
    from app.nlp.keyword_extractor import keyword_candidates
    from app.nlp.preprocessing import clean_text
    from app.nlp.stopwords import BROADCAST_FILLER
    from app.nlp.tokenizer import get_tokenizer

    rows = messages_in(db, segment)
    if not rows:
        return Reaction(summary=segment.chat_summary or "")

    counts = Counter(row.sentiment for row in rows if row.sentiment)

    tokenizer = get_tokenizer()
    words: Counter[str] = Counter()
    for row in rows:
        tokens = tokenizer.tokenize(clean_text(row.text))
        words.update(keyword_candidates(tokens, exclude=BROADCAST_FILLER))

    # The middle of the story, where a reaction is more likely than a greeting.
    middle = rows[len(rows) // 4 : len(rows) // 4 + SAMPLE_SIZE] or rows[:SAMPLE_SIZE]

    return Reaction(
        total=len(rows),
        sentiment_counts=dict(counts),
        keywords=[word for word, _ in words.most_common(8)],
        samples=[
            {
                "text": row.text,
                "sentiment": row.sentiment or "neutral",
                # Carried so the UI can fade a prediction it should not be
                # presenting as a verdict; see app.config.sentiment_unclear_below.
                "confidence": row.sentiment_confidence or 0.0,
                "offset_ms": row.offset_ms or 0,
            }
            for row in middle
        ],
        summary=segment.chat_summary or "",
    )
