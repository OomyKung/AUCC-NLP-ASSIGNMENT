"""Turn collected chat into analysed documents.

Bridges Phase 2 (collection) and the NLP pipeline: messages are grouped into
windows, each window is analysed as one document and stored as a
:class:`~app.models.news.NewsArticle`, and every message keeps its own sentiment
so the UI can drill from a window down to what individual viewers said.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from app.config import settings
from app.models.analysis import NLPAnalysis
from app.models.chat import ChatMessage, ChatStream
from app.models.news import NewsArticle, SourceType
from app.nlp.keyword_extractor import TfidfKeywordExtractor
from app.nlp.preprocessing import is_noise
from app.services.collectors.base import RawChatMessage
from app.services.pipeline import NLPPipeline, get_pipeline
from app.services.windowing import build_windows, synthesise_title
from app.taxonomy import FALLBACK_SENTIMENT, SENTIMENT_SLUGS


@dataclass(slots=True)
class IngestReport:
    """What an analysis run produced."""

    streams: int = 0
    windows_created: int = 0
    messages_scored: int = 0
    messages_skipped_noise: int = 0

    def as_dict(self) -> dict:
        return {
            "streams": self.streams,
            "windows_created": self.windows_created,
            "messages_scored": self.messages_scored,
            "messages_skipped_noise": self.messages_skipped_noise,
        }


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _to_raw(message: ChatMessage) -> RawChatMessage:
    """Adapt a stored message back to the windowing input shape."""
    return RawChatMessage(
        text=message.text,
        published_at=message.published_at,
        message_id=message.message_id,
        author=message.author,
        offset_ms=message.offset_ms,
    )


# A window needs this share of opinionated messages before it is called
# positive or negative at all.
OPINIONATED_FLOOR = 0.15

# Positive and negative shares closer than this are a genuinely mixed reaction.
POLARITY_MARGIN = 0.04


def _aggregate_sentiment(
    messages: list[ChatMessage],
) -> tuple[str, dict[str, float], float]:
    """Aggregate member-message sentiments into a window-level verdict.

    Returns the label and the share of each class among scored messages.
    Messages filtered as noise are excluded so junk cannot sway a window.

    Labelling is deliberately **not** a plain majority vote. Most short chat
    messages contain no sentiment word, so ~69% of them score neutral and a
    majority vote makes essentially every window neutral -- true but useless.
    What is informative about an audience reaction is the balance among the
    messages that *did* express something, so a window is called positive or
    negative when opinionated messages are numerous enough and clearly
    one-sided, and neutral when they are sparse or evenly split.

    Returns ``(label, shares, confidence)``. The distribution is the factual
    share of each class, which is what the charts display; the label is the
    net-reaction verdict over it. Confidence expresses certainty in that
    verdict, not the raw share -- for a polar label it is the winning
    polarity's share *among opinionated messages* ("of viewers who expressed an
    opinion, 58% were positive"), because reporting the raw 0.22 next to the
    word "Positive" reads as a broken result.
    """
    counts = {slug: 0 for slug in SENTIMENT_SLUGS}
    total = 0
    for message in messages:
        if message.sentiment in counts:
            counts[message.sentiment] += 1
            total += 1

    if total == 0:
        empty = {
            slug: 1.0 if slug == FALLBACK_SENTIMENT else 0.0 for slug in SENTIMENT_SLUGS
        }
        return FALLBACK_SENTIMENT, empty, 1.0

    shares = {slug: counts[slug] / total for slug in SENTIMENT_SLUGS}
    positive = shares["positive"]
    negative = shares["negative"]
    opinionated = positive + negative

    if opinionated < OPINIONATED_FLOOR or abs(positive - negative) < POLARITY_MARGIN:
        return "neutral", shares, shares["neutral"]

    label = "positive" if positive > negative else "negative"
    confidence = shares[label] / opinionated if opinionated else 0.0
    return label, shares, confidence


def fit_keyword_idf(db: Session, pipeline: NLPPipeline) -> int:
    """Learn keyword IDF from the stored corpus and cache it.

    Keyword extraction degrades to term frequency without corpus statistics, so
    this is run before analysis to make the IDF term real.

    Returns:
        The number of documents the statistics were built from.
    """
    tokenizer = pipeline.components.tokenizer
    documents: list[list[str]] = []

    for stream in db.scalars(select(ChatStream)):
        messages = list(
            db.scalars(
                select(ChatMessage)
                .where(ChatMessage.stream_id == stream.id)
                .order_by(ChatMessage.published_at)
            )
        )
        usable = [_to_raw(m) for m in messages if not is_noise(m.text)]
        for window in build_windows(usable):
            documents.append(tokenizer.tokenize(window.as_document()))

    if not documents:
        return 0

    extractor = TfidfKeywordExtractor().fit(documents)
    extractor.save()

    # Make the freshly fitted statistics active for this process.
    pipeline.components.keywords = extractor
    return len(documents)


def analyse_stream(
    db: Session,
    stream: ChatStream,
    pipeline: NLPPipeline,
    *,
    replace: bool = True,
) -> IngestReport:
    """Analyse one stream's chat into windowed documents.

    Args:
        replace: Delete this stream's existing windows first, so re-running is
            idempotent rather than duplicating documents.
    """
    report = IngestReport(streams=1)

    if replace:
        existing = list(
            db.scalars(select(NewsArticle.id).where(NewsArticle.stream_id == stream.id))
        )
        if existing:
            # Detach messages before the documents disappear.
            for message in db.scalars(
                select(ChatMessage).where(ChatMessage.news_id.in_(existing))
            ):
                message.news_id = None
                message.window_index = None
            db.flush()
            db.execute(delete(NewsArticle).where(NewsArticle.id.in_(existing)))
            db.flush()

    messages = list(
        db.scalars(
            select(ChatMessage)
            .where(ChatMessage.stream_id == stream.id)
            .order_by(ChatMessage.published_at)
        )
    )
    if not messages:
        return report

    # Per-message sentiment for every message, including ones too thin to window.
    texts = [m.text for m in messages]
    predictions = pipeline.sentiment_only(texts)
    by_message_id: dict[int, ChatMessage] = {m.id: m for m in messages}

    for message, prediction in zip(messages, predictions, strict=True):
        noisy = is_noise(message.text)
        message.is_spam = noisy
        if noisy:
            report.messages_skipped_noise += 1
            # A noise message gets no sentiment rather than a misleading one.
            message.sentiment = None
            message.sentiment_confidence = None
        else:
            message.sentiment = prediction.label
            message.sentiment_confidence = round(prediction.confidence, 4)
            report.messages_scored += 1

    # Windows are built from analysable messages only.
    usable = [m for m in messages if not m.is_spam]
    windows = build_windows([_to_raw(m) for m in usable])

    # Map window membership back onto the stored rows by position.
    cursor = 0
    for window in windows:
        window_messages = usable[cursor : cursor + window.count]
        cursor += window.count

        document = window.as_document()
        result = pipeline.analyse(document, title=None)

        # Window sentiment is aggregated from its member messages rather than
        # taken from the concatenated text. Summing polarity across 200 messages
        # saturates: every window came out positive or negative and none
        # neutral, even though the messages themselves are ~69% neutral. The
        # share of member sentiments is both accurate and directly explainable
        # ("of 200 messages, 69% were neutral").
        (
            window_sentiment,
            window_probabilities,
            window_confidence,
        ) = _aggregate_sentiment(window_messages)

        title = synthesise_title(
            window,
            stream_title=stream.title,
            keywords=result.keyword_words,
        )

        article = NewsArticle(
            title=title,
            content=document,
            summary=result.summary.text if result.summary else "",
            topic=result.topic.label,
            topic_confidence=round(result.topic.confidence, 4),
            sentiment=window_sentiment,
            sentiment_confidence=round(window_confidence, 4),
            keywords=[keyword.as_dict() for keyword in result.keywords],
            source=stream.channel or "YouTube",
            url=stream.url,
            published_at=window.start,
            source_type=SourceType.CHAT_WINDOW.value,
            content_hash=_content_hash(f"{stream.video_id}:{window.index}:{document}"),
            stream_id=stream.id,
            message_count=window.count,
            window_start=window.start,
            window_end=window.end,
        )
        article.analysis = NLPAnalysis(
            cleaned_text=result.preprocessing.cleaned_text,
            tokens=result.preprocessing.tokens[:600],
            filtered_tokens=result.preprocessing.filtered_tokens[:600],
            token_count=result.preprocessing.token_count,
            unique_token_count=result.preprocessing.unique_token_count,
            stopword_removed_count=result.preprocessing.stopword_removed_count,
            topic_probabilities=result.topic.probabilities,
            sentiment_probabilities=window_probabilities,
            entities=[entity.as_dict() for entity in result.entities],
            keyword_scores=[keyword.as_dict() for keyword in result.keywords],
            sentences=result.preprocessing.sentences[:80],
            model_versions=result.model_versions,
            processing_ms=round(result.processing_ms, 2),
        )
        db.add(article)
        db.flush()

        for message in window_messages:
            stored = by_message_id.get(message.id)
            if stored is not None:
                stored.news_id = article.id
                stored.window_index = window.index

        report.windows_created += 1

    stream.window_count = report.windows_created
    db.commit()
    return report


def analyse_all_streams(
    db: Session,
    pipeline: NLPPipeline | None = None,
    *,
    fit_idf: bool = True,
) -> IngestReport:
    """Analyse every collected stream. Returns a combined report."""
    pipeline = pipeline or get_pipeline()

    if fit_idf:
        fit_keyword_idf(db, pipeline)

    combined = IngestReport()
    for stream in db.scalars(select(ChatStream).order_by(ChatStream.id)):
        report = analyse_stream(db, stream, pipeline)
        combined.streams += report.streams
        combined.windows_created += report.windows_created
        combined.messages_scored += report.messages_scored
        combined.messages_skipped_noise += report.messages_skipped_noise
    return combined


def analyse_text(
    db: Session,
    *,
    title: str,
    content: str,
    source: str | None = None,
    url: str | None = None,
    published_at=None,
    pipeline: NLPPipeline | None = None,
) -> NewsArticle:
    """Analyse a single pasted article and store it.

    Backs the "Analyze News" page. Re-analysing identical text returns the
    existing row rather than creating a duplicate.
    """
    pipeline = pipeline or get_pipeline()
    result = pipeline.analyse(content, title=title)

    digest = _content_hash(result.preprocessing.cleaned_text)
    existing = db.scalar(select(NewsArticle).where(NewsArticle.content_hash == digest))
    if existing is not None:
        return existing

    article = NewsArticle(
        title=title.strip(),
        content=content,
        summary=result.summary.text if result.summary else "",
        topic=result.topic.label,
        topic_confidence=round(result.topic.confidence, 4),
        sentiment=result.sentiment.label,
        sentiment_confidence=round(result.sentiment.confidence, 4),
        keywords=[keyword.as_dict() for keyword in result.keywords],
        source=(source or "ผู้ใช้ป้อนข้อมูล").strip(),
        url=url,
        published_at=published_at or None,
        source_type=SourceType.ARTICLE.value,
        content_hash=digest,
    )
    if published_at is None:
        # Let the column default apply.
        article.published_at = None  # type: ignore[assignment]

    article.analysis = NLPAnalysis(
        cleaned_text=result.preprocessing.cleaned_text,
        tokens=result.preprocessing.tokens[:600],
        filtered_tokens=result.preprocessing.filtered_tokens[:600],
        token_count=result.preprocessing.token_count,
        unique_token_count=result.preprocessing.unique_token_count,
        stopword_removed_count=result.preprocessing.stopword_removed_count,
        topic_probabilities=result.topic.probabilities,
        sentiment_probabilities=result.sentiment.probabilities,
        entities=[entity.as_dict() for entity in result.entities],
        keyword_scores=[keyword.as_dict() for keyword in result.keywords],
        sentences=result.preprocessing.sentences[:80],
        model_versions=result.model_versions,
        processing_ms=round(result.processing_ms, 2),
    )

    db.add(article)
    db.commit()
    db.refresh(article)
    return article


__all__ = [
    "IngestReport",
    "analyse_all_streams",
    "analyse_stream",
    "analyse_text",
    "fit_keyword_idf",
    "settings",
]
