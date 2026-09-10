"""Dashboard aggregation.

Every figure is computed in SQL over the stored documents, so the cards, the
charts and the Explorer can never disagree with each other.
"""

from __future__ import annotations

from collections import Counter

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.chat import ChatMessage, ChatStream
from app.models.news import NewsArticle, SourceType
from app.schemas.statistics import (
    KeywordCount,
    LabelCount,
    SentimentSlice,
    StatisticsResponse,
    TopicSentimentRow,
    TrendPoint,
    TrendResponse,
)
from app.taxonomy import (
    SENTIMENT_BY_SLUG,
    SENTIMENT_SLUGS,
    TOPIC_BY_SLUG,
    TOPICS,
)

# SQLite strftime patterns per granularity.
_TREND_FORMATS = {
    "daily": "%Y-%m-%d",
    "weekly": "%Y-W%W",
    "monthly": "%Y-%m",
}


def build_statistics(db: Session, *, keyword_limit: int = 40) -> StatisticsResponse:
    """Compute every dashboard figure in one pass."""
    total = db.scalar(select(func.count()).select_from(NewsArticle)) or 0

    sentiment_counts = dict(
        db.execute(
            select(NewsArticle.sentiment, func.count()).group_by(NewsArticle.sentiment)
        ).all()
    )
    topic_counts = dict(
        db.execute(
            select(NewsArticle.topic, func.count()).group_by(NewsArticle.topic)
        ).all()
    )

    # Topic bars keep taxonomy order and include zero-count categories, so the
    # chart does not reshuffle as data arrives.
    by_topic = [
        LabelCount(
            key=topic.slug,
            label=topic.thai,
            label_en=topic.english,
            color=topic.color,
            count=int(topic_counts.get(topic.slug, 0)),
        )
        for topic in TOPICS
    ]

    by_sentiment = []
    for slug in SENTIMENT_SLUGS:
        info = SENTIMENT_BY_SLUG[slug]
        count = int(sentiment_counts.get(slug, 0))
        by_sentiment.append(
            SentimentSlice(
                key=slug,
                label=info.thai,
                label_en=info.english,
                color=info.color,
                count=count,
                percentage=round(count / total * 100, 1) if total else 0.0,
            )
        )

    # Topic x sentiment: the chart that shows the two classifiers relating.
    cross: dict[str, dict[str, int]] = {}
    for topic, sentiment, count in db.execute(
        select(NewsArticle.topic, NewsArticle.sentiment, func.count()).group_by(
            NewsArticle.topic, NewsArticle.sentiment
        )
    ):
        cross.setdefault(topic, {})[sentiment] = int(count)

    topic_sentiment = []
    for topic in TOPICS:
        row = cross.get(topic.slug, {})
        subtotal = sum(row.values())
        if subtotal == 0:
            continue  # an empty stacked bar is noise
        topic_sentiment.append(
            TopicSentimentRow(
                topic=topic.slug,
                label=topic.thai,
                label_en=topic.english,
                color=topic.color,
                positive=row.get("positive", 0),
                neutral=row.get("neutral", 0),
                negative=row.get("negative", 0),
                total=subtotal,
            )
        )
    topic_sentiment.sort(key=lambda item: -item.total)

    most_common_topic = None
    most_common_count = 0
    if topic_counts:
        most_common_topic, most_common_count = max(
            topic_counts.items(), key=lambda kv: kv[1]
        )
        most_common_count = int(most_common_count)

    return StatisticsResponse(
        total_news=total,
        positive=int(sentiment_counts.get("positive", 0)),
        neutral=int(sentiment_counts.get("neutral", 0)),
        negative=int(sentiment_counts.get("negative", 0)),
        most_common_topic=most_common_topic,
        most_common_topic_label=(
            TOPIC_BY_SLUG[most_common_topic].thai if most_common_topic else None
        ),
        most_common_topic_count=most_common_count,
        total_messages=db.scalar(select(func.count()).select_from(ChatMessage)) or 0,
        scored_messages=db.scalar(
            select(func.count())
            .select_from(ChatMessage)
            .where(ChatMessage.sentiment.isnot(None))
        )
        or 0,
        noise_messages=db.scalar(
            select(func.count()).select_from(ChatMessage).where(ChatMessage.is_spam)
        )
        or 0,
        total_streams=db.scalar(select(func.count()).select_from(ChatStream)) or 0,
        chat_windows=db.scalar(
            select(func.count())
            .select_from(NewsArticle)
            .where(NewsArticle.source_type == SourceType.CHAT_WINDOW.value)
        )
        or 0,
        articles=db.scalar(
            select(func.count())
            .select_from(NewsArticle)
            .where(NewsArticle.source_type != SourceType.CHAT_WINDOW.value)
        )
        or 0,
        by_topic=by_topic,
        by_sentiment=by_sentiment,
        topic_sentiment=topic_sentiment,
        top_keywords=top_keywords(db, limit=keyword_limit),
        avg_topic_confidence=round(
            float(db.scalar(select(func.avg(NewsArticle.topic_confidence))) or 0.0), 4
        ),
        avg_sentiment_confidence=round(
            float(db.scalar(select(func.avg(NewsArticle.sentiment_confidence))) or 0.0),
            4,
        ),
    )


def top_keywords(
    db: Session, *, limit: int = 40, topic: str | None = None
) -> list[KeywordCount]:
    """Aggregate keywords across documents for the word cloud.

    Keywords live in a JSON column, so this aggregates in Python. That is
    acceptable at this corpus size and keeps the storage model simple; a
    dedicated keyword table would be the move if the corpus grew large.
    """
    query = select(NewsArticle.keywords)
    if topic:
        query = query.where(NewsArticle.topic == topic)

    counts: Counter[str] = Counter()
    scores: dict[str, float] = {}

    for (payload,) in db.execute(query):
        for item in payload or []:
            if isinstance(item, dict):
                word = str(item.get("word") or "").strip()
                score = float(item.get("score") or 0.0)
            else:
                word, score = str(item).strip(), 0.0
            if not word:
                continue
            counts[word] += 1
            scores[word] = max(scores.get(word, 0.0), score)

    return [
        KeywordCount(word=word, count=count, score=round(scores.get(word, 0.0), 4))
        for word, count in counts.most_common(limit)
    ]


def build_trend(db: Session, granularity: str = "daily") -> TrendResponse:
    """Document volume over time, split by sentiment.

    Bucketing is done in SQL with ``strftime`` so the aggregation is real rather
    than computed over a page of rows.
    """
    fmt = _TREND_FORMATS.get(granularity, _TREND_FORMATS["daily"])
    bucket = func.strftime(fmt, NewsArticle.published_at)

    rows = db.execute(
        select(bucket, NewsArticle.sentiment, func.count())
        .group_by(bucket, NewsArticle.sentiment)
        .order_by(bucket)
    ).all()

    buckets: dict[str, TrendPoint] = {}
    for period, sentiment, count in rows:
        if period is None:
            continue
        point = buckets.setdefault(period, TrendPoint(period=period, count=0))
        point.count += int(count)
        if sentiment == "positive":
            point.positive += int(count)
        elif sentiment == "negative":
            point.negative += int(count)
        else:
            point.neutral += int(count)

    return TrendResponse(
        granularity=granularity,  # type: ignore[arg-type]
        points=[buckets[key] for key in sorted(buckets)],
    )
