"""Single source of truth for the topic and sentiment label sets.

Every layer (NLP models, API schemas, seeds, frontend types) derives its labels
from here so a label can never drift between the model and the UI.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Topic:
    """One news category."""

    slug: str
    thai: str
    english: str
    color: str  # tailwind-friendly hex used by charts and badges


# Order is meaningful: it is the display order in filters and charts.
TOPICS: tuple[Topic, ...] = (
    Topic("accident", "อุบัติเหตุ", "Accident", "#F97316"),
    Topic("crime", "อาชญากรรม", "Crime", "#DC2626"),
    Topic("politics", "การเมือง", "Politics", "#7C3AED"),
    Topic("economy", "เศรษฐกิจ", "Economy", "#0891B2"),
    Topic("technology", "เทคโนโลยี", "Technology", "#2563EB"),
    Topic("sports", "กีฬา", "Sports", "#16A34A"),
    Topic("health", "สุขภาพ", "Health", "#0D9488"),
    Topic("society", "สังคม", "Society", "#D97706"),
    Topic("disaster", "ภัยพิบัติ", "Disaster", "#B91C1C"),
    Topic("environment", "สิ่งแวดล้อม", "Environment", "#65A30D"),
    Topic("education", "การศึกษา", "Education", "#4F46E5"),
    Topic("entertainment", "บันเทิง", "Entertainment", "#DB2777"),
    Topic("international", "ต่างประเทศ", "International", "#0369A1"),
    Topic("business", "ธุรกิจ", "Business", "#059669"),
    Topic("other", "อื่นๆ", "Other", "#64748B"),
)

TOPIC_SLUGS: tuple[str, ...] = tuple(t.slug for t in TOPICS)
TOPIC_BY_SLUG: dict[str, Topic] = {t.slug: t for t in TOPICS}
FALLBACK_TOPIC = "other"


@dataclass(frozen=True, slots=True)
class Sentiment:
    """One sentiment class."""

    slug: str
    thai: str
    english: str
    color: str


SENTIMENTS: tuple[Sentiment, ...] = (
    Sentiment("positive", "เชิงบวก", "Positive", "#16A34A"),
    Sentiment("neutral", "เป็นกลาง", "Neutral", "#64748B"),
    Sentiment("negative", "เชิงลบ", "Negative", "#DC2626"),
)

SENTIMENT_SLUGS: tuple[str, ...] = tuple(s.slug for s in SENTIMENTS)
SENTIMENT_BY_SLUG: dict[str, Sentiment] = {s.slug: s for s in SENTIMENTS}
FALLBACK_SENTIMENT = "neutral"


def topic_label(slug: str, lang: str = "thai") -> str:
    """Human label for a topic slug; unknown slugs degrade to the slug itself."""
    topic = TOPIC_BY_SLUG.get(slug)
    if topic is None:
        return slug
    return topic.thai if lang == "thai" else topic.english


def sentiment_label(slug: str, lang: str = "thai") -> str:
    """Human label for a sentiment slug."""
    sentiment = SENTIMENT_BY_SLUG.get(slug)
    if sentiment is None:
        return slug
    return sentiment.thai if lang == "thai" else sentiment.english


def normalise_topic(value: str | None) -> str:
    """Coerce arbitrary input (Thai or English, any case) to a valid topic slug."""
    if not value:
        return FALLBACK_TOPIC
    needle = value.strip().lower()
    for topic in TOPICS:
        if needle in {topic.slug, topic.english.lower(), topic.thai}:
            return topic.slug
    return FALLBACK_TOPIC


def normalise_sentiment(value: str | None) -> str:
    """Coerce arbitrary input to a valid sentiment slug."""
    if not value:
        return FALLBACK_SENTIMENT
    needle = value.strip().lower()
    for sentiment in SENTIMENTS:
        if needle in {sentiment.slug, sentiment.english.lower(), sentiment.thai}:
            return sentiment.slug
    # Common dataset spellings.
    return {"pos": "positive", "neg": "negative", "neu": "neutral"}.get(
        needle, FALLBACK_SENTIMENT
    )
