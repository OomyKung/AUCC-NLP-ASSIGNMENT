"""Tests for label normalisation, the guard against label drift."""

from __future__ import annotations

import pytest

from app.taxonomy import (
    FALLBACK_SENTIMENT,
    FALLBACK_TOPIC,
    TOPICS,
    normalise_sentiment,
    normalise_topic,
    topic_label,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("อุบัติเหตุ", "accident"),  # Thai label
        ("Accident", "accident"),  # English label
        ("ACCIDENT", "accident"),  # any case
        ("  sports  ", "sports"),  # padded slug
        ("ต่างประเทศ", "international"),
        ("not-a-topic", FALLBACK_TOPIC),
        (None, FALLBACK_TOPIC),
        ("", FALLBACK_TOPIC),
    ],
)
def test_normalise_topic(value, expected):
    assert normalise_topic(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("positive", "positive"),
        ("เชิงลบ", "negative"),
        ("NEG", "negative"),
        ("pos", "positive"),
        ("neu", "neutral"),
        ("nonsense", FALLBACK_SENTIMENT),
        (None, FALLBACK_SENTIMENT),
    ],
)
def test_normalise_sentiment(value, expected):
    assert normalise_sentiment(value) == expected


def test_topic_slugs_are_unique_and_have_distinct_colors():
    slugs = [t.slug for t in TOPICS]
    colors = [t.color for t in TOPICS]
    assert len(set(slugs)) == len(slugs)
    assert len(set(colors)) == len(colors), "chart colours must be distinguishable"


def test_topic_label_falls_back_to_slug_for_unknown_input():
    assert topic_label("nope") == "nope"
    assert topic_label("crime") == "อาชญากรรม"
    assert topic_label("crime", "en") == "Crime"
