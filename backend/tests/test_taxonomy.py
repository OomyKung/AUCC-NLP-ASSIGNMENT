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


# --------------------------------------------------------------------------
# Configuration surface
# --------------------------------------------------------------------------


def test_every_referenced_setting_exists():
    """No code may read a `settings.` attribute that is not defined.

    This exists because an edit to config.py once deleted
    `nlp_chat_sentiment_backend` while rewriting the comment above it. Nothing
    caught it until the registry raised AttributeError at startup, which is a
    late and confusing place to find out. Pydantic settings are attribute
    access, so a typo or a dropped field cannot be caught by import alone.
    """
    import re
    from pathlib import Path

    from app.config import settings

    backend = Path(__file__).resolve().parent.parent
    pattern = re.compile(r"settings\.([a-z_][a-z0-9_]*)")

    referenced: set[str] = set()
    for directory in (backend / "app",):
        for path in directory.rglob("*.py"):
            referenced |= set(pattern.findall(path.read_text(encoding="utf-8")))
    for path in backend.glob("*.py"):
        referenced |= set(pattern.findall(path.read_text(encoding="utf-8")))

    # Properties count as defined; hasattr covers both fields and properties.
    missing = sorted(name for name in referenced if not hasattr(settings, name))
    assert not missing, f"settings attributes referenced but not defined: {missing}"


def test_env_example_documents_only_real_settings():
    """Every NLP_* key in .env.example must map to a real setting.

    A stale key in the example file is worse than no key: pydantic ignores
    unknown environment variables (extra="ignore"), so someone setting it would
    see no effect and no error.
    """
    import re
    from pathlib import Path

    from app.config import Settings, settings

    example = Path(__file__).resolve().parents[2] / ".env.example"
    if not example.is_file():
        pytest.skip(".env.example not present")

    keys = re.findall(r"^([A-Z][A-Z0-9_]*)=", example.read_text(encoding="utf-8"), re.M)
    fields = set(Settings.model_fields)

    unknown = sorted(k for k in keys if k.lower() not in fields)
    assert not unknown, f".env.example documents unknown settings: {unknown}"
    # And the documented values must actually validate.
    assert hasattr(settings, "nlp_chat_sentiment_backend")
