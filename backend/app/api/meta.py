"""Health and reference-data endpoints.

These need no database, so they double as the liveness probe used by the
frontend to confirm the API is reachable.
"""

from __future__ import annotations

import sys

from fastapi import APIRouter

from app.config import settings
from app.taxonomy import SENTIMENTS, TOPICS

router = APIRouter(tags=["meta"])


@router.get("/health", summary="Liveness probe")
def health() -> dict:
    """Report service health plus the versions that matter for debugging."""
    return {
        "status": "ok",
        "app": settings.app_name,
        "version": settings.app_version,
        "python": sys.version.split()[0],
        "nlp_backends": {
            "tokenizer": settings.nlp_tokenizer,
            "topic": settings.nlp_topic_backend,
            "sentiment": settings.nlp_sentiment_backend,
            "summarizer": settings.nlp_summarizer_backend,
            "ner": settings.nlp_ner_backend,
        },
        "llm_summarizer_configured": settings.has_llm,
    }


@router.get("/topics", summary="The 15 news categories")
def list_topics() -> list[dict]:
    """Return the topic taxonomy so the frontend never hard-codes labels."""
    return [
        {"slug": t.slug, "thai": t.thai, "english": t.english, "color": t.color}
        for t in TOPICS
    ]


@router.get("/sentiments", summary="The 3 sentiment classes")
def list_sentiments() -> list[dict]:
    """Return the sentiment taxonomy with display colours."""
    return [
        {
            "slug": s.slug,
            "thai": s.thai,
            "english": s.english,
            "color": s.color,
            "color_dark": s.color_dark,
        }
        for s in SENTIMENTS
    ]
