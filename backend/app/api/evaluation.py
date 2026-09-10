"""Model evaluation endpoint.

Serves whatever ``evaluate.py`` wrote to ``models/metrics.json``. When that file
does not exist the endpoint says so explicitly rather than inventing figures --
the Evaluation page needs to be able to state honestly that no metrics have been
produced yet.
"""

from __future__ import annotations

import json

from fastapi import APIRouter

from app.config import settings
from app.taxonomy import SENTIMENT_BY_SLUG, TOPIC_BY_SLUG

router = APIRouter(tags=["evaluation"])

METRICS_FILENAME = "metrics.json"


@router.get("/evaluation", summary="Model evaluation metrics")
def evaluation() -> dict:
    """Return the stored evaluation metrics, with display labels attached."""
    path = settings.model_dir / METRICS_FILENAME

    if not path.is_file():
        return {
            "available": False,
            "reason": (
                "No metrics have been generated yet. Train the models and "
                "evaluate them first."
            ),
            "how_to": [
                "cd backend",
                "python build_dataset.py",
                "python train.py",
                "python evaluate.py",
            ],
        }

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {
            "available": False,
            "reason": f"Metrics file could not be read: {exc}",
            "how_to": ["cd backend", "python evaluate.py"],
        }

    # Attach Thai display labels so the UI does not have to join taxonomies.
    for task, entry in (payload.get("tasks") or {}).items():
        lookup = TOPIC_BY_SLUG if task == "topic" else SENTIMENT_BY_SLUG
        entry["label_names"] = {
            slug: {
                "thai": lookup[slug].thai if slug in lookup else slug,
                "english": lookup[slug].english if slug in lookup else slug,
                "color": lookup[slug].color if slug in lookup else "#64748B",
            }
            for slug in entry.get("labels", [])
        }

    payload["available"] = True
    return payload
