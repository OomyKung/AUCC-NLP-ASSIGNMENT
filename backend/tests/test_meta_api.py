"""Tests for the health and reference-data endpoints."""

from __future__ import annotations

from app.taxonomy import SENTIMENT_SLUGS, TOPIC_SLUGS


def test_health_reports_ok(client):
    response = client.get("/api/health")
    assert response.status_code == 200

    body = response.json()
    assert body["status"] == "ok"
    # The active NLP backends must be reported so misconfiguration is visible.
    assert set(body["nlp_backends"]) == {
        "tokenizer",
        "topic",
        "sentiment",
        "summarizer",
        "ner",
    }


def test_topics_endpoint_returns_all_fifteen_categories(client):
    response = client.get("/api/topics")
    assert response.status_code == 200

    body = response.json()
    assert len(body) == 15
    assert [t["slug"] for t in body] == list(TOPIC_SLUGS)
    # Thai labels must survive JSON encoding intact.
    assert body[0]["thai"] == "อุบัติเหตุ"
    assert all(t["color"].startswith("#") for t in body)


def test_sentiments_endpoint_returns_three_classes(client):
    response = client.get("/api/sentiments")
    assert response.status_code == 200

    body = response.json()
    assert [s["slug"] for s in body] == list(SENTIMENT_SLUGS)


def test_root_points_at_docs(client):
    body = client.get("/").json()
    assert body["docs"] == "/docs"
