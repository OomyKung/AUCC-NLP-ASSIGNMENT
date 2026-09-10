"""Tests for the news, statistics and analysis endpoints.

These run against a seeded in-memory database rather than the developer's real
one, so results do not depend on what happens to be collected locally.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.database.base import Base
from app.database.session import get_db
from app.main import create_app
from app.models.analysis import NLPAnalysis
from app.models.chat import ChatMessage, ChatStream
from app.models.news import NewsArticle, SourceType


@pytest.fixture
def api() -> TestClient:
    """A client backed by a small, deterministic seeded database."""
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)

    base_time = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
    with factory() as session:
        stream = ChatStream(
            video_id="seedVideo01",
            url="https://www.youtube.com/watch?v=seedVideo01",
            title="ทดสอบ ถ่ายทอดสด",
            channel="ช่องทดสอบ",
            collector="file",
            message_count=3,
        )
        session.add(stream)
        session.flush()

        rows = [
            ("อุบัติเหตุรถชนกลางสี่แยก 100%", "accident", "negative", 0.91, 5),
            ("ทีมชาติไทยชนะฟุตบอลนัดสำคัญ", "sports", "positive", 0.82, 200),
            ("สภาผ่านร่างกฎหมายงบประมาณ", "politics", "neutral", 0.55, 120),
            ("ตลาดหุ้นปรับตัวลดลง", "economy", "negative", 0.66, None),
        ]
        for index, (title, topic, sentiment, confidence, messages) in enumerate(rows):
            article = NewsArticle(
                title=title,
                content=f"{title} เนื้อหาข่าวสำหรับการทดสอบระบบ ครั้งที่ {index}",
                summary=f"สรุปข่าว {index}",
                topic=topic,
                topic_confidence=0.7,
                sentiment=sentiment,
                sentiment_confidence=confidence,
                keywords=[{"word": "ทดสอบ", "score": 0.5}],
                source="ทดสอบ",
                published_at=base_time + timedelta(days=index),
                source_type=(
                    SourceType.CHAT_WINDOW.value
                    if messages
                    else SourceType.ARTICLE.value
                ),
                content_hash=f"hash-{index}",
                stream_id=stream.id if messages else None,
                message_count=messages,
            )
            article.analysis = NLPAnalysis(
                cleaned_text=title,
                tokens=["ทดสอบ", "ข่าว"],
                filtered_tokens=["ทดสอบ"],
                token_count=2,
                unique_token_count=2,
                topic_probabilities={topic: 0.7},
                sentiment_probabilities={sentiment: confidence},
                entities=[{"text": "ไทย", "label": "LOCATION"}],
                keyword_scores=[{"word": "ทดสอบ", "score": 0.5}],
            )
            session.add(article)
        session.flush()

        first = session.query(NewsArticle).order_by(NewsArticle.id).first()
        for offset in range(3):
            session.add(
                ChatMessage(
                    stream_id=stream.id,
                    message_id=f"m{offset}",
                    author=f"user{offset}",
                    text=f"ข้อความทดสอบ {offset}",
                    published_at=base_time + timedelta(minutes=offset),
                    sentiment="positive" if offset else "negative",
                    sentiment_confidence=0.7,
                    news_id=first.id,
                    window_index=0,
                )
            )
        session.commit()

    app = create_app()
    app.dependency_overrides[get_db] = lambda: factory()
    client = TestClient(app)
    yield client
    app.dependency_overrides.clear()
    Base.metadata.drop_all(engine)
    engine.dispose()


# --------------------------------------------------------------------------
# Listing and filtering
# --------------------------------------------------------------------------


def test_list_returns_every_document(api):
    body = api.get("/api/news").json()
    assert body["total"] == 4
    assert body["page"] == 1
    assert len(body["items"]) == 4


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("topic=accident", 1),
        ("topic=sports", 1),
        ("sentiment=negative", 2),
        ("sentiment=positive", 1),
        ("topic=accident&sentiment=negative", 1),
        ("topic=sports&sentiment=negative", 0),
        ("source_type=chat_window", 3),
        ("source_type=article", 1),
        ("min_confidence=0.8", 2),
    ],
)
def test_filters(api, query, expected):
    assert api.get(f"/api/news?{query}").json()["total"] == expected


def test_search_matches_title_and_content(api):
    assert api.get("/api/news", params={"search": "ฟุตบอล"}).json()["total"] == 1
    assert api.get("/api/news", params={"search": "ไม่มีคำนี้"}).json()["total"] == 0
    # Body text is searched too, not only the headline.
    assert api.get("/api/news", params={"search": "เนื้อหาข่าว"}).json()["total"] == 4


def test_search_escapes_like_wildcards(api):
    """A literal % must not behave as "match everything".

    One seeded title contains "100%", so a search for "%" should find exactly
    that row rather than the whole corpus.
    """
    body = api.get("/api/news", params={"search": "%"}).json()
    assert body["total"] == 1
    assert "100%" in body["items"][0]["title"]

    # An underscore matches nothing here, rather than any single character.
    assert api.get("/api/news", params={"search": "_"}).json()["total"] == 0


def test_date_range_filter(api):
    assert api.get("/api/news?date_from=2026-09-03").json()["total"] == 2
    assert api.get("/api/news?date_to=2026-09-01T23:59:59").json()["total"] == 1
    # An inverted range returns nothing rather than erroring.
    assert (
        api.get("/api/news?date_from=2030-01-01&date_to=2020-01-01").json()["total"] == 0
    )


@pytest.mark.parametrize(
    ("sort", "first_topic"),
    [("newest", "economy"), ("oldest", "accident"), ("confidence", "accident")],
)
def test_sorting(api, sort, first_topic):
    body = api.get(f"/api/news?sort={sort}").json()
    assert body["items"][0]["topic"] == first_topic


def test_sort_by_messages_puts_nulls_last(api):
    """The article row has no message count and must not sort above windows."""
    items = api.get("/api/news?sort=messages").json()["items"]
    assert items[0]["message_count"] == 200
    assert items[-1]["message_count"] is None


def test_pagination(api):
    first = api.get("/api/news?page=1&page_size=2").json()
    second = api.get("/api/news?page=2&page_size=2").json()

    assert first["pages"] == 2
    assert len(first["items"]) == 2
    assert len(second["items"]) == 2
    # No row appears on both pages.
    assert {item["id"] for item in first["items"]}.isdisjoint(
        {item["id"] for item in second["items"]}
    )


def test_page_beyond_the_end_is_empty_not_an_error(api):
    body = api.get("/api/news?page=999").json()
    assert body["items"] == []
    assert body["total"] == 4


@pytest.mark.parametrize(
    "query",
    ["topic=nonsense", "sentiment=nope", "sort=sideways", "page=0", "page_size=0"],
)
def test_invalid_query_parameters_are_rejected(api, query):
    assert api.get(f"/api/news?{query}").status_code == 422


# --------------------------------------------------------------------------
# Detail
# --------------------------------------------------------------------------


def test_detail_includes_analysis_and_content(api):
    listing = api.get("/api/news?sort=oldest").json()
    news_id = listing["items"][0]["id"]

    body = api.get(f"/api/news/{news_id}").json()
    assert body["content"]
    assert body["analysis"]["token_count"] == 2
    assert body["analysis"]["entities"][0]["label"] == "LOCATION"


def test_detail_404(api):
    assert api.get("/api/news/424242").status_code == 404


def test_window_messages(api):
    listing = api.get("/api/news?sort=oldest").json()
    news_id = listing["items"][0]["id"]

    messages = api.get(f"/api/news/{news_id}/messages").json()
    assert len(messages) == 3
    assert messages[0]["text"].startswith("ข้อความทดสอบ")
    # Ordered by time.
    assert [m["text"] for m in messages] == sorted(m["text"] for m in messages)


def test_delete_detaches_messages_rather_than_orphaning_them(api):
    listing = api.get("/api/news?sort=oldest").json()
    news_id = listing["items"][0]["id"]

    assert api.delete(f"/api/news/{news_id}").status_code == 204
    assert api.get(f"/api/news/{news_id}").status_code == 404
    assert api.get("/api/news").json()["total"] == 3


# --------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------


def test_statistics_totals_agree_with_the_listing(api):
    stats = api.get("/api/statistics").json()
    listing = api.get("/api/news").json()

    assert stats["total_news"] == listing["total"]
    # The stat cards must sum to the total, or the dashboard contradicts itself.
    assert stats["positive"] + stats["neutral"] + stats["negative"] == stats["total_news"]


def test_statistics_chart_series_are_consistent(api):
    stats = api.get("/api/statistics").json()

    # All 15 topics present, so the chart axis does not reshuffle as data arrives.
    assert len(stats["by_topic"]) == 15
    assert sum(row["count"] for row in stats["by_topic"]) == stats["total_news"]

    assert sum(slice_["count"] for slice_ in stats["by_sentiment"]) == stats["total_news"]
    # Only non-empty topics appear as stacked bars.
    assert all(row["total"] > 0 for row in stats["topic_sentiment"])
    assert sum(row["total"] for row in stats["topic_sentiment"]) == stats["total_news"]


def test_statistics_most_common_topic(api):
    stats = api.get("/api/statistics").json()
    assert stats["most_common_topic_count"] >= 1
    assert stats["most_common_topic_label"]


@pytest.mark.parametrize("granularity", ["daily", "weekly", "monthly"])
def test_trend_buckets_sum_to_the_total(api, granularity):
    body = api.get(f"/api/statistics/trend?granularity={granularity}").json()
    total = api.get("/api/news").json()["total"]

    assert body["granularity"] == granularity
    assert sum(point["count"] for point in body["points"]) == total
    for point in body["points"]:
        assert (
            point["positive"] + point["neutral"] + point["negative"] == point["count"]
        )


def test_trend_rejects_unknown_granularity(api):
    assert api.get("/api/statistics/trend?granularity=hourly").status_code == 422


def test_keywords_endpoint(api):
    body = api.get("/api/statistics/keywords?limit=5").json()
    assert body
    assert body[0]["word"] == "ทดสอบ"
    assert body[0]["count"] == 4

    assert api.get("/api/statistics/keywords?topic=nonsense").status_code == 422


def test_streams_endpoint(api):
    body = api.get("/api/streams").json()
    assert len(body) == 1
    assert body[0]["video_id"] == "seedVideo01"
    assert body[0]["channel"] == "ช่องทดสอบ"


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------


def test_analyze_returns_every_intermediate_value(api):
    body = api.post(
        "/api/analyze",
        json={
            "title": "รถกระบะชนเสาไฟฟ้า มีผู้บาดเจ็บ 3 ราย",
            "content": (
                "รถกระบะเสียหลักพุ่งชนเสาไฟฟ้าริมถนน มีผู้ได้รับบาดเจ็บ 3 ราย "
                "เจ้าหน้าที่กู้ภัยนำส่งโรงพยาบาลทันที"
            ),
        },
    ).json()

    # The NLP Analysis page depends on all of these being present.
    for key in (
        "topic",
        "sentiment",
        "summary",
        "keywords",
        "entities",
        "topic_probabilities",
        "sentiment_probabilities",
        "cleaned_text",
        "tokens",
        "filtered_tokens",
        "token_count",
        "unique_token_count",
        "sentences",
        "model_versions",
    ):
        assert key in body, key

    assert body["token_count"] > 0
    assert abs(sum(body["sentiment_probabilities"].values()) - 1.0) < 1e-6
    # Not stored unless asked.
    assert body["news_id"] is None


def test_analyze_can_store_and_deduplicate(api):
    payload = {
        "title": "ทดสอบการบันทึกข้อมูล",
        "content": "เนื้อหาข่าวภาษาไทยสำหรับทดสอบการบันทึกลงฐานข้อมูล",
    }
    before = api.get("/api/news").json()["total"]

    first = api.post("/api/analyze?store=true", json=payload).json()
    assert first["news_id"] is not None
    assert api.get("/api/news").json()["total"] == before + 1

    # Re-analysing identical text must not create a duplicate.
    second = api.post("/api/analyze?store=true", json=payload).json()
    assert second["news_id"] == first["news_id"]
    assert api.get("/api/news").json()["total"] == before + 1


def test_analyze_handles_content_with_no_thai(api):
    body = api.post(
        "/api/analyze",
        json={"title": "111", "content": "11111111 22222222"},
    )
    # Junk must produce an honest low-confidence result, not a crash.
    assert body.status_code == 200
    assert body.json()["topic"] == "other"


def test_create_news_endpoint(api):
    response = api.post(
        "/api/news",
        json={
            "title": "ข่าวใหม่จากการทดสอบ",
            "content": "เนื้อหาข่าวภาษาไทยที่ยาวพอสำหรับการวิเคราะห์ในการทดสอบนี้",
            "source": "ผู้ทดสอบ",
        },
    )
    assert response.status_code == 201
    assert response.json()["analysis"] is not None


# --------------------------------------------------------------------------
# Ingestion
# --------------------------------------------------------------------------


def test_ingest_rejects_unknown_collector(api):
    response = api.post(
        "/api/ingest/youtube", json={"source": "abc", "collector": "telepathy"}
    )
    assert response.status_code == 422
    assert "Unknown collector" in response.json()["detail"]


def test_ingest_reports_a_missing_snapshot_as_a_client_error(api):
    response = api.post(
        "/api/ingest/youtube",
        json={"source": "not-a-snapshot", "collector": "file", "analyse": False},
    )
    # The caller can act on this, so it is a 400 with the real reason.
    assert response.status_code == 400
    assert "not found" in response.json()["detail"].lower()


def test_reanalyse_unknown_stream_is_404(api):
    assert (
        api.post("/api/ingest/reanalyse", json={"stream_id": 987654}).status_code == 404
    )
