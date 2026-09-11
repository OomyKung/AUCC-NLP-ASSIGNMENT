"""Tests for the trained models, the registry's routing, and evaluation output."""

from __future__ import annotations

import json

import pytest

from app.config import settings
from app.nlp.backends.gazetteer_topic import GazetteerTopicBackend
from app.nlp.backends.lexicon_sentiment import LexiconSentimentBackend
from app.nlp.backends.sklearn_model import ARTEFACT_VERSION, load_model
from app.nlp.features import build_vectorizer, thai_word_analyzer
from app.nlp.registry import get_components, reset_components
from app.taxonomy import SENTIMENT_SLUGS, TOPIC_SLUGS

TRAINED = load_model("topic") is not None and load_model("sentiment") is not None
needs_models = pytest.mark.skipif(
    not TRAINED, reason="no trained artefacts; run python train.py"
)


# --------------------------------------------------------------------------
# Features
# --------------------------------------------------------------------------


def test_word_analyzer_produces_unigrams_and_bigrams():
    features = thai_word_analyzer("ทีมไทยเล่นไม่ดีเลยครับ")

    assert "ไทย" in features
    # Negation survives into the features, or sentiment would invert.
    assert "ไม่" in features
    # Filler is dropped.
    assert "ครับ" not in features
    # Bigrams are present and use the underscore join.
    assert any("_" in feature for feature in features)


def test_analyzer_is_picklable():
    """The analyser must be a module-level function.

    A lambda would let the model save but fail to load, which is the classic way
    this kind of pipeline breaks in production.
    """
    import pickle

    restored = pickle.loads(pickle.dumps(thai_word_analyzer))
    assert restored("ไทยชนะ") == thai_word_analyzer("ไทยชนะ")


def test_vectorizer_round_trips_through_joblib(tmp_path):
    import joblib

    documents = [
        "ทีมชาติไทยชนะการแข่งขันฟุตบอลนัดสำคัญ",
        "รถกระบะชนเสาไฟฟ้ามีผู้บาดเจ็บสามราย",
        "ตลาดหุ้นไทยปรับตัวเพิ่มขึ้นหลังนักลงทุนคลายกังวล",
        "กระทรวงสาธารณสุขเตือนประชาชนระวังโรคตามฤดูกาล",
    ]
    # min_df=1 because this is a four-document fixture.
    vectorizer = build_vectorizer(word_min_df=1, char_min_df=1)
    vectorizer.fit(documents)

    path = tmp_path / "vectorizer.pkl"
    joblib.dump(vectorizer, path)
    restored = joblib.load(path)

    assert restored.transform(["ทดสอบภาษาไทย"]).shape[1] == (
        vectorizer.transform(["ทดสอบภาษาไทย"]).shape[1]
    )


# --------------------------------------------------------------------------
# Trained artefacts
# --------------------------------------------------------------------------


def test_load_model_returns_none_when_missing(tmp_path):
    """A missing artefact must degrade, not raise: the registry relies on it."""
    assert load_model("topic", tmp_path) is None


def test_load_model_rejects_a_corrupt_artefact(tmp_path):
    (tmp_path / "topic_model.pkl").write_bytes(b"this is not a joblib file")
    assert load_model("topic", tmp_path) is None


def test_load_model_rejects_a_version_mismatch(tmp_path):
    import joblib

    joblib.dump(
        {"version": ARTEFACT_VERSION + 99, "pipeline": object()},
        tmp_path / "topic_model.pkl",
    )
    assert load_model("topic", tmp_path) is None


@needs_models
def test_trained_topic_model_returns_a_full_distribution():
    from app.nlp.backends.sklearn_topic import SklearnTopicBackend

    backend = SklearnTopicBackend.load()
    assert backend is not None
    assert backend.is_trained

    prediction = backend.predict(
        "รถกระบะเสียหลักพุ่งชนเสาไฟฟ้า มีผู้ได้รับบาดเจ็บ 3 ราย"
    )

    assert prediction.label in TOPIC_SLUGS
    # All 15 categories present, so the detail page's table is complete.
    assert set(prediction.probabilities) == set(TOPIC_SLUGS)
    assert abs(sum(prediction.probabilities.values()) - 1.0) < 1e-6
    assert 0.0 <= prediction.confidence <= 1.0


@needs_models
def test_trained_sentiment_model_batch_matches_single():
    from app.nlp.backends.sklearn_sentiment import SklearnSentimentBackend

    backend = SklearnSentimentBackend.load()
    assert backend is not None

    texts = [
        "ทีมชาติไทยคว้าชัยชนะอย่างงดงาม",
        "มีผู้เสียชีวิต 3 ราย จากอุบัติเหตุ",
        "",
    ]
    batch = backend.predict_many(texts)
    singles = [backend.predict(text) for text in texts]

    assert [item.label for item in batch] == [item.label for item in singles]
    assert set(batch[0].probabilities) == set(SENTIMENT_SLUGS)


@needs_models
def test_registry_prefers_trained_models_when_present():
    reset_components()
    try:
        components = get_components()
        assert components.status["topic"].trained is True
        assert components.status["sentiment"].trained is True
        # The active backend must be built on the fitted model, not the
        # rule-based cold-start path. Asserted by substring rather than prefix
        # so wrapping the model (as the blend does) does not break the test
        # while still failing if the gazetteer is serving alone.
        assert "sklearn:" in components.status["topic"].active
    finally:
        reset_components()


# --------------------------------------------------------------------------
# Domain routing
# --------------------------------------------------------------------------


def test_chat_sentiment_uses_its_own_backend():
    """Per-message chat scoring must not use the news-trained model.

    Measured reason: the news-trained model labelled 59% of real chat messages
    positive and called "แย่ที่สุด" ("the worst") positive. Cross-domain
    accuracy was later quantified in both directions -- the news model scores
    0.319 on the Wisesight test split, and a Wisesight-trained model scores
    0.287 on the news split -- so the two stages must stay on separate models.

    The assertion is about routing, not about which chat model wins: whether
    chat is served by the Wisesight model or by the lexicon fallback, it must
    not be the news-trained object.
    """
    reset_components()
    try:
        components = get_components()
        assert components.chat_sentiment is not components.sentiment
        assert components.status["chat_sentiment"].active != components.status[
            "sentiment"
        ].active
        # Never the news artefact.
        assert components.status["chat_sentiment"].active in {
            "lexicon",
            "sklearn:logreg-wisesight",
            "sklearn:svc-wisesight",
        }
    finally:
        reset_components()


def test_chat_lexicon_handles_informal_negatives():
    backend = LexiconSentimentBackend()
    for text in ["เล่นห่วยมาก", "กากจริง", "ไม่ไหวแล้ว", "แย่ที่สุด"]:
        assert backend.predict(text).label == "negative", text


def test_pipeline_sentiment_only_routes_to_chat_backend():
    from app.services.pipeline import get_pipeline

    reset_components()
    try:
        pipeline = get_pipeline()
        components = get_components()
        predictions = pipeline.sentiment_only(["แย่ที่สุด", "สุดยอดมาก"])
        assert [p.label for p in predictions] == ["negative", "positive"]
        # Routed through the chat backend, whichever one is configured -- the
        # point of the test is that news sentiment is not used for chat.
        assert all(
            p.model_name == components.chat_sentiment.name for p in predictions
        )
    finally:
        reset_components()


# --------------------------------------------------------------------------
# Baselines still work, so a fresh clone is functional
# --------------------------------------------------------------------------


def test_gazetteer_baseline_remains_usable():
    backend = GazetteerTopicBackend()
    assert backend.is_trained is False
    prediction = backend.predict("ทีมชาติไทยคว้าชัยในการแข่งขันฟุตบอล")
    assert prediction.label == "sports"
    assert set(prediction.probabilities) == set(TOPIC_SLUGS)


# --------------------------------------------------------------------------
# Evaluation endpoint
# --------------------------------------------------------------------------


def test_evaluation_endpoint_reports_unavailable_without_metrics(client, monkeypatch, tmp_path):
    """With no metrics file the endpoint must say so, never invent numbers."""
    monkeypatch.setattr(settings, "model_dir", tmp_path)
    body = client.get("/api/evaluation").json()

    assert body["available"] is False
    assert "reason" in body
    assert any("train.py" in step for step in body["how_to"])


@needs_models
def test_evaluation_endpoint_shape(client):
    body = client.get("/api/evaluation").json()
    if not body.get("available"):
        pytest.skip("metrics.json not generated; run python evaluate.py")

    assert set(body["tasks"]) == {"topic", "sentiment"}

    for task, entry in body["tasks"].items():
        expected = TOPIC_SLUGS if task == "topic" else SENTIMENT_SLUGS
        assert entry["labels"] == [s for s in expected if s in entry["labels"]]

        # The baseline is always present, so the trained model can be judged
        # against something rather than presented in isolation.
        assert "baseline" in entry["models"]

        for metrics in entry["models"].values():
            for key in (
                "accuracy",
                "precision_macro",
                "recall_macro",
                "f1_macro",
            ):
                assert 0.0 <= metrics[key] <= 1.0

            matrix = metrics["confusion_matrix"]
            assert len(matrix["matrix"]) == len(matrix["labels"])
            assert all(len(row) == len(matrix["labels"]) for row in matrix["matrix"])
            # Every held-out row appears exactly once in the matrix.
            assert sum(sum(row) for row in matrix["matrix"]) == metrics["support"]

        # Thai display labels are attached so the UI need not join taxonomies.
        assert set(entry["label_names"]) == set(entry["labels"])


@needs_models
def test_metrics_file_is_valid_json():
    path = settings.model_dir / "metrics.json"
    if not path.is_file():
        pytest.skip("metrics.json not generated")

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["split"]["random_state"] == 42
    assert payload["dataset"]["documents"] > 0
