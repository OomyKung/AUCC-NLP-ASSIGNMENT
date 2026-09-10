"""Per-document NLP analysis detail.

Split from ``NewsArticle`` because these columns are large (token lists,
probability distributions) and are only needed on the detail and NLP-analysis
pages, not in list views.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import JSON, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.base import Base, UtcDateTime, utcnow

if TYPE_CHECKING:
    from app.models.news import NewsArticle


class NLPAnalysis(Base):
    """The explainable intermediate output of one pipeline run."""

    __tablename__ = "nlp_analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    news_id: Mapped[int] = mapped_column(
        ForeignKey("news.id", ondelete="CASCADE"), unique=True, index=True
    )

    # ------------------------------------------- preprocessing intermediates
    cleaned_text: Mapped[str | None] = mapped_column(Text)
    tokens: Mapped[list] = mapped_column(JSON, default=list)
    # Tokens surviving stopword/punctuation removal -- the model's real input.
    filtered_tokens: Mapped[list] = mapped_column(JSON, default=list)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    unique_token_count: Mapped[int] = mapped_column(Integer, default=0)
    stopword_removed_count: Mapped[int] = mapped_column(Integer, default=0)

    # --------------------------------------------- model output distributions
    # Probability over all 15 topics, e.g. {"crime": 0.82, "society": 0.09}.
    topic_probabilities: Mapped[dict] = mapped_column(JSON, default=dict)
    # e.g. {"negative": 0.91, "neutral": 0.07, "positive": 0.02}
    sentiment_probabilities: Mapped[dict] = mapped_column(JSON, default=dict)

    # e.g. [{"text": "กรุงเทพมหานคร", "label": "LOCATION"}]
    entities: Mapped[list] = mapped_column(JSON, default=list)
    # The full scored keyword ranking, e.g. [{"word": ..., "score": ...}].
    keyword_scores: Mapped[list] = mapped_column(JSON, default=list)
    sentences: Mapped[list] = mapped_column(JSON, default=list)

    # ---------------------------------------------------- reproducibility
    # Which backend produced each field, so results stay traceable after a swap.
    model_versions: Mapped[dict] = mapped_column(JSON, default=dict)
    pipeline_version: Mapped[str] = mapped_column(String(20), default="1.0.0")
    processing_ms: Mapped[float] = mapped_column(Float, default=0.0)

    created_at: Mapped[datetime] = mapped_column(UtcDateTime, default=utcnow)

    article: Mapped[NewsArticle] = relationship(back_populates="analysis")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<NLPAnalysis news_id={self.news_id} tokens={self.token_count}>"
