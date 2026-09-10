"""Keyword extraction for Thai documents.

Uses TF-IDF when corpus statistics are available and degrades to weighted term
frequency when they are not, so extraction works on the very first document
analysed as well as on a populated database.

Document frequencies are learned from the corpus by :meth:`fit` and cached to
``models/keyword_idf.json``, so the IDF term is real corpus statistics rather
than a guess.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from app.config import settings
from app.nlp.base import Keyword
from app.nlp.preprocessing import THAI_CHARS, filter_tokens

IDF_FILENAME = "keyword_idf.json"

# Terms shorter than this are rarely informative on their own in Thai.
MIN_TERM_LENGTH = 2

# Multiplier applied to terms that also appear in the document title.
TITLE_BOOST = 1.6


class TfidfKeywordExtractor:
    """Rank a document's terms by TF-IDF."""

    name = "tfidf"

    def __init__(
        self,
        document_frequencies: dict[str, int] | None = None,
        document_count: int = 0,
    ) -> None:
        self.document_frequencies = document_frequencies or {}
        self.document_count = document_count

    # -------------------------------------------------------------------- state
    @property
    def is_fitted(self) -> bool:
        """True when real corpus statistics are backing the IDF term."""
        return self.document_count > 0 and bool(self.document_frequencies)

    def fit(self, documents: list[list[str]]) -> TfidfKeywordExtractor:
        """Learn document frequencies from tokenised documents."""
        frequencies: Counter[str] = Counter()
        for tokens in documents:
            frequencies.update(set(filter_tokens(tokens)))
        self.document_frequencies = dict(frequencies)
        self.document_count = len(documents)
        return self

    def save(self, directory: Path | None = None) -> Path:
        """Persist the learned IDF statistics."""
        target = (directory or settings.model_dir) / IDF_FILENAME
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "document_count": self.document_count,
                    "document_frequencies": self.document_frequencies,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, directory: Path | None = None) -> TfidfKeywordExtractor:
        """Load cached IDF statistics, or return an unfitted extractor."""
        source = (directory or settings.model_dir) / IDF_FILENAME
        if not source.is_file():
            return cls()
        try:
            payload = json.loads(source.read_text(encoding="utf-8"))
            return cls(
                document_frequencies=payload.get("document_frequencies") or {},
                document_count=int(payload.get("document_count") or 0),
            )
        except (json.JSONDecodeError, ValueError, OSError):
            # A corrupt cache must not break analysis.
            return cls()

    # ------------------------------------------------------------------ scoring
    def _idf(self, term: str) -> float:
        """Inverse document frequency, or a length-based proxy when unfitted."""
        if not self.is_fitted:
            # Without corpus statistics, prefer longer (more specific) terms.
            return 1.0 + min(len(term), 12) / 12
        frequency = self.document_frequencies.get(term, 0)
        # Smoothed IDF: never zero, never division by zero.
        return math.log((1 + self.document_count) / (1 + frequency)) + 1.0

    def extract(
        self,
        text: str,
        tokens: list[str] | None = None,
        count: int | None = None,
        *,
        title: str | None = None,
    ) -> list[Keyword]:
        """Return the top keywords, highest score first.

        Args:
            text: The document text (used only if ``tokens`` is omitted).
            tokens: Pre-tokenised document, avoiding a second tokenisation pass.
            count: How many keywords to return.
            title: Optional headline; terms appearing in it are boosted, since a
                headline word is usually what the document is about.
        """
        limit = count or settings.keyword_max_count

        if tokens is None:
            from app.nlp.tokenizer import get_tokenizer

            tokens = get_tokenizer().tokenize(text or "")

        candidates = [
            token
            for token in filter_tokens(tokens)
            if len(token) >= MIN_TERM_LENGTH and THAI_CHARS.search(token)
        ]
        if not candidates:
            return []

        frequencies = Counter(candidates)
        total = sum(frequencies.values())

        title_terms: set[str] = set()
        if title:
            from app.nlp.tokenizer import get_tokenizer

            title_terms = set(filter_tokens(get_tokenizer().tokenize(title)))

        scored: list[Keyword] = []
        for term, occurrences in frequencies.items():
            term_frequency = occurrences / total
            score = term_frequency * self._idf(term)
            if term in title_terms:
                score *= TITLE_BOOST
            scored.append(Keyword(word=term, score=score))

        scored.sort(key=lambda keyword: (-keyword.score, keyword.word))
        return scored[:limit]
