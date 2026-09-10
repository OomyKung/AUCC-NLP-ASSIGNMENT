"""Extractive summarisation for Thai text.

The default summarizer is fully offline, as the specification requires: the
application must work with no external LLM configured.

Method: split into candidate sentences, score each by TF-IDF centrality (how
much it shares with the rest of the document) plus a lead bias (Thai news, like
most news, front-loads the important facts), then return the highest-scoring
sentences **in their original order** so the summary reads naturally.

Chat windows are handled specially: each message is its own unit, so the summary
becomes the most representative messages rather than sentence fragments.
"""

from __future__ import annotations

import math
import re
from collections import Counter

from app.config import settings
from app.nlp.base import Summary
from app.nlp.preprocessing import filter_tokens

# Sentences shorter than this carry too little to be worth a summary slot.
MIN_SENTENCE_CHARS = 12

# Weight of the lead bias relative to centrality.
LEAD_WEIGHT = 0.35

_NEWLINES = re.compile(r"[\r\n]+")


class ExtractiveSummarizer:
    """Select the most representative sentences from a document."""

    name = "extractive"

    def __init__(self) -> None:
        from app.nlp.tokenizer import get_tokenizer

        self._tokenizer = get_tokenizer()

    # ---------------------------------------------------------------- sentences
    def candidates(self, text: str) -> list[str]:
        """Split ``text`` into summary candidates.

        Newlines are treated as hard boundaries first, because a chat window is
        newline-joined messages and a message is a complete unit that must not
        be merged with its neighbours.
        """
        if not text or not text.strip():
            return []

        units: list[str] = []
        for line in _NEWLINES.split(text):
            line = line.strip()
            if not line:
                continue
            # Within a line, fall back to Thai sentence segmentation.
            for sentence in self._tokenizer.sentences(line) or [line]:
                sentence = sentence.strip()
                if sentence:
                    units.append(sentence)
        return units

    # ------------------------------------------------------------------ scoring
    def _score(self, sentences: list[str]) -> list[float]:
        """Score sentences by TF-IDF centrality plus lead bias."""
        token_sets = [set(filter_tokens(self._tokenizer.tokenize(s))) for s in sentences]

        # Document frequency across sentences, used as a local IDF.
        document_frequency: Counter[str] = Counter()
        for tokens in token_sets:
            document_frequency.update(tokens)

        count = len(sentences)
        weights: list[dict[str, float]] = []
        for tokens in token_sets:
            weights.append(
                {
                    token: math.log((1 + count) / (1 + document_frequency[token])) + 1.0
                    for token in tokens
                }
            )

        scores: list[float] = []
        for index, tokens in enumerate(token_sets):
            if not tokens:
                scores.append(0.0)
                continue

            # Centrality: weighted overlap with every other sentence, normalised
            # by length so long sentences do not win automatically.
            centrality = 0.0
            for other_index, other in enumerate(token_sets):
                if other_index == index or not other:
                    continue
                shared = tokens & other
                if shared:
                    centrality += sum(weights[index][t] for t in shared) / math.sqrt(
                        len(tokens) * len(other)
                    )
            centrality /= max(count - 1, 1)

            # Lead bias: earlier sentences matter more.
            lead = 1.0 - (index / max(count, 1))
            scores.append(centrality + LEAD_WEIGHT * lead)

        return scores

    # ---------------------------------------------------------------- summarize
    def summarize(
        self,
        text: str,
        *,
        min_sentences: int | None = None,
        max_sentences: int | None = None,
    ) -> Summary:
        """Summarise ``text`` in roughly ``min``..``max`` sentences."""
        floor = min_sentences or settings.summary_min_sentences
        ceiling = max_sentences or settings.summary_max_sentences

        sentences = self.candidates(text)
        if not sentences:
            return Summary(text="", sentences=[], method=self.name)

        # Prefer substantial sentences, but keep short ones if that is all there is.
        substantial = [s for s in sentences if len(s) >= MIN_SENTENCE_CHARS]
        pool = substantial or sentences

        if len(pool) <= floor:
            return Summary(
                text=" ".join(pool), sentences=list(pool), method=self.name
            )

        scores = self._score(pool)
        wanted = max(floor, min(ceiling, len(pool)))

        # Take the highest scoring, then restore original order for readability.
        ranked = sorted(range(len(pool)), key=lambda i: -scores[i])[:wanted]
        chosen = [pool[i] for i in sorted(ranked)]

        return Summary(text=" ".join(chosen), sentences=chosen, method=self.name)


class LLMSummarizer:
    """Abstractive summarisation via an external LLM.

    Opt-in only: selected with ``NLP_SUMMARIZER_BACKEND=llm`` and requires
    ``LLM_API_KEY``. Falls back to the extractive summarizer on any failure, so
    a network problem degrades quality instead of breaking the request.
    """

    name = "llm"

    def __init__(self) -> None:
        self._fallback = ExtractiveSummarizer()

    def summarize(
        self,
        text: str,
        *,
        min_sentences: int | None = None,
        max_sentences: int | None = None,
    ) -> Summary:
        """Summarise via the configured LLM, falling back when unavailable."""
        if not settings.has_llm:
            return self._fallback.summarize(
                text, min_sentences=min_sentences, max_sentences=max_sentences
            )

        ceiling = max_sentences or settings.summary_max_sentences
        try:
            import httpx

            response = httpx.post(
                settings.llm_base_url,
                headers={
                    "x-api-key": settings.llm_api_key or "",
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": settings.llm_model,
                    "max_tokens": 512,
                    "messages": [
                        {
                            "role": "user",
                            "content": (
                                f"สรุปข่าวภาษาไทยต่อไปนี้ให้กระชับใน {ceiling} ประโยค "
                                f"โดยตอบเป็นภาษาไทยและไม่ต้องขึ้นต้นด้วยคำอธิบาย:\n\n{text}"
                            ),
                        }
                    ],
                },
                timeout=30.0,
            )
            response.raise_for_status()
            blocks = response.json().get("content") or []
            summary_text = "".join(
                block.get("text", "") for block in blocks if isinstance(block, dict)
            ).strip()
            if not summary_text:
                raise ValueError("empty completion")

            return Summary(
                text=summary_text,
                sentences=[s for s in summary_text.split(" ") if s][:ceiling],
                method=self.name,
            )
        except Exception:
            # Any failure (network, auth, quota, malformed response) degrades to
            # the offline summarizer rather than failing the request.
            return self._fallback.summarize(
                text, min_sentences=min_sentences, max_sentences=max_sentences
            )
