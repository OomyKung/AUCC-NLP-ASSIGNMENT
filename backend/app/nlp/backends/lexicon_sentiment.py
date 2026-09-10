"""Lexicon-based Thai sentiment analysis.

Real, explainable inference that requires no trained model, so the application
produces genuine sentiment on a fresh clone. It also stays useful after
training as an interpretable comparison for the Evaluation page.

Negation is handled explicitly rather than ignored: ``ไม่ดี`` scores negative
even though ``ดี`` alone is positive. This is why negators are protected from
stopword removal in :mod:`app.nlp.stopwords`.
"""

from __future__ import annotations

import math

from app.nlp.base import Prediction
from app.nlp.lexicon import (
    DIMINISHERS,
    INTENSIFIERS,
    NEGATION_WINDOW,
    NEGATORS,
    lexicon_size,
    match_terms,
)
from app.nlp.tokenizer import get_tokenizer
from app.taxonomy import SENTIMENT_SLUGS

# Total polarity below this magnitude is reported as neutral. Tuned so that a
# single mild word does not swing a whole document.
NEUTRAL_BAND = 0.35

# Controls how quickly confidence saturates as evidence accumulates.
_CONFIDENCE_SCALE = 1.6


class LexiconSentimentBackend:
    """Score sentiment by summing lexicon weights with negation handling."""

    name = "lexicon"

    def __init__(self, neutral_band: float = NEUTRAL_BAND) -> None:
        self.neutral_band = neutral_band
        self._tokenizer = get_tokenizer()

    @property
    def is_trained(self) -> bool:
        """False: this is a rule-based baseline, not a fitted model."""
        return False

    # ------------------------------------------------------------------ scoring
    def score(self, tokens: list[str]) -> tuple[float, list[dict]]:
        """Return the total polarity and the per-word evidence behind it.

        The evidence list is what makes the result explainable in the UI: it
        records which words contributed, and whether a negator or intensifier
        modified them.
        """
        # Sentiment terms, found with phrase and compound awareness so that
        # multi-token entries and Thai compounds (ผู้เสียชีวิต) are not missed.
        term_at: dict[int, tuple[int, str, float]] = {
            start: (span, term, weight)
            for start, span, term, weight in match_terms(tokens)
        }

        total = 0.0
        evidence: list[dict] = []
        negate_until = -1
        pending_multiplier = 1.0

        index = 0
        while index < len(tokens):
            token = tokens[index]
            found = term_at.get(index)

            # A lexicon phrase is checked before the negator branch, because
            # some entries begin with a negator ("ไม่โอเค", "ไม่ไหว",
            # "รับไม่ได้"). Treating the leading ไม่ as a bare negator would
            # consume it and lose the phrase entirely.
            if found is None:
                if token in NEGATORS:
                    negate_until = index + NEGATION_WINDOW
                    index += 1
                    continue

                if token in INTENSIFIERS:
                    pending_multiplier *= INTENSIFIERS[token]
                    index += 1
                    continue
                if token in DIMINISHERS:
                    pending_multiplier *= DIMINISHERS[token]
                    index += 1
                    continue

                index += 1
                continue

            span, term, base = found

            # An intensifier may also follow the word it modifies, which is the
            # usual order in Thai: "ดี มาก".
            following = index + span
            if following < len(tokens):
                next_token = tokens[following]
                if next_token in INTENSIFIERS:
                    pending_multiplier *= INTENSIFIERS[next_token]
                elif next_token in DIMINISHERS:
                    pending_multiplier *= DIMINISHERS[next_token]

            negated = index <= negate_until
            value = base * pending_multiplier
            if negated:
                # Negation flips polarity but softens magnitude: "ไม่ดี" is
                # negative, though less emphatically than "แย่มาก".
                value = -value * 0.8

            total += value
            evidence.append(
                {
                    "word": term,
                    "matched": "".join(tokens[index : index + span]),
                    "base": round(base, 3),
                    "score": round(value, 3),
                    "negated": negated,
                    "multiplier": round(pending_multiplier, 2),
                }
            )
            pending_multiplier = 1.0
            index += span

        return total, evidence

    def _distribution(self, total: float) -> dict[str, float]:
        """Turn a raw polarity sum into a probability distribution.

        Uses a softmax over three pseudo-logits so the result is a genuine
        normalised distribution rather than an ad-hoc rescaling.
        """
        magnitude = abs(total)
        logits = {
            "positive": total * _CONFIDENCE_SCALE,
            "negative": -total * _CONFIDENCE_SCALE,
            # Neutral competes on how *little* evidence there is.
            "neutral": (self.neutral_band - magnitude) * _CONFIDENCE_SCALE * 1.4,
        }
        largest = max(logits.values())
        exponentials = {k: math.exp(v - largest) for k, v in logits.items()}
        denominator = sum(exponentials.values()) or 1.0
        return {k: v / denominator for k, v in exponentials.items()}

    # ------------------------------------------------------------------ predict
    def predict(self, text: str, tokens: list[str] | None = None) -> Prediction:
        """Classify the sentiment of ``text``."""
        if tokens is None:
            tokens = self._tokenizer.tokenize(text or "")

        if not tokens:
            return Prediction(
                label="neutral",
                confidence=1.0,
                probabilities={"positive": 0.0, "neutral": 1.0, "negative": 0.0},
                model_name=self.name,
            )

        total, _evidence = self.score(tokens)
        probabilities = self._distribution(total)

        if abs(total) < self.neutral_band:
            label = "neutral"
        else:
            label = "positive" if total > 0 else "negative"

        return Prediction(
            label=label,
            confidence=probabilities[label],
            probabilities={key: probabilities[key] for key in SENTIMENT_SLUGS},
            model_name=self.name,
        )

    def predict_many(self, texts: list[str]) -> list[Prediction]:
        """Classify a batch. Per-message chat sentiment runs over tens of
        thousands of rows, so this avoids repeated setup."""
        return [self.predict(text) for text in texts]

    def explain(self, text: str, tokens: list[str] | None = None) -> dict:
        """Full explanation of one prediction, for the NLP Analysis page."""
        if tokens is None:
            tokens = self._tokenizer.tokenize(text or "")
        total, evidence = self.score(tokens)
        prediction = self.predict(text, tokens)
        return {
            "label": prediction.label,
            "confidence": prediction.confidence,
            "probabilities": prediction.probabilities,
            "total_score": round(total, 3),
            "neutral_band": self.neutral_band,
            "evidence": evidence,
            "lexicon": lexicon_size(),
        }
