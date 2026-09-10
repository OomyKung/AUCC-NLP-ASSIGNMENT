"""Term matching for Thai text.

Exact token lookup is not sufficient for Thai, for two reasons:

1. A term may span several tokens. ``เป็นกำลังใจ`` tokenises as
   ``['เป็น', 'กำลังใจ']`` and is only found by joining adjacent tokens.
2. Thai builds compounds by concatenation, so a term often arrives welded to a
   prefix. ``มีผู้เสียชีวิต`` tokenises to ``['มี', 'ผู้เสียชีวิต']`` -- the term
   ``เสียชีวิต`` is present only as a *substring*.

:class:`TermMatcher` handles both. It is shared by the sentiment lexicon and
the topic gazetteer so the two cannot drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass

# Longest phrase, in tokens, attempted when joining adjacent tokens.
DEFAULT_MAX_PHRASE_TOKENS = 3

# Minimum term length eligible for substring matching. Short words such as ดี
# (2) or ชนะ (3) would fire inside unrelated compounds; long terms like
# เสียชีวิต (9) or บาดเจ็บ (7) are specific enough to be safe.
DEFAULT_MIN_SUBSTRING_LENGTH = 5


@dataclass(frozen=True, slots=True)
class Match:
    """One matched term."""

    start: int  # token index where the match begins
    span: int  # how many tokens it covers
    term: str  # the lexicon/gazetteer key that matched
    weight: float
    via: str  # "exact", "phrase" or "compound" -- shown in explanations


class TermMatcher:
    """Finds terms from a weighted vocabulary inside a token list."""

    def __init__(
        self,
        terms: dict[str, float],
        *,
        max_phrase_tokens: int = DEFAULT_MAX_PHRASE_TOKENS,
        min_substring_length: int = DEFAULT_MIN_SUBSTRING_LENGTH,
    ) -> None:
        self.terms = dict(terms)
        self.max_phrase_tokens = max(1, max_phrase_tokens)
        self.min_substring_length = min_substring_length

        # Substring candidates, longest first so the most specific term wins.
        self._substring_terms: tuple[tuple[str, float], ...] = tuple(
            sorted(
                (
                    (term, weight)
                    for term, weight in self.terms.items()
                    if len(term) >= min_substring_length
                ),
                key=lambda pair: -len(pair[0]),
            )
        )

    def find(self, tokens: list[str]) -> list[Match]:
        """Return non-overlapping matches in token order.

        The longest match at each position wins, so a phrase is scored once as
        a whole rather than once per constituent word.
        """
        matches: list[Match] = []
        index = 0
        count = len(tokens)

        while index < count:
            # 1. Longest exact match on joined adjacent tokens.
            phrase_span = 0
            for span in range(min(self.max_phrase_tokens, count - index), 0, -1):
                joined = "".join(tokens[index : index + span])
                if joined in self.terms:
                    matches.append(
                        Match(
                            start=index,
                            span=span,
                            term=joined,
                            weight=self.terms[joined],
                            via="exact" if span == 1 else "phrase",
                        )
                    )
                    phrase_span = span
                    break
            if phrase_span:
                index += phrase_span
                continue

            # 2. Compound fallback: a long term embedded inside this token.
            token = tokens[index]
            if len(token) > self.min_substring_length:
                for term, weight in self._substring_terms:
                    if len(term) < len(token) and term in token:
                        matches.append(
                            Match(
                                start=index,
                                span=1,
                                term=term,
                                weight=weight,
                                via="compound",
                            )
                        )
                        break

            index += 1

        return matches
