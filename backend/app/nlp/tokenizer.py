"""Thai word and sentence tokenisation.

Thai is written without spaces between words, so tokenisation is a real
modelling step rather than a ``str.split``. PyThaiNLP's ``newmm`` engine
(dictionary-based maximum matching with TCC boundaries) is the default: it is
pure Python, needs no model download, and handled the collected chat corpus
correctly in spot checks.

Wrapped behind :class:`~app.nlp.base.Tokenizer` so the engine can be swapped
(``NLP_TOKENIZER=longest``) or replaced with a neural tokenizer later.
"""

from __future__ import annotations

from functools import lru_cache

from pythainlp.tokenize import sent_tokenize, word_tokenize

from app.config import settings
from app.nlp.preprocessing import (
    PreprocessResult,
    clean_text,
    filter_tokens,
)

# Engines verified to work with only the pinned requirements installed.
# PyThaiNLP's other engines (attacut, deepcut, nlpo3, icu, sefr_cut, oskut,
# tltk) each need an extra package, so they are deliberately not offered here.
SUPPORTED_ENGINES = ("newmm", "newmm-safe", "longest", "mm")


class ThaiTokenizer:
    """Tokenises Thai text with PyThaiNLP."""

    def __init__(self, engine: str | None = None) -> None:
        requested = (engine or settings.nlp_tokenizer or "newmm").lower()
        if requested not in SUPPORTED_ENGINES:
            raise ValueError(
                f"Unsupported tokenizer engine {requested!r}. "
                f"Choose one of: {', '.join(SUPPORTED_ENGINES)}."
            )
        self.engine = requested
        self.name = f"pythainlp:{requested}"

    # ---------------------------------------------------------------- tokens
    def tokenize(self, text: str) -> list[str]:
        """Split ``text`` into non-whitespace tokens."""
        if not text or not text.strip():
            return []
        tokens = word_tokenize(text, engine=self.engine, keep_whitespace=False)
        return [token for token in tokens if token.strip()]

    def sentences(self, text: str) -> list[str]:
        """Split ``text`` into sentences.

        Uses PyThaiNLP's CRF sentence segmenter, falling back to whitespace and
        newline splitting if the CRF model is unavailable -- the summarizer must
        keep working either way.
        """
        if not text or not text.strip():
            return []
        try:
            parts = sent_tokenize(text)
        except Exception:
            parts = sent_tokenize(text, engine="whitespace+newline")
        return [part.strip() for part in parts if part and part.strip()]

    # -------------------------------------------------------------- pipeline
    def preprocess(
        self,
        text: str,
        *,
        keep_emoji: bool = True,
        protect_polarity: bool = True,
        with_sentences: bool = True,
    ) -> PreprocessResult:
        """Run the full clean -> tokenise -> filter chain.

        This is the single entry point the pipeline and the NLP Analysis page
        use, so the intermediate values shown in the UI are exactly the ones the
        models consumed.
        """
        cleaned = clean_text(text, keep_emoji=keep_emoji)
        tokens = self.tokenize(cleaned)
        return PreprocessResult(
            raw_text=text or "",
            cleaned_text=cleaned,
            tokens=tokens,
            filtered_tokens=filter_tokens(tokens, protect_polarity=protect_polarity),
            sentences=self.sentences(cleaned) if with_sentences else [],
        )


@lru_cache(maxsize=4)
def get_tokenizer(engine: str | None = None) -> ThaiTokenizer:
    """Cached tokenizer accessor.

    Cached because ``newmm`` builds a dictionary trie on first use; sharing one
    instance keeps per-request cost near zero.
    """
    return ThaiTokenizer(engine)
