"""Feature extraction for the trained Thai classifiers.

The analyser functions here are deliberately **module-level**, not lambdas or
closures. ``TfidfVectorizer(analyzer=...)`` stores a reference to whatever it is
given, and joblib cannot pickle a lambda -- a model trained with one would save
but fail to load, which is the classic way this kind of pipeline breaks.

Two views of the text are combined:

* **word n-grams** over PyThaiNLP tokens, which carry topical meaning
* **character n-grams**, which are robust to the tokenisation mistakes and
  spelling variation that are common in chat text, and which give the model
  something to work with for words it has never seen
"""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def thai_word_analyzer(text: str) -> list[str]:
    """Tokenise and filter Thai text into content-word unigrams and bigrams.

    Used as the ``analyzer`` for the word-level vectoriser.
    """
    # Imported lazily so this module stays cheap to import for unpickling.
    from app.nlp.preprocessing import clean_text, filter_tokens
    from app.nlp.tokenizer import get_tokenizer

    cleaned = clean_text(text or "")
    tokens = filter_tokens(
        get_tokenizer().tokenize(cleaned),
        # Keep negation: for sentiment, removing ไม่ inverts the label.
        protect_polarity=True,
    )

    # Bigrams capture short Thai collocations such as "ไม่ ดี" or "ทีม ชาติ".
    bigrams = [f"{a}_{b}" for a, b in zip(tokens, tokens[1:], strict=False)]
    return tokens + bigrams


def thai_char_preprocessor(text: str) -> str:
    """Cleaned text for the character-level vectoriser."""
    from app.nlp.preprocessing import clean_text

    return clean_text(text or "")


def build_vectorizer(
    *,
    word_min_df: int = 2,
    char_min_df: int = 3,
    char_ngram_range: tuple[int, int] = (2, 4),
    max_features: int | None = 60_000,
) -> FeatureUnion:
    """Build the combined word + character TF-IDF feature extractor.

    Args:
        word_min_df: Ignore word features appearing in fewer documents. At 2, a
            term seen once in the whole corpus cannot become a shortcut the
            model memorises.
        char_min_df: Same, for character n-grams.
        char_ngram_range: Character n-gram sizes. Wider ranges capture more
            Thai morphology at the cost of a much larger vocabulary.
        max_features: Cap on each view's vocabulary size.
    """
    word = TfidfVectorizer(
        analyzer=thai_word_analyzer,
        min_df=word_min_df,
        max_features=max_features,
        sublinear_tf=True,
    )
    char = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=char_ngram_range,
        preprocessor=thai_char_preprocessor,
        min_df=char_min_df,
        max_features=max_features,
        sublinear_tf=True,
    )
    return FeatureUnion([("word", word), ("char", char)])
