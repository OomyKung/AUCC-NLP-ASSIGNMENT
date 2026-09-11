r"""Repair ASR-mangled Thai names using evidence the project already holds.

The problem
-----------
YouTube's speech recognition writes Thai names phonetically and gets them wrong.
For one volleyball player -- **ออมสิน ศศิภาพร จันทวิสูตร** -- a single sentence
produced three different error rates:

=================  ==================  ===========================================
Spoken             ASR wrote           Recoverable?
=================  ==================  ===========================================
ออมสิน (nickname)   ``อำสิน``           ``ออมสิน`` *is* written correctly in the
                                       chat of the same video, but reaching it
                                       needs two character substitutions, and
                                       substitution proved unsafe -- see
                                       :func:`correct_name`. Needs the LLM.
จันทวิสูตร (family)  ``จันทวิสูตร``      already correct
ศศิภาพร (given)     ``สักสิภาพร``       **no** -- the correct form appears nowhere
                                       in any source this project holds
=================  ==================  ===========================================

Two repairs are therefore possible without a language model, and one is not.

What this module does
---------------------
1. **Builds a lexicon** of names that are probably spelled correctly, from the
   sources most likely to be right:

   * **chat messages** -- typed by people, so the spelling is a human's, not an
     ASR's. This is the strongest evidence available.
   * **every stored transcript** -- the ASR is inconsistent, writing a name
     correctly in one place and not another, so a form that recurs across
     programmes outranks a one-off.

2. **Trims welded suffixes** against that lexicon. If ``ไอซ์`` is attested and
   the ASR wrote ``ไอซ์ดำ``, the name is ``ไอซ์``. No character is ever
   substituted -- general edit-distance correction was built, measured at
   roughly 40% precision on this corpus, and rejected. See
   :func:`correct_name`.

3. **Collapses variants onto their stem.** The tokeniser welds the following
   word onto a name, so one person becomes many: ``กัน`` appeared as ``กันจอม``,
   ``กันติด``, ``กันพูด`` and six more. The stable prefix is the name -- but only
   when the extractor also saw that prefix *standing alone*, because otherwise
   six different people whose names begin ``สุ`` collapse into one.

What it deliberately does not do
--------------------------------
Guess. A name with no lexicon support is left exactly as the ASR wrote it.
Inventing a plausible Thai name would make the output look better and be
worse, and there is no way for a reader to tell the two apart.

``ศศิภาพร`` and ``อำสิน`` need knowledge of which Thai names exist, which is what
``LLM_API_KEY`` and :mod:`app.nlp.llm_enrich` are for.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict

# Maximum edit distance for a correction. 1 catches the common ASR slip
# (อำสิน/ออมสิน); 2 starts matching genuinely different names.
MAX_EDITS = 2

# A lexicon entry must be at least this many characters before it may correct
# anything, or a single edit is most of the word and the match is meaningless.
MIN_LEXICON_LENGTH = 4

# The correct form must be this many times more common than the suspect one.
# Without it two spellings of equal weight can flip each other.
# Weight a lexicon entry needs before it may correct anything. The suspect
# form must itself be unattested, so this is the whole of the evidence.
MIN_SUPPORT = 3

# Distinct welded variants needed before a stem is treated as the real name.
MIN_VARIANTS = 3

_THAI = re.compile(r"[฀-๿]")

# Titles and address words that introduce a name in chat. Chat is informal, so
# น้อง/พี่ appear far more often than นาย.
_CHAT_TITLES: frozenset[str] = frozenset(
    {"น้อง", "พี่", "คุณ", "นาย", "นาง", "นางสาว", "โค้ช", "กัปตัน", "ป้า", "ลุง"}
)


def _edit_distance(left: str, right: str, *, cap: int) -> int:
    """Levenshtein distance, abandoned once it exceeds ``cap``.

    The cap matters: this runs over a lexicon of thousands against hundreds of
    candidates, and almost every pair is nowhere near a match.
    """
    if abs(len(left) - len(right)) > cap:
        return cap + 1
    previous = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        current = [i]
        for j, b in enumerate(right, 1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (a != b),
                )
            )
        if min(current) > cap:
            return cap + 1
        previous = current
    return previous[-1]


def build_lexicon(
    chat_texts: list[str], transcript_texts: list[str]
) -> Counter[str]:
    """Names that are probably spelled correctly, with their weight.

    Chat counts for more than transcript text because a person typed it. A form
    appearing in both is the strongest evidence available here.

    The harvest **tokenises** rather than pattern-matching. A regex like
    ``น้อง([ก-ฮ]...)`` looks right and is not: Thai has no word boundaries, so
    the character class runs straight through the following words and
    ``น้องออมสินสวยมาก`` yields ``ออมสินสวยมาก`` instead of ``ออมสิน``. The
    tokeniser is the only thing that knows where the name stops.
    """
    from app.nlp.entities import _NAME_STOP_WORDS, _NOT_A_PERSON
    from app.nlp.preprocessing import clean_text
    from app.nlp.stopwords import get_stopwords
    from app.nlp.tokenizer import get_tokenizer

    tokenizer = get_tokenizer()
    # A title is followed by a particle or a verb at least as often as by a
    # name, and an unfiltered lexicon poisons the correction: "ชาดา" was being
    # rewritten to "มาหา" ("come and find", weight 84) purely on frequency.
    blocked = get_stopwords(protect_polarity=False) | _NAME_STOP_WORDS | _NOT_A_PERSON
    lexicon: Counter[str] = Counter()

    def harvest(texts: list[str], weight: int) -> None:
        for text in texts:
            if not text:
                continue
            tokens = tokenizer.tokenize(clean_text(text))
            for index, token in enumerate(tokens):
                if token not in _CHAT_TITLES:
                    continue
                following = tokens[index + 1] if index + 1 < len(tokens) else ""
                if (
                    len(following) >= 2
                    and _THAI.search(following)
                    and following not in _CHAT_TITLES
                    and following not in blocked
                ):
                    lexicon[following] += weight

    harvest(chat_texts, weight=3)
    harvest(transcript_texts, weight=1)
    return lexicon


def correct_name(name: str, lexicon: Counter[str]) -> str:
    """Trim a welded suffix off a name, using the lexicon as the authority.

    **Only suffix removal.** General edit-distance correction was built,
    measured on the real corpus and rejected: at a distance of 2 it repaired
    ``อำสิน`` to ``ออมสิน`` and ``อัถสิทธิ์`` to ``อภิสิทธิ์``, but in the same
    pass it turned ``สุภาพร`` into ``สุภา``, ``ชาดา`` into ``มาหา`` and
    ``รัชนก`` into ``รัชดา`` -- roughly 40% precision. A name that is
    confidently wrong is worse than one that is visibly garbled, because a
    reader cannot tell it happened.

    Removing a suffix substitutes no characters. If ``ไอซ์`` is an attested
    name and the ASR produced ``ไอซ์ดำ``, the name is ``ไอซ์`` and ``ดำ`` is the
    next word welded on -- a claim the lexicon fully supports.

    Fixing ``อำสิน`` -> ``ออมสิน`` needs a substitution, so it is out of reach
    here and belongs to :mod:`app.nlp.llm_enrich`.
    """
    if not name or len(name) <= MIN_LEXICON_LENGTH:
        return name

    # An attested spelling is not ours to overrule.
    if lexicon.get(name, 0) > 0:
        return name

    from app.nlp.entities import _NAME_STOP_WORDS, _NOT_A_PERSON

    droppable = _NAME_STOP_WORDS | _NOT_A_PERSON

    for size in range(len(name) - 1, MIN_LEXICON_LENGTH - 1, -1):
        stem, suffix = name[:size], name[size:]
        if lexicon.get(stem, 0) < MIN_SUPPORT:
            continue
        # The suffix must be something that is plainly not part of a name.
        # Without this the rule eats real names that merely begin with a
        # shorter attested one: สุภาพร -> สุภา, เลิศศักดิ์ -> เลิศ,
        # มัลิกา -> มัลิ. A Thai given name is routinely built from
        # components that are themselves names, so "the stem is attested" is
        # not on its own evidence that the rest is not part of it.
        if suffix in droppable:
            return stem
    return name


def collapse_variants(bodies: list[str]) -> dict[str, str]:
    """Map welded name variants onto the stem that is the real name.

    ``bodies`` is every extracted name (title removed), *with* repeats: the
    counts are the evidence. A stem is accepted only when the extractor saw it
    standing alone, which is what stops ``สุพนัส``, ``สุภาพร`` and ``สุริยัน`` --
    three different people -- collapsing onto the shared prefix ``สุ``.
    """
    seen = Counter(bodies)
    distinct = set(bodies)

    starts: defaultdict[str, int] = defaultdict(int)
    for body in distinct:
        for size in range(2, len(body)):
            starts[body[:size]] += 1

    mapping: dict[str, str] = {}
    for body in distinct:
        stem = body
        for size in range(len(body) - 1, 1, -1):
            candidate = body[:size]
            if seen.get(candidate, 0) >= 1 and starts[candidate] >= MIN_VARIANTS:
                stem = candidate
                break
        mapping[body] = stem
    return mapping
