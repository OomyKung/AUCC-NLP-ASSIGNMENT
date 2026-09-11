r"""Split a news programme's transcript into individual stories.

The problem
-----------
A Thai morning news programme runs for hours and covers dozens of unrelated
stories back to back, with no markup saying where one ends. To point a user at
"the bit about the road accident" the boundaries have to be found from the
content itself.

The method: three independent signals, combined
-----------------------------------------------
No single signal is reliable on live broadcast speech, so three are combined and
each one's contribution is recorded on the boundary it produced, which is what
makes a result explainable rather than a black box.

1. **Lexical cohesion (TextTiling, Hearst 1997).** Adjacent blocks of transcript
   are turned into TF-IDF vectors and compared by cosine similarity. Where a
   story ends, vocabulary changes, so similarity dips. A *depth score* measures
   how deep each dip is relative to the peaks either side of it, which finds
   local minima without needing an absolute similarity threshold -- important
   because absolute similarity varies with how chatty a presenter is.

2. **Topic-label change.** Each block is classified by the trained topic model.
   A run of blocks agreeing on a new label is evidence of a new story. Used with
   hysteresis: a single disagreeing block is noise and is ignored, because the
   classifier sees only ~30 seconds of speech at a time.

3. **Non-speech markers.** Thai news programmes put a music sting between
   stories, and YouTube's ASR emits it as ``[เพลง]``. This is a strong, cheap
   prior that costs nothing to use: 167 of them appear in one 249-minute
   programme. Speaker changes (``>>``) were considered and rejected as a primary
   signal -- at 1,531 occurrences in the same programme they fire far too often
   to mark a story boundary, so they only break ties.

A minimum segment length is enforced afterwards. Without it the depth score
produces runs of adjacent boundaries around a single transition, and the
dashboard fills with eight-second "stories".

Why not a neural segmenter: there is no Thai topic-segmentation training data in
this project, and the transformer experiment already showed that 668 labelled
rows is not enough to fine-tune on. TextTiling needs no training data at all,
which is the right trade at this scale, and it is citable.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

from app.config import settings
from app.services.transcripts.base import Transcript, TranscriptCue

# ---------------------------------------------------------------- parameters

# Duration of one comparison block. 30s is a compromise: short enough to locate
# a boundary usefully, long enough that a block holds sixty-odd Thai words and
# the topic classifier has something to work with.
BLOCK_MS = 30_000

# Blocks compared either side of a candidate gap. 3 blocks = 90s per side, which
# is about the length of a short news item.
COMPARISON_BLOCKS = 3

# A block with less text than this is mostly silence or music; it gets no vote.
MIN_BLOCK_CHARS = 40

# Consecutive blocks that must agree on a new topic before the label signal
# fires. 1 would make it fire on classifier noise.
LABEL_RUN = 2

# Segments shorter than this are merged into their neighbour.
MIN_SEGMENT_MS = 60_000

# A music gap at least this long is treated as a programme transition.
MUSIC_GAP_MS = 4_000

# Depth-score cutoff, as the classic TextTiling heuristic: mean + std/2 of the
# depth distribution. Expressed as a multiplier so it can be tuned from config
# without touching code.
DEPTH_STD_MULTIPLIER = 0.5

_THAI = re.compile(r"[฀-๿]")


# -------------------------------------------------------------------- types


@dataclass(slots=True)
class Block:
    """One fixed-duration slice of transcript, ready to compare."""

    index: int
    start_ms: int
    end_ms: int
    text: str
    tokens: list[str] = field(default_factory=list)
    topic: str | None = None
    topic_confidence: float = 0.0
    # Music/applause time inside this block, used for the non-speech signal.
    non_speech_ms: int = 0

    @property
    def usable(self) -> bool:
        return len(self.text) >= MIN_BLOCK_CHARS


@dataclass(slots=True)
class Boundary:
    """A detected story boundary and the evidence behind it."""

    block_index: int  # boundary sits *before* this block
    time_ms: int
    depth: float = 0.0
    reasons: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        """How much to trust this boundary, from how many signals agreed.

        Deliberately simple and monotone: each signal adds a fixed amount. A
        learned combiner would need boundary-labelled Thai broadcast data, which
        does not exist here.
        """
        return min(1.0, 0.34 * len(self.reasons))


@dataclass(slots=True)
class Segment:
    """One story: a time range with its own analysis."""

    index: int
    start_ms: int
    end_ms: int
    text: str
    topic: str = ""
    topic_confidence: float = 0.0
    sentiment: str = ""
    sentiment_confidence: float = 0.0
    headline: str = ""
    summary: str = ""
    keywords: list[str] = field(default_factory=list)
    entities: list[dict] = field(default_factory=list)
    boundary_confidence: float = 0.0
    boundary_reasons: list[str] = field(default_factory=list)
    # Character offsets inside `text` where a transcript cue begins. Used to
    # start a headline on a real unit of speech rather than mid-name.
    cue_starts: set[int] = field(default_factory=set)

    @property
    def duration_ms(self) -> int:
        return max(0, self.end_ms - self.start_ms)

    @property
    def start_seconds(self) -> int:
        return self.start_ms // 1000

    def youtube_url(self, video_id: str) -> str:
        """Deep link to the moment this story starts.

        A couple of seconds are shaved off so the click lands just *before* the
        first word rather than halfway through it.
        """
        seconds = max(0, self.start_seconds - 2)
        return f"https://www.youtube.com/watch?v={video_id}&t={seconds}s"

    @property
    def timecode(self) -> str:
        """``H:MM:SS`` label, as a viewer would read it."""
        total = self.start_seconds
        hours, remainder = divmod(total, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours}:{minutes:02d}:{seconds:02d}"
        return f"{minutes}:{seconds:02d}"


# ---------------------------------------------------------------- block build


def build_blocks(transcript: Transcript, *, block_ms: int = BLOCK_MS) -> list[Block]:
    """Slice a transcript into fixed-duration blocks."""
    if not transcript.cues:
        return []

    from app.nlp.preprocessing import clean_text, filter_tokens
    from app.nlp.tokenizer import get_tokenizer

    tokenizer = get_tokenizer()
    end_ms = max(cue.end_ms for cue in transcript.cues)

    blocks: list[Block] = []
    for index, start in enumerate(range(0, end_ms, block_ms)):
        stop = start + block_ms
        text = transcript.text_between(start, stop)
        non_speech = sum(
            cue.duration_ms
            for cue in transcript.cues
            if cue.non_speech and cue.start_ms < stop and cue.end_ms > start
        )
        block = Block(
            index=index,
            start_ms=start,
            end_ms=min(stop, end_ms),
            text=text,
            non_speech_ms=non_speech,
        )
        if block.usable:
            # protect_polarity=False: these tokens are used for *topic*
            # similarity, where negators carry no topical information.
            block.tokens = filter_tokens(
                tokenizer.tokenize(clean_text(text)), protect_polarity=False
            )
        blocks.append(block)
    return blocks


# ------------------------------------------------------- signal 1: cohesion


def _vector(tokens: list[str]) -> Counter[str]:
    return Counter(token for token in tokens if len(token) > 1 and _THAI.search(token))


def _cosine(left: Counter[str], right: Counter[str]) -> float:
    """Cosine similarity between two bag-of-words counters."""
    if not left or not right:
        return 0.0
    shared = set(left) & set(right)
    if not shared:
        return 0.0
    numerator = sum(left[term] * right[term] for term in shared)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def cohesion_scores(
    blocks: list[Block], *, window: int = COMPARISON_BLOCKS
) -> list[float]:
    """Similarity across the gap before each block.

    ``scores[i]`` compares the ``window`` blocks ending at ``i-1`` with the
    ``window`` blocks starting at ``i``. Index 0 has nothing to its left, so it
    is 1.0 (perfectly cohesive, i.e. not a boundary).
    """
    scores = [1.0] * len(blocks)
    for gap in range(1, len(blocks)):
        left_tokens: list[str] = []
        for block in blocks[max(0, gap - window) : gap]:
            left_tokens.extend(block.tokens)
        right_tokens: list[str] = []
        for block in blocks[gap : gap + window]:
            right_tokens.extend(block.tokens)
        scores[gap] = _cosine(_vector(left_tokens), _vector(right_tokens))
    return scores


def depth_scores(similarities: list[float]) -> list[float]:
    """TextTiling depth: how deep each dip is relative to its surrounding peaks.

    For each gap, walk left while similarity keeps rising to find the left peak,
    then walk right for the right peak. Depth is the sum of the two climbs. This
    is what makes the method robust to an overall similarity level that drifts
    across a programme -- only the *shape* matters.
    """
    depths = [0.0] * len(similarities)
    for index in range(1, len(similarities) - 1):
        valley = similarities[index]

        left = index
        while left > 0 and similarities[left - 1] >= similarities[left]:
            left -= 1
        right = index
        last = len(similarities) - 1
        while right < last and similarities[right + 1] >= similarities[right]:
            right += 1

        depths[index] = max(0.0, similarities[left] - valley) + max(
            0.0, similarities[right] - valley
        )
    return depths


# ----------------------------------------------------- signal 2: topic label


def classify_blocks(blocks: list[Block]) -> None:
    """Attach a topic label to every usable block, in one batched pass."""
    from app.nlp.registry import get_components

    backend = get_components().topic
    for block in blocks:
        if not block.usable:
            continue
        prediction = backend.predict(block.text, tokens=block.tokens or None)
        block.topic = prediction.label
        block.topic_confidence = prediction.confidence


def smooth_labels(labels: list[str]) -> list[str]:
    """Remove isolated single-block labels from a label sequence.

    A block is 30 seconds of speech, which the topic classifier regularly gets
    wrong on its own; the sequence ``crime crime sports crime crime`` is one
    story with a blip, not three stories.

    Looking only forwards is not enough to catch this. Forward hysteresis
    correctly refuses to split at the blip -- ``sports`` is not sustained -- but
    then splits at the *return*, because from ``sports`` the following
    ``crime crime`` looks like a perfectly sustained change. Both halves of one
    story end up separated by a boundary that exists only because of a single
    misclassified block.

    So isolated labels are rewritten to their neighbour before any change
    detection runs. Only a label that differs from both sides is replaced, so a
    genuine two-block story survives.
    """
    if len(labels) < 3:
        return list(labels)

    smoothed = list(labels)
    for index in range(1, len(labels) - 1):
        previous, current, following = labels[index - 1], labels[index], labels[index + 1]
        if current != previous and current != following and previous == following:
            smoothed[index] = previous
    return smoothed


def label_change_indices(blocks: list[Block], *, run: int = LABEL_RUN) -> set[int]:
    """Gaps where the topic label changes and *stays* changed.

    Two defences against classifier noise, because a 30-second block is thin
    evidence: isolated labels are smoothed away first (see :func:`smooth_labels`),
    then a change must be backed by ``run`` consecutive agreeing blocks.
    """
    changes: set[int] = set()
    labelled = [(block.index, block.topic) for block in blocks if block.topic]
    if len(labelled) < run + 1:
        return changes

    indices = [index for index, _label in labelled]
    smoothed = smooth_labels([label for _index, label in labelled])

    for position in range(1, len(smoothed)):
        previous = smoothed[position - 1]
        upcoming = smoothed[position : position + run]
        if len(upcoming) < run:
            break
        if upcoming[0] != previous and len(set(upcoming)) == 1:
            changes.add(indices[position])
    return changes


# ------------------------------------------------- signal 3: non-speech gaps


def music_gap_indices(blocks: list[Block], *, threshold_ms: int = MUSIC_GAP_MS) -> set[int]:
    """Gaps preceded by a substantial music or applause break."""
    return {
        block.index
        for block in blocks
        if block.non_speech_ms >= threshold_ms and block.index > 0
    }


# ------------------------------------------------------------- combination


def detect_boundaries(
    blocks: list[Block],
    *,
    depth_multiplier: float = DEPTH_STD_MULTIPLIER,
) -> list[Boundary]:
    """Combine the three signals into a boundary list."""
    if len(blocks) < 3:
        return []

    similarities = cohesion_scores(blocks)
    depths = depth_scores(similarities)

    # Classic TextTiling cutoff: mean + std/2 over the depth distribution.
    scored = [d for d in depths if d > 0]
    if scored:
        mean = sum(scored) / len(scored)
        variance = sum((d - mean) ** 2 for d in scored) / len(scored)
        cutoff = mean + depth_multiplier * math.sqrt(variance)
    else:
        cutoff = math.inf

    cohesion_hits = {
        index for index, depth in enumerate(depths) if depth >= cutoff and index > 0
    }
    label_hits = label_change_indices(blocks)
    music_hits = music_gap_indices(blocks)

    # The signals must not be allowed to contradict each other. A label change
    # where the *vocabulary carried straight on* is the classifier wobbling on 30
    # seconds of speech, not a new story -- measured on one programme, 37 of 89
    # label-change gaps sat at above-median cohesion, and they were visibly
    # splitting single stories into three (one report on a shot monkey became
    # crime / politics / crime with identical keywords throughout).
    #
    # So a boundary proposed by the label signal *alone* is vetoed when
    # similarity across that gap is above the median. When any other signal
    # agrees, the boundary stands regardless: two signals agreeing is exactly
    # the evidence this veto is looking for.
    usable_similarities = [
        similarities[index]
        for index in range(1, len(blocks))
        if blocks[index].usable
    ]
    median_similarity = (
        sorted(usable_similarities)[len(usable_similarities) // 2]
        if usable_similarities
        else 0.0
    )

    boundaries: list[Boundary] = []
    for index in sorted(cohesion_hits | label_hits | music_hits):
        reasons: list[str] = []
        if index in cohesion_hits:
            reasons.append("lexical-cohesion")
        if index in label_hits:
            reasons.append("topic-change")
        if index in music_hits:
            reasons.append("music-break")

        if (
            reasons == ["topic-change"]
            and index < len(similarities)
            and similarities[index] > median_similarity
        ):
            continue

        boundaries.append(
            Boundary(
                block_index=index,
                time_ms=blocks[index].start_ms,
                depth=depths[index] if index < len(depths) else 0.0,
                reasons=reasons,
            )
        )
    return boundaries


def _merge_short(
    spans: list[tuple[int, int, Boundary | None]], *, minimum_ms: int
) -> list[tuple[int, int, Boundary | None]]:
    """Absorb spans shorter than ``minimum_ms`` into the previous one.

    Runs until stable, because merging two short spans can still leave a short
    one. The first span has no previous, so it absorbs forwards instead.
    """
    if not spans:
        return spans

    changed = True
    while changed and len(spans) > 1:
        changed = False
        for position, (start, end, boundary) in enumerate(spans):
            if end - start >= minimum_ms:
                continue
            if position == 0:
                nxt = spans[1]
                spans[0] = (start, nxt[1], boundary)
                del spans[1]
            else:
                prev = spans[position - 1]
                spans[position - 1] = (prev[0], end, prev[2])
                del spans[position]
            changed = True
            break
    return spans


def segment_transcript(
    transcript: Transcript,
    *,
    block_ms: int = BLOCK_MS,
    min_segment_ms: int | None = None,
    analyse: bool = True,
) -> tuple[list[Segment], list[Boundary], list[Block]]:
    """Split ``transcript`` into stories and analyse each one.

    Returns ``(segments, boundaries, blocks)``. The blocks and boundaries come
    back too so the UI can show *why* each split happened rather than only the
    result.
    """
    minimum = min_segment_ms or settings.segment_min_seconds * 1000

    blocks = build_blocks(transcript, block_ms=block_ms)
    if not blocks:
        return [], [], []

    classify_blocks(blocks)
    boundaries = detect_boundaries(blocks)

    # Turn boundaries into spans covering the whole programme.
    cut_points = [0] + [b.block_index for b in boundaries]
    by_index = {b.block_index: b for b in boundaries}

    spans: list[tuple[int, int, Boundary | None]] = []
    for position, cut in enumerate(cut_points):
        start_ms = blocks[cut].start_ms
        if position + 1 < len(cut_points):
            end_ms = blocks[cut_points[position + 1]].start_ms
        else:
            end_ms = blocks[-1].end_ms
        spans.append((start_ms, end_ms, by_index.get(cut)))

    spans = _merge_short(spans, minimum_ms=minimum)

    segments: list[Segment] = []
    for index, (start_ms, end_ms, boundary) in enumerate(spans):
        text = transcript.text_between(start_ms, end_ms)
        if len(text) < MIN_BLOCK_CHARS:
            # A span with no speech at all (pure music) is not a story.
            continue
        segment = Segment(
            index=len(segments),
            start_ms=start_ms,
            end_ms=end_ms,
            text=text,
            cue_starts=transcript.cue_start_offsets(start_ms, end_ms),
            boundary_confidence=boundary.confidence if boundary else 1.0,
            boundary_reasons=list(boundary.reasons) if boundary else ["programme-start"],
        )
        segments.append(segment)

    if analyse:
        analyse_segments(segments)
    return segments, boundaries, blocks


def analyse_segments(segments: list[Segment]) -> None:
    """Run the full NLP pipeline over each segment, in place."""
    from app.services.pipeline import get_pipeline

    from app.nlp.stopwords import BROADCAST_FILLER

    pipeline = get_pipeline()
    for segment in segments:
        # Broadcast filler is excluded from keywords only. It is not in the
        # general stopword list because that list defines the trained models'
        # feature space; see app.nlp.stopwords.BROADCAST_FILLER.
        result = pipeline.analyse(
            segment.text,
            with_summary=True,
            with_entities=True,
            keyword_exclude=BROADCAST_FILLER,
        )
        segment.topic = result.topic.label
        segment.topic_confidence = result.topic.confidence
        segment.sentiment = result.sentiment.label
        segment.sentiment_confidence = result.sentiment.confidence
        segment.keywords = [keyword.word for keyword in result.keywords]
        segment.summary = result.summary.text if result.summary else ""
        segment.entities = [entity.as_dict() for entity in result.entities]
        segment.headline = synthesise_headline(segment)


def synthesise_headline(segment: Segment, *, max_length: int = 120) -> str:
    """A readable one-line description of the story.

    Prefers a real phrase pulled out of what was actually said (see
    :mod:`app.nlp.headline`). Joining the top keywords -- which is what this
    used to do -- produced titles like ``ติดตาม · นิติ · ศุกร์ · เช้านี้``, four
    disconnected words that describe nothing.

    Falls back through summary, then keywords, then the timecode, so a story is
    never nameless: on a rambling or badly transcribed segment there may be no
    dense span to find, and a weak title beats a blank one.
    """
    from app.nlp.headline import extract_headline
    from app.nlp.keyword_extractor import TfidfKeywordExtractor

    extractor = TfidfKeywordExtractor.load()
    headline = extract_headline(
        segment.text,
        keywords=segment.keywords,
        idf=extractor if extractor.is_fitted else None,
        cue_starts=segment.cue_starts or None,
    )

    if not headline and segment.summary:
        headline = segment.summary
    if not headline and segment.keywords:
        headline = " · ".join(segment.keywords[:4])

    headline = re.sub(r"\s+", " ", headline or "").strip()
    if len(headline) > max_length:
        headline = headline[: max_length - 1].rstrip() + "…"
    return headline or f"ช่วง {segment.timecode}"
