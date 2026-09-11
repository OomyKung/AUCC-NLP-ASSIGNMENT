"""Keep LLM-written headlines on disk, so a rebuild needs neither the model nor
the time.

Why this exists
---------------
A local model spends about 20 seconds per story, so re-analysing the two stored
programmes -- 112 stories -- is about 40 minutes. Every rebuild would pay that
again, and a fresh clone with Ollama installed would appear to hang during
seeding. Worse, the good headlines would exist only in whoever's database
happened to run it.

So the enrichment is cached beside the transcript snapshot and the captured
frames, for exactly the same reason those are: **the expensive, non-reproducible
part of the pipeline is committed, so the demo works offline and identically on
every machine.**

What the key is
---------------
The hash of the segment's own text, plus the model that wrote it. Both halves
matter:

* **text** -- if segmentation moves a boundary, the story is no longer the same
  story, so the cache must miss rather than staple the old headline to new
  content.
* **model** -- switching models must not silently serve the previous one's work.

The *prompt* is deliberately not part of the key. Re-wording it costs 40 minutes
of model time to regenerate 112 headlines, for a change that is usually cosmetic
-- and the cleanup in :func:`app.nlp.llm_enrich._clean_headline` runs on cached
headlines too, so the common wording failures are fixed without re-running
anything.

What goes in it
---------------
Only what the *model* returned. In the default headline-only mode that is the
headline and nothing else -- the entities on a segment come from the rule-based
extractor on every run, and caching those would freeze them: a later improvement
to :mod:`app.nlp.entities` would silently never reach a cached story. (This is
not hypothetical. The first programme's cache was seeded from the database,
which carried 327 extractor entities along with the headlines, and they had to
be cleared.)

Nothing is cached unless a caller passes a cache in, which keeps tests that run
the pipeline from writing into committed data.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from app.config import settings

# Bump when the stored shape changes, so old entries are ignored rather than
# misread.
SCHEMA = 1

_WHITESPACE = re.compile(r"\s+")


def _model_id() -> str:
    """Which model the headline came from, as it goes into the key."""
    if settings.llm_provider == "ollama":
        return f"ollama:{settings.ollama_model}"
    return f"{settings.llm_provider}:{settings.llm_model}"


def cache_path(video_id: str) -> Path:
    return settings.data_dir / "enrichments" / f"{video_id}.json"


class EnrichmentCache:
    """Headlines already written for one video, keyed by story text."""

    def __init__(
        self,
        video_id: str,
        entries: dict[str, dict] | None = None,
        *,
        autosave: bool = False,
    ) -> None:
        self.video_id = video_id
        self.entries: dict[str, dict] = entries or {}
        # Entries actually *used*, counted by the caller rather than by the
        # lookup: an entry can be read and then rejected (a headline that cleans
        # down to nothing), and counting the read would make the reported totals
        # exceed the number of stories.
        self.hits = 0
        # Headlines written into this cache during the run, which is exactly the
        # number the model was asked for -- reported instead of inferred by
        # subtraction, which could not tell a stale entry from a fresh call.
        self.writes = 0
        # Flush after every new headline. A 76-story programme is 25 minutes of
        # model time, and saving only at the end means a Ctrl+C at minute 24
        # throws all of it away.
        self.autosave = autosave
        self._dirty = False

    # ------------------------------------------------------------------ keys
    @staticmethod
    def key(text: str) -> str:
        """Content hash of one story, insensitive to whitespace only.

        Whitespace is normalised because the ASR's spacing is not meaningful and
        should not cause a miss; nothing else is, because any other difference
        means the segment really did change.
        """
        normalised = _WHITESPACE.sub(" ", text or "").strip()
        digest = hashlib.sha1(normalised.encode("utf-8")).hexdigest()
        return f"{digest[:20]}:{_model_id()}"

    # ----------------------------------------------------------------- reads
    def get(self, text: str) -> dict | None:
        """The stored entry for one story, or None. Counts nothing -- see
        :meth:`note_used`."""
        entry = self.entries.get(self.key(text))
        if not isinstance(entry, dict) or entry.get("schema") != SCHEMA:
            return None
        return entry

    def note_used(self) -> None:
        """Record that an entry from :meth:`get` was accepted and applied."""
        self.hits += 1

    # ---------------------------------------------------------------- writes
    def put(self, text: str, *, headline: str, entities: list, corrections: list) -> None:
        self.entries[self.key(text)] = {
            "schema": SCHEMA,
            "headline": headline,
            "entities": entities,
            "corrections": [list(pair) for pair in corrections],
        }
        self.writes += 1
        self._dirty = True
        if self.autosave:
            save(self)

    @property
    def dirty(self) -> bool:
        return self._dirty

    def mark_saved(self) -> None:
        self._dirty = False


def load(video_id: str, *, autosave: bool = False) -> EnrichmentCache:
    """Read the cache for one video. A missing or corrupt file is simply empty."""
    path = cache_path(video_id)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return EnrichmentCache(video_id, autosave=autosave)
    entries = data.get("entries") if isinstance(data, dict) else None
    return EnrichmentCache(
        video_id,
        entries if isinstance(entries, dict) else None,
        autosave=autosave,
    )


def save(cache: EnrichmentCache) -> None:
    """Write the cache back, only when something new was added."""
    if not cache.dirty:
        return
    path = cache_path(cache.video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"video_id": cache.video_id, "entries": cache.entries}
    # Written whole to a temporary file and moved into place, so an interrupted
    # run leaves the previous cache intact rather than a truncated one.
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    temporary.replace(path)
    cache.mark_saved()
