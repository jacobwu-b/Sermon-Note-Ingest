"""Deterministic title -> filename-stem slug.

Used to name a sermon's transcript file in the Content repo
(``transcripts/<church>/<preached_on>_<slug>_<guid>.txt``). The only invariant that
matters is determinism — the same title always produces the same stem, so a
rediscovered sermon always maps to the same file (the idempotency the whole
Poller-to-Content handoff depends on) — plus a stem safe on any filesystem: ASCII,
lowercase, hyphen-separated, length-capped.
"""

from __future__ import annotations

import re
import unicodedata

# Filesystem-friendly cap on the stem. Sermon titles are short; this only guards
# against a pathological title blowing past path-length limits.
_MAX_LENGTH = 80

# Anything that is not an ASCII letter or digit becomes a separator.
_NON_ALNUM = re.compile(r"[^a-z0-9]+")

# Stem used when a title carries no transliterable alphanumerics at all.
_FALLBACK = "untitled"


def slugify(title: str) -> str:
    """Reduce ``title`` to a lowercase, ASCII, hyphen-separated filename stem.

    Folds accents to their ASCII base (``Café`` -> ``cafe``), lowercases, turns every
    run of non-alphanumerics into a single hyphen, trims leading/trailing hyphens,
    and caps the length without leaving a dangling hyphen. A title with nothing
    transliterable yields :data:`_FALLBACK` so the filename is always well-formed.
    """
    ascii_text = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode("ascii")
    stem = _NON_ALNUM.sub("-", ascii_text.lower()).strip("-")
    if len(stem) > _MAX_LENGTH:
        stem = stem[:_MAX_LENGTH].rstrip("-")
    return stem or _FALLBACK
