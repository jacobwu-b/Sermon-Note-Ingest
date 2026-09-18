"""Build the hotwords a sermon's transcription is prompted with.

Pure: no I/O, no config reads. :mod:`poller.transcriber` assembles the inputs (the
sermon's own record, the church's ledger, the operator's ``CHURCHES`` vocabulary) and
:mod:`poller.transcribe` is the one place that hands the result to faster-whisper.

``hotwords`` is the vehicle because faster-whisper re-injects it into every decoding
window. It is kept deliberately short: measured on large-v3, a list of a few dozen
terms made the model skip stretches of speech and run a third slower, and an
``initial_prompt`` sentence sent it into repetition loops — while a handful of terms
fixed proper-noun spelling at no cost (spec 0003).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence

HOTWORDS_MAX_TERMS = 6


def _clean(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def church_vocabulary(records: dict[str, dict]) -> list[str]:
    """Distinct speakers and series across a church's ledger, most frequent first."""
    counts: Counter[str] = Counter()
    for record in records.values():
        for field in ("speaker", "series"):
            term = _clean(record.get(field))
            if term:
                counts[term] += 1
    return [term for term, _count in counts.most_common()]


def _dedupe(terms: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    kept: list[str] = []
    for term in terms:
        term = term.strip()
        if term and term.casefold() not in seen:
            seen.add(term.casefold())
            kept.append(term)
    return kept


def build_hotwords(record: dict, *, church_terms: Sequence[str], vocabulary: Sequence[str]) -> str | None:
    """Comma-joined hotwords for one sermon: its own metadata first, then the church's.

    Priority order — record ``speaker``/``series``/``title``, then ``vocabulary`` (the
    operator's list from ``CHURCHES``), then ``church_terms`` (:func:`church_vocabulary`)
    — cut to :data:`HOTWORDS_MAX_TERMS`, so the sermon-specific terms always survive.
    ``blurb`` is deliberately unused: on most feeds it is a paragraph, not a term.
    """
    own = (_clean(record.get(field)) for field in ("speaker", "series", "title"))
    terms = _dedupe([*own, *vocabulary, *church_terms])[:HOTWORDS_MAX_TERMS]
    return ", ".join(terms) or None
