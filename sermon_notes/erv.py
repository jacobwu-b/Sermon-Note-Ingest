"""Verbatim ERV passage text for "Scripture engaged" (spec 0016, ADR-0021, ADR-0026,
ADR-0028).

Turns a canonical reference string ("Luke 10:25-37") into the verbatim ERV text
for that range, read from the vendored local dataset under ``data/processed/erv/``
— no network, no external boundary. The dataset is produced by
``scripts/import_erv.py`` from the cursor-bible processed ERV; each book file is a
flat ``{chapter: {verse-key: text}}`` map, where a verse key is the ERV's own
display number: usually a single verse ("14"), sometimes a merged span ("14-15")
for verses the translation renders as one block.

Because the ERV's blocks do not always align with the verse numbers a citation
uses, resolution returns a :class:`Passage` — the text *and* the reference that
truthfully describes it. Asking for a verse inside a merged block yields the whole
block and a reference widened to match, so a rendered citation always names exactly
the verses printed beside it. A resolved span also widens forward, one block at a
time, when its text doesn't end on a sentence boundary — the ERV sometimes runs a
sentence past a verse with no merged-block signal to catch it (ADR-0028) — up to a
small cap; ``Passage.complete`` is ``False`` when the cap or the chapter's end was
hit before a sentence finished, so the caller can log the outlier.

Reference parsing is delegated to :func:`sermon_notes.scripture.parse_reference`.
A same-book reference crossing a chapter boundary ("Revelation 21:1-22:5")
resolves here (spec 0016, ADR-0079) by concatenating every chapter the span
touches, even though the omnibible URL contract addresses one chapter and so
cannot link the whole thing — :func:`reference_url` below offers a best-effort
link to the leading chapter instead. A reference the grammar rejects outright,
or one whose book/chapter/verses are absent from the dataset, returns ``None``
so the caller can record the miss (there is no fallback text).
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from sermon_notes import scripture
from sermon_notes.scripture import DASH_CHARS, parse_reference

# Vendored ERV dataset, resolved like the other repo-relative data dirs
# (transcribe.DEFAULT_TRANSCRIPTS_DIR, render.DEFAULT_NOTES_DIR).
DEFAULT_ERV_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "erv"

# Book slugs whose canonical file name differs from scripture.py's slugged
# alternate spelling: the omnibible reader redirects these aliases, but the local
# files have one canonical name each, so we map before opening.
_SLUG_ALIASES = {"psalm": "psalms", "song-of-songs": "song-of-solomon"}

_WHITESPACE_RE = re.compile(r"\s+")

# Typographic double quotes, which delimit speech in the ERV. Single quotes mark
# speech nested inside speech and are never touched.
_OPEN_QUOTE = "“"
_CLOSE_QUOTE = "”"

# Sentence-terminal punctuation for the forward widen (ADR-0028). Checked after the
# trailing close-quote _flatten already re-balances, so a quotation's closing mark
# never masks an incomplete sentence underneath it.
_SENTENCE_END = (".", "!", "?")

# How many blocks the forward widen (ADR-0028) may append beyond the original
# resolution before giving up — a guard against a heuristic with no dataset to check
# itself against pulling in a runaway amount of text.
_MAX_FORWARD_WIDEN = 5

# The trailing verse portion of a reference ("14" in "John 10:14", "41-43a" in
# "Luke 9:41-43a", "3" in the single-chapter shorthand "Jude 3"). Substituting only
# this leaves the book name as the caller spelled it — "Psalm 23" must never come
# back as "Psalms 23" — and keeps the shorthand's chapterless form intact.
_VERSE_PORTION_RE = re.compile(
    r"\d+[a-z]?(?:\s*[" + DASH_CHARS + r"]\s*\d+[a-z]?)?$",
)

# The trailing chapter:verse-chapter:verse portion of a cross-chapter reference
# ("21:1-22:5" in "Revelation 21:1-22:5"). Mirrors _VERSE_PORTION_RE's substitution
# style but for the two-chapter shape, so widening a cross-chapter span still
# preserves the caller's book spelling.
_CROSS_CHAPTER_PORTION_RE = re.compile(
    r"\d+:\d+[a-z]?\s*[" + DASH_CHARS + r"]\s*\d+:\d+[a-z]?$",
)


class VerseSegment(NamedTuple):
    """One verse's (or merged block's) text, keyed by its ERV display number.

    ``number`` is the dataset's own verse key — a single verse ("14") or a merged
    span ("14-15") — never re-derived from the citation, so it always matches what
    the vendored data actually keys the block by.
    """

    number: str
    text: str


class Passage(NamedTuple):
    """Resolved ERV text and the reference that accurately describes it.

    ``reference`` is the caller's string unchanged whenever the resolved verses are
    exactly the ones asked for, and a widened span when the ERV merges a verse in
    the requested range into a larger block, or when the text is widened forward to
    complete a sentence (ADR-0028).

    ``complete`` is ``False`` when a sentence-boundary widen (ADR-0028) stopped
    without reaching terminal punctuation — the chapter ran out or the widen cap was
    hit — so the caller can log the outlier for a human to look at.

    ``segments`` is ``text`` broken into its per-verse (or per-merged-block,
    ADR-0026) pieces, in order, so a renderer can print each verse's number as a
    superscript ahead of its own text (ADR-0070). Space-joining every segment's
    ``text`` reproduces ``text`` exactly.

    ``verse_count`` is the true number of verses covered — summed across every
    chapter a cross-chapter span touches, not a subtraction of verse numbers that
    reset at each chapter boundary — so :data:`sermon_notes.generate.MAX_QUOTED_VERSES`
    (ADR-0055) can be applied correctly regardless of how many chapters a passage
    spans. ``None`` for a chapter-only passage, which names an unbounded span and
    is never quotable (ADR-0055).
    """

    reference: str
    text: str
    complete: bool = True
    segments: tuple[VerseSegment, ...] = ()
    verse_count: int | None = None


class _Block(NamedTuple):
    """One stored unit of ERV text and the verse range it covers."""

    start: int
    end: int
    text: str


@lru_cache(maxsize=None)
def _load_book(book_slug: str, data_dir: Path) -> dict[str, dict[str, str]] | None:
    """Load one ``{chapter: {verse-key: text}}`` book file, or ``None`` if absent."""
    path = data_dir / f"{_SLUG_ALIASES.get(book_slug, book_slug)}.json"
    if not path.is_file():
        return None
    book: dict[str, dict[str, str]] = json.loads(path.read_text(encoding="utf-8"))
    return book


def _blocks(chapter: dict[str, str]) -> list[_Block]:
    """Parse a chapter's verse keys into blocks ordered by starting verse.

    A key is either a single verse ("14") or the span the ERV merged into one block
    ("14-15"); both become a ``_Block`` covering the verses they address.
    """
    blocks = []
    for key, text in chapter.items():
        first, _, last = key.partition("-")
        blocks.append(_Block(int(first), int(last or first), text))
    return sorted(blocks)


def _flatten_segments(blocks: Iterable[_Block]) -> list[VerseSegment]:
    """Split ``blocks`` into one :class:`VerseSegment` each, leaving one balanced
    quotation across the whole set (spec 0016, ADR-0070).

    The ERV punctuates speech that runs across paragraphs the conventional way:
    reopen ``“`` at every paragraph, close ``”`` only where the speech ends. Read as
    paragraphs that is correct, but a study note prints each verse as one
    continuous line, where each reopening becomes a stray mark mid-sentence — a
    17-verse discourse arrives with four opening quotes and no closing one.

    So the paragraph structure is consumed here, before whitespace is collapsed,
    with quotation depth carried across block boundaries exactly as it already
    carries across paragraphs within one block:

    - An opening mark that begins a paragraph while a quotation is already open is
      a reopening, and is dropped. Line breaks are the signal, not verse or block
      boundaries — the ERV breaks paragraphs inside a verse too.
    - A closing mark with nothing open belongs to speech that began before the
      excerpt, and is dropped.
    - A quotation still open after the last block is closed there, since the
      excerpt ends before the speech does.

    Marks that survive are the ones the excerpt itself opened, so each segment
    reads as part of one quotation of exactly what is shown.
    """
    blocks = list(blocks)
    depth = 0
    segments: list[VerseSegment] = []
    for index, block in enumerate(blocks):
        paragraphs: list[str] = []
        for paragraph in block.text.split("\n"):
            paragraph = _WHITESPACE_RE.sub(" ", paragraph).strip()
            if not paragraph:
                continue
            if depth and paragraph.startswith(_OPEN_QUOTE):
                paragraph = paragraph[1:].lstrip()
            kept: list[str] = []
            for char in paragraph:
                if char == _OPEN_QUOTE:
                    depth += 1
                elif char == _CLOSE_QUOTE:
                    if not depth:
                        continue
                    depth -= 1
                kept.append(char)
            paragraphs.append("".join(kept))
        text = " ".join(paragraphs).strip()
        if index == len(blocks) - 1:
            text += _CLOSE_QUOTE * depth
        segments.append(VerseSegment(_block_number(block), text))
    return segments


def _block_number(block: _Block) -> str:
    """The ERV display number for ``block`` — a single verse or a merged span."""
    return str(block.start) if block.start == block.end else f"{block.start}-{block.end}"


def _flatten(blocks: Iterable[_Block]) -> str:
    """Join ``blocks``' text into one line, leaving a single balanced quotation.

    A thin join over :func:`_flatten_segments` — see its docstring for the
    quotation-balancing rule. Used where only the flat text is needed: the
    sentence-boundary widen's boundary check, and building :attr:`Passage.text`.
    """
    return " ".join(s.text for s in _flatten_segments(blocks) if s.text).strip()


def widen_reference(reference: str, start: int, end: int) -> str:
    """Rewrite ``reference``'s verse portion to the span ``start``–``end``.

    Public so a caller outside this module can build a reference string for a span
    it already knows resolves — ``generate_notes`` merging two ``scripture_engaged``
    entries whose ranges overlap (ADR-0029) — without re-deriving the verse-portion
    regex. A single verse (``start == end``) renders without a dash.
    """
    span = str(start) if start == end else f"{start}-{end}"
    return _VERSE_PORTION_RE.sub(span, reference)


def widen_cross_chapter_reference(
    reference: str, start_chapter: int, start: int, end_chapter: int, end: int
) -> str:
    """Rewrite a cross-chapter ``reference``'s verse portion to ``start``–``end``.

    The two-chapter counterpart to :func:`widen_reference`: both boundary
    chapters can widen (a merged block at either end, ADR-0026) but the chapters
    themselves never do — a widen never crosses *into* a third chapter (ADR-0028
    stays bounded to the resolved end chapter). Only the verse portion changes.
    """
    span = f"{start_chapter}:{start}-{end_chapter}:{end}"
    return _CROSS_CHAPTER_PORTION_RE.sub(span, reference)


def _ends_sentence(text: str) -> bool:
    """Whether ``text`` ends on a sentence boundary, ignoring trailing wrappers.

    A closing double quote can sit a space after the punctuation it closes ("God.’
    ”" — the ERV's own paragraph-close spacing), so quote marks and whitespace strip
    together as one trailing set rather than in two separate passes.

    Closing brackets strip with them: a parenthetical wraps a sentence exactly as a
    quotation does, and ERV ``John 1:42`` ends ``(“Cephas” means “Peter. ”)`` — a
    finished sentence whose last character is a bracket. Of the 205 verse blocks in
    the dataset that end in ``)``, 204 close a complete sentence, so reading the
    bracket as terminal punctuation is the common case rather than the exception.
    Stripping cannot mask an unfinished sentence, because what the bracket wraps is
    re-checked against :data:`_SENTENCE_END` in turn.
    """
    stripped = text.rstrip(_CLOSE_QUOTE + "’" + ")]" + " \t\n")
    return stripped.endswith(_SENTENCE_END)


def _chapter_length(chapter: dict[str, str]) -> int:
    """The highest verse number a loaded chapter dict covers."""
    return max(b.end for b in _blocks(chapter))


def _widen_forward(
    selected: list[_Block], span_end: int, by_start: dict[int, _Block]
) -> tuple[list[_Block], int, bool]:
    """Append blocks from ``by_start`` (ADR-0028) until ``selected`` ends on a
    sentence boundary, the available blocks run out, or the widen cap is hit.

    Shared by the single-chapter and cross-chapter paths in :func:`passage` —
    ``by_start`` is scoped to whichever chapter the widen is allowed to reach
    into (the resolved chapter itself, never a further one), so the boundary
    ADR-0028 describes ("in this chapter only") holds for both.

    Returns the possibly extended block list, its new end verse, and whether it
    stopped on a genuine sentence boundary rather than running out of room.
    """
    complete = True
    appended = 0
    while not _ends_sentence(_flatten(selected)):
        next_block = by_start.get(span_end + 1)
        if next_block is None or appended >= _MAX_FORWARD_WIDEN:
            complete = False
            break
        selected = [*selected, next_block]
        span_end = next_block.end
        appended += 1
    return selected, span_end, complete


def passage(reference: str, *, data_dir: Path = DEFAULT_ERV_DIR) -> Passage | None:
    """Return the verbatim ERV text for ``reference``, or ``None`` if unresolvable.

    Unresolvable means the reference is out of scope for the shared grammar (bare
    book, unparseable text), or its book, chapter, or verses are not present in the
    dataset — including any chapter a cross-chapter span passes through. A
    chapter-only reference returns the whole chapter. Poetry line breaks and
    indentation are collapsed to single spaces so the passage renders as one
    continuous line.

    Every ERV block *overlapping* the requested verses is returned, so a reference
    landing anywhere inside a merged block resolves to the whole block. The
    returned reference widens to the span actually covered; it is the caller's
    string untouched when nothing widened.

    A same-book reference crossing a chapter boundary (spec 0016, ADR-0079)
    resolves by gathering every block the span touches: the requested verses
    onward in the starting chapter, every chapter strictly between in full, and
    the requested verses up to the end verse in the ending chapter. Widening
    (merged block, or the ADR-0028 forward widen) can still move ``span_start``
    within the starting chapter and ``span_end`` within the ending chapter, but
    the chapters themselves never move — a widen never reaches a *third*
    chapter looking for a sentence boundary.
    """
    parsed = parse_reference(reference)
    if parsed is None:
        return None

    book = _load_book(parsed.book_slug, data_dir)
    if book is None:
        return None
    chapter = book.get(str(parsed.chapter))
    if chapter is None:
        return None

    all_blocks = _blocks(chapter)
    if parsed.start is None:
        resolved = reference  # chapter-only: no verse portion to widen
        segments = tuple(_flatten_segments(all_blocks))
        text = " ".join(s.text for s in segments if s.text).strip()
        return Passage(resolved, text, segments=segments)

    end = parsed.end if parsed.end is not None else parsed.start

    if parsed.end_chapter is None:
        selected = [b for b in all_blocks if b.start <= end and b.end >= parsed.start]
        if not selected:
            return None
        span_start, span_end = selected[0].start, max(b.end for b in selected)

        by_start = {b.start: b for b in all_blocks}
        selected, span_end, complete = _widen_forward(selected, span_end, by_start)

        resolved = (
            reference
            if (span_start, span_end) == (parsed.start, end)
            else widen_reference(reference, span_start, span_end)
        )
        verse_count = span_end - span_start + 1
        segments = tuple(_flatten_segments(selected))
        text = " ".join(s.text for s in segments if s.text).strip()
        return Passage(resolved, text, complete, segments, verse_count)

    # Cross-chapter (ADR-0079): the starting chapter's blocks were already loaded
    # above as `chapter`/`all_blocks`; load the ending chapter and every chapter
    # strictly between, in full — an absent one anywhere in the span is the same
    # "unresolvable" contract as a missing chapter today.
    end_chapter_dict = book.get(str(parsed.end_chapter))
    if end_chapter_dict is None:
        return None
    end_blocks = _blocks(end_chapter_dict)

    start_selected = [b for b in all_blocks if b.end >= parsed.start]
    if not start_selected:
        return None
    span_start = start_selected[0].start

    end_selected = [b for b in end_blocks if b.start <= end]
    if not end_selected:
        return None
    span_end = max(b.end for b in end_selected)

    middle_chapter_blocks: list[list[_Block]] = []
    for mid_chapter in range(parsed.chapter + 1, parsed.end_chapter):
        mid_dict = book.get(str(mid_chapter))
        if mid_dict is None:
            return None
        middle_chapter_blocks.append(_blocks(mid_dict))

    selected = [
        *start_selected,
        *(b for blocks in middle_chapter_blocks for b in blocks),
        *end_selected,
    ]

    by_start = {b.start: b for b in end_blocks}
    selected, span_end, complete = _widen_forward(selected, span_end, by_start)

    resolved = (
        reference
        if (span_start, span_end) == (parsed.start, end)
        else widen_cross_chapter_reference(
            reference, parsed.chapter, span_start, parsed.end_chapter, span_end
        )
    )
    verse_count = (
        (_chapter_length(chapter) - span_start + 1)
        + sum(max(b.end for b in blocks) for blocks in middle_chapter_blocks)
        + span_end
    )
    segments = tuple(_flatten_segments(selected))
    text = " ".join(s.text for s in segments if s.text).strip()
    return Passage(resolved, text, complete, segments, verse_count)


def reference_url(reference: str, *, data_dir: Path = DEFAULT_ERV_DIR) -> str | None:
    """Best-effort omnibible URL for ``reference`` (spec 0016, ADR-0079).

    Delegates to :func:`sermon_notes.scripture.reference_url` for anything that
    contract already links unchanged. A same-book reference crossing a chapter
    boundary is not linkable whole — the ``?v=`` query addresses one chapter only
    — so this links just its leading chapter, truncated at that chapter's own
    last verse, rather than yielding no link at all. Never returns a URL naming
    two chapters. A missing book or chapter still returns ``None``, the same
    "unresolvable" contract :func:`passage` uses.
    """
    parsed = parse_reference(reference)
    if parsed is None:
        return None
    if parsed.end_chapter is None:
        return scripture.reference_url(reference)

    book = _load_book(parsed.book_slug, data_dir)
    if book is None:
        return None
    chapter = book.get(str(parsed.chapter))
    if chapter is None or parsed.start is None:
        return None
    return scripture.chapter_url(
        parsed.book_slug, parsed.chapter, parsed.start, _chapter_length(chapter)
    )
