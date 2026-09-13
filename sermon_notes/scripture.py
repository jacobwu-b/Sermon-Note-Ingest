"""Scripture references: the canonical book set and omnibible links (spec 0010).

This module owns the 66-book name set used across the pipeline — :mod:`feed`
imports it for reference extraction — and turns a canonical reference string
("Romans 12:1-2") into an omnibible reader URL for that exact book, chapter,
and verse range.

The URL shape is the issue #26 contract for external link authors:
``/bible/{version}/{book}/{chapter}?v={start}[-{end}]``. The translation is
pinned to ERV. The book segment is the matched name slugged (spaces → hyphens);
the omnibible app resolves aliases to its canonical slug. A verse may carry a
trailing section-marker letter (e.g. "43a"); the marker is dropped from the
URL since the ``?v=`` range only addresses whole verses. References the
contract cannot represent — cross-chapter or cross-book ranges, or anything that
is not a single recognized book + chapter — return ``None`` so the caller can
fall back to plain text.

A same-book cross-chapter reference is parsed (see ``ParsedReference.end_chapter``)
even though :func:`reference_url` still cannot link it — the omnibible URL
addresses one chapter only, and no query string names two. :mod:`sermon_notes.erv`
builds on the parse this module already recognizes to offer a best-effort link for
that case (spec 0016, ADR-0079); this module's own contract is unchanged.
"""

from __future__ import annotations

import re
from typing import NamedTuple

# Canonical 66-book names (plus common variants) shared with feed.py extraction.
BOOK_NAMES = (
    "Genesis",
    "Exodus",
    "Leviticus",
    "Numbers",
    "Deuteronomy",
    "Joshua",
    "Judges",
    "Ruth",
    "1 Samuel",
    "2 Samuel",
    "1 Kings",
    "2 Kings",
    "1 Chronicles",
    "2 Chronicles",
    "Ezra",
    "Nehemiah",
    "Esther",
    "Job",
    "Psalms",
    "Psalm",
    "Proverbs",
    "Ecclesiastes",
    "Song of Solomon",
    "Song of Songs",
    "Isaiah",
    "Jeremiah",
    "Lamentations",
    "Ezekiel",
    "Daniel",
    "Hosea",
    "Joel",
    "Amos",
    "Obadiah",
    "Jonah",
    "Micah",
    "Nahum",
    "Habakkuk",
    "Zephaniah",
    "Haggai",
    "Zechariah",
    "Malachi",
    "Matthew",
    "Mark",
    "Luke",
    "John",
    "Acts",
    "Romans",
    "1 Corinthians",
    "2 Corinthians",
    "Galatians",
    "Ephesians",
    "Philippians",
    "Colossians",
    "1 Thessalonians",
    "2 Thessalonians",
    "1 Timothy",
    "2 Timothy",
    "Titus",
    "Philemon",
    "Hebrews",
    "James",
    "1 Peter",
    "2 Peter",
    "1 John",
    "2 John",
    "3 John",
    "Jude",
    "Revelation",
)

# omnibible reader base; ERV is the pinned translation for every link (issue #26).
# Host migrated from omnibible.vercel.app to the app's own domain (issue #516).
_BASE_URL = "https://omnibible.online/bible"
_VERSION = "erv"

# Dash variants that may separate a verse range: the ASCII hyphen-minus plus the
# Unicode hyphen / figure dash / en dash / em dash seen in pasted references.
# Exposed (with the alternation below) so reference matching has one source of
# truth — feed.py reuses both for free-text extraction (issue #41).
DASH_CHARS = "‐-―-"

# Book-name alternation, longest names first so "Song of Solomon" wins over "Song
# of Songs" and a bare "Song" never matches on its own.
_BOOK_ALTERNATION = "|".join(re.escape(b) for b in sorted(BOOK_NAMES, key=len, reverse=True))

# The core reference grammar: a known book, a chapter, and an optional single
# contiguous verse range within that chapter — or, when the range's end falls in
# a later chapter of the same book, a range across those two chapters ("Revelation
# 21:1-22:5"). A verse number may carry a trailing section-marker letter (e.g.
# "43a", "43b") splitting one verse into sub-parts; the marker is captured but
# dropped when building a URL, since omnibible's ?v= range only addresses whole
# verses. Un-anchored, with named groups, so callers can wrap it for whole-string
# parsing (here) or free-text scanning (feed.py) without re-deriving the book set
# or dash handling.
REFERENCE_PATTERN = (
    r"(?P<book>" + _BOOK_ALTERNATION + r")\s+(?P<chapter>\d+)"
    r"(?::(?P<start>\d+)[a-z]?"
    r"(?:\s*[" + DASH_CHARS + r"]\s*(?:(?P<end_chapter>\d+):)?(?P<end>\d+)[a-z]?)?"
    r")?"
)

# Anchored end-to-end, so trailing text past a well-formed reference (e.g. a
# third ":" or stray words) fails the match and the reference is treated as
# unparseable.
_REFERENCE_RE = re.compile(r"^" + REFERENCE_PATTERN + r"$")

# The five single-chapter books. Their customary citation drops the chapter — "Jude
# 3" means Jude 1:3, "Philemon 4-6" means Philemon 1:4-6 — so the bare number after
# the book is a verse, not a chapter. Slugs, to match the parsed book slug.
_SINGLE_CHAPTER_BOOKS = frozenset({"obadiah", "philemon", "2-john", "3-john", "jude"})

# Splits a reference into its leading book name and the remainder, to detect the
# single-chapter shorthand before the main grammar (which would misread the verse
# number as a chapter).
_BOOK_PREFIX_RE = re.compile(r"^(?P<book>" + _BOOK_ALTERNATION + r")\s+(?P<rest>\S.*)$")


def _expand_single_chapter(reference: str) -> str:
    """Insert the implicit ``1:`` into a single-chapter book's verse shorthand.

    "Jude 3" -> "Jude 1:3", "Philemon 4-6" -> "Philemon 1:4-6". Only when the book
    is single-chapter and the reference carries no explicit ``chapter:verse`` colon;
    everything else (including the fully-qualified "Jude 1:3") is returned unchanged.
    """
    match = _BOOK_PREFIX_RE.match(reference)
    if match is None:
        return reference
    book_slug = match.group("book").lower().replace(" ", "-")
    rest = match.group("rest")
    if book_slug in _SINGLE_CHAPTER_BOOKS and ":" not in rest:
        return f"{match.group('book')} 1:{rest}"
    return reference


class ParsedReference(NamedTuple):
    """A resolved reference: the shared unit both consumers key on.

    ``book_slug`` is the matched book name lowercased with spaces hyphenated (the
    omnibible path segment). ``chapter`` is the starting (or only) chapter number.
    ``start`` and ``end`` bound the verse range: both ``None`` for a chapter-only
    reference, ``end`` ``None`` for a single verse. A whole-verse range only —
    trailing section-marker letters (``43a``) are dropped, matching the URL
    contract.

    ``end_chapter`` is ``None`` for a reference contained in one chapter (the
    common case, and the only shape :func:`reference_url` can link) and the
    later chapter number when the range crosses a chapter boundary within the
    same book (e.g. ``chapter=21, end_chapter=22`` for "Revelation 21:1-22:5");
    ``end`` is then the verse within ``end_chapter``, not ``chapter``. A
    cross-*book* range has no parse path at all — the grammar names one book.
    """

    book_slug: str
    chapter: int
    start: int | None
    end: int | None
    end_chapter: int | None = None


def parse_reference(reference: str) -> ParsedReference | None:
    """Parse a canonical reference into its parts, or ``None`` if unparseable.

    The single source of truth for what a resolvable reference is, shared by
    :func:`reference_url` (link target) and :mod:`sermon_notes.erv` (verbatim
    text and, for a cross-chapter span, a best-effort link). Unparseable means
    the reference is not a single recognized book + chapter (a bare book name
    or unrecognized text), or a verse range whose end does not ascend past its
    start. A single-chapter book's chapterless shorthand ("Jude 3") is expanded
    to its implicit chapter 1 first.

    A same-book cross-chapter range ("Revelation 21:1-22:5") parses; a range
    naming the same chapter on both sides of the dash ("Luke 10:25-10:37") is
    read as the ordinary single-chapter form. A range whose end chapter is
    earlier than its start is out of scope, same as a non-ascending verse range.
    """
    match = _REFERENCE_RE.match(_expand_single_chapter(reference.strip()))
    if match is None:
        return None

    book_slug = match.group("book").lower().replace(" ", "-")
    chapter = int(match.group("chapter"))
    start = match.group("start")
    end = match.group("end")
    end_chapter = match.group("end_chapter")
    start_num = int(start) if start is not None else None
    end_num = int(end) if end is not None else None
    end_chapter_num = int(end_chapter) if end_chapter is not None else None
    if end_chapter_num == chapter:
        end_chapter_num = None  # written out redundantly; treat as one chapter
    if end_chapter_num is not None and end_chapter_num < chapter:
        return None  # a chapter range running backward is not a valid span
    if (
        end_chapter_num is None
        and start_num is not None
        and end_num is not None
        and end_num <= start_num
    ):
        return None  # a non-ascending single-chapter range is not a valid span
    return ParsedReference(book_slug, chapter, start_num, end_num, end_chapter_num)


def chapter_url(
    book_slug: str, chapter: int, start: int | None = None, end: int | None = None
) -> str:
    """Build the omnibible ERV URL for one already-resolved chapter and verse range.

    The single-chapter half of the issue #26 contract, factored out of
    :func:`reference_url` so a caller that has already resolved its own parts —
    :func:`sermon_notes.erv.reference_url`, truncating a cross-chapter span at a
    chapter boundary (spec 0016, ADR-0079) — can build a URL without reaching into
    this module's private constants.
    """
    url = f"{_BASE_URL}/{_VERSION}/{book_slug}/{chapter}"
    if start is None:
        return url
    if end is None:
        return f"{url}?v={start}"
    return f"{url}?v={start}-{end}"


def reference_url(reference: str) -> str | None:
    """Return the omnibible ERV URL for ``reference``, or ``None`` if out of scope.

    Out of scope means the reference is not a single recognized book + chapter
    (e.g. a bare book name, unrecognized text), a verse range whose end does not
    ascend past its start, or a range crossing a chapter boundary — the ``?v=``
    query addresses one chapter only, so a cross-chapter parse (see
    :attr:`ParsedReference.end_chapter`) is rejected here even though
    :func:`parse_reference` recognizes it. :func:`sermon_notes.erv.reference_url`
    offers a best-effort single-chapter link for that case instead.
    """
    parsed = parse_reference(reference)
    if parsed is None or parsed.end_chapter is not None:
        return None
    return chapter_url(parsed.book_slug, parsed.chapter, parsed.start, parsed.end)
