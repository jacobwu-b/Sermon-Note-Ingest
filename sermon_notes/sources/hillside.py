"""The Hillside Church (San Jose) source adapter: Subsplash podcast feed +
Sunday-only classification, with the sermon title read out of the description
(spec 0023, ADR-0013).

This is the Hillside ingestion boundary (CLAUDE.md §6): it owns the only feed
fetch for this source. Its feed url is supplied by
:func:`sources.enabled_adapters` from ``CHURCHES`` (ADR-0074) rather than
read here directly, and it follows the retry posture in PRD §11.1 — three
attempts with exponential backoff, then the source defers to the next schedule
rather than crashing. Subsplash needs no WAF-clearing User-Agent (unlike PBC).

Two things about this feed are unlike every other source, and both are
load-bearing:

**The sermon's name is not in ``<title>``.** The current title format is
``"{M.D.YY} | {scripture} | {speaker}"`` — e.g. ``"9.6.26 | Hebrews 13:7–17 |
Keith Crosby"`` — and the sermon's actual name is the first paragraph of the
HTML ``<description>`` (``"A Parting Conversation"``). Per the owner's
resolved decision (spec 0023), the description's first paragraph is the record
``title``; the raw feed title supplies the scripture references and a speaker
fallback instead. ``notes/`` is append-only, so this choice names every
Hillside artifact permanently.

**The title's date is the wrong signal to classify on.** This is the mirror
image of Westgate. Spec 0019 classified Westgate on its title date because its
``pubDate`` lagged the true service day 24% of the time. Here it is the title
date that drifts: of the 23 current-format items sampled, 22 agree with
``pubDate`` and one names the day *after* the actual Sunday. So this adapter
classifies on ``pubDate``, like North Point — importing Westgate's rule here
would import the wrong lesson from it.
"""

from __future__ import annotations

import html
import re
from datetime import datetime
from typing import Any, Callable

import feedparser

from sermon_notes.logging import get_logger
from sermon_notes.registry import Registry, SermonRecord
from sermon_notes.sources.base import FeedItem, PollResult, SourceAdapter
from sermon_notes.sources.feedbase import (
    FeedFetchError,
    entry_audio_url,
    entry_has_identity,
    extract_scripture_refs,
    fetch_feed,
    now,
    parse_pubdate,
    parse_pubdate_at,
)

logger = get_logger()

# The current feed title is "{M.D.YY} | {scripture} | {speaker}"; older archive
# items are a bare sermon title with no separator. Anything that is not exactly
# three segments contributes neither scripture refs nor a speaker fallback, and
# the item is classified and titled exactly the same way regardless (spec 0023).
_TITLE_SEP = " | "
_TITLE_SEGMENTS = 3
# email.utils weekday convention used throughout the feed layer: Sunday == 6.
_SUNDAY = 6
# The feed's own <guid> is an opaque 32-char hex string with no namespace (like
# North Point's and Lakepointe's), so this adapter namespaces it to guarantee no
# cross-source collision (spec 0023, issue #206).
_GUID_PREFIX = "hillside:"
# Subsplash writes the description as HTML paragraphs. Block-level boundaries
# become line breaks so the first paragraph can be taken as the title; the
# remaining tags are then dropped. This is per-church wiring and deliberately
# local to this adapter — Hillside is the only source whose description is HTML,
# and ADR-0013 puts per-church handling in the adapter, not in `feedbase`.
_BLOCK_BOUNDARY_RE = re.compile(r"</p\s*>|<br\s*/?>|</div\s*>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def _description_lines(raw_description: str) -> list[str]:
    """Split an HTML description into its non-empty plain-text lines.

    Block boundaries become newlines before the remaining tags are stripped, so
    ``"<p>A</p><p>B</p>"`` yields ``["A", "B"]`` rather than the run-together
    ``"AB"`` a bare tag-strip would produce.
    """
    text = _BLOCK_BOUNDARY_RE.sub("\n", raw_description)
    text = _TAG_RE.sub("", text)
    return [line.strip() for line in html.unescape(text).splitlines() if line.strip()]


def _split_title(raw_title: str) -> tuple[str | None, str | None]:
    """Split a current-format feed title into ``(scripture_segment, speaker)``.

    Returns ``(None, None)`` for any title that is not exactly three segments —
    the pre-2026-04 archive format, and whatever the church moves to next. The
    caller treats both as "no refs, no speaker fallback" rather than as an
    error, which is what keeps classification independent of the title grammar
    (spec 0023).
    """
    parts = raw_title.split(_TITLE_SEP)
    if len(parts) != _TITLE_SEGMENTS:
        return None, None
    _date, scripture_segment, speaker = (part.strip() for part in parts)
    return scripture_segment or None, speaker or None


def _entry_to_item(entry: Any) -> FeedItem:
    """Convert a feedparser entry into a :class:`FeedItem`.

    ``title`` is the description's first paragraph, falling back to the raw feed
    title when the description is empty — an empty description must never leave
    the record with an empty title, since that title names the artifact. The
    full description text is kept as the ``blurb``; scripture references come
    from the raw title's middle segment instead, run through the shared
    :func:`extract_scripture_refs` so the recognized book set and dash
    normalization keep one source of truth (issue #41).
    """
    raw_title = (entry.get("title") or "").strip()
    scripture_segment, title_speaker = _split_title(raw_title)
    lines = _description_lines(entry.get("summary") or "")
    published_on, weekday = parse_pubdate(entry.get("published"))
    return FeedItem(
        guid=_GUID_PREFIX + entry.get("id", ""),
        raw_title=raw_title,
        title=lines[0] if lines else raw_title,
        # The feed carries no series field, and the description's later paragraphs
        # are sometimes a part-marker and sometimes a subtitle — not reliably a
        # series name, so nothing is inferred (spec 0023).
        series=None,
        # itunes:author names the individual preacher (North Point's situation,
        # not PBC's); the title's trailing segment covers the items that omit it.
        speaker=(entry.get("author") or "").strip() or title_speaker,
        published_on=published_on,
        published_weekday=weekday,
        published_at=parse_pubdate_at(entry.get("published")),
        episode_url=entry.get("link") or "",
        audio_url=entry_audio_url(entry),
        blurb=" ".join(lines),
        scripture_refs=extract_scripture_refs(scripture_segment or ""),
    )


def parse_feed(content: bytes) -> tuple[list[FeedItem], int]:
    """Parse raw feed bytes into :class:`FeedItem` objects, in feed order.

    Returns ``(items, skipped)``: an entry with neither ``<guid>`` nor ``<link>``
    has no usable identity for the ledger's guid-keyed primary key, so it is
    skipped (logged) rather than defaulted to a shared, prefix-only guid that
    every other id-less item would also carry (issue #206) — the identity check
    runs on the raw entry, ahead of this adapter's namespace prefix, so the
    prefix can never mask the gap. ``skipped`` counts how many were dropped.
    """
    parsed = feedparser.parse(content)
    items: list[FeedItem] = []
    skipped = 0
    for entry in parsed.entries:
        if not entry_has_identity(entry):
            logger.warning(
                "skip identity-less feed item (no guid or link): %r",
                (entry.get("title") or "").strip(),
            )
            skipped += 1
            continue
        items.append(_entry_to_item(entry))
    return items, skipped


# --- classification (spec 0023) ---------------------------------------------


def is_main_sunday_sermon(item: FeedItem) -> bool:
    """Return whether ``item`` is Hillside's main Sunday sermon (spec 0023).

    A bare ``pubDate`` weekday check, with no denylist: the feed's only
    non-Sunday content is holiday services (Good Friday, Christmas Eve), which
    the weekday check already excludes, and no non-sermon series shares the
    feed. An item carrying no usable ``pubDate`` scores ``-1`` and so is never
    a match — nine live items are in that state.
    """
    return item.published_weekday == _SUNDAY


# --- adapter ------------------------------------------------------------


class HillsideAdapter(SourceAdapter):
    """Ingest Hillside Church (San Jose) sermons from the Subsplash feed (ADR-0013).

    The feed URL and fetch function are injectable so the HTTP boundary is
    mocked in tests; in production ``url`` is always supplied by
    :func:`sources.enabled_adapters` from ``CHURCHES`` (ADR-0074), and the
    fetch is the retrying :func:`fetch_feed`.
    """

    source = "hillside"

    def __init__(
        self,
        *,
        url: str | None = None,
        fetch: Callable[[str], bytes] = fetch_feed,
    ) -> None:
        self._url = url
        self._fetch = fetch

    def _to_record(self, item: FeedItem, *, now: str) -> SermonRecord:
        """Build a freshly-discovered :class:`SermonRecord` from a feed item."""
        return SermonRecord(
            guid=item.guid,
            source=self.source,
            episode_url=item.episode_url,
            audio_url=item.audio_url,
            published_on=item.published_on,
            published_at=item.published_at.isoformat() if item.published_at is not None else None,
            title=item.title,
            series=item.series,
            speaker=item.speaker,
            scripture_refs=list(item.scripture_refs),
            blurb=item.blurb,
            transcript_hash=None,
            state="discovered",
            artifact_path=None,
            first_seen_at=now,
            last_state_change_at=now,
            runs=[],
        )

    def poll(self, registry: Registry, *, limit: int | None = None) -> PollResult:
        """Fetch, parse, select, and upsert main Hillside Sunday sermons as ``discovered``.

        Sunday items are upserted as ``discovered`` (idempotent on guid, so a
        re-poll never duplicates); holiday services and undated items are
        excluded. When ``limit`` is given only the ``limit`` most recent
        qualifying sermons (by publication date) are upserted — the v1 working
        set (PRD §6.6) — while exclusion still covers the whole feed. A
        :class:`FeedFetchError` defers the source — no crash, no records
        written — leaving the next schedule to retry.
        """
        if self._url is None:
            raise ValueError(
                f"{type(self).__name__}.poll() requires a feed url; "
                "sources.enabled_adapters() always supplies one in production."
            )
        try:
            content = self._fetch(self._url)
        except FeedFetchError as exc:
            logger.warning("feed unavailable; deferring %s poll to next run: %s", self.source, exc)
            return PollResult(deferred=True)

        polled_at = now()
        items, id_less = parse_feed(content)
        included: list[FeedItem] = []
        excluded = id_less
        for item in items:
            if is_main_sunday_sermon(item):
                included.append(item)
            else:
                excluded += 1

        included.sort(key=lambda item: item.published_on, reverse=True)
        if limit is not None:
            included = included[:limit]
        discovered: list[str] = []
        publish_times: dict[str, datetime] = {}
        for item in included:
            registry.upsert(self._to_record(item, now=polled_at))
            discovered.append(item.guid)
            if item.published_at is not None:
                publish_times[item.guid] = item.published_at

        logger.info(
            "%s poll complete: %d discovered, %d excluded",
            self.source,
            len(discovered),
            excluded,
        )
        return PollResult(discovered=discovered, excluded=excluded, publish_times=publish_times)
