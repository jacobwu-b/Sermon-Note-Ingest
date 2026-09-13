"""The WestGate Church (San Jose) source adapter: Squarespace podcast feed +
title-date Sunday classification (spec 0019, ADR-0013).

This is the Westgate ingestion boundary (CLAUDE.md §6): it owns the only
feed fetch for this source. Its feed url is supplied by
:func:`sources.enabled_adapters` from ``CHURCHES`` (ADR-0074) rather than
read here directly, and it follows the retry posture in PRD §11.1 — three
attempts with exponential backoff, then the source defers to the next
schedule rather than crashing.

Westgate publishes through a Squarespace-hosted podcast feed
(``westgatechurch.org/westgate-teaching?format=rss``); no WAF-clearing
User-Agent is needed (unlike PBC) — the shared default clears it fine.

Unlike every other adapter, Westgate does **not** classify off the feed's
``pubDate``: a live sample of 300 feed items found 24% whose ``pubDate``,
converted to Pacific time, lands on the day *after* the actual Sunday
service — Squarespace's publish step regularly lags. The feed's title
instead embeds the true service date directly (``"{series} | {title} |
{Month D[D], YYYY}"``, or without a series segment), and a live sample of
its current-format items found that trailing date is a Sunday 100% of the
time — so that's the signal this adapter classifies on, not ``pubDate``
(spec 0019).
"""

from __future__ import annotations

import re
from datetime import date, datetime
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
    parse_pubdate_at,
)

logger = get_logger()

# The feed's guid is an opaque, unnamespaced colon-delimited hash (unlike Menlo's
# "podbean-ep-..." or PBC's "https://pbc.org?..." guids), so this adapter
# namespaces it itself to guarantee no cross-source collision (spec 0019, issue #206).
_GUID_PREFIX = "westgate:"

# The title's trailing "{Month D[D], YYYY}" segment names the actual service date
# (spec 0019) — e.g. "The Gospel of John (Part 2) | To Whom Shall We Go | August 16,
# 2026". Zero-padded and non-padded days both appear in the live feed.
_TITLE_DATE_RE = re.compile(r"^(?P<rest>.+) \| (?P<date>[A-Za-z]+ \d{1,2}, \d{4})$")
_TITLE_DATE_FMT = "%B %d, %Y"

# A distinct kids/family program that happens to publish on a Sunday — excluded per
# the owner's resolved decision (spec 0019), not the main congregational teaching.
_NEXTGEN_PREFIX = "NextGen Sunday"


def _parse_title(raw_title: str) -> tuple[str, str | None, date | None]:
    """Split ``raw_title`` into ``(title, series, service_date)``.

    ``service_date`` is the parsed trailing date, or ``None`` when the title
    doesn't match the current format (an older archive item, or something
    unrecognized) — callers treat a ``None`` date as unclassifiable and exclude
    the item (spec 0019: no backfill, so only the current format matters).
    """
    match = _TITLE_DATE_RE.match(raw_title)
    if not match:
        return raw_title, None, None
    try:
        service_date = datetime.strptime(match.group("date"), _TITLE_DATE_FMT).date()
    except ValueError:
        return raw_title, None, None
    rest = match.group("rest")
    series, sep, title = rest.partition(" | ")
    if not sep:
        return rest, None, service_date
    return title, series, service_date


def _entry_to_item(entry: Any) -> FeedItem:
    """Convert a feedparser entry into a :class:`FeedItem`.

    ``published_on`` and ``published_weekday`` are derived from the title's
    parsed service date, not the feed's ``pubDate`` (spec 0019) — a title that
    fails to parse leaves both blank, matching :func:`~sermon_notes.sources.
    feedbase.parse_pubdate`'s "never mistaken for a Sunday" convention for an
    undated item. ``published_at`` still comes from the feed's own ``pubDate``:
    it measures actual feed availability for the attempt-cap (ADR-0031), which
    is a different instant than the nominal service date.
    """
    raw_title = (entry.get("title") or "").strip()
    title, series, service_date = _parse_title(raw_title)
    published_on = service_date.isoformat() if service_date is not None else ""
    published_weekday = service_date.weekday() if service_date is not None else -1
    blurb = (entry.get("summary") or "").strip()
    speaker = (entry.get("author") or "").strip() or None
    return FeedItem(
        guid=_GUID_PREFIX + entry.get("id", ""),
        raw_title=raw_title,
        title=title,
        series=series,
        speaker=speaker,
        published_on=published_on,
        published_weekday=published_weekday,
        published_at=parse_pubdate_at(entry.get("published")),
        episode_url=entry.get("link") or "",
        audio_url=entry_audio_url(entry),
        blurb=blurb,
        scripture_refs=extract_scripture_refs(blurb),
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


# --- classification (spec 0019) ---------------------------------------------


def is_main_sunday_sermon(item: FeedItem) -> bool:
    """Return whether ``item`` is Westgate's main Sunday sermon (spec 0019).

    An item counts when its title's trailing date parsed to a Sunday
    (``published_weekday == 6``, the same email.utils convention every other
    adapter uses) and its title isn't a "NextGen Sunday" item — a distinct
    kids/family program excluded per the owner's resolved decision. An item
    whose title didn't parse a date at all (``published_weekday == -1``) is
    never a match.
    """
    return item.published_weekday == 6 and not item.raw_title.startswith(_NEXTGEN_PREFIX)


# --- adapter ------------------------------------------------------------


class WestgateAdapter(SourceAdapter):
    """Ingest WestGate Church (San Jose) sermons from the Squarespace feed (ADR-0013).

    The feed URL and fetch function are injectable so the HTTP boundary is
    mocked in tests; in production ``url`` is always supplied by
    :func:`sources.enabled_adapters` from ``CHURCHES`` (ADR-0074), and the
    fetch is the retrying :func:`fetch_feed`.
    """

    source = "westgate"

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
        """Fetch, parse, select, and upsert main Westgate Sunday sermons as ``discovered``.

        Sunday items are upserted as ``discovered`` (idempotent on guid, so a
        re-poll never duplicates) whether or not their audio enclosure has
        appeared yet — a missing enclosure defers at download time instead
        (ADR-0009), needing no special handling here (spec 0019). Non-Sunday
        and "NextGen Sunday" items are excluded. When ``limit`` is given only
        the ``limit`` most recent qualifying sermons (by service date) are
        upserted — the v1 working set (PRD §6.6) — while exclusion still
        covers the whole feed. A :class:`FeedFetchError` defers the source —
        no crash, no records written — leaving the next schedule to retry.
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
