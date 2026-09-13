"""The Lakepointe Church (Rockwall, TX) source adapter: Megaphone podcast feed +
Sunday-only classification with a non-sermon denylist (spec 0022, ADR-0013).

This is the Lakepointe ingestion boundary (CLAUDE.md §6): it owns the only
feed fetch for this source. Its feed url is supplied by
:func:`sources.enabled_adapters` from ``CHURCHES`` (ADR-0074) rather than
read here directly, and it follows the retry posture in PRD §11.1 — three
attempts with exponential backoff, then the source defers to the next
schedule rather than crashing.

Lakepointe publishes through a Megaphone-hosted podcast feed
(``feeds.megaphone.fm/...``); like North Point and unlike PBC there is no WAF
to clear, so the shared default User-Agent is used.

Classification is North Point's rule plus a denylist. The feed's `pubDate` is
trustworthy — unlike Westgate's, which lags the service by a day 24% of the
time — so the weekday check runs on it directly. What North Point does not
have is a set of non-sermon series sharing the feed: "Bonus Podcast",
"Church At Home", and "Bonus Q&A" episodes. Every one of them published on a
non-Sunday across the sampled feed, so the weekday check alone would exclude
them today; they are named anyway, because a bonus episode that one week
publishes on a Sunday would otherwise become a sermon and spend an LLM
attempt on itself (spec 0022).

Two fields the other adapters populate are permanently empty here: the feed
emits no ``<link>`` and no ``<description>`` on any item, so ``episode_url``
and ``blurb`` are always ``""`` and ``scripture_refs`` is always empty at
ingest. The empty link is North Point's situation, resolved the same way
(ADR-0050): carry nothing rather than fabricate a URL.
"""

from __future__ import annotations

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

# The current feed title is "{title} | {series} | {speaker}"; an older archive
# item is a bare title with no separator at all. Anything that is not exactly
# three segments keeps the whole string as the title (spec 0022).
_TITLE_SEP = " | "
_TITLE_SEGMENTS = 3
# email.utils weekday convention used throughout the feed layer: Sunday == 6.
_SUNDAY = 6
# The feed's own <guid> is a bare UUID with no namespace (like North Point's hex
# string and Westgate's colon-delimited hash), so this adapter namespaces it to
# guarantee no cross-source collision (spec 0022, issue #206).
_GUID_PREFIX = "lakepointe:"
# Non-sermon series that share the feed. Matched on the title text rather than on
# segment count, because the bonus series uses a *double* pipe ("… || Bonus
# Podcast with Pastor …") and so already fails the three-segment split.
_NON_SERMON_MARKERS = (
    "|| Bonus Podcast",
    " | Church At Home | ",
    " | Bonus Q&A ",
)


def _split_title(raw_title: str) -> tuple[str, str | None, str | None]:
    """Split a ``"{title} | {series} | {speaker}"`` feed title into its three parts.

    A title with any other number of segments keeps the whole string as the
    title and leaves series and speaker null, which the record tolerates. That
    is what the pre-2026-03 archive format needs, and it is also what keeps
    classification independent of the title grammar: the church changed this
    format once already, and the next change must degrade rather than exclude
    (spec 0022).
    """
    parts = raw_title.split(_TITLE_SEP)
    if len(parts) != _TITLE_SEGMENTS:
        return raw_title, None, None
    title, series, speaker = (part.strip() for part in parts)
    return title, series or None, speaker or None


def _entry_to_item(entry: Any) -> FeedItem:
    """Convert a feedparser entry into a :class:`FeedItem`.

    ``blurb`` and ``episode_url`` come out empty for every Lakepointe item —
    the feed carries neither — so ``scripture_refs`` is empty at ingest too.
    Both are read through the same accessors the other adapters use rather than
    hard-coded to ``""``, so the day the feed starts carrying them, they land.
    """
    raw_title = (entry.get("title") or "").strip()
    title, series, speaker = _split_title(raw_title)
    published_on, weekday = parse_pubdate(entry.get("published"))
    blurb = (entry.get("summary") or "").strip()
    return FeedItem(
        guid=_GUID_PREFIX + entry.get("id", ""),
        raw_title=raw_title,
        title=title,
        series=series,
        # itunes:author names the church ("Lakepointe Church"), not the preacher,
        # so unlike North Point the author field is no help — the title's third
        # segment is the only speaker source this feed has (spec 0022).
        speaker=speaker,
        published_on=published_on,
        published_weekday=weekday,
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
    prefix can never mask the gap. Lakepointe is doubly exposed here: its
    ``<link>`` is empty on every item, so feedparser's usual ``<guid>``-less
    fallback has nothing to fall back to. ``skipped`` counts how many were
    dropped.
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


# --- classification (spec 0022) ---------------------------------------------


def is_main_sunday_sermon(item: FeedItem) -> bool:
    """Return whether ``item`` is Lakepointe's main Sunday sermon (spec 0022).

    An item counts when it published on a Sunday and its title carries none of
    the feed's non-sermon series markers. The weekday check alone matches every
    sampled item's verdict; the denylist is what keeps that true if one of those
    series ever publishes on a Sunday.
    """
    if item.published_weekday != _SUNDAY:
        return False
    return not any(marker in item.raw_title for marker in _NON_SERMON_MARKERS)


# --- adapter ------------------------------------------------------------


class LakepointeAdapter(SourceAdapter):
    """Ingest Lakepointe Church sermons from the Megaphone feed (ADR-0013).

    The feed URL and fetch function are injectable so the HTTP boundary is
    mocked in tests; in production ``url`` is always supplied by
    :func:`sources.enabled_adapters` from ``CHURCHES`` (ADR-0074), and the
    fetch is the retrying :func:`fetch_feed`.
    """

    source = "lakepointe"

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
        """Fetch, parse, select, and upsert main Lakepointe Sunday sermons as ``discovered``.

        Sunday items outside the non-sermon denylist are upserted as
        ``discovered`` (idempotent on guid, so a re-poll never duplicates);
        everything else is excluded. When ``limit`` is given only the ``limit``
        most recent qualifying sermons (by publication date) are upserted — the
        v1 working set (PRD §6.6) — while exclusion still covers the whole feed.
        A :class:`FeedFetchError` defers the source — no crash, no records
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
