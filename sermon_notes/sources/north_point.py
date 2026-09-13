"""The North Point Community Church source adapter: Sardius podcast feed +
Sunday-only classification (spec 0015, ADR-0013).

This is the North Point ingestion boundary (CLAUDE.md §6): it owns the only
feed fetch for this source. Its feed url is supplied by
:func:`sources.enabled_adapters` from ``CHURCHES`` (ADR-0074) rather than
read here directly, and it follows the retry posture in PRD §11.1 — three
attempts with exponential backoff, then the source defers to the next
schedule rather than crashing.

North Point publishes through a Sardius-hosted podcast feed
(``api.sardius.media/feeds/...``). Unlike PBC there is no WAF to clear (no
special User-Agent needed) and unlike PBC the feed already carries the
per-episode speaker in ``<itunes:author>``, so no sermons-page scrape
fallback is needed.

North Point's classification is the simplest of the three churches (spec
0015): the feed has no service-segment or title marker distinguishing its
main Sunday teaching from guest-pastor content the way PBC's ``Main_Service``
segment does — and per the owner's resolved decision, it doesn't need one.
Any sermon delivered at the main campus on a Sunday is in scope regardless of
who is teaching; only non-Sunday content (e.g. a midweek Christmas Eve
special) is excluded. A spike against the live feed (100 most recent items)
found 99/100 published on a Sunday, confirming a bare weekday check is
sufficient with no denylist.

:class:`NorthPointAdapter` ties the unit together: fetch → parse → select
Sunday items → upsert as ``discovered`` (idempotent on the namespaced guid).
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

# The feed titles a sermon as "{title} // {speaker}"; this splits the trailing
# speaker credit. Unlike PBC's " - {series}" suffix, North Point has no clean
# series delimiter, so series is left null.
_TITLE_SPEAKER_SEP = " // "
# email.utils weekday convention used throughout the feed layer: Sunday == 6.
_SUNDAY = 6
# The feed's own <guid> is an opaque, unnamespaced hex string (unlike Menlo's
# "podbean-ep-..." or PBC's "https://pbc.org?..." guids), so this adapter
# namespaces it itself to guarantee no cross-source collision (spec 0015).
_GUID_PREFIX = "north_point:"


def _split_title(raw_title: str) -> tuple[str, str | None]:
    """Split a ``"{title} // {speaker}"`` feed title into its title and speaker parts.

    A title with no such suffix keeps the whole string as the title and leaves
    the speaker null, which the record tolerates.
    """
    title, sep, speaker = raw_title.rpartition(_TITLE_SPEAKER_SEP)
    if not sep:
        return raw_title, None
    return title, speaker


def _entry_to_item(entry: Any) -> FeedItem:
    """Convert a feedparser entry into a :class:`FeedItem`."""
    raw_title = (entry.get("title") or "").strip()
    title, title_speaker = _split_title(raw_title)
    published_on, weekday = parse_pubdate(entry.get("published"))
    blurb = (entry.get("summary") or "").strip()
    # itunes:author names the individual preacher (unlike PBC, which names the
    # church); fall back to the title's trailing speaker credit if absent.
    speaker = (entry.get("author") or "").strip() or title_speaker
    return FeedItem(
        guid=_GUID_PREFIX + entry.get("id", ""),
        raw_title=raw_title,
        title=title,
        series=None,
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


# --- classification (spec 0015) ---------------------------------------------


def is_main_sunday_sermon(item: FeedItem) -> bool:
    """Return whether ``item`` is North Point's main Sunday sermon (spec 0015).

    Per the owner's resolved decision, any item published on a Sunday counts
    regardless of speaker — guest teaching pastors included; only non-Sunday
    content is excluded.
    """
    return item.published_weekday == _SUNDAY


# --- adapter ------------------------------------------------------------


class NorthPointAdapter(SourceAdapter):
    """Ingest North Point Community Church sermons from the Sardius feed (ADR-0013).

    The feed URL and fetch function are injectable so the HTTP boundary is
    mocked in tests; in production ``url`` is always supplied by
    :func:`sources.enabled_adapters` from ``CHURCHES`` (ADR-0074), and the
    fetch is the retrying :func:`fetch_feed`.
    """

    source = "north_point"

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
        """Fetch, parse, select, and upsert main North Point Sunday sermons as ``discovered``.

        Sunday items are upserted as ``discovered`` (idempotent on guid, so a
        re-poll never duplicates); non-Sunday items are excluded. When
        ``limit`` is given only the ``limit`` most recent qualifying sermons
        (by publication date) are upserted — the v1 working set (PRD §6.6) —
        while exclusion still covers the whole feed. A :class:`FeedFetchError`
        defers the source — no crash, no records written — leaving the next
        schedule to retry.
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
