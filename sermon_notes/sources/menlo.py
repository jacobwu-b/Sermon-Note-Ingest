"""The Menlo Church source adapter: Podbean RSS feed + Menlo classification (ADR-0013).

This is the Menlo ingestion boundary (CLAUDE.md §6): it owns the only RSS fetch
for this source. Its feed url is supplied by :func:`sources.enabled_adapters`
from ``CHURCHES`` (ADR-0074) rather than read here directly, and it follows
the retry posture in PRD §11.1 — three attempts with exponential backoff, then
the source defers to the next schedule rather than crashing.

The classification rule is Menlo-specific (PRD §5.3): the feed mixes main Sunday
sermons with the Legacy (traditional) service and the Midweek Podcast; only main
Sunday sermons are in scope. Legacy and Midweek content is EXCLUDEd (recognized by
title only — main-sermon blurbs cross-promote the Midweek Podcast by name, so the
blurb is not a reliable exclude signal, issue #113); any item published on a
Sunday that is neither is INCLUDEd; anything else is AMBIGUOUS and skip-and-flagged.
Within a Sunday-anchored week with no Sunday item, the AMBIGUOUS item closest to
Sunday is promoted as the late-posted main sermon (ADR-0006).

:class:`MenloPodbeanAdapter` ties the unit together: fetch → parse → classify →
upsert the main Sunday sermons as ``discovered`` (idempotent on the feed ``guid``).
"""

from __future__ import annotations

import dataclasses
import datetime
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Iterable

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

# The Legacy service names itself in the final title suffix (PRD §5.3).
_LEGACY_SUFFIX = "Menlo Church Legacy Service"
# The Midweek Podcast names itself in the title or description.
_MIDWEEK_MARKER = "menlo midweek podcast"
# email.utils weekday convention used throughout the feed layer: Sunday == 6.
_SUNDAY = 6


def _split_title(raw_title: str) -> tuple[str, str | None, str | None]:
    """Split a ``Title | Series | Speaker`` feed title into its parts.

    The feed sometimes appends a trailing suffix (``Title | Series | Speaker |
    Menlo Church Podcasts``); that fourth segment is discarded when present. A
    title with no series (``Title | Speaker``) still yields a speaker. Anything
    shorter keeps the leading segment as the title and leaves the rest unparsed.
    """
    parts = [p.strip() for p in raw_title.split("|")]
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], None, parts[1]
    return (parts[0] if parts else ""), None, None


def _entry_to_item(entry: Any) -> FeedItem:
    """Convert a feedparser entry into a :class:`FeedItem`."""
    raw_title = (entry.get("title") or "").strip()
    title, series, speaker = _split_title(raw_title)
    published_on, weekday = parse_pubdate(entry.get("published"))
    blurb = (entry.get("summary") or "").strip()
    return FeedItem(
        guid=entry.get("id", ""),
        raw_title=raw_title,
        title=title,
        series=series,
        speaker=speaker,
        published_on=published_on,
        published_weekday=weekday,
        published_at=parse_pubdate_at(entry.get("published")),
        episode_url=entry.get("link", ""),
        audio_url=entry_audio_url(entry),
        blurb=blurb,
        scripture_refs=extract_scripture_refs(blurb),
    )


def parse_feed(content: bytes) -> tuple[list[FeedItem], int]:
    """Parse raw feed bytes into :class:`FeedItem` objects, in feed order.

    Returns ``(items, skipped)``: an entry with neither ``<guid>`` nor ``<link>``
    has no usable identity for the ledger's guid-keyed primary key, so it is
    skipped (logged) rather than defaulted to an empty, collision-prone guid
    (issue #206); ``skipped`` counts how many were dropped this way.
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


# --- classification (PRD §5.3, ADR-0006) -----------------------------------


class Classification(Enum):
    """The verdict for a feed item (PRD §5.3)."""

    INCLUDE = "include"
    EXCLUDE = "exclude"
    AMBIGUOUS = "ambiguous"


def classify(item: FeedItem) -> Classification:
    """Return the classification for ``item`` per the PRD §5.3 rules.

    Order matters: Legacy and Midweek are recognized first so their content can
    never fall through to INCLUDE — including a Midweek item that happens to
    carry a Sunday date. A Sunday item that is neither is a main sermon
    regardless of its title suffix; everything else is AMBIGUOUS and left for
    skip-and-flag.
    """
    if item.raw_title.strip().endswith(_LEGACY_SUFFIX):
        return Classification.EXCLUDE
    if _MIDWEEK_MARKER in item.raw_title.lower():
        return Classification.EXCLUDE
    if item.published_weekday == _SUNDAY:
        return Classification.INCLUDE
    return Classification.AMBIGUOUS


def _days_since_sunday(item: FeedItem) -> int:
    """Distance of ``item`` from the Sunday that opens its week (Sunday == 0).

    The feed states weekdays in the email.utils convention (Monday == 0 …
    Sunday == 6); shifting by one wraps Sunday to 0 so the main sermon's preferred
    day sorts first and each later weekday follows in order.
    """
    return (item.published_weekday + 1) % 7


def _week_anchor(item: FeedItem) -> datetime.date:
    """The date of the Sunday that opens ``item``'s liturgical week (Sun–Sat)."""
    published = datetime.date.fromisoformat(item.published_on)
    return published - datetime.timedelta(days=_days_since_sunday(item))


def classify_feed(items: Iterable[FeedItem]) -> dict[str, Classification]:
    """Classify a whole feed, promoting a late-posted main sermon per week (PRD §5.3).

    Each item first gets its per-item :func:`classify` verdict. The church
    sometimes posts the main sermon a day or two after Sunday, where the per-item
    rule can only mark it AMBIGUOUS. So within each Sunday-anchored week that has no
    INCLUDE (no Sunday item), the AMBIGUOUS item closest to Sunday — walking
    Sunday → Monday → … → Saturday — is promoted to INCLUDE as the main sermon; any
    other AMBIGUOUS items that week stay flagged. Weeks already claimed by a Sunday
    sermon are left untouched, and EXCLUDE (Legacy/Midweek) items never participate.
    Returns a verdict per ``guid``.
    """
    items = list(items)
    verdicts = {item.guid: classify(item) for item in items}

    # An undated/malformed item (published_on == "", weekday == -1) is never Sunday,
    # so it is always AMBIGUOUS. It has no week anchor, so it can neither claim a week
    # nor be promoted — it must stay flagged. Excluding it here keeps _week_anchor from
    # raising on fromisoformat("") and crashing the whole poll (issue #33).
    weeks_with_include: set[datetime.date] = {
        _week_anchor(item)
        for item in items
        if item.published_on and verdicts[item.guid] is Classification.INCLUDE
    }
    ambiguous_by_week: dict[datetime.date, list[FeedItem]] = defaultdict(list)
    for item in items:
        if item.published_on and verdicts[item.guid] is Classification.AMBIGUOUS:
            ambiguous_by_week[_week_anchor(item)].append(item)

    for anchor, candidates in ambiguous_by_week.items():
        if anchor in weeks_with_include:
            continue
        winner = min(candidates, key=lambda it: (_days_since_sunday(it), it.published_on, it.guid))
        verdicts[winner.guid] = Classification.INCLUDE

    return verdicts


# --- adapter ---------------------------------------------------------------


class MenloPodbeanAdapter(SourceAdapter):
    """Ingest Menlo Church sermons from the Podbean RSS feed (ADR-0013).

    The feed URL and fetch function are injectable so the HTTP boundary is mocked
    in tests; in production ``url`` is always supplied by
    :func:`sources.enabled_adapters` from ``CHURCHES`` (ADR-0074), and the
    fetch is the retrying :func:`fetch_feed`.
    """

    source = "menlo"

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
        """Fetch, parse, classify, and upsert main Menlo sermons as ``discovered``.

        Main Sunday sermons are upserted as ``discovered`` (idempotent on guid, so a
        re-poll never duplicates); ambiguous items are skipped and flagged; Legacy
        and Midweek items are excluded. When ``limit`` is given only the ``limit``
        most recent qualifying sermons (by publication date) are upserted — the v1
        working set (PRD §6.6) — while flagging and exclusion still cover the whole
        feed. A :class:`FeedFetchError` defers the source — no crash, no records
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
        verdicts = classify_feed(items)
        included: list[FeedItem] = []
        flagged: list[str] = []
        excluded = id_less
        for item in items:
            verdict = verdicts[item.guid]
            if verdict is Classification.INCLUDE:
                if item.published_weekday != _SUNDAY:
                    # A promoted late-posted item (ADR-0006): the feed's own pubDate is
                    # an internal detail, never a user-visible date (#479) — the sermon
                    # was preached on the Sunday classify_feed anchored it to.
                    item = dataclasses.replace(item, published_on=_week_anchor(item).isoformat())
                included.append(item)
            elif verdict is Classification.AMBIGUOUS:
                logger.warning(
                    "skip-and-flag ambiguous feed item: %r (%s)", item.raw_title, item.guid
                )
                flagged.append(item.guid)
            else:
                excluded += 1

        included.sort(key=lambda item: item.published_on, reverse=True)
        if limit is not None:
            included = included[:limit]
        discovered: list[str] = []
        publish_times: dict[str, datetime.datetime] = {}
        for item in included:
            registry.upsert(self._to_record(item, now=polled_at))
            discovered.append(item.guid)
            if item.published_at is not None:
                publish_times[item.guid] = item.published_at

        logger.info(
            "%s poll complete: %d discovered, %d flagged, %d excluded",
            self.source,
            len(discovered),
            len(flagged),
            excluded,
        )
        return PollResult(
            discovered=discovered, flagged=flagged, excluded=excluded, publish_times=publish_times
        )
