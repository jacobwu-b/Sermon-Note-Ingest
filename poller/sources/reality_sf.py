"""Reality San Francisco: Podbean feed, earliest upload of each week.

``pubDate`` is the upload instant, and the Sunday sermon usually goes up on
Monday (546 of 999 live items), sometimes Tuesday, and once (Easter 2025) the
Wednesday after. So ``preached_on`` is the Sunday on or before the upload, in
the church's own timezone, as for Grace Presbyterian.

The feed also carries the church's side programs, uploaded later in the same
week: the 2020–21 numbered podcast "Episode N", daily "BREAD" readings,
"Still Processing" conversations, Lent and fasting reflections, interviews and
letters. Classification keeps, per Sunday-anchored week, the earliest upload
that is not a named side program. A weekday window alone would drop the
Tuesday and Wednesday sermons, and a denylist alone would admit the
interviews; across the 999-item feed this rule keeps one sermon a week with no
known false positive since 2021. Lent reflections are named because they go
up on Monday alongside the sermon.

``itunes:author`` names the church; the speaker leads the blurb ("Ruthie Kim
teaches from Esther 4…"), sometimes after a series clause.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import feedparser

from poller.net import FeedFetchError, fetch_feed
from poller.sources.base import ExcludedItem, PollResult, SermonItem, SourceAdapter
from poller.sources.common import (
    entry_audio_url,
    entry_has_identity,
    excluded_entry,
    excluded_item,
    parse_pubdate_at,
    sunday_on_or_before,
)

_GUID_PREFIX = "reality_sf:"
_CHURCH_TZ = ZoneInfo("America/Los_Angeles")

_NON_SERMON_RE = re.compile(r"still processing|reflection|letter from|^bread\b|^episode \d", re.IGNORECASE)

_TAG_RE = re.compile(r"<[^>]+>")
_SPEAKER_RE = re.compile(
    r"^(?:In our series [^,]+,\s*)?(?:Dr\.?\s+)?"
    r"(?P<name>[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.-]*){1,2})\s+(?:teaches|taught|preaches|preached)\b"
)


def is_non_sermon_title(raw_title: str) -> bool:
    """A title naming one of the feed's side programs rather than a Sunday sermon."""
    return bool(_NON_SERMON_RE.search(raw_title))


def _speaker_from_blurb(blurb: str) -> str | None:
    match = _SPEAKER_RE.match(_TAG_RE.sub(" ", blurb).strip())
    return match.group("name") if match else None


def _entry_to_item(entry: Any) -> SermonItem:
    raw_title = (entry.get("title") or "").strip()
    feed_published_at = parse_pubdate_at(entry.get("published"))
    blurb = (entry.get("summary") or "").strip()
    preached_on = (
        sunday_on_or_before(feed_published_at.astimezone(_CHURCH_TZ).date()).isoformat()
        if feed_published_at is not None
        else ""
    )
    return SermonItem(
        guid=_GUID_PREFIX + entry.get("id", ""),
        title=raw_title,
        raw_title=raw_title,
        series=None,
        speaker=_speaker_from_blurb(blurb),
        preached_on=preached_on,
        feed_published_at=feed_published_at,
        episode_url=entry.get("link") or "",
        audio_url=entry_audio_url(entry),
        blurb=blurb,
    )


def _parse(content: bytes) -> tuple[list[SermonItem], list[ExcludedItem]]:
    parsed = feedparser.parse(content)
    items: list[SermonItem] = []
    skipped: list[ExcludedItem] = []
    for entry in parsed.entries:
        if not entry_has_identity(entry):
            skipped.append(excluded_entry(entry))
            continue
        items.append(_entry_to_item(entry))
    return items, skipped


def select_sermons(items: list[SermonItem]) -> tuple[list[SermonItem], list[SermonItem]]:
    """Split items into ``(sermons, rest)``: each week's earliest non-side-program upload.

    Items keep feed order. An item with no ``pubDate`` cannot be placed in a week
    and is never a sermon.
    """
    earliest: dict[str, tuple[datetime, SermonItem]] = {}
    for item in items:
        if item.feed_published_at is None or is_non_sermon_title(item.raw_title):
            continue
        current = earliest.get(item.preached_on)
        if current is None or item.feed_published_at < current[0]:
            earliest[item.preached_on] = (item.feed_published_at, item)
    chosen = {id(item) for _, item in earliest.values()}
    sermons = [item for item in items if id(item) in chosen]
    rest = [item for item in items if id(item) not in chosen]
    return sermons, rest


class RealitySfAdapter(SourceAdapter):
    """Ingest Reality San Francisco sermons from its Podbean feed."""

    source = "reality_sf"
    # Measured since 2025: sermons land 42 h (median) to 93 h after Sunday 00:00 UTC.
    publish_grace = timedelta(hours=120)

    def poll(self) -> PollResult:
        try:
            content = fetch_feed(self.url)
        except FeedFetchError:
            return PollResult(deferred=True)

        items, id_less = _parse(content)
        sermons, rest = select_sermons(items)
        return PollResult(items=sermons, excluded=id_less + [excluded_item(item) for item in rest])
