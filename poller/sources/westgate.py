"""WestGate Church (San Jose): Squarespace podcast feed, title-date classification.

Unlike every other adapter, Westgate does not classify off the feed's
``pubDate``: a live sample of 300 feed items found 24% whose ``pubDate``
lands on the day *after* the actual Sunday service — Squarespace's publish
step regularly lags. The feed's title instead embeds the true service date
directly (``"{series} | {title} | {Month D[D], YYYY}"``), and that trailing
date is a Sunday 100% of the time in a live sample — so that's the signal
this adapter classifies on, not ``pubDate``.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any

import feedparser

from poller.net import FeedFetchError, fetch_feed
from poller.sources.base import ExcludedItem, PollResult, SermonItem, SourceAdapter
from poller.sources.common import (
    entry_audio_url,
    entry_has_identity,
    excluded_entry,
    excluded_item,
    is_service_date,
    parse_pubdate_at,
)

_GUID_PREFIX = "westgate:"

# e.g. "The Gospel of John (Part 2) | To Whom Shall We Go | August 16, 2026". Titles are
# typed by hand, so the live feed also varies the spacing around "|", abbreviates the month
# ("Sep. 6, 2026") and drops the comma ("March 09 2025"); each variant once dropped a sermon.
_TITLE_DATE_RE = re.compile(
    r"^(?P<rest>.+?)\s*\|\s*(?P<month>[A-Za-z]+)\.?\s+(?P<day>\d{1,2}),?\s+(?P<year>\d{4})\s*$"
)
_SEPARATOR_RE = re.compile(r"\s*\|\s*")
_TITLE_DATE_FMTS = ("%B %d %Y", "%b %d %Y")

# A distinct kids/family program that happens to publish on a Sunday —
# excluded per the owner's resolved decision, not the main congregational teaching.
_NEXTGEN_PREFIX = "NextGen Sunday"


def _parse_title(raw_title: str) -> tuple[str, str | None, date | None]:
    """Split into ``(title, series, service_date)``; ``service_date`` is ``None``
    when the title doesn't match the current format (an item is then unclassifiable
    and excluded — no backfill of the old format)."""
    match = _TITLE_DATE_RE.match(raw_title)
    if not match:
        return raw_title, None, None
    service_date = _parse_title_date(f"{match['month']} {match['day']} {match['year']}")
    if service_date is None:
        return raw_title, None, None
    rest = match.group("rest")
    parts = _SEPARATOR_RE.split(rest, maxsplit=1)
    if len(parts) == 1:
        return rest, None, service_date
    series, title = parts
    return title, series, service_date


def _parse_title_date(text: str) -> date | None:
    """``text`` (``"<month> <day> <year>"``) as a date, with the month full or abbreviated."""
    for fmt in _TITLE_DATE_FMTS:
        try:
            # Only the calendar date is kept (.date() below), so the naive instant
            # strptime produces is never used as a real datetime.
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def _entry_to_item(entry: Any) -> SermonItem:
    """``preached_on`` comes from the title's parsed service date, not
    ``pubDate``; ``feed_published_at`` still comes from the feed's own ``pubDate`` — it
    measures actual feed availability, a different instant than the nominal
    service date."""
    raw_title = (entry.get("title") or "").strip()
    title, series, service_date = _parse_title(raw_title)
    preached_on = service_date.isoformat() if service_date is not None else ""
    return SermonItem(
        guid=_GUID_PREFIX + entry.get("id", ""),
        title=title,
        raw_title=raw_title,
        series=series,
        speaker=(entry.get("author") or "").strip() or None,
        preached_on=preached_on,
        feed_published_at=parse_pubdate_at(entry.get("published")),
        episode_url=entry.get("link") or "",
        audio_url=entry_audio_url(entry),
        blurb=(entry.get("summary") or "").strip(),
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


def is_main_sunday_sermon(raw_title: str, preached_on: str) -> bool:
    """A title date that parsed to a service date (a Sunday, or a special service such as
    Good Friday), excluding the "NextGen Sunday" program."""
    if not preached_on or raw_title.startswith(_NEXTGEN_PREFIX):
        return False
    return is_service_date(date.fromisoformat(preached_on))


class WestgateAdapter(SourceAdapter):
    """Ingest WestGate Church (San Jose) sermons from the Squarespace feed."""

    source = "westgate"
    # Measured: 27–66 h after Sunday 00:00 UTC (June–October 2026).
    publish_grace = timedelta(hours=96)

    def poll(self) -> PollResult:
        try:
            content = fetch_feed(self.url)
        except FeedFetchError:
            return PollResult(deferred=True)

        items, id_less = _parse(content)
        included: list[SermonItem] = []
        excluded = id_less
        for item in items:
            if is_main_sunday_sermon(item.raw_title, item.preached_on):
                included.append(item)
            else:
                excluded.append(excluded_item(item))
        return PollResult(items=included, excluded=excluded)
