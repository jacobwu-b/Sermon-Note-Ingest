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
from datetime import date, datetime
from typing import Any

import feedparser

from poller.net import FeedFetchError, fetch_feed
from poller.sources.base import PollResult, SermonItem, SourceAdapter
from poller.sources.common import entry_audio_url, entry_has_identity, parse_pubdate_at

_GUID_PREFIX = "westgate:"

# e.g. "The Gospel of John (Part 2) | To Whom Shall We Go | August 16, 2026".
_TITLE_DATE_RE = re.compile(r"^(?P<rest>.+) \| (?P<date>[A-Za-z]+ \d{1,2}, \d{4})$")
_TITLE_DATE_FMT = "%B %d, %Y"

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
    try:
        # Only the calendar date is kept (.date() below), so the naive instant
        # strptime produces is never used as a real datetime.
        service_date = datetime.strptime(match.group("date"), _TITLE_DATE_FMT).date()  # noqa: DTZ007
    except ValueError:
        return raw_title, None, None
    rest = match.group("rest")
    series, sep, title = rest.partition(" | ")
    if not sep:
        return rest, None, service_date
    return title, series, service_date


def _entry_to_item(entry: Any) -> tuple[SermonItem, int]:
    """``preached_on``/weekday come from the title's parsed service date, not
    ``pubDate``; ``feed_published_at`` still comes from the feed's own ``pubDate`` — it
    measures actual feed availability, a different instant than the nominal
    service date."""
    raw_title = (entry.get("title") or "").strip()
    title, series, service_date = _parse_title(raw_title)
    preached_on = service_date.isoformat() if service_date is not None else ""
    weekday = service_date.weekday() if service_date is not None else -1
    item = SermonItem(
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
    return item, weekday


def _parse(content: bytes) -> tuple[list[tuple[SermonItem, int]], int]:
    parsed = feedparser.parse(content)
    items: list[tuple[SermonItem, int]] = []
    skipped = 0
    for entry in parsed.entries:
        if not entry_has_identity(entry):
            skipped += 1
            continue
        items.append(_entry_to_item(entry))
    return items, skipped


def is_main_sunday_sermon(raw_title: str, weekday: int) -> bool:
    """A title date that parsed to a Sunday, excluding the "NextGen Sunday" program."""
    return weekday == 6 and not raw_title.startswith(_NEXTGEN_PREFIX)


class WestgateAdapter(SourceAdapter):
    """Ingest WestGate Church (San Jose) sermons from the Squarespace feed."""

    source = "westgate"

    def poll(self) -> PollResult:
        try:
            content = fetch_feed(self.url)
        except FeedFetchError:
            return PollResult(deferred=True)

        pairs, id_less = _parse(content)
        included: list[SermonItem] = []
        excluded = id_less
        for item, weekday in pairs:
            if is_main_sunday_sermon(item.raw_title, weekday):
                included.append(item)
            else:
                excluded += 1
        return PollResult(items=included, excluded=excluded)
