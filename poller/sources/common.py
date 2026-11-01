"""Feed-parsing helpers shared by every adapter (pubdate parsing, identity, enclosure)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any

from poller.sources.base import ExcludedItem, SermonItem

_SUNDAY = 6


def parse_pubdate(raw: str | None) -> tuple[str, int]:
    """Parse an RFC-822 ``pubDate`` into ``(YYYY-MM-DD, weekday)`` in its own timezone.

    ``weekday`` uses the ``email.utils`` convention: Monday=0 … Sunday=6. A
    missing or unparseable date yields ``("", -1)`` so the item is never
    mistaken for a Sunday.
    """
    if not raw:
        return "", -1
    try:
        when = parsedate_to_datetime(raw)
    except ValueError, TypeError:
        return "", -1
    return when.date().isoformat(), when.weekday()


def parse_pubdate_at(raw: str | None) -> datetime | None:
    """Parse an RFC-822 ``pubDate`` into its full instant, or ``None`` if unusable."""
    if not raw:
        return None
    try:
        when = parsedate_to_datetime(raw)
    except ValueError, TypeError:
        return None
    return when if when.tzinfo is not None else when.replace(tzinfo=UTC)


_FIXED_SPECIAL_SERVICES = frozenset({(12, 24), (12, 25), (12, 31), (1, 1)})


def _easter(year: int) -> date:
    """Western Easter Sunday for ``year`` (the anonymous Gregorian computus)."""
    golden = year % 19
    century, year_of_century = divmod(year, 100)
    leap_centuries, century_rem = divmod(century, 4)
    lunar_correction = (century - (century + 8) // 25 + 1) // 3
    epact = (19 * golden + century - leap_centuries - lunar_correction + 15) % 30
    leap_years, year_rem = divmod(year_of_century, 4)
    weekday_offset = (32 + 2 * century_rem + 2 * leap_years - epact - year_rem) % 7
    shift = (golden + 11 * epact + 22 * weekday_offset) // 451
    month, day = divmod(epact + weekday_offset - 7 * shift + 114, 31)
    return date(year, month, day + 1)


def _thanksgiving(year: int) -> date:
    """US Thanksgiving Day: the fourth Thursday of November."""
    first = date(year, 11, 1)
    return first + timedelta(days=(3 - first.weekday()) % 7 + 21)


def is_service_date(day: date) -> bool:
    """Whether a sermon can have been preached on ``day``: a Sunday, or a special
    service churches hold off-Sunday — Christmas Eve/Day, New Year's Eve/Day,
    Good Friday, or Thanksgiving Day. Easter is omitted because it is always a Sunday.
    """
    return (
        day.weekday() == _SUNDAY
        or (day.month, day.day) in _FIXED_SPECIAL_SERVICES
        or day == _easter(day.year) - timedelta(days=2)
        or day == _thanksgiving(day.year)
    )


def entry_audio_url(entry: Any) -> str:
    """Return the first enclosure URL on a feed entry, or ``""`` if none is present."""
    for enclosure in entry.get("enclosures", []):
        href = enclosure.get("href")
        if href:
            return str(href)
    return ""


def entry_has_identity(entry: Any) -> bool:
    """Whether a feed entry carries a usable id (a ``<guid>``, or its ``<link>`` fallback).

    feedparser derives ``entry.id`` from ``<guid>``, falling back to ``<link>``;
    an item with neither yields ``""``. Every id-less item in a feed would
    otherwise collapse onto the same (or, for a namespaced adapter, prefix-only)
    key on upsert, so adapters must skip these before building a
    :class:`~poller.sources.base.SermonItem`.
    """
    return bool(entry.get("id", ""))


def excluded_entry(entry: Any) -> ExcludedItem:
    """A feed entry skipped before it became a :class:`SermonItem` (e.g. one with no id)."""
    day, _weekday = parse_pubdate(entry.get("published"))
    return ExcludedItem(
        raw_title=(entry.get("title") or "").strip(),
        published_on=date.fromisoformat(day) if day else None,
    )


def excluded_item(item: SermonItem) -> ExcludedItem:
    """A parsed item the classifier rejected, dated by its own ``feed_published_at``."""
    published = item.feed_published_at
    return ExcludedItem(raw_title=item.raw_title, published_on=published.date() if published else None)
