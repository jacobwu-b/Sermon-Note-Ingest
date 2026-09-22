"""Feed-parsing helpers shared by every adapter (pubdate parsing, identity, enclosure)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any
from zoneinfo import ZoneInfo

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
    except (ValueError, TypeError):
        return "", -1
    return when.date().isoformat(), when.weekday()


def parse_pubdate_at(raw: str | None) -> datetime | None:
    """Parse an RFC-822 ``pubDate`` into its full instant, or ``None`` if unusable."""
    if not raw:
        return None
    try:
        when = parsedate_to_datetime(raw)
    except (ValueError, TypeError):
        return None
    return when if when.tzinfo is not None else when.replace(tzinfo=UTC)


def derive_preached_on(
    published_at: datetime, *, title_date: date | None = None, tz: ZoneInfo | None = None
) -> date:
    """The Sunday a sermon was preached: an explicit Sunday title date if the caller
    has one, else the nearest Sunday on or before ``published_at``'s own calendar day.

    Generalizes the rule already used by ``gracepres.py``'s ``service_date`` (a title
    date is trusted only when it's itself a Sunday — a non-Sunday title date is more
    likely a typo or an unrelated number than a real override) and, for a
    ``published_at`` that already falls on a Sunday, reduces to that day exactly —
    the common case for every adapter that classifies on ``pubDate`` directly.

    ``tz``, when given, converts ``published_at`` to that zone before taking its
    calendar day — required for a feed whose ``published_at`` isn't already in the
    church's own local time (e.g. GracePres's SoundCloud upload instant, stored in
    UTC): a midnight-UTC timestamp on a Sunday is still Saturday evening in Pacific
    time, and skipping the conversion silently derives the wrong week entirely,
    not just the wrong day (confirmed via the docs/specs/0007 backtest). Omit it
    for a feed whose ``published_at`` is already meaningful in the church's own day
    (the common case).

    Not yet wired into any adapter (see docs/specs/0007) — a per-church switch-over
    is a separate decision made from backtesting this against each church's history.
    """
    if title_date is not None and title_date.weekday() == _SUNDAY:
        return title_date
    when = published_at.astimezone(tz) if tz is not None else published_at
    day = when.date()
    return day - timedelta(days=(day.weekday() + 1) % 7)


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
