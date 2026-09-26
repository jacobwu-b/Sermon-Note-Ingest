"""Grace Presbyterian Church of Silicon Valley (Palo Alto): SoundCloud feed.

Two things about this feed are unlike the other sources:

- ``pubDate`` is the SoundCloud *upload* instant, not the service date: a
  Sunday sermon typically lands Monday–Wednesday (19 of 500 live items fell on
  a Sunday), so the weekday check every other adapter classifies on is useless
  here. Instead, ``preached_on`` is *derived*: the ``YYMMDD`` the church
  appended to titles through late 2025 when it parses to a Sunday, otherwise
  the most recent Sunday on or before the upload in the church's own timezone.
  Ordering both ways round matters — one live title carried a typo'd
  non-Sunday date, and the older archive was uploaded in batches weeks late.
- Classification is a title denylist, not a weekday: the feed also carries the
  church's daily Lent meditations ("Journey to the Cross: Day N"), retreat
  talks, and Good Friday meditations. Everything else in a 500-item live
  sample was a Sunday sermon.

``itunes:author`` names the church, so the speaker is read from the blurb's
leading honorific ("Pastor Matt Mobley ends our series…"), present on every
live item since mid-2025 and absent from most of the archive. The series is
mentioned only in that same free prose, so it is not inferred.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import feedparser

from poller.net import FeedFetchError, fetch_feed
from poller.sources.base import PollResult, SermonItem, SourceAdapter
from poller.sources.common import entry_audio_url, entry_has_identity, parse_pubdate_at

_GUID_PREFIX = "gracepres:"
_SUNDAY = 6
_CHURCH_TZ = ZoneInfo("America/Los_Angeles")

# e.g. "Living God's Mission: Growing Pains 251109" or "A Vision of Beauty _ 201011".
_TITLE_DATE_RE = re.compile(r"^(?P<title>.*?)[\s_-]*(?P<date>\d{6})$")
_TITLE_DATE_FMT = "%y%m%d"

# The first two capitalized words after the honorific; a capitalized verb
# ("Pastor Iron Kim Begins…") is left behind by the two-word cap.
_SPEAKER_RE = re.compile(r"^(?:Pastor|Rev\.|Reverend|Dr\.)\s+(?P<name>[A-Z][\w'’.-]*\s+[A-Z][\w'’.-]*)")

_NON_SERMON_MARKERS = ("journey to the cross", "retreat", "meditation")


def _split_title(raw_title: str) -> tuple[str, date | None]:
    """Split off a trailing ``YYMMDD``; ``None`` when there is none or it is not a real date."""
    match = _TITLE_DATE_RE.match(raw_title)
    if not match:
        return raw_title, None
    try:
        # Only the calendar date is kept, so strptime's naive instant is never used.
        title_date = datetime.strptime(match.group("date"), _TITLE_DATE_FMT).date()  # noqa: DTZ007
    except ValueError:
        return raw_title, None
    return match.group("title").strip() or raw_title, title_date


def service_date(title_date: date | None, feed_published_at: datetime | None) -> date | None:
    """The Sunday this sermon was preached: the title's date if it is one, else the
    most recent Sunday on or before the upload, in the church's local day."""
    if title_date is not None and title_date.weekday() == _SUNDAY:
        return title_date
    if feed_published_at is None:
        return None
    local_day = feed_published_at.astimezone(_CHURCH_TZ).date()
    return local_day - timedelta(days=(local_day.weekday() + 1) % 7)


def _speaker_from_blurb(blurb: str) -> str | None:
    match = _SPEAKER_RE.match(blurb.strip())
    return match.group("name") if match else None


def _entry_to_item(entry: Any) -> SermonItem:
    raw_title = (entry.get("title") or "").strip()
    title, title_date = _split_title(raw_title)
    feed_published_at = parse_pubdate_at(entry.get("published"))
    blurb = (entry.get("summary") or "").strip()
    when = service_date(title_date, feed_published_at)
    return SermonItem(
        guid=_GUID_PREFIX + entry.get("id", ""),
        title=title,
        raw_title=raw_title,
        series=None,
        speaker=_speaker_from_blurb(blurb),
        preached_on=when.isoformat() if when is not None else "",
        feed_published_at=feed_published_at,
        episode_url=entry.get("link") or "",
        audio_url=entry_audio_url(entry),
        blurb=blurb,
    )


def _parse(content: bytes) -> tuple[list[SermonItem], int]:
    parsed = feedparser.parse(content)
    items: list[SermonItem] = []
    skipped = 0
    for entry in parsed.entries:
        if not entry_has_identity(entry):
            skipped += 1
            continue
        items.append(_entry_to_item(entry))
    return items, skipped


def is_main_sunday_sermon(raw_title: str) -> bool:
    """A title carrying none of the feed's non-sermon program markers."""
    lowered = raw_title.lower()
    return not any(marker in lowered for marker in _NON_SERMON_MARKERS)


class GracepresAdapter(SourceAdapter):
    """Ingest Grace Presbyterian Church of Silicon Valley sermons from the SoundCloud feed."""

    source = "gracepres"

    def poll(self) -> PollResult:
        try:
            content = fetch_feed(self.url)
        except FeedFetchError:
            return PollResult(deferred=True)

        items, id_less = _parse(content)
        included = [item for item in items if is_main_sunday_sermon(item.raw_title)]
        excluded = id_less + (len(items) - len(included))
        return PollResult(items=included, excluded=excluded)
