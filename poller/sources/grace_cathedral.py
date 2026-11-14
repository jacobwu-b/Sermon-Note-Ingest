"""Grace Cathedral (San Francisco): Libsyn "Sermons from Grace Cathedral" feed.

The feed carries every sermon preached at the cathedral, not only the Sunday
one: Holy Week and Christmas services, the Easter Vigil, Evensong, Pride Mass,
diocesan ordinations and confirmations, and on some feasts a second sermon by
the bishop. The scope kept here is the Sunday sermons (both, on a Sunday with
two preachers) and the special services every adapter accepts
(:func:`~poller.sources.common.is_service_date`: Christmas, Good Friday, New
Year, Thanksgiving). Ash Wednesday, Maundy Thursday and the named special
programs are out.

``preached_on`` comes from the blurb's header when it states a calendar date
within two weeks of the upload (about a fifth of items do, in several
formats; one carried a year typo), else from the upload's local
day: a service date as-is, a Monday upload the Sunday before, a Saturday upload
the Sunday after (sermons sometimes go up the afternoon before). A midweek
upload with no header date and no feast is not a sermon in scope.

The blurb header is the short lines before any prose: speaker, "Grace
Cathedral, San Francisco, CA", then a date or liturgical day. Special programs
are named in the title or that header. The church re-posts a sermon now and
then under a new guid with the same audio file; the first one in feed order
is kept.

About a third of items are titled with the preacher's name alone, in ``<title>``
and ``itunes:title`` both. Such an item is titled from its blurb instead: by the
service the header names just above the lectionary readings, else by the Gospel
reading (spec 0012). ``raw_title`` keeps the feed's.
"""

from __future__ import annotations

import html
import re
from datetime import date, datetime, timedelta
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
    is_service_date,
    parse_pubdate_at,
)

_GUID_PREFIX = "grace_cathedral:"
_CHURCH_TZ = ZoneInfo("America/Los_Angeles")
_MONDAY = 0
_SATURDAY = 5
# The latest live upload after its stated date is 3 days; two weeks leaves room.
_STATED_DATE_WINDOW_DAYS = 14

_HEADER_LINES = 4
_HEADER_LINE_MAX = 150
_BREAK_RE = re.compile(r"<br\s*/?>|</p>|</div>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")

_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
_MONTH_DAY_YEAR_RE = re.compile(
    rf"\b(?P<month>{_MONTHS})\s+(?P<day>\d{{1,2}})(?:st|nd|rd|th)?,?\s+(?P<year>\d{{4}})\b"
)
_DAY_MONTH_YEAR_RE = re.compile(rf"\b(?P<day>\d{{1,2}})\s+(?P<month>{_MONTHS})\s+(?P<year>\d{{4}})\b")

_SPECIAL_PROGRAM_RE = re.compile(
    r"pride mass|evensong|ordination|confirmation|vigil|maundy thursday|ash wednesday", re.IGNORECASE
)

_HONORIFIC_RE = re.compile(
    r"^(?:The\s+)?(?:(?:Very|Rt\.?|Right|Most)\s+)?(?:Rev\.?|Revd\.?|Reverend)\s+(?:(?:Dr\.?|Canon|Br\.?)\s+)*"
)
_ROLE_SUFFIX_RE = re.compile(r"\s*(?:,| - ).*$")
_BARE_NAME_RE = re.compile(r"^[A-Z][\w'’.-]*(?:\s+[A-Z][\w'’.()-]*){1,3}$")

# A whole line that is a Bible reference: "Psalm 121", "1 Thessalonians 4:13-18", "Ps. 118:1-2, 14-24".
_READING_RE = re.compile(
    r"^(?P<book>(?:[1-3]\s*)?[A-Z][a-z]+\.?(?:\s+(?:of\s+)?[A-Z][a-z]+)*)\s*:?\s*\d+(?:[:.]\d+)?[\d:.,;\s\-–—ab]*$"
)
_GOSPELS = frozenset({"Matthew", "Mark", "Luke", "John"})


def _blurb_lines(blurb: str) -> list[str]:
    """Every non-blank line of the blurb's text."""
    text = html.unescape(_TAG_RE.sub("\n", _BREAK_RE.sub("\n", blurb)))
    return [line for line in (part.strip() for part in text.split("\n")) if line]


def _header_lines(blurb: str) -> list[str]:
    """The blurb's leading short lines, before the first line of prose."""
    lines: list[str] = []
    for line in _blurb_lines(blurb):
        if len(line) > _HEADER_LINE_MAX or len(lines) == _HEADER_LINES:
            break
        lines.append(line)
    return lines


def header_date(lines: list[str]) -> date | None:
    """The first calendar date the blurb header states, or ``None``."""
    for line in lines:
        match = _MONTH_DAY_YEAR_RE.search(line) or _DAY_MONTH_YEAR_RE.search(line)
        if match:
            try:
                return datetime.strptime(
                    f"{match.group('year')} {match.group('month')} {match.group('day')}", "%Y %B %d"
                ).date()
            except ValueError:
                continue
    return None


def service_date(stated: date | None, feed_published_at: datetime | None) -> date | None:
    """The day an in-scope sermon was preached, or ``None`` if it is not one.

    A header date near the upload is trusted either way: on a service date it is the
    answer, off one (Ash Wednesday) the sermon is out of scope. A header date far
    from the upload is a typo (one live header was a year out) and the upload decides.
    """
    if feed_published_at is None:
        return stated if stated is not None and is_service_date(stated) else None
    day = feed_published_at.astimezone(_CHURCH_TZ).date()
    if stated is not None and abs((day - stated).days) <= _STATED_DATE_WINDOW_DAYS:
        return stated if is_service_date(stated) else None
    if is_service_date(day):
        return day
    if day.weekday() == _MONDAY:
        return day - timedelta(days=1)
    if day.weekday() == _SATURDAY:
        return day + timedelta(days=1)
    return None


def is_special_program(lines: list[str]) -> bool:
    """Whether the title or blurb header names a service out of scope."""
    return any(_SPECIAL_PROGRAM_RE.search(line) for line in lines)


def _name_from_honorific(line: str) -> str | None:
    match = _HONORIFIC_RE.match(line)
    if not match:
        return None
    return _ROLE_SUFFIX_RE.sub("", line[match.end() :]).strip() or None


def _speaker(raw_title: str, lines: list[str]) -> str | None:
    """The preacher: an honorific-led header line, a bare name above the cathedral line, or the title."""
    for index, line in enumerate(lines):
        name = _name_from_honorific(line)
        if name:
            return name
        following = lines[index + 1] if index + 1 < len(lines) else ""
        if following.startswith("Grace Cathedral") and _BARE_NAME_RE.match(line):
            return line
    return _name_from_honorific(raw_title)


def _is_date_line(line: str) -> bool:
    return bool(_MONTH_DAY_YEAR_RE.search(line) or _DAY_MONTH_YEAR_RE.search(line))


def _reading_book(line: str) -> str | None:
    """The book a lectionary-reading line cites, or ``None`` if the line is not one."""
    match = _READING_RE.match(line)
    return match.group("book") if match and not _is_date_line(line) else None


def _is_name_only(title: str, speaker: str | None) -> bool:
    if speaker and title == speaker:
        return True
    match = _HONORIFIC_RE.match(title)
    return match is not None and _BARE_NAME_RE.match(title[match.end() :]) is not None


def _service_name(lines: list[str]) -> str | None:
    """The one line between the header's date and the first reading, when it is a name."""
    date_at = next((i for i, line in enumerate(lines) if _is_date_line(line)), None)
    if date_at is None:
        return None
    reading_at = next((i for i in range(date_at + 1, len(lines)) if _reading_book(lines[i])), None)
    if reading_at is None:
        return None
    between = [line for line in lines[date_at + 1 : reading_at] if not line.startswith("Grace Cathedral")]
    if len(between) != 1 or len(between[0]) > _HEADER_LINE_MAX or any(c.isdigit() for c in between[0]):
        return None
    return between[0]


def _gospel_reading(lines: list[str]) -> str | None:
    gospels = [line for line in lines if _reading_book(line) in _GOSPELS]
    return " ".join(gospels[-1].split()) if gospels else None


def sermon_title(raw_title: str, speaker: str | None, blurb: str) -> str:
    """The feed title, or for a name-only one the blurb's service name, else its Gospel reading."""
    if not _is_name_only(raw_title, speaker):
        return raw_title
    lines = _blurb_lines(blurb)
    return _service_name(lines) or _gospel_reading(lines) or raw_title


def _entry_to_item(entry: Any) -> SermonItem:
    raw_title = (entry.get("title") or "").strip()
    feed_published_at = parse_pubdate_at(entry.get("published"))
    blurb = (entry.get("summary") or "").strip()
    lines = _header_lines(blurb)
    when = service_date(header_date(lines), feed_published_at)
    speaker = _speaker(raw_title, lines)
    return SermonItem(
        guid=_GUID_PREFIX + entry.get("id", ""),
        title=sermon_title(raw_title, speaker, blurb),
        raw_title=raw_title,
        series=None,
        speaker=speaker,
        preached_on=when.isoformat() if when is not None else "",
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
    """Split items into ``(sermons, rest)``, in feed order.

    A sermon has a service date (an empty ``preached_on`` means it had none), names no
    special program, and is the first item in the feed with its audio file.
    """
    seen_audio: set[str] = set()
    sermons: list[SermonItem] = []
    rest: list[SermonItem] = []
    for item in items:
        in_scope = bool(item.preached_on) and not is_special_program(
            [item.raw_title, *_header_lines(item.blurb)]
        )
        if in_scope and item.audio_url not in seen_audio:
            sermons.append(item)
        else:
            rest.append(item)
        seen_audio.add(item.audio_url)
    return sermons, rest


class GraceCathedralAdapter(SourceAdapter):
    """Ingest Grace Cathedral (San Francisco) sermons from its Libsyn feed."""

    source = "grace_cathedral"
    # Measured April 2025–November 2026: 22 h (median) to 75 h after 00:00 UTC on the service day.
    publish_grace = timedelta(hours=120)

    def poll(self) -> PollResult:
        try:
            content = fetch_feed(self.url)
        except FeedFetchError:
            return PollResult(deferred=True)

        items, id_less = _parse(content)
        sermons, rest = select_sermons(items)
        return PollResult(items=sermons, excluded=id_less + [excluded_item(item) for item in rest])
