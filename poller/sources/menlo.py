"""Menlo Church: Podbean RSS feed + Legacy/Midweek exclusion.

The feed mixes main Sunday sermons with the Legacy (traditional) service and
the Midweek Podcast; only main Sunday sermons are in scope. Legacy and Midweek
content is recognized by title only (main-sermon blurbs cross-promote the
Midweek Podcast by name, so the blurb is not a reliable exclude signal). Any
item published on a Sunday that is neither is included; anything else is
ambiguous. Within a Sunday-anchored week with no Sunday item, the ambiguous
item closest to Sunday is promoted as the late-posted main sermon.
"""

from __future__ import annotations

import dataclasses
import datetime
from collections import defaultdict
from collections.abc import Iterable
from enum import Enum
from typing import Any

import feedparser

from poller.net import FeedFetchError, fetch_feed
from poller.sources.base import PollResult, SermonItem, SourceAdapter
from poller.sources.common import (
    entry_audio_url,
    entry_has_identity,
    parse_pubdate,
    parse_pubdate_at,
)

# The Legacy service names itself in the final title suffix.
_LEGACY_SUFFIX = "Menlo Church Legacy Service"
# The Midweek Podcast names itself in the title.
_MIDWEEK_MARKER = "menlo midweek podcast"
# email.utils weekday convention used throughout: Sunday == 6.
_SUNDAY = 6


def _split_title(raw_title: str) -> tuple[str, str | None, str | None]:
    """Split a ``Title | Series | Speaker`` feed title into its parts.

    The feed sometimes appends a trailing suffix (``Title | Series | Speaker |
    Menlo Church Podcasts``); that fourth segment is discarded when present.
    """
    parts = [p.strip() for p in raw_title.split("|")]
    if len(parts) >= 3:
        return parts[0], parts[1], parts[2]
    if len(parts) == 2:
        return parts[0], None, parts[1]
    return (parts[0] if parts else ""), None, None


def _entry_to_item(entry: Any) -> tuple[SermonItem, int]:
    raw_title = (entry.get("title") or "").strip()
    title, series, speaker = _split_title(raw_title)
    preached_on, weekday = parse_pubdate(entry.get("published"))
    return SermonItem(
        guid=entry.get("id", ""),
        title=title,
        raw_title=raw_title,
        series=series,
        speaker=speaker,
        preached_on=preached_on,
        feed_published_at=parse_pubdate_at(entry.get("published")),
        episode_url=entry.get("link", ""),
        audio_url=entry_audio_url(entry),
        blurb=(entry.get("summary") or "").strip(),
    ), weekday


def _parse(content: bytes) -> tuple[list[tuple[SermonItem, int]], int]:
    """Parse raw feed bytes into ``(item, weekday)`` pairs, in feed order."""
    parsed = feedparser.parse(content)
    items: list[tuple[SermonItem, int]] = []
    skipped = 0
    for entry in parsed.entries:
        if not entry_has_identity(entry):
            skipped += 1
            continue
        items.append(_entry_to_item(entry))
    return items, skipped


class Classification(Enum):
    INCLUDE = "include"
    EXCLUDE = "exclude"
    AMBIGUOUS = "ambiguous"


def classify(raw_title: str, weekday: int) -> Classification:
    """Return the classification for one item.

    Order matters: Legacy and Midweek are recognized first so their content can
    never fall through to INCLUDE. A Sunday item that is neither is a main
    sermon regardless of title suffix; everything else is AMBIGUOUS.
    """
    if raw_title.strip().endswith(_LEGACY_SUFFIX):
        return Classification.EXCLUDE
    if _MIDWEEK_MARKER in raw_title.lower():
        return Classification.EXCLUDE
    if weekday == _SUNDAY:
        return Classification.INCLUDE
    return Classification.AMBIGUOUS


def _days_since_sunday(weekday: int) -> int:
    return (weekday + 1) % 7


def _week_anchor(preached_on: str, weekday: int) -> datetime.date:
    published = datetime.date.fromisoformat(preached_on)
    return published - datetime.timedelta(days=_days_since_sunday(weekday))


def classify_feed(
    items: Iterable[tuple[SermonItem, int]],
) -> dict[str, Classification]:
    """Classify a whole feed, promoting one late-posted main sermon per week.

    The church sometimes posts the main sermon a day or two after Sunday, where
    the per-item rule can only mark it AMBIGUOUS. So within each Sunday-anchored
    week with no Sunday item, the AMBIGUOUS item closest to Sunday is promoted
    to INCLUDE; any other AMBIGUOUS items that week stay flagged.
    """
    items = list(items)
    verdicts = {item.guid: classify(item.raw_title, weekday) for item, weekday in items}

    # An undated/malformed item (weekday == -1) has no week anchor, so it can
    # neither claim a week nor be promoted — excluding it here keeps
    # _week_anchor from raising on an empty date string.
    weeks_with_include = {
        _week_anchor(item.preached_on, weekday)
        for item, weekday in items
        if item.preached_on and verdicts[item.guid] is Classification.INCLUDE
    }
    ambiguous_by_week: dict[datetime.date, list[tuple[SermonItem, int]]] = defaultdict(list)
    for item, weekday in items:
        if item.preached_on and verdicts[item.guid] is Classification.AMBIGUOUS:
            ambiguous_by_week[_week_anchor(item.preached_on, weekday)].append((item, weekday))

    for anchor, candidates in ambiguous_by_week.items():
        if anchor in weeks_with_include:
            continue
        winner, _winner_weekday = min(
            candidates,
            key=lambda pair: (
                _days_since_sunday(pair[1]),
                pair[0].preached_on,
                pair[0].guid,
            ),
        )
        verdicts[winner.guid] = Classification.INCLUDE

    return verdicts


class MenloPodbeanAdapter(SourceAdapter):
    """Ingest Menlo Church sermons from the Podbean RSS feed."""

    source = "menlo"

    def poll(self) -> PollResult:
        try:
            content = fetch_feed(self.url)
        except FeedFetchError:
            return PollResult(deferred=True)

        pairs, id_less = _parse(content)
        verdicts = classify_feed(pairs)
        included: list[SermonItem] = []
        excluded = id_less
        for item, weekday in pairs:
            verdict = verdicts[item.guid]
            if verdict is Classification.INCLUDE:
                if weekday != _SUNDAY:
                    # A promoted late-posted item: the feed's own pubDate is an
                    # internal detail, never the true service date — the sermon
                    # was preached on the Sunday classify_feed anchored it to.
                    item = dataclasses.replace(
                        item,
                        preached_on=_week_anchor(item.preached_on, weekday).isoformat(),
                    )
                included.append(item)
            elif verdict is Classification.EXCLUDE:
                excluded += 1
            # AMBIGUOUS items are neither included nor counted excluded — they
            # are simply dropped (no downstream "flagged for review" state here).

        return PollResult(items=included, excluded=excluded)
