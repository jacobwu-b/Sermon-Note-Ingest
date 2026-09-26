"""Lakepointe Church (Rockwall, TX): Megaphone podcast feed, Sunday + denylist.

Classification is North Point's rule plus a denylist. The feed's ``pubDate``
is trustworthy here (unlike Westgate's), so the weekday check runs on it
directly. What North Point does not have is a set of non-sermon series
sharing the feed: "Bonus Podcast", "Church At Home", and "Bonus Q&A"
episodes — every one of them published on a non-Sunday in a sampled feed, so
the weekday check alone would exclude them today, but they are named anyway
so a bonus episode that one week publishes on a Sunday doesn't get mistaken
for a sermon.

Two fields are permanently empty for Lakepointe: the feed emits no ``<link>``
and no ``<description>`` on any item, so ``episode_url`` and ``blurb`` are
always ``""``.
"""

from __future__ import annotations

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

# The current feed title is "{title} | {series} | {speaker}"; an older archive
# item is a bare title with no separator at all.
_TITLE_SEP = " | "
_TITLE_SEGMENTS = 3
_SUNDAY = 6
_GUID_PREFIX = "lakepointe:"
# Matched on title text rather than segment count, because the bonus series
# uses a double pipe and so already fails the three-segment split.
_NON_SERMON_MARKERS = (
    "|| Bonus Podcast",
    " | Church At Home | ",
    " | Bonus Q&A ",
)


def _split_title(raw_title: str) -> tuple[str, str | None, str | None]:
    parts = raw_title.split(_TITLE_SEP)
    if len(parts) != _TITLE_SEGMENTS:
        return raw_title, None, None
    title, series, speaker = (part.strip() for part in parts)
    return title, series or None, speaker or None


def _entry_to_item(entry: Any) -> tuple[SermonItem, int]:
    raw_title = (entry.get("title") or "").strip()
    title, series, speaker = _split_title(raw_title)
    preached_on, weekday = parse_pubdate(entry.get("published"))
    item = SermonItem(
        guid=_GUID_PREFIX + entry.get("id", ""),
        title=title,
        raw_title=raw_title,
        series=series,
        # itunes:author names the church, not the preacher — the title's third
        # segment is the only speaker source this feed has.
        speaker=speaker,
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
    """A Sunday item whose title carries none of the feed's non-sermon series markers."""
    if weekday != _SUNDAY:
        return False
    return not any(marker in raw_title for marker in _NON_SERMON_MARKERS)


class LakepointeAdapter(SourceAdapter):
    """Ingest Lakepointe Church sermons from the Megaphone feed."""

    source = "lakepointe"

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
