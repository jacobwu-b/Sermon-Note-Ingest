"""North Point Community Church: Sardius podcast feed, Sunday-only.

The feed has no service-segment or title marker distinguishing its main Sunday
teaching from guest-pastor content — per the owner's resolved decision, it
doesn't need one. Any sermon delivered on a Sunday is in scope regardless of
who is teaching; only non-Sunday content (e.g. a midweek special) is excluded.
The feed already carries the per-episode speaker in ``<itunes:author>``, so no
scrape fallback is needed.
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

# The feed titles a sermon as "{title} // {speaker}"; unlike PBC there's no
# clean series delimiter, so series is left null.
_TITLE_SPEAKER_SEP = " // "
_SUNDAY = 6
# The feed's own <guid> is an opaque, unnamespaced hex string, so this adapter
# namespaces it to guarantee no cross-source collision.
_GUID_PREFIX = "north_point:"


def _split_title(raw_title: str) -> tuple[str, str | None]:
    title, sep, speaker = raw_title.rpartition(_TITLE_SPEAKER_SEP)
    if not sep:
        return raw_title, None
    return title, speaker


def _entry_to_item(entry: Any) -> tuple[SermonItem, int]:
    raw_title = (entry.get("title") or "").strip()
    title, title_speaker = _split_title(raw_title)
    preached_on, weekday = parse_pubdate(entry.get("published"))
    # itunes:author names the individual preacher; fall back to the title's
    # trailing speaker credit if absent.
    speaker = (entry.get("author") or "").strip() or title_speaker
    item = SermonItem(
        guid=_GUID_PREFIX + entry.get("id", ""),
        title=title,
        raw_title=raw_title,
        series=None,
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


def is_main_sunday_sermon(weekday: int) -> bool:
    """Any Sunday item counts, regardless of speaker (owner's resolved decision)."""
    return weekday == _SUNDAY


class NorthPointAdapter(SourceAdapter):
    """Ingest North Point Community Church sermons from the Sardius feed."""

    source = "north_point"

    def poll(self) -> PollResult:
        try:
            content = fetch_feed(self.url)
        except FeedFetchError:
            return PollResult(deferred=True)

        pairs, id_less = _parse(content)
        included = [item for item, weekday in pairs if is_main_sunday_sermon(weekday)]
        excluded = id_less + (len(pairs) - len(included))
        return PollResult(items=included, excluded=excluded)
