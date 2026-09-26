"""Hillside Church (San Jose): Subsplash podcast feed, title read out of description.

Two things about this feed are unlike every other source:

- The sermon's name is not in ``<title>``. The current title format is
  ``"{M.D.YY} | {scripture} | {speaker}"`` (e.g. ``"9.6.26 | Hebrews 13:7–17 |
  Keith Crosby"``) and the sermon's actual name is the first paragraph of the
  HTML ``<description>`` (e.g. ``"A Parting Conversation"``). The description's
  first paragraph becomes the record title; the raw feed title supplies a
  speaker fallback instead.
- The title's date is not the classification signal here (the opposite of
  Westgate): of a 23-item sample, 22 agreed with ``pubDate`` and one named the
  day *after* the actual Sunday. So this adapter classifies on ``pubDate``.
"""

from __future__ import annotations

import html
import re
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

_TITLE_SEP = " | "
_TITLE_SEGMENTS = 3
_SUNDAY = 6
_GUID_PREFIX = "hillside:"
# Subsplash writes the description as HTML paragraphs; block-level boundaries
# become line breaks so the first paragraph can be taken as the title.
_BLOCK_BOUNDARY_RE = re.compile(r"</p\s*>|<br\s*/?>|</div\s*>", re.IGNORECASE)
_TAG_RE = re.compile(r"<[^>]+>")


def _description_lines(raw_description: str) -> list[str]:
    text = _BLOCK_BOUNDARY_RE.sub("\n", raw_description)
    text = _TAG_RE.sub("", text)
    return [line.strip() for line in html.unescape(text).splitlines() if line.strip()]


def _split_title_speaker(raw_title: str) -> str | None:
    """Pull the trailing speaker segment out of a current-format title.

    Returns ``None`` for any title that is not exactly three segments — the
    pre-2026-04 archive format, or whatever the church moves to next. The
    scripture segment itself is not parsed out separately: ``raw_title`` is
    kept in full on the record, so nothing is lost by not splitting it here.
    """
    parts = raw_title.split(_TITLE_SEP)
    if len(parts) != _TITLE_SEGMENTS:
        return None
    _date, _scripture, speaker = (part.strip() for part in parts)
    return speaker or None


def _entry_to_item(entry: Any) -> tuple[SermonItem, int]:
    """``title`` falls back to the raw feed title when the description is empty —
    an empty description must never leave the record with an empty title."""
    raw_title = (entry.get("title") or "").strip()
    title_speaker = _split_title_speaker(raw_title)
    lines = _description_lines(entry.get("summary") or "")
    preached_on, weekday = parse_pubdate(entry.get("published"))
    item = SermonItem(
        guid=_GUID_PREFIX + entry.get("id", ""),
        title=lines[0] if lines else raw_title,
        raw_title=raw_title,
        # The feed carries no series field, and the description's later
        # paragraphs are sometimes a part-marker and sometimes a subtitle —
        # not reliably a series name, so nothing is inferred.
        series=None,
        speaker=(entry.get("author") or "").strip() or title_speaker,
        preached_on=preached_on,
        feed_published_at=parse_pubdate_at(entry.get("published")),
        episode_url=entry.get("link") or "",
        audio_url=entry_audio_url(entry),
        blurb=" ".join(lines),
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
    """A bare ``pubDate`` weekday check, with no denylist; an undated item
    scores ``-1`` and so is never a match."""
    return weekday == _SUNDAY


class HillsideAdapter(SourceAdapter):
    """Ingest Hillside Church (San Jose) sermons from the Subsplash feed."""

    source = "hillside"

    def poll(self) -> PollResult:
        try:
            content = fetch_feed(self.url)
        except FeedFetchError:
            return PollResult(deferred=True)

        pairs, id_less = _parse(content)
        included = [item for item, weekday in pairs if is_main_sunday_sermon(weekday)]
        excluded = id_less + (len(pairs) - len(included))
        return PollResult(items=included, excluded=excluded)
