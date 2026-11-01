"""Every registered adapter reports what it excluded, by raw title and pubDate (spec 0011)."""

import sys
from datetime import date

import pytest

from poller.sources import ADAPTERS
from poller.sources.base import ExcludedItem

# One feed every adapter reads the same way: a main Sunday sermon, a weekday clip in the same
# week that each adapter's classifier rejects (no title date for Westgate, non-Main_Service
# audio for PBC, a program marker for Grace Pres, a weekday for the rest, and not promoted by
# Menlo since its week already has a Sunday item), and an item with no guid or link.
_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Example Church</title>
<item>
  <title>The Gospel of John (Part 3) | I am the Light of the World | October 25, 2026</title>
  <guid isPermaLink="false">sermon-2026-10-25</guid>
  <link>https://example.org/sermons/light-of-the-world?enmse_mid=4700</link>
  <pubDate>Sun, 25 Oct 2026 18:30:00 +0000</pubDate>
  <enclosure url="https://cdn.example.org/Main_Service/light-of-the-world.mp3" type="audio/mpeg" length="1"/>
</item>
<item>
  <title>Journey to the Cross: Day 12</title>
  <guid isPermaLink="false">clip-2026-10-27</guid>
  <link>https://example.org/clips/day-12</link>
  <pubDate>Tue, 27 Oct 2026 16:00:00 -0700</pubDate>
  <enclosure url="https://cdn.example.org/Youth_Service/day-12.mp3" type="audio/mpeg" length="1"/>
</item>
<item>
  <title>Untitled upload</title>
  <pubDate>Wed, 28 Oct 2026 09:15:00 -0700</pubDate>
  <enclosure url="https://cdn.example.org/Youth_Service/untitled.mp3" type="audio/mpeg" length="1"/>
</item>
</channel></rss>"""


@pytest.mark.parametrize("name", sorted(ADAPTERS))
def test_poll_lists_each_excluded_item_with_its_raw_title_and_pubdate_day(name, monkeypatch):
    adapter_cls = ADAPTERS[name]
    monkeypatch.setattr(sys.modules[adapter_cls.__module__], "fetch_feed", lambda url, **_: _FEED)

    result = adapter_cls(url="https://example.org/feed.xml").poll()

    assert [item.raw_title for item in result.items] == [
        "The Gospel of John (Part 3) | I am the Light of the World | October 25, 2026"
    ]
    assert sorted(result.excluded, key=lambda item: item.raw_title) == [
        ExcludedItem(raw_title="Journey to the Cross: Day 12", published_on=date(2026, 10, 27)),
        ExcludedItem(raw_title="Untitled upload", published_on=date(2026, 10, 28)),
    ]
