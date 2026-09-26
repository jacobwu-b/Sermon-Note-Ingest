from datetime import UTC, date, datetime

from poller.sources.gracepres import (
    _parse,
    _speaker_from_blurb,
    _split_title,
    is_main_sunday_sermon,
    service_date,
)


def test_split_title_strips_trailing_yymmdd_service_date():
    title, when = _split_title("Living God's Mission: Growing Pains 251109")
    assert title == "Living God's Mission: Growing Pains"
    assert when == date(2025, 11, 9)


def test_split_title_strips_underscore_separator_before_date():
    title, when = _split_title("A Vision of Beauty _ 201011")
    assert title == "A Vision of Beauty"
    assert when == date(2020, 10, 11)


def test_split_title_without_date_keeps_whole_title():
    assert _split_title("Protected By The King") == ("Protected By The King", None)


def test_split_title_ignores_six_digits_that_are_not_a_date():
    assert _split_title("Ten Words 991399") == ("Ten Words 991399", None)


def test_service_date_uses_title_date_when_it_is_a_sunday():
    uploaded = datetime(2025, 11, 11, 22, 1, tzinfo=UTC)  # Tuesday
    assert service_date(date(2025, 11, 9), uploaded) == date(2025, 11, 9)


def test_service_date_falls_back_when_title_date_is_not_a_sunday():
    # A real feed typo: "...250807" on a sermon uploaded Tue 2025-09-09.
    uploaded = datetime(2025, 9, 9, 21, 51, tzinfo=UTC)
    assert service_date(date(2025, 8, 7), uploaded) == date(2025, 9, 7)


def test_service_date_is_most_recent_sunday_on_or_before_upload():
    uploaded = datetime(2026, 9, 15, 23, 13, tzinfo=UTC)  # Tuesday
    assert service_date(None, uploaded) == date(2026, 9, 13)


def test_service_date_same_day_for_a_sunday_upload():
    uploaded = datetime(2026, 7, 19, 20, 0, tzinfo=UTC)  # Sunday 1pm Pacific
    assert service_date(None, uploaded) == date(2026, 7, 19)


def test_service_date_uses_church_local_day_not_utc():
    # Sun 03:00 UTC is still Saturday evening in Palo Alto: that sermon was
    # preached the Sunday before, not the one about to happen.
    uploaded = datetime(2026, 9, 13, 3, 0, tzinfo=UTC)
    assert service_date(None, uploaded) == date(2026, 9, 6)


def test_service_date_is_none_without_any_date():
    assert service_date(None, None) is None


def test_speaker_from_blurb_reads_leading_pastor_honorific():
    blurb = "Pastor Matt Mobley ends our series in the book of Psalm with a message from Psalm 2"
    assert _speaker_from_blurb(blurb) == "Matt Mobley"


def test_speaker_from_blurb_reads_rev_honorific():
    blurb = "Rev. Iron Kim concludes our series on the Minor Prophets with a word from Joel 1:1-4"
    assert _speaker_from_blurb(blurb) == "Iron Kim"


def test_speaker_from_blurb_does_not_swallow_a_capitalized_verb():
    blurb = "Pastor Iron Kim Begins our sermon series in The Book of Psalms"
    assert _speaker_from_blurb(blurb) == "Iron Kim"


def test_speaker_from_blurb_is_none_for_scripture_only_blurb():
    assert _speaker_from_blurb("Luke 20:40-21:4\nhttps://www.biblegateway.com/passage/") is None


def test_speaker_from_blurb_is_none_for_soundclouds_default_blurb():
    assert _speaker_from_blurb("The Gospel of the Kingdom by Grace Presbyterian Church of SV") is None


def test_is_main_sunday_sermon_includes_ordinary_sermon_title():
    assert is_main_sunday_sermon("The Secret Of Contentment") is True


def test_is_main_sunday_sermon_excludes_lent_daily_meditations():
    assert is_main_sunday_sermon("Journey to the Cross: Day 40") is False


def test_is_main_sunday_sermon_excludes_retreat_talks():
    assert is_main_sunday_sermon("All-Church Retreat 2025: Saturday Night") is False
    assert is_main_sunday_sermon("Retreat: Keeping In Step With the Spirit 240503") is False


def test_is_main_sunday_sermon_excludes_good_friday_meditation():
    assert is_main_sunday_sermon("Good Friday Meditation 240329") is False


_FEED = b"""<?xml version='1.0' encoding='UTF-8'?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
<channel>
  <title>Grace Presbyterian Church of Silicon Valley</title>
  <item>
    <guid isPermaLink="false">tag:soundcloud,2010:tracks/2401192710</guid>
    <title>Protected By The King</title>
    <pubDate>Tue, 15 Sep 2026 23:13:02 +0000</pubDate>
    <link>https://soundcloud.com/gracepres-sv/protected-by-the-king</link>
    <itunes:author>Grace Presbyterian Church of SV</itunes:author>
    <description>Pastor Matt Mobley ends our series in the book of Psalm with a message from Psalm 2</description>
    <enclosure type="audio/mpeg" url="https://feeds.soundcloud.com/stream/2401192710-gracepres-sv-protected-by-the-king.mp3" length="79276268"/>
  </item>
  <item>
    <guid isPermaLink="false">tag:soundcloud,2010:tracks/2036000001</guid>
    <title>Journey to the Cross: Day 40</title>
    <pubDate>Tue, 15 Apr 2025 18:00:00 +0000</pubDate>
    <link>https://soundcloud.com/gracepres-sv/journey-to-the-cross-day-40</link>
    <description>Today's meditation was written and read by Bethany Nichols.</description>
    <enclosure type="audio/mpeg" url="https://feeds.soundcloud.com/stream/2036000001-day-40.mp3" length="1"/>
  </item>
  <item>
    <title>No identity here</title>
    <pubDate>Tue, 08 Sep 2026 23:30:56 +0000</pubDate>
    <enclosure type="audio/mpeg" url="https://feeds.soundcloud.com/stream/0-orphan.mp3" length="1"/>
  </item>
</channel>
</rss>
"""


def test_parse_maps_feed_entries_to_items_and_counts_id_less_ones():
    items, id_less = _parse(_FEED)
    assert id_less == 1
    assert [item.title for item in items] == ["Protected By The King", "Journey to the Cross: Day 40"]
    sermon = items[0]
    assert sermon.guid == "gracepres:tag:soundcloud,2010:tracks/2401192710"
    assert sermon.raw_title == "Protected By The King"
    assert sermon.series is None
    assert sermon.speaker == "Matt Mobley"
    assert sermon.preached_on == "2026-09-13"
    assert sermon.feed_published_at == datetime(2026, 9, 15, 23, 13, 2, tzinfo=UTC)
    assert sermon.episode_url == "https://soundcloud.com/gracepres-sv/protected-by-the-king"
    assert sermon.audio_url.endswith("protected-by-the-king.mp3")
    assert sermon.blurb.startswith("Pastor Matt Mobley ends")
