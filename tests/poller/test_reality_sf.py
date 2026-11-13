from datetime import UTC, date, datetime

from poller.sources.base import ExcludedItem
from poller.sources.reality_sf import (
    _parse,
    _speaker_from_blurb,
    is_non_sermon_title,
    select_sermons,
)


def test_speaker_from_blurb_reads_leading_name_before_teaches():
    blurb = "<p>Ruthie Kim teaches from Esther 4 on how God is working in the ordinary moments of life.</p>"
    assert _speaker_from_blurb(blurb) == "Ruthie Kim"


def test_speaker_from_blurb_reads_past_tense_taught():
    blurb = "Sergio Gomes taught from 1 Peter 5:8-14 on Peter's final encouragement toward vigilance."
    assert _speaker_from_blurb(blurb) == "Sergio Gomes"


def test_speaker_from_blurb_skips_a_leading_series_clause():
    blurb = "In our series Becoming Who You Are, Mike Gleim teaches from 1 Peter 1:17-21 on the challenge."
    assert _speaker_from_blurb(blurb) == "Mike Gleim"


def test_speaker_from_blurb_drops_a_dr_honorific():
    blurb = "Dr. Amy Orr-Ewing teaches on what it means to read Scripture well."
    assert _speaker_from_blurb(blurb) == "Amy Orr-Ewing"


def test_speaker_from_blurb_is_none_when_no_teaching_verb_follows_a_name():
    blurb = "For Monday, April 14, Melissa Mueller guides us in a reflection on this week's theme."
    assert _speaker_from_blurb(blurb) is None


def test_is_non_sermon_title_flags_the_feeds_side_programs():
    for title in (
        "Still Processing: Episode 8",
        "Week 3: Lent Reflection",
        "Week 3: Fasting Reflection",
        "Letter from Dave Lomas",
        "BREAD 2022: Monday, January 10 - Luke 1",
        "Episode 108: A Manifesto Against Civil Religion",
    ):
        assert is_non_sermon_title(title) is True, title


def test_is_non_sermon_title_keeps_ordinary_sermon_titles():
    for title in ("Esther, the Hidden Hand", "The Rational & The Holy", "Believe"):
        assert is_non_sermon_title(title) is False, title


def _feed(*items: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">'
        "<channel><title>Reality San Francisco</title>" + "".join(items) + "</channel></rss>"
    ).encode()


def _item(guid: str, title: str, pub: str, blurb: str = "") -> str:
    return (
        f"<item><title>{title}</title>"
        f'<guid isPermaLink="false">realitysf.podbean.com/{guid}</guid>'
        f"<link>https://realitysf.podbean.com/e/{guid}/</link>"
        f"<pubDate>{pub}</pubDate>"
        f"<description><![CDATA[{blurb}]]></description>"
        f'<enclosure url="https://mcdn.podbean.com/mf/web/{guid}.mp3" length="1" type="audio/mpeg"/>'
        "</item>"
    )


def test_parse_maps_a_monday_upload_to_the_sunday_it_was_preached():
    content = _feed(
        _item(
            "04eb633d",
            "Esther, the Hidden Hand",
            "Mon, 09 Nov 2026 07:14:53 -0800",
            "<p>Ruthie Kim teaches from Esther 4 on how God is working in the ordinary.</p>",
        )
    )
    items, id_less = _parse(content)
    assert id_less == []
    (sermon,) = items
    assert sermon.guid == "reality_sf:realitysf.podbean.com/04eb633d"
    assert sermon.title == sermon.raw_title == "Esther, the Hidden Hand"
    assert sermon.series is None
    assert sermon.speaker == "Ruthie Kim"
    assert sermon.preached_on == "2026-11-08"
    assert sermon.feed_published_at == datetime(2026, 11, 9, 15, 14, 53, tzinfo=UTC)
    assert sermon.episode_url == "https://realitysf.podbean.com/e/04eb633d/"
    assert sermon.audio_url == "https://mcdn.podbean.com/mf/web/04eb633d.mp3"
    assert sermon.blurb.startswith("<p>Ruthie Kim teaches")


def test_parse_dates_a_sunday_evening_upload_by_the_churchs_local_day():
    # 03:55 UTC Monday is still Sunday evening in San Francisco.
    items, _ = _parse(_feed(_item("a1", "Monsters & Seraphim", "Mon, 26 Oct 2026 03:55:29 +0000")))
    assert items[0].preached_on == "2026-10-25"


def test_parse_lists_items_with_no_guid_or_link_as_excluded():
    no_identity = (
        "<item><title>Untitled upload</title><pubDate>Wed, 28 Oct 2026 09:15:00 -0700</pubDate>"
        '<enclosure url="https://mcdn.podbean.com/mf/web/x.mp3" length="1" type="audio/mpeg"/></item>'
    )
    items, id_less = _parse(_feed(no_identity))
    assert items == []
    assert id_less == [ExcludedItem(raw_title="Untitled upload", published_on=date(2026, 10, 28))]


def test_select_sermons_keeps_the_earliest_upload_of_each_week():
    items, _ = _parse(
        _feed(
            _item("annual", "Annual Vision & Prayer 2025", "Thu, 06 Feb 2025 12:00:00 -0800"),
            _item("sermon", "Proclaiming Christ", "Tue, 04 Feb 2025 10:00:00 -0800"),
        )
    )
    included, excluded = select_sermons(items)
    assert [item.title for item in included] == ["Proclaiming Christ"]
    assert [item.raw_title for item in excluded] == ["Annual Vision & Prayer 2025"]


def test_select_sermons_keeps_a_late_upload_when_its_week_has_nothing_earlier():
    # Easter 2025's sermon went up the Wednesday after.
    items, _ = _parse(_feed(_item("believe", "Believe", "Wed, 23 Apr 2025 14:05:03 -0700")))
    included, excluded = select_sermons(items)
    assert [(item.title, item.preached_on) for item in included] == [("Believe", "2025-04-20")]
    assert excluded == []


def test_select_sermons_skips_a_side_program_posted_before_the_sermon():
    items, _ = _parse(
        _feed(
            _item("sermon", "Look Up & Live", "Mon, 10 Mar 2025 11:00:00 -0700"),
            _item("lent", "Week 1: Lent Reflection", "Mon, 10 Mar 2025 06:00:00 -0700"),
        )
    )
    included, excluded = select_sermons(items)
    assert [item.title for item in included] == ["Look Up & Live"]
    assert [item.raw_title for item in excluded] == ["Week 1: Lent Reflection"]


def test_select_sermons_keeps_one_sermon_per_week_across_weeks():
    items, _ = _parse(
        _feed(
            _item("w2", "The Rational & The Holy", "Mon, 02 Nov 2026 12:05:30 -0800"),
            _item("w1", "Monsters & Seraphim", "Sun, 25 Oct 2026 19:55:29 -0700"),
        )
    )
    included, excluded = select_sermons(items)
    assert [item.preached_on for item in included] == ["2026-11-01", "2026-10-25"]
    assert excluded == []
