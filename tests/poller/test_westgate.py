from poller.sources import westgate
from poller.sources.westgate import WestgateAdapter, _parse_title, is_main_sunday_sermon


def test_parse_title_extracts_series_title_and_date():
    title, series, service_date = _parse_title(
        "The Gospel of John (Part 2) | To Whom Shall We Go | August 16, 2026"
    )
    assert title == "To Whom Shall We Go"
    assert series == "The Gospel of John (Part 2)"
    assert service_date.isoformat() == "2026-08-16"


def test_parse_title_tolerates_irregular_spacing_around_the_separators():
    # Live feed title for the 2026-10-25 service.
    title, series, service_date = _parse_title(
        "The Gospel of John (Part 3) |    I am the Light of the World |October 25, 2026"
    )
    assert title == "I am the Light of the World"
    assert series == "The Gospel of John (Part 3)"
    assert service_date.isoformat() == "2026-10-25"


def test_parse_title_accepts_an_abbreviated_month_with_a_period():
    # Live feed title for the 2026-09-06 service.
    title, series, service_date = _parse_title(
        "The Pace Of Peace | Come to Me, Learn from Me, and Rest | Sep. 6, 2026"
    )
    assert title == "Come to Me, Learn from Me, and Rest"
    assert series == "The Pace Of Peace"
    assert service_date.isoformat() == "2026-09-06"


def test_parse_title_accepts_a_date_without_a_comma():
    title, series, service_date = _parse_title("Hunger and Thirst | Pray Continually | March 09 2025")
    assert title == "Pray Continually"
    assert series == "Hunger and Thirst"
    assert service_date.isoformat() == "2025-03-09"


def test_parse_title_with_a_word_that_is_not_a_month_is_unclassifiable():
    title, series, service_date = _parse_title("Series | Title | Someday 5, 2026")
    assert service_date is None
    assert series is None
    assert title == "Series | Title | Someday 5, 2026"


def test_parse_title_without_trailing_date_is_unclassifiable():
    title, series, service_date = _parse_title("An Old Archive Sermon")
    assert service_date is None
    assert series is None
    assert title == "An Old Archive Sermon"


def test_is_main_sunday_sermon_requires_sunday_weekday():
    assert is_main_sunday_sermon("Series | Title | August 16, 2026", 6) is True
    assert is_main_sunday_sermon("Series | Title | August 17, 2026", 0) is False


def test_is_main_sunday_sermon_excludes_nextgen_program():
    assert is_main_sunday_sermon("NextGen Sunday | Title | August 16, 2026", 6) is False


def test_is_main_sunday_sermon_excludes_unparseable_title():
    assert is_main_sunday_sermon("An Old Archive Sermon", -1) is False


_FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>WestGate Church</title>
<item>
  <title>The Gospel of John (Part 3) |    I am the Light of the World |October 25, 2026</title>
  <guid isPermaLink="false">5f1a2b3c4d5e6f7a8b9c0d1e:64a1b2c3d4e5f6a7b8c9d0e1</guid>
  <link>https://www.westgatechurch.org/sermons/i-am-the-light-of-the-world</link>
  <pubDate>Mon, 26 Oct 2026 02:52:52 +0000</pubDate>
  <author>Pastor Example</author>
  <enclosure url="https://static1.squarespace.com/static/example/light-of-the-world.mp3" type="audio/mpeg"/>
</item>
</channel></rss>"""


def test_poll_includes_a_sunday_sermon_whose_title_spacing_varies(monkeypatch):
    monkeypatch.setattr(westgate, "fetch_feed", lambda url: _FEED)

    result = WestgateAdapter(url="https://www.westgatechurch.org/sermons?format=rss").poll()

    assert result.excluded == 0
    assert [(item.title, item.preached_on) for item in result.items] == [
        ("I am the Light of the World", "2026-10-25")
    ]
