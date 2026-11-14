from datetime import UTC, date, datetime

from poller.sources.grace_cathedral import (
    _header_lines,
    _parse,
    _speaker,
    header_date,
    is_special_program,
    select_sermons,
    sermon_title,
    service_date,
)

_REQUIEM_BLURB = (
    "<p>The Very Rev. Dr. Malcolm Clemens Young, Dean<br /> Grace Cathedral, San Francisco, CA<br />"
    " Sunday, November 8th, 2026 11 a. m. Eucharist<br /> Solemn Requiem For All Souls<br /> <br />"
    " Wisdom 3:1-9<br /> Psalm 121<br /> 1 Thessalonians 4:13-18<br /> John 5:24-27</p>"
)
_ERICKSON_BLURB = (
    "<p>The Rev. Susan Erickson<br /> Grace Cathedral, San Francisco, CA<br /> Sunday, October 25th, 2026</p>"
    " <p><br /> Deuteronomy 34:1-12<br /> Psalm 90:1-6, 13-17<br /> 1 Thessalonians 2:1-8<br />"
    " Matthew 22:34-46</p>"
)


def test_header_lines_splits_breaks_and_paragraphs_and_skips_long_prose():
    blurb = (
        "<p>The Very Rev. Dr. Malcolm Clemens Young, Dean<br /> Grace Cathedral, San Francisco, CA<br />"
        " Sunday, November 8th, 2026 11 a. m. Eucharist</p><p>" + "Long sermon summary prose. " * 10 + "</p>"
    )
    assert _header_lines(blurb) == [
        "The Very Rev. Dr. Malcolm Clemens Young, Dean",
        "Grace Cathedral, San Francisco, CA",
        "Sunday, November 8th, 2026 11 a. m. Eucharist",
    ]


def test_header_date_reads_month_day_year_with_ordinal():
    assert header_date(["Grace Cathedral", "March 15th, 2026, 11 a.m. Eucharist"]) == date(2026, 3, 15)


def test_header_date_reads_day_month_year():
    lines = [
        "Grace Cathedral, San Francisco, CA 8 Pentecost (Proper 10A) 11:00 a.m. Eucharist Sunday 19 July 2026"
    ]
    assert header_date(lines) == date(2026, 7, 19)


def test_header_date_reads_a_date_after_a_feast_name():
    assert header_date(["Christmas Eve, December 24th, 2025"]) == date(2025, 12, 24)


def test_header_date_is_none_for_a_liturgical_heading_without_a_calendar_date():
    assert (
        header_date(
            ["Grace Cathedral, San Francisco, CA 2G32", "20 Pentecost (Proper 23A) 11:00 a.m. Eucharist"]
        )
        is None
    )


def test_service_date_prefers_the_header_date():
    uploaded = datetime(2025, 6, 18, 17, 0, tzinfo=UTC)  # a Wednesday
    assert service_date(date(2025, 6, 15), uploaded) == date(2025, 6, 15)


def test_service_date_rejects_a_header_date_that_is_not_a_service_date():
    # Ash Wednesday and Maundy Thursday are out of scope.
    uploaded = datetime(2026, 2, 19, 18, 0, tzinfo=UTC)
    assert service_date(date(2026, 2, 18), uploaded) is None


def test_service_date_falls_back_to_the_upload_when_the_header_year_is_a_typo():
    # A live header read "Sunday January 4th, 2025" on a sermon uploaded that Sunday evening in 2026.
    uploaded = datetime(2026, 1, 5, 0, 0, tzinfo=UTC)
    assert service_date(date(2025, 1, 4), uploaded) == date(2026, 1, 4)


def test_service_date_ignores_a_header_sunday_far_from_the_upload():
    uploaded = datetime(2026, 1, 5, 0, 0, tzinfo=UTC)
    assert service_date(date(2025, 1, 5), uploaded) == date(2026, 1, 4)


def test_service_date_keeps_a_christmas_day_header_date():
    uploaded = datetime(2025, 12, 26, 18, 0, tzinfo=UTC)
    assert service_date(date(2025, 12, 25), uploaded) == date(2025, 12, 25)


def test_service_date_uses_a_sunday_upload_day_in_church_local_time():
    uploaded = datetime(2026, 11, 8, 22, 1, tzinfo=UTC)
    assert service_date(None, uploaded) == date(2026, 11, 8)


def test_service_date_maps_a_monday_upload_to_the_sunday_before():
    uploaded = datetime(2026, 1, 12, 18, 0, tzinfo=UTC)
    assert service_date(None, uploaded) == date(2026, 1, 11)


def test_service_date_maps_a_saturday_upload_to_the_next_morning():
    # Sermons sometimes go up the afternoon before they are preached.
    uploaded = datetime(2026, 9, 26, 21, 0, tzinfo=UTC)
    assert service_date(None, uploaded) == date(2026, 9, 27)


def test_service_date_keeps_a_christmas_eve_upload():
    uploaded = datetime(2025, 12, 25, 6, 0, tzinfo=UTC)  # 22:00 on the 24th in San Francisco
    assert service_date(None, uploaded) == date(2025, 12, 24)


def test_service_date_is_none_for_an_undated_midweek_upload():
    uploaded = datetime(2025, 9, 12, 2, 0, tzinfo=UTC)  # Thursday evening, a 9/11 Evensong
    assert service_date(None, uploaded) is None


def test_is_special_program_flags_services_out_of_scope():
    assert is_special_program(["Pride Mass"]) is True
    assert (
        is_special_program(["Joe C. Williams", "Grace Cathedral, San Francisco, CA", "Pride Mass 6:00 p.m."])
        is True
    )
    assert is_special_program(["Diocesan Ordinations of a Deacon and Priests"]) is True
    assert (
        is_special_program(["The Right Rev. Austin Keith Rios - Sermon from Diocesan Confirmation"]) is True
    )
    assert is_special_program(["Why Do We Have an Easter Vigil?"]) is True
    assert is_special_program(["Evensong 60 5:30 p.m. Choral Evensong"]) is True
    assert is_special_program(["The Great Three Days: Maundy Thursday"]) is True


def test_is_special_program_keeps_a_sunday_eucharist():
    assert (
        is_special_program(
            ["Pride Sunday: It's All in How you Read It", "5 Pentecost (Proper 8A) 11:00 a.m. Eucharist"]
        )
        is False
    )


def test_speaker_strips_honorifics_and_role():
    assert (
        _speaker("Coming Out", ["The Very Rev. Dr. Malcolm Clemens Young", "Grace Cathedral"])
        == "Malcolm Clemens Young"
    )
    assert (
        _speaker("x", ["The Rev. Canon Mary Carter Greene, Canon Pastor", "Grace Cathedral"])
        == "Mary Carter Greene"
    )
    assert (
        _speaker("x", ["The Rev. Br. Richard Edward Helmer, Precentor", "Grace Cathedral"])
        == "Richard Edward Helmer"
    )
    assert (
        _speaker("x", ["The Very Revd Andrew Tremlett", "Dean of St Paul's Cathedral, London UK"])
        == "Andrew Tremlett"
    )


def test_speaker_reads_a_bare_name_line_above_the_cathedral_line():
    assert (
        _speaker("x", ["Malcolm Clemens Young", "Grace Cathedral, San Francisco, CA 2G30"])
        == "Malcolm Clemens Young"
    )


def test_speaker_falls_back_to_a_title_that_names_the_preacher():
    assert _speaker("The Right Rev. Austin Keith Rios - Easter Sunday Sermon", []) == "Austin Keith Rios"


def test_speaker_is_none_when_nothing_names_one():
    assert _speaker("Hosanna Means Help Us", ["What does Hosanna really mean?"]) is None


def _feed(*items: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">'
        "<channel><title>Sermons from Grace Cathedral</title>" + "".join(items) + "</channel></rss>"
    ).encode()


def _item(guid: str, title: str, pub: str, blurb: str = "", audio: str | None = None) -> str:
    audio = audio or f"https://traffic.libsyn.com/secure/gracecathedral/{guid}.mp3"
    return (
        f"<item><title>{title}</title>"
        f'<guid isPermaLink="false">{guid}</guid>'
        f"<link>https://gracecathedral.libsyn.com/{guid}</link>"
        f"<pubDate>{pub}</pubDate>"
        f"<description><![CDATA[{blurb}]]></description>"
        f'<enclosure url="{audio}" length="1" type="audio/mpeg"/>'
        "</item>"
    )


def test_parse_maps_a_sunday_sermon():
    blurb = _REQUIEM_BLURB
    items, id_less = _parse(
        _feed(
            _item(
                "f2ef981c",
                "The Very Rev. Dr. Malcolm Clemens Young",
                "Sun, 08 Nov 2026 22:01:00 +0000",
                blurb,
            )
        )
    )
    assert id_less == []
    (sermon,) = items
    assert sermon.guid == "grace_cathedral:f2ef981c"
    assert sermon.title == "Solemn Requiem For All Souls"
    assert sermon.raw_title == "The Very Rev. Dr. Malcolm Clemens Young"
    assert sermon.series is None
    assert sermon.speaker == "Malcolm Clemens Young"
    assert sermon.preached_on == "2026-11-08"
    assert sermon.feed_published_at == datetime(2026, 11, 8, 22, 1, tzinfo=UTC)
    assert sermon.episode_url == "https://gracecathedral.libsyn.com/f2ef981c"
    assert sermon.audio_url == "https://traffic.libsyn.com/secure/gracecathedral/f2ef981c.mp3"


def test_select_sermons_keeps_sundays_and_feasts_and_drops_the_rest():
    items, _ = _parse(
        _feed(
            _item("easter-dean", "Easter and What Heaven Really Is", "Sun, 05 Apr 2026 22:00:00 +0000"),
            _item(
                "easter-bishop",
                "The Right Rev. Austin Keith Rios - Easter Sunday Sermon",
                "Sun, 05 Apr 2026 22:30:00 +0000",
            ),
            _item("vigil", "Why Do We Have an Easter Vigil?", "Sun, 05 Apr 2026 05:00:00 +0000"),
            _item(
                "good-friday",
                "The Abstract Cross - A Reflection for Good Friday",
                "Fri, 03 Apr 2026 23:00:00 +0000",
            ),
            _item(
                "maundy",
                "Nobody Asked Judas Why",
                "Fri, 03 Apr 2026 04:00:00 +0000",
                "<p>The Rev. Canon Mary Carter Greene<br/>Grace Cathedral, San Francisco, CA<br/>April 2nd, 2026</p>",
            ),
        )
    )
    included, excluded = select_sermons(items)
    assert sorted(item.title for item in included) == [
        "Easter and What Heaven Really Is",
        "The Abstract Cross - A Reflection for Good Friday",
        "The Right Rev. Austin Keith Rios - Easter Sunday Sermon",
    ]
    assert sorted(item.raw_title for item in excluded) == [
        "Nobody Asked Judas Why",
        "Why Do We Have an Easter Vigil?",
    ]


def test_select_sermons_drops_a_repost_of_the_same_audio():
    shared = "https://traffic.libsyn.com/secure/gracecathedral/singer.mp3"
    items, _ = _parse(
        _feed(
            _item("first", "The Rev. Susanna Singer", "Sun, 22 Jun 2025 22:00:00 +0000", audio=shared),
            _item("repost", "The Rev. Susanna Singer", "Sun, 22 Jun 2025 22:00:00 +0000", audio=shared),
        )
    )
    included, excluded = select_sermons(items)
    assert [item.guid for item in included] == ["grace_cathedral:first"]
    assert [item.raw_title for item in excluded] == ["The Rev. Susanna Singer"]


def test_sermon_title_names_a_name_only_sermon_by_its_service():
    assert (
        sermon_title("The Very Rev. Dr. Malcolm Clemens Young", "Malcolm Clemens Young", _REQUIEM_BLURB)
        == "Solemn Requiem For All Souls"
    )


def test_sermon_title_names_a_name_only_sermon_by_its_gospel_reading():
    assert sermon_title("The Rev. Susan Erickson", "Susan Erickson", _ERICKSON_BLURB) == "Matthew 22:34-46"


def test_sermon_title_reads_past_a_role_line_and_a_title_spelled_unlike_the_blurb():
    blurb = (
        "<p>The Rt. Rev William Swing<br /> VII Bishop of California<br /> Grace Cathedral, San Francisco, CA<br />"
        " Sunday, September 13th, 2026</p><p>Exodus 14:19-31<br /> Psalm 114<br /> Romans 14:1-12<br />"
        " Matthew 18:21-35</p>"
    )
    assert sermon_title("The Rt. Rev. William Swing", "William Swing", blurb) == "Matthew 18:21-35"


def test_sermon_title_ignores_a_feast_named_before_the_date_line():
    blurb = (
        "<p>The Rev. Canon J. Sierra Reyes</p> <p>Canon to the Ordinary, The Diocese of California</p>"
        " <p>The Feast of St. Michael and All Angels</p> <p>11 am Eucharist</p> <p>Sunday 28 September 2025</p>"
        " <p>\xa0</p> <p>Genesis 28:10-17</p> <p>Psalm 103:19-22</p> <p>Revelation 12:7-12</p> <p>John 1:47-51</p>"
    )
    assert sermon_title("The Rev. Canon J. Sierra Reyes", "J. Sierra Reyes", blurb) == "John 1:47-51"


def test_sermon_title_skips_an_epistle_of_john_for_the_gospel():
    blurb = (
        "<p>The Rev. Susanna Singer<br />Grace Cathedral, San Francisco, CA<br />June 22, 2025</p>"
        "<p>Acts 2:1-4<br />Psalm 42:1-7<br />Luke 8:26-39<br />1 John 3:1-3</p>"
    )
    assert sermon_title("The Rev. Susanna Singer", "Susanna Singer", blurb) == "Luke 8:26-39"


def test_sermon_title_uses_the_gospel_when_two_lines_sit_before_the_readings():
    blurb = (
        "<p>The Rev. Susan Erickson<br />Grace Cathedral, San Francisco, CA<br />Sunday, October 25th, 2026<br />"
        "Proper Twenty-Five<br />Commemoration of the Faithful Departed<br />Psalm 90<br />Matthew 22:34-46</p>"
    )
    assert sermon_title("The Rev. Susan Erickson", "Susan Erickson", blurb) == "Matthew 22:34-46"


def test_sermon_title_ignores_a_reference_quoted_inside_prose():
    blurb = (
        "<p>The Very Revd Andrew Tremlett</p><p>Dean of St Paul's Cathedral, London UK</p>"
        '<p>"But we speak God\'s wisdom, a hidden mystery" (1 Cor. 2.7) and so John 3:16 too.</p>'
        "<p>Preached on Sunday February 8th, 2026</p><p>Grace Cathedral San Francisco, CA</p>"
    )
    assert (
        sermon_title("The Very Revd Andrew Tremlett", "Andrew Tremlett", blurb)
        == "The Very Revd Andrew Tremlett"
    )


def test_sermon_title_keeps_the_feed_title_when_the_blurb_names_no_reading():
    blurb = (
        "<p>The Rev. Canon Dr. Stephanie Spellers</p><p>Preached Sunday January 18th, 2026</p>"
        "<p>The Second Sunday After the Epiphany</p><p>Grace Cathedral San Francisco, CA</p>"
    )
    title = "The Rev. Canon Dr. Stephanie Spellers"
    assert sermon_title(title, "Stephanie Spellers", blurb) == title


def test_sermon_title_names_a_title_that_is_the_bare_speaker():
    assert sermon_title("Susan Erickson", "Susan Erickson", _ERICKSON_BLURB) == "Matthew 22:34-46"


def test_sermon_title_keeps_a_real_title():
    assert sermon_title("Why Did Jesus Have to Die?", "Susan Erickson", _ERICKSON_BLURB) == (
        "Why Did Jesus Have to Die?"
    )


def test_sermon_title_keeps_a_title_with_more_than_the_name():
    title = "The Right Rev. Austin Keith Rios - Easter Sunday Sermon"
    blurb = (
        "<p>The Right Rev. Austin Keith Rios<br />Grace Cathedral, San Francisco, CA<br />Sunday 5 April 2026<br />"
        "Acts 10:34-43<br />Ps. 118:1-2, 14-24<br />Colossians 3:1-4<br />John 20:1-18</p>"
    )
    assert sermon_title(title, "Austin Keith Rios", blurb) == title
