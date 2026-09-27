from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

import pytest

from poller.sources.common import (
    derive_preached_on,
    entry_audio_url,
    entry_has_identity,
    is_service_date,
    parse_pubdate,
    parse_pubdate_at,
)


def test_parse_pubdate_returns_date_and_weekday():
    on, weekday = parse_pubdate("Sun, 06 Sep 2026 10:00:00 -0700")
    assert on == "2026-09-06"
    assert weekday == 6


def test_parse_pubdate_missing_value_is_never_sunday():
    assert parse_pubdate(None) == ("", -1)


def test_parse_pubdate_malformed_value_is_never_sunday():
    assert parse_pubdate("not a date") == ("", -1)


def test_parse_pubdate_at_returns_a_tz_aware_instant():
    when = parse_pubdate_at("Sun, 06 Sep 2026 10:00:00 -0700")
    assert when is not None
    assert when.tzinfo is not None


def test_parse_pubdate_at_missing_value_is_none():
    assert parse_pubdate_at(None) is None


def test_entry_audio_url_returns_first_enclosure():
    entry = {"enclosures": [{"href": "https://example.org/a.mp3"}, {"href": "https://example.org/b.mp3"}]}
    assert entry_audio_url(entry) == "https://example.org/a.mp3"


def test_entry_audio_url_with_no_enclosures_is_empty():
    assert entry_audio_url({}) == ""


def test_entry_has_identity_true_when_id_present():
    assert entry_has_identity({"id": "guid-1"}) is True


def test_entry_has_identity_false_when_id_missing():
    assert entry_has_identity({}) is False


def test_derive_preached_on_prefers_a_sunday_title_date():
    feed_published_at = datetime(2026, 9, 8, 10, 0, tzinfo=UTC)  # Tuesday
    title_date = date(2026, 9, 6)  # Sunday
    assert derive_preached_on(feed_published_at, title_date=title_date) == title_date


def test_derive_preached_on_ignores_a_non_sunday_title_date():
    """A title date that isn't itself a Sunday isn't trustworthy as the service
    date (e.g. a typo, or a date that means something else) — fall through to
    the arrival-based derivation instead of trusting it verbatim."""
    feed_published_at = datetime(2026, 9, 8, 10, 0, tzinfo=UTC)  # Tuesday
    title_date = date(2026, 9, 7)  # Monday
    assert derive_preached_on(feed_published_at, title_date=title_date) == date(2026, 9, 6)


def test_derive_preached_on_falls_back_to_nearest_sunday_on_or_before_arrival():
    feed_published_at = datetime(2026, 9, 9, 10, 0, tzinfo=UTC)  # Wednesday
    assert derive_preached_on(feed_published_at) == date(2026, 9, 6)


def test_derive_preached_on_is_a_no_op_when_arrival_is_already_sunday():
    feed_published_at = datetime(2026, 9, 6, 10, 0, tzinfo=UTC)  # Sunday
    assert derive_preached_on(feed_published_at) == date(2026, 9, 6)


def test_derive_preached_on_converts_to_the_given_timezone_before_deriving():
    """Regression: a midnight-UTC Sunday is still Saturday evening in Pacific —
    skipping the conversion derives the wrong week entirely, not just off by a
    day (found via the docs/specs/0007 backtest against GracePres's archive)."""
    feed_published_at = datetime(2018, 12, 2, 0, 0, tzinfo=UTC)  # Sunday in UTC
    assert derive_preached_on(feed_published_at, tz=ZoneInfo("America/Los_Angeles")) == date(2018, 11, 25)


def test_derive_preached_on_without_a_timezone_uses_feed_published_at_as_is():
    feed_published_at = datetime(2018, 12, 2, 0, 0, tzinfo=UTC)
    assert derive_preached_on(feed_published_at) == date(2018, 12, 2)


def test_is_service_date_accepts_an_ordinary_sunday():
    assert is_service_date(date(2026, 9, 20))


def test_is_service_date_rejects_an_ordinary_weekday():
    assert not is_service_date(date(2026, 9, 23))


@pytest.mark.parametrize(
    "day",
    [
        date(2025, 12, 24),  # Christmas Eve, a Wednesday
        date(2026, 12, 25),  # Christmas Day, a Friday
        date(2026, 12, 31),  # New Year's Eve, a Thursday
        date(2027, 1, 1),  # New Year's Day, a Friday
        date(2026, 4, 3),  # Good Friday
        date(2027, 3, 26),  # Good Friday, Easter falling in March
        date(2026, 11, 26),  # Thanksgiving Day
        date(2025, 11, 27),  # Thanksgiving Day, November starting on a Saturday
    ],
)
def test_is_service_date_accepts_off_sunday_special_services(day):
    assert is_service_date(day)


@pytest.mark.parametrize(
    "day",
    [
        date(2026, 4, 2),  # Maundy Thursday — not on the list
        date(2026, 4, 6),  # Easter Monday
        date(2026, 11, 19),  # third Thursday of November
        date(2026, 11, 27),  # the day after Thanksgiving
        date(2026, 12, 23),
        date(2027, 1, 2),
    ],
)
def test_is_service_date_rejects_days_next_to_a_special_service(day):
    assert not is_service_date(day)


@pytest.mark.parametrize(
    ("year", "easter"),
    [
        (1818, date(1818, 3, 22)),  # earliest possible Easter
        (1943, date(1943, 4, 25)),  # latest possible Easter
        (2024, date(2024, 3, 31)),
        (2025, date(2025, 4, 20)),
        (2026, date(2026, 4, 5)),
        (2027, date(2027, 3, 28)),
    ],
)
def test_is_service_date_places_good_friday_two_days_before_easter(year, easter):
    good_friday = date.fromordinal(easter.toordinal() - 2)
    assert is_service_date(good_friday)
    assert not is_service_date(date.fromordinal(easter.toordinal() - 1))
