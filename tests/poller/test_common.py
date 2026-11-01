from datetime import date

import pytest

from poller.sources.common import (
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
