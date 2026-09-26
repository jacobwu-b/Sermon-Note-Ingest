from datetime import UTC, date, datetime
from zoneinfo import ZoneInfo

from poller.sources.common import (
    derive_preached_on,
    entry_audio_url,
    entry_has_identity,
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
