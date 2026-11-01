from datetime import UTC, date, datetime, timedelta

from poller import discovery_health
from poller.discovery_health import ChurchStatus
from poller.sources.base import ExcludedItem

# Thursday after Westgate's 2026-10-25 sermon was dropped; its newest ledgered sermon was 10-18.
_THU = datetime(2026, 10, 29, 15, 0, tzinfo=UTC)


def _status(
    name: str = "westgate",
    *,
    newest: date | None = date(2026, 10, 18),
    grace_hours: int = 96,
    feed_healthy: bool | None = True,
    snoozed_until: date | None = None,
    excluded: tuple[ExcludedItem, ...] = (),
) -> ChurchStatus:
    return ChurchStatus(
        name=name,
        publish_grace=timedelta(hours=grace_hours),
        newest_preached_on=newest,
        feed_healthy=feed_healthy,
        snoozed_until=snoozed_until,
        excluded=excluded,
    )


def _evaluate(statuses, state=None, now=_THU):
    return discovery_health.evaluate(statuses, state or {}, now=now)


def _send_all(alerts, state):
    for alert in alerts:
        state = discovery_health.mark_alerted(state, alert)
    return state


# --- latest_due_sunday ---


def test_latest_due_sunday_is_last_sunday_once_its_grace_has_passed():
    # 2026-10-25 00:00 UTC + 96h = 2026-10-29 00:00 UTC, before Thursday 15:00.
    assert discovery_health.latest_due_sunday(_THU, timedelta(hours=96)) == date(2026, 10, 25)


def test_latest_due_sunday_falls_back_a_week_while_last_sundays_grace_is_running():
    # 2026-10-25 + 120h = 2026-10-30 00:00 UTC, still ahead of Thursday 15:00.
    assert discovery_health.latest_due_sunday(_THU, timedelta(hours=120)) == date(2026, 10, 18)


def test_latest_due_sunday_is_due_exactly_at_the_grace_boundary():
    boundary = datetime(2026, 10, 27, 0, 0, tzinfo=UTC)
    assert discovery_health.latest_due_sunday(boundary, timedelta(hours=48)) == date(2026, 10, 25)
    just_before = boundary - timedelta(seconds=1)
    assert discovery_health.latest_due_sunday(just_before, timedelta(hours=48)) == date(2026, 10, 18)


# --- newest_preached_on ---


def test_newest_preached_on_takes_the_latest_parseable_date_and_skips_the_rest():
    records = {
        "a": {"preached_on": "2026-10-11"},
        "b": {"preached_on": "2026-10-18"},
        "c": {"preached_on": None},
        "d": {"preached_on": ""},
        "e": {"preached_on": "not a date"},
    }
    assert discovery_health.newest_preached_on(records, on_or_before=_THU.date()) == date(2026, 10, 18)


def test_newest_preached_on_ignores_a_date_after_today_so_a_typo_cannot_cover_every_sunday():
    records = {
        "a": {"preached_on": "2026-10-18"},
        # A typed title date a year out ("October 24, 2027" is a Sunday) passes every adapter check.
        "b": {"preached_on": "2027-10-24"},
    }
    assert discovery_health.newest_preached_on(records, on_or_before=_THU.date()) == date(2026, 10, 18)


def test_newest_preached_on_counts_a_sermon_dated_today():
    records = {"a": {"preached_on": "2026-10-29"}}
    assert discovery_health.newest_preached_on(records, on_or_before=date(2026, 10, 29)) == date(2026, 10, 29)


def test_newest_preached_on_is_none_for_a_ledger_with_no_dated_sermon():
    assert discovery_health.newest_preached_on({}, on_or_before=_THU.date()) is None
    assert discovery_health.newest_preached_on({"a": {"preached_on": None}}, on_or_before=_THU.date()) is None


# --- missing Sunday ---


def test_a_church_missing_its_latest_due_sunday_raises_one_alert_naming_both_dates():
    alerts, _ = _evaluate([_status()])

    assert [(a.church, a.kind, a.sunday) for a in alerts] == [
        ("westgate", "missing_sunday", date(2026, 10, 25))
    ]
    text = alerts[0].subject + " ".join(alerts[0].lines)
    assert "2026-10-25" in text
    assert "2026-10-18" in text


def test_a_missing_sunday_alert_lists_the_excluded_items_dated_on_or_after_that_sunday():
    excluded = (
        ExcludedItem("NextGen Sunday | Light in the Dark | October 18, 2026", date(2026, 10, 18)),
        ExcludedItem(
            "The Gospel of John (Part 3) |    I am the Light of the World |Oct 25 2026", date(2026, 10, 26)
        ),
        ExcludedItem("NextGen Sunday | Walking in Light | October 25, 2026", date(2026, 10, 25)),
        ExcludedItem("Untitled upload", None),
    )
    alerts, _ = _evaluate([_status(excluded=excluded)])

    lines = alerts[0].lines
    assert "2026-10-26 — The Gospel of John (Part 3) |    I am the Light of the World |Oct 25 2026" in lines
    assert "2026-10-25 — NextGen Sunday | Walking in Light | October 25, 2026" in lines
    text = " ".join(lines)
    assert "October 18, 2026" not in text
    assert "Untitled upload" not in text


def test_a_missing_sunday_alert_says_so_when_the_feed_excluded_nothing_since_that_sunday():
    excluded = (ExcludedItem("NextGen Sunday | Light in the Dark | October 18, 2026", date(2026, 10, 18)),)
    alerts, _ = _evaluate([_status(excluded=excluded)])

    assert any("no item excluded" in line and "2026-10-25" in line for line in alerts[0].lines)


def test_a_missing_sunday_alert_says_nothing_about_exclusions_when_the_feed_was_not_read():
    for feed_healthy in (False, None):
        excluded = (ExcludedItem("Untitled upload", date(2026, 10, 27)),)
        alerts, _ = _evaluate([_status(feed_healthy=feed_healthy, excluded=excluded)])

        missing = [a for a in alerts if a.kind == "missing_sunday"]
        text = " ".join(missing[0].lines)
        assert "Untitled upload" not in text, feed_healthy
        assert "no item excluded" not in text, feed_healthy


def test_a_church_with_a_sermon_for_its_latest_due_sunday_raises_nothing():
    alerts, state = _evaluate([_status(newest=date(2026, 10, 25))])
    assert alerts == []
    assert state == {}


def test_a_missing_sunday_alert_fires_once_then_again_for_the_next_sunday():
    alerts, state = _evaluate([_status()])
    state = _send_all(alerts, state)

    later_same_week = _THU + timedelta(days=2)
    alerts, state = _evaluate([_status()], state, now=later_same_week)
    assert alerts == []

    next_week = _THU + timedelta(days=7)
    alerts, _ = _evaluate([_status()], state, now=next_week)
    assert [a.sunday for a in alerts] == [date(2026, 11, 1)]


def test_an_unsent_missing_sunday_alert_is_raised_again_on_the_next_tick():
    alerts, state = _evaluate([_status()])
    # Not passed through mark_alerted: the send failed.
    alerts, _ = _evaluate([_status()], state, now=_THU + timedelta(minutes=5))
    assert [a.kind for a in alerts] == ["missing_sunday"]


def test_a_church_snoozed_through_the_due_sunday_raises_nothing():
    alerts, _ = _evaluate([_status(snoozed_until=date(2026, 10, 25))])
    assert alerts == []


def test_a_snooze_ending_before_the_due_sunday_does_not_suppress_it():
    alerts, _ = _evaluate([_status(snoozed_until=date(2026, 10, 24))])
    assert [a.kind for a in alerts] == ["missing_sunday"]


def test_a_church_with_no_dated_sermon_raises_no_missing_sunday_alert():
    alerts, _ = _evaluate([_status(newest=None)])
    assert alerts == []


def test_each_churchs_own_grace_decides_whether_last_sunday_is_due():
    # Both last ledgered 10-18. Westgate's 96h grace has passed for 10-25; PBC's 120h hasn't.
    alerts, _ = _evaluate([_status("westgate", grace_hours=96), _status("pbc", grace_hours=120)])
    assert [a.church for a in alerts] == ["westgate"]


# --- feed failing ---


def _failing(name: str = "westgate") -> ChurchStatus:
    # A healthy ledger, so only the feed check can fire.
    return _status(name, newest=date(2026, 10, 25), feed_healthy=False)


def test_a_first_failing_poll_records_when_the_failure_started_and_raises_nothing():
    alerts, state = _evaluate([_failing()])
    assert alerts == []
    assert state == {"westgate": {"failing_since": _THU.isoformat()}}


def test_a_feed_failing_for_just_under_six_hours_raises_nothing():
    _, state = _evaluate([_failing()])
    alerts, _ = _evaluate([_failing()], state, now=_THU + timedelta(hours=6) - timedelta(seconds=1))
    assert alerts == []


def test_a_feed_failing_for_six_hours_raises_one_alert_naming_when_it_started():
    _, state = _evaluate([_failing()])
    alerts, state = _evaluate([_failing()], state, now=_THU + timedelta(hours=6))

    assert [(a.church, a.kind) for a in alerts] == [("westgate", "feed_failing")]
    assert _THU.isoformat() in alerts[0].subject + " ".join(alerts[0].lines)

    state = _send_all(alerts, state)
    alerts, _ = _evaluate([_failing()], state, now=_THU + timedelta(hours=9))
    assert alerts == []


def test_a_healthy_poll_ends_the_incident_and_a_later_failure_starts_a_new_one():
    _, state = _evaluate([_failing()])
    alerts, state = _evaluate([_failing()], state, now=_THU + timedelta(hours=6))
    state = _send_all(alerts, state)

    healthy = _status(newest=date(2026, 10, 25), feed_healthy=True)
    alerts, state = _evaluate([healthy], state, now=_THU + timedelta(hours=7))
    assert alerts == []
    assert state == {}

    restart = _THU + timedelta(hours=8)
    _, state = _evaluate([_failing()], state, now=restart)
    alerts, _ = _evaluate([_failing()], state, now=restart + timedelta(hours=6))
    assert [a.kind for a in alerts] == ["feed_failing"]


def test_a_church_not_polled_this_tick_leaves_its_feed_state_untouched():
    _, state = _evaluate([_failing()])
    not_polled = _status(newest=date(2026, 10, 25), feed_healthy=None)
    alerts, after = _evaluate([not_polled], state, now=_THU + timedelta(hours=7))
    assert alerts == []
    assert after == state


def test_evaluate_does_not_mutate_the_state_it_was_given():
    _, state = _evaluate([_failing()])
    snapshot = {church: dict(entry) for church, entry in state.items()}
    alerts, _ = _evaluate([_failing()], state, now=_THU + timedelta(hours=6))
    discovery_health.mark_alerted(state, alerts[0])
    assert state == snapshot


def test_the_state_of_a_church_no_longer_selected_is_kept():
    other = {"menlo": {"missing_alerted_for": "2026-10-18"}}
    _, state = _evaluate([_status(newest=date(2026, 10, 25))], other)
    assert state == other
