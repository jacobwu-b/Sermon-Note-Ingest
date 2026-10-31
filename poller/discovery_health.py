"""Outcome-level discovery checks: a missed Sunday, or a feed that keeps failing (ADR-0025).

Discovery fails in ways that each look like "no new sermon this week" — a feed that defers
every run, one that parses to nothing, a classifier that rejects the real sermon. Rather than
guard each cause, this asserts the outcome: every church has a sermon for every Sunday its
publish grace has passed, and its feed is healthy. Each incident alerts once; the state that
remembers which incidents have already alerted is a plain dict that ``poller.store`` persists.

Everything here is pure: the runner gathers the inputs and does the sending and saving.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any

# Not measured: deferrals are only logged, so there is no history to size this from.
FEED_FAILING_AFTER = timedelta(hours=6)

MISSING_SUNDAY = "missing_sunday"
FEED_FAILING = "feed_failing"

_SUNDAY = 6

State = dict[str, dict[str, Any]]


@dataclass(frozen=True)
class ChurchStatus:
    """What one tick knows about one church.

    ``feed_healthy`` is ``None`` when the church wasn't polled to completion this tick, which
    leaves its feed state untouched.
    """

    name: str
    publish_grace: timedelta
    newest_preached_on: date | None
    feed_healthy: bool | None
    snoozed_until: date | None = None


@dataclass(frozen=True)
class Alert:
    church: str
    kind: str
    subject: str
    lines: tuple[str, ...]
    sunday: date | None = None


def latest_due_sunday(now: datetime, publish_grace: timedelta) -> date:
    """The latest Sunday ``S`` with ``S`` 00:00 UTC + ``publish_grace`` at or before ``now``."""
    cutoff = (now - publish_grace).date()
    return cutoff - timedelta(days=(cutoff.weekday() - _SUNDAY) % 7)


def newest_preached_on(records: Mapping[str, Mapping[str, Any]]) -> date | None:
    """The latest parseable ``preached_on`` in a ledger, or ``None`` if it has none."""
    newest = None
    for record in records.values():
        try:
            when = date.fromisoformat(record.get("preached_on") or "")
        except ValueError:
            continue
        if newest is None or when > newest:
            newest = when
    return newest


def evaluate(statuses: Iterable[ChurchStatus], state: State, *, now: datetime) -> tuple[list[Alert], State]:
    """The alerts due this tick, and the next state with none of them marked sent yet.

    The caller marks each alert with :func:`mark_alerted` only once it is delivered, so an
    undelivered alert is raised again on the next tick.
    """
    next_state = {church: dict(entry) for church, entry in state.items()}
    alerts: list[Alert] = []
    for status in statuses:
        entry = next_state.get(status.name, {})
        alert = _missing_sunday(status, entry, now)
        if alert is not None:
            alerts.append(alert)

        if status.feed_healthy is True:
            entry.pop("failing_since", None)
            entry.pop("failing_alerted", None)
        elif status.feed_healthy is False:
            since = entry.setdefault("failing_since", now.isoformat())
            if not entry.get("failing_alerted") and now - datetime.fromisoformat(since) >= FEED_FAILING_AFTER:
                alerts.append(
                    Alert(
                        church=status.name,
                        kind=FEED_FAILING,
                        subject=f"Discovery: {status.name} feed failing since {since}",
                        lines=(
                            (
                                f"{status.name}: every poll since {since} has found the feed unreachable "
                                "or parsed it to no entries."
                            ),
                        ),
                    )
                )

        if entry:
            next_state[status.name] = entry
        else:
            next_state.pop(status.name, None)
    return alerts, next_state


def _missing_sunday(status: ChurchStatus, entry: Mapping[str, Any], now: datetime) -> Alert | None:
    # A church with no dated sermon yet has nothing to compare; an empty feed is the
    # feed-failing check's to catch.
    if status.newest_preached_on is None:
        return None
    sunday = latest_due_sunday(now, status.publish_grace)
    if status.newest_preached_on >= sunday:
        return None
    if status.snoozed_until is not None and sunday <= status.snoozed_until:
        return None
    if entry.get("missing_alerted_for") == sunday.isoformat():
        return None
    return Alert(
        church=status.name,
        kind=MISSING_SUNDAY,
        subject=f"Discovery: no sermon from {status.name} for Sunday {sunday.isoformat()}",
        lines=(
            (
                f"{status.name}: the newest ledgered sermon is {status.newest_preached_on.isoformat()}; "
                f"one for {sunday.isoformat()} was due within {status.publish_grace} of that Sunday."
            ),
            "Check the feed for an item the adapter excluded, or snooze the church in data/alerts/snooze.json.",
        ),
        sunday=sunday,
    )


def mark_alerted(state: State, alert: Alert) -> State:
    """``state`` with ``alert`` recorded as delivered; ``state`` itself is left unchanged."""
    entry = dict(state.get(alert.church, {}))
    if alert.kind == MISSING_SUNDAY and alert.sunday is not None:
        entry["missing_alerted_for"] = alert.sunday.isoformat()
    elif alert.kind == FEED_FAILING:
        entry["failing_alerted"] = True
    return {**state, alert.church: entry}
