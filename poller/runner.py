"""The poll entry point: fetch every enabled church, ledger new sermons, notify.

One run touches every church the church table marks ``enabled`` (or the subset
named by ``--church``), independently — one church's feed being down, or its
notification failing, never stops the others from being polled and recorded.
``--backfill`` seeds a church's ledger from its full feed history without
sending any notification for what it adds, so a first run (or a run after
adding a new church) doesn't blast an inbox with years of back-catalog. Every
run ends by regenerating ``data/README.md``'s stats section from whatever the
ledgers now hold, whether or not this run found anything new, and by checking each
polled church for a missed Sunday or a feed that keeps failing (ADR-0025).
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime

from poller import batch_poll, config, discovery_health, notify, stats, store
from poller.net import now
from poller.sources import ADAPTERS

logger = logging.getLogger("poller")


def poll_church(
    name: str,
    entry: config.ChurchConfig,
    *,
    backfill: bool,
    discovered_out: list[str] | None = None,
    health_out: dict[str, bool] | None = None,
) -> bool:
    """Poll one church, ledger any new sermons, refresh known ones, and notify if configured.

    Returns ``True`` on success (including "feed unavailable, deferred" — that
    is an expected, retried-next-run outcome, not a failure of this run).

    ``discovered_out``, when given, gets ``name`` appended iff this poll ledgered
    at least one new sermon (whether or not it was a backfill or notified) — used
    by :func:`run` to report which churches ``poll.yml`` should dispatch
    transcription for (ADR-0007).

    ``health_out``, when given, gets ``name`` mapped to whether its feed was healthy: ``False``
    when it deferred or parsed to no entries at all (ADR-0025).
    """
    adapter_cls = ADAPTERS.get(name)
    if adapter_cls is None:
        logger.warning("%s: no adapter registered for this church; skipping", name)
        return True

    adapter = adapter_cls(url=entry.rss)
    result = adapter.poll()
    if result.deferred:
        logger.warning("%s: feed unavailable, deferring to next run", name)
        if health_out is not None:
            health_out[name] = False
        return True
    # Every real feed carries history, so a fetch that yields nothing at all — not even an
    # excluded item — is a challenge page or a changed format, not a quiet week.
    feed_empty = not result.items and result.excluded == 0
    if feed_empty:
        logger.warning("%s: feed fetched but parsed to no entries", name)
    if health_out is not None:
        health_out[name] = not feed_empty

    records = store.load(name)
    retrieved_at = now()
    new_items = [item for item in result.items if item.guid not in records]
    existing_items = [item for item in result.items if item.guid in records]
    for item in new_items:
        feed_published_at = adapter.resolve_feed_published_at(item)
        records[item.guid] = store.item_to_record(
            item,
            first_seen_at=retrieved_at,
            feed_published_at=feed_published_at.isoformat() if feed_published_at is not None else None,
        )
    # Refreshes audio_url/title/etc. in place for guids the feed still lists — a feed
    # can rotate an enclosure URL after first discovery, and a pending sermon must not
    # be stuck retrying a URL the feed no longer serves (store.refresh_record).
    for item in existing_items:
        store.refresh_record(records[item.guid], item)

    logger.info(
        "%s: %d discovered, %d new, %d excluded",
        name,
        len(result.items),
        len(new_items),
        result.excluded,
    )

    if not new_items:
        if existing_items:
            store.save(name, records)
        return True

    if discovered_out is not None:
        discovered_out.append(name)

    if backfill:
        # Backfill seeds history without emailing — mark it already-notified so
        # a later normal run never sends for these.
        for item in new_items:
            records[item.guid]["notified_at"] = retrieved_at
        store.save(name, records)
        return True

    store.save(name, records)

    if not entry.notify:
        return True

    try:
        notify_cfg = config.load_notify_config()
        notify.send_new_sermons(name, new_items, notify_cfg)
    except (config.ConfigError, notify.NotifyError) as exc:
        logger.error(
            "%s: notification failed for %d new sermon(s): %s",
            name,
            len(new_items),
            exc,
        )
        return False

    for item in new_items:
        records[item.guid]["notified_at"] = now()
    store.save(name, records)
    return True


def run(
    *,
    church_names: list[str] | None,
    backfill: bool,
    discovered_out: list[str] | None = None,
) -> bool:
    """Poll every selected, enabled church. Returns ``True`` iff all succeeded.

    ``discovered_out``, when given, collects the name of every church that had at
    least one new sermon this run (ADR-0007) — the set ``poll.yml`` dispatches
    transcription for.
    """
    churches = config.load_churches()
    selected = {
        name: entry
        for name, entry in churches.items()
        if entry.enabled and (church_names is None or name in church_names)
    }
    if not selected:
        logger.warning("no enabled churches matched the selection; nothing to poll")
        return True

    all_ok = True
    health: dict[str, bool] = {}
    for name, entry in selected.items():
        try:
            ok = poll_church(name, entry, backfill=backfill, discovered_out=discovered_out, health_out=health)
        except Exception:
            logger.exception("%s: poll crashed unexpectedly", name)
            ok = False
        all_ok = all_ok and ok

    try:
        stats.regenerate()
    except stats.StatsError:
        logger.exception("failed to regenerate data/README.md stats")
        all_ok = False

    try:
        all_ok = check_discovery(list(selected), health) and all_ok
    except Exception:
        logger.exception("discovery check crashed unexpectedly")
        all_ok = False

    # Best-effort latency optimization (docs/decisions/0010), never a correctness
    # dependency — Pipeline's own cron is the backstop, so a crash here must not
    # fail an otherwise-successful poll run.
    try:
        batch_poll.poll_pending_batches()
    except Exception:
        logger.exception("batch poll crashed unexpectedly")

    return all_ok


def check_discovery(church_names: list[str], health: dict[str, bool]) -> bool:
    """Raise this tick's discovery alerts (ADR-0025). Returns ``False`` iff any was due.

    An alert fails exactly the run that raises it: once delivered it is recorded and not
    raised again, so cron-job.org's failure email fires once per incident rather than every
    tick. An undelivered one stays unrecorded and is raised again next tick.
    """
    snoozes = store.load_alert_snoozes()
    statuses = [
        discovery_health.ChurchStatus(
            name=name,
            publish_grace=ADAPTERS[name].publish_grace,
            newest_preached_on=discovery_health.newest_preached_on(store.load(name)),
            feed_healthy=health.get(name),
            snoozed_until=date.fromisoformat(snoozes[name]) if name in snoozes else None,
        )
        for name in church_names
        if name in ADAPTERS
    ]
    state = store.load_alert_state()
    alerts, next_state = discovery_health.evaluate(statuses, state, now=datetime.fromisoformat(now()))

    notify_cfg = None
    if alerts:
        try:
            notify_cfg = config.load_notify_config()
        except config.ConfigError as exc:
            logger.error("cannot send %d discovery alert(s): %s", len(alerts), exc)
    for alert in alerts:
        logger.error("%s", alert.subject)
        if notify_cfg is None:
            continue
        try:
            notify.send_discovery_alert(alert.subject, alert.lines, notify_cfg)
        except notify.NotifyError as exc:
            logger.error("%s: discovery alert failed to send: %s", alert.church, exc)
            continue
        next_state = discovery_health.mark_alerted(next_state, alert)

    # Written only on a change, so a quiet tick leaves poll.yml nothing to commit.
    if next_state != state:
        store.save_alert_state(next_state)
    return not alerts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--church",
        action="append",
        dest="churches",
        help="Only poll this church (repeatable). Default: every enabled church.",
    )
    parser.add_argument(
        "--backfill",
        action="store_true",
        help="Seed the ledger from the full feed history without sending notifications.",
    )
    parser.add_argument("--verbose", action="store_true", help="Debug-level logging.")
    parser.add_argument(
        "--print-discovered",
        action="store_true",
        help=(
            "After polling, print a comma-joined list of churches that had a new sermon "
            "this run (empty line if none). Used by poll.yml to know which churches to "
            "dispatch transcription for (ADR-0007)."
        ),
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    discovered: list[str] | None = [] if args.print_discovered else None
    ok = run(church_names=args.churches, backfill=args.backfill, discovered_out=discovered)
    if args.print_discovered:
        # Machine-readable stdout, not a log line — same designed-output exemption as
        # transcriber.py's --print-shard-count (CLAUDE.md §6).
        print(",".join(discovered or []))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
