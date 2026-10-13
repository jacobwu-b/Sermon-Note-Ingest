"""Check the ledger against Content after a re-transcription run (ADR-0022).

The invariant: every ``done`` record's Content file exists and hashes to the record's
``transcript_hash``; every replaced record (one with a ``superseded_transcript_hash``)
names the full profile that produced it; and no guid owns two Content files. Any
violation fails the run and is emailed.

Also reports the backlog — stale, in scope, replaced and failed per church — to the step
summary. On the scheduled pass (``--scheduled``) it alerts on absence: nothing replaced in
24 hours while stale sermons are still in scope, and once, on the run that empties the
backlog, that the pass is complete.

Logs and emails name churches, guids, paths and hash prefixes only, never text (ADR-0004).

``python -m poller.verify_content [--summary-md PATH] [--scheduled --planned N]``
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath

from poller import config, content_repo, notify, store, transcribe, transcriber

logger = logging.getLogger("poller")

SendAlert = Callable[[str, Sequence[str]], None]

_DEAD_MAN_WINDOW = timedelta(hours=24)
# Every profile field a replacement records; ``hotwords`` is ``None`` whenever none were sent.
_REQUIRED_ON_REPLACEMENT = tuple(key for key in store.PROFILE_KEYS if key != "hotwords")


def _guid_part(path: str) -> str | None:
    """The sanitized guid at the end of ``<preached_on>_<slug>_<guid>.txt``.

    Neither the date (or ``undated``) nor the slug holds an underscore, so everything
    after the second one is the guid.
    """
    parts = PurePosixPath(path).stem.split("_", 2)
    return parts[2] if len(parts) == 3 else None


def violations(ledgers: Mapping[str, Mapping[str, dict]], content: Mapping[str, str]) -> list[str]:
    """Every way ``ledgers`` (church → guid → record) and Content's ``transcripts/`` disagree."""
    found = []
    for church, records in ledgers.items():
        for guid, record in records.items():
            if record.get("transcription_status") != "done":
                continue
            path = record.get("content_path")
            text = content.get(path) if path else None
            if text is None:
                found.append(f"{church}/{guid}: {path} is missing from Content")
                continue
            actual = transcribe.transcript_hash(text)
            if actual != record.get("transcript_hash"):
                found.append(
                    f"{church}/{guid}: {path} hashes to {actual[:12]}, "
                    f"ledger names {(record.get('transcript_hash') or '')[:12]}"
                )
            if record.get("superseded_transcript_hash"):
                missing = [
                    key for key in _REQUIRED_ON_REPLACEMENT if record.get(f"transcription_{key}") is None
                ]
                if missing:
                    found.append(f"{church}/{guid}: replaced transcript records no {', '.join(missing)}")

        by_guid: dict[str, list[str]] = {}
        for path in content:
            if path.startswith(f"transcripts/{church}/") and (guid_part := _guid_part(path)):
                by_guid.setdefault(guid_part, []).append(path)
        for paths in by_guid.values():
            if len(paths) > 1:
                found.append(
                    f"{church}: one guid owns {len(paths)} Content files: {', '.join(sorted(paths))}"
                )
    return found


@dataclass
class Backlog:
    church: str
    stale: int = 0
    in_scope: int = 0
    replaced: int = 0
    failed: Counter[str] = field(default_factory=Counter)
    last_replaced_at: str | None = None


def backlog(churches: Iterable[str]) -> list[Backlog]:
    """Per church: stale ``done`` records, how many the pass would still select, how many
    it replaced, and failures under the running profile by reason."""
    running = transcriber._running_profile()
    digest = store.profile_digest(running)
    rows = []
    for church in churches:
        records = store.load(church)
        row = Backlog(church=church)
        row.in_scope = len(transcriber.select_stale(records, running=running, digest=digest))
        for record in records.values():
            if record.get("transcription_status") != "done":
                continue
            if store.profile_is_stale(record, running):
                row.stale += 1
                if record.get("retranscription_failed_profile") == digest:
                    row.failed[record.get("retranscription_last_error") or "unknown"] += 1
            if record.get("superseded_transcript_hash"):
                row.replaced += 1
                at = record.get("transcribed_at")
                if at and (row.last_replaced_at is None or at > row.last_replaced_at):
                    row.last_replaced_at = at
        rows.append(row)
    return rows


def render_summary(rows: Sequence[Backlog], found: Sequence[str]) -> str:
    lines = [
        "### Re-transcription backlog",
        "",
        "| Church | Stale | In scope | Replaced | Failed under the running profile |",
        "|---|---|---|---|---|",
    ]
    for row in rows:
        failed = ", ".join(f"{reason} {n}" for reason, n in sorted(row.failed.items()))
        lines.append(f"| {row.church} | {row.stale} | {row.in_scope} | {row.replaced} | {failed} |")
    lines += ["", f"**{len(found)} violations** of the ledger ↔ Content invariant."]
    lines += [f"- {violation}" for violation in found]
    return "\n".join(lines) + "\n"


def _send_alert(subject: str, lines: Sequence[str]) -> None:
    notify.send_retranscribe_alert(subject, lines, config.load_notify_config())


def _absence_alert(rows: Sequence[Backlog], *, planned: int, now: datetime) -> tuple[str, list[str]] | None:
    """The dead-man or completion alert the scheduled pass owes, if any."""
    in_scope = sum(row.in_scope for row in rows)
    stale = sum(row.stale for row in rows)
    if in_scope == 0:
        if planned == 0:
            return None
        return (
            "Re-transcription pass complete",
            [f"{stale} stale sermon(s) left for review (failed under the running profile)"]
            + [f"{row.church}: {reason} {n}" for row in rows for reason, n in sorted(row.failed.items())],
        )
    stamps = [row.last_replaced_at for row in rows if row.last_replaced_at]
    last = max((datetime.fromisoformat(at) for at in stamps), default=None)
    if last is not None and now - last < _DEAD_MAN_WINDOW:
        return None
    return (
        "Re-transcription stalled: no transcript replaced in 24 hours",
        [
            f"{in_scope} stale sermon(s) still in scope",
            f"last replacement: {last.isoformat() if last else 'never'}",
        ],
    )


def main(
    argv: list[str] | None = None,
    *,
    fetch_all: Callable[[], dict[str, str]] = content_repo.all_transcripts,
    send_alert: SendAlert = _send_alert,
    now: Callable[[], datetime] = lambda: datetime.now(UTC),
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-md", help="Append the backlog and any violations to this path.")
    parser.add_argument(
        "--scheduled",
        action="store_true",
        help="The scheduled pass: also alert when it stalls, and once when it completes.",
    )
    parser.add_argument(
        "--planned", type=int, default=0, help="Sermons this run's plan put in scope (completion detection)."
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=config.load_log_level(), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    churches = list(config.load_churches())
    try:
        content = fetch_all()
    except content_repo.ContentPublishError as exc:
        logger.error("could not read Content to verify the ledger: %s", exc)
        return 1
    found = violations({church: store.load(church) for church in churches}, content)
    rows = backlog(churches)
    for violation in found:
        logger.error("ledger/Content violation: %s", violation)
    logger.info(
        "verified: %d violation(s); %d stale, %d in scope, %d replaced",
        len(found),
        sum(r.stale for r in rows),
        sum(r.in_scope for r in rows),
        sum(r.replaced for r in rows),
    )
    if args.summary_md:
        with open(args.summary_md, "a", encoding="utf-8") as f:
            f.write(render_summary(rows, found))

    alerts: list[tuple[str, list[str]]] = []
    if found:
        noun = "violation" if len(found) == 1 else "violations"
        alerts.append(
            (f"Re-transcription verify: {len(found)} {noun} of the ledger ↔ Content invariant", found)
        )
    if args.scheduled and (absence := _absence_alert(rows, planned=args.planned, now=now())):
        alerts.append(absence)
    ok = not found
    for subject, lines in alerts:
        try:
            send_alert(subject, lines)
        except (config.ConfigError, notify.NotifyError) as exc:
            logger.error("alert %r not sent: %s", subject, exc)
            ok = False
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
