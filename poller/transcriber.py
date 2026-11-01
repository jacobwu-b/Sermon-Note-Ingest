"""The transcription entry point: transcribe pending sermons and push them to Content.

One run works down at most ``--limit`` pending sermons *per church* (``--church``,
default: every enabled church), in deterministic (newest-``preached_on``-first)
order — so the most recent sermon is always the one a small ``--limit`` (e.g. 1)
surfaces. This is also this repo's backfill tool: a large historical backlog is
worked down by repeated bounded runs, catching up from most-recent backward
rather than a special unbounded mode. A sermon becomes "pending" the moment it's
ledgered by ``poller.runner`` and stays pending until its ``transcription_status``
is ``"done"`` or ``"failed"``.

Durability ordering (ADR-0004): every sermon transcribed in a run is pushed to
Sermon-Note-Content in one batch, and only sermons whose push succeeds are marked
``"done"`` in the local ledger. A push failure leaves the whole run's transcriptions
pending rather than mark one done that never durably landed.

Sharding (ADR-0005): ``--shard-index``/``--shard-count`` split that same bounded
selection into disjoint pieces so a large backfill can be dispatched as several
parallel runs instead of one. Sharding never changes *which* sermons are in scope
(``--limit``/``--church`` still decide that) or their processing order — it only
decides which of them this particular invocation is responsible for. The default
(index 0, count 1) is every sermon in scope, identical to omitting both flags.

``--retranscribe-stale`` (ADR-0022) is a second mode over ``done`` records instead of
pending ones: it re-transcribes each one whose recorded decoding profile differs from the
running one, and replaces it in Content only by compare-and-swap. It never dispatches
Pipeline. See :func:`poller.retranscribe.run_retranscribe`.
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import re
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta

from poller import (
    config,
    content_repo,
    net,
    notify,
    pipeline_dispatch,
    prompting,
    quality,
    retranscribe,
    selection,
    store,
    transcribe,
)
from poller.slug import slugify

TranscribeAudio = Callable[[str, str | None], tuple[str, str, transcribe.DecodingProfile]]
PushTranscripts = Callable[[dict[str, str]], None]
FetchExisting = Callable[[list[str]], dict[str, str]]
DispatchIngestEvent = Callable[[dict[str, object], str], None]
SendQualityAlert = Callable[[str, list[notify.FlaggedTranscript]], None]

logger = logging.getLogger("poller")

_DEFAULT_LIMIT = 5
_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
_DISPATCH_RECENCY = timedelta(days=7)


def _recently_published(record: dict, *, now: datetime) -> bool:
    """Whether ``record`` was published within :data:`_DISPATCH_RECENCY` of ``now``.

    Gates the pipeline dispatch (spec 0005) so a backfill run — transcribing an old
    sermon, potentially years after it aired — never triggers note generation; only
    genuine discovery of a new sermon does. Falls back to ``preached_on`` (date-only)
    when ``feed_published_at`` is absent (PBC, archive backfills).
    """
    raw = record.get("feed_published_at") or record.get("preached_on")
    if not raw:
        return False
    try:
        when = datetime.fromisoformat(raw)
    except ValueError:
        return False
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return timedelta() <= now - when <= _DISPATCH_RECENCY


def _safe_guid(guid: str) -> str:
    """The stable part of a feed guid, sanitized for use in a filesystem path."""
    return _UNSAFE_FILENAME.sub("_", guid)


def _content_path(church: str, record: dict) -> str:
    """Map a record to its deterministic path inside Sermon-Note-Content.

    Guid-derived, so a sermon rediscovered many times always maps to the same
    file (the idempotency the Content push depends on, ADR-0004).
    """
    stem = f"{record.get('preached_on') or 'undated'}_{slugify(record.get('title') or '')}_{_safe_guid(record['guid'])}"
    return f"transcripts/{church}/{stem}.txt"


def _ingest_event(
    name: str, guid: str, record: dict, *, content_path: str, transcript_hash: str, transcribed_at: str
) -> dict[str, object]:
    """Build the ``sermon_detected`` payload Sermon-Note-Pipeline's ``ingest_event`` expects.

    Field shape follows Pipeline's own spec 0026, with the date fields per ADR-0015:
    ``preached_on`` is this ledger's service date, which Pipeline adopts as authoritative
    for the matching sermon; ``feed_published_at`` is ``None`` when the feed carries no
    real instant, never back-filled from the date.
    ``external_id`` is ``guid`` verbatim: this repo's per-church guid scheme (raw for
    Menlo/PBC, ``<source>:``-prefixed for the rest) already matches Pipeline's own.
    """
    return {
        "event": "sermon_detected",
        "source": name,
        "external_id": guid,
        "url": record.get("episode_url") or "",
        "preached_on": record.get("preached_on") or None,
        "feed_published_at": record.get("feed_published_at"),
        "detected_at": transcribed_at,
        "transcript": {
            "content_path": content_path,
            "transcript_hash": transcript_hash,
            "transcribed_at": transcribed_at,
        },
    }


def _dispatch_ingest_event(event: dict[str, object], source: str) -> None:
    pipeline_config = config.load_pipeline_config()
    pipeline_dispatch.dispatch_ingest_event(event, source=source, config=pipeline_config)


def _send_quality_alert(church: str, flagged: list[notify.FlaggedTranscript]) -> None:
    notify.send_quality_alert(church, flagged, config.load_notify_config())


def _select_pending(records: dict[str, dict]) -> list[tuple[str, dict]]:
    """Records not yet transcribed or failed, with a fetchable enclosure, newest first.

    A record missing ``preached_on`` sorts last regardless of direction — it carries
    the least scheduling information, so it's the lowest priority either way (mirrors
    ``store._ordered_items``).
    """
    return selection.newest_first(
        (guid, record)
        for guid, record in records.items()
        if record.get("transcription_status") is None
        and net.is_fetchable_enclosure(record.get("audio_url") or "")
    )


def transcribe_church(
    name: str,
    records: dict[str, dict],
    batch: list[tuple[str, dict]],
    *,
    transcribe_audio: TranscribeAudio = transcribe.transcribe_audio,
    push: PushTranscripts = content_repo.push_transcripts,
    dispatch_ingest_event: DispatchIngestEvent = _dispatch_ingest_event,
    fetch_existing: FetchExisting = content_repo.existing_transcripts,
    vocabulary: Sequence[str] = (),
    pipeline_enabled: bool = True,
    send_quality_alert: SendQualityAlert = _send_quality_alert,
) -> bool:
    """Transcribe ``batch``, push the successes to Content in one commit, then ledger them.

    A sermon whose transcript Content already holds is ledgered from Content's copy
    instead — no Whisper, no push, no pipeline dispatch (the run that pushed it already
    dispatched). The ledger saying "pending" while Content holds the file is reachable
    from more than one cause (a queued run reading a stale ledger, a ledger push that
    failed after its Content push landed); re-transcribing would only be refused by
    :func:`content_repo.default_push`'s overwrite guard, failing every run (issue #84).
    If Content can't be read at all, the whole batch stays pending.

    Each transcript is scored (ADR-0023) before the push. A flagged one is still pushed,
    ledgered and dispatched; once the push lands, the church's flagged sermons go out in
    one ``send_quality_alert`` email.

    Returns ``True`` iff no sermon in ``batch`` ended terminally failed, the Content
    push (if there was anything to push) succeeded, and any quality alert was delivered.
    A download failure is not a failure of this run — it's an expected, retried-next-run
    outcome. ``vocabulary`` is the
    church's configured term list (its ``ingest.vocabulary`` in the church table), prompted alongside
    what ``records`` already knows about the church. ``pipeline_enabled`` is False for a
    church Pipeline does not run: it is transcribed, but Pipeline is not dispatched, since
    its plan has no ledger record to fan out and would report a broken handoff.
    """
    to_push: dict[str, str] = {}
    pending_marks: dict[str, tuple[str, str, dict[str, object], quality.QualityScore]] = {}
    flagged: list[notify.FlaggedTranscript] = []
    all_ok = True
    church_terms = prompting.church_vocabulary(records)

    paths = {guid: _content_path(name, record) for guid, record in batch}
    try:
        existing = fetch_existing(list(paths.values()))
    except content_repo.ContentPublishError as exc:
        logger.error("%s: could not read Content, leaving %d sermon(s) pending: %s", name, len(batch), exc)
        return False

    for guid, record in batch:
        path = paths[guid]
        if path in existing:
            logger.warning(
                "%s/%s: transcript already in Content but ledgered pending; recording Content's copy",
                name,
                guid,
            )
            # When Content's copy was produced isn't known; this records when it was adopted.
            adopted_hash = transcribe.transcript_hash(existing[path])
            adopted_at = net.now()
            store.mark_transcribed(
                record,
                content_path=path,
                transcript_hash=adopted_hash,
                transcribed_at=adopted_at,
                profile=None,
                adopted=True,
            )
            continue
        hotwords = prompting.build_hotwords(record, church_terms=church_terms, vocabulary=vocabulary)
        logger.debug("%s/%s: hotwords: %s", name, guid, hotwords)
        try:
            text, digest, decoded_under = transcribe_audio(record["audio_url"], hotwords)
        except net.AudioDownloadError as exc:
            logger.warning("%s/%s: audio download failed, retrying next run: %s", name, guid, exc)
            store.mark_download_failed(record)
            continue
        except (transcribe.TranscriptionError, transcribe.EmptyTranscriptError) as exc:
            logger.warning("%s/%s: transcription failed terminally: %s", name, guid, exc)
            error = (
                store.EMPTY_TRANSCRIPT_ERROR
                if isinstance(exc, transcribe.EmptyTranscriptError)
                else store.TRANSCRIPTION_ERROR
            )
            store.mark_transcription_failed(record, error=error)
            all_ok = False
            continue
        to_push[path] = text
        score = quality.score(text, hotwords=decoded_under.hotwords)
        pending_marks[guid] = (path, digest, dataclasses.asdict(decoded_under), score)

    if to_push:
        try:
            push(to_push)
        except content_repo.ContentPublishError as exc:
            logger.error(
                "%s: content push failed for %d sermon(s), leaving them pending: %s",
                name,
                len(to_push),
                exc,
            )
            store.save(name, records)
            return False
        transcribed_at = net.now()
        now_dt = datetime.fromisoformat(transcribed_at)
        for guid, (path, digest, profile, score) in pending_marks.items():
            record = records[guid]
            counts = dataclasses.asdict(score)
            store.mark_transcribed(
                record,
                content_path=path,
                transcript_hash=digest,
                transcribed_at=transcribed_at,
                profile=profile,
                quality=counts,
            )
            tripped = quality.flags(score)
            if tripped:
                logger.warning(
                    "%s/%s: transcript quality flagged %s: %s", name, guid, ",".join(tripped), counts
                )
                flagged.append(
                    notify.FlaggedTranscript(
                        guid=guid,
                        title=record.get("title"),
                        preached_on=record.get("preached_on"),
                        score=score,
                        flags=tripped,
                    )
                )
            if not pipeline_enabled:
                logger.debug("%s/%s: church not enabled in pipeline, skipping pipeline dispatch", name, guid)
                continue
            if not _recently_published(record, now=now_dt):
                logger.debug(
                    "%s/%s: published more than %d days ago, skipping pipeline dispatch (backfill)",
                    name,
                    guid,
                    _DISPATCH_RECENCY.days,
                )
                continue
            event = _ingest_event(
                name, guid, record, content_path=path, transcript_hash=digest, transcribed_at=transcribed_at
            )
            try:
                dispatch_ingest_event(event, name)
            except (config.ConfigError, pipeline_dispatch.PipelineDispatchError) as exc:
                logger.warning(
                    "%s/%s: pipeline dispatch failed, pipeline's own cron will still catch it: %s",
                    name,
                    guid,
                    exc,
                )

    store.save(name, records)
    logger.info("%s: transcribed %d/%d selected", name, len(pending_marks), len(batch))
    if flagged:
        try:
            send_quality_alert(name, flagged)
        except (config.ConfigError, notify.NotifyError) as exc:
            logger.error(
                "%s: quality alert for %d flagged transcript(s) not sent: %s", name, len(flagged), exc
            )
            all_ok = False
    return all_ok


def count_in_scope(
    *,
    church_names: list[str] | None,
    limit: int,
    retranscribe_stale: bool = False,
    guids: Sequence[str] | None = None,
) -> int:
    """Number of sermons :func:`run` (or, with ``retranscribe_stale``,
    :func:`poller.retranscribe.run_retranscribe`) would process for the same arguments.

    Pure selection — no transcription, no writes. Used by ``transcribe.yml``'s plan job to
    size the shard matrix (ADR-0005) before any shard runs, so shard count always matches the
    actual backlog instead of being guessed at dispatch time.
    """
    selected = selection.selected_churches(church_names)
    if not selected:
        return 0
    select = retranscribe.stale_selector(guids) if retranscribe_stale else _select_pending
    _loaded, ordered = selection.select_in_scope(selected, limit=limit, select=select)
    return len(ordered)


def run(
    *,
    church_names: list[str] | None,
    limit: int,
    shard_index: int = 0,
    shard_count: int = 1,
    transcribe_audio: TranscribeAudio = transcribe.transcribe_audio,
    push: PushTranscripts = content_repo.push_transcripts,
    dispatch_ingest_event: DispatchIngestEvent = _dispatch_ingest_event,
    fetch_existing: FetchExisting = content_repo.existing_transcripts,
    send_quality_alert: SendQualityAlert = _send_quality_alert,
) -> bool:
    """Transcribe at most ``limit`` pending sermons per selected church.

    ``shard_index``/``shard_count`` (ADR-0005) narrow that same bounded selection to
    the ``shard_index``-th of ``shard_count`` disjoint pieces, by position in the
    scoped, ordered selection (:func:`poller.selection.select_in_scope`) — every
    in-scope sermon lands in exactly one shard, and the default (0, 1) is every sermon,
    unchanged from before sharding existed.

    ``transcribe_audio``/``push``/``dispatch_ingest_event``/``fetch_existing``/
    ``send_quality_alert`` are threaded through to :func:`transcribe_church` rather than
    relied on as its defaults, so a caller (or a test) can replace them without reaching
    into another module's attributes.
    """
    selected = selection.selected_churches(church_names)
    if not selected:
        logger.warning("no enabled churches matched the selection; nothing to transcribe")
        return True

    loaded, ordered = selection.select_in_scope(selected, limit=limit, select=_select_pending)
    shard = [pair for i, pair in enumerate(ordered) if i % shard_count == shard_index]

    batches: dict[str, list[tuple[str, dict]]] = {}
    for name, guid in shard:
        batches.setdefault(name, []).append((guid, loaded[name][guid]))

    all_ok = True
    for name, batch in batches.items():
        try:
            ok = transcribe_church(
                name,
                loaded[name],
                batch,
                transcribe_audio=transcribe_audio,
                push=push,
                dispatch_ingest_event=dispatch_ingest_event,
                fetch_existing=fetch_existing,
                vocabulary=selected[name].vocabulary,
                pipeline_enabled=selected[name].pipeline_enabled,
                send_quality_alert=send_quality_alert,
            )
        except Exception:
            logger.exception("%s: transcription crashed unexpectedly", name)
            ok = False
        all_ok = all_ok and ok

    return all_ok


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--church",
        action="append",
        dest="churches",
        help="Only transcribe this church (repeatable). Default: every enabled church.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=_DEFAULT_LIMIT,
        help=f"Maximum sermons to transcribe this run, per selected church (default: {_DEFAULT_LIMIT}).",
    )
    parser.add_argument(
        "--print-shard-count",
        action="store_true",
        help=(
            "Print the number of sermons in scope for --limit/--church and exit "
            "(no transcription, no writes). Used by transcribe.yml to size its shard matrix."
        ),
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=0,
        help="This run's shard, 0-based (default: 0). Used with --shard-count for parallel backfills.",
    )
    parser.add_argument(
        "--shard-count",
        type=int,
        default=1,
        help="Number of disjoint shards to split the selection into (default: 1, i.e. no sharding).",
    )
    parser.add_argument(
        "--retranscribe-stale",
        action="store_true",
        help=(
            "Re-transcribe done sermons whose recorded decoding profile differs from the running "
            "one, replacing them in Content by compare-and-swap (ADR-0022), instead of "
            "transcribing pending sermons."
        ),
    )
    parser.add_argument(
        "--guid",
        action="append",
        dest="guids",
        help="With --retranscribe-stale: only this sermon's guid (repeatable).",
    )
    parser.add_argument(
        "--summary-md",
        help="With --retranscribe-stale: append a markdown table of each sermon's outcome to this path.",
    )
    parser.add_argument("--verbose", action="store_true", help="Debug-level logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level="DEBUG" if args.verbose else config.load_log_level(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.limit <= 0:
        parser.error("--limit must be a positive integer")
    if (args.guids or args.summary_md) and not args.retranscribe_stale:
        parser.error("--guid and --summary-md need --retranscribe-stale")
    if args.print_shard_count:
        # Machine-readable stdout, not a log line: transcribe.yml's plan job captures
        # this via `$(...)` to size its shard matrix (§6 exempts a command's own
        # designed output contract from the "no print" rule, which targets ad-hoc
        # debug prints in place of logging).
        print(
            count_in_scope(
                church_names=args.churches,
                limit=args.limit,
                retranscribe_stale=args.retranscribe_stale,
                guids=args.guids,
            )
        )
        return 0
    if args.shard_count <= 0:
        parser.error("--shard-count must be a positive integer")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")

    if args.retranscribe_stale:
        rows: list[retranscribe.SummaryRow] = []
        ok = retranscribe.run_retranscribe(
            church_names=args.churches,
            limit=args.limit,
            guids=args.guids,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            summary_rows=rows,
        )
        for row in rows:
            if row.outcome in retranscribe.WARNING_OUTCOMES:
                # A workflow annotation, not a log line: GitHub reads it only at the start of stdout.
                print(f"::warning::{row.church}/{row.guid}: re-transcription {row.outcome}")
        if args.summary_md and rows:
            with open(args.summary_md, "a", encoding="utf-8") as f:
                f.write(retranscribe.render_summary(rows))
    else:
        ok = run(
            church_names=args.churches,
            limit=args.limit,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
        )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
