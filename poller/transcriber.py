"""The transcription entry point: transcribe pending sermons and push them to Content.

One run works down at most ``--limit`` pending sermons *per church* (``--church``,
default: every enabled church), in deterministic (newest-``published_on``-first)
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
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta

from poller import config, content_repo, net, pipeline_dispatch, prompting, store, transcribe
from poller.slug import slugify

TranscribeAudio = Callable[[str, str | None], tuple[str, str]]
PushTranscripts = Callable[[dict[str, str]], None]
DispatchIngestEvent = Callable[[dict[str, object], str], None]

logger = logging.getLogger("poller")

_DEFAULT_LIMIT = 5
_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
_DISPATCH_RECENCY = timedelta(days=7)


def _recently_published(record: dict, *, now: datetime) -> bool:
    """Whether ``record`` was published within :data:`_DISPATCH_RECENCY` of ``now``.

    Gates the pipeline dispatch (spec 0005) so a backfill run — transcribing an old
    sermon, potentially years after it aired — never triggers note generation; only
    genuine discovery of a new sermon does. Falls back to ``published_on`` (date-only)
    when ``published_at`` is absent, mirroring ``_ingest_event``'s own fallback.
    """
    raw = record.get("published_at") or record.get("published_on")
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
    stem = f"{record.get('published_on') or 'undated'}_{slugify(record.get('title') or '')}_{_safe_guid(record['guid'])}"
    return f"transcripts/{church}/{stem}.txt"


def _ingest_event(
    name: str, guid: str, record: dict, *, content_path: str, transcript_hash: str, transcribed_at: str
) -> dict[str, object]:
    """Build the ``sermon_detected`` payload Sermon-Note-Pipeline's ``ingest_event`` expects.

    Field shape and names come from Pipeline's own spec 0026 — ``source``/``external_id``
    are the only fields Pipeline's code reads; the rest is logged there for traceability.
    ``external_id`` is ``guid`` verbatim: this repo's per-church guid scheme (raw for
    Menlo/PBC, ``<source>:``-prefixed for the rest) already matches Pipeline's own.
    """
    return {
        "event": "sermon_detected",
        "source": name,
        "external_id": guid,
        "url": record.get("episode_url") or "",
        "published_at": record.get("published_at") or record.get("published_on") or "",
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


def _select_pending(records: dict[str, dict]) -> list[tuple[str, dict]]:
    """Records not yet transcribed or failed, with a fetchable enclosure, newest first.

    A record missing ``published_on`` sorts last regardless of direction — it carries
    the least scheduling information, so it's the lowest priority either way (mirrors
    ``store._ordered_items``).
    """
    pending = [
        (guid, record)
        for guid, record in records.items()
        if record.get("transcription_status") is None
        and net.is_fetchable_enclosure(record.get("audio_url") or "")
    ]
    dated = [(guid, record) for guid, record in pending if record.get("published_on")]
    undated = [(guid, record) for guid, record in pending if not record.get("published_on")]
    dated.sort(key=lambda kv: kv[0])
    dated.sort(key=lambda kv: kv[1]["published_on"], reverse=True)
    undated.sort(key=lambda kv: kv[0])
    return dated + undated


def transcribe_church(
    name: str,
    records: dict[str, dict],
    batch: list[tuple[str, dict]],
    *,
    transcribe_audio: TranscribeAudio = transcribe.transcribe_audio,
    push: PushTranscripts = content_repo.push_transcripts,
    dispatch_ingest_event: DispatchIngestEvent = _dispatch_ingest_event,
    vocabulary: Sequence[str] = (),
    marks_out: dict[str, dict] | None = None,
) -> bool:
    """Transcribe ``batch``, push the successes to Content in one commit, then ledger them.

    Returns ``True`` iff no sermon in ``batch`` ended terminally failed and the Content
    push (if there was anything to push) succeeded. A download failure is not a failure
    of this run — it's an expected, retried-next-run outcome. ``vocabulary`` is the
    church's configured term list (``CHURCHES[name].vocabulary``), prompted alongside
    what ``records`` already knows about the church.

    ``marks_out``, when given, is filled with every ledger mutation this call actually
    durably makes (guid -> ``{"kind": "done", ...mark_transcribed kwargs}`` or
    ``{"kind": "failed"}``) — never a transcription whose Content push didn't succeed.
    :func:`replay_marks` re-applies this record verbatim onto a freshly-loaded ledger, so
    transcribe.yml's push-conflict recovery never has to re-run Whisper or re-push Content
    just to recover from a losing ``data/*.json`` race (issue #60).
    """
    to_push: dict[str, str] = {}
    pending_marks: dict[str, tuple[str, str]] = {}
    all_ok = True
    church_terms = prompting.church_vocabulary(records)
    whisper_cfg = config.load_whisper_config()

    for guid, record in batch:
        hotwords = prompting.build_hotwords(record, church_terms=church_terms, vocabulary=vocabulary)
        logger.debug("%s/%s: hotwords: %s", name, guid, hotwords)
        try:
            text, digest = transcribe_audio(record["audio_url"], hotwords)
        except net.AudioDownloadError as exc:
            logger.warning("%s/%s: audio download failed, retrying next run: %s", name, guid, exc)
            continue
        except (transcribe.TranscriptionError, transcribe.EmptyTranscriptError) as exc:
            logger.warning("%s/%s: transcription failed terminally: %s", name, guid, exc)
            store.mark_transcription_failed(record)
            if marks_out is not None:
                marks_out[guid] = {"kind": "failed"}
            all_ok = False
            continue
        path = _content_path(name, record)
        to_push[path] = text
        pending_marks[guid] = (path, digest)

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
        for guid, (path, digest) in pending_marks.items():
            record = records[guid]
            store.mark_transcribed(
                record,
                content_path=path,
                transcript_hash=digest,
                transcribed_at=transcribed_at,
                model=whisper_cfg.model,
                domain_prompt=whisper_cfg.domain_prompt,
            )
            if marks_out is not None:
                marks_out[guid] = {
                    "kind": "done",
                    "content_path": path,
                    "transcript_hash": digest,
                    "transcribed_at": transcribed_at,
                    "model": whisper_cfg.model,
                    "domain_prompt": whisper_cfg.domain_prompt,
                }
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
    return all_ok


def _select_in_scope(
    selected: dict[str, config.ChurchConfig], *, limit: int
) -> tuple[dict[str, dict], list[tuple[str, str]]]:
    """Every (church, guid) in scope for this run, in processing order, plus each
    touched church's full record set (for :func:`transcribe_church` to save back).

    Each church in ``selected`` independently contributes up to ``limit`` of its own
    newest-``published_on``-first pending sermons — ``limit`` is a per-church cap, not
    a budget shared across churches (ADR-0005, amended). This is the list :func:`run`
    shards — sharding only filters it, never reorders it, so shard-count 1 reproduces
    this order byte-for-byte.
    """
    loaded: dict[str, dict] = {}
    ordered: list[tuple[str, str]] = []
    for name in selected:
        records = store.load(name)
        batch = _select_pending(records)[:limit]
        if not batch:
            continue
        loaded[name] = records
        ordered.extend((name, guid) for guid, _record in batch)
    return loaded, ordered


def replay_marks(path: str) -> bool:
    """Re-apply a JSON marks file (:func:`transcribe_church`'s ``marks_out``) onto a
    freshly-loaded ledger — no transcription, no Content push.

    Used by transcribe.yml's push-conflict recovery (issue #60): after a genuine
    ``git rebase`` conflict against ``origin/main``, ``git reset --hard`` discards this
    shard's local ledger commit, but the marks it already durably pushed to Content
    (or the model failures it already hit) must not be redone from scratch — that
    re-runs Whisper on the whole batch and can push a second, differently-worded
    version of a transcript Pipeline may have already consumed. Replaying is a pure,
    idempotent record-level merge: a guid missing from the freshly-loaded ledger (e.g.
    manually removed) is skipped with a warning rather than raised.
    """
    try:
        with open(path, encoding="utf-8") as f:
            marks: dict[str, dict[str, dict]] = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("replay-marks: could not read %s: %s", path, exc)
        return False

    for church, guid_marks in marks.items():
        records = store.load(church)
        for guid, fields in guid_marks.items():
            record = records.get(guid)
            if record is None:
                logger.warning("%s/%s: replay-marks: guid not in ledger, skipping", church, guid)
                continue
            kind = fields.get("kind")
            if kind == "done":
                store.mark_transcribed(
                    record,
                    content_path=fields["content_path"],
                    transcript_hash=fields["transcript_hash"],
                    transcribed_at=fields["transcribed_at"],
                    model=fields["model"],
                    domain_prompt=fields["domain_prompt"],
                )
            elif kind == "failed":
                store.mark_transcription_failed(record)
            else:
                logger.warning("%s/%s: replay-marks: unknown mark kind %r, skipping", church, guid, kind)
                continue
        store.save(church, records)
    return True


def _selected_churches(church_names: list[str] | None) -> dict[str, config.ChurchConfig]:
    """Enabled churches, narrowed to ``church_names`` if given (default: all enabled)."""
    churches = config.load_churches()
    return {
        name: entry
        for name, entry in churches.items()
        if entry.enabled and (church_names is None or name in church_names)
    }


def count_in_scope(*, church_names: list[str] | None, limit: int) -> int:
    """Number of sermons :func:`run` would process for the same ``church_names``/``limit``.

    Pure selection — no transcription, no writes. Used by ``transcribe.yml``'s plan job to
    size the shard matrix (ADR-0005) before any shard runs, so shard count always matches the
    actual backlog instead of being guessed at dispatch time.
    """
    selected = _selected_churches(church_names)
    if not selected:
        return 0
    _loaded, ordered = _select_in_scope(selected, limit=limit)
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
    marks_out: dict[str, dict[str, dict]] | None = None,
) -> bool:
    """Transcribe at most ``limit`` pending sermons per selected church.

    ``shard_index``/``shard_count`` (ADR-0005) narrow that same bounded selection to
    the ``shard_index``-th of ``shard_count`` disjoint pieces, by position in the
    scoped, ordered selection (``_select_in_scope``) — every in-scope sermon lands in
    exactly one shard, and the default (0, 1) is every sermon, unchanged from before
    sharding existed.

    ``transcribe_audio``/``push``/``dispatch_ingest_event`` are threaded through to
    :func:`transcribe_church` rather than relied on as its defaults, so a caller (or a
    test) can replace them without reaching into another module's attributes.

    ``marks_out``, when given, is filled per church with this run's
    :func:`transcribe_church` marks (see there) for :func:`replay_marks` to later
    re-apply without redoing any transcription or Content push.
    """
    selected = _selected_churches(church_names)
    if not selected:
        logger.warning("no enabled churches matched the selection; nothing to transcribe")
        return True

    loaded, ordered = _select_in_scope(selected, limit=limit)
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
                vocabulary=selected[name].vocabulary,
                marks_out=marks_out.setdefault(name, {}) if marks_out is not None else None,
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
        "--marks-out",
        help=(
            "Write a JSON record of every mark this run applied (guid -> done/failed "
            "fields) to this path. Used by transcribe.yml's push-conflict recovery."
        ),
    )
    parser.add_argument(
        "--replay-marks",
        help=(
            "Re-apply a --marks-out file onto a freshly-loaded ledger and exit — no "
            "transcription, no Content push. Used by transcribe.yml's push-conflict "
            "recovery after a git reset --hard discards this shard's local commit."
        ),
    )
    parser.add_argument("--verbose", action="store_true", help="Debug-level logging.")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level="DEBUG" if args.verbose else config.load_log_level(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if args.replay_marks:
        return 0 if replay_marks(args.replay_marks) else 1

    if args.limit <= 0:
        parser.error("--limit must be a positive integer")
    if args.print_shard_count:
        # Machine-readable stdout, not a log line: transcribe.yml's plan job captures
        # this via `$(...)` to size its shard matrix (§6 exempts a command's own
        # designed output contract from the "no print" rule, which targets ad-hoc
        # debug prints in place of logging).
        print(count_in_scope(church_names=args.churches, limit=args.limit))
        return 0
    if args.shard_count <= 0:
        parser.error("--shard-count must be a positive integer")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")

    marks: dict[str, dict[str, dict]] = {}
    ok = run(
        church_names=args.churches,
        limit=args.limit,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
        marks_out=marks if args.marks_out else None,
    )
    if args.marks_out:
        with open(args.marks_out, "w", encoding="utf-8") as f:
            json.dump(marks, f)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
