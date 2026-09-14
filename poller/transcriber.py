"""The transcription entry point: transcribe pending sermons and push them to Content.

One run works down at most ``--limit`` pending sermons, in deterministic
(oldest-``published_on``-first) order, across the selected churches (``--church``,
default: every enabled church) — this is also this repo's backfill tool: a large
historical backlog is worked down by repeated bounded runs, not a special
unbounded mode. A sermon becomes "pending" the moment it's ledgered by
``poller.runner`` and stays pending until its ``transcription_status`` is
``"done"`` or ``"failed"``.

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
import logging
import re
import sys
from collections.abc import Callable

from poller import config, content_repo, net, store, transcribe
from poller.slug import slugify

TranscribeAudio = Callable[[str], tuple[str, str]]
PushTranscripts = Callable[[dict[str, str]], None]

logger = logging.getLogger("poller")

_DEFAULT_LIMIT = 5
_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")


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


def _select_pending(records: dict[str, dict]) -> list[tuple[str, dict]]:
    """Records not yet transcribed or failed, with a fetchable enclosure, oldest first."""
    pending = [
        (guid, record)
        for guid, record in records.items()
        if record.get("transcription_status") is None
        and net.is_fetchable_enclosure(record.get("audio_url") or "")
    ]
    pending.sort(key=lambda kv: kv[1].get("published_on") or "9999-99-99")
    return pending


def transcribe_church(
    name: str,
    records: dict[str, dict],
    batch: list[tuple[str, dict]],
    *,
    transcribe_audio: TranscribeAudio = transcribe.transcribe_audio,
    push: PushTranscripts = content_repo.push_transcripts,
) -> bool:
    """Transcribe ``batch``, push the successes to Content in one commit, then ledger them.

    Returns ``True`` iff no sermon in ``batch`` ended terminally failed and the Content
    push (if there was anything to push) succeeded. A download failure is not a failure
    of this run — it's an expected, retried-next-run outcome.
    """
    to_push: dict[str, str] = {}
    pending_marks: dict[str, tuple[str, str]] = {}
    all_ok = True

    for guid, record in batch:
        try:
            text, digest = transcribe_audio(record["audio_url"])
        except net.AudioDownloadError as exc:
            logger.warning("%s/%s: audio download failed, retrying next run: %s", name, guid, exc)
            continue
        except (transcribe.TranscriptionError, transcribe.EmptyTranscriptError) as exc:
            logger.warning("%s/%s: transcription failed terminally: %s", name, guid, exc)
            store.mark_transcription_failed(record)
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
        for guid, (path, digest) in pending_marks.items():
            store.mark_transcribed(
                records[guid], content_path=path, transcript_hash=digest, transcribed_at=transcribed_at
            )

    store.save(name, records)
    logger.info("%s: transcribed %d/%d selected", name, len(pending_marks), len(batch))
    return all_ok


def _select_in_scope(
    selected: dict[str, config.ChurchConfig], *, limit: int
) -> tuple[dict[str, dict], list[tuple[str, str]]]:
    """Every (church, guid) in scope for this run, in processing order, plus each
    touched church's full record set (for :func:`transcribe_church` to save back).

    Order matches today's unsharded ``run()`` exactly: churches in ``selected``'s
    order, each contributing its own oldest-``published_on``-first pending sermons
    up to whatever of ``limit`` remains. This is the list :func:`run` shards —
    sharding only filters it, never reorders it, so shard-count 1 reproduces this
    order byte-for-byte.
    """
    loaded: dict[str, dict] = {}
    ordered: list[tuple[str, str]] = []
    remaining = limit
    for name in selected:
        if remaining <= 0:
            break
        records = store.load(name)
        batch = _select_pending(records)[:remaining]
        if not batch:
            continue
        loaded[name] = records
        remaining -= len(batch)
        ordered.extend((name, guid) for guid, _record in batch)
    return loaded, ordered


def run(
    *,
    church_names: list[str] | None,
    limit: int,
    shard_index: int = 0,
    shard_count: int = 1,
    transcribe_audio: TranscribeAudio = transcribe.transcribe_audio,
    push: PushTranscripts = content_repo.push_transcripts,
) -> bool:
    """Transcribe at most ``limit`` pending sermons across the selected churches.

    ``shard_index``/``shard_count`` (ADR-0005) narrow that same bounded selection to
    the ``shard_index``-th of ``shard_count`` disjoint pieces, by position in the
    scoped, ordered selection (``_select_in_scope``) — every in-scope sermon lands in
    exactly one shard, and the default (0, 1) is every sermon, unchanged from before
    sharding existed.

    ``transcribe_audio``/``push`` are threaded through to :func:`transcribe_church`
    rather than relied on as its defaults, so a caller (or a test) can replace them
    without reaching into another module's attributes.
    """
    churches = config.load_churches()
    selected = {
        name: entry
        for name, entry in churches.items()
        if entry.enabled and (church_names is None or name in church_names)
    }
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
            ok = transcribe_church(name, loaded[name], batch, transcribe_audio=transcribe_audio, push=push)
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
        help=f"Maximum sermons to transcribe this run, across all selected churches (default: {_DEFAULT_LIMIT}).",
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
    parser.add_argument("--verbose", action="store_true", help="Debug-level logging.")
    args = parser.parse_args(argv)

    if args.limit <= 0:
        parser.error("--limit must be a positive integer")
    if args.shard_count <= 0:
        parser.error("--shard-count must be a positive integer")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must be in [0, --shard-count)")

    logging.basicConfig(
        level="DEBUG" if args.verbose else config.load_log_level(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ok = run(
        church_names=args.churches,
        limit=args.limit,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
