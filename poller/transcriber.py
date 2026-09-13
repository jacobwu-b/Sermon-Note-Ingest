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


def run(
    *,
    church_names: list[str] | None,
    limit: int,
    transcribe_audio: TranscribeAudio = transcribe.transcribe_audio,
    push: PushTranscripts = content_repo.push_transcripts,
) -> bool:
    """Transcribe at most ``limit`` pending sermons across the selected churches.

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

    remaining = limit
    all_ok = True
    for name in selected:
        if remaining <= 0:
            break
        records = store.load(name)
        batch = _select_pending(records)[:remaining]
        if not batch:
            continue
        remaining -= len(batch)
        try:
            ok = transcribe_church(name, records, batch, transcribe_audio=transcribe_audio, push=push)
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
    parser.add_argument("--verbose", action="store_true", help="Debug-level logging.")
    args = parser.parse_args(argv)

    if args.limit <= 0:
        parser.error("--limit must be a positive integer")

    logging.basicConfig(
        level="DEBUG" if args.verbose else config.load_log_level(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ok = run(church_names=args.churches, limit=args.limit)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
