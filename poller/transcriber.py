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
Pipeline. See :func:`run_retranscribe`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import re
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import PurePosixPath
from typing import Any

from poller import config, content_repo, net, notify, pipeline_dispatch, prompting, quality, store, transcribe
from poller.slug import slugify

TranscribeAudio = Callable[[str, str | None], tuple[str, str, transcribe.DecodingProfile]]
PushTranscripts = Callable[[dict[str, str]], None]
FetchExisting = Callable[[list[str]], dict[str, str]]
DispatchIngestEvent = Callable[[dict[str, object], str], None]
SendQualityAlert = Callable[[str, list[notify.FlaggedTranscript]], None]
LastCommitTrailers = Callable[[str], dict[str, str]]
ReplaceTranscript = Callable[..., None]
KeepCandidate = Callable[..., None]

logger = logging.getLogger("poller")

_DEFAULT_LIMIT = 5
_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")
_DISPATCH_RECENCY = timedelta(days=7)
# ADR-0022 rule 6: a replacement carrying under 90% of the old text's clean words lost speech.
_MIN_CLEAN_WORDS_RATIO = 0.9
_CANDIDATES_DIR = "retranscription-candidates"
# Failures that re-running the same profile on the same audio would only repeat.
_TERMINAL_RETRANSCRIBE_ERRORS = frozenset(
    {
        store.TRANSCRIPTION_ERROR,
        store.EMPTY_TRANSCRIPT_ERROR,
        store.QUALITY_REGRESSION_ERROR,
        store.CONTENT_CONFLICT_ERROR,
    }
)


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
    return _newest_first(
        (guid, record)
        for guid, record in records.items()
        if record.get("transcription_status") is None
        and net.is_fetchable_enclosure(record.get("audio_url") or "")
    )


def _newest_first(pairs: Iterable[tuple[str, dict]]) -> list[tuple[str, dict]]:
    """Newest ``preached_on`` first, ties by guid; undated records last, by guid."""
    pairs = list(pairs)
    dated = [(guid, record) for guid, record in pairs if record.get("preached_on")]
    undated = [(guid, record) for guid, record in pairs if not record.get("preached_on")]
    dated.sort(key=lambda kv: kv[0])
    dated.sort(key=lambda kv: kv[1]["preached_on"], reverse=True)
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
    fetch_existing: FetchExisting = content_repo.existing_transcripts,
    vocabulary: Sequence[str] = (),
    pipeline_enabled: bool = True,
    marks_out: dict[str, dict] | None = None,
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

    ``marks_out``, when given, is filled with every ledger mutation this call actually
    durably makes (guid -> ``{"kind": "done", ...mark_transcribed kwargs}``, its
    ``profile`` and ``quality`` as plain dicts, or
    ``{"kind": "failed", "error": ...}``, or ``{"kind": "download_failed"}``) — never a
    transcription whose Content push didn't succeed.
    :func:`replay_marks` re-applies this record verbatim onto a freshly-loaded ledger, so
    transcribe.yml's push-conflict recovery never has to re-run Whisper or re-push Content
    just to recover from a losing ``data/*.json`` race (issue #60).
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
            if marks_out is not None:
                marks_out[guid] = {
                    "kind": "done",
                    "content_path": path,
                    "transcript_hash": adopted_hash,
                    "transcribed_at": adopted_at,
                    "profile": None,
                    "adopted": True,
                    "quality": None,
                }
            continue
        hotwords = prompting.build_hotwords(record, church_terms=church_terms, vocabulary=vocabulary)
        logger.debug("%s/%s: hotwords: %s", name, guid, hotwords)
        try:
            text, digest, decoded_under = transcribe_audio(record["audio_url"], hotwords)
        except net.AudioDownloadError as exc:
            logger.warning("%s/%s: audio download failed, retrying next run: %s", name, guid, exc)
            store.mark_download_failed(record)
            if marks_out is not None:
                marks_out[guid] = {"kind": "download_failed"}
            continue
        except (transcribe.TranscriptionError, transcribe.EmptyTranscriptError) as exc:
            logger.warning("%s/%s: transcription failed terminally: %s", name, guid, exc)
            error = (
                store.EMPTY_TRANSCRIPT_ERROR
                if isinstance(exc, transcribe.EmptyTranscriptError)
                else store.TRANSCRIPTION_ERROR
            )
            store.mark_transcription_failed(record, error=error)
            if marks_out is not None:
                marks_out[guid] = {"kind": "failed", "error": error}
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
            if marks_out is not None:
                marks_out[guid] = {
                    "kind": "done",
                    "content_path": path,
                    "transcript_hash": digest,
                    "transcribed_at": transcribed_at,
                    "profile": profile,
                    "adopted": False,
                    "quality": counts,
                }
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


def _select_in_scope(
    selected: dict[str, config.ChurchConfig],
    *,
    limit: int,
    select: Callable[[dict[str, dict]], list[tuple[str, dict]]] = _select_pending,
) -> tuple[dict[str, dict], list[tuple[str, str]]]:
    """Every (church, guid) in scope for this run, in processing order, plus each
    touched church's full record set (for :func:`transcribe_church` to save back).

    Each church in ``selected`` independently contributes up to ``limit`` of its own
    newest-``preached_on``-first pending sermons — ``limit`` is a per-church cap, not
    a budget shared across churches (ADR-0005, amended). This is the list :func:`run`
    shards — sharding only filters it, never reorders it, so shard-count 1 reproduces
    this order byte-for-byte. ``select`` is the mode's selection (pending by default).
    """
    loaded: dict[str, dict] = {}
    ordered: list[tuple[str, str]] = []
    for name in selected:
        records = store.load(name)
        batch = select(records)[:limit]
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
                # A marks file from before ADR-0022 carries only the ADR-0012 pair; the rest is unknown.
                profile = (
                    fields["profile"]
                    if "profile" in fields
                    else {"model": fields["model"], "domain_prompt": fields["domain_prompt"]}
                )
                store.mark_transcribed(
                    record,
                    content_path=fields["content_path"],
                    transcript_hash=fields["transcript_hash"],
                    transcribed_at=fields["transcribed_at"],
                    profile=profile,
                    adopted=fields.get("adopted", False),
                    # A marks file from before ADR-0023 carries no counts: not scored.
                    quality=fields.get("quality"),
                )
            elif kind == "failed":
                # A marks file from before ADR-0018 carries no reason; the model is all it covered.
                store.mark_transcription_failed(record, error=fields.get("error", store.TRANSCRIPTION_ERROR))
            elif kind == "download_failed":
                store.mark_download_failed(record)
            elif kind in ("retranscribed", "retranscribe_failed"):
                if not _apply_retranscribe_mark(record, fields):
                    logger.warning(
                        "%s/%s: replay-marks: transcript hash moved since the attempt, skipping", church, guid
                    )
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


def count_in_scope(
    *,
    church_names: list[str] | None,
    limit: int,
    retranscribe_stale: bool = False,
    guids: Sequence[str] | None = None,
) -> int:
    """Number of sermons :func:`run` (or, with ``retranscribe_stale``, :func:`run_retranscribe`)
    would process for the same arguments.

    Pure selection — no transcription, no writes. Used by ``transcribe.yml``'s plan job to
    size the shard matrix (ADR-0005) before any shard runs, so shard count always matches the
    actual backlog instead of being guessed at dispatch time.
    """
    selected = _selected_churches(church_names)
    if not selected:
        return 0
    select = _stale_selector(guids) if retranscribe_stale else _select_pending
    _loaded, ordered = _select_in_scope(selected, limit=limit, select=select)
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
    marks_out: dict[str, dict[str, dict]] | None = None,
    send_quality_alert: SendQualityAlert = _send_quality_alert,
) -> bool:
    """Transcribe at most ``limit`` pending sermons per selected church.

    ``shard_index``/``shard_count`` (ADR-0005) narrow that same bounded selection to
    the ``shard_index``-th of ``shard_count`` disjoint pieces, by position in the
    scoped, ordered selection (``_select_in_scope``) — every in-scope sermon lands in
    exactly one shard, and the default (0, 1) is every sermon, unchanged from before
    sharding existed.

    ``transcribe_audio``/``push``/``dispatch_ingest_event``/``fetch_existing``/
    ``send_quality_alert`` are threaded through to :func:`transcribe_church` rather than
    relied on as its defaults, so a caller (or a test) can replace them without reaching
    into another module's attributes.

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
                fetch_existing=fetch_existing,
                vocabulary=selected[name].vocabulary,
                pipeline_enabled=selected[name].pipeline_enabled,
                marks_out=marks_out.setdefault(name, {}) if marks_out is not None else None,
                send_quality_alert=send_quality_alert,
            )
        except Exception:
            logger.exception("%s: transcription crashed unexpectedly", name)
            ok = False
        all_ok = all_ok and ok

    return all_ok


# --- Re-transcription of stale-profile transcripts (ADR-0022) ------------------------------


def _running_profile() -> dict[str, Any]:
    """The staleness keys of the profile a model call would run under right now."""
    profile = transcribe.decoding_profile(config.load_whisper_config(), None)
    return {key: getattr(profile, key) for key in store.STALENESS_KEYS}


def select_stale(
    records: dict[str, dict], *, running: Mapping[str, Any], digest: str, guids: Sequence[str] | None = None
) -> list[tuple[str, dict]]:
    """``done`` records whose profile is stale against ``running``, in processing order.

    Skips a record without a Content file, a hash or a fetchable enclosure, and one that
    already failed terminally under ``digest`` (the running profile). Records never
    attempted under ``digest`` come first, so a dead enclosure retrying every run can't
    hold the head of the queue; each group is newest ``preached_on`` first. ``guids``,
    when given, narrows the set to those records.
    """
    eligible = [
        (guid, record)
        for guid, record in records.items()
        if record.get("transcription_status") == "done"
        and record.get("content_path")
        and record.get("transcript_hash")
        and net.is_fetchable_enclosure(record.get("audio_url") or "")
        and (guids is None or guid in guids)
        and store.profile_is_stale(record, running)
        and not (
            record.get("retranscription_failed_profile") == digest
            and record.get("retranscription_last_error") in _TERMINAL_RETRANSCRIBE_ERRORS
        )
    ]
    fresh = [kv for kv in eligible if kv[1].get("retranscription_failed_profile") != digest]
    retries = [kv for kv in eligible if kv[1].get("retranscription_failed_profile") == digest]
    return _newest_first(fresh) + _newest_first(retries)


def _stale_selector(guids: Sequence[str] | None) -> Callable[[dict[str, dict]], list[tuple[str, dict]]]:
    running = _running_profile()
    digest = store.profile_digest(running)
    return lambda records: select_stale(records, running=running, digest=digest, guids=guids)


def _trailer_name(prefix: str, key: str) -> str:
    return prefix + "-".join(part.capitalize() for part in key.split("_"))


_PROFILE_TRAILERS = {key: _trailer_name("Ingest-Profile-", key) for key in store.PROFILE_KEYS}
_QUALITY_TRAILERS = {key: _trailer_name("Ingest-Quality-", key) for key in store.QUALITY_KEYS}


def _provenance_trailers(
    guid: str,
    *,
    superseded: str,
    transcript_hash: str,
    transcribed_at: str,
    profile: Mapping[str, Any],
    counts: Mapping[str, int],
) -> dict[str, str]:
    """The ``Ingest-*`` trailers of a replacement commit (ADR-0022 rule 5); values are JSON."""
    return {
        "Ingest-Guid": guid,
        "Ingest-Supersedes": superseded,
        "Ingest-Transcript-Hash": transcript_hash,
        "Ingest-Transcribed-At": transcribed_at,
        **{name: json.dumps(profile[key]) for key, name in _PROFILE_TRAILERS.items()},
        **{name: json.dumps(counts[key]) for key, name in _QUALITY_TRAILERS.items()},
    }


def _mark_from_trailers(
    trailers: Mapping[str, str], *, guid: str, ledger_hash: str, content_hash: str
) -> dict[str, Any] | None:
    """The ``retranscribed`` mark a previous run's Content commit proves, or ``None``.

    Proof is a commit naming this guid, superseding the ledger's hash and producing the
    text Content holds now, with every profile and quality trailer present.
    """
    if (
        trailers.get("Ingest-Guid") != guid
        or trailers.get("Ingest-Supersedes") != ledger_hash
        or trailers.get("Ingest-Transcript-Hash") != content_hash
        or "Ingest-Transcribed-At" not in trailers
    ):
        return None
    try:
        profile = {key: json.loads(trailers[name]) for key, name in _PROFILE_TRAILERS.items()}
        counts = {key: json.loads(trailers[name]) for key, name in _QUALITY_TRAILERS.items()}
    except (KeyError, json.JSONDecodeError):
        return None
    return {
        "kind": "retranscribed",
        "expected_hash": ledger_hash,
        "transcript_hash": content_hash,
        "transcribed_at": trailers["Ingest-Transcribed-At"],
        "profile": profile,
        "quality": counts,
        "superseded": ledger_hash,
    }


def _apply_retranscribe_mark(record: dict, mark: Mapping[str, Any]) -> bool:
    """Apply a ``retranscribed``/``retranscribe_failed`` mark if the record still matches it.

    The ledger-side compare-and-swap: a replacement applies while the record holds the
    hash the attempt saw or the hash it produced (so replaying it twice is a no-op); a
    failure applies only while the record holds the hash the attempt saw.
    """
    current = record.get("transcript_hash")
    if mark["kind"] == "retranscribed":
        if current not in (mark["expected_hash"], mark["transcript_hash"]):
            return False
        store.mark_retranscribed(
            record,
            transcript_hash=mark["transcript_hash"],
            transcribed_at=mark["transcribed_at"],
            profile=mark["profile"],
            quality=mark["quality"],
            superseded=mark["superseded"],
        )
        return True
    if current != mark["expected_hash"]:
        return False
    store.mark_retranscribe_failed(
        record,
        error=mark["error"],
        failed_profile=mark["failed_profile"],
        candidate_path=mark.get("candidate_path"),
        candidate_hash=mark.get("candidate_hash"),
    )
    return True


@dataclass
class SummaryRow:
    """One sermon's outcome: identifiers, hashes and counts, never text."""

    church: str
    guid: str
    outcome: str
    seconds: float | None = None
    old_hash: str | None = None
    new_hash: str | None = None
    old_score: quality.QualityScore | None = None
    new_score: quality.QualityScore | None = None


# Outcomes that need nobody: the record is done, or will be retried, or its refused
# candidate waits in Content for review.
_SETTLED_OUTCOMES = frozenset(
    {"replaced", "identical", "adopted", store.DOWNLOAD_ERROR, store.QUALITY_REGRESSION_ERROR}
)
_WARNING_OUTCOMES = frozenset({store.DOWNLOAD_ERROR, store.QUALITY_REGRESSION_ERROR})


def _change(old: object, new: object) -> str:
    if old is None and new is None:
        return ""
    return f"{'' if old is None else old}→{'' if new is None else new}"


def render_summary(rows: Sequence[SummaryRow]) -> str:
    """A markdown table of ``rows`` for the step summary."""
    lines = [
        "| Church | Guid | Outcome | Seconds | Old hash | New hash | Words | Loop words | Credit hits | Hotword pairs |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        old, new = row.old_score, row.new_score
        counts = [_change(getattr(old, key, None), getattr(new, key, None)) for key in store.QUALITY_KEYS]
        lines.append(
            f"| {row.church} | {row.guid} | {row.outcome} | "
            f"{'' if row.seconds is None else f'{row.seconds:.0f}'} | "
            f"{(row.old_hash or '')[:12]} | {(row.new_hash or '')[:12]} | " + " | ".join(counts) + " |"
        )
    return "\n".join(lines) + "\n"


@dataclass
class _Attempt:
    mark: dict[str, Any] | None
    row: SummaryRow
    model_called: bool = False


def _retranscribe_sermon(
    name: str,
    guid: str,
    record: dict,
    *,
    hotwords: str | None,
    digest: str,
    transcribe_audio: TranscribeAudio,
    fetch_existing: FetchExisting,
    last_commit_trailers: LastCommitTrailers,
    replace: ReplaceTranscript,
    keep_candidate: KeepCandidate,
) -> _Attempt:
    """One sermon through the ADR-0022 flow; returns the mark to apply, if any.

    ``mark`` is ``None`` only when Content couldn't be read or a push was exhausted: the
    ledger is left exactly as it was, and the next run selects the record again.
    """
    path = record["content_path"]
    old_hash = record["transcript_hash"]
    row = SummaryRow(church=name, guid=guid, outcome="", old_hash=old_hash)

    def fail(error: str, **candidate: str) -> dict[str, Any]:
        row.outcome = error
        return {
            "kind": "retranscribe_failed",
            "expected_hash": old_hash,
            "error": error,
            "failed_profile": digest,
            **candidate,
        }

    trailers: dict[str, str] = {}
    try:
        current = fetch_existing([path]).get(path)
        if current is not None and transcribe.transcript_hash(current) != old_hash:
            trailers = last_commit_trailers(path)
    except content_repo.ContentPublishError as exc:
        logger.error("%s/%s: could not read Content, leaving the record as it is: %s", name, guid, exc)
        row.outcome = "content_unreadable"
        return _Attempt(None, row)

    if current is None:
        logger.error("%s/%s: %s is missing from Content", name, guid, path)
        return _Attempt(fail(store.CONTENT_CONFLICT_ERROR), row)
    current_hash = transcribe.transcript_hash(current)
    if current_hash != old_hash:
        adopted = _mark_from_trailers(trailers, guid=guid, ledger_hash=old_hash, content_hash=current_hash)
        if adopted is None:
            logger.error(
                "%s/%s: Content's text (%s) is not the ledger's (%s) and no Ingest commit explains it",
                name,
                guid,
                current_hash[:12],
                old_hash[:12],
            )
            return _Attempt(fail(store.CONTENT_CONFLICT_ERROR), row)
        row.outcome, row.new_hash = "adopted", current_hash
        return _Attempt(adopted, row)

    started = time.monotonic()
    try:
        text, new_hash, decoded_under = transcribe_audio(record["audio_url"], hotwords)
    except net.AudioDownloadError as exc:
        logger.warning("%s/%s: audio download failed, retrying on a later run: %s", name, guid, exc)
        return _Attempt(fail(store.DOWNLOAD_ERROR), row)
    except (transcribe.TranscriptionError, transcribe.EmptyTranscriptError) as exc:
        logger.error("%s/%s: re-transcription failed, keeping the old transcript: %s", name, guid, exc)
        error = (
            store.EMPTY_TRANSCRIPT_ERROR
            if isinstance(exc, transcribe.EmptyTranscriptError)
            else store.TRANSCRIPTION_ERROR
        )
        return _Attempt(fail(error), row, model_called=True)
    row.seconds = time.monotonic() - started
    row.new_hash = new_hash
    profile = dataclasses.asdict(decoded_under)
    new_score = quality.score(text, hotwords=decoded_under.hotwords)
    old_score = quality.score(current, hotwords=record.get("transcription_hotwords"))
    row.old_score, row.new_score = old_score, new_score
    counts = dataclasses.asdict(new_score)
    transcribed_at = net.now()
    replaced = {
        "kind": "retranscribed",
        "expected_hash": old_hash,
        "transcript_hash": new_hash,
        "transcribed_at": transcribed_at,
        "profile": profile,
        "quality": counts,
        "superseded": None,
    }

    if new_hash == old_hash:
        row.outcome = "identical"
        return _Attempt(replaced, row, model_called=True)

    old_clean = quality.clean_words(current, hotwords=record.get("transcription_hotwords"))
    new_clean = quality.clean_words(text, hotwords=decoded_under.hotwords)
    refusal = list(quality.flags(new_score))
    if new_clean < _MIN_CLEAN_WORDS_RATIO * old_clean:
        refusal.append("clean_words")
    if refusal:
        candidate = f"{_CANDIDATES_DIR}/{name}/{PurePosixPath(path).stem}.{new_hash[:12]}.txt"
        trailers = {
            "Ingest-Guid": guid,
            "Ingest-Refusal": ",".join(refusal),
            "Ingest-Current-Hash": old_hash,
            "Ingest-Transcript-Hash": new_hash,
            "Ingest-Old-Clean-Words": json.dumps(old_clean),
            "Ingest-New-Clean-Words": json.dumps(new_clean),
            **{name_: json.dumps(profile[key]) for key, name_ in _PROFILE_TRAILERS.items()},
            **{
                _trailer_name("Ingest-Old-Quality-", key): json.dumps(getattr(old_score, key))
                for key in store.QUALITY_KEYS
            },
            **{
                _trailer_name("Ingest-New-Quality-", key): json.dumps(counts[key])
                for key in store.QUALITY_KEYS
            },
        }
        try:
            keep_candidate(candidate, text, trailers=trailers)
        except content_repo.ContentPublishError as exc:
            logger.error("%s/%s: refused candidate could not be kept in Content: %s", name, guid, exc)
            row.outcome = "candidate_push_failed"
            return _Attempt(None, row, model_called=True)
        logger.warning(
            "%s/%s: replacement refused (%s); candidate kept at %s", name, guid, ",".join(refusal), candidate
        )
        return _Attempt(
            fail(store.QUALITY_REGRESSION_ERROR, candidate_path=candidate, candidate_hash=new_hash),
            row,
            model_called=True,
        )

    trailers = _provenance_trailers(
        guid,
        superseded=old_hash,
        transcript_hash=new_hash,
        transcribed_at=transcribed_at,
        profile=profile,
        counts=counts,
    )
    try:
        replace(path, text, expect_hash=old_hash, trailers=trailers)
    except content_repo.ContentConflictError as exc:
        logger.error("%s/%s: Content changed under the replacement: %s", name, guid, exc)
        return _Attempt(fail(store.CONTENT_CONFLICT_ERROR), row, model_called=True)
    except content_repo.ContentPublishError as exc:
        logger.error("%s/%s: replacement push exhausted, leaving the record as it is: %s", name, guid, exc)
        row.outcome = "push_failed"
        return _Attempt(None, row, model_called=True)
    row.outcome = "replaced"
    return _Attempt({**replaced, "superseded": old_hash}, row, model_called=True)


def _log_attempt(row: SummaryRow) -> None:
    old, new = row.old_score, row.new_score
    logger.info(
        "church=%s guid=%s seconds=%s old_hash=%s new_hash=%s %s outcome=%s",
        row.church,
        row.guid,
        "" if row.seconds is None else f"{row.seconds:.0f}",
        (row.old_hash or "")[:12],
        (row.new_hash or "")[:12],
        " ".join(
            f"{key}={_change(getattr(old, key, None), getattr(new, key, None))}" for key in store.QUALITY_KEYS
        ),
        row.outcome,
    )


def _replace_transcript(path: str, text: str, *, expect_hash: str, trailers: Mapping[str, str]) -> None:
    content_repo.replace_transcript(path, text, expect_hash=expect_hash, trailers=trailers)


def _keep_candidate(path: str, text: str, *, trailers: Mapping[str, str]) -> None:
    content_repo.keep_candidate(path, text, trailers=trailers)


def run_retranscribe(
    *,
    church_names: list[str] | None,
    limit: int,
    guids: Sequence[str] | None = None,
    shard_index: int = 0,
    shard_count: int = 1,
    transcribe_audio: TranscribeAudio = transcribe.transcribe_audio,
    fetch_existing: FetchExisting = content_repo.existing_transcripts,
    last_commit_trailers: LastCommitTrailers = content_repo.last_commit_trailers,
    replace: ReplaceTranscript = _replace_transcript,
    keep_candidate: KeepCandidate = _keep_candidate,
    marks_out: dict[str, dict[str, dict]] | None = None,
    summary_rows: list[SummaryRow] | None = None,
) -> bool:
    """Re-transcribe at most ``limit`` stale ``done`` sermons per church (ADR-0022).

    Selection is :func:`select_stale`, sharded like :func:`run`. Each sermon's ledger
    record changes only through a mark: after Content took the new text, the text was
    identical, a previous run's commit proved it ours, or the attempt failed (which
    touches only the ``retranscription_*`` fields). Each record is saved as soon as it
    is marked. Pipeline is never dispatched.

    Returns ``False`` when a person is needed: a conflict, a model failure, Content
    unreadable, a push exhausted, a crash, or Whisper having run on a sermon that ended
    with no mark. A download failure or a refused replacement is an expected outcome.
    """
    selected = _selected_churches(church_names)
    if not selected:
        logger.warning("no enabled churches matched the selection; nothing to re-transcribe")
        return True
    loaded, ordered = _select_in_scope(selected, limit=limit, select=_stale_selector(guids))
    digest = store.profile_digest(_running_profile())
    shard = [pair for i, pair in enumerate(ordered) if i % shard_count == shard_index]

    all_ok = True
    model_calls = marked = 0
    church_terms = {name: prompting.church_vocabulary(loaded[name]) for name in {name for name, _g in shard}}
    for name, guid in shard:
        record = loaded[name][guid]
        hotwords = prompting.build_hotwords(
            record, church_terms=church_terms[name], vocabulary=selected[name].vocabulary
        )
        try:
            attempt = _retranscribe_sermon(
                name,
                guid,
                record,
                hotwords=hotwords,
                digest=digest,
                transcribe_audio=transcribe_audio,
                fetch_existing=fetch_existing,
                last_commit_trailers=last_commit_trailers,
                replace=replace,
                keep_candidate=keep_candidate,
            )
        except Exception:
            logger.exception("%s/%s: re-transcription crashed unexpectedly", name, guid)
            all_ok = False
            continue
        model_calls += attempt.model_called
        if attempt.mark is not None:
            _apply_retranscribe_mark(record, attempt.mark)
            store.save(name, loaded[name])
            marked += attempt.model_called
            if marks_out is not None:
                marks_out.setdefault(name, {})[guid] = attempt.mark
        _log_attempt(attempt.row)
        if summary_rows is not None:
            summary_rows.append(attempt.row)
        all_ok = all_ok and attempt.row.outcome in _SETTLED_OUTCOMES

    if model_calls != marked:
        logger.error("Whisper ran on %d sermon(s) that ended with no ledger mark", model_calls - marked)
        all_ok = False
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

    if args.replay_marks:
        return 0 if replay_marks(args.replay_marks) else 1

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

    marks: dict[str, dict[str, dict]] = {}
    if args.retranscribe_stale:
        rows: list[SummaryRow] = []
        ok = run_retranscribe(
            church_names=args.churches,
            limit=args.limit,
            guids=args.guids,
            shard_index=args.shard_index,
            shard_count=args.shard_count,
            marks_out=marks,
            summary_rows=rows,
        )
        for row in rows:
            if row.outcome in _WARNING_OUTCOMES:
                # A workflow annotation, not a log line: GitHub reads it only at the start of stdout.
                print(f"::warning::{row.church}/{row.guid}: re-transcription {row.outcome}")
        if args.summary_md and rows:
            with open(args.summary_md, "a", encoding="utf-8") as f:
                f.write(render_summary(rows))
    else:
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
