"""Re-transcription of stale-profile transcripts (ADR-0022).

The ``--retranscribe-stale`` mode of :mod:`poller.transcriber`'s CLI: it re-transcribes each
``done`` sermon whose recorded decoding profile differs from the running one, and replaces it
in Content only by compare-and-swap. It never dispatches Pipeline. See :func:`run_retranscribe`.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Any

from poller import config, content_repo, net, prompting, quality, selection, store, transcribe

TranscribeAudio = Callable[[str, str | None], tuple[str, str, transcribe.DecodingProfile]]
FetchExisting = Callable[[list[str]], dict[str, str]]
LastCommitTrailers = Callable[[str], dict[str, str]]
ReplaceTranscript = Callable[..., None]
KeepCandidate = Callable[..., None]

logger = logging.getLogger("poller")

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


def running_profile() -> dict[str, Any]:
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
    return selection.newest_first(fresh) + selection.newest_first(retries)


def stale_selector(guids: Sequence[str] | None) -> Callable[[dict[str, dict]], list[tuple[str, dict]]]:
    running = running_profile()
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
    except KeyError, json.JSONDecodeError:
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


def apply_mark(record: dict, mark: Mapping[str, Any]) -> bool:
    """Apply a ``retranscribed``/``retranscribe_failed`` mark if the record still matches it.

    The ledger-side compare-and-swap: a replacement applies while the record holds the
    hash the attempt saw or the hash it produced (so applying it twice is a no-op); a
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
WARNING_OUTCOMES = frozenset({store.DOWNLOAD_ERROR, store.QUALITY_REGRESSION_ERROR})


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
    refusal = list(quality.worse_flags(old_score, new_score))
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
    summary_rows: list[SummaryRow] | None = None,
) -> bool:
    """Re-transcribe at most ``limit`` stale ``done`` sermons per church (ADR-0022).

    Selection is :func:`select_stale`, sharded like :func:`poller.transcriber.run`. Each
    sermon's ledger record changes only through a mark: after Content took the new text,
    the text was identical, a previous run's commit proved it ours, or the attempt failed
    (which touches only the ``retranscription_*`` fields). Each record is saved as soon as
    it is marked. Pipeline is never dispatched.

    Returns ``False`` when a person is needed: a conflict, a model failure, Content
    unreadable, a push exhausted, a crash, or Whisper having run on a sermon that ended
    with no mark. A download failure or a refused replacement is an expected outcome.
    """
    selected = selection.selected_churches(church_names)
    if not selected:
        logger.warning("no enabled churches matched the selection; nothing to re-transcribe")
        return True
    loaded, ordered = selection.select_in_scope(selected, limit=limit, select=stale_selector(guids))
    digest = store.profile_digest(running_profile())
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
            apply_mark(record, attempt.mark)
            store.save(name, loaded[name])
            marked += attempt.model_called
        _log_attempt(attempt.row)
        if summary_rows is not None:
            summary_rows.append(attempt.row)
        all_ok = all_ok and attempt.row.outcome in _SETTLED_OUTCOMES

    if model_calls != marked:
        logger.error("Whisper ran on %d sermon(s) that ended with no ledger mark", model_calls - marked)
        all_ok = False
    return all_ok
