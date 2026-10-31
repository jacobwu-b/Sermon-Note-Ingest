"""The per-church JSON ledger: one file per church, keyed by guid.

Each record keeps every real field the source gave us — link, title, speaker,
series, the service date, the instant the feed first made it available (when
known), and the instant this poller first retrieved it — and nothing else.
Upsert is idempotent on guid: a re-poll of an already-known sermon never
duplicates it, and never clobbers ``first_seen_at``/``feed_published_at`` or any
``transcription_*``/``notified_at`` progress field. Every other feed-sourced
field (``audio_url``, ``title``, ``series``, ``speaker``, ``episode_url``,
``blurb``, ``preached_on``) is refreshed on every re-poll via
:func:`refresh_record`, since a feed can rotate or correct these after first
discovery (e.g. a CDN reissuing a signed enclosure URL) — a record must not be
stuck retrying a URL the feed no longer serves just because it was ledgered
once. ``title`` stops refreshing once transcription is ``"done"``: the pushed
transcript's title text is already fixed as of that push, and a later feed
rename must not desync the ledger from it. Every write re-sorts the whole file
newest-``preached_on``-first, so opening a ledger always shows the latest
sermon at the top.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from poller.quality import QualityScore
from poller.sources.base import SermonItem

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# The values of a record's ``transcription_last_error`` (ADR-0018).
DOWNLOAD_ERROR = "audio_download"
TRANSCRIPTION_ERROR = "transcription"
EMPTY_TRANSCRIPT_ERROR = "empty_transcript"
# Re-transcription only (ADR-0022): the new text was refused by the quality gate, or
# Content no longer held the text the ledger names.
QUALITY_REGRESSION_ERROR = "quality_regression"
CONTENT_CONFLICT_ERROR = "content_conflict"

# What a transcript records about how it was produced, each as ``transcription_<key>``
# (ADR-0012, ADR-0022); ``poller.transcribe.DecodingProfile`` has exactly these fields.
PROFILE_KEYS = (
    "model",
    "compute_type",
    "beam_size",
    "condition_on_previous_text",
    "hallucination_silence_threshold",
    "vad_filter",
    "domain_prompt",
    "hotwords",
    "faster_whisper_version",
    "ctranslate2_version",
)

# The profile keys that make a transcript stale when they differ from the running config
# (ADR-0022 rule 2). Engine versions and the hotwords string are recorded but left out, so a
# dependency bump or a vocabulary edit never queues the whole corpus.
STALENESS_KEYS = (
    "model",
    "compute_type",
    "beam_size",
    "condition_on_previous_text",
    "hallucination_silence_threshold",
    "vad_filter",
    "domain_prompt",
)

# A transcript's quality counts, each as ``transcript_<key>`` (ADR-0023);
# ``poller.quality.QualityScore`` has exactly these fields.
QUALITY_KEYS = ("words", "loop_words", "credit_hits", "hotword_pairs")

logger = logging.getLogger("poller")


def _record_path(church: str) -> Path:
    return DATA_DIR / f"{church}.json"


def _overrides_path(church: str) -> Path:
    return DATA_DIR / "overrides" / f"{church}.json"


# ADR-0015: the pre-rename names. A one-shot migration rewrote every committed ledger,
# but a poll/transcribe run on the old code can still land a record under these names
# after that merge — load() renames them so the next save() finishes the job.
_LEGACY_KEYS = {"published_on": "preached_on", "published_at": "feed_published_at"}


def _rename_legacy_keys(entry: dict[str, Any]) -> dict[str, Any]:
    """``entry`` with any pre-ADR-0015 key renamed, each keeping its position."""
    if not _LEGACY_KEYS.keys() & entry.keys():
        return entry
    return {_LEGACY_KEYS.get(key, key): value for key, value in entry.items()}


def _load_overrides(church: str) -> dict[str, dict[str, Any]]:
    """Manual field corrections for ``church``, keyed by guid (ADR-0008).

    Merged onto the base record by :func:`load` on every read, never written back by
    :func:`save` — a correction survives the next feed refresh without editing the ledger
    itself. Keys starting with ``_`` (``_reason``, ``_added_at``) are documentation for
    whoever edits the file by hand and are never merged into a record.
    """
    path = _overrides_path(church)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        overrides: dict[str, dict[str, Any]] = json.load(f)
    return {guid: _rename_legacy_keys(fields) for guid, fields in overrides.items()}


def _load_raw(church: str) -> dict[str, dict[str, Any]]:
    path = _record_path(church)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        records: dict[str, dict[str, Any]] = json.load(f)
    return records


def load(church: str) -> dict[str, dict[str, Any]]:
    """Load a church's ledger as ``{guid: record}``, or ``{}`` if it has none yet.

    Any matching entry in ``data/overrides/<church>.json`` is merged onto the returned
    records field by field (ADR-0008); an override for a guid absent from the ledger is
    skipped with a warning rather than applied or raised. A record or override still on
    the pre-ADR-0015 key names is returned under the new ones.
    """
    if not _record_path(church).exists():
        return {}
    records = {guid: _rename_legacy_keys(record) for guid, record in _load_raw(church).items()}

    for guid, fields in _load_overrides(church).items():
        record = records.get(guid)
        if record is None:
            logger.warning("%s: override for unknown guid %r ignored", church, guid)
            continue
        for key, value in fields.items():
            if key.startswith("_"):
                continue
            record[key] = value

    return records


def legacy_key_guids(church: str) -> list[str]:
    """Guids whose record in ``church``'s ledger file still uses a pre-ADR-0015 key name.

    Reads the file as written, not through :func:`load` (which renames on the way in), so
    ADR-0015's migration can tell which ledgers still need a rewrite.
    """
    return [guid for guid, record in _load_raw(church).items() if _LEGACY_KEYS.keys() & record.keys()]


def _ordered_items(records: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Newest-``preached_on`` first; records missing it sort after every dated one.

    Ties (same ``preached_on``, or no ``preached_on`` at all) break by
    ``feed_published_at`` then guid, both ascending — Python's sort is stable, so
    sorting by guid first and then by date leaves equal-date groups in
    guid-ascending order.
    """
    dated = [(guid, r) for guid, r in records.items() if r.get("preached_on")]
    undated = [(guid, r) for guid, r in records.items() if not r.get("preached_on")]
    undated.sort(key=lambda kv: kv[0])
    dated.sort(key=lambda kv: kv[0])
    dated.sort(key=lambda kv: (kv[1]["preached_on"], kv[1].get("feed_published_at") or ""), reverse=True)
    return dated + undated


def save(church: str, records: dict[str, dict[str, Any]]) -> None:
    """Write a church's ledger back, newest ``preached_on`` first, for an easy-to-scan file.

    Writes to a ``.tmp`` sibling and ``os.replace``s it onto the real path, so a process
    killed mid-write (a job timeout, a runner eviction) never leaves a truncated ledger on
    disk — the previous, complete ledger stays in place until the new one is fully written.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _record_path(church)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    ordered = dict(_ordered_items(records))
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(ordered, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def item_to_record(item: SermonItem, *, first_seen_at: str, feed_published_at: str | None) -> dict[str, Any]:
    """Build the JSON record for a newly-discovered item.

    The ``transcription_*``/``content_path`` fields are absent-by-default operational
    metadata (ADR-0004, ADR-0012): their value never holds the transcript text itself,
    only whether/where/how transcription happened, so this public repo can ledger
    transcription state without ever holding transcript content. An older record saved
    before these fields existed simply lacks them — :func:`load` and :func:`save` are
    field-agnostic, so no migration is needed.
    """
    return {
        "guid": item.guid,
        "title": item.title or None,
        "raw_title": item.raw_title or None,
        "series": item.series,
        "speaker": item.speaker,
        "preached_on": item.preached_on or None,
        "feed_published_at": feed_published_at,
        "episode_url": item.episode_url or None,
        "audio_url": item.audio_url or None,
        "blurb": item.blurb or None,
        "first_seen_at": first_seen_at,
        "notified_at": None,
        "transcription_status": None,
        "transcribed_at": None,
        "transcript_hash": None,
        "content_path": None,
        **{f"transcription_{key}": None for key in PROFILE_KEYS},
        "transcription_last_error": None,
        "transcript_adopted": None,
        **{f"transcript_{key}": None for key in QUALITY_KEYS},
    }


_REFRESHABLE_FIELDS = (
    "title",
    "raw_title",
    "series",
    "speaker",
    "preached_on",
    "episode_url",
    "audio_url",
    "blurb",
)

_DONE_FROZEN_FIELDS = frozenset({"title"})


def refresh_record(record: dict[str, Any], item: SermonItem) -> None:
    """Refresh ``record``'s feed-sourced fields in place from a rediscovered ``item``.

    Mirrors ``sermon_notes.registry.Registry.upsert``: on an already-ledgered guid,
    only feed-sourced metadata is refreshed — ``first_seen_at``, ``feed_published_at``, and
    every ``notified_at``/``transcription_*`` progress field are pipeline state, not
    feed data, and are left untouched here. ``feed_published_at`` in particular is resolved
    once at first discovery (some adapters do a network call for it) and is never
    recomputed on a re-poll — see ``SourceAdapter.resolve_feed_published_at``.

    ``title`` also stops refreshing once ``transcription_status`` is ``"done"``, same
    reasoning as the upstream's post-publish title freeze: the text already pushed to
    Content is fixed, so a later feed rename must not desync the ledger from it.
    """
    frozen = _DONE_FROZEN_FIELDS if record.get("transcription_status") == "done" else frozenset()
    values = {
        "title": item.title or None,
        "raw_title": item.raw_title or None,
        "series": item.series,
        "speaker": item.speaker,
        "preached_on": item.preached_on or None,
        "episode_url": item.episode_url or None,
        "audio_url": item.audio_url or None,
        "blurb": item.blurb or None,
    }
    for name in _REFRESHABLE_FIELDS:
        if name in frozen:
            continue
        record[name] = values[name]


def mark_transcribed(
    record: dict[str, Any],
    *,
    content_path: str,
    transcript_hash: str,
    transcribed_at: str,
    profile: Mapping[str, Any] | None,
    adopted: bool = False,
    quality: Mapping[str, int] | None = None,
) -> None:
    """Advance ``record`` to transcribed, recording where its text landed in Content.

    Called only once this record's content is durably in Content — the transcript
    text itself is never written here or anywhere else in this repo.
    ``profile`` is the decoding profile that produced this text, keyed by
    :data:`PROFILE_KEYS` (ADR-0012, ADR-0022), so a later re-transcription pass can select
    precisely instead of inferring from timestamps. A key it lacks, or a ``None`` profile
    (a transcript adopted from Content that this run didn't produce), records "unknown";
    a key the ledger doesn't know raises ``ValueError``. ``adopted`` marks the adopted case: its
    ``transcribed_at`` is when it was adopted, not produced, so latency reports skip it
    (ADR-0018). ``quality`` is the transcript's counts keyed by :data:`QUALITY_KEYS`
    (ADR-0023); ``None`` records "not scored", as for an adopted transcript.
    """
    profile = profile or {}
    quality = quality or {}
    unknown = sorted(set(profile) - set(PROFILE_KEYS)) + sorted(set(quality) - set(QUALITY_KEYS))
    if unknown:
        raise ValueError(f"not a ledger profile or quality field: {', '.join(unknown)}")
    record["transcription_status"] = "done"
    record["content_path"] = content_path
    record["transcript_hash"] = transcript_hash
    record["transcribed_at"] = transcribed_at
    for key in PROFILE_KEYS:
        record[f"transcription_{key}"] = profile.get(key)
    record["transcript_adopted"] = adopted
    for key in QUALITY_KEYS:
        record[f"transcript_{key}"] = quality.get(key)
    record["transcription_last_error"] = None


def transcript_quality(record: Mapping[str, Any]) -> QualityScore | None:
    """``record``'s transcript counts, or ``None`` if it was never scored (ADR-0023).

    Unscored covers pending and failed records, adopted transcripts and anything
    transcribed before scoring existed. None of these may read as clean.
    """
    counts: dict[str, Any] = {}
    for key in QUALITY_KEYS:
        value = record.get(f"transcript_{key}")
        if value is None:
            return None
        counts[key] = value
    return QualityScore(**counts)


def profile_is_stale(record: Mapping[str, Any], running: Mapping[str, Any]) -> bool:
    """Whether ``record``'s transcript was produced under a profile other than ``running``.

    ``running`` is keyed by :data:`STALENESS_KEYS`. A field the record lacks, or holds as
    ``None``, is unknown and counts as stale (ADR-0022 rule 2).
    """
    for key in STALENESS_KEYS:
        recorded = record.get(f"transcription_{key}")
        if recorded is None or recorded != running[key]:
            return True
    return False


def profile_digest(profile: Mapping[str, Any]) -> str:
    """A short key naming ``profile``'s :data:`STALENESS_KEYS` values.

    Stored as ``retranscription_failed_profile`` so a record that failed terminally isn't
    selected again until the profile changes. Only compared for equality, never queried.
    """
    values = {key: profile[key] for key in STALENESS_KEYS}
    values["hallucination_silence_threshold"] = float(values["hallucination_silence_threshold"])
    canonical = json.dumps(values, sort_keys=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


def mark_retranscribed(
    record: dict[str, Any],
    *,
    transcript_hash: str,
    transcribed_at: str,
    profile: Mapping[str, Any],
    quality: Mapping[str, int],
    superseded: str | None,
) -> None:
    """Record that ``record``'s transcript was re-produced under ``profile`` (ADR-0022 rule 3).

    The only writer of a ``done`` record's transcript fields in re-transcription. Called
    only once Content holds the new text, or the new text was byte-identical to the old
    (``superseded`` ``None``: there is nothing to supersede, so an earlier
    ``superseded_transcript_hash`` stays), or a previous run's Content commit proved the
    text is ours. ``content_path`` and ``transcription_status`` never change. Any earlier
    re-transcription failure, and its candidate pointer, is cleared.
    """
    unknown = sorted(set(profile) - set(PROFILE_KEYS)) + sorted(set(quality) - set(QUALITY_KEYS))
    if unknown:
        raise ValueError(f"not a ledger profile or quality field: {', '.join(unknown)}")
    record["transcript_hash"] = transcript_hash
    record["transcribed_at"] = transcribed_at
    for key in PROFILE_KEYS:
        record[f"transcription_{key}"] = profile.get(key)
    record["transcript_adopted"] = False
    for key in QUALITY_KEYS:
        record[f"transcript_{key}"] = quality.get(key)
    if superseded is not None:
        record["superseded_transcript_hash"] = superseded
    record["retranscription_last_error"] = None
    record["retranscription_failed_profile"] = None
    record["retranscription_candidate_path"] = None
    record["retranscription_candidate_hash"] = None


def mark_retranscribe_failed(
    record: dict[str, Any],
    *,
    error: str,
    failed_profile: str,
    candidate_path: str | None = None,
    candidate_hash: str | None = None,
) -> None:
    """Record a failed re-transcription attempt, leaving the current transcript's fields alone.

    ``failed_profile`` is :func:`profile_digest` of the profile the attempt ran under.
    ``candidate_path``/``candidate_hash`` point at a refused candidate kept in Content; a
    later failure without one leaves an earlier pointer in place.
    """
    record["retranscription_last_error"] = error
    record["retranscription_failed_profile"] = failed_profile
    if candidate_path is not None:
        record["retranscription_candidate_path"] = candidate_path
        record["retranscription_candidate_hash"] = candidate_hash


def mark_download_failed(record: dict[str, Any]) -> None:
    """Record that ``record``'s audio could not be downloaded; it stays pending for retry.

    Only a still-pending record is touched, so a replayed mark cannot relabel one another
    run has since settled. No attempt time is stored: a download failing on every run must
    leave the ledger byte-identical rather than commit on every run (ADR-0018).
    """
    if record.get("transcription_status") is None:
        record["transcription_last_error"] = DOWNLOAD_ERROR


def mark_transcription_failed(record: dict[str, Any], *, error: str) -> None:
    """Send ``record``'s transcription attempt terminal — the model failed after its retries.

    Distinct from a download failure, which leaves ``transcription_status`` untouched
    (``None``) so the next run retries it; this is for a failure that retrying the
    same audio would not fix. ``error`` is :data:`TRANSCRIPTION_ERROR` or
    :data:`EMPTY_TRANSCRIPT_ERROR`.
    """
    record["transcription_status"] = "failed"
    record["transcription_last_error"] = error


def _alerts_dir() -> Path:
    return DATA_DIR / "alerts"


def load_alert_state() -> dict[str, dict[str, Any]]:
    """Discovery alert state (ADR-0025), keyed by church, or ``{}`` before any has been saved."""
    path = _alerts_dir() / "state.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        state: dict[str, dict[str, Any]] = json.load(f)
    return state


def save_alert_state(state: Mapping[str, Mapping[str, Any]]) -> None:
    """Write discovery alert state, atomically like :func:`save`."""
    path = _alerts_dir() / "state.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    try:
        with tmp_path.open("w", encoding="utf-8") as f:
            json.dump(dict(sorted(state.items())), f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def load_alert_snoozes() -> dict[str, str]:
    """Each snoozed church's ``until`` date from the hand-edited ``data/alerts/snooze.json``.

    Like an override (ADR-0008), ``_``-prefixed keys are notes for whoever edits the file.
    """
    path = _alerts_dir() / "snooze.json"
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        snoozes: dict[str, dict[str, Any]] = json.load(f)
    return {church: str(entry["until"]) for church, entry in snoozes.items()}


def validate_all() -> list[str]:
    """Strict-load every ``data/*.json`` ledger and ``data/alerts/*.json`` file; return the
    paths, relative to ``data/``, that fail to parse.

    Run by both poll.yml's and transcribe.yml's commit steps before ``git add data/``, so a
    truncated or corrupted ledger fails the step instead of landing on main with `[skip ci]`.
    """
    if not DATA_DIR.exists():
        return []
    bad = []
    for path in sorted(DATA_DIR.glob("*.json")) + sorted(_alerts_dir().glob("*.json")):
        try:
            with path.open(encoding="utf-8") as f:
                json.load(f)
        except (OSError, json.JSONDecodeError):
            bad.append(path.relative_to(DATA_DIR).as_posix())
    return bad


def main(argv: list[str] | None = None) -> int:
    """``python -m poller.store validate`` — non-zero if any ledger fails to parse."""
    argv = sys.argv[1:] if argv is None else argv
    if argv != ["validate"]:
        print("usage: python -m poller.store validate", file=sys.stderr)
        return 2

    bad = validate_all()
    if bad:
        for name in bad:
            print(f"::error::data/{name} failed to parse as JSON", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
