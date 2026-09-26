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

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

from poller.sources.base import SermonItem

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

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
        "transcription_model": None,
        "transcription_domain_prompt": None,
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
    model: str,
    domain_prompt: bool,
) -> None:
    """Advance ``record`` to transcribed, recording where its text landed in Content.

    Called only after :func:`poller.content_repo.push_transcripts` has already
    succeeded for this record's content — the transcript text itself is never
    written here or anywhere else in this repo. ``model``/``domain_prompt`` are the
    ``WhisperConfig`` values active for this transcription (ADR-0012), so a later
    re-transcription pass can select precisely instead of inferring from timestamps.
    """
    record["transcription_status"] = "done"
    record["content_path"] = content_path
    record["transcript_hash"] = transcript_hash
    record["transcribed_at"] = transcribed_at
    record["transcription_model"] = model
    record["transcription_domain_prompt"] = domain_prompt


def mark_transcription_failed(record: dict[str, Any]) -> None:
    """Send ``record``'s transcription attempt terminal — the model failed after its retries.

    Distinct from a download failure, which leaves ``transcription_status`` untouched
    (``None``) so the next run retries it; this is for a failure that retrying the
    same audio would not fix.
    """
    record["transcription_status"] = "failed"


def validate_all() -> list[str]:
    """Strict-load every ``data/*.json`` ledger; return the filenames that fail to parse.

    Run by both poll.yml's and transcribe.yml's commit steps before ``git add data/``, so a
    truncated or corrupted ledger fails the step instead of landing on main with `[skip ci]`.
    """
    if not DATA_DIR.exists():
        return []
    bad = []
    for path in sorted(DATA_DIR.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as f:
                json.load(f)
        except (OSError, json.JSONDecodeError):
            bad.append(path.name)
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
