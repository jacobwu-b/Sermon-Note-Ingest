"""The per-church JSON ledger: one file per church, keyed by guid.

Each record keeps every real field the source gave us — link, title, speaker,
series, the service date, the instant the feed first made it available (when
known), and the instant this poller first retrieved it — and nothing else.
Upsert is idempotent on guid: a re-poll of an already-known sermon never
duplicates it, and never clobbers ``first_seen_at``/``published_at`` or any
``transcription_*``/``notified_at`` progress field. Every other feed-sourced
field (``audio_url``, ``title``, ``series``, ``speaker``, ``episode_url``,
``blurb``, ``published_on``) is refreshed on every re-poll via
:func:`refresh_record`, since a feed can rotate or correct these after first
discovery (e.g. a CDN reissuing a signed enclosure URL) — a record must not be
stuck retrying a URL the feed no longer serves just because it was ledgered
once. ``title`` stops refreshing once transcription is ``"done"``: the pushed
transcript's title text is already fixed as of that push, and a later feed
rename must not desync the ledger from it. Every write re-sorts the whole file
newest-``published_on``-first, so opening a ledger always shows the latest
sermon at the top.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from poller.sources.base import SermonItem

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _record_path(church: str) -> Path:
    return DATA_DIR / f"{church}.json"


def load(church: str) -> dict[str, dict[str, Any]]:
    """Load a church's ledger as ``{guid: record}``, or ``{}`` if it has none yet."""
    path = _record_path(church)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _ordered_items(records: dict[str, dict[str, Any]]) -> list[tuple[str, dict[str, Any]]]:
    """Newest-``published_on`` first; records missing it sort after every dated one.

    Ties (same ``published_on``, or no ``published_on`` at all) break by
    ``published_at`` then guid, both ascending — Python's sort is stable, so
    sorting by guid first and then by date leaves equal-date groups in
    guid-ascending order.
    """
    dated = [(guid, r) for guid, r in records.items() if r.get("published_on")]
    undated = [(guid, r) for guid, r in records.items() if not r.get("published_on")]
    undated.sort(key=lambda kv: kv[0])
    dated.sort(key=lambda kv: kv[0])
    dated.sort(key=lambda kv: (kv[1]["published_on"], kv[1].get("published_at") or ""), reverse=True)
    return dated + undated


def save(church: str, records: dict[str, dict[str, Any]]) -> None:
    """Write a church's ledger back, newest ``published_on`` first, for an easy-to-scan file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = _record_path(church)
    ordered = dict(_ordered_items(records))
    with path.open("w", encoding="utf-8") as f:
        json.dump(ordered, f, indent=2, ensure_ascii=False)
        f.write("\n")


def item_to_record(item: SermonItem, *, first_seen_at: str, published_at: str | None) -> dict[str, Any]:
    """Build the JSON record for a newly-discovered item.

    The four ``transcription_*``/``content_path`` fields are absent-by-default
    operational metadata (ADR-0004): their value never holds the transcript text
    itself, only whether/where transcription happened, so this public repo can
    ledger transcription state without ever holding transcript content. An older
    record saved before these fields existed simply lacks them — :func:`load` and
    :func:`save` are field-agnostic, so no migration is needed.
    """
    return {
        "guid": item.guid,
        "title": item.title or None,
        "raw_title": item.raw_title or None,
        "series": item.series,
        "speaker": item.speaker,
        "published_on": item.published_on or None,
        "published_at": published_at,
        "episode_url": item.episode_url or None,
        "audio_url": item.audio_url or None,
        "blurb": item.blurb or None,
        "first_seen_at": first_seen_at,
        "notified_at": None,
        "transcription_status": None,
        "transcribed_at": None,
        "transcript_hash": None,
        "content_path": None,
    }


_REFRESHABLE_FIELDS = (
    "title",
    "raw_title",
    "series",
    "speaker",
    "published_on",
    "episode_url",
    "audio_url",
    "blurb",
)

_DONE_FROZEN_FIELDS = frozenset({"title"})


def refresh_record(record: dict[str, Any], item: SermonItem) -> None:
    """Refresh ``record``'s feed-sourced fields in place from a rediscovered ``item``.

    Mirrors ``sermon_notes.registry.Registry.upsert``: on an already-ledgered guid,
    only feed-sourced metadata is refreshed — ``first_seen_at``, ``published_at``, and
    every ``notified_at``/``transcription_*`` progress field are pipeline state, not
    feed data, and are left untouched here. ``published_at`` in particular is resolved
    once at first discovery (some adapters do a network call for it) and is never
    recomputed on a re-poll — see ``SourceAdapter.resolve_published_at``.

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
        "published_on": item.published_on or None,
        "episode_url": item.episode_url or None,
        "audio_url": item.audio_url or None,
        "blurb": item.blurb or None,
    }
    for name in _REFRESHABLE_FIELDS:
        if name in frozen:
            continue
        record[name] = values[name]


def mark_transcribed(
    record: dict[str, Any], *, content_path: str, transcript_hash: str, transcribed_at: str
) -> None:
    """Advance ``record`` to transcribed, recording where its text landed in Content.

    Called only after :func:`poller.content_repo.push_transcripts` has already
    succeeded for this record's content — the transcript text itself is never
    written here or anywhere else in this repo.
    """
    record["transcription_status"] = "done"
    record["content_path"] = content_path
    record["transcript_hash"] = transcript_hash
    record["transcribed_at"] = transcribed_at


def mark_transcription_failed(record: dict[str, Any]) -> None:
    """Send ``record``'s transcription attempt terminal — the model failed after its retries.

    Distinct from a download failure, which leaves ``transcription_status`` untouched
    (``None``) so the next run retries it; this is for a failure that retrying the
    same audio would not fix.
    """
    record["transcription_status"] = "failed"
