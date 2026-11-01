"""Which sermons a transcription run is responsible for, shared by both of its modes.

:mod:`poller.transcriber` (pending sermons) and :mod:`poller.retranscribe` (stale
``done`` ones) each supply their own per-church selection; this module scopes it to the
enabled churches, caps it per church and orders it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from poller import config, store


def newest_first(pairs: Iterable[tuple[str, dict]]) -> list[tuple[str, dict]]:
    """Newest ``preached_on`` first, ties by guid; undated records last, by guid."""
    pairs = list(pairs)
    dated = [(guid, record) for guid, record in pairs if record.get("preached_on")]
    undated = [(guid, record) for guid, record in pairs if not record.get("preached_on")]
    dated.sort(key=lambda kv: kv[0])
    dated.sort(key=lambda kv: kv[1]["preached_on"], reverse=True)
    undated.sort(key=lambda kv: kv[0])
    return dated + undated


def selected_churches(church_names: list[str] | None) -> dict[str, config.ChurchConfig]:
    """Enabled churches, narrowed to ``church_names`` if given (default: all enabled)."""
    churches = config.load_churches()
    return {
        name: entry
        for name, entry in churches.items()
        if entry.enabled and (church_names is None or name in church_names)
    }


def select_in_scope(
    selected: dict[str, config.ChurchConfig],
    *,
    limit: int,
    select: Callable[[dict[str, dict]], list[tuple[str, dict]]],
) -> tuple[dict[str, dict], list[tuple[str, str]]]:
    """Every (church, guid) in scope for this run, in processing order, plus each
    touched church's full record set (for the run to save back).

    Each church in ``selected`` independently contributes up to ``limit`` of its own
    newest-``preached_on``-first selected sermons — ``limit`` is a per-church cap, not
    a budget shared across churches (ADR-0005, amended). This is the list each run
    shards — sharding only filters it, never reorders it, so shard-count 1 reproduces
    this order byte-for-byte. ``select`` is the mode's selection.
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
