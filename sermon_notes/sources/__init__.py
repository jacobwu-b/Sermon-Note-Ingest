"""Source adapters and the enabled-sources registry (ADR-0013, ADR-0074).

Ingestion is pluggable: the orchestrator polls the set of adapters this module
resolves from :mod:`sermon_notes.church_config`'s ``CHURCHES`` table — one JSON
object naming, per church, its feed URL and whether it is on. Adding a church
already covered by an adapter is registering it in ``CHURCHES`` — a GitHub
Actions variable, no workflow change. Adding a genuinely new church is still
writing a new adapter and registering it in :data:`_ADAPTERS`.

A manual ``workflow_dispatch`` may additionally narrow *which* enabled churches
a given run touches (the ``sources`` input, wired to ``ENABLED_SOURCES`` — a
one-run filter, not a persistent enablement mechanism; that lives in
``CHURCHES`` itself now).
"""

from __future__ import annotations

from typing import Protocol

from sermon_notes import church_config, config
from sermon_notes.logging import get_logger
from sermon_notes.sources.base import FeedItem, PollResult, SourceAdapter
from sermon_notes.sources.feedbase import FeedFetchError
from sermon_notes.sources.hillside import HillsideAdapter
from sermon_notes.sources.lakepointe import LakepointeAdapter
from sermon_notes.sources.menlo import MenloPodbeanAdapter
from sermon_notes.sources.north_point import NorthPointAdapter
from sermon_notes.sources.pbc import PbcAdapter
from sermon_notes.sources.westgate import WestgateAdapter

__all__ = [
    "FeedItem",
    "PollResult",
    "SourceAdapter",
    "FeedFetchError",
    "HillsideAdapter",
    "LakepointeAdapter",
    "MenloPodbeanAdapter",
    "NorthPointAdapter",
    "PbcAdapter",
    "WestgateAdapter",
    "enabled_adapters",
    "enabled_source_names",
]

logger = get_logger()


class _AdapterFactory(Protocol):
    """What every adapter class looks like from here: constructible with ``url=``.

    Each concrete adapter's ``__init__`` takes ``url`` (plus other, adapter-specific
    keyword defaults this call site never passes) — a narrower view than
    ``type[SourceAdapter]`` gives mypy, since the base class declares no constructor
    at all.
    """

    def __call__(self, *, url: str) -> SourceAdapter: ...


# The set of known source adapters, keyed by source identity. A new church adds
# its adapter factory here so ``CHURCHES`` can name it.
_ADAPTERS: dict[str, _AdapterFactory] = {
    "menlo": MenloPodbeanAdapter,
    "pbc": PbcAdapter,
    "north_point": NorthPointAdapter,
    "westgate": WestgateAdapter,
    "lakepointe": LakepointeAdapter,
    "hillside": HillsideAdapter,
}


def _dispatch_override() -> list[str] | None:
    """``ENABLED_SOURCES`` as a one-run narrowing filter, or ``None`` if unset.

    ``None`` — the ordinary case, every scheduled run — means every church
    ``CHURCHES`` marks ``enabled`` is in play. Set only by a manual
    ``workflow_dispatch``'s ``sources`` input (never a repository variable, since
    persistent enablement now lives in ``CHURCHES`` itself), it narrows that set to
    the named churches for one targeted run — it can never *widen* it: a church
    ``CHURCHES`` marks disabled stays disabled regardless of what this names.
    """
    raw = config.get("ENABLED_SOURCES", "")
    names = [name.strip() for name in raw.split(",") if name.strip()]
    return names or None


def _unknown_source_error(source: str) -> config.ConfigError:
    return config.ConfigError(
        f"{source!r} names a source this codebase has no adapter for; "
        f"known sources: {sorted(_ADAPTERS)}."
    )


def enabled_adapters(*, raw_churches: str | None = None) -> list[SourceAdapter]:
    """Instantiate every adapter ``CHURCHES`` marks enabled (ADR-0074).

    Each named source's feed URL and on/off state come from ``CHURCHES``; a source
    marked enabled whose ``rss`` is missing, empty, or not a usable URL is skipped —
    logged at ``WARNING``, never raised. Naming an unrecognized source identity
    (not one this codebase has an adapter for at all) is still a configuration
    error *when that entry is enabled* — an operator turning on a typo should hear
    about it; a disabled placeholder for a church with no adapter yet (staged ahead
    of time) is not an error.

    ``ENABLED_SOURCES`` (:func:`_dispatch_override`), when a manual dispatch set it,
    further narrows the result to the named churches — naming an unrecognized
    source there is a configuration error unconditionally, the same reasoning as
    an enabled unknown church above.

    ``raw_churches`` overrides the live ``CHURCHES`` value; tests pass it so a test
    never depends on process environment. The real caller leaves it ``None``.
    """
    entries = church_config.churches(raw_churches)
    for name, entry in entries.items():
        if entry.enabled and name not in _ADAPTERS:
            raise _unknown_source_error(name)

    override = _dispatch_override()
    if override is not None:
        for name in override:
            if name not in _ADAPTERS:
                raise _unknown_source_error(name)

    adapters: list[SourceAdapter] = []
    for name, entry in entries.items():
        if not entry.enabled:
            continue
        if override is not None and name not in override:
            continue
        if not church_config.is_usable_rss(entry.rss):
            logger.warning(
                "CHURCHES entry for %r has no usable rss url, skipping it: %r", name, entry.rss
            )
            continue
        assert entry.rss is not None  # is_usable_rss(None) is False, so this always holds
        adapters.append(_ADAPTERS[name](url=entry.rss))
    return adapters


def enabled_source_names() -> list[str]:
    """The church names :func:`enabled_adapters` would attempt, before feed validity.

    Split out so a caller can tell "nothing was requested" (an operator's
    deliberate choice — no church enabled, or a dispatch override naming none of
    them) from "something was requested but none of it resolved" (issue #496)
    without re-deriving the same enablement logic.
    """
    entries = church_config.churches(None)
    override = _dispatch_override()
    names = [name for name, entry in entries.items() if entry.enabled]
    if override is not None:
        names = [name for name in names if name in override]
    return names
