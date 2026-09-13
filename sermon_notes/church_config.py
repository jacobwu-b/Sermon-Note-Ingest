"""Parses ``CHURCHES``: the per-church config table (ADR-0074).

One JSON object, keyed by source identity, replacing both the six per-church
``*_FEED_URL`` vars and ``ENABLED_SOURCES`` (feed URL + enablement) and the
hardcoded ``_BATCH_SOURCES`` set in :mod:`pipeline` (ADR-0063's per-church
Batches-API eligibility, now config-driven instead of code-driven)::

    {"menlo": {"rss": "https://...", "enabled": true, "api": "normal"},
     "north_point": {"rss": "https://...", "enabled": true, "api": "batch"}}

This module owns parsing that one variable and nothing else — it does not
decide which parsed entries are *usable* (a missing/malformed ``rss`` is
:mod:`sources`' call, since only it knows what "usable" means for an
adapter) or which entries a given run *wants* (a manual dispatch's narrowing
override is :mod:`sources`' call too). Two very different consumers read the
same parse: :mod:`sources` (feed URL + enablement) and :mod:`pipeline` (the
generation API mode) — this module is their one shared source of truth so
neither re-parses or drifts from the other.
"""

from __future__ import annotations

import json
import urllib.parse
from dataclasses import dataclass
from typing import Literal

from sermon_notes import config
from sermon_notes.logging import get_logger

logger = get_logger()

ApiMode = Literal["batch", "normal"]

_DEFAULT_API: ApiMode = "normal"
_VALID_API_MODES = frozenset({"batch", "normal"})


@dataclass(frozen=True)
class ChurchEntry:
    """One church's parsed config. Always present; individual fields may be defaulted."""

    #: The feed URL, or ``None`` if missing/not a string. Shape (http(s) + host) is
    #: not validated here — see :func:`is_usable_rss`.
    rss: str | None
    #: Whether this church is on. Defaults to ``False`` on a malformed value: an
    #: entry this module cannot make sense of should never accidentally poll.
    enabled: bool
    #: Which path this church's generation takes. Defaults to ``"normal"``
    #: (synchronous) on a malformed value — the safe direction, since the failure
    #: mode of wrongly defaulting to ``"batch"`` is a silent latency regression
    #: nobody asked for, not a crash.
    api: ApiMode


def _parse_entry(source: str, raw_entry: object) -> ChurchEntry | None:
    """One church's raw JSON value into a :class:`ChurchEntry`, or ``None`` to drop it."""
    if not isinstance(raw_entry, dict):
        logger.warning("CHURCHES entry for %r is not an object, skipping it", source)
        return None
    rss = raw_entry.get("rss")
    if rss is not None and not isinstance(rss, str):
        logger.warning("CHURCHES entry for %r has a non-string rss, ignoring it", source)
        rss = None
    enabled = raw_entry.get("enabled", False)
    if not isinstance(enabled, bool):
        logger.warning(
            "CHURCHES entry for %r has a non-boolean enabled (%r), treating as false",
            source,
            enabled,
        )
        enabled = False
    api = raw_entry.get("api", _DEFAULT_API)
    if api not in _VALID_API_MODES:
        logger.warning(
            "CHURCHES entry for %r has an unrecognized api %r, defaulting to %r",
            source,
            api,
            _DEFAULT_API,
        )
        api = _DEFAULT_API
    return ChurchEntry(rss=rss, enabled=enabled, api=api)


def parse_churches(raw: str | None) -> dict[str, ChurchEntry]:
    """Parse ``CHURCHES`` into ``{source: ChurchEntry}``, or ``{}``.

    Mirrors :func:`notify_channels._parse_channels` (ADR-0067): unset/empty, invalid
    JSON, or a non-object top level all degrade to "nothing configured" rather than
    raising. A malformed *church's* entry is dropped (or field-defaulted — see
    :class:`ChurchEntry`) rather than discarding every other church with it. Every
    failure logs once at ``WARNING``; an unset variable does not, since that is the
    ordinary not-yet-configured case.
    """
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        logger.warning("CHURCHES is not valid JSON, treating as unconfigured: %s", exc)
        return {}
    if not isinstance(payload, dict):
        logger.warning("CHURCHES is not a JSON object, treating as unconfigured")
        return {}
    entries: dict[str, ChurchEntry] = {}
    for source, raw_entry in payload.items():
        entry = _parse_entry(source, raw_entry)
        if entry is not None:
            entries[source] = entry
    return entries


def churches(raw: str | None = None) -> dict[str, ChurchEntry]:
    """The parsed ``CHURCHES`` table. ``raw`` overrides the live config; tests pass it.

    The real callers (:mod:`sources`, :mod:`pipeline`) leave ``raw`` ``None`` and this
    reads config fresh on every call — deliberately not cached, so a test's
    ``monkeypatch.setenv`` is always seen and a mid-run config edit (there isn't one
    today) could never observe a stale table.
    """
    return parse_churches(raw if raw is not None else config.get("CHURCHES", None))


def is_usable_rss(url: str | None) -> bool:
    """Whether ``url`` is shaped like a fetchable feed URL — scheme and host only.

    A syntactically fine URL that 404s or times out is a poll-time failure, already
    covered by each adapter's own fetch retries and by ``_poll``'s per-adapter
    isolation (ADR-0013) — not this check, which only screens the config-shape
    failures ``CHURCHES`` can itself contain (missing, empty, or not a URL at all).
    """
    if not url:
        return False
    parsed = urllib.parse.urlsplit(url)
    return parsed.scheme in ("http", "https") and bool(parsed.netloc)


def api_mode(source: str, *, raw: str | None = None) -> ApiMode:
    """Which generation path ``source`` takes — ``"batch"`` or ``"normal"`` (ADR-0063).

    A source absent from an otherwise-populated ``CHURCHES`` — a sermon from before
    this migration, or a source since removed from the table — defaults to
    ``"normal"`` silently, the same safe direction an unrecognized ``api`` value
    defaults to.

    A wholly *empty* table is different: no legitimate run has zero churches
    configured, so that shape means the config never arrived (e.g. ``CHURCHES``
    wasn't wired into this job) rather than a deliberate per-source omission. That
    case still defaults to ``"normal"``, but logs at ``ERROR`` first — see #540.
    """
    table = churches(raw)
    if not table:
        logger.error(
            "CHURCHES has no entries at all; api_mode(%r) is falling back to %r. "
            "No legitimate run has zero churches configured — check that CHURCHES "
            "is wired into this job.",
            source,
            _DEFAULT_API,
        )
        return _DEFAULT_API
    entry = table.get(source)
    if entry is None:
        return _DEFAULT_API
    return entry.api
