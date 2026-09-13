"""The church-to-channel routing table for published-note delivery (ADR-0067, spec 0024).

``NOTIFY_CHANNELS_JSON`` is one JSON object mapping a church's source key to a list of
``{"url": ..., "name": ...}`` webhook entries — any number of channels per church, mixing
kinds freely. This module is the only place that reads that variable, the only place that
infers a channel's *kind* (Discord vs. Google Chat) from its URL's host, and the seam
:mod:`pipeline` calls through instead of the per-source env-var dict this replaced.

Every failure mode here — an unset or malformed ``NOTIFY_CHANNELS_JSON``, a church with no
entry, an entry whose host matches neither known kind — resolves to "no channel," logged at
``WARNING`` where it isn't the ordinary unconfigured case, and never raises. The pipeline
never halts on missing or bad delivery configuration (#465).
"""

from __future__ import annotations

import json
import urllib.parse
from dataclasses import dataclass

from sermon_notes import config
from sermon_notes.logging import get_logger
from sermon_notes.registry import SermonRecord

logger = get_logger()

_DISCORD_HOST = "discord.com"
_GOOGLE_CHAT_HOST = "chat.googleapis.com"


@dataclass(frozen=True)
class Channel:
    """One resolved, dispatchable channel: its kind, its webhook URL, and an optional
    name carried through only for log identification — never logged itself is the URL."""

    kind: str
    url: str
    name: str | None


def _parse_channels(raw: str | None) -> dict[str, list[dict[str, object]]]:
    """Parse ``NOTIFY_CHANNELS_JSON`` into ``{church: [entry, ...]}``, or ``{}``.

    Total: unset/empty, invalid JSON, a non-object, or a church value that isn't a list
    all degrade to "no channels configured" rather than raising. A parse or shape failure
    logs once at ``WARNING`` so a misconfiguration is visible without halting the run —
    an unset variable does not log, since that is the ordinary "nothing configured yet"
    case every source starts in.
    """
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        logger.warning("NOTIFY_CHANNELS_JSON is not valid JSON, treating as unconfigured: %s", exc)
        return {}
    if not isinstance(payload, dict):
        logger.warning("NOTIFY_CHANNELS_JSON is not a JSON object, treating as unconfigured")
        return {}
    churches: dict[str, list[dict[str, object]]] = {}
    for church, entries in payload.items():
        if not isinstance(entries, list):
            logger.warning("NOTIFY_CHANNELS_JSON entry for %r is not a list, skipping it", church)
            continue
        churches[church] = entries
    return churches


def _kind_for(url: str) -> str | None:
    """Which boundary handles ``url``, inferred from its host; ``None`` if neither."""
    host = urllib.parse.urlsplit(url).hostname
    if host == _DISCORD_HOST:
        return "discord"
    if host == _GOOGLE_CHAT_HOST:
        return "google_chat"
    return None


def channels_for(source: str, *, raw: str | None = None) -> list[Channel]:
    """The channels configured for ``source``'s church, each with its kind resolved.

    ``raw`` overrides the live ``NOTIFY_CHANNELS_JSON`` value; tests pass it so a test
    never depends on process environment. The real caller (:mod:`pipeline`) leaves it
    ``None`` and this reads config fresh on every call. A malformed entry (not an object,
    or missing a string ``url``) or one whose host matches neither known kind is skipped
    and logged here, naming the church and the entry's ``name`` when it has one — never
    the URL — so every caller downstream sees only channels it can actually deliver to.
    """
    churches = _parse_channels(raw if raw is not None else config.get("NOTIFY_CHANNELS_JSON", None))
    channels: list[Channel] = []
    for entry in churches.get(source, []):
        url = entry.get("url") if isinstance(entry, dict) else None
        if not isinstance(url, str):
            logger.warning(
                "NOTIFY_CHANNELS_JSON has a malformed channel entry for %r, skipping it", source
            )
            continue
        name = entry.get("name") if isinstance(entry, dict) else None
        name = name if isinstance(name, str) else None
        kind = _kind_for(url)
        if kind is None:
            logger.warning(
                "channel %r configured for %r has an unrecognized webhook host, skipping it",
                name or "<unnamed>",
                source,
            )
            continue
        channels.append(Channel(kind=kind, url=url, name=name))
    return channels


def already_sent(sermon: SermonRecord, channel: Channel) -> bool:
    """Whether ``sermon``'s note has already been posted to ``channel``.

    The single source of truth for "don't send this again," shared by the regular
    publish flow (:mod:`pipeline`) and the one-shot backfill script — both call this
    before posting so a sermon reprocessed by a second run (a manual re-run of an
    already-succeeded workflow attempt, a redispatched reconcile) can't double-post
    (incident: a re-run redelivered three sermons a moment after its own earlier
    attempt had already delivered them).

    "Already sent" is per channel, not per sermon: a church with both a channel this
    predates and a brand-new one needs the old one skipped and the new one sent, for
    the same sermon. A named channel is checked against ``channel_message_ids``
    (spec 0024 amendment, ADR-0067 amendment); Discord additionally falls back to the
    legacy single ``discord_message_id`` field for the single-webhook-per-church era
    that predates the routing table. An unnamed non-Discord channel has no field to
    check and is therefore never recognized as already-sent — give every channel a
    ``name`` in ``NOTIFY_CHANNELS_JSON`` to get this guard's protection.
    """
    if channel.name and channel.name in sermon.channel_message_ids:
        return True
    return channel.kind == "discord" and sermon.discord_message_id is not None
