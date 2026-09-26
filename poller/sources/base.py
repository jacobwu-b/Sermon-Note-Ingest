"""The per-church adapter contract.

A :class:`SourceAdapter` fetches one church's feed and returns the main
sermons it finds as normalized :class:`SermonItem` records. Everything a church
adapter needs to decide — which items are the main Sunday sermon versus a
different program sharing the same feed — lives in the adapter; everything
downstream of that decision (storing it, notifying about it) is
church-agnostic and lives in :mod:`poller.runner`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


@dataclass(frozen=True)
class SermonItem:
    """One normalized main-sermon record — the contract between adapters and the store.

    ``preached_on`` is the sermon's service date (``YYYY-MM-DD``), which for
    some churches is derived from the title rather than the feed's ``pubDate``
    (see the Westgate and Hillside adapters). ``feed_published_at`` is the real
    instant the feed first made the item available, when the feed carries one —
    left ``None`` rather than filled with a placeholder (e.g. PBC's ``pubDate``
    is always a fixed nominal value, never the true publish time).
    """

    guid: str
    title: str
    raw_title: str
    series: str | None
    speaker: str | None
    preached_on: str
    feed_published_at: datetime | None
    episode_url: str
    audio_url: str
    blurb: str


@dataclass(frozen=True)
class PollResult:
    """What one adapter's poll produced: the qualifying items, plus a defer flag."""

    items: list[SermonItem] = field(default_factory=list)
    excluded: int = 0
    deferred: bool = False


class SourceAdapter(ABC):
    """One church's ingestion: fetch its feed and return its main sermons.

    A poll that cannot reach the source defers (``PollResult(deferred=True)``)
    rather than raising, so the runner can carry on with the other churches.
    """

    #: Globally-unique source identity (e.g. ``"menlo"``), matching its ``CHURCHES`` key.
    source: str

    def __init__(self, *, url: str) -> None:
        self.url = url

    @abstractmethod
    def poll(self) -> PollResult:
        """Fetch, parse, and classify this source's feed into main sermons."""
        raise NotImplementedError

    def resolve_feed_published_at(self, item: SermonItem) -> datetime | None:
        """The real "first made available" instant for a newly-discovered item.

        The default is a no-op returning ``item.feed_published_at`` unchanged: most
        feeds' own ``pubDate`` already is that instant. An adapter overrides
        this only when the feed's ``pubDate`` is a placeholder (PBC) and the
        real instant must be resolved some other way. The runner calls this
        once, for a guid's first discovery only — never on a re-poll of an
        already-known item — so an adapter that resolves it via network can do
        so without turning into a per-poll cost.
        """
        return item.feed_published_at
