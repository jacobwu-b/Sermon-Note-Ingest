"""The source-agnostic ingestion contract (ADR-0013).

A :class:`SourceAdapter` polls one church's source and upserts its main sermons
into the registry as ``discovered``. Everything downstream of ``discovered`` —
transcription, generation, render, index, delivery — is already
church-agnostic, so the adapter is the only place per-church ingestion logic
lives. :class:`FeedItem` is the normalized discovered-record shape that adapters
produce; :class:`PollResult` is the per-source poll summary the orchestrator
merges across adapters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

from sermon_notes.registry import Registry


@dataclass(frozen=True)
class FeedItem:
    """One normalized discovered record (the contract between adapters and the registry).

    ``raw_title`` keeps the full source title for suffix matching; ``title`` is the
    sermon-title portion that lands in the registry. ``published_weekday`` is in
    the source's stated timezone (email.utils convention: Monday=0 … Sunday=6).
    """

    guid: str
    raw_title: str
    title: str
    series: str | None
    speaker: str | None
    published_on: str
    published_weekday: int
    episode_url: str
    audio_url: str
    blurb: str
    scripture_refs: list[str] = field(default_factory=list)
    #: Full publication instant from the feed, kept in memory only (ADR-0031). The ledger
    #: stores ``published_on`` to the day; the automated-attempt cap needs the time of
    #: day, because a sermon published Sunday evening would otherwise be measured from
    #: that morning and read as hundreds of runs old on the polling cadence. ``None``
    #: when the feed date is missing or malformed.
    published_at: datetime | None = None


@dataclass(frozen=True)
class PollResult:
    """Summary of one source's POLL: what was discovered, flagged, excluded, or deferred."""

    discovered: list[str] = field(default_factory=list)
    flagged: list[str] = field(default_factory=list)
    excluded: int = 0
    deferred: bool = False
    #: Publication instant per upserted guid, in memory only (ADR-0031). Carries the
    #: feed's time of day to the attempt cap without persisting it: the cap's baseline
    #: must come from the feed, since a ledger the pipeline failed to write is exactly
    #: what it has to survive.
    publish_times: dict[str, datetime] = field(default_factory=dict)


class SourceAdapter(ABC):
    """One church's ingestion: poll its source and upsert main sermons as ``discovered``.

    Each concrete adapter owns its own fetch mechanism, classification rule, and
    source configuration (read through the config layer), and tags its records
    with its own :attr:`source`. A poll that cannot reach the source defers that
    source — :meth:`poll` returns ``PollResult(deferred=True)`` rather than raising —
    so the orchestrator can carry on with the other adapters and downstream stages.
    """

    #: Globally-unique source identity persisted on every record (e.g. ``"menlo"``).
    source: str

    @abstractmethod
    def poll(self, registry: Registry, *, limit: int | None = None) -> PollResult:
        """Fetch, classify, and upsert this source's main sermons as ``discovered``.

        Upserts are idempotent on guid, so a re-poll never duplicates a record or
        resets its lifecycle. When ``limit`` is given only the ``limit`` most recent
        qualifying sermons (by publication date) are upserted, while flagging and
        exclusion still cover the whole source.
        """
        raise NotImplementedError
