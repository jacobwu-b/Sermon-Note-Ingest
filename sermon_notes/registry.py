"""The sermon ledger: sole reader/writer of ``state/registry.json``.

This module is the ONLY code that opens ``state/registry.json`` (CLAUDE.md §6,
ADR-0002). It loads the ledger into typed records, enforces the linear state
machine (PRD §6.3), keeps run history append-only, and exposes the
``transcript_hash`` re-generation guard (PRD §11.3). The on-disk shape is fixed
by PRD §6.2; any change to it is a migration shipped in the same PR.

Mutations are in-memory; callers :meth:`Registry.save` to flush them. The write is
atomic (``os.replace``), so the orchestrator saves after each sermon advances to
bound crash loss to a single sermon (#35) — "each run reads the ledger, advances
sermons, and writes it back" still holds, just incrementally.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from sermon_notes import audio_fingerprint

# Where the ledger lives inside a repo root — the only place that name is spelled
# (CLAUDE.md §6). DEFAULT_REGISTRY_PATH resolves it against the package's own root, two
# parents above this file; a caller holding some *other* checkout (the workflow's
# artifact commit, #269) joins this onto that root instead.
REGISTRY_RELPATH = Path("state") / "registry.json"
DEFAULT_REGISTRY_PATH = Path(__file__).resolve().parents[2] / REGISTRY_RELPATH

# Church display names keyed by source identity (ADR-0012). The single source of
# truth shared by every consumer that turns a record's ``source`` into a human name —
# the ``.docx`` header band/provenance and the README index. A new church adds its
# entry here so no church name is ever hardcoded at a call site.
CHURCH_NAMES = {
    "menlo": "Menlo Church",
    "pbc": "Peninsula Bible Church",
    "north_point": "North Point Community Church",
    "westgate": "WestGate Church",
    "lakepointe": "Lakepointe Church",
    "hillside": "Hillside Church",
}


def church_name(source: str) -> str:
    """The church display name for a record's ``source`` (ADR-0012).

    Raises :class:`KeyError` for a source with no registered name; callers that
    need a domain-specific failure (e.g. render) wrap it.
    """
    return CHURCH_NAMES[source]


# PRD §6.3 lifecycle. Each state maps to the states it may legally advance to.
# Linear path plus any→failed; ``published`` and ``failed`` are terminal sinks.
# ``pending_batch`` is a detour off ``transcribed`` for a backfill sermon awaiting an
# async Anthropic Batches API result (spec 0020, ADR-0061): it resolves to
# ``generated`` either from the batch result or, after the 24h fallback, from an
# ordinary synchronous call — the state machine can't tell the two apart, and doesn't
# need to.
_TRANSITIONS: dict[str, frozenset[str]] = {
    "discovered": frozenset({"transcribed", "failed"}),
    "transcribed": frozenset({"generated", "pending_batch", "failed"}),
    "pending_batch": frozenset({"generated", "failed"}),
    "generated": frozenset({"published", "failed"}),
    "published": frozenset(),
    "failed": frozenset(),
}

# The non-terminal lifecycle states a stranded ``failed`` record may be reset to by
# :meth:`Registry.recover` (ADR-0014). ``failed`` and ``published`` are terminal and
# are never recovery targets. ``pending_batch`` is deliberately excluded too: recovery
# resumes a sermon from ``transcribed`` so the pipeline re-decides whether to submit a
# fresh batch or call synchronously, rather than resurrecting a batch id that may no
# longer resolve to anything on Anthropic's side.
_RECOVERABLE_STATES = frozenset({"discovered", "transcribed", "generated"})

# Feed-sourced fields refreshed on re-discovery; lifecycle fields are preserved.
_METADATA_FIELDS = (
    "episode_url",
    "audio_url",
    "published_on",
    "published_at",
    "title",
    "series",
    "speaker",
    "scripture_refs",
    "blurb",
)

# ``title`` is excluded from the refresh once a record is ``published`` (issue #183):
# the title is already baked into the rendered artifact by then, and the feed
# renaming an episode after the fact must not desync the ledger from what a reader
# actually sees in the file. Every other metadata field keeps refreshing.
_PUBLISHED_FROZEN_FIELDS = frozenset({"title"})


class RegistryError(RuntimeError):
    """Base class for registry errors."""


class IllegalTransitionError(RegistryError):
    """Raised when a state transition is not permitted by PRD §6.3."""


class UnknownSermonError(RegistryError):
    """Raised when an operation targets a guid not present in the ledger."""


class InvalidGuidError(RegistryError):
    """Raised when an operation supplies an empty guid.

    ``guid`` is the ledger's primary key; an empty guid is a valid-looking key
    that every id-less caller would share, silently collapsing distinct sermons
    into one record (issue #206). Adapters are expected to filter these out
    before ever reaching :meth:`Registry.upsert`, but the check lives here too
    so the invariant is enforced by the module that owns it (CLAUDE.md §6),
    not just by callers that remember to.
    """


@dataclass(kw_only=True)
class RunRecord:
    """One pipeline attempt against a sermon (PRD §6.2 ``runs[]`` entry).

    Field order matches PRD §6.2 so serialization stays diff-stable.

    ``outcome`` is ``success``, ``retry_scheduled`` (ADR-0009), ``failed_terminal``, or
    ``billed`` — the last written by :func:`~sermon_notes.generate.generate_notes` the
    moment a paid LLM call returns and amended to ``success`` when the note validates, so
    a generation that fails after the model was billed still records what it cost
    (ADR-0035). A record left at ``billed`` means exactly that: money spent, no note.

    ``llm_stop_reason`` is the SDK's ``message.stop_reason`` verbatim (``null`` for
    non-LLM runs), stamped at the same ``billed`` point as ``llm_model`` so a
    truncated completion (``"max_tokens"``) is a recorded fact rather than an
    inference from a token count that happens to match the cap (ADR-0054, #368).

    ``llm_billing`` is ``LLMResponse.billing`` verbatim (``"sync"`` or ``"batch"``,
    ``null`` for non-LLM runs), and ``llm_batch_id`` is the Batches API id that
    produced the completion, or ``null`` for a synchronous call or a non-LLM run.
    Both are stamped at the same ``billed`` point as ``llm_model`` so a resolved run
    names the rate and the batch that actually billed it, rather than leaving that
    fact recoverable only by reverse-engineering the recorded cost (ADR-0082, #543).
    """

    attempted_state: str
    outcome: str
    llm_input_tokens: int = 0
    llm_output_tokens: int = 0
    llm_cost_usd: float = 0.0
    llm_model: str | None = None
    llm_stop_reason: str | None = None
    llm_billing: str | None = None
    llm_batch_id: str | None = None
    error_class: str | None = None
    error_detail: str | None = None
    started_at: str
    finished_at: str


# The ``RunRecord.error_class`` a sermon retired by the attempt cap carries (#240).
# Ledger vocabulary, so it lives with the record that holds it rather than with the guard
# that writes it: the pipeline decides *when* a sermon is retired this way, and
# ``scripts/recover_sermon.py`` reads it back off the ledger to tell an operator whether
# a recovery still needs its ``workflow_dispatch`` half (#267). A ledger-only tool should
# not have to import the orchestrator to know one of its own field's values.
ATTEMPT_BUDGET_ERROR_CLASS = "AttemptBudgetExceeded"


@dataclass(kw_only=True)
class SermonRecord:
    """A single sermon and its embedded run history (PRD §6.2).

    Field order matches PRD §6.2 so serialization stays diff-stable.
    """

    guid: str
    source: str
    episode_url: str
    audio_url: str
    published_on: str
    #: Full publication instant, ISO-8601, or ``None`` when the source carries no
    #: usable signal (ADR-0056). Menlo/North Point read it from a trustworthy feed
    #: ``pubDate``; PBC — whose ``pubDate`` is a constant nominal value — reads it from
    #: the audio enclosure's CDN ``Last-Modified`` instead. Makes realised discovery
    #: time (``first_seen_at`` minus this) computable from the ledger, closing the
    #: feedback loop ADR-0048 flagged as missing.
    published_at: str | None = None
    title: str
    series: str | None
    speaker: str | None
    scripture_refs: list[str]
    blurb: str
    transcript_hash: str | None
    state: str
    #: Anthropic Batches API id while ``state`` is ``pending_batch``; ``None``
    #: otherwise (spec 0020, ADR-0061). Set on submission, read back on poll.
    batch_id: str | None = None
    #: ISO-8601 instant the batch above was submitted; the 24h fallback compares
    #: against this, not against ``last_state_change_at``, so a later legal
    #: transition through ``pending_batch`` (there is only ever one) can't reset the
    #: clock a poll depends on.
    batch_submitted_at: str | None = None
    artifact_path: str | None
    #: The id Discord assigned this sermon's note post, or ``None`` when nothing was
    #: ever posted (spec 0017 amendment, ADR-0062). ``None`` is the ordinary value for
    #: a source with no channel and for anything published before its channel shipped;
    #: it is not an error state. Captured at the publish transition and never
    #: recomputed — a webhook credential cannot read channel history, so an id lost
    #: here cannot be recovered from Discord.
    discord_message_id: str | None = None
    #: The posted message id for each *named* channel this sermon has been sent to,
    #: keyed by the channel's `name` in `NOTIFY_CHANNELS_JSON` (spec 0024 amendment,
    #: ADR-0067 amendment). Additive alongside `discord_message_id` above, which stays
    #: Discord's single legacy field — this generalizes id capture to any number of
    #: channels of either kind. A channel with no configured `name` is never a key here;
    #: there is no other stable, non-secret value to key on.
    channel_message_ids: dict[str, str] = field(default_factory=dict)
    #: This record's own audio-content fingerprint (ADR-0077, #517), set the first
    #: time its enclosure is downloaded and fingerprinted — whether or not a match
    #: was found — so a later record has something to compare against without
    #: re-downloading this one. ``None`` until then.
    audio_fingerprint: str | None = None
    #: The guid of another record this one's audio fingerprint matched (ADR-0077,
    #: #517), or ``None``. Set only when a match was found; the plan withholds a
    #: ``discovered`` record with this set, exactly like ADR-0066's missing-enclosure
    #: case, so a confirmed duplicate is not re-sharded and re-downloaded every poll.
    #: Cleared by :meth:`Registry.upsert` when the feed's ``audio_url`` for this guid
    #: actually changes, re-admitting the record for a fresh check.
    suspected_duplicate_of: str | None = None
    first_seen_at: str
    last_state_change_at: str
    runs: list[RunRecord] = field(default_factory=list)


@dataclass(kw_only=True)
class NotifiedAlert:
    """When a condition-level alert last fired, and what condition it fired for.

    One entry per alert class in :attr:`Registry` — a run-level alert (e.g.
    ``AttemptCapUnenforced``) has no per-sermon record to hang a "already told you
    this" flag on, so the ledger carries a small map keyed by alert class instead
    (ADR-0046, #324). ``fingerprint`` is the alert's own summary of the condition's
    current severity (e.g. a missing-count string, or a constant for a binary
    alert) — a change here means the condition changed shape, not just persisted.
    """

    last_sent_at: str
    fingerprint: str


def _now() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def parse_instant(value: str) -> datetime:
    """Parse a ledger timestamp field, coercing a naive result to UTC.

    Every writer in this module stamps an aware value via :func:`_now`, so a naive
    string can only reach the ledger through a hand-edit, a migration script (the
    ADR-0016 exemption), or a restored backup. Without this, comparing it against an
    aware ``datetime`` raises ``TypeError`` deep inside a caller (#274) — the run-level
    spend guard and the dead-man's-switch staleness check both do exactly that.
    Mirrors :func:`~sermon_notes.sources.feedbase.parse_pubdate_at`, which makes the
    same assumption for feed-sourced dates so the value is always comparable.

    Use it for *every* timestamp string that meets an aware ``datetime``, including an
    injected ``now``. Normalizing only the stored side leaves the same ``TypeError``
    one argument away: both operands have to be aware for the comparison to hold, so a
    guard that coerces one of them is not yet total.
    """
    when = datetime.fromisoformat(value)
    return when if when.tzinfo is not None else when.replace(tzinfo=timezone.utc)


def _sermon_from_dict(data: dict[str, Any]) -> SermonRecord:
    """Build a :class:`SermonRecord` from a plain dict, converting nested runs."""
    runs = [RunRecord(**run) for run in data.get("runs", [])]
    return SermonRecord(**{**data, "runs": runs})


class LedgerValidationError(RuntimeError):
    """Raised when a ledger file is not one this module could load back (ADR-0044)."""


def validate_ledger(path: Path = DEFAULT_REGISTRY_PATH) -> int:
    """Check the ledger at ``path`` is sound, and return how many sermons it holds.

    The pipeline's artifact commit goes straight to the default branch over a deploy
    key, so it never passes through a pull request and ``ci.yml`` sees it only on
    ``push``, after it has landed. The one gate that costs us is this module's own
    suite, which strict-loads the committed ledger on every PR: the ledger is the
    pipeline's only persistent state and the merge job is its sole writer, so a shape
    :meth:`Registry.load` rejects would land and every later run would then fail on
    load with the bad ledger already published.

    Lives here rather than in the caller because CLAUDE.md §6 puts every read of the
    ledger file behind this module. Raises :class:`LedgerValidationError` for the
    checks it makes itself, and lets the loader's own ``TypeError``/``ValueError``
    escape unchanged — the caller reports both the same way.
    """
    if not path.exists():
        # `load` treats an absent file as an empty ledger by design, so the strict load
        # below cannot catch this. The ledger is append-only (CLAUDE.md §6), which makes
        # a vanished one a corruption rather than a state worth publishing.
        raise LedgerValidationError(f"the ledger is missing at {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        # Valid JSON of the wrong shape. `save` can only ever write an object, but the
        # ADR-0016 migration exemption lets a script write this file directly, and a
        # shape validator that lets a malformed shape through as `AttributeError` is
        # reporting a bug in itself rather than in the ledger.
        raise LedgerValidationError(f"the ledger at {path} is not a JSON object")
    sermons = data.get("sermons", [])
    for sermon in sermons:
        guid = sermon.get("guid", "<no guid>")
        for run in sermon.get("runs", []):
            if "llm_model" not in run:
                # What the rolling spend guard prices a window by (#241, #256). A run
                # without it loads fine but is invisible to the caps, so the ledger stays
                # readable while the guard quietly under-counts.
                raise LedgerValidationError(f"a run on {guid} carries no llm_model")
            if "llm_billing" not in run:
                # What answers "did batch actually work?" (ADR-0082, #543). A run without
                # it loads fine but is silently unattributable to a rate.
                raise LedgerValidationError(f"a run on {guid} carries no llm_billing")
            if "llm_batch_id" not in run:
                # The only surviving evidence of which batch produced a run once
                # SermonRecord.batch_id is cleared on resolution (ADR-0082, #543).
                raise LedgerValidationError(f"a run on {guid} carries no llm_batch_id")
            if "quote_check_passed" in run:
                raise LedgerValidationError(f"a run on {guid} carries a retired field")

    Registry.load(path)
    return len(sermons)


class Registry:
    """In-memory view of the ledger, owning all reads and writes of its file."""

    def __init__(self, path: Path = DEFAULT_REGISTRY_PATH) -> None:
        self.path = path
        self._sermons: list[SermonRecord] = []
        self._notified_alerts: dict[str, NotifiedAlert] = {}

    @classmethod
    def load(cls, path: Path = DEFAULT_REGISTRY_PATH) -> Registry:
        """Load the ledger at ``path``; an absent or empty file yields no sermons."""
        registry = cls(path)
        if path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            registry._sermons = [_sermon_from_dict(s) for s in data.get("sermons", [])]
            registry._notified_alerts = {
                alert_class: NotifiedAlert(**entry)
                for alert_class, entry in data.get("notified_alerts", {}).items()
            }
        return registry

    def save(self) -> None:
        """Write the ledger atomically as human-readable, diff-stable JSON.

        Cleans up its ``.tmp`` sibling on an in-process exception rather than
        leaving it for the next ``git add state/registry.json`` to pick up (#272).
        """
        payload = {
            "sermons": [asdict(s) for s in self._sermons],
            "notified_alerts": {
                alert_class: asdict(entry) for alert_class, entry in self._notified_alerts.items()
            },
        }
        text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, self.path)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

    def sermons(self) -> list[SermonRecord]:
        """Return all sermon records in ledger order."""
        return self._sermons

    def get(self, guid: str) -> SermonRecord | None:
        """Return the sermon with ``guid``, or ``None`` if absent."""
        return next((s for s in self._sermons if s.guid == guid), None)

    def _require(self, guid: str) -> SermonRecord:
        sermon = self.get(guid)
        if sermon is None:
            raise UnknownSermonError(f"no sermon with guid {guid!r} in the ledger")
        return sermon

    def upsert(self, sermon: SermonRecord) -> SermonRecord:
        """Insert a new sermon, or refresh feed metadata on an existing guid.

        On an existing guid only the feed-sourced fields are updated; ``state``,
        ``transcript_hash``, ``artifact_path``, ``runs`` and the timestamps are
        preserved so re-discovery never clobbers pipeline progress. Once the record
        is ``published``, ``title`` also stops refreshing (issue #183) — the artifact
        is already rendered with the title as of publication, and a feed rename
        afterward must not desync the ledger from it.

        A refreshed ``audio_url`` that actually changes also clears
        ``audio_fingerprint``/``suspected_duplicate_of`` (ADR-0077, #517): both were a
        verdict about the *previous* enclosure, and the record needs a fresh check
        against whatever the feed is serving now, not a stale one carried forward.
        """
        if not sermon.guid:
            raise InvalidGuidError("cannot upsert a sermon with an empty guid")
        existing = self.get(sermon.guid)
        if existing is None:
            self._sermons.append(sermon)
            return sermon
        frozen = _PUBLISHED_FROZEN_FIELDS if existing.state == "published" else frozenset()
        previous_audio_url = existing.audio_url
        for name in _METADATA_FIELDS:
            if name in frozen:
                continue
            setattr(existing, name, getattr(sermon, name))
        if existing.audio_url != previous_audio_url:
            existing.audio_fingerprint = None
            existing.suspected_duplicate_of = None
        return existing

    def replace(self, sermon: SermonRecord) -> SermonRecord:
        """Insert ``sermon``, or wholesale-overwrite an existing guid's record with it.

        The deliberate escape hatch :meth:`upsert` refuses to be: that method exists to
        protect pipeline progress from ordinary feed re-discovery, so it preserves
        ``state``, ``transcript_hash``, ``artifact_path`` and ``runs`` on an existing
        guid no matter what the caller passes. That protection is exactly wrong for
        ``scripts/replay_unpushed_artifacts.py`` (ADR-0045), whose caller has already
        decided — by comparing ``last_state_change_at`` — that ``sermon`` is the more
        advanced record and the committed one is stale. Calling :meth:`upsert` there
        silently keeps the committed record's lifecycle fields, so a sermon a lost push
        had carried all the way to ``published`` reverts to whatever partial state
        ``main`` last saw, while its rendered artifact lands on disk unreferenced. This
        method is for that one caller: it replaces the full record, not just its
        metadata.
        """
        if not sermon.guid:
            raise InvalidGuidError("cannot replace a sermon with an empty guid")
        for index, existing in enumerate(self._sermons):
            if existing.guid == sermon.guid:
                self._sermons[index] = sermon
                return sermon
        self._sermons.append(sermon)
        return sermon

    def advance(self, guid: str, to_state: str, *, now: str | None = None) -> SermonRecord:
        """Move a sermon to ``to_state`` if the transition is legal (PRD §6.3)."""
        sermon = self._require(guid)
        if to_state not in _TRANSITIONS.get(sermon.state, frozenset()):
            raise IllegalTransitionError(
                f"illegal transition {sermon.state!r} → {to_state!r} for guid {guid!r}"
            )
        sermon.state = to_state
        sermon.last_state_change_at = now if now is not None else _now()
        return sermon

    def recover(self, guid: str, to_state: str, *, now: str | None = None) -> SermonRecord:
        """Reset a stranded ``failed`` record to an earlier state for re-attempt (ADR-0014).

        The sole sanctioned exit from the otherwise-terminal ``failed`` sink: only a
        record currently in ``failed`` may be recovered, and only to a non-terminal
        lifecycle state (``discovered``/``transcribed``/``generated``) it can resume
        from. Kept distinct from :meth:`advance` — which keeps ``failed`` terminal — so
        recovery is an explicit, auditable operation, not a loosening of the normal
        state machine. Only ``state`` and ``last_state_change_at`` change; the
        append-only run history (including the terminal failure) is preserved as the
        audit trail.
        """
        sermon = self._require(guid)
        if sermon.state != "failed":
            raise IllegalTransitionError(
                f"recover requires state 'failed', not {sermon.state!r} for guid {guid!r}"
            )
        if to_state not in _RECOVERABLE_STATES:
            raise IllegalTransitionError(
                f"cannot recover guid {guid!r} to terminal/unknown state {to_state!r}"
            )
        sermon.state = to_state
        sermon.last_state_change_at = now if now is not None else _now()
        return sermon

    def record_discord_message(self, guid: str, message_id: str) -> None:
        """Record the id of the Discord message carrying this sermon's note (ADR-0062).

        Not a lifecycle event: ``published`` stays terminal and the run history is
        untouched: this records what a delivery already did, rather than advancing the
        sermon. Callers must :meth:`save` afterwards, which is the half that matters —
        both run paths save *before* the Discord step, so an unsaved id dies with the
        runner (CLAUDE.md §10).
        """
        self._require(guid).discord_message_id = message_id

    def record_channel_message(self, guid: str, channel_name: str, message_id: str) -> None:
        """Record the id of a posted message under its channel's ``name`` (spec 0024
        amendment, ADR-0067 amendment).

        Generalizes :meth:`record_discord_message` to any number of channels of either
        kind — each channel keeps its own key, so one channel's id never overwrites
        another's for the same sermon. Not a lifecycle event, same posture as
        :meth:`record_discord_message`: callers must :meth:`save` afterwards.
        """
        self._require(guid).channel_message_ids[channel_name] = message_id

    def append_run(self, guid: str, run: RunRecord) -> None:
        """Append a run record to a sermon's history, preserving prior runs."""
        self._require(guid).runs.append(run)

    def has_successful_generation(
        self, transcript_hash: str, *, exclude_guid: str | None = None
    ) -> bool:
        """Whether a sermon with this hash already generated successfully (PRD §11.3).

        The orchestrator uses this to skip a second LLM call for an unchanged
        transcript. ``exclude_guid`` omits one record from the scan, for callers asking
        the *cross-guid* question the duplicate guard was written for (#86): without it
        a record that already generated answers ``True`` about its own transcript and
        is treated as a duplicate of itself (#196).
        """
        return self.find_generated_duplicate(transcript_hash, exclude_guid=exclude_guid) is not None

    def find_generated_duplicate(
        self, transcript_hash: str, *, exclude_guid: str | None = None
    ) -> str | None:
        """The guid of another sermon that already generated from this hash, if any.

        Same cross-guid scan as :meth:`has_successful_generation`, but hands back
        *which* record it collided with instead of only whether one exists — the
        escalation the post-transcription dedup guard now sends (#517 child 2) names
        both sermons, not just the one the guard sent terminal.
        """
        return next(
            (
                s.guid
                for s in self._sermons
                if s.guid != exclude_guid
                and s.transcript_hash == transcript_hash
                and any(r.attempted_state == "generated" and r.outcome == "success" for r in s.runs)
            ),
            None,
        )

    def find_duplicate_audio(
        self,
        source: str,
        fingerprint: str,
        *,
        exclude_guid: str,
        threshold: float = audio_fingerprint.DEFAULT_SIMILARITY_THRESHOLD,
    ) -> str | None:
        """The guid of another of ``source``'s records whose own audio matches ``fingerprint``.

        Detects a duplicate *before* transcription (ADR-0077, #517), pre-empting the
        cross-guid guard :meth:`has_successful_generation` runs after it. Only compares
        against a **confirmed-distinct** record — one with its own stored
        ``audio_fingerprint`` and no ``suspected_duplicate_of`` of its own — so a false
        positive can't propagate: a record already flagged as someone else's duplicate
        never becomes a candidate a third record matches against. Returns ``None`` when
        nothing matches, including when ``source`` has no fingerprinted records yet.
        """
        for s in self._sermons:
            if (
                s.guid != exclude_guid
                and s.source == source
                and s.audio_fingerprint is not None
                and s.suspected_duplicate_of is None
                and audio_fingerprint.is_same_audio(
                    s.audio_fingerprint, fingerprint, threshold=threshold
                )
            ):
                return s.guid
        return None

    def should_escalate_alert(
        self, alert_class: str, fingerprint: str, *, cooldown_hours: int, now: str | None = None
    ) -> bool:
        """Whether a condition-level alert should send now, recording that it did.

        Suppresses only when the *same* condition (``fingerprint`` unchanged) was
        already reported within ``cooldown_hours`` — a worsening condition (a
        changed fingerprint) or an elapsed cooldown always sends, and records the
        new fingerprint/timestamp as the most recent notification (ADR-0046,
        #324).
        """
        when = now if now is not None else _now()
        previous = self._notified_alerts.get(alert_class)
        if previous is not None and previous.fingerprint == fingerprint:
            elapsed = parse_instant(when) - parse_instant(previous.last_sent_at)
            if elapsed < timedelta(hours=cooldown_hours):
                return False
        self._notified_alerts[alert_class] = NotifiedAlert(
            last_sent_at=when, fingerprint=fingerprint
        )
        return True
