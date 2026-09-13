"""The shard→merge delta: the state one parallel shard hands back to the serial merge.

Parallel backfill (ADR-0023) splits the ``run`` pipeline into a preflight ``plan``, a
per-sermon ``shard`` fan-out, and a single serial ``merge`` fan-in. A shard transcribes
and generates one sermon into deterministic files (``transcripts/…txt`` +
``notes/…json``) but **never opens the registry** — the merge is the sole registry
writer. This module carries the one thing the files cannot: the per-run telemetry and
lifecycle outcome the shard produced, so the merge can advance the ledger exactly as
the serial pipeline would (ADR-0010 keeps token/cost/model on the run record).

A :class:`Delta` is a transient inter-job artifact, not persistent schema (the
committed ledger's record shape is unchanged, so the ADR-0002 migration gate does
not fire). It snapshots the discovered record the shard processed, the resulting
``transcript_hash``, the state it reached, and the run records it appended.
:func:`apply_delta` replays that onto the merge's single loaded registry through the
same :class:`~sermon_notes.registry.Registry` calls the serial stages use — including
the cross-registry ``has_successful_generation`` dedup guard, which can only be
evaluated in the merge where the whole ledger is present (PRD §11.3, #86).
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from sermon_notes.registry import Registry, RunRecord, SermonRecord

# (attempted_state, outcome) → the lifecycle state that outcome advances a sermon to.
# A ``retry_scheduled`` transcribe leaves the sermon ``discovered`` (no entry), matching
# transcribe._go_recoverable (ADR-0009). A ``billed`` generate has no entry either: it is
# the cost of a paid call recorded before the attempt concluded (ADR-0035), so it lands in
# the ledger as telemetry while the sermon's state comes from the run that concluded it.
# Mirrors the serial stages' transitions so a replayed delta lands a sermon in the same
# state a serial run would.
# A ``pending_batch`` success is a *submission*, not a completion: the money is not spent
# and no note exists yet, so it advances the sermon onto ADR-0061's detour rather than to
# ``generated``. The run that eventually resolves the batch reports ``("generated",
# "success")`` like any other generation, whether it came back from the Batches API or
# from the 24h synchronous fallback — the ledger cannot tell those apart and does not
# need to (spec 0020).
_ADVANCE_ON: dict[tuple[str, str], str] = {
    ("transcribed", "success"): "transcribed",
    ("transcribed", "failed_terminal"): "failed",
    ("pending_batch", "success"): "pending_batch",
    ("generated", "success"): "generated",
    ("generated", "failed_terminal"): "failed",
}

# Public: the merge (pipeline._merge_escalation, #517 child 2) checks a terminal
# failure's error class against this to build the two-sermon collision escalation
# instead of the generic one-sermon terminal-failure email.
DUPLICATE_ERROR_CLASS = "DuplicateTranscript"


def _now() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _terminal_of(runs: list[RunRecord]) -> tuple[str, str] | None:
    """The ``(error_class, error_detail)`` of the last terminal failure in ``runs``."""
    terminal: tuple[str, str] | None = None
    for run in runs:
        if run.outcome == "failed_terminal":
            terminal = (run.error_class or "Unknown", run.error_detail or "")
    return terminal


@dataclass(frozen=True)
class Delta:
    """One shard's result: the sermon it processed, its outcome, and its run records.

    ``sermon`` is the discovered/transcribed record the shard was handed (its metadata
    seeds the merge's ledger via idempotent upsert); ``reached_state`` and ``runs`` are
    what the shard's transcribe+generate produced. Transient — serialized to a per-shard
    artifact, applied once by the merge, never committed.

    ``cap_unenforced`` carries why this shard spent without the attempt cap in force, or
    ``None`` when it was in force (ADR-0036). The cap fails open on an unreadable claim
    listing — a guard's dependency must not halt the pipeline — and the shard has no way
    to alert on its own, so the reason rides back to the merge, which escalates it once
    for the run. It is *not* set for the dormant off-Actions case, which is documented
    behaviour rather than a fault.

    ``batch_id``/``batch_submitted_at`` are set only by a shard that submitted a backfill
    generation to the Anthropic Batches API (spec 0020, ADR-0061). They are the one piece
    of the record a shard changes that its run records cannot express, and the merge — the
    sole registry writer — is where they become durable.

    ``audio_fingerprint``/``suspected_duplicate_of`` are the shard's own audio-content
    fingerprint and, when it matched a sibling record, that record's guid (ADR-0077,
    #517). Carried the same way as ``batch_id`` above and for the same reason: a shard
    mutates its own throwaway record's fields directly, which ``reached_state``/``runs``
    cannot express, so the merge needs them spelled out to make the mutation durable.
    """

    sermon: SermonRecord
    transcript_hash: str | None
    reached_state: str
    runs: list[RunRecord] = field(default_factory=list)
    cap_unenforced: str | None = None
    batch_id: str | None = None
    batch_submitted_at: str | None = None
    audio_fingerprint: str | None = None
    suspected_duplicate_of: str | None = None

    def to_json(self) -> str:
        """Serialize to deterministic JSON (sorted keys) for the shard artifact."""
        payload = {
            "sermon": asdict(self.sermon),
            "transcript_hash": self.transcript_hash,
            "reached_state": self.reached_state,
            "runs": [asdict(run) for run in self.runs],
            "cap_unenforced": self.cap_unenforced,
            "batch_id": self.batch_id,
            "batch_submitted_at": self.batch_submitted_at,
            "audio_fingerprint": self.audio_fingerprint,
            "suspected_duplicate_of": self.suspected_duplicate_of,
        }
        return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> Delta:
        """Rebuild a :class:`Delta` from :meth:`to_json` output."""
        data = json.loads(text)
        sermon_data = dict(data["sermon"])
        sermon = SermonRecord(
            **{**sermon_data, "runs": [RunRecord(**r) for r in sermon_data.get("runs", [])]}
        )
        runs = [RunRecord(**r) for r in data.get("runs", [])]
        return cls(
            sermon=sermon,
            transcript_hash=data.get("transcript_hash"),
            reached_state=data["reached_state"],
            runs=runs,
            # `.get`, not `[...]`: a merge re-run applies deltas an earlier shard wrote
            # (#196, #230), which on the run that ships this field predate it.
            cap_unenforced=data.get("cap_unenforced"),
            # `.get` for the same reason as the line above: a re-run applies deltas an
            # earlier shard wrote, which on the run that ships these fields predate them.
            batch_id=data.get("batch_id"),
            batch_submitted_at=data.get("batch_submitted_at"),
            # `.get` for the same reason: a re-run may replay a delta written before #517.
            audio_fingerprint=data.get("audio_fingerprint"),
            suspected_duplicate_of=data.get("suspected_duplicate_of"),
        )


@dataclass(frozen=True)
class AppliedDelta:
    """What :func:`apply_delta` did, so the merge can escalate and tidy files.

    ``terminal`` is ``(error_class, error_detail)`` when the applied outcome sent the
    sermon to ``failed`` (the merge escalates it once, PRD §11.2); ``deduped`` marks the
    cross-guid duplicate case, whose shard-written note the merge discards (#86).
    """

    guid: str
    deduped: bool = False
    terminal: tuple[str, str] | None = None


def _already_applied(record: SermonRecord, delta: Delta) -> bool:
    """Whether the ledger already carries this delta's effects (#196, #230).

    Replay is routine, not exotic: the workflow deliberately commits partial progress
    when the merge fails (#125) while the shard artifacts survive, so "Re-run failed
    jobs" hands the merge deltas it has already applied. Re-applying one is destructive —
    the advance would be an illegal self-transition, or the dedup guard would mistake the
    record's own prior generation for a duplicate and delete the note JSON that is the
    whole point of spec 0009.

    Two signals, because a delta need not have moved the record at all:

    - **State.** A shard starts from the record's committed state and reports only the
      runs that advance it *from there* (``run_shard`` seeds a clean history), so the
      ledger sitting at any other state means an earlier merge already applied this delta.
    - **Runs.** An exhausted download leaves the sermon ``discovered`` and records a
      ``retry_scheduled`` run (ADR-0009), so the state is unchanged and only the history
      shows the delta landed. A replayed delta carries byte-identical run records — the
      shard's timestamps were fixed when it ran — while two genuinely distinct attempts
      always differ in ``started_at``/``finished_at``, so value equality separates them
      and a real second attempt still appends (#230).
    """
    if record.state != delta.sermon.state:
        return True
    return bool(delta.runs) and all(run in record.runs for run in delta.runs)


def apply_delta(delta: Delta, registry: Registry, *, now: str | None = None) -> AppliedDelta:
    """Replay one shard's ``delta`` onto the merge's loaded ``registry`` (ADR-0023).

    Seeds the sermon's metadata (idempotent upsert), then applies the shard's runs
    through the same :class:`Registry` transitions the serial stages use, so the ledger
    integrity, transition enforcement, and telemetry are identical — only relocated to
    the single serial writer. The cross-registry ``has_successful_generation`` dedup
    guard is evaluated here (the shard's one-record view could not see it): a transcript
    that already generated elsewhere sends this sermon terminal ``DuplicateTranscript``
    and escalates once, discarding the shard's work, exactly as the serial transcribe
    stage would (PRD §11.3, #86).

    Applying a delta is idempotent: the merge job is re-runnable, so a delta already in
    the committed ledger is recognized and skipped (see :func:`_already_applied`).
    """
    registry.upsert(copy.deepcopy(delta.sermon))
    guid = delta.sermon.guid
    at = now if now is not None else _now()

    record = registry.get(guid)
    assert record is not None  # just upserted
    if _already_applied(record, delta):
        # Report the terminal again — a merge that died before escalating never mailed
        # it — but never `deduped`, which would discard a note this record owns (#196).
        return AppliedDelta(guid=guid, terminal=_terminal_of(delta.runs))

    # Dedup first, before applying any run — mirrors transcribe._go_duplicate, which
    # sends a duplicate terminal *before* the transcribed advance, so a deduped record
    # carries only the DuplicateTranscript run (no transcribed/generated success run).
    # The record's own runs are excluded: the guard asks whether *another* sermon
    # already generated this transcript (#86), not whether this one did (#196).
    if delta.audio_fingerprint is not None:
        record.audio_fingerprint = delta.audio_fingerprint
    if delta.suspected_duplicate_of is not None:
        record.suspected_duplicate_of = delta.suspected_duplicate_of

    if delta.transcript_hash is not None and registry.has_successful_generation(
        delta.transcript_hash, exclude_guid=guid
    ):
        record.transcript_hash = delta.transcript_hash
        detail = (
            f"transcript identical to an already-generated sermon (hash {delta.transcript_hash})"
        )
        registry.advance(guid, "failed", now=now)
        registry.append_run(
            guid,
            RunRecord(
                attempted_state="transcribed",
                outcome="failed_terminal",
                error_class=DUPLICATE_ERROR_CLASS,
                error_detail=detail,
                started_at=at,
                finished_at=at,
            ),
        )
        return AppliedDelta(guid=guid, deduped=True, terminal=(DUPLICATE_ERROR_CLASS, detail))

    if delta.transcript_hash is not None:
        record.transcript_hash = delta.transcript_hash

    if delta.batch_id is not None:
        record.batch_id = delta.batch_id
        record.batch_submitted_at = delta.batch_submitted_at

    for run in delta.runs:
        target = _ADVANCE_ON.get((run.attempted_state, run.outcome))
        if target is not None:
            registry.advance(guid, target, now=now)
        registry.append_run(guid, run)

    if record.state != "pending_batch":
        # PRD §6.2 defines both fields as set only while the record is on ADR-0061's
        # detour, so the state and the fields are kept true to each other here rather
        # than at each of the several places a sermon can leave it (resolved, fallen
        # back, or failed). A record that never entered it clears two `None`s.
        record.batch_id = None
        record.batch_submitted_at = None

    return AppliedDelta(guid=guid, terminal=_terminal_of(delta.runs))
