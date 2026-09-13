"""The orchestrator: drive every sermon through the state machine each run (U10).

This is the integration unit (spec 0007, PRD §6.3/§6.4/§11). It composes the stage
modules — source poll, transcription, generation, docx render, README index, and the
delivery boundaries — into one pass that advances each sermon as far as it can:
POLL/SEED → TRANSCRIBE → GENERATE → RENDER/COMMIT → LOG & ESCALATE. Each stage
module already owns its own retry posture (download ×3, transcription ×2, LLM ×3,
render ×1); the orchestrator owns the five-most-recent seeding of the initial run
(PRD §6.6), idempotency (skip ``published``; ``transcript_hash`` guards
regeneration), and the single escalation email per terminal failure (PRD §11.2).
The mechanical quote gate is retired (ADR-0008).

The git commit/push that publishes the working tree is the scheduled workflow's job
(U11), not a pipeline boundary — this module's responsibility ends at writing the
``.docx``, refreshing the README index, and persisting the ledger.

After a run produces newly ``published`` sermons, a run-level publish step renders the
secret-free content feed (spec 0014) and ships it across the two web-publishing
boundaries — push to ``sermon-notes-content`` then the Vercel deploy hook (ADR-0018,
U4) — under the PRD §11.2 escalation contract. It runs only when those boundaries are
configured, so the pipeline stays dark until the content repo and secrets are
provisioned (U5/U8).

Each newly published note is then delivered: by email for every source (spec 0011), and
additionally to whichever Discord and Google Chat channels its church has configured in the
``NOTIFY_CHANNELS_JSON`` routing table (spec 0024, ADR-0067; :mod:`notify_channels` owns the
lookup). Every channel is best-effort and config-gated — a dropped delivery, or a church with
no configured channel, is logged at most, never fatal.
"""

from __future__ import annotations

import copy
import functools
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from textwrap import indent
from typing import Callable, Sequence

from sermon_notes import (
    attempt_claims,
    audio,
    church_config,
    config,
    content_publish,
    deploy_hook,
    discord_notify,
    feed,
    google_chat_notify,
    llm_client,
    notify,
    notify_channels,
    sources,
    workflow_runs,
)
from sermon_notes.artifacts import note_json_path, read_note_json
from sermon_notes.generate import (
    GenerationError,
    SchemaValidationError,
    generate_notes,
    submit_note_batch,
)
from sermon_notes.llm_client import (
    BatchRequest,
    BatchResult,
    BatchStatus,
    LLMError,
    LLMResponse,
    PermanentLLMError,
    TransientLLMError,
)
from sermon_notes.logging import get_logger
from sermon_notes.readme_index import DEFAULT_README_PATH, update_readme
from sermon_notes.registry import (
    DEFAULT_REGISTRY_PATH,
    Registry,
    ATTEMPT_BUDGET_ERROR_CLASS,
    RegistryError,
    RunRecord,
    SermonRecord,
    parse_instant,
)
from sermon_notes.render import DEFAULT_NOTES_DIR, RenderError, render_note
from sermon_notes.render_pdf import render_note_pdf
from sermon_notes.shard import DUPLICATE_ERROR_CLASS, Delta, apply_delta
from sermon_notes.sources import PollResult, SourceAdapter
from sermon_notes.transcribe import (
    DEFAULT_TRANSCRIPTS_DIR,
    cache_path,
    default_transcribe,
    transcribe_sermon,
)

logger = get_logger()

_DEFAULT_INITIAL_LIMIT = 5

# Fallback for SERMON_NOTES_WEBSITE_BASE_URL (#472): the website's production domain,
# so a channel's note link works out of the box without an operator setting anything.
_DEFAULT_WEBSITE_BASE_URL = "https://www.sermon-note.online/"


def note_url_for(sermon: SermonRecord) -> str:
    """The website's page for ``sermon``'s note: ``{base_url}/{source}/{slug}``.

    Shared by every channel that links the note itself rather than (or in addition to)
    attaching it — Google Chat (spec 0024 amendment, #472), Discord (spec 0017
    amendment, #574), and the ``discord-test`` manual verification command — so all
    three derive the same URL for the same sermon rather than risking drift between
    separate implementations. Built from the same slug the content feed's own manifest
    carries (:func:`feed.sermon_slug`).
    """
    base_url = config.get("SERMON_NOTES_WEBSITE_BASE_URL", _DEFAULT_WEBSITE_BASE_URL)
    return f"{base_url.rstrip('/')}/{sermon.source}/{feed.sermon_slug(sermon)}"


@dataclass(frozen=True)
class PipelineResult:
    """Summary of one pipeline run, for the CLI and the tests."""

    deferred: bool = False
    discovered: tuple[str, ...] = ()
    transcribed: tuple[str, ...] = ()
    generated: tuple[str, ...] = ()
    published: tuple[str, ...] = ()
    failed: tuple[str, ...] = ()
    escalated: int = 0
    delivered: int = 0
    discord_delivered: int = 0
    google_chat_delivered: int = 0
    feed_published: bool = False
    deploy_triggered: bool = False
    # The plan refused to fan out on a rolling spend cap and the merge escalated it
    # (ADR-0037). Kept off `failed`, which is a tuple of guids: no sermon failed — none
    # was planned. It flips the run red on its own, so the refusal keeps the red Actions
    # run it always had as a second channel behind the best-effort email.
    spend_refused: bool = False
    # The plan refused discovery because every enabled source resolved to zero usable
    # feeds, and the merge escalated it (ADR-0074, #496). Same reasoning as
    # `spend_refused` — no sermon failed, so it stays off `failed`, and it flips the
    # run red on its own as a second channel behind the best-effort email.
    sources_refused: bool = False
    # How many planned shard deltas never reached the fan-in (#266). Non-zero only where a
    # shard job failed — an unexplained shortfall refuses the merge before it runs — so it
    # deliberately does not flip the run red on its own: that path is already red from the
    # failed shard job, and the sermons behind the missing deltas did not fail, they were
    # never processed. Kept off `failed` for the same reason the two alerts above are: it
    # is a tuple of guids, and the merge never learns which sermons went missing.
    delta_shortfall: int = 0


@dataclass(frozen=True)
class ReconcileResult:
    """Summary of one reconciliation pass (PRD §6.5)."""

    deferred: bool = False
    discovered: tuple[str, ...] = ()
    flagged: tuple[str, ...] = ()
    missing_artifacts: tuple[str, ...] = ()
    recovered: tuple[str, ...] = ()
    escalated: int = 0
    # The dead-man's switch (#242): how many days since the last note published, when
    # that exceeds the threshold; ``None`` when it does not, or when nothing ever has.
    publish_stale_days: int | None = None
    # The staleness alert was tripped and its send was refused. Fatal on its own terms:
    # an alert of last resort that silently fails to deliver is the outage twice over.
    publish_stale_alert_failed: bool = False
    # The tighter dead-man's switch (#283): how many trailing completed pipeline.yml
    # runs on main were red, when that meets or exceeds RED_RUN_STREAK_COUNT; ``None``
    # otherwise — not tripped, not enough run history yet, or the run listing was
    # dormant/unreadable this pass (fails open, like the attempt cap).
    red_streak_count: int | None = None
    # The red-streak alert was tripped and its send was refused — fatal on its own
    # terms, same as publish_stale_alert_failed.
    red_streak_alert_failed: bool = False
    # The third switch (ADR-0066): guids discovered but still without an audio
    # enclosure past MISSING_ENCLOSURE_DAYS. Deliberately NOT part of the exit code —
    # the church has not uploaded audio, which is not a pipeline failure, and a red run
    # here would feed the red-run-streak switch above until it paged about a healthy
    # pipeline. The email is the whole signal.
    awaiting_enclosure: tuple[str, ...] = ()
    # The fourth switch (ADR-0077, #517): guids withheld as a suspected duplicate of
    # another record's audio past SUSPECTED_DUPLICATE_AUDIO_DAYS. Not part of the exit
    # code for the same reason as awaiting_enclosure above — this is either a correct,
    # permanent verdict or a false positive, and either way it is not a run failure;
    # the email is what surfaces it for a human to confirm or override.
    suspected_duplicate_audio: tuple[str, ...] = ()


# Failure classes the current pipeline can no longer raise (ADR-0014). A record
# stranded ``failed`` by one of these would now succeed, so reconciliation resets it
# to re-attempt. Seeded with the retired quote gate (ADR-0008); add a class here when
# its check is retired from the pipeline.
#
# The test is "the pipeline cannot raise this any more", not "we would like this record
# re-attempted". A live class listed here is re-attempted nightly and fails again
# forever; :data:`ATTEMPT_BUDGET_ERROR_CLASS` is the one that would cost real money
# doing it, which is why it is deliberately absent (#240).
RETIRED_FAILURE_CLASSES = frozenset({"QuoteGateError"})

# The lifecycle state a sermon resumes from after a terminal failure at each step:
# the predecessor in the linear chain (PRD §6.3).
_RECOVERY_TARGET = {
    "transcribed": "discovered",
    "generated": "transcribed",
    "published": "generated",
}


def _stranded_by_retired_logic(sermon: SermonRecord) -> str | None:
    """The state to resume a ``failed`` record from if a retired class stranded it (ADR-0014).

    Returns the recovery target when ``sermon``'s last run is a terminal failure whose
    ``error_class`` the current pipeline can no longer raise; otherwise ``None`` — so a
    record that failed for a still-live reason (or for any other state) is left alone.
    """
    if sermon.state != "failed" or not sermon.runs:
        return None
    last = sermon.runs[-1]
    if last.outcome != "failed_terminal" or last.error_class not in RETIRED_FAILURE_CLASSES:
        return None
    return _RECOVERY_TARGET.get(last.attempted_state)


@dataclass(frozen=True)
class _Escalation:
    """A queued terminal failure awaiting its single escalation email (PRD §11.2)."""

    guid: str
    title: str
    date: str
    failure_class: str
    failure_message: str


def _now() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _run_url() -> str:
    """Build the GitHub Actions run link for escalation, from the runner's env.

    The three variables are provided automatically by Actions; outside CI they fall
    back to a clearly-marked placeholder so a local run still produces an email.
    """
    server = config.get("GITHUB_SERVER_URL", "https://github.com")
    repository = config.get("GITHUB_REPOSITORY", "")
    run_id = config.get("GITHUB_RUN_ID", "")
    if repository and run_id:
        return f"{server}/{repository}/actions/runs/{run_id}"
    return "(local run — no GitHub Actions context)"


def _escalation(
    sermon: SermonRecord, failure_class: str | None, message: str | None
) -> _Escalation:
    """Build an escalation payload from a sermon and its failure details."""
    return _Escalation(
        guid=sermon.guid,
        title=sermon.title,
        date=sermon.published_on,
        failure_class=failure_class or "Unknown",
        failure_message=message or "",
    )


# Alert class for the collision escalation below, distinct from the run-record
# `DUPLICATE_ERROR_CLASS` it fires on: this is the `notified_alerts` (ADR-0046) key,
# not a lifecycle error class, so it never collides with one.
_DUPLICATE_TRANSCRIPT_ALERT_CLASS = "DuplicateTranscriptCollision"


def _duplicate_transcript_escalation(
    sermon: SermonRecord, other: SermonRecord, transcript_hash: str
) -> _Escalation:
    """One escalation naming both sermons in a transcript-hash collision (#517 child 2).

    The post-transcription dedup guard (PRD §11.3) keeps whichever record generated
    first and sends the other terminal — arbitrary with respect to which one's own
    metadata actually describes the audio (ADR-0077). With the pre-transcription
    fingerprint check (ADR-0077) now catching the common case, this guard firing at
    all should be rare, so the alert names both records instead of just the one sent
    terminal, so a human can confirm which is correct.
    """
    return _Escalation(
        guid=sermon.guid,
        title=sermon.title,
        date=sermon.published_on,
        failure_class=_DUPLICATE_TRANSCRIPT_ALERT_CLASS,
        failure_message=(
            f"Transcript identical to an already-generated sermon (hash {transcript_hash}).\n\n"
            f"Sent terminal: {sermon.title} ({sermon.guid}), {sermon.published_on}\n"
            f"Already generated: {other.title} ({other.guid}), {other.published_on}\n\n"
            "Neither record auto-resolves as correct: the guard keeps whichever "
            "generated first, which is not necessarily the one whose own metadata "
            "actually describes the audio. Confirm against the source, and if the "
            "terminal record is actually right, `recover` it back to `transcribed` "
            "and retire the other by hand."
        ),
    )


def _send_escalations(
    escalations: list[_Escalation], *, send_fn: Callable[[notify.EmailMessage], None]
) -> int:
    """Send one best-effort email per terminal failure; return how many were attempted."""
    run_url = _run_url()
    for esc in escalations:
        notify.send_escalation(
            sermon_title=esc.title,
            sermon_date=esc.date,
            failure_class=esc.failure_class,
            failure_message=esc.failure_message,
            run_url=run_url,
            send_fn=send_fn,
        )
    return len(escalations)


def _send_note_deliveries(
    registry: Registry,
    published: list[str],
    *,
    repo_root: Path,
    send_fn: Callable[[notify.EmailMessage], None],
) -> int:
    """Email each newly published note's ``.docx`` to Jacob; return how many were attempted.

    Best-effort delivery (PRD §11.2): runs once per sermon at the publish transition —
    a ``published`` sermon is terminal and never reprocessed, so a re-run never re-sends.
    A record missing its artifact path is skipped (nothing to attach).
    """
    attempted = 0
    for guid in published:
        sermon = registry.get(guid)
        if sermon is None or sermon.artifact_path is None:
            continue
        notify.send_note(
            sermon_title=sermon.title,
            sermon_date=sermon.published_on,
            artifact_path=repo_root / sermon.artifact_path,
            send_fn=send_fn,
        )
        attempted += 1
    return attempted


def _send_discord_deliveries(
    registry: Registry,
    published: list[str],
    *,
    repo_root: Path,
    send_fn: Callable[[discord_notify.DiscordMessage], str | None] | None,
) -> int:
    """Post each newly published note to every Discord channel its church has
    configured (:mod:`notify_channels`, ADR-0067), recording the id of each message
    posted; return how many deliveries were attempted.

    Best-effort delivery (spec 0017, generalized by spec 0024): notes for a church with
    no configured Discord channel reach Jacob by email alone. Like the email delivery it
    runs once per sermon at the publish transition — a ``published`` sermon is terminal
    and never reprocessed, so a re-run never re-posts. A record missing its artifact
    path is skipped (nothing to upload); each church's channels are resolved
    independently, so one church's missing or bad configuration never affects another's.

    ``send_fn`` is the caller's transport override (tests inject one to capture posts
    without a network); ``None`` means "use the real transport". In that ``None`` case
    each configured channel's webhook is bound fresh via ``functools.partial`` over
    :func:`discord_notify.send_to_url`, so a note is never posted to another channel's
    webhook. A church with more than one Discord channel gets a post on each.

    The channel gets the ``.pdf``, which Discord previews inline; the ``.docx`` recorded
    on the registry is the fallback for when the best-effort PDF render was lost
    (spec 0017 amendment, #173). The post also names the record's ``episode_url`` so the
    channel can reach the sermon itself (omitted when the source publishes none), and
    the website's page for the note itself, ``note_url_for(sermon)`` (spec 0017
    amendment, #574) — always present, since it is derived rather than feed-sourced.

    Each delivered message's id is written to the record (ADR-0062): to the legacy
    ``discord_message_id`` field exactly as before (a church with more than one Discord
    channel retains only the last channel's id there — a known limitation), and, when the
    channel carries a configured ``name``, also to ``channel_message_ids[name]`` (spec 0024
    amendment), which keeps every channel's id distinct regardless of how many a church
    has. Writing it here only puts it in memory — **the caller must save afterwards**, and
    both run paths save before this function rather than after, which is why each gained a
    save of its own.

    A channel :func:`notify_channels.already_sent` reports as delivered is skipped rather
    than posted again — the guard that makes this idempotent when ``published`` names a
    sermon this run already delivered in an earlier pass over the *same* in-memory
    registry, and also when a channel was recorded by an entirely separate run whose
    commit this run's checkout picked up. It does not by itself cover every cause of
    redelivery (a checkout pinned to a stale ref, as a re-run of an already-succeeded
    workflow attempt produces, cannot see a sibling attempt's later commit), so a
    duplicate is still possible when that happens — see the landmine in CLAUDE.md §10.
    """
    attempted = 0
    for guid in published:
        sermon = registry.get(guid)
        if sermon is None or sermon.artifact_path is None:
            continue
        docx_path = repo_root / sermon.artifact_path
        pdf_path = docx_path.with_suffix(".pdf")
        note_url = note_url_for(sermon)
        for channel in notify_channels.channels_for(sermon.source):
            if channel.kind != "discord":
                continue
            if notify_channels.already_sent(sermon, channel):
                logger.info(
                    "skipping discord delivery for %r on %r: already sent",
                    sermon.guid,
                    channel.name or "<unnamed>",
                )
                continue
            sermon_send_fn = (
                functools.partial(discord_notify.send_to_url, channel.url)
                if send_fn is None
                else send_fn
            )
            result = discord_notify.send_note(
                sermon_title=sermon.title,
                sermon_date=sermon.published_on,
                artifact_path=pdf_path if pdf_path.exists() else docx_path,
                episode_url=sermon.episode_url or None,
                note_url=note_url,
                send_fn=sermon_send_fn,
            )
            if result.message_id is not None:
                registry.record_discord_message(guid, result.message_id)
                if channel.name:
                    registry.record_channel_message(guid, channel.name, result.message_id)
            attempted += 1
    return attempted


def _send_google_chat_deliveries(
    registry: Registry,
    published: list[str],
    *,
    send_fn: Callable[[google_chat_notify.GoogleChatMessage], str | None] | None,
) -> int:
    """Post each newly published note to every Google Chat channel its church has
    configured (:mod:`notify_channels`, ADR-0067); return how many were attempted.

    Mirrors :func:`_send_discord_deliveries`'s shape and posture (spec 0024): best-effort,
    scoped to whatever channels are configured, one church's missing or bad configuration
    never affecting another's, and one channel's failure never blocking a sibling's.
    Carries no attachment — Google Chat's webhook API has none to offer, so the message
    links the website's note page instead (spec 0024 amendment, #472; see
    :mod:`google_chat_notify`), built from the sermon's source and slug
    (:func:`feed.sermon_slug` — the same derivation the website's own manifest uses) under
    ``SERMON_NOTES_WEBSITE_BASE_URL``. A record with no ``artifact_path`` is skipped, same
    as Discord: the slug is derived from the artifact stem and there is none to derive it
    from.

    A delivered message's id is written to ``channel_message_ids[name]`` (spec 0024
    amendment, ADR-0067 amendment) when the channel carries a configured ``name`` — there
    is no other stable, non-secret value to key on. Writing it here only puts it in
    memory — **the caller must save afterwards**, same contract as
    :func:`_send_discord_deliveries`.

    Skips a channel :func:`notify_channels.already_sent` reports as delivered, same
    idempotency guard and same limitation as :func:`_send_discord_deliveries`'s copy.
    """
    attempted = 0
    for guid in published:
        sermon = registry.get(guid)
        if sermon is None or sermon.artifact_path is None:
            continue
        note_url = note_url_for(sermon)
        for channel in notify_channels.channels_for(sermon.source):
            if channel.kind != "google_chat":
                continue
            if notify_channels.already_sent(sermon, channel):
                logger.info(
                    "skipping google chat delivery for %r on %r: already sent",
                    sermon.guid,
                    channel.name or "<unnamed>",
                )
                continue
            sermon_send_fn = (
                functools.partial(google_chat_notify.default_send, channel.url)
                if send_fn is None
                else send_fn
            )
            result = google_chat_notify.send_note(
                sermon_title=sermon.title,
                sermon_date=sermon.published_on,
                note_url=note_url,
                send_fn=sermon_send_fn,
            )
            if result.message_id is not None and channel.name:
                registry.record_channel_message(guid, channel.name, result.message_id)
            attempted += 1
    return attempted


def _go_terminal(
    registry: Registry,
    sermon: SermonRecord,
    attempted_state: str,
    *,
    error_class: str,
    error_detail: str,
    now: str | None,
) -> None:
    """Advance ``sermon`` to ``failed`` and append its terminal run record (PRD §11.1)."""
    registry.advance(sermon.guid, "failed", now=now)
    registry.append_run(
        sermon.guid,
        RunRecord(
            attempted_state=attempted_state,
            outcome="failed_terminal",
            error_class=error_class,
            error_detail=error_detail,
            started_at=now if now is not None else _now(),
            finished_at=now if now is not None else _now(),
        ),
    )
    logger.warning("sermon %s terminal at %s: %s", sermon.guid, attempted_state, error_detail)


# --- POLL / SEED -----------------------------------------------------------


def _poll(
    registry: Registry,
    *,
    adapters: Sequence[SourceAdapter],
    initial_limit: int,
) -> PollResult:
    """Poll every enabled source, merging their results (PRD §6.6, ADR-0013).

    Each adapter resolves its own source config, defers on a fetch failure, and
    upserts only the newest ``initial_limit`` qualifying sermons — idempotently, so
    a re-poll never resets a sermon's progress. Capping every poll (not just the
    first) keeps the v1 working set small, away from rate-limit/bot-scraping exposure
    downstream. One source's poll failure defers that source (``deferred`` set on the
    merged result) without aborting the others.
    """
    discovered: list[str] = []
    flagged: list[str] = []
    excluded = 0
    deferred = False
    publish_times: dict[str, datetime] = {}
    for adapter in adapters:
        result = adapter.poll(registry, limit=initial_limit)
        discovered.extend(result.discovered)
        flagged.extend(result.flagged)
        excluded += result.excluded
        deferred = deferred or result.deferred
        publish_times.update(result.publish_times)
    return PollResult(
        discovered=discovered,
        flagged=flagged,
        excluded=excluded,
        deferred=deferred,
        publish_times=publish_times,
    )


# --- Stages ----------------------------------------------------------------


def _transcribe_stage(
    registry: Registry,
    *,
    transcribe_fn: Callable[[Path], str],
    download: Callable[[str, Path], None],
    transcripts_dir: Path,
    sleep: Callable[[float], None],
    now: str | None,
    escalations: list[_Escalation],
    duplicate_lookup_registry: Registry | None = None,
) -> list[str]:
    """Transcribe every ``discovered`` sermon; collect terminal failures (PRD §6.4).

    ``duplicate_lookup_registry`` passes through to :func:`transcribe_sermon` for the
    audio-fingerprint duplicate check (ADR-0077, #517) — a shard's own ``registry`` is
    a throwaway single-record ledger, so :func:`run_shard` supplies a read-only load
    of the committed ledger here instead. ``None`` (the serial pipeline's case, where
    ``registry`` already is the whole ledger) defers to ``transcribe_sermon``'s own
    default of comparing against ``registry`` itself.
    """
    advanced: list[str] = []
    for sermon in [s for s in registry.sermons() if s.state == "discovered"]:
        started = time.perf_counter()
        result = transcribe_sermon(
            registry,
            sermon.guid,
            transcribe=transcribe_fn,
            download=download,
            duplicate_lookup_registry=duplicate_lookup_registry,
            transcripts_dir=transcripts_dir,
            sleep=sleep,
            now=now,
        )
        logger.info(
            "transcribe stage: %s %s in %.0fs",
            sermon.guid,
            result.outcome,
            time.perf_counter() - started,
        )
        if result.outcome == "failed":
            escalations.append(_escalation(sermon, result.error_class, result.error_detail))
        elif result.outcome == "success":
            advanced.append(sermon.guid)
        registry.save()  # Persist per sermon so a crash loses at most one (#35).
    return advanced


def _generate_stage(
    registry: Registry,
    *,
    llm_call: Callable[[str, str], LLMResponse],
    transcripts_dir: Path,
    notes_dir: Path,
    now: str | None,
    escalations: list[_Escalation],
    from_state: str = "transcribed",
) -> list[str]:
    """Generate a note for every ``transcribed`` sermon, advancing each (PRD §6.4).

    The mechanical quote gate is retired (ADR-0008): a note always advances to
    ``generated``. A failed LLM call or a failed note-JSON write (spec 0009) is
    terminal. ``generate_notes`` persists the JSON sidecar before this stage advances
    the sermon, so a ``generated`` sermon always has its artifact on disk for render.

    ``from_state`` is ``pending_batch`` when a backfill's Batches API result has come
    back (spec 0020): the completion arrives through ``llm_call`` like any other, and
    everything after it — parse, validate, sidecar, cost, the advance to ``generated``
    — is deliberately the same code, so a batch-generated note is indistinguishable
    downstream from a synchronous one.
    """
    advanced: list[str] = []
    for sermon in [s for s in registry.sermons() if s.state == from_state]:
        started = time.perf_counter()
        try:
            generate_notes(
                registry,
                sermon.guid,
                llm_call=llm_call,
                transcripts_dir=transcripts_dir,
                notes_dir=notes_dir,
                now=now,
            )
        except (GenerationError, LLMError) as exc:
            _go_terminal(
                registry,
                sermon,
                "generated",
                error_class=type(exc).__name__,
                error_detail=str(exc),
                now=now,
            )
            escalations.append(_escalation(sermon, type(exc).__name__, str(exc)))
            registry.save()  # Persist the terminal failure before moving on (#35).
            continue

        registry.advance(sermon.guid, "generated", now=now)
        advanced.append(sermon.guid)
        logger.info("generate stage: %s in %.0fs", sermon.guid, time.perf_counter() - started)
        registry.save()  # Persist the generated advance + run so a crash never re-bills the LLM (#35).
    return advanced


def _render_stage(
    registry: Registry,
    *,
    notes_dir: Path,
    now: str | None,
    escalations: list[_Escalation],
) -> list[str]:
    """Render and publish every ``generated`` sermon from its persisted note JSON (PRD §6.4).

    The note is read from the ``notes/YYYY/YYYY-MM-DD_<slug>.json`` sidecar written by
    the generate stage (spec 0009), not from in-memory state — so a ``.docx`` can be
    rebuilt without another LLM call. A missing sidecar (no LLM fallback) and a
    corrupt one are both terminal render failures that escalate once (#82).
    """
    repo_root = notes_dir.parent
    published: list[str] = []
    for sermon in [s for s in registry.sermons() if s.state == "generated"]:
        try:
            note = read_note_json(note_json_path(notes_dir, sermon))
        except FileNotFoundError as exc:
            # No persisted note (a prior run crashed before the JSON was written). The
            # transcript_hash guard blocks regeneration, so there is no automatic
            # recovery: fail terminal and escalate for manual attention rather than
            # silently stalling in `generated` forever (#82).
            detail = f"missing note JSON sidecar at {note_json_path(notes_dir, sermon)}"
            _go_terminal(
                registry,
                sermon,
                "published",
                error_class=type(exc).__name__,
                error_detail=detail,
                now=now,
            )
            escalations.append(_escalation(sermon, type(exc).__name__, detail))
            registry.save()  # Persist the terminal failure before moving on (#35).
            continue
        except SchemaValidationError as exc:
            _go_terminal(
                registry,
                sermon,
                "published",
                error_class=type(exc).__name__,
                error_detail=str(exc),
                now=now,
            )
            escalations.append(_escalation(sermon, type(exc).__name__, str(exc)))
            registry.save()  # Persist the terminal failure before moving on (#35).
            continue
        try:
            path = render_note(note, sermon, notes_dir=notes_dir)
        except RenderError as exc:
            _go_terminal(
                registry,
                sermon,
                "published",
                error_class=type(exc).__name__,
                error_detail=str(exc),
                now=now,
            )
            escalations.append(_escalation(sermon, type(exc).__name__, str(exc)))
            registry.save()  # Persist the terminal failure before moving on (#35).
            continue

        # The note's second rendering (#173, ADR-0025). Best-effort by contract: it
        # returns None on any failure, so a lost PDF costs the Discord preview and
        # nothing else — the .docx is the deliverable of record and publish proceeds.
        render_note_pdf(note, sermon, notes_dir=notes_dir)

        sermon.artifact_path = path.relative_to(repo_root).as_posix()
        registry.advance(sermon.guid, "published", now=now)
        registry.append_run(
            sermon.guid,
            RunRecord(
                attempted_state="published",
                outcome="success",
                started_at=now if now is not None else _now(),
                finished_at=now if now is not None else _now(),
            ),
        )
        registry.save()  # Persist the published advance per sermon (#35).
        published.append(sermon.guid)
    return published


# --- Publish step (feed push + deploy hook) --------------------------------


def _publishing_configured() -> bool:
    """True only when every publish-step credential and target is present in config.

    The two web-publishing boundaries (ADR-0018) stay dark until ``sermon-notes-content``
    and its write token plus the Vercel hook URL are provisioned (U5/U8); an unconfigured
    pipeline publishes notes to the repo as before and skips the feed push entirely.
    """
    return all(
        config.get(name, None)
        for name in ("CONTENT_REPO", "CONTENT_REPO_TOKEN", "VERCEL_DEPLOY_HOOK_URL")
    )


def _should_escalate(
    registry: Registry, alert_class: str, fingerprint: str, *, now: str | None
) -> bool:
    """Whether to send a condition-level alert now (ADR-0046, #324).

    Thin wrapper around :meth:`Registry.should_escalate_alert` reading the shared
    cooldown from config, so the per-run alert builders below don't each read it
    themselves — the three run-level condition alerts plus the duplicate-transcript
    collision escalation (#517 child 2), keyed per colliding guid pair rather than
    per run. The two reconcile dead-man's switches already run once/day and are
    unaffected (see the ADR's open questions).
    """
    cooldown_hours = config.get_int(
        "ALERT_COOLDOWN_HOURS", _DEFAULT_ALERT_COOLDOWN_HOURS, minimum=1
    )
    return registry.should_escalate_alert(
        alert_class, fingerprint, cooldown_hours=cooldown_hours, now=now
    )


def _unenforced_cap_escalation(deltas: Sequence[Delta], *, now: str | None) -> _Escalation | None:
    """One run-level alert when any shard spent without the attempt cap (ADR-0036, #264).

    The cap is the load-bearing bound on unattended spend, and it is the only one that
    survives the ledger not being written — so a run where it could not be consulted is
    a run with no bound at all, since ``_enforce_spend_budget`` reads the same ledger the
    #235 failure mode stops writing. Failing open is deliberate; failing open with no
    alert, no red run, and no non-zero exit is what made it invisible.

    One escalation for the whole run rather than one per sermon: the condition is a
    property of the run — a revoked scope or an API outage hits every shard at once — so
    per-sermon mail would send a dozen copies of one fault. ``None`` when every shard's
    cap was in force, including the dormant off-Actions case.
    """
    unenforced = [(d.sermon.guid, d.cap_unenforced) for d in deltas if d.cap_unenforced]
    if not unenforced:
        return None
    logger.error(
        "merge: the attempt cap was not enforced for %d sermon(s) this run", len(unenforced)
    )
    return _Escalation(
        guid="",
        title=f"Attempt cap not enforced for {len(unenforced)} sermon(s)",
        date=(now or _now())[:10],
        failure_class=_CAP_UNENFORCED_ERROR_CLASS,
        failure_message=(
            "The attempt cap (ADR-0033) could not be consulted, so these sermons were "
            "transcribed and generated with no bound on repeat spend:\n"
            + "\n".join(f"- {guid}: {reason}" for guid, reason in unenforced)
            + "\n\nThe registry-derived spend windows are not a backstop here — they read "
            "the committed ledger, which is exactly what stops being written in the "
            "failure the claim cap exists to survive (#235). Check the shard job's "
            "`actions: read` scope and the Actions artifacts API before the next run."
        ),
    )


def _spend_refusal_escalation(refusal: str | None, *, now: str | None) -> _Escalation | None:
    """One run-level alert when the plan refused to fan out on a spend cap (ADR-0037, #265).

    ``refusal`` is :class:`SpendGuardError`'s own message, handed across the plan → merge
    seam by the workflow because the plan job holds no mail credentials (ADR-0033) and a
    plan that failed would skip this job entirely. It is carried verbatim: it already
    names the window, the total, the cap, the env var to raise, and the fact that a manual
    dispatch is exempt — the whole point of #265 is that nobody was ever delivered it.

    ``None`` when the run was not refused, which is every ordinary run.
    """
    if not refusal:
        return None
    logger.error("merge: the plan refused to fan out on a spend cap — %s", refusal)
    return _Escalation(
        guid="",
        title="LLM spend cap reached — the pipeline is not processing sermons",
        date=(now or _now())[:10],
        failure_class=_SPEND_REFUSAL_ERROR_CLASS,
        failure_message=(
            f"{refusal}\n\nNo sermon was polled, planned, or spent on this run, and none "
            "will be until the window clears or the cap is raised. The rolling window "
            "self-clears as the runs that filled it age out, so a one-off surge resolves "
            "on its own — but a persistent cause (a model swap, a pricing change, a wide "
            "INITIAL_RUN_LIMIT left in a repository variable) holds the refusal "
            "indefinitely, and every Sunday it holds is a sermon the pipeline never "
            "publishes."
        ),
    )


def _no_sources_escalation(refusal: str | None, *, now: str | None) -> _Escalation | None:
    """One run-level alert when every enabled source resolved to zero usable feeds (#496).

    ``refusal`` is :class:`NoSourcesConfiguredError`'s own message, handed across the
    plan → merge seam the same way ``spend_refusal`` is (ADR-0037): the plan job holds
    no mail credentials, so this is that condition's only delivery channel. Distinct
    from an ordinary per-source skip (ADR-0074), which needs no alert of its own — this
    fires only when at least one church was enabled and none of them resolved, the
    condition that would otherwise poll nothing forever with no signal.

    ``None`` when the run was not refused, which is every ordinary run.
    """
    if not refusal:
        return None
    logger.error("merge: every enabled source resolved to zero usable feeds — %s", refusal)
    return _Escalation(
        guid="",
        title="No church feeds configured — the pipeline is discovering nothing",
        date=(now or _now())[:10],
        failure_class=_NO_SOURCES_ERROR_CLASS,
        failure_message=(
            f"{refusal}\n\nNo new sermon was discovered on this run, and none will be "
            "until CHURCHES names at least one usable feed for an enabled source. "
            "Sermons already in progress are unaffected — this only stops new discovery."
        ),
    )


def _delta_shortfall_escalation(
    missing: int, expected: int, *, now: str | None
) -> _Escalation | None:
    """One run-level alert for the deltas a broken fan-out never delivered (#266).

    Reached only when a shard job *failed*: an unexplained shortfall never gets this far,
    because the CLI refuses the merge outright (``__main__._run_merge``). What is left is
    the case ADR-0023 requires to still merge — one shard crashing must not discard the
    other nine — where the sermons behind the missing deltas are otherwise unmentioned
    anywhere a person reads. A shard that fails *inside* Python hands back a terminal
    delta and is escalated per sermon; a shard that dies at the job level (OOM, a lost
    runner, a failed install, an unrecordable claim) writes nothing at all, so without
    this the only trace is a red matrix leg in a run log.

    Their guids are deliberately not named: the merge downloads deltas, not the working
    set, so it knows how many sermons were planned but not which ones. The count plus the
    run link is what it can say truthfully.

    No exit-code effect, unlike the refusals either side of it: this alert exists only on
    the path where a shard job already failed, which is already a red run. ``None`` on
    every ordinary run, where every planned delta arrived.
    """
    if missing <= 0:
        return None
    logger.error(
        "merge: %d of %d planned shard delta(s) never reached the fan-in", missing, expected
    )
    return _Escalation(
        guid="",
        title=f"{missing} of {expected} shard deltas never reached the merge",
        date=(now or _now())[:10],
        failure_class=_DELTA_SHORTFALL_ERROR_CLASS,
        failure_message=(
            f"The run planned {expected} sermon(s) and the merge received {missing} "
            "fewer delta(s) than that. A shard job reported a failure, so these sermons "
            "were most likely lost with the runner rather than in transit — their "
            "records stay where they were and the next run will re-attempt them, at the "
            "cost of re-transcribing and re-generating each one.\n\n"
            "The deltas that did arrive were applied and committed (ADR-0023 failure "
            "isolation). Check the failed shard job(s) in this run: a repeated shortfall "
            "for the same sermon burns one of MAX_SERMON_ATTEMPTS on every tick, and the "
            "third one retires it."
        ),
    )


# The fallback diagnosis, for a push that failed without leaving the step anything to
# read back. It names candidates for a human to work through — which is exactly the work
# the quoted refusal below exists to save, so it is what happens when there is no quote.
_ARTIFACT_PUSH_UNCLASSIFIED_DIAGNOSIS = (
    "The step captured nothing from the remote, so what refused the push has to be found "
    "rather than read. Look for a branch protection rule or ruleset now covering the "
    "pusher, a revoked or read-only credential, or a repository in a state that rejects "
    "writes; the push output is in the run log linked below."
)

# How much of the refusal to quote. A rejection is a handful of `remote:` lines; anything
# longer is a transport dumping hints, and the tail is the part that names the cause.
_PUSH_ERROR_QUOTE_LIMIT = 2000


def _artifact_push_diagnosis(push_error: str | None) -> str:
    """What the operator should look at next, in the remote's own words where possible.

    The refusal text is the one piece of evidence that names the cause outright — a
    ruleset requiring a pull request says so in plain English — and the step used to
    discard it, leaving this alert to describe by prose what the remote had already
    stated (#316). Quoting it is the difference between "look for something that refuses
    the push" and "the default branch's ruleset requires a pull request".
    """
    refusal = (push_error or "").strip()
    if not refusal:
        return _ARTIFACT_PUSH_UNCLASSIFIED_DIAGNOSIS
    if len(refusal) > _PUSH_ERROR_QUOTE_LIMIT:
        refusal = "(earlier output trimmed)\n" + refusal[-_PUSH_ERROR_QUOTE_LIMIT:]
    return (
        "The remote refused the push. What it said:\n\n"
        f"{indent(refusal, '    ')}\n\n"
        "Start there rather than from a list of candidates. A rule or branch protection "
        "naming the default branch means the push needs a pull request, or an exemption "
        "the pusher does not have; a permission or credential error means the deploy key "
        "is revoked, read-only, or no longer on the repository."
    )


_ARTIFACT_PUSH_UNADVANCED = (
    "Until it lands, the ledger on the default branch is as it was before the run, so the "
    "next scheduled run re-processes these sermons and bills for them again — and each "
    "retry burns one of MAX_SERMON_ATTEMPTS, so a push that keeps failing retires them "
    "rather than publishing them."
)


def escalate_artifact_push_failure(
    *,
    preserved: bool = False,
    push_error: str | None = None,
    send_fn: Callable[[notify.EmailMessage], None] = notify.default_send,
    now: str | None = None,
) -> int:
    """Mail the run-level alert that the artifact push was abandoned (#269, #315).

    The one escalation whose trigger lives in the workflow rather than in a pipeline
    pass: the push happens after ``merge`` has returned, in the ``Commit artifacts``
    step, and until this existed a push that never landed was reported only as a red run
    and an Actions annotation.

    ``push_error`` is what the remote said when it refused, captured by the step that
    ran the push. It replaces the guesswork the diagnosis used to offer with the refusal
    itself; empty when the push failed without a readable message (#316).

    ``preserved`` is what the step observed, not what it intended — whether it managed to
    stage the run's output for the upload that follows it. It picks the whole alert, both
    subject and body, because the two outcomes are different events: with the tree
    preserved this is a recovery to perform and the sermons cost nothing to land, and
    without it this is #235's failure mode again, where every sermon the run processed is
    re-transcribed, re-generated and re-billed on the next tick. Telling an operator the
    work is safe when it is not is worse than the alert this replaced.

    Run-level like its siblings above, so nothing reaches the ledger: the sermons
    themselves are fine, and the ledger recording that fact is exactly what did not land.

    Returns how many escalations were attempted, matching :func:`_send_escalations`.
    """
    if preserved:
        logger.error(
            "the artifact push was abandoned; this run's artifacts were preserved as %r",
            ARTIFACT_PRESERVATION_NAME,
        )
        title = ARTIFACT_PUSH_FAILURE_TITLE
        failure_message = (
            "The merge job could not push what this run produced. The runner has been "
            "reclaimed, but nothing went with it: the notes, the transcripts, the ledger "
            "and the regenerated README index were uploaded from that job as the workflow "
            f"artifact '{ARTIFACT_PRESERVATION_NAME}', kept for 90 days — the same window "
            "as the attempt claims that would otherwise block a replay.\n\n"
            "To land them: download that artifact from the run linked below, then from a "
            "branch cut off the default branch run\n\n"
            f"    python {ARTIFACT_REPLAY_SCRIPT} <the downloaded directory>\n\n"
            "and open a pull request with what it stages. Nothing is re-transcribed and "
            "nothing is re-generated — the replay is a file copy plus a ledger merge, so "
            "it costs nothing and spends no attempt.\n\n"
            f"{_ARTIFACT_PUSH_UNADVANCED}\n\n"
            f"{_artifact_push_diagnosis(push_error)}"
        )
    else:
        logger.error(
            "the artifact push was abandoned and could not be preserved; "
            "this run's artifacts stay on the runner"
        )
        title = ARTIFACT_PUSH_LOSS_TITLE
        failure_message = (
            "The merge job could not push what this run produced, and could not stage it "
            "for preservation either, so the runner has been reclaimed with the notes, "
            "transcripts and ledger advances still on it. The run's log says which of the "
            "two failed first.\n\n"
            "Nothing is corrupt: the ledger on the default branch is simply as it was "
            "before the run. The cost is that every sermon this run processed is "
            "unadvanced, so the next scheduled run re-transcribes and re-generates each "
            "one and bills for it again — and each retry burns one of "
            "MAX_SERMON_ATTEMPTS, so a push that keeps failing retires the sermons "
            "rather than publishing them.\n\n"
            f"{_artifact_push_diagnosis(push_error)}"
        )
    return _send_escalations(
        [
            _Escalation(
                guid="",
                title=title,
                date=(now or _now())[:10],
                failure_class=_ARTIFACT_PUSH_ERROR_CLASS,
                failure_message=failure_message,
            )
        ],
        send_fn=send_fn,
    )


# Default stall threshold, mirrored from queue-stall-alert.yml's QUEUE_STALL_HOURS.
# Not a config.py entry: nothing in src/ reads the workflow's own env var directly,
# this only backstops a --threshold-hours flag the workflow always passes explicitly
# (ADR-0057) so the email's "Threshold: ..." line still has something sane if it did not.
_DEFAULT_QUEUE_STALL_HOURS = 24


def escalate_stalled_queue_pr(
    *,
    pr_number: int,
    pr_title: str,
    hours_stale: int,
    pr_url: str,
    threshold_hours: int = _DEFAULT_QUEUE_STALL_HOURS,
    send_fn: Callable[[notify.EmailMessage], None] = notify.default_send,
) -> bool:
    """Mail the alert that a PR has stalled in the merge train's queue (#339, ADR-0057).

    The trigger lives entirely in ``queue-stall-alert.yml``, not in any pass this
    module runs: the workflow finds PRs wearing ``queue:ready`` past the threshold
    and invokes ``python -m sermon_notes alert-stalled-queue`` once per stale PR,
    naming what it found in flags. This function is the thin span that crosses from
    that CLI call into the single email boundary (:mod:`notify`) — mirroring
    :func:`escalate_artifact_push_failure`'s shape, the other escalation whose
    trigger is a workflow step rather than a pipeline pass.

    Never tied to a sermon or the ledger: a stalled PR is a repo-governance fact,
    not a `SermonRecord`. Returns whether the send succeeded, so the CLI can fail the
    run on ``False`` — a dropped alert of last resort is itself the incident.
    """
    delivered = notify.send_stalled_queue_alert(
        pr_number=pr_number,
        pr_title=pr_title,
        hours_stale=hours_stale,
        threshold_hours=threshold_hours,
        pr_url=pr_url,
        send_fn=send_fn,
    )
    if not delivered:
        logger.error("stalled-queue alert for PR #%d was not delivered", pr_number)
    return delivered


def _publish_escalation(step: str, exc: Exception, now: str | None) -> _Escalation:
    """A run-level escalation for a failed publish boundary (not tied to one sermon)."""
    return _Escalation(
        guid="",
        title=f"Content feed publish — {step}",
        date=(now or _now())[:10],
        failure_class=type(exc).__name__,
        failure_message=str(exc),
    )


def _publish_feed_step(
    *,
    registry_path: Path,
    notes_dir: Path,
    published: list[str],
    render_feed: Callable[..., object],
    publish_feed: Callable[[Path], None],
    fire_deploy_hook: Callable[[], None],
    escalations: list[_Escalation],
    now: str | None,
) -> tuple[bool, bool]:
    """Render, push, and trigger a rebuild for a run that produced new published notes.

    Runs only when at least one sermon newly reached ``published`` and the publish
    boundaries are configured. Renders the feed to a temp directory, pushes it to the
    content repo, then fires the Vercel deploy hook, in that order (spec 0014). A render
    failure escalates and skips both — there is no tree to push; a push failure escalates
    and skips the hook — nothing new reached the content repo to deploy; a hook failure
    escalates. Each boundary retries internally before raising (PRD §11.1/§11.2), so a
    failure escalates rather than silently dropping the publish.
    Returns whether the feed was pushed and whether the rebuild was triggered.
    """
    if not published:
        return (False, False)
    if not _publishing_configured():
        logger.info("content publishing not configured; skipping feed push and deploy hook")
        return (False, False)

    with tempfile.TemporaryDirectory() as workdir:
        feed_dir = Path(workdir) / "feed"
        try:
            render_feed(registry_path=registry_path, notes_dir=notes_dir, output_dir=feed_dir)
        except Exception as exc:
            # The render reads every archived note, so its failure surface grows with the
            # archive; it escalates like the two boundaries below rather than aborting the
            # run and taking the queued escalations with it (#197).
            logger.error("content feed render failed; escalating: %s", exc)
            escalations.append(_publish_escalation("content feed render", exc, now))
            return (False, False)
        try:
            publish_feed(feed_dir)
        except content_publish.ContentPublishError as exc:
            logger.error("content feed push failed; escalating: %s", exc)
            escalations.append(_publish_escalation("content feed publish", exc, now))
            return (False, False)

    try:
        fire_deploy_hook()
    except deploy_hook.DeployHookError as exc:
        logger.error("deploy hook failed; escalating: %s", exc)
        escalations.append(_publish_escalation("Vercel deploy hook", exc, now))
        return (True, False)
    return (True, True)


# --- Entry points ----------------------------------------------------------


def run_pipeline(
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    transcripts_dir: Path = DEFAULT_TRANSCRIPTS_DIR,
    notes_dir: Path = DEFAULT_NOTES_DIR,
    readme_path: Path = DEFAULT_README_PATH,
    adapters: Sequence[SourceAdapter] | None = None,
    transcribe: Callable[[Path], str] = default_transcribe,
    download: Callable[[str, Path], None] = audio.http_download,
    llm_call: Callable[[str, str], LLMResponse] = llm_client.call,
    send_fn: Callable[[notify.EmailMessage], None] = notify.default_send,
    discord_send_fn: Callable[[discord_notify.DiscordMessage], str | None] | None = None,
    google_chat_send_fn: Callable[[google_chat_notify.GoogleChatMessage], str | None] | None = None,
    render_feed: Callable[..., object] = feed.render_feed,
    publish_feed: Callable[[Path], None] = content_publish.publish_feed,
    fire_deploy_hook: Callable[[], None] = deploy_hook.fire_deploy_hook,
    sleep: Callable[[float], None] = time.sleep,
    now: str | None = None,
) -> PipelineResult:
    """Run the full pipeline once, advancing every sermon as far as it can (PRD §6.4).

    Boundaries (source fetch, audio download, transcription, the LLM, email, and the
    feed render/push/deploy-hook publish step) are all injectable so the run is
    exercised end to end with nothing real touched. The
    enabled source adapters default to :func:`sources.enabled_adapters`. One source's
    poll failure defers that source without aborting the others or the downstream
    stages (ADR-0013); the merged ``deferred`` flag is carried into the result.
    Terminal failures are collected and escalated once each after the ledger is saved.
    """
    initial_limit = config.get_int("INITIAL_RUN_LIMIT", _DEFAULT_INITIAL_LIMIT, minimum=1)
    if adapters is None:
        adapters = sources.enabled_adapters()
    registry = Registry.load(registry_path)
    escalations: list[_Escalation] = []

    poll_result = _poll(
        registry,
        adapters=adapters,
        initial_limit=initial_limit,
    )

    transcribed = _transcribe_stage(
        registry,
        transcribe_fn=transcribe,
        download=download,
        transcripts_dir=transcripts_dir,
        sleep=sleep,
        now=now,
        escalations=escalations,
    )
    generated = _generate_stage(
        registry,
        llm_call=llm_call,
        transcripts_dir=transcripts_dir,
        notes_dir=notes_dir,
        now=now,
        escalations=escalations,
    )
    published = _render_stage(registry, notes_dir=notes_dir, now=now, escalations=escalations)

    registry.save()

    # Notify before the index refresh and the publish step: on an unattended run the
    # escalation email is the only operator signal, so nothing downstream may preempt it
    # (#197). The publish step's own escalations follow it below.
    escalated = _send_escalations(escalations, send_fn=send_fn)
    delivered = _send_note_deliveries(
        registry, published, repo_root=notes_dir.parent, send_fn=send_fn
    )
    discord_delivered = _send_discord_deliveries(
        registry, published, repo_root=notes_dir.parent, send_fn=discord_send_fn
    )
    if discord_delivered:
        # The ledger was saved *before* the delivery above, so the message ids it just
        # recorded exist only in memory until now. Without this the ids die with the
        # runner and the next repaint is back to copying them out of Discord by hand
        # (ADR-0062, CLAUDE.md §10).
        registry.save()
    google_chat_delivered = _send_google_chat_deliveries(
        registry, published, send_fn=google_chat_send_fn
    )
    if google_chat_delivered:
        # Same reasoning as the Discord save above: a channel-message-id capture
        # (spec 0024 amendment) that lived only in memory would die with the runner.
        registry.save()

    if published:
        update_readme(registry_path=registry_path, readme_path=readme_path)

    publish_escalations: list[_Escalation] = []
    feed_published, deploy_triggered = _publish_feed_step(
        registry_path=registry_path,
        notes_dir=notes_dir,
        published=published,
        render_feed=render_feed,
        publish_feed=publish_feed,
        fire_deploy_hook=fire_deploy_hook,
        escalations=publish_escalations,
        now=now,
    )
    escalated += _send_escalations(publish_escalations, send_fn=send_fn)

    return PipelineResult(
        deferred=poll_result.deferred,
        discovered=tuple(poll_result.discovered),
        transcribed=tuple(transcribed),
        generated=tuple(generated),
        published=tuple(published),
        failed=tuple(esc.guid for esc in escalations),
        escalated=escalated,
        delivered=delivered,
        discord_delivered=discord_delivered,
        google_chat_delivered=google_chat_delivered,
        feed_published=feed_published,
        deploy_triggered=deploy_triggered,
    )


# Days a discovered sermon may wait for its audio enclosure before the third switch
# reports it (ADR-0066). Generous on purpose: the wait is normally a day or two, and
# this alert exists for the case where the audio never arrives at all, not to narrate
# an ordinary ingestion lag.
_DEFAULT_MISSING_ENCLOSURE_DAYS = 14
_AWAITING_ENCLOSURE_ALERT_CLASS = "SermonAwaitingEnclosure"

# Days a discovered sermon may sit withheld as a suspected audio duplicate before the
# fourth switch reports it (ADR-0077, #517). Same default and reasoning as
# MISSING_ENCLOSURE_DAYS above — this is either a correct, permanent verdict (a stale
# enclosure will never resolve on its own) or a false positive, and either way silent
# indefinite withholding is the failure mode this alert exists to replace.
_DEFAULT_SUSPECTED_DUPLICATE_AUDIO_DAYS = 14
_SUSPECTED_DUPLICATE_AUDIO_ALERT_CLASS = "SermonSuspectedDuplicateAudio"

# Days without a published note before the dead-man's switch trips (#242).
#
# Every other alert fires on a failure the pipeline recognised. In #235 nothing was
# recognised: ~20 runs went green over eight days while nothing published, and a human
# found it. This one fires on the absence of success, which is the only signal that
# covers the failure modes nobody predicted.
#
# 10 days is set by the false-positive cost, not by detection speed. Sermons are
# weekly, so a gap of 8 days is routine; legitimate two-week gaps happen (a holiday, a
# series break, a feed pausing). An alert that cries wolf gets filtered, and a filtered
# alert of last resort is worse than none — so the threshold clears one missed Sunday
# and trips on the second. It is deliberately conservative: this is a net for unknown
# failure modes, not a fast detector and not a cost bound (#240/#241 bound cost).
_DEFAULT_PUBLISH_STALENESS_DAYS = 10


def _check_awaiting_enclosure(
    registry: Registry,
    *,
    now: str | None,
    send_fn: Callable[[notify.EmailMessage], None],
) -> tuple[str, ...]:
    """Report sermons withheld for want of an audio enclosure for too long (ADR-0066).

    The bound on :func:`plan_shards`'s withholding. Withholding is indefinite by
    design — the sermon resumes when the enclosure appears — so without this a sermon
    whose audio never arrives waits silently forever, which is the failure the
    withholding was meant to replace, not reproduce.

    Returns the tripped guids, oldest wait first, whether or not the mail was delivered:
    unlike the two dead-man's switches this does not fail the run, so there is no
    "alert failed" flag to act on. The wait is measured from ``first_seen_at``, which
    ``upsert`` preserves across re-polls, so it survives the daily rediscovery that
    keeps the record fresh.
    """
    threshold = config.get_int("MISSING_ENCLOSURE_DAYS", _DEFAULT_MISSING_ENCLOSURE_DAYS, minimum=1)
    at = parse_instant(now) if now is not None else datetime.now(timezone.utc)
    waiting = [
        (sermon.guid, sermon.title, (at - parse_instant(sermon.first_seen_at)).days)
        for sermon in registry.sermons()
        if _awaiting_enclosure(sermon)
    ]
    tripped = sorted(
        (entry for entry in waiting if entry[2] >= threshold), key=lambda e: (-e[2], e[0])
    )
    if not tripped:
        return ()

    logger.warning(
        "reconcile: %d sermon(s) still have no audio enclosure past %d days: %s",
        len(tripped),
        threshold,
        ", ".join(guid for guid, _title, _days in tripped),
    )
    if _should_escalate(
        registry,
        _AWAITING_ENCLOSURE_ALERT_CLASS,
        ",".join(sorted(guid for guid, _title, _days in tripped)),
        now=now,
    ):
        notify.send_awaiting_enclosure_alert(
            waiting=tripped, threshold_days=threshold, run_url=_run_url(), send_fn=send_fn
        )
    return tuple(sorted(guid for guid, _title, _days in tripped))


def _check_suspected_duplicate_audio(
    registry: Registry,
    *,
    now: str | None,
    send_fn: Callable[[notify.EmailMessage], None],
) -> tuple[str, ...]:
    """Report sermons withheld as a suspected audio duplicate for too long (ADR-0077, #517).

    The bound on :func:`plan_shards`'s persisted-marker withholding, parallel to
    :func:`_check_awaiting_enclosure` — withholding is indefinite by design, so without
    this a false positive (or a real one nobody looks at) waits silently forever.
    """
    threshold = config.get_int(
        "SUSPECTED_DUPLICATE_AUDIO_DAYS", _DEFAULT_SUSPECTED_DUPLICATE_AUDIO_DAYS, minimum=1
    )
    at = parse_instant(now) if now is not None else datetime.now(timezone.utc)
    waiting = [
        (
            sermon.guid,
            sermon.title,
            sermon.suspected_duplicate_of or "",
            (at - parse_instant(sermon.first_seen_at)).days,
        )
        for sermon in registry.sermons()
        if _suspected_duplicate_audio(sermon)
    ]
    tripped = sorted(
        (entry for entry in waiting if entry[3] >= threshold), key=lambda e: (-e[3], e[0])
    )
    if not tripped:
        return ()

    logger.warning(
        "reconcile: %d sermon(s) withheld as a suspected audio duplicate past %d days: %s",
        len(tripped),
        threshold,
        ", ".join(guid for guid, _title, _dup, _days in tripped),
    )
    if _should_escalate(
        registry,
        _SUSPECTED_DUPLICATE_AUDIO_ALERT_CLASS,
        ",".join(sorted(guid for guid, _title, _dup, _days in tripped)),
        now=now,
    ):
        notify.send_suspected_duplicate_audio_alert(
            waiting=tripped, threshold_days=threshold, run_url=_run_url(), send_fn=send_fn
        )
    return tuple(sorted(guid for guid, _title, _dup, _days in tripped))


def _check_publish_staleness(
    registry: Registry,
    *,
    now: str | None,
    send_fn: Callable[[notify.EmailMessage], None],
) -> tuple[int | None, bool]:
    """Alert once when nothing has published in too long (#242); the dead-man's switch.

    Returns ``(days_stale, alert_failed)`` — ``(None, False)`` when the pipeline is
    producing, or has never produced. One check per reconcile pass, one email at most:
    the condition is a property of the ledger as a whole, not of any sermon, so there
    is nothing to fan out over.

    Staleness is measured from ``last_state_change_at`` — *when the pipeline published*
    — not from ``published_on``, which is when the church released the sermon. The
    switch asks whether the pipeline is alive, and a backfill that publishes a 2024
    sermon today is proof that it is; ``published_on`` would still read stale and alert
    on a healthy run.

    An empty published set is pre-go-live, not an outage: a fresh repo has never
    published, so there is no last success for an absence to be measured against.
    """
    threshold = config.get_int("PUBLISH_STALENESS_DAYS", _DEFAULT_PUBLISH_STALENESS_DAYS, minimum=1)
    published_at = [
        parse_instant(sermon.last_state_change_at)
        for sermon in registry.sermons()
        if sermon.state == "published"
    ]
    if not published_at:
        logger.info("reconcile: ledger has never published a note; staleness check skipped.")
        return None, False

    latest = max(published_at)
    at = parse_instant(now) if now is not None else datetime.now(timezone.utc)
    days_stale = (at - latest).days
    logger.info(
        "reconcile: last note published %s — %d day(s) ago, against a %d-day threshold.",
        latest.isoformat(),
        days_stale,
        threshold,
    )
    if days_stale < threshold:
        return None, False

    logger.error(
        "reconcile: no note has published in %d days (threshold %d) and nothing failed — "
        "the pipeline has stopped producing.",
        days_stale,
        threshold,
    )
    delivered = notify.send_staleness_alert(
        last_published_at=latest.isoformat(),
        days_stale=days_stale,
        threshold_days=threshold,
        run_url=_run_url(),
        send_fn=send_fn,
    )
    if not delivered:
        logger.error(
            "reconcile: the publish-staleness alert was NOT delivered — the outage is "
            "unreported by email; the run's non-zero exit is the only remaining signal."
        )
    return days_stale, not delivered


# Consecutive red completed pipeline.yml runs before the tighter dead-man's switch
# trips (#283, ADR-0039). A second switch beside _check_publish_staleness: where that
# one fires on an ABSENCE of success measured in days, this one fires on an
# AFFIRMATIVE run of red completed runs, measured in runs, so it can catch a red-run
# cause with no dedicated escalation (a crash before notify is reached, a broken
# install step, a runner fault) in hours rather than PUBLISH_STALENESS_DAYS' 10 days.
#
# Unlike staleness, this has low false-positive risk against a legitimate feed gap: a
# quiet week still merges green (empty fan-out, trivially-successful empty merge), so
# only an actual non-zero exit trips this — which is what lets it be so much tighter.
_DEFAULT_RED_RUN_STREAK = 5

# Conclusions GitHub's Actions API reports for a completed run that did not do its job.
_RED_CONCLUSIONS = frozenset({"failure", "timed_out", "startup_failure", "action_required"})

# `cancelled` is skipped from the window entirely — neither counted red nor treated as
# breaking the streak. Production's `concurrency: {cancel-in-progress: false}` means
# the workflow never cancels itself, so a `cancelled` conclusion can only be a human's
# manual action (debugging, a verification dispatch) — not a pipeline failure. Counting
# it red would let a debugging session trip the alert it is trying to avoid; counting
# it as breaking the streak would just as wrongly read "nothing happened" as "healthy".
_EXCLUDED_CONCLUSIONS = frozenset({"cancelled"})

_STREAK_WORKFLOW_FILE = "pipeline.yml"
_STREAK_BRANCH = "main"

# Fetch past the threshold so a skipped `cancelled` run doesn't cost the streak its
# length — the check still needs to see the red runs on either side of it.
_STREAK_LOOKBACK_MULTIPLIER = 4


def _check_red_run_streak(
    *,
    send_fn: Callable[[notify.EmailMessage], None],
    fetch: Callable[[str, str], bytes] = workflow_runs.default_fetch,
) -> tuple[int | None, bool]:
    """Alert once when the trailing completed pipeline.yml runs on main are all red (#283).

    Returns ``(streak_count, alert_failed)`` — ``(None, False)`` when the streak has
    not met the threshold, there is not yet enough run history, or the run listing was
    dormant (off Actions) or unreadable this pass. Fails open like `attempt_claims`: a
    guard's own dependency must not be able to take the pipeline down, and the 10-day
    staleness switch remains the backstop of last resort regardless of this one's
    health.

    Reports the *actual* leading streak length, which can exceed the threshold, since
    the lookback fetches well past it — free diagnostic value at no extra API cost.
    """
    threshold = config.get_int("RED_RUN_STREAK_COUNT", _DEFAULT_RED_RUN_STREAK, minimum=1)
    lookback = min(threshold * _STREAK_LOOKBACK_MULTIPLIER, 100)
    try:
        runs = workflow_runs.recent_runs(
            count=lookback,
            workflow_file=_STREAK_WORKFLOW_FILE,
            branch=_STREAK_BRANCH,
            fetch=fetch,
        )
    except workflow_runs.WorkflowRunsError as exc:
        logger.error("reconcile: could not read the pipeline.yml run history: %s", exc)
        return None, False
    if runs is None:
        return None, False

    considered = [run for run in runs if run.conclusion not in _EXCLUDED_CONCLUSIONS]
    streak: list[workflow_runs.RunSummary] = []
    for run in considered:
        if run.conclusion not in _RED_CONCLUSIONS:
            break
        streak.append(run)

    logger.info(
        "reconcile: %d consecutive red pipeline.yml run(s) against a %d-run threshold.",
        len(streak),
        threshold,
    )
    if len(streak) < threshold:
        return None, False

    logger.error(
        "reconcile: the last %d completed pipeline.yml run(s) on main all came back red "
        "(threshold %d) — the pipeline appears to be failing without producing.",
        len(streak),
        threshold,
    )
    delivered = notify.send_red_streak_alert(
        streak_count=len(streak),
        threshold=threshold,
        newest_run_url=streak[0].html_url,
        oldest_run_at=streak[-1].created_at,
        run_url=_run_url(),
        send_fn=send_fn,
    )
    if not delivered:
        logger.error(
            "reconcile: the red-run-streak alert was NOT delivered — the outage is "
            "unreported by email; the run's non-zero exit is the only remaining signal."
        )
    return len(streak), not delivered


def reconcile(
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    notes_dir: Path = DEFAULT_NOTES_DIR,
    readme_path: Path = DEFAULT_README_PATH,
    adapters: Sequence[SourceAdapter] | None = None,
    send_fn: Callable[[notify.EmailMessage], None] = notify.default_send,
    runs_fetch: Callable[[str, str], bytes] = workflow_runs.default_fetch,
    now: str | None = None,
) -> ReconcileResult:
    """Reconcile the ledger against reality (PRD §6.5).

    Re-polls every enabled source to re-flag ambiguous items and pick up anything
    newly classifiable, confirms every ``published`` record still has its artifact on
    disk — escalating any that do not — and resets records stranded ``failed`` by a
    retired failure class (ADR-0014) so the next run re-attempts them. The reset is a
    self-heal, not a failure: it runs no stages and sends no escalation; the next
    scheduled run regenerates from the cached transcript.

    Finally it trips two dead-man's switches — neither needs the pipeline to have
    recognised its own failure: the staleness switch (#242) when the ledger has
    published nothing in ``PUBLISH_STALENESS_DAYS``, and the tighter, complementary
    red-run-streak switch (#283) when the last ``RED_RUN_STREAK_COUNT`` completed
    ``pipeline.yml`` runs on ``main`` all came back red.
    """
    initial_limit = config.get_int("INITIAL_RUN_LIMIT", _DEFAULT_INITIAL_LIMIT, minimum=1)
    if adapters is None:
        adapters = sources.enabled_adapters()
    registry = Registry.load(registry_path)
    poll_result = _poll(registry, adapters=adapters, initial_limit=initial_limit)

    repo_root = notes_dir.parent
    missing: list[str] = []
    recovered: list[str] = []
    escalations: list[_Escalation] = []
    for sermon in registry.sermons():
        if sermon.state == "published":
            artifact = sermon.artifact_path
            if artifact is None or not (repo_root / artifact).exists():
                missing.append(sermon.guid)
                detail = f"published record has no artifact on disk at {artifact!r}"
                logger.warning("reconcile: %s — %s", sermon.guid, detail)
                escalations.append(_escalation(sermon, "MissingArtifact", detail))
            continue
        target = _stranded_by_retired_logic(sermon)
        if target is not None:
            error_class = sermon.runs[-1].error_class
            registry.recover(sermon.guid, target, now=now)
            recovered.append(sermon.guid)
            logger.info(
                "reconcile: recovered %s stranded by retired %s; reset failed → %s",
                sermon.guid,
                error_class,
                target,
            )

    registry.save()
    escalated = _send_escalations(escalations, send_fn=send_fn)
    awaiting = _check_awaiting_enclosure(registry, now=now, send_fn=send_fn)
    suspected_duplicate = _check_suspected_duplicate_audio(registry, now=now, send_fn=send_fn)
    registry.save()  # The switches write `notified_alerts`; persist it with everything else.
    stale_days, alert_failed = _check_publish_staleness(registry, now=now, send_fn=send_fn)
    red_streak_count, red_streak_alert_failed = _check_red_run_streak(
        send_fn=send_fn, fetch=runs_fetch
    )
    return ReconcileResult(
        deferred=poll_result.deferred,
        discovered=tuple(poll_result.discovered),
        flagged=tuple(poll_result.flagged),
        missing_artifacts=tuple(missing),
        recovered=tuple(recovered),
        escalated=escalated,
        publish_stale_days=stale_days,
        publish_stale_alert_failed=alert_failed,
        red_streak_count=red_streak_count,
        red_streak_alert_failed=red_streak_alert_failed,
        awaiting_enclosure=awaiting,
        suspected_duplicate_audio=suspected_duplicate,
    )


# --- Parallel fan-out/fan-in (ADR-0023) ------------------------------------
#
# The `run` command splits into three entry points the workflow drives across jobs:
# `plan_shards` (serial preflight — poll, emit the working set), `run_shard` (one
# parallel job per sermon — transcribe+generate into files + a `Delta`, no registry
# write), and `run_merge` (serial fan-in — the sole registry writer: apply each delta,
# render, publish, escalate, deliver). `reconcile`/`feed` stay serial and unchanged.

# The lifecycle states a shard still has work to do on: `discovered` needs
# transcribe+generate, `transcribed` needs generate. `generated` needs only render
# (the merge's job) and terminal states are done — none of those are sharded.
_SHARDABLE_STATES = frozenset({"discovered", "transcribed"})


def _awaiting_enclosure(sermon: SermonRecord) -> bool:
    """Whether ``sermon``'s next step is a download it cannot perform yet (ADR-0066).

    Some feeds publish an item before its audio finishes processing, leaving
    ``audio_url`` empty (spec 0019). Only ``discovered`` qualifies: every later state
    has its transcript and needs no audio.
    """
    return sermon.state == "discovered" and not audio.is_fetchable_enclosure(sermon.audio_url)


def _reuses_a_claimed_enclosure(sermon: SermonRecord, others: Sequence[SermonRecord]) -> bool:
    """Whether another of this source's records already claims this exact ``audio_url``
    and precedes it (ADR-0077, #517) — the cheap, exact-reuse half of the duplicate-audio
    withholding, stateless like :func:`_awaiting_enclosure` and requiring no download.

    "Precedes" is ``(published_on, guid)`` order, not discovery order: the earlier
    service date is the one whose audio this genuinely is (Squarespace re-serves a
    *previous* week's file, never a future one), so the later item is the one withheld
    regardless of which happened to be polled first.
    """
    if sermon.state != "discovered" or not sermon.audio_url:
        return False
    key = (sermon.published_on, sermon.guid)
    return any(
        other.guid != sermon.guid
        and other.source == sermon.source
        and other.audio_url == sermon.audio_url
        and (other.published_on, other.guid) < key
        for other in others
    )


def _suspected_duplicate_audio(sermon: SermonRecord) -> bool:
    """Whether ``sermon`` was already fingerprinted as a probable duplicate (ADR-0077, #517).

    The persisted half of the duplicate-audio withholding: once a shard has downloaded
    and fingerprinted this audio and found a match, the verdict lives on the record
    (``suspected_duplicate_of``) so every later poll withholds it for free, exactly like
    :func:`_awaiting_enclosure` — no re-download, no re-spent claim. ``Registry.upsert``
    clears the marker when the feed's ``audio_url`` for this guid actually changes,
    which is what re-admits the record here once that happens.
    """
    return sermon.state == "discovered" and sermon.suspected_duplicate_of is not None


# The state each shardable state is trying to reach — the ``attempted_state`` its run
# records carry. Attempts are counted against the step the sermon is stuck on, not its
# whole history, so clearing one step gives the next a fresh budget.
_ATTEMPTED_STATE = {
    "discovered": "transcribed",
    "transcribed": "generated",
    # A sermon waiting on a batch is still trying to reach `generated`; the detour
    # through `pending_batch` (ADR-0061) does not give it a fresh budget for the same
    # step it was already attempting when it was submitted.
    "pending_batch": "generated",
}

# The `--to` value _refused_delta suggests in its recovery command, keyed by the state
# the sermon was refused at. Every such state is already a valid `_RECOVERABLE_STATES`
# target (registry.py) except `pending_batch`, which recovery deliberately excludes —
# resuming from it could resurrect a batch id that no longer resolves to anything on
# Anthropic's side (ADR-0061). Recovery should resume from `transcribed` instead, the
# step the batch was submitted from (#457).
_REFUSAL_RECOVERY_TO = {
    "pending_batch": "transcribed",
}

# The env var carrying `github.event.inputs.limit` verbatim — blank on every scheduled
# run, because a `schedule:` event has no inputs (spec 0020, ADR-0061 §1). Deliberately
# NOT a `GITHUB_*` name: the runner owns that namespace and a workflow `env:` entry there
# renders into the log and never reaches the process (CLAUDE.md §10).
_BACKFILL_LIMIT_ENV = "BACKFILL_LIMIT"

# How long a submitted batch may go unresolved before the pipeline stops waiting for it
# and generates the sermon synchronously instead (ADR-0061 §4). Measured from
# `batch_submitted_at`, never from `last_state_change_at` — see the field's own comment
# in `registry`. Because it is only evaluated when a run happens to poll, the fallback
# can fire up to a cron gap late; backfills are defined as not time-sensitive.
_BATCH_FALLBACK_AFTER = timedelta(hours=24)

# How many attempts a sermon may have claimed before the shard refuses to spend on it
# again (#240, ADR-0033).
#
# PRD §11.1's retry posture is *within* a run. Between runs there was no bound at all: a
# non-terminal failure deliberately leaves the sermon shardable so the next run
# re-attempts it (ADR-0009 for a transient CDN rejection; ADR-0014 for a self-heal
# reset), and nothing ever stopped that repeating. On the Sunday cron — a run requested
# every 5 minutes — a sermon that never advances is re-downloaded, re-transcribed and
# re-generated on every tick, paying the LLM each time, for as long as the condition
# lasts. Three attempts is the between-run budget; the fourth is refused.
#
# The count comes from `attempt_claims`, not from the ledger. #235 spent money and then
# failed to record the spend, so any counter kept in `runs[]` reads zero for exactly as
# long as the outage lasts. A claim is written before the money moves and outside the
# path that broke, so the budget still runs down when every downstream write fails.
_DEFAULT_MAX_SERMON_ATTEMPTS = 3

# :data:`~sermon_notes.registry.ATTEMPT_BUDGET_ERROR_CLASS` — the class a
# budget-exhausted sermon goes terminal with — is defined in `registry`, because it is a
# value of a ledger field. The policy around it is this module's, and it is this:
#
# It must NEVER join :data:`RETIRED_FAILURE_CLASSES`: that set is what the nightly
# reconcile resets a record out of (ADR-0014), so a budget failure listed there would be
# un-stranded every night, re-attempted, and re-exhausted — the same unbounded loop with
# extra ceremony and a nightly email. A retired sermon resumes only by hand:
# ``scripts/recover_sermon.py`` to leave ``failed``, then a manual ``workflow_dispatch``,
# which is exempt from the cap.

# The class the merge escalates a run under when the cap could not be consulted at all
# (ADR-0036, #264). Not a sermon's failure — the sermons processed — so it never reaches
# the ledger; it names the run in which the spend guard was inert.
_CAP_UNENFORCED_ERROR_CLASS = "AttemptCapUnenforced"

# The class the merge escalates a run under when the plan refused to fan out on a rolling
# spend cap (ADR-0037, #265). Like the one above it names the run, not a sermon, so it
# never reaches the ledger — the refusal happens before any sermon is planned at all.
_SPEND_REFUSAL_ERROR_CLASS = "SpendGuardRefusal"

# The class the merge escalates a run under when every enabled source resolved to zero
# usable feeds (ADR-0074, #496). Like the two above it names the run, not a sermon.
_NO_SOURCES_ERROR_CLASS = "NoSourcesConfigured"

# The class the merge escalates a run under when fewer deltas arrived than the plan fanned
# out and a failed shard job explains the difference (#266). A run-level class like the two
# above: the sermons behind the missing deltas were never processed, so nothing about them
# reaches the ledger — they simply stay where they were, to be retried next run.
_DELTA_SHORTFALL_ERROR_CLASS = "ShardDeltaShortfall"

# The class the merge escalates a run under when it produced artifacts it could not push
# (#269). Run-level like the three above — the sermons were processed correctly and the
# ledger recording that is precisely what did not survive.
_ARTIFACT_PUSH_ERROR_CLASS = "ArtifactPushAbandoned"

# How long a condition-level alert (the three classes above) stays quiet after firing for
# an unchanged occurrence, before ADR-0046's dedup lets it send again (#324, #323). A
# worsening occurrence — a changed fingerprint — always sends immediately regardless.
_DEFAULT_ALERT_COOLDOWN_HOURS = 24

# The subjects that escalation carries — one per outcome, because the two are read very
# differently at 2am: one is a recovery to perform, the other is a loss to absorb. Public
# because the caller is the workflow's `Commit artifacts` step rather than a pipeline
# pass, so these strings are the only handle a test of that step has on which alert it
# actually sent.
ARTIFACT_PUSH_FAILURE_TITLE = "Artifact push abandoned — this run's work is preserved for replay"
ARTIFACT_PUSH_LOSS_TITLE = "Artifact push abandoned — this run's work is lost"

# Where the merge job uploads what it could not push, and what lands it afterwards (#315,
# ADR-0045). Named in the alert because an operator reading it hours later has no other
# way to find either.
ARTIFACT_PRESERVATION_NAME = "unpushed-artifacts"
ARTIFACT_REPLAY_SCRIPT = "scripts/replay_unpushed_artifacts.py"


@dataclass(frozen=True)
class _SpendWindow:
    """One rolling window :func:`plan_shards` measures recorded LLM spend over (#241, #256).

    Every window is rolling, deliberately — a calendar reset hands a loop that starts at
    23:00 two full budgets back to back, which is the granularity mistake that makes the
    provider's monthly ceiling useless as a runaway detector in the first place.
    """

    label: str  # how the window is named in a log line and in a refusal
    period: str  # the noun in "far more than a normal <period>'s volume"
    env_var: str  # the config key overriding this window's cap
    span: timedelta
    default_usd: float


# The windows the guard enforces, narrowest first — checked in that order, so a run
# that busts both is refused naming the tighter and more specific of the two. Both
# defaults are sized from measured per-sermon cost and legitimate weekly volume, not
# from the #235 incident's burn — see ADR-0058, its ADR-0075 amendment (six sources,
# #461), and ADR-0084's correction (Sonnet 5's list price, #541) for the full
# derivation. ADR-0084 sizes each cap at the worst-legitimate sermon count times the
# *mean* per-sermon cost, rounded to the nearest half dollar, rather than pricing that
# count at the single costliest sermon ever billed or padding it with a coverage
# margin on top: the worst-legitimate count is itself a tail assumption that has never
# actually occurred in the ledger's history, so compounding it with a second layer of
# cost-tail safety is redundant, not conservative.
_SPEND_WINDOWS = (
    _SpendWindow(
        label="24h",
        period="day",
        env_var="LLM_DAILY_BUDGET_USD",
        span=timedelta(hours=24),
        default_usd=1.00,
    ),
    _SpendWindow(
        label="7d",
        period="week",
        env_var="LLM_WEEKLY_BUDGET_USD",
        span=timedelta(days=7),
        default_usd=1.50,
    ),
)


class SpendGuardError(RuntimeError):
    """Raised when the plan refuses to fan out on a rolling spend window (#241, #256).

    Covers every refusal the guard can make, against either window: the window total is
    at or above the cap, or the total cannot be trusted — because a model in the window
    has no list price, or because a call in the window exhausted its retries with its
    billing status unknown (#279). The message names which window refused.

    ``resolvable`` is the working set the refusal does *not* cover: a ``pending_batch``
    sermon whose Batches API request has already ended costs nothing to retrieve, because
    the money was committed when it was submitted (spec 0020). Refusing it would strand
    work already paid for without saving a cent, so the plan still fans those out while
    everything that would spend — a fresh submission, the 24h synchronous fallback,
    transcription, an ordinary generation — stays refused. Empty on every refusal that
    has no such sermon, which is every refusal outside a backfill.
    """

    def __init__(self, message: str, *, resolvable: Sequence[SermonRecord] = ()) -> None:
        super().__init__(message)
        self.resolvable = list(resolvable)


class NoSourcesConfiguredError(RuntimeError):
    """Raised when every source :func:`sources.enabled_source_names` named resolved to zero.

    ``sources.enabled_adapters()`` skips (logs, doesn't raise) any one source
    whose ``CHURCHES`` entry is missing or malformed (ADR-0074) — that per-
    source skip is the intended fix for issue #496 and needs no escalation of
    its own. This is the run-level condition on top of it: at least one church
    was enabled (in ``CHURCHES``, narrowed by a manual dispatch's
    ``ENABLED_SOURCES`` override if one was given), and *none* of them
    resolved, which is indistinguishable at the poll from "nothing to do"
    unless raised here — exactly the silent-forever failure mode the
    per-source skip would otherwise create.

    ``resolvable`` mirrors :class:`SpendGuardError`'s: work this refusal does
    not cover. Discovering *new* sermons needs a feed; sermons already in a
    shardable state, or a pending batch that has ended or expired, need
    neither — refusing them too would strand work already in flight over a
    condition that has nothing to do with it.
    """

    def __init__(self, message: str, *, resolvable: Sequence[SermonRecord] = ()) -> None:
        super().__init__(message)
        self.resolvable = list(resolvable)


def _window_spend(
    registry: Registry, *, since: datetime, until: datetime
) -> tuple[float, list[str], list[str]]:
    """Recorded LLM spend in ``[since, until]``, and what makes that total a lower bound.

    Sums :attr:`~sermon_notes.registry.RunRecord.llm_cost_usd` across every sermon's
    runs whose ``started_at`` falls in the window, and separately collects two signals
    that the sum understates the real bill:

    - any ``llm_model`` in the window with no :data:`~sermon_notes.llm_client.MODEL_PRICING`
      entry. Those runs cost ``0.0`` in the ledger by design (`llm_client.cost_usd`
      refuses to fail a sermon after the paid call already succeeded, #128).
    - the guid of any sermon whose ``generated`` attempt in the window ended
      ``failed_terminal`` with ``error_class`` naming
      :class:`~sermon_notes.llm_client.RetriesExhaustedError`. That terminal record
      carries the dataclass defaults (``llm_cost_usd=0.0``) because the call that
      exhausted its retries reported no usable token count — the request may or may not
      have been billed, so there is nothing truthful to sum (#279).

    Either signal means the caller needs to know the sum is a lower bound rather than
    read it as "under budget".

    Bounded at both ends, not just at ``since``. On a live run the two are equivalent —
    nothing in the ledger is stamped later than now — but a one-sided window makes the
    sum unreplayable against any past instant, which is how the guard gets checked
    against real history when its cap is tuned. It also means a single record stamped in
    the future (a skewed runner clock, a hand-edited ledger) would otherwise count
    against every window from then on, with no way to age out.
    """
    total = 0.0
    unpriced: set[str] = set()
    retries_exhausted: list[str] = []
    for sermon in registry.sermons():
        for run in sermon.runs:
            if not since <= parse_instant(run.started_at) <= until:
                continue
            total += run.llm_cost_usd
            if run.llm_model is not None and run.llm_model not in llm_client.MODEL_PRICING:
                unpriced.add(run.llm_model)
            if run.error_class == llm_client.RetriesExhaustedError.__name__:
                retries_exhausted.append(sermon.guid)
    return total, sorted(unpriced), retries_exhausted


def _spend_budget_refusal(registry: Registry, *, now: str | None) -> str | None:
    """The message to refuse this run with, or ``None`` when rolling spend is in budget.

    Returns rather than raises so the caller can decide what the refusal covers: since
    ADR-0061 a run can carry work that costs nothing (retrieving an already-billed batch
    result), and a guard that raises past its own call site cannot express that.
    :func:`plan_shards` is the one caller and it raises :class:`SpendGuardError`.

    The run-level companion to the per-sermon attempt cap. Claims bound how many times
    we pay for *one* sermon; they say nothing about dollars, so a model swap that costs
    ten times as much, a pricing change, or a retry storm spread across many sermons all
    stay inside the attempt cap while the bill multiplies. This bounds the bill directly.

    Two windows, because one granularity cannot see both failures. The 24h cap bounds a
    day-of surge and is blind to the days around it: a fault that spends half a day's
    budget every day reports green on every single run. The 7d cap bounds that, and is in
    turn too coarse to catch a single runaway afternoon. Each window is measured and
    enforced independently against its own cap, narrowest first, so a run that busts both
    is refused naming the tighter and more specific of the two.

    Enforced here rather than in the shard because a shard never opens the registry
    (ADR-0023) and so cannot see a global total; the plan already loads the whole ledger
    read-only and already decides what work happens. The refusal is a raise, not a drop:
    failing the plan job leaves no matrix, skips every shard, and skips the merge, so the
    run ends red with nothing spent and no record silently stranded.

    An unpriced model, or a call that exhausted its retries with an unknown bill, in a
    window refuses too. "Under budget" and "unmeasurable" must not return the same
    answer at the call site — a pricing-table gap, or a billed-but-uncounted timeout,
    would otherwise widen the cap to infinity for the affected runs while the guard
    reported green. Either check is per window, so a gap or an exhausted retry that the
    24h window can no longer see still refuses on the 7d one it remains inside: a week
    whose total is a lower bound has no enforceable cap either.

    A manual ``workflow_dispatch`` is exempt from every window, exactly as it is from the
    attempt cap: both guards govern *unattended* spend, and a human asking for a run has
    already decided to spend. That exemption is what keeps backfill possible at all: a
    backfill is the one legitimate way to exceed these caps on purpose. Seeding three
    sources at the default ``INITIAL_RUN_LIMIT`` came to $0.92 in a day and $1.20 across
    the week it fell in — under today's six-source $1.00 daily / $1.50 weekly caps
    (ADR-0084) with little room to spare, but a wider ``limit`` or more sources
    backfilled at once is not bounded the same way. Without the exemption the only way
    to run one would be to merge a change raising the defaults,
    backfill, then merge another lowering them. Every window total is logged either way,
    so a dispatched run still exercises the measurement rather than skipping it — a guard
    whose arithmetic only runs on the path that refuses is a guard nothing checks until
    the run where it matters.
    """
    at = parse_instant(now) if now is not None else datetime.now(timezone.utc)
    measured = []
    for window in _SPEND_WINDOWS:
        budget = config.get_float(window.env_var, window.default_usd, minimum=0.0)
        spent, unpriced, retries_exhausted = _window_spend(
            registry, since=at - window.span, until=at
        )
        logger.info(
            "plan: $%.2f of LLM spend recorded in the %s to %s, against the %s cap of $%.2f",
            spent,
            window.label,
            at.isoformat(),
            window.env_var,
            budget,
        )
        measured.append((window, budget, spent, unpriced, retries_exhausted))

    if _dispatch_is_manual():
        logger.info("plan: manual dispatch — the run is exempt from every spend cap")
        return None

    for window, budget, spent, unpriced, retries_exhausted in measured:
        if unpriced:
            return (
                f"the rolling {window.label} spend window contains run(s) produced by a model "
                f"with no entry in llm_client.MODEL_PRICING: {', '.join(unpriced)}. Those runs "
                f"were recorded at $0.00, so the window total of ${spent:.2f} is a lower bound "
                f"and the {window.env_var} cap of ${budget:.2f} cannot be enforced against it. "
                "Refusing to fan out. Add the model's list price to MODEL_PRICING (see PRD §12) "
                "and re-run."
            )
        if retries_exhausted:
            return (
                f"the rolling {window.label} spend window contains {len(retries_exhausted)} "
                f"sermon(s) whose LLM call exhausted its retries with the last attempt's "
                f"billing unknown: {', '.join(retries_exhausted)}. Those runs were recorded at "
                f"$0.00, so the window total of ${spent:.2f} is a lower bound and the "
                f"{window.env_var} cap of ${budget:.2f} cannot be enforced against it. Refusing "
                "to fan out. The window clears on its own as these runs age out; re-run by hand "
                "if the spend is known to be safe — a workflow_dispatch is exempt from this cap."
            )
        if spent >= budget:
            return (
                f"${spent:.2f} of LLM spend is recorded in the rolling {window.label} window, at "
                f"or above the {window.env_var} cap of ${budget:.2f}. Refusing to fan out: no "
                "sermon is planned and nothing is spent this run. The window is rolling, so it "
                "clears on its own as the runs that filled it age out. If the spend is "
                "legitimate — a backfill, a widened INITIAL_RUN_LIMIT — re-run by hand: a "
                "workflow_dispatch is exempt from this cap. Investigate first if it is not: at "
                f"roughly $0.09 a sermon, ${spent:.2f} is far more than a normal "
                f"{window.period}'s volume."
            )
    return None


def _batch_expired(sermon: SermonRecord, *, at: datetime) -> bool:
    """Whether ``sermon``'s batch is past the point of being worth waiting for.

    ``batch_submitted_at`` is the clock, not ``last_state_change_at``: only the former
    is pinned to the submission the deadline is about (see the field's comment in
    ``registry``). A ``pending_batch`` record carrying neither a batch id nor a
    submission time cannot be polled or timed at all — there is no state in which the
    ledger should hold one, so it is treated as expired and completed synchronously
    rather than left to wait on a clock that does not exist.
    """
    if sermon.batch_id is None or sermon.batch_submitted_at is None:
        logger.error(
            "sermon %s is pending_batch with no batch id or submission time; "
            "generating it synchronously instead of waiting",
            sermon.guid,
        )
        return True
    return at - parse_instant(sermon.batch_submitted_at) > _BATCH_FALLBACK_AFTER


def _poll_pending_batches(
    registry: Registry,
    *,
    now: str | None,
    poll_batch: Callable[[str], BatchStatus],
) -> tuple[list[str], list[str]]:
    """``(guids whose batch has ended, guids past the fallback deadline)`` (spec 0020).

    The cross-run half of ADR-0061, and the reason polling lives in ``plan`` rather than
    in a second scheduled workflow: this job already runs on every tick and is the only
    one that can decide a sermon is *not* worth a runner. That decision is the point. A
    ``pending_batch`` sermon fanned out while its batch is still processing would claim
    one of ``MAX_SERMON_ATTEMPTS`` (ADR-0033) and do nothing with it, so a batch that
    took its full day would retire the sermon it was about to complete.

    An expired batch is never polled: its answer no longer decides anything, and the
    shard falls back to a synchronous call either way. A poll that fails with an
    :class:`~sermon_notes.llm_client.LLMError` leaves the sermon where it is — the next
    run polls it again, and the deadline bounds how long that can repeat. Anything the
    boundary does not map (auth, a 5xx) aborts the run loudly, as it does everywhere
    else in this pipeline.
    """
    at = parse_instant(now) if now is not None else datetime.now(timezone.utc)
    ended: list[str] = []
    expired: list[str] = []
    for sermon in registry.sermons():
        if sermon.state != "pending_batch":
            continue
        if _batch_expired(sermon, at=at):
            logger.info(
                "plan: %s has waited longer than %s on batch %s; planning the synchronous fallback",
                sermon.guid,
                _BATCH_FALLBACK_AFTER,
                sermon.batch_id,
            )
            expired.append(sermon.guid)
            continue
        assert sermon.batch_id is not None  # _batch_expired returns True without one
        try:
            status = poll_batch(sermon.batch_id)
        except LLMError as exc:
            logger.warning(
                "plan: could not poll batch %s for %s (%s); leaving it pending",
                sermon.batch_id,
                sermon.guid,
                exc,
            )
            continue
        logger.info(
            "plan: batch %s for %s is %s", sermon.batch_id, sermon.guid, status.processing_status
        )
        if status.ended:
            ended.append(sermon.guid)
    return ended, expired


def plan_shards(
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    adapters: Sequence[SourceAdapter] | None = None,
    now: str | None = None,
    poll_batch: Callable[[str], BatchStatus] = llm_client.poll_batch,
) -> list[SermonRecord]:
    """Preflight: poll every source and return the sermons needing shard work (ADR-0023).

    Loads the committed ledger, re-polls (idempotent ``upsert`` — ADR-0013) to fold in
    newly-published sermons, and returns every record in a :data:`_SHARDABLE_STATES`
    state — the fan-out working set, one shard per record. It deliberately does **not**
    persist the ledger: the plan job never commits (the merge is the sole writer), and
    each returned record carries the metadata a shard needs, which the merge re-seeds
    from the delta. ``now`` fixes the clock the spend windows are measured back from.

    The plan applies no *per-sermon* cap (ADR-0033). It cannot: it never writes the
    ledger, so the only outcome available to it against one record is a silent drop,
    which leaves that record live to be rediscovered, re-judged and re-alerted on every
    subsequent run. The attempt cap therefore lives in :func:`run_shard`, which can
    refuse to spend *and* hand back a terminal delta that retires the sermon for good.

    It does withhold a sermon still awaiting its audio enclosure (ADR-0066), which is
    that same silent drop and is nonetheless right here. The objection above is to
    dropping a record the plan has *judged*: the verdict is lost with it, so the next
    run re-reaches it and re-alerts. A record with no enclosure is judged by nobody and
    alerts nothing — being rediscovered every run is exactly the intended behaviour, and
    the next poll's ``upsert`` is what heals it. Withholding has to happen here because
    the workflow uploads the attempt claim *before* :func:`run_shard` is reached, so
    anything downstream can decline the spend but not the claim, and three polls of an
    audio-less sermon retire it having fetched nothing. A manual dispatch overrides the
    withholding, as it overrides the cap. :func:`reconcile` bounds the wait.

    What the plan *can* enforce is the run-level one, because refusing the whole run
    needs no per-record verdict and no ledger write: :func:`_enforce_spend_budget` runs
    first and raises :class:`SpendGuardError` when the rolling 24h or 7d LLM spend is at
    its cap (#241, #256), before the poll touches a feed. Both guards govern unattended
    spend and
    step aside for a manual dispatch, and both read only config and the committed ledger,
    so the plan job still holds no Actions token and no mail credentials. It does hold
    ``ANTHROPIC_API_KEY`` since ADR-0061, because :func:`_poll_pending_batches` below
    reads the Batches API — the one boundary this job reaches.

    Pending batches are polled before the feeds are (spec 0020) and widen the working set
    with the sermons whose results are ready, or whose 24h deadline has passed. They
    never narrow it: an unresolved batch simply is not planned this run.
    """
    initial_limit = config.get_int("INITIAL_RUN_LIMIT", _DEFAULT_INITIAL_LIMIT, minimum=1)
    adapters_requested = adapters is None
    if adapters is None:
        adapters = sources.enabled_adapters()
    registry = Registry.load(registry_path)
    ended, expired = _poll_pending_batches(registry, now=now, poll_batch=poll_batch)

    refusal = _spend_budget_refusal(registry, now=now)
    if refusal is not None:
        # Retrieving an ended batch is free, so the refusal does not cover it (see
        # SpendGuardError). `expired` is deliberately excluded: its fallback is an
        # ordinary synchronous call, which is exactly what the cap governs. `ended` is
        # monotonic, so a sermon planned here can only take the free path in the shard.
        raise SpendGuardError(
            refusal, resolvable=[s for s in registry.sermons() if s.guid in ended]
        )

    # Only when the plan resolved its own adapters (ADR-0074, #496): an explicit
    # `adapters=` injection is the caller's own controlled scenario, not a config
    # problem to escalate. `sources.enabled_source_names()` is which churches
    # CHURCHES marked enabled (narrowed by a dispatch's ENABLED_SOURCES override,
    # if any); an empty `adapters` despite a non-empty request means every one of
    # them was skipped for a missing or malformed rss url — the condition that
    # would otherwise poll nothing forever with no signal.
    if adapters_requested and not adapters and sources.enabled_source_names():
        raise NoSourcesConfiguredError(
            "A church is enabled in CHURCHES, but none of them has a usable rss url.",
            resolvable=[
                s for s in registry.sermons() if s.state in _SHARDABLE_STATES or s.guid in ended
            ],
        )

    _poll(registry, adapters=adapters, initial_limit=initial_limit)

    ready = set(ended) | set(expired)
    working = [s for s in registry.sermons() if s.state in _SHARDABLE_STATES or s.guid in ready]
    if _dispatch_is_manual():
        return working
    all_sermons = registry.sermons()
    planned: list[SermonRecord] = []
    for sermon in working:
        if _awaiting_enclosure(sermon):
            logger.info(
                "plan: %s has no audio enclosure yet; withholding it from the fan-out",
                sermon.guid,
            )
            continue
        if _reuses_a_claimed_enclosure(sermon, all_sermons):
            logger.info(
                "plan: %s's enclosure is already claimed by an earlier sermon; "
                "withholding it from the fan-out (ADR-0077, #517)",
                sermon.guid,
            )
            continue
        if _suspected_duplicate_audio(sermon):
            logger.info(
                "plan: %s's audio was already fingerprinted as a probable duplicate of "
                "%s; withholding it from the fan-out (ADR-0077, #517)",
                sermon.guid,
                sermon.suspected_duplicate_of,
            )
            continue
        planned.append(sermon)
    return planned


def _dispatch_is_manual() -> bool:
    """Whether this run was started by hand rather than by the cron (ADR-0031, ADR-0033).

    The attempt cap governs *unattended* spend. A human dispatching a run has already
    decided to spend, so the cap steps aside — which is what keeps backfill working and
    what makes a retired sermon resumable: recover the record out of ``failed``, then
    dispatch a run by hand.
    """
    return config.get("GITHUB_EVENT_NAME", "") == "workflow_dispatch"


def _claimed_attempts(sermon: SermonRecord) -> tuple[int, str | None]:
    """``(claims on sermon, why the cap is not in force)`` — the reason is ``None`` if it is.

    Never blocks on the boundary being unavailable. Off Actions, or when the claim
    listing cannot be read after its retries, the count is ``UNKNOWN`` and the sermon
    proceeds — a guard's dependency must not be able to halt the pipeline, and the worst
    case is one run behaving as it did before the cap existed (ADR-0031, kept by
    ADR-0033).

    Only the *error* path yields a reason. Off Actions the cap is dormant by design and
    :func:`~sermon_notes.attempt_claims.claims_for` reports that at INFO without raising;
    alerting on it would mail on every local run. An unreadable listing on a runner is a
    fault, and the reason is what the merge escalates once for the run (ADR-0036, #264) —
    without it the guard's own failure is invisible: green run, zero exit, no mail.
    """
    try:
        return attempt_claims.claims_for(sermon.guid), None
    except attempt_claims.AttemptClaimsError as exc:
        reason = (
            f"could not read the attempt claims for {sermon.guid} ({exc}); it was processed "
            "without the attempt cap in force"
        )
        logger.error("shard: %s", reason)
        return attempt_claims.UNKNOWN, reason


def _budget_spent(sermon: SermonRecord, *, claimed: int, max_attempts: int) -> int | None:
    """The claim count when ``sermon`` is at or past its budget, else ``None``.

    The current run's own claim is uploaded by the workflow *before* this runs, so the
    count already includes the attempt about to be spent: at ``max_attempts`` of 3, a
    sermon on its third attempt reads 3 and is refused, having been processed twice. That
    is the conservative direction, and pinning it here keeps the arithmetic in one place.

    The count is read and logged on *every* run, including a manual dispatch that is
    exempt from the refusal. Exempting the read too would leave the boundary exercised
    only when it is about to retire a sermon — so a broken listing (a revoked scope, a
    changed endpoint) would stay invisible until the one run where the cap mattered, and
    would then fail open silently. Reading it always makes a dispatched run a genuine
    check of the boundary, and the log line the evidence — which is what the fan-out
    smoke's shard now leans on to exercise the counting half outside production
    (ADR-0036): it pins ``GITHUB_EVENT_NAME`` to a dispatch so the count is read and
    logged for real while the refusal steps aside.
    """
    logger.info(
        "shard: %s has %s claimed attempt(s) against a cap of %d",
        sermon.guid,
        "an unreadable number of" if claimed == attempt_claims.UNKNOWN else claimed,
        max_attempts,
    )
    if _dispatch_is_manual():
        logger.info("shard: manual dispatch — %s is exempt from the cap", sermon.guid)
        return None
    if claimed == attempt_claims.UNKNOWN or claimed < max_attempts:
        return None
    return claimed


def _seed_cached_transcript(
    sermon: SermonRecord, *, committed_dir: Path, staging_dir: Path
) -> None:
    """Copy ``sermon``'s already-committed transcript into the shard's staging tree (#228).

    A shard writes into a per-runner staging tree so its artifact upload carries only the
    files it produced rather than the whole committed archive (#202). That tree starts
    empty, which leaves the committed ``transcripts/`` cache invisible to the stages: a
    ``transcribed`` sermon would find no transcript to generate from and go terminal, and a
    ``discovered`` one would re-download the enclosure and re-run whisper for a transcript
    already in the repo. Copying the one relevant entry across restores the cache
    semantics of PRD §11.3 while keeping the upload scoped — the staged copy is
    byte-identical, so the merge committing it back is a no-op.

    A no-op when the two directories resolve to the same file (the serial and local
    default, where staging *is* the committed tree) or when nothing is cached yet.
    """
    source = cache_path(committed_dir, sermon)
    dest = cache_path(staging_dir, sermon)
    if dest == source or not source.is_file():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    logger.info("seeded staged transcript for %s from the committed cache", sermon.guid)


def _refused_delta(
    sermon: SermonRecord, *, attempts: int, max_attempts: int, now: str | None
) -> Delta:
    """The delta a shard returns instead of spending a spent sermon's budget (#240).

    Carries a single terminal run at the step the sermon is stuck on, so the merge applies
    it through the ordinary terminal path (:data:`~sermon_notes.shard._ADVANCE_ON` sends
    ``attempted_state``/``failed_terminal`` to ``failed``) and escalates it exactly once,
    like any other terminal failure. Going terminal — rather than being dropped from the
    working set — is what ends the loop: ``failed`` is not shardable, so the next run does
    not re-plan the sermon at all, and the ledger records why it stopped.

    ``transcript_hash`` is ``None`` because this shard produced no transcript. That is the
    honest value and it also keeps the merge's cross-guid dedup guard out of the way, so
    the record's terminal failure is the budget, not an incidental ``DuplicateTranscript``.
    """
    at = now if now is not None else _now()
    target = _ATTEMPTED_STATE[sermon.state]
    recover_to = _REFUSAL_RECOVERY_TO.get(sermon.state, sermon.state)
    detail = (
        f"{attempts} attempt(s) have been claimed on this sermon at {sermon.state} → "
        f"{target} without it advancing, at or past the cap of {max_attempts} "
        "(MAX_SERMON_ATTEMPTS). Refusing to spend transcription or LLM budget on it again. "
        "Investigate the earlier runs in the ledger and the shard job logs, then resume it "
        f"by hand: `python scripts/recover_sermon.py {sermon.guid} --to {recover_to}`, "
        "then re-run the pipeline via workflow_dispatch — a manual run is exempt from this "
        "cap. Both steps are needed; recovering alone leaves the claims spent."
    )
    logger.error("shard: refusing to process %s — %s", sermon.guid, detail)
    return Delta(
        sermon=copy.deepcopy(sermon),
        transcript_hash=None,
        reached_state="failed",
        runs=[
            RunRecord(
                attempted_state=target,
                outcome="failed_terminal",
                error_class=ATTEMPT_BUDGET_ERROR_CLASS,
                error_detail=detail,
                started_at=at,
                finished_at=at,
            )
        ],
    )


def _is_backfill() -> bool:
    """Whether this run is a backfill, and so generates through the Batches API.

    Two conditions, both of which a ``schedule:`` trigger fails: the run was started by
    hand (the runner's own ``GITHUB_EVENT_NAME``), and the dispatch set ``limit``
    (:data:`_BACKFILL_LIMIT_ENV`, which carries that input verbatim). ADR-0061 §1 chose
    ``limit`` as the signal because it already *is* the backfill signal in this
    workflow, rather than adding a second input that has to be kept in sync with it.

    The event name is checked as well as the input, so a repository variable or a stale
    export named the same thing can never route the weekly cron onto a path that
    publishes a day late — the cron's latency guarantee is the thing being protected.
    """
    return _dispatch_is_manual() and bool(config.get(_BACKFILL_LIMIT_ENV, ""))


def _is_batch_source(sermon: SermonRecord) -> bool:
    """Whether ``sermon``'s source generates through the Batches API on every run.

    Independent of :func:`_is_backfill`: a church whose ``CHURCHES`` entry sets
    ``"api": "batch"`` (ADR-0063, generalized from a hardcoded two-source set to
    config by ADR-0074) is batch-eligible on a scheduled cron tick, not only during
    a manual backfill dispatch. The attempt cap grants a sermon only two real
    dispatches, not three (``_budget_spent``), so "submit a batch, then
    resolve-or-fall-back-to-sync on the sermon's last real attempt" is what both a
    backfill and a regular batch-mode church's run share — only the trigger
    differs. A church absent from ``CHURCHES``, or whose ``api`` is anything but
    ``"batch"``, defaults to ``"normal"`` (:func:`church_config.api_mode`) — the
    safe direction, matching what ADR-0061 shipped for Menlo/PBC before this.
    """
    return church_config.api_mode(sermon.source) == "batch"


def _raising_call(error: LLMError) -> Callable[[str, str], LLMResponse]:
    """An ``llm_call`` seam that raises ``error`` when the generate stage reaches it.

    A batch whose result is a failure has to fail *as a generation* — recorded on the
    run, sent terminal once, escalated once — and the generate stage is what does all
    three. Handing it a seam that raises is how a per-item batch failure joins the
    existing per-sermon isolation instead of getting a path of its own (ADR-0019).
    """

    def call(_system: str, _user: str) -> LLMResponse:
        raise error

    return call


def _batch_result_call(
    sermon: SermonRecord,
    *,
    retrieve_batch: Callable[[str], list[BatchResult]],
) -> Callable[[str, str], LLMResponse] | None:
    """The ``llm_call`` seam handing the generate stage this sermon's batch completion.

    ``None`` means the result could not be fetched *this run* and nothing was billed:
    the sermon stays ``pending_batch`` and the next plan polls it again. Every other
    outcome — the batch rejected outright, this sermon's item failed, or the results
    carry no item for it at all — is decided here and re-raised inside the seam, so it
    lands in the ledger as a generation failure rather than as a crashed shard job.
    """
    assert sermon.batch_id is not None  # only reached for a record carrying one
    try:
        results = retrieve_batch(sermon.batch_id)
    except TransientLLMError as exc:
        logger.warning(
            "shard: could not retrieve batch %s for %s (%s); leaving it pending",
            sermon.batch_id,
            sermon.guid,
            exc,
        )
        return None
    except PermanentLLMError as exc:
        return _raising_call(exc)

    for result in results:
        if result.custom_id != llm_client.batch_custom_id(sermon.guid):
            continue
        response = result.response
        if response is not None:
            return lambda _system, _user: response
        return _raising_call(
            PermanentLLMError(
                f"batch {sermon.batch_id} returned no completion for {sermon.guid}: {result.error}"
            )
        )
    return _raising_call(
        PermanentLLMError(f"batch {sermon.batch_id} carried no result for {sermon.guid}")
    )


def _submit_batch_stage(
    registry: Registry,
    guid: str,
    *,
    submit_batch: Callable[[Sequence[BatchRequest]], str],
    transcripts_dir: Path,
    now: str | None,
    escalations: list[_Escalation],
) -> tuple[str | None, str | None]:
    """Submit this sermon's generation as a batch; return ``(batch id, submitted at)``.

    The backfill replacement for :func:`_generate_stage`, and the reason submission
    lands in the shard rather than in the plan as ADR-0061 first sketched: a batch
    request carries the transcript, which does not exist until this job has produced it,
    and the plan never writes the ledger, so it has no way to record a submission
    (ADR-0023). The delta this shard hands back is exactly the channel for a state
    change the merge must persist.

    A failed submission is a failed generation attempt — same terminal record, same
    escalation, same ``attempted_state`` — so the attempt cap keeps counting the step
    the sermon is actually stuck on. ``(None, None)`` on that path.
    """
    sermon = _require_record(registry, guid)
    at = now if now is not None else _now()
    try:
        batch_id = submit_note_batch(
            registry,
            guid,
            submit_fn=submit_batch,
            transcripts_dir=transcripts_dir,
            now=at,
        )
    except (GenerationError, LLMError) as exc:
        _go_terminal(
            registry,
            sermon,
            "generated",
            error_class=type(exc).__name__,
            error_detail=str(exc),
            now=now,
        )
        escalations.append(_escalation(sermon, type(exc).__name__, str(exc)))
        return None, None

    registry.advance(guid, "pending_batch", now=at)
    return batch_id, at


def _resolve_batch_stage(
    registry: Registry,
    guid: str,
    *,
    poll_batch: Callable[[str], BatchStatus],
    retrieve_batch: Callable[[str], list[BatchResult]],
    llm_call: Callable[[str, str], LLMResponse],
    transcripts_dir: Path,
    notes_dir: Path,
    now: str | None,
    escalations: list[_Escalation],
) -> None:
    """Turn a resolved (or timed-out) batch into a generated note (spec 0020).

    The plan already polled this sermon to decide it was worth a runner; this polls
    again rather than inferring what the plan concluded, so the shard is correct on its
    own and stays correct if the plan's rule ever changes. A status read costs nothing.

    Past :data:`_BATCH_FALLBACK_AFTER` the batch is abandoned and the note is generated
    synchronously, which is ADR-0061's guarantee that batch mode can never do worse than
    the path it replaces — worst case a backfill costs full price and takes longer, never
    fails where the synchronous call would have succeeded.
    """
    sermon = _require_record(registry, guid)
    at = parse_instant(now) if now is not None else datetime.now(timezone.utc)

    if _batch_expired(sermon, at=at):
        logger.warning(
            "shard: batch %s for %s is past the %s deadline; generating it synchronously",
            sermon.batch_id,
            guid,
            _BATCH_FALLBACK_AFTER,
        )
        call = llm_call
    else:
        assert sermon.batch_id is not None  # _batch_expired returns True without one
        try:
            ended = poll_batch(sermon.batch_id).ended
        except LLMError as exc:
            logger.warning(
                "shard: could not poll batch %s for %s (%s); leaving it pending",
                sermon.batch_id,
                guid,
                exc,
            )
            return
        if not ended:
            logger.info(
                "shard: batch %s for %s has not ended; nothing to do", sermon.batch_id, guid
            )
            return
        resolved = _batch_result_call(sermon, retrieve_batch=retrieve_batch)
        if resolved is None:
            return
        call = resolved

    _generate_stage(
        registry,
        llm_call=call,
        transcripts_dir=transcripts_dir,
        notes_dir=notes_dir,
        now=now,
        escalations=escalations,
        from_state="pending_batch",
    )


def run_shard(
    sermon: SermonRecord,
    *,
    transcripts_dir: Path = DEFAULT_TRANSCRIPTS_DIR,
    notes_dir: Path = DEFAULT_NOTES_DIR,
    committed_transcripts_dir: Path = DEFAULT_TRANSCRIPTS_DIR,
    committed_registry_path: Path = DEFAULT_REGISTRY_PATH,
    transcribe: Callable[[Path], str] = default_transcribe,
    download: Callable[[str, Path], None] = audio.http_download,
    llm_call: Callable[[str, str], LLMResponse] = llm_client.call,
    submit_batch: Callable[[Sequence[BatchRequest]], str] = llm_client.submit_batch,
    poll_batch: Callable[[str], BatchStatus] = llm_client.poll_batch,
    retrieve_batch: Callable[[str], list[BatchResult]] = llm_client.retrieve_batch,
    sleep: Callable[[float], None] = time.sleep,
    now: str | None = None,
) -> Delta:
    """Transcribe and generate one sermon into deterministic files, returning a ``Delta``.

    Runs the same transcribe and generate stages the serial pipeline uses, but against a
    throwaway single-record ledger at a temp path — so the deterministic artifacts
    (``transcripts/…txt``, ``notes/…json``) land under the real ``transcripts_dir`` /
    ``notes_dir`` for upload, while **no write ever touches the committed
    ledger** (the merge is the sole registry writer, ADR-0023). The
    sermon's resulting ``transcript_hash``, reached state, and run records (telemetry and
    all) are captured in a :class:`~sermon_notes.shard.Delta` for the merge to replay.

    ``transcripts_dir`` is where this shard *writes*; ``committed_transcripts_dir`` is the
    repo's existing archive it may *read* an earlier transcript from. The fan-out job
    points the first at a staging tree and leaves the second at the checkout, so the
    upload stays scoped without the transcript cache going cold (#228, see
    :func:`_seed_cached_transcript`). They default to the same directory, which makes the
    seeding a no-op for a serial or local run.

    Before any of that, the attempt budget is checked: a sermon at or past
    ``MAX_SERMON_ATTEMPTS`` *claimed* attempts spends nothing and gets a terminal
    ``AttemptBudgetExceeded`` delta instead (ADR-0033, see :func:`_refused_delta`). The
    claim for this run is uploaded by the workflow before this function is reached, so the
    budget runs down even when the ledger never records the attempt — which is the failure
    that made #235 unbounded.

    On a backfill (:func:`_is_backfill`), or for a source ``CHURCHES`` marks
    ``"api": "batch"`` (:func:`_is_batch_source`, #439/ADR-0063, config-driven since
    ADR-0074), generation is submitted to the Batches API and the sermon stops at
    ``pending_batch`` instead
    (ADR-0061); transcription is unchanged and still happens here, synchronously, because
    a batch request needs the transcript. A sermon handed to this function already
    ``pending_batch`` is one the plan judged resolvable — its result is ready, or its 24h
    deadline has passed — and it is generated from that result, or from a synchronous
    fallback call.

    Escalation is the merge's job, not the shard's — a terminal failure here is recorded
    in the delta's runs and escalated once downstream, so parallel shards never fan out
    duplicate emails. The cross-registry dedup guard (PRD §11.3) likewise defers to the
    merge, which alone sees the whole ledger.

    The audio-fingerprint duplicate check (ADR-0077, #517) is the one cross-record
    question this function *can* answer, because — unlike the transcript-hash guard —
    it has to run before transcription to save the cost, not after. It reads
    ``committed_registry_path`` **read-only** for that comparison, alongside
    ``committed_transcripts_dir``: the checkout is already present for the transcript
    cache read, so this is the same kind of read, not a new write surface (the sole
    write path is still ``run_merge`` via ``Registry.save``).
    """
    max_attempts = config.get_int("MAX_SERMON_ATTEMPTS", _DEFAULT_MAX_SERMON_ATTEMPTS, minimum=1)
    claimed, cap_unenforced = _claimed_attempts(sermon)
    spent = _budget_spent(sermon, claimed=claimed, max_attempts=max_attempts)
    if spent is not None:
        return _refused_delta(sermon, attempts=spent, max_attempts=max_attempts, now=now)

    _seed_cached_transcript(
        sermon, committed_dir=committed_transcripts_dir, staging_dir=transcripts_dir
    )
    with tempfile.TemporaryDirectory() as workdir:
        shard_registry = Registry(Path(workdir) / "shard-ledger.json")
        seed = copy.deepcopy(sermon)
        seed.runs = []  # start clean so the delta captures only this shard's new runs
        shard_registry.upsert(seed)
        discarded: list[_Escalation] = []  # collected by the stages; the shard sends none

        batch_id: str | None = None
        batch_submitted_at: str | None = None

        if seed.state == "discovered":
            committed_registry = (
                Registry.load(committed_registry_path)
                if committed_registry_path.exists()
                else Registry(committed_registry_path)
            )
            _transcribe_stage(
                shard_registry,
                transcribe_fn=transcribe,
                download=download,
                transcripts_dir=transcripts_dir,
                sleep=sleep,
                now=now,
                escalations=discarded,
                duplicate_lookup_registry=committed_registry,
            )
        state = _require_record(shard_registry, sermon.guid).state
        if state == "transcribed" and (_is_backfill() or _is_batch_source(sermon)):
            batch_id, batch_submitted_at = _submit_batch_stage(
                shard_registry,
                sermon.guid,
                submit_batch=submit_batch,
                transcripts_dir=transcripts_dir,
                now=now,
                escalations=discarded,
            )
        elif state == "transcribed":
            _generate_stage(
                shard_registry,
                llm_call=llm_call,
                transcripts_dir=transcripts_dir,
                notes_dir=notes_dir,
                now=now,
                escalations=discarded,
            )
        elif state == "pending_batch":
            _resolve_batch_stage(
                shard_registry,
                sermon.guid,
                poll_batch=poll_batch,
                retrieve_batch=retrieve_batch,
                llm_call=llm_call,
                transcripts_dir=transcripts_dir,
                notes_dir=notes_dir,
                now=now,
                escalations=discarded,
            )

        final = _require_record(shard_registry, sermon.guid)
        return Delta(
            sermon=copy.deepcopy(sermon),
            transcript_hash=final.transcript_hash,
            reached_state=final.state,
            runs=list(final.runs),
            cap_unenforced=cap_unenforced,
            batch_id=batch_id,
            batch_submitted_at=batch_submitted_at,
            audio_fingerprint=final.audio_fingerprint,
            suspected_duplicate_of=final.suspected_duplicate_of,
        )


def run_merge(
    deltas: Sequence[Delta],
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    notes_dir: Path = DEFAULT_NOTES_DIR,
    readme_path: Path = DEFAULT_README_PATH,
    send_fn: Callable[[notify.EmailMessage], None] = notify.default_send,
    discord_send_fn: Callable[[discord_notify.DiscordMessage], str | None] | None = None,
    google_chat_send_fn: Callable[[google_chat_notify.GoogleChatMessage], str | None] | None = None,
    render_feed: Callable[..., object] = feed.render_feed,
    publish_feed: Callable[[Path], None] = content_publish.publish_feed,
    fire_deploy_hook: Callable[[], None] = deploy_hook.fire_deploy_hook,
    spend_refusal: str | None = None,
    no_sources_refusal: str | None = None,
    expected_deltas: int = 0,
    now: str | None = None,
) -> PipelineResult:
    """Fan-in: apply every shard ``delta`` to the ledger, then render, publish, deliver.

    The sole registry writer (ADR-0023). Loads the committed ledger once, applies each
    delta through :func:`~sermon_notes.shard.apply_delta` — which advances state, appends
    the shard's run records (telemetry preserved), and enforces the cross-registry
    ``has_successful_generation`` dedup guard the shard could not (PRD §11.3, #86) — then
    runs the serial render stage (``generated → published``), refreshes the README,
    ships the content feed, and sends one escalation per terminal failure plus the
    published-note deliveries. The git commit/push of the working tree is the workflow's
    job, as it is for :func:`run_pipeline`.

    It also escalates one run-level alert when any shard reports it spent without the
    attempt cap in force (:func:`_unenforced_cap_escalation`). The shard cannot mail —
    it holds no credentials — so the delta is the only channel, and the merge is where a
    run-wide condition can be reported once rather than per sermon (ADR-0036).

    ``spend_refusal`` carries the same kind of signal from the other end of the run: the
    message :func:`plan_shards` refused with, when a rolling spend cap was reached. The
    plan holds no credentials either, and there are no deltas on a refused run, so the
    workflow hands it here as a job output (ADR-0037). A refused run merges nothing —
    ``deltas`` is empty — and exists only to send that one email and end red.

    ``no_sources_refusal`` is the same kind of signal for a different condition
    (ADR-0074, #496): :func:`plan_shards` refuses discovery, not spend, when at
    least one church was enabled but ``CHURCHES`` had no usable feed for any of
    them. Unlike a spend refusal it may still carry deltas — sermons already in
    progress are unaffected — so this alert can arrive alongside an otherwise
    ordinary merge.

    ``expected_deltas`` is how many shards the plan fanned out. Fewer deltas than that
    means sermons went unprocessed, and the merge escalates the difference once
    (:func:`_delta_shortfall_escalation`). It only ever arrives short here when a shard
    job failed: the CLI refuses an unexplained shortfall before calling this at all
    (#266). Zero — the default — on a reconcile, a feed render, or any run off Actions.

    The transcript/note files a shard produced are expected already in place under
    ``notes_dir`` / the transcripts cache (the workflow's artifact download); a deduped
    delta's note is discarded here so a duplicate leaves no orphaned sidecar, matching the
    serial pipeline, which never writes one.
    """
    registry = Registry.load(registry_path)
    escalations: list[_Escalation] = []
    applied_guids: list[str] = []

    for delta in deltas:
        try:
            applied = apply_delta(delta, registry, now=now)
        except RegistryError as exc:
            # Mirrors the serial stages' per-sermon isolation (ADR-0019): a delta the
            # ledger can't accept — e.g. the plan's snapshot and the committed state
            # have since diverged — must not drop every delta queued after it (#198).
            logger.error("delta for %s could not be applied: %s", delta.sermon.guid, exc)
            escalations.append(_escalation(delta.sermon, type(exc).__name__, str(exc)))
            registry.save()  # Persist whatever the upsert landed before the failure (#35).
            continue
        applied_guids.append(applied.guid)
        if applied.deduped:
            _discard_note(notes_dir, registry.get(applied.guid))
        if applied.terminal is not None:
            sermon = registry.get(applied.guid)
            if sermon is not None:
                other = None
                if applied.terminal[0] == DUPLICATE_ERROR_CLASS:
                    other_guid = registry.find_generated_duplicate(
                        sermon.transcript_hash or "", exclude_guid=sermon.guid
                    )
                    other = registry.get(other_guid) if other_guid is not None else None
                if other is not None:
                    # `other` is always found in practice — the guard that produced this
                    # terminal state just matched against it — but a defensive fallback to
                    # the generic one-sermon escalation keeps PRD §11.2's "one escalation
                    # per terminal failure" true even if the collision can't be resolved.
                    fingerprint = "|".join(sorted((sermon.guid, other.guid)))
                    if _should_escalate(
                        registry, _DUPLICATE_TRANSCRIPT_ALERT_CLASS, fingerprint, now=now
                    ):
                        escalations.append(
                            _duplicate_transcript_escalation(
                                sermon, other, sermon.transcript_hash or ""
                            )
                        )
                else:
                    escalations.append(
                        _escalation(sermon, applied.terminal[0], applied.terminal[1])
                    )
        registry.save()  # Persist per delta so a crash loses at most one (#35).

    published = _render_stage(registry, notes_dir=notes_dir, now=now, escalations=escalations)
    registry.save()

    # As in run_pipeline: the whole fan-out's escalations and deliveries go out before
    # the index refresh and the publish step can fail (#197). The merge is the only place
    # a parallel run escalates at all, so a raise here would silence every shard.
    escalated = _send_escalations(escalations, send_fn=send_fn)

    # Kept in its own list so the run-level alert never reaches `failed` below, which is
    # a tuple of guids that actually failed — these sermons processed (ADR-0036). Deduped
    # against a repeat of the same condition (ADR-0046, #324): no severity axis to
    # fingerprint, so the cooldown alone gates a persistent, unchanged occurrence.
    cap_alert = _unenforced_cap_escalation(deltas, now=now)
    if cap_alert and not _should_escalate(registry, _CAP_UNENFORCED_ERROR_CLASS, "active", now=now):
        cap_alert = None
    escalated += _send_escalations([cap_alert] if cap_alert else [], send_fn=send_fn)

    # Same reasoning, from the plan's end of the run (ADR-0037): a refusal is a property
    # of the run, no sermon failed, and this is the only job that can mail it at all.
    # Deduped the same way as the cap alert above — a persistent refusal stays quiet.
    refusal_alert = _spend_refusal_escalation(spend_refusal, now=now)
    was_refused = refusal_alert is not None
    if refusal_alert and not _should_escalate(
        registry, _SPEND_REFUSAL_ERROR_CLASS, "active", now=now
    ):
        refusal_alert = None
    escalated += _send_escalations([refusal_alert] if refusal_alert else [], send_fn=send_fn)

    # Same reasoning again, from the fan-out's discovery step this time (ADR-0074, #496):
    # a fully-unresolved CHURCHES is a property of the run, no sermon failed, and this
    # is the only job that can mail it. Deduped the same way — a persistent misconfiguration
    # stays quiet after the first alert.
    no_sources_alert = _no_sources_escalation(no_sources_refusal, now=now)
    was_sources_refused = no_sources_alert is not None
    if no_sources_alert and not _should_escalate(
        registry, _NO_SOURCES_ERROR_CLASS, "active", now=now
    ):
        no_sources_alert = None
    escalated += _send_escalations([no_sources_alert] if no_sources_alert else [], send_fn=send_fn)

    # And from the fan-out's end (#266): the sermons whose shards died without a delta.
    # Run-level for the same reason again — the merge knows the count, not the guids.
    # Fingerprinted on the missing count (ADR-0046): a shortfall that grows re-alerts
    # immediately even inside the cooldown; an unchanged shortfall stays quiet.
    shortfall = max(expected_deltas - len(deltas), 0)
    shortfall_alert = _delta_shortfall_escalation(shortfall, expected_deltas, now=now)
    if shortfall_alert and not _should_escalate(
        registry, _DELTA_SHORTFALL_ERROR_CLASS, str(shortfall), now=now
    ):
        shortfall_alert = None
    escalated += _send_escalations([shortfall_alert] if shortfall_alert else [], send_fn=send_fn)
    registry.save()  # Persist the alert dedup state alongside everything else this run.

    delivered = _send_note_deliveries(
        registry, published, repo_root=notes_dir.parent, send_fn=send_fn
    )
    discord_delivered = _send_discord_deliveries(
        registry, published, repo_root=notes_dir.parent, send_fn=discord_send_fn
    )
    if discord_delivered:
        # The ledger was saved *before* the delivery above, so the message ids it just
        # recorded exist only in memory until now. Without this the ids die with the
        # runner and the next repaint is back to copying them out of Discord by hand
        # (ADR-0062, CLAUDE.md §10).
        registry.save()
    google_chat_delivered = _send_google_chat_deliveries(
        registry, published, send_fn=google_chat_send_fn
    )
    if google_chat_delivered:
        # Same reasoning as the Discord save above: a channel-message-id capture
        # (spec 0024 amendment) that lived only in memory would die with the runner.
        registry.save()

    if published:
        update_readme(registry_path=registry_path, readme_path=readme_path)

    publish_escalations: list[_Escalation] = []
    feed_published, deploy_triggered = _publish_feed_step(
        registry_path=registry_path,
        notes_dir=notes_dir,
        published=published,
        render_feed=render_feed,
        publish_feed=publish_feed,
        fire_deploy_hook=fire_deploy_hook,
        escalations=publish_escalations,
        now=now,
    )
    escalated += _send_escalations(publish_escalations, send_fn=send_fn)

    transcribed = [g for g in applied_guids if _state_of(registry, g) == "transcribed"]
    generated = [g for g in applied_guids if _state_of(registry, g) == "generated"]
    return PipelineResult(
        transcribed=tuple(transcribed),
        generated=tuple(generated),
        published=tuple(published),
        failed=tuple(esc.guid for esc in escalations),
        escalated=escalated,
        delivered=delivered,
        discord_delivered=discord_delivered,
        google_chat_delivered=google_chat_delivered,
        feed_published=feed_published,
        deploy_triggered=deploy_triggered,
        spend_refused=was_refused,
        sources_refused=was_sources_refused,
        delta_shortfall=shortfall,
    )


def _require_record(registry: Registry, guid: str) -> SermonRecord:
    """Return the record for ``guid``, asserting it exists (a shard always seeds one)."""
    record = registry.get(guid)
    assert record is not None, f"sermon {guid!r} missing from the shard ledger"
    return record


def _state_of(registry: Registry, guid: str) -> str | None:
    """The current lifecycle state of ``guid``, or ``None`` if it is absent."""
    record = registry.get(guid)
    return record.state if record is not None else None


def _discard_note(notes_dir: Path, sermon: SermonRecord | None) -> None:
    """Remove a shard-written note JSON for a deduped sermon (it never reaches render).

    Sole exemption to CLAUDE.md §6's notes/ append-only invariant — see the
    "Soft-delete" row there. No published artifact is ever removed.
    """
    if sermon is None:
        return
    path = note_json_path(notes_dir, sermon)
    if path.exists():
        path.unlink()
