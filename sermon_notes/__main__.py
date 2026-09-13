"""CLI entry for the pipeline — what the scheduled workflow invokes (U10/U11).

``python -m sermon_notes`` runs the full pipeline once (the every-N-hours
cron); ``python -m sermon_notes reconcile`` runs the daily reconciliation
pass (PRD §6.5); ``python -m sermon_notes feed`` renders the content feed for
every published sermon (spec 0014 backfill / seeding). All default every
boundary to its real client, reading configuration through
:mod:`sermon_notes.config`.

Three further commands drive the parallel fan-out/fan-in ``run`` (ADR-0023), one
per Actions job: ``plan-shards`` (preflight — poll, write the working set, emit the
matrix), ``shard`` (one per sermon — transcribe+generate into files + a delta), and
``merge`` (fan-in — apply every delta, render, publish, commit). Their artifact paths
are read from config so the workflow can point the three jobs at shared paths.

``discord-test`` is a manual verification command (spec 0017): it re-sends the most
recently published PBC sermon's note through the real Discord boundary
(:func:`discord_notify.default_send`), so the webhook can be exercised end-to-end from
a one-click ``workflow_dispatch`` without waiting for a sermon to actually publish.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sermon_notes import config
from sermon_notes.logging import get_logger
from sermon_notes.readme_index import README_RELPATH, update_readme
from sermon_notes.registry import (
    DEFAULT_REGISTRY_PATH,
    REGISTRY_RELPATH,
    LedgerValidationError,
    Registry,
    RunRecord,
    SermonRecord,
    validate_ledger,
)

# attempt_claims, discord_notify, feed, pipeline, and shard are imported inside the
# branches that need them, not here. `pipeline` alone pulls in render_pdf (reportlab),
# render (python-docx), transcribe, and sources — importing it unconditionally cost
# every invocation ~200ms even for `readme-index` and `validate-ledger`, which the
# `Commit artifacts` step (pipeline.yml) calls up to four times per merge-job run.
# The two below are for type annotations only (`from __future__ import annotations`
# defers them at runtime, but ruff/mypy still resolve the names statically).
if TYPE_CHECKING:
    from sermon_notes import discord_notify, pipeline

logger = get_logger()

# Default artifact paths for the fan-out/fan-in jobs; overridable via config so the
# workflow can stage them where upload/download-artifact expects (Child C).
_DEFAULT_WORKING_SET_PATH = "shard/working-set.json"
# A shard writes its transcript/note under a per-runner staging dir rather than the
# real transcripts/notes trees the checkout populates, so its artifact upload carries
# only the files it produced instead of the whole committed archive (#202). The merge
# job overlays this staging dir onto its checkout before rendering.
_DEFAULT_STAGING_DIR = "shard/staging"
# The delta lives *inside* that staging dir so the shard job can upload a single
# directory. `upload-artifact` roots an artifact at the least common ancestor of its
# search paths, so a second path silently re-roots the archive and moves everything in
# it — which is how #235 lost every delta between the shard and the merge. One path
# keeps the root pinned to the staging dir no matter what the shard writes into it.
_DEFAULT_DELTAS_DIR = f"{_DEFAULT_STAGING_DIR}/deltas"


def _sermon_from_dict(data: dict[str, Any]) -> SermonRecord:
    """Rebuild a :class:`SermonRecord` (with nested runs) from its serialized dict."""
    return SermonRecord(**{**data, "runs": [RunRecord(**run) for run in data.get("runs", [])]})


def _working_set_path() -> Path:
    return Path(config.get("SHARD_WORKING_SET_PATH", _DEFAULT_WORKING_SET_PATH))


def _deltas_dir() -> Path:
    return Path(config.get("SHARD_DELTAS_DIR", _DEFAULT_DELTAS_DIR))


def _staging_dir() -> Path:
    return Path(config.get("SHARD_STAGING_DIR", _DEFAULT_STAGING_DIR))


# The delimiter the free-form `spend_refusal` output is written with. GitHub's multi-line
# output form needs one the value cannot contain; the value is a guard message this
# codebase composes, so a fixed sentinel is enough — nothing external reaches it.
_OUTPUT_DELIMITER = "SPEND_REFUSAL_EOF"

# Same reasoning, for `no_sources_refusal` (ADR-0074, #496): a distinct sentinel so the
# two multi-line outputs can never be mistaken for one another in the outputs file.
_NO_SOURCES_OUTPUT_DELIMITER = "NO_SOURCES_REFUSAL_EOF"

# The `needs.shard.result` values that mean the fan-out itself broke, so a delta short of
# the plan's shard count is explained by a shard dying rather than by an artifact going
# missing between the jobs (#266). Everything else — `success`, `skipped`, and the empty
# string an unwired job output leaves behind — is treated as "the shards accounted for
# their deltas", which is the direction that keeps a lost delta loud.
_FAILED_SHARD_RESULTS = frozenset({"failure", "cancelled"})


def _write_working_set(records: Sequence[SermonRecord]) -> None:
    """Persist the working set for the shard jobs to resolve their sermon from.

    Written even when it is empty — including on a refusal — so the file the shard job
    reads always exists and says the same thing as the matrix.
    """
    path = _working_set_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(r) for r in records], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _matrix(records: Sequence[SermonRecord]) -> str:
    """The GitHub matrix JSON for ``records`` — one entry per sermon needing work."""
    from sermon_notes import attempt_claims

    return json.dumps(
        {
            "include": [
                {"index": i, "guid": r.guid, "claim": attempt_claims.claim_name(r.guid)}
                for i, r in enumerate(records)
            ]
        }
    )


def _emit_plan_outputs(
    matrix: str,
    *,
    shard_count: int,
    spend_refusal: str | None = None,
    no_sources_refusal: str | None = None,
) -> None:
    """Append the plan job's step outputs to ``$GITHUB_OUTPUT``; a no-op off Actions.

    ``shard_count`` is emitted twice over, as the ``has_shards`` boolean the ``if:``
    gates read and as the count itself. Both come from the one number so they can never
    disagree about whether the run fanned out; the count is what lets the merge check its
    fan-in against how many shards were planned rather than merely against zero (#266).

    ``spend_refusal`` and ``no_sources_refusal`` are each written in GitHub's multi-line
    form rather than as ``k=v``: each carries a guard message, not a token, so a newline
    in it must not be able to truncate the value or forge a second key. Each is omitted
    entirely on an ordinary run, which leaves the job output empty and the merge with
    nothing to escalate (ADR-0037, ADR-0074).
    """
    github_output = config.get("GITHUB_OUTPUT", None)
    if not github_output:
        return
    with Path(github_output).open("a", encoding="utf-8") as fh:
        fh.write(f"matrix={matrix}\n")
        fh.write(f"has_shards={'true' if shard_count else 'false'}\n")
        fh.write(f"shard_count={shard_count}\n")
        if spend_refusal is not None:
            fh.write(f"spend_refusal<<{_OUTPUT_DELIMITER}\n{spend_refusal}\n{_OUTPUT_DELIMITER}\n")
        if no_sources_refusal is not None:
            fh.write(
                f"no_sources_refusal<<{_NO_SOURCES_OUTPUT_DELIMITER}\n"
                f"{no_sources_refusal}\n{_NO_SOURCES_OUTPUT_DELIMITER}\n"
            )


def _run_plan_shards() -> None:
    """Poll, persist the working set, and emit the per-sermon matrix (ADR-0023).

    Writes the working-set records to :func:`_working_set_path` for the shard jobs to
    read, and appends the GitHub matrix ``{"include": [...]}`` plus a ``has_shards``
    boolean and a ``shard_count`` to ``$GITHUB_OUTPUT`` — so the shard job fans out one
    runner per sermon via ``fromJSON``, is gated off when the working set is empty, and
    the merge knows how many deltas to expect back (#266). Also logs the matrix for a
    local run.

    Each matrix entry carries the sermon's ``claim`` artifact name (ADR-0033). The shard
    job uploads a claim under that name before it spends, and
    :func:`~sermon_notes.attempt_claims.claims_for` counts artifacts under the same name —
    deriving both from :func:`~sermon_notes.attempt_claims.claim_name` here is what stops
    the upload and the count from ever drifting apart and silently disabling the cap.

    A rolling spend cap refusal (:class:`~sermon_notes.pipeline.SpendGuardError`) is
    caught rather than propagated, and emitted as a ``spend_refusal`` output alongside the
    matrix (ADR-0037). Exiting zero is the point: the plan job holds no mail
    credentials, and ``merge`` — which does — is gated on the plan succeeding, so a raise
    here refused the run *and* skipped the only job that could announce the refusal. The
    run still ends red, from the merge, once the escalation has been sent.

    That matrix is normally empty, but a refusal covers spend and not work: a backfill
    sermon whose batch has already ended is retrieved for free, so the guard hands those
    back on the exception and they are fanned out anyway (spec 0020). The merge job runs
    on either signal, so a refusal with shards escalates and merges in the same run.

    A source-configuration refusal
    (:class:`~sermon_notes.pipeline.NoSourcesConfiguredError`, ADR-0074, #496) is caught
    the same way, emitted as a ``no_sources_refusal`` output instead — every enabled
    source resolved to zero usable feeds. Its ``resolvable`` sermons are already-working
    ones a missing feed does not affect, so a run with in-flight sermons still fans them
    out even while discovery is refused.
    """
    from sermon_notes import pipeline

    try:
        records = pipeline.plan_shards()
    except pipeline.SpendGuardError as exc:
        logger.error("plan-shards: %s", exc)
        resolvable = exc.resolvable
        _write_working_set(resolvable)
        _emit_plan_outputs(_matrix(resolvable), shard_count=len(resolvable), spend_refusal=str(exc))
        return
    except pipeline.NoSourcesConfiguredError as exc:
        logger.error("plan-shards: %s", exc)
        resolvable = exc.resolvable
        _write_working_set(resolvable)
        _emit_plan_outputs(
            _matrix(resolvable), shard_count=len(resolvable), no_sources_refusal=str(exc)
        )
        return
    _write_working_set(records)
    matrix_json = _matrix(records)
    _emit_plan_outputs(matrix_json, shard_count=len(records))
    logger.info("plan-shards: %d sermon(s) to process; matrix=%s", len(records), matrix_json)


def _run_shard(index: int) -> None:
    """Transcribe+generate the working-set sermon at ``index`` and write its delta.

    Only the *write* directories are redirected into staging (#202). ``run_shard`` still
    reads an already-committed transcript from the checkout's ``transcripts/`` via its
    ``committed_transcripts_dir`` default, so scoping the upload never costs the cache
    a re-download and a second whisper run (#228).
    """
    from sermon_notes import pipeline

    records = json.loads(_working_set_path().read_text(encoding="utf-8"))
    sermon = _sermon_from_dict(records[index])
    staging_dir = _staging_dir()
    delta = pipeline.run_shard(
        sermon,
        transcripts_dir=staging_dir / "transcripts",
        notes_dir=staging_dir / "notes",
    )
    deltas_dir = _deltas_dir()
    deltas_dir.mkdir(parents=True, exist_ok=True)
    (deltas_dir / f"{index}.json").write_text(delta.to_json(), encoding="utf-8")
    logger.info("shard %d: %s reached %s", index, sermon.guid, delta.reached_state)


def _run_merge() -> pipeline.PipelineResult:
    """Apply every shard delta (in matrix order) and render/publish/commit (ADR-0023).

    Refuses a fan-in shorter than the fan-out unless the shards' own failures account for
    it. ``SHARD_DELTAS_EXPECTED`` is the plan's shard count and ``SHARD_JOB_RESULT`` is
    what GitHub reports for the shard matrix as a whole, and both are needed because a
    shard that *crashed* and a delta *lost in transit* look identical from a count alone:

    * shards all succeeded, deltas missing → transit loss, and every affected sermon
      stays unadvanced. The ledger is what stops a sermon being reprocessed, so the next
      cron tick re-transcribes and re-generates each one at full LLM cost. #235 did that
      for 14 hours behind a green run and an INFO log, so this raises.
    * a shard job failed → its sermon is the one that suffers, and discarding the
      surviving deltas would throw away work that was already paid for (ADR-0023 failure
      isolation, #125). The merge proceeds and reports the shortfall instead.

    Refusing when the result is *unknown* is deliberate: an unset ``SHARD_JOB_RESULT``
    reads exactly like a green fan-out at the call site, and a guard that cannot tell the
    two apart is the §10 landmine — so anything but a reported failure is treated as
    "the shards accounted for their deltas".

    ``SPEND_REFUSAL`` carries the plan job's spend-cap refusal, when it made one
    (ADR-0037). A refused run fans out no shards, so it arrives here with no deltas and
    nothing to apply; the merge runs anyway because it is the only job holding the mail
    credentials, and delivering that refusal is what it is here to do.

    ``NO_SOURCES_REFUSAL`` carries the plan job's source-configuration refusal the same
    way (ADR-0074, #496) — unlike a spend refusal it may still carry deltas from
    already-in-progress sermons, so this run is not necessarily otherwise empty.
    """
    from sermon_notes import pipeline
    from sermon_notes.shard import Delta

    deltas_dir = _deltas_dir()
    all_paths = deltas_dir.glob("*.json")
    delta_paths: list[Path] = []
    for path in all_paths:
        if path.stem.isdigit():
            delta_paths.append(path)
        else:
            logger.warning("merge: skipping non-delta file %s in %s", path.name, deltas_dir)
    paths = sorted(delta_paths, key=lambda p: int(p.stem))
    deltas = [Delta.from_json(p.read_text(encoding="utf-8")) for p in paths]
    expected = config.get_int("SHARD_DELTAS_EXPECTED", 0, minimum=0)
    shards_failed = config.get("SHARD_JOB_RESULT", "") in _FAILED_SHARD_RESULTS
    if len(deltas) < expected and not shards_failed:
        raise RuntimeError(
            f"merge: the run fanned out {expected} shard job(s) but only {len(deltas)} of "
            f"the {expected} delta(s) reached {deltas_dir}, and no shard job reported a "
            "failure to account for the rest. Refusing to publish a short merge — the "
            "sermons whose deltas went missing would stay unadvanced and be "
            "re-transcribed and re-billed on the next run (#235, #266). Check that the "
            "shard artifact upload and the merge's download resolve to the same "
            "directory."
        )
    logger.info("merge: applying %d shard delta(s) of %d planned", len(deltas), expected)
    return pipeline.run_merge(
        deltas,
        expected_deltas=expected,
        spend_refusal=config.get("SPEND_REFUSAL", None),
        no_sources_refusal=config.get("NO_SOURCES_REFUSAL", None),
    )


def _run_readme_index(*, repo_root: Path = Path(".")) -> None:
    """Regenerate the README sermon index from the ledger in ``repo_root`` (#269).

    The workflow's ``Commit artifacts`` step calls this between rebasing and pushing.
    README.md is the one committed artifact path a human also writes — the fences
    :mod:`~sermon_notes.readme_index` rewrites exist because the prose around them is
    theirs — so replaying the run's copy of it over a squash-merge that landed mid-run is
    the only way that commit can conflict. Re-deriving the table instead removes README
    from the rebase entirely, which is sound because the table is a pure function of the
    ledger: rebase the ledger, re-derive the table, and there is nothing left to merge.

    Paths resolve against the working directory, not the package's own repo root: the
    checkout being committed is the one that has to be rewritten.
    """
    update_readme(
        registry_path=repo_root / REGISTRY_RELPATH,
        readme_path=repo_root / README_RELPATH,
    )


def _run_validate_ledger(*, repo_root: Path = Path(".")) -> int:
    """Refuse to stage a ledger the default branch could not load back (ADR-0044).

    :func:`~sermon_notes.registry.validate_ledger` does the checking; this decides what
    a failure is worth. The dataclass constructors behind the strict loader raise
    ``TypeError`` on a missing or unexpected field and ``json.loads`` raises
    ``ValueError`` on a truncated write, so both join the module's own error here.

    Paths resolve against the working directory, not the package's own root — the
    checkout being committed is the one that has to be sound, exactly as for
    :func:`_run_readme_index`.

    Returns the process exit code, so a rejection stops ``Commit artifacts`` under
    ``bash -e`` before ``git add`` rather than after ``git push``.
    """
    try:
        count = validate_ledger(repo_root / REGISTRY_RELPATH)
    except (LedgerValidationError, TypeError, ValueError) as exc:
        logger.error("ledger validation failed, refusing to commit or push: %s", exc)
        return 1

    logger.info("ledger validated: %d sermon(s) load strictly", count)
    return 0


def _run_discord_test(
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    repo_root: Path = Path("."),
    send_fn: Callable[[discord_notify.DiscordMessage], str | None] | None = None,
) -> int:
    """Re-post the newest published PBC note through the real Discord boundary.

    Scoped to ``source == "pbc"`` regardless of what other sources Discord delivery
    now covers (spec 0017 amendment adds Menlo's own channel) — this command exercises
    ``DISCORD_WEBHOOK_URL`` only, unchanged by that amendment. It resolves the
    candidate's artifact exactly as the publish-time delivery does — the ``.pdf`` when
    present, else the ``.docx`` (#173) — then hands it to
    :func:`discord_notify.send_note`, the same function the orchestrator calls at the
    publish transition. Nothing in the registry changes; this only re-sends a note
    that already published. That includes the message id the boundary now reports
    (ADR-0062): this command creates a *second* message for a sermon whose real one is
    already recorded, so writing its id would point every future repaint at the test
    post instead of the note the channel actually reads. Returns a non-zero exit code
    when there is no eligible sermon, the webhook isn't configured, or the post fails,
    so a failed manual dispatch shows red in Actions.
    """
    from sermon_notes import discord_notify
    from sermon_notes.pipeline import note_url_for

    if send_fn is None:
        send_fn = discord_notify.default_send
    registry = Registry.load(registry_path)
    candidates = [
        sermon
        for sermon in registry.sermons()
        if sermon.source == "pbc" and sermon.state == "published" and sermon.artifact_path
    ]
    if not candidates:
        logger.error("discord-test: no published PBC sermon in the ledger to test with.")
        return 1
    if not discord_notify.is_configured():
        logger.error("discord-test: DISCORD_WEBHOOK_URL is not configured.")
        return 1
    sermon = max(candidates, key=lambda s: s.published_on)
    assert sermon.artifact_path is not None  # guaranteed by the `candidates` filter above
    docx_path = repo_root / sermon.artifact_path
    pdf_path = docx_path.with_suffix(".pdf")
    result = discord_notify.send_note(
        sermon_title=sermon.title,
        sermon_date=sermon.published_on,
        artifact_path=pdf_path if pdf_path.exists() else docx_path,
        episode_url=sermon.episode_url or None,
        note_url=note_url_for(sermon),
        send_fn=send_fn,
    )
    logger.info(
        "discord-test: %s for %r (%s)",
        "delivered" if result.delivered else "failed to deliver",
        sermon.title,
        sermon.guid,
    )
    return 0 if result.delivered else 1


def _exit_code(result: object) -> int:
    """Non-zero when a run left terminal failures behind (#200).

    A deferred poll (source unreachable) is the designed transient path and stays
    ``0`` — only sermons that reached the terminal ``failed`` state, or a
    reconciliation sweep that found artifacts missing from disk, should flip the
    workflow to failure.

    A tripped dead-man's switch (#242) fails the run too. Email is best-effort by
    contract, so a red run in Actions is the switch's second delivery channel — and
    the only one left when the alert itself fails to send, which is why
    ``publish_stale_alert_failed`` needs no separate branch: it is only ever set on a
    pass that already returns 1.

    A run the plan refused on a spend cap (#265) fails too, for the same reason: the
    refusal's escalation is best-effort mail, so the red run is its second channel — and
    the only one left when the send itself fails. The red simply moves from the plan job,
    which now exits zero so the merge can send that mail at all, to the merge (ADR-0037).

    A tripped red-run-streak switch (#283) fails the run too, for the same
    best-effort-mail-needs-a-second-channel reason — ``red_streak_alert_failed`` needs
    no separate branch, as it is only ever set on a pass that already returns 1.

    A run the plan refused because every enabled source resolved to zero usable feeds
    (ADR-0074, #496) fails too, for the same reason as the spend refusal above.
    """
    from sermon_notes import pipeline

    if isinstance(result, pipeline.PipelineResult):
        return 1 if result.failed or result.spend_refused or result.sources_refused else 0
    if isinstance(result, pipeline.ReconcileResult):
        return (
            1
            if result.missing_artifacts
            or result.publish_stale_days is not None
            or result.red_streak_count is not None
            else 0
        )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Parse the command and run the pipeline; return a process exit code."""
    parser = argparse.ArgumentParser(
        prog="sermon_notes",
        description=(
            "Detect, transcribe, and write church sermon study notes across "
            "the configured podcast feeds."
        ),
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="run",
        choices=(
            "run",
            "reconcile",
            "feed",
            "plan-shards",
            "shard",
            "merge",
            "discord-test",
            "readme-index",
            "escalate-push-failure",
            "validate-ledger",
            "alert-stalled-queue",
        ),
        help=(
            "run the full pipeline (default), the daily reconciliation pass, render "
            "the content feed, a fan-out/fan-in job (plan-shards / shard / merge), "
            "re-send the newest published PBC note to Discord (discord-test), one of "
            "the three the artifact commit drives (validate-ledger / readme-index / "
            "escalate-push-failure), or mail queue-stall-alert.yml's per-PR finding "
            "(alert-stalled-queue)"
        ),
    )
    parser.add_argument(
        "--index",
        type=int,
        default=None,
        help="the working-set index for a `shard` job (else read from SHARD_INDEX)",
    )
    parser.add_argument(
        "--artifacts-preserved",
        action="store_true",
        help=(
            "for `escalate-push-failure`: the step managed to stage this run's output for "
            "the preservation upload, so the alert describes a replay rather than a loss"
        ),
    )
    parser.add_argument(
        "--push-error",
        default="",
        help=(
            "for `escalate-push-failure`: what the remote said when it refused the push, "
            "so the alert names the cause instead of listing candidates"
        ),
    )
    parser.add_argument(
        "--pr",
        type=int,
        default=None,
        help="for `alert-stalled-queue`: the stalled pull request's number",
    )
    parser.add_argument(
        "--hours",
        type=int,
        default=None,
        help=(
            "for `alert-stalled-queue`: how many hours the PR has worn `queue:ready`, "
            "as the workflow computed it"
        ),
    )
    parser.add_argument(
        "--url",
        default="",
        help="for `alert-stalled-queue`: the stalled pull request's URL",
    )
    parser.add_argument(
        "--title",
        default="",
        help="for `alert-stalled-queue`: the stalled pull request's title",
    )
    parser.add_argument(
        "--threshold-hours",
        type=int,
        # Resolved against pipeline._DEFAULT_QUEUE_STALL_HOURS below, not here: reading
        # it as this argument's default would import `pipeline` (and everything it
        # drags in) to build the parser for every command, not just this one.
        default=None,
        help=(
            "for `alert-stalled-queue`: the QUEUE_STALL_HOURS the workflow checked "
            "against, named in the alert's body"
        ),
    )
    args = parser.parse_args(argv)

    result: object
    if args.command == "reconcile":
        from sermon_notes import pipeline

        result = pipeline.reconcile()
    elif args.command == "feed":
        from sermon_notes import feed

        result = feed.render_feed()
    elif args.command == "plan-shards":
        _run_plan_shards()
        return 0
    elif args.command == "shard":
        index = (
            args.index if args.index is not None else config.get_int("SHARD_INDEX", 0, minimum=0)
        )
        _run_shard(index)
        return 0
    elif args.command == "merge":
        result = _run_merge()
    elif args.command == "discord-test":
        return _run_discord_test()
    elif args.command == "readme-index":
        _run_readme_index()
        return 0
    elif args.command == "validate-ledger":
        # Non-zero is the whole point: `Commit artifacts` runs under `bash -e`, so a
        # rejection here stops the step before `git add` rather than after `git push`.
        return _run_validate_ledger()
    elif args.command == "escalate-push-failure":
        from sermon_notes import pipeline

        # Zero whatever the mail did: escalation is best-effort by contract (PRD §11.2),
        # and the caller is a workflow step that is already on its way to exiting 1.
        # The flag is passed only when the step has already staged the tree, so the
        # default — a bare invocation — is the alert that claims nothing (#315).
        pipeline.escalate_artifact_push_failure(
            preserved=args.artifacts_preserved, push_error=args.push_error
        )
        return 0
    elif args.command == "alert-stalled-queue":
        from sermon_notes import pipeline

        # Non-zero on a dropped send, unlike escalate-push-failure: this alert has no
        # other trigger and no other run to catch it on, so a failed send is itself
        # the incident and the red run is its second channel (ADR-0057).
        if args.pr is None or args.hours is None:
            logger.error("alert-stalled-queue: --pr and --hours are required.")
            return 1
        threshold_hours = (
            args.threshold_hours
            if args.threshold_hours is not None
            else pipeline._DEFAULT_QUEUE_STALL_HOURS
        )
        delivered = pipeline.escalate_stalled_queue_pr(
            pr_number=args.pr,
            pr_title=args.title,
            hours_stale=args.hours,
            pr_url=args.url,
            threshold_hours=threshold_hours,
        )
        return 0 if delivered else 1
    else:
        from sermon_notes import pipeline

        result = pipeline.run_pipeline()

    logger.info("pipeline %s complete: %s", args.command, result)
    return _exit_code(result)


if __name__ == "__main__":  # pragma: no cover - exercised via the module entry point
    raise SystemExit(main())
