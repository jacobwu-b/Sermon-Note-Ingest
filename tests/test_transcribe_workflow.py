"""Contract test on transcribe.yml's own declared trigger/timeout config.

The repo has no YAML-parsing dependency (stdlib-only stack), so this matches on
the file's text rather than a parsed structure — the same tradeoff
test_poll_workflow.py makes for poll.yml.
"""

import os
import re
import subprocess
from pathlib import Path

import pytest

WORKFLOW = Path(__file__).parent.parent / ".github" / "workflows" / "transcribe.yml"

STEPS_OUTPUTS_RE = re.compile(r"\$\{\{\s*steps\.[\w-]+\.outputs\.")


def workflow_text() -> str:
    return WORKFLOW.read_text()


def run_block_bodies(text: str) -> list[str]:
    """Each `run: |` script body, by indentation."""
    lines = text.splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        stripped = lines[i].lstrip()
        if stripped == "run: |":
            indent = len(lines[i]) - len(stripped)
            body = []
            i += 1
            while i < len(lines) and (
                lines[i].strip() == "" or len(lines[i]) - len(lines[i].lstrip()) > indent
            ):
                body.append(lines[i])
                i += 1
            blocks.append("\n".join(body))
            continue
        i += 1
    return blocks


def test_transcribe_workflow_declares_its_own_timeout():
    # Whisper transcription cannot fit inside poll.yml's 4-minute budget — this
    # job needs (and must bound) its own, separate timeout (spec 0003, §13).
    assert "timeout-minutes:" in workflow_text()


def test_transcribe_workflow_accepts_workflow_dispatch_with_limit_and_church_inputs():
    text = workflow_text()
    assert "workflow_dispatch:" in text
    assert "limit:" in text
    assert "church:" in text


def test_transcribe_workflow_wires_dispatch_inputs_to_the_transcriber_cli():
    text = workflow_text()
    assert "inputs.limit" in text
    assert "inputs.church" in text
    assert "python -m poller.transcriber" in text


def test_transcribe_workflow_caches_whisper_model_weights():
    # A scheduled run must not re-download model weights every invocation
    # (§13 unattended cost).
    assert "actions/cache" in workflow_text()


def test_transcribe_workflow_does_not_share_poll_yml_s_concurrency_group():
    # Transcription and polling must never block each other.
    assert "group: transcribe-sermons" in workflow_text()


def test_transcribe_workflow_wires_pipeline_dispatch_config():
    # poller/pipeline_dispatch.py (spec 0005, ADR-0009) needs both to trigger
    # Sermon-Note-Pipeline's ingest-event dispatch after a successful transcription,
    # and every job needs them to read the church table (ADR-0016) — so they are
    # workflow-level env, before the first job.
    text = workflow_text()
    workflow_env = text[: text.index("\njobs:")]
    assert "PIPELINE_REPO: ${{ vars.PIPELINE_REPO }}" in workflow_env
    assert "PIPELINE_DISPATCH_TOKEN: ${{ secrets.PIPELINE_DISPATCH_TOKEN }}" in workflow_env


def test_transcribe_workflow_reads_the_church_table_from_pipeline_not_a_variable():
    """A `CHURCHES` variable here would be a second copy of Pipeline's table (ADR-0016)."""
    assert "CHURCHES" not in workflow_text()


def test_transcribe_workflow_wires_hf_token_secret():
    # HF_TOKEN authenticates faster-whisper's Hugging Face Hub model download
    # (read by huggingface_hub itself, not this repo's code) — the secret already
    # exists in GitHub but must be passed into the job's environment to take effect.
    assert "HF_TOKEN: ${{ secrets.HF_TOKEN }}" in workflow_text()


def test_transcribe_workflow_does_not_accept_a_shards_dispatch_input():
    # Shard count is computed from the actual backlog (ADR-0005, amended), not
    # guessed by an operator at dispatch time.
    text = workflow_text()
    assert "      shards:\n" not in text  # a workflow_dispatch input, indented under `inputs:`
    assert "inputs.shards" not in text


def test_transcribe_workflow_computes_a_shard_matrix_from_the_actual_backlog():
    # A dynamic matrix, not a static one — the shard count is only known once the
    # plan job counts sermons in scope, so a separate job must turn it into an
    # array the transcribe job's strategy.matrix consumes.
    text = workflow_text()
    assert "--print-shard-count" in text
    assert "strategy:" in text
    assert "fromJson(needs.plan.outputs.shards)" in text
    assert "fail-fast: false" in text


def test_transcribe_workflow_wires_shard_index_and_count_to_the_transcriber_cli():
    text = workflow_text()
    assert "--shard-index" in text
    assert "--shard-count" in text
    assert "matrix.shard" in text
    assert "needs.plan.outputs.shard_count" in text


def test_transcribe_workflow_scales_ledger_push_retries_with_shard_count():
    # More shards means more concurrent writers to data/*.json; a fixed retry
    # budget sized for one concurrent writer (poll.yml) would run out headroom
    # faster as shard count grows (ADR-0005 Decision 3).
    text = workflow_text()
    assert "attempts=" in text
    assert 'attempts="${{ needs.plan.outputs.shard_count }}"' in text


def test_transcribe_workflow_rebases_onto_main_before_replaying_marks_on_conflict():
    # Regression (issue #60): a rejected push used to unconditionally `git reset
    # --hard` and re-run the whole shard's transcriber invocation on conflict —
    # re-transcribing the whole batch with Whisper and risking a second,
    # differently-worded push overwriting a transcript Pipeline may have already
    # consumed. A rejected push must try a rebase first, and a genuine conflict
    # must replay just this shard's own already-durable marks
    # (poller.transcriber --replay-marks, ADR-0014) rather than redoing any
    # transcription.
    text = workflow_text()
    assert "git rebase origin/main" in text
    assert "git rebase --abort" in text
    assert "--replay-marks" in text
    # The replay path must be reached only from a failed rebase, not
    # unconditionally on every rejected push.
    reset_index = text.index("git reset --hard origin/main")
    rebase_abort_index = text.index("git rebase --abort")
    replay_index = text.index("python -m poller.transcriber --replay-marks")
    assert rebase_abort_index < reset_index < replay_index


def test_transcribe_workflow_never_redoes_transcription_on_a_push_conflict():
    # The old full-batch redo (issue #60) is removed entirely, not just guarded:
    # no invocation of the transcriber with the original run's arguments, and no
    # `|| true` masking a failed recovery step.
    text = workflow_text()
    assert "TRANSCRIBE_ARGS" not in text
    assert "|| true" not in text


def test_transcribe_workflow_writes_marks_out_for_the_push_conflict_recovery():
    text = workflow_text()
    assert "--marks-out" in text
    assert "MARKS_FILE" in text


def test_transcribe_workflow_preserves_unpushed_marks_as_an_artifact_on_failure():
    # If the push retry loop still fails, this shard's already-Content-pushed
    # marks must survive the runner rather than being lost (§13 durable output).
    text = workflow_text()
    assert "actions/upload-artifact@043fb46d1a93c77aae656e7c1c64a875d1fc6a0a # v7" in text
    assert "if: failure()" in text


def test_transcribe_workflow_commits_the_ledger_update_once_before_the_push_retry_loop():
    # The commit must happen once, outside the retry loop — a successful rebase
    # already carries the commit forward, so re-committing on every loop
    # iteration would either duplicate it or (worse) silently no-op the "nothing
    # staged" check and report success without ever pushing.
    text = workflow_text()
    commit_count = text.count('git commit -m "chore(data): record transcription progress [skip ci]"')
    assert commit_count == 2  # once up front, once after a redo following a real conflict


def test_transcribe_workflow_wires_whisper_decoding_vars():
    # poller/config.py reads these (spec 0003); unset repo variables arrive as empty
    # strings and fall back to the defaults, so wiring them costs nothing until set.
    text = workflow_text()
    assert "WHISPER_BEAM_SIZE: ${{ vars.WHISPER_BEAM_SIZE }}" in text
    assert "WHISPER_CONDITION_ON_PREVIOUS_TEXT: ${{ vars.WHISPER_CONDITION_ON_PREVIOUS_TEXT }}" in text
    assert "WHISPER_DOMAIN_PROMPT: ${{ vars.WHISPER_DOMAIN_PROMPT }}" in text
    assert (
        "WHISPER_HALLUCINATION_SILENCE_THRESHOLD: ${{ vars.WHISPER_HALLUCINATION_SILENCE_THRESHOLD }}" in text
    )


def test_transcribe_workflow_does_not_interpolate_dispatch_inputs_into_run_scripts():
    # `${{ inputs.x }}` inside a `run:` body is text substitution before the
    # shell sees the script — a crafted dispatch input becomes code with
    # access to every secret in the job (CONTENT_REPO_TOKEN, HF_TOKEN, etc.).
    # Inputs must be passed through `env:` and referenced as shell variables.
    for body in run_block_bodies(workflow_text()):
        assert "${{ inputs." not in body
        assert not STEPS_OUTPUTS_RE.search(body)


def test_transcribe_workflow_passes_inputs_through_env_not_interpolation():
    # Once, at workflow level, so the plan and every shard read the same values.
    workflow_env = workflow_text()[: workflow_text().index("\njobs:")]
    assert "  LIMIT: ${{ inputs.limit || 5 }}" in workflow_env
    assert "  CHURCH: ${{ inputs.church }}" in workflow_env
    assert "  GUIDS: ${{ inputs.guids }}" in workflow_env
    for job in ("plan", "transcribe"):
        assert "LIMIT:" not in _job_text(job), "a step-level LIMIT would let the plan and shards disagree"


def test_transcribe_workflow_validates_limit_and_church_inputs_before_use():
    # Defense in depth even with the injection vector closed above: an
    # unvalidated LIMIT/CHURCH still reaches the CLI as an arbitrary string.
    text = workflow_text()
    assert "^[1-9][0-9]*$" in text
    assert "^[a-z_]*$" in text


def test_transcribe_workflow_validates_ledgers_before_every_git_add():
    # A truncated ledger must fail the push step rather than reach `git add
    # data/` and land on main with `[skip ci]` — see poller/store.py:validate_all.
    # Two `git add data/` sites: the normal commit and the post-rebase-conflict
    # redo — both must be guarded.
    text = workflow_text()
    git_add_positions = [m.start() for m in re.finditer(re.escape("git add data/"), text)]
    validate_positions = [m.start() for m in re.finditer(re.escape("python -m poller.store validate"), text)]
    assert len(git_add_positions) == 2
    assert len(validate_positions) == 2
    for git_add_pos in git_add_positions:
        assert any(validate_pos < git_add_pos for validate_pos in validate_positions)


def _job_text(job: str) -> str:
    """The text of one top-level job (`plan` or `transcribe`), up to the next job."""
    text = workflow_text()
    start = text.index(f"\n  {job}:\n")
    next_job = re.search(r"\n  [\w-]+:\n", text[start + 1 :])
    return text[start : start + 1 + next_job.start()] if next_job else text[start:]


def test_transcribe_workflow_plans_against_the_live_ledger_not_the_queued_snapshot():
    # Regression (issue #84): a run queued behind another in the concurrency group
    # checked out github.sha — main as of when it was *queued* — so it re-selected
    # sermons the run ahead of it had already marked done, re-transcribed them, and
    # failed on Content's overwrite guard. The plan job must read main's tip as of
    # when it actually starts.
    plan = _job_text("plan")
    assert "ref: ${{ github.ref }}" in plan
    assert "git rev-parse HEAD" in plan
    assert "ledger_sha: ${{ steps.compute.outputs.ledger_sha }}" in plan


def test_transcribe_workflow_shards_select_from_the_same_ledger_the_plan_counted():
    # Shards partition the plan's ordered selection by position (ADR-0005); a poll.yml
    # commit landing between plan and shard checkout would shift that order and let
    # two shards transcribe the same sermon, so every shard pins the plan's commit.
    transcribe = _job_text("transcribe")
    assert "ref: ${{ needs.plan.outputs.ledger_sha }}" in transcribe


def test_transcribe_workflow_pushes_the_ledger_from_a_detached_checkout():
    # Checking out a commit SHA leaves HEAD detached, where a bare `git push` has no
    # branch to push — every push must name main explicitly.
    transcribe = _job_text("transcribe")
    assert "if git push origin HEAD:main; then" in transcribe
    assert not re.search(r"git push\s*;", transcribe)


def test_transcribe_job_is_skipped_when_the_plan_counts_no_shards():
    # An empty matrix (`fromJson('[]')`) fails the whole run even though every job
    # that ran is green, so an idle tick reads as an outage. Skipping the job leaves
    # the run green and keeps a red run meaning something broke.
    job = _job_text("transcribe")
    assert "if: ${{ needs.plan.outputs.shard_count != '0' }}" in job


def test_status_publishes_on_every_planned_tick_even_with_no_shards():
    # A tick with nothing pending runs zero shards, so the transcribe job is skipped;
    # the status file must still refresh (spec 0009), so the job keys off the plan
    # succeeding and a status function, not off transcribe having run.
    job = _job_text("publish-status")
    assert "needs: [plan, transcribe]" in job
    assert "if: ${{ !cancelled() && needs.plan.result == 'success' }}" in job
    assert "timeout-minutes:" in job


def test_status_publishes_from_the_ledger_every_shard_already_pushed():
    # Read main's tip when the job starts, after the shards' ledger pushes — not the
    # plan's pinned commit, which predates this run's own progress.
    job = _job_text("publish-status")
    assert "ref: ${{ github.ref }}" in job
    assert "ledger_sha" not in job
    assert "python -m poller.status" in job


_NOTIFY_SECRETS = ("NOTIFY_EMAIL_FROM", "NOTIFY_EMAIL_TO", "RESEND_API_KEY")


def _job(text: str, name: str) -> str:
    """One top-level job's block, by its two-space-indented key."""
    match = re.search(rf"^  {name}:\n(.*?)(?=^  [\w-]+:\n|\Z)", text, re.DOTALL | re.MULTILINE)
    assert match, f"no job {name!r}"
    return match.group(1)


def test_the_transcribe_job_receives_the_notify_secrets_for_quality_alerts():
    # ADR-0023: a flagged transcript is emailed from the shard that produced it.
    job = _job(workflow_text(), "transcribe")
    for var in _NOTIFY_SECRETS:
        assert f"{var}: ${{{{ secrets.{var} }}}}" in job


def test_the_transcribe_step_fails_fast_when_a_notify_secret_is_missing():
    # Missing wiring must show on the first run, not on the first flagged sermon.
    [body] = [b for b in run_block_bodies(_job(workflow_text(), "transcribe")) if "--shard-index" in b]
    check = body.index("::error::notify secret")
    assert check < body.index("python -m poller.transcriber")
    for var in _NOTIFY_SECRETS:
        assert var in body[:check]


def test_no_secret_is_interpolated_into_a_run_script():
    for body in run_block_bodies(workflow_text()):
        assert "secrets." not in body


# --- Re-transcription mode (ADR-0022) ------------------------------------------------------

_MODE_EXPR = "${{ inputs.mode || 'pending' }}"


def test_only_the_pending_tick_is_scheduled_and_the_pass_is_dispatched_by_an_external_scheduler():
    # GitHub's own cron is delayed or dropped under load (ADR-0011); the re-transcription
    # pass is driven by workflow_dispatch from cron-job.org, like poll.yml.
    text = workflow_text()
    on_block = text[text.index("\non:") : text.index("\nconcurrency:")]
    assert on_block.count("cron:") == 1
    assert '    - cron: "*/15 * * * *"' in on_block
    assert "github.event.schedule" not in text


def test_mode_is_a_dispatch_choice_with_pending_as_the_default():
    text = workflow_text()
    inputs = text[text.index("    inputs:") : text.index("\nconcurrency:")]
    assert "      mode:\n" in inputs
    assert "        type: choice" in inputs
    assert "          - pending\n          - retranscribe-stale\n        default: pending" in inputs
    assert "      guids:\n" in inputs


def test_each_mode_has_its_own_concurrency_group():
    # A multi-hour pass must never hold up a new sermon's transcription.
    text = workflow_text()
    assert f"  group: transcribe-sermons-{_MODE_EXPR}\n" in text
    assert f"  TRANSCRIBE_MODE: {_MODE_EXPR}\n" in text


def test_the_pass_caps_parallelism_and_bounds_each_shard():
    job = _job_text("transcribe")
    assert "max-parallel: ${{ fromJSON(needs.plan.outputs.max_parallel) }}" in job
    assert "timeout-minutes: ${{ needs.plan.outputs.mode == 'retranscribe-stale' && 180 || 360 }}" in job


def test_verify_runs_after_every_retranscribe_run_against_mains_tip():
    job = _job_text("verify")
    assert "needs: [plan, transcribe]" in job
    assert (
        "if: ${{ !cancelled() && needs.plan.result == 'success' && needs.plan.outputs.mode == 'retranscribe-stale' }}"
        in job
    )
    assert "timeout-minutes: 10" in job
    assert "ref: ${{ github.ref }}" in job
    assert "python -m poller.verify_content" in job


def test_verify_receives_the_notify_secrets_and_checks_them_before_verifying():
    job = _job(workflow_text(), "verify")
    for var in _NOTIFY_SECRETS:
        assert f"{var}: ${{{{ secrets.{var} }}}}" in job
    [body] = [b for b in run_block_bodies(job) if "verify_content" in b]
    assert body.index("::error::notify secret") < body.index("python -m poller.verify_content")


def _run_script(
    body: str, tmp_path, env: dict[str, str], *, count: str = "3"
) -> tuple[int, str, list[str], str]:
    """Run a `run:` body under bash with a stub `python` that logs its argv.

    Returns the exit code, stdout, the logged python invocations and $GITHUB_OUTPUT.
    """
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    log = tmp_path / "python.log"
    stub = bin_dir / "python"
    stub.write_text(
        "#!/bin/bash\n"
        f'printf "%s\\x1f" "$@" >> "{log}"\n'
        f'echo >> "{log}"\n'
        f'if [[ " $* " == *" --print-shard-count "* ]]; then echo {count}; fi\n'
    )
    stub.chmod(0o755)
    output = tmp_path / "github_output"
    output.write_text("")
    summary = tmp_path / "summary"
    result = subprocess.run(
        ["bash", "-e", "-c", body],
        check=False,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "GITHUB_OUTPUT": str(output),
            "GITHUB_ENV": str(tmp_path / "github_env"),
            "GITHUB_STEP_SUMMARY": str(summary),
            **env,
        },
    )
    calls = [line.split("\x1f")[:-1] for line in log.read_text().splitlines()] if log.exists() else []
    return result.returncode, result.stdout, calls, output.read_text()


def _plan_body() -> str:
    [body] = [b for b in run_block_bodies(_job_text("plan")) if "--print-shard-count" in b]
    return body.replace("$(git rev-parse HEAD)", "deadbeef")


_PLAN_ENV = {
    "RETRANSCRIBE_MAX_PARALLEL": "",
    "TRANSCRIBE_MODE": "pending",
    "LIMIT": "5",
    "CHURCH": "",
    "GUIDS": "",
}


def test_the_plan_counts_the_pending_selection_by_default(tmp_path):
    code, _out, calls, output = _run_script(_plan_body(), tmp_path, _PLAN_ENV)
    assert code == 0
    assert calls == [["-m", "poller.transcriber", "--limit", "5", "--print-shard-count"]]
    assert "mode=pending\n" in output
    assert "shards=[0,1,2]\n" in output


def test_the_plan_counts_the_stale_selection_narrowed_to_the_dispatched_guids(tmp_path):
    env = {
        **_PLAN_ENV,
        "TRANSCRIBE_MODE": "retranscribe-stale",
        "GUIDS": "menlo:a1, part 2;gracepres:x/y?z=1",
    }
    code, _out, calls, output = _run_script(_plan_body(), tmp_path, env)
    assert code == 0
    assert calls == [
        [
            "-m",
            "poller.transcriber",
            "--limit",
            "5",
            "--retranscribe-stale",
            "--guid",
            "menlo:a1, part 2",
            "--guid",
            "gracepres:x/y?z=1",
            "--print-shard-count",
        ]
    ]
    assert "mode=retranscribe-stale\n" in output


@pytest.mark.parametrize("guid", ["a$(id)", "a`id`", 'a"b', "-x", " a"])
def test_the_plan_refuses_a_guid_outside_the_ledger_charset(tmp_path, guid):
    env = {**_PLAN_ENV, "TRANSCRIBE_MODE": "retranscribe-stale", "GUIDS": guid}
    code, out, calls, _output = _run_script(_plan_body(), tmp_path, env)
    assert code != 0
    assert calls == []
    assert "::error::GUIDS entry" in out


def test_the_plan_refuses_guids_in_pending_mode_and_an_unknown_mode(tmp_path):
    assert _run_script(_plan_body(), tmp_path, {**_PLAN_ENV, "GUIDS": "menlo:a"})[0] != 0
    assert _run_script(_plan_body(), tmp_path, {**_PLAN_ENV, "TRANSCRIBE_MODE": "everything"})[0] != 0


def test_an_unnarrowed_retranscribe_dispatch_is_the_unattended_pass_that_owes_absence_alerts(tmp_path):
    # cron-job.org dispatches mode and limit only; a canary names guids and is a one-off.
    env = {**_PLAN_ENV, "TRANSCRIBE_MODE": "retranscribe-stale"}
    code, _out, calls, output = _run_script(_plan_body(), tmp_path, env)
    assert code == 0
    assert "--retranscribe-stale" in calls[0]
    assert "mode=retranscribe-stale\n" in output
    assert "scheduled=true\n" in output

    (tmp_path / "python.log").unlink()
    code, _out, calls, output = _run_script(_plan_body(), tmp_path, {**env, "GUIDS": "menlo:a"})
    assert code == 0
    assert "scheduled=false\n" in output


def test_a_pending_run_is_never_the_scheduled_pass(tmp_path):
    code, _out, _calls, output = _run_script(_plan_body(), tmp_path, _PLAN_ENV)
    assert code == 0
    assert "scheduled=false\n" in output


def test_there_is_no_repo_variable_kill_switch_for_the_pass():
    # Disabling the cron-job.org job is the off switch; a variable would make a dispatched
    # run that does nothing look like a healthy green run (CLAUDE.md section 6, Controls).
    assert "RETRANSCRIBE_ENABLED" not in workflow_text()
    assert "RETRANSCRIBE_LIMIT" not in workflow_text()


def test_the_plan_reads_the_passs_parallelism_from_a_variable_defaulting_to_32(tmp_path):
    assert "RETRANSCRIBE_MAX_PARALLEL: ${{ vars.RETRANSCRIBE_MAX_PARALLEL }}" in _job_text("plan")
    stale = {**_PLAN_ENV, "TRANSCRIBE_MODE": "retranscribe-stale"}

    assert "max_parallel=256\n" in _run_script(_plan_body(), tmp_path, _PLAN_ENV)[3]
    assert "max_parallel=32\n" in _run_script(_plan_body(), tmp_path, stale)[3]
    assert (
        "max_parallel=12\n"
        in _run_script(_plan_body(), tmp_path, {**stale, "RETRANSCRIBE_MAX_PARALLEL": "12"})[3]
    )


@pytest.mark.parametrize("value", ["0", "-4", "eight", "257", "1e3"])
def test_the_plan_refuses_a_parallelism_github_would_not_accept(tmp_path, value):
    env = {**_PLAN_ENV, "TRANSCRIBE_MODE": "retranscribe-stale", "RETRANSCRIBE_MAX_PARALLEL": value}
    code, out, _calls, _output = _run_script(_plan_body(), tmp_path, env)
    assert code != 0
    assert "RETRANSCRIBE_MAX_PARALLEL" in out


def test_the_job_reads_the_parallelism_the_plan_validated():
    assert "max_parallel: ${{ steps.compute.outputs.max_parallel }}" in _job_text("plan")


def test_the_plan_refuses_more_shards_than_github_allows_in_a_matrix(tmp_path):
    code, out, _calls, _output = _run_script(_plan_body(), tmp_path, _PLAN_ENV, count="257")
    assert code != 0
    assert "within 256 shards" in out


def _shard_body() -> str:
    [body] = [b for b in run_block_bodies(_job(workflow_text(), "transcribe")) if "--shard-index" in b]
    return (
        body.replace("${{ runner.temp }}", ".")
        .replace("${{ matrix.shard }}", "1")
        .replace("${{ needs.plan.outputs.shard_count }}", "2")
    )


_SHARD_ENV = {
    **{var: "set" for var in _NOTIFY_SECRETS},
    "LIMIT": "1",
    "CHURCH": "",
    "GUIDS": "",
}


def test_a_pending_shard_runs_the_pending_selection(tmp_path):
    code, _out, calls, _output = _run_script(_shard_body(), tmp_path, {**_SHARD_ENV, "MODE": "pending"})
    assert code == 0
    [call] = calls
    assert "--retranscribe-stale" not in call
    assert "--summary-md" not in call


def test_a_retranscribe_shard_runs_the_same_selection_the_plan_counted(tmp_path):
    env = {**_SHARD_ENV, "MODE": "retranscribe-stale", "GUIDS": "menlo:a1, part 2;pbc:b"}
    code, _out, calls, _output = _run_script(_shard_body(), tmp_path, env)
    assert code == 0
    [call] = calls
    assert call[:9] == [
        "-m",
        "poller.transcriber",
        "--limit",
        "1",
        "--shard-index",
        "1",
        "--shard-count",
        "2",
        "--marks-out",
    ]
    assert call[10:] == [
        "--retranscribe-stale",
        "--summary-md",
        str(tmp_path / "summary"),
        "--guid",
        "menlo:a1, part 2",
        "--guid",
        "pbc:b",
    ]


def _verify_body() -> str:
    [body] = [b for b in run_block_bodies(_job(workflow_text(), "verify")) if "verify_content" in b]
    return body


def test_verify_alerts_on_absence_only_on_the_scheduled_pass(tmp_path):
    env = {**{var: "set" for var in _NOTIFY_SECRETS}, "PLANNED": "4"}
    _code, _out, calls, _output = _run_script(_verify_body(), tmp_path, {**env, "SCHEDULED": "false"})
    assert calls == [["-m", "poller.verify_content", "--summary-md", str(tmp_path / "summary")]]

    (tmp_path / "python.log").unlink()
    _code, _out, calls, _output = _run_script(_verify_body(), tmp_path, {**env, "SCHEDULED": "true"})
    assert calls == [
        [
            "-m",
            "poller.verify_content",
            "--summary-md",
            str(tmp_path / "summary"),
            "--scheduled",
            "--planned",
            "4",
        ]
    ]
