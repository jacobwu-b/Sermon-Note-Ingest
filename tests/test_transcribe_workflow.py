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
from push_job import PushJob, run_block

from poller.config import CHURCH_KEY_RE

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
    # and every job needs them to read the church table (ADR-0016). The repo is a
    # workflow-level variable; the token reaches each job through its own step env.
    text = workflow_text()
    workflow_env = text[: text.index("\njobs:")]
    assert "PIPELINE_REPO: ${{ vars.PIPELINE_REPO }}" in workflow_env
    for job in ("plan", "transcribe", "verify", "publish-status"):
        assert "PIPELINE_DISPATCH_TOKEN: ${{ secrets.PIPELINE_DISPATCH_TOKEN }}" in _job_text(job)


# Every secret each job may receive. HF_TOKEN only where weights download; the Content
# token only where Content is read or written; the plan only reads the church table; the
# deploy key only in the job that installs nothing (ADR-0027).
_JOB_SECRETS = {
    "plan": {"PIPELINE_DISPATCH_TOKEN"},
    "transcribe": {
        "CONTENT_REPO_TOKEN",
        "HF_TOKEN",
        "NOTIFY_EMAIL_FROM",
        "NOTIFY_EMAIL_TO",
        "PIPELINE_DISPATCH_TOKEN",
        "RESEND_API_KEY",
    },
    "verify": {
        "CONTENT_REPO_TOKEN",
        "NOTIFY_EMAIL_FROM",
        "NOTIFY_EMAIL_TO",
        "PIPELINE_DISPATCH_TOKEN",
        "RESEND_API_KEY",
    },
    "publish-status": {"CONTENT_REPO_TOKEN", "PIPELINE_DISPATCH_TOKEN"},
    "push": {"DEPLOY_KEY"},
}

_SECRET_RE = re.compile(r"\$\{\{\s*secrets\.(\w+)\s*\}\}")


def test_no_secret_is_set_in_the_workflow_level_env():
    # Workflow-level env reaches every job, so a secret there is handed to jobs that never use it.
    text = workflow_text()
    assert not _SECRET_RE.search(text[: text.index("\njobs:")])


def test_each_job_receives_exactly_its_allowlisted_secrets():
    text = workflow_text()
    jobs = re.findall(r"^  ([\w-]+):\n", text[text.index("\njobs:") :], re.MULTILINE)
    assert set(jobs) == set(_JOB_SECRETS), "a new job needs an entry in _JOB_SECRETS"
    for job, allowed in _JOB_SECRETS.items():
        assert set(_SECRET_RE.findall(_job_text(job))) == allowed, job


def test_no_secret_reaches_a_job_wide_env_or_the_dependency_install():
    # Step-level only, so `pip install` (third-party code) runs with none of them.
    for job in _JOB_SECRETS.keys() - {"push"}:
        text = _job_text(job)
        job_env = re.search(r"^    env:\n((?:      .*\n)+)", text, re.MULTILINE)
        assert not (job_env and _SECRET_RE.search(job_env.group(1))), job
        install = re.search(r"- name: Install dependencies\n((?:        .*\n)+)", text)
        assert install and not _SECRET_RE.search(install.group(1)), job


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


def _code(text: str) -> str:
    """``text`` without its comment lines, which may name what the code must not do."""
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def _push_script() -> str:
    push = _job_text("push")
    return run_block(push[push.index("- name: Push ledger update") :])


def test_no_shard_holds_the_deploy_key():
    # A shard installs the whole Whisper stack; the key bypasses main's ruleset (ADR-0027).
    assert "DEPLOY_KEY" not in _code(_job_text("transcribe"))
    assert workflow_text().count("secrets.DEPLOY_KEY") == 1


def test_the_push_job_installs_nothing_and_restores_no_pip_cache():
    push = _code(_job_text("push"))
    assert "ssh-key: ${{ secrets.DEPLOY_KEY }}" in push
    assert "pip install" not in push
    assert "cache:" not in push
    assert "requirements" not in push


def test_the_push_job_runs_python_only_without_site_packages():
    calls = re.findall(r"python[^\n]*", _push_script())
    assert calls
    assert all(call.startswith("python -E -S -m poller.ledger_push ") for call in calls), calls


def test_every_shard_uploads_its_ledger_writes_even_when_transcription_fails():
    shard = _job_text("transcribe")
    stage = shard[shard.index("- name: Stage ledger changes") :]
    assert stage.index("if: always()") < stage.index("python -m poller.ledger_push stage")
    upload = shard[shard.index("- name: Upload ledger changes") :]
    assert upload.index("if: always()") < upload.index("uses: actions/upload-artifact@")
    assert "name: ledger-shard-${{ matrix.shard }}" in upload
    assert "if-no-files-found: error" in upload


def test_the_push_job_merges_every_shard_once_the_matrix_is_done():
    # always(): a cancelled run's finished shards already pushed to Content, so their marks
    # must still reach the ledger or the next run re-selects them.
    push = _job_text("push")
    assert "needs: [plan, transcribe]" in push
    assert (
        "if: ${{ always() && needs.plan.result == 'success' && needs.plan.outputs.shard_count != '0' }}"
        in push
    )
    assert "pattern: ledger-shard-*" in push
    assert "EXPECTED: ${{ needs.plan.outputs.shard_count }}" in push
    assert "timeout-minutes:" in push


def test_the_push_job_checks_out_the_live_tip_and_pushes_to_it():
    push = _job_text("push")
    assert "ref: ${{ github.ref }}" in push
    assert "ledger_sha" not in push
    script = _push_script()
    assert 'git push origin "HEAD:$GITHUB_REF_NAME"' in script
    assert not re.search(r"git push\s*$", script, re.MULTILINE)


def test_the_push_job_never_rebases_replays_or_redoes_transcription():
    # The merge needs neither the transcriber's import graph nor a Whisper call (ADR-0027).
    text = workflow_text()
    assert "git rebase" not in text
    assert "--replay-marks" not in text
    assert "poller.transcriber" not in _job_text("push")


def test_transcribe_workflow_never_redoes_transcription_on_a_push_conflict():
    # The old full-batch redo (issue #60) stays gone: no invocation of the transcriber with
    # the original run's arguments, and no `|| true` masking a failed recovery step.
    text = workflow_text()
    assert "TRANSCRIBE_ARGS" not in text
    assert "|| true" not in text


def test_transcribe_workflow_wires_whisper_decoding_vars():
    # poller/config.py reads these (spec 0003); unset repo variables arrive as empty
    # strings and fall back to the defaults, so wiring them costs nothing until set.
    text = workflow_text()
    assert "WHISPER_BEAM_SIZE: ${{ vars.WHISPER_BEAM_SIZE }}" in text
    assert "WHISPER_CONDITION_ON_PREVIOUS_TEXT: ${{ vars.WHISPER_CONDITION_ON_PREVIOUS_TEXT }}" in text
    assert "WHISPER_DOMAIN_PROMPT: ${{ vars.WHISPER_DOMAIN_PROMPT }}" in text
    assert "vars.WHISPER_HALLUCINATION_SILENCE_THRESHOLD }}" in text


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
    # The same rule load_churches applies to the table, so a church Pipeline adds is
    # never one this input rejects. Two sites: the plan job and each shard.
    assert re.findall(r'\[\[ "\$CHURCH" =~ (\S+) \]\]', text) == [CHURCH_KEY_RE.pattern] * 2


@pytest.mark.parametrize(
    "key", ["menlo", "north_point", "grace2", "Menlo", "2grace", "st-marks", "../x", "a/b"]
)
def test_bash_s_church_input_check_agrees_with_load_churches(key):
    """Bash's ERE and Python's ``re`` must read ``CHURCH_KEY_RE`` the same way."""
    result = subprocess.run(["bash", "-c", f'[[ "$1" =~ {CHURCH_KEY_RE.pattern} ]]', "_", key], check=False)
    assert (result.returncode == 0) == bool(CHURCH_KEY_RE.fullmatch(key))


def test_the_push_job_refuses_staged_paths_outside_the_ledger_before_committing():
    # apply parses every artifact file as a ledger before writing, and check-staged refuses
    # anything else that reached the index, so neither a truncated ledger nor a non-ledger
    # path lands on main with `[skip ci]`.
    script = _push_script()
    assert script.index("ledger_push apply") < script.index("git add data/")
    assert (
        script.index("git add data/") < script.index("ledger_push check-staged") < script.index("git commit")
    )


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
    assert "needs: [plan, transcribe, push]" in job
    assert "if: ${{ !cancelled() && needs.plan.result == 'success' }}" in job
    assert "timeout-minutes:" in job


def test_status_publishes_from_the_ledger_every_shard_already_pushed():
    # Read main's tip when the job starts, after the ledger push — not the
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
    assert '    - cron: "7 1 * * 1,2,4,6"' in on_block
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
    assert "needs: [plan, transcribe, push]" in job
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


_CANARY_ENV = {
    **_PLAN_ENV,
    "TRANSCRIBE_MODE": "retranscribe-stale",
    "GUIDS": "gracepres:tag:soundcloud,2010:tracks/1448755930",
}


def test_the_silence_guard_override_is_a_dispatch_input_that_beats_the_repo_variable():
    text = workflow_text()
    assert "      silence_guard:" in text
    workflow_env = text[: text.index("\njobs:")]
    assert "  SILENCE_GUARD: ${{ inputs.silence_guard }}" in workflow_env
    assert (
        "WHISPER_HALLUCINATION_SILENCE_THRESHOLD: "
        "${{ inputs.silence_guard || vars.WHISPER_HALLUCINATION_SILENCE_THRESHOLD }}" in workflow_env
    )


def test_a_guid_narrowed_dispatch_may_override_the_silence_guard_including_to_zero(tmp_path):
    for value in ("0", "1.5"):
        env = {**_CANARY_ENV, "SILENCE_GUARD": value, "WHISPER_HALLUCINATION_SILENCE_THRESHOLD": value}
        workdir = tmp_path / f"v{value}"
        workdir.mkdir()
        code, out, calls, _output = _run_script(_plan_body(), workdir, env)
        assert code == 0, out
        assert calls and calls[0][-1] == "--print-shard-count"


def test_the_silence_guard_override_is_named_in_the_run_summary(tmp_path):
    env = {**_CANARY_ENV, "SILENCE_GUARD": "0", "WHISPER_HALLUCINATION_SILENCE_THRESHOLD": "0"}
    code, _out, _calls, _output = _run_script(_plan_body(), tmp_path, env)
    assert code == 0
    assert "silence guard" in (tmp_path / "summary").read_text().lower()


def test_the_plan_refuses_a_silence_guard_override_that_names_no_guids(tmp_path):
    # Every record decoded under the default would turn stale against an unnarrowed override
    # and be re-transcribed under it.
    for mode in ("retranscribe-stale", "pending"):
        env = {
            **_PLAN_ENV,
            "TRANSCRIBE_MODE": mode,
            "SILENCE_GUARD": "0",
            "WHISPER_HALLUCINATION_SILENCE_THRESHOLD": "0",
        }
        workdir = tmp_path / mode
        workdir.mkdir()
        code, out, calls, _output = _run_script(_plan_body(), workdir, env)
        assert code != 0
        assert calls == []
        assert "::error::SILENCE_GUARD" in out


@pytest.mark.parametrize("value", ["abc", "-1", "1e3", "1.", "$(id)", "2 "])
def test_the_plan_refuses_a_silence_guard_override_that_is_not_a_number_of_seconds(tmp_path, value):
    env = {**_CANARY_ENV, "SILENCE_GUARD": value, "WHISPER_HALLUCINATION_SILENCE_THRESHOLD": value}
    code, out, calls, _output = _run_script(_plan_body(), tmp_path, env)
    assert code != 0
    assert calls == []
    assert "::error::SILENCE_GUARD" in out


def test_the_plan_fails_when_the_override_did_not_reach_the_decoder_setting(tmp_path):
    # An expression that dropped the override would fall back to the variable and the pass
    # would run, exit zero and record a profile nobody asked for.
    env = {**_CANARY_ENV, "SILENCE_GUARD": "0", "WHISPER_HALLUCINATION_SILENCE_THRESHOLD": ""}
    code, out, calls, _output = _run_script(_plan_body(), tmp_path, env)
    assert code != 0
    assert calls == []
    assert "did not reach" in out


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
    assert call[:8] == [
        "-m",
        "poller.transcriber",
        "--limit",
        "1",
        "--shard-index",
        "1",
        "--shard-count",
        "2",
    ]
    assert call[8:] == [
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


# --- The push job's step, run end to end ---------------------------------------------------------

ABIDE = {
    "guid": "https://pbc.org?enmse_mid=4691",
    "title": "Abide in Me",
    "preached_on": "2026-10-18",
    "feed_published_at": "2026-10-19T18:02:11+00:00",
    "transcription_status": "pending",
}
CAPTIVES = {
    "guid": "https://pbc.org?enmse_mid=4701",
    "title": "Setting Captives Free",
    "preached_on": "2026-10-25",
    "feed_published_at": "2026-10-26T18:21:06+00:00",
    "transcription_status": "pending",
}


def _done(record: dict, slug: str) -> dict:
    return {
        **record,
        "transcription_status": "done",
        "transcript_hash": f"{slug:0<64}"[:64],
        "content_path": f"transcripts/pbc/{record['preached_on']}_{slug}.txt",
    }


def _two_pending() -> dict:
    return {"data/pbc.json": {CAPTIVES["guid"]: CAPTIVES, ABIDE["guid"]: ABIDE}}


def test_the_push_step_merges_every_shards_writes_onto_a_tip_the_poll_moved(tmp_path):
    job = PushJob(tmp_path, _two_pending())
    job.produce(
        "ledger-shard-0",
        {"data/pbc.json": {CAPTIVES["guid"]: CAPTIVES, ABIDE["guid"]: _done(ABIDE, "abide")}},
    )
    job.produce(
        "ledger-shard-1",
        {"data/pbc.json": {CAPTIVES["guid"]: _done(CAPTIVES, "captives"), ABIDE["guid"]: ABIDE}},
    )
    rooted = {"guid": "hillside-2026-11-01", "title": "Rooted", "preached_on": "2026-11-01"}
    job.land_first(
        {"data/hillside.json": {rooted["guid"]: rooted}}, "chore(data): record poll results [skip ci]"
    )

    result = job.run(_push_script(), expected="2")

    assert result.returncode == 0, result.stdout + result.stderr
    assert job.main_json("data/pbc.json") == {
        CAPTIVES["guid"]: _done(CAPTIVES, "captives"),
        ABIDE["guid"]: _done(ABIDE, "abide"),
    }
    assert job.main_json("data/hillside.json") == {rooted["guid"]: rooted}
    # One ledger commit for the whole run, on top of the seed and the poll's.
    assert job.commit_count() == 3


def test_the_push_step_records_the_shards_that_arrived_and_fails_for_the_one_that_did_not(tmp_path):
    job = PushJob(tmp_path, _two_pending())
    job.produce(
        "ledger-shard-0",
        {"data/pbc.json": {CAPTIVES["guid"]: CAPTIVES, ABIDE["guid"]: _done(ABIDE, "abide")}},
    )

    result = job.run(_push_script(), expected="2")

    assert result.returncode != 0
    assert "1 of 2 staged ledger artifacts arrived" in result.stdout
    assert job.main_json("data/pbc.json")[ABIDE["guid"]]["transcription_status"] == "done"


def test_the_push_step_fails_closed_when_two_shards_wrote_one_field_differently(tmp_path):
    job = PushJob(tmp_path, _two_pending())
    job.produce(
        "ledger-shard-0",
        {"data/pbc.json": {CAPTIVES["guid"]: CAPTIVES, ABIDE["guid"]: _done(ABIDE, "abide")}},
    )
    job.produce(
        "ledger-shard-1",
        {"data/pbc.json": {CAPTIVES["guid"]: CAPTIVES, ABIDE["guid"]: _done(ABIDE, "other")}},
    )

    result = job.run(_push_script(), expected="2")

    assert result.returncode != 0
    assert "changed on both sides" in result.stderr
    assert job.commit_count() == 1
