"""Contract test on poll.yml's own declared trigger/concurrency config.

The repo has no YAML-parsing dependency (stdlib-only stack), so this matches
on the file's text rather than a parsed structure. See
docs/plans/chore-poll-external-dispatch-trigger.md for the tradeoff.
"""

import re
from pathlib import Path

from push_job import PushJob, run_block

from poller.config import CHURCH_KEY_RE

WORKFLOW = Path(__file__).parent.parent / ".github" / "workflows" / "poll.yml"

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


def test_poll_workflow_does_not_cancel_in_progress_runs():
    # A dispatch (e.g. from an external scheduler) must queue behind an
    # in-flight run rather than kill it — see ADR-0002.
    assert "cancel-in-progress: false" in workflow_text()


def test_poll_workflow_accepts_workflow_dispatch():
    # External schedulers trigger polling via the workflow_dispatch REST API,
    # not GitHub's schedule: event — see ADR-0002.
    assert "workflow_dispatch:" in workflow_text()


def test_poll_workflow_has_no_schedule_trigger():
    # GitHub's schedule: event was dropped as a fallback — it is delayed or
    # dropped under load for the same reason it isn't the primary trigger,
    # so workflow_dispatch (cron-job.org) is now the sole trigger — ADR-0011.
    assert "\n  schedule:" not in workflow_text()


def test_poll_workflow_dispatches_transcription_for_discovered_churches():
    # After a successful ledger push, dispatch transcribe.yml for each newly-
    # discovered church rather than waiting for its own schedule — see ADR-0007.
    assert "gh workflow run transcribe.yml" in workflow_text()


def test_poll_workflow_dispatch_uses_the_dedicated_dispatch_token():
    # The default GITHUB_TOKEN cannot trigger another workflow's
    # workflow_dispatch event — a dedicated PAT is required (ADR-0007).
    assert "secrets.TRANSCRIBE_DISPATCH_TOKEN" in workflow_text()


def test_poll_workflow_dispatch_is_gated_on_the_commit_step_succeeding():
    # Dispatching against a ledger update that failed to push would have
    # transcribe.yml find nothing pending — the dispatch step must depend on
    # the commit step's own outcome (ADR-0007).
    assert "steps.commit.outcome == 'success'" in workflow_text()


def test_poll_workflow_commit_step_has_an_id():
    # The dispatch step's gate above depends on this id existing.
    assert "id: commit" in workflow_text()


def test_poll_workflow_passes_the_claude_batch_poll_config_to_the_poll_step():
    # runner.run() now also checks pending Claude batches every poll cycle
    # (ADR-0010) — without these, poller/batch_poll.py silently no-ops.
    text = workflow_text()
    assert "ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}" in text
    assert "PIPELINE_REPO: ${{ vars.PIPELINE_REPO }}" in text
    assert "PIPELINE_DISPATCH_TOKEN: ${{ secrets.PIPELINE_DISPATCH_TOKEN }}" in text


def test_poll_workflow_does_not_interpolate_dispatch_inputs_into_run_scripts():
    # `${{ inputs.x }}` inside a `run:` body is text substitution before the
    # shell sees the script — a crafted dispatch input becomes code with
    # access to every secret in the job. Inputs must be passed through `env:`
    # and referenced as shell variables instead.
    for body in run_block_bodies(workflow_text()):
        assert "${{ inputs." not in body
        assert not STEPS_OUTPUTS_RE.search(body)


def test_poll_workflow_passes_inputs_through_env_not_interpolation():
    text = workflow_text()
    assert "BACKFILL: ${{ inputs.backfill }}" in text
    assert "CHURCH: ${{ inputs.church }}" in text
    assert "DISCOVERED: ${{ needs.poll.outputs.discovered }}" in text
    assert "discovered: ${{ steps.poll.outputs.discovered }}" in _job_text("poll")


def test_poll_workflow_validates_church_input_before_use():
    # Defense in depth even with the injection vector closed above: an
    # unvalidated CHURCH still reaches the CLI as an arbitrary string.
    # The same rule load_churches applies to the table, so a church Pipeline adds is
    # never one this input rejects.
    assert re.findall(r'\[\[ "\$CHURCH" =~ (\S+) \]\]', workflow_text()) == [CHURCH_KEY_RE.pattern]


def test_poll_workflow_refuses_staged_paths_outside_the_ledger_before_committing():
    # apply parses every artifact file as a ledger before writing, and check-staged refuses
    # anything else that reached the index (ADR-0027), so neither a truncated ledger nor a
    # non-ledger path lands on main with `[skip ci]`.
    script = commit_step_script()
    assert script.index("ledger_push apply") < script.index("git add data/")
    assert (
        script.index("git add data/") < script.index("ledger_push check-staged") < script.index("git commit")
    )


def test_poll_workflow_reads_the_church_table_from_pipeline_not_a_variable():
    """The table is Pipeline's config/churches.json (ADR-0016); the Poll step needs its credentials.

    `poller.runner` loads it through PIPELINE_REPO/PIPELINE_DISPATCH_TOKEN, so a Poll step
    without them fails every run. A `CHURCHES` variable here would be a second copy.
    """
    text = workflow_text()
    assert "CHURCHES" not in text
    poll_step = text[text.index("- name: Poll\n") : text.index("- name: Commit newly-discovered sermons")]
    assert "PIPELINE_REPO: ${{ vars.PIPELINE_REPO }}" in poll_step
    assert "PIPELINE_DISPATCH_TOKEN: ${{ secrets.PIPELINE_DISPATCH_TOKEN }}" in poll_step


def test_poll_workflow_checks_out_the_live_branch_tip_not_the_trigger_sha():
    """A run queued behind another must read that run's committed alert state (ADR-0025).

    `actions/checkout` defaults to `github.sha`, the commit as of dispatch, so a queued run
    would re-send an alert the run ahead of it already sent and recorded.
    """
    text = workflow_text()
    checkout = text[text.index("uses: actions/checkout@") : text.index("uses: actions/setup-python@")]
    assert "ref: ${{ github.ref }}" in checkout


def _job_text(job: str) -> str:
    """One top-level job's text, up to the next job."""
    text = workflow_text()
    start = text.index(f"\n  {job}:\n")
    next_job = re.search(r"\n  [\w-]+:\n", text[start + 1 :])
    return text[start : start + 1 + next_job.start()] if next_job else text[start:]


def _code(text: str) -> str:
    """``text`` without its comment lines, which may name what the code must not do."""
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def commit_step_script() -> str:
    push = _job_text("push")
    return run_block(push[push.index("- name: Commit newly-discovered sermons") :])


# --- The push credential stays out of every job that runs third-party code (ADR-0027) -------


def test_only_the_push_job_holds_the_deploy_key():
    assert "DEPLOY_KEY" not in _code(_job_text("poll"))
    assert "ssh-key: ${{ secrets.DEPLOY_KEY }}" in _job_text("push")
    assert workflow_text().count("secrets.DEPLOY_KEY") == 1


def test_the_push_job_installs_nothing_and_restores_no_pip_cache():
    push = _code(_job_text("push"))
    assert "pip install" not in push
    assert "cache:" not in push
    assert "requirements" not in push


def test_the_push_job_runs_python_only_without_site_packages():
    # `-S`: a third-party import on this path fails instead of running next to the key.
    calls = re.findall(r"python[^\n]*", commit_step_script())
    assert calls
    assert all(call.startswith("python -E -S -m poller.ledger_push ") for call in calls), calls


def test_the_poll_job_uploads_its_ledger_writes_even_when_the_poll_step_fails():
    poll = _job_text("poll")
    stage = poll[poll.index("- name: Stage ledger changes") :]
    assert stage.index("if: always()") < stage.index("python -m poller.ledger_push stage")
    upload = poll[poll.index("- name: Upload ledger changes") :]
    assert upload.index("if: always()") < upload.index("uses: actions/upload-artifact@")
    assert "name: ledger-poll" in upload
    assert "if-no-files-found: error" in upload


def test_the_push_job_runs_after_a_failed_poll_but_not_a_cancelled_one():
    push = _job_text("push")
    assert "needs: poll" in push
    assert "if: ${{ !cancelled() }}" in push
    assert "timeout-minutes:" in push
    assert "name: ledger-poll" in push


def test_the_push_job_checks_out_the_live_tip_and_pushes_to_it():
    push = _job_text("push")
    assert "ref: ${{ github.ref }}" in push
    script = commit_step_script()
    assert 'git push origin "HEAD:$GITHUB_REF_NAME"' in script
    assert not re.search(r"git push\s*$", script, re.MULTILINE)


def test_the_push_job_never_repolls():
    # A re-poll re-sends that run's emails, and would run feedparser next to the key.
    assert "poller.runner" not in _job_text("push")


# --- The push job's step, run end to end ---------------------------------------------------------

ABIDE = {
    "guid": "https://pbc.org?enmse_mid=4691",
    "title": "Abide in Me",
    "preached_on": "2026-10-18",
    "feed_published_at": "2026-10-19T18:02:11+00:00",
    "audio_url": "https://s3.us-west-2.amazonaws.com/cdn.pbc.org/Main_Service/2026/10/18/a.mp3",
    "notified_at": "2026-10-19T18:05:00+00:00",
    "transcription_status": "pending",
}
ROOTED = {
    "guid": "hillside-2026-11-01",
    "title": "Rooted",
    "preached_on": "2026-11-01",
    "feed_published_at": "2026-11-01T22:00:00+00:00",
    "notified_at": "2026-11-01T22:04:31+00:00",
    "transcription_status": "pending",
}
DONE = {
    "transcription_status": "done",
    "transcript_hash": "e23512004d655969cec6b45873c4399fe68a16b95f2e5e6343c3e9ae4399b74d",
    "content_path": "transcripts/pbc/2026-10-18_abide-in-me_https_pbc.org_enmse_mid_4691.txt",
}
SEED = {"data/pbc.json": {ABIDE["guid"]: ABIDE}, "data/hillside.json": {}}


def test_the_push_step_lands_a_discovery_on_a_tip_a_shard_moved(tmp_path):
    job = PushJob(tmp_path, SEED)
    job.produce("poll", {"data/hillside.json": {ROOTED["guid"]: ROOTED}})
    job.land_first({"data/pbc.json": {ABIDE["guid"]: {**ABIDE, **DONE}}})

    result = job.run(commit_step_script(), expected="1")

    assert result.returncode == 0, result.stdout + result.stderr
    assert job.main_json("data/pbc.json") == {ABIDE["guid"]: {**ABIDE, **DONE}}
    assert job.main_json("data/hillside.json") == {ROOTED["guid"]: ROOTED}


def test_the_push_step_merges_a_feed_refresh_and_a_transcription_of_one_sermon(tmp_path):
    # As text these collide (adjacent lines), which used to cost a re-poll and a second email.
    job = PushJob(tmp_path, SEED)
    job.produce("poll", {"data/pbc.json": {ABIDE["guid"]: {**ABIDE, "title": "Abide in Me (Part 1)"}}})
    job.land_first({"data/pbc.json": {ABIDE["guid"]: {**ABIDE, **DONE}}})

    result = job.run(commit_step_script(), expected="1")

    assert result.returncode == 0, result.stdout + result.stderr
    assert job.main_json("data/pbc.json") == {
        ABIDE["guid"]: {**ABIDE, **DONE, "title": "Abide in Me (Part 1)"}
    }


def test_the_push_step_retries_a_rejected_push_onto_the_fresh_tip(tmp_path):
    job = PushJob(tmp_path, SEED)
    job.produce("poll", {"data/hillside.json": {ROOTED["guid"]: ROOTED}})
    job.reject_pushes(times=1)

    result = job.run(commit_step_script(), expected="1")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "push rejected (attempt 1/3)" in result.stdout
    assert job.main_json("data/hillside.json") == {ROOTED["guid"]: ROOTED}


def test_the_push_step_fails_when_every_push_attempt_is_rejected(tmp_path):
    job = PushJob(tmp_path, SEED)
    job.produce("poll", {"data/hillside.json": {ROOTED["guid"]: ROOTED}})
    job.reject_pushes()

    result = job.run(commit_step_script(), expected="1")

    assert result.returncode != 0
    assert "failed to push ledger update after 3 attempts" in result.stdout
    assert job.main_json("data/hillside.json") == {}


def test_the_push_step_refuses_an_artifact_carrying_a_workflow(tmp_path):
    job = PushJob(tmp_path, SEED)
    job.produce("poll", {"data/hillside.json": {ROOTED["guid"]: ROOTED}})
    evil = job.ledger_dir / "poll" / ".github" / "workflows" / "poll.yml"
    evil.parent.mkdir(parents=True)
    evil.write_text("on: push\n")

    result = job.run(commit_step_script(), expected="1")

    assert result.returncode != 0
    assert ".github/workflows/poll.yml" in result.stderr
    assert job.main_paths() == ["data/hillside.json", "data/pbc.json"]
    assert job.commit_count() == 1


def test_the_push_step_fails_when_the_poll_artifact_never_arrived(tmp_path):
    job = PushJob(tmp_path, SEED)
    job.ledger_dir.mkdir()

    result = job.run(commit_step_script(), expected="1")

    assert result.returncode != 0
    assert "0 of 1 staged ledger artifacts arrived" in result.stdout


def test_the_push_step_with_nothing_discovered_commits_nothing(tmp_path):
    job = PushJob(tmp_path, SEED)
    job.produce("poll", {})

    result = job.run(commit_step_script(), expected="1")

    assert result.returncode == 0, result.stdout + result.stderr
    assert "No new sermons discovered this run." in result.stdout
    assert job.commit_count() == 1
