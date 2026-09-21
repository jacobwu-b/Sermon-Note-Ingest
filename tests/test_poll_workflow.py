"""Contract test on poll.yml's own declared trigger/concurrency config.

The repo has no YAML-parsing dependency (stdlib-only stack), so this matches
on the file's text rather than a parsed structure. See
docs/plans/chore-poll-external-dispatch-trigger.md for the tradeoff.
"""

import re
from pathlib import Path

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
    assert "DISCOVERED: ${{ steps.poll.outputs.discovered }}" in text


def test_poll_workflow_validates_church_input_before_use():
    # Defense in depth even with the injection vector closed above: an
    # unvalidated CHURCH still reaches the CLI as an arbitrary string.
    text = workflow_text()
    assert "^[a-z_]*$" in text
