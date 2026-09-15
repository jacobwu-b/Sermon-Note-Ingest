"""Contract test on poll.yml's own declared trigger/concurrency config.

The repo has no YAML-parsing dependency (stdlib-only stack), so this matches
on the file's text rather than a parsed structure. See
docs/plans/chore-poll-external-dispatch-trigger.md for the tradeoff.
"""

from pathlib import Path

WORKFLOW = Path(__file__).parent.parent / ".github" / "workflows" / "poll.yml"


def workflow_text() -> str:
    return WORKFLOW.read_text()


def test_poll_workflow_does_not_cancel_in_progress_runs():
    # A dispatch (e.g. from an external scheduler) must queue behind an
    # in-flight run rather than kill it — see ADR-0002.
    assert "cancel-in-progress: false" in workflow_text()


def test_poll_workflow_accepts_workflow_dispatch():
    # External schedulers trigger polling via the workflow_dispatch REST API,
    # not GitHub's schedule: event — see ADR-0002.
    assert "workflow_dispatch:" in workflow_text()


def test_poll_workflow_keeps_schedule_fallback():
    # Retained as a zero-cost fallback in case the external trigger fails
    # silently — see ADR-0002.
    assert "schedule:" in workflow_text()


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
