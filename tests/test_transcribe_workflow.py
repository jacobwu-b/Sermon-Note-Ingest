"""Contract test on transcribe.yml's own declared trigger/timeout config.

The repo has no YAML-parsing dependency (stdlib-only stack), so this matches on
the file's text rather than a parsed structure — the same tradeoff
test_poll_workflow.py makes for poll.yml.
"""

from pathlib import Path

WORKFLOW = Path(__file__).parent.parent / ".github" / "workflows" / "transcribe.yml"


def workflow_text() -> str:
    return WORKFLOW.read_text()


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
