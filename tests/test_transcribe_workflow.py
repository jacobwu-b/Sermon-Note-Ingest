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
