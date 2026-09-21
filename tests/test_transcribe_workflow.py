"""Contract test on transcribe.yml's own declared trigger/timeout config.

The repo has no YAML-parsing dependency (stdlib-only stack), so this matches on
the file's text rather than a parsed structure — the same tradeoff
test_poll_workflow.py makes for poll.yml.
"""

import re
from pathlib import Path

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
    # Sermon-Note-Pipeline's ingest-event dispatch after a successful transcription.
    text = workflow_text()
    assert "PIPELINE_REPO: ${{ vars.PIPELINE_REPO }}" in text
    assert "PIPELINE_DISPATCH_TOKEN: ${{ secrets.PIPELINE_DISPATCH_TOKEN }}" in text


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


def test_transcribe_workflow_rebases_onto_main_before_redoing_a_shard_s_batch():
    # Regression: a rejected push used to unconditionally `git reset --hard` and
    # re-run the whole shard's transcriber invocation, discarding this shard's
    # own already-completed (expensive, Whisper) work on every race with
    # poll.yml's independent commits — not just on a genuine conflict. That
    # turned a routine push race into a repeated ~50-minute redo, exhausting the
    # retry budget in hours. A rejected push must try a rebase first, so an
    # unrelated writer's commit (almost always a different record) merges
    # cleanly instead of discarding this shard's progress.
    text = workflow_text()
    assert "git rebase origin/main" in text
    assert "git rebase --abort" in text
    # The expensive redo path must be reached only from a failed rebase, not
    # unconditionally on every rejected push.
    reset_index = text.index("git reset --hard origin/main")
    rebase_abort_index = text.index("git rebase --abort")
    redo_index = text.index("python -m poller.transcriber $TRANSCRIBE_ARGS")
    assert rebase_abort_index < reset_index < redo_index


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


def test_transcribe_workflow_does_not_interpolate_dispatch_inputs_into_run_scripts():
    # `${{ inputs.x }}` inside a `run:` body is text substitution before the
    # shell sees the script — a crafted dispatch input becomes code with
    # access to every secret in the job (CONTENT_REPO_TOKEN, HF_TOKEN, etc.).
    # Inputs must be passed through `env:` and referenced as shell variables.
    for body in run_block_bodies(workflow_text()):
        assert "${{ inputs." not in body
        assert not STEPS_OUTPUTS_RE.search(body)


def test_transcribe_workflow_passes_inputs_through_env_not_interpolation():
    text = workflow_text()
    assert "LIMIT: ${{ inputs.limit || 5 }}" in text
    assert "CHURCH: ${{ inputs.church }}" in text


def test_transcribe_workflow_validates_limit_and_church_inputs_before_use():
    # Defense in depth even with the injection vector closed above: an
    # unvalidated LIMIT/CHURCH still reaches the CLI as an arbitrary string.
    text = workflow_text()
    assert "^[1-9][0-9]*$" in text
    assert "^[a-z_]*$" in text
