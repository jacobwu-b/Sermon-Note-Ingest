"""Contract test on poll.yml's own declared trigger/concurrency config.

The repo has no YAML-parsing dependency (stdlib-only stack), so this matches
on the file's text rather than a parsed structure. See
docs/plans/chore-poll-external-dispatch-trigger.md for the tradeoff.
"""

import os
import re
import subprocess
from pathlib import Path

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
    assert "DISCOVERED: ${{ steps.poll.outputs.discovered }}" in text


def test_poll_workflow_validates_church_input_before_use():
    # Defense in depth even with the injection vector closed above: an
    # unvalidated CHURCH still reaches the CLI as an arbitrary string.
    # The same rule load_churches applies to the table, so a church Pipeline adds is
    # never one this input rejects.
    assert re.findall(r'\[\[ "\$CHURCH" =~ (\S+) \]\]', workflow_text()) == [CHURCH_KEY_RE.pattern]


def test_poll_workflow_validates_ledgers_before_git_add():
    # A truncated ledger must fail the commit step rather than reach `git add
    # data/` and land on main with `[skip ci]` — see poller/store.py:validate_all.
    text = workflow_text()
    assert text.index("python -m poller.store validate") < text.index("git add data/")


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


def commit_step_script() -> str:
    text = workflow_text()
    step = text[
        text.index("- name: Commit newly-discovered sermons") : text.index("- name: Dispatch transcription")
    ]
    (body,) = run_block_bodies(step)
    return body


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, env=GIT_ENV
    ).stdout


# The developer's own git config (signing, hooks, default branch) must not reach these repos.
GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "Ledger Writer",
    "GIT_AUTHOR_EMAIL": "writer@example.com",
    "GIT_COMMITTER_NAME": "Ledger Writer",
    "GIT_COMMITTER_EMAIL": "writer@example.com",
}

PBC_LEDGER = (
    '[\n  {\n    "guid": "pbc-2026-10-25",\n    "title": "Abide in Me",\n'
    '    "transcription_status": "pending"\n  }\n]\n'
)

HILLSIDE_DISCOVERY = '[\n  {\n    "guid": "hillside-2026-11-01",\n    "title": "Rooted"\n  }\n]\n'

# Stands in for `python -m poller.<module>` at the process boundary: validate passes, and a
# re-poll records its argv and adds a sermon the stale run had also found.
FAKE_PYTHON = """#!/usr/bin/env bash
if [ "$2" = poller.runner ]; then
  shift 2
  echo "$*" >> "$RUNNER_LOG"
  printf '[\\n  {\\n    "guid": "hillside-2026-11-01",\\n    "title": "Rooted"\\n  }\\n]\\n' > data/hillside.json
fi
"""


class PollCheckout:
    """A poll run's checkout of `main`, plus a second clone for whoever pushes ahead of it."""

    def __init__(self, root: Path):
        self.origin = root / "origin.git"
        self.runner = root / "runner"
        self.other = root / "other"
        self.bin = root / "bin"
        self.runner_log = root / "runner.log"
        seed = root / "seed"
        subprocess.run(
            ["git", "init", "-q", "--bare", "-b", "main", str(self.origin)], check=True, env=GIT_ENV
        )
        subprocess.run(["git", "clone", "-q", str(self.origin), str(seed)], check=True, env=GIT_ENV)
        (seed / "data").mkdir()
        (seed / "data" / "pbc.json").write_text(PBC_LEDGER)
        (seed / "data" / "hillside.json").write_text("[]\n")
        git(seed, "add", "data")
        git(seed, "commit", "-q", "-m", "seed ledger")
        git(seed, "push", "-q", "origin", "main")
        for clone in (self.runner, self.other):
            subprocess.run(["git", "clone", "-q", str(self.origin), str(clone)], check=True, env=GIT_ENV)
        self.bin.mkdir()
        (self.bin / "python").write_text(FAKE_PYTHON)
        (self.bin / "python").chmod(0o755)

    def land_first(self, path: str, content: str) -> None:
        (self.other / path).write_text(content)
        git(self.other, "commit", "-q", "-am", "chore(data): record transcription progress [skip ci]")
        git(self.other, "push", "-q", "origin", "main")

    def run_commit_step(
        self, *, backfill: str = "false", church: str = ""
    ) -> subprocess.CompletedProcess[str]:
        env = {
            **GIT_ENV,
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "RUNNER_LOG": str(self.runner_log),
            "GITHUB_REF_NAME": "main",
            "BACKFILL": backfill,
            "CHURCH": church,
        }
        return subprocess.run(
            ["bash", "-eo", "pipefail", "-c", commit_step_script()],
            cwd=self.runner,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def main_file(self, path: str) -> str:
        return git(self.origin, "show", f"main:{path}")

    def reruns(self) -> list[str]:
        return self.runner_log.read_text().splitlines() if self.runner_log.exists() else []


def test_poll_commit_step_rebases_onto_a_ledger_push_that_landed_first(tmp_path):
    """A transcribe shard landing between checkout and push must not drop this run's discoveries."""
    checkout = PollCheckout(tmp_path)
    (checkout.runner / "data" / "hillside.json").write_text(HILLSIDE_DISCOVERY)
    transcribed = PBC_LEDGER.replace('"pending"', '"done"')
    checkout.land_first("data/pbc.json", transcribed)

    result = checkout.run_commit_step()

    assert result.returncode == 0, result.stdout + result.stderr
    assert checkout.main_file("data/pbc.json") == transcribed
    assert '"hillside-2026-11-01"' in checkout.main_file("data/hillside.json")
    # A clean rebase keeps this run's own commit — no second poll, so no second email.
    assert checkout.reruns() == []


def test_poll_commit_step_repolls_against_the_fresh_tip_when_the_rebase_conflicts(tmp_path):
    checkout = PollCheckout(tmp_path)
    # The feed retitled the sermon on the line next to the one the shard marked done.
    (checkout.runner / "data" / "pbc.json").write_text(
        PBC_LEDGER.replace('"Abide in Me"', '"Abide in Me (Part 1)"')
    )
    transcribed = PBC_LEDGER.replace('"pending"', '"done"')
    checkout.land_first("data/pbc.json", transcribed)

    result = checkout.run_commit_step()

    assert result.returncode == 0, result.stdout + result.stderr
    # The other writer's marks survive; this run's stale edit is redone by the re-poll, not merged.
    assert checkout.main_file("data/pbc.json") == transcribed
    assert '"hillside-2026-11-01"' in checkout.main_file("data/hillside.json")
    assert checkout.reruns() == [""]


def test_poll_commit_step_repoll_keeps_the_backfill_and_church_selection(tmp_path):
    # A backfill re-poll that dropped --backfill would email a church's whole back-catalog.
    checkout = PollCheckout(tmp_path)
    (checkout.runner / "data" / "pbc.json").write_text(PBC_LEDGER.replace('"pending"', '"failed"'))
    checkout.land_first("data/pbc.json", PBC_LEDGER.replace('"pending"', '"done"'))

    result = checkout.run_commit_step(backfill="true", church="pbc")

    assert result.returncode == 0, result.stdout + result.stderr
    assert checkout.reruns() == ["--backfill --church pbc"]


def test_poll_commit_step_fails_when_every_push_attempt_is_rejected(tmp_path):
    checkout = PollCheckout(tmp_path)
    (checkout.runner / "data" / "hillside.json").write_text(HILLSIDE_DISCOVERY)
    hook = checkout.origin / "hooks" / "pre-receive"
    hook.write_text("#!/usr/bin/env bash\necho 'ruleset: push declined' >&2\nexit 1\n")
    hook.chmod(0o755)

    result = checkout.run_commit_step()

    assert result.returncode != 0
    assert "failed to push ledger update after 3 attempts" in result.stdout
    assert checkout.main_file("data/hillside.json") == "[]\n"


def test_poll_commit_step_pushes_to_the_branch_it_checked_out():
    script = commit_step_script()
    assert 'git push origin "HEAD:$GITHUB_REF_NAME"' in script
    assert not re.search(r"git push\s*$", script, re.MULTILINE)
