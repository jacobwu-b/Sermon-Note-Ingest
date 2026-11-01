"""Contract tests on the repo's own governance config: the controls it declares are run.

Workflows are matched on their text rather than parsed — the same stdlib-only tradeoff
test_poll_workflow.py makes for poll.yml.
"""

import json
import os
import re
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
GITHUB_DIR = REPO_ROOT / ".github"
WORKFLOWS = sorted((GITHUB_DIR / "workflows").glob("*.yml"))
SHELL_TESTS = sorted(GITHUB_DIR.rglob("*.test.sh"))
DRIFT_STEP = "Live ruleset matches default-ruleset.json"


def test_shell_contract_tests_are_discovered():
    assert SHELL_TESTS, "no .github/**/*.test.sh found — the runner below would pass vacuously"


@pytest.mark.parametrize("script", SHELL_TESTS, ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_shell_contract_test_passes(script: Path):
    # The scripts call `python3`; resolve it to this interpreter, which has the dev deps.
    env = {**os.environ, "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}"}
    result = subprocess.run(
        ["sh", str(script)], check=False, capture_output=True, text=True, env=env, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _jobs(text: str) -> dict[str, str]:
    """Each top-level job's key mapped to its block, by indentation."""
    jobs_section = re.search(r"^jobs:\n((?:[ #\n].*\n?)*)", text, re.MULTILINE)
    assert jobs_section
    parts = re.split(r"^  ([\w-]+):[ \t]*\n", jobs_section.group(1), flags=re.MULTILINE)
    return dict(zip(parts[1::2], parts[2::2], strict=True))


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda p: p.name)
def test_every_job_declares_its_own_timeout(workflow: Path):
    # GitHub's default is six hours, billed in full by a job that hangs (CLAUDE.md §13).
    jobs = _jobs(workflow.read_text())
    assert jobs, f"{workflow.name}: no jobs parsed"
    missing = [
        name
        for name, block in jobs.items()
        if not re.search(r"^    timeout-minutes:\s*\S", block, re.MULTILINE)
    ]
    assert not missing, f"{workflow.name}: jobs without timeout-minutes: {missing}"


def test_static_checks_target_the_python_version_workflows_run():
    versions = {v for wf in WORKFLOWS for v in re.findall(r'python-version:\s*"([\d.]+)"', wf.read_text())}
    assert len(versions) == 1, f"workflows disagree on python-version: {versions}"
    (version,) = versions
    config = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
    assert config["tool"]["ruff"]["target-version"] == "py" + version.replace(".", "")
    assert config["tool"]["mypy"]["python_version"] == version


def _step_block(text: str, name: str) -> str:
    """The named step's text, up to the next step at the same indentation."""
    match = re.search(rf"^(\s*)- name: {re.escape(name)}\n", text, re.MULTILINE)
    assert match, f"no step named {name!r}"
    indent = match.group(1)
    rest = text[match.end() :]
    end = re.search(rf"^{indent}- |^\S", rest, re.MULTILINE)
    return match.group(0) + (rest[: end.start()] if end else rest)


def _run_body(step: str) -> str:
    match = re.search(r"^(\s*)run: \|\n((?:\1  .*\n|\s*\n)+)", step, re.MULTILINE)
    assert match, "step has no `run: |` block"
    return textwrap.dedent(match.group(2))


def _drift_step() -> str:
    return _step_block((GITHUB_DIR / "workflows" / "ci.yml").read_text(), DRIFT_STEP)


def test_ruleset_drift_check_runs_on_the_daily_schedule_and_never_on_a_pull_request():
    # Repairing drift takes an admin, not a PR — gating PRs on it would deadlock Dependabot.
    ci = (GITHUB_DIR / "workflows" / "ci.yml").read_text()
    assert re.search(r"^  schedule:\n    - cron:", ci, re.MULTILINE)
    assert "if: always() && github.event_name != 'pull_request'" in _drift_step()


DOCUMENTED_TYPES = [
    rule["type"] for rule in json.loads((GITHUB_DIR / "default-ruleset.json").read_text())["rules"]
]


def _run_drift_check(tmp_path: Path, live_types: list[str]) -> subprocess.CompletedProcess[str]:
    """Run the step's own script against a stand-in `gh` serving `live_types` as main's rules."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (tmp_path / "rules.json").write_text(json.dumps([{"type": t, "ruleset_id": 1} for t in live_types]))
    gh = bin_dir / "gh"
    gh.write_text(f'#!/usr/bin/env bash\necho "$*" >> {tmp_path}/gh.log\ncat {tmp_path}/rules.json\n')
    gh.chmod(0o755)
    script = tmp_path / "step.sh"
    script.write_text(_run_body(_drift_step()))
    env = {
        **os.environ,
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "GH_REPO": "jacobwu-b/Sermon-Note-Ingest",
    }
    return subprocess.run(
        ["bash", "-e", str(script)], check=False, cwd=REPO_ROOT, capture_output=True, text=True, env=env
    )


def test_ruleset_drift_check_passes_when_live_rule_types_match_the_documented_set(tmp_path: Path):
    result = _run_drift_check(tmp_path, list(reversed(DOCUMENTED_TYPES)))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "repos/jacobwu-b/Sermon-Note-Ingest/rules/branches/main" in (tmp_path / "gh.log").read_text()


def test_ruleset_drift_check_fails_on_a_live_rule_the_file_does_not_declare(tmp_path: Path):
    result = _run_drift_check(tmp_path, [*DOCUMENTED_TYPES, "required_signatures"])
    assert result.returncode != 0
    assert "required_signatures" in result.stdout + result.stderr


def test_ruleset_drift_check_fails_when_a_declared_rule_is_missing_live(tmp_path: Path):
    result = _run_drift_check(tmp_path, [t for t in DOCUMENTED_TYPES if t != "code_scanning"])
    assert result.returncode != 0


def test_documented_ruleset_declares_the_codeql_rule_main_enforces():
    assert "code_scanning" in DOCUMENTED_TYPES
