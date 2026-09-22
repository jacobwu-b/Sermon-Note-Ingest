"""Contract tests for ADR-0013: hash-pinned dependency installs.

Every requirement line in the compiled locks must carry a hash, and every CI
install step must require one — `--require-hashes` is the enforcement itself,
not a convention to remember (CLAUDE.md §6 "Controls").
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

LOCK_FILES = ["requirements.txt", "requirements-dev.txt"]

REQUIREMENT_LINE_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==")
PIP_INSTALL_RE = re.compile(r"pip install\s+(.*-r\s+requirements\S*\.txt)")


def _requirement_blocks(text: str) -> list[str]:
    """Each top-level requirement line plus its continuation (`\\`) lines."""
    lines = text.splitlines()
    blocks: list[str] = []
    current: list[str] = []
    for line in lines:
        if REQUIREMENT_LINE_RE.match(line):
            if current:
                blocks.append("\n".join(current))
            current = [line]
        elif current and (line.startswith(" ") or line.rstrip().endswith("\\")):
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def test_every_lock_file_requirement_has_a_hash() -> None:
    for name in LOCK_FILES:
        text = (REPO_ROOT / name).read_text()
        blocks = _requirement_blocks(text)
        assert blocks, f"{name}: found no requirement lines to check"
        missing = [b.splitlines()[0] for b in blocks if "--hash=sha256:" not in b]
        assert not missing, f"{name}: requirements with no --hash entry: {missing}"


def test_lock_files_have_no_unpinned_source_references() -> None:
    for name in LOCK_FILES:
        text = (REPO_ROOT / name).read_text()
        for line in text.splitlines():
            stripped = line.strip()
            assert not stripped.startswith("-r "), (
                f"{name}: '-r' reference ({stripped!r}) defeats --require-hashes "
                "unless the referenced file is itself fully hashed"
            )
            assert not stripped.startswith("-e "), f"{name}: editable install ({stripped!r})"


def test_every_workflow_pip_install_requires_hashes() -> None:
    offenders = []
    for workflow in sorted(WORKFLOWS_DIR.glob("*.yml")):
        text = workflow.read_text()
        for match in PIP_INSTALL_RE.finditer(text):
            install_args = match.group(1)
            if "--require-hashes" not in install_args:
                line_no = text[: match.start()].count("\n") + 1
                offenders.append(f"{workflow.name}:{line_no}")
    assert not offenders, f"pip install without --require-hashes: {offenders}"
