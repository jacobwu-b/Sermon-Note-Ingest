"""Contract tests for ADR-0013: hash-pinned dependency installs.

Every requirement line in the compiled locks must carry a hash, and every CI
install step must require one — `--require-hashes` is the enforcement itself,
not a convention to remember (CLAUDE.md §6 "Controls").
"""

import ast
import re
import sys
from importlib.metadata import packages_distributions
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

LOCK_FILES = ["requirements.txt", "requirements-poll.txt", "requirements-dev.txt"]

REQUIREMENT_LINE_RE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==")
PIP_INSTALL_RE = re.compile(r"pip install\s+(.*-r\s+requirements\S*\.txt)")
INCLUDE_RE = re.compile(r"^-r\s+(\S+)")


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


def _normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _declared_requirements(in_file: str) -> set[str]:
    """Direct requirements of an `.in` file, following its `-r` includes."""
    declared: set[str] = set()
    for line in (REPO_ROOT / in_file).read_text().splitlines():
        if m := INCLUDE_RE.match(line):
            declared |= _declared_requirements(m.group(1))
        elif m := REQUIREMENT_LINE_RE.match(line):
            declared.add(_normalize(m.group(1)))
    return declared


def _locked_versions(lock_file: str) -> dict[str, str]:
    pins = {}
    for line in (REPO_ROOT / lock_file).read_text().splitlines():
        if m := re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)==(\S+)", line):
            pins[_normalize(m.group(1))] = m.group(2)
    return pins


def _top_level_imports(source: Path) -> set[str]:
    """Every module a file imports, at any depth — lazy in-function imports included."""
    names: set[str] = set()
    for node in ast.walk(ast.parse(source.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _module_file(module: str) -> Path | None:
    path = REPO_ROOT.joinpath(*module.split("."))
    for candidate in (path.with_suffix(".py"), path / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def _third_party_imports_reachable_from(entrypoints: list[str]) -> set[str]:
    """Third-party top-level modules imported anywhere in the poller modules these reach."""
    seen: set[str] = set()
    third_party: set[str] = set()
    pending = list(entrypoints)
    while pending:
        module = pending.pop()
        source = _module_file(module)
        if module in seen or source is None:
            continue
        seen.add(module)
        for name in _top_level_imports(source):
            top = name.split(".")[0]
            if top == "poller":
                pending.append(name)
            elif top not in sys.stdlib_module_names:
                third_party.add(top)
    return third_party


def _third_party_imports(package: Path) -> set[str]:
    names = {name.split(".")[0] for source in package.rglob("*.py") for name in _top_level_imports(source)}
    return {n for n in names if n not in sys.stdlib_module_names and n != package.name}


def test_every_third_party_import_in_poller_is_a_declared_direct_dependency() -> None:
    # A module that only arrives transitively breaks the day the package pulling it
    # in drops it, and pip-compile never warns.
    from importlib.metadata import packages_distributions

    declared = _declared_requirements("requirements.in")
    module_to_dists = packages_distributions()
    imports = _third_party_imports(REPO_ROOT / "poller")
    assert imports, "found no third-party imports in poller/ to check"
    undeclared = {
        module: module_to_dists.get(module, [])
        for module in sorted(imports)
        if not {_normalize(d) for d in module_to_dists.get(module, [])} & declared
    }
    assert not undeclared, f"imported by poller/ but not declared in requirements.in: {undeclared}"


POLL_WORKFLOW = WORKFLOWS_DIR / "poll.yml"


def test_poll_workflow_installs_the_poll_lock_not_the_transcription_stack() -> None:
    # The poll runs every few minutes with every notify/dispatch secret in its environment;
    # installing faster-whisper's ~30 packages there buys only PyPI outages and third-party
    # code next to those credentials.
    installs = PIP_INSTALL_RE.findall(POLL_WORKFLOW.read_text())
    assert installs == ["--require-hashes -r requirements-poll.txt"]


def test_poll_workflow_pip_cache_is_keyed_on_the_lock_it_installs() -> None:
    # setup-python's pip cache key hashes requirements.txt by default and names no workflow,
    # so without this the poll and transcribe jobs share one key, and whichever saves first
    # leaves the other restoring a cache that lacks its wheels.
    text = POLL_WORKFLOW.read_text()
    setup = text[text.index("uses: actions/setup-python@") : text.index("- name: Install dependencies")]
    assert "cache-dependency-path: requirements-poll.txt" in setup


def test_every_third_party_import_on_the_poll_path_is_in_the_poll_lock() -> None:
    # Walks lazy imports too (anthropic is one), which a runtime import check would miss.
    entrypoints = sorted(set(re.findall(r"python -m (poller\.\w+)", POLL_WORKFLOW.read_text())))
    assert entrypoints == ["poller.ledger_push", "poller.runner"]
    imports = _third_party_imports_reachable_from(entrypoints)
    assert {"feedparser", "anthropic"} <= imports, f"walk found too little to trust: {imports}"
    locked = set(_locked_versions("requirements-poll.txt"))
    module_to_dists = packages_distributions()
    missing = {
        module: module_to_dists.get(module, [])
        for module in sorted(imports)
        if not {_normalize(d) for d in module_to_dists.get(module, [])} & locked
    }
    assert not missing, f"imported on poll.yml's path but not in requirements-poll.txt: {missing}"


def test_poll_lock_carries_no_transcription_dependency() -> None:
    locked = _locked_versions("requirements-poll.txt")
    assert not {"faster-whisper", "ctranslate2", "onnxruntime", "av", "huggingface-hub"} & set(locked)


def test_poll_lock_pins_agree_with_the_full_lock() -> None:
    # Two locks over overlapping inputs: relocking one without the other would leave the
    # poll and transcription jobs running different versions of the same package.
    poll = _locked_versions("requirements-poll.txt")
    full = _locked_versions("requirements.txt")
    disagree = {
        name: (version, full.get(name)) for name, version in poll.items() if full.get(name) != version
    }
    assert not disagree, f"requirements-poll.txt vs requirements.txt (poll, full): {disagree}"


def test_requirements_in_includes_the_poll_requirements() -> None:
    assert "-r requirements-poll.in" in (REPO_ROOT / "requirements.in").read_text().splitlines()
