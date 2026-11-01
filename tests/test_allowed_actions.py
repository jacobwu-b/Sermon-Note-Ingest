"""Contract test: the Actions allowlist and the workflows' `uses:` refs agree.

`.github/allowed-actions.json` is the body applied to the repo's
`actions/permissions/selected-actions` endpoint (.github/repo-settings.md →
Actions). GitHub refuses at dispatch any action the allowlist does not cover, so
an action added to a workflow without an entry would first fail on whichever
unattended run reached it. Like `sha_pinning_required`, the live setting needs
Administration read, so it is verified by hand, not here; this test is the
review-time half.
"""

import json
import re
from fnmatch import fnmatchcase
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
ALLOWLIST = json.loads((REPO_ROOT / ".github" / "allowed-actions.json").read_text())

USES_RE = re.compile(r"^\s*-?\s*uses:\s*(\S+@\S+)", re.MULTILINE)
GITHUB_OWNED_ORGS = ("actions/", "github/")


def _workflow_refs() -> set[str]:
    return {
        match.group(1)
        for workflow in WORKFLOWS_DIR.glob("*.yml")
        for match in USES_RE.finditer(workflow.read_text())
    }


def _third_party_refs() -> set[str]:
    return {ref for ref in _workflow_refs() if not ref.startswith(GITHUB_OWNED_ORGS)}


def test_allowlist_lets_github_owned_actions_run() -> None:
    assert ALLOWLIST["github_owned_allowed"] is True


def test_allowlist_does_not_admit_every_verified_creator() -> None:
    assert ALLOWLIST["verified_allowed"] is False


def test_every_third_party_workflow_action_is_on_the_allowlist() -> None:
    uncovered = sorted(
        ref
        for ref in _third_party_refs()
        if not any(fnmatchcase(ref, pattern) for pattern in ALLOWLIST["patterns_allowed"])
    )
    assert not uncovered, f"actions GitHub would refuse at dispatch: {uncovered}"


def test_every_allowlist_pattern_is_used_by_a_workflow() -> None:
    refs = _third_party_refs()
    unused = [
        pattern
        for pattern in ALLOWLIST["patterns_allowed"]
        if not any(fnmatchcase(ref, pattern) for ref in refs)
    ]
    assert not unused, f"allowlist entries no workflow uses: {unused}"
