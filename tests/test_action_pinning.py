"""Contract test: every `uses:` reference is pinned to a commit SHA.

The repo's Actions setting `sha_pinning_required: true` enforces this on GitHub's
side, but a setting is prose to this test suite until something here checks it
(CLAUDE.md §6 "Controls") — and this is the half a green setting can't show: a
workflow edit that reintroduces a tag ref fails at dispatch time, not at review
time, for whichever schedule or push trips it next.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

USES_RE = re.compile(r"^(\s*)uses:\s*(\S+)@(\S+)", re.MULTILINE)
FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def test_every_workflow_action_is_pinned_to_a_commit_sha() -> None:
    offenders = []
    for workflow in sorted(WORKFLOWS_DIR.glob("*.yml")):
        text = workflow.read_text()
        for match in USES_RE.finditer(text):
            ref = match.group(3)
            if not FULL_SHA_RE.match(ref):
                line_no = text[: match.start()].count("\n") + 1
                offenders.append(f"{workflow.name}:{line_no} uses {match.group(2)}@{ref}")
    assert not offenders, f"action refs not pinned to a full commit SHA: {offenders}"
