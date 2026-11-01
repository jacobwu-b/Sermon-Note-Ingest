"""Runs a workflow's push-job step against real git repos and the real ``poller.ledger_push``.

Shared by test_poll_workflow.py and test_transcribe_workflow.py. The step's ``python`` is this
interpreter, invoked as the step invokes it (``-E -S``), so these runs also prove the push path
loads without site-packages (ADR-0027).
"""

import json
import os
import subprocess
import sys
from pathlib import Path

from poller import ledger_push

REPO_ROOT = Path(__file__).resolve().parent.parent

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


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, env=GIT_ENV
    ).stdout


def run_block(step: str) -> str:
    """The `run: |` body of one step's text."""
    lines = step.splitlines()
    start = next(i for i, line in enumerate(lines) if line.strip() == "run: |")
    indent = len(lines[start]) - len(lines[start].lstrip())
    body = []
    for line in lines[start + 1 :]:
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        body.append(line)
    return "\n".join(body)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


class PushJob:
    """An origin holding ``seed`` on main, producers that clone it, and the push job's checkout."""

    def __init__(self, root: Path, seed: dict[str, dict]):
        self.root = root
        self.origin = root / "origin.git"
        self.ledger_dir = root / "ledger"
        self.bin = root / "bin"
        git(root, "init", "-q", "--bare", "-b", "main", str(self.origin))
        # GitHub serves any reachable sha, which the push job's shallow checkout relies on.
        git(self.origin, "config", "uploadpack.allowAnySHA1InWant", "true")
        seeder = self.clone("seeder")
        for path, data in seed.items():
            write_json(seeder / path, data)
        git(seeder, "add", "data")
        git(seeder, "commit", "-q", "-m", "seed ledger")
        git(seeder, "push", "-q", "origin", "main")
        self.bin.mkdir()
        (self.bin / "python").write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
        (self.bin / "python").chmod(0o755)

    def clone(self, name: str, *args: str) -> Path:
        path = self.root / name
        git(self.root, "clone", "-q", *args, f"file://{self.origin}", str(path))
        return path

    def produce(self, name: str, writes: dict[str, dict]) -> Path:
        """A producer job: check out the tip, write ``writes``, and stage them as artifact ``name``."""
        producer = self.clone(f"producer-{name}")
        for path, data in writes.items():
            write_json(producer / path, data)
        ledger_push.stage(producer, self.ledger_dir / name)
        return producer

    def land_first(self, writes: dict[str, dict], message: str = "another writer") -> None:
        other = self.clone(f"other-{len(list(self.root.glob('other-*')))}")
        for path, data in writes.items():
            write_json(other / path, data)
        git(other, "add", "data")
        git(other, "commit", "-q", "-m", message)
        git(other, "push", "-q", "origin", "main")

    def reject_pushes(self, times: int | None = None) -> None:
        """A pre-receive hook standing in for the ruleset: decline ``times`` pushes, or all of them."""
        hook = self.origin / "hooks" / "pre-receive"
        counter = self.root / "rejections"
        limit = "" if times is None else f'[ "$(wc -l < {counter})" -gt {times} ] && exit 0\n'
        hook.write_text(f"#!/bin/sh\necho x >> {counter}\n{limit}echo 'ruleset: push declined' >&2\nexit 1\n")
        hook.chmod(0o755)

    def run(self, script: str, *, expected: str) -> subprocess.CompletedProcess[str]:
        checkout = self.clone("push-checkout", "--depth=1")
        (checkout / "poller").symlink_to(REPO_ROOT / "poller")
        env = {
            **GIT_ENV,
            "PATH": f"{self.bin}{os.pathsep}{os.environ['PATH']}",
            "GITHUB_REF_NAME": "main",
            "LEDGER_DIR": str(self.ledger_dir),
            "EXPECTED": expected,
        }
        return subprocess.run(
            ["bash", "-eo", "pipefail", "-c", script],
            cwd=checkout,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )

    def main_json(self, path: str) -> dict:
        return json.loads(git(self.origin, "show", f"main:{path}"))

    def main_paths(self) -> list[str]:
        return git(self.origin, "ls-tree", "-r", "--name-only", "main").split()

    def commit_count(self) -> int:
        return int(git(self.origin, "rev-list", "--count", "main"))
