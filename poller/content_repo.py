"""Push transcripts to the private Sermon-Note-Content repo — the Content boundary.

This is the Content-repo push boundary (CLAUDE.md §6): the only place this repo
writes to ``Sermon-Note-Content``. The push is additive, not a mirror — Content
also holds Pipeline's and Web's own output, so a run here may only add or update
the specific paths it produced, never touch anything else. A rewrite identical to
what's already there diffs to nothing and pushes nothing, which is what makes a
guid-derived path idempotent: a sermon rediscovered many times always maps to the
same file and is represented in Content exactly once (ADR-0004).

The git work lives in :func:`default_push` — the boundary the tests replace via
``push`` — so no token is read and no repository is touched in a unit test.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

from poller import config

_PUSH_ATTEMPTS = 3
_PUSH_BACKOFF_BASE = 1.0
_GIT_TIMEOUT = 120
_COMMIT_AUTHOR_NAME = "github-actions[bot]"
_COMMIT_AUTHOR_EMAIL = "github-actions[bot]@users.noreply.github.com"
_COMMIT_MESSAGE = "chore(transcripts): add newly-transcribed sermons"


class ContentPublishError(RuntimeError):
    """Raised when the push to Sermon-Note-Content fails after all attempts."""


def _remote_url(repo: str, token: str) -> str:
    """The authenticated HTTPS remote for ``repo`` (``owner/repo``); never logged."""
    return f"https://x-access-token:{token}@github.com/{repo}.git"


def _run_git(args: list[str], *, cwd: Path, token: str) -> None:
    """Run one git command, redacting ``token`` from any failure before raising.

    The token appears only in the clone remote URL, which git may echo into an error.
    The message is built from the captured output with the token masked, and the
    original exception is dropped (``from None``) so the token cannot re-leak through
    the chained ``__cause__``.
    """
    try:
        subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, timeout=_GIT_TIMEOUT
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raw = getattr(exc, "stderr", None) or getattr(exc, "stdout", None) or str(exc)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        detail = raw.strip().replace(token, "***")
        raise ContentPublishError(f"git {args[0]} failed: {detail}") from None


def _has_staged_changes(checkout: Path) -> bool:
    """True when the index differs from HEAD (``git diff --cached --quiet`` exits non-zero).

    Carries no token (the diff runs against the local checkout), so no redaction is needed.
    """
    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=checkout,
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT,
        check=False,
    )
    return result.returncode != 0


def default_push(files: dict[str, str]) -> None:
    """Clone Sermon-Note-Content, write only ``files``, and commit+push if anything changed.

    ``files`` maps a repo-relative path to the text to write there. Reads
    ``CONTENT_REPO``, ``CONTENT_REPO_TOKEN``, and ``CONTENT_REPO_BRANCH`` from config
    (the mocked boundary). The token rides only in the clone remote URL, is never
    logged, and is redacted from any git error.
    """
    cfg = config.load_content_repo_config()
    remote = _remote_url(cfg.repo, cfg.token)

    with tempfile.TemporaryDirectory() as workdir:
        checkout = Path(workdir) / "content"
        _run_git(
            ["clone", "--depth", "1", "--branch", cfg.branch, remote, str(checkout)],
            cwd=Path(workdir),
            token=cfg.token,
        )
        for rel_path, text in files.items():
            dest = checkout / rel_path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text, encoding="utf-8")
        _run_git(["add", "-A"], cwd=checkout, token=cfg.token)
        if not _has_staged_changes(checkout):
            return
        _run_git(
            [
                "-c",
                f"user.name={_COMMIT_AUTHOR_NAME}",
                "-c",
                f"user.email={_COMMIT_AUTHOR_EMAIL}",
                "commit",
                "-m",
                _COMMIT_MESSAGE,
            ],
            cwd=checkout,
            token=cfg.token,
        )
        _run_git(["push", "origin", cfg.branch], cwd=checkout, token=cfg.token)


def push_transcripts(
    files: dict[str, str],
    *,
    push: Callable[[dict[str, str]], None] = default_push,
    sleep: Callable[[float], None] = time.sleep,
    attempts: int = _PUSH_ATTEMPTS,
    backoff_base: float = _PUSH_BACKOFF_BASE,
) -> None:
    """Push ``files`` to Sermon-Note-Content, retrying transient failures with backoff.

    An empty ``files`` is a no-op that never touches the network. Exhausting all
    attempts raises :class:`ContentPublishError` so the caller can leave every sermon
    transcribed this run pending rather than mark one done that never durably landed.
    """
    if not files:
        return
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            push(files)
            return
        except Exception as exc:  # noqa: BLE001 — any push failure retries then escalates.
            last_error = exc
            if attempt < attempts:
                sleep(backoff_base * 2 ** (attempt - 1))
    raise ContentPublishError(f"content push failed after {attempts} attempts: {last_error}") from last_error
