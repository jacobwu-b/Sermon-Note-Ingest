"""Push the rendered content feed to ``sermon-notes-content`` — the content-repo boundary.

This is the content-repo publish boundary (CLAUDE.md §6, ADR-0018): the only place the
pipeline writes to the external content repository. It mirrors the audio-download
posture (PRD §11.1) — up to three push attempts with exponential backoff, then it
raises :class:`ContentPublishError` so the publish step can escalate (PRD §11.2) rather
than silently dropping the publish. Credentials are read only through :mod:`config`; the
write token rides in the git remote URL, is never logged, and is redacted from any
subprocess error before it surfaces.

The git work lives in :func:`default_push` — the boundary the tests replace via
``push`` — so no token is read and no repository is touched in a unit test.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Callable

from sermon_notes import config
from sermon_notes.logging import get_logger

logger = get_logger()

_PUSH_ATTEMPTS = 3
_PUSH_BACKOFF_BASE = 1.0
_GIT_TIMEOUT = 120
# The feed is bot-authored, so the commit carries the pipeline's bot identity — the same
# author the artifact-commit step uses (pipeline.yml).
_COMMIT_AUTHOR_NAME = "sermon-note-bot"
_COMMIT_AUTHOR_EMAIL = "sermon-note-bot@users.noreply.github.com"
_COMMIT_MESSAGE = "chore(feed): publish content feed"


class ContentPublishError(RuntimeError):
    """Raised when the feed cannot be pushed to the content repo after all attempts."""


def _remote_url(repo: str, token: str) -> str:
    """The authenticated HTTPS remote for ``repo`` (``owner/repo``); never logged."""
    return f"https://x-access-token:{token}@github.com/{repo}.git"


def _run_git(args: list[str], *, cwd: Path, token: str) -> None:
    """Run one git command, redacting ``token`` from any failure before raising.

    The token appears only in the clone remote URL, which git may echo into an error.
    Both branches build their own message from the captured output with the token
    masked and drop the original exception (``from None``) so the token cannot re-leak
    through the chained ``__cause__`` (which carries the raw command).
    """
    try:
        subprocess.run(
            ["git", *args],
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT,
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raw = getattr(exc, "stderr", None) or getattr(exc, "stdout", None) or str(exc)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        detail = raw.strip().replace(token, "***")
        raise ContentPublishError(f"git {args[0]} failed: {detail}") from None


def _sync_tree(feed_dir: Path, checkout: Path) -> None:
    """Make ``checkout`` mirror ``feed_dir`` exactly, preserving only its ``.git``.

    The content repo holds the feed and nothing else, so a stale feed file (e.g. a
    sermon removed upstream) must not survive — the working tree is replaced wholesale
    rather than merged.
    """
    for entry in checkout.iterdir():
        if entry.name == ".git":
            continue
        if entry.is_dir():
            shutil.rmtree(entry)
        else:
            entry.unlink()
    for entry in feed_dir.iterdir():
        dest = checkout / entry.name
        if entry.is_dir():
            shutil.copytree(entry, dest)
        else:
            shutil.copy2(entry, dest)


def _has_staged_changes(checkout: Path) -> bool:
    """True when the index differs from HEAD (``git diff --cached --quiet`` exits non-zero).

    Carries no token (the diff runs against the local checkout), so it does not go
    through :func:`_run_git`'s redaction path.
    """
    result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=checkout,
        capture_output=True,
        text=True,
        timeout=_GIT_TIMEOUT,
    )
    return result.returncode != 0


def default_push(feed_dir: Path) -> None:
    """Clone the content repo, replace its tree with ``feed_dir``, commit, and push.

    Reads ``CONTENT_REPO`` (``owner/repo``), ``CONTENT_REPO_TOKEN``, and
    ``CONTENT_REPO_BRANCH`` (default ``main``) from config — this is the mocked
    boundary. The token rides only in the clone remote URL, is never logged, and is
    redacted from any git error. A tree identical to what is already published is a
    no-op: nothing is committed or pushed, so an unchanged-feed re-run pushes nothing
    (idempotent, matching the deterministic feed render of spec 0014).
    """
    repo = config.get("CONTENT_REPO")
    token = config.get("CONTENT_REPO_TOKEN")
    branch = config.get("CONTENT_REPO_BRANCH", "main")
    remote = _remote_url(repo, token)

    with tempfile.TemporaryDirectory() as workdir:
        checkout = Path(workdir) / "content"
        _run_git(
            ["clone", "--depth", "1", "--branch", branch, remote, str(checkout)],
            cwd=Path(workdir),
            token=token,
        )
        _sync_tree(feed_dir, checkout)
        _run_git(["add", "-A"], cwd=checkout, token=token)
        if not _has_staged_changes(checkout):
            logger.info("content feed unchanged; nothing to push to %s (%s)", repo, branch)
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
            token=token,
        )
        _run_git(["push", "origin", branch], cwd=checkout, token=token)
    logger.info("pushed content feed to %s (%s)", repo, branch)


def publish_feed(
    feed_dir: Path,
    *,
    push: Callable[[Path], None] = default_push,
    sleep: Callable[[float], None] = time.sleep,
    attempts: int = _PUSH_ATTEMPTS,
    backoff_base: float = _PUSH_BACKOFF_BASE,
) -> None:
    """Push the rendered feed at ``feed_dir`` to the content repo, retrying with backoff.

    Per PRD §11.1: up to ``attempts`` tries, backing off ``backoff_base * 2**n`` seconds
    between them. Every failure class retries uniformly (a bad token is escalated the
    same as a transient network drop, per spec 0014's publish AC); exhausting all
    attempts raises :class:`ContentPublishError` so the publish step can escalate
    (PRD §11.2). No secret is logged on any path — :func:`default_push` redacts the
    token before the error reaches this loop.
    """
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            push(feed_dir)
            return
        except Exception as exc:  # noqa: BLE001 — any push failure retries then escalates.
            last_error = exc
            logger.warning("content feed push attempt %d/%d failed: %s", attempt, attempts, exc)
            if attempt < attempts:
                sleep(backoff_base * 2 ** (attempt - 1))
    raise ContentPublishError(
        f"content feed push failed after {attempts} attempts: {last_error}"
    ) from last_error
