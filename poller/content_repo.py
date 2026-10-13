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

Re-transcription (ADR-0022) is the one path that replaces a transcript, and only by
compare-and-swap (:func:`replace_transcript`): Content must still hold the text the
ledger names. Its commits carry ``Ingest-*`` trailers, read back by
:func:`last_commit_trailers`, so a run that died between the Content and ledger pushes
can be recognised as ours. A refused replacement is kept outside ``transcripts/`` by
:func:`keep_candidate`.
"""

from __future__ import annotations

import random
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path

from poller import config
from poller.transcribe import transcript_hash

_PUSH_ATTEMPTS = 3
_PUSH_BACKOFF_BASE = 1.0
# Pipeline publishes to Content every few minutes, so a re-transcription push losing the
# race is expected; five jittered attempts outlast it without lockstep retries.
_REPLACE_ATTEMPTS = 5
_GIT_TIMEOUT = 120
_COMMIT_AUTHOR_NAME = "github-actions[bot]"
_COMMIT_AUTHOR_EMAIL = "github-actions[bot]@users.noreply.github.com"
_COMMIT_MESSAGE = "chore(transcripts): add newly-transcribed sermons"
_STATUS_COMMIT_MESSAGE = "chore(status): publish ingest status"
_REPLACE_COMMIT_MESSAGE = "chore(transcripts): replace a stale-profile transcript"
_CANDIDATE_COMMIT_MESSAGE = "chore(transcripts): keep a refused re-transcription candidate"
_TRANSCRIPTS_DIR = "transcripts"


class ContentPublishError(RuntimeError):
    """Raised when the push to Sermon-Note-Content fails after all attempts."""


class ContentConflictError(ContentPublishError):
    """Content no longer holds the text a replacement expected; never retried (ADR-0022)."""


def _remote_url(repo: str, token: str) -> str:
    """The authenticated HTTPS remote for ``repo`` (``owner/repo``); never logged."""
    return f"https://x-access-token:{token}@github.com/{repo}.git"


def _run_git(args: list[str], *, cwd: Path, token: str) -> str:
    """Run one git command and return its stdout, redacting ``token`` from any failure.

    The token appears only in the clone remote URL, which git may echo into an error.
    The message is built from the captured output with the token masked, and the
    original exception is dropped (``from None``) so the token cannot re-leak through
    the chained ``__cause__``.
    """
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, timeout=_GIT_TIMEOUT
        )
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raw = getattr(exc, "stderr", None) or getattr(exc, "stdout", None) or str(exc)
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", "replace")
        detail = raw.strip().replace(token, "***")
        raise ContentPublishError(f"git {args[0]} failed: {detail}") from None
    return result.stdout


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

    Refuses (raises :class:`ContentPublishError`) to overwrite a path that already
    exists in Content with different text, rather than silently replacing it — Whisper
    is not bit-for-bit deterministic, so a second transcription of an already-"done"
    sermon must never quietly become a second, different version of what a downstream
    consumer (Pipeline) may already have fetched and hashed (issue #60). A rewrite with
    identical text is unaffected: it diffs to nothing and stays a no-op push.
    """
    _push(files, overwrite=False, message=_COMMIT_MESSAGE)


def default_push_status(files: dict[str, str]) -> None:
    """Like :func:`default_push`, but for the status file, which is meant to change.

    No overwrite guard: ``status/ingest.json`` is regenerated whole from the ledger
    each run (spec 0009), so replacing it is the point. Still additive — nothing but
    ``files`` is written — and still a no-op push when the text is unchanged.
    """
    _push(files, overwrite=True, message=_STATUS_COMMIT_MESSAGE)


def _push(files: dict[str, str], *, overwrite: bool, message: str) -> None:
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
            if not overwrite and dest.exists() and dest.read_text(encoding="utf-8") != text:
                raise ContentPublishError(
                    f"refusing to overwrite existing Content file {rel_path!r} with different "
                    "text — a sermon must be represented in Content exactly once"
                )
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
                message,
            ],
            cwd=checkout,
            token=cfg.token,
        )
        _run_git(["push", "origin", cfg.branch], cwd=checkout, token=cfg.token)


def _clone(workdir: str, cfg: config.ContentRepoConfig, *extra: str) -> Path:
    checkout = Path(workdir) / "content"
    _run_git(
        ["clone", *extra, "--branch", cfg.branch, _remote_url(cfg.repo, cfg.token), str(checkout)],
        cwd=Path(workdir),
        token=cfg.token,
    )
    return checkout


def _commit_message(subject: str, trailers: Mapping[str, str]) -> str:
    """``subject`` with ``trailers`` as a final ``Key: value`` paragraph git parses as trailers."""
    if not trailers:
        return subject
    return subject + "\n\n" + "\n".join(f"{key}: {value}" for key, value in trailers.items())


def _commit_and_push(checkout: Path, cfg: config.ContentRepoConfig, message: str) -> None:
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
            message,
        ],
        cwd=checkout,
        token=cfg.token,
    )
    _run_git(["push", "origin", cfg.branch], cwd=checkout, token=cfg.token)


def default_replace(path: str, text: str, *, expect_hash: str, trailers: Mapping[str, str]) -> None:
    """Replace ``path`` with ``text`` only if Content's copy hashes to ``expect_hash``.

    The check runs inside the clone the push is made from. If Content already holds
    ``text`` (an earlier attempt landed but reported failure), nothing is pushed. A missing
    file or any other text raises :class:`ContentConflictError`.
    """
    cfg = config.load_content_repo_config()
    with tempfile.TemporaryDirectory() as workdir:
        checkout = _clone(workdir, cfg, "--depth", "1")
        dest = checkout / path
        current = dest.read_text(encoding="utf-8") if dest.exists() else None
        if current == text:
            return
        if current is None or transcript_hash(current) != expect_hash:
            raise ContentConflictError(
                f"Content's {path!r} is {'missing' if current is None else 'not the text the ledger names'}; "
                "refusing to replace it"
            )
        dest.write_text(text, encoding="utf-8")
        _commit_and_push(checkout, cfg, _commit_message(_REPLACE_COMMIT_MESSAGE, trailers))


def default_last_commit_trailers(path: str) -> dict[str, str]:
    """The ``Key: value`` trailers of the last Content commit that touched ``path``.

    A blobless clone: the history is needed, the file contents are not.
    """
    cfg = config.load_content_repo_config()
    with tempfile.TemporaryDirectory() as workdir:
        checkout = _clone(workdir, cfg, "--filter=blob:none", "--no-checkout")
        out = _run_git(
            ["log", "-1", "--format=%(trailers:only,unfold)", "--", path], cwd=checkout, token=cfg.token
        )
    trailers = {}
    for line in out.splitlines():
        key, sep, value = line.partition(": ")
        if sep:
            trailers[key] = value
    return trailers


def default_keep_candidate(path: str, text: str, *, trailers: Mapping[str, str]) -> None:
    """Add ``text`` at ``path``; never overwrite a different file already there.

    The same text already at ``path`` is a no-op, so a retried keep is idempotent.
    """
    cfg = config.load_content_repo_config()
    with tempfile.TemporaryDirectory() as workdir:
        checkout = _clone(workdir, cfg, "--depth", "1")
        dest = checkout / path
        if dest.exists():
            if dest.read_text(encoding="utf-8") == text:
                return
            raise ContentPublishError(f"refusing to overwrite kept candidate {path!r} with different text")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        _commit_and_push(checkout, cfg, _commit_message(_CANDIDATE_COMMIT_MESSAGE, trailers))


def default_fetch_all_transcripts() -> dict[str, str]:
    """Every file under Content's ``transcripts/``, keyed by repo-relative path. Read-only."""
    cfg = config.load_content_repo_config()
    with tempfile.TemporaryDirectory() as workdir:
        checkout = _clone(workdir, cfg, "--depth", "1")
        root = checkout / _TRANSCRIPTS_DIR
        if not root.exists():
            return {}
        return {
            file.relative_to(checkout).as_posix(): file.read_text(encoding="utf-8")
            for file in sorted(root.rglob("*"))
            if file.is_file()
        }


def default_fetch_existing(paths: list[str]) -> dict[str, str]:
    """Clone Sermon-Note-Content and return the text of each of ``paths`` it already holds.

    Read-only: nothing is committed or pushed. A path absent from Content is simply
    absent from the result.
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
        return {
            rel_path: (checkout / rel_path).read_text(encoding="utf-8")
            for rel_path in paths
            if (checkout / rel_path).exists()
        }


def _with_retries[T](
    action: Callable[[], T],
    *,
    what: str,
    sleep: Callable[[float], None],
    attempts: int,
    backoff_base: float,
    give_up_on: tuple[type[Exception], ...] = (),
) -> T:
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except give_up_on:
            raise
        except Exception as exc:  # noqa: BLE001 — any Content failure retries then escalates.
            last_error = exc
            if attempt < attempts:
                sleep(backoff_base * 2 ** (attempt - 1))
    raise ContentPublishError(
        f"content {what} failed after {attempts} attempts: {last_error}"
    ) from last_error


def existing_transcripts(
    paths: list[str],
    *,
    fetch: Callable[[list[str]], dict[str, str]] = default_fetch_existing,
    sleep: Callable[[float], None] = time.sleep,
    attempts: int = _PUSH_ATTEMPTS,
    backoff_base: float = _PUSH_BACKOFF_BASE,
) -> dict[str, str]:
    """Which of ``paths`` Sermon-Note-Content already holds, mapped to their text.

    An empty ``paths`` is a no-op that never touches the network. Exhausting all
    attempts raises :class:`ContentPublishError`.
    """
    if not paths:
        return {}
    return _with_retries(
        lambda: fetch(paths), what="read", sleep=sleep, attempts=attempts, backoff_base=backoff_base
    )


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
    _with_retries(lambda: push(files), what="push", sleep=sleep, attempts=attempts, backoff_base=backoff_base)


def push_status(
    files: dict[str, str],
    *,
    push: Callable[[dict[str, str]], None] = default_push_status,
    sleep: Callable[[float], None] = time.sleep,
    attempts: int = _PUSH_ATTEMPTS,
    backoff_base: float = _PUSH_BACKOFF_BASE,
) -> None:
    """Push the status file, retrying with backoff; each attempt starts from a fresh clone.

    Pipeline pushes to Content too, so a push rejected because Content moved is routine:
    the retry re-clones the new tip and re-applies the file. Exhausting all attempts
    raises :class:`ContentPublishError`.
    """
    _with_retries(
        lambda: push(files), what="status push", sleep=sleep, attempts=attempts, backoff_base=backoff_base
    )


def _jittered(sleep: Callable[[float], None]) -> Callable[[float], None]:
    return lambda seconds: sleep(seconds * random.uniform(0.5, 1.5))


def replace_transcript(
    path: str,
    text: str,
    *,
    expect_hash: str,
    trailers: Mapping[str, str],
    replace: Callable[..., None] = default_replace,
    sleep: Callable[[float], None] = time.sleep,
    attempts: int = _REPLACE_ATTEMPTS,
    backoff_base: float = _PUSH_BACKOFF_BASE,
) -> None:
    """Compare-and-swap ``path`` to ``text`` (ADR-0022 rule 4), retrying a lost push race.

    Each attempt re-clones and checks again. :class:`ContentConflictError` is raised at
    once, never retried; exhausting the attempts raises :class:`ContentPublishError`.
    """
    _with_retries(
        lambda: replace(path, text, expect_hash=expect_hash, trailers=trailers),
        what="replace",
        sleep=_jittered(sleep),
        attempts=attempts,
        backoff_base=backoff_base,
        give_up_on=(ContentConflictError,),
    )


def last_commit_trailers(
    path: str,
    *,
    fetch: Callable[[str], dict[str, str]] = default_last_commit_trailers,
    sleep: Callable[[float], None] = time.sleep,
    attempts: int = _PUSH_ATTEMPTS,
    backoff_base: float = _PUSH_BACKOFF_BASE,
) -> dict[str, str]:
    """The trailers of the last Content commit to ``path``, retrying transient failures."""
    return _with_retries(
        lambda: fetch(path), what="history read", sleep=sleep, attempts=attempts, backoff_base=backoff_base
    )


def keep_candidate(
    path: str,
    text: str,
    *,
    trailers: Mapping[str, str],
    keep: Callable[..., None] = default_keep_candidate,
    sleep: Callable[[float], None] = time.sleep,
    attempts: int = _REPLACE_ATTEMPTS,
    backoff_base: float = _PUSH_BACKOFF_BASE,
) -> None:
    """Keep a refused re-transcription in Content (ADR-0022 rule 6), retrying a lost push race."""
    _with_retries(
        lambda: keep(path, text, trailers=trailers),
        what="candidate push",
        sleep=_jittered(sleep),
        attempts=attempts,
        backoff_base=backoff_base,
    )


def all_transcripts(
    *,
    fetch: Callable[[], dict[str, str]] = default_fetch_all_transcripts,
    sleep: Callable[[float], None] = time.sleep,
    attempts: int = _PUSH_ATTEMPTS,
    backoff_base: float = _PUSH_BACKOFF_BASE,
) -> dict[str, str]:
    """Every transcript Content holds, keyed by path, retrying transient failures."""
    return _with_retries(fetch, what="read", sleep=sleep, attempts=attempts, backoff_base=backoff_base)
