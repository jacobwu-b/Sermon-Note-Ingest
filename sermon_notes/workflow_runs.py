"""The workflow-run-history boundary: what did recent pipeline.yml runs conclude? (#283)

Ninth external boundary (CLAUDE.md §6). Distinct from the retired ADR-0031
``actions_runs.py`` (deleted by ADR-0033): that module tried to attribute run history to
a *specific sermon's* attempt count and could not do so reliably across reruns. This
module answers a coarser, stateless question with no attribution problem — "were the
last N *completed* runs of this workflow red?" — so nothing here needs to survive
between calls or agree with a counter kept anywhere else.

Unlike :mod:`attempt_claims`, there is no persisted count to keep in sync: GitHub's own
run history already lives entirely outside the ledger/registry/git failure domain this
switch exists to detect (#235), so each check queries it live rather than maintaining
state of its own.

Credentials come only from :mod:`config`: ``GITHUB_TOKEN`` (needs ``actions: read``)
plus the ``GITHUB_REPOSITORY`` the runner sets. Off Actions, :func:`recent_runs` returns
``None`` without a request (dormant, ADR-0036's fail-loud-on-Actions /
fail-silent-off-Actions pattern). On Actions, a missing ``GITHUB_REPOSITORY``/
``GITHUB_TOKEN`` is a misconfiguration and raises rather than reporting dormant.

The HTTP call lives in :func:`default_fetch` — the boundary the tests replace via
``fetch`` — so no token is read and no network is touched in a unit test.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Callable

from sermon_notes import config
from sermon_notes.logging import get_logger

logger = get_logger()

_LIST_ATTEMPTS = 3
_LIST_BACKOFF_BASE = 1.0
_LIST_TIMEOUT = 30
_API_ROOT = "https://api.github.com"

# GitHub caps `per_page` at 100. The streak window this feeds is always a small
# multiple of a small threshold, so one page is plenty.
_PER_PAGE_CAP = 100


class WorkflowRunsError(RuntimeError):
    """Raised when the run listing cannot be read after the configured attempts."""


@dataclass(frozen=True)
class RunSummary:
    """One completed workflow run — as much as the streak check needs, no more."""

    conclusion: str  # GitHub's vocabulary verbatim: "success", "failure", "cancelled", …
    html_url: str
    created_at: str  # ISO-8601, for the alert's "streak began around" line
    run_number: int


def default_fetch(url: str, token: str) -> bytes:
    """GET ``url`` with the workflow token (the mocked boundary).

    Same shape as :func:`attempt_claims.default_fetch`: the token appears only in the
    request header — never in the raised error or the logs — and both failure paths
    drop the original exception (``from None``) so the URL and headers cannot re-leak
    through the chained :class:`~urllib.error.HTTPError`.
    """
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "sermon-note-pipeline",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=_LIST_TIMEOUT) as response:
            body: bytes = response.read()
            return body
    except urllib.error.HTTPError as exc:
        raise WorkflowRunsError(f"run listing returned HTTP {exc.code}") from None
    except urllib.error.URLError as exc:
        raise WorkflowRunsError(f"run listing request failed: {exc.reason}") from None


def _endpoint(repository: str, workflow_file: str, branch: str, per_page: int) -> str:
    """The list-workflow-runs URL, filtered to completed runs of ``workflow_file``."""
    query = urllib.parse.urlencode({"status": "completed", "branch": branch, "per_page": per_page})
    return f"{_API_ROOT}/repos/{repository}/actions/workflows/{workflow_file}/runs?{query}"


def _runs_from(payload: dict[str, object]) -> tuple[RunSummary, ...]:
    """The ordered run list in ``payload``; raises rather than guessing at its shape.

    A malformed or absent ``workflow_runs`` array must not be read as "zero red
    runs" — that would silently disable the switch behind a green reconcile pass.
    """
    runs = payload.get("workflow_runs")
    if not isinstance(runs, list):
        raise WorkflowRunsError("run listing response had no 'workflow_runs' array")
    summaries: list[RunSummary] = []
    for entry in runs:
        if not isinstance(entry, dict):
            raise WorkflowRunsError("run listing response contained a non-object run entry")
        try:
            summaries.append(
                RunSummary(
                    conclusion=str(entry["conclusion"]),
                    html_url=str(entry["html_url"]),
                    created_at=str(entry["created_at"]),
                    run_number=int(entry["run_number"]),
                )
            )
        except KeyError as exc:
            raise WorkflowRunsError(f"run listing entry missing {exc.args[0]!r}") from None
    return tuple(summaries)


def recent_runs(
    *,
    count: int,
    workflow_file: str,
    branch: str,
    fetch: Callable[[str, str], bytes] = default_fetch,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[RunSummary, ...] | None:
    """The most recent ``count`` COMPLETED runs of ``workflow_file`` on ``branch``.

    Ordered newest first, matching the API's own default ordering. Returns ``None``
    off Actions — the check this feeds is dormant by design there, mirroring
    :func:`attempt_claims.claims_for`'s :data:`~attempt_claims.UNKNOWN`, but as ``None``
    since every caller branches on "was this consulted" before touching the tuple, so
    there is no numeric sentinel to guard against misuse as a run count.

    *On* Actions a missing ``GITHUB_REPOSITORY``/``GITHUB_TOKEN`` is a misconfiguration
    rather than an absent context (ADR-0036), so it raises like an unreadable listing
    instead of silently reporting dormant.

    Per PRD §11.1 a failed request retries up to three times with exponential backoff;
    exhausting them raises :class:`WorkflowRunsError`.
    """
    if config.get("GITHUB_ACTIONS", "") != "true":
        logger.info(
            "run listing not consulted: not running on Actions, so the red-run-streak "
            "switch is dormant this pass"
        )
        return None

    repository = config.get("GITHUB_REPOSITORY", "")
    token = config.get("GITHUB_TOKEN", "")
    missing = [n for n, v in (("GITHUB_REPOSITORY", repository), ("GITHUB_TOKEN", token)) if not v]
    if missing:
        raise WorkflowRunsError(
            f"the run listing needs {' and '.join(missing)} on Actions; "
            "the red-run-streak switch cannot be evaluated without it"
        )

    url = _endpoint(repository, workflow_file, branch, min(count, _PER_PAGE_CAP))
    last_error: Exception | None = None
    for attempt in range(1, _LIST_ATTEMPTS + 1):
        try:
            payload = json.loads(fetch(url, token))
            return _runs_from(payload)[:count]
        except (WorkflowRunsError, ValueError) as exc:
            last_error = exc
            logger.warning("run listing attempt %d/%d failed: %s", attempt, _LIST_ATTEMPTS, exc)
            if attempt < _LIST_ATTEMPTS:
                sleep(_LIST_BACKOFF_BASE * 2 ** (attempt - 1))
    raise WorkflowRunsError(
        f"run listing unavailable after {_LIST_ATTEMPTS} attempts: {last_error}"
    )
