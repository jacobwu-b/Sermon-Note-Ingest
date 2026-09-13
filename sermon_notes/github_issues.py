"""The GitHub Issues boundary: auto-create or group a failure notice (ADR-0080, #533).

Eleventh external boundary (CLAUDE.md §6). Every occurrence that already sends a
best-effort email (:func:`notify.send_escalation` and its alert siblings) also calls
:func:`create_or_group_issue` here, so "read the email, hand-write an issue" (#533)
needs one fewer manual step. This boundary is purely additive: it never delays,
blocks, or suppresses the paired email, and a failure here is logged and swallowed
exactly like a dropped email (PRD §11.2's best-effort posture) — it never raises.

Every issue this module writes carries the fixed :data:`LABEL`, so the search that
decides create-vs-comment can never match the unrelated `dependency-reconciliation`
tracking issue (`dependabot-reconciliation.yml`) or a hand-filed one. The caller's
failure class is the dedup key, embedded verbatim in the issue title
(``"[Sermon Notes] {failure_class}"``); an open, labeled issue with that exact title
created within the last :data:`GROUP_WINDOW_HOURS` gets a new comment, otherwise a new
issue opens. This window is its own constant, not `ALERT_COOLDOWN_HOURS` — that
governs a different property (suppressing a repeat *email* for three specific
run-level conditions), and tuning one must not silently move the other (ADR-0080).

Credentials come only from :mod:`config`: ``GITHUB_TOKEN`` (needs ``issues: write``)
and ``GITHUB_REPOSITORY`` — the same pair :mod:`attempt_claims`/:mod:`workflow_runs`
already read for their own, read-only calls. Off Actions, or either variable unset,
this module is dormant and does nothing: unlike those two modules, a missing issue is
not a safety property that must fail loudly (ADR-0036) — it degrades to "email only,"
today's status quo, which is why this module never raises.

The HTTP calls live in :func:`default_fetch` (GET, list issues) and
:func:`default_post` (POST, create an issue or a comment) — the boundary tests
replace via ``fetch``/``post`` so no token is read and no network is touched.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from typing import Callable

from sermon_notes import config
from sermon_notes.logging import get_logger

logger = get_logger()

_API_ROOT = "https://api.github.com"
_TIMEOUT = 30

# Every issue (and only every issue) this module writes carries this label — the
# scope that keeps its search from ever matching an issue it didn't create.
LABEL = "pipeline-failure"

# How recent an existing issue must be to group into rather than duplicate (#533).
# Deliberately its own constant — see the module docstring on `ALERT_COOLDOWN_HOURS`.
GROUP_WINDOW_HOURS = 24

# A handful of open failure-class issues at once is already anomalous; one page is
# always enough to find a match within the group window.
_PER_PAGE = 20


def _headers(token: str) -> dict[str, str]:
    return {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "sermon-note-pipeline",
        "Content-Type": "application/json",
    }


def default_fetch(url: str, token: str) -> bytes:
    """GET ``url`` with the workflow token (the mocked boundary). Raises on HTTP error."""
    request = urllib.request.Request(url, headers=_headers(token))
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
        body: bytes = response.read()
        return body


def default_post(url: str, token: str, payload: dict[str, object]) -> bytes:
    """POST ``payload`` to ``url`` with the workflow token (the mocked boundary)."""
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=_headers(token),
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
        body: bytes = response.read()
        return body


def issue_title(failure_class: str) -> str:
    """The title every issue/search for ``failure_class`` uses — the dedup key."""
    return f"[Sermon Notes] {failure_class}"


def _find_recent_issue(
    repository: str,
    token: str,
    title: str,
    *,
    fetch: Callable[[str, str], bytes],
    now: datetime,
) -> int | None:
    """The number of the most recent open, :data:`LABEL`-tagged issue titled ``title``
    created within :data:`GROUP_WINDOW_HOURS` — or ``None`` if none matches.
    """
    query = urllib.parse.urlencode(
        {
            "labels": LABEL,
            "state": "open",
            "per_page": _PER_PAGE,
            "sort": "created",
            "direction": "desc",
        }
    )
    url = f"{_API_ROOT}/repos/{repository}/issues?{query}"
    payload = json.loads(fetch(url, token))
    if not isinstance(payload, list):
        return None
    cutoff = now - timedelta(hours=GROUP_WINDOW_HOURS)
    for entry in payload:
        if not isinstance(entry, dict) or entry.get("title") != title:
            continue
        # Defense in depth against the server silently ignoring `?labels=` (the same
        # failure `attempt_claims._count_from` guards against for `?name=`): an entry
        # that reached here without actually carrying the label is not trusted as a
        # match, so a filter failure widens the search rather than mis-grouping into
        # an unrelated issue that merely happens to share this failure class's title.
        labels = entry.get("labels")
        names = (
            {lbl.get("name") for lbl in labels if isinstance(lbl, dict)}
            if isinstance(labels, list)
            else set()
        )
        if LABEL not in names:
            continue
        created_at = entry.get("created_at")
        number = entry.get("number")
        if not isinstance(created_at, str) or not isinstance(number, int):
            continue
        created = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        if created >= cutoff:
            return number
    return None


def create_or_group_issue(
    *,
    failure_class: str,
    body: str,
    fetch: Callable[[str, str], bytes] = default_fetch,
    post: Callable[[str, str, dict[str, object]], bytes] = default_post,
    now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
) -> bool:
    """Create a GitHub issue for ``failure_class``, or comment on a recent match.

    Never raises. Returns ``True`` when the API accepted the create/comment call and
    ``False`` on any drop (off Actions, missing credentials, or a failed request) —
    logged, not escalated further, matching every other notify boundary's best-effort
    posture (ADR-0080). This must never affect the run or the paired email: the two
    are independent, not sequenced on each other's success.
    """
    if config.get("GITHUB_ACTIONS", "") != "true":
        logger.info("issue create/group for %r skipped: not running on Actions", failure_class)
        return False

    repository = config.get("GITHUB_REPOSITORY", "")
    token = config.get("GITHUB_TOKEN", "")
    if not repository or not token:
        logger.warning(
            "issue create/group for %r skipped: GITHUB_REPOSITORY/GITHUB_TOKEN not set",
            failure_class,
        )
        return False

    title = issue_title(failure_class)
    try:
        existing = _find_recent_issue(repository, token, title, fetch=fetch, now=now())
        if existing is not None:
            post(
                f"{_API_ROOT}/repos/{repository}/issues/{existing}/comments", token, {"body": body}
            )
            logger.info("grouped occurrence of %r into issue #%d", failure_class, existing)
        else:
            post(
                f"{_API_ROOT}/repos/{repository}/issues",
                token,
                {"title": title, "body": body, "labels": [LABEL]},
            )
            logger.info("created issue %r", title)
    except Exception as exc:  # noqa: BLE001 — best-effort side channel (ADR-0080).
        logger.error("issue create/group for %r failed: %s", failure_class, exc)
        return False
    return True
