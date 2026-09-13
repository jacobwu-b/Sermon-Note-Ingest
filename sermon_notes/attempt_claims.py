"""The attempt-claim boundary: counting a sermon's claimed attempts (ADR-0033).

An eighth external boundary (CLAUDE.md §6), replacing the run-history boundary
ADR-0031 introduced. It answers one question: how many times has a shard job *claimed*
an attempt on this sermon?

A claim is a small artifact the shard job uploads **before** it transcribes or calls the
LLM. That ordering is the whole point. #235 spent money, then tried to record the spend,
and the recording failed — so every run rediscovered the same sermons and paid again.
A counter made of our own records inherits that hole: ``runs[]`` reads zero during the
outage because appending to it is the step that isn't happening. A claim is written
first, and to a store that is not on the delta → merge → registry → git path that broke,
so the count still rises when every downstream write fails.

Credentials come only from :mod:`config`: ``GITHUB_TOKEN`` (needs ``actions: read``) plus
the ``GITHUB_REPOSITORY`` the runner sets. Off Actions — ``GITHUB_ACTIONS`` unset —
:func:`claims_for` reports :data:`UNKNOWN` without a request, so local runs and tests never
reach the network. On a runner missing either variable is a fault rather than an absent
context, and raises (ADR-0036).

Per PRD §11.1 the request retries three times with exponential backoff. Exhausting them
raises :class:`AttemptClaimsError`; callers treat that as "unknown" and proceed rather
than halt, since a guard's dependency must not be able to take the pipeline down
(ADR-0031, retained by ADR-0033).

The HTTP call lives in :func:`default_fetch` — the boundary the tests replace via
``fetch`` — so no token is read and no network is touched in a unit test.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Callable

from sermon_notes import config
from sermon_notes.logging import get_logger

logger = get_logger()

_COUNT_ATTEMPTS = 3
_COUNT_BACKOFF_BASE = 1.0
_COUNT_TIMEOUT = 30
_API_ROOT = "https://api.github.com"

# GitHub caps `per_page` at 100. The count is only ever compared against a small cap, so
# one page is plenty — and the page is also what proves the name filter was applied
# (see :func:`claims_for`).
_PER_PAGE = 100

# How much of the guid's digest names the claim. Guids are URLs, which cannot go in an
# artifact name; a digest is filename-safe and fixed-width. 16 hex chars is 64 bits —
# far past collision risk for a ledger that holds hundreds of sermons, and short enough
# to stay readable in the Actions UI.
_NAME_DIGEST_CHARS = 16

# Reported when the claim count cannot be consulted: off Actions, or misconfigured. It is
# not a count, and callers must not compare it against a threshold.
UNKNOWN = -1


class AttemptClaimsError(RuntimeError):
    """Raised when the claim count cannot be read after the configured attempts."""


def claim_name(guid: str) -> str:
    """The artifact name that claims an attempt on ``guid``.

    The single source of truth for the name, shared by the counter here and by the
    workflow — ``plan-shards`` emits this into the fan-out matrix so the shard job's
    upload and this count can never drift apart.
    """
    digest = hashlib.sha256(guid.encode("utf-8")).hexdigest()[:_NAME_DIGEST_CHARS]
    return f"attempt-{digest}"


def default_fetch(url: str, token: str) -> bytes:
    """GET ``url`` with the workflow token (the mocked boundary).

    The token is a credential, so it appears only in the request header — never in the
    raised error or the logs. Both failure paths drop the original exception
    (``from None``) so the URL and headers cannot re-leak through the chained
    :class:`~urllib.error.HTTPError`.
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
        with urllib.request.urlopen(request, timeout=_COUNT_TIMEOUT) as response:
            body: bytes = response.read()
            return body
    except urllib.error.HTTPError as exc:
        raise AttemptClaimsError(f"claim listing returned HTTP {exc.code}") from None
    except urllib.error.URLError as exc:
        raise AttemptClaimsError(f"claim listing request failed: {exc.reason}") from None


def _endpoint(repository: str, name: str) -> str:
    """The list-artifacts URL for the repository, filtered to claims named ``name``."""
    query = urllib.parse.urlencode({"name": name, "per_page": _PER_PAGE})
    return f"{_API_ROOT}/repos/{repository}/actions/artifacts?{query}"


def _count_from(payload: dict[str, object], name: str) -> int:
    """The claim count in ``payload``, verifying the server applied the name filter.

    An endpoint that silently ignored ``?name=`` would answer with the repository's
    *entire* artifact list, whose ``total_count`` runs to thousands — every sermon would
    read as far past its cap and be retired on sight. That failure is worse than not
    counting at all and would look exactly like a legitimate large count, so it is
    checked rather than assumed: any returned artifact under a different name means the
    filter was not applied, and the count is refused.
    """
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, list):
        foreign = [
            entry.get("name")
            for entry in artifacts
            if isinstance(entry, dict) and entry.get("name") != name
        ]
        if foreign:
            raise AttemptClaimsError(
                f"claim listing ignored the name filter (returned {foreign[0]!r} for {name!r})"
            )
    total = payload.get("total_count")
    if isinstance(total, int):
        # The server's own count for the filter, so it stays correct past one page.
        return total
    return len(artifacts) if isinstance(artifacts, list) else 0


def claims_for(
    guid: str,
    *,
    fetch: Callable[[str, str], bytes] = default_fetch,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """How many attempts have been claimed on ``guid``.

    Returns :data:`UNKNOWN` off Actions, where the cap is dormant by design — so a caller
    can tell "no attempts claimed" from "cannot tell", and never treats the latter as zero.

    *On* Actions the two outcomes are separated (ADR-0036): a missing
    ``GITHUB_REPOSITORY``/``GITHUB_TOKEN`` is a misconfiguration rather than an absent
    context, so it raises like an unreadable listing instead of silently reporting
    :data:`UNKNOWN`. That collapse is what let a revoked ``actions: read`` scope disable
    the cap for every sermon of every run behind a green run (#264).

    Per PRD §11.1 a failed request retries up to three times with exponential backoff;
    exhausting them raises :class:`AttemptClaimsError`.
    """
    if config.get("GITHUB_ACTIONS", "") != "true":
        # Dormant, not broken: off a runner there are no artifacts and no API, so the cap
        # has nothing to consult (ADR-0033). INFO rather than DEBUG because LOG_LEVEL
        # defaults to "info" — at DEBUG this path left no line at all, and a guard that
        # did not run must be readable as such rather than inferred from silence (#264).
        logger.info(
            "claim listing not consulted: not running on Actions, so the attempt cap is "
            "dormant this run"
        )
        return UNKNOWN

    repository = config.get("GITHUB_REPOSITORY", "")
    token = config.get("GITHUB_TOKEN", "")
    missing = [n for n, v in (("GITHUB_REPOSITORY", repository), ("GITHUB_TOKEN", token)) if not v]
    if missing:
        # On a runner both are always set unless something is wrong — a revoked
        # `actions: read` scope, a dropped `env:` entry — so this is a fault, not an
        # absent context, and takes the same loud path an unreadable listing does
        # (ADR-0036). Only the *names* are reported; the token's value never is.
        raise AttemptClaimsError(
            f"the claim listing needs {' and '.join(missing)} on Actions; "
            "the attempt cap cannot be enforced without it"
        )

    name = claim_name(guid)
    url = _endpoint(repository, name)
    last_error: Exception | None = None
    for attempt in range(1, _COUNT_ATTEMPTS + 1):
        try:
            payload = json.loads(fetch(url, token))
            return _count_from(payload, name)
        except (AttemptClaimsError, ValueError) as exc:
            last_error = exc
            logger.warning("claim listing attempt %d/%d failed: %s", attempt, _COUNT_ATTEMPTS, exc)
            if attempt < _COUNT_ATTEMPTS:
                sleep(_COUNT_BACKOFF_BASE * 2 ** (attempt - 1))
    raise AttemptClaimsError(
        f"claim listing unavailable after {_COUNT_ATTEMPTS} attempts: {last_error}"
    )
