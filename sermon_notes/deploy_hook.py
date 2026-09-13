"""Fire the Vercel deploy hook — the site-rebuild trigger boundary.

This is the deploy-hook boundary (CLAUDE.md §6, ADR-0018): the only place the pipeline
triggers a website rebuild. It POSTs the hook URL read only through :mod:`config`,
following PRD §11.1 — up to three attempts with exponential backoff, then it raises
:class:`DeployHookError` so the publish step can escalate (PRD §11.2). The hook URL is
itself a secret capability token, so it is never logged and never placed in a raised
error: a rejection carries the provider status and body, never the URL.

The HTTP call lives in :func:`default_fire` — the boundary the tests replace via
``fire`` — so no URL is read and no network is touched in a unit test.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.request
from typing import Callable

from sermon_notes import config
from sermon_notes.logging import get_logger

logger = get_logger()

_FIRE_ATTEMPTS = 3
_FIRE_BACKOFF_BASE = 1.0
_FIRE_TIMEOUT = 30


class DeployHookError(RuntimeError):
    """Raised when the deploy hook cannot be fired after the configured attempts."""


def default_fire() -> None:
    """POST the configured Vercel deploy hook (the mocked boundary).

    Reads ``VERCEL_DEPLOY_HOOK_URL`` from config and POSTs it with an empty body. The
    URL is a secret capability token, so neither the error nor the logs ever carry it:
    an HTTP rejection raises with the status and response body only, and a transport
    error raises with the reason only. Both drop the original exception (``from None``)
    so the URL cannot re-leak through the chained :class:`~urllib.error.HTTPError`'s
    ``url`` attribute.
    """
    url = config.get("VERCEL_DEPLOY_HOOK_URL")
    request = urllib.request.Request(url, data=b"", method="POST")
    try:
        with urllib.request.urlopen(request, timeout=_FIRE_TIMEOUT) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace").strip()
        raise DeployHookError(f"deploy hook returned HTTP {exc.code}: {detail}") from None
    except urllib.error.URLError as exc:
        raise DeployHookError(f"deploy hook request failed: {exc.reason}") from None


def fire_deploy_hook(
    *,
    fire: Callable[[], None] = default_fire,
    sleep: Callable[[float], None] = time.sleep,
    attempts: int = _FIRE_ATTEMPTS,
    backoff_base: float = _FIRE_BACKOFF_BASE,
) -> None:
    """Fire the deploy hook, retrying transient failures with backoff.

    Per PRD §11.1: up to ``attempts`` tries, backing off ``backoff_base * 2**n`` seconds
    between them; exhausting all attempts raises :class:`DeployHookError` so the publish
    step can escalate (PRD §11.2). No secret is logged on any path — :func:`default_fire`
    keeps the hook URL out of the error this loop logs.
    """
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            fire()
            return
        except Exception as exc:  # noqa: BLE001 — any hook failure retries then escalates.
            last_error = exc
            logger.warning("deploy hook attempt %d/%d failed: %s", attempt, attempts, exc)
            if attempt < attempts:
                sleep(backoff_base * 2 ** (attempt - 1))
    raise DeployHookError(
        f"deploy hook failed after {attempts} attempts: {last_error}"
    ) from last_error
