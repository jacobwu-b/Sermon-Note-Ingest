"""The Anthropic boundary — batch-completion status checks only.

Ported from Sermon-Note-Pipeline's own ``llm_client.py`` (its ADR-0004/ADR-0019):
the same ``BatchStatus`` shape and transient/permanent error classes, so a
batch's status means the same thing whichever repo checks it. Only the
status-check subset is needed here — this repo never submits a batch or
retrieves its results, only polls whether one has ended (docs/decisions/0010).

Authentication mirrors Pipeline's own Workload Identity Federation support
(ADR-0087): an explicit ``ANTHROPIC_API_KEY`` wins if set (local development's
only path), otherwise ``ANTHROPIC_FEDERATION_RULE_ID`` exchanges a GitHub Actions
OIDC token for a short-lived Anthropic access token bound to a service account.
See :func:`_fetch_github_oidc_token` for why the token is fetched fresh on every
exchange rather than read once from a file.

The Anthropic SDK is imported lazily inside :func:`_default_poll_batch` so unit
tests never construct a client or need a key; tests inject ``poll_fn``.
"""

from __future__ import annotations

import functools
import json
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from poller import config
from poller.net import DEFAULT_USER_AGENT

if TYPE_CHECKING:
    from anthropic import Anthropic

_client: Anthropic | None = None
_OIDC_TIMEOUT = 10


def _fetch_github_oidc_token(request_url: str, request_token: str) -> str:
    """Fetch one fresh GitHub Actions OIDC token for the Anthropic WIF exchange (ADR-0087).

    Called on every credential exchange rather than read once from a file written
    early in the job, mirroring Sermon-Note-Pipeline's own ``llm_client.py``: a
    GitHub-issued JWT expires ~5 minutes after issuance, but
    ``ACTIONS_ID_TOKEN_REQUEST_URL`` stays valid for the whole job.
    """
    separator = "&" if urllib.parse.urlsplit(request_url).query else "?"
    request = urllib.request.Request(
        request_url + separator + urllib.parse.urlencode({"audience": "https://api.anthropic.com"}),
        headers={"Authorization": f"Bearer {request_token}", "User-Agent": DEFAULT_USER_AGENT},
    )
    with urllib.request.urlopen(request, timeout=_OIDC_TIMEOUT) as response:
        return str(json.loads(response.read())["value"])


class LLMError(RuntimeError):
    """Base class for a mapped Anthropic API error."""


class TransientLLMError(LLMError):
    """A retryable Anthropic error (timeout, rate limit, connection, 5xx)."""


class PermanentLLMError(LLMError):
    """A non-retryable Anthropic rejection (any other 4xx, or an unusable response)."""


@dataclass(frozen=True)
class BatchStatus:
    """A submitted batch's processing state as of one poll."""

    id: str
    ended: bool
    processing_status: str


def _get_client() -> Anthropic:
    """Build (once per process) and return the Anthropic client, keyed from config.

    An explicit ``api_key`` wins if set; otherwise ``federation_rule_id`` selects
    Workload Identity Federation (ADR-0087) — the client exchanges a GitHub Actions
    OIDC token (fetched fresh per call, see :func:`_fetch_github_oidc_token`) for a
    short-lived Anthropic access token bound to the configured service account.
    """
    global _client
    if _client is None:
        import anthropic

        cfg = config.load_anthropic_config()
        if cfg.api_key:
            _client = anthropic.Anthropic(api_key=cfg.api_key)
        else:
            # load_anthropic_config() already guarantees all four are set whenever
            # api_key is not — the only other case that clears its own ConfigError.
            assert cfg.federation_rule_id is not None
            assert cfg.organization_id is not None
            assert cfg.oidc_request_url is not None
            assert cfg.oidc_request_token is not None
            _client = anthropic.Anthropic(
                credentials=anthropic.WorkloadIdentityCredentials(
                    identity_token_provider=functools.partial(
                        _fetch_github_oidc_token, cfg.oidc_request_url, cfg.oidc_request_token
                    ),
                    federation_rule_id=cfg.federation_rule_id,
                    organization_id=cfg.organization_id,
                    service_account_id=cfg.service_account_id,
                    workspace_id=cfg.workspace_id,
                )
            )
    return _client


def _default_poll_batch(batch_id: str) -> BatchStatus:
    """Check one batch's processing status, mapping every SDK error to an :class:`LLMError`.

    Every one, so a single bad batch (say, one Anthropic has since deleted) is a
    logged skip in ``batch_poll``'s loop rather than an exception that ends it.
    """
    import anthropic

    try:
        batch = _get_client().messages.batches.retrieve(batch_id)
    except (anthropic.APIConnectionError, anthropic.RateLimitError) as exc:
        # APIConnectionError covers timeouts too, and is what the SDK raises when
        # the OIDC fetch feeding the WIF exchange (_fetch_github_oidc_token) fails.
        raise TransientLLMError(str(exc)) from exc
    except anthropic.APIStatusError as exc:
        if exc.status_code >= 500:
            raise TransientLLMError(str(exc)) from exc
        raise PermanentLLMError(str(exc)) from exc
    except anthropic.AnthropicError as exc:
        raise PermanentLLMError(str(exc)) from exc
    return BatchStatus(
        id=batch.id,
        ended=batch.processing_status == "ended",
        processing_status=batch.processing_status,
    )


def poll_batch(
    batch_id: str,
    *,
    poll_fn: Callable[[str], BatchStatus] = _default_poll_batch,
) -> BatchStatus:
    """Check whether a submitted batch has finished processing."""
    return poll_fn(batch_id)
