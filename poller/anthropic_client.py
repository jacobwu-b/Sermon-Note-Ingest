"""The Anthropic boundary — batch-completion status checks only.

Ported from Sermon-Note-Pipeline's own ``llm_client.py`` (its ADR-0004/ADR-0019):
the same ``BatchStatus`` shape and transient/permanent error mapping, so a
batch's status means the same thing whichever repo checks it. Only the
status-check subset is needed here — this repo never submits a batch or
retrieves its results, only polls whether one has ended (docs/decisions/0010).
No Workload Identity Federation support (Pipeline's ADR-0087) — ``ANTHROPIC_API_KEY``
only, since this boundary never spends (no completions calls).

The Anthropic SDK is imported lazily inside :func:`_default_poll_batch` so unit
tests never construct a client or need a key; tests inject ``poll_fn``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from poller import config

if TYPE_CHECKING:
    from anthropic import Anthropic

_client: Anthropic | None = None


class LLMError(RuntimeError):
    """Base class for a mapped Anthropic API error."""


class TransientLLMError(LLMError):
    """A retryable Anthropic error (timeout, rate limit, connection)."""


class PermanentLLMError(LLMError):
    """A non-retryable, per-request Anthropic rejection (400/422)."""


@dataclass(frozen=True)
class BatchStatus:
    """A submitted batch's processing state as of one poll."""

    id: str
    ended: bool
    processing_status: str


def _get_client() -> Anthropic:
    """Build (once per process) and return the Anthropic client, keyed from config."""
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic(api_key=config.load_anthropic_config().api_key)
    return _client


def _default_poll_batch(batch_id: str) -> BatchStatus:
    """Check one batch's processing status, mapping transient SDK errors."""
    import anthropic

    try:
        batch = _get_client().messages.batches.retrieve(batch_id)  # type: ignore[attr-defined]
    except (anthropic.APITimeoutError, anthropic.RateLimitError, anthropic.APIConnectionError) as exc:
        raise TransientLLMError(str(exc)) from exc
    except (anthropic.BadRequestError, anthropic.UnprocessableEntityError) as exc:
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
