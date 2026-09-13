"""The Anthropic boundary — the only module that calls the LLM.

This is the LLM external-call boundary (CLAUDE.md §6, ADR-0004): the single place
the pipeline talks to Anthropic. It reads its credentials and model only through
:mod:`sermon_notes.config`, wraps one Messages call behind an injectable
boundary, and follows PRD §11.1 — a timeout or rate-limit retries three times with
exponential backoff, then raises.

The Anthropic SDK is imported lazily inside :func:`_default_call` so unit tests and
tooling never construct a client or need a key; tests inject ``call_fn``. The SDK's
transient exceptions are mapped to :class:`TransientLLMError` and its per-request
rejections (400/422) to :class:`PermanentLLMError` at this boundary, so callers catch
one :class:`LLMError` hierarchy and never depend on the SDK's exception classes
(#126, ADR-0019). Genuinely global SDK errors (auth, permission, server 5xx) are left
raw on purpose, so a run aborts loudly rather than sending every sermon terminal.

:func:`submit_batch`, :func:`poll_batch`, and :func:`retrieve_batch` add the Anthropic
Batches API surface for backfill generation (ADR-0061, spec 0020), following the same
lazy-import/injectable-callable/error-mapping shape as :func:`call`. Unlike ``call()``,
none of the three retries internally — a submission failure risks double-billing on
retry, and polling/retrieval get their retry for free from the next pipeline run that
touches the registry. A per-item batch failure (errored/canceled/expired) is decoded
into :class:`BatchResult.error`, not raised, so one bad item doesn't abort retrieval of
the rest of the batch's results.

:func:`_get_client` authenticates via Workload Identity Federation when
``ANTHROPIC_FEDERATION_RULE_ID`` is configured and no ``ANTHROPIC_API_KEY`` overrides it
(ADR-0087) — see :func:`_fetch_github_oidc_token` for why the GitHub OIDC token is fetched
fresh on every exchange rather than read once from a file.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Callable, Sequence

from sermon_notes import config
from sermon_notes.config import ConfigError
from sermon_notes.logging import get_logger

if TYPE_CHECKING:
    from anthropic import Anthropic

logger = get_logger()

# Default model per PRD §6.1 / decision #14 — latest Sonnet, overridable via config
# (ADR-0022 moved this to Sonnet 5 with adaptive thinking).
DEFAULT_MODEL = "claude-sonnet-5"

# List price per model (USD per 1M tokens: input, output), PRD §12 / decision #14.
# Keyed by the resolved model so the cost ledger tracks whatever ANTHROPIC_MODEL
# selects; pricing lives beside the model it prices so the two never drift (#34).
# Priced at standard list rates. Sonnet 5's rate is $2/$10 (verified against the
# pricing page): ADR-0022 modelled $2/$10 as a temporary introductory rate expiring
# 2026-08-31 and deliberately priced at $3/$15 to stay conservative past it — that
# premise was wrong, $2/$10 is the standard rate, not an intro one (ADR-0084, #541).
# claude-sonnet-4-6 stays at $3/$15 so pre-migration run records remain priced at
# the rate that actually billed them (ADR-0010) — that model's rate is unaffected.
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-opus-4-7": (5.00, 25.00),
    "claude-opus-4-6": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
    "claude-fable-5": (10.00, 50.00),
}

# Adaptive thinking shares this budget on Sonnet 5, so the cap has headroom for
# both the reasoning and the ~3.5k-token three-section note (ADR-0022). Raised from
# 16000 (ADR-0081) after a sermon truncated twice — once mid-JSON, once with thinking
# consuming the entire cap and leaving zero text. The Python SDK's non-streaming guard
# is expected_time = 3600 * max_tokens / 128_000, erroring past a 600s estimate, which
# puts its ceiling at ~21,333 tokens; 20000 stays clear of it so the call stays
# synchronous. This still isn't a structural fix — adaptive thinking has no reserved
# floor for output text, so a sufficiently verbose reasoning pass can still exhaust it.
_MAX_OUTPUT_TOKENS = 20000
_CALL_ATTEMPTS = 3
_CALL_BACKOFF_BASE = 1.0

# Batches API discount off MODEL_PRICING's synchronous list rates (ADR-0061, spec 0020).
_BATCH_DISCOUNT = 0.5

# Built once on first real call; tests inject ``call_fn`` and never reach this.
_client: Anthropic | None = None


@dataclass(frozen=True)
class LLMResponse:
    """One model completion plus the token usage the call reported.

    ``model`` is the resolved model that produced the completion; :func:`call`
    stamps it so the cost ledger can be priced against the model actually billed
    (#34). It defaults to ``""`` for hand-built responses in tests.

    ``stop_reason`` is read verbatim off the SDK response (e.g. ``"end_turn"``,
    ``"max_tokens"``) with no interpretation applied at this boundary, so a
    caller can tell a truncated completion apart from any other malformed-reply
    cause instead of inferring it from a token count that happens to match the
    cap (#368).

    ``billing`` records which rate actually applied — ``"sync"`` for a plain
    :func:`call`, ``"batch"`` for a completion resolved through the Batches API
    (ADR-0061). A batch result reuses this same dataclass rather than a parallel
    one so downstream generation code needs no branching on how a completion
    arrived; :func:`cost_usd` reads this field to pick the right rate.
    """

    text: str
    input_tokens: int
    output_tokens: int
    model: str = ""
    stop_reason: str | None = None
    billing: str = "sync"


class LLMError(RuntimeError):
    """Base class for LLM boundary failures."""


class TransientLLMError(LLMError):
    """A retryable failure — timeout, rate limit, or connection error (PRD §11.1)."""


class RetriesExhaustedError(LLMError):
    """:func:`call` exhausted its retry budget on repeated :class:`TransientLLMError`.

    A client-side timeout or connection error reports no ``usage`` block, so a request
    Anthropic already served and billed can fail here with no way to know its cost — the
    last attempt is indistinguishable, at this boundary, from one that was never billed
    at all (#279). Kept as its own subclass of :class:`LLMError`, rather than the plain
    ``LLMError`` this used to raise, so a caller that records run history can tell "the
    call never returned anything billable" apart from every other failure shape and mark
    the ledger accordingly, instead of the two reading as the same $0.00.
    """


class PermanentLLMError(LLMError):
    """A per-request failure the SDK rejected outright — e.g. a 400 from a transcript that
    overruns the context window (#126).

    Not retryable and not global: it is specific to *this* request, so it surfaces as a
    catchable :class:`LLMError` the generate stage sends terminal for one sermon, instead
    of a raw SDK type escaping to abort the whole batch (ADR-0019). Genuinely global SDK
    errors (auth, permission, server 5xx) are left raw so the run aborts loudly instead.
    """


@dataclass(frozen=True)
class BatchRequest:
    """One sermon's generation request to submit inside a batch (spec 0020).

    ``custom_id`` is sent to the Batches API verbatim and comes back on every
    resolved item, so a caller matches a result back to the sermon that
    requested it without any side table.
    """

    custom_id: str
    system: str
    user: str


def batch_custom_id(guid: str) -> str:
    """Derive a Batches-API-legal ``custom_id`` from a sermon's guid.

    The API requires ``custom_id`` to match ``^[a-zA-Z0-9_-]{1,64}$``, but a guid is
    whatever shape its source feed uses — Menlo/PBC's are URLs, Westgate/North Point's
    carry a colon-delimited source prefix (e.g. ``"north_point:34eF..."``) — so it can't
    be sent verbatim (#454). A sha256 hex digest is always exactly 64 lowercase
    hex characters, which is both legal and length-safe regardless of the guid's shape.
    Callers derive this the same way at submission and at resolution time, so no reverse
    mapping is needed to match a result back to its sermon.
    """
    return hashlib.sha256(guid.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class BatchStatus:
    """A submitted batch's processing state as of one poll (spec 0020's poll step)."""

    id: str
    ended: bool
    processing_status: str


@dataclass(frozen=True)
class BatchResult:
    """One resolved item from a batch: a completion, or a per-item failure.

    Mirrors the Batches API's succeeded/errored/canceled/expired result union so a
    caller sees exactly one of ``response`` or ``error`` set, never both, without
    depending on the SDK's own result types (ADR-0019's boundary-hides-the-SDK
    contract, extended to batch results). A per-item failure is data returned
    alongside the batch's other results, not an exception — it should not abort
    retrieval of the rest of the batch.
    """

    custom_id: str
    response: LLMResponse | None = None
    error: str | None = None


def _fetch_github_oidc_token() -> str:
    """Fetch one fresh GitHub Actions OIDC token for the Anthropic WIF exchange (ADR-0087).

    Called on every credential exchange rather than read once from a file written early
    in the job: a GitHub-issued JWT expires ~5 minutes after issuance, but
    ``ACTIONS_ID_TOKEN_REQUEST_URL`` stays valid for the whole job. The ``shard`` job runs
    up to 90 minutes (transcription, then generation) — a token fetched once upfront would
    already be stale by the time the LLM call fires. Refetching here instead survives that.
    """
    import httpx

    url = config.get("ACTIONS_ID_TOKEN_REQUEST_URL")
    token = config.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN")
    response = httpx.get(
        url,
        params={"audience": "https://api.anthropic.com"},
        headers={"Authorization": f"Bearer {token}"},
        timeout=10.0,
    )
    response.raise_for_status()
    return str(response.json()["value"])


def _get_client() -> Anthropic:
    """Build (once) and return the Anthropic client, keyed from config (ADR-0087).

    Three auth paths, in order: an explicit ``ANTHROPIC_API_KEY`` wins if set — this is the
    rollback lever during the Workload Identity Federation migration, and the only path
    local development uses. Otherwise, an ``ANTHROPIC_FEDERATION_RULE_ID`` selects WIF: the
    client exchanges a GitHub Actions OIDC token (fetched fresh per call, see
    :func:`_fetch_github_oidc_token`) for a short-lived Anthropic access token bound to the
    configured service account. With neither set, the SDK's own zero-argument resolution
    applies (e.g. an ``ANTHROPIC_PROFILE`` or active CLI profile) rather than failing here —
    a clearer error than this boundary could construct is left to the SDK.
    """
    global _client
    if _client is None:
        import anthropic

        api_key = config.get("ANTHROPIC_API_KEY", None)
        federation_rule_id = config.get("ANTHROPIC_FEDERATION_RULE_ID", None)
        if api_key:
            _client = anthropic.Anthropic(api_key=api_key)
        elif federation_rule_id:
            _client = anthropic.Anthropic(
                credentials=anthropic.WorkloadIdentityCredentials(
                    identity_token_provider=_fetch_github_oidc_token,
                    federation_rule_id=federation_rule_id,
                    organization_id=config.get("ANTHROPIC_ORGANIZATION_ID"),
                    service_account_id=config.get("ANTHROPIC_SERVICE_ACCOUNT_ID", None),
                    workspace_id=config.get("ANTHROPIC_WORKSPACE_ID", None),
                )
            )
        else:
            _client = anthropic.Anthropic()
    return _client


def _default_call(
    model: str, system: str, user: str, *, output_config: dict[str, Any] | None = None
) -> LLMResponse:
    """Make one real Messages call, mapping transient SDK errors to retryable ones.

    ``output_config`` (ADR-0083, #542), when given, is forwarded verbatim — the caller
    (``generate.py``) owns the note's JSON Schema and effort level; this boundary only
    passes it through. Omitted entirely (not sent as ``None``) when not given, so a
    caller with no opinion on structured output sees the SDK's own default.
    """
    import anthropic
    import httpx

    kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": _MAX_OUTPUT_TOKENS,
        "thinking": {"type": "adaptive"},
        # No cache_control breakpoint here — deliberately (ADR-0085). Concurrent
        # shards in the fan-out (ADR-0023) all miss and all pay the write premium.
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }
    if output_config is not None:
        kwargs["output_config"] = output_config

    try:
        message = _get_client().messages.create(**kwargs)
    except (
        anthropic.APITimeoutError,
        anthropic.RateLimitError,
        anthropic.APIConnectionError,
        # A hiccup fetching the GitHub OIDC token that feeds the WIF exchange (ADR-0087,
        # _fetch_github_oidc_token) surfaces here, on the same first call that triggers
        # the exchange — mapped transient for the same reason an Anthropic-side timeout
        # is: retrying is the right response, not aborting the sermon.
        httpx.TimeoutException,
        httpx.ConnectError,
    ) as exc:
        raise TransientLLMError(str(exc)) from exc
    except (
        anthropic.BadRequestError,
        anthropic.UnprocessableEntityError,
    ) as exc:
        # 400/422 — the SDK rejected *this* request (e.g. an over-long transcript). A
        # per-content failure: surface it as a catchable LLMError, not a raw SDK type,
        # so one sermon goes terminal without aborting the batch (#126, ADR-0019). Global
        # SDK errors (auth, 5xx) are deliberately left to propagate and abort the run.
        raise PermanentLLMError(str(exc)) from exc

    text = "".join(block.text for block in message.content if block.type == "text")
    return LLMResponse(
        text=text,
        input_tokens=message.usage.input_tokens,
        output_tokens=message.usage.output_tokens,
        stop_reason=message.stop_reason,
    )


def _resolve_configured_model() -> str:
    """Resolve ``ANTHROPIC_MODEL`` from config, allowlisted against :data:`MODEL_PRICING`.

    Only the config-resolved path is guarded — an explicit ``model=`` argument (today,
    only test call sites) is unvalidated. ``ANTHROPIC_MODEL`` is a live repository
    variable (#444, ADR-0064); a typo or unsupported value must be refused before any
    API request, not discovered mid-run after a batch has already started billing.
    """
    resolved = config.get("ANTHROPIC_MODEL", DEFAULT_MODEL)
    if resolved not in MODEL_PRICING:
        raise ConfigError(
            f"ANTHROPIC_MODEL={resolved!r} is not a known model — add it to "
            "MODEL_PRICING in llm_client.py before using it as a live override "
            "(ADR-0064)."
        )
    return resolved


def call(
    system: str,
    user: str,
    *,
    model: str | None = None,
    output_config: dict[str, Any] | None = None,
    call_fn: Callable[..., LLMResponse] = _default_call,
    sleep: Callable[[float], None] = time.sleep,
    attempts: int = _CALL_ATTEMPTS,
    backoff_base: float = _CALL_BACKOFF_BASE,
) -> LLMResponse:
    """Run one prompt through the model, retrying transient failures with backoff.

    The model defaults to ``ANTHROPIC_MODEL`` from config (falling back to
    :data:`DEFAULT_MODEL`, allowlisted against :data:`MODEL_PRICING` — see
    :func:`_resolve_configured_model`) and is stamped onto the returned
    :class:`LLMResponse` so callers can price usage against the model actually
    billed (#34). Per PRD
    §11.1, a :class:`TransientLLMError` (timeout, rate limit, connection) retries
    up to ``attempts`` times, backing off ``backoff_base * 2**n`` seconds between
    tries; exhausting them raises :class:`RetriesExhaustedError`. Non-transient errors
    propagate immediately.

    ``output_config`` (ADR-0083, #542) is forwarded to ``call_fn`` as a keyword
    argument only when given, so a ``call_fn`` double that doesn't know about it
    (every pre-existing test fake) keeps working unchanged — this is an additive
    boundary, not a breaking one.
    """
    resolved_model = model or _resolve_configured_model()
    last_error: TransientLLMError | None = None
    for attempt in range(1, attempts + 1):
        try:
            if output_config is not None:
                response = call_fn(resolved_model, system, user, output_config=output_config)
            else:
                response = call_fn(resolved_model, system, user)
            return replace(response, model=resolved_model)
        except TransientLLMError as exc:
            last_error = exc
            logger.warning("LLM call attempt %d/%d failed: %s", attempt, attempts, exc)
            if attempt < attempts:
                sleep(backoff_base * 2 ** (attempt - 1))
    raise RetriesExhaustedError(
        f"LLM call failed after {attempts} attempts: {last_error}"
    ) from last_error


def _default_submit_batch(
    requests: Sequence[BatchRequest], model: str, *, output_config: dict[str, Any] | None = None
) -> str:
    """Submit one Batches API request per item, mapping transient SDK errors.

    ``output_config`` (ADR-0083, #542), when given, is applied to every item's
    ``params`` identically — the same schema and effort as the synchronous path,
    since the batch and sync paths generate the same note shape (spec 0020).
    """
    import anthropic
    import httpx

    def _params(r: BatchRequest) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": model,
            "max_tokens": _MAX_OUTPUT_TOKENS,
            "thinking": {"type": "adaptive"},
            "system": r.system,
            "messages": [{"role": "user", "content": r.user}],
        }
        if output_config is not None:
            params["output_config"] = output_config
        return params

    try:
        # mypy can verify a dict *literal* built inline against the SDK's TypedDict, but
        # not one assembled by a helper — _params's conditional output_config key is what
        # forces it out of literal form. The runtime call accepts a plain dict either way
        # (the TypedDict is a static-only hint), so this is a boundary mismatch, not a
        # real type error.
        requests_payload = [{"custom_id": r.custom_id, "params": _params(r)} for r in requests]
        batch = _get_client().messages.batches.create(
            requests=requests_payload  # type: ignore[arg-type]  # see comment above
        )
    except (
        anthropic.APITimeoutError,
        anthropic.RateLimitError,
        anthropic.APIConnectionError,
        # A hiccup fetching the GitHub OIDC token that feeds the WIF exchange (ADR-0087,
        # _fetch_github_oidc_token) surfaces here, on the same first call that triggers
        # the exchange — mapped transient for the same reason an Anthropic-side timeout
        # is: retrying is the right response, not aborting the sermon.
        httpx.TimeoutException,
        httpx.ConnectError,
    ) as exc:
        raise TransientLLMError(str(exc)) from exc
    except (
        anthropic.BadRequestError,
        anthropic.UnprocessableEntityError,
    ) as exc:
        raise PermanentLLMError(str(exc)) from exc
    return batch.id


def submit_batch(
    requests: Sequence[BatchRequest],
    *,
    model: str | None = None,
    output_config: dict[str, Any] | None = None,
    submit_fn: Callable[..., str] = _default_submit_batch,
) -> str:
    """Submit one batch covering every request, returning the id to persist (spec 0020).

    The model defaults to ``ANTHROPIC_MODEL`` from config, same as :func:`call`. Unlike
    :func:`call`, a submission failure is not retried here — the request may already
    have reached Anthropic before a client-side error surfaced, and re-submitting risks
    double-billing a whole batch. A caller that wants a retry re-invokes this on a later
    pipeline run, the same way polling and the 24h fallback already work (ADR-0061).

    ``output_config`` (ADR-0083, #542) is forwarded to ``submit_fn`` as a keyword
    argument only when given, the same additive-not-breaking contract as :func:`call`.
    """
    resolved_model = model or _resolve_configured_model()
    if output_config is not None:
        return submit_fn(requests, resolved_model, output_config=output_config)
    return submit_fn(requests, resolved_model)


def _default_poll_batch(batch_id: str) -> BatchStatus:
    """Check one batch's processing status, mapping transient SDK errors."""
    import anthropic
    import httpx

    try:
        batch = _get_client().messages.batches.retrieve(batch_id)
    except (
        anthropic.APITimeoutError,
        anthropic.RateLimitError,
        anthropic.APIConnectionError,
        # A hiccup fetching the GitHub OIDC token that feeds the WIF exchange (ADR-0087,
        # _fetch_github_oidc_token) surfaces here, on the same first call that triggers
        # the exchange — mapped transient for the same reason an Anthropic-side timeout
        # is: retrying is the right response, not aborting the sermon.
        httpx.TimeoutException,
        httpx.ConnectError,
    ) as exc:
        raise TransientLLMError(str(exc)) from exc
    except (
        anthropic.BadRequestError,
        anthropic.UnprocessableEntityError,
    ) as exc:
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
    """Check whether a submitted batch has finished processing (spec 0020's poll step)."""
    return poll_fn(batch_id)


def _default_retrieve_batch(batch_id: str) -> list[BatchResult]:
    """Fetch an ended batch's results, decoding each item, mapping transient SDK errors."""
    import anthropic
    import httpx

    try:
        items = list(_get_client().messages.batches.results(batch_id))
    except (
        anthropic.APITimeoutError,
        anthropic.RateLimitError,
        anthropic.APIConnectionError,
        # A hiccup fetching the GitHub OIDC token that feeds the WIF exchange (ADR-0087,
        # _fetch_github_oidc_token) surfaces here, on the same first call that triggers
        # the exchange — mapped transient for the same reason an Anthropic-side timeout
        # is: retrying is the right response, not aborting the sermon.
        httpx.TimeoutException,
        httpx.ConnectError,
    ) as exc:
        raise TransientLLMError(str(exc)) from exc
    except (
        anthropic.BadRequestError,
        anthropic.UnprocessableEntityError,
    ) as exc:
        raise PermanentLLMError(str(exc)) from exc

    resolved: list[BatchResult] = []
    for item in items:
        if item.result.type == "succeeded":
            message = item.result.message
            text = "".join(block.text for block in message.content if block.type == "text")
            resolved.append(
                BatchResult(
                    custom_id=item.custom_id,
                    response=LLMResponse(
                        text=text,
                        input_tokens=message.usage.input_tokens,
                        output_tokens=message.usage.output_tokens,
                        model=message.model,
                        stop_reason=message.stop_reason,
                        billing="batch",
                    ),
                )
            )
        elif item.result.type == "errored":
            resolved.append(
                BatchResult(custom_id=item.custom_id, error=item.result.error.error.message)
            )
        else:
            # canceled / expired carry no message of their own — the type is the reason.
            resolved.append(BatchResult(custom_id=item.custom_id, error=item.result.type))
    return resolved


def retrieve_batch(
    batch_id: str,
    *,
    retrieve_fn: Callable[[str], list[BatchResult]] = _default_retrieve_batch,
) -> list[BatchResult]:
    """Retrieve an ended batch's results as one :class:`BatchResult` per item (spec 0020)."""
    return retrieve_fn(batch_id)


def cost_usd(response: LLMResponse) -> float:
    """USD cost of one completion, by the model and rate that actually billed it (PRD §12).

    Looks up :data:`MODEL_PRICING` for ``response.model`` so an ``ANTHROPIC_MODEL``
    override is billed at its own rate rather than Sonnet's (#34). A model with no
    registered price logs a loud warning and costs ``0.0`` in the ledger rather than
    failing the sermon after the paid LLM call already succeeded (#128) — the pricing
    table is a manual sync point and a lookup miss shouldn't burn LLM budget on every
    sermon for the rest of the batch.

    ``response.billing == "batch"`` applies :data:`_BATCH_DISCOUNT` on top of the
    synchronous list rate, so a backfill sermon resolved through the Batches API is
    priced at the rate that actually applied — never the synchronous path's rate,
    and never an average of the two (ADR-0061, spec 0020).

    The result is rounded for a stable ledger diff.
    """
    try:
        input_price, output_price = MODEL_PRICING[response.model]
    except KeyError:
        logger.warning(
            "no list price registered for model %r; recording cost as $0.00 "
            "(add it to MODEL_PRICING, see PRD §12)",
            response.model,
        )
        return 0.0
    dollars = (
        response.input_tokens / 1_000_000 * input_price
        + response.output_tokens / 1_000_000 * output_price
    )
    if response.billing == "batch":
        dollars *= _BATCH_DISCOUNT
    return round(dollars, 6)
