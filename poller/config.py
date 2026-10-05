"""Configuration: the church table and the environment.

All env vars this app reads are listed in ``.env.example``. The church table is not
one of them: it is Sermon-Note-Pipeline's checked-in ``config/churches.json``, read
through :mod:`poller.pipeline_registry` with ``PIPELINE_REPO``/``PIPELINE_DISPATCH_TOKEN``
(docs/decisions/0016). ``NOTIFY_EMAIL_FROM``, ``NOTIFY_EMAIL_TO``, and
``RESEND_API_KEY`` are repository secrets.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or malformed."""


@dataclass(frozen=True)
class ChurchConfig:
    """One church's entry from the church table: its ``rss`` and ``ingest`` section."""

    name: str
    rss: str
    enabled: bool
    notify: bool
    # Operator-supplied terms the feed can't tell us (the church's full name, campus
    # names); prompted into every transcription of this church (poller/prompting.py).
    vocabulary: tuple[str, ...] = ()
    # Whether Pipeline generates notes for this church. Ingest transcribes a church
    # either way, but only dispatches Pipeline for one it runs.
    pipeline_enabled: bool = True


def _fetch_church_table() -> str:
    """Pipeline's church table, read from its ``main`` branch. Raises on any failure.

    There is no fallback copy: polling a stale table is the drift the shared file exists
    to end, so a failed read fails the run (docs/decisions/0016).
    """
    # Imported here because pipeline_registry imports PipelineConfig from this module.
    from poller import pipeline_registry

    return pipeline_registry.read_church_table(config=load_pipeline_config())


def load_churches(raw: str | None = None) -> dict[str, ChurchConfig]:
    """Parse the church table into ``{name: ChurchConfig}``.

    ``raw`` overrides the table's text; tests pass it. Each church carries its ``rss``
    and one section per repo; only the ``ingest`` section is this repo's. A church
    with no ``ingest`` section is one only Pipeline uses, so it is disabled here. A
    malformed table is a configuration error — there is nothing sensible to poll
    without it.
    """
    text = raw if raw is not None else _fetch_church_table()
    if not text.strip():
        raise ConfigError("the church table is empty")
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(f"the church table is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError("the church table must be a JSON object keyed by church name")

    churches: dict[str, ChurchConfig] = {}
    for name, entry in data.items():
        if not isinstance(entry, dict):
            raise ConfigError(f"church table entry {name!r} must be a JSON object")
        section = entry.get("ingest", {})
        if not isinstance(section, dict):
            raise ConfigError(f"church table entry {name!r} has a non-object ingest section")
        pipeline_section = entry.get("pipeline", {})
        if not isinstance(pipeline_section, dict):
            raise ConfigError(f"church table entry {name!r} has a non-object pipeline section")
        vocabulary = section.get("vocabulary", [])
        if not isinstance(vocabulary, list) or not all(isinstance(term, str) for term in vocabulary):
            raise ConfigError(f"church table entry {name!r}: ingest.vocabulary must be a list of strings")
        churches[name] = ChurchConfig(
            name=name,
            rss=str(entry.get("rss", "")),
            enabled=bool(section.get("enabled", False)),
            notify=bool(section.get("notify", False)),
            vocabulary=tuple(vocabulary),
            pipeline_enabled=bool(pipeline_section.get("enabled", False)),
        )
    return churches


def _env_flag(name: str, default: bool) -> bool:
    """A boolean env var; unset or empty (an unset Actions ``vars.*``) reads as ``default``."""
    raw = (os.environ.get(name) or "").strip().lower()
    if not raw:
        return default
    if raw in ("true", "1", "yes"):
        return True
    if raw in ("false", "0", "no"):
        return False
    raise ConfigError(f"{name} must be true or false, got {raw!r}")


@dataclass(frozen=True)
class NotifyConfig:
    """The Resend email secrets, read once at startup."""

    from_addr: str
    to_addr: str
    api_key: str


def load_notify_config() -> NotifyConfig:
    """Read the three notify secrets, raising if any are missing.

    Only called when a poll actually has something to notify about, so a repo
    that never enables ``notify`` for any church never needs these secrets set.
    """
    missing = [
        var for var in ("NOTIFY_EMAIL_FROM", "NOTIFY_EMAIL_TO", "RESEND_API_KEY") if not os.environ.get(var)
    ]
    if missing:
        raise ConfigError(f"missing required env var(s) for notification: {', '.join(missing)}")
    return NotifyConfig(
        from_addr=os.environ["NOTIFY_EMAIL_FROM"],
        to_addr=os.environ["NOTIFY_EMAIL_TO"],
        api_key=os.environ["RESEND_API_KEY"],
    )


@dataclass(frozen=True)
class WhisperConfig:
    """The faster-whisper accuracy/runtime knobs, read once at startup."""

    model: str
    compute_type: str
    cpu_threads: int
    beam_size: int
    condition_on_previous_text: bool
    # The kill switch for poller/prompting.py's hotwords (spec 0003).
    domain_prompt: bool


def load_whisper_config() -> WhisperConfig:
    """Read the ``WHISPER_*`` env vars; every one is optional.

    Defaults: ``small``/``int8``/``0`` threads — ``0`` tells faster-whisper to pick its
    own thread count (every core it can see). A single-run GitHub Actions job never needs
    to override this; a local run fanning out several parallel shards (scripts/
    transcribe_local.py) sets it per-shard so shards divide cores instead of each
    claiming every core faster-whisper can see. ``beam_size`` 5 and
    ``condition_on_previous_text`` true are faster-whisper's own defaults, exposed so a
    production run can be retuned without a code change.
    """
    return WhisperConfig(
        model=os.environ.get("WHISPER_MODEL") or "small",
        compute_type=os.environ.get("WHISPER_COMPUTE_TYPE") or "int8",
        cpu_threads=int(os.environ.get("WHISPER_CPU_THREADS") or "0"),
        beam_size=int(os.environ.get("WHISPER_BEAM_SIZE") or "5"),
        condition_on_previous_text=_env_flag("WHISPER_CONDITION_ON_PREVIOUS_TEXT", True),
        domain_prompt=_env_flag("WHISPER_DOMAIN_PROMPT", True),
    )


@dataclass(frozen=True)
class ContentRepoConfig:
    """Where and how to push transcripts to the private Sermon-Note-Content repo."""

    repo: str
    token: str
    branch: str


def load_content_repo_config() -> ContentRepoConfig:
    """Read ``CONTENT_REPO``/``CONTENT_REPO_TOKEN`` (required) and ``CONTENT_REPO_BRANCH``
    (default ``main``), raising if either required var is missing.
    """
    repo = os.environ.get("CONTENT_REPO", "")
    token = os.environ.get("CONTENT_REPO_TOKEN", "")
    missing = [name for name, value in (("CONTENT_REPO", repo), ("CONTENT_REPO_TOKEN", token)) if not value]
    if missing:
        raise ConfigError(f"missing required env var(s) for content repo push: {', '.join(missing)}")
    return ContentRepoConfig(repo=repo, token=token, branch=os.environ.get("CONTENT_REPO_BRANCH") or "main")


@dataclass(frozen=True)
class PipelineConfig:
    """Where and how to dispatch an ingest event to the Sermon-Note-Pipeline repo."""

    repo: str
    token: str


def load_pipeline_config() -> PipelineConfig:
    """Read ``PIPELINE_REPO``/``PIPELINE_DISPATCH_TOKEN`` (both required), raising if either is missing."""
    repo = os.environ.get("PIPELINE_REPO", "")
    token = os.environ.get("PIPELINE_DISPATCH_TOKEN", "")
    missing = [
        name for name, value in (("PIPELINE_REPO", repo), ("PIPELINE_DISPATCH_TOKEN", token)) if not value
    ]
    if missing:
        raise ConfigError(f"missing required env var(s) for pipeline dispatch: {', '.join(missing)}")
    return PipelineConfig(repo=repo, token=token)


@dataclass(frozen=True)
class AnthropicConfig:
    """Anthropic auth used to check Claude batch status (docs/decisions/0010).

    Two auth paths, same precedence as Sermon-Note-Pipeline's ADR-0087: an explicit
    ``api_key`` wins if set; otherwise ``federation_rule_id`` (with
    ``organization_id``) selects Workload Identity Federation. ``service_account_id``
    and ``workspace_id`` are optional narrowing on the federation exchange.
    """

    api_key: str | None
    federation_rule_id: str | None
    organization_id: str | None
    service_account_id: str | None
    workspace_id: str | None


def load_anthropic_config() -> AnthropicConfig:
    """Read Anthropic auth: ``ANTHROPIC_API_KEY`` or ``ANTHROPIC_FEDERATION_RULE_ID``
    (with ``ANTHROPIC_ORGANIZATION_ID``), raising if neither path is configured.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY") or None
    federation_rule_id = os.environ.get("ANTHROPIC_FEDERATION_RULE_ID") or None
    organization_id = os.environ.get("ANTHROPIC_ORGANIZATION_ID") or None
    if not api_key and not federation_rule_id:
        raise ConfigError(
            "missing required env var for Claude batch poll: ANTHROPIC_API_KEY "
            "(or ANTHROPIC_FEDERATION_RULE_ID for Workload Identity Federation)"
        )
    if federation_rule_id and not api_key and not organization_id:
        raise ConfigError(
            "missing required env var for Claude batch poll: ANTHROPIC_ORGANIZATION_ID "
            "(required alongside ANTHROPIC_FEDERATION_RULE_ID)"
        )
    return AnthropicConfig(
        api_key=api_key,
        federation_rule_id=federation_rule_id,
        organization_id=organization_id,
        service_account_id=os.environ.get("ANTHROPIC_SERVICE_ACCOUNT_ID") or None,
        workspace_id=os.environ.get("ANTHROPIC_WORKSPACE_ID") or None,
    )


def load_log_level(default: str = "INFO") -> str:
    """Return ``LOG_LEVEL`` capitalized, defaulting to ``INFO`` — the sole place this repo reads that var."""
    value = os.environ.get("LOG_LEVEL")
    return value.upper() if value else default
