"""Reads Sermon-Note-Pipeline's committed files: its registry and its church table.

Pipeline is the single writer of its own registry (Pipeline's ADR-0023); this
module only ever reads it, and only ever reads *status* (``state``/``batch_id``),
never batch *results* — retrieval and generation stay entirely inside Pipeline.
It also reads Pipeline's ``config/churches.json``, the one church table both repos
run on (docs/decisions/0016). Each read is a point-in-time fetch of one committed
GitHub blob via the Contents API, so it never sees a torn write (docs/decisions/0010).
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from poller.config import PipelineConfig
from poller.net import DEFAULT_USER_AGENT

_TIMEOUT = 15
_REGISTRY_PATH = "state/registry.json"
_CHURCH_TABLE_PATH = "config/churches.json"
_PENDING_STATE = "pending_batch"


class PipelineRegistryError(RuntimeError):
    """Raised when Pipeline's registry cannot be fetched or parsed."""


@dataclass(frozen=True)
class PendingBatch:
    """One Pipeline sermon record currently awaiting a Claude batch's completion."""

    guid: str
    source: str
    batch_id: str


def _read_file(path: str, *, config: PipelineConfig) -> bytes:
    """One committed file from Pipeline's default branch, decoded from the Contents API."""
    url = f"https://api.github.com/repos/{config.repo}/contents/{path}"
    request = urllib.request.Request(
        url,
        headers={
            "Authorization": f"Bearer {config.token}",
            "Accept": "application/vnd.github+json",
            # GitHub's edge blocks the stdlib default UA as a bot signature —
            # same fix as poller/net.py and poller/pipeline_dispatch.py.
            "User-Agent": DEFAULT_USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            if response.status >= 300:
                raise PipelineRegistryError(f"GitHub returned status {response.status} for {path}")
            raw = response.read()
    except urllib.error.HTTPError as exc:
        raise PipelineRegistryError(f"GitHub rejected the read of {path}: {exc.code} {exc.read()!r}") from exc
    except urllib.error.URLError as exc:
        raise PipelineRegistryError(f"could not reach GitHub: {exc}") from exc

    try:
        return base64.b64decode(json.loads(raw)["content"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise PipelineRegistryError(f"malformed Contents API response for {path}: {exc}") from exc


def read_church_table(*, config: PipelineConfig) -> str:
    """The text of Pipeline's ``config/churches.json``, the church table both repos read."""
    try:
        return _read_file(_CHURCH_TABLE_PATH, config=config).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PipelineRegistryError(f"{_CHURCH_TABLE_PATH} is not UTF-8: {exc}") from exc


def list_pending_batches(*, config: PipelineConfig) -> list[PendingBatch]:
    """Return every Pipeline registry record whose ``state`` is ``"pending_batch"``."""
    try:
        registry = json.loads(_read_file(_REGISTRY_PATH, config=config))
    except (json.JSONDecodeError, ValueError) as exc:
        raise PipelineRegistryError(f"malformed registry.json: {exc}") from exc

    sermons = registry.get("sermons") if isinstance(registry, dict) else None
    if not isinstance(sermons, list):
        raise PipelineRegistryError("registry.json has no 'sermons' list")

    pending: list[PendingBatch] = []
    for record in sermons:
        if not isinstance(record, dict) or record.get("state") != _PENDING_STATE:
            continue
        guid, source, batch_id = record.get("guid"), record.get("source"), record.get("batch_id")
        if not guid or not source or not batch_id:
            continue
        pending.append(PendingBatch(guid=guid, source=source, batch_id=batch_id))
    return pending
