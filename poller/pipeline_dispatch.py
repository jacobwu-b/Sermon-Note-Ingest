"""Dispatches an ingest event to Sermon-Note-Pipeline's ``pipeline.yml`` workflow.

Sermon-Note-Pipeline's own cron (its spec 0026) is the correctness backstop for
picking up a newly-pushed transcript, so a dispatch failure here is the
caller's to log and swallow — never a reason to fail a transcription run or
undo the ledger mark already written (same non-fatal shape as ADR-0007's
poll -> transcribe dispatch, moved into Python since this needs per-sermon
granularity that a workflow-level shell step can't see).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from poller.config import PipelineConfig
from poller.net import DEFAULT_USER_AGENT

_TIMEOUT = 15


class PipelineDispatchError(RuntimeError):
    """Raised when the GitHub Actions dispatch API rejects or cannot be reached."""


def dispatch_ingest_event(event: dict[str, object], *, source: str, config: PipelineConfig) -> None:
    """POST a ``workflow_dispatch`` of ``pipeline.yml`` on the Pipeline repo, carrying ``event``.

    ``event`` is serialized as the ``ingest_event`` workflow input (a JSON string,
    per Pipeline's own contract, spec 0026 in that repo); ``source`` narrows the
    dispatched run's ``sources`` input to the one church this sermon belongs to.
    """
    url = f"https://api.github.com/repos/{config.repo}/actions/workflows/pipeline.yml/dispatches"
    payload = json.dumps(
        {"ref": "main", "inputs": {"sources": source, "ingest_event": json.dumps(event, sort_keys=True)}}
    ).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {config.token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            # GitHub's edge blocks the stdlib default UA as a bot signature —
            # same fix as poller/net.py and poller/notify.py.
            "User-Agent": DEFAULT_USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            if response.status >= 300:
                raise PipelineDispatchError(f"GitHub returned status {response.status}")
    except urllib.error.HTTPError as exc:
        raise PipelineDispatchError(f"GitHub rejected the dispatch: {exc.code} {exc.read()!r}") from exc
    except urllib.error.URLError as exc:
        raise PipelineDispatchError(f"could not reach GitHub: {exc}") from exc
