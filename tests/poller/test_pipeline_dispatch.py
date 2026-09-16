import io
import json
import urllib.error

import pytest

from poller import pipeline_dispatch
from poller.config import PipelineConfig


def _config() -> PipelineConfig:
    return PipelineConfig(repo="owner/sermon-note-pipeline", token="ghp_456")


class _FakeResponse:
    def __init__(self, status: int = 204) -> None:
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_dispatch_ingest_event_posts_the_expected_request(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
        captured["body"] = json.loads(request.data)
        return _FakeResponse(204)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    event = {"event": "sermon_detected", "source": "menlo", "external_id": "g1"}
    pipeline_dispatch.dispatch_ingest_event(event, source="menlo", config=_config())

    assert captured["url"] == (
        "https://api.github.com/repos/owner/sermon-note-pipeline/actions/workflows/pipeline.yml/dispatches"
    )
    assert captured["headers"]["authorization"] == "Bearer ghp_456"
    assert captured["headers"]["accept"] == "application/vnd.github+json"
    assert captured["body"]["ref"] == "main"
    assert captured["body"]["inputs"]["sources"] == "menlo"
    assert json.loads(captured["body"]["inputs"]["ingest_event"]) == event


def test_dispatch_ingest_event_sets_a_non_default_user_agent(monkeypatch):
    """Regression: the stdlib default UA gets blocked as a bot signature by
    GitHub's edge, the same failure mode poller/net.py and poller/notify.py
    already work around."""
    captured = {}

    def fake_urlopen(request, timeout):
        captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
        return _FakeResponse(204)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    pipeline_dispatch.dispatch_ingest_event({"event": "sermon_detected"}, source="menlo", config=_config())

    assert "user-agent" in captured["headers"]
    assert "python-urllib" not in captured["headers"]["user-agent"].lower()


def test_dispatch_ingest_event_raises_on_http_error(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, io.BytesIO(b"denied"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(pipeline_dispatch.PipelineDispatchError):
        pipeline_dispatch.dispatch_ingest_event(
            {"event": "sermon_detected"}, source="menlo", config=_config()
        )


def test_dispatch_ingest_event_raises_on_network_failure(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(pipeline_dispatch.PipelineDispatchError):
        pipeline_dispatch.dispatch_ingest_event(
            {"event": "sermon_detected"}, source="menlo", config=_config()
        )


def test_dispatch_ingest_event_raises_on_non_2xx_status(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: _FakeResponse(404))
    with pytest.raises(pipeline_dispatch.PipelineDispatchError):
        pipeline_dispatch.dispatch_ingest_event(
            {"event": "sermon_detected"}, source="menlo", config=_config()
        )
