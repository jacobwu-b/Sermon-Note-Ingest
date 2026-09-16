import base64
import io
import json
import urllib.error

import pytest

from poller import pipeline_registry
from poller.config import PipelineConfig


def _config() -> PipelineConfig:
    return PipelineConfig(repo="owner/sermon-note-pipeline", token="ghp_456")


class _FakeResponse:
    def __init__(self, body: bytes, status: int = 200) -> None:
        self._body = body
        self.status = status

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _contents_response(registry: dict) -> _FakeResponse:
    content = json.dumps(registry).encode("utf-8")
    payload = json.dumps({"content": base64.b64encode(content).decode("ascii")}).encode("utf-8")
    return _FakeResponse(payload)


def test_list_pending_batches_returns_only_pending_batch_records(monkeypatch):
    registry = {
        "sermons": [
            {
                "guid": "g1",
                "source": "menlo",
                "state": "pending_batch",
                "batch_id": "batch_1",
            },
            {
                "guid": "g2",
                "source": "menlo",
                "state": "published",
                "batch_id": None,
            },
            {
                "guid": "g3",
                "source": "north_point",
                "state": "pending_batch",
                "batch_id": "batch_2",
            },
        ],
        "notified_alerts": [],
    }

    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
        return _contents_response(registry)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    pending = pipeline_registry.list_pending_batches(config=_config())

    assert captured["url"] == (
        "https://api.github.com/repos/owner/sermon-note-pipeline/contents/state/registry.json"
    )
    assert captured["headers"]["authorization"] == "Bearer ghp_456"
    assert pending == [
        pipeline_registry.PendingBatch(guid="g1", source="menlo", batch_id="batch_1"),
        pipeline_registry.PendingBatch(guid="g3", source="north_point", batch_id="batch_2"),
    ]


def test_list_pending_batches_returns_empty_list_when_none_pending(monkeypatch):
    registry = {"sermons": [{"guid": "g1", "source": "menlo", "state": "published", "batch_id": None}]}
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: _contents_response(registry))
    assert pipeline_registry.list_pending_batches(config=_config()) == []


def test_list_pending_batches_skips_a_pending_record_missing_batch_id(monkeypatch):
    registry = {"sermons": [{"guid": "g1", "source": "menlo", "state": "pending_batch", "batch_id": None}]}
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: _contents_response(registry))
    assert pipeline_registry.list_pending_batches(config=_config()) == []


def test_list_pending_batches_raises_on_http_error(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, io.BytesIO(b"missing"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(pipeline_registry.PipelineRegistryError):
        pipeline_registry.list_pending_batches(config=_config())


def test_list_pending_batches_raises_on_network_failure(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(pipeline_registry.PipelineRegistryError):
        pipeline_registry.list_pending_batches(config=_config())


def test_list_pending_batches_raises_on_malformed_response(monkeypatch):
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request, timeout: _FakeResponse(b'{"not_content": true}')
    )
    with pytest.raises(pipeline_registry.PipelineRegistryError):
        pipeline_registry.list_pending_batches(config=_config())


def test_list_pending_batches_sets_a_non_default_user_agent(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
        return _contents_response({"sermons": []})

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    pipeline_registry.list_pending_batches(config=_config())

    assert "user-agent" in captured["headers"]
    assert "python-urllib" not in captured["headers"]["user-agent"].lower()
