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
    # The raw media type returns the file body itself, not the base64 JSON envelope.
    return _FakeResponse(json.dumps(registry).encode("utf-8"))


@pytest.fixture
def sleeps(monkeypatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(pipeline_registry.time, "sleep", recorded.append)
    return recorded


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


def test_list_pending_batches_raises_on_http_error(monkeypatch, sleeps):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, io.BytesIO(b"missing"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(pipeline_registry.PipelineRegistryError):
        pipeline_registry.list_pending_batches(config=_config())


def test_list_pending_batches_raises_on_network_failure(monkeypatch, sleeps):
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


def test_read_church_table_reads_pipeline_s_checked_in_table(monkeypatch):
    table = '{"menlo": {"rss": "https://example.org/feed.xml", "ingest": {"enabled": true}}}'
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
        return _FakeResponse(table.encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert pipeline_registry.read_church_table(config=_config()) == table
    assert captured["url"] == (
        "https://api.github.com/repos/owner/sermon-note-pipeline/contents/config/churches.json"
    )
    assert captured["headers"]["authorization"] == "Bearer ghp_456"


def test_read_church_table_raises_on_http_error(monkeypatch, sleeps):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, io.BytesIO(b"no access"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(pipeline_registry.PipelineRegistryError):
        pipeline_registry.read_church_table(config=_config())


def test_read_church_table_raises_when_the_table_is_not_utf8(monkeypatch):
    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: _FakeResponse(b"\xff\xfe{}"))
    with pytest.raises(pipeline_registry.PipelineRegistryError, match="not UTF-8"):
        pipeline_registry.read_church_table(config=_config())


def test_reads_request_the_raw_media_type_so_files_past_one_megabyte_arrive_whole(monkeypatch):
    # Past 1 MB the default JSON media type returns an empty `content` with
    # `encoding: "none"`; the raw media type returns the body for files up to 100 MB.
    sermons = [
        {"guid": f"guid-{i:06d}", "source": "menlo", "state": "published", "batch_id": None}
        for i in range(20_000)
    ]
    sermons.append({"guid": "g-last", "source": "menlo", "state": "pending_batch", "batch_id": "batch_9"})
    body = json.dumps({"sermons": sermons}).encode("utf-8")
    assert len(body) > 1024 * 1024
    captured = {}

    def fake_urlopen(request, timeout):
        captured["accept"] = request.headers.get("Accept")
        return _FakeResponse(body)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    pending = pipeline_registry.list_pending_batches(config=_config())

    assert captured["accept"] == "application/vnd.github.raw+json"
    assert pending == [pipeline_registry.PendingBatch(guid="g-last", source="menlo", batch_id="batch_9")]


def test_list_pending_batches_raises_clearly_on_an_over_limit_json_envelope(monkeypatch):
    envelope = {
        "name": "registry.json",
        "path": "state/registry.json",
        "size": 1_250_000,
        "encoding": "none",
        "content": "",
    }
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda request, timeout: _FakeResponse(json.dumps(envelope).encode())
    )
    with pytest.raises(pipeline_registry.PipelineRegistryError, match="no 'sermons' list"):
        pipeline_registry.list_pending_batches(config=_config())


@pytest.mark.parametrize(
    "transient",
    [
        urllib.error.HTTPError("https://api.github.com", 502, "Bad Gateway", {}, io.BytesIO(b"")),
        urllib.error.HTTPError("https://api.github.com", 429, "Too Many Requests", {}, io.BytesIO(b"")),
        urllib.error.URLError("connection reset by peer"),
        TimeoutError("The read operation timed out"),
    ],
    ids=["5xx", "429", "network", "timeout"],
)
def test_read_church_table_retries_a_transient_failure_then_succeeds(monkeypatch, sleeps, transient):
    table = '{"menlo": {"rss": "https://example.org/feed.xml", "ingest": {"enabled": true}}}'
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        if len(calls) == 1:
            raise transient
        return _FakeResponse(table.encode("utf-8"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert pipeline_registry.read_church_table(config=_config()) == table
    assert len(calls) == 2
    assert sleeps == [1.0]


def test_read_church_table_gives_up_after_bounded_attempts_with_backoff(monkeypatch, sleeps):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 503, "Service Unavailable", {}, io.BytesIO(b""))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(pipeline_registry.PipelineRegistryError, match="after 3 attempts"):
        pipeline_registry.read_church_table(config=_config())

    assert len(calls) == 3
    assert sleeps == [1.0, 2.0]


def test_read_church_table_does_not_retry_a_permanent_rejection(monkeypatch, sleeps):
    calls = []

    def fake_urlopen(request, timeout):
        calls.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 404, "Not Found", {}, io.BytesIO(b"missing"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(pipeline_registry.PipelineRegistryError, match="404"):
        pipeline_registry.read_church_table(config=_config())

    assert len(calls) == 1
    assert sleeps == []
