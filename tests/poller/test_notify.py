import io
import json
import urllib.error

import pytest

from poller import notify
from poller.config import NotifyConfig
from poller.sources.base import SermonItem


def _item(title: str = "Hear and Do", published_on: str = "2026-09-06") -> SermonItem:
    return SermonItem(
        guid="g1",
        title=title,
        raw_title=title,
        series="Luke",
        speaker="Jane Doe",
        published_on=published_on,
        published_at=None,
        episode_url="https://example.org/ep1",
        audio_url="https://example.org/ep1.mp3",
        blurb="A sermon.",
    )


def _config() -> NotifyConfig:
    return NotifyConfig(from_addr="alerts@example.org", to_addr="team@example.org", api_key="re_123")


class _FakeResponse:
    def __init__(self, status: int = 200) -> None:
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_send_new_sermons_with_no_items_does_not_call_the_network(monkeypatch):
    calls = []
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **k: calls.append(1))
    notify.send_new_sermons("menlo", [], _config())
    assert calls == []


def test_send_new_sermons_posts_the_expected_payload(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
        captured["body"] = json.loads(request.data)
        return _FakeResponse(200)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    notify.send_new_sermons("menlo", [_item()], _config())

    assert captured["url"] == "https://api.resend.com/emails"
    assert captured["headers"]["authorization"] == "Bearer re_123"
    assert captured["body"]["from"] == "alerts@example.org"
    assert captured["body"]["to"] == ["team@example.org"]
    assert "Hear and Do" in captured["body"]["subject"]
    assert "Jane Doe" in captured["body"]["html"]


def test_send_new_sermons_sets_a_non_default_user_agent(monkeypatch):
    """Regression: the stdlib default UA gets the request blocked by Resend's
    edge as a bot signature (Cloudflare error 1010, 403) on every real send."""
    captured = {}

    def fake_urlopen(request, timeout):
        captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
        return _FakeResponse(200)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    notify.send_new_sermons("menlo", [_item()], _config())

    assert "user-agent" in captured["headers"]
    assert "python-urllib" not in captured["headers"]["user-agent"].lower()


def test_send_new_sermons_uses_a_count_only_subject_for_a_large_backlog(monkeypatch):
    """Regression: Resend rejects subjects at 2000+ chars, which a real
    backlog of concatenated titles reaches long before a handful of items."""
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data)
        return _FakeResponse(200)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    items = [_item(f"A Very Long Sermon Title Number {i}" * 3) for i in range(50)]
    notify.send_new_sermons("menlo", items, _config())

    assert captured["body"]["subject"] == "New 50 sermons from menlo"
    assert len(captured["body"]["subject"]) < notify._MAX_SUBJECT_LEN


def test_send_new_sermons_raises_notify_error_on_http_error(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 401, "Unauthorized", {}, io.BytesIO(b"denied"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(notify.NotifyError):
        notify.send_new_sermons("menlo", [_item()], _config())


def test_send_new_sermons_raises_notify_error_on_network_failure(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.URLError("no route to host")

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(notify.NotifyError):
        notify.send_new_sermons("menlo", [_item()], _config())


def test_non_sunday_items_returns_only_the_non_sunday_ones():
    sunday = _item(title="Sunday sermon", published_on="2026-09-06")
    friday = _item(title="Good Friday", published_on="2026-04-03")
    assert notify._non_sunday_items([sunday, friday]) == [friday]


def test_non_sunday_items_ignores_an_undated_item():
    """A missing/unparseable published_on is a different condition than this
    alert's concern (a dated-but-wrong-weekday value) and must not be flagged."""
    undated = _item(title="Untitled", published_on="")
    assert notify._non_sunday_items([undated]) == []


def test_send_new_sermons_html_carries_a_warning_for_a_non_sunday_item(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data)
        return _FakeResponse(200)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    friday = _item(title="Good Friday Service", published_on="2026-04-03")
    notify.send_new_sermons("pbc", [friday], _config())

    assert "Good Friday Service" in captured["body"]["html"]
    assert "not a Sunday" in captured["body"]["html"]


def test_send_new_sermons_html_has_no_warning_when_all_items_are_sunday(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data)
        return _FakeResponse(200)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    notify.send_new_sermons("menlo", [_item()], _config())

    assert "not a Sunday" not in captured["body"]["html"]
