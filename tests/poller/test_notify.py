import dataclasses
import io
import json
import urllib.error

import pytest

from poller import notify, quality
from poller.config import NotifyConfig
from poller.sources.base import SermonItem


def _item(title: str = "Hear and Do", preached_on: str = "2026-09-06") -> SermonItem:
    return SermonItem(
        guid="g1",
        title=title,
        raw_title=title,
        series="Luke",
        speaker="Jane Doe",
        preached_on=preached_on,
        feed_published_at=None,
        episode_url="https://example.org/ep1",
        audio_url="https://example.org/ep1.mp3",
        blurb="A sermon.",
    )


def _config() -> NotifyConfig:
    return NotifyConfig(from_addr="alerts@example.org", to_addr="team@example.org", api_key="re_123")


@pytest.fixture(autouse=True)
def _no_backoff_sleep(monkeypatch) -> list[float]:
    delays: list[float] = []
    monkeypatch.setattr(notify.time, "sleep", delays.append)
    return delays


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


def test_send_new_sermons_keeps_a_quote_in_the_episode_url_inside_the_href(monkeypatch):
    """Regression: a feed ``<link>`` carrying ``"`` closed the href and added its own attributes."""
    captured = _capture(monkeypatch)
    url = 'https://example.org/ep1" onmouseover="alert(1)'
    notify.send_new_sermons("menlo", [dataclasses.replace(_item(), episode_url=url)], _config())

    html = captured["body"]["html"]
    assert 'href="https://example.org/ep1&quot; onmouseover=&quot;alert(1)"' in html
    assert 'onmouseover="' not in html


def test_send_new_sermons_shows_a_non_http_episode_url_as_plain_text(monkeypatch):
    captured = _capture(monkeypatch)
    url = "javascript:alert(document.cookie)"
    notify.send_new_sermons("menlo", [dataclasses.replace(_item(), episode_url=url)], _config())

    html = captured["body"]["html"]
    assert "href" not in html
    assert f"<br>{url}</li>" in html


def test_send_new_sermons_escapes_the_church_and_preached_on(monkeypatch):
    captured = _capture(monkeypatch)
    notify.send_new_sermons("<b>menlo</b>", [_item(preached_on="2026-09-06<i>")], _config())

    html = captured["body"]["html"]
    assert "&lt;b&gt;menlo&lt;/b&gt; published" in html
    assert "2026-09-06&lt;i&gt;" in html


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


def _flaky_urlopen(failures: list[BaseException]):
    """A urlopen that raises each of ``failures`` in turn, then accepts; records every request."""
    requests = []

    def fake_urlopen(request, timeout):
        requests.append(request)
        if len(requests) <= len(failures):
            raise failures[len(requests) - 1]
        return _FakeResponse(200)

    return fake_urlopen, requests


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(_RESEND, code, "error", {}, io.BytesIO(b'{"message":"try later"}'))


_RESEND = "https://api.resend.com/emails"


@pytest.mark.parametrize(
    "failure",
    [
        _http_error(500),
        _http_error(503),
        _http_error(429),
        urllib.error.URLError("no route to host"),
        TimeoutError("The read operation timed out"),
    ],
)
def test_send_new_sermons_retries_a_transient_failure_and_then_succeeds(
    monkeypatch, failure, _no_backoff_sleep
):
    fake_urlopen, requests = _flaky_urlopen([failure])
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    notify.send_new_sermons("menlo", [_item()], _config())

    assert len(requests) == 2
    assert _no_backoff_sleep == [1.0]


def test_send_new_sermons_does_not_retry_a_rejected_request(monkeypatch):
    fake_urlopen, requests = _flaky_urlopen([_http_error(422)])
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(notify.NotifyError, match="422"):
        notify.send_new_sermons("menlo", [_item()], _config())
    assert len(requests) == 1


def test_send_new_sermons_raises_notify_error_once_every_attempt_times_out(monkeypatch, _no_backoff_sleep):
    fake_urlopen, requests = _flaky_urlopen([TimeoutError("timed out")] * 3)
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    with pytest.raises(notify.NotifyError, match="3 attempts"):
        notify.send_new_sermons("menlo", [_item()], _config())
    assert len(requests) == 3
    assert _no_backoff_sleep == [1.0, 2.0]


def test_send_reuses_one_idempotency_key_across_retries_but_not_across_sends(monkeypatch):
    fake_urlopen, requests = _flaky_urlopen([TimeoutError("timed out")])
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    notify.send_new_sermons("menlo", [_item()], _config())
    notify.send_new_sermons("menlo", [_item()], _config())

    keys = [request.get_header("Idempotency-key") for request in requests]
    assert len(keys) == 3
    assert keys[0] and keys[0] == keys[1]
    assert keys[2] and keys[2] != keys[0]


def test_send_quality_alert_retries_a_server_error(monkeypatch):
    fake_urlopen, requests = _flaky_urlopen([_http_error(502)])
    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    notify.send_quality_alert("menlo", [_flagged()], _config())

    assert len(requests) == 2


def test_off_day_items_returns_only_items_off_a_sunday_or_special_service():
    sunday = _item(title="Sunday sermon", preached_on="2026-09-06")
    good_friday = _item(title="Good Friday", preached_on="2026-04-03")
    wednesday = _item(title="Late-posted sermon", preached_on="2026-09-09")
    assert notify._off_day_items([sunday, good_friday, wednesday]) == [wednesday]


def test_off_day_items_ignores_an_undated_item():
    """A missing/unparseable preached_on is a different condition than this
    alert's concern (a dated-but-wrong-weekday value) and must not be flagged."""
    undated = _item(title="Untitled", preached_on="")
    assert notify._off_day_items([undated]) == []


def test_send_new_sermons_html_carries_a_warning_for_an_off_day_item(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data)
        return _FakeResponse(200)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    wednesday = _item(title="Late-posted sermon", preached_on="2026-09-09")
    notify.send_new_sermons("pbc", [wednesday], _config())

    assert "Late-posted sermon" in captured["body"]["html"]
    assert "Heads up" in captured["body"]["html"]


def test_send_new_sermons_html_has_no_warning_for_sunday_and_special_service_items(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["body"] = json.loads(request.data)
        return _FakeResponse(200)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    christmas_eve = _item(title="Christmas Eve Service", preached_on="2025-12-24")
    notify.send_new_sermons("menlo", [_item(), christmas_eve], _config())

    assert "Heads up" not in captured["body"]["html"]


def _flagged(title: str = "Hear and Do") -> notify.FlaggedTranscript:
    return notify.FlaggedTranscript(
        guid="menlochurchvideo.podbean.com/3a5d9a10",
        title=title,
        preached_on="2026-08-23",
        score=quality.QualityScore(words=6606, loop_words=72, credit_hits=2, hotword_pairs=0),
        flags=("loops", "credits"),
    )


def _capture(monkeypatch) -> dict:
    captured: dict = {}

    def fake_urlopen(request, timeout):
        captured["url"] = request.full_url
        captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
        captured["body"] = json.loads(request.data)
        return _FakeResponse(200)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    return captured


def test_send_quality_alert_names_each_flagged_sermon_with_its_counts_and_flags(monkeypatch):
    captured = _capture(monkeypatch)

    notify.send_quality_alert("menlo", [_flagged()], _config())

    body = captured["body"]
    assert captured["url"] == "https://api.resend.com/emails"
    assert body["to"] == ["team@example.org"]
    assert body["subject"] == "Transcript quality: 1 sermon from menlo flagged"
    for expected in ("Hear and Do", "2026-08-23", "menlochurchvideo.podbean.com/3a5d9a10", "loops, credits"):
        assert expected in body["html"]
    for count in ("6606 words", "72 loop words", "2 credit hits", "0 hotword pairs"):
        assert count in body["html"]


def test_send_quality_alert_sets_the_project_user_agent(monkeypatch):
    captured = _capture(monkeypatch)
    notify.send_quality_alert("menlo", [_flagged()], _config())
    assert captured["headers"]["user-agent"] == notify.DEFAULT_USER_AGENT


def test_send_quality_alert_escapes_a_title_with_markup(monkeypatch):
    captured = _capture(monkeypatch)
    notify.send_quality_alert("menlo", [_flagged(title="Faith <and> Works")], _config())
    assert "Faith &lt;and&gt; Works" in captured["body"]["html"]


def test_send_quality_alert_raises_notify_error_on_http_error(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 422, "Unprocessable", {}, io.BytesIO(b"bad"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(notify.NotifyError, match="422"):
        notify.send_quality_alert("menlo", [_flagged()], _config())


def test_send_retranscribe_alert_sends_the_subject_and_one_escaped_item_per_line(monkeypatch):
    captured = _capture(monkeypatch)

    notify.send_retranscribe_alert(
        "Re-transcription verify: 2 violations",
        [
            "menlo/g1: transcripts/menlo/a.txt hashes to 0123456789ab, ledger names ba9876543210",
            "pbc <x> & y",
        ],
        _config(),
    )

    body = captured["body"]
    assert body["subject"] == "Re-transcription verify: 2 violations"
    assert "<li>menlo/g1: transcripts/menlo/a.txt hashes to 0123456789ab" in body["html"]
    assert "pbc &lt;x&gt; &amp; y" in body["html"]
    assert captured["headers"]["user-agent"] == notify.DEFAULT_USER_AGENT


def test_send_retranscribe_alert_raises_notify_error_on_http_error(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 500, "Server Error", {}, io.BytesIO(b"down"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(notify.NotifyError, match="500"):
        notify.send_retranscribe_alert("subject", ["line"], _config())


def test_send_discovery_alert_posts_the_subject_and_one_item_per_line(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
        captured["body"] = json.loads(request.data)
        return _FakeResponse(200)

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    notify.send_discovery_alert(
        "Discovery: no sermon from westgate for Sunday 2026-10-25",
        ["westgate: newest ledgered sermon is 2026-10-18 <check the feed>"],
        _config(),
    )

    assert captured["body"]["subject"] == "Discovery: no sermon from westgate for Sunday 2026-10-25"
    assert captured["body"]["html"] == (
        "<ul><li>westgate: newest ledgered sermon is 2026-10-18 &lt;check the feed&gt;</li></ul>"
    )
    assert captured["headers"]["user-agent"] == notify.DEFAULT_USER_AGENT


def test_send_discovery_alert_raises_notify_error_on_http_error(monkeypatch):
    def fake_urlopen(request, timeout):
        raise urllib.error.HTTPError(request.full_url, 500, "boom", {}, io.BytesIO(b"down"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    with pytest.raises(notify.NotifyError):
        notify.send_discovery_alert("subject", ["line"], _config())
