"""``poller.verify_content``: the ledger ↔ Content invariant after a re-transcription run (ADR-0022).

Synthetic text only; real transcripts never enter this repo (ADR-0004).
"""

import logging
from datetime import UTC, datetime

import pytest

from poller import store, transcribe, verify_content
from poller.sources.base import SermonItem

_PROFILE = {
    "model": "large-v3",
    "compute_type": "int8",
    "beam_size": 5,
    "condition_on_previous_text": False,
    "hallucination_silence_threshold": 2.0,
    "vad_filter": True,
    "domain_prompt": False,
    "hotwords": None,
    "faster_whisper_version": "1.2.1",
    "ctranslate2_version": "4.8.2",
}
_STALE_PROFILE = {**_PROFILE, "condition_on_previous_text": True}
_TEXT = "Verse 1 reminds us that the gospel of grace reaches people we did not expect. Zebulon marmalade."
_NOW = datetime(2026, 10, 14, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def env(monkeypatch, tmp_path, church_table):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    for key, value in {
        "WHISPER_MODEL": "large-v3",
        "WHISPER_CONDITION_ON_PREVIOUS_TEXT": "false",
        "WHISPER_DOMAIN_PROMPT": "false",
        "WHISPER_HALLUCINATION_SILENCE_THRESHOLD": "2.0",
    }.items():
        monkeypatch.setenv(key, value)
    church_table(
        '{"menlo": {"rss": "https://example.org/menlo.xml", "ingest": {"enabled": true}},'
        ' "pbc": {"rss": "https://example.org/pbc.xml", "ingest": {"enabled": false}}}'
    )


def _done(guid: str, *, text: str = _TEXT, profile: dict = _PROFILE, church: str = "menlo") -> dict:
    record = store.item_to_record(
        SermonItem(
            guid=guid,
            title=f"Sermon {guid}",
            raw_title=None,
            series=None,
            speaker=None,
            preached_on="2026-09-06",
            feed_published_at=None,
            episode_url=None,
            audio_url="https://example.org/a.mp3",
            blurb=None,
        ),
        first_seen_at="t",
        feed_published_at=None,
    )
    store.mark_transcribed(
        record,
        content_path=f"transcripts/{church}/2026-09-06_sermon_{guid}.txt",
        transcript_hash=transcribe.transcript_hash(text),
        transcribed_at="2026-09-20T12:00:00+00:00",
        profile=profile,
    )
    return record


def _ledgers_and_content(*records: dict, church: str = "menlo") -> tuple[dict, dict]:
    ledger = {r["guid"]: r for r in records}
    content = {r["content_path"]: _TEXT for r in records}
    return {church: ledger}, content


def test_a_ledger_that_matches_content_has_no_violations():
    ledgers, content = _ledgers_and_content(_done("a"), _done("b"))
    assert verify_content.violations(ledgers, content) == []


def test_a_done_record_whose_content_file_is_missing_is_a_violation():
    ledgers, _content = _ledgers_and_content(_done("a"))
    [violation] = verify_content.violations(ledgers, {})
    assert "menlo/a" in violation
    assert "missing" in violation


def test_a_done_record_whose_content_text_hashes_differently_is_a_violation():
    record = _done("a")
    ledgers = {"menlo": {"a": record}}
    [violation] = verify_content.violations(ledgers, {record["content_path"]: "edited by hand"})
    assert "menlo/a" in violation
    assert record["transcript_hash"][:12] in violation


def test_a_replaced_record_missing_a_profile_field_is_a_violation():
    record = _done("a")
    record["superseded_transcript_hash"] = "oldhash"
    record["transcription_beam_size"] = None
    ledgers = {"menlo": {"a": record}}
    [violation] = verify_content.violations(ledgers, {record["content_path"]: _TEXT})
    assert "beam_size" in violation


def test_a_replaced_record_may_record_that_no_hotwords_were_sent():
    record = _done("a")
    record["superseded_transcript_hash"] = "oldhash"
    assert verify_content.violations({"menlo": {"a": record}}, {record["content_path"]: _TEXT}) == []


def test_two_content_files_for_one_guid_are_a_violation():
    record = _done("menlo_x")
    ledgers = {"menlo": {"menlo_x": record}}
    renamed = "transcripts/menlo/2026-09-06_a-renamed-title_menlo_x.txt"
    other_guid = "transcripts/menlo/2026-09-06_another_x.txt"
    [violation] = verify_content.violations(
        ledgers, {record["content_path"]: _TEXT, renamed: _TEXT, other_guid: "different sermon"}
    )
    assert renamed in violation
    assert record["content_path"] in violation
    assert other_guid not in violation


def test_no_violation_names_any_transcript_text():
    record = _done("a")
    found = verify_content.violations(
        {"menlo": {"a": record}}, {record["content_path"]: "Zebulon marmalade edit"}
    )
    assert "zebulon" not in " ".join(found).lower()


def test_backlog_counts_stale_in_scope_failed_and_replaced_per_church():
    stale = _done("stale", profile=_STALE_PROFILE)
    refused = _done("refused", profile=_STALE_PROFILE)
    running = verify_content.transcriber._running_profile()
    store.mark_retranscribe_failed(
        refused, error=store.QUALITY_REGRESSION_ERROR, failed_profile=store.profile_digest(running)
    )
    replaced = _done("replaced")
    replaced["superseded_transcript_hash"] = "oldhash"
    replaced["transcribed_at"] = "2026-10-14T09:00:00+00:00"
    store.save("menlo", {r["guid"]: r for r in (stale, refused, replaced)})

    [row] = [r for r in verify_content.backlog(["menlo"]) if r.church == "menlo"]

    assert (row.stale, row.in_scope, row.replaced) == (2, 1, 1)
    assert row.failed == {store.QUALITY_REGRESSION_ERROR: 1}
    assert row.last_replaced_at == "2026-10-14T09:00:00+00:00"


class _Alerts:
    def __init__(self):
        self.sent: list[tuple[str, list[str]]] = []

    def __call__(self, subject, lines):
        self.sent.append((subject, list(lines)))


def _main(*args: str, content: dict, alerts: _Alerts, tmp_path=None) -> int:
    return verify_content.main(list(args), fetch_all=lambda: content, send_alert=alerts, now=lambda: _NOW)


def test_main_passes_and_sends_nothing_when_the_ledger_matches_content():
    record = _done("a")
    store.save("menlo", {"a": record})
    alerts = _Alerts()
    assert _main(content={record["content_path"]: _TEXT}, alerts=alerts) == 0
    assert alerts.sent == []


def test_main_fails_the_run_and_emails_every_violation(caplog):
    record = _done("a")
    store.save("menlo", {"a": record})
    alerts = _Alerts()

    with caplog.at_level(logging.INFO, logger="poller"):
        assert _main(content={record["content_path"]: "Zebulon marmalade edit"}, alerts=alerts) == 1

    [(subject, lines)] = alerts.sent
    assert "1 violation" in subject
    assert "menlo/a" in lines[0]
    assert "zebulon" not in caplog.text.lower()


def test_main_still_fails_when_the_violation_alert_cannot_be_sent():
    record = _done("a")
    store.save("menlo", {"a": record})

    def broken(subject, lines):
        raise verify_content.notify.NotifyError("Resend down")

    assert verify_content.main([], fetch_all=dict, send_alert=broken, now=lambda: _NOW) == 1


def test_main_fails_when_content_cannot_be_read():
    def unreadable():
        raise verify_content.content_repo.ContentPublishError("content read failed after 3 attempts")

    assert verify_content.main([], fetch_all=unreadable, send_alert=_Alerts(), now=lambda: _NOW) == 1


def test_a_scheduled_pass_with_nothing_replaced_in_24_hours_sends_a_dead_man_alert():
    stale = _done("stale", profile=_STALE_PROFILE)
    replaced = _done("replaced")
    replaced["superseded_transcript_hash"] = "oldhash"
    replaced["transcribed_at"] = "2026-10-13T11:00:00+00:00"
    store.save("menlo", {"stale": stale, "replaced": replaced})
    content = {stale["content_path"]: _TEXT, replaced["content_path"]: _TEXT}
    alerts = _Alerts()

    assert _main("--scheduled", "--planned", "1", content=content, alerts=alerts) == 0

    [(subject, lines)] = alerts.sent
    assert "no transcript replaced in 24 hours" in subject
    assert any("1 stale sermon(s) still in scope" in line for line in lines)


def test_a_scheduled_pass_that_replaced_something_recently_sends_nothing():
    stale = _done("stale", profile=_STALE_PROFILE)
    replaced = _done("replaced")
    replaced["superseded_transcript_hash"] = "oldhash"
    replaced["transcribed_at"] = "2026-10-14T11:00:00+00:00"
    store.save("menlo", {"stale": stale, "replaced": replaced})
    alerts = _Alerts()

    _main(
        "--scheduled",
        "--planned",
        "1",
        content={r["content_path"]: _TEXT for r in (stale, replaced)},
        alerts=alerts,
    )

    assert alerts.sent == []


def test_a_dispatched_run_never_sends_the_dead_man_alert():
    stale = _done("stale", profile=_STALE_PROFILE)
    store.save("menlo", {"stale": stale})
    alerts = _Alerts()
    _main("--planned", "1", content={stale["content_path"]: _TEXT}, alerts=alerts)
    assert alerts.sent == []


def test_the_scheduled_run_that_empties_the_backlog_sends_one_completion_email():
    refused = _done("refused", profile=_STALE_PROFILE)
    running = verify_content.transcriber._running_profile()
    store.mark_retranscribe_failed(
        refused, error=store.QUALITY_REGRESSION_ERROR, failed_profile=store.profile_digest(running)
    )
    store.save("menlo", {"refused": refused})
    content = {refused["content_path"]: _TEXT}

    finishing, idle = _Alerts(), _Alerts()
    _main("--scheduled", "--planned", "3", content=content, alerts=finishing)
    _main("--scheduled", "--planned", "0", content=content, alerts=idle)

    [(subject, lines)] = finishing.sent
    assert "complete" in subject
    assert any("1 stale sermon(s) left for review" in line for line in lines)
    assert idle.sent == []


def test_main_writes_the_backlog_and_violation_count_to_the_summary(tmp_path):
    stale = _done("stale", profile=_STALE_PROFILE)
    store.save("menlo", {"stale": stale})
    summary = tmp_path / "summary.md"

    verify_content.main(
        ["--summary-md", str(summary)],
        fetch_all=lambda: {stale["content_path"]: _TEXT},
        send_alert=_Alerts(),
        now=lambda: _NOW,
    )

    text = summary.read_text()
    assert "| menlo | 1 | 1 | 0 |" in text
    assert "0 violations" in text
