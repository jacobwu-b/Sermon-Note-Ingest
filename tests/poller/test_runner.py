import json
import os
import subprocess
from datetime import UTC, date, datetime, timedelta
from typing import ClassVar

from poller import batch_poll, config, ledger_push, notify, runner, store
from poller.sources.base import ExcludedItem, PollResult, SermonItem, SourceAdapter

_MARKED_README = "# Sermon ledger\n\n<!-- STATS:START -->\nplaceholder\n<!-- STATS:END -->\n"

# The developer's own git config (signing, hooks, default branch) must not reach these repos.
_GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "Ledger Writer",
    "GIT_AUTHOR_EMAIL": "writer@example.com",
    "GIT_COMMITTER_NAME": "Ledger Writer",
    "GIT_COMMITTER_EMAIL": "writer@example.com",
}


def _item(guid: str) -> SermonItem:
    return SermonItem(
        guid=guid,
        title="Hear and Do",
        raw_title="Hear and Do",
        series=None,
        speaker="Jane Doe",
        preached_on="2026-09-06",
        feed_published_at=None,
        episode_url="https://example.org/ep",
        audio_url="https://example.org/ep.mp3",
        blurb="",
    )


class _FakeAdapter(SourceAdapter):
    source = "fake"
    items: ClassVar[list[SermonItem]] = []
    excluded: ClassVar[list[ExcludedItem]] = []
    deferred = False
    # Older than any fixture date, so the polling tests here never trip a missing-Sunday
    # alert; the alert tests below use _AlertingAdapter.
    publish_grace = timedelta(days=3650)

    def poll(self) -> PollResult:
        if self.deferred:
            return PollResult(deferred=True)
        return PollResult(items=list(self.items), excluded=list(self.excluded))


def _church(*, notify_flag: bool) -> config.ChurchConfig:
    return config.ChurchConfig(
        name="fake", rss="https://example.org/feed.xml", enabled=True, notify=notify_flag
    )


def test_poll_church_stores_new_items_and_skips_when_none_are_new(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(runner, "ADAPTERS", {"fake": _FakeAdapter})
    _FakeAdapter.items = [_item("g1")]
    _FakeAdapter.deferred = False

    assert runner.poll_church("fake", _church(notify_flag=False), backfill=False) is True
    records = store.load("fake")
    assert set(records) == {"g1"}
    assert records["g1"]["first_seen_at"]

    # A re-poll of the same item must not duplicate or touch first_seen_at.
    first_seen = records["g1"]["first_seen_at"]
    assert runner.poll_church("fake", _church(notify_flag=False), backfill=False) is True
    assert store.load("fake")["g1"]["first_seen_at"] == first_seen


def test_poll_church_refreshes_a_rotated_audio_url_on_an_existing_guid(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(runner, "ADAPTERS", {"fake": _FakeAdapter})
    _FakeAdapter.items = [_item("g1")]
    _FakeAdapter.deferred = False

    assert runner.poll_church("fake", _church(notify_flag=False), backfill=False) is True
    assert store.load("fake")["g1"]["audio_url"] == "https://example.org/ep.mp3"

    rotated = _item("g1").__class__(
        **{**_item("g1").__dict__, "audio_url": "https://cdn.example.org/rotated.mp3"}
    )
    _FakeAdapter.items = [rotated]
    assert runner.poll_church("fake", _church(notify_flag=False), backfill=False) is True
    assert store.load("fake")["g1"]["audio_url"] == "https://cdn.example.org/rotated.mp3"


def test_poll_church_saves_a_refresh_even_when_nothing_new_is_discovered(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(runner, "ADAPTERS", {"fake": _FakeAdapter})
    _FakeAdapter.items = [_item("g1")]
    _FakeAdapter.deferred = False
    runner.poll_church("fake", _church(notify_flag=False), backfill=False)

    renamed = _item("g1").__class__(**{**_item("g1").__dict__, "title": "A new title"})
    _FakeAdapter.items = [renamed]
    assert runner.poll_church("fake", _church(notify_flag=False), backfill=False) is True
    assert store.load("fake")["g1"]["title"] == "A new title"


def test_poll_church_defers_without_writing_when_feed_unavailable(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(runner, "ADAPTERS", {"fake": _FakeAdapter})
    _FakeAdapter.items = []
    _FakeAdapter.deferred = True

    assert runner.poll_church("fake", _church(notify_flag=True), backfill=False) is True
    assert store.load("fake") == {}


def test_poll_church_sends_notification_for_new_items_when_notify_is_on(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(runner, "ADAPTERS", {"fake": _FakeAdapter})
    _FakeAdapter.items = [_item("g1")]
    _FakeAdapter.deferred = False
    monkeypatch.setenv("NOTIFY_EMAIL_FROM", "alerts@example.org")
    monkeypatch.setenv("NOTIFY_EMAIL_TO", "team@example.org")
    monkeypatch.setenv("RESEND_API_KEY", "re_123")

    sent = []
    monkeypatch.setattr(notify, "send_new_sermons", lambda church, items, cfg: sent.append((church, items)))

    assert runner.poll_church("fake", _church(notify_flag=True), backfill=False) is True
    assert sent == [("fake", [_item("g1")])]
    assert store.load("fake")["g1"]["notified_at"] is not None


def test_poll_church_does_not_notify_when_notify_is_off(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(runner, "ADAPTERS", {"fake": _FakeAdapter})
    _FakeAdapter.items = [_item("g1")]
    _FakeAdapter.deferred = False

    sent = []
    monkeypatch.setattr(notify, "send_new_sermons", lambda *a, **k: sent.append(1))

    assert runner.poll_church("fake", _church(notify_flag=False), backfill=False) is True
    assert sent == []
    assert store.load("fake")["g1"]["notified_at"] is None


def test_poll_church_backfill_seeds_ledger_without_notifying(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(runner, "ADAPTERS", {"fake": _FakeAdapter})
    _FakeAdapter.items = [_item("g1"), _item("g2")]
    _FakeAdapter.deferred = False

    sent = []
    monkeypatch.setattr(notify, "send_new_sermons", lambda *a, **k: sent.append(1))

    assert runner.poll_church("fake", _church(notify_flag=True), backfill=True) is True
    assert sent == []
    records = store.load("fake")
    assert set(records) == {"g1", "g2"}
    assert records["g1"]["notified_at"] is not None  # marked so a later normal run won't re-send


def test_poll_church_returns_false_when_notification_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(runner, "ADAPTERS", {"fake": _FakeAdapter})
    _FakeAdapter.items = [_item("g1")]
    _FakeAdapter.deferred = False
    monkeypatch.setenv("NOTIFY_EMAIL_FROM", "alerts@example.org")
    monkeypatch.setenv("NOTIFY_EMAIL_TO", "team@example.org")
    monkeypatch.setenv("RESEND_API_KEY", "re_123")

    def _boom(*a, **k):
        raise notify.NotifyError("resend is down")

    monkeypatch.setattr(notify, "send_new_sermons", _boom)

    assert runner.poll_church("fake", _church(notify_flag=True), backfill=False) is False
    # The ledger write still happened — discovery is never rolled back for a failed send.
    assert store.load("fake")["g1"]["notified_at"] is None


def _notify_setup(tmp_path, monkeypatch, items: list[SermonItem]) -> None:
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(runner, "ADAPTERS", {"fake": _FakeAdapter})
    _FakeAdapter.items = items
    _FakeAdapter.deferred = False
    monkeypatch.setenv("NOTIFY_EMAIL_FROM", "alerts@example.org")
    monkeypatch.setenv("NOTIFY_EMAIL_TO", "team@example.org")
    monkeypatch.setenv("RESEND_API_KEY", "re_123")


def test_poll_church_resends_a_failed_notification_on_the_next_poll(tmp_path, monkeypatch):
    _notify_setup(tmp_path, monkeypatch, [_item("g1")])

    def _boom(*a, **k):
        raise notify.NotifyError("Resend rejected the request: 503")

    monkeypatch.setattr(notify, "send_new_sermons", _boom)
    assert runner.poll_church("fake", _church(notify_flag=True), backfill=False) is False

    sent = []
    monkeypatch.setattr(
        notify, "send_new_sermons", lambda church, items, cfg: sent.append([i.guid for i in items])
    )
    assert runner.poll_church("fake", _church(notify_flag=True), backfill=False) is True
    assert sent == [["g1"]]
    assert store.load("fake")["g1"]["notified_at"] is not None

    # Once marked, a later poll sends nothing more.
    assert runner.poll_church("fake", _church(notify_flag=True), backfill=False) is True
    assert sent == [["g1"]]


def test_poll_church_sends_a_left_over_sermon_together_with_a_new_one(tmp_path, monkeypatch):
    _notify_setup(tmp_path, monkeypatch, [_item("g1")])
    runner.poll_church("fake", _church(notify_flag=False), backfill=False)  # g1 ledgered, never notified

    _FakeAdapter.items = [_item("g1"), _item("g2")]
    sent = []
    monkeypatch.setattr(
        notify, "send_new_sermons", lambda church, items, cfg: sent.append({i.guid for i in items})
    )
    assert runner.poll_church("fake", _church(notify_flag=True), backfill=False) is True
    assert sent == [{"g1", "g2"}]


def test_poll_church_never_notifies_a_sermon_first_seen_outside_the_window(tmp_path, monkeypatch):
    _notify_setup(tmp_path, monkeypatch, [_item("old"), _item("recent")])
    first_seen = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
    _at(monkeypatch, first_seen)
    runner.poll_church("fake", _church(notify_flag=False), backfill=False)
    records = store.load("fake")
    records["recent"]["first_seen_at"] = (first_seen + timedelta(days=2)).isoformat()
    store.save("fake", records)

    sent = []
    monkeypatch.setattr(
        notify, "send_new_sermons", lambda church, items, cfg: sent.append([i.guid for i in items])
    )
    _at(monkeypatch, first_seen + runner.NOTIFY_WINDOW + timedelta(days=1))
    assert runner.poll_church("fake", _church(notify_flag=True), backfill=False) is True

    assert sent == [["recent"]]
    assert store.load("fake")["old"]["notified_at"] is None


def test_run_skips_disabled_and_unselected_churches(tmp_path, monkeypatch, church_table):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    (tmp_path / "README.md").write_text(_MARKED_README, encoding="utf-8")
    monkeypatch.setattr(runner, "ADAPTERS", {"fake": _FakeAdapter})
    _FakeAdapter.items = [_item("g1")]
    _FakeAdapter.deferred = False
    raw = (
        '{"fake": {"rss": "https://example.org/feed.xml", "ingest": {"enabled": true, "notify": false}},'
        ' "off": {"rss": "https://example.org/other.xml", "ingest": {"enabled": false, "notify": false}}}'
    )
    church_table(raw)

    assert runner.run(church_names=None, backfill=False) is True
    assert set(store.load("fake")) == {"g1"}
    assert store.load("off") == {}


def test_a_poll_that_discovers_a_sermon_stages_every_data_file_it_wrote(tmp_path, monkeypatch, church_table):
    # The poll job uploads whatever it changed under data/ and the stage refuses any path outside
    # the ledger allowlist (ADR-0027), so a file the poll writes that the allowlist lacks fails
    # every run that discovers something, and that discovery is never recorded.
    repo = tmp_path / "repo"
    data = repo / "data"
    data.mkdir(parents=True)
    (data / "README.md").write_text(_MARKED_README, encoding="utf-8")
    (data / "fake.json").write_text("{}\n", encoding="utf-8")
    for args in (["init", "-q", "-b", "main"], ["add", "data"], ["commit", "-q", "-m", "seed ledger"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True, env=_GIT_ENV)
    monkeypatch.setattr(store, "DATA_DIR", data)
    monkeypatch.setattr(runner, "ADAPTERS", {"fake": _FakeAdapter})
    _FakeAdapter.items = [_item("g1")]
    _FakeAdapter.deferred = False
    church_table(
        '{"fake": {"rss": "https://example.org/feed.xml", "ingest": {"enabled": true, "notify": false}}}'
    )

    assert runner.run(church_names=None, backfill=False) is True

    assert ledger_push.stage(repo, tmp_path / "ledger") == ["data/fake.json"]
    assert (data / "README.md").read_text(encoding="utf-8") == _MARKED_README


def test_poll_church_records_discovery_when_new_items_are_found(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(runner, "ADAPTERS", {"fake": _FakeAdapter})
    _FakeAdapter.items = [_item("g1")]
    _FakeAdapter.deferred = False

    discovered: list[str] = []
    runner.poll_church("fake", _church(notify_flag=False), backfill=False, discovered_out=discovered)
    assert discovered == ["fake"]


def test_poll_church_does_not_record_discovery_when_nothing_new(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(runner, "ADAPTERS", {"fake": _FakeAdapter})
    _FakeAdapter.items = [_item("g1")]
    _FakeAdapter.deferred = False
    runner.poll_church("fake", _church(notify_flag=False), backfill=False)  # seeds the ledger

    discovered: list[str] = []
    runner.poll_church("fake", _church(notify_flag=False), backfill=False, discovered_out=discovered)
    assert discovered == []


def test_poll_church_records_discovery_on_backfill_too(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(runner, "ADAPTERS", {"fake": _FakeAdapter})
    _FakeAdapter.items = [_item("g1")]
    _FakeAdapter.deferred = False

    discovered: list[str] = []
    runner.poll_church("fake", _church(notify_flag=False), backfill=True, discovered_out=discovered)
    assert discovered == ["fake"]


def test_poll_church_does_not_record_discovery_when_deferred(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(runner, "ADAPTERS", {"fake": _FakeAdapter})
    _FakeAdapter.items = [_item("g1")]
    _FakeAdapter.deferred = True

    discovered: list[str] = []
    runner.poll_church("fake", _church(notify_flag=False), backfill=False, discovered_out=discovered)
    assert discovered == []


def test_run_aggregates_discovered_churches_across_the_selection(tmp_path, monkeypatch, church_table):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    (tmp_path / "README.md").write_text(_MARKED_README, encoding="utf-8")
    monkeypatch.setattr(runner, "ADAPTERS", {"fake": _FakeAdapter})
    _FakeAdapter.items = [_item("g1")]
    _FakeAdapter.deferred = False
    church_table(
        '{"fake": {"rss": "https://example.org/feed.xml", "ingest": {"enabled": true, "notify": false}}}'
    )

    discovered: list[str] = []
    assert runner.run(church_names=None, backfill=False, discovered_out=discovered) is True
    assert discovered == ["fake"]


def test_main_prints_discovered_churches_when_requested(tmp_path, monkeypatch, capsys, church_table):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    (tmp_path / "README.md").write_text(_MARKED_README, encoding="utf-8")
    monkeypatch.setattr(runner, "ADAPTERS", {"fake": _FakeAdapter})
    _FakeAdapter.items = [_item("g1")]
    _FakeAdapter.deferred = False
    church_table(
        '{"fake": {"rss": "https://example.org/feed.xml", "ingest": {"enabled": true, "notify": false}}}'
    )

    exit_code = runner.main(["--print-discovered"])
    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "fake"


def test_main_prints_nothing_when_no_churches_were_discovered(tmp_path, monkeypatch, capsys, church_table):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    (tmp_path / "README.md").write_text(_MARKED_README, encoding="utf-8")
    monkeypatch.setattr(runner, "ADAPTERS", {"fake": _FakeAdapter})
    _FakeAdapter.items = []
    _FakeAdapter.deferred = False
    church_table(
        '{"fake": {"rss": "https://example.org/feed.xml", "ingest": {"enabled": true, "notify": false}}}'
    )

    exit_code = runner.main(["--print-discovered"])
    assert exit_code == 0
    assert capsys.readouterr().out.strip() == ""


def test_run_polls_pending_claude_batches_once(tmp_path, monkeypatch, church_table):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    (tmp_path / "README.md").write_text(_MARKED_README, encoding="utf-8")
    monkeypatch.setattr(runner, "ADAPTERS", {"fake": _FakeAdapter})
    _FakeAdapter.items = []
    _FakeAdapter.deferred = False
    church_table(
        '{"fake": {"rss": "https://example.org/feed.xml", "ingest": {"enabled": true, "notify": false}}}'
    )

    calls = []
    monkeypatch.setattr(batch_poll, "poll_pending_batches", lambda: calls.append(1))

    assert runner.run(church_names=None, backfill=False) is True
    assert calls == [1]


def test_run_survives_an_unexpected_batch_poll_crash(tmp_path, monkeypatch, church_table):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    (tmp_path / "README.md").write_text(_MARKED_README, encoding="utf-8")
    monkeypatch.setattr(runner, "ADAPTERS", {"fake": _FakeAdapter})
    _FakeAdapter.items = []
    _FakeAdapter.deferred = False
    church_table(
        '{"fake": {"rss": "https://example.org/feed.xml", "ingest": {"enabled": true, "notify": false}}}'
    )

    def _boom():
        raise RuntimeError("unexpected crash")

    monkeypatch.setattr(batch_poll, "poll_pending_batches", _boom)

    assert runner.run(church_names=None, backfill=False) is True


# --- discovery absence alerts (spec 0011, ADR-0025) ---

# Thursday after Westgate's 2026-10-25 sermon was dropped.
_THU = datetime(2026, 10, 29, 15, 0, tzinfo=UTC)
_ONE_CHURCH = (
    '{"fake": {"rss": "https://example.org/feed.xml", "ingest": {"enabled": true, "notify": false}}}'
)


class _AlertingAdapter(_FakeAdapter):
    publish_grace = timedelta(hours=48)


def _dated(guid: str, preached_on: str) -> SermonItem:
    return _item(guid).__class__(**{**_item(guid).__dict__, "preached_on": preached_on})


def _alert_setup(tmp_path, monkeypatch, church_table, *, items, deferred=False, excluded=()):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    (tmp_path / "README.md").write_text(_MARKED_README, encoding="utf-8")
    monkeypatch.setattr(runner, "ADAPTERS", {"fake": _AlertingAdapter})
    monkeypatch.setattr(batch_poll, "poll_pending_batches", lambda: None)
    _AlertingAdapter.items = items
    _AlertingAdapter.deferred = deferred
    _AlertingAdapter.excluded = list(excluded)
    church_table(_ONE_CHURCH)
    monkeypatch.setenv("NOTIFY_EMAIL_FROM", "alerts@example.org")
    monkeypatch.setenv("NOTIFY_EMAIL_TO", "team@example.org")
    monkeypatch.setenv("RESEND_API_KEY", "re_123")
    sent: list[tuple[str, list[str]]] = []
    monkeypatch.setattr(
        notify, "send_discovery_alert", lambda subject, lines, cfg: sent.append((subject, list(lines)))
    )
    return sent


def _at(monkeypatch, when: datetime) -> None:
    monkeypatch.setattr(runner, "now", lambda: when.isoformat())


def test_run_alerts_once_for_a_missing_sunday_and_fails_only_that_run(tmp_path, monkeypatch, church_table):
    sent = _alert_setup(
        tmp_path,
        monkeypatch,
        church_table,
        items=[_dated("g1", "2026-10-18")],
        excluded=[ExcludedItem("NextGen Sunday | Light in the Dark | October 18, 2026", date(2026, 10, 18))],
    )

    _at(monkeypatch, _THU)
    assert runner.run(church_names=None, backfill=False) is False
    assert len(sent) == 1
    assert "fake" in sent[0][0]
    assert "2026-10-25" in sent[0][0] + " ".join(sent[0][1])

    _at(monkeypatch, _THU + timedelta(minutes=5))
    assert runner.run(church_names=None, backfill=False) is True
    assert len(sent) == 1


def test_run_names_the_excluded_item_that_was_likely_the_missing_sermon(tmp_path, monkeypatch, church_table):
    dropped = "The Gospel of John (Part 3) |    I am the Light of the World |October 25 2026"
    sent = _alert_setup(
        tmp_path,
        monkeypatch,
        church_table,
        items=[_dated("g1", "2026-10-18")],
        excluded=[ExcludedItem(dropped, date(2026, 10, 26))],
    )
    _at(monkeypatch, _THU)

    assert runner.run(church_names=None, backfill=False) is False
    assert len(sent) == 1
    assert f"2026-10-26 — {dropped}" in sent[0][1]


def test_run_alerts_for_a_missing_sunday_despite_a_future_dated_record(tmp_path, monkeypatch, church_table):
    sent = _alert_setup(
        tmp_path,
        monkeypatch,
        church_table,
        items=[_dated("g1", "2026-10-18"), _dated("g2", "2027-10-24")],
    )
    _at(monkeypatch, _THU)

    assert runner.run(church_names=None, backfill=False) is False
    assert len(sent) == 1
    assert "2026-10-25" in sent[0][0] + " ".join(sent[0][1])


def test_run_stays_green_when_the_due_sunday_is_ledgered(tmp_path, monkeypatch, church_table):
    sent = _alert_setup(tmp_path, monkeypatch, church_table, items=[_dated("g1", "2026-10-25")])
    _at(monkeypatch, _THU)

    assert runner.run(church_names=None, backfill=False) is True
    assert sent == []
    # Nothing changed, so nothing is written for poll.yml to commit.
    assert not (tmp_path / "alerts" / "state.json").exists()


def test_run_retries_an_alert_whose_send_failed(tmp_path, monkeypatch, church_table):
    sent = _alert_setup(tmp_path, monkeypatch, church_table, items=[_dated("g1", "2026-10-18")])

    def _rejected(subject, lines, cfg):
        raise notify.NotifyError("Resend rejected the request: 500")

    monkeypatch.setattr(notify, "send_discovery_alert", _rejected)
    _at(monkeypatch, _THU)
    assert runner.run(church_names=None, backfill=False) is False

    monkeypatch.setattr(
        notify, "send_discovery_alert", lambda subject, lines, cfg: sent.append((subject, lines))
    )
    _at(monkeypatch, _THU + timedelta(minutes=5))
    assert runner.run(church_names=None, backfill=False) is False
    assert len(sent) == 1

    _at(monkeypatch, _THU + timedelta(minutes=10))
    assert runner.run(church_names=None, backfill=False) is True


def test_run_fails_and_keeps_the_alert_pending_when_notify_secrets_are_missing(
    tmp_path, monkeypatch, church_table
):
    sent = _alert_setup(tmp_path, monkeypatch, church_table, items=[_dated("g1", "2026-10-18")])
    monkeypatch.delenv("RESEND_API_KEY")
    _at(monkeypatch, _THU)

    assert runner.run(church_names=None, backfill=False) is False
    assert sent == []

    monkeypatch.setenv("RESEND_API_KEY", "re_123")
    _at(monkeypatch, _THU + timedelta(minutes=5))
    assert runner.run(church_names=None, backfill=False) is False
    assert len(sent) == 1


def test_run_honors_a_snooze_through_the_due_sunday(tmp_path, monkeypatch, church_table):
    sent = _alert_setup(tmp_path, monkeypatch, church_table, items=[_dated("g1", "2026-10-18")])
    (tmp_path / "alerts").mkdir()
    (tmp_path / "alerts" / "snooze.json").write_text(
        json.dumps({"fake": {"until": "2026-11-01", "_reason": "fall break"}}), encoding="utf-8"
    )
    _at(monkeypatch, _THU)

    assert runner.run(church_names=None, backfill=False) is True
    assert sent == []


def test_run_alerts_once_a_feed_has_deferred_for_six_hours(tmp_path, monkeypatch, church_table):
    sent = _alert_setup(tmp_path, monkeypatch, church_table, items=[], deferred=True)

    _at(monkeypatch, _THU)
    assert runner.run(church_names=None, backfill=False) is True
    _at(monkeypatch, _THU + timedelta(hours=6))
    assert runner.run(church_names=None, backfill=False) is False
    assert len(sent) == 1
    _at(monkeypatch, _THU + timedelta(hours=7))
    assert runner.run(church_names=None, backfill=False) is True
    assert len(sent) == 1


def test_run_alerts_once_a_feed_has_parsed_empty_for_six_hours(tmp_path, monkeypatch, church_table):
    sent = _alert_setup(tmp_path, monkeypatch, church_table, items=[])

    _at(monkeypatch, _THU)
    assert runner.run(church_names=None, backfill=False) is True
    _at(monkeypatch, _THU + timedelta(hours=6))
    assert runner.run(church_names=None, backfill=False) is False
    assert len(sent) == 1


def test_run_alert_state_lands_under_data_for_poll_yml_to_commit(tmp_path, monkeypatch, church_table):
    _alert_setup(tmp_path, monkeypatch, church_table, items=[], deferred=True)
    _at(monkeypatch, _THU)
    runner.run(church_names=None, backfill=False)

    state = json.loads((tmp_path / "alerts" / "state.json").read_text(encoding="utf-8"))
    assert state == {"fake": {"failing_since": _THU.isoformat()}}
    assert store.validate_all() == []


def test_poll_church_reports_feed_health(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(runner, "ADAPTERS", {"fake": _FakeAdapter})
    cases = [
        ("deferred", [], [], True, False),
        ("parsed empty", [], [], False, False),
        ("only excluded items", [], [ExcludedItem("Untitled upload", None)], False, True),
        ("sermons", [_item("g1")], [], False, True),
    ]
    for label, items, excluded, deferred, healthy in cases:
        _FakeAdapter.items, _FakeAdapter.excluded, _FakeAdapter.deferred = items, excluded, deferred
        health: dict[str, bool] = {}
        assert (
            runner.poll_church("fake", _church(notify_flag=False), backfill=False, health_out=health) is True
        )
        assert health == {"fake": healthy}, label
    _FakeAdapter.excluded = []

    assert "parsed to no entries" in caplog.text
