from typing import ClassVar

from poller import config, notify, runner, stats, store
from poller.sources.base import PollResult, SermonItem, SourceAdapter

_MARKED_README = "# Sermon ledger\n\n<!-- STATS:START -->\nplaceholder\n<!-- STATS:END -->\n"


def _item(guid: str) -> SermonItem:
    return SermonItem(
        guid=guid,
        title="Hear and Do",
        raw_title="Hear and Do",
        series=None,
        speaker="Jane Doe",
        published_on="2026-09-06",
        published_at=None,
        episode_url="https://example.org/ep",
        audio_url="https://example.org/ep.mp3",
        blurb="",
    )


class _FakeAdapter(SourceAdapter):
    source = "fake"
    items: ClassVar[list[SermonItem]] = []
    deferred = False

    def poll(self) -> PollResult:
        if self.deferred:
            return PollResult(deferred=True)
        return PollResult(items=list(self.items), excluded=0)


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


def test_run_skips_disabled_and_unselected_churches(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    (tmp_path / "README.md").write_text(_MARKED_README, encoding="utf-8")
    monkeypatch.setattr(runner, "ADAPTERS", {"fake": _FakeAdapter})
    _FakeAdapter.items = [_item("g1")]
    _FakeAdapter.deferred = False
    raw = (
        '{"fake": {"rss": "https://example.org/feed.xml", "enabled": true, "notify": false},'
        ' "off": {"rss": "https://example.org/other.xml", "enabled": false, "notify": false}}'
    )
    monkeypatch.setenv("CHURCHES", raw)

    assert runner.run(church_names=None, backfill=False) is True
    assert set(store.load("fake")) == {"g1"}
    assert store.load("off") == {}


def test_run_regenerates_readme_stats_even_when_nothing_new_is_found(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    (tmp_path / "README.md").write_text(_MARKED_README, encoding="utf-8")
    monkeypatch.setattr(runner, "ADAPTERS", {"fake": _FakeAdapter})
    _FakeAdapter.items = []
    _FakeAdapter.deferred = False
    monkeypatch.setenv(
        "CHURCHES", '{"fake": {"rss": "https://example.org/feed.xml", "enabled": true, "notify": false}}'
    )

    assert runner.run(church_names=None, backfill=False) is True
    assert "placeholder" not in (tmp_path / "README.md").read_text(encoding="utf-8")


def test_run_reports_failure_when_stats_regeneration_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    (tmp_path / "README.md").write_text(_MARKED_README, encoding="utf-8")
    monkeypatch.setattr(runner, "ADAPTERS", {"fake": _FakeAdapter})
    _FakeAdapter.items = [_item("g1")]
    _FakeAdapter.deferred = False
    monkeypatch.setenv(
        "CHURCHES", '{"fake": {"rss": "https://example.org/feed.xml", "enabled": true, "notify": false}}'
    )

    def _boom():
        raise stats.StatsError("markers missing")

    monkeypatch.setattr(stats, "regenerate", _boom)

    # The ledger write itself still succeeded; only the stats step failed.
    assert runner.run(church_names=None, backfill=False) is False
    assert set(store.load("fake")) == {"g1"}


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


def test_run_aggregates_discovered_churches_across_the_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    (tmp_path / "README.md").write_text(_MARKED_README, encoding="utf-8")
    monkeypatch.setattr(runner, "ADAPTERS", {"fake": _FakeAdapter})
    _FakeAdapter.items = [_item("g1")]
    _FakeAdapter.deferred = False
    monkeypatch.setenv(
        "CHURCHES", '{"fake": {"rss": "https://example.org/feed.xml", "enabled": true, "notify": false}}'
    )

    discovered: list[str] = []
    assert runner.run(church_names=None, backfill=False, discovered_out=discovered) is True
    assert discovered == ["fake"]


def test_main_prints_discovered_churches_when_requested(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    (tmp_path / "README.md").write_text(_MARKED_README, encoding="utf-8")
    monkeypatch.setattr(runner, "ADAPTERS", {"fake": _FakeAdapter})
    _FakeAdapter.items = [_item("g1")]
    _FakeAdapter.deferred = False
    monkeypatch.setenv(
        "CHURCHES", '{"fake": {"rss": "https://example.org/feed.xml", "enabled": true, "notify": false}}'
    )

    exit_code = runner.main(["--print-discovered"])
    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "fake"


def test_main_prints_nothing_when_no_churches_were_discovered(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    (tmp_path / "README.md").write_text(_MARKED_README, encoding="utf-8")
    monkeypatch.setattr(runner, "ADAPTERS", {"fake": _FakeAdapter})
    _FakeAdapter.items = []
    _FakeAdapter.deferred = False
    monkeypatch.setenv(
        "CHURCHES", '{"fake": {"rss": "https://example.org/feed.xml", "enabled": true, "notify": false}}'
    )

    exit_code = runner.main(["--print-discovered"])
    assert exit_code == 0
    assert capsys.readouterr().out.strip() == ""
