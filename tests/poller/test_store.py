from poller import store
from poller.sources.base import SermonItem


def _item(guid: str = "g1", published_on: str = "2026-09-06") -> SermonItem:
    return SermonItem(
        guid=guid,
        title="Hear and Do",
        raw_title="Hear and Do - Luke",
        series="Luke",
        speaker="Jane Doe",
        published_on=published_on,
        published_at=None,
        episode_url="https://example.org/ep1",
        audio_url="https://example.org/ep1.mp3",
        blurb="A sermon on Luke.",
    )


def _record(guid: str, *, published_on: str | None, published_at: str | None = None) -> dict:
    item = _item(guid, published_on=published_on or "")
    return store.item_to_record(item, first_seen_at="t", published_at=published_at)


def test_load_returns_empty_dict_for_a_church_with_no_ledger_yet(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    assert store.load("menlo") == {}


def test_save_then_load_round_trips_a_record(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    record = store.item_to_record(_item(), first_seen_at="2026-09-06T12:00:00+00:00", published_at=None)
    store.save("menlo", {"g1": record})

    loaded = store.load("menlo")
    assert loaded == {"g1": record}
    assert loaded["g1"]["title"] == "Hear and Do"
    assert loaded["g1"]["speaker"] == "Jane Doe"
    assert loaded["g1"]["notified_at"] is None


def test_item_to_record_discards_empty_strings_as_placeholders():
    item = _item()
    item = item.__class__(**{**item.__dict__, "episode_url": "", "blurb": ""})
    record = store.item_to_record(item, first_seen_at="now", published_at=None)
    assert record["episode_url"] is None
    assert record["blurb"] is None


def test_item_to_record_treats_an_empty_title_as_null_like_episode_url_and_blurb():
    item = _item()
    item = item.__class__(**{**item.__dict__, "title": "", "raw_title": ""})
    record = store.item_to_record(item, first_seen_at="now", published_at=None)
    assert record["title"] is None
    assert record["raw_title"] is None


def test_save_is_idempotent_and_stable_on_key_order(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    record_b = store.item_to_record(_item("b"), first_seen_at="t1", published_at=None)
    record_a = store.item_to_record(_item("a"), first_seen_at="t2", published_at=None)
    store.save("menlo", {"b": record_b, "a": record_a})

    path = tmp_path / "menlo.json"
    first_write = path.read_text(encoding="utf-8")
    store.save("menlo", store.load("menlo"))
    assert path.read_text(encoding="utf-8") == first_write
    assert list(store.load("menlo").keys()) == ["a", "b"]


def test_save_orders_records_newest_published_on_first(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {
        "old": _record("old", published_on="2026-07-05"),
        "new": _record("new", published_on="2026-09-06"),
        "mid": _record("mid", published_on="2026-08-16"),
    }
    store.save("menlo", records)
    assert list(store.load("menlo").keys()) == ["new", "mid", "old"]


def test_save_breaks_same_date_ties_by_published_at_then_guid(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {
        "b": _record("b", published_on="2026-09-06", published_at="2026-09-06T10:00:00+00:00"),
        "a": _record("a", published_on="2026-09-06", published_at="2026-09-06T12:00:00+00:00"),
        "c": _record("c", published_on="2026-09-06", published_at=None),
    }
    store.save("menlo", records)
    # "a" has the later published_at; "c" has none and ties break by guid among
    # remaining same-date records without a published_at.
    assert list(store.load("menlo").keys()) == ["a", "b", "c"]


def test_save_sorts_records_missing_published_on_after_every_dated_record(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {
        "undated_b": _record("undated_b", published_on=None),
        "dated": _record("dated", published_on="2026-01-04"),
        "undated_a": _record("undated_a", published_on=None),
    }
    store.save("menlo", records)
    assert list(store.load("menlo").keys()) == ["dated", "undated_a", "undated_b"]


def test_item_to_record_defaults_transcription_fields_to_absent():
    record = store.item_to_record(_item(), first_seen_at="now", published_at=None)
    assert record["transcription_status"] is None
    assert record["transcribed_at"] is None
    assert record["transcript_hash"] is None
    assert record["content_path"] is None


def test_mark_transcribed_sets_exactly_its_own_fields():
    record = store.item_to_record(_item(), first_seen_at="now", published_at=None)
    store.mark_transcribed(
        record,
        content_path="transcripts/menlo/2026-09-06_hear-and-do_g1.txt",
        transcript_hash="deadbeef",
        transcribed_at="2026-09-06T12:00:00+00:00",
    )
    assert record["transcription_status"] == "done"
    assert record["content_path"] == "transcripts/menlo/2026-09-06_hear-and-do_g1.txt"
    assert record["transcript_hash"] == "deadbeef"
    assert record["transcribed_at"] == "2026-09-06T12:00:00+00:00"


def test_mark_transcription_failed_sets_only_the_status():
    record = store.item_to_record(_item(), first_seen_at="now", published_at=None)
    store.mark_transcription_failed(record)
    assert record["transcription_status"] == "failed"
    assert record["content_path"] is None
    assert record["transcript_hash"] is None


def test_refresh_record_updates_feed_sourced_fields():
    record = store.item_to_record(_item(), first_seen_at="now", published_at=None)
    rotated = _item(published_on="2026-09-06").__class__(
        **{
            **_item(published_on="2026-09-06").__dict__,
            "audio_url": "https://cdn.example.org/rotated.mp3",
            "title": "Hear and Do (rebroadcast)",
        }
    )
    store.refresh_record(record, rotated)
    assert record["audio_url"] == "https://cdn.example.org/rotated.mp3"
    assert record["title"] == "Hear and Do (rebroadcast)"


def test_refresh_record_never_touches_first_seen_at_published_at_or_progress_fields():
    record = store.item_to_record(
        _item(), first_seen_at="2026-01-01T00:00:00+00:00", published_at="2026-01-01T00:00:00+00:00"
    )
    record["notified_at"] = "2026-01-02T00:00:00+00:00"
    store.mark_transcribed(
        record,
        content_path="transcripts/x.txt",
        transcript_hash="abc",
        transcribed_at="2026-01-03T00:00:00+00:00",
    )

    store.refresh_record(record, _item(published_on="2026-09-06"))

    assert record["first_seen_at"] == "2026-01-01T00:00:00+00:00"
    assert record["published_at"] == "2026-01-01T00:00:00+00:00"
    assert record["notified_at"] == "2026-01-02T00:00:00+00:00"
    assert record["transcript_hash"] == "abc"
    assert record["content_path"] == "transcripts/x.txt"
    assert record["transcribed_at"] == "2026-01-03T00:00:00+00:00"


def test_refresh_record_freezes_title_once_transcription_is_done_but_not_other_fields():
    record = store.item_to_record(_item(), first_seen_at="now", published_at=None)
    store.mark_transcribed(
        record, content_path="transcripts/x.txt", transcript_hash="abc", transcribed_at="now"
    )

    renamed_and_rotated = _item(published_on="2026-09-06").__class__(
        **{
            **_item(published_on="2026-09-06").__dict__,
            "title": "A totally different title",
            "audio_url": "https://cdn.example.org/rotated.mp3",
        }
    )
    store.refresh_record(record, renamed_and_rotated)

    assert record["title"] == "Hear and Do"  # frozen once done
    assert record["audio_url"] == "https://cdn.example.org/rotated.mp3"  # still refreshes


def test_an_old_record_without_transcription_fields_round_trips_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    old_record = _record("g1", published_on="2026-01-04")
    for key in ("transcription_status", "transcribed_at", "transcript_hash", "content_path"):
        del old_record[key]

    store.save("menlo", {"g1": old_record})
    loaded = store.load("menlo")
    assert "transcription_status" not in loaded["g1"]
    assert loaded["g1"]["title"] == "Hear and Do"
