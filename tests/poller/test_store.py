import json

import pytest

from poller import store
from poller.sources.base import SermonItem


def _item(guid: str = "g1", preached_on: str = "2026-09-06") -> SermonItem:
    return SermonItem(
        guid=guid,
        title="Hear and Do",
        raw_title="Hear and Do - Luke",
        series="Luke",
        speaker="Jane Doe",
        preached_on=preached_on,
        feed_published_at=None,
        episode_url="https://example.org/ep1",
        audio_url="https://example.org/ep1.mp3",
        blurb="A sermon on Luke.",
    )


def _record(guid: str, *, preached_on: str | None, feed_published_at: str | None = None) -> dict:
    item = _item(guid, preached_on=preached_on or "")
    return store.item_to_record(item, first_seen_at="t", feed_published_at=feed_published_at)


def test_load_returns_empty_dict_for_a_church_with_no_ledger_yet(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    assert store.load("menlo") == {}


def test_save_then_load_round_trips_a_record(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    record = store.item_to_record(_item(), first_seen_at="2026-09-06T12:00:00+00:00", feed_published_at=None)
    store.save("menlo", {"g1": record})

    loaded = store.load("menlo")
    assert loaded == {"g1": record}
    assert loaded["g1"]["title"] == "Hear and Do"
    assert loaded["g1"]["speaker"] == "Jane Doe"
    assert loaded["g1"]["notified_at"] is None


def test_item_to_record_discards_empty_strings_as_placeholders():
    item = _item()
    item = item.__class__(**{**item.__dict__, "episode_url": "", "blurb": ""})
    record = store.item_to_record(item, first_seen_at="now", feed_published_at=None)
    assert record["episode_url"] is None
    assert record["blurb"] is None


def test_item_to_record_treats_an_empty_title_as_null_like_episode_url_and_blurb():
    item = _item()
    item = item.__class__(**{**item.__dict__, "title": "", "raw_title": ""})
    record = store.item_to_record(item, first_seen_at="now", feed_published_at=None)
    assert record["title"] is None
    assert record["raw_title"] is None


def test_save_is_idempotent_and_stable_on_key_order(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    record_b = store.item_to_record(_item("b"), first_seen_at="t1", feed_published_at=None)
    record_a = store.item_to_record(_item("a"), first_seen_at="t2", feed_published_at=None)
    store.save("menlo", {"b": record_b, "a": record_a})

    path = tmp_path / "menlo.json"
    first_write = path.read_text(encoding="utf-8")
    store.save("menlo", store.load("menlo"))
    assert path.read_text(encoding="utf-8") == first_write
    assert list(store.load("menlo").keys()) == ["a", "b"]


def test_save_orders_records_newest_preached_on_first(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {
        "old": _record("old", preached_on="2026-07-05"),
        "new": _record("new", preached_on="2026-09-06"),
        "mid": _record("mid", preached_on="2026-08-16"),
    }
    store.save("menlo", records)
    assert list(store.load("menlo").keys()) == ["new", "mid", "old"]


def test_save_breaks_same_date_ties_by_feed_published_at_then_guid(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {
        "b": _record("b", preached_on="2026-09-06", feed_published_at="2026-09-06T10:00:00+00:00"),
        "a": _record("a", preached_on="2026-09-06", feed_published_at="2026-09-06T12:00:00+00:00"),
        "c": _record("c", preached_on="2026-09-06", feed_published_at=None),
    }
    store.save("menlo", records)
    # "a" has the later feed_published_at; "c" has none and ties break by guid among
    # remaining same-date records without a feed_published_at.
    assert list(store.load("menlo").keys()) == ["a", "b", "c"]


def test_save_sorts_records_missing_preached_on_after_every_dated_record(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {
        "undated_b": _record("undated_b", preached_on=None),
        "dated": _record("dated", preached_on="2026-01-04"),
        "undated_a": _record("undated_a", preached_on=None),
    }
    store.save("menlo", records)
    assert list(store.load("menlo").keys()) == ["dated", "undated_a", "undated_b"]


def test_item_to_record_defaults_transcription_fields_to_absent():
    record = store.item_to_record(_item(), first_seen_at="now", feed_published_at=None)
    assert record["transcription_status"] is None
    assert record["transcribed_at"] is None
    assert record["transcript_hash"] is None
    assert record["content_path"] is None
    assert record["transcription_model"] is None
    assert record["transcription_domain_prompt"] is None


def test_mark_transcribed_sets_exactly_its_own_fields():
    record = store.item_to_record(_item(), first_seen_at="now", feed_published_at=None)
    store.mark_transcribed(
        record,
        content_path="transcripts/menlo/2026-09-06_hear-and-do_g1.txt",
        transcript_hash="deadbeef",
        transcribed_at="2026-09-06T12:00:00+00:00",
        model="large-v3",
        domain_prompt=True,
    )
    assert record["transcription_status"] == "done"
    assert record["content_path"] == "transcripts/menlo/2026-09-06_hear-and-do_g1.txt"
    assert record["transcript_hash"] == "deadbeef"
    assert record["transcribed_at"] == "2026-09-06T12:00:00+00:00"
    assert record["transcription_model"] == "large-v3"
    assert record["transcription_domain_prompt"] is True


def test_mark_transcription_failed_sets_only_the_status():
    record = store.item_to_record(_item(), first_seen_at="now", feed_published_at=None)
    store.mark_transcription_failed(record)
    assert record["transcription_status"] == "failed"
    assert record["content_path"] is None
    assert record["transcript_hash"] is None


def test_refresh_record_updates_feed_sourced_fields():
    record = store.item_to_record(_item(), first_seen_at="now", feed_published_at=None)
    rotated = _item(preached_on="2026-09-06").__class__(
        **{
            **_item(preached_on="2026-09-06").__dict__,
            "audio_url": "https://cdn.example.org/rotated.mp3",
            "title": "Hear and Do (rebroadcast)",
        }
    )
    store.refresh_record(record, rotated)
    assert record["audio_url"] == "https://cdn.example.org/rotated.mp3"
    assert record["title"] == "Hear and Do (rebroadcast)"


def test_refresh_record_never_touches_first_seen_at_feed_published_at_or_progress_fields():
    record = store.item_to_record(
        _item(), first_seen_at="2026-01-01T00:00:00+00:00", feed_published_at="2026-01-01T00:00:00+00:00"
    )
    record["notified_at"] = "2026-01-02T00:00:00+00:00"
    store.mark_transcribed(
        record,
        content_path="transcripts/x.txt",
        transcript_hash="abc",
        transcribed_at="2026-01-03T00:00:00+00:00",
        model="large-v3",
        domain_prompt=True,
    )

    store.refresh_record(record, _item(preached_on="2026-09-06"))

    assert record["first_seen_at"] == "2026-01-01T00:00:00+00:00"
    assert record["feed_published_at"] == "2026-01-01T00:00:00+00:00"
    assert record["notified_at"] == "2026-01-02T00:00:00+00:00"
    assert record["transcript_hash"] == "abc"
    assert record["content_path"] == "transcripts/x.txt"
    assert record["transcribed_at"] == "2026-01-03T00:00:00+00:00"


def test_refresh_record_freezes_title_once_transcription_is_done_but_not_other_fields():
    record = store.item_to_record(_item(), first_seen_at="now", feed_published_at=None)
    store.mark_transcribed(
        record,
        content_path="transcripts/x.txt",
        transcript_hash="abc",
        transcribed_at="now",
        model="large-v3",
        domain_prompt=True,
    )

    renamed_and_rotated = _item(preached_on="2026-09-06").__class__(
        **{
            **_item(preached_on="2026-09-06").__dict__,
            "title": "A totally different title",
            "audio_url": "https://cdn.example.org/rotated.mp3",
        }
    )
    store.refresh_record(record, renamed_and_rotated)

    assert record["title"] == "Hear and Do"  # frozen once done
    assert record["audio_url"] == "https://cdn.example.org/rotated.mp3"  # still refreshes


def test_an_old_record_without_transcription_fields_round_trips_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    old_record = _record("g1", preached_on="2026-01-04")
    for key in (
        "transcription_status",
        "transcribed_at",
        "transcript_hash",
        "content_path",
        "transcription_model",
        "transcription_domain_prompt",
    ):
        del old_record[key]

    store.save("menlo", {"g1": old_record})
    loaded = store.load("menlo")
    assert "transcription_status" not in loaded["g1"]
    assert "transcription_model" not in loaded["g1"]
    assert loaded["g1"]["title"] == "Hear and Do"


def _write_overrides(tmp_path, church: str, overrides: dict) -> None:
    overrides_dir = tmp_path / "overrides"
    overrides_dir.mkdir(parents=True, exist_ok=True)
    (overrides_dir / f"{church}.json").write_text(json.dumps(overrides), encoding="utf-8")


def test_load_with_no_overrides_file_is_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    store.save("menlo", {"g1": _record("g1", preached_on="2026-09-06")})
    loaded = store.load("menlo")
    assert loaded["g1"]["audio_url"] == "https://example.org/ep1.mp3"


def test_load_merges_an_override_field_onto_a_matching_guid(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    store.save("menlo", {"g1": _record("g1", preached_on="2026-09-06")})
    _write_overrides(tmp_path, "menlo", {"g1": {"audio_url": "https://cdn.example.org/fixed.mp3"}})

    loaded = store.load("menlo")
    assert loaded["g1"]["audio_url"] == "https://cdn.example.org/fixed.mp3"


def test_load_merges_multiple_override_fields_at_once(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    store.save("menlo", {"g1": _record("g1", preached_on="2026-09-06")})
    _write_overrides(
        tmp_path,
        "menlo",
        {"g1": {"audio_url": "https://cdn.example.org/fixed.mp3", "title": "Corrected Title"}},
    )

    loaded = store.load("menlo")
    assert loaded["g1"]["audio_url"] == "https://cdn.example.org/fixed.mp3"
    assert loaded["g1"]["title"] == "Corrected Title"


def test_load_ignores_underscore_prefixed_override_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    store.save("menlo", {"g1": _record("g1", preached_on="2026-09-06")})
    _write_overrides(
        tmp_path,
        "menlo",
        {
            "g1": {
                "audio_url": "https://cdn.example.org/fixed.mp3",
                "_reason": "feed serves a 404ing path",
                "_added_at": "2026-09-15",
            }
        },
    )

    loaded = store.load("menlo")
    assert loaded["g1"]["audio_url"] == "https://cdn.example.org/fixed.mp3"
    assert "_reason" not in loaded["g1"]
    assert "_added_at" not in loaded["g1"]


def test_load_warns_and_skips_an_override_for_an_unknown_guid(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    store.save("menlo", {"g1": _record("g1", preached_on="2026-09-06")})
    _write_overrides(tmp_path, "menlo", {"unknown-guid": {"audio_url": "https://cdn.example.org/x.mp3"}})

    with caplog.at_level("WARNING"):
        loaded = store.load("menlo")

    assert list(loaded.keys()) == ["g1"]
    assert loaded["g1"]["audio_url"] == "https://example.org/ep1.mp3"
    assert any("unknown-guid" in message for message in caplog.messages)


def test_save_does_not_read_or_touch_the_overrides_file(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    _write_overrides(tmp_path, "menlo", {"g1": {"audio_url": "https://cdn.example.org/fixed.mp3"}})

    store.save("menlo", {"g1": _record("g1", preached_on="2026-09-06")})

    on_disk = json.loads((tmp_path / "menlo.json").read_text(encoding="utf-8"))
    assert on_disk["g1"]["audio_url"] == "https://example.org/ep1.mp3"


def test_save_interrupted_mid_write_leaves_the_previous_ledger_intact(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    store.save("menlo", {"g1": _record("g1", preached_on="2026-09-06")})
    before = (tmp_path / "menlo.json").read_text(encoding="utf-8")

    def _boom(*args, **kwargs):
        raise RuntimeError("killed mid-write")

    monkeypatch.setattr(store.json, "dump", _boom)
    with pytest.raises(RuntimeError):
        store.save("menlo", {"g2": _record("g2", preached_on="2026-09-07")})

    assert (tmp_path / "menlo.json").read_text(encoding="utf-8") == before
    assert not (tmp_path / "menlo.json.tmp").exists()


def test_save_leaves_no_tmp_file_behind_on_success(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    store.save("menlo", {"g1": _record("g1", preached_on="2026-09-06")})
    assert not (tmp_path / "menlo.json.tmp").exists()


def test_validate_all_returns_empty_list_when_data_dir_is_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path / "does-not-exist")
    assert store.validate_all() == []


def test_validate_all_returns_empty_list_for_well_formed_ledgers(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    store.save("menlo", {"g1": _record("g1", preached_on="2026-09-06")})
    store.save("pbc", {"g2": _record("g2", preached_on="2026-09-07")})
    assert store.validate_all() == []


def test_validate_all_reports_a_truncated_ledger_by_filename(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    store.save("menlo", {"g1": _record("g1", preached_on="2026-09-06")})
    (tmp_path / "menlo.json").write_text('{"g1": {"guid":', encoding="utf-8")

    assert store.validate_all() == ["menlo.json"]


def test_main_validate_exits_zero_for_well_formed_ledgers(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    store.save("menlo", {"g1": _record("g1", preached_on="2026-09-06")})
    assert store.main(["validate"]) == 0


def test_main_validate_exits_non_zero_and_reports_a_truncated_ledger(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    (tmp_path / "menlo.json").write_text("not json", encoding="utf-8")

    assert store.main(["validate"]) == 1
    assert "menlo.json" in capsys.readouterr().err


def _legacy_record(guid: str, *, published_on: str, published_at: str | None) -> dict:
    """A ledger record exactly as written before ADR-0015's rename."""
    return {
        "guid": guid,
        "title": "Hear and Do",
        "raw_title": "Hear and Do - Luke",
        "series": "Luke",
        "speaker": "Jane Doe",
        "published_on": published_on,
        "published_at": published_at,
        "episode_url": "https://example.org/ep1",
        "audio_url": "https://example.org/ep1.mp3",
        "blurb": "A sermon on Luke.",
        "first_seen_at": "2026-09-06T20:00:00+00:00",
        "notified_at": "2026-09-06T20:00:00+00:00",
    }


def _write_ledger(tmp_path, church: str, records: dict) -> None:
    (tmp_path / f"{church}.json").write_text(json.dumps(records), encoding="utf-8")


def test_load_renames_legacy_date_keys_in_place_preserving_field_order(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    legacy = _legacy_record("g1", published_on="2026-09-06", published_at="2026-09-06T17:03:00+00:00")
    _write_ledger(tmp_path, "menlo", {"g1": legacy})

    record = store.load("menlo")["g1"]

    assert record["preached_on"] == "2026-09-06"
    assert record["feed_published_at"] == "2026-09-06T17:03:00+00:00"
    assert "published_on" not in record
    assert "published_at" not in record
    expected_order = [
        {"published_on": "preached_on", "published_at": "feed_published_at"}.get(k, k) for k in legacy
    ]
    assert list(record) == expected_order


def test_load_reads_a_ledger_mixing_legacy_and_renamed_records(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    renamed = _record("new", preached_on="2026-09-13", feed_published_at="2026-09-13T17:00:00+00:00")
    legacy = _legacy_record("old", published_on="2026-09-06", published_at=None)
    _write_ledger(tmp_path, "menlo", {"new": renamed, "old": legacy})

    loaded = store.load("menlo")

    assert loaded["new"] == renamed
    assert loaded["old"]["preached_on"] == "2026-09-06"
    assert loaded["old"]["feed_published_at"] is None


def test_load_renames_legacy_date_keys_in_an_override_entry(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    store.save("pbc", {"g1": _record("g1", preached_on="2026-09-20")})
    _write_overrides(
        tmp_path, "pbc", {"g1": {"published_on": "2026-09-13", "_reason": "feed shows wrong week"}}
    )

    record = store.load("pbc")["g1"]

    assert record["preached_on"] == "2026-09-13"
    assert "published_on" not in record


def test_save_after_loading_a_legacy_ledger_writes_only_the_renamed_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    _write_ledger(
        tmp_path, "menlo", {"g1": _legacy_record("g1", published_on="2026-09-06", published_at=None)}
    )

    store.save("menlo", store.load("menlo"))

    text = (tmp_path / "menlo.json").read_text(encoding="utf-8")
    assert '"published_on"' not in text
    assert '"published_at"' not in text
    assert '"preached_on": "2026-09-06"' in text


def test_legacy_key_guids_names_only_records_still_on_the_old_keys(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    _write_ledger(
        tmp_path,
        "menlo",
        {
            "new": _record("new", preached_on="2026-09-13"),
            "old": _legacy_record("old", published_on="2026-09-06", published_at=None),
        },
    )

    assert store.legacy_key_guids("menlo") == ["old"]


def test_legacy_key_guids_is_empty_for_a_church_with_no_ledger(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    assert store.legacy_key_guids("menlo") == []


def test_committed_ledgers_and_overrides_carry_no_legacy_date_keys():
    paths = sorted(store.DATA_DIR.glob("*.json")) + sorted((store.DATA_DIR / "overrides").glob("*.json"))
    assert paths, "expected committed ledgers under data/"
    offenders = []
    for path in paths:
        with path.open(encoding="utf-8") as f:
            entries = json.load(f)
        for guid, entry in entries.items():
            if {"published_on", "published_at"} & entry.keys():
                offenders.append(f"{path.name}:{guid}")
    assert offenders == [], f"legacy keys (ADR-0015) in {offenders[:5]}"


def test_load_prefers_the_legacy_value_when_a_record_carries_both_names(tmp_path, monkeypatch):
    # An old-code poll in flight across the rename merge refreshes `published_on` onto an
    # already-renamed record: that write is the newer feed value.
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    record = _record("g1", preached_on="2026-09-20")
    record["published_on"] = "2026-09-13"
    _write_ledger(tmp_path, "pbc", {"g1": record})

    loaded = store.load("pbc")["g1"]

    assert loaded["preached_on"] == "2026-09-13"
    assert "published_on" not in loaded
