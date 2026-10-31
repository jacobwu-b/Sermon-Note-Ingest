import json

import pytest

from poller import quality, store
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
    assert record["transcription_last_error"] is None
    assert record["transcript_adopted"] is None


def test_item_to_record_carries_every_decoding_profile_field_as_unknown():
    record = store.item_to_record(_item(), first_seen_at="now", feed_published_at=None)
    assert {key: record[f"transcription_{key}"] for key in store.PROFILE_KEYS} == dict.fromkeys(
        store.PROFILE_KEYS
    )


def test_mark_transcribed_sets_exactly_its_own_fields():
    record = store.item_to_record(_item(), first_seen_at="now", feed_published_at=None)
    store.mark_transcribed(
        record,
        content_path="transcripts/menlo/2026-09-06_hear-and-do_g1.txt",
        transcript_hash="deadbeef",
        transcribed_at="2026-09-06T12:00:00+00:00",
        profile={"model": "large-v3", "domain_prompt": True},
    )
    assert record["transcription_status"] == "done"
    assert record["content_path"] == "transcripts/menlo/2026-09-06_hear-and-do_g1.txt"
    assert record["transcript_hash"] == "deadbeef"
    assert record["transcribed_at"] == "2026-09-06T12:00:00+00:00"
    assert record["transcription_model"] == "large-v3"
    assert record["transcription_domain_prompt"] is True
    assert record["transcript_adopted"] is False


_FULL_PROFILE = {
    "model": "large-v3",
    "compute_type": "int8",
    "beam_size": 5,
    "condition_on_previous_text": False,
    "hallucination_silence_threshold": 2.0,
    "vad_filter": True,
    "domain_prompt": True,
    "hotwords": "Keith Crosby, Luke",
    "faster_whisper_version": "1.2.1",
    "ctranslate2_version": "4.8.2",
}


def test_mark_transcribed_records_every_decoding_profile_field():
    record = store.item_to_record(_item(), first_seen_at="now", feed_published_at=None)
    store.mark_transcribed(
        record, content_path="x", transcript_hash="h", transcribed_at="t", profile=_FULL_PROFILE
    )
    assert {key: record[f"transcription_{key}"] for key in store.PROFILE_KEYS} == _FULL_PROFILE


def test_mark_transcribed_records_a_missing_profile_field_as_unknown():
    record = store.item_to_record(_item(), first_seen_at="now", feed_published_at=None)
    store.mark_transcribed(
        record, content_path="x", transcript_hash="h", transcribed_at="t", profile={"model": "large-v3"}
    )
    assert record["transcription_model"] == "large-v3"
    assert record["transcription_beam_size"] is None


def test_mark_transcribed_refuses_a_profile_field_the_ledger_does_not_know():
    record = store.item_to_record(_item(), first_seen_at="now", feed_published_at=None)
    with pytest.raises(ValueError, match="temperature"):
        store.mark_transcribed(
            record, content_path="x", transcript_hash="h", transcribed_at="t", profile={"temperature": 0.0}
        )
    assert record["transcription_status"] is None


def test_mark_transcribed_records_a_transcript_adopted_from_content():
    record = store.item_to_record(_item(), first_seen_at="now", feed_published_at=None)
    store.mark_transcribed(
        record,
        content_path="transcripts/menlo/2026-09-06_hear-and-do_g1.txt",
        transcript_hash="deadbeef",
        transcribed_at="2026-09-19T15:49:16+00:00",
        profile=None,
        adopted=True,
    )
    assert record["transcript_adopted"] is True


_COUNTS = {"words": 6423, "loop_words": 72, "credit_hits": 0, "hotword_pairs": 0}


def test_item_to_record_carries_every_quality_count_as_not_scored():
    record = store.item_to_record(_item(), first_seen_at="now", feed_published_at=None)
    assert {key: record[f"transcript_{key}"] for key in store.QUALITY_KEYS} == dict.fromkeys(
        store.QUALITY_KEYS
    )


def test_mark_transcribed_records_the_transcripts_quality_counts():
    record = store.item_to_record(_item(), first_seen_at="now", feed_published_at=None)
    store.mark_transcribed(
        record,
        content_path="x",
        transcript_hash="h",
        transcribed_at="t",
        profile=_FULL_PROFILE,
        quality=_COUNTS,
    )
    assert {key: record[f"transcript_{key}"] for key in store.QUALITY_KEYS} == _COUNTS
    assert store.transcript_quality(record) == quality.QualityScore(**_COUNTS)


def test_an_adopted_transcript_records_its_quality_as_not_scored():
    record = store.item_to_record(_item(), first_seen_at="now", feed_published_at=None)
    store.mark_transcribed(
        record, content_path="x", transcript_hash="h", transcribed_at="t", profile=None, adopted=True
    )
    assert {key: record[f"transcript_{key}"] for key in store.QUALITY_KEYS} == dict.fromkeys(
        store.QUALITY_KEYS
    )
    assert store.transcript_quality(record) is None


def test_mark_transcribed_refuses_a_quality_count_the_ledger_does_not_know():
    record = store.item_to_record(_item(), first_seen_at="now", feed_published_at=None)
    with pytest.raises(ValueError, match="wpm"):
        store.mark_transcribed(
            record,
            content_path="x",
            transcript_hash="h",
            transcribed_at="t",
            profile=None,
            quality={"wpm": 150},
        )
    assert record["transcription_status"] is None


def test_a_record_saved_before_quality_counts_existed_reads_as_not_scored():
    record = store.item_to_record(_item(), first_seen_at="now", feed_published_at=None)
    store.mark_transcribed(record, content_path="x", transcript_hash="h", transcribed_at="t", profile=None)
    for key in store.QUALITY_KEYS:
        del record[f"transcript_{key}"]
    assert store.transcript_quality(record) is None


def test_a_successful_transcription_clears_the_earlier_failure_reason():
    record = store.item_to_record(_item(), first_seen_at="now", feed_published_at=None)
    store.mark_download_failed(record)
    store.mark_transcribed(
        record,
        content_path="transcripts/menlo/2026-09-06_hear-and-do_g1.txt",
        transcript_hash="deadbeef",
        transcribed_at="2026-09-06T12:00:00+00:00",
        profile={"model": "large-v3", "domain_prompt": True},
    )
    assert record["transcription_last_error"] is None


def test_mark_download_failed_records_the_reason_and_leaves_the_record_pending():
    record = store.item_to_record(_item(), first_seen_at="now", feed_published_at=None)
    store.mark_download_failed(record)
    assert record["transcription_last_error"] == "audio_download"
    assert record["transcription_status"] is None


def test_mark_download_failed_never_overwrites_a_settled_record():
    """A replayed download mark landing after another run finished must not relabel it."""
    record = store.item_to_record(_item(), first_seen_at="now", feed_published_at=None)
    store.mark_transcribed(
        record,
        content_path="transcripts/menlo/2026-09-06_hear-and-do_g1.txt",
        transcript_hash="deadbeef",
        transcribed_at="2026-09-06T12:00:00+00:00",
        profile={"model": "large-v3", "domain_prompt": True},
    )
    store.mark_download_failed(record)
    assert record["transcription_last_error"] is None
    assert record["transcription_status"] == "done"


def test_a_repeated_download_failure_leaves_the_record_byte_identical():
    """No attempt timestamp: a download failing every run must not rewrite the ledger every run."""
    record = store.item_to_record(_item(), first_seen_at="now", feed_published_at=None)
    store.mark_download_failed(record)
    first = json.dumps(record, sort_keys=True)
    store.mark_download_failed(record)
    assert json.dumps(record, sort_keys=True) == first


def test_mark_transcription_failed_records_the_status_and_the_reason():
    record = store.item_to_record(_item(), first_seen_at="now", feed_published_at=None)
    store.mark_transcription_failed(record, error="empty_transcript")
    assert record["transcription_status"] == "failed"
    assert record["transcription_last_error"] == "empty_transcript"
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
        profile={"model": "large-v3", "domain_prompt": True},
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
        profile={"model": "large-v3", "domain_prompt": True},
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


def _on_disk(tmp_path, church: str) -> dict:
    return json.loads((tmp_path / f"{church}.json").read_text(encoding="utf-8"))


def test_a_load_then_save_round_trip_keeps_the_override_out_of_the_ledger_file(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    store.save("menlo", {"g1": _record("g1", preached_on="2026-09-06")})
    _write_overrides(tmp_path, "menlo", {"g1": {"audio_url": "https://cdn.example.org/fixed.mp3"}})

    store.save("menlo", store.load("menlo"))

    assert _on_disk(tmp_path, "menlo")["g1"]["audio_url"] == "https://example.org/ep1.mp3"
    assert store.load("menlo")["g1"]["audio_url"] == "https://cdn.example.org/fixed.mp3"


def test_removing_an_override_after_a_round_trip_restores_the_ledger_value(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    store.save("menlo", {"g1": _record("g1", preached_on="2026-09-06")})
    _write_overrides(
        tmp_path, "menlo", {"g1": {"audio_url": "https://cdn.example.org/fixed.mp3", "title": "Fixed"}}
    )
    records = store.load("menlo")
    store.mark_download_failed(records["g1"])
    store.save("menlo", records)

    (tmp_path / "overrides" / "menlo.json").unlink()

    reloaded = store.load("menlo")["g1"]
    assert reloaded["audio_url"] == "https://example.org/ep1.mp3"
    assert reloaded["title"] == "Hear and Do"
    assert reloaded["transcription_last_error"] == store.DOWNLOAD_ERROR


def test_an_override_adding_a_field_the_ledger_lacks_is_not_written_back(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    store.save("menlo", {"g1": _record("g1", preached_on="2026-09-06")})
    _write_overrides(tmp_path, "menlo", {"g1": {"service_note": "Combined campus service"}})

    store.save("menlo", store.load("menlo"))

    assert "service_note" not in _on_disk(tmp_path, "menlo")["g1"]


def test_save_writes_an_overridden_field_a_caller_changed_since_load(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    store.save("menlo", {"g1": _record("g1", preached_on="2026-09-06")})
    _write_overrides(tmp_path, "menlo", {"g1": {"audio_url": "https://cdn.example.org/fixed.mp3"}})
    records = store.load("menlo")
    records["g1"]["audio_url"] = "https://cdn.example.org/rotated.mp3"

    store.save("menlo", records)

    assert _on_disk(tmp_path, "menlo")["g1"]["audio_url"] == "https://cdn.example.org/rotated.mp3"


def test_save_leaves_the_callers_merged_records_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    store.save("menlo", {"g1": _record("g1", preached_on="2026-09-06")})
    _write_overrides(tmp_path, "menlo", {"g1": {"audio_url": "https://cdn.example.org/fixed.mp3"}})
    records = store.load("menlo")

    store.save("menlo", records)

    assert records["g1"]["audio_url"] == "https://cdn.example.org/fixed.mp3"


def test_save_writes_a_newly_discovered_record_as_given_even_if_it_matches_an_override(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    store.save("menlo", {"g1": _record("g1", preached_on="2026-09-06")})
    _write_overrides(tmp_path, "menlo", {"g2": {"audio_url": "https://example.org/ep1.mp3"}})
    records = store.load("menlo")
    records["g2"] = _record("g2", preached_on="2026-09-13")

    store.save("menlo", records)

    assert _on_disk(tmp_path, "menlo")["g2"]["audio_url"] == "https://example.org/ep1.mp3"


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


_RUNNING = {
    "model": "large-v3",
    "compute_type": "int8",
    "beam_size": 5,
    "condition_on_previous_text": False,
    "hallucination_silence_threshold": 2.0,
    "vad_filter": True,
    "domain_prompt": False,
}


def _done_record(**profile) -> dict:
    record = store.item_to_record(_item(), first_seen_at="now", feed_published_at=None)
    store.mark_transcribed(
        record,
        content_path="transcripts/menlo/2026-09-06_hear-and-do_g1.txt",
        transcript_hash="oldhash",
        transcribed_at="2026-09-20T12:00:00+00:00",
        profile={**_RUNNING, "hotwords": None, **profile},
        quality={"words": 5000, "loop_words": 120, "credit_hits": 1, "hotword_pairs": 0},
    )
    return record


def test_a_record_produced_under_the_running_profile_is_not_stale():
    assert store.profile_is_stale(_done_record(), _RUNNING) is False


@pytest.mark.parametrize(
    ("key", "old_value"),
    [
        ("model", "small"),
        ("compute_type", "float32"),
        ("beam_size", 1),
        ("condition_on_previous_text", True),
        ("hallucination_silence_threshold", 0),
        ("vad_filter", False),
        ("domain_prompt", True),
    ],
)
def test_a_record_is_stale_when_any_one_decoding_knob_differs(key, old_value):
    assert store.profile_is_stale(_done_record(**{key: old_value}), _RUNNING) is True


@pytest.mark.parametrize("key", store.STALENESS_KEYS)
def test_a_missing_decoding_knob_counts_as_stale(key):
    record = _done_record()
    del record[f"transcription_{key}"]
    assert store.profile_is_stale(record, _RUNNING) is True


def test_engine_versions_and_the_hotwords_string_never_make_a_record_stale():
    # ADR-0022 rule 2: a Dependabot bump or a vocabulary edit must not queue the corpus.
    record = _done_record(
        hotwords="Keith Crosby, Luke", faster_whisper_version="0.9", ctranslate2_version=None
    )
    assert store.profile_is_stale(record, _RUNNING) is False


def test_the_profile_digest_names_the_decoding_knobs_only():
    full = {
        **_RUNNING,
        "hotwords": "Luke, Acts",
        "faster_whisper_version": "1.2.1",
        "ctranslate2_version": "4.8",
    }
    assert store.profile_digest(full) == store.profile_digest(_RUNNING)
    assert store.profile_digest({**_RUNNING, "beam_size": 1}) != store.profile_digest(_RUNNING)


def test_the_profile_digest_reads_a_whole_number_threshold_the_same_as_its_float():
    # JSON round-trips 2.0 as 2.0, but config can hand over 0 or 0.0 for the guard off.
    off = {**_RUNNING, "hallucination_silence_threshold": 0}
    assert store.profile_digest(off) == store.profile_digest({**off, "hallucination_silence_threshold": 0.0})


_NEW_PROFILE = {
    **_RUNNING,
    "hotwords": None,
    "faster_whisper_version": "1.2.1",
    "ctranslate2_version": "4.8.2",
}
_NEW_QUALITY = {"words": 5200, "loop_words": 0, "credit_hits": 0, "hotword_pairs": 0}


def test_mark_retranscribed_replaces_the_transcript_fields_and_keeps_the_old_hash():
    record = _done_record(condition_on_previous_text=True)
    store.mark_retranscribed(
        record,
        transcript_hash="newhash",
        transcribed_at="2026-10-13T01:00:00+00:00",
        profile=_NEW_PROFILE,
        quality=_NEW_QUALITY,
        superseded="oldhash",
    )
    assert record["transcript_hash"] == "newhash"
    assert record["transcribed_at"] == "2026-10-13T01:00:00+00:00"
    assert record["superseded_transcript_hash"] == "oldhash"
    assert {key: record[f"transcription_{key}"] for key in store.PROFILE_KEYS} == _NEW_PROFILE
    assert {key: record[f"transcript_{key}"] for key in store.QUALITY_KEYS} == _NEW_QUALITY
    assert record["transcription_status"] == "done"
    assert record["content_path"] == "transcripts/menlo/2026-09-06_hear-and-do_g1.txt"
    assert record["transcript_adopted"] is False


def test_mark_retranscribed_with_identical_text_leaves_any_earlier_superseded_hash():
    record = _done_record(condition_on_previous_text=True)
    record["superseded_transcript_hash"] = "olderhash"
    store.mark_retranscribed(
        record,
        transcript_hash="oldhash",
        transcribed_at="2026-10-13T01:00:00+00:00",
        profile=_NEW_PROFILE,
        quality=_NEW_QUALITY,
        superseded=None,
    )
    assert record["superseded_transcript_hash"] == "olderhash"
    assert record["transcription_condition_on_previous_text"] is False


def test_mark_retranscribed_clears_an_earlier_retranscription_failure():
    record = _done_record(condition_on_previous_text=True)
    store.mark_retranscribe_failed(
        record,
        error=store.QUALITY_REGRESSION_ERROR,
        failed_profile="digest",
        candidate_path="retranscription-candidates/menlo/x.abc.txt",
        candidate_hash="abc",
    )
    store.mark_retranscribed(
        record,
        transcript_hash="newhash",
        transcribed_at="t",
        profile=_NEW_PROFILE,
        quality=_NEW_QUALITY,
        superseded="oldhash",
    )
    for field in (
        "retranscription_last_error",
        "retranscription_failed_profile",
        "retranscription_candidate_path",
        "retranscription_candidate_hash",
    ):
        assert record[field] is None


def test_mark_retranscribed_refuses_a_field_the_ledger_does_not_know():
    with pytest.raises(ValueError, match="temperature"):
        store.mark_retranscribed(
            _done_record(),
            transcript_hash="h",
            transcribed_at="t",
            profile={**_NEW_PROFILE, "temperature": 0},
            quality=_NEW_QUALITY,
            superseded="oldhash",
        )


def test_mark_retranscribe_failed_leaves_every_current_transcript_field_untouched():
    record = _done_record(condition_on_previous_text=True)
    before = {key: value for key, value in record.items() if not key.startswith("retranscription_")}
    store.mark_retranscribe_failed(
        record,
        error=store.QUALITY_REGRESSION_ERROR,
        failed_profile="digest",
        candidate_path="retranscription-candidates/menlo/x.abc.txt",
        candidate_hash="abc",
    )
    assert {key: value for key, value in record.items() if not key.startswith("retranscription_")} == before
    assert record["retranscription_last_error"] == "quality_regression"
    assert record["retranscription_failed_profile"] == "digest"
    assert record["retranscription_candidate_path"] == "retranscription-candidates/menlo/x.abc.txt"
    assert record["retranscription_candidate_hash"] == "abc"


def test_a_later_retranscription_failure_keeps_the_pointer_to_an_earlier_candidate():
    record = _done_record()
    store.mark_retranscribe_failed(
        record, error="quality_regression", failed_profile="d1", candidate_path="p", candidate_hash="h"
    )
    store.mark_retranscribe_failed(record, error=store.DOWNLOAD_ERROR, failed_profile="d2")
    assert record["retranscription_last_error"] == "audio_download"
    assert record["retranscription_candidate_path"] == "p"


# --- discovery alert state (ADR-0025) ---


def test_alert_state_is_empty_before_any_alert_and_round_trips(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    assert store.load_alert_state() == {}

    state = {
        "westgate": {"missing_alerted_for": "2026-10-25"},
        "pbc": {"failing_since": "2026-10-29T15:00:00+00:00"},
    }
    store.save_alert_state(state)
    assert store.load_alert_state() == state
    assert not list((tmp_path / "alerts").glob("*.tmp"))


def test_alert_snoozes_are_empty_without_a_snooze_file(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    assert store.load_alert_snoozes() == {}


def test_alert_snoozes_read_each_churchs_until_date_and_skip_notes(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    (tmp_path / "alerts").mkdir()
    (tmp_path / "alerts" / "snooze.json").write_text(
        json.dumps({"lakepointe": {"until": "2026-08-02", "_reason": "summer break"}}), encoding="utf-8"
    )
    assert store.load_alert_snoozes() == {"lakepointe": "2026-08-02"}


def test_validate_all_reports_a_corrupt_alert_state_file(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    (tmp_path / "alerts").mkdir()
    (tmp_path / "alerts" / "state.json").write_text('{"westgate": {', encoding="utf-8")
    assert store.validate_all() == ["alerts/state.json"]
