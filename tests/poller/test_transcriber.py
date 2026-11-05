import dataclasses
import hashlib

import pytest

from poller import net, notify, pipeline_dispatch, store, transcribe, transcriber
from poller.content_repo import ContentPublishError
from poller.net import AudioDownloadError
from poller.sources.base import SermonItem
from poller.transcribe import EmptyTranscriptError

_PROFILE = transcribe.DecodingProfile(
    model="large-v3",
    compute_type="int8",
    beam_size=5,
    condition_on_previous_text=False,
    hallucination_silence_threshold=2.0,
    vad_filter=True,
    domain_prompt=False,
    hotwords=None,
    faster_whisper_version="1.2.1",
    ctranslate2_version="4.8.2",
)
_AUDIO = hashlib.sha256(b"sermon audio").hexdigest()


def _item(guid: str, *, preached_on: str, audio_url: str = "https://example.org/a.mp3") -> SermonItem:
    return SermonItem(
        guid=guid,
        title=f"Sermon {guid}",
        raw_title=f"Sermon {guid}",
        series=None,
        speaker=None,
        preached_on=preached_on,
        feed_published_at=None,
        episode_url="https://example.org/ep",
        audio_url=audio_url,
        blurb="",
    )


def _record(guid: str, *, preached_on: str, audio_url: str = "https://example.org/a.mp3") -> dict:
    return store.item_to_record(
        _item(guid, preached_on=preached_on, audio_url=audio_url), first_seen_at="t", feed_published_at=None
    )


def _nothing_in_content(paths):
    return {}


@pytest.fixture(autouse=True)
def churches_env(church_table):
    church_table(
        '{"menlo": {"rss": "https://example.org/menlo.xml", "ingest": {"enabled": true}},'
        ' "pbc": {"rss": "https://example.org/pbc.xml", "ingest": {"enabled": true}}}'
    )


def test_select_pending_skips_done_failed_and_unfetchable():
    records = {
        "done": _record("done", preached_on="2026-01-01"),
        "failed": _record("failed", preached_on="2026-01-02"),
        "no_audio": _record("no_audio", preached_on="2026-01-03", audio_url=""),
        "pending": _record("pending", preached_on="2026-01-04"),
    }
    store.mark_transcribed(
        records["done"],
        content_path="x",
        transcript_hash="h",
        transcribed_at="t",
        profile={"model": "large-v3", "domain_prompt": True},
    )
    store.mark_transcription_failed(records["failed"], error="transcription")

    pending = transcriber._select_pending(records)
    assert [guid for guid, _r in pending] == ["pending"]


def test_select_pending_orders_newest_preached_on_first():
    records = {
        "new": _record("new", preached_on="2026-09-06"),
        "old": _record("old", preached_on="2026-01-04"),
        "mid": _record("mid", preached_on="2026-05-01"),
    }
    pending = transcriber._select_pending(records)
    assert [guid for guid, _r in pending] == ["new", "mid", "old"]


def test_select_pending_sorts_undated_records_last_regardless_of_direction():
    records = {
        "undated": _record("undated", preached_on=""),
        "new": _record("new", preached_on="2026-09-06"),
        "old": _record("old", preached_on="2026-01-04"),
    }
    pending = transcriber._select_pending(records)
    assert [guid for guid, _r in pending] == ["new", "old", "undated"]


def test_content_path_is_deterministic_for_the_same_guid():
    record = _record("guid-1", preached_on="2026-09-06")
    record["title"] = "Hear and Do"
    path_a = transcriber._content_path("menlo", record)
    path_b = transcriber._content_path("menlo", record)
    assert path_a == path_b
    assert path_a == "transcripts/menlo/2026-09-06_hear-and-do_guid-1.txt"


def test_transcribe_church_marks_success_only_after_push_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {"g1": _record("g1", preached_on="2026-01-01")}
    pushed = {}

    def fake_transcribe_audio(url, hotwords, known_audio):
        return "the transcript", "hash123", _PROFILE, _AUDIO

    def fake_push(files):
        pushed.update(files)

    ok = transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"])],
        transcribe_audio=fake_transcribe_audio,
        push=fake_push,
        fetch_existing=_nothing_in_content,
    )

    assert ok is True
    assert pushed == {"transcripts/menlo/2026-01-01_sermon-g1_g1.txt": "the transcript"}
    saved = store.load("menlo")
    assert saved["g1"]["transcription_status"] == "done"
    assert saved["g1"]["transcript_hash"] == "hash123"
    assert saved["g1"]["content_path"] == "transcripts/menlo/2026-01-01_sermon-g1_g1.txt"


def test_transcribe_church_records_the_decoding_profile_the_transcription_returned(tmp_path, monkeypatch):
    # ADR-0022: the ledger records what the model call was given, not the config
    # re-read afterwards — the running env deliberately disagrees here.
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setenv("WHISPER_MODEL", "small")
    monkeypatch.setenv("WHISPER_DOMAIN_PROMPT", "true")
    records = {"g1": _record("g1", preached_on="2026-01-01")}
    prompted = dataclasses.replace(_PROFILE, domain_prompt=True, hotwords="Keith Crosby, Luke")

    transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"])],
        transcribe_audio=lambda url, hotwords, known_audio: ("the transcript", "hash123", prompted, _AUDIO),
        push=lambda files: None,
        fetch_existing=_nothing_in_content,
    )

    saved = store.load("menlo")["g1"]
    assert {key: saved[f"transcription_{key}"] for key in store.PROFILE_KEYS} == dataclasses.asdict(prompted)


def test_transcribe_church_passes_per_sermon_hotwords_built_from_the_record_and_church_vocabulary(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {
        "g1": _record("g1", preached_on="2026-01-01") | {"speaker": "Keith Crosby", "series": "Luke"},
        "g2": _record("g2", preached_on="2025-12-01") | {"speaker": "Jesse Fenn", "series": None},
    }
    seen = {}

    def fake_transcribe_audio(url, hotwords, known_audio):
        seen[url] = hotwords
        return "the transcript", "hash123", _PROFILE, _AUDIO

    ok = transcriber.transcribe_church(
        "hillside",
        records,
        [("g1", records["g1"])],
        transcribe_audio=fake_transcribe_audio,
        push=lambda files: None,
        fetch_existing=_nothing_in_content,
        vocabulary=("Hillside Church",),
    )

    assert ok is True
    # The sermon's own metadata leads, then the operator's vocabulary, then what the
    # rest of the ledger knows about this church (g2's speaker).
    assert seen["https://example.org/a.mp3"] == "Keith Crosby, Luke, Sermon g1, Hillside Church, Jesse Fenn"


def test_run_passes_the_configured_church_vocabulary_through_to_the_hotwords(
    tmp_path, monkeypatch, church_table
):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    church_table(
        '{"pbc": {"rss": "https://example.org/pbc.xml", "ingest": {"enabled": true,'
        ' "vocabulary": ["Peninsula Bible Church"]}}}'
    )
    store.save("pbc", {"p1": _record("p1", preached_on="2026-01-01")})
    seen = []

    def fake_transcribe_audio(url, hotwords, known_audio):
        seen.append(hotwords)
        return "text", "hash", _PROFILE, _AUDIO

    transcriber.run(
        church_names=["pbc"],
        limit=1,
        transcribe_audio=fake_transcribe_audio,
        push=lambda files: None,
        fetch_existing=_nothing_in_content,
    )

    assert seen == ["Sermon p1, Peninsula Bible Church"]


def test_transcribe_church_dispatches_an_ingest_event_per_successfully_transcribed_item(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    # _recently_published compares against the real wall clock (poller.net.now), so the
    # fixture's publish date must be pinned to a fixed "now" rather than a fixed calendar
    # date — otherwise this test silently starts failing once real time drifts more than
    # _DISPATCH_RECENCY past whatever date was hardcoded here.
    monkeypatch.setattr(net, "now", lambda: "2026-09-15T12:00:00+00:00")
    records = {"g1": _record("g1", preached_on="2026-09-15")}
    dispatched = []

    def fake_transcribe_audio(url, hotwords, known_audio):
        return "the transcript", "hash123", _PROFILE, _AUDIO

    ok = transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"])],
        transcribe_audio=fake_transcribe_audio,
        push=lambda files: None,
        fetch_existing=_nothing_in_content,
        dispatch_ingest_event=lambda event, source: dispatched.append((event, source)),
    )

    assert ok is True
    assert len(dispatched) == 1
    event, source = dispatched[0]
    assert source == "menlo"
    assert event["event"] == "sermon_detected"
    assert event["source"] == "menlo"
    assert event["external_id"] == "g1"
    assert event["transcript"]["content_path"] == "transcripts/menlo/2026-09-15_sermon-g1_g1.txt"
    assert event["transcript"]["transcript_hash"] == "hash123"


def test_transcribe_church_does_not_dispatch_for_a_church_pipeline_does_not_run(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(net, "now", lambda: "2026-09-15T12:00:00+00:00")
    records = {"g1": _record("g1", preached_on="2026-09-15")}
    dispatched = []

    ok = transcriber.transcribe_church(
        "lakepointe",
        records,
        [("g1", records["g1"])],
        transcribe_audio=lambda url, hotwords, known_audio: ("the transcript", "hash123", _PROFILE, _AUDIO),
        push=lambda files: None,
        fetch_existing=_nothing_in_content,
        dispatch_ingest_event=lambda event, source: dispatched.append((event, source)),
        pipeline_enabled=False,
    )

    assert ok is True
    assert dispatched == []
    assert store.load("lakepointe")["g1"]["transcription_status"] == "done"


def test_run_dispatches_only_for_churches_the_table_enables_under_pipeline(
    tmp_path, monkeypatch, church_table
):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(net, "now", lambda: "2026-09-15T12:00:00+00:00")
    church_table(
        '{"menlo": {"rss": "https://example.org/m.xml", "pipeline": {"enabled": true},'
        ' "ingest": {"enabled": true}},'
        ' "lakepointe": {"rss": "https://example.org/l.xml", "pipeline": {"enabled": false},'
        ' "ingest": {"enabled": true}}}'
    )
    for name in ("menlo", "lakepointe"):
        store.save(name, {"g1": _record("g1", preached_on="2026-09-15")})
    dispatched = []

    transcriber.run(
        church_names=None,
        limit=1,
        transcribe_audio=lambda url, hotwords, known_audio: ("the transcript", "hash123", _PROFILE, _AUDIO),
        push=lambda files: None,
        fetch_existing=_nothing_in_content,
        dispatch_ingest_event=lambda event, source: dispatched.append(source),
    )

    assert dispatched == ["menlo"]


def test_transcribe_church_survives_a_dispatch_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {"g1": _record("g1", preached_on="2026-09-15")}

    def fake_transcribe_audio(url, hotwords, known_audio):
        return "the transcript", "hash123", _PROFILE, _AUDIO

    def failing_dispatch(event, source):
        raise pipeline_dispatch.PipelineDispatchError("pipeline unreachable")

    ok = transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"])],
        transcribe_audio=fake_transcribe_audio,
        push=lambda files: None,
        fetch_existing=_nothing_in_content,
        dispatch_ingest_event=failing_dispatch,
    )

    assert ok is True
    saved = store.load("menlo")
    assert saved["g1"]["transcription_status"] == "done"


def test_transcribe_church_never_dispatches_for_an_item_whose_push_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {"g1": _record("g1", preached_on="2026-09-15")}
    dispatched = []

    def fake_transcribe_audio(url, hotwords, known_audio):
        return "text", "hash", _PROFILE, _AUDIO

    def failing_push(files):
        raise ContentPublishError("content repo unreachable")

    ok = transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"])],
        transcribe_audio=fake_transcribe_audio,
        push=failing_push,
        fetch_existing=_nothing_in_content,
        dispatch_ingest_event=lambda event, source: dispatched.append((event, source)),
    )

    assert ok is False
    assert dispatched == []


def test_transcribe_church_never_dispatches_for_a_backfill_sermon(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {"g1": _record("g1", preached_on="2026-01-01")}
    dispatched = []

    def fake_transcribe_audio(url, hotwords, known_audio):
        return "the transcript", "hash123", _PROFILE, _AUDIO

    ok = transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"])],
        transcribe_audio=fake_transcribe_audio,
        push=lambda files: None,
        fetch_existing=_nothing_in_content,
        dispatch_ingest_event=lambda event, source: dispatched.append((event, source)),
    )

    assert ok is True
    assert dispatched == []
    saved = store.load("menlo")
    assert saved["g1"]["transcription_status"] == "done"


def test_transcribe_church_dispatches_when_feed_published_at_is_recent_even_if_preached_on_is_not(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(net, "now", lambda: "2026-09-15T12:00:00+00:00")
    records = {"g1": _record("g1", preached_on="2026-01-01")}
    records["g1"]["feed_published_at"] = "2026-09-15T10:00:00+00:00"
    dispatched = []

    def fake_transcribe_audio(url, hotwords, known_audio):
        return "the transcript", "hash123", _PROFILE, _AUDIO

    ok = transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"])],
        transcribe_audio=fake_transcribe_audio,
        push=lambda files: None,
        fetch_existing=_nothing_in_content,
        dispatch_ingest_event=lambda event, source: dispatched.append((event, source)),
    )

    assert ok is True
    assert len(dispatched) == 1


def test_ingest_event_carries_the_preached_date_and_feed_instant_as_separate_fields(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(net, "now", lambda: "2026-09-15T12:00:00+00:00")
    records = {"g1": _record("g1", preached_on="2026-09-13")}
    records["g1"]["feed_published_at"] = "2026-09-14T09:30:00-07:00"
    dispatched = []

    transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"])],
        transcribe_audio=lambda url, hotwords, known_audio: ("the transcript", "hash123", _PROFILE, _AUDIO),
        push=lambda files: None,
        fetch_existing=_nothing_in_content,
        dispatch_ingest_event=lambda event, source: dispatched.append((event, source)),
    )

    event, _ = dispatched[0]
    assert event["preached_on"] == "2026-09-13"
    assert event["feed_published_at"] == "2026-09-14T09:30:00-07:00"
    assert "published_at" not in event


def test_ingest_event_sends_a_null_feed_instant_rather_than_the_preached_date(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(net, "now", lambda: "2026-09-15T12:00:00+00:00")
    records = {"g1": _record("g1", preached_on="2026-09-13")}
    dispatched = []

    transcriber.transcribe_church(
        "pbc",
        records,
        [("g1", records["g1"])],
        transcribe_audio=lambda url, hotwords, known_audio: ("the transcript", "hash123", _PROFILE, _AUDIO),
        push=lambda files: None,
        fetch_existing=_nothing_in_content,
        dispatch_ingest_event=lambda event, source: dispatched.append((event, source)),
    )

    event, _ = dispatched[0]
    assert event["preached_on"] == "2026-09-13"
    assert event["feed_published_at"] is None


def test_transcribe_church_leaves_batch_pending_when_push_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {"g1": _record("g1", preached_on="2026-01-01")}

    def fake_transcribe_audio(url, hotwords, known_audio):
        return "text", "hash", _PROFILE, _AUDIO

    def failing_push(files):
        raise ContentPublishError("content repo unreachable")

    ok = transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"])],
        transcribe_audio=fake_transcribe_audio,
        push=failing_push,
        fetch_existing=_nothing_in_content,
    )

    assert ok is False
    saved = store.load("menlo")
    assert saved["g1"]["transcription_status"] is None


def test_transcribe_church_leaves_a_download_failure_pending_not_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {"g1": _record("g1", preached_on="2026-01-01")}

    def failing_transcribe_audio(url, hotwords, known_audio):
        raise AudioDownloadError("cdn rejected")

    ok = transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"])],
        transcribe_audio=failing_transcribe_audio,
        push=lambda files: None,
        fetch_existing=_nothing_in_content,
    )

    assert ok is True
    saved = store.load("menlo")
    assert saved["g1"]["transcription_status"] is None
    assert saved["g1"]["transcription_last_error"] == "audio_download"


@pytest.mark.parametrize(
    ("error", "reason"),
    [
        (EmptyTranscriptError("nothing but silence"), "empty_transcript"),
        (transcribe.TranscriptionError("whisper crashed twice"), "transcription"),
    ],
)
def test_transcribe_church_records_why_a_transcription_failed(tmp_path, monkeypatch, error, reason):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {"g1": _record("g1", preached_on="2026-01-01")}

    def failing_transcribe_audio(url, hotwords, known_audio):
        raise error

    transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"])],
        transcribe_audio=failing_transcribe_audio,
        push=lambda files: None,
        fetch_existing=_nothing_in_content,
    )

    assert store.load("menlo")["g1"]["transcription_last_error"] == reason


def test_transcribe_church_records_a_whisper_transcript_as_not_adopted(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {"g1": _record("g1", preached_on="2026-01-01")}

    transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"])],
        transcribe_audio=lambda url, hotwords, known_audio: ("the transcript", "hash123", _PROFILE, _AUDIO),
        push=lambda files: None,
        fetch_existing=_nothing_in_content,
    )

    assert store.load("menlo")["g1"]["transcript_adopted"] is False


def test_transcribe_church_marks_a_terminal_model_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {"g1": _record("g1", preached_on="2026-01-01")}

    def failing_transcribe_audio(url, hotwords, known_audio):
        raise EmptyTranscriptError("nothing but silence")

    ok = transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"])],
        transcribe_audio=failing_transcribe_audio,
        push=lambda files: None,
        fetch_existing=_nothing_in_content,
    )

    assert ok is False
    saved = store.load("menlo")
    assert saved["g1"]["transcription_status"] == "failed"


def test_run_caps_each_church_independently_not_a_shared_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    store.save(
        "menlo",
        {"m1": _record("m1", preached_on="2026-01-01"), "m2": _record("m2", preached_on="2026-01-02")},
    )
    store.save(
        "pbc",
        {"p1": _record("p1", preached_on="2026-01-01"), "p2": _record("p2", preached_on="2026-01-02")},
    )

    transcribed = []

    def fake_transcribe_audio(url, hotwords, known_audio):
        transcribed.append(url)
        return "text", "hash", _PROFILE, _AUDIO

    transcriber.run(
        church_names=None,
        limit=1,
        transcribe_audio=fake_transcribe_audio,
        push=lambda files: None,
        fetch_existing=_nothing_in_content,
    )
    # limit=1 caps each church at 1, not the pair combined at 1 — a shared budget
    # would let the first church (menlo) exhaust it and leave pbc untouched. Each
    # church's newest sermon (m2/p2) is the one selected, not its oldest.
    assert len(transcribed) == 2
    assert store.load("menlo")["m2"]["transcription_status"] == "done"
    assert store.load("menlo")["m1"]["transcription_status"] is None
    assert store.load("pbc")["p2"]["transcription_status"] == "done"
    assert store.load("pbc")["p1"]["transcription_status"] is None


def test_run_is_a_noop_on_a_second_run_against_already_transcribed_records(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    store.save("menlo", {"m1": _record("m1", preached_on="2026-01-01")})

    calls = []

    def fake_transcribe_audio(url, hotwords, known_audio):
        calls.append(url)
        return "text", "hash", _PROFILE, _AUDIO

    transcriber.run(
        church_names=["menlo"],
        limit=5,
        transcribe_audio=fake_transcribe_audio,
        push=lambda files: None,
        fetch_existing=_nothing_in_content,
    )
    assert len(calls) == 1

    transcriber.run(
        church_names=["menlo"],
        limit=5,
        transcribe_audio=fake_transcribe_audio,
        push=lambda files: None,
        fetch_existing=_nothing_in_content,
    )
    assert len(calls) == 1


def test_run_default_sharding_is_a_noop_matching_unsharded_order(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path / "unsharded")
    store.save(
        "menlo",
        {"m1": _record("m1", preached_on="2026-01-01"), "m2": _record("m2", preached_on="2026-01-02")},
    )
    store.save("pbc", {"p1": _record("p1", preached_on="2026-01-01")})

    unsharded_calls: list[str] = []

    def fake_unsharded(url, hotwords, known_audio):
        unsharded_calls.append(url)
        return "text", "hash", _PROFILE, _AUDIO

    transcriber.run(
        church_names=None,
        limit=5,
        transcribe_audio=fake_unsharded,
        push=lambda files: None,
        fetch_existing=_nothing_in_content,
    )

    monkeypatch.setattr(store, "DATA_DIR", tmp_path / "sharded")
    store.save(
        "menlo",
        {"m1": _record("m1", preached_on="2026-01-01"), "m2": _record("m2", preached_on="2026-01-02")},
    )
    store.save("pbc", {"p1": _record("p1", preached_on="2026-01-01")})

    sharded_calls: list[str] = []

    def fake_sharded(url, hotwords, known_audio):
        sharded_calls.append(url)
        return "text", "hash", _PROFILE, _AUDIO

    transcriber.run(
        church_names=None,
        limit=5,
        shard_index=0,
        shard_count=1,
        transcribe_audio=fake_sharded,
        push=lambda files: None,
        fetch_existing=_nothing_in_content,
    )

    assert len(unsharded_calls) == 3
    assert sharded_calls == unsharded_calls


def test_run_shards_partition_every_pending_sermon_exactly_once(tmp_path, monkeypatch):
    shard_count = 3
    seen: list[str] = []
    for shard_index in range(shard_count):
        base = tmp_path / f"shard-{shard_index}"
        base.mkdir()
        monkeypatch.setattr(store, "DATA_DIR", base)
        store.save(
            "menlo",
            {
                "m1": _record("m1", preached_on="2026-01-01"),
                "m2": _record("m2", preached_on="2026-01-02"),
                "m3": _record("m3", preached_on="2026-01-03"),
            },
        )
        store.save(
            "pbc",
            {"p1": _record("p1", preached_on="2026-01-01"), "p2": _record("p2", preached_on="2026-01-02")},
        )

        def fake_transcribe_audio(url, hotwords, known_audio, _seen=seen):
            _seen.append(url)
            return "text", "hash", _PROFILE, _AUDIO

        transcriber.run(
            church_names=None,
            limit=5,
            shard_index=shard_index,
            shard_count=shard_count,
            transcribe_audio=fake_transcribe_audio,
            push=lambda files: None,
            fetch_existing=_nothing_in_content,
        )

    assert len(seen) == 5


def test_run_shard_count_more_than_pending_leaves_extra_shards_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    store.save("menlo", {"m1": _record("m1", preached_on="2026-01-01")})

    calls = []

    def fake_transcribe_audio(url, hotwords, known_audio):
        calls.append(url)
        return "text", "hash", _PROFILE, _AUDIO

    ok = transcriber.run(
        church_names=None,
        limit=5,
        shard_index=1,
        shard_count=5,
        transcribe_audio=fake_transcribe_audio,
        push=lambda files: None,
        fetch_existing=_nothing_in_content,
    )

    assert ok is True
    assert calls == []
    assert store.load("menlo")["m1"]["transcription_status"] is None


def test_main_rejects_invalid_shard_arguments():
    with pytest.raises(SystemExit):
        transcriber.main(["--shard-count", "0"])
    with pytest.raises(SystemExit):
        transcriber.main(["--shard-index", "2", "--shard-count", "2"])
    with pytest.raises(SystemExit):
        transcriber.main(["--shard-index", "-1"])


def test_count_in_scope_matches_the_number_of_sermons_run_would_process(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    store.save(
        "menlo",
        {"m1": _record("m1", preached_on="2026-01-01"), "m2": _record("m2", preached_on="2026-01-02")},
    )
    store.save("pbc", {"p1": _record("p1", preached_on="2026-01-01")})

    assert transcriber.count_in_scope(church_names=None, limit=1) == 2
    assert transcriber.count_in_scope(church_names=None, limit=5) == 3
    assert transcriber.count_in_scope(church_names=["pbc"], limit=5) == 1


def test_count_in_scope_is_zero_when_nothing_is_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {"m1": _record("m1", preached_on="2026-01-01")}
    store.mark_transcribed(
        records["m1"],
        content_path="x",
        transcript_hash="h",
        transcribed_at="t",
        profile={"model": "large-v3", "domain_prompt": True},
    )
    store.save("menlo", records)

    assert transcriber.count_in_scope(church_names=None, limit=5) == 0


def test_main_print_shard_count_prints_the_count_and_transcribes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    store.save("menlo", {"m1": _record("m1", preached_on="2026-01-01")})

    exit_code = transcriber.main(["--print-shard-count"])

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "1"
    assert store.load("menlo")["m1"]["transcription_status"] is None


def test_run_narrows_to_the_named_church(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    store.save("menlo", {"m1": _record("m1", preached_on="2026-01-01")})
    store.save("pbc", {"p1": _record("p1", preached_on="2026-01-01")})

    calls = []

    def fake_transcribe_audio(url, hotwords, known_audio):
        calls.append(url)
        return "text", "hash", _PROFILE, _AUDIO

    transcriber.run(
        church_names=["menlo"],
        limit=5,
        transcribe_audio=fake_transcribe_audio,
        push=lambda files: None,
        fetch_existing=_nothing_in_content,
    )
    assert len(calls) == 1
    assert store.load("pbc")["p1"]["transcription_status"] is None


def test_transcribe_church_records_a_sermon_already_in_content_without_retranscribing_it(
    tmp_path, monkeypatch
):
    # Regression (issue #84): a pending sermon whose transcript already landed in
    # Content (a queued run reading a stale ledger, or a ledger mark lost after the
    # Content push) was re-transcribed, then refused by the overwrite guard — failing
    # every run forever. Content's copy is authoritative; the ledger adopts it.
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {"g1": _record("g1", preached_on="2019-01-06")}
    path = "transcripts/menlo/2019-01-06_sermon-g1_g1.txt"
    whisper_calls = []
    pushed = []

    ok = transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"])],
        transcribe_audio=lambda url, hotwords, known_audio: (
            whisper_calls.append(url) or ("new text", "new-hash", _PROFILE, _AUDIO)
        ),
        push=pushed.append,
        fetch_existing=lambda paths: {path: "the transcript already in Content"},
    )

    assert ok is True
    assert whisper_calls == []
    assert pushed == []
    saved = store.load("menlo")["g1"]
    assert saved["transcription_status"] == "done"
    assert saved["content_path"] == path
    assert saved["transcript_hash"] == transcribe.transcript_hash("the transcript already in Content")
    # ADR-0012: null means "unknown" — this run didn't produce the text, so it can't vouch for any of it.
    assert all(saved[f"transcription_{key}"] is None for key in store.PROFILE_KEYS)


def test_transcribe_church_asks_content_about_exactly_the_batch_s_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {"g1": _record("g1", preached_on="2026-01-01"), "g2": _record("g2", preached_on="2025-12-01")}
    asked = []

    transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"]), ("g2", records["g2"])],
        transcribe_audio=lambda url, hotwords, known_audio: ("text", "hash", _PROFILE, _AUDIO),
        push=lambda files: None,
        fetch_existing=lambda paths: asked.append(sorted(paths)) or {},
    )

    assert asked == [
        ["transcripts/menlo/2025-12-01_sermon-g2_g2.txt", "transcripts/menlo/2026-01-01_sermon-g1_g1.txt"]
    ]


def test_transcribe_church_pushes_only_the_sermons_content_does_not_already_hold(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {"g1": _record("g1", preached_on="2026-01-01"), "g2": _record("g2", preached_on="2025-12-01")}
    pushed = {}

    ok = transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"]), ("g2", records["g2"])],
        transcribe_audio=lambda url, hotwords, known_audio: (
            "fresh transcript",
            "fresh-hash",
            _PROFILE,
            _AUDIO,
        ),
        push=pushed.update,
        fetch_existing=lambda paths: {"transcripts/menlo/2025-12-01_sermon-g2_g2.txt": "older transcript"},
    )

    assert ok is True
    assert pushed == {"transcripts/menlo/2026-01-01_sermon-g1_g1.txt": "fresh transcript"}
    saved = store.load("menlo")
    assert saved["g1"]["transcript_hash"] == "fresh-hash"
    assert saved["g2"]["transcript_hash"] == transcribe.transcript_hash("older transcript")


def test_transcribe_church_never_dispatches_for_a_sermon_already_in_content(tmp_path, monkeypatch):
    # The run that pushed it to Content already dispatched its ingest event.
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(net, "now", lambda: "2026-09-27T21:40:00+00:00")
    records = {"g1": _record("g1", preached_on="2026-09-27")}
    dispatched = []

    transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"])],
        transcribe_audio=lambda url, hotwords, known_audio: ("new text", "new-hash", _PROFILE, _AUDIO),
        push=lambda files: None,
        dispatch_ingest_event=lambda event, source: dispatched.append(event),
        fetch_existing=lambda paths: dict.fromkeys(paths, "already there"),
    )

    assert dispatched == []


def test_transcribe_church_leaves_the_batch_pending_when_content_cannot_be_read(tmp_path, monkeypatch):
    # Without knowing what Content holds, transcribing risks a second, different
    # version of a sermon — so nothing is transcribed and nothing is marked.
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {"g1": _record("g1", preached_on="2026-01-01")}
    whisper_calls = []

    def unreadable(paths):
        raise ContentPublishError("content read failed after 3 attempts: clone timed out")

    ok = transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"])],
        transcribe_audio=lambda url, hotwords, known_audio: (
            whisper_calls.append(url) or ("text", "hash", _PROFILE, _AUDIO)
        ),
        push=lambda files: None,
        fetch_existing=unreadable,
    )

    assert ok is False
    assert whisper_calls == []
    assert records["g1"]["transcription_status"] is None


def test_run_threads_fetch_existing_through_to_each_church(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    store.save("menlo", {"m1": _record("m1", preached_on="2026-01-01")})
    whisper_calls = []

    ok = transcriber.run(
        church_names=["menlo"],
        limit=5,
        transcribe_audio=lambda url, hotwords, known_audio: (
            whisper_calls.append(url) or ("text", "hash", _PROFILE, _AUDIO)
        ),
        push=lambda files: None,
        fetch_existing=lambda paths: dict.fromkeys(paths, "already there"),
    )

    assert ok is True
    assert whisper_calls == []
    assert store.load("menlo")["m1"]["transcription_status"] == "done"


# ADR-0023: score every transcript before push, record the counts, alert on a flag.

_CLEAN_TEXT = (
    "This morning we are continuing in the book of Romans, chapter eight. Paul has spent seven "
    "chapters telling us what God has done, and now he turns to what that means for the way we live."
)
# "We lead with mercy." x12 adds 44 loop words: over the 40-word flag.
_LOOPED_TEXT = _CLEAN_TEXT + " " + "We lead with mercy. " * 12 + "And that is the gospel."


def _transcribe_returning(text: str, profile: transcribe.DecodingProfile = _PROFILE):
    return lambda url, hotwords, known_audio: (text, transcribe.transcript_hash(text), profile, _AUDIO)


def _collecting_alerts(sent: list):
    return lambda church, flagged: sent.append((church, flagged))


def test_a_looped_transcript_is_still_pushed_marked_done_dispatched_and_alerted(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(net, "now", lambda: "2026-09-15T12:00:00+00:00")
    records = {"g1": _record("g1", preached_on="2026-09-15")}
    pushed, dispatched, alerts = {}, [], []

    ok = transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"])],
        transcribe_audio=_transcribe_returning(_LOOPED_TEXT),
        push=pushed.update,
        fetch_existing=_nothing_in_content,
        dispatch_ingest_event=lambda event, source: dispatched.append(event["external_id"]),
        send_quality_alert=_collecting_alerts(alerts),
    )

    assert ok is True
    assert list(pushed.values()) == [_LOOPED_TEXT]
    assert dispatched == ["g1"]
    saved = store.load("menlo")["g1"]
    assert saved["transcription_status"] == "done"
    assert saved["transcript_loop_words"] == 44
    assert saved["transcript_words"] == 36 + 48 + 5
    assert [church for church, _flagged in alerts] == ["menlo"]
    [(_church, [flagged])] = alerts
    assert flagged.guid == "g1"
    assert flagged.title == "Sermon g1"
    assert flagged.preached_on == "2026-09-15"
    assert flagged.flags == ("loops",)
    assert flagged.score.loop_words == 44


def test_a_clean_transcript_records_its_counts_and_sends_no_alert(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {"g1": _record("g1", preached_on="2026-01-01")}
    alerts: list = []

    ok = transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"])],
        transcribe_audio=_transcribe_returning(_CLEAN_TEXT),
        push=lambda files: None,
        fetch_existing=_nothing_in_content,
        send_quality_alert=_collecting_alerts(alerts),
    )

    assert ok is True
    assert alerts == []
    saved = store.load("menlo")["g1"]
    assert {key: saved[f"transcript_{key}"] for key in store.QUALITY_KEYS} == {
        "words": 36,
        "loop_words": 0,
        "credit_hits": 0,
        "hotword_pairs": 0,
    }


@pytest.mark.parametrize(("sent_hotwords", "pairs"), [("Keith Crosby, Luke", 1), (None, 0)])
def test_hotword_pairs_are_scored_only_against_hotwords_the_model_was_sent(
    tmp_path, monkeypatch, sent_hotwords, pairs
):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {"g1": _record("g1", preached_on="2026-01-01") | {"speaker": "Keith Crosby", "series": "Luke"}}
    profile = dataclasses.replace(_PROFILE, domain_prompt=sent_hotwords is not None, hotwords=sent_hotwords)

    transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"])],
        transcribe_audio=_transcribe_returning(_CLEAN_TEXT + " Keith Crosby, Luke.", profile),
        push=lambda files: None,
        fetch_existing=_nothing_in_content,
        send_quality_alert=lambda church, flagged: None,
    )

    assert store.load("menlo")["g1"]["transcript_hotword_pairs"] == pairs


def test_a_transcript_adopted_from_content_is_neither_scored_nor_alerted(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {"g1": _record("g1", preached_on="2019-01-06")}
    alerts: list = []

    transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"])],
        transcribe_audio=_transcribe_returning(_LOOPED_TEXT),
        push=lambda files: None,
        fetch_existing=lambda paths: dict.fromkeys(paths, _LOOPED_TEXT),
        send_quality_alert=_collecting_alerts(alerts),
    )

    assert alerts == []
    assert store.transcript_quality(store.load("menlo")["g1"]) is None


def test_a_transcript_whose_push_failed_is_neither_recorded_nor_alerted(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {"g1": _record("g1", preached_on="2026-01-01")}
    alerts: list = []

    def failing_push(files):
        raise ContentPublishError("content repo unreachable")

    ok = transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"])],
        transcribe_audio=_transcribe_returning(_LOOPED_TEXT),
        push=failing_push,
        fetch_existing=_nothing_in_content,
        send_quality_alert=_collecting_alerts(alerts),
    )

    assert ok is False
    assert alerts == []
    assert store.transcript_quality(store.load("menlo")["g1"]) is None


def test_an_undelivered_alert_fails_the_run_but_keeps_the_push_and_the_mark(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {"g1": _record("g1", preached_on="2026-01-01")}
    pushed: dict = {}

    def failing_alert(church, flagged):
        raise notify.NotifyError("Resend rejected the request: 422")

    ok = transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"])],
        transcribe_audio=_transcribe_returning(_LOOPED_TEXT),
        push=pushed.update,
        fetch_existing=_nothing_in_content,
        send_quality_alert=failing_alert,
    )

    assert ok is False
    assert list(pushed.values()) == [_LOOPED_TEXT]
    assert store.load("menlo")["g1"]["transcription_status"] == "done"
    assert "quality alert" in caplog.text
    assert "422" in caplog.text


def test_missing_notify_secrets_fail_the_run_when_a_transcript_is_flagged(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    for var in ("NOTIFY_EMAIL_FROM", "NOTIFY_EMAIL_TO", "RESEND_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    records = {"g1": _record("g1", preached_on="2026-01-01")}

    ok = transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"])],
        transcribe_audio=_transcribe_returning(_LOOPED_TEXT),
        push=lambda files: None,
        fetch_existing=_nothing_in_content,
    )

    assert ok is False
    assert store.load("menlo")["g1"]["transcription_status"] == "done"
    assert "RESEND_API_KEY" in caplog.text


def test_run_threads_the_alert_sender_through_to_each_church(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    store.save("pbc", {"p1": _record("p1", preached_on="2026-01-01")})
    alerts: list = []

    transcriber.run(
        church_names=["pbc"],
        limit=1,
        transcribe_audio=_transcribe_returning(_LOOPED_TEXT),
        push=lambda files: None,
        fetch_existing=_nothing_in_content,
        send_quality_alert=_collecting_alerts(alerts),
    )

    assert [church for church, _flagged in alerts] == ["pbc"]


def test_no_transcript_text_reaches_the_logs_or_the_alert(tmp_path, monkeypatch, caplog):
    # This repo's Actions logs are public (ADR-0004): counts and names only.
    caplog.set_level("DEBUG", logger="poller")
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {"g1": _record("g1", preached_on="2026-01-01")}
    alerts: list = []
    sentinel = "Zebulun carried the lanterns across the brook. "

    transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"])],
        transcribe_audio=_transcribe_returning(_CLEAN_TEXT + " " + sentinel * 7),
        push=lambda files: None,
        fetch_existing=_nothing_in_content,
        send_quality_alert=_collecting_alerts(alerts),
    )

    assert len(alerts) == 1
    assert "Zebulun" not in caplog.text
    assert "Zebulun" not in repr(alerts)


# Re-posted audio (ADR-0028). North Point re-posted 2026-10-18's sermon under new guids, each
# with its own enclosure URL but the same file byte for byte.

_ORIGINAL = "north_point:9e1EFEED61D064276595A0B4B931"
_REPOST = "north_point:47F507bB27DC524dc65B128CD49F"
_REPOSTED_AUDIO = hashlib.sha256(b"all good things, part 1").hexdigest()
_ORIGINAL_TEXT = "So, if you haven't heard already, maybe you came in late, today's a little bit different."
_ORIGINAL_COUNTS = {"words": 6423, "loop_words": 0, "credit_hits": 0, "hotword_pairs": 0}


def _np_record(guid: str) -> dict:
    record = _record(
        guid,
        preached_on="2026-10-18",
        audio_url=f"https://storage.sardius.media/archives/{guid.split(':')[1]}/media/sermon.mp3",
    )
    record["title"] = "All Good Things, Part 1: All Good Things"
    return record


def _np_transcribed(
    guid: str, *, audio_sha256: str | None = _REPOSTED_AUDIO, copied_from: str | None = None
) -> dict:
    record = _np_record(guid)
    store.mark_transcribed(
        record,
        content_path=transcriber._content_path("north_point", record),
        transcript_hash=transcribe.transcript_hash(_ORIGINAL_TEXT),
        transcribed_at="2026-10-18T20:40:00+00:00",
        profile=dataclasses.asdict(_PROFILE),
        quality=_ORIGINAL_COUNTS,
        audio_sha256=audio_sha256,
        copied_from=copied_from,
    )
    return record


def _whisper_by_audio(audio: dict[str, str], text: dict[str, str], calls: list):
    """A ``transcribe_audio`` that behaves like the real one: same bytes as a known hash, no Whisper."""

    def fake(url, hotwords, known_audio):
        calls.append((url, dict(known_audio)))
        if audio[url] in known_audio:
            raise transcribe.DuplicateAudioError(audio[url], known_audio[audio[url]])
        return text[url], transcribe.transcript_hash(text[url]), _PROFILE, audio[url]

    return fake


def test_a_repost_of_audio_already_transcribed_reuses_its_transcript_without_whisper(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(net, "now", lambda: "2026-10-18T21:00:00+00:00")
    records = {_ORIGINAL: _np_transcribed(_ORIGINAL), _REPOST: _np_record(_REPOST)}
    original_path = records[_ORIGINAL]["content_path"]
    repost_path = transcriber._content_path("north_point", records[_REPOST])
    calls: list = []
    pushed: dict = {}
    dispatched: list = []
    alerts: list = []

    ok = transcriber.transcribe_church(
        "north_point",
        records,
        [(_REPOST, records[_REPOST])],
        transcribe_audio=_whisper_by_audio({records[_REPOST]["audio_url"]: _REPOSTED_AUDIO}, {}, calls),
        push=pushed.update,
        fetch_existing=lambda paths: {p: _ORIGINAL_TEXT for p in paths if p == original_path},
        dispatch_ingest_event=lambda event, source: dispatched.append(event),
        send_quality_alert=_collecting_alerts(alerts),
    )

    assert ok is True
    assert calls == [(records[_REPOST]["audio_url"], {_REPOSTED_AUDIO: _ORIGINAL})]
    assert pushed == {repost_path: _ORIGINAL_TEXT}
    saved = store.load("north_point")[_REPOST]
    assert saved["transcription_status"] == "done"
    assert saved["content_path"] == repost_path
    assert saved["transcript_hash"] == transcribe.transcript_hash(_ORIGINAL_TEXT)
    assert saved["transcript_copied_from"] == _ORIGINAL
    assert saved["audio_sha256"] == _REPOSTED_AUDIO
    assert saved["transcript_adopted"] is False
    assert {key: saved[f"transcription_{key}"] for key in store.PROFILE_KEYS} == dataclasses.asdict(_PROFILE)
    assert {key: saved[f"transcript_{key}"] for key in store.QUALITY_KEYS} == _ORIGINAL_COUNTS
    assert [event["external_id"] for event in dispatched] == [_REPOST]
    assert dispatched[0]["transcript"]["transcript_hash"] == transcribe.transcript_hash(_ORIGINAL_TEXT)
    assert alerts == []


def test_reposts_in_one_batch_run_whisper_once(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    first, second = _ORIGINAL, _REPOST
    records = {first: _np_record(first), second: _np_record(second)}
    audio = {records[first]["audio_url"]: _REPOSTED_AUDIO, records[second]["audio_url"]: _REPOSTED_AUDIO}
    calls: list = []
    pushed: dict = {}
    alerts: list = []

    ok = transcriber.transcribe_church(
        "north_point",
        records,
        [(first, records[first]), (second, records[second])],
        transcribe_audio=_whisper_by_audio(audio, {records[first]["audio_url"]: _LOOPED_TEXT}, calls),
        push=pushed.update,
        fetch_existing=_nothing_in_content,
        send_quality_alert=_collecting_alerts(alerts),
    )

    assert ok is True
    assert calls[1] == (records[second]["audio_url"], {_REPOSTED_AUDIO: first})
    assert list(pushed.values()) == [_LOOPED_TEXT, _LOOPED_TEXT]
    saved = store.load("north_point")
    assert saved[first]["audio_sha256"] == _REPOSTED_AUDIO
    assert saved[first]["transcript_copied_from"] is None
    assert saved[second]["transcript_copied_from"] == first
    assert saved[second]["transcript_hash"] == saved[first]["transcript_hash"]
    assert store.transcript_quality(saved[second]) == store.transcript_quality(saved[first])
    # The looped original is alerted on once; its copy is not alerted again.
    assert [[f.guid for f in flagged] for _church, flagged in alerts] == [[first]]


def test_a_repost_whose_original_is_gone_from_content_is_transcribed_after_all(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {_ORIGINAL: _np_transcribed(_ORIGINAL), _REPOST: _np_record(_REPOST)}
    url = records[_REPOST]["audio_url"]
    calls: list = []
    pushed: dict = {}

    ok = transcriber.transcribe_church(
        "north_point",
        records,
        [(_REPOST, records[_REPOST])],
        transcribe_audio=_whisper_by_audio({url: _REPOSTED_AUDIO}, {url: "the repost, transcribed"}, calls),
        push=pushed.update,
        fetch_existing=_nothing_in_content,
    )

    assert ok is True
    assert calls == [(url, {_REPOSTED_AUDIO: _ORIGINAL}), (url, {})]
    assert list(pushed.values()) == ["the repost, transcribed"]
    saved = store.load("north_point")[_REPOST]
    assert saved["transcript_copied_from"] is None
    assert saved["audio_sha256"] == _REPOSTED_AUDIO
    assert _ORIGINAL in caplog.text


def test_a_repost_stays_pending_when_content_cannot_be_read_for_its_original(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {_ORIGINAL: _np_transcribed(_ORIGINAL), _REPOST: _np_record(_REPOST)}
    original_path = records[_ORIGINAL]["content_path"]
    pushed: dict = {}

    def content_down_for_the_original(paths):
        if original_path in paths:
            raise ContentPublishError("content repo unreachable")
        return {}

    ok = transcriber.transcribe_church(
        "north_point",
        records,
        [(_REPOST, records[_REPOST])],
        transcribe_audio=_whisper_by_audio({records[_REPOST]["audio_url"]: _REPOSTED_AUDIO}, {}, []),
        push=pushed.update,
        fetch_existing=content_down_for_the_original,
    )

    assert ok is False
    assert pushed == {}
    saved = store.load("north_point")[_REPOST]
    assert saved["transcription_status"] is None
    assert saved["transcript_copied_from"] is None


def test_known_audio_names_only_done_transcripts_that_were_not_themselves_copied():
    copy = "north_point:BFF77112EF464803B69032F1FFF8"
    older = "north_point:248D1e023fBD11EF0A0f7E9533f0"
    records = {
        _ORIGINAL: _np_transcribed(_ORIGINAL),
        copy: _np_transcribed(copy, copied_from=_ORIGINAL),
        older: _np_transcribed(older, audio_sha256=None),
        _REPOST: _np_record(_REPOST),
    }
    assert transcriber._known_audio(records) == {_REPOSTED_AUDIO: _ORIGINAL}


def _hashing(audio: dict[str, str], calls: list):
    """A ``hash_audio`` over ``audio``, recording each enclosure it was asked to download."""

    def fake(url):
        calls.append(url)
        return audio[url]

    return fake


def test_a_repost_of_a_transcript_with_no_audio_hash_is_caught_without_whisper(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(net, "now", lambda: "2026-10-18T21:00:00+00:00")
    records = {_ORIGINAL: _np_transcribed(_ORIGINAL, audio_sha256=None), _REPOST: _np_record(_REPOST)}
    original_path = records[_ORIGINAL]["content_path"]
    repost_path = transcriber._content_path("north_point", records[_REPOST])
    original_url = records[_ORIGINAL]["audio_url"]
    repost_url = records[_REPOST]["audio_url"]
    calls: list = []
    hashed: list = []
    pushed: dict = {}

    ok = transcriber.transcribe_church(
        "north_point",
        records,
        [(_REPOST, records[_REPOST])],
        transcribe_audio=_whisper_by_audio({repost_url: _REPOSTED_AUDIO}, {}, calls),
        hash_audio=_hashing({original_url: _REPOSTED_AUDIO}, hashed),
        push=pushed.update,
        fetch_existing=lambda paths: {p: _ORIGINAL_TEXT for p in paths if p == original_path},
        dispatch_ingest_event=lambda event, source: None,
    )

    assert ok is True
    assert hashed == [original_url]
    assert calls == [(repost_url, {_REPOSTED_AUDIO: _ORIGINAL})]
    assert pushed == {repost_path: _ORIGINAL_TEXT}
    saved = store.load("north_point")
    assert saved[_REPOST]["transcript_copied_from"] == _ORIGINAL
    assert saved[_ORIGINAL]["audio_sha256"] == _REPOSTED_AUDIO


def test_a_same_titled_sermon_with_different_audio_is_transcribed_and_the_older_hash_is_kept(
    tmp_path, monkeypatch
):
    # A church's edited re-upload: same title and date, different bytes, different words.
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(net, "now", lambda: "2026-10-18T21:00:00+00:00")
    records = {_ORIGINAL: _np_transcribed(_ORIGINAL, audio_sha256=None), _REPOST: _np_record(_REPOST)}
    older_audio = hashlib.sha256(b"before the edit").hexdigest()
    repost_url = records[_REPOST]["audio_url"]
    calls: list = []
    pushed: dict = {}

    transcriber.transcribe_church(
        "north_point",
        records,
        [(_REPOST, records[_REPOST])],
        transcribe_audio=_whisper_by_audio(
            {repost_url: _REPOSTED_AUDIO}, {repost_url: "The edited sermon."}, calls
        ),
        hash_audio=lambda url: older_audio,
        push=pushed.update,
        fetch_existing=_nothing_in_content,
        dispatch_ingest_event=lambda event, source: None,
    )

    assert list(pushed.values()) == ["The edited sermon."]
    saved = store.load("north_point")
    assert saved[_REPOST]["transcript_copied_from"] is None
    assert saved[_ORIGINAL]["audio_sha256"] == older_audio


def test_an_expired_enclosure_on_the_older_record_leaves_the_sermon_to_be_transcribed(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(net, "now", lambda: "2026-10-18T21:00:00+00:00")
    records = {_ORIGINAL: _np_transcribed(_ORIGINAL, audio_sha256=None), _REPOST: _np_record(_REPOST)}
    repost_url = records[_REPOST]["audio_url"]
    pushed: dict = {}

    def expired(url):
        raise AudioDownloadError("HTTP Error 404: Not Found")

    ok = transcriber.transcribe_church(
        "north_point",
        records,
        [(_REPOST, records[_REPOST])],
        transcribe_audio=_whisper_by_audio({repost_url: _REPOSTED_AUDIO}, {repost_url: "The sermon."}, []),
        hash_audio=expired,
        push=pushed.update,
        fetch_existing=_nothing_in_content,
        dispatch_ingest_event=lambda event, source: None,
    )

    assert ok is True
    assert list(pushed.values()) == ["The sermon."]
    assert store.load("north_point")[_ORIGINAL]["audio_sha256"] is None


def test_only_done_unhashed_originals_with_the_same_title_are_downloaded(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(net, "now", lambda: "2026-10-18T21:00:00+00:00")
    hashed_already = "north_point:AAAA"
    copy = "north_point:BBBB"
    other_title = "north_point:CCCC"
    pending = "north_point:DDDD"
    no_enclosure = "north_point:EEEE"
    records = {
        _ORIGINAL: _np_transcribed(_ORIGINAL, audio_sha256=None),
        hashed_already: _np_transcribed(hashed_already, audio_sha256=_AUDIO),
        copy: _np_transcribed(copy, audio_sha256=None, copied_from=_ORIGINAL),
        other_title: _np_transcribed(other_title, audio_sha256=None) | {"title": "Mind Games, Part 3"},
        pending: _np_record(pending),
        no_enclosure: _np_transcribed(no_enclosure, audio_sha256=None) | {"audio_url": None},
        _REPOST: _np_record(_REPOST),
    }
    repost_url = records[_REPOST]["audio_url"]
    hashed: list = []

    transcriber.transcribe_church(
        "north_point",
        records,
        [(_REPOST, records[_REPOST])],
        transcribe_audio=_whisper_by_audio({repost_url: _REPOSTED_AUDIO}, {repost_url: "The sermon."}, []),
        hash_audio=_hashing({records[_ORIGINAL]["audio_url"]: hashlib.sha256(b"older").hexdigest()}, hashed),
        push=lambda files: None,
        fetch_existing=_nothing_in_content,
        dispatch_ingest_event=lambda event, source: None,
    )

    assert hashed == [records[_ORIGINAL]["audio_url"]]


def test_a_sermon_with_no_title_downloads_nothing_extra(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(net, "now", lambda: "2026-10-18T21:00:00+00:00")
    records = {
        _ORIGINAL: _np_transcribed(_ORIGINAL, audio_sha256=None) | {"title": None},
        _REPOST: _np_record(_REPOST) | {"title": None},
    }
    repost_url = records[_REPOST]["audio_url"]
    hashed: list = []

    transcriber.transcribe_church(
        "north_point",
        records,
        [(_REPOST, records[_REPOST])],
        transcribe_audio=_whisper_by_audio({repost_url: _REPOSTED_AUDIO}, {repost_url: "The sermon."}, []),
        hash_audio=_hashing({}, hashed),
        push=lambda files: None,
        fetch_existing=_nothing_in_content,
        dispatch_ingest_event=lambda event, source: None,
    )

    assert hashed == []


def test_run_threads_hash_audio_through_to_each_church(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    monkeypatch.setattr(net, "now", lambda: "2026-10-18T21:00:00+00:00")
    older = _record("g1", preached_on="2026-10-11", audio_url="https://example.org/older.mp3")
    store.mark_transcribed(
        older,
        content_path="transcripts/menlo/older.txt",
        transcript_hash="h",
        transcribed_at="t",
        profile=None,
    )
    newer = _record("g2", preached_on="2026-10-18", audio_url="https://example.org/newer.mp3")
    newer["title"] = older["title"]
    store.save("menlo", {"g1": older, "g2": newer})
    hashed: list = []

    transcriber.run(
        church_names=["menlo"],
        limit=1,
        transcribe_audio=lambda url, hotwords, known_audio: ("text", "hash", _PROFILE, _AUDIO),
        hash_audio=_hashing({"https://example.org/older.mp3": _REPOSTED_AUDIO}, hashed),
        push=lambda files: None,
        fetch_existing=_nothing_in_content,
        dispatch_ingest_event=lambda event, source: None,
    )

    assert hashed == ["https://example.org/older.mp3"]
