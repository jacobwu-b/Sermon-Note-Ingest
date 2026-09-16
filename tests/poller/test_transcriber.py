import pytest

from poller import pipeline_dispatch, store, transcriber
from poller.content_repo import ContentPublishError
from poller.net import AudioDownloadError
from poller.sources.base import SermonItem
from poller.transcribe import EmptyTranscriptError


def _item(guid: str, *, published_on: str, audio_url: str = "https://example.org/a.mp3") -> SermonItem:
    return SermonItem(
        guid=guid,
        title=f"Sermon {guid}",
        raw_title=f"Sermon {guid}",
        series=None,
        speaker=None,
        published_on=published_on,
        published_at=None,
        episode_url="https://example.org/ep",
        audio_url=audio_url,
        blurb="",
    )


def _record(guid: str, *, published_on: str, audio_url: str = "https://example.org/a.mp3") -> dict:
    return store.item_to_record(
        _item(guid, published_on=published_on, audio_url=audio_url), first_seen_at="t", published_at=None
    )


@pytest.fixture(autouse=True)
def churches_env(monkeypatch):
    monkeypatch.setenv(
        "CHURCHES",
        '{"menlo": {"rss": "https://example.org/menlo.xml", "enabled": true},'
        ' "pbc": {"rss": "https://example.org/pbc.xml", "enabled": true}}',
    )


def test_select_pending_skips_done_failed_and_unfetchable():
    records = {
        "done": _record("done", published_on="2026-01-01"),
        "failed": _record("failed", published_on="2026-01-02"),
        "no_audio": _record("no_audio", published_on="2026-01-03", audio_url=""),
        "pending": _record("pending", published_on="2026-01-04"),
    }
    store.mark_transcribed(records["done"], content_path="x", transcript_hash="h", transcribed_at="t")
    store.mark_transcription_failed(records["failed"])

    pending = transcriber._select_pending(records)
    assert [guid for guid, _r in pending] == ["pending"]


def test_select_pending_orders_newest_published_on_first():
    records = {
        "new": _record("new", published_on="2026-09-06"),
        "old": _record("old", published_on="2026-01-04"),
        "mid": _record("mid", published_on="2026-05-01"),
    }
    pending = transcriber._select_pending(records)
    assert [guid for guid, _r in pending] == ["new", "mid", "old"]


def test_select_pending_sorts_undated_records_last_regardless_of_direction():
    records = {
        "undated": _record("undated", published_on=""),
        "new": _record("new", published_on="2026-09-06"),
        "old": _record("old", published_on="2026-01-04"),
    }
    pending = transcriber._select_pending(records)
    assert [guid for guid, _r in pending] == ["new", "old", "undated"]


def test_content_path_is_deterministic_for_the_same_guid():
    record = _record("guid-1", published_on="2026-09-06")
    record["title"] = "Hear and Do"
    path_a = transcriber._content_path("menlo", record)
    path_b = transcriber._content_path("menlo", record)
    assert path_a == path_b
    assert path_a == "transcripts/menlo/2026-09-06_hear-and-do_guid-1.txt"


def test_transcribe_church_marks_success_only_after_push_succeeds(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {"g1": _record("g1", published_on="2026-01-01")}
    pushed = {}

    def fake_transcribe_audio(url):
        return "the transcript", "hash123"

    def fake_push(files):
        pushed.update(files)

    ok = transcriber.transcribe_church(
        "menlo", records, [("g1", records["g1"])], transcribe_audio=fake_transcribe_audio, push=fake_push
    )

    assert ok is True
    assert pushed == {"transcripts/menlo/2026-01-01_sermon-g1_g1.txt": "the transcript"}
    saved = store.load("menlo")
    assert saved["g1"]["transcription_status"] == "done"
    assert saved["g1"]["transcript_hash"] == "hash123"
    assert saved["g1"]["content_path"] == "transcripts/menlo/2026-01-01_sermon-g1_g1.txt"


def test_transcribe_church_dispatches_an_ingest_event_per_successfully_transcribed_item(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {"g1": _record("g1", published_on="2026-09-15")}
    dispatched = []

    def fake_transcribe_audio(url):
        return "the transcript", "hash123"

    ok = transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"])],
        transcribe_audio=fake_transcribe_audio,
        push=lambda files: None,
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


def test_transcribe_church_survives_a_dispatch_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {"g1": _record("g1", published_on="2026-09-15")}

    def fake_transcribe_audio(url):
        return "the transcript", "hash123"

    def failing_dispatch(event, source):
        raise pipeline_dispatch.PipelineDispatchError("pipeline unreachable")

    ok = transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"])],
        transcribe_audio=fake_transcribe_audio,
        push=lambda files: None,
        dispatch_ingest_event=failing_dispatch,
    )

    assert ok is True
    saved = store.load("menlo")
    assert saved["g1"]["transcription_status"] == "done"


def test_transcribe_church_never_dispatches_for_an_item_whose_push_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {"g1": _record("g1", published_on="2026-09-15")}
    dispatched = []

    def fake_transcribe_audio(url):
        return "text", "hash"

    def failing_push(files):
        raise ContentPublishError("content repo unreachable")

    ok = transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"])],
        transcribe_audio=fake_transcribe_audio,
        push=failing_push,
        dispatch_ingest_event=lambda event, source: dispatched.append((event, source)),
    )

    assert ok is False
    assert dispatched == []


def test_transcribe_church_never_dispatches_for_a_backfill_sermon(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {"g1": _record("g1", published_on="2026-01-01")}
    dispatched = []

    def fake_transcribe_audio(url):
        return "the transcript", "hash123"

    ok = transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"])],
        transcribe_audio=fake_transcribe_audio,
        push=lambda files: None,
        dispatch_ingest_event=lambda event, source: dispatched.append((event, source)),
    )

    assert ok is True
    assert dispatched == []
    saved = store.load("menlo")
    assert saved["g1"]["transcription_status"] == "done"


def test_transcribe_church_dispatches_when_published_at_is_recent_even_if_published_on_is_not(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {"g1": _record("g1", published_on="2026-01-01")}
    records["g1"]["published_at"] = "2026-09-15T10:00:00+00:00"
    dispatched = []

    def fake_transcribe_audio(url):
        return "the transcript", "hash123"

    ok = transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"])],
        transcribe_audio=fake_transcribe_audio,
        push=lambda files: None,
        dispatch_ingest_event=lambda event, source: dispatched.append((event, source)),
    )

    assert ok is True
    assert len(dispatched) == 1


def test_transcribe_church_leaves_batch_pending_when_push_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {"g1": _record("g1", published_on="2026-01-01")}

    def fake_transcribe_audio(url):
        return "text", "hash"

    def failing_push(files):
        raise ContentPublishError("content repo unreachable")

    ok = transcriber.transcribe_church(
        "menlo", records, [("g1", records["g1"])], transcribe_audio=fake_transcribe_audio, push=failing_push
    )

    assert ok is False
    saved = store.load("menlo")
    assert saved["g1"]["transcription_status"] is None


def test_transcribe_church_leaves_a_download_failure_pending_not_failed(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {"g1": _record("g1", published_on="2026-01-01")}

    def failing_transcribe_audio(url):
        raise AudioDownloadError("cdn rejected")

    ok = transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"])],
        transcribe_audio=failing_transcribe_audio,
        push=lambda files: None,
    )

    assert ok is True
    saved = store.load("menlo")
    assert saved["g1"]["transcription_status"] is None


def test_transcribe_church_marks_a_terminal_model_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {"g1": _record("g1", published_on="2026-01-01")}

    def failing_transcribe_audio(url):
        raise EmptyTranscriptError("nothing but silence")

    ok = transcriber.transcribe_church(
        "menlo",
        records,
        [("g1", records["g1"])],
        transcribe_audio=failing_transcribe_audio,
        push=lambda files: None,
    )

    assert ok is False
    saved = store.load("menlo")
    assert saved["g1"]["transcription_status"] == "failed"


def test_run_caps_each_church_independently_not_a_shared_budget(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    store.save(
        "menlo",
        {"m1": _record("m1", published_on="2026-01-01"), "m2": _record("m2", published_on="2026-01-02")},
    )
    store.save(
        "pbc",
        {"p1": _record("p1", published_on="2026-01-01"), "p2": _record("p2", published_on="2026-01-02")},
    )

    transcribed = []

    def fake_transcribe_audio(url):
        transcribed.append(url)
        return "text", "hash"

    transcriber.run(
        church_names=None, limit=1, transcribe_audio=fake_transcribe_audio, push=lambda files: None
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
    store.save("menlo", {"m1": _record("m1", published_on="2026-01-01")})

    calls = []

    def fake_transcribe_audio(url):
        calls.append(url)
        return "text", "hash"

    transcriber.run(
        church_names=["menlo"], limit=5, transcribe_audio=fake_transcribe_audio, push=lambda files: None
    )
    assert len(calls) == 1

    transcriber.run(
        church_names=["menlo"], limit=5, transcribe_audio=fake_transcribe_audio, push=lambda files: None
    )
    assert len(calls) == 1


def test_run_default_sharding_is_a_noop_matching_unsharded_order(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path / "unsharded")
    store.save(
        "menlo",
        {"m1": _record("m1", published_on="2026-01-01"), "m2": _record("m2", published_on="2026-01-02")},
    )
    store.save("pbc", {"p1": _record("p1", published_on="2026-01-01")})

    unsharded_calls: list[str] = []

    def fake_unsharded(url):
        unsharded_calls.append(url)
        return "text", "hash"

    transcriber.run(church_names=None, limit=5, transcribe_audio=fake_unsharded, push=lambda files: None)

    monkeypatch.setattr(store, "DATA_DIR", tmp_path / "sharded")
    store.save(
        "menlo",
        {"m1": _record("m1", published_on="2026-01-01"), "m2": _record("m2", published_on="2026-01-02")},
    )
    store.save("pbc", {"p1": _record("p1", published_on="2026-01-01")})

    sharded_calls: list[str] = []

    def fake_sharded(url):
        sharded_calls.append(url)
        return "text", "hash"

    transcriber.run(
        church_names=None,
        limit=5,
        shard_index=0,
        shard_count=1,
        transcribe_audio=fake_sharded,
        push=lambda files: None,
    )

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
                "m1": _record("m1", published_on="2026-01-01"),
                "m2": _record("m2", published_on="2026-01-02"),
                "m3": _record("m3", published_on="2026-01-03"),
            },
        )
        store.save(
            "pbc",
            {"p1": _record("p1", published_on="2026-01-01"), "p2": _record("p2", published_on="2026-01-02")},
        )

        def fake_transcribe_audio(url, _seen=seen):
            _seen.append(url)
            return "text", "hash"

        transcriber.run(
            church_names=None,
            limit=5,
            shard_index=shard_index,
            shard_count=shard_count,
            transcribe_audio=fake_transcribe_audio,
            push=lambda files: None,
        )

    assert len(seen) == 5


def test_run_shard_count_more_than_pending_leaves_extra_shards_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    store.save("menlo", {"m1": _record("m1", published_on="2026-01-01")})

    calls = []

    def fake_transcribe_audio(url):
        calls.append(url)
        return "text", "hash"

    ok = transcriber.run(
        church_names=None,
        limit=5,
        shard_index=1,
        shard_count=5,
        transcribe_audio=fake_transcribe_audio,
        push=lambda files: None,
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
        {"m1": _record("m1", published_on="2026-01-01"), "m2": _record("m2", published_on="2026-01-02")},
    )
    store.save("pbc", {"p1": _record("p1", published_on="2026-01-01")})

    assert transcriber.count_in_scope(church_names=None, limit=1) == 2
    assert transcriber.count_in_scope(church_names=None, limit=5) == 3
    assert transcriber.count_in_scope(church_names=["pbc"], limit=5) == 1


def test_count_in_scope_is_zero_when_nothing_is_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    records = {"m1": _record("m1", published_on="2026-01-01")}
    store.mark_transcribed(records["m1"], content_path="x", transcript_hash="h", transcribed_at="t")
    store.save("menlo", records)

    assert transcriber.count_in_scope(church_names=None, limit=5) == 0


def test_main_print_shard_count_prints_the_count_and_transcribes_nothing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    store.save("menlo", {"m1": _record("m1", published_on="2026-01-01")})

    exit_code = transcriber.main(["--print-shard-count"])

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == "1"
    assert store.load("menlo")["m1"]["transcription_status"] is None


def test_run_narrows_to_the_named_church(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    store.save("menlo", {"m1": _record("m1", published_on="2026-01-01")})
    store.save("pbc", {"p1": _record("p1", published_on="2026-01-01")})

    calls = []

    def fake_transcribe_audio(url):
        calls.append(url)
        return "text", "hash"

    transcriber.run(
        church_names=["menlo"], limit=5, transcribe_audio=fake_transcribe_audio, push=lambda files: None
    )
    assert len(calls) == 1
    assert store.load("pbc")["p1"]["transcription_status"] is None
