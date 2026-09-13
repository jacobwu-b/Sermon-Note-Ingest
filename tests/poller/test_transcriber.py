import pytest

from poller import store, transcriber
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


def test_select_pending_orders_oldest_published_on_first():
    records = {
        "new": _record("new", published_on="2026-09-06"),
        "old": _record("old", published_on="2026-01-04"),
        "mid": _record("mid", published_on="2026-05-01"),
    }
    pending = transcriber._select_pending(records)
    assert [guid for guid, _r in pending] == ["old", "mid", "new"]


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


def test_run_caps_total_sermons_across_churches_not_per_church(tmp_path, monkeypatch):
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
        church_names=None, limit=3, transcribe_audio=fake_transcribe_audio, push=lambda files: None
    )
    assert len(transcribed) == 3


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
