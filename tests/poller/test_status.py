"""`status/ingest.json`: Ingest's half of the web report (spec 0009, Pipeline spec 0029).

A pure projection of every church's ledger plus the church table, pushed to Content
only when it changed. These pin the contract the web app joins against.
"""

import json

import pytest

from poller import config, content_repo, status, store
from poller.sources.base import SermonItem

_TABLE = (
    '{"menlo": {"rss": "https://feed.podbean.com/menlochurchvideo/feed.xml",'
    ' "pipeline": {"enabled": true}, "ingest": {"enabled": true}},'
    ' "hillside": {"rss": "https://podcasts.subsplash.com/82xq2z3/podcast.rss",'
    ' "pipeline": {"enabled": false}, "ingest": {"enabled": false}}}'
)

_CONTRACT_KEYS = {
    "guid",
    "source",
    "title",
    "preached_on",
    "feed_published_at",
    "first_seen_at",
    "audio_available",
    "transcription_status",
    "transcribed_at",
    "transcript_adopted",
    "last_error",
}


@pytest.fixture(autouse=True)
def churches_env(church_table, tmp_path, monkeypatch):
    church_table(_TABLE)
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)


def _record(guid: str, *, preached_on: str, audio_url: str | None = None) -> dict:
    item = SermonItem(
        guid=guid,
        title="The Purity of Reality",
        raw_title="The Purity of Reality | Repentance For the Rest Of Us | Mark Swarner",
        series="Repentance For the Rest Of Us",
        speaker="Mark Swarner",
        preached_on=preached_on,
        feed_published_at=None,
        episode_url="https://menlochurchvideo.podbean.com/e/the-purity-of-reality/",
        audio_url=audio_url
        if audio_url is not None
        else "https://mcdn.podbean.com/mf/web/abc/2026-09-27_SermonPodcastAudio.mp3",
        blurb="Text Our Team (650) 600-0402",
    )
    return store.item_to_record(
        item, first_seen_at="2026-09-27T20:12:24+00:00", feed_published_at="2026-09-27T13:08:11-07:00"
    )


def _only(payload: dict, guid: str) -> dict:
    [entry] = [s for s in payload["sermons"] if s["guid"] == guid]
    return entry


def test_a_transcribed_sermon_carries_its_instants_and_no_error():
    record = _record("menlochurchvideo.podbean.com/3a5d9a10", preached_on="2026-09-27")
    store.mark_transcribed(
        record,
        content_path="transcripts/menlo/2026-09-27_the-purity-of-reality_x.txt",
        transcript_hash="2ce57804",
        transcribed_at="2026-09-27T21:34:56+00:00",
        model="large-v3",
        domain_prompt=True,
    )
    store.save("menlo", {record["guid"]: record})

    entry = _only(status.build_status(config.load_churches()), record["guid"])

    assert entry == {
        "guid": "menlochurchvideo.podbean.com/3a5d9a10",
        "source": "menlo",
        "title": "The Purity of Reality",
        "preached_on": "2026-09-27",
        "feed_published_at": "2026-09-27T13:08:11-07:00",
        "first_seen_at": "2026-09-27T20:12:24+00:00",
        "audio_available": True,
        "transcription_status": "done",
        "transcribed_at": "2026-09-27T21:34:56+00:00",
        "transcript_adopted": False,
        "last_error": None,
    }


def test_a_failing_download_is_reported_with_its_reason():
    record = _record("menlochurchvideo.podbean.com/failing", preached_on="2026-10-04")
    store.mark_download_failed(record)
    store.save("menlo", {record["guid"]: record})

    entry = _only(status.build_status(config.load_churches()), record["guid"])

    assert entry["transcription_status"] is None
    assert entry["last_error"] == "audio_download"


def test_a_record_with_no_fetchable_enclosure_has_no_audio():
    """Westgate's audio-less items are never attempted, so the URL is the only evidence."""
    record = _record("menlochurchvideo.podbean.com/no-audio", preached_on="2026-10-04", audio_url="")
    store.save("menlo", {record["guid"]: record})

    entry = _only(status.build_status(config.load_churches()), record["guid"])

    assert entry["audio_available"] is False


def test_a_record_saved_before_failure_reasons_existed_reads_as_unknown():
    record = _record("menlochurchvideo.podbean.com/legacy", preached_on="2025-01-05")
    del record["transcription_last_error"]
    del record["transcript_adopted"]
    store.save("menlo", {record["guid"]: record})

    entry = _only(status.build_status(config.load_churches()), record["guid"])

    assert entry["last_error"] is None
    assert entry["transcript_adopted"] is None


def test_every_church_in_the_table_is_listed_with_its_enablement_and_records():
    store.save("hillside", {"hillside:e7dc32": {**_record("hillside:e7dc32", preached_on="2026-09-13")}})

    payload = status.build_status(config.load_churches())

    assert payload["churches"] == {
        "menlo": {"ingest_enabled": True},
        "hillside": {"ingest_enabled": False},
    }
    assert [s["source"] for s in payload["sermons"]] == ["hillside"]


def test_nothing_but_the_contract_crosses_into_content():
    record = _record("menlochurchvideo.podbean.com/3a5d9a10", preached_on="2026-09-27")
    store.save("menlo", {record["guid"]: record})

    payload = status.build_status(config.load_churches())

    assert set(payload) == {"churches", "sermons"}
    assert set(payload["sermons"][0]) == _CONTRACT_KEYS


def test_an_unchanged_ledger_renders_byte_identical_status():
    """No timestamp: a tick that changed nothing must render the same bytes, so the push no-ops."""
    record = _record("menlochurchvideo.podbean.com/3a5d9a10", preached_on="2026-09-27")
    store.save("menlo", {record["guid"]: record})

    first = status.render_status(config.load_churches())
    second = status.render_status(config.load_churches())

    assert first == second
    assert json.loads(first)["sermons"][0]["guid"] == record["guid"]
    assert first.endswith("\n")


def test_main_pushes_the_rendered_file_to_its_contract_path():
    record = _record("menlochurchvideo.podbean.com/3a5d9a10", preached_on="2026-09-27")
    store.save("menlo", {record["guid"]: record})
    pushed: list[dict[str, str]] = []

    assert status.main([], push=pushed.append) == 0

    [files] = pushed
    assert list(files) == ["status/ingest.json"]
    assert json.loads(files["status/ingest.json"])["sermons"][0]["guid"] == record["guid"]


def test_main_exits_non_zero_when_the_push_never_lands():
    def refuse(files):
        raise content_repo.ContentPublishError("content push failed after 3 attempts: rejected")

    assert status.main([], push=refuse) == 1
