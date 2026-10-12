"""`transcripts/index.json`: each transcript's metadata for the web corpus (spec 0010).

A pure projection of every church's done records. Web loads each ``path`` as written
and fails its build on a malformed file, so these pin the shape exactly.
"""

import json

import pytest

from poller import config, store, transcript_index
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
    "preached_on",
    "title",
    "speaker",
    "series",
    "episode_url",
    "blurb",
    "path",
}


@pytest.fixture(autouse=True)
def churches_env(church_table, tmp_path, monkeypatch):
    church_table(_TABLE)
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)


def _record(guid: str, *, preached_on: str = "2026-09-27", **overrides: str | None) -> dict:
    fields: dict = {
        "title": "The Purity of Reality",
        "raw_title": "The Purity of Reality | Repentance For the Rest Of Us | Mark Swarner",
        "series": "Repentance For the Rest Of Us",
        "speaker": "Mark Swarner",
        "episode_url": "https://menlochurchvideo.podbean.com/e/the-purity-of-reality/",
        "blurb": "Text Our Team (650) 600-0402",
        **overrides,
    }
    item = SermonItem(
        guid=guid,
        preached_on=preached_on,
        feed_published_at=None,
        audio_url="https://mcdn.podbean.com/mf/web/abc/2026-09-27_SermonPodcastAudio.mp3",
        **fields,
    )
    return store.item_to_record(
        item, first_seen_at="2026-09-27T20:12:24+00:00", feed_published_at="2026-09-27T13:08:11-07:00"
    )


def _transcribed(record: dict, content_path: str, *, adopted: bool = False) -> dict:
    store.mark_transcribed(
        record,
        content_path=content_path,
        transcript_hash="2ce57804",
        transcribed_at="2026-09-27T21:34:56+00:00",
        profile=None if adopted else {"model": "large-v3", "domain_prompt": False},
        adopted=adopted,
    )
    return record


def _guids(payload: dict) -> list[str]:
    return [entry["guid"] for entry in payload["transcripts"]]


def test_a_transcribed_sermon_is_indexed_with_its_metadata_and_content_path():
    path = "transcripts/menlo/2026-09-27_the-purity-of-reality_menlochurchvideo.podbean.com_3a5d9a10.txt"
    record = _transcribed(_record("menlochurchvideo.podbean.com/3a5d9a10"), path)
    store.save("menlo", {record["guid"]: record})

    payload = transcript_index.build_index(config.load_churches())

    assert payload == {
        "transcripts": [
            {
                "guid": "menlochurchvideo.podbean.com/3a5d9a10",
                "source": "menlo",
                "preached_on": "2026-09-27",
                "title": "The Purity of Reality",
                "speaker": "Mark Swarner",
                "series": "Repentance For the Rest Of Us",
                "episode_url": "https://menlochurchvideo.podbean.com/e/the-purity-of-reality/",
                "blurb": "Text Our Team (650) 600-0402",
                "path": path,
            }
        ]
    }


def test_only_done_records_are_indexed():
    done = _transcribed(_record("menlochurchvideo.podbean.com/done"), "transcripts/menlo/done.txt")
    adopted = _transcribed(
        _record("menlochurchvideo.podbean.com/adopted"), "transcripts/menlo/adopted.txt", adopted=True
    )
    pending = _record("menlochurchvideo.podbean.com/pending")
    download_failed = _record("menlochurchvideo.podbean.com/no-download")
    store.mark_download_failed(download_failed)
    failed = _record("menlochurchvideo.podbean.com/failed")
    store.mark_transcription_failed(failed, error=store.EMPTY_TRANSCRIPT_ERROR)
    store.save("menlo", {r["guid"]: r for r in (done, adopted, pending, download_failed, failed)})

    payload = transcript_index.build_index(config.load_churches())

    assert sorted(_guids(payload)) == [
        "menlochurchvideo.podbean.com/adopted",
        "menlochurchvideo.podbean.com/done",
    ]


def test_a_church_with_ingest_disabled_still_has_its_transcripts_indexed():
    """Content keeps a disabled church's transcripts, so the index must still name them."""
    record = _transcribed(_record("hillside:e7dc32"), "transcripts/hillside/2026-09-13_x_hillside_e7dc32.txt")
    store.save("hillside", {record["guid"]: record})

    payload = transcript_index.build_index(config.load_churches())

    assert [(e["source"], e["path"]) for e in payload["transcripts"]] == [
        ("hillside", "transcripts/hillside/2026-09-13_x_hillside_e7dc32.txt")
    ]


def test_absent_metadata_is_null_never_a_missing_key():
    """PBC titles and most gracepres speakers are absent; Web needs the key regardless."""
    record = _transcribed(
        _record(
            "https://pbc.org?enmse_mid=4696",
            title=None,
            speaker=None,
            series=None,
            episode_url=None,
            blurb=None,
        ),
        "transcripts/menlo/2026-10-04_untitled_https_pbc.org_enmse_mid_4696.txt",
    )
    del record["series"]  # a record saved before the field existed
    store.save("menlo", {record["guid"]: record})

    [entry] = transcript_index.build_index(config.load_churches())["transcripts"]

    assert set(entry) == _CONTRACT_KEYS
    assert {k: entry[k] for k in ("title", "speaker", "series", "episode_url", "blurb")} == {
        "title": None,
        "speaker": None,
        "series": None,
        "episode_url": None,
        "blurb": None,
    }


def test_nothing_but_the_contract_crosses_into_content():
    """Metadata only (ADR-0004): hashes, model and ledger bookkeeping stay out."""
    record = _transcribed(_record("menlochurchvideo.podbean.com/3a5d9a10"), "transcripts/menlo/x.txt")
    store.save("menlo", {record["guid"]: record})

    payload = transcript_index.build_index(config.load_churches())

    assert set(payload) == {"transcripts"}
    assert set(payload["transcripts"][0]) == _CONTRACT_KEYS


def test_a_done_record_without_a_content_path_refuses_to_render():
    """Web loads every path as written; an index naming no file must never be published."""
    record = _transcribed(_record("menlochurchvideo.podbean.com/broken"), "transcripts/menlo/broken.txt")
    del record["content_path"]
    store.save("menlo", {record["guid"]: record})

    with pytest.raises(KeyError):
        transcript_index.build_index(config.load_churches())


def test_records_follow_church_table_order_then_ledger_order():
    menlo = [
        _transcribed(_record(f"menlochurchvideo.podbean.com/{n}"), f"transcripts/menlo/{n}.txt")
        for n in ("b", "a")
    ]
    hillside = _transcribed(_record("hillside:e7dc32"), "transcripts/hillside/e7dc32.txt")
    store.save("hillside", {hillside["guid"]: hillside})
    store.save("menlo", {r["guid"]: r for r in menlo})
    ledger_order = list(store.load("menlo"))

    payload = transcript_index.build_index(config.load_churches())

    assert _guids(payload) == [*ledger_order, "hillside:e7dc32"]


def test_an_unchanged_ledger_renders_byte_identical_index():
    """No timestamp: a tick that changed nothing must render the same bytes, so the push no-ops."""
    record = _transcribed(_record("menlochurchvideo.podbean.com/3a5d9a10"), "transcripts/menlo/x.txt")
    store.save("menlo", {record["guid"]: record})

    first = transcript_index.render_index(config.load_churches())
    second = transcript_index.render_index(config.load_churches())

    assert first == second
    assert json.loads(first)["transcripts"][0]["path"] == "transcripts/menlo/x.txt"
    assert first.endswith("\n")


def test_an_empty_ledger_renders_an_empty_list_not_a_missing_file():
    assert json.loads(transcript_index.render_index(config.load_churches())) == {"transcripts": []}
