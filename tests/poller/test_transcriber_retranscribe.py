"""``--retranscribe-stale`` (ADR-0022): selection, the per-sermon flow, marks and replay.

Synthetic sermon-like text only; real transcripts never enter this repo (ADR-0004).
"""

import dataclasses
import json
import logging

import pytest

from poller import content_repo, pipeline_dispatch, quality, store, transcribe, transcriber
from poller.net import AudioDownloadError
from poller.sources.base import SermonItem

_RUNNING = {
    "model": "large-v3",
    "compute_type": "int8",
    "beam_size": 5,
    "condition_on_previous_text": False,
    "hallucination_silence_threshold": 2.0,
    "vad_filter": True,
    "domain_prompt": False,
}
_RUNNING_DIGEST = store.profile_digest(_RUNNING)
_NEW_PROFILE = transcribe.DecodingProfile(
    **_RUNNING, hotwords=None, faster_whisper_version="1.2.1", ctranslate2_version="4.8.2"
)
_OLD_PROFILE = {
    **_RUNNING,
    "condition_on_previous_text": True,
    "hallucination_silence_threshold": 0,
    "domain_prompt": True,
    "hotwords": None,
    "faster_whisper_version": None,
    "ctranslate2_version": None,
}

_SENTINEL = "Zebulon marmalade covenant"


def _sentence(n: int) -> str:
    return f"Verse {n} reminds us that the gospel of grace reaches people we did not expect. "


def _speech(sentences: int) -> str:
    """Sermon-like prose with no back-to-back repeat, so it scores no loop words."""
    return "".join(_sentence(n) for n in range(sentences))


_OLD_TEXT = f"{_SENTINEL}. " + _speech(20) + "We lead with mercy. " * 30
_NEW_TEXT = f"{_SENTINEL}. " + _speech(21)
_OLD_HASH = transcribe.transcript_hash(_OLD_TEXT)
_NEW_HASH = transcribe.transcript_hash(_NEW_TEXT)
_CURRENT_FIELDS = (
    "transcription_status",
    "content_path",
    "transcript_hash",
    "transcribed_at",
    "transcript_adopted",
    "superseded_transcript_hash",
    *(f"transcription_{key}" for key in store.PROFILE_KEYS),
    *(f"transcript_{key}" for key in store.QUALITY_KEYS),
)


@pytest.fixture(autouse=True)
def running_profile(monkeypatch, church_table, tmp_path):
    for key, value in {
        "WHISPER_MODEL": "large-v3",
        "WHISPER_COMPUTE_TYPE": "int8",
        "WHISPER_BEAM_SIZE": "5",
        "WHISPER_CONDITION_ON_PREVIOUS_TEXT": "false",
        "WHISPER_HALLUCINATION_SILENCE_THRESHOLD": "2.0",
        "WHISPER_DOMAIN_PROMPT": "false",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    church_table(
        '{"menlo": {"rss": "https://example.org/menlo.xml", "ingest": {"enabled": true}},'
        ' "pbc": {"rss": "https://example.org/pbc.xml", "ingest": {"enabled": true}}}'
    )

    def no_dispatch(*_a, **_k):
        raise AssertionError("re-transcription never dispatches Pipeline")

    monkeypatch.setattr(pipeline_dispatch, "dispatch_ingest_event", no_dispatch)


def _record(
    guid: str, *, preached_on: str = "2026-09-06", audio_url: str = "https://example.org/a.mp3"
) -> dict:
    item = SermonItem(
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
    return store.item_to_record(item, first_seen_at="t", feed_published_at=None)


def _done(guid: str, *, profile: dict | None = None, preached_on: str = "2026-09-06", **kw) -> dict:
    record = _record(guid, preached_on=preached_on, **kw)
    store.mark_transcribed(
        record,
        content_path=f"transcripts/menlo/{preached_on}_sermon-{guid}_{guid}.txt",
        transcript_hash=_OLD_HASH,
        transcribed_at="2026-09-20T12:00:00+00:00",
        profile=_OLD_PROFILE if profile is None else profile,
    )
    return record


# --- Selection -----------------------------------------------------------------------------


def _selected(records: dict) -> list[str]:
    return [guid for guid, _r in transcriber.select_stale(records, running=_RUNNING, digest=_RUNNING_DIGEST)]


def test_a_record_produced_under_the_running_profile_is_not_selected():
    assert _selected({"g": _done("g", profile=dataclasses.asdict(_NEW_PROFILE))}) == []


_OLDER_KNOBS = {
    "model": "small",
    "compute_type": "float32",
    "beam_size": 1,
    "condition_on_previous_text": True,
    "hallucination_silence_threshold": 0,
    "vad_filter": False,
    "domain_prompt": True,
}


@pytest.mark.parametrize("key", store.STALENESS_KEYS)
def test_a_record_differing_in_one_decoding_knob_is_selected(key):
    old = {**dataclasses.asdict(_NEW_PROFILE), key: _OLDER_KNOBS[key]}
    assert _selected({"g": _done("g", profile=old)}) == ["g"]


def test_a_record_missing_a_profile_field_is_selected():
    record = _done("g", profile=dataclasses.asdict(_NEW_PROFILE))
    del record["transcription_vad_filter"]
    assert _selected({"g": record}) == ["g"]


def test_an_adopted_transcript_with_an_unknown_profile_is_selected():
    record = _record("g")
    store.mark_transcribed(
        record,
        content_path="transcripts/menlo/x.txt",
        transcript_hash="h",
        transcribed_at="t",
        profile=None,
        adopted=True,
    )
    assert _selected({"g": record}) == ["g"]


def test_pending_failed_and_audio_less_records_are_never_selected():
    failed = _record("failed")
    store.mark_transcription_failed(failed, error=store.TRANSCRIPTION_ERROR)
    no_audio = _done("no_audio", audio_url="")
    records = {"pending": _record("pending"), "failed": failed, "no_audio": no_audio, "stale": _done("stale")}
    assert _selected(records) == ["stale"]


@pytest.mark.parametrize(
    "error",
    [
        store.TRANSCRIPTION_ERROR,
        store.EMPTY_TRANSCRIPT_ERROR,
        store.QUALITY_REGRESSION_ERROR,
        store.CONTENT_CONFLICT_ERROR,
    ],
)
def test_a_terminal_failure_under_the_running_profile_is_not_selected_again(error):
    record = _done("g")
    store.mark_retranscribe_failed(record, error=error, failed_profile=_RUNNING_DIGEST)
    assert _selected({"g": record}) == []


def test_a_terminal_failure_under_an_older_profile_is_selected_again():
    record = _done("g")
    store.mark_retranscribe_failed(
        record, error=store.QUALITY_REGRESSION_ERROR, failed_profile="an-older-digest"
    )
    assert _selected({"g": record}) == ["g"]


def test_never_attempted_records_come_before_download_retries_then_newest_first():
    retry = _done("retry", preached_on="2026-10-04")
    store.mark_retranscribe_failed(retry, error=store.DOWNLOAD_ERROR, failed_profile=_RUNNING_DIGEST)
    records = {
        "retry": retry,
        "old": _done("old", preached_on="2025-01-05"),
        "new": _done("new", preached_on="2026-09-27"),
        "undated": _done("undated", preached_on=""),
    }
    assert _selected(records) == ["new", "old", "undated", "retry"]


def _save(church: str, records: dict) -> None:
    store.save(church, records)


def test_count_in_scope_caps_stale_records_per_church_and_narrows_by_guid():
    _save("menlo", {g: _done(g, preached_on=f"2026-0{i + 1}-01") for i, g in enumerate(["a", "b", "c"])})
    _save("pbc", {"d": _done("d"), "pending": _record("pending")})

    assert transcriber.count_in_scope(church_names=None, limit=2, retranscribe_stale=True) == 3
    assert (
        transcriber.count_in_scope(church_names=None, limit=5, retranscribe_stale=True, guids=["a", "d"]) == 2
    )
    # The pending path is unchanged: it sees only the never-transcribed record.
    assert transcriber.count_in_scope(church_names=None, limit=5) == 1


# --- The per-sermon flow ------------------------------------------------------------------


class _Content:
    """A fake Content: its files, the last commit's trailers per path, and every write."""

    def __init__(self, files: dict[str, str], trailers: dict[str, dict[str, str]] | None = None):
        self.files = dict(files)
        self.trailers = trailers or {}
        self.replaced: list[tuple[str, str, str, dict]] = []
        self.kept: list[tuple[str, str, dict]] = []
        self.replace_error: Exception | None = None
        self.keep_error: Exception | None = None
        self.read_error: Exception | None = None

    def fetch_existing(self, paths):
        if self.read_error:
            raise self.read_error
        return {p: self.files[p] for p in paths if p in self.files}

    def last_commit_trailers(self, path):
        return self.trailers.get(path, {})

    def replace(self, path, text, *, expect_hash, trailers):
        if self.replace_error:
            raise self.replace_error
        self.replaced.append((path, text, expect_hash, dict(trailers)))
        self.files[path] = text
        self.trailers[path] = dict(trailers)

    def keep_candidate(self, path, text, *, trailers):
        if self.keep_error:
            raise self.keep_error
        self.kept.append((path, text, dict(trailers)))


def _transcribes(text: str, profile=_NEW_PROFILE):
    calls = []

    def fake(url, hotwords):
        calls.append(url)
        return text, transcribe.transcript_hash(text), profile

    fake.calls = calls
    return fake


def _raises(exc):
    calls = []

    def fake(url, hotwords):
        calls.append(url)
        raise exc

    fake.calls = calls
    return fake


def _run(content: _Content, transcribe_audio, *, marks=None, rows=None, guids=None) -> bool:
    return transcriber.run_retranscribe(
        church_names=None,
        limit=5,
        guids=guids,
        transcribe_audio=transcribe_audio,
        fetch_existing=content.fetch_existing,
        last_commit_trailers=content.last_commit_trailers,
        replace=content.replace,
        keep_candidate=content.keep_candidate,
        marks_out=marks,
        summary_rows=rows,
    )


def _one_stale_sermon() -> tuple[dict, str]:
    record = _done("g1")
    _save("menlo", {"g1": record})
    return record, record["content_path"]


def test_a_better_transcript_replaces_the_old_one_by_compare_and_swap():
    _record_before, path = _one_stale_sermon()
    content = _Content({path: _OLD_TEXT})
    fake = _transcribes(_NEW_TEXT)

    assert _run(content, fake) is True

    [(replaced_path, text, expect_hash, _trailers)] = content.replaced
    assert (replaced_path, text, expect_hash) == (path, _NEW_TEXT, _OLD_HASH)
    saved = store.load("menlo")["g1"]
    assert saved["transcript_hash"] == _NEW_HASH
    assert saved["superseded_transcript_hash"] == _OLD_HASH
    assert saved["content_path"] == path
    assert saved["transcription_status"] == "done"
    assert {key: saved[f"transcription_{key}"] for key in store.PROFILE_KEYS} == dataclasses.asdict(
        _NEW_PROFILE
    )
    assert saved["transcript_loop_words"] == 0
    assert saved["transcript_words"] == quality.score(_NEW_TEXT, hotwords=None).words


def test_the_replacement_commit_carries_trailers_that_parse_back_to_the_recorded_profile():
    _one_stale_sermon()
    path = store.load("menlo")["g1"]["content_path"]
    content = _Content({path: _OLD_TEXT})
    _run(content, _transcribes(_NEW_TEXT))
    [(_p, _t, _e, trailers)] = content.replaced
    saved = store.load("menlo")["g1"]

    mark = transcriber._mark_from_trailers(trailers, guid="g1", ledger_hash=_OLD_HASH, content_hash=_NEW_HASH)

    assert mark is not None
    assert mark["profile"] == dataclasses.asdict(_NEW_PROFILE)
    assert mark["transcribed_at"] == saved["transcribed_at"]
    assert mark["quality"] == {key: saved[f"transcript_{key}"] for key in store.QUALITY_KEYS}


def test_identical_text_records_the_profile_without_pushing():
    _record_before, path = _one_stale_sermon()
    content = _Content({path: _OLD_TEXT})

    assert _run(content, _transcribes(_OLD_TEXT)) is True

    assert content.replaced == []
    saved = store.load("menlo")["g1"]
    assert saved["transcript_hash"] == _OLD_HASH
    assert saved["transcription_condition_on_previous_text"] is False
    assert saved.get("superseded_transcript_hash") is None


def test_a_content_commit_whose_ledger_push_was_lost_is_adopted_from_its_trailers_without_whisper():
    _record_before, path = _one_stale_sermon()
    trailers = transcriber._provenance_trailers(
        "g1",
        superseded=_OLD_HASH,
        transcript_hash=_NEW_HASH,
        transcribed_at="2026-10-13T02:00:00+00:00",
        profile=dataclasses.asdict(_NEW_PROFILE),
        counts=dataclasses.asdict(quality.score(_NEW_TEXT, hotwords=None)),
    )
    content = _Content({path: _NEW_TEXT}, {path: trailers})
    fake = _transcribes(_NEW_TEXT)

    assert _run(content, fake) is True

    assert fake.calls == []
    saved = store.load("menlo")["g1"]
    assert saved["transcript_hash"] == _NEW_HASH
    assert saved["superseded_transcript_hash"] == _OLD_HASH
    assert saved["transcribed_at"] == "2026-10-13T02:00:00+00:00"
    assert saved["transcription_hallucination_silence_threshold"] == 2.0


def _failed_without_touching_the_transcript(before: dict, error: str) -> dict:
    saved = store.load("menlo")["g1"]
    assert {f: saved.get(f) for f in _CURRENT_FIELDS} == {f: before.get(f) for f in _CURRENT_FIELDS}
    assert saved["retranscription_last_error"] == error
    assert saved["retranscription_failed_profile"] == _RUNNING_DIGEST
    return saved


def test_content_holding_someone_elses_text_is_a_conflict_and_is_not_overwritten():
    record, path = _one_stale_sermon()
    before = dict(record)
    content = _Content({path: "A hand-corrected transcript of the sermon on Romans eight."})
    fake = _transcribes(_NEW_TEXT)

    assert _run(content, fake) is False

    assert fake.calls == []
    assert content.replaced == []
    _failed_without_touching_the_transcript(before, store.CONTENT_CONFLICT_ERROR)


def test_trailers_naming_a_different_superseded_hash_are_a_conflict():
    record, path = _one_stale_sermon()
    before = dict(record)
    trailers = transcriber._provenance_trailers(
        "g1",
        superseded="some-other-hash",
        transcript_hash=_NEW_HASH,
        transcribed_at="t",
        profile=dataclasses.asdict(_NEW_PROFILE),
        counts=dataclasses.asdict(quality.score(_NEW_TEXT, hotwords=None)),
    )
    content = _Content({path: _NEW_TEXT}, {path: trailers})

    assert _run(content, _transcribes(_NEW_TEXT)) is False
    _failed_without_touching_the_transcript(before, store.CONTENT_CONFLICT_ERROR)


def test_a_transcript_missing_from_content_is_a_conflict():
    record, _path = _one_stale_sermon()
    before = dict(record)
    assert _run(_Content({}), _transcribes(_NEW_TEXT)) is False
    _failed_without_touching_the_transcript(before, store.CONTENT_CONFLICT_ERROR)


def test_a_compare_and_swap_refused_at_push_time_is_a_conflict():
    record, path = _one_stale_sermon()
    before = dict(record)
    content = _Content({path: _OLD_TEXT})
    content.replace_error = content_repo.ContentConflictError("Content moved")

    assert _run(content, _transcribes(_NEW_TEXT)) is False
    _failed_without_touching_the_transcript(before, store.CONTENT_CONFLICT_ERROR)


def test_a_download_failure_is_recorded_and_retried_later_without_failing_the_run():
    record, path = _one_stale_sermon()
    before = dict(record)
    assert _run(_Content({path: _OLD_TEXT}), _raises(AudioDownloadError("timed out"))) is True
    _failed_without_touching_the_transcript(before, store.DOWNLOAD_ERROR)


@pytest.mark.parametrize(
    ("exc", "error"),
    [
        (transcribe.TranscriptionError("ffmpeg could not decode"), store.TRANSCRIPTION_ERROR),
        (transcribe.EmptyTranscriptError("no text"), store.EMPTY_TRANSCRIPT_ERROR),
    ],
)
def test_a_model_failure_keeps_the_old_transcript_and_fails_the_run(exc, error):
    record, path = _one_stale_sermon()
    before = dict(record)
    assert _run(_Content({path: _OLD_TEXT}), _raises(exc)) is False
    _failed_without_touching_the_transcript(before, error)


def test_a_new_transcript_that_trips_a_flag_is_refused_and_kept_as_a_candidate():
    record, path = _one_stale_sermon()
    before = dict(record)
    worse = _NEW_TEXT + " Subtitles by the Amara.org community"
    content = _Content({path: _OLD_TEXT})

    assert _run(content, _transcribes(worse)) is True

    assert content.replaced == []
    [(candidate_path, text, trailers)] = content.kept
    worse_hash = transcribe.transcript_hash(worse)
    assert candidate_path == f"retranscription-candidates/menlo/2026-09-06_sermon-g1_g1.{worse_hash[:12]}.txt"
    assert text == worse
    assert trailers["Ingest-Refusal"] == "credits"
    assert trailers["Ingest-Guid"] == "g1"
    assert trailers["Ingest-New-Quality-Credit-Hits"] == str(quality.credit_hits(worse))
    assert trailers["Ingest-Old-Quality-Loop-Words"] == str(quality.loop_words(_OLD_TEXT))
    saved = _failed_without_touching_the_transcript(before, store.QUALITY_REGRESSION_ERROR)
    assert saved["retranscription_candidate_path"] == candidate_path
    assert saved["retranscription_candidate_hash"] == worse_hash


def test_a_new_transcript_with_under_90_percent_of_the_old_speech_is_refused():
    record, path = _one_stale_sermon()
    before = dict(record)
    shorter = _speech(10)
    content = _Content({path: _OLD_TEXT})

    assert _run(content, _transcribes(shorter)) is True

    [(_p, _t, trailers)] = content.kept
    assert trailers["Ingest-Refusal"] == "clean_words"
    _failed_without_touching_the_transcript(before, store.QUALITY_REGRESSION_ERROR)


def test_the_old_text_is_scored_against_the_hotwords_it_was_produced_with():
    # A hotword-leaking old transcript must not count its leak as speech the new one lost.
    old = "".join(_sentence(n) + "Ready To Serve, Grace Presbyterian. " for n in range(20))
    new = _speech(20)
    record = _done(
        "g1", profile={**_OLD_PROFILE, "hotwords": "Ready To Serve, Grace Presbyterian, David Jones"}
    )
    record["transcript_hash"] = transcribe.transcript_hash(old)
    _save("menlo", {"g1": record})
    content = _Content({record["content_path"]: old})

    assert _run(content, _transcribes(new)) is True
    assert content.kept == []
    assert len(content.replaced) == 1


def test_an_exhausted_replacement_push_leaves_the_ledger_untouched_and_fails_the_run():
    record, path = _one_stale_sermon()
    before = dict(record)
    content = _Content({path: _OLD_TEXT})
    content.replace_error = content_repo.ContentPublishError("content replace failed after 5 attempts")
    marks: dict = {}

    assert _run(content, _transcribes(_NEW_TEXT), marks=marks) is False

    assert store.load("menlo")["g1"] == before
    assert marks.get("menlo", {}) == {}


def test_an_exhausted_candidate_push_leaves_the_ledger_untouched_so_the_next_run_redoes_it():
    record, path = _one_stale_sermon()
    before = dict(record)
    content = _Content({path: _OLD_TEXT})
    content.keep_error = content_repo.ContentPublishError("candidate push failed after 5 attempts")

    assert _run(content, _transcribes(_NEW_TEXT + " Thanks for watching")) is False
    assert store.load("menlo")["g1"] == before


def test_unreadable_content_spends_no_whisper_and_leaves_the_ledger_untouched():
    record, path = _one_stale_sermon()
    before = dict(record)
    content = _Content({path: _OLD_TEXT})
    content.read_error = content_repo.ContentPublishError("content read failed after 3 attempts")
    fake = _transcribes(_NEW_TEXT)

    assert _run(content, fake) is False
    assert fake.calls == []
    assert store.load("menlo")["g1"] == before


def test_guid_narrows_the_pass_to_the_named_sermons():
    _save("menlo", {"a": _done("a", preached_on="2026-09-06"), "b": _done("b", preached_on="2026-09-13")})
    content = _Content(
        {
            _done("a")["content_path"]: _OLD_TEXT,
            _done("b", preached_on="2026-09-13")["content_path"]: _OLD_TEXT,
        }
    )
    fake = _transcribes(_NEW_TEXT)

    assert _run(content, fake, guids=["a"]) is True
    assert [p for p, *_ in content.replaced] == [_done("a")["content_path"]]


# --- Marks and replay ---------------------------------------------------------------------


def _marks_for(content: _Content, transcribe_audio) -> dict:
    marks: dict = {}
    _run(content, transcribe_audio, marks=marks)
    return json.loads(json.dumps(marks))


def test_replaying_a_replacement_mark_onto_a_ledger_that_lost_it_restores_it(tmp_path):
    record, path = _one_stale_sermon()
    before = dict(record)
    marks = _marks_for(_Content({path: _OLD_TEXT}), _transcribes(_NEW_TEXT))
    after = store.load("menlo")["g1"]
    _save("menlo", {"g1": before})
    marks_file = tmp_path / "marks.json"
    marks_file.write_text(json.dumps(marks))

    assert transcriber.replay_marks(str(marks_file)) is True
    assert store.load("menlo")["g1"] == after
    assert transcriber.replay_marks(str(marks_file)) is True
    assert store.load("menlo")["g1"] == after


def test_a_replacement_mark_is_not_replayed_over_a_hash_that_has_since_moved(tmp_path):
    record, path = _one_stale_sermon()
    marks = _marks_for(_Content({path: _OLD_TEXT}), _transcribes(_NEW_TEXT))
    moved = {**record, "transcript_hash": "someone-elses-hash"}
    _save("menlo", {"g1": moved})
    marks_file = tmp_path / "marks.json"
    marks_file.write_text(json.dumps(marks))

    assert transcriber.replay_marks(str(marks_file)) is True
    assert store.load("menlo")["g1"] == moved


def test_a_failure_mark_replays_only_onto_the_hash_the_attempt_saw(tmp_path):
    record, path = _one_stale_sermon()
    content = _Content({path: _OLD_TEXT})
    marks = _marks_for(content, _transcribes(_NEW_TEXT + " Thanks for watching"))
    marks_file = tmp_path / "marks.json"
    marks_file.write_text(json.dumps(marks))

    _save("menlo", {"g1": dict(record)})
    assert transcriber.replay_marks(str(marks_file)) is True
    assert store.load("menlo")["g1"]["retranscription_last_error"] == store.QUALITY_REGRESSION_ERROR

    moved = {**record, "transcript_hash": "someone-elses-hash"}
    _save("menlo", {"g1": moved})
    assert transcriber.replay_marks(str(marks_file)) is True
    assert store.load("menlo")["g1"] == moved


# --- Observability ------------------------------------------------------------------------


@pytest.mark.parametrize(
    "branch",
    ["replaced", "identical", "refused", "conflict", "download", "model", "push_exhausted"],
)
def test_no_log_line_or_summary_row_carries_transcript_text(branch, caplog):
    _record_before, path = _one_stale_sermon()
    content = _Content({path: _OLD_TEXT})
    fake = _transcribes(_NEW_TEXT)
    if branch == "identical":
        fake = _transcribes(_OLD_TEXT)
    elif branch == "refused":
        fake = _transcribes(_NEW_TEXT + " Thanks for watching")
    elif branch == "conflict":
        content.files[path] = f"{_SENTINEL} edited by hand"
    elif branch == "download":
        fake = _raises(AudioDownloadError("404 Not Found"))
    elif branch == "model":
        fake = _raises(transcribe.TranscriptionError("decode failed"))
    elif branch == "push_exhausted":
        content.replace_error = content_repo.ContentPublishError("rejected")
    rows: list = []

    with caplog.at_level(logging.DEBUG, logger="poller"):
        _run(content, fake, rows=rows)

    assert _SENTINEL.lower() not in caplog.text.lower()
    assert _SENTINEL.lower() not in transcriber.render_summary(rows).lower()


def test_the_summary_has_one_row_per_sermon_with_its_outcome():
    _record_before, path = _one_stale_sermon()
    rows: list = []
    _run(_Content({path: _OLD_TEXT}), _transcribes(_NEW_TEXT), rows=rows)

    summary = transcriber.render_summary(rows)

    assert "| menlo | g1 | replaced |" in summary
    assert _OLD_HASH[:12] in summary
    assert _NEW_HASH[:12] in summary


def test_main_counts_stale_sermons_in_scope_for_the_plan_job(capsys):
    _save("menlo", {"a": _done("a"), "b": _done("b", profile=dataclasses.asdict(_NEW_PROFILE))})

    assert transcriber.main(["--retranscribe-stale", "--limit", "5", "--print-shard-count"]) == 0
    assert capsys.readouterr().out.strip() == "1"


def test_main_passes_its_arguments_through_and_writes_the_summary_and_marks_files(tmp_path, monkeypatch):
    calls = []

    def fake_run(**kwargs):
        calls.append(kwargs)
        kwargs["marks_out"].setdefault("menlo", {})["g1"] = {"kind": "retranscribed"}
        kwargs["summary_rows"].append(transcriber.SummaryRow(church="menlo", guid="g1", outcome="replaced"))
        return True

    monkeypatch.setattr(transcriber, "run_retranscribe", fake_run)
    summary = tmp_path / "summary.md"
    marks = tmp_path / "marks.json"

    code = transcriber.main(
        [
            "--retranscribe-stale",
            "--limit",
            "1",
            "--guid",
            "menlo:a, b",
            "--guid",
            "c",
            "--shard-index",
            "1",
            "--shard-count",
            "2",
            "--summary-md",
            str(summary),
            "--marks-out",
            str(marks),
        ]
    )

    assert code == 0
    [kwargs] = calls
    assert (kwargs["limit"], kwargs["guids"], kwargs["shard_index"], kwargs["shard_count"]) == (
        1,
        ["menlo:a, b", "c"],
        1,
        2,
    )
    assert "| menlo | g1 | replaced |" in summary.read_text()
    assert json.loads(marks.read_text()) == {"menlo": {"g1": {"kind": "retranscribed"}}}


def test_main_returns_non_zero_when_the_pass_needs_a_human(tmp_path, monkeypatch):
    monkeypatch.setattr(transcriber, "run_retranscribe", lambda **_k: False)
    assert transcriber.main(["--retranscribe-stale", "--limit", "1"]) == 1
