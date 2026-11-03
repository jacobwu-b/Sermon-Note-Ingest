import json
import subprocess
from pathlib import Path

import pytest

from poller import content_repo

# Content always holds a schema/; this one accepts any document, for tests about
# something other than the schema check (a push with no schema is refused).
_PERMISSIVE_SCHEMAS = {f"schema/{name}": "{}" for name in content_repo._SCHEMAS.values()}
# Content's real schemas for the files this repo pushes, as a test copy (ADR-0029).
_SCHEMA_COPY = Path(__file__).resolve().parents[1] / "fixtures" / "content_schema"


def _init_bare_repo_with_content(tmp_path):
    """A local bare repo (the fixture 'Sermon-Note-Content') seeded with one commit."""
    bare = tmp_path / "content.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(bare)], check=True, capture_output=True)

    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", str(bare), str(seed)], check=True, capture_output=True)
    (seed / "notes").mkdir()
    (seed / "notes" / "existing.txt").write_text("pre-existing pipeline output\n")
    (seed / "schema").mkdir()
    for rel_path, text in _PERMISSIVE_SCHEMAS.items():
        (seed / rel_path).write_text(text)
    subprocess.run(["git", "add", "-A"], cwd=seed, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.name=seed", "-c", "user.email=seed@example.org", "commit", "-m", "seed"],
        cwd=seed,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "push", "origin", "main"], cwd=seed, check=True, capture_output=True)
    # Serve partial clones as GitHub does, including the on-demand fetch of a skipped blob.
    for setting in ("uploadpack.allowFilter", "uploadpack.allowAnySHA1InWant"):
        subprocess.run(["git", "config", setting, "true"], cwd=bare, check=True, capture_output=True)
    return bare


def _read_file_at_head(bare, rel_path):
    result = subprocess.run(
        ["git", "show", f"main:{rel_path}"], cwd=bare, capture_output=True, text=True, check=True
    )
    return result.stdout


@pytest.fixture
def content_repo_env(tmp_path, monkeypatch):
    """Point ``default_push`` at a local bare repo instead of a real github.com remote.

    ``CONTENT_REPO``/``CONTENT_REPO_TOKEN`` are still read (proving config wiring), but
    ``_remote_url`` is patched to return the local bare repo rather than build a
    github.com URL from them, so this exercises the real git mechanics with no network.
    It is a ``file://`` URL because git ignores ``--depth`` and ``--filter`` when cloning
    a plain local path.
    """
    bare = _init_bare_repo_with_content(tmp_path)
    monkeypatch.setenv("CONTENT_REPO", "owner/sermon-note-content")
    monkeypatch.setenv("CONTENT_REPO_TOKEN", "unused-in-local-git-fixture")
    monkeypatch.setenv("CONTENT_REPO_BRANCH", "main")
    monkeypatch.setattr(content_repo, "_remote_url", lambda repo, token: f"file://{bare}")
    return bare


def test_default_push_adds_a_new_file_without_touching_existing_ones(content_repo_env):
    content_repo.default_push({"transcripts/menlo/sermon-1.txt": "hello world"})

    assert _read_file_at_head(content_repo_env, "transcripts/menlo/sermon-1.txt") == "hello world"
    assert _read_file_at_head(content_repo_env, "notes/existing.txt") == "pre-existing pipeline output\n"


def test_default_push_is_a_noop_when_content_is_unchanged(content_repo_env):
    content_repo.default_push({"transcripts/menlo/sermon-1.txt": "hello world"})
    before = subprocess.run(
        ["git", "rev-parse", "main"], cwd=content_repo_env, capture_output=True, text=True, check=True
    ).stdout

    content_repo.default_push({"transcripts/menlo/sermon-1.txt": "hello world"})
    after = subprocess.run(
        ["git", "rev-parse", "main"], cwd=content_repo_env, capture_output=True, text=True, check=True
    ).stdout

    assert before == after


def test_default_push_refuses_to_overwrite_an_existing_transcript_with_different_text(content_repo_env):
    # Whisper is not bit-for-bit deterministic across runs; a second transcription
    # of an already-"done" sermon must never silently replace one a downstream
    # consumer (Pipeline) may already have fetched and hashed (issue #60).
    content_repo.default_push({"transcripts/menlo/sermon-1.txt": "first version"})

    with pytest.raises(content_repo.ContentRefusedError):
        content_repo.default_push({"transcripts/menlo/sermon-1.txt": "second, different version"})

    assert _read_file_at_head(content_repo_env, "transcripts/menlo/sermon-1.txt") == "first version"


def test_push_transcripts_gives_up_on_the_overwrite_guard_after_one_attempt(content_repo_env):
    # A fresh clone can't change the refusal, so retrying it only burns backoff and clones.
    content_repo.default_push({"transcripts/menlo/sermon-1.txt": "first version"})
    sleeps = []

    with pytest.raises(content_repo.ContentRefusedError, match="refusing to overwrite"):
        content_repo.push_transcripts(
            {"transcripts/menlo/sermon-1.txt": "second, different version"}, sleep=sleeps.append
        )
    assert sleeps == []


def test_push_transcripts_fails_at_once_as_a_publish_error_when_content_is_unconfigured(monkeypatch):
    monkeypatch.delenv("CONTENT_REPO", raising=False)
    monkeypatch.delenv("CONTENT_REPO_TOKEN", raising=False)
    sleeps = []

    with pytest.raises(content_repo.ContentPublishError, match="CONTENT_REPO"):
        content_repo.push_transcripts({"transcripts/menlo/sermon-1.txt": "hello world"}, sleep=sleeps.append)
    assert sleeps == []


def test_default_fetch_existing_returns_the_text_of_only_the_paths_content_already_holds(content_repo_env):
    content_repo.default_push({"transcripts/pbc/2019-01-06_untitled_a.txt": "already pushed text"})
    head_before = subprocess.run(
        ["git", "rev-parse", "main"], cwd=content_repo_env, capture_output=True, text=True, check=True
    ).stdout

    existing = content_repo.default_fetch_existing(
        ["transcripts/pbc/2019-01-06_untitled_a.txt", "transcripts/pbc/2019-01-13_untitled_b.txt"]
    )
    head_after = subprocess.run(
        ["git", "rev-parse", "main"], cwd=content_repo_env, capture_output=True, text=True, check=True
    ).stdout

    assert existing == {"transcripts/pbc/2019-01-06_untitled_a.txt": "already pushed text"}
    assert head_before == head_after


def test_existing_transcripts_is_a_noop_for_no_paths():
    calls = []
    assert content_repo.existing_transcripts([], fetch=lambda paths: calls.append(paths) or {}) == {}
    assert calls == []


def test_existing_transcripts_retries_a_transient_failure():
    attempts = []

    def flaky(paths):
        attempts.append(paths)
        if len(attempts) == 1:
            raise content_repo.ContentPublishError("git clone failed: connection reset")
        return {"transcripts/menlo/sermon-1.txt": "text"}

    existing = content_repo.existing_transcripts(
        ["transcripts/menlo/sermon-1.txt"], fetch=flaky, sleep=lambda _s: None, attempts=3
    )

    assert existing == {"transcripts/menlo/sermon-1.txt": "text"}
    assert len(attempts) == 2


def test_existing_transcripts_raises_after_exhausting_attempts():
    def always_fails(paths):
        raise content_repo.ContentPublishError("git clone failed: connection reset")

    with pytest.raises(content_repo.ContentPublishError):
        content_repo.existing_transcripts(
            ["transcripts/menlo/sermon-1.txt"], fetch=always_fails, sleep=lambda _s: None, attempts=3
        )


def test_push_transcripts_is_a_noop_for_empty_files(monkeypatch):
    calls = []
    content_repo.push_transcripts({}, push=lambda files: calls.append(files))
    assert calls == []


def test_push_transcripts_retries_then_raises_after_exhaustion():
    attempts = []

    def always_fails(files):
        attempts.append(files)
        raise content_repo.ContentPublishError("network is down")

    with pytest.raises(content_repo.ContentPublishError):
        content_repo.push_transcripts(
            {"transcripts/menlo/sermon-1.txt": "text"},
            push=always_fails,
            sleep=lambda _s: None,
            attempts=3,
        )
    assert len(attempts) == 3


def test_run_git_redacts_token_from_a_failure_message(tmp_path):
    with pytest.raises(content_repo.ContentPublishError) as exc_info:
        content_repo._run_git(
            [
                "clone",
                "--depth",
                "1",
                "https://x-access-token:super-secret-token@github.com/nope/nope.git",
                "dest",
            ],
            cwd=tmp_path,
            token="super-secret-token",
        )
    assert "super-secret-token" not in str(exc_info.value)


def test_default_push_status_replaces_the_status_file_and_nothing_else(content_repo_env):
    """Unlike a transcript, the status file is meant to change: no overwrite guard."""
    content_repo.default_push_status({"status/ingest.json": '{"sermons": []}\n'})
    content_repo.default_push_status({"status/ingest.json": '{"sermons": [1]}\n'})

    assert _read_file_at_head(content_repo_env, "status/ingest.json") == '{"sermons": [1]}\n'
    assert _read_file_at_head(content_repo_env, "notes/existing.txt") == "pre-existing pipeline output\n"


def test_default_push_status_is_a_noop_when_the_status_is_unchanged(content_repo_env):
    content_repo.default_push_status({"status/ingest.json": '{"sermons": []}\n'})
    before = subprocess.run(
        ["git", "rev-parse", "main"], cwd=content_repo_env, capture_output=True, text=True, check=True
    ).stdout

    content_repo.default_push_status({"status/ingest.json": '{"sermons": []}\n'})
    after = subprocess.run(
        ["git", "rev-parse", "main"], cwd=content_repo_env, capture_output=True, text=True, check=True
    ).stdout

    assert before == after


def test_push_status_retries_a_rejected_push_from_a_fresh_clone():
    """Pipeline pushes to Content too; a push that lost the race is retried, not lost."""
    calls = []

    def flaky(files):
        calls.append(files)
        if len(calls) == 1:
            raise content_repo.ContentPublishError("! [rejected] main -> main (fetch first)")

    content_repo.push_status({"status/ingest.json": "{}\n"}, push=flaky, sleep=lambda _s: None)

    assert len(calls) == 2


def test_push_status_raises_after_exhausting_attempts():
    def refuse(files):
        raise content_repo.ContentPublishError("rejected")

    with pytest.raises(content_repo.ContentPublishError, match="after 3 attempts"):
        content_repo.push_status({"status/ingest.json": "{}\n"}, push=refuse, sleep=lambda _s: None)


# --- Re-transcription (ADR-0022) ---------------------------------------------------------

_PATH = "transcripts/menlo/2026-09-06_hear-and-do_g1.txt"
_OLD = "This morning we turn to Luke chapter six. Hear and do. Hear and do. Hear and do."
_NEW = "This morning we turn to Luke chapter six, where Jesus tells us to hear and to do."
_TRAILERS = {
    "Ingest-Guid": "menlo:g1, part 1",
    "Ingest-Supersedes": "oldhash",
    "Ingest-Profile-Model": '"large-v3"',
}


def _head(bare):
    return subprocess.run(
        ["git", "rev-parse", "main"], cwd=bare, capture_output=True, text=True, check=True
    ).stdout.strip()


def _last_message(bare):
    return subprocess.run(
        ["git", "log", "-1", "--format=%B", "main"], cwd=bare, capture_output=True, text=True, check=True
    ).stdout


def _changed_paths(bare):
    return subprocess.run(
        ["git", "show", "--name-only", "--format=", "main"],
        cwd=bare,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()


def _hash(text):
    from poller.transcribe import transcript_hash

    return transcript_hash(text)


def test_default_replace_overwrites_only_the_named_path_when_content_holds_the_expected_text(
    content_repo_env,
):
    content_repo.default_push({_PATH: _OLD})

    content_repo.default_replace(_PATH, _NEW, expect_hash=_hash(_OLD), trailers=_TRAILERS)

    assert _read_file_at_head(content_repo_env, _PATH) == _NEW
    assert _changed_paths(content_repo_env) == [_PATH]
    message = _last_message(content_repo_env)
    for key, value in _TRAILERS.items():
        assert f"{key}: {value}" in message


def test_default_replace_refuses_when_content_holds_text_the_ledger_does_not_name(content_repo_env):
    # Someone else's edit (Pipeline, a person) is never clobbered: CAS on the ledger's hash.
    content_repo.default_push({_PATH: "a hand-corrected transcript"})
    head = _head(content_repo_env)

    with pytest.raises(content_repo.ContentConflictError):
        content_repo.default_replace(_PATH, _NEW, expect_hash=_hash(_OLD), trailers=_TRAILERS)

    assert _head(content_repo_env) == head


def test_default_replace_refuses_when_the_file_is_missing_from_content(content_repo_env):
    with pytest.raises(content_repo.ContentConflictError):
        content_repo.default_replace(_PATH, _NEW, expect_hash=_hash(_OLD), trailers=_TRAILERS)
    with pytest.raises(subprocess.CalledProcessError):
        _read_file_at_head(content_repo_env, _PATH)


def test_default_replace_is_a_noop_when_content_already_holds_the_new_text(content_repo_env):
    # A retry after a push that landed but reported failure must not read as a conflict.
    content_repo.default_push({_PATH: _NEW})
    head = _head(content_repo_env)

    content_repo.default_replace(_PATH, _NEW, expect_hash=_hash(_OLD), trailers=_TRAILERS)

    assert _head(content_repo_env) == head


def test_default_replace_redacts_the_token_from_a_failure(monkeypatch):
    monkeypatch.setenv("CONTENT_REPO", "owner/sermon-note-content")
    monkeypatch.setenv("CONTENT_REPO_TOKEN", "super-secret-token")
    monkeypatch.setattr(content_repo, "_remote_url", lambda repo, token: f"/nonexistent/{token}/content.git")

    with pytest.raises(content_repo.ContentPublishError) as exc_info:
        content_repo.default_replace(_PATH, _NEW, expect_hash=_hash(_OLD), trailers=_TRAILERS)
    assert "super-secret-token" not in str(exc_info.value)


def test_replace_transcript_retries_a_rejected_push_and_checks_again_each_time():
    calls = []

    def flaky(path, text, *, expect_hash, trailers):
        calls.append(expect_hash)
        if len(calls) < 3:
            raise content_repo.ContentPublishError("! [rejected] main -> main (fetch first)")

    content_repo.replace_transcript(
        _PATH, _NEW, expect_hash="oldhash", trailers=_TRAILERS, replace=flaky, sleep=lambda _s: None
    )

    assert calls == ["oldhash", "oldhash", "oldhash"]


def test_replace_transcript_never_retries_a_conflict():
    calls = []

    def moved(path, text, *, expect_hash, trailers):
        calls.append(path)
        raise content_repo.ContentConflictError("Content moved")

    with pytest.raises(content_repo.ContentConflictError):
        content_repo.replace_transcript(
            _PATH, _NEW, expect_hash="oldhash", trailers=_TRAILERS, replace=moved, sleep=lambda _s: None
        )
    assert len(calls) == 1


def test_replace_transcript_raises_after_exhausting_five_attempts():
    calls = []

    def refuse(path, text, *, expect_hash, trailers):
        calls.append(path)
        raise content_repo.ContentPublishError("rejected")

    with pytest.raises(content_repo.ContentPublishError, match="after 5 attempts"):
        content_repo.replace_transcript(
            _PATH, _NEW, expect_hash="oldhash", trailers=_TRAILERS, replace=refuse, sleep=lambda _s: None
        )
    assert len(calls) == 5


def test_default_last_commit_trailers_reads_the_last_commit_that_touched_the_path(content_repo_env):
    content_repo.default_push({_PATH: _OLD})
    content_repo.default_replace(_PATH, _NEW, expect_hash=_hash(_OLD), trailers=_TRAILERS)
    content_repo.default_push({"transcripts/menlo/another.txt": "a later sermon"})

    assert content_repo.default_last_commit_trailers(_PATH) == _TRAILERS


def test_default_last_commit_trailers_is_empty_for_a_commit_without_trailers(content_repo_env):
    content_repo.default_push({_PATH: _OLD})
    assert content_repo.default_last_commit_trailers(_PATH) == {}


def test_last_commit_trailers_retries_a_transient_failure():
    calls = []

    def flaky(path):
        calls.append(path)
        if len(calls) == 1:
            raise content_repo.ContentPublishError("connection reset")
        return {"Ingest-Guid": "g1"}

    assert content_repo.last_commit_trailers(_PATH, fetch=flaky, sleep=lambda _s: None) == {
        "Ingest-Guid": "g1"
    }


_CANDIDATE = "retranscription-candidates/menlo/2026-09-06_hear-and-do_g1.abcdef012345.txt"


def test_default_keep_candidate_adds_the_file_with_its_trailers_outside_transcripts(content_repo_env):
    content_repo.default_push({_PATH: _OLD})

    content_repo.default_keep_candidate(_CANDIDATE, _NEW, trailers={"Ingest-Refusal": "loops"})

    assert _read_file_at_head(content_repo_env, _CANDIDATE) == _NEW
    assert _read_file_at_head(content_repo_env, _PATH) == _OLD
    assert _changed_paths(content_repo_env) == [_CANDIDATE]
    assert "Ingest-Refusal: loops" in _last_message(content_repo_env)


def test_default_keep_candidate_never_overwrites_a_kept_candidate(content_repo_env):
    content_repo.default_keep_candidate(_CANDIDATE, _NEW, trailers={"Ingest-Refusal": "loops"})
    head = _head(content_repo_env)

    content_repo.default_keep_candidate(_CANDIDATE, _NEW, trailers={"Ingest-Refusal": "loops"})
    assert _head(content_repo_env) == head
    with pytest.raises(content_repo.ContentRefusedError):
        content_repo.default_keep_candidate(_CANDIDATE, "different text", trailers={})
    assert _read_file_at_head(content_repo_env, _CANDIDATE) == _NEW


def test_keep_candidate_gives_up_on_a_different_kept_candidate_after_one_attempt(content_repo_env):
    content_repo.default_keep_candidate(_CANDIDATE, _NEW, trailers={})
    sleeps = []

    with pytest.raises(content_repo.ContentRefusedError, match="kept candidate"):
        content_repo.keep_candidate(_CANDIDATE, "different text", trailers={}, sleep=sleeps.append)
    assert sleeps == []
    assert _read_file_at_head(content_repo_env, _CANDIDATE) == _NEW


def test_keep_candidate_retries_then_raises():
    calls = []

    def refuse(path, text, *, trailers):
        calls.append(path)
        raise content_repo.ContentPublishError("rejected")

    with pytest.raises(content_repo.ContentPublishError):
        content_repo.keep_candidate(_CANDIDATE, _NEW, trailers={}, keep=refuse, sleep=lambda _s: None)
    assert len(calls) == 5


def test_default_fetch_all_transcripts_returns_every_transcript_and_nothing_else(content_repo_env):
    content_repo.default_push({_PATH: _OLD, "transcripts/pbc/2019-01-06_untitled_a.txt": "pbc text"})
    content_repo.default_keep_candidate(_CANDIDATE, _NEW, trailers={})

    assert content_repo.default_fetch_all_transcripts() == {
        _PATH: _OLD,
        "transcripts/pbc/2019-01-06_untitled_a.txt": "pbc text",
    }


def test_provenance_trailers_survive_a_real_git_commit_and_prove_the_replacement(content_repo_env):
    # The crash-between-pushes repair (ADR-0022 rule 5) depends on git's own trailer
    # parser giving back exactly what the replacement wrote.
    from poller import retranscribe

    profile = {
        "model": "large-v3",
        "compute_type": "int8",
        "beam_size": 5,
        "condition_on_previous_text": False,
        "hallucination_silence_threshold": 2.0,
        "vad_filter": True,
        "domain_prompt": True,
        "hotwords": "Keith Crosby, Luke: Hear and Do, Grace #2",
        "faster_whisper_version": "1.2.1",
        "ctranslate2_version": None,
    }
    counts = {"words": 5210, "loop_words": 0, "credit_hits": 0, "hotword_pairs": 0}
    guid = "gracepres:sid:2023-07-30,slot:sermon/0ab4"
    content_repo.default_push({_PATH: _OLD})
    trailers = retranscribe._provenance_trailers(
        guid,
        superseded=_hash(_OLD),
        transcript_hash=_hash(_NEW),
        transcribed_at="2026-10-13T02:00:00+00:00",
        profile=profile,
        counts=counts,
    )

    content_repo.default_replace(_PATH, _NEW, expect_hash=_hash(_OLD), trailers=trailers)
    mark = retranscribe._mark_from_trailers(
        content_repo.default_last_commit_trailers(_PATH),
        guid=guid,
        ledger_hash=_hash(_OLD),
        content_hash=_hash(_NEW),
    )

    assert mark is not None
    assert mark["profile"] == profile
    assert mark["quality"] == counts
    assert mark["transcribed_at"] == "2026-10-13T02:00:00+00:00"


def _checkout_on_disk(monkeypatch):
    """Record the files the clone holds on disk when a write stages its commit."""
    seen = []
    original = content_repo._has_staged_changes

    def spy(checkout):
        seen.extend(
            sorted(
                p.relative_to(checkout).as_posix()
                for p in checkout.rglob("*")
                if p.is_file() and ".git" not in p.relative_to(checkout).parts
            )
        )
        return original(checkout)

    monkeypatch.setattr(content_repo, "_has_staged_changes", spy)
    return seen


def test_default_push_checks_out_only_the_paths_it_writes_and_keeps_the_rest(content_repo_env, monkeypatch):
    """Content is mostly transcripts, so a write fetches only its own paths' blobs.

    The paths the sparse clone skipped are not on disk, and must not be pushed as deleted.
    """
    content_repo.default_push({_PATH: _OLD, "transcripts/pbc/2019-01-06_untitled_a.txt": "pbc text"})
    checkout = _checkout_on_disk(monkeypatch)

    content_repo.default_push_status({"status/ingest.json": '{"sermons": []}\n'})

    assert checkout == [*sorted(_PERMISSIVE_SCHEMAS), "status/ingest.json"]
    assert _read_file_at_head(content_repo_env, _PATH) == _OLD
    assert _read_file_at_head(content_repo_env, "transcripts/pbc/2019-01-06_untitled_a.txt") == "pbc text"
    assert _read_file_at_head(content_repo_env, "notes/existing.txt") == "pre-existing pipeline output\n"


def test_default_replace_checks_out_only_the_transcript_it_replaces(content_repo_env, monkeypatch):
    content_repo.default_push({_PATH: _OLD, "transcripts/pbc/2019-01-06_untitled_a.txt": "pbc text"})
    checkout = _checkout_on_disk(monkeypatch)

    content_repo.default_replace(_PATH, _NEW, expect_hash=_hash(_OLD), trailers={})

    assert checkout == [_PATH]
    assert _read_file_at_head(content_repo_env, _PATH) == _NEW
    assert _read_file_at_head(content_repo_env, "transcripts/pbc/2019-01-06_untitled_a.txt") == "pbc text"


def test_default_push_reads_a_path_with_glob_characters_literally(content_repo_env, monkeypatch):
    """A sparse pattern is gitignore syntax; ``[1]`` must not also match a sibling ``1``."""
    literal = "transcripts/menlo/2026-10-04_part-[1]_a.txt"
    content_repo.default_push({literal: "bracketed", "transcripts/menlo/2026-10-04_part-1_a.txt": "plain"})
    checkout = _checkout_on_disk(monkeypatch)

    content_repo.default_push({literal: "bracketed"})

    assert checkout == [literal]


def _schema_checkout(tmp_path, names):
    """A stand-in clone whose ``schema/`` holds the real copies of ``names``."""
    (tmp_path / "schema").mkdir()
    for name in names:
        (tmp_path / "schema" / name).write_text((_SCHEMA_COPY / name).read_text())
    return tmp_path


def test_a_status_file_that_fits_the_schema_passes(tmp_path):
    checkout = _schema_checkout(tmp_path, content_repo._SCHEMAS.values())

    content_repo._refuse_invalid(
        {"status/ingest.json": json.dumps({"churches": {}, "sermons": []}), "transcripts/x.txt": "text"},
        checkout,
    )


def test_an_undeclared_field_is_refused_naming_file_and_field(tmp_path):
    """Every schema object is closed, so a ledger field cannot leak into Content by accident."""
    checkout = _schema_checkout(tmp_path, content_repo._SCHEMAS.values())
    status = {"churches": {}, "sermons": [], "audio_url": "https://cdn.example/a.mp3"}

    with pytest.raises(content_repo.ContentSchemaError, match=r"status/ingest\.json: \(root\): .*audio_url"):
        content_repo._refuse_invalid({"status/ingest.json": json.dumps(status)}, checkout)


def test_a_missing_schema_is_refused(tmp_path):
    """Fail closed: the check never passes because Content has no contract for the file."""
    with pytest.raises(
        content_repo.ContentSchemaError, match=r"schema/status-ingest\.schema\.json: unusable"
    ):
        content_repo._refuse_invalid({"status/ingest.json": "{}"}, tmp_path)


def test_a_json_file_no_schema_describes_is_refused(tmp_path):
    with pytest.raises(content_repo.ContentSchemaError, match=r"status/extra\.json: no schema describes"):
        content_repo._refuse_invalid({"status/extra.json": "{}"}, tmp_path)


def _head(bare):
    return subprocess.run(
        ["git", "rev-parse", "main"], cwd=bare, capture_output=True, text=True, check=True
    ).stdout


def test_default_push_status_refuses_an_invalid_file_and_leaves_content_unchanged(content_repo_env, tmp_path):
    """The refusal comes before any write, against the schema Content holds now."""
    seed = tmp_path / "schema-seed"
    subprocess.run(["git", "clone", str(content_repo_env), str(seed)], check=True, capture_output=True)
    name = "status-ingest.schema.json"
    (seed / "schema" / name).write_text((_SCHEMA_COPY / name).read_text())
    subprocess.run(
        ["git", "-c", "user.name=seed", "-c", "user.email=seed@example.org", "commit", "-qam", "schema"],
        cwd=seed,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "push", "origin", "main"], cwd=seed, check=True, capture_output=True)
    before = _head(content_repo_env)

    with pytest.raises(content_repo.ContentSchemaError, match=r"sermons/0: 'first_seen_at' is a required"):
        content_repo.default_push_status(
            {"status/ingest.json": json.dumps({"churches": {}, "sermons": [{"guid": "g"}]})}
        )

    assert _head(content_repo_env) == before


def test_push_status_does_not_retry_a_schema_refusal():
    """Re-cloning cannot change the verdict, so the refusal surfaces at once."""
    calls = []

    def refuses(files):
        calls.append(files)
        raise content_repo.ContentSchemaError("refusing to push 1 schema violation(s)")

    with pytest.raises(content_repo.ContentSchemaError):
        content_repo.push_status({"status/ingest.json": "{}"}, push=refuses, sleep=lambda _s: None)

    assert len(calls) == 1
