import subprocess

import pytest

from poller import content_repo


def _init_bare_repo_with_content(tmp_path):
    """A local bare repo (the fixture 'Sermon-Note-Content') seeded with one commit."""
    bare = tmp_path / "content.git"
    subprocess.run(["git", "init", "--bare", "-b", "main", str(bare)], check=True, capture_output=True)

    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", str(bare), str(seed)], check=True, capture_output=True)
    (seed / "notes").mkdir()
    (seed / "notes" / "existing.txt").write_text("pre-existing pipeline output\n")
    subprocess.run(["git", "add", "-A"], cwd=seed, check=True, capture_output=True)
    subprocess.run(
        ["git", "-c", "user.name=seed", "-c", "user.email=seed@example.org", "commit", "-m", "seed"],
        cwd=seed,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "push", "origin", "main"], cwd=seed, check=True, capture_output=True)
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
    ``_remote_url`` is patched to return the local bare path directly rather than build a
    github.com URL from them, so this exercises the real git mechanics with no network.
    """
    bare = _init_bare_repo_with_content(tmp_path)
    monkeypatch.setenv("CONTENT_REPO", "owner/sermon-note-content")
    monkeypatch.setenv("CONTENT_REPO_TOKEN", "unused-in-local-git-fixture")
    monkeypatch.setenv("CONTENT_REPO_BRANCH", "main")
    monkeypatch.setattr(content_repo, "_remote_url", lambda repo, token: str(bare))
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

    with pytest.raises(content_repo.ContentPublishError):
        content_repo.default_push({"transcripts/menlo/sermon-1.txt": "second, different version"})

    assert _read_file_at_head(content_repo_env, "transcripts/menlo/sermon-1.txt") == "first version"


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
