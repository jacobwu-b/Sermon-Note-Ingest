"""poller.ledger_push: a producer job's ledger writes, merged onto the live tip by the push job (ADR-0027)."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from poller import ledger_push

REPO_ROOT = Path(__file__).resolve().parents[2]

# The developer's own git config (signing, hooks, default branch) must not reach these repos.
GIT_ENV = {
    **os.environ,
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_AUTHOR_NAME": "Ledger Writer",
    "GIT_AUTHOR_EMAIL": "writer@example.com",
    "GIT_COMMITTER_NAME": "Ledger Writer",
    "GIT_COMMITTER_EMAIL": "writer@example.com",
}

ABIDE = {
    "guid": "https://pbc.org?enmse_mid=4691",
    "title": "Abide in Me",
    "speaker": "Dan Westman",
    "preached_on": "2026-10-18",
    "feed_published_at": "2026-10-19T18:02:11+00:00",
    "audio_url": "https://s3.us-west-2.amazonaws.com/cdn.pbc.org/Main_Service/2026/10/18/a.mp3",
    "notified_at": "2026-10-19T18:05:00+00:00",
    "transcription_status": "pending",
}
CAPTIVES = {
    "guid": "https://pbc.org?enmse_mid=4701",
    "title": "Setting Captives Free",
    "speaker": "Dan Westman",
    "preached_on": "2026-10-25",
    "feed_published_at": "2026-10-26T18:21:06+00:00",
    "audio_url": "https://s3.us-west-2.amazonaws.com/cdn.pbc.org/Main_Service/2026/10/25/b.mp3",
    "notified_at": None,
    "transcription_status": "pending",
}
DONE_FIELDS = {
    "transcription_status": "done",
    "transcribed_at": "2026-10-26T20:06:24.657009+00:00",
    "transcript_hash": "e23512004d655969cec6b45873c4399fe68a16b95f2e5e6343c3e9ae4399b74d",
    "content_path": "transcripts/pbc/2026-10-18_abide-in-me_https_pbc.org_enmse_mid_4691.txt",
}


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True, env=GIT_ENV
    ).stdout


def ledger(*records: dict) -> dict:
    return {r["guid"]: r for r in records}


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


@pytest.fixture
def repo(tmp_path, monkeypatch) -> Path:
    """A checkout whose last commit holds one pbc ledger, standing in for main's tip."""
    # ledger_push runs git itself, so the isolation has to reach the real process environment.
    for key, value in GIT_ENV.items():
        if key.startswith("GIT_"):
            monkeypatch.setenv(key, value)
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q", "-b", "main")
    write_json(root / "data" / "pbc.json", ledger(ABIDE))
    (root / "data" / "overrides").mkdir()
    (root / "data" / "README.md").write_text("# Ledger\n")
    git(root, "add", "data")
    git(root, "commit", "-q", "-m", "seed ledger")
    return root


def head(repo: Path) -> str:
    return git(repo, "rev-parse", "HEAD").strip()


def staged_artifact(tmp_path: Path, repo: Path, edit) -> Path:
    """What a producer job uploads: ``edit`` applied to a clone of ``repo``, then staged."""
    producer = tmp_path / f"producer-{len(list(tmp_path.glob('producer-*')))}"
    git(tmp_path, "clone", "-q", str(repo), str(producer))
    edit(producer)
    out = tmp_path / "ledger" / producer.name
    ledger_push.stage(producer, out)
    return out


# --- stage -------------------------------------------------------------------------------


def test_stage_records_the_base_commit_and_copies_only_the_changed_ledgers(tmp_path, repo):
    def discover(producer: Path) -> None:
        write_json(producer / "data" / "pbc.json", ledger(CAPTIVES, ABIDE))
        write_json(producer / "data" / "hillside.json", ledger({**CAPTIVES, "guid": "hillside-2026-10-25"}))

    out = staged_artifact(tmp_path, repo, discover)

    assert (out / ledger_push.BASE_FILE).read_text().strip() == head(repo)
    staged = sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file())
    assert staged == ["data/hillside.json", "data/pbc.json", ledger_push.BASE_FILE]


def test_stage_with_nothing_changed_still_records_the_base(tmp_path, repo):
    out = staged_artifact(tmp_path, repo, lambda producer: None)
    assert [p.name for p in out.iterdir()] == [ledger_push.BASE_FILE]


def test_stage_refuses_a_change_outside_the_allowlist(tmp_path, repo):
    producer = tmp_path / "producer"
    git(tmp_path, "clone", "-q", str(repo), str(producer))
    write_json(producer / "data" / "overrides" / "pbc.json", {ABIDE["guid"]: {"title": "Abide"}})

    with pytest.raises(ledger_push.LedgerPushError, match="data/overrides/pbc.json"):
        ledger_push.stage(producer, tmp_path / "out")


# --- allowlist -----------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["data/pbc.json", "data/north_point.json", "data/alerts/state.json"])
def test_the_allowlist_admits_what_automation_writes(path):
    assert ledger_push.allowed(path)


@pytest.mark.parametrize(
    "path",
    [
        ".github/workflows/poll.yml",
        "poller/runner.py",
        "data/README.md",
        "data/overrides/pbc.json",
        "data/alerts/snooze.json",
        "data/../poller/store.py",
        "data/Pbc.json",
        "data/pbc.json.tmp",
        "data/sub/pbc.json",
        "requirements.txt",
    ],
)
def test_the_allowlist_refuses_everything_else(path):
    assert not ledger_push.allowed(path)


# --- apply: the merge ------------------------------------------------------------------------


def test_apply_lands_a_new_sermon_on_a_tip_another_writer_moved(tmp_path, repo):
    out = staged_artifact(
        tmp_path, repo, lambda p: write_json(p / "data" / "pbc.json", ledger(CAPTIVES, ABIDE))
    )
    # A transcribe shard marked the older sermon done after the poll checked out.
    write_json(repo / "data" / "pbc.json", ledger({**ABIDE, **DONE_FIELDS}))
    git(repo, "commit", "-q", "-am", "chore(data): record transcription progress [skip ci]")

    ledger_push.apply(repo, out.parent)

    merged = read_json(repo / "data" / "pbc.json")
    assert merged == ledger(CAPTIVES, {**ABIDE, **DONE_FIELDS})
    # The store's own order: newest preached_on first.
    assert list(merged) == [CAPTIVES["guid"], ABIDE["guid"]]


def test_apply_merges_field_by_field_when_both_writers_touched_one_record(tmp_path, repo):
    # The poll refreshed a rotated enclosure URL; a shard marked the same sermon done.
    rotated = "https://s3.us-west-2.amazonaws.com/cdn.pbc.org/Main_Service/2026/10/18/a-v2.mp3"
    out = staged_artifact(
        tmp_path, repo, lambda p: write_json(p / "data" / "pbc.json", ledger({**ABIDE, "audio_url": rotated}))
    )
    write_json(repo / "data" / "pbc.json", ledger({**ABIDE, **DONE_FIELDS}))
    git(repo, "commit", "-q", "-am", "transcribed")

    ledger_push.apply(repo, out.parent)

    assert read_json(repo / "data" / "pbc.json") == ledger({**ABIDE, **DONE_FIELDS, "audio_url": rotated})


def test_apply_carries_a_field_removal_over(tmp_path, repo):
    def drop_notified(p: Path) -> None:
        record = {k: v for k, v in ABIDE.items() if k != "notified_at"}
        write_json(p / "data" / "pbc.json", ledger(record))

    out = staged_artifact(tmp_path, repo, drop_notified)
    ledger_push.apply(repo, out.parent)
    assert "notified_at" not in read_json(repo / "data" / "pbc.json")[ABIDE["guid"]]


def test_apply_merges_every_shards_artifact_onto_one_tip(tmp_path, repo):
    write_json(repo / "data" / "pbc.json", ledger(CAPTIVES, ABIDE))
    git(repo, "commit", "-q", "-am", "two pending")
    first = staged_artifact(
        tmp_path,
        repo,
        lambda p: write_json(p / "data" / "pbc.json", ledger(CAPTIVES, {**ABIDE, **DONE_FIELDS})),
    )
    captives_done = {**CAPTIVES, **DONE_FIELDS, "content_path": "transcripts/pbc/2026-10-25_captives.txt"}
    staged_artifact(
        tmp_path, repo, lambda p: write_json(p / "data" / "pbc.json", ledger(captives_done, ABIDE))
    )

    ledger_push.apply(repo, first.parent)

    assert read_json(repo / "data" / "pbc.json") == ledger(captives_done, {**ABIDE, **DONE_FIELDS})


def test_apply_fails_closed_when_both_writers_changed_one_field_differently(tmp_path, repo):
    out = staged_artifact(
        tmp_path,
        repo,
        lambda p: write_json(p / "data" / "pbc.json", ledger({**ABIDE, "transcription_status": "failed"})),
    )
    write_json(repo / "data" / "pbc.json", ledger({**ABIDE, **DONE_FIELDS}))
    git(repo, "commit", "-q", "-am", "transcribed")
    before = (repo / "data" / "pbc.json").read_text()

    with pytest.raises(ledger_push.LedgerPushError, match="transcription_status"):
        ledger_push.apply(repo, out.parent)
    assert (repo / "data" / "pbc.json").read_text() == before


def test_apply_writes_raw_ledger_values_never_an_override(tmp_path, repo):
    # store.load merges overrides in; the merge works on the files, so a correction is never baked in.
    write_json(repo / "data" / "overrides" / "pbc.json", {ABIDE["guid"]: {"title": "Abide in Me (Part 1)"}})
    git(repo, "add", "data")
    git(repo, "commit", "-q", "-m", "override")
    out = staged_artifact(
        tmp_path, repo, lambda p: write_json(p / "data" / "pbc.json", ledger(CAPTIVES, ABIDE))
    )

    ledger_push.apply(repo, out.parent)

    assert read_json(repo / "data" / "pbc.json")[ABIDE["guid"]]["title"] == "Abide in Me"


def test_apply_merges_alert_state_by_church_and_drops_a_cleared_incident(tmp_path, repo):
    state = repo / "data" / "alerts" / "state.json"
    write_json(state, {"menlo": {"failing_since": "2026-10-30T12:00:00+00:00"}, "pbc": {}})
    git(repo, "add", "data")
    git(repo, "commit", "-q", "-m", "alert state")

    def clear_menlo(p: Path) -> None:
        write_json(
            p / "data" / "alerts" / "state.json",
            {"pbc": {}, "westgate": {"missing_alerted_for": "2026-10-26"}},
        )

    out = staged_artifact(tmp_path, repo, clear_menlo)
    ledger_push.apply(repo, out.parent)

    assert list(read_json(state).items()) == [
        ("pbc", {}),
        ("westgate", {"missing_alerted_for": "2026-10-26"}),
    ]


def test_apply_fetches_a_base_commit_a_shallow_checkout_lacks(tmp_path, repo):
    out = staged_artifact(
        tmp_path, repo, lambda p: write_json(p / "data" / "pbc.json", ledger(CAPTIVES, ABIDE))
    )
    write_json(repo / "data" / "pbc.json", ledger({**ABIDE, **DONE_FIELDS}))
    git(repo, "commit", "-q", "-am", "transcribed")
    git(repo, "config", "uploadpack.allowAnySHA1InWant", "true")
    shallow = tmp_path / "shallow"
    git(tmp_path, "clone", "-q", "--depth=1", f"file://{repo}", str(shallow))

    ledger_push.apply(shallow, out.parent)

    assert read_json(shallow / "data" / "pbc.json") == ledger(CAPTIVES, {**ABIDE, **DONE_FIELDS})


# --- apply: refusing what an artifact must not carry -----------------------------------------


def test_apply_refuses_an_artifact_carrying_a_workflow_and_writes_nothing(tmp_path, repo):
    out = staged_artifact(
        tmp_path, repo, lambda p: write_json(p / "data" / "pbc.json", ledger(CAPTIVES, ABIDE))
    )
    evil = out / ".github" / "workflows" / "poll.yml"
    evil.parent.mkdir(parents=True)
    evil.write_text("on: push\n")
    before = (repo / "data" / "pbc.json").read_text()

    with pytest.raises(ledger_push.LedgerPushError, match=r"\.github/workflows/poll\.yml"):
        ledger_push.apply(repo, out.parent)
    assert (repo / "data" / "pbc.json").read_text() == before
    assert not (repo / ".github").exists()


def test_apply_refuses_a_symlink_in_an_artifact(tmp_path, repo):
    out = staged_artifact(tmp_path, repo, lambda p: None)
    (out / "data").mkdir()
    (out / "data" / "hillside.json").symlink_to(REPO_ROOT / "poller" / "runner.py")

    with pytest.raises(ledger_push.LedgerPushError, match="data/hillside.json"):
        ledger_push.apply(repo, out.parent)


@pytest.mark.parametrize("base", ["HEAD", "main", "0" * 39, "$(id)"])
def test_apply_refuses_a_base_that_is_not_a_full_commit_sha(tmp_path, repo, base):
    out = staged_artifact(tmp_path, repo, lambda p: None)
    (out / ledger_push.BASE_FILE).write_text(base + "\n")

    with pytest.raises(ledger_push.LedgerPushError, match="base"):
        ledger_push.apply(repo, out.parent)


@pytest.mark.parametrize("text", ["[]", '{"guid": "not a record"}', "{truncated"])
def test_apply_refuses_a_ledger_that_is_not_records_keyed_by_guid(tmp_path, repo, text):
    out = staged_artifact(
        tmp_path, repo, lambda p: write_json(p / "data" / "pbc.json", ledger(CAPTIVES, ABIDE))
    )
    (out / "data" / "pbc.json").write_text(text)

    with pytest.raises(ledger_push.LedgerPushError, match="data/pbc.json"):
        ledger_push.apply(repo, out.parent)


# --- check-staged --------------------------------------------------------------------------


def test_check_staged_names_every_staged_path_outside_the_allowlist(repo):
    write_json(repo / "data" / "pbc.json", ledger(CAPTIVES, ABIDE))
    (repo / "data" / "README.md").write_text("# Ledger, edited\n")
    git(repo, "add", "data")

    assert ledger_push.check_staged(repo) == ["data/README.md"]


def test_check_staged_passes_a_ledger_only_change(repo):
    write_json(repo / "data" / "pbc.json", ledger(CAPTIVES, ABIDE))
    git(repo, "add", "data")
    assert ledger_push.check_staged(repo) == []


# --- the push job's interpreter ----------------------------------------------------------------


def test_the_push_path_imports_nothing_outside_the_standard_library():
    """`-S` drops site-packages, as the push job runs it: a third-party import here fails (ADR-0027)."""
    result = subprocess.run(
        [
            sys.executable,
            "-E",
            "-S",
            "-c",
            "import json, sys, poller.ledger_push; print(json.dumps(sorted(sys.modules)))",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    loaded = {name.split(".")[0] for name in json.loads(result.stdout)}
    assert loaded - set(sys.stdlib_module_names) - {"poller", "__main__"} == set()
