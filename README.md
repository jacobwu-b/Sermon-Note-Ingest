# Sermon Note Ingest

The intake stage of the Sermon Note platform. It discovers each church's new
Sunday sermon, transcribes it, hands the transcript to the content repo, and
wakes [Sermon-Note-Pipeline](https://github.com/jacobwu-b/Sermon-Note-Pipeline),
which writes the study note. It never generates notes itself.

## How it works

- **Discover.** [`.github/workflows/poll.yml`](./.github/workflows/poll.yml)
  runs [`poller/runner.py`](./poller/runner.py) about every 5 minutes. An
  external scheduler dispatches it, not GitHub's `schedule:`.
  - Each church has its own adapter in [`poller/sources/`](./poller/sources)
    that picks out the main Sunday sermon from other content sharing the feed
    (a traditional-service stream, a midweek podcast, a kids' program, ...).
    See each adapter's module docstring for its rule.
  - A new sermon is upserted into `data/<church>.json`, keyed by the feed's own
    guid, so a re-poll never duplicates it. Each record keeps every real field
    the feed gave: title, speaker, series, the service date, the instant the
    feed first made it available (only when the feed provides one), the
    episode and audio links, and the instant this repo first retrieved it.
  - If the church's `notify` flag is on, one email per poll lists its new
    sermons, via [Resend](https://resend.com).
- **Transcribe.** [`.github/workflows/transcribe.yml`](./.github/workflows/transcribe.yml)
  runs every 15 minutes, and `poll.yml` also dispatches it right after a
  discovery. It downloads each untranscribed sermon's audio and transcribes it
  with [faster-whisper](https://github.com/SYSTRAN/faster-whisper), spreading
  the work across parallel jobs.
- **Hand off.** Each transcript is pushed to
  [Sermon-Note-Content](https://github.com/jacobwu-b/Sermon-Note-Content) as
  `transcripts/<church>/<date>_<slug>_<guid>.txt`. Then Sermon-Note-Pipeline's
  `pipeline.yml` is dispatched with an `ingest_event` naming that sermon.
- **Watch Claude batches.** Each poll also reads Sermon-Note-Pipeline's
  registry for Claude batches it is waiting on and checks their status. When
  one has ended, the poll wakes the pipeline early. This is a speed-up only:
  the pipeline's own schedule still picks the batch up.

**Churches.** Which churches to poll, their feed URLs, and each church's
`enabled`/`notify` flags are not configured here. They come from
Sermon-Note-Pipeline's checked-in `config/churches.json` (each church's `rss`
and `ingest` sections), read at the start of every run. A run fails if the
table cannot be read.

## Configuration

| Name | Kind | Purpose |
|---|---|---|
| `PIPELINE_REPO` | repository variable | `owner/repo` of Sermon-Note-Pipeline |
| `PIPELINE_DISPATCH_TOKEN` | repository secret | Reads Pipeline's church table and registry; dispatches its `pipeline.yml` |
| `CONTENT_REPO` / `CONTENT_REPO_BRANCH` | repository variables | Where transcripts are pushed |
| `CONTENT_REPO_TOKEN` | repository secret | Push access to the content repo |
| `TRANSCRIBE_DISPATCH_TOKEN` | repository secret | Lets `poll.yml` dispatch `transcribe.yml` |
| `HF_TOKEN` | repository secret | Authenticated Whisper model download |
| `WHISPER_*`, `LOG_LEVEL` | repository variables | Transcription tuning |
| `NOTIFY_EMAIL_FROM` / `NOTIFY_EMAIL_TO` / `RESEND_API_KEY` | repository secrets | New-sermon email |
| `ANTHROPIC_API_KEY` | repository secret (optional) | Batch-status check. Without it, `poll.yml` uses Workload Identity Federation on `main` |

See [`.env.example`](./.env.example) for every variable, with details, for local runs.

## Backfill

Adding a church (or running the poller for the first time) would otherwise
notify about years of back-catalog. Seed a church's ledger from its full feed
history without sending any notification:

```bash
python -m poller.runner --backfill --church menlo
```

or via the "Poll church feeds" workflow's manual dispatch, with its `backfill`
input checked.

## Local development

```bash
pip install -r requirements-dev.txt

# The church table is read from Sermon-Note-Pipeline, so a local poll needs
# a token with Contents: read on it.
PIPELINE_REPO=jacobwu-b/Sermon-Note-Pipeline PIPELINE_DISPATCH_TOKEN=<token> \
  python -m poller.runner

pytest
ruff check .
ruff format --check .
```

## Working in this repo

- **PR protocol**: branch from `main`, squash-merge back. See
  [`.github/PULL_REQUEST_TEMPLATE.md`](./.github/PULL_REQUEST_TEMPLATE.md).
- **Repository GitHub settings** are documented in
  [`.github/repo-settings.md`](./.github/repo-settings.md).

## License

Copyright (c) 2026 Zhengyuan Wu. All Rights Reserved.

This product is protected by copyright and distributed under licenses restricting copying, distribution, and decompilation. See [`LICENSE`](./LICENSE) for full terms.
