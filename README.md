# Sermon Note Ingest

Polls each church's sermon RSS feed on a schedule, records every newly-published
sermon to a per-church JSON file in [`data/`](./data), and emails about it when
that church's `notify` flag is on.

## How it works

- [`.github/workflows/poll.yml`](./.github/workflows/poll.yml) runs every 5
  minutes (GitHub Actions' minimum schedule granularity) via
  [`poller/runner.py`](./poller/runner.py).
- Which churches to poll, their feed URLs, and whether each is enabled/notified
  comes from the `CHURCHES` repository **variable** — one JSON object keyed by
  church name: `{"rss": <feed url>, "enabled": bool, "notify": bool}`.
- Each church has its own adapter in [`poller/sources/`](./poller/sources) that
  knows how to tell that church's main Sunday sermon apart from other content
  sharing the same feed (a traditional-service stream, a midweek podcast, a
  kids' program, ...). See each adapter's module docstring for its rule.
- A newly-discovered sermon is upserted into `data/<church>.json`, keyed by the
  feed's own guid — a re-poll of an already-known sermon never duplicates it.
  Each record keeps every real field the feed gave us: title, speaker, series,
  the service date, the instant it was first made available (when the feed
  provides one — never a placeholder value), the episode/audio links, and the
  instant this poller first retrieved it.
- If the church's `notify` flag is on, one email goes out per poll for that
  church's newly-discovered sermons, via [Resend](https://resend.com).

## Configuration

| Name | Kind | Purpose |
|---|---|---|
| `CHURCHES` | repository variable | JSON: `{name: {rss, enabled, notify}}` per church |
| `NOTIFY_EMAIL_FROM` | repository secret | Resend "from" address |
| `NOTIFY_EMAIL_TO` | repository secret | notification recipient |
| `RESEND_API_KEY` | repository secret | Resend API key |

See [`.env.example`](./.env.example) for the same schema, for local runs.

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

CHURCHES='{"menlo": {"rss": "https://feed.podbean.com/menlochurchvideo/feed.xml", "enabled": true, "notify": false}}' \
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
