# Ledger overrides

Manual, human-maintained corrections for a bad feed-sourced field on an already-discovered
sermon (ADR-0008: `docs/decisions/0008-manual-ledger-field-overrides.md`).

Nothing in this repo writes here — an entry is added and removed by hand, reviewed like any
other change, and merged onto the matching `data/<church>.json` record every time
`poller.store.load` reads it.

## When to add one

Only when the feed itself is serving wrong data for an already-discovered sermon, and waiting
for the church to fix it isn't acceptable (e.g. it's blocking transcription indefinitely). Not
for anything the poller can fix on its own re-poll.

## Shape

`data/overrides/<church>.json`, keyed by guid, holding only the field(s) being corrected:

```json
{
  "<guid>": {
    "audio_url": "https://correct.example.org/audio.mp3",
    "_reason": "feed serves a 404ing path (off by 7 days) as of 2026-09-15",
    "_added_at": "2026-09-15"
  }
}
```

- Only the fields actually wrong need to be listed — this is a partial record, not a full one.
- A guid not present in that church's `data/<church>.json` is ignored (logged as a warning), not
  applied — the sermon has to already be ledgered.
- `_reason` and `_added_at` are required by convention (not enforced in code): every entry must
  be traceable to why it exists and when, since nothing here expires automatically.

## Removing an entry

Delete it once the upstream feed is confirmed fixed. Nothing else needs to change — the next
`store.load` simply stops applying it.
