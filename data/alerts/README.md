# Discovery alerts

Each poll checks every church for a missed Sunday and for a feed that keeps failing, and emails
once per incident (ADR-0025: `docs/decisions/0025-discovery-absence-alerts.md`).

## `state.json`

Written by `poller.store` only; don't edit it. It records which incidents have already alerted:

- `missing_alerted_for`: the Sunday the last missing-sermon alert was for.
- `failing_since` / `failing_alerted`: when the current feed failure started, and whether it has
  alerted. Cleared by the next healthy poll.

## `snooze.json`

Hand-edited. Suppresses the missing-Sunday alert for a church on a known break, for every Sunday
on or before `until`:

```json
{
  "lakepointe": {
    "until": "2026-08-02",
    "_reason": "no services posted during the July series break"
  }
}
```

Keys starting with `_` are notes and are ignored. Remove the entry once the break is over. A
church that routinely posts later than its grace needs its adapter's `publish_grace` raised
instead.
