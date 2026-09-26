"""Backtests ``derive_preached_on`` against every historical live-feed record in
``data/*.json`` (docs/specs/0007). Two families are excluded, both documented in
the spec's "PBC feed_published_at" research (see the #51 discussion) rather than
silently dropped:

- PBC's archive-backfill records: their ``preached_on`` came from an S3 path
  date, never from any adapter's date logic (ADR-0003), so they aren't a fair
  comparison for a helper that generalizes adapter logic.
- **All** of PBC's live records: PBC's ``feed_published_at`` comes from the audio
  file's CDN ``Last-Modified`` header (pbc.py's ``resolve_feed_published_at``), which
  answers "when was this object last written", not "when did this sermon first
  become available" — a re-encode or CDN cache-bust can rewrite it long after the
  original air date. Measured directly against this ledger: 5 of PBC's 11 live
  records (45%) have a ``feed_published_at`` more than a day after ``preached_on``,
  two of them by over a week — far too frequent to be an edge case, and
  structurally the wrong signal rather than a noisy version of the right one.
  PBC's own ``preached_on`` is already reliable (100% Sunday across every live
  record) because it's classified directly off the feed's own ``pubDate``, so
  PBC never needed ``derive_preached_on`` to begin with — this backtest isn't
  the place to resolve what *should* anchor PBC's ``feed_published_at`` (a separate,
  already-scoped follow-up: PBC's own sermons page carries a human-readable
  service date per ``enmse_mid``, alongside the speaker name the adapter already
  scrapes from that same page).

This does not wire the helper into any adapter — it only confirms whether the
helper reproduces what each adapter already computed, for review before that
decision is made.
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from poller.sources.common import derive_preached_on
from poller.sources.gracepres import _split_title as _gracepres_title
from poller.sources.westgate import _parse_title as _westgate_title

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_GRACEPRES_TZ = ZoneInfo("America/Los_Angeles")

# Only Westgate and GracePres derive preached_on from the title; every other
# adapter classifies and dates off pubDate directly (title_date=None).
_TITLE_DATE_EXTRACTORS = {
    "westgate": lambda raw_title: _westgate_title(raw_title)[2],
    "gracepres": lambda raw_title: _gracepres_title(raw_title)[1],
}

# GracePres's feed_published_at is a SoundCloud upload instant in UTC, meaningless as
# a calendar day without converting to the church's own timezone first (see the
# module docstring's backtest finding). Every other church's feed_published_at is
# already in a timezone where its own calendar day is the relevant one.
_TZ_BY_CHURCH = {"gracepres": _GRACEPRES_TZ}

# PBC's feed_published_at is excluded entirely — see the module docstring.
_EXCLUDED_CHURCHES = {"pbc"}


def _live_records():
    for path in sorted(_DATA_DIR.glob("*.json")):
        church = path.stem
        if church in _EXCLUDED_CHURCHES:
            continue
        records = json.loads(path.read_text(encoding="utf-8"))
        for guid, record in records.items():
            if guid.startswith("pbc-archive:"):
                continue
            if not record.get("feed_published_at") or not record.get("preached_on"):
                continue
            yield church, guid, record


def test_derive_preached_on_matches_every_historical_live_record():
    mismatches = []
    for church, guid, record in _live_records():
        feed_published_at = datetime.fromisoformat(record["feed_published_at"])
        extractor = _TITLE_DATE_EXTRACTORS.get(church)
        title_date = extractor(record["raw_title"]) if extractor else None
        tz = _TZ_BY_CHURCH.get(church)
        computed = derive_preached_on(feed_published_at, title_date=title_date, tz=tz)
        stored = date.fromisoformat(record["preached_on"])
        if computed != stored:
            mismatches.append(
                {"church": church, "guid": guid, "stored": str(stored), "computed": str(computed)}
            )

    assert not mismatches, (
        f"derive_preached_on disagrees with the stored preached_on for "
        f"{len(mismatches)} historical record(s): {mismatches}"
    )
