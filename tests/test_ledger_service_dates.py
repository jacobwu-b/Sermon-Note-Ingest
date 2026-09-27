"""Every ledgered sermon's ``preached_on`` is a date a sermon can be preached on.

Asserted over the committed ledgers, read through ``store.load`` so overrides apply,
rather than per adapter: an off-day date has reached them from both an adapter rule
and a backfill, and this is the one place every path ends up.
"""

from __future__ import annotations

from datetime import date

import pytest

from poller import store
from poller.sources.common import is_service_date

_CHURCHES = sorted(path.stem for path in store.DATA_DIR.glob("*.json"))


def test_the_committed_ledgers_are_found():
    assert "menlo" in _CHURCHES
    assert "pbc" in _CHURCHES


@pytest.mark.parametrize("church", _CHURCHES)
def test_every_ledgered_preached_on_is_a_sunday_or_special_service(church):
    off_day = {
        guid: record["preached_on"]
        for guid, record in store.load(church).items()
        if record.get("preached_on") and not is_service_date(date.fromisoformat(record["preached_on"]))
    }
    assert off_day == {}, (
        f"{church} ledgers {len(off_day)} sermon(s) on neither a Sunday nor a special service: "
        f"{off_day}. Correct the date with a data/overrides/{church}.json entry, or, if the "
        "service is real, add its day to poller/sources/common.py:is_service_date."
    )
