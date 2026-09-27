"""Suite-wide isolation from Sermon-Note-Pipeline's live church table (ADR-0016)."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from poller import config


@pytest.fixture(autouse=True)
def real_church_table_fetch(monkeypatch: pytest.MonkeyPatch) -> Callable[[], str]:
    """No test reads the live table over the network; each returns the real fetch.

    A test that loads churches without supplying a table fails here rather than polling
    whatever Pipeline's ``main`` enables today. The few tests of the fetch itself put the
    returned function back.
    """
    real = config._fetch_church_table

    def refuse() -> str:
        raise AssertionError("a test loaded the live church table; use the church_table fixture")

    monkeypatch.setattr(config, "_fetch_church_table", refuse)
    return real


@pytest.fixture
def church_table(monkeypatch: pytest.MonkeyPatch) -> Callable[[str], None]:
    """Make ``config.load_churches()`` read the given JSON text as Pipeline's table."""

    def use(text: str) -> None:
        monkeypatch.setattr(config, "_fetch_church_table", lambda: text)

    return use
