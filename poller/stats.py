"""Aggregate ledger stats, rendered into the generated section of ``data/README.md``.

Recomputed from scratch from every ``data/*.json`` ledger (via
:func:`poller.store.load`, never by reading the files directly — see
``poller/store.py``'s data-access invariant) at the end of every poll run, so
the numbers always reflect the ledger's current state, not just what one run
discovered. Each church's "latest sermon" is simply its first record, relying
on ``store.save`` having already written every ledger newest-``preached_on``
first.
"""

from __future__ import annotations

from typing import Any

from poller import store

_STATS_START = "<!-- STATS:START -->"
_STATS_END = "<!-- STATS:END -->"
_README_NAME = "README.md"


class StatsError(RuntimeError):
    """Raised when ``data/README.md`` can't be regenerated (e.g. markers missing)."""


def _church_names() -> list[str]:
    return sorted(path.stem for path in store.DATA_DIR.glob("*.json"))


def _latest(records: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    return next(iter(records.values()), None)


def _render(per_church: dict[str, dict[str, dict[str, Any]]]) -> str:
    total = sum(len(records) for records in per_church.values())
    lines = [
        "| Church | Sermons | Latest sermon | Latest date |",
        "|---|---|---|---|",
    ]
    for name, records in sorted(per_church.items()):
        latest = _latest(records)
        title = (latest or {}).get("title") or "—"
        preached_on = (latest or {}).get("preached_on") or "—"
        lines.append(f"| {name} | {len(records)} | {title} | {preached_on} |")
    lines.append("")
    lines.append(f"**Total sermons across all churches: {total}**")
    return "\n".join(lines)


def _replace_marked_section(readme: str, generated: str) -> str:
    start = readme.find(_STATS_START)
    end = readme.find(_STATS_END)
    if start == -1 or end == -1 or end < start:
        raise StatsError(
            f"data/{_README_NAME} is missing the {_STATS_START}/{_STATS_END} markers; cannot regenerate stats"
        )
    before = readme[: start + len(_STATS_START)]
    after = readme[end:]
    return f"{before}\n{generated}\n{after}"


def regenerate() -> None:
    """Recompute every church's stats and rewrite the generated section of ``data/README.md``."""
    per_church = {name: store.load(name) for name in _church_names()}
    generated = _render(per_church)

    readme_path = store.DATA_DIR / _README_NAME
    if not readme_path.exists():
        raise StatsError(f"data/{_README_NAME} does not exist; cannot regenerate stats")
    current = readme_path.read_text(encoding="utf-8")
    readme_path.write_text(_replace_marked_section(current, generated), encoding="utf-8")
