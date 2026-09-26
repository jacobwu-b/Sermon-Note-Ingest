import pytest

from poller import stats, store
from poller.sources.base import SermonItem

_MARKED_README = """# Sermon ledger

Hand-written schema docs live here.

## Ledger stats

<!-- STATS:START -->
placeholder
<!-- STATS:END -->

These files are committed automatically.
"""


def _item(guid: str, *, title: str, preached_on: str) -> SermonItem:
    return SermonItem(
        guid=guid,
        title=title,
        raw_title=title,
        series=None,
        speaker=None,
        preached_on=preached_on,
        feed_published_at=None,
        episode_url="https://example.org/ep",
        audio_url="https://example.org/ep.mp3",
        blurb="",
    )


def _seed(church: str, *items: SermonItem) -> None:
    records = {
        item.guid: store.item_to_record(item, first_seen_at="t", feed_published_at=None) for item in items
    }
    store.save(church, records)


def test_regenerate_writes_totals_and_latest_sermon_per_church(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    (tmp_path / "README.md").write_text(_MARKED_README, encoding="utf-8")
    _seed(
        "menlo",
        _item("g1", title="Hear and Do", preached_on="2026-08-30"),
        _item("g2", title="Take Up Your Cross", preached_on="2026-09-06"),
    )
    _seed("pbc", _item("g3", title="On the Mountaintop", preached_on="2026-07-12"))

    stats.regenerate()

    content = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "Hand-written schema docs live here." in content
    assert "These files are committed automatically." in content
    assert "menlo" in content and "2" in content
    assert "Take Up Your Cross" in content  # menlo's latest, by preached_on
    assert "pbc" in content and "On the Mountaintop" in content
    assert "3" in content  # repo-wide total across both churches


def test_regenerate_replaces_only_the_marked_section_on_rerun(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    (tmp_path / "README.md").write_text(_MARKED_README, encoding="utf-8")
    _seed("menlo", _item("g1", title="Hear and Do", preached_on="2026-08-30"))
    stats.regenerate()
    first = (tmp_path / "README.md").read_text(encoding="utf-8")

    _seed("menlo", _item("g1", title="Hear and Do", preached_on="2026-08-30"))
    _seed("pbc", _item("g2", title="New One", preached_on="2026-09-06"))
    stats.regenerate()
    second = (tmp_path / "README.md").read_text(encoding="utf-8")

    assert first != second
    assert "Hand-written schema docs live here." in second
    assert "These files are committed automatically." in second
    assert second.count("<!-- STATS:START -->") == 1
    assert second.count("<!-- STATS:END -->") == 1


def test_regenerate_handles_a_church_with_no_ledger_records(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    (tmp_path / "README.md").write_text(_MARKED_README, encoding="utf-8")
    (tmp_path / "empty.json").write_text("{}", encoding="utf-8")

    stats.regenerate()  # must not raise

    content = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "empty" in content


def test_regenerate_raises_when_readme_markers_are_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(store, "DATA_DIR", tmp_path)
    (tmp_path / "README.md").write_text("# Sermon ledger\n\nNo markers here.\n", encoding="utf-8")
    _seed("menlo", _item("g1", title="Hear and Do", preached_on="2026-08-30"))

    with pytest.raises(stats.StatsError):
        stats.regenerate()
