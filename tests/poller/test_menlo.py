from poller.sources.base import SermonItem
from poller.sources.menlo import Classification, classify, classify_feed

_SUNDAY = 6
_MONDAY = 0


def _item(guid: str, raw_title: str, preached_on: str) -> SermonItem:
    return SermonItem(
        guid=guid,
        title=raw_title,
        raw_title=raw_title,
        series=None,
        speaker=None,
        preached_on=preached_on,
        feed_published_at=None,
        episode_url="",
        audio_url="",
        blurb="",
    )


def test_classify_excludes_legacy_service_by_title_suffix():
    assert (
        classify("Grace Alone | John | Pastor Dave | Menlo Church Legacy Service", _SUNDAY)
        is Classification.EXCLUDE
    )


def test_classify_excludes_midweek_podcast_by_title_marker():
    assert classify("Menlo Midweek Podcast: A Chat", _SUNDAY) is Classification.EXCLUDE


def test_classify_includes_sunday_item_that_is_neither():
    assert classify("Grace Alone | John | Pastor Dave", _SUNDAY) is Classification.INCLUDE


def test_classify_flags_non_sunday_non_excluded_item_as_ambiguous():
    assert classify("Grace Alone | John | Pastor Dave", _MONDAY) is Classification.AMBIGUOUS


def test_classify_feed_promotes_late_posted_sermon_when_week_has_no_sunday_item():
    monday = _item("g1", "Late Sermon | John | Pastor Dave", "2026-09-07")  # Monday
    verdicts = classify_feed([(monday, _MONDAY)])
    assert verdicts["g1"] is Classification.INCLUDE


def test_classify_feed_does_not_promote_when_sunday_item_already_claims_the_week():
    sunday = _item("g1", "Grace Alone | John | Pastor Dave", "2026-09-06")  # Sunday
    monday = _item("g2", "Random Clip", "2026-09-07")  # same week, Monday
    verdicts = classify_feed([(sunday, _SUNDAY), (monday, _MONDAY)])
    assert verdicts["g1"] is Classification.INCLUDE
    assert verdicts["g2"] is Classification.AMBIGUOUS


def test_classify_feed_promotes_the_ambiguous_item_closest_to_sunday():
    monday = _item("closer", "Clip A", "2026-09-07")  # Monday: 1 day after Sunday
    wednesday = _item("farther", "Clip B", "2026-09-09")  # Wednesday: 3 days after
    verdicts = classify_feed([(monday, _MONDAY), (wednesday, 2)])
    assert verdicts["closer"] is Classification.INCLUDE
    assert verdicts["farther"] is Classification.AMBIGUOUS


def test_classify_feed_never_promotes_an_undated_item():
    undated = _item("g1", "Untitled Clip", "")
    verdicts = classify_feed([(undated, -1)])
    assert verdicts["g1"] is Classification.AMBIGUOUS
