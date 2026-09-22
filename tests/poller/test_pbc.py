from datetime import UTC, datetime

from poller.sources.base import SermonItem
from poller.sources.pbc import (
    PbcAdapter,
    _split_title,
    episode_url_from_link,
    is_main_sunday_sermon,
    parse_sermons_page_dates,
    parse_sermons_page_speakers,
)

_SUNDAY = 6
_MONDAY = 0

_SERMONS_PAGE_HTML = b"""
<div class="card">
  <h6>September 6, 2026</h6>
  <h5>No Middle Ground</h5>
  <p class="enmse-speaker-name">Dan Westman</p>
  <a href="?enmse_mid=4690">Watch</a>
</div>
"""


def _item(*, episode_url: str, audio_url: str = "https://cdn.pbc.org/Main_Service/ep.mp3") -> SermonItem:
    return SermonItem(
        guid="guid-1",
        title="No Middle Ground",
        raw_title="No Middle Ground",
        series=None,
        speaker=None,
        published_on="2026-09-06",
        published_at=None,
        episode_url=episode_url,
        audio_url=audio_url,
        blurb="",
    )


def test_is_main_sunday_sermon_requires_main_service_segment_and_sunday():
    assert is_main_sunday_sermon("https://cdn.pbc.org/Main_Service/ep1.mp3", _SUNDAY) is True


def test_is_main_sunday_sermon_excludes_other_service_segments():
    assert is_main_sunday_sermon("https://cdn.pbc.org/Youth_Service/ep1.mp3", _SUNDAY) is False


def test_is_main_sunday_sermon_excludes_non_sunday_main_service():
    assert is_main_sunday_sermon("https://cdn.pbc.org/Main_Service/ep1.mp3", _MONDAY) is False


def test_split_title_separates_trailing_series():
    assert _split_title("Hear and Do - Luke") == ("Hear and Do", "Luke")


def test_split_title_with_no_series_suffix_keeps_whole_title():
    assert _split_title("Hear and Do") == ("Hear and Do", None)


def test_episode_url_from_link_rebuilds_against_sermons_page():
    raw = "https://pbc.org?enmse_mid=4687"
    assert episode_url_from_link(raw) == "https://pbc.org/sermons?enmse=1&enmse_am=1&enmse_mid=4687"


def test_episode_url_from_link_passes_through_when_no_mid():
    raw = "https://pbc.org/some-page"
    assert episode_url_from_link(raw) == raw


def test_parse_sermons_page_speakers_extracts_mid_to_speaker_mapping():
    html = b"""
    <div class="card">
      <h6>September 6, 2026</h6>
      <h5>Hear and Do</h5>
      <p class="enmse-speaker-name">Jane Doe</p>
      <a href="?enmse_mid=4687">Watch</a>
    </div>
    """
    assert parse_sermons_page_speakers(html) == {"4687": "Jane Doe"}


def test_parse_sermons_page_dates_extracts_mid_to_air_date_mapping():
    assert parse_sermons_page_dates(_SERMONS_PAGE_HTML) == {"4690": datetime(2026, 9, 6, tzinfo=UTC)}


def test_parse_sermons_page_dates_skips_a_card_whose_date_does_not_parse():
    html = b"""
    <div class="card">
      <h6>not a date</h6>
      <h5>Hear and Do</h5>
      <p class="enmse-speaker-name">Jane Doe</p>
      <a href="?enmse_mid=4687">Watch</a>
    </div>
    """
    assert parse_sermons_page_dates(html) == {}


def test_resolve_published_at_prefers_the_sermons_page_date_over_cdn_last_modified():
    adapter = PbcAdapter(
        url="https://pbc.org/feed",
        fetch_last_modified=lambda url: datetime(2026, 9, 9, tzinfo=UTC),
    )
    adapter._page_dates = parse_sermons_page_dates(_SERMONS_PAGE_HTML)
    item = _item(episode_url="https://pbc.org/sermons?enmse=1&enmse_am=1&enmse_mid=4690")

    assert adapter.resolve_published_at(item) == datetime(2026, 9, 6, tzinfo=UTC)


def test_resolve_published_at_falls_back_to_cdn_last_modified_when_mid_not_on_page():
    fallback = datetime(2026, 9, 9, tzinfo=UTC)
    adapter = PbcAdapter(url="https://pbc.org/feed", fetch_last_modified=lambda url: fallback)
    adapter._page_dates = parse_sermons_page_dates(_SERMONS_PAGE_HTML)
    item = _item(episode_url="https://pbc.org/sermons?enmse=1&enmse_am=1&enmse_mid=9999")

    assert adapter.resolve_published_at(item) == fallback
