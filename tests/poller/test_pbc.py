import functools
import io
import urllib.error
from datetime import UTC, datetime

from poller import net
from poller.sources import pbc
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

_FEED_URL = "https://pbc.org/?feed=seriesengine&enmse_pid=9000"
_SERMONS_URL = "https://pbc.org/sermons/"
_FEED_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>PBC Sermons</title>
<item>
  <title>No Middle Ground - Luke</title>
  <guid>https://pbc.org/?enmse_mid=4690</guid>
  <link>https://pbc.org/?enmse=1&amp;enmse_mid=4690</link>
  <pubDate>Sun, 06 Sep 2026 17:00:00 +0000</pubDate>
  <enclosure url="https://cdn.pbc.org/Main_Service/2026/09/06/No_Middle_Ground.mp3" type="audio/mpeg" length="1"/>
</item>
</channel></rss>"""


def _serve(monkeypatch, pages: dict[str, bytes]) -> dict[str, str | None]:
    """Answer ``urlopen`` from ``pages`` (unknown URLs fail) and record each request's User-Agent."""
    seen: dict[str, str | None] = {}

    def fake_urlopen(request, timeout=None):
        seen[request.full_url] = request.get_header("User-agent")
        if request.full_url not in pages:
            raise urllib.error.URLError("connection refused")
        return io.BytesIO(pages[request.full_url])

    monkeypatch.setattr(net.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(pbc, "fetch_feed", functools.partial(net.fetch_feed, sleep=lambda _: None))
    return seen


def _item(*, episode_url: str, audio_url: str = "https://cdn.pbc.org/Main_Service/ep.mp3") -> SermonItem:
    return SermonItem(
        guid="guid-1",
        title="No Middle Ground",
        raw_title="No Middle Ground",
        series=None,
        speaker=None,
        preached_on="2026-09-06",
        feed_published_at=None,
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


def test_resolve_feed_published_at_prefers_the_sermons_page_date_over_cdn_last_modified():
    adapter = PbcAdapter(
        url="https://pbc.org/feed",
        fetch_last_modified=lambda url: datetime(2026, 9, 9, tzinfo=UTC),
    )
    adapter._page_dates = parse_sermons_page_dates(_SERMONS_PAGE_HTML)
    item = _item(episode_url="https://pbc.org/sermons?enmse=1&enmse_am=1&enmse_mid=4690")

    assert adapter.resolve_feed_published_at(item) == datetime(2026, 9, 6, tzinfo=UTC)


def test_resolve_feed_published_at_falls_back_to_cdn_last_modified_when_mid_not_on_page():
    fallback = datetime(2026, 9, 9, tzinfo=UTC)
    adapter = PbcAdapter(url="https://pbc.org/feed", fetch_last_modified=lambda url: fallback)
    adapter._page_dates = parse_sermons_page_dates(_SERMONS_PAGE_HTML)
    item = _item(episode_url="https://pbc.org/sermons?enmse=1&enmse_am=1&enmse_mid=9999")

    assert adapter.resolve_feed_published_at(item) == fallback


def test_poll_identifies_itself_with_the_project_user_agent_on_the_feed_and_sermons_page(monkeypatch):
    seen = _serve(monkeypatch, {_FEED_URL: _FEED_XML, _SERMONS_URL: _SERMONS_PAGE_HTML})

    result = PbcAdapter(url=_FEED_URL, sermons_url=_SERMONS_URL).poll()

    assert seen == {_FEED_URL: net.DEFAULT_USER_AGENT, _SERMONS_URL: net.DEFAULT_USER_AGENT}
    assert [item.speaker for item in result.items] == ["Dan Westman"]


def test_poll_warns_and_still_returns_feed_items_when_the_sermons_page_fetch_fails(monkeypatch, caplog):
    _serve(monkeypatch, {_FEED_URL: _FEED_XML})

    with caplog.at_level("WARNING", logger="poller"):
        result = PbcAdapter(url=_FEED_URL, sermons_url=_SERMONS_URL).poll()

    assert [(item.title, item.speaker) for item in result.items] == [("No Middle Ground", None)]
    assert any(_SERMONS_URL in record.getMessage() for record in caplog.records)
