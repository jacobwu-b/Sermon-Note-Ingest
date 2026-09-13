"""The Peninsula Bible Church source adapter: SeriesEngine podcast feed + PBC
classification (ADR-0013, spec 0013).

This is the PBC ingestion boundary (CLAUDE.md §6): it owns the only feed fetch for
this source. Its feed url is supplied by :func:`sources.enabled_adapters` from
``CHURCHES`` (ADR-0074) rather than read here directly, and it follows the
retry posture in PRD §11.1 — three attempts with exponential backoff, then the
source defers to the next schedule rather than crashing.

PBC publishes through the WordPress SeriesEngine podcast feed
(``pbc.org/?feed=seriesengine&enmse_pid=9000``) rather than the Podbean feed the
Menlo path expects. The feed's channel description carries a stale "no longer
updated" note, but the feed is in fact current (verified against live data) and is
the only in-scope pbc.org sermon feed — pbcc.org (Cupertino) is a different church.
Daily reconciliation catches any future drift.

PBC's classification is simpler than Menlo's (PRD §5.3): there is no Legacy/Midweek
denylist and no late-posted-sermon promotion. PBC organizes its CDN audio by service
type, so the main Sunday sermon is exactly an item whose audio is served from the
``Main_Service`` segment and whose publication day is Sunday; every other service
type or non-Sunday item is excluded.

:class:`PbcAdapter` ties the unit together: fetch → parse → select the main Sunday
sermons → upsert as ``discovered`` (idempotent on the feed ``guid``).
"""

from __future__ import annotations

from datetime import datetime

import re
import urllib.parse
from typing import Any, Callable

import feedparser

from sermon_notes import config
from sermon_notes.logging import get_logger
from sermon_notes.registry import Registry, SermonRecord
from sermon_notes.sources.base import FeedItem, PollResult, SourceAdapter
from sermon_notes.sources.feedbase import (
    FeedFetchError,
    entry_audio_url,
    entry_has_identity,
    extract_scripture_refs,
    fetch_feed,
    http_get,
    http_head_last_modified,
    now,
    parse_pubdate,
    parse_pubdate_at,
)

logger = get_logger()

# A real-browser User-Agent: PBC sits behind a CDN/WAF that can reject bot-shaped
# agents (spec 0013), the same posture the audio-download stage already uses.
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)

# PBC organizes its CDN audio by service type; the main Sunday sermon is served from
# this segment. Other service types live under other segments and are excluded.
_MAIN_SERVICE_SEGMENT = "/Main_Service/"
# The feed titles a sermon as "{title} - {series}"; this splits the trailing series.
_TITLE_SERIES_SEP = " - "
# email.utils weekday convention used throughout the feed layer: Sunday == 6.
_SUNDAY = 6
# The scrape-fallback page (spec 0013) that carries the per-episode speaker the feed
# omits; overridable via PBC_SERMONS_URL for the same reason the feed URL is.
_DEFAULT_SERMONS_URL = "https://pbc.org/sermons/"

# The feed's own <link> (e.g. "https://pbc.org?enmse_mid=4687") does not resolve to the
# episode on pbc.org's current site — it 404s through to the homepage (issue #332). The
# sermons page's own "Watch"/"Listen" card links use this query shape against
# /sermons instead, and that shape does land on the specific episode, so episode_url is
# built from it rather than passed through from the feed verbatim.
_EPISODE_URL_TEMPLATE = "https://pbc.org/sermons?enmse=1&enmse_am=1&enmse_mid={mid}"

# PBC's SeriesEngine feed carries no per-episode speaker (spec 0013's "metadata
# gaps" risk); pbc.org/sermons does, in its "card" markup, keyed by the same
# ``enmse_mid`` the feed uses in its item links. This narrow regex reads just that
# — mirroring the spec's documented scrape-fallback approach — without adding an
# HTML-parsing dependency (CLAUDE.md §6 approval-gates any such dependency).
_SERMONS_PAGE_CARD_RE = re.compile(
    r"<h5>(?P<title>[^<]*)</h5>\s*"
    r'<p class="enmse-speaker-name">(?P<speaker>[^<]*)</p>.*?'
    r"enmse_mid=(?P<mid>\d+)",
    re.DOTALL,
)


def _fetch(url: str) -> bytes:
    """The adapter's default feed fetch: shared retry over a UA-bound hardened GET.

    PBC binds its own browser User-Agent (the module's ``_USER_AGENT``) to clear the
    CDN/WAF (spec 0013); the http(s) guard, body cap, and backoff are the shared
    transport in :mod:`sermon_notes.sources.feedbase`.
    """
    return fetch_feed(url, get=lambda target: http_get(target, user_agent=_USER_AGENT))


def _fetch_sermons_page(url: str) -> bytes:
    """The adapter's default sermons-page fetch: same hardened transport as ``_fetch``."""
    return fetch_feed(url, get=lambda target: http_get(target, user_agent=_USER_AGENT))


def _mid_from_url(url: str) -> str | None:
    """Extract the ``enmse_mid`` query parameter from a PBC episode URL, if present."""
    values = urllib.parse.parse_qs(urllib.parse.urlsplit(url).query).get("enmse_mid")
    return values[0] if values else None


def parse_sermons_page_speakers(content: bytes) -> dict[str, str]:
    """Map each ``enmse_mid`` on the PBC sermons page to its speaker name.

    Reads the SeriesEngine "card" markup on ``pbc.org/sermons`` — the scrape
    fallback spec 0013 documents — to recover the per-episode speaker the podcast
    feed itself never publishes. A sermon whose mid isn't found here is simply
    absent from the result; callers fall back to a null speaker.
    """
    text = content.decode("utf-8", errors="replace")
    speakers: dict[str, str] = {}
    for match in _SERMONS_PAGE_CARD_RE.finditer(text):
        speakers[match.group("mid")] = match.group("speaker").strip()
    return speakers


def _split_title(raw_title: str) -> tuple[str, str | None]:
    """Split a ``"{title} - {series}"`` feed title into its title and series parts.

    PBC titles append the series after a final ``" - "`` (e.g. ``"Hear and Do -
    Luke"``). A title with no such suffix keeps the whole string as the title and
    leaves the series null, which the record tolerates.
    """
    title, sep, series = raw_title.rpartition(_TITLE_SERIES_SEP)
    if not sep:
        return raw_title, None
    return title, series


def episode_url_from_link(raw_link: str) -> str:
    """The episode link to publish: the feed's ``enmse_mid`` rebuilt against
    ``/sermons`` (the shape that actually resolves to the episode), or the feed's raw
    link unchanged when it carries no ``enmse_mid`` to rebuild from.

    Public (not module-private) because ``scripts/backfill_pbc_episode_url.py`` (#332)
    reuses this exact rebuild logic to repair already-published records — the fix and
    the backfill must compute the identical URL, so there is one function, not two.
    """
    mid = _mid_from_url(raw_link)
    return _EPISODE_URL_TEMPLATE.format(mid=mid) if mid is not None else raw_link


def _entry_to_item(entry: Any) -> FeedItem:
    """Convert a feedparser entry into a :class:`FeedItem`."""
    raw_title = (entry.get("title") or "").strip()
    title, series = _split_title(raw_title)
    published_on, weekday = parse_pubdate(entry.get("published"))
    blurb = (entry.get("summary") or "").strip()
    return FeedItem(
        guid=entry.get("id", ""),
        raw_title=raw_title,
        title=title,
        series=series,
        # The PBC feed names the church as author, not the preacher (spec 0013); the
        # individual speaker is not published in the feed, so it is left null.
        speaker=None,
        published_on=published_on,
        published_weekday=weekday,
        published_at=parse_pubdate_at(entry.get("published")),
        episode_url=episode_url_from_link(entry.get("link", "")),
        audio_url=entry_audio_url(entry),
        blurb=blurb,
        scripture_refs=extract_scripture_refs(blurb),
    )


def parse_feed(content: bytes) -> tuple[list[FeedItem], int]:
    """Parse raw feed bytes into :class:`FeedItem` objects, in feed order.

    Returns ``(items, skipped)``: an entry with neither ``<guid>`` nor ``<link>``
    has no usable identity for the ledger's guid-keyed primary key, so it is
    skipped (logged) rather than defaulted to an empty, collision-prone guid
    (issue #206); ``skipped`` counts how many were dropped this way.
    """
    parsed = feedparser.parse(content)
    items: list[FeedItem] = []
    skipped = 0
    for entry in parsed.entries:
        if not entry_has_identity(entry):
            logger.warning(
                "skip identity-less feed item (no guid or link): %r",
                (entry.get("title") or "").strip(),
            )
            skipped += 1
            continue
        items.append(_entry_to_item(entry))
    return items, skipped


# --- classification (spec 0013) --------------------------------------------


def is_main_sunday_sermon(item: FeedItem) -> bool:
    """Return whether ``item`` is PBC's main Sunday sermon (spec 0013).

    The audio served from the ``Main_Service`` CDN segment is PBC's main service;
    other service types live under other segments. Combined with a Sunday
    publication day, that uniquely identifies the main Sunday sermon — every other
    service type or non-Sunday item is excluded.
    """
    return _MAIN_SERVICE_SEGMENT in item.audio_url and item.published_weekday == _SUNDAY


# --- adapter ---------------------------------------------------------------


class PbcAdapter(SourceAdapter):
    """Ingest Peninsula Bible Church sermons from the SeriesEngine feed (ADR-0013).

    The feed URL and fetch function are injectable so the HTTP boundary is mocked
    in tests; in production ``url`` is always supplied by
    :func:`sources.enabled_adapters` from ``CHURCHES`` (ADR-0074), and the
    fetch is :func:`_fetch` — the shared retry transport bound to PBC's browser
    User-Agent. The sermons-page URL and fetch (``PBC_SERMONS_URL`` /
    :func:`_fetch_sermons_page`) are unaffected by ADR-0074 and still resolve from
    their own config var, enriching each item with the speaker the feed itself
    omits (spec 0013).
    """

    source = "pbc"

    def __init__(
        self,
        *,
        url: str | None = None,
        fetch: Callable[[str], bytes] = _fetch,
        sermons_url: str | None = None,
        fetch_sermons_page: Callable[[str], bytes] = _fetch_sermons_page,
        fetch_last_modified: Callable[[str], datetime | None] = http_head_last_modified,
    ) -> None:
        self._url = url
        self._fetch = fetch
        self._sermons_url = sermons_url
        self._fetch_sermons_page = fetch_sermons_page
        self._fetch_last_modified = fetch_last_modified

    def _to_record(
        self,
        item: FeedItem,
        *,
        now: str,
        speakers: dict[str, str],
        published_at: str | None,
    ) -> SermonRecord:
        """Build a freshly-discovered :class:`SermonRecord` from a feed item.

        ``speakers`` is the mid-to-name mapping scraped from the sermons page; a
        matching mid fills in what the feed itself leaves null. ``published_at`` is the
        already-resolved ISO-8601 instant to persist (ADR-0056) — the audio enclosure's
        CDN ``Last-Modified`` on first discovery, or an existing record's value carried
        forward on re-poll. PBC's own ``pubDate`` (``item.published_at``) is a constant
        nominal value and is never used here, unlike the other adapters.
        """
        mid = _mid_from_url(item.episode_url)
        speaker = item.speaker or (speakers.get(mid) if mid is not None else None)
        return SermonRecord(
            guid=item.guid,
            source=self.source,
            episode_url=item.episode_url,
            audio_url=item.audio_url,
            published_on=item.published_on,
            published_at=published_at,
            title=item.title,
            series=item.series,
            speaker=speaker,
            scripture_refs=list(item.scripture_refs),
            blurb=item.blurb,
            transcript_hash=None,
            state="discovered",
            artifact_path=None,
            first_seen_at=now,
            last_state_change_at=now,
            runs=[],
        )

    def _fetch_speakers(self) -> dict[str, str]:
        """Best-effort mid-to-speaker mapping from the sermons page.

        Enrichment is never load-bearing for discovery: a fetch failure logs a
        warning and yields an empty mapping rather than deferring the whole poll.
        """
        sermons_url = (
            self._sermons_url
            if self._sermons_url is not None
            else config.get("PBC_SERMONS_URL", _DEFAULT_SERMONS_URL)
        )
        try:
            content = self._fetch_sermons_page(sermons_url)
        except FeedFetchError as exc:
            logger.warning(
                "sermons page unavailable; %s poll continues without speaker enrichment: %s",
                self.source,
                exc,
            )
            return {}
        return parse_sermons_page_speakers(content)

    def poll(self, registry: Registry, *, limit: int | None = None) -> PollResult:
        """Fetch, parse, select, and upsert main PBC Sunday sermons as ``discovered``.

        Main Sunday sermons are upserted as ``discovered`` (idempotent on guid, so a
        re-poll never duplicates); every other service type or non-Sunday item is
        excluded. When ``limit`` is given only the ``limit`` most recent qualifying
        sermons (by publication date) are upserted — the v1 working set (PRD §6.6) —
        while exclusion still covers the whole feed. A :class:`FeedFetchError` defers
        the source — no crash, no records written — leaving the next schedule to retry.
        """
        if self._url is None:
            raise ValueError(
                f"{type(self).__name__}.poll() requires a feed url; "
                "sources.enabled_adapters() always supplies one in production."
            )
        try:
            content = self._fetch(self._url)
        except FeedFetchError as exc:
            logger.warning("feed unavailable; deferring %s poll to next run: %s", self.source, exc)
            return PollResult(deferred=True)

        polled_at = now()
        items, id_less = parse_feed(content)
        included: list[FeedItem] = []
        excluded = id_less
        for item in items:
            if is_main_sunday_sermon(item):
                included.append(item)
            else:
                excluded += 1

        included.sort(key=lambda item: item.published_on, reverse=True)
        if limit is not None:
            included = included[:limit]
        speakers = self._fetch_speakers()
        discovered: list[str] = []
        publish_times: dict[str, datetime] = {}
        for item in included:
            # The CDN HEAD (ADR-0056) is only worth its cost the first time a guid is
            # seen: an already-known guid keeps whatever ``published_at`` its first
            # discovery recorded (metadata refresh must not clobber it back to ``None``
            # on a re-poll), and re-fetching on every re-poll would turn a ~1/week cost
            # into one per poll for no new information.
            existing = registry.get(item.guid)
            if existing is None:
                fetched = self._fetch_last_modified(item.audio_url)
                published_at = fetched.isoformat() if fetched is not None else None
            else:
                published_at = existing.published_at
            registry.upsert(
                self._to_record(item, now=polled_at, speakers=speakers, published_at=published_at)
            )
            discovered.append(item.guid)
            if item.published_at is not None:
                publish_times[item.guid] = item.published_at

        logger.info(
            "%s poll complete: %d discovered, %d excluded",
            self.source,
            len(discovered),
            excluded,
        )
        return PollResult(discovered=discovered, excluded=excluded, publish_times=publish_times)
