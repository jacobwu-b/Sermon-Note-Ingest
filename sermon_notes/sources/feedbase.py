"""Shared feed-fetch transport and parsing helpers for source adapters (ADR-0013, issue #67).

Every source adapter pulls its sermons from an HTTP feed and normalizes the same
handful of fields. The fetch transport in particular carries security and retry
hardening that must not drift between adapters: the http(s) scheme guard and 64 MiB
body cap (issue #36) and the three-try exponential backoff (PRD §11.1). Those, plus
the small feed helpers every adapter shares (pubdate parsing, enclosure URL, scripture
reference extraction), live here so a fix lands once.

Each adapter still owns its own fetch *configuration* — its feed URL (read through the
config layer) and its User-Agent — and composes this transport with that configuration;
PBC, for instance, binds a browser User-Agent to clear its CDN/WAF (spec 0013). What is
shared is the hardened mechanism, not the per-church wiring (ADR-0013).
"""

from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
from datetime import datetime as _datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Callable

from sermon_notes import scripture
from sermon_notes.logging import get_logger
from sermon_notes.net_guard import UnsafeHostError, build_guarded_opener, guard_public_host

logger = get_logger()

# The pipeline's default feed User-Agent. Menlo's Podbean feed accepts it as-is; PBC
# binds its own browser User-Agent instead to clear a CDN/WAF (spec 0013).
_DEFAULT_USER_AGENT = "sermon-notes/0.1.0 (+https://github.com/jacobwu-b/Sermon-Note-Pipeline)"
_FETCH_ATTEMPTS = 3
_FETCH_BACKOFF_BASE = 1.0
_FETCH_TIMEOUT = 30
# Defense-in-depth on a privileged runner (issue #36): only fetch over http(s),
# and cap the buffered body. A podcast RSS feed is a few MB at most; 64 MiB is
# generous headroom while still refusing a body crafted to exhaust memory/disk.
# The host is also validated (issue #129) so a hostile feed URL can't reach an
# internal address, and that validation is re-applied to any redirect target the feed
# sends back (issue #203); see sermon_notes.net_guard for the guard and its scope.
_ALLOWED_SCHEMES = frozenset({"http", "https"})
_MAX_FEED_BYTES = 64 * 1024 * 1024

# Shared across calls so redirect handling is configured once (issue #203).
_opener = build_guarded_opener()

# Scripture matching lives in the scripture module so the recognized book set and
# dash handling have one source of truth (issue #41). Here we scan free text, so we
# wrap the shared grammar with a leading word boundary instead of anchoring it.
_SCRIPTURE_RE = re.compile(r"\b" + scripture.REFERENCE_PATTERN)


class FeedFetchError(RuntimeError):
    """Raised when a feed cannot be fetched after the configured retries."""


def now() -> str:
    """Current UTC time as an ISO-8601 string."""
    return _datetime.now(timezone.utc).isoformat()


def http_get(url: str, *, user_agent: str = _DEFAULT_USER_AGENT) -> bytes:
    """Fetch ``url`` over HTTP and return the raw body (the mocked boundary).

    Every refusal leaves as :class:`FeedFetchError`, including one the guarded opener
    raises part-way through the request when it refuses a redirect target (issue #229).
    An :class:`UnsafeHostError` escaping here would be a bare ``ValueError`` outside this
    boundary's contract: neither :func:`fetch_feed` (which catches ``OSError``) nor an
    adapter's ``poll`` (which catches ``FeedFetchError``) would handle it, so a single
    hostile or misconfigured redirect would abort the whole run instead of deferring one
    source (ADR-0013).
    """
    scheme = urllib.parse.urlsplit(url).scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise FeedFetchError(f"refusing to fetch non-http(s) feed URL: {url!r}")
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        guard_public_host(url)
        with _opener.open(request, timeout=_FETCH_TIMEOUT) as response:
            body: bytes = response.read(_MAX_FEED_BYTES + 1)
    except UnsafeHostError as exc:
        raise FeedFetchError(str(exc)) from exc
    if len(body) > _MAX_FEED_BYTES:
        raise FeedFetchError(f"feed body exceeds {_MAX_FEED_BYTES}-byte cap: {url!r}")
    return body


def http_head_last_modified(url: str, *, user_agent: str = _DEFAULT_USER_AGENT) -> _datetime | None:
    """HEAD ``url`` and return its ``Last-Modified`` instant, or ``None`` if unusable.

    Reuses :func:`http_get`'s scheme and host guards (ADR-0056): a HEAD carries the same
    SSRF exposure as a GET, so it goes through the same ``guard_public_host`` check and
    the same guarded opener rather than a bare ``urllib`` call. Unlike :func:`http_get`
    this never raises :class:`FeedFetchError` — a missing header, a non-2xx response, or
    a network error all yield ``None``, since the timestamp is advisory (measurement),
    not load-bearing for discovery (PBC's poll must not defer because a CDN omitted a
    header).
    """
    scheme = urllib.parse.urlsplit(url).scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        return None
    request = urllib.request.Request(url, headers={"User-Agent": user_agent}, method="HEAD")
    try:
        guard_public_host(url)
        with _opener.open(request, timeout=_FETCH_TIMEOUT) as response:
            raw = response.headers.get("Last-Modified")
    except UnsafeHostError, OSError:
        return None
    if not raw:
        return None
    try:
        when = parsedate_to_datetime(raw)
    except ValueError:
        return None
    return when if when.tzinfo is not None else when.replace(tzinfo=timezone.utc)


def fetch_feed(
    url: str,
    *,
    get: Callable[[str], bytes] = http_get,
    sleep: Callable[[float], None] = time.sleep,
    attempts: int = _FETCH_ATTEMPTS,
    backoff_base: float = _FETCH_BACKOFF_BASE,
) -> bytes:
    """Fetch the feed body, retrying transient errors with exponential backoff.

    Per PRD §11.1: up to ``attempts`` tries, backing off ``backoff_base * 2**n``
    seconds between them. Exhausting all attempts raises :class:`FeedFetchError`
    so the caller can defer to the next scheduled run.
    """
    last_error: OSError | None = None
    for attempt in range(1, attempts + 1):
        try:
            return get(url)
        except OSError as exc:  # URLError/HTTPError/timeouts all subclass OSError.
            last_error = exc
            logger.warning("feed fetch attempt %d/%d failed: %s", attempt, attempts, exc)
            if attempt < attempts:
                sleep(backoff_base * 2 ** (attempt - 1))
    raise FeedFetchError(
        f"feed fetch failed after {attempts} attempts: {last_error}"
    ) from last_error


def extract_scripture_refs(text: str) -> list[str]:
    """Pull scripture references out of free text, de-duplicated and dash-normalized."""
    refs: list[str] = []
    for match in _SCRIPTURE_RE.finditer(text):
        ref = re.sub(r"\s*[" + scripture.DASH_CHARS + r"]\s*", "-", match.group(0))
        ref = re.sub(r"\s+", " ", ref).strip()
        if ref not in refs:
            refs.append(ref)
    return refs


def parse_pubdate(raw: str | None) -> tuple[str, int]:
    """Parse an RFC-822 ``pubDate`` into (YYYY-MM-DD, weekday) in its own timezone.

    A missing or unparseable date yields ``("", -1)`` so the item can never be
    mistaken for a Sunday; each adapter decides what that verdict means for selection.
    """
    if not raw:
        return "", -1
    try:
        when = parsedate_to_datetime(raw)
    except ValueError:
        # Malformed RFC-822 date — treat as undated, never Sunday.
        return "", -1
    return when.date().isoformat(), when.weekday()


def parse_pubdate_at(raw: str | None) -> _datetime | None:
    """Parse an RFC-822 ``pubDate`` into its full instant, or ``None`` if unusable.

    Kept separate from :func:`parse_pubdate` rather than widening its return tuple: the
    day and weekday drive classification at three call sites, while the instant is needed
    only by the automated-attempt cap (ADR-0031), which measures from the moment a sermon
    became available. A naive datetime is assumed UTC so the value is always comparable.
    """
    if not raw:
        return None
    try:
        when = parsedate_to_datetime(raw)
    except ValueError:
        return None
    return when if when.tzinfo is not None else when.replace(tzinfo=timezone.utc)


def entry_audio_url(entry: Any) -> str:
    """Return the first enclosure URL on a feed entry, or '' if none is present."""
    for enclosure in entry.get("enclosures", []):
        href = enclosure.get("href")
        if href:
            return str(href)
    return ""


def entry_has_identity(entry: Any) -> bool:
    """Whether a feed entry carries a usable id (a ``<guid>``, or its ``<link>`` fallback).

    feedparser derives ``entry.id`` from ``<guid>`` when present, falling back to
    ``<link>`` otherwise; an item with neither yields ``""``. ``guid`` is the
    ledger's primary key (CLAUDE.md §6, PRD §6.2), so an id-less item cannot be
    safely upserted — every id-less item in a feed would default to the same
    empty (or, for a namespaced adapter, prefix-only) key and silently collapse
    into a single record on upsert (issue #206). Adapters check this before
    building a :class:`~sermon_notes.sources.base.FeedItem` so the identity gap
    is caught on the raw entry, ahead of any namespace prefix that would
    otherwise mask it.
    """
    return bool(entry.get("id", ""))
