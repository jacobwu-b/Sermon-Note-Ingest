"""Shared feed-fetch transport: retries, scheme guard, and a body size cap.

Every adapter fetches its feed (and, for PBC, an auxiliary HTML page and a HEAD
request) through this module so the hardening lives in one place. The feed URLs
themselves are fixed, operator-configured values from ``CHURCHES``, not
attacker-controlled input, so the guard here is a simple defense-in-depth cap
rather than a full SSRF-hardened opener.

This is also the audio-enclosure download boundary (:func:`http_download`,
:func:`download_audio`): the only place the transcriber fetches a sermon's
audio, mirroring the feed-fetch boundary above.
"""

from __future__ import annotations

import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from pathlib import Path

DEFAULT_USER_AGENT = "sermon-rss-monitor/1.0 (+https://github.com/jacobwu-b/Sermon-RSS-Feed-Monitor)"
_ALLOWED_SCHEMES = frozenset({"http", "https"})
_MAX_BODY_BYTES = 32 * 1024 * 1024  # a podcast RSS feed is a few MB at most
_TIMEOUT = 30
_ATTEMPTS = 3
_BACKOFF_BASE = 1.0

# An audio enclosure is tens of MB; 1 GiB leaves ample headroom over any real one
# while still capping a hostile or misconfigured enclosure (issue-#36-style defense
# in depth on a shared runner).
_MAX_DOWNLOAD_BYTES = 1024 * 1024 * 1024
_DOWNLOAD_CHUNK = 1024 * 1024
_DOWNLOAD_TIMEOUT = 120
_DOWNLOAD_ATTEMPTS = 3
_DOWNLOAD_BACKOFF_BASE = 1.0


class AudioDownloadError(RuntimeError):
    """Raised when an enclosure cannot be downloaded, or is unsafe to fetch."""


class FeedFetchError(RuntimeError):
    """Raised when a URL cannot be fetched after retries, or is unsafe to fetch."""


def http_get(url: str, *, user_agent: str = DEFAULT_USER_AGENT) -> bytes:
    """Fetch ``url`` over HTTP(S) and return the raw body (the mocked boundary)."""
    scheme = urllib.parse.urlsplit(url).scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise FeedFetchError(f"refusing to fetch non-http(s) url: {url!r}")
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            body = response.read(_MAX_BODY_BYTES + 1)
    except (urllib.error.URLError, OSError) as exc:
        raise FeedFetchError(f"fetch failed for {url!r}: {exc}") from exc
    if len(body) > _MAX_BODY_BYTES:
        raise FeedFetchError(f"body exceeds {_MAX_BODY_BYTES}-byte cap: {url!r}")
    return body


def http_head_last_modified(url: str, *, user_agent: str = DEFAULT_USER_AGENT) -> datetime | None:
    """HEAD ``url`` and return its ``Last-Modified`` instant, or ``None`` if unusable.

    Never raises: a missing header, non-2xx response, or network error all yield
    ``None`` since this is enrichment (PBC's real publish instant), never
    load-bearing for discovery.
    """
    scheme = urllib.parse.urlsplit(url).scheme.lower()
    if scheme not in _ALLOWED_SCHEMES or not url:
        return None
    request = urllib.request.Request(url, headers={"User-Agent": user_agent}, method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            raw = response.headers.get("Last-Modified")
    except (urllib.error.URLError, OSError):
        return None
    if not raw:
        return None
    try:
        when = parsedate_to_datetime(raw)
    except (ValueError, TypeError):
        return None
    return when if when.tzinfo is not None else when.replace(tzinfo=UTC)


def fetch_feed(
    url: str,
    *,
    get: Callable[[str], bytes] = http_get,
    sleep: Callable[[float], None] = time.sleep,
    attempts: int = _ATTEMPTS,
    backoff_base: float = _BACKOFF_BASE,
) -> bytes:
    """Fetch a URL, retrying transient failures with exponential backoff.

    Exhausting all attempts raises :class:`FeedFetchError` so the caller can
    defer this source to the next scheduled run rather than crash the poll.
    """
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            return get(url)
        except FeedFetchError as exc:
            last_error = exc
            if attempt < attempts:
                sleep(backoff_base * 2 ** (attempt - 1))
    raise FeedFetchError(f"fetch failed after {attempts} attempts: {last_error}") from last_error


def now() -> str:
    """Current UTC instant as an ISO-8601 string — the "first retrieved" timestamp."""
    return datetime.now(UTC).isoformat()


def is_fetchable_enclosure(url: str) -> bool:
    """Whether :func:`http_download` would attempt ``url`` at all.

    The transcriber withholds a sermon this returns ``False`` for rather than attempting
    (and failing) a download, so it has to answer the same question the download guard
    below does, from the same constant.
    """
    return bool(url) and urllib.parse.urlsplit(url).scheme.lower() in _ALLOWED_SCHEMES


def http_download(url: str, dest: Path, *, user_agent: str = DEFAULT_USER_AGENT) -> None:
    """Stream ``url`` to ``dest`` over HTTP(S) (the mocked boundary), capping the body size."""
    if not is_fetchable_enclosure(url):
        raise AudioDownloadError(f"refusing to fetch non-http(s) enclosure url: {url!r}")
    request = urllib.request.Request(url, headers={"User-Agent": user_agent})
    try:
        with urllib.request.urlopen(request, timeout=_DOWNLOAD_TIMEOUT) as response, open(dest, "wb") as out:
            read = 0
            while chunk := response.read(_DOWNLOAD_CHUNK):
                read += len(chunk)
                if read > _MAX_DOWNLOAD_BYTES:
                    raise AudioDownloadError(
                        f"enclosure body exceeds {_MAX_DOWNLOAD_BYTES}-byte cap: {url!r}"
                    )
                out.write(chunk)
    except (urllib.error.URLError, OSError) as exc:
        raise AudioDownloadError(f"audio download failed for {url!r}: {exc}") from exc


def download_audio(
    url: str,
    dest: Path,
    *,
    download: Callable[[str, Path], None] = http_download,
    sleep: Callable[[float], None] = time.sleep,
    attempts: int = _DOWNLOAD_ATTEMPTS,
    backoff_base: float = _DOWNLOAD_BACKOFF_BASE,
) -> None:
    """Download the enclosure at ``url`` to ``dest``, retrying transient failures with backoff.

    Exhausting all attempts raises :class:`AudioDownloadError` so the caller can leave the
    sermon pending to retry on the next scheduled run rather than mark it terminally failed.
    """
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            download(url, dest)
            return
        except AudioDownloadError as exc:
            last_error = exc
            if attempt < attempts:
                sleep(backoff_base * 2 ** (attempt - 1))
    raise AudioDownloadError(f"audio download failed after {attempts} attempts: {last_error}") from last_error
