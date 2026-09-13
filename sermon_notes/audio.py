"""Download a sermon's audio enclosure — the Podbean CDN boundary.

This is the audio-download boundary (CLAUDE.md §6): the only place the pipeline
fetches enclosure media. It streams the file to a caller-provided path and
follows PRD §11.1 — three download attempts with exponential backoff, then it
raises so the caller can leave the sermon ``discovered`` to retry next run
(ADR-0009). No credentials are required (the enclosure URL is public); the URL
itself comes from the registry record set during feed ingest. The request carries
a browser-like User-Agent because the Podbean CDN edge rejects bot-shaped agents
from datacenter IPs with HTTP 403 (issue #29).
"""

from __future__ import annotations

import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Callable

from sermon_notes.logging import get_logger
from sermon_notes.net_guard import UnsafeHostError, build_guarded_opener, guard_public_host

logger = get_logger()

# Browser-like UA: the Podbean CDN edge returns HTTP 403 to bot-shaped agents from
# datacenter/CI IPs (issue #29). A real-browser UA passes the WAF reliably.
_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
)
_DOWNLOAD_ATTEMPTS = 3
_DOWNLOAD_BACKOFF_BASE = 1.0
_DOWNLOAD_TIMEOUT = 120
# Defense-in-depth on a privileged runner (issue #36): only fetch over http(s),
# and cap the streamed body so a hostile enclosure can't fill the runner disk.
# A sermon enclosure is tens of MB; 1 GiB leaves ample headroom over any real one.
# The host is also validated (issue #129) so a hostile enclosure URL can't reach an
# internal address, and that validation is re-applied to any redirect target the CDN
# sends back (issue #203); see sermon_notes.net_guard for the guard and its scope.
_ALLOWED_SCHEMES = frozenset({"http", "https"})
_MAX_DOWNLOAD_BYTES = 1024 * 1024 * 1024
_DOWNLOAD_CHUNK = 1024 * 1024

# Shared across calls so redirect handling is configured once (issue #203).
_opener = build_guarded_opener()


class AudioDownloadError(RuntimeError):
    """Raised when the enclosure cannot be downloaded after the configured attempts."""


def is_fetchable_enclosure(url: str) -> bool:
    """Whether :func:`http_download` would attempt ``url`` at all.

    The plan withholds a sermon this returns ``False`` for (ADR-0066), so it has to
    answer the same question the download guard below does, from the same constant —
    a filter that disagreed with the downloader would either strand sermons the
    downloader could have fetched, or fan out ones it is about to refuse.
    """
    return urllib.parse.urlsplit(url).scheme.lower() in _ALLOWED_SCHEMES


def http_download(url: str, dest: Path) -> None:
    """Stream ``url`` to ``dest`` (the mocked boundary).

    Every refusal leaves as :class:`AudioDownloadError`, including one the guarded opener
    raises part-way through the request when it refuses a redirect target (issue #229).
    An :class:`UnsafeHostError` escaping here would be a bare ``ValueError`` outside this
    boundary's contract, which :func:`~sermon_notes.transcribe.transcribe_sermon` does not
    catch — so a hostile or misconfigured CDN redirect would abort the whole batch instead
    of leaving one sermon ``discovered`` to retry (ADR-0009).
    """
    if not is_fetchable_enclosure(url):
        raise AudioDownloadError(f"refusing to fetch non-http(s) enclosure URL: {url!r}")
    headers = {"User-Agent": _USER_AGENT, "Accept": "*/*"}
    request = urllib.request.Request(url, headers=headers)
    try:
        guard_public_host(url)
        with _opener.open(request, timeout=_DOWNLOAD_TIMEOUT) as response:
            with open(dest, "wb") as out:
                read = 0
                while chunk := response.read(_DOWNLOAD_CHUNK):
                    read += len(chunk)
                    if read > _MAX_DOWNLOAD_BYTES:
                        raise AudioDownloadError(
                            f"enclosure body exceeds {_MAX_DOWNLOAD_BYTES}-byte cap: {url!r}"
                        )
                    out.write(chunk)
    except UnsafeHostError as exc:
        raise AudioDownloadError(str(exc)) from exc


def download_audio(
    url: str,
    dest: Path,
    *,
    download: Callable[[str, Path], None] = http_download,
    sleep: Callable[[float], None] = time.sleep,
    attempts: int = _DOWNLOAD_ATTEMPTS,
    backoff_base: float = _DOWNLOAD_BACKOFF_BASE,
) -> Path:
    """Download the enclosure at ``url`` to ``dest``, retrying with backoff.

    Per PRD §11.1: up to ``attempts`` tries, backing off ``backoff_base * 2**n``
    seconds between them. Exhausting all attempts raises :class:`AudioDownloadError`
    so the caller can leave the sermon ``discovered`` to retry next run (ADR-0009).
    Returns ``dest`` on success.
    """
    last_error: OSError | None = None
    for attempt in range(1, attempts + 1):
        try:
            download(url, dest)
            return dest
        except OSError as exc:  # URLError/HTTPError/timeouts all subclass OSError.
            last_error = exc
            logger.warning("audio download attempt %d/%d failed: %s", attempt, attempts, exc)
            if attempt < attempts:
                sleep(backoff_base * 2 ** (attempt - 1))
    raise AudioDownloadError(
        f"audio download failed after {attempts} attempts: {last_error}"
    ) from last_error
