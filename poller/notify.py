"""Email notification via the Resend REST API (stdlib HTTP client — no SDK dependency).

Notification is best-effort relative to the ledger write: the ledger is the
system of record, so a poll always writes discovered sermons to JSON first and
only then attempts to notify. A notify failure is logged and surfaced to the
caller, never silently swallowed, but it does not undo the ledger write —
undoing a real discovery to match a failed email would be the wrong failure
mode (CLAUDE.md §6, never swallow an error to pass a test).
"""

from __future__ import annotations

import datetime
import hashlib
import html
import json
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass

from poller.config import NotifyConfig
from poller.net import DEFAULT_USER_AGENT, is_fetchable_enclosure
from poller.quality import QualityScore
from poller.sources.base import SermonItem
from poller.sources.common import is_service_date

_RESEND_URL = "https://api.resend.com/emails"
_TIMEOUT = 15
_ATTEMPTS = 3
_BACKOFF_BASE = 1.0
# Resend rejects a subject at 2000+ chars; a real backlog of titles blows past
# that long before then, so fall back to a count-only subject well under it.
_MAX_SUBJECT_LEN = 200


class NotifyError(RuntimeError):
    """Raised when the Resend API rejects or cannot be reached for a send."""


def _off_day_items(items: list[SermonItem]) -> list[SermonItem]:
    """Items whose ``preached_on`` is a real, parseable date that is neither a Sunday
    nor a special service (``is_service_date``).

    Such a date is the anomaly worth surfacing here, distinct from a
    missing/unparseable ``preached_on`` (a different, pre-existing condition this
    alert doesn't concern itself with).
    """
    anomalies = []
    for item in items:
        if not item.preached_on:
            continue
        try:
            when = datetime.date.fromisoformat(item.preached_on)
        except ValueError:
            continue
        if not is_service_date(when):
            anomalies.append(item)
    return anomalies


def _format_email(church: str, items: list[SermonItem]) -> tuple[str, str]:
    """Return ``(subject, html_body)`` for one church's newly-discovered sermons."""
    noun = "sermon" if len(items) == 1 else "sermons"
    subject = f"New {noun} from {church}: " + "; ".join(item.title for item in items)
    if len(subject) > _MAX_SUBJECT_LEN:
        subject = f"New {len(items)} {noun} from {church}"

    warning = ""
    off_day = _off_day_items(items)
    if off_day:
        names = ", ".join(_escape(item.title) for item in off_day)
        warning = f"<p><strong>Heads up:</strong> preached_on is neither a Sunday nor a special service for: {names}.</p>"

    rows = []
    for item in items:
        details = [f"<strong>{_escape(item.title)}</strong>"]
        if item.speaker:
            details.append(_escape(item.speaker))
        if item.series:
            details.append(_escape(item.series))
        if item.preached_on:
            details.append(_escape(item.preached_on))
        line = " — ".join(details)
        if item.episode_url:
            url = _escape(item.episode_url)
            # Only http(s) gets a link: a feed-supplied javascript:/data: URL stays inert text.
            if is_fetchable_enclosure(item.episode_url):
                line += f'<br><a href="{url}">{url}</a>'
            else:
                line += f"<br>{url}"
        rows.append(f"<li>{line}</li>")

    body = (
        warning + f"<p>{_escape(church)} published {len(items)} new {noun}:</p><ul>" + "".join(rows) + "</ul>"
    )
    return subject, body


def _escape(text: str) -> str:
    return html.escape(text, quote=True)


def send_new_sermons(church: str, items: list[SermonItem], config: NotifyConfig) -> None:
    """Send one notification email for a church's newly-discovered sermons.

    Raises :class:`NotifyError` on any failure — the caller decides whether
    that should fail the run (it should never roll back the ledger write).
    """
    if not items:
        return
    subject, html_body = _format_email(church, items)
    _send(subject, html_body, config)


@dataclass(frozen=True)
class FlaggedTranscript:
    """A transcript that tripped a quality flag (ADR-0023): identifiers and counts, never its text."""

    guid: str
    title: str | None
    preached_on: str | None
    score: QualityScore
    flags: tuple[str, ...]


def _format_quality_alert(church: str, flagged: list[FlaggedTranscript]) -> tuple[str, str]:
    """Return ``(subject, html_body)`` for one church's flagged transcripts."""
    noun = "sermon" if len(flagged) == 1 else "sermons"
    subject = f"Transcript quality: {len(flagged)} {noun} from {church} flagged"
    rows = []
    for entry in flagged:
        score = entry.score
        details = [f"<strong>{_escape(entry.title or '(untitled)')}</strong>"]
        if entry.preached_on:
            details.append(_escape(entry.preached_on))
        details.append(_escape(entry.guid))
        rows.append(
            "<li>"
            + " — ".join(details)
            + f"<br>flags: {', '.join(entry.flags)}; {score.words} words, {score.loop_words} loop words, "
            f"{score.credit_hits} credit hits, {score.hotword_pairs} hotword pairs</li>"
        )
    body = (
        f"<p>{_escape(church)}: {len(flagged)} {noun} pushed to Content tripped a transcript quality flag. "
        "They were kept and marked done; review them in Content.</p><ul>" + "".join(rows) + "</ul>"
    )
    return subject, body


def send_quality_alert(church: str, flagged: list[FlaggedTranscript], config: NotifyConfig) -> None:
    """Send one email naming a church's flagged transcripts. Raises :class:`NotifyError` on failure."""
    subject, html_body = _format_quality_alert(church, flagged)
    _send(subject, html_body, config)


def send_retranscribe_alert(subject: str, lines: Sequence[str], config: NotifyConfig) -> None:
    """Send a re-transcription alert (ADR-0022): ``subject`` and one list item per line.

    Lines name churches, guids, paths and counts, never transcript text. Raises
    :class:`NotifyError` on failure.
    """
    body = "<ul>" + "".join(f"<li>{_escape(line)}</li>" for line in lines) + "</ul>"
    _send(subject, body, config)


def send_discovery_alert(subject: str, lines: Sequence[str], config: NotifyConfig) -> None:
    """Send a discovery absence alert (ADR-0025): ``subject`` and one list item per line.

    Raises :class:`NotifyError` on failure.
    """
    body = "<ul>" + "".join(f"<li>{_escape(line)}</li>" for line in lines) + "</ul>"
    _send(subject, body, config)


def _send(subject: str, html_body: str, config: NotifyConfig) -> None:
    """POST one email, retrying a 409, 429, 5xx or network failure with backoff (as ``net.fetch_feed``).

    The ``Idempotency-Key`` is derived from the request body, so a retry after a timeout that
    Resend had in fact accepted, or a later poll re-sending an email whose ``notified_at`` never
    reached the ledger, gets the original send back from Resend instead of a second email.
    Resend keeps keys for 24 hours, and a key always names one body, so its "same key,
    different payload" 409 can't occur.
    """
    payload = json.dumps(
        {
            "from": config.from_addr,
            "to": [config.to_addr],
            "subject": subject,
            "html": html_body,
        }
    ).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {config.api_key}",
        "Content-Type": "application/json",
        "Idempotency-Key": f"email/{hashlib.sha256(payload).hexdigest()}",
        # Resend's edge blocks the stdlib default UA as a bot signature
        # (Cloudflare error 1010) — same fix as poller/net.py.
        "User-Agent": DEFAULT_USER_AGENT,
    }
    last_error: NotifyError | None = None
    for attempt in range(1, _ATTEMPTS + 1):
        request = urllib.request.Request(_RESEND_URL, data=payload, method="POST", headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
                if response.status >= 300:
                    raise NotifyError(f"Resend returned status {response.status}")
            return
        except urllib.error.HTTPError as exc:
            error = NotifyError(f"Resend rejected the request: {exc.code} {exc.read()!r}")
            # A 409 can only mean an earlier attempt with this key is still in flight.
            if exc.code not in (409, 429) and exc.code < 500:
                raise error from exc
            last_error = error
        except OSError as exc:  # URLError, and a bare TimeoutError from a slow read
            last_error = NotifyError(f"could not reach Resend: {exc}")
        if attempt < _ATTEMPTS:
            time.sleep(_BACKOFF_BASE * 2 ** (attempt - 1))
    raise NotifyError(f"send failed after {_ATTEMPTS} attempts: {last_error}") from last_error
