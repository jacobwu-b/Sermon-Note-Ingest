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
import json
import urllib.error
import urllib.request

from poller.config import NotifyConfig
from poller.net import DEFAULT_USER_AGENT
from poller.sources.base import SermonItem

_SUNDAY = 6

_RESEND_URL = "https://api.resend.com/emails"
_TIMEOUT = 15
# Resend rejects a subject at 2000+ chars; a real backlog of titles blows past
# that long before then, so fall back to a count-only subject well under it.
_MAX_SUBJECT_LEN = 200


class NotifyError(RuntimeError):
    """Raised when the Resend API rejects or cannot be reached for a send."""


def _non_sunday_items(items: list[SermonItem]) -> list[SermonItem]:
    """Items whose ``published_on`` is a real, parseable date that isn't a Sunday.

    The owner's stated rule (#51) is that every church's sermon is preached on a
    Sunday — a non-Sunday date is the anomaly worth surfacing here, distinct from a
    missing/unparseable ``published_on`` (a different, pre-existing condition this
    alert doesn't concern itself with).
    """
    anomalies = []
    for item in items:
        if not item.published_on:
            continue
        try:
            when = datetime.date.fromisoformat(item.published_on)
        except ValueError:
            continue
        if when.weekday() != _SUNDAY:
            anomalies.append(item)
    return anomalies


def _format_email(church: str, items: list[SermonItem]) -> tuple[str, str]:
    """Return ``(subject, html_body)`` for one church's newly-discovered sermons."""
    noun = "sermon" if len(items) == 1 else "sermons"
    subject = f"New {noun} from {church}: " + "; ".join(item.title for item in items)
    if len(subject) > _MAX_SUBJECT_LEN:
        subject = f"New {len(items)} {noun} from {church}"

    warning = ""
    non_sunday = _non_sunday_items(items)
    if non_sunday:
        names = ", ".join(_escape(item.title) for item in non_sunday)
        warning = f"<p><strong>Heads up:</strong> published_on is not a Sunday for: {names}.</p>"

    rows = []
    for item in items:
        details = [f"<strong>{_escape(item.title)}</strong>"]
        if item.speaker:
            details.append(_escape(item.speaker))
        if item.series:
            details.append(_escape(item.series))
        if item.published_on:
            details.append(item.published_on)
        line = " — ".join(details)
        if item.episode_url:
            line += f'<br><a href="{_escape(item.episode_url)}">{_escape(item.episode_url)}</a>'
        rows.append(f"<li>{line}</li>")

    body = warning + f"<p>{church} published {len(items)} new {noun}:</p><ul>" + "".join(rows) + "</ul>"
    return subject, body


def _escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def send_new_sermons(church: str, items: list[SermonItem], config: NotifyConfig) -> None:
    """Send one notification email for a church's newly-discovered sermons.

    Raises :class:`NotifyError` on any failure — the caller decides whether
    that should fail the run (it should never roll back the ledger write).
    """
    if not items:
        return
    subject, html_body = _format_email(church, items)
    payload = json.dumps(
        {
            "from": config.from_addr,
            "to": [config.to_addr],
            "subject": subject,
            "html": html_body,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        _RESEND_URL,
        data=payload,
        method="POST",
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
            # Resend's edge blocks the stdlib default UA as a bot signature
            # (Cloudflare error 1010) — same fix as poller/net.py.
            "User-Agent": DEFAULT_USER_AGENT,
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=_TIMEOUT) as response:
            if response.status >= 300:
                raise NotifyError(f"Resend returned status {response.status}")
    except urllib.error.HTTPError as exc:
        raise NotifyError(f"Resend rejected the request: {exc.code} {exc.read()!r}") from exc
    except urllib.error.URLError as exc:
        raise NotifyError(f"could not reach Resend: {exc}") from exc
