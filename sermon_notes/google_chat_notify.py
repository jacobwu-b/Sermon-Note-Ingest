"""The Google Chat webhook boundary — the only module that posts to Google Chat.

This is the Google Chat note-delivery boundary (CLAUDE.md §6, ADR-0067): one of the channel
kinds a published sermon note can fan out to (spec 0024), alongside Discord
(:mod:`discord_notify`). It carries the same two facts every channel post carries — which
sermon published, and a link to it — as a plain text message. Google Chat's incoming-webhook
API takes a JSON payload only, with no file-upload mechanism, so unlike Discord's post this
one links the website's rendered note page (``note_url``, spec 0024 amendment, #472) rather
than attaching the file. Delivery is best-effort (PRD §11.2): a failure is logged and
swallowed so a dropped post never crashes the run, and the note is committed to the repo and
emailed regardless.

The webhook URL is a secret capability token exactly as Discord's is — read only by the
caller (:mod:`notify_channels`, which resolves it from the ``NOTIFY_CHANNELS_JSON`` routing
table) and never logged, never present in a raised error. This module reads no configuration
of its own; every URL it ever sees is handed to it directly.

The HTTP call lives in :func:`default_send`, which tests replace via ``send_fn`` so no URL is
read and no network is touched. Google Chat's incoming-webhook endpoint is the same REST
resource the Chat API's create-message call uses, so it already answers with a ``Message``
object carrying a ``name`` — Google Chat's equivalent of Discord's message ``id`` — which
:func:`default_send` reads back exactly as :mod:`discord_notify` does (spec 0024 amendment,
ADR-0067 amendment). Any future repaint capability for this boundary would target that id.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Callable

from sermon_notes.logging import get_logger

logger = get_logger()

_SEND_TIMEOUT = 30
_USER_AGENT = "sermon-note-pipeline (+https://github.com/jacobwu-b/Sermon-Note-Pipeline)"


class GoogleChatDeliveryError(RuntimeError):
    """Raised when Google Chat rejects a post or the request never reaches it."""


@dataclass(frozen=True)
class GoogleChatMessage:
    """One outbound webhook post: its message text."""

    text: str


@dataclass(frozen=True)
class DeliveryResult:
    """Whether a note reached Google Chat, and the id of the message that carries it.

    Mirrors :class:`discord_notify.DeliveryResult`: two separate facts, kept separate — a
    transport may deliver without reporting an id (every fake predating this amendment
    returns ``None``, as does a response whose body cannot be read).
    """

    delivered: bool
    message_id: str | None = None


def _message_id_from(body: bytes) -> str | None:
    """The ``name`` of the message Google Chat reports creating, or ``None`` if unreadable.

    Deliberately total, mirroring :func:`discord_notify._message_id_from`: every malformed
    shape yields ``None`` rather than raising. The id is a convenience for a later repaint,
    so failing a post the channel already received over an unparseable response would trade
    a real delivery for a cosmetic one.
    """
    try:
        payload = json.loads(body)
    except ValueError:
        return None
    name = payload.get("name") if isinstance(payload, dict) else None
    return name if isinstance(name, str) else None


def default_send(url: str, message: GoogleChatMessage) -> str | None:
    """POST ``message`` to the Google Chat webhook at ``url``, returning the id Google Chat
    assigned the created message (its ``name``), or ``None`` if the response carried none.

    ``url`` is handed in by the caller (:mod:`notify_channels`), resolved from the
    ``NOTIFY_CHANNELS_JSON`` routing table — this module never reads config itself. Same
    secret-never-leaks posture as :mod:`discord_notify`: a rejection raises with the status
    and response body only, a transport error with the reason only, both dropping the
    chained exception (``from None``) so ``url`` cannot re-leak through it.
    """
    body = json.dumps({"text": message.text}).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json; charset=UTF-8",
            "User-Agent": _USER_AGENT,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_SEND_TIMEOUT) as response:
            response_body: bytes = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace").strip()
        raise GoogleChatDeliveryError(f"Google Chat returned HTTP {exc.code}: {detail}") from None
    except urllib.error.URLError as exc:
        raise GoogleChatDeliveryError(f"Google Chat request failed: {exc.reason}") from None
    return _message_id_from(response_body)


def _build_note_message(sermon_title: str, sermon_date: str, note_url: str) -> GoogleChatMessage:
    """Assemble the published-note post text.

    ``note_url`` is the website's page for the note (spec 0024 amendment, #472) — the
    whole point of the post is to send a reader there, not to the original sermon audio,
    so unlike :mod:`discord_notify` (which attaches the file directly) this channel
    always carries the link. There is no attachment line — Google Chat's webhook API has
    no attachment to describe.
    """
    text = (
        f"A new sermon study note has been published.\n{sermon_title} — preached {sermon_date}"
        f"\nView the sermon note: {note_url}"
    )
    return GoogleChatMessage(text=text)


def send_note(
    *,
    sermon_title: str,
    sermon_date: str,
    note_url: str,
    send_fn: Callable[[GoogleChatMessage], str | None],
) -> DeliveryResult:
    """Post the published-note message to Google Chat; never raise.

    ``note_url`` is the website's page for the note (spec 0024 amendment, #472) — the
    caller (:mod:`pipeline`) builds it from the sermon's source and slug, so it is always
    present here. Delivery is best-effort (spec 0024): a transport failure is logged and
    swallowed — the note is committed to the repo and emailed regardless, so a dropped
    post loses only the channel copy.

    Returns a :class:`DeliveryResult` carrying both whether the transport accepted the post
    and the id of the message it created (spec 0024 amendment, ADR-0067 amendment). A
    transport that reports no id still delivered: the two are separate facts, and every
    ``send_fn`` written before the id existed returns ``None``.
    """
    try:
        message = _build_note_message(sermon_title, sermon_date, note_url)
        message_id = send_fn(message)
    except Exception as exc:  # noqa: BLE001 — note delivery is best-effort (PRD §11.2).
        logger.error("google chat delivery for %r failed to post: %s", sermon_title, exc)
        return DeliveryResult(delivered=False)
    logger.info("google chat delivery posted for %r", sermon_title)
    return DeliveryResult(delivered=True, message_id=message_id)
