"""The Discord webhook boundary — the only module that posts to Discord.

This is the Discord note-delivery boundary (CLAUDE.md §6, ADR-0024): the single place
the pipeline posts to a chat channel. It carries a published PBC sermon's rendered
note, the link back to the sermon itself, and the link to the note's own page on the
website (spec 0017, amended #574) — to the study group that reads it. It transmits whatever
artifact the orchestrator hands it, declaring the content type its filename implies;
that is the ``.pdf`` Discord can preview inline, falling back to the ``.docx`` when the
PDF rendering was lost (spec 0017 amendment, #173). Delivery is
best-effort, exactly as the email channel is (PRD §11.2): a failure is logged and
swallowed so a dropped post never crashes the run, and the note is committed to the
repo and emailed regardless.

The credential is a Discord *incoming webhook* URL. A webhook is bound to one channel at
creation, so "a specific channel" is enforced by the credential rather than by code. The URL
is itself a secret capability token — anyone holding it can post — so it is never logged and
never appears in a raised error, the same posture :mod:`sermon_notes.deploy_hook` keeps for
the Vercel hook. :func:`default_send` reads it from :mod:`config` by env var name;
:func:`send_to_url` takes an already-resolved URL instead, for :mod:`notify_channels`
(ADR-0067), which resolves URLs from the ``NOTIFY_CHANNELS_JSON`` routing table rather than
one named env var per church.

The HTTP call lives in :func:`default_send`, which tests replace via ``send_fn`` so no
URL is read and no network is touched. :func:`edit_note` (via :func:`default_edit`) is
the same webhook's edit-message call, used only by one-off maintenance scripts to
repaint an already-posted message after a rendering change — never by the regular
publish flow, which sends each note exactly once.
"""

from __future__ import annotations

import json
import secrets
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from sermon_notes import config
from sermon_notes.logging import get_logger

logger = get_logger()

_SEND_TIMEOUT = 30
# Cloudflare fronts discord.com and blocks requests carrying urllib's default
# "Python-urllib/x.y" User-Agent with a 403 (Cloudflare error 1010) before the request
# ever reaches Discord's application layer — reproduced directly against the real
# webhook endpoint. Any other User-Agent clears it, so `default_send` declares one.
_USER_AGENT = "sermon-note-pipeline (+https://github.com/jacobwu-b/Sermon-Note-Pipeline)"
_DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
# The two renderings a published note has (ADR-0025). Declaring the right type is what
# lets Discord preview the PDF inline instead of offering it as a download.
_MIME_TYPES = {".pdf": "application/pdf", ".docx": _DOCX_MIME}


def _mime_type(filename: str) -> str:
    """The content type to declare for an uploaded artifact, keyed by its suffix."""
    return _MIME_TYPES.get(Path(filename).suffix, "application/octet-stream")


def _with_wait(url: str) -> str:
    """``url`` with ``wait=true``, the parameter that makes Discord return the message.

    Discord's Execute Webhook answers ``204 No Content`` by default; with ``wait`` it
    answers ``200`` carrying the created message object, which is the only moment the
    message's id is ever knowable (ADR-0062) — a webhook credential cannot read channel
    history afterwards. Rebuilt through :mod:`urllib.parse` rather than string-appended
    so a URL that already carries a query (an operator-set ``thread_id``, say) keeps it.
    """
    parts = urllib.parse.urlsplit(url)
    query = [(k, v) for k, v in urllib.parse.parse_qsl(parts.query) if k != "wait"]
    query.append(("wait", "true"))
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(query)))


def _message_id_from(body: bytes) -> str | None:
    """The ``id`` of the message Discord reports creating, or ``None`` if unreadable.

    Deliberately total: every malformed shape yields ``None`` rather than raising. The
    id is a convenience for a later repaint, so failing a post the channel has already
    received — over a proxy's HTML error page, a ``204``'s empty body, or a response
    shape that changes — would trade a real delivery for a cosmetic one. Coerced to
    ``str`` because the ledger field is typed ``str | None``; Discord sends snowflakes
    as strings precisely because they exceed 2**53.
    """
    try:
        payload = json.loads(body)
    except ValueError:
        return None
    if not isinstance(payload, dict) or payload.get("id") is None:
        return None
    return str(payload["id"])


@dataclass(frozen=True)
class DeliveryResult:
    """Whether a note reached Discord, and the id of the message that carries it.

    Two separate facts, kept separate: a transport may deliver without reporting an id
    (every fake predating ADR-0062 returns ``None``, as does a response whose body
    cannot be read). Collapsing them into one truthy value would report each of those
    successful deliveries as a failure.
    """

    delivered: bool
    message_id: str | None = None


def _escape_header_filename(filename: str) -> str:
    """Escape ``filename`` for safe interpolation into a quoted header value.

    A bare ``"`` or ``\\`` would terminate or alter the quoted-string early, and a bare
    CR/LF would start a new header line — each lets an attacker-controlled filename
    inject extra MIME headers or parts into the request body (#275, RFC 6266 §4.1). CR
    and LF are stripped outright since a filename has no legitimate use for a line
    break; ``"`` and ``\\`` are backslash-escaped, the same treatment ``email.utils``
    applies.
    """
    stripped = filename.replace("\r", "").replace("\n", "")
    return stripped.replace("\\", "\\\\").replace('"', '\\"')


class DiscordDeliveryError(RuntimeError):
    """Raised when Discord rejects a post or the request never reaches it."""


@dataclass(frozen=True)
class DiscordAttachment:
    """One uploaded file: a filename and its raw bytes."""

    filename: str
    content: bytes


@dataclass(frozen=True)
class DiscordMessage:
    """One outbound webhook post: its message text and uploaded files."""

    content: str
    attachments: tuple[DiscordAttachment, ...] = field(default_factory=tuple)


def is_configured(webhook_env_var: str = "DISCORD_WEBHOOK_URL") -> bool:
    """Whether the webhook named by ``webhook_env_var`` is provisioned.

    Unset leaves the boundary dark — the pipeline publishes and emails as before and
    posts nothing — mirroring how the web-publishing boundaries (ADR-0018) stay dark
    until their secrets exist. Menlo's channel (spec 0017 amendment) is a second,
    independent webhook read the same way: either can be absent while the other
    still delivers.
    """
    return bool(config.get(webhook_env_var, None))


def _encode_multipart(
    message: DiscordMessage, boundary: str, *, clear_attachments: bool = False
) -> bytes:
    """Encode ``message`` as a ``multipart/form-data`` body for Discord's webhook API.

    Discord takes the message itself as a ``payload_json`` part and each upload as a
    ``files[n]`` part. Assembled here by hand over the stdlib rather than pulled in
    from an HTTP library, so the boundary adds no dependency (ADR-0024).

    ``clear_attachments`` sets ``"attachments": []`` on the payload — Discord's
    edit-message contract otherwise keeps an existing attachment beside the new
    ``files[n]`` part rather than replacing it, so an edit that is meant to swap the
    file declares the old one gone.
    """
    payload: dict[str, object] = {"content": message.content}
    if clear_attachments:
        payload["attachments"] = []
    marker = f"--{boundary}".encode("ascii")
    parts: list[bytes] = [
        marker,
        b'Content-Disposition: form-data; name="payload_json"',
        b"Content-Type: application/json",
        b"",
        json.dumps(payload).encode("utf-8"),
    ]
    for index, attachment in enumerate(message.attachments):
        disposition = (
            f'Content-Disposition: form-data; name="files[{index}]"; '
            f'filename="{_escape_header_filename(attachment.filename)}"'
        )
        parts.extend(
            [
                marker,
                disposition.encode("utf-8"),
                f"Content-Type: {_mime_type(attachment.filename)}".encode("ascii"),
                b"",
                attachment.content,
            ]
        )
    parts.append(f"--{boundary}--".encode("ascii"))
    return b"\r\n".join(parts) + b"\r\n"


def _post_multipart(url: str, body: bytes, boundary: str, *, method: str) -> bytes:
    """Send an already-encoded multipart body to ``url``, returning Discord's response
    body and raising on rejection.

    Shared by :func:`default_send` and :func:`default_edit` — same headers, same
    error handling, only the verb and target URL differ. The response is returned
    rather than discarded so ``default_send`` can read back the id of the message it
    just created (ADR-0062); a caller with no use for it ignores it.
    """
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": _USER_AGENT,
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(request, timeout=_SEND_TIMEOUT) as response:
            response_body: bytes = response.read()
            return response_body
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace").strip()
        raise DiscordDeliveryError(f"Discord returned HTTP {exc.code}: {detail}") from None
    except urllib.error.URLError as exc:
        raise DiscordDeliveryError(f"Discord request failed: {exc.reason}") from None


def send_to_url(url: str, message: DiscordMessage) -> str | None:
    """POST ``message`` to the Discord webhook at ``url`` directly, returning the id
    Discord assigned the created message.

    The URL-taking half of :func:`default_send`, split out so a caller that already
    holds a resolved webhook URL — :mod:`notify_channels`, routing a church's
    configured channels (ADR-0067) — can post without a named env var to read it
    through. Same secret-never-leaks posture: a rejection raises with the status and
    response body only, a transport error with the reason only, both dropping the
    original exception (``from None``) so ``url`` cannot re-leak through the chained
    :class:`~urllib.error.HTTPError`'s ``url`` attribute.

    The request carries ``wait=true`` so Discord answers with the message rather than
    ``204 No Content`` — see :func:`_with_wait`. That makes the post synchronous, which
    surfaces rejections (429s especially) a fire-and-forget call never observed; they
    raise here and are swallowed by :func:`send_note` like any other failure. ``None``
    comes back when the response carried no readable id, which is not a delivery
    failure (ADR-0062).
    """
    prepared_url = _with_wait(url)
    boundary = secrets.token_hex(16)
    body = _post_multipart(
        prepared_url, _encode_multipart(message, boundary), boundary, method="POST"
    )
    return _message_id_from(body)


def default_send(
    message: DiscordMessage, *, webhook_env_var: str = "DISCORD_WEBHOOK_URL"
) -> str | None:
    """POST ``message`` to the Discord webhook named by ``webhook_env_var``, returning
    the id Discord assigned the created message.

    Reads that variable from config — ``DISCORD_WEBHOOK_URL`` (PBC) by default, or
    ``DISCORD_WEBHOOK_URL_MENLO`` for Menlo's independent channel (spec 0017
    amendment) — then delegates to :func:`send_to_url`.
    """
    return send_to_url(config.get(webhook_env_var), message)


def edit_to_url(url: str, message_id: str, message: DiscordMessage) -> None:
    """PATCH the already-posted message ``message_id`` on the webhook at ``url`` to
    carry ``message`` instead.

    The URL-taking half of :func:`default_edit`, split out the same way
    :func:`send_to_url` is split from :func:`default_send`: a caller that already
    holds a resolved webhook URL — :mod:`notify_channels`, routing a church's
    ``NOTIFY_CHANNELS_JSON`` channels (ADR-0067) — can edit without a named env var
    to read it through. Targets Discord's edit-webhook-message endpoint
    (``{url}/messages/{message_id}``); only a webhook's own messages can be edited
    this way, which is exactly the ones this pipeline ever posts. Same secret-never-leaks
    posture as ``send_to_url``.
    """
    target = f"{url}/messages/{message_id}"
    boundary = secrets.token_hex(16)
    body = _encode_multipart(message, boundary, clear_attachments=True)
    _post_multipart(target, body, boundary, method="PATCH")


def default_edit(
    message_id: str, message: DiscordMessage, *, webhook_env_var: str = "DISCORD_WEBHOOK_URL"
) -> None:
    """PATCH an already-posted webhook message to carry ``message`` instead.

    Reads that variable from config — ``DISCORD_WEBHOOK_URL`` (PBC) by default, or
    ``DISCORD_WEBHOOK_URL_MENLO`` for Menlo's independent channel (spec 0017
    amendment) — then delegates to :func:`edit_to_url`.
    """
    edit_to_url(config.get(webhook_env_var), message_id, message)


def _build_note_message(
    sermon_title: str,
    sermon_date: str,
    artifact_path: Path,
    note_url: str,
    episode_url: str | None = None,
) -> DiscordMessage:
    """Assemble the published-note post with the rendered note uploaded alongside it.

    ``episode_url`` links back to the sermon the note was made from, so a reader can
    check a quote or hear the delivery without hunting for it. A source whose feed
    publishes no episode link (North Point's does not) passes ``None`` and the line is
    left out entirely — no link beats a link to nowhere.

    ``note_url`` links the website's own page for this note (spec 0017 amendment,
    #574) — the same URL Google Chat's message carries (spec 0024 amendment, #472).
    Unlike ``episode_url`` it is derived from the sermon, not read off a feed, so it is
    always present and the line is never omitted.

    Both URLs are wrapped in angle brackets, Discord's own syntax for suppressing a
    link's embed/preview card — a source church's page metadata (title, description, a
    vendor byline) or a second embed card stacking under the attached PDF preview has
    no business appearing under a note-delivery message.
    """
    suffix = artifact_path.suffix
    format_name = "PDF" if suffix == ".pdf" else suffix
    episode_line = f"Listen to the original sermon: <{episode_url}>\n" if episode_url else ""
    note_line = f"Read the sermon note online: <{note_url}>\n"
    content = (
        "**A new sermon study note has been published.**\n"
        f"{sermon_title} — preached {sermon_date}\n"
        f"{episode_line}"
        f"{note_line}"
        f"The {format_name} study note is attached."
    )
    return DiscordMessage(
        content=content,
        attachments=(
            DiscordAttachment(filename=artifact_path.name, content=artifact_path.read_bytes()),
        ),
    )


def send_note(
    *,
    sermon_title: str,
    sermon_date: str,
    artifact_path: Path,
    note_url: str,
    episode_url: str | None = None,
    send_fn: Callable[[DiscordMessage], str | None] = default_send,
) -> DeliveryResult:
    """Post the published note at ``artifact_path`` to the Discord channel; never raise.

    Reads the rendered artifact, uploads it under its own filename and declared type,
    and transmits through ``send_fn``. ``note_url`` is the website's page for this note
    (spec 0017 amendment, #574) — the caller (:mod:`pipeline`) builds it from the
    sermon's source and slug, so it is always present here. ``episode_url`` is the link
    back to the original sermon; omit it (or pass ``""``) when the source's feed
    publishes none. Delivery is best-effort (spec 0017): an unreadable artifact or a
    transport failure is logged and swallowed — the note is committed to the repo and
    emailed regardless, so a dropped post loses only the channel copy.

    Returns a :class:`DeliveryResult` carrying both whether the transport accepted the
    post and the id of the message it created (ADR-0062). A transport that reports no
    id still delivered: the two are separate facts, and every ``send_fn`` written before
    the id existed returns ``None``.
    """
    try:
        message = _build_note_message(
            sermon_title, sermon_date, artifact_path, note_url, episode_url
        )
        message_id = send_fn(message)
    except Exception as exc:  # noqa: BLE001 — note delivery is best-effort (PRD §11.2).
        logger.error("discord delivery for %r failed to post: %s", sermon_title, exc)
        return DeliveryResult(delivered=False)
    logger.info("discord delivery posted for %r", sermon_title)
    return DeliveryResult(delivered=True, message_id=message_id)


def edit_note(
    *,
    message_id: str,
    sermon_title: str,
    sermon_date: str,
    artifact_path: Path,
    note_url: str,
    episode_url: str | None = None,
    edit_fn: Callable[[str, DiscordMessage], None] = default_edit,
) -> bool:
    """Rebuild an already-posted note message from the current artifact and edit it in
    place; never raise.

    Not part of the pipeline's regular flow (delivery is send-once, spec 0017) — this is
    for a one-time repaint of already-sent messages after a rendering change (e.g. #391),
    driven by a maintenance script that supplies the target ``message_id``. It builds
    exactly the message :func:`send_note` would build for the same arguments today,
    ``note_url`` included (spec 0017 amendment, #574), so an edited post always matches
    what a fresh publish would send. Same best-effort, logged-and-swallowed posture as
    ``send_note``.
    """
    try:
        message = _build_note_message(
            sermon_title, sermon_date, artifact_path, note_url, episode_url
        )
        edit_fn(message_id, message)
    except Exception as exc:  # noqa: BLE001 — note delivery is best-effort (PRD §11.2).
        logger.error("discord edit for %r failed to post: %s", sermon_title, exc)
        return False
    logger.info("discord edit posted for %r", sermon_title)
    return True
