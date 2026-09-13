"""The transactional-email boundary — the only module that sends email.

This is the transactional-email boundary (CLAUDE.md §6): the single place the
pipeline emails Jacob. It serves two purposes, both best-effort (PRD §11.2): a
terminal-failure escalation naming the sermon, its failure class and message, and a
link to the GitHub Actions run; and a published-note delivery carrying the rendered
``.docx`` as an attachment. Each assembles one email and hands it to an injectable
transport. The real transport posts to the Resend HTTP API, reading
``RESEND_API_KEY`` only through :mod:`config`; the key is never logged. A transport
failure is logged and swallowed so a dropped email never crashes the run.

The HTTP call lives in :func:`default_send`, which tests replace via ``send_fn``
so no key is read and no network is touched.

Every actionable failure/alert send (escalation, staleness, red-run-streak,
stalled-queue — not the two self-resolving withholding notices) also pairs with a
call to :mod:`github_issues` (ADR-0080, #533): a sibling, independent, best-effort
GitHub issue create-or-comment that reuses the email's own body text. Neither call
gates the other.
"""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from sermon_notes import config, github_issues
from sermon_notes.logging import get_logger

logger = get_logger()

_RESEND_ENDPOINT = "https://api.resend.com/emails"
_SEND_TIMEOUT = 30
# Cloudflare fronts the Resend API and rejects urllib's default ``Python-urllib``
# signature with a 403 (error 1010) before the request reaches Resend, so the
# transport must identify itself with a named agent.
_USER_AGENT = "sermon-note-pipeline/1.0"


@dataclass(frozen=True)
class Attachment:
    """One email attachment: a filename and its base64-encoded content."""

    filename: str
    content_b64: str


@dataclass(frozen=True)
class EmailMessage:
    """One outbound email: who it is from and to, its subject, body, and attachments."""

    from_addr: str
    to_addr: str
    subject: str
    body: str
    attachments: tuple[Attachment, ...] = field(default_factory=tuple)


def default_send(message: EmailMessage) -> None:
    """Post ``message`` to the Resend API (the mocked boundary).

    Reads ``RESEND_API_KEY`` from config and authorizes with it; the key is used
    only in the request header and is never logged. Any attachments ride along as
    Resend's base64 ``content`` entries. Raises on any HTTP error so the caller can
    record the drop; the raised error includes Resend's response body, which names
    the actual reason (e.g. an unverified sender domain behind a ``403``) so the
    failure is diagnosable from the log alone. The body never contains the key.
    """
    body: dict[str, object] = {
        "from": message.from_addr,
        "to": [message.to_addr],
        "subject": message.subject,
        "text": message.body,
    }
    if message.attachments:
        body["attachments"] = [
            {"filename": attachment.filename, "content": attachment.content_b64}
            for attachment in message.attachments
        ]
    payload = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(
        _RESEND_ENDPOINT,
        data=payload,
        headers={
            "Authorization": f"Bearer {config.get('RESEND_API_KEY')}",
            "Content-Type": "application/json",
            "User-Agent": _USER_AGENT,
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=_SEND_TIMEOUT) as response:
            response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace").strip()
        raise RuntimeError(f"Resend API returned HTTP {exc.code}: {detail}") from exc


def _escalation_body(
    sermon_title: str, sermon_date: str, failure_class: str, failure_message: str, run_url: str
) -> str:
    """The escalation's body text — shared by the email and the paired issue (ADR-0080)."""
    return (
        "A sermon failed terminally and needs manual attention.\n\n"
        f"Sermon: {sermon_title} ({sermon_date})\n"
        f"Failure class: {failure_class}\n"
        f"Message: {failure_message}\n\n"
        f"GitHub Actions run: {run_url}\n"
    )


def _build_message(
    sermon_title: str,
    sermon_date: str,
    failure_class: str,
    failure_message: str,
    run_url: str,
) -> EmailMessage:
    """Assemble the escalation email body and subject from the failure details."""
    return EmailMessage(
        from_addr=config.get("NOTIFY_EMAIL_FROM"),
        to_addr=config.get("NOTIFY_EMAIL_TO"),
        subject=f"[Sermon Notes] Terminal failure: {sermon_title}",
        body=_escalation_body(sermon_title, sermon_date, failure_class, failure_message, run_url),
    )


def send_escalation(
    *,
    sermon_title: str,
    sermon_date: str,
    failure_class: str,
    failure_message: str,
    run_url: str,
    send_fn: Callable[[EmailMessage], None] = default_send,
    create_issue: Callable[..., bool] = github_issues.create_or_group_issue,
) -> bool:
    """Send one terminal-failure escalation email; never raise.

    Builds the message from config-supplied addresses (PRD §11.2) and the failure
    details, then transmits it through ``send_fn``. Returns ``True`` when the
    transport accepts it. A transport failure is logged and swallowed — a dropped
    escalation must not crash the run — and reported as ``False``.

    Independently of the email's own outcome, also calls ``create_issue`` to create
    or group a GitHub issue for this failure class (ADR-0080, #533) — a best-effort
    side channel that never affects this function's return value.
    """
    try:
        message = _build_message(sermon_title, sermon_date, failure_class, failure_message, run_url)
        send_fn(message)
    except Exception as exc:  # noqa: BLE001 — escalation is best-effort (PRD §11.2).
        logger.error("escalation email for %r failed to send: %s", sermon_title, exc)
        delivered = False
    else:
        logger.info("escalation email sent for %r", sermon_title)
        delivered = True
    create_issue(
        failure_class=failure_class,
        body=_escalation_body(sermon_title, sermon_date, failure_class, failure_message, run_url),
    )
    return delivered


# The failure-class strings the two dead-man's-switch alerts pass to `github_issues`
# as their dedup key (ADR-0080) — these alerts carry no per-occurrence `failure_class`
# of their own, unlike a sermon escalation, so each gets one fixed name.
_STALENESS_FAILURE_CLASS = "PublishStaleness"
_RED_STREAK_FAILURE_CLASS = "RedRunStreak"


def _staleness_body(
    last_published_at: str, days_stale: int, threshold_days: int, run_url: str
) -> str:
    """The staleness alert's body text — shared by the email and the paired issue."""
    return (
        f"The pipeline has published no sermon note in {days_stale} days.\n\n"
        f"Last note published: {last_published_at}\n"
        f"Threshold: PUBLISH_STALENESS_DAYS = {threshold_days} days\n\n"
        "Nothing reported a failure. This alert fires on the absence of success, so "
        "whatever stopped the pipeline is a failure mode it does not recognise — "
        "start from the recent run logs, not from the ledger's failed records.\n\n"
        "A legitimate multi-week gap in the feeds (holidays, a series break) looks "
        "identical from here. Confirm against the sources before treating it as an "
        "outage.\n\n"
        f"GitHub Actions run: {run_url}\n"
    )


def _build_staleness_message(
    last_published_at: str, days_stale: int, threshold_days: int, run_url: str
) -> EmailMessage:
    """Assemble the dead-man's-switch email — an absence of success, not a failure."""
    return EmailMessage(
        from_addr=config.get("NOTIFY_EMAIL_FROM"),
        to_addr=config.get("NOTIFY_EMAIL_TO"),
        subject=f"[Sermon Notes] No note published in {days_stale} days",
        body=_staleness_body(last_published_at, days_stale, threshold_days, run_url),
    )


def send_staleness_alert(
    *,
    last_published_at: str,
    days_stale: int,
    threshold_days: int,
    run_url: str,
    send_fn: Callable[[EmailMessage], None] = default_send,
    create_issue: Callable[..., bool] = github_issues.create_or_group_issue,
) -> bool:
    """Send the publish-staleness alert (#242); never raise, but report the drop.

    The dead-man's switch's delivery arm. Assembled and transmitted exactly like
    :func:`send_escalation`, and it swallows a transport failure for the same reason:
    reconciliation's remaining work must still finish. The difference is what the
    caller does with ``False`` — for an alert of last resort a dropped send is itself
    the incident, so :func:`pipeline.reconcile` fails the run on it rather than
    logging and moving on.

    It is a separate function from :func:`send_escalation` because the two say
    different things. An escalation names a sermon that failed terminally; this names
    no sermon, and sending it through the escalation body would tell the reader a
    sermon failed when none did. The paired GitHub issue (independent of this
    function's own return value, ADR-0080) uses the fixed class
    ``_STALENESS_FAILURE_CLASS`` rather than a per-occurrence one, for the same reason.
    """
    try:
        message = _build_staleness_message(last_published_at, days_stale, threshold_days, run_url)
        send_fn(message)
    except Exception as exc:  # noqa: BLE001 — the caller fails the run on the False.
        logger.error("publish-staleness alert failed to send: %s", exc)
        delivered = False
    else:
        logger.info("publish-staleness alert sent (%d days since the last note)", days_stale)
        delivered = True
    create_issue(
        failure_class=_STALENESS_FAILURE_CLASS,
        body=_staleness_body(last_published_at, days_stale, threshold_days, run_url),
    )
    return delivered


def _build_awaiting_enclosure_message(
    waiting: list[tuple[str, str, int]], threshold_days: int, run_url: str
) -> EmailMessage:
    """Assemble the awaiting-enclosure email — a sermon nobody can transcribe yet.

    ``waiting`` is ``(guid, title, days_waited)`` per sermon, longest wait first. Unlike
    an escalation this reports no failure: the pipeline is withholding the sermon on
    purpose, and the church has not published its audio.
    """
    count = len(waiting)
    longest = waiting[0][2]
    noun = "sermon" if count == 1 else "sermons"
    possessive = "its" if count == 1 else "their"
    listing = "\n".join(
        f"  - {title} ({guid}) — waiting {days} days" for guid, title, days in waiting
    )
    body = (
        f"{count} {noun} discovered in the feed still {possessive} audio enclosure.\n\n"
        f"{listing}\n\n"
        f"Threshold: MISSING_ENCLOSURE_DAYS = {threshold_days} days\n\n"
        "Nothing has failed. A feed item can publish before its audio finishes "
        "processing, so the pipeline withholds it from the fan-out rather than "
        "spending an attempt on a download it cannot perform (ADR-0066). It resumes on "
        "its own the moment the enclosure appears — no recovery step is needed.\n\n"
        "Past this threshold the audio may simply never arrive. Check the episode page: "
        "if the sermon was published without audio, the record stays discovered and "
        "waiting indefinitely, and retiring it is a manual decision.\n\n"
        f"GitHub Actions run: {run_url}\n"
    )
    return EmailMessage(
        from_addr=config.get("NOTIFY_EMAIL_FROM"),
        to_addr=config.get("NOTIFY_EMAIL_TO"),
        subject=(
            f"[Sermon Notes] {count} {noun} still awaiting {possessive} audio ({longest} days)"
        ),
        body=body,
    )


def send_awaiting_enclosure_alert(
    *,
    waiting: list[tuple[str, str, int]],
    threshold_days: int,
    run_url: str,
    send_fn: Callable[[EmailMessage], None] = default_send,
) -> bool:
    """Send the awaiting-enclosure alert (ADR-0066); never raise, but report the drop.

    Returns whether it was delivered. A dropped send is not fatal here, unlike the two
    dead-man's switches: this alert reports a condition outside the pipeline's control,
    and the run stays green either way (see :func:`pipeline._check_awaiting_enclosure`).
    """
    try:
        send_fn(_build_awaiting_enclosure_message(waiting, threshold_days, run_url))
    except Exception as exc:  # noqa: BLE001 — reconciliation's remaining work must finish.
        logger.error("awaiting-enclosure alert failed to send: %s", exc)
        return False
    logger.info("awaiting-enclosure alert sent (%d sermon(s) waiting)", len(waiting))
    return True


def _build_suspected_duplicate_audio_message(
    waiting: list[tuple[str, str, str, int]], threshold_days: int, run_url: str
) -> EmailMessage:
    """Assemble the suspected-duplicate-audio email (ADR-0077, #517).

    ``waiting`` is ``(guid, title, duplicate_of_guid, days_waited)`` per sermon, longest
    wait first. Like the awaiting-enclosure alert, this reports no failure: the
    pipeline is withholding the sermon on purpose, pending either the feed correcting
    the enclosure or a human confirming the duplicate.
    """
    count = len(waiting)
    longest = waiting[0][3]
    noun = "sermon" if count == 1 else "sermons"
    listing = "\n".join(
        f"  - {title} ({guid}) — matches {duplicate_of}, waiting {days} days"
        for guid, title, duplicate_of, days in waiting
    )
    body = (
        f"{count} {noun} withheld because its audio fingerprint matched another "
        f"record's.\n\n"
        f"{listing}\n\n"
        f"Threshold: SUSPECTED_DUPLICATE_AUDIO_DAYS = {threshold_days} days\n\n"
        "Nothing has failed. Before transcribing, the pipeline compares each sermon's "
        "audio against its source's other records; a match withholds it rather than "
        "spending a transcription on what is very likely a re-served enclosure "
        "(ADR-0077). It resumes on its own if the feed later serves different audio — "
        "no recovery step is needed.\n\n"
        "Past this threshold, confirm which record is correct: check the episode page "
        "against the guid listed above. If this is a false positive — two genuinely "
        "different sermons whose audio coincidentally fingerprinted alike — a "
        "workflow_dispatch run overrides the withholding, exactly as it does for a "
        "missing enclosure (ADR-0066).\n\n"
        f"GitHub Actions run: {run_url}\n"
    )
    return EmailMessage(
        from_addr=config.get("NOTIFY_EMAIL_FROM"),
        to_addr=config.get("NOTIFY_EMAIL_TO"),
        subject=f"[Sermon Notes] {count} {noun} withheld as a suspected audio duplicate ({longest} days)",
        body=body,
    )


def send_suspected_duplicate_audio_alert(
    *,
    waiting: list[tuple[str, str, str, int]],
    threshold_days: int,
    run_url: str,
    send_fn: Callable[[EmailMessage], None] = default_send,
) -> bool:
    """Send the suspected-duplicate-audio alert (ADR-0077, #517); never raise, but report the drop.

    Returns whether it was delivered. A dropped send is not fatal here, unlike the two
    dead-man's switches: this alert reports a condition outside the pipeline's own
    failure modes, and the run stays green either way (see
    :func:`pipeline._check_suspected_duplicate_audio`).
    """
    try:
        send_fn(_build_suspected_duplicate_audio_message(waiting, threshold_days, run_url))
    except Exception as exc:  # noqa: BLE001 — reconciliation's remaining work must finish.
        logger.error("suspected-duplicate-audio alert failed to send: %s", exc)
        return False
    logger.info("suspected-duplicate-audio alert sent (%d sermon(s) waiting)", len(waiting))
    return True


def _red_streak_body(
    streak_count: int, threshold: int, newest_run_url: str, oldest_run_at: str, run_url: str
) -> str:
    """The red-run-streak alert's body text — shared by the email and the paired issue."""
    return (
        f"The last {streak_count} completed pipeline.yml run(s) on main all came back red.\n\n"
        f"Threshold: RED_RUN_STREAK_COUNT = {threshold}\n"
        f"Streak began around: {oldest_run_at}\n"
        f"Most recent run: {newest_run_url}\n\n"
        "This is the tighter half of the dead-man's switch (#283): it fires on the run's "
        "own recorded outcome, not on a failure the pipeline recognised, so it also "
        "catches a red run nothing else explains — a crash before any escalation code "
        "ran, a broken install step, a runner fault. If a specific per-run escalation "
        "already named a cause for one of these runs, start there; if not, start from "
        "the run log linked above.\n\n"
        f"GitHub Actions run: {run_url}\n"
    )


def _build_red_streak_message(
    streak_count: int, threshold: int, newest_run_url: str, oldest_run_at: str, run_url: str
) -> EmailMessage:
    """Assemble the red-run-streak email — cause-agnostic, like the staleness alert."""
    return EmailMessage(
        from_addr=config.get("NOTIFY_EMAIL_FROM"),
        to_addr=config.get("NOTIFY_EMAIL_TO"),
        subject=f"[Sermon Notes] {streak_count} consecutive red pipeline runs",
        body=_red_streak_body(streak_count, threshold, newest_run_url, oldest_run_at, run_url),
    )


def send_red_streak_alert(
    *,
    streak_count: int,
    threshold: int,
    newest_run_url: str,
    oldest_run_at: str,
    run_url: str,
    send_fn: Callable[[EmailMessage], None] = default_send,
    create_issue: Callable[..., bool] = github_issues.create_or_group_issue,
) -> bool:
    """Send the red-run-streak alert (#283); never raise, but report the drop.

    The second, tighter dead-man's switch's delivery arm — parallel to
    :func:`send_staleness_alert`, not :func:`send_escalation`: this switch by
    definition does not know *why* the runs are red, the same way the staleness switch
    does not know why nothing published, so it gets its own cause-agnostic message
    rather than borrowing the escalation's ``failure_class`` shape — and, for the same
    reason, its paired GitHub issue (ADR-0080) uses the fixed
    ``_RED_STREAK_FAILURE_CLASS`` rather than a per-occurrence one.

    Like the staleness alert, a dropped send is itself the incident for a switch of
    last resort, so :func:`pipeline.reconcile` fails the run on a ``False`` here rather
    than logging and moving on. The paired issue call is independent of that outcome.
    """
    try:
        message = _build_red_streak_message(
            streak_count, threshold, newest_run_url, oldest_run_at, run_url
        )
        send_fn(message)
    except Exception as exc:  # noqa: BLE001 — the caller fails the run on the False.
        logger.error("red-run-streak alert failed to send: %s", exc)
        delivered = False
    else:
        logger.info("red-run-streak alert sent (%d consecutive red runs)", streak_count)
        delivered = True
    create_issue(
        failure_class=_RED_STREAK_FAILURE_CLASS,
        body=_red_streak_body(streak_count, threshold, newest_run_url, oldest_run_at, run_url),
    )
    return delivered


def _stalled_queue_body(
    pr_number: int, pr_title: str, hours_stale: int, threshold_hours: int, pr_url: str
) -> str:
    """The stalled-queue alert's body text — shared by the email and the paired issue."""
    return (
        f"Pull request #{pr_number} ({pr_title!r}) has worn `queue:ready` for "
        f"{hours_stale} hours without merging.\n\n"
        f"Threshold: QUEUE_STALL_HOURS = {threshold_hours}\n"
        f"Pull request: {pr_url}\n\n"
        "The merge train (ADR-0043) evicts the failures it predicts — a conflict, a "
        "red head — with a comment explaining why. A PR that is neither evicted nor "
        "merging is stuck on something the train did not predict: a revoked or "
        "expired App credential, a drift in main's branch protection, an auto-merge "
        "that never satisfies the base, the queue:ready label itself missing from the "
        "repository, or an eviction whose head_sha no longer matched. Start from the "
        "most recent merge-train.yml run, not from this PR's own checks.\n"
    )


def _build_stalled_queue_message(
    pr_number: int, pr_title: str, hours_stale: int, threshold_hours: int, pr_url: str
) -> EmailMessage:
    """Assemble the stalled-merge-train email — cause-agnostic, like the other two.

    Names no cause because the check that trips it (``queue-stall-alert.yml``,
    ADR-0057) does not know one: the PR could be stuck behind a revoked App
    credential, a ruleset drift, an auto-merge that never satisfies its base, or a
    missed eviction — the switch exists precisely because none of those produce any
    other signal.
    """
    return EmailMessage(
        from_addr=config.get("NOTIFY_EMAIL_FROM"),
        to_addr=config.get("NOTIFY_EMAIL_TO"),
        subject=f"[Sermon Notes] PR #{pr_number} stalled in the merge queue ({hours_stale}h)",
        body=_stalled_queue_body(pr_number, pr_title, hours_stale, threshold_hours, pr_url),
    )


def send_stalled_queue_alert(
    *,
    pr_number: int,
    pr_title: str,
    hours_stale: int,
    threshold_hours: int,
    pr_url: str,
    send_fn: Callable[[EmailMessage], None] = default_send,
) -> bool:
    """Send the stalled-merge-queue alert (#339); never raise, but report the drop.

    A third dead-man's switch's delivery arm, parallel to :func:`send_staleness_alert`
    and :func:`send_red_streak_alert`: it fires on the merge train's own governance
    state rather than anything about a sermon, and like those two it does not know
    why the PR is stuck — only that it has been wearing the label too long.

    A dropped send is itself the incident, so the caller (the ``alert-stalled-queue``
    CLI command) exits non-zero on a ``False`` here, the same fail-loud-second-channel
    posture as the other two.

    Unlike :func:`send_escalation`, :func:`send_staleness_alert`, and
    :func:`send_red_streak_alert`, this one has no paired GitHub issue (ADR-0080): the
    ``queue-stall-alert.yml`` job's token stays read-only by design (ADR-0057), and
    this alert must not be the reason that invariant is given up.
    """
    try:
        message = _build_stalled_queue_message(
            pr_number, pr_title, hours_stale, threshold_hours, pr_url
        )
        send_fn(message)
    except Exception as exc:  # noqa: BLE001 — the caller fails the run on the False.
        logger.error("stalled-queue alert for PR #%d failed to send: %s", pr_number, exc)
        return False
    logger.info("stalled-queue alert sent for PR #%d (%d hours stale)", pr_number, hours_stale)
    return True


def _build_note_message(sermon_title: str, sermon_date: str, artifact_path: Path) -> EmailMessage:
    """Assemble the published-note delivery email with the ``.docx`` attached."""
    content_b64 = base64.b64encode(artifact_path.read_bytes()).decode("ascii")
    body = (
        "A new sermon study note has been published.\n\n"
        f"Sermon: {sermon_title} ({sermon_date})\n\n"
        "The .docx study note is attached.\n"
    )
    return EmailMessage(
        from_addr=config.get("NOTIFY_EMAIL_FROM"),
        to_addr=config.get("NOTIFY_EMAIL_TO"),
        subject=f"[Sermon Notes] New note: {sermon_title} ({sermon_date})",
        body=body,
        attachments=(Attachment(filename=artifact_path.name, content_b64=content_b64),),
    )


def send_note(
    *,
    sermon_title: str,
    sermon_date: str,
    artifact_path: Path,
    send_fn: Callable[[EmailMessage], None] = default_send,
) -> bool:
    """Email the published note's ``.docx`` to ``NOTIFY_EMAIL_TO``; never raise.

    Reads the rendered ``.docx`` from ``artifact_path``, attaches it base64-encoded,
    and transmits through ``send_fn``. Returns ``True`` when the transport accepts it.
    Delivery is best-effort (PRD §11.2): an unreadable artifact or a transport failure
    is logged and swallowed — the ``.docx`` is committed to the repo regardless, so a
    dropped email loses only the convenience copy, never the artifact — and reported as
    ``False``.
    """
    try:
        message = _build_note_message(sermon_title, sermon_date, artifact_path)
        send_fn(message)
    except Exception as exc:  # noqa: BLE001 — note delivery is best-effort (PRD §11.2).
        logger.error("note delivery email for %r failed to send: %s", sermon_title, exc)
        return False
    logger.info("note delivery email sent for %r", sermon_title)
    return True
