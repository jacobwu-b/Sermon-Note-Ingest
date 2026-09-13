"""Shared SSRF host guard for the fetch boundaries (issue #129, defense-in-depth).

The feed-fetch and audio-download boundaries (:mod:`sermon_notes.sources.feedbase`,
:mod:`sermon_notes.audio`) already restrict the URL scheme and cap the response body
(issue #36), but a hostile ``audio_url``/feed URL could still point at an internal
address (cloud metadata, loopback, RFC-1918 ranges) that the privileged pipeline
runner would happily connect to. :func:`guard_public_host` resolves the URL's host
and refuses to proceed unless every resolved address is globally routable.

:func:`build_guarded_opener` closes the redirect half of that gap (issue #203): both
boundaries fetch through the returned opener instead of bare ``urlopen``, so a 3xx
response from an otherwise-public host re-runs the same scheme and host guards on the
``Location`` target before following it, and redirect depth is bounded explicitly.

Scope note: the remaining residual is DNS rebinding — this does not pin the resolved
IP for the actual connection, so an attacker who can swap the DNS answer between the
guard check and the real connect is not stopped. Given the current blast radius —
ephemeral GitHub-hosted runners with no metadata service or internal network to
reach — that residual risk is accepted for this P2 fix.
"""

from __future__ import annotations

import ipaddress
import socket
import urllib.parse
import urllib.request

# Mirrors the scheme allow-list each boundary already applies to the original URL;
# a redirect must be held to the same standard (issue #203).
_ALLOWED_REDIRECT_SCHEMES = frozenset({"http", "https"})
# Bounded explicitly rather than relying on urllib's default (10).
_MAX_REDIRECTS = 5


class UnsafeHostError(ValueError):
    """Raised when a URL's host cannot be resolved or resolves to a non-public address."""


def guard_public_host(url: str) -> None:
    """Raise :class:`UnsafeHostError` unless ``url``'s host resolves only to public IPs."""
    host = urllib.parse.urlsplit(url).hostname
    if not host:
        raise UnsafeHostError(f"URL has no host to validate: {url!r}")
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise UnsafeHostError(f"could not resolve host {host!r}: {exc}") from exc
    for *_rest, sockaddr in infos:
        ip = ipaddress.ip_address(sockaddr[0])
        if not ip.is_global:
            raise UnsafeHostError(f"refusing to fetch non-public host {host!r} -> {ip}")


class _GuardedRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Re-applies the scheme and host guards to every redirect hop (issue #203)."""

    max_redirections = _MAX_REDIRECTS

    def redirect_request(  # type: ignore[override]
        self, req: object, fp: object, code: int, msg: str, headers: object, newurl: str
    ) -> object:
        scheme = urllib.parse.urlsplit(newurl).scheme.lower()
        if scheme not in _ALLOWED_REDIRECT_SCHEMES:
            raise UnsafeHostError(f"refusing redirect to non-http(s) URL: {newurl!r}")
        guard_public_host(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)  # type: ignore[arg-type]


def build_guarded_opener() -> urllib.request.OpenerDirector:
    """Build an opener whose redirect handling re-runs the scheme/host guards on each hop."""
    return urllib.request.build_opener(_GuardedRedirectHandler)
