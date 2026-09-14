"""Download a sermon's audio and transcribe it with faster-whisper — the model boundary.

Downloads go through :mod:`poller.net` (the sole HTTP boundary, CLAUDE.md §6); this
module owns only the model call and its retry/error classification. Nothing here
caches the transcript to disk — no sermon-specific content may persist in this
public repo (ADR-0004) — so a cache-hit concept doesn't exist: every call downloads
and transcribes, and the caller (:mod:`poller.transcriber`) is what makes a repeat
run cheap, by never selecting a sermon whose ledger record already says done.

faster-whisper is imported lazily inside :func:`_get_model` so unit tests and
tooling never load model weights; the model call itself is mocked at the
:func:`transcribe_audio` boundary via its ``transcribe`` parameter.
"""

from __future__ import annotations

import hashlib
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from poller import config
from poller.net import download_audio, http_download

_TRANSCRIBE_ATTEMPTS = 2
_TRANSCRIBE_BACKOFF_BASE = 1.0

# Built once on first real transcription; mocked away in tests.
_model: Any = None


class EmptyTranscriptError(RuntimeError):
    """Raised when transcription yields empty/whitespace text after its retries."""


class TranscriptionError(RuntimeError):
    """The model call failed on this file after its retries — a per-content failure.

    A corrupt or undecodable enclosure makes faster-whisper raise (ffmpeg/ctranslate2
    internals); this wraps that so a model failure on one sermon never aborts a batch.
    """


def _get_model() -> Any:
    """Build (once) and return the faster-whisper model, sized from config."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        whisper_cfg = config.load_whisper_config()
        _model = WhisperModel(
            whisper_cfg.model,
            device="cpu",
            compute_type=whisper_cfg.compute_type,
            cpu_threads=whisper_cfg.cpu_threads,
        )
    return _model


def default_transcribe(audio_path: Path) -> str:
    """Run faster-whisper over ``audio_path`` and return the joined transcript text."""
    segments, _info = _get_model().transcribe(str(audio_path), vad_filter=True, log_progress=True)
    return " ".join(segment.text.strip() for segment in segments)


def _sha256(text: str) -> str:
    """Return the sha256 hex digest of ``text``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _transcribe_with_retries(
    audio_path: Path,
    *,
    transcribe: Callable[[Path], str],
    sleep: Callable[[float], None],
    attempts: int = _TRANSCRIBE_ATTEMPTS,
    backoff_base: float = _TRANSCRIBE_BACKOFF_BASE,
) -> str:
    """Transcribe ``audio_path``, treating an empty result or a raising model call as failure.

    Up to ``attempts`` runs, backing off between them. Empty/whitespace output exhausts to
    :class:`EmptyTranscriptError`; a model call that keeps raising exhausts to
    :class:`TranscriptionError`. Returns the stripped transcript on success.
    """
    model_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            text = transcribe(audio_path).strip()
        except Exception as exc:  # noqa: BLE001 — a per-file model failure; classified below.
            model_error = exc
        else:
            if text:
                return text
            model_error = None
        if attempt < attempts:
            sleep(backoff_base * 2 ** (attempt - 1))
    if model_error is not None:
        raise TranscriptionError(
            f"transcription failed after {attempts} attempts: {model_error}"
        ) from model_error
    raise EmptyTranscriptError(f"transcription produced no text after {attempts} attempts")


def transcribe_audio(
    audio_url: str,
    *,
    transcribe: Callable[[Path], str] = default_transcribe,
    download: Callable[[str, Path], None] = http_download,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[str, str]:
    """Download ``audio_url``'s enclosure and transcribe it, returning ``(text, sha256_hash)``.

    Raises :class:`~poller.net.AudioDownloadError` when the download is exhausted (the
    caller should leave the sermon pending to retry next run), or
    :class:`TranscriptionError`/:class:`EmptyTranscriptError` when the model itself fails
    after its retries (the caller should mark the sermon terminally failed). The downloaded
    file lives only in a temp directory removed before this returns.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir) / "audio"
        download_audio(audio_url, dest, download=download, sleep=sleep)
        text = _transcribe_with_retries(dest, transcribe=transcribe, sleep=sleep)
    return text, _sha256(text)
