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
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

from poller import config
from poller.net import download_audio, http_download

_TRANSCRIBE_ATTEMPTS = 2
_TRANSCRIBE_BACKOFF_BASE = 1.0

# Built once on first real transcription; mocked away in tests.
_model: Any = None


@dataclass(frozen=True)
class DecodingProfile:
    """Every input that can change a transcript's text (ADR-0022).

    Built once per call by :func:`decoding_profile`; the model call's arguments come from
    it and the ledger records it, so what is recorded is what the model was given. Field
    names are ``store.PROFILE_KEYS``. ``hallucination_silence_threshold`` is ``0`` when the
    guard is off; ``hotwords`` is ``None`` when none were sent; a version is ``None`` when
    the package isn't installed.
    """

    model: str
    compute_type: str
    beam_size: int
    condition_on_previous_text: bool
    hallucination_silence_threshold: float
    vad_filter: bool
    domain_prompt: bool
    hotwords: str | None
    faster_whisper_version: str | None
    ctranslate2_version: str | None

    def model_kwargs(self) -> dict[str, Any]:
        """The decoding arguments for ``WhisperModel.transcribe``."""
        kwargs: dict[str, Any] = {
            "vad_filter": self.vad_filter,
            "beam_size": self.beam_size,
            "condition_on_previous_text": self.condition_on_previous_text,
            "hotwords": self.hotwords,
        }
        if self.hallucination_silence_threshold:
            # faster-whisper applies the threshold only when word timestamps are on.
            kwargs["word_timestamps"] = True
            kwargs["hallucination_silence_threshold"] = self.hallucination_silence_threshold
        return kwargs


def _installed_version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def decoding_profile(whisper_cfg: config.WhisperConfig, hotwords: str | None) -> DecodingProfile:
    """The profile a call with ``whisper_cfg`` and ``hotwords`` decodes under.

    ``hotwords`` is dropped here unless ``WHISPER_DOMAIN_PROMPT`` is on, so the profile
    never records terms the model wasn't sent.
    """
    return DecodingProfile(
        model=whisper_cfg.model,
        compute_type=whisper_cfg.compute_type,
        beam_size=whisper_cfg.beam_size,
        condition_on_previous_text=whisper_cfg.condition_on_previous_text,
        hallucination_silence_threshold=whisper_cfg.hallucination_silence_threshold or 0,
        vad_filter=True,
        domain_prompt=whisper_cfg.domain_prompt,
        hotwords=hotwords if whisper_cfg.domain_prompt else None,
        faster_whisper_version=_installed_version("faster-whisper"),
        ctranslate2_version=_installed_version("ctranslate2"),
    )


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


def default_transcribe(audio_path: Path, hotwords: str | None) -> tuple[str, DecodingProfile]:
    """Run faster-whisper over ``audio_path``; return the joined text and the profile it ran under.

    This is the one place that talks to the model. ``hotwords`` (poller/prompting.py)
    reaches it only through :func:`decoding_profile`.
    """
    profile = decoding_profile(config.load_whisper_config(), hotwords)
    segments, _info = _get_model().transcribe(str(audio_path), log_progress=True, **profile.model_kwargs())
    return " ".join(segment.text.strip() for segment in segments), profile


def transcript_hash(text: str) -> str:
    """Return the sha256 hex digest of ``text`` — the ledger's ``transcript_hash``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _transcribe_with_retries(
    audio_path: Path,
    hotwords: str | None,
    *,
    transcribe: Callable[[Path, str | None], tuple[str, DecodingProfile]],
    sleep: Callable[[float], None],
    attempts: int = _TRANSCRIBE_ATTEMPTS,
    backoff_base: float = _TRANSCRIBE_BACKOFF_BASE,
) -> tuple[str, DecodingProfile]:
    """Transcribe ``audio_path``, treating an empty result or a raising model call as failure.

    Up to ``attempts`` runs, backing off between them. Empty/whitespace output exhausts to
    :class:`EmptyTranscriptError`; a model call that keeps raising exhausts to
    :class:`TranscriptionError`. Returns the stripped transcript and its profile on success.
    """
    model_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            text, profile = transcribe(audio_path, hotwords)
        except Exception as exc:  # noqa: BLE001 — a per-file model failure; classified below.
            model_error = exc
        else:
            if text.strip():
                return text.strip(), profile
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
    hotwords: str | None = None,
    *,
    transcribe: Callable[[Path, str | None], tuple[str, DecodingProfile]] = default_transcribe,
    download: Callable[[str, Path], None] = http_download,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[str, str, DecodingProfile]:
    """Download ``audio_url``'s enclosure and transcribe it, returning ``(text, sha256_hash, profile)``.

    Raises :class:`~poller.net.AudioDownloadError` when the download is exhausted (the
    caller should leave the sermon pending to retry next run), or
    :class:`TranscriptionError`/:class:`EmptyTranscriptError` when the model itself fails
    after its retries (the caller should mark the sermon terminally failed). The downloaded
    file lives only in a temp directory removed before this returns.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir) / "audio"
        download_audio(audio_url, dest, download=download, sleep=sleep)
        text, profile = _transcribe_with_retries(dest, hotwords, transcribe=transcribe, sleep=sleep)
    return text, transcript_hash(text), profile
