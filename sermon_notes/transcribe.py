"""Transcribe a sermon with faster-whisper, cache it, and advance its state.

This wraps the faster-whisper engine (ADR-0003): it downloads the enclosure via
:mod:`sermon_notes.audio`, runs the model on the CPU runner, caches the
plain transcript under ``transcripts/`` (named ``YYYY-MM-DD_<title-slug>_<id>.txt``,
deterministic per sermon for idempotency), sets
``transcript_hash`` (sha256), and advances the sermon ``discovered → transcribed``
(PRD §6.3). An empty transcript or a model call that keeps failing on a corrupt file
sends the sermon terminal after its PRD §11.1 retries, while an exhausted download
leaves it ``discovered`` to retry next run (ADR-0009). A distinct sermon whose transcript hash already produced a successful
generation is deduplicated — sent terminal and escalated once, never re-generated
(PRD §11.3, #86).

faster-whisper is imported lazily inside :func:`default_transcribe` so unit tests
and tooling never load model weights; the model is mocked at the
:func:`transcribe_sermon` boundary.
"""

from __future__ import annotations

import hashlib
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from sermon_notes import audio_fingerprint, config
from sermon_notes.audio import AudioDownloadError, download_audio, http_download
from sermon_notes.logging import get_logger
from sermon_notes.registry import Registry, RunRecord, SermonRecord, UnknownSermonError
from sermon_notes.slug import slugify

logger = get_logger()

# transcripts/ lives at the repo root (two parents above this package file).
DEFAULT_TRANSCRIPTS_DIR = Path(__file__).resolve().parents[2] / "transcripts"
_TRANSCRIBE_ATTEMPTS = 2
_TRANSCRIBE_BACKOFF_BASE = 1.0
_UNSAFE_FILENAME = re.compile(r"[^A-Za-z0-9._-]+")

# Built once on first real transcription; mocked away in tests. The batched
# pipeline (when enabled) wraps the same model, so it shares its weights.
_model: Any = None
_batched_model: Any = None

# Truthy env spellings for the boolean transcription knobs.
_TRUE = frozenset({"1", "true", "yes", "on"})


def _bool_config(name: str, default: str) -> bool:
    """Read ``name`` as a boolean env flag (``1/true/yes/on`` are true, case-insensitive)."""
    return config.get(name, default).strip().lower() in _TRUE


def _audio_duplicate_threshold() -> float:
    """The audio-fingerprint similarity threshold, overridable via config (ADR-0077)."""
    return config.get_float(
        "AUDIO_DUPLICATE_THRESHOLD", audio_fingerprint.DEFAULT_SIMILARITY_THRESHOLD
    )


class EmptyTranscriptError(RuntimeError):
    """Raised when transcription yields empty/whitespace text after its retries."""


class DuplicateAudioError(RuntimeError):
    """Raised by the download hook when a fresh download's fingerprint matches a
    sibling's (ADR-0077, #517) — aborts before the expensive transcription call."""

    def __init__(self, fingerprint: str, duplicate_of: str) -> None:
        super().__init__(f"audio fingerprint matches already-known guid {duplicate_of!r}")
        self.fingerprint = fingerprint
        self.duplicate_of = duplicate_of


class TranscriptionError(RuntimeError):
    """The model call failed on this file after its retries — a per-content failure (#126).

    A corrupt or undecodable enclosure makes faster-whisper raise a runtime error
    (ffmpeg/ctranslate2). Per PRD §11.1 that is retried, then terminal for *this* sermon;
    wrapping it here keeps the model's library exceptions out of the orchestrator so one
    bad file never aborts the batch.
    """


@dataclass(frozen=True)
class TranscribeResult:
    """Outcome of transcribing one sermon (``outcome`` is success|failed|retry_scheduled)."""

    guid: str
    outcome: str
    transcript_hash: str | None = None
    error_class: str | None = None
    error_detail: str | None = None


def _now() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _cpu_threads() -> int:
    """Threads for ctranslate2 to decode on.

    Defaults to the runner's detected core count (4 on ``ubuntu-latest``) rather than
    leaving ctranslate2's ``0`` auto-detect, which can under-count cores inside a CI
    container and silently under-thread. Overridable via ``WHISPER_CPU_THREADS``.
    """
    return config.get_int("WHISPER_CPU_THREADS", os.cpu_count() or 4, minimum=1)


def _get_model() -> Any:
    """Build (once) and return the faster-whisper model from config."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        _model = WhisperModel(
            config.get("WHISPER_MODEL", "small"),
            device="cpu",
            compute_type=config.get("WHISPER_COMPUTE_TYPE", "int8"),
            cpu_threads=_cpu_threads(),
        )
    return _model


def _get_batched_model() -> Any:
    """Build (once) and return the batched pipeline wrapping the shared model."""
    global _batched_model
    if _batched_model is None:
        from faster_whisper import BatchedInferencePipeline

        _batched_model = BatchedInferencePipeline(model=_get_model())
    return _batched_model


def default_transcribe(audio_path: Path) -> str:
    """Run faster-whisper over ``audio_path`` and return the joined transcript text.

    The accuracy/runtime knobs are read from config (ADR-0003 frames these as tunable):
    ``WHISPER_BEAM_SIZE`` (decode beam width), ``WHISPER_VAD_FILTER`` (skip non-speech
    so silence/music is never decoded — usually both faster and cleaner), and
    ``WHISPER_BATCHED`` to route through faster-whisper's batched pipeline
    (``WHISPER_BATCH_SIZE`` chunks; the pipeline VAD-segments internally, so its VAD is
    always on). Logs the realtime factor so the Actions log shows where a run's time
    goes.
    """
    beam_size = config.get_int("WHISPER_BEAM_SIZE", 5, minimum=1)
    batched = _bool_config("WHISPER_BATCHED", "false")
    vad = _bool_config("WHISPER_VAD_FILTER", "true")

    started = time.perf_counter()
    if batched:
        segments, info = _get_batched_model().transcribe(
            str(audio_path),
            beam_size=beam_size,
            batch_size=config.get_int("WHISPER_BATCH_SIZE", 8, minimum=1),
        )
    else:
        segments, info = _get_model().transcribe(
            str(audio_path),
            beam_size=beam_size,
            vad_filter=vad,
        )
    # The generator does the decoding work; timing must span the join, not just the call.
    text = " ".join(segment.text.strip() for segment in segments)
    elapsed = time.perf_counter() - started
    logger.info(
        "whisper: %.0fs audio in %.0fs (%.1fx realtime) beam=%d vad=%s batched=%s",
        info.duration,
        elapsed,
        (info.duration / elapsed) if elapsed else 0.0,
        beam_size,
        "n/a" if batched else vad,
        batched,
    )
    return text


def _guid_identifier(guid: str) -> str:
    """The stable part of a feed guid: the whole guid sanitized for the filesystem.

    Source-agnostic (ADR-0012): no church-specific host prefix is stripped, so every
    source's guid maps deterministically and uniquely.
    """
    return _UNSAFE_FILENAME.sub("_", guid)


def cache_path(transcripts_dir: Path, sermon: SermonRecord) -> Path:
    """Map a sermon to its cache file: ``<source>/YYYY-MM-DD_<title-slug>_<id>.txt``.

    The name carries the publish date and a slugified title for human legibility, plus
    the sanitized guid as a stable, unique identifier, all under the church's
    ``<source>`` segment (ADR-0012). It is deterministic — the same sermon always
    yields the same path — preserving the idempotency the cache and the §11.3 skip
    guard depend on.
    """
    stem = f"{sermon.published_on}_{slugify(sermon.title)}_{_guid_identifier(sermon.guid)}"
    return transcripts_dir / sermon.source / f"{stem}.txt"


def _write_cache_atomic(cache_file: Path, transcript: str) -> None:
    """Write ``transcript`` to ``cache_file`` via a ``.tmp`` sibling + ``os.replace``.

    A process killed mid-write (job cancellation, the 6-hour Actions cap, a runner
    eviction) must never leave a partial transcript at ``cache_file`` — the next run's
    cache-hit guard trusts existence alone (#199), and a truncated transcript would be
    silently transcribed, generated, and published as a note. Same contract as
    :func:`sermon_notes.registry.Registry.save`. A hard kill can still leave the
    ``.tmp`` sibling behind (nothing in-process can prevent that), which is why
    ``.gitignore`` excludes ``*.tmp`` — but an in-process exception cleans up after
    itself here so the common failure path doesn't litter the tree (#272).
    """
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache_file.with_suffix(cache_file.suffix + ".tmp")
    try:
        tmp.write_text(transcript, encoding="utf-8")
        os.replace(tmp, cache_file)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


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

    Per PRD §11.1: up to ``attempts`` runs, backing off between them. Empty/whitespace
    output exhausts to :class:`EmptyTranscriptError`; a model call that keeps raising
    (a corrupt/undecodable file) exhausts to :class:`TranscriptionError`. Both are
    terminal for this sermon (#126). Returns the stripped transcript on success.
    """
    model_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            text = transcribe(audio_path).strip()
        except Exception as exc:  # noqa: BLE001 — a per-file model failure; classified below.
            model_error = exc
            logger.warning("transcription error on attempt %d/%d: %s", attempt, attempts, exc)
        else:
            if text:
                return text
            model_error = None
            logger.warning("empty transcript on attempt %d/%d", attempt, attempts)
        if attempt < attempts:
            sleep(backoff_base * 2 ** (attempt - 1))
    if model_error is not None:
        raise TranscriptionError(
            f"transcription failed after {attempts} attempts: {model_error}"
        ) from model_error
    raise EmptyTranscriptError(f"transcription produced no text after {attempts} attempts")


def _produce_transcript(
    audio_url: str,
    *,
    transcribe: Callable[[Path], str],
    download: Callable[[str, Path], None],
    sleep: Callable[[float], None],
    on_downloaded: Callable[[Path], None] | None = None,
) -> str:
    """Download the enclosure to a temp file and transcribe it, returning the text.

    ``on_downloaded``, when given, runs on the downloaded file before transcription —
    :func:`transcribe_sermon` uses it to fingerprint the audio and check for a
    duplicate before spending the (expensive) transcription call (ADR-0077, #517).
    Raising from it aborts before ``transcribe`` is ever reached.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        dest = Path(tmpdir) / "audio"
        download_audio(audio_url, dest, download=download, sleep=sleep)
        if on_downloaded is not None:
            on_downloaded(dest)
        return _transcribe_with_retries(dest, transcribe=transcribe, sleep=sleep)


def _go_terminal(
    registry: Registry,
    guid: str,
    exc: Exception,
    *,
    started_at: str,
    now: str | None,
) -> TranscribeResult:
    """Advance ``guid`` to ``failed`` and record the terminal run (PRD §11.1)."""
    error_class = type(exc).__name__
    error_detail = str(exc)
    registry.advance(guid, "failed", now=now)
    registry.append_run(
        guid,
        RunRecord(
            attempted_state="transcribed",
            outcome="failed_terminal",
            error_class=error_class,
            error_detail=error_detail,
            started_at=started_at,
            finished_at=now if now is not None else _now(),
        ),
    )
    logger.warning("transcription terminal for %s: %s: %s", guid, error_class, error_detail)
    return TranscribeResult(
        guid=guid, outcome="failed", error_class=error_class, error_detail=error_detail
    )


def _go_duplicate(
    registry: Registry,
    sermon: SermonRecord,
    transcript_hash: str,
    *,
    started_at: str,
    now: str | None,
) -> TranscribeResult:
    """Send a cross-guid duplicate terminal and queue its single escalation (#86, PRD §11.3).

    An identical transcript never triggers a second LLM call. But when a *distinct*
    sermon (a re-broadcast, re-upload, or cross-source duplicate) hashes to one already
    generated, leaving it ``discovered`` strands it: it is re-polled, re-hashed, and
    re-skipped every run with no operator signal. Instead it advances to terminal
    ``failed`` with a ``DuplicateTranscript`` class so the orchestrator escalates it
    exactly once, and its hash is recorded for audit. The append-only run history
    captures why.
    """
    error_class = "DuplicateTranscript"
    error_detail = f"transcript identical to an already-generated sermon (hash {transcript_hash})"
    sermon.transcript_hash = transcript_hash
    registry.advance(sermon.guid, "failed", now=now)
    registry.append_run(
        sermon.guid,
        RunRecord(
            attempted_state="transcribed",
            outcome="failed_terminal",
            error_class=error_class,
            error_detail=error_detail,
            started_at=started_at,
            finished_at=now if now is not None else _now(),
        ),
    )
    logger.warning("duplicate transcript for %s, sending terminal: %s", sermon.guid, error_detail)
    return TranscribeResult(
        guid=sermon.guid,
        outcome="failed",
        transcript_hash=transcript_hash,
        error_class=error_class,
        error_detail=error_detail,
    )


def _go_suspected_duplicate(
    registry: Registry,
    sermon: SermonRecord,
    exc: DuplicateAudioError,
    *,
    started_at: str,
    now: str | None,
) -> TranscribeResult:
    """Record a suspected-duplicate-audio detection and leave the sermon ``discovered``.

    Unlike :func:`_go_duplicate` (a *post*-transcription transcript-hash collision,
    which is terminal), this fires *before* transcription and is not terminal
    (ADR-0077, #517): the sermon resumes on its own if the feed later serves a
    different enclosure (``Registry.upsert`` clears the marker), and plan-time
    filtering withholds it meanwhile without re-downloading or re-spending a claim,
    mirroring ADR-0066's missing-enclosure withholding.
    """
    error_class = "SuspectedDuplicateAudio"
    error_detail = f"audio fingerprint matches {exc.duplicate_of}; withheld pending distinct audio"
    sermon.audio_fingerprint = exc.fingerprint
    sermon.suspected_duplicate_of = exc.duplicate_of
    registry.append_run(
        sermon.guid,
        RunRecord(
            attempted_state="transcribed",
            outcome="retry_scheduled",
            error_class=error_class,
            error_detail=error_detail,
            started_at=started_at,
            finished_at=now if now is not None else _now(),
        ),
    )
    logger.warning(
        "suspected duplicate audio for %s (matches %s): withheld, not transcribed",
        sermon.guid,
        exc.duplicate_of,
    )
    return TranscribeResult(
        guid=sermon.guid,
        outcome="retry_scheduled",
        error_class=error_class,
        error_detail=error_detail,
    )


def _go_recoverable(
    registry: Registry,
    guid: str,
    exc: Exception,
    *,
    started_at: str,
    now: str | None,
) -> TranscribeResult:
    """Record a non-terminal download failure and leave ``guid`` in ``discovered``.

    Per ADR-0009: a transient CDN/edge rejection (e.g. a 403 from the Podbean CDN)
    must not be terminal. The sermon stays ``discovered`` so the next scheduled run
    re-attempts it; the attempt is logged as a ``retry_scheduled`` run for observability
    and is not escalated.
    """
    error_class = type(exc).__name__
    error_detail = str(exc)
    registry.append_run(
        guid,
        RunRecord(
            attempted_state="transcribed",
            outcome="retry_scheduled",
            error_class=error_class,
            error_detail=error_detail,
            started_at=started_at,
            finished_at=now if now is not None else _now(),
        ),
    )
    logger.warning(
        "download failed for %s, retry next run: %s: %s", guid, error_class, error_detail
    )
    return TranscribeResult(
        guid=guid, outcome="retry_scheduled", error_class=error_class, error_detail=error_detail
    )


def transcribe_and_cache(
    sermon: SermonRecord,
    *,
    transcribe: Callable[[Path], str] = default_transcribe,
    download: Callable[[str, Path], None] = http_download,
    transcripts_dir: Path = DEFAULT_TRANSCRIPTS_DIR,
    sleep: Callable[[float], None] = time.sleep,
    on_downloaded: Callable[[Path], None] | None = None,
) -> str:
    """Return ``sermon``'s cached transcript, producing and caching it on a miss.

    The registry-free half of :func:`transcribe_sermon` (spec 0025): a cache hit is
    reused unchanged; a miss downloads the enclosure, runs the model, and writes the
    cache atomically. It never touches a :class:`Registry`, so a local bulk-backfill
    script can call this to pre-populate ``transcripts/`` ahead of time without any
    way to advance a sermon's state — that stays with the pipeline's sole-writer
    ``merge`` job (CLAUDE.md §6). The next real run's cache-hit check here picks up
    the pre-populated file and skips the download and the model entirely.

    ``on_downloaded`` never fires on a cache hit — nothing was downloaded to hand it.
    """
    cache_file = cache_path(transcripts_dir, sermon)
    if cache_file.exists():
        return cache_file.read_text(encoding="utf-8")
    transcript = _produce_transcript(
        sermon.audio_url,
        transcribe=transcribe,
        download=download,
        sleep=sleep,
        on_downloaded=on_downloaded,
    )
    _write_cache_atomic(cache_file, transcript)
    return transcript


def transcribe_sermon(
    registry: Registry,
    guid: str,
    *,
    transcribe: Callable[[Path], str] = default_transcribe,
    download: Callable[[str, Path], None] = http_download,
    fingerprint: Callable[[Path], str] = audio_fingerprint.compute_fingerprint,
    duplicate_lookup_registry: Registry | None = None,
    transcripts_dir: Path = DEFAULT_TRANSCRIPTS_DIR,
    sleep: Callable[[float], None] = time.sleep,
    now: str | None = None,
) -> TranscribeResult:
    """Download, transcribe, cache, hash, and advance one discovered sermon.

    Idempotent: a transcript already cached for ``guid`` is reused without
    re-downloading or re-running the model, and a distinct sermon whose hash already
    produced a successful generation is deduplicated to terminal ``failed`` and
    escalated once rather than left ``discovered`` (PRD §11.3, #86). On success the sermon
    advances ``discovered → transcribed``; an empty transcript advances it to the
    terminal ``failed`` state, while an exhausted download leaves it ``discovered`` to
    retry on the next run (PRD §11.1, ADR-0009). The caller persists the ledger.

    On a cache miss, the freshly downloaded audio is fingerprinted and checked against
    this source's other confirmed-distinct records *before* transcription (ADR-0077,
    #517) — a match withholds the sermon (still ``discovered``) instead of spending the
    transcription. ``duplicate_lookup_registry``, when given, is checked instead of
    ``registry`` for that comparison: a shard's own ``registry`` is a throwaway
    single-record ledger with no sibling records to compare against (ADR-0023), so
    :func:`~sermon_notes.pipeline.run_shard` passes a read-only load of the committed
    ledger here. Defaults to ``registry`` itself, which is correct for the serial
    pipeline, where ``registry`` already is the whole ledger.
    """
    sermon = registry.get(guid)
    if sermon is None:
        raise UnknownSermonError(f"no sermon with guid {guid!r} in the ledger")

    started_at = now if now is not None else _now()
    lookup_registry = (
        duplicate_lookup_registry if duplicate_lookup_registry is not None else registry
    )

    def _check_for_duplicate_audio(audio_path: Path) -> None:
        try:
            computed = fingerprint(audio_path)
        except Exception as exc:  # noqa: BLE001 — a corrupt/undecodable download degrades
            # to "no fingerprint," not a transcription failure: faster-whisper's own
            # decode attempt right after this is the real word on whether the file is
            # usable, and it already has its own retry/terminal handling (PRD §11.1).
            logger.warning(
                "audio fingerprint skipped for %s: %s: %s", guid, type(exc).__name__, exc
            )
            return
        match = lookup_registry.find_duplicate_audio(
            sermon.source,
            computed,
            exclude_guid=guid,
            threshold=_audio_duplicate_threshold(),
        )
        if match is not None:
            raise DuplicateAudioError(computed, match)
        sermon.audio_fingerprint = computed

    try:
        transcript = transcribe_and_cache(
            sermon,
            transcribe=transcribe,
            download=download,
            transcripts_dir=transcripts_dir,
            sleep=sleep,
            on_downloaded=_check_for_duplicate_audio,
        )
    except DuplicateAudioError as exc:
        return _go_suspected_duplicate(registry, sermon, exc, started_at=started_at, now=now)
    except AudioDownloadError as exc:
        return _go_recoverable(registry, guid, exc, started_at=started_at, now=now)
    except (EmptyTranscriptError, TranscriptionError) as exc:
        return _go_terminal(registry, guid, exc, started_at=started_at, now=now)

    transcript_hash = _sha256(transcript)
    if registry.has_successful_generation(transcript_hash):
        return _go_duplicate(registry, sermon, transcript_hash, started_at=started_at, now=now)

    sermon.transcript_hash = transcript_hash
    registry.advance(guid, "transcribed", now=now)
    registry.append_run(
        guid,
        RunRecord(
            attempted_state="transcribed",
            outcome="success",
            started_at=started_at,
            finished_at=now if now is not None else _now(),
        ),
    )
    logger.info("transcribed %s (%d chars)", guid, len(transcript))
    return TranscribeResult(guid=guid, outcome="success", transcript_hash=transcript_hash)
