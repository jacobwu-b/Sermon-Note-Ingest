"""A dependency-free audio-content fingerprint, for duplicate-audio detection (ADR-0077).

Squarespace occasionally serves a feed item whose enclosure is the *previous*
week's audio, or a re-encoded copy of an already-seen recording (#517). This
computes a fingerprint from the audio's own decoded content — not from any
source's filename or URL convention — so a duplicate is detectable before
spending a transcription on it, and generalizes past Westgate.

The fingerprint is a **sequence** of per-time-window spectral-energy vectors,
not a single track-wide average. That shape is deliberate (ADR-0077): sermons
from one church share a room, microphones, and often an intro/outro bed, so an
averaged spectrum is a weak discriminator between two distinct talks recorded
the same way. A temporal sequence keyed to *when* things happen in the
recording discriminates far better — two different sermons diverge in their
moment-to-moment energy pattern almost immediately, while the same recording,
even re-encoded, preserves it.

Built entirely on ``numpy`` and ``faster_whisper.audio.decode_audio`` — both
already dependencies of this pipeline, so this adds none (ADR-0077,
Alternative A, rejects Chromaprint for exactly that reason).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

# faster-whisper's own default decode rate; reusing it means a fingerprint's window
# count is directly comparable to another fingerprint's without a resample step.
SAMPLE_RATE = 16000

# Coarser than Whisper's own frame size on purpose: this only needs to resolve
# "does the talk's content change here," not model phonemes. ~10s balances
# temporal resolution (catches a truncated clip within one window's slop) against
# the up-front FFT cost.
_WINDOW_SECONDS = 10.0
_WINDOW_SAMPLES = int(_WINDOW_SECONDS * SAMPLE_RATE)

# Log-spaced spectral bands per window. Coarse on purpose — a rough shape
# discriminates speech content plenty well and stays robust to a re-encode's
# quantization noise, which a fine per-bin FFT comparison would be sensitive to.
_BANDS = 16
_QUANT_MAX = 255

# Two fingerprints must agree on window count within this many windows (~10-20s of
# runtime) before content is compared at all — anything wider is a different-length
# recording, not the same one clipped or padded (see the truncated-clip test).
_WINDOW_COUNT_TOLERANCE = 1

# Cosine similarity above this counts as "the same recording." High on purpose
# (ADR-0077): favors missing a real duplicate (still caught later by the existing
# transcript-hash guard) over flagging two distinct sermons as one.
DEFAULT_SIMILARITY_THRESHOLD = 0.92


def _band_energies(window: np.ndarray) -> np.ndarray:
    """The log-spaced spectral-band energy vector for one time window, normalized.

    Normalizing each window's vector to unit length (L2) makes the fingerprint
    compare *shape*, not loudness — a re-encode's bitrate/loudness-normalization
    changes overall level without changing where the energy sits across the
    spectrum. A silent window normalizes to the zero vector rather than raising.
    """
    spectrum = np.abs(np.fft.rfft(window))
    # Log-spaced band edges across the available frequency bins (bin 0 excluded —
    # DC offset carries no spectral shape).
    n_bins = len(spectrum)
    edges = np.unique(np.geomspace(1, n_bins, num=_BANDS + 1).astype(int).clip(1, n_bins))
    if len(edges) < 2:
        edges = np.array([1, n_bins])
    energies = np.zeros(_BANDS, dtype=np.float64)
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        energies[i] = spectrum[lo:hi].sum() if hi > lo else 0.0
    norm = np.linalg.norm(energies)
    if norm > 0:
        energies = energies / norm
    return energies


def fingerprint_from_samples(samples: np.ndarray, *, sample_rate: int = SAMPLE_RATE) -> str:
    """Compute a fingerprint from already-decoded mono float samples.

    Returns a comma-separated string of quantized band-energy values, ``_BANDS``
    per window — a plain string so it round-trips through the JSON ledger like any
    other field. Resampling to a rate other than :data:`SAMPLE_RATE` is the
    caller's job (:func:`compute_fingerprint` does it via ``decode_audio``); this
    function trusts ``sample_rate`` for window sizing only.
    """
    window_samples = int(_WINDOW_SECONDS * sample_rate)
    if window_samples <= 0:
        window_samples = len(samples) or 1
    n_windows = max(1, int(np.ceil(len(samples) / window_samples)))
    values: list[int] = []
    for i in range(n_windows):
        start = i * window_samples
        end = min(start + window_samples, len(samples))
        window = samples[start:end]
        if len(window) == 0:
            energies = np.zeros(_BANDS, dtype=np.float64)
        else:
            energies = _band_energies(window)
        quantized = np.clip(np.round(energies * _QUANT_MAX), 0, _QUANT_MAX).astype(int)
        values.extend(int(v) for v in quantized)
    return ",".join(str(v) for v in values)


def compute_fingerprint(audio_path: Path) -> str:
    """Decode ``audio_path`` and compute its fingerprint (the file-facing entry point).

    Lazily imports ``faster_whisper`` for the same reason :mod:`sermon_notes.transcribe`
    does: unit tests exercising :func:`fingerprint_from_samples` never need model
    weights or a decoder on the import path.
    """
    from faster_whisper.audio import decode_audio

    samples = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
    return fingerprint_from_samples(samples, sample_rate=SAMPLE_RATE)


def _parse(fingerprint: str) -> np.ndarray:
    """Parse a fingerprint string back into a ``(n_windows, _BANDS)`` array."""
    values = [int(v) for v in fingerprint.split(",")] if fingerprint else []
    n_windows = len(values) // _BANDS
    return np.array(values[: n_windows * _BANDS], dtype=np.float64).reshape(n_windows, _BANDS)


def fingerprint_similarity(a: str, b: str) -> float:
    """Mean per-window cosine similarity between two fingerprints, in ``[0, 1]``.

    Returns ``0.0`` outright when the window counts disagree by more than
    :data:`_WINDOW_COUNT_TOLERANCE` — a duration mismatch that large means these are
    different-length recordings, not the same one clipped or padded, and comparing
    only their overlapping prefix would misreport a truncated clip as a match.
    """
    matrix_a, matrix_b = _parse(a), _parse(b)
    if abs(len(matrix_a) - len(matrix_b)) > _WINDOW_COUNT_TOLERANCE:
        return 0.0
    n = min(len(matrix_a), len(matrix_b))
    if n == 0:
        return 1.0 if len(matrix_a) == len(matrix_b) else 0.0
    matrix_a, matrix_b = matrix_a[:n], matrix_b[:n]
    norms_a = np.linalg.norm(matrix_a, axis=1)
    norms_b = np.linalg.norm(matrix_b, axis=1)
    dot = np.einsum("ij,ij->i", matrix_a, matrix_b)
    denom = norms_a * norms_b
    # Both-silent windows (denom == 0) agree by definition; only one silent is
    # maximally different — encode both without dividing by zero.
    per_window = np.where(denom > 0, dot / np.where(denom > 0, denom, 1), 1.0)
    per_window = np.where((norms_a == 0) != (norms_b == 0), 0.0, per_window)
    return float(np.clip(per_window.mean(), 0.0, 1.0))


def is_same_audio(a: str, b: str, *, threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> bool:
    """Whether two fingerprints likely describe the same recording."""
    return fingerprint_similarity(a, b) >= threshold
