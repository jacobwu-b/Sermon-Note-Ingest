"""Score a transcript for the decoding failures Whisper ships silently (issue 98).

Pure text detectors, no model call: repetition loops, Whisper's stock credit lines, and the
hotwords list read back as speech. A score holds only counts, so it is safe to log and to
record on the public ledger; the text itself never leaves memory (ADR-0004).

Definitions match the ADR-0022 benchmark, so a gate built on them refuses what that
benchmark measured.
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"[a-z0-9']+")
_CREDIT_RE = re.compile(
    r"amara\.org|subtitles by|thanks? (?:you )?for watching|transcribed by|transcription by|captions? by",
    re.IGNORECASE,
)
# Under four words a back-to-back repeat is usually the congregation ("Amen. Amen. Amen."),
# not the decoder; past forty, no observed loop.
_MIN_LOOP_PHRASE, _MAX_LOOP_PHRASE = 4, 40
_MIN_LOOP_REPEATS = 3

# A transcript is flagged at these counts (ADR-0022 rule 6, ADR-0023). The only copy: the
# alert, the status file and the re-transcription gate all go through :func:`flags`.
LOOP_WORDS_FLAG = 40
CREDIT_HITS_FLAG = 1
HOTWORD_PAIRS_FLAG = 1


@dataclass(frozen=True)
class QualityScore:
    words: int
    loop_words: int
    credit_hits: int
    hotword_pairs: int


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def loop_words(text: str) -> int:
    """Words spent repeating a 4–40-word phrase 3+ times back to back.

    Counts only the repeats beyond the first occurrence — the words a loop added — and
    counts each word once however many phrase lengths it repeats under.
    """
    words = _words(text)
    flagged = bytearray(len(words))
    for n in range(_MIN_LOOP_PHRASE, _MAX_LOOP_PHRASE + 1):
        i = 0
        while i + _MIN_LOOP_REPEATS * n <= len(words):
            reps = 1
            while words[i + reps * n : i + (reps + 1) * n] == words[i : i + n]:
                reps += 1
            if reps >= _MIN_LOOP_REPEATS:
                flagged[i + n : i + reps * n] = b"\x01" * ((reps - 1) * n)
                i += reps * n
            else:
                i += 1
    return sum(flagged)


def credit_hits(text: str) -> int:
    """Matches of Whisper's training-data credit strings ("Subtitles by the Amara.org community")."""
    return len(_CREDIT_RE.findall(text))


def hotword_pairs(text: str, hotwords: str | None) -> int:
    """Occurrences of two adjacent hotword terms emitted as ``"A, B"`` — the list read back as speech."""
    if not hotwords:
        return 0
    terms = [t.strip() for t in hotwords.split(",") if t.strip()]
    lowered = text.lower()
    return sum(lowered.count(f"{a}, {b}".lower()) for a, b in itertools.pairwise(terms))


def score(text: str, *, hotwords: str | None) -> QualityScore:
    """Every detector over ``text``; ``hotwords`` is the exact string sent to the model, if any."""
    return QualityScore(
        words=len(_words(text)),
        loop_words=loop_words(text),
        credit_hits=credit_hits(text),
        hotword_pairs=hotword_pairs(text, hotwords),
    )


def flags(score: QualityScore) -> tuple[str, ...]:
    """The names of the flags ``score`` trips, in a fixed order; ``()`` when it is clean."""
    tripped = (
        ("loops", score.loop_words >= LOOP_WORDS_FLAG),
        ("credits", score.credit_hits >= CREDIT_HITS_FLAG),
        ("hotword_leak", score.hotword_pairs >= HOTWORD_PAIRS_FLAG),
    )
    return tuple(name for name, hit in tripped if hit)
