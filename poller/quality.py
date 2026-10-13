"""Score a transcript for the decoding failures Whisper ships silently (issue 98).

Pure text detectors, no model call: repetition loops, Whisper's stock credit lines, and the
hotwords list read back as speech. A score holds only counts, so it is safe to log and to
record on the public ledger; the text itself never leaves memory (ADR-0004).

Definitions match the ADR-0022 benchmark, so a gate built on them refuses what that
benchmark measured.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass

_WORD_RE = re.compile(r"[a-z0-9']+")
# An attribution line never belongs in a sermon, wherever it lands.
_ATTRIBUTION_RE = re.compile(
    r"amara\.org|subtitles by|transcribed by|transcription by|captions? by",
    re.IGNORECASE,
)
# "Thanks for watching" can be speech ("she goes, OK, thanks for watching, Dad"); Whisper's
# stock outro is the phrase as a sentence on its own, so only that counts.
_OUTRO_RE = re.compile(
    r"(?:^|(?<=[.!?])\s+)thanks? (?:you )?for watching\s*(?:[.!?]+|$)",
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

# Each flag's count, in flag order.
_FLAG_COUNTS = (("loops", "loop_words"), ("credits", "credit_hits"), ("hotword_leak", "hotword_pairs"))
# How unlikely a rise in a count must be, if both transcripts produced it at the same rate per
# word, before a replacement counts as worse (ADR-0022 rule 6, amended).
WORSE_ALPHA = 0.01


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
    """Matches of Whisper's training-data credit strings ("Subtitles by the Amara.org community").

    An attribution counts anywhere; "thanks for watching" only as a sentence of its own.
    """
    return len(_ATTRIBUTION_RE.findall(text)) + len(_OUTRO_RE.findall(text))


def _hotword_pair_strings(hotwords: str | None) -> list[str]:
    if not hotwords:
        return []
    terms = [t.strip() for t in hotwords.split(",") if t.strip()]
    return [f"{a}, {b}".lower() for a, b in itertools.pairwise(terms)]


def hotword_pairs(text: str, hotwords: str | None) -> int:
    """Occurrences of two adjacent hotword terms emitted as ``"A, B"`` — the list read back as speech."""
    lowered = text.lower()
    return sum(lowered.count(pair) for pair in _hotword_pair_strings(hotwords))


def score(text: str, *, hotwords: str | None) -> QualityScore:
    """Every detector over ``text``; ``hotwords`` is the exact string sent to the model, if any."""
    return QualityScore(
        words=len(_words(text)),
        loop_words=loop_words(text),
        credit_hits=credit_hits(text),
        hotword_pairs=hotword_pairs(text, hotwords),
    )


def clean_words(text: str, *, hotwords: str | None) -> int:
    """Words left once credit lines, hotword pairs and loop repeats are taken out.

    The speech a transcript actually carries, which ADR-0022's re-transcription gate
    compares between the old and new text. Junk is removed before loops are counted so
    a looping credit line isn't subtracted twice.
    """
    residue = _OUTRO_RE.sub(" ", _ATTRIBUTION_RE.sub(" ", text.lower()))
    for pair in _hotword_pair_strings(hotwords):
        residue = residue.replace(pair, " ")
    return len(_words(residue)) - loop_words(residue)


def flags(score: QualityScore) -> tuple[str, ...]:
    """The names of the flags ``score`` trips, in a fixed order; ``()`` when it is clean."""
    tripped = (
        ("loops", score.loop_words >= LOOP_WORDS_FLAG),
        ("credits", score.credit_hits >= CREDIT_HITS_FLAG),
        ("hotword_leak", score.hotword_pairs >= HOTWORD_PAIRS_FLAG),
    )
    return tuple(name for name, hit in tripped if hit)


def _rises_beyond_chance(old_count: int, old_words: int, new_count: int, new_words: int) -> bool:
    """Whether ``new_count`` is too high to be the old rate per word plus chance.

    The conditional test for two Poisson rates: given both counts' total, under equal rates
    the new count is binomial with the new text's share of the words. The rise is beyond
    chance when its upper tail falls under :data:`WORSE_ALPHA`.
    """
    total = old_count + new_count
    if new_count == 0 or new_words == 0 or old_words == 0:
        return False
    share = new_words / (old_words + new_words)
    log_share, log_rest = math.log(share), math.log1p(-share)
    log_choose_total = math.lgamma(total + 1)
    tail = sum(
        math.exp(
            log_choose_total
            - math.lgamma(k + 1)
            - math.lgamma(total - k + 1)
            + k * log_share
            + (total - k) * log_rest
        )
        for k in range(new_count, total + 1)
    )
    return tail < WORSE_ALPHA


def worse_flags(old: QualityScore, new: QualityScore) -> tuple[str, ...]:
    """The flags ``new`` trips on a count that rose beyond chance over ``old``'s, in flag order.

    Empty when ``new`` is clean, or no worse than ``old`` past what chance explains. ADR-0022's
    re-transcription gate refuses a replacement on any name returned here.
    """
    tripped = flags(new)
    return tuple(
        name
        for name, count in _FLAG_COUNTS
        if name in tripped
        and _rises_beyond_chance(getattr(old, count), old.words, getattr(new, count), new.words)
    )
