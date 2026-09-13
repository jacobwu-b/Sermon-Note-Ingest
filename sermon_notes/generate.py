"""Single-pass note generation: transcript → validated three-section JSON.

The GENERATE stage (PRD §6.4, ADR-0004, spec 0005): for one ``transcribed`` sermon
it builds the prompt from the sermon's feed metadata and cached transcript, makes a
single Anthropic call through the :mod:`sermon_notes.llm_client` boundary,
computes ``llm_cost_usd`` from the reported token usage (PRD §12) and appends a run
record carrying those numbers the moment the call returns (PRD §6.2, ADR-0035), then
parses and validates the response against the fixed at_a_glance/
exposition/devotional schema (PRD §8).

Scope is U5 only. This module does not advance ``transcribed → generated`` — that
wiring lives in the orchestrator (U10). The mechanical quote gate is retired
(ADR-0008). A schema-invalid response raises so U10 can apply retry policy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, NamedTuple, Sequence

from sermon_notes import erv, llm_client, prompts
from sermon_notes.artifacts import note_artifact_path, note_json_path, write_note_json
from sermon_notes.llm_client import LLMResponse
from sermon_notes.logging import get_logger
from sermon_notes.registry import Registry, RunRecord, UnknownSermonError
from sermon_notes.render import DEFAULT_NOTES_DIR
from sermon_notes.scripture import parse_reference
from sermon_notes.transcribe import DEFAULT_TRANSCRIPTS_DIR, cache_path

logger = get_logger()

# Suffix for the raw-response debug sidecar written on SchemaValidationError (ADR-0054,
# #368). Shares the note JSON's stem via note_artifact_path so it can never drift.
_DEBUG_SIDECAR_SUFFIX = ".debug.txt"

# The longest passage a "Scripture engaged" entry will quote verbatim (ADR-0055). Past
# it the citation is kept and its text dropped, so the block names what the preacher
# opened without transcribing a chapter he only pointed at. Calibrated against the
# published corpus, whose longest genuinely-engaged passage is 17 verses
# (John 15:1-17) — clear of the work, tight against the failure.
MAX_QUOTED_VERSES = 20

# The note's shape as a JSON Schema, mirroring prompts.SCHEMA_INSTRUCTION's example
# object field-for-field (ADR-0083, #542). Sent to the Anthropic API via
# ``output_config.format`` so a malformed or truncated-mid-string reply is structurally
# unreachable rather than caught after billing by ``_parse_json``/``validate_note``.
# ``additionalProperties: False`` mirrors SCHEMA_INSTRUCTION's "do not add, rename, or
# omit keys" instruction — this is the API enforcing the same rule the prompt already
# states. Kept independent of ``validate_note`` (which stays the semantic contract the
# renderer depends on, and is forward-compatible with pre-#542 committed notes);
# ``test_note_json_schema_required_fields_are_accepted_by_validate_note`` in
# ``tests/test_generate.py`` is the contract test keeping the two from drifting apart.
NOTE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["at_a_glance", "exposition", "devotional", "formation"],
    "properties": {
        "at_a_glance": {
            "type": "object",
            "additionalProperties": False,
            "required": ["thesis", "takeaways", "pull_quote"],
            "properties": {
                "thesis": {"type": "string"},
                "takeaways": {"type": "array", "items": {"type": "string"}},
                "pull_quote": {"type": "string"},
            },
        },
        "exposition": {
            "type": "object",
            "additionalProperties": False,
            "required": ["scripture_engaged", "argument", "cross_references"],
            "properties": {
                "scripture_engaged": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["reference"],
                        "properties": {"reference": {"type": "string"}},
                    },
                },
                "argument": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["heading", "subtitle", "body", "insight"],
                        "properties": {
                            "heading": {"type": "string"},
                            "subtitle": {"type": "string"},
                            "body": {"type": "string"},
                            "insight": {"type": "string"},
                        },
                    },
                },
                "cross_references": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["reference", "usage"],
                        "properties": {
                            "reference": {"type": "string"},
                            "usage": {"type": "string"},
                        },
                    },
                },
            },
        },
        "devotional": {
            "type": "object",
            "additionalProperties": False,
            "required": ["meditation", "reflection_prompts", "closing_prayer"],
            "properties": {
                "meditation": {"type": "string"},
                "reflection_prompts": {"type": "array", "items": {"type": "string"}},
                "closing_prayer": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["source", "text"],
                    "properties": {
                        "source": {"type": "string", "enum": ["distilled", "original"]},
                        "text": {"type": "string"},
                    },
                },
            },
        },
        "formation": {
            "type": "object",
            "additionalProperties": False,
            "required": ["one_thing", "step", "anchor"],
            "properties": {
                "one_thing": {"type": "string"},
                "step": {"type": "string"},
                "anchor": {"type": "string"},
            },
        },
    },
}

# `effort` defaults to "high" when unset (ADR-0083, #542) — output (including thinking)
# is 73% of total generation spend per the issue's cost breakdown, and "medium" is
# evaluated here as the mitigation for the max_tokens-exhausted-by-thinking failure
# mode that ``output_config.format`` alone cannot close. Kept as its own dict key
# (rather than folded into a single flag) so it is independently revertible from the
# schema if Phase 1.5 calibration finds a quality regression.
NOTE_OUTPUT_CONFIG: dict[str, Any] = {
    "effort": "medium",
    "format": {"type": "json_schema", "schema": NOTE_JSON_SCHEMA},
}


def _default_llm_call(system: str, user: str) -> LLMResponse:
    """The real ``llm_call``: :func:`llm_client.call` with the note's structured output
    config applied (ADR-0083). Broken out as its own default so callers/tests that
    inject their own ``llm_call`` — the existing mocking boundary — are unaffected."""
    return llm_client.call(system, user, output_config=NOTE_OUTPUT_CONFIG)


def _default_submit_fn(requests: Sequence[llm_client.BatchRequest]) -> str:
    """The real batch ``submit_fn``: :func:`llm_client.submit_batch` with the same
    structured output config as :func:`_default_llm_call` (ADR-0083) — the batch and
    sync paths generate the same note shape."""
    return llm_client.submit_batch(requests, output_config=NOTE_OUTPUT_CONFIG)


class GenerationError(RuntimeError):
    """Base class for generation-stage failures."""


class TranscriptMissingError(GenerationError):
    """Raised when no cached transcript exists for a sermon being generated."""


class SchemaValidationError(GenerationError):
    """Raised when the model response is not JSON matching the PRD §8 schema."""


class NoteArtifactError(GenerationError):
    """Raised when the validated note JSON cannot be persisted to ``notes/`` (spec 0009)."""


@dataclass(frozen=True)
class GenerationResult:
    """Outcome of generating one sermon's note (gate and state advance are U6/U10)."""

    guid: str
    note: dict[str, Any]
    input_tokens: int
    output_tokens: int
    cost_usd: float


def _now() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


# The C0 control characters XML 1.0 forbids outright — a docx built from any of them
# raises at serialization ("All strings must be XML compatible"). Tab, newline and
# carriage return are the legal exceptions and are kept: the model uses them as real
# content (e.g. a multi-paragraph meditation) and python-docx renders them as tabs and
# line breaks. Mapped to ``None`` for :meth:`str.translate`, which then deletes them.
_XML_ILLEGAL_CONTROLS = {c: None for c in range(0x20) if c not in (0x09, 0x0A, 0x0D)}


def _strip_illegal_controls(value: Any) -> Any:
    """Recursively delete XML-incompatible control characters from parsed JSON.

    ``strict=False`` lets a raw control character into a string value; the C0 controls
    outside tab/newline/CR are illegal in the OOXML the docx renderer emits, so left in
    place they only move the terminal failure downstream from parse to render (#157).
    Deleting them here — at the boundary where the model's text enters the system —
    keeps both the persisted note JSON and the rendered docx well-formed.
    """
    if isinstance(value, str):
        return value.translate(_XML_ILLEGAL_CONTROLS)
    if isinstance(value, list):
        return [_strip_illegal_controls(item) for item in value]
    if isinstance(value, dict):
        return {key: _strip_illegal_controls(item) for key, item in value.items()}
    return value


# Bound on how many unescaped-quote repairs a single reply gets (#441). Two or three
# cover every case seen in practice (one per embedded quotation mark); this just keeps
# a pathologically malformed reply from looping instead of raising.
_MAX_QUOTE_REPAIRS = 20


def _is_escaped(text: str, index: int) -> bool:
    """True if ``text[index]`` is preceded by an odd run of backslashes (a real escape)."""
    count = 0
    i = index - 1
    while i >= 0 and text[i] == "\\":
        count += 1
        i -= 1
    return count % 2 == 1


def _repair_unescaped_quote(text: str, error_pos: int) -> str | None:
    """Escape the nearest unescaped ``"`` before ``error_pos``, or ``None`` if there isn't one.

    The schema asks the model to embed quoted material in prose (a pull quote, a
    scare-quoted phrase in the meditation) and it occasionally writes that quote as a
    literal ``"`` instead of the JSON-escaped ``\\"`` (#441). That closes the string
    early, so ``json.loads`` fails one token later with "Expecting ',' delimiter" (or
    similar) at ``error_pos`` — the character right after the wrongly-closing quote.
    The quote itself is the nearest unescaped ``"`` behind that position.
    """
    i = error_pos - 1
    while i >= 0:
        if text[i] == '"' and not _is_escaped(text, i):
            return text[:i] + "\\" + text[i:]
        i -= 1
    return None


def _parse_json(text: str) -> dict[str, Any]:
    """Parse the model's reply as a JSON object, tolerating a stray code fence.

    Parsing is non-strict so an unescaped control character the model occasionally
    emits inside a string value — a raw newline or tab where the JSON escape was
    meant — is accepted rather than failing the whole note terminally (#157). The
    parsed strings are then swept of XML-incompatible control characters so a stray
    one cannot resurface as a render-time crash (see :func:`_strip_illegal_controls`).

    A literal, unescaped ``"`` inside a string value survives neither ``strict`` mode —
    it is a real syntax error, not a lenient-mode control character — so it is repaired
    before reparsing: the nearest unescaped quote behind the error position is escaped
    and parsing retried, bounded by ``_MAX_QUOTE_REPAIRS`` (#441). Every other
    malformation still raises unchanged: a reply with no such quote to repair, or one
    that runs out of attempts, surfaces its original ``JSONDecodeError``.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        # Drop a leading ```/```json fence and its closing fence, if present.
        stripped = stripped.split("\n", 1)[-1] if "\n" in stripped else ""
        if stripped.endswith("```"):
            stripped = stripped[: -len("```")]
        stripped = stripped.strip()

    candidate = stripped
    first_exc: json.JSONDecodeError | None = None
    for _ in range(_MAX_QUOTE_REPAIRS + 1):
        try:
            parsed = json.loads(candidate, strict=False)
            break
        except json.JSONDecodeError as exc:
            if first_exc is None:
                first_exc = exc
            repaired = _repair_unescaped_quote(candidate, exc.pos)
            if repaired is None:
                raise SchemaValidationError(f"response was not valid JSON: {first_exc}") from exc
            candidate = repaired
    else:
        raise SchemaValidationError(f"response was not valid JSON: {first_exc}") from first_exc

    if not isinstance(parsed, dict):
        raise SchemaValidationError(f"response JSON was {type(parsed).__name__}, expected object")
    return {key: _strip_illegal_controls(value) for key, value in parsed.items()}


def _require(condition: bool, path: str, expected: str) -> None:
    """Raise :class:`SchemaValidationError` naming ``path`` unless ``condition`` holds."""
    if not condition:
        raise SchemaValidationError(f"{path}: expected {expected}")


def _require_obj(value: Any, path: str) -> dict[str, Any]:
    """Assert ``value`` is an object and return it (narrowing the type for callers)."""
    if not isinstance(value, dict):
        raise SchemaValidationError(f"{path}: expected an object")
    return value


def _require_str_list(value: Any, path: str) -> None:
    """Assert ``value`` is a list of strings."""
    _require(isinstance(value, list), path, "a list")
    for i, item in enumerate(value):
        _require(isinstance(item, str), f"{path}[{i}]", "a string")


def _require_obj_list(value: Any, path: str, keys: tuple[str, ...]) -> None:
    """Assert ``value`` is a list of objects each carrying string ``keys``."""
    _require(isinstance(value, list), path, "a list")
    for i, item in enumerate(value):
        _require(isinstance(item, dict), f"{path}[{i}]", "an object")
        for key in keys:
            _require(isinstance(item.get(key), str), f"{path}[{i}].{key}", "a string")


def validate_note(note: dict[str, Any]) -> None:
    """Validate ``note`` against the PRD §8 schema; raise on the first violation.

    Checks presence and types of every field the docx renderer depends on. Advisory
    metadata the renderer does not key on (``subtitle``/``insight`` on a movement,
    ``source`` on the prayer) is left to the prompt. A ``scripture_engaged`` entry
    is required to carry only its ``reference``; its ``text`` is verbatim ERV
    injected after validation (spec 0016), not part of the LLM contract. Counts
    (how many takeaways, movements, prompts) are deliberately not enforced — those
    budgets are a calibration knob, not a schema invariant.

    The ``formation`` element (#140, ADR-0020) is validated *forward-only*:
    the prompt requires it for every new note, but validation accepts a note without
    it — so the committed legacy notes that predate it stay valid and re-renderable —
    and checks its three sub-fields only when the object is present.
    """
    briefing = _require_obj(note.get("at_a_glance"), "at_a_glance")
    _require(isinstance(briefing.get("thesis"), str), "at_a_glance.thesis", "a string")
    _require_str_list(briefing.get("takeaways"), "at_a_glance.takeaways")
    _require(isinstance(briefing.get("pull_quote"), str), "at_a_glance.pull_quote", "a string")

    study = _require_obj(note.get("exposition"), "exposition")
    # Only ``reference`` is part of the LLM contract; ``text`` is derived from the
    # ERV dataset after validation (spec 0016), so it is not required on input.
    _require_obj_list(
        study.get("scripture_engaged"), "exposition.scripture_engaged", ("reference",)
    )
    _require_obj_list(study.get("argument"), "exposition.argument", ("heading", "body"))
    _require_obj_list(
        study.get("cross_references"), "exposition.cross_references", ("reference", "usage")
    )

    devotional = _require_obj(note.get("devotional"), "devotional")
    _require(isinstance(devotional.get("meditation"), str), "devotional.meditation", "a string")
    _require_str_list(devotional.get("reflection_prompts"), "devotional.reflection_prompts")
    prayer = _require_obj(devotional.get("closing_prayer"), "devotional.closing_prayer")
    _require(isinstance(prayer.get("text"), str), "devotional.closing_prayer.text", "a string")

    formation = note.get("formation")
    if formation is not None:
        formation = _require_obj(formation, "formation")
        for key in ("one_thing", "step", "anchor"):
            _require(isinstance(formation.get(key), str), f"formation.{key}", "a string")


def _quotable(resolved: erv.Passage) -> bool:
    """Whether ``resolved`` is brief enough to print its verses verbatim (ADR-0055).

    "Scripture engaged" quotes the passage the preacher opened; a citation naming more
    than :data:`MAX_QUOTED_VERSES` verses is a chapter he pointed at, not a passage he
    read, and printing it buries the note's key passage under narrative he never
    engaged. A chapter-only reference names an unbounded span (``verse_count is None``)
    and is never quotable.

    Reads ``resolved.verse_count`` rather than re-parsing the reference string, so a
    same-book cross-chapter span (ADR-0079) is measured by its true verse count —
    summed across every chapter it touches — instead of a raw verse-number
    subtraction that would be meaningless once numbers reset at a chapter boundary.
    """
    return resolved.verse_count is not None and resolved.verse_count <= MAX_QUOTED_VERSES


def _cite_without_quoting(entry: dict[str, Any], guid: str) -> None:
    """Drop an over-long entry's verse text, keeping its citation (ADR-0055).

    The reference stays exactly as resolved, so the reader still gets the citation and
    its omnibible link — the renderers already print a reference alone when its text is
    empty (spec 0016) — and the warning names the guid so a recurring over-reach is
    visible without reading the note.
    """
    logger.warning(
        "engaged passage %r for %s spans more than %d verses; citing it without quoting it",
        entry["reference"],
        guid,
        MAX_QUOTED_VERSES,
    )
    entry["text"] = []


def _segments_with_attribution(resolved: erv.Passage) -> list[dict[str, str]]:
    """A resolved passage's verse segments as note-JSON dicts (ADR-0070).

    The " (ERV)" attribution (ADR-0027) lands on the last segment's text only —
    the excerpt is attributed once, where it ends, not once per verse.
    """
    segments = [{"number": s.number, "text": s.text} for s in resolved.segments]
    if segments:
        segments[-1]["text"] += " (ERV)"
    return segments


def _inject_scripture_text(note: dict[str, Any], guid: str) -> None:
    """Populate each ``scripture_engaged`` entry's ``text`` with verbatim ERV.

    The LLM emits only the ``reference``; the passage text is the vendored ERV for
    that reference (spec 0016, ADR-0021), not the transcript. A reference the ERV
    dataset cannot resolve gets an empty ``text`` and a logged warning — the agreed
    signal to revisit if it recurs — never a transcript-derived fallback.

    The resolved reference is written back too. Where the ERV renders several verses
    as one block, the model's citation of a single verse would otherwise introduce
    text it does not name (ADR-0026); widening it here fixes the docx and PDF
    scripture blocks, their omnibible links, and the title-block scripture line at
    once, since all three read this field. References carrying no verse text —
    cross-references, the Formation anchor — keep the model's wording.

    A citation can also widen because its text ran past a verse boundary mid-sentence
    with no merged-block signal to catch it (ADR-0028); if that widen stops short of a
    sentence boundary (chapter end or the widen cap), a second warning names the
    guid and both references so the outlier gets a human look.

    A resolved quotation carries a trailing " (ERV)" attribution (ADR-0027), the
    publisher's required initials for non-saleable media, rendered wherever this
    field is: docx, PDF, note JSON, web feed. An unresolved reference's text stays
    empty — there is no quotation to attribute.

    Section-marker rounding and forward widening can each independently resolve two
    adjacent entries to overlapping verse ranges, printing the overlap twice in the
    same note; adjacent entries whose resolved ranges overlap are merged into one
    (ADR-0029) after every entry is individually resolved here.
    """
    for entry in note["exposition"]["scripture_engaged"]:
        reference = entry["reference"]
        resolved = erv.passage(reference)
        if resolved is None:
            logger.warning("no ERV text for %s reference %r; leaving it blank", guid, reference)
            entry["text"] = []
            continue
        entry["reference"] = resolved.reference
        if not _quotable(resolved):
            _cite_without_quoting(entry, guid)
            continue
        entry["text"] = _segments_with_attribution(resolved)
        if not resolved.complete:
            logger.warning(
                "ERV sentence-boundary widen stopped incomplete for %s reference %r (resolved %r)",
                guid,
                reference,
                resolved.reference,
            )

    note["exposition"]["scripture_engaged"] = _merge_overlapping_engaged(
        note["exposition"]["scripture_engaged"], guid
    )


class _Span(NamedTuple):
    """A resolved entry's verse range, keyed by book and chapter for overlap checks."""

    book_slug: str
    chapter: int
    start: int
    end: int


def _entry_span(entry: dict[str, Any]) -> _Span | None:
    """The resolved verse range an injected entry covers, or ``None`` if it has none.

    An entry with no text (unresolvable reference, or over the ADR-0055 quotable
    cap), a chapter-only reference (no verse portion to overlap on), or a
    cross-chapter reference (ADR-0079 — its span isn't expressible as one chapter's
    verse range, and merging across a chapter boundary is out of scope for this
    same-chapter overlap check) has no span to compare.
    """
    if not entry["text"]:
        return None
    parsed = parse_reference(entry["reference"])
    if parsed is None or parsed.start is None or parsed.end_chapter is not None:
        return None
    return _Span(parsed.book_slug, parsed.chapter, parsed.start, parsed.end or parsed.start)


def _merge_overlapping_engaged(entries: list[dict[str, Any]], guid: str) -> list[dict[str, Any]]:
    """Merge adjacent ``scripture_engaged`` entries whose resolved ranges overlap (ADR-0029).

    Only immediately adjacent entries are compared — a later, non-adjacent entry
    citing the same verses elsewhere in the note is a deliberate repeat, not a
    splitting artifact. A merge cascades: combining two entries can widen the span
    far enough to overlap the next one too, so each merged entry is checked again
    against the entry that follows it.
    """
    merged: list[dict[str, Any]] = []
    spans: list[_Span | None] = []

    for entry in entries:
        span = _entry_span(entry)
        prev_span = spans[-1] if spans else None
        if (
            span is not None
            and prev_span is not None
            and (span.book_slug, span.chapter) == (prev_span.book_slug, prev_span.chapter)
            and span.start <= prev_span.end
            and span.end >= prev_span.start
        ):
            start, end = min(prev_span.start, span.start), max(prev_span.end, span.end)
            merged_reference = erv.widen_reference(merged[-1]["reference"], start, end)
            resolved = erv.passage(merged_reference)
            if resolved is None:
                logger.warning(
                    "could not merge overlapping scripture_engaged entries for %s "
                    "(%r); leaving them separate",
                    guid,
                    merged_reference,
                )
                merged.append(entry)
                spans.append(span)
                continue

            merged[-1]["reference"] = resolved.reference
            if not _quotable(resolved):
                # Two brief spans can union into one that is not (ADR-0055); the merged
                # citation is subject to the cap the same as any other.
                _cite_without_quoting(merged[-1], guid)
                spans[-1] = _entry_span(merged[-1])
                continue
            merged[-1]["text"] = _segments_with_attribution(resolved)
            if not resolved.complete:
                logger.warning(
                    "ERV sentence-boundary widen stopped incomplete for %s reference %r "
                    "(resolved %r)",
                    guid,
                    merged_reference,
                    resolved.reference,
                )
            spans[-1] = _entry_span(merged[-1])
            continue

        merged.append(entry)
        spans.append(span)

    return merged


def _write_debug_sidecar(path: Path, text: str) -> None:
    """Best-effort persist of the model's raw reply beside a would-be note (ADR-0054).

    Never raises: a write failure here must not turn a diagnosable
    :class:`SchemaValidationError` into an undiagnosable one, so an ``OSError`` is
    logged and swallowed rather than propagated.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    except OSError as exc:
        logger.warning("could not write debug sidecar to %s: %s", path, exc)


def _clear_debug_sidecar(path: Path) -> None:
    """Remove a stale debug sidecar left by a prior failed attempt (ADR-0054).

    Best-effort, matching :func:`_write_debug_sidecar`: a cleanup failure must not
    fail a generation that otherwise succeeded.
    """
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        logger.warning("could not clear debug sidecar at %s: %s", path, exc)


def _prompt_for(sermon: Any, transcript: str) -> str:
    """The user prompt for one sermon — shared by the synchronous and batch paths.

    Both paths must send the model the same thing, or a backfill's notes would differ
    from the cron's for reasons no reader could see. One builder, two callers.
    """
    return prompts.build_user_prompt(
        title=sermon.title,
        series=sermon.series,
        speaker=sermon.speaker,
        published_on=sermon.published_on,
        blurb=sermon.blurb,
        scripture_refs=sermon.scripture_refs,
        transcript=transcript,
    )


def _transcript_for(sermon: Any, guid: str, transcripts_dir: Path) -> str:
    """The cached transcript for ``sermon``, or :class:`TranscriptMissingError`."""
    cache_file = cache_path(transcripts_dir, sermon)
    if not cache_file.exists():
        raise TranscriptMissingError(f"no cached transcript for guid {guid!r} at {cache_file}")
    return cache_file.read_text(encoding="utf-8")


def submit_note_batch(
    registry: Registry,
    guid: str,
    *,
    submit_fn: Callable[[Sequence[llm_client.BatchRequest]], str] = _default_submit_fn,
    transcripts_dir: Path = DEFAULT_TRANSCRIPTS_DIR,
    now: str | None = None,
) -> str:
    """Submit one sermon's generation to the Batches API, returning the batch id (spec 0020).

    The backfill counterpart to :func:`generate_notes`: same transcript, same prompt, same
    model — only the delivery is asynchronous, at ADR-0061's discount. It returns the id
    the caller persists on the record as ``batch_id``; the note itself arrives on a later
    pipeline run, which feeds the retrieved completion back through
    :func:`generate_notes` so parsing, validation, the sidecar write and the cost ledger
    are the ones that already exist.

    The run record appended here carries no tokens, no model and no cost, because none of
    those are known yet and none has been billed: a submission is a commitment to spend,
    not a spend. The run that resolves the batch records what it actually cost, at the
    batch rate (:func:`~sermon_notes.llm_client.cost_usd` reads ``LLMResponse.billing``).

    ``custom_id`` is derived from the sermon's guid (:func:`llm_client.batch_custom_id`),
    which is how the resolving run finds this sermon's completion among the batch's
    results without a side table.
    """
    sermon = registry.get(guid)
    if sermon is None:
        raise UnknownSermonError(f"no sermon with guid {guid!r} in the ledger")

    transcript = _transcript_for(sermon, guid, transcripts_dir)
    started_at = now if now is not None else _now()
    batch_id = submit_fn(
        [
            llm_client.BatchRequest(
                custom_id=llm_client.batch_custom_id(guid),
                system=prompts.SYSTEM_PROMPT,
                user=_prompt_for(sermon, transcript),
            )
        ]
    )
    registry.append_run(
        guid,
        RunRecord(
            attempted_state="pending_batch",
            outcome="success",
            started_at=started_at,
            finished_at=now if now is not None else _now(),
        ),
    )
    logger.info("submitted %s for batch generation as %s", guid, batch_id)
    return batch_id


def generate_notes(
    registry: Registry,
    guid: str,
    *,
    llm_call: Callable[[str, str], LLMResponse] = _default_llm_call,
    transcripts_dir: Path = DEFAULT_TRANSCRIPTS_DIR,
    notes_dir: Path = DEFAULT_NOTES_DIR,
    now: str | None = None,
) -> GenerationResult:
    """Generate one sermon's structured note and record its token usage and cost.

    Reads the cached transcript for ``guid``, makes a single LLM call, validates the
    JSON against the PRD §8 schema (raising :class:`SchemaValidationError` on a bad
    shape), persists the validated note to its ``notes/YYYY/YYYY-MM-DD_<slug>.json``
    sidecar (spec 0009), and returns the validated note. The ``transcribed → generated``
    advance (U10) is wired by the orchestrator.

    The ``generated`` run record — input/output tokens, ``llm_cost_usd``, and the
    resolved ``llm_model`` that produced them, so the cost ledger is self-describing
    (PRD §6.2, #47) — is appended the moment the paid call returns, with
    ``outcome="billed"``, and amended to ``"success"`` once the note validates and its
    sidecar is written. So a successful generation still leaves exactly one run record,
    while a failure *after* the call (bad JSON, schema violation, a failed write) leaves
    the bill recorded rather than costing $0.00 in the ledger the rolling spend guard
    sums (#263, ADR-0035). The terminal record beside it is the orchestrator's
    (``_generate_stage``); the two facts — what the call cost, and that it produced no
    note — are recorded separately. The run also carries ``llm_stop_reason`` verbatim off
    the SDK response, so a truncated completion is a recorded fact rather than an
    inference (ADR-0054, #368); and ``llm_billing``/``llm_batch_id``, so a resolved run
    names the rate and batch that actually billed it (ADR-0082, #543). ``llm_batch_id``
    is read off ``sermon.batch_id``, which is still set here for a sermon resolving out
    of ``pending_batch`` — the merge only clears it once the record leaves that state —
    so no plumbing beyond the record already in scope is needed to carry it through.

    On :class:`SchemaValidationError` (bad JSON or a schema violation), the model's raw
    reply is persisted to a ``.debug.txt`` sidecar beside where the note JSON would have
    gone, so a failure is inspectable after the run instead of lost with the process
    (ADR-0054, #368); a stale sidecar from a prior failure is cleared on the next success.
    """
    sermon = registry.get(guid)
    if sermon is None:
        raise UnknownSermonError(f"no sermon with guid {guid!r} in the ledger")

    transcript = _transcript_for(sermon, guid, transcripts_dir)

    started_at = now if now is not None else _now()
    response = llm_call(prompts.SYSTEM_PROMPT, _prompt_for(sermon, transcript))

    # The money is spent the instant that call returns, so the ledger records it here —
    # before parse, validation, scripture injection, or the sidecar write, any of which
    # can raise. Recording it with the success instead made every post-call failure cost
    # $0.00 in the ledger, which is what the rolling spend guard sums (#263, ADR-0035).
    cost = llm_client.cost_usd(response)
    run = RunRecord(
        attempted_state="generated",
        outcome="billed",
        llm_input_tokens=response.input_tokens,
        llm_output_tokens=response.output_tokens,
        llm_cost_usd=cost,
        llm_model=response.model,
        llm_stop_reason=response.stop_reason,
        llm_billing=response.billing,
        llm_batch_id=sermon.batch_id,
        started_at=started_at,
        finished_at=now if now is not None else _now(),
    )
    registry.append_run(guid, run)

    debug_path = note_artifact_path(notes_dir, sermon, _DEBUG_SIDECAR_SUFFIX)
    try:
        note = _parse_json(response.text)
        validate_note(note)
    except SchemaValidationError:
        # The raw reply is otherwise lost the moment this process exits — only the
        # JSONDecodeError/schema message survives in run.error_detail, which isn't
        # enough to tell a truncated (stop_reason == "max_tokens") response apart from
        # any other malformed shape (ADR-0054, #368). Best-effort: a write failure here
        # must not mask the SchemaValidationError that triggered it.
        _write_debug_sidecar(debug_path, response.text)
        raise
    _inject_scripture_text(note, guid)
    write_note_json(note, note_json_path(notes_dir, sermon))

    # A stale sidecar from a prior failed attempt on this sermon no longer applies to a
    # note that just validated and was persisted — clear it so it doesn't sit beside a
    # published note implying a failure that isn't current (ADR-0054).
    _clear_debug_sidecar(debug_path)

    # The note is validated and persisted, so the billed run becomes the success run —
    # amended, not appended, so a successful generation still leaves exactly one record.
    run.outcome = "success"
    run.finished_at = now if now is not None else _now()
    logger.info(
        "generated note for %s (in=%d out=%d tokens, $%.4f)",
        guid,
        response.input_tokens,
        response.output_tokens,
        cost,
    )
    return GenerationResult(
        guid=guid,
        note=note,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        cost_usd=cost,
    )
