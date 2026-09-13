"""The generated-note JSON sidecar: derive its path, write it, read it back (spec 0009).

The GENERATE stage validates an LLM note (PRD §8) and persists it next to the
rendered artifact as ``notes/YYYY/YYYY-MM-DD_<slug>.json``. That sidecar decouples an
expensive, quality-sensitive generation from a repeatable docx render: the renderer
reads the persisted JSON rather than holding a note in memory, so a ``.docx`` can be
rebuilt without another LLM call.

This module owns the path derivation (so the ``.json`` stem can never drift from the
``.docx`` stem — both go through :func:`note_artifact_path`) and the deterministic,
atomic (de)serialization. The note schema validator lives in
:mod:`sermon_notes.generate`; it is imported lazily here to avoid a circular
import, since generation writes through this module.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from sermon_notes.registry import SermonRecord
from sermon_notes.slug import slugify


def note_artifact_path(notes_dir: Path, sermon: SermonRecord, suffix: str) -> Path:
    """Compute ``notes/<source>/YYYY/YYYY-MM-DD_<slug><suffix>`` for ``sermon`` (PRD §7.3).

    The single owner of the dated, slugged stem. The renderer's ``.docx`` and the
    generator's ``.json`` both derive their path here, so the two artifacts always
    share a stem (spec 0009). The leading ``<source>`` segment namespaces each church
    so two sources never collide on a shared date and title (ADR-0012).

    Once a record has been published, its stem comes from the ``artifact_path`` the
    ledger recorded rather than from the title. Feeds edit episode titles after the
    fact — two Menlo sermons have been renamed since publication — and re-deriving the
    stem from the current title would address a file that was never written, orphaning
    the published artifact beside it. Only the stem is taken; the directory is still
    computed, so an injected ``notes_dir`` and the source namespace still hold.
    """
    year = sermon.published_on[:4]
    if sermon.artifact_path:
        stem = Path(sermon.artifact_path).stem
    else:
        stem = f"{sermon.published_on}_{slugify(sermon.title)}"
    return notes_dir / sermon.source / year / f"{stem}{suffix}"


def note_json_path(notes_dir: Path, sermon: SermonRecord) -> Path:
    """The ``.json`` sidecar path for ``sermon`` (beside its ``.docx``)."""
    return note_artifact_path(notes_dir, sermon, ".json")


def write_note_json(note: dict[str, Any], path: Path) -> None:
    """Atomically write ``note`` to ``path`` with a deterministic, git-friendly shape.

    Pretty-printed with sorted keys and a trailing newline so equivalent notes
    produce byte-identical files (a stable diff). Written to a ``.tmp`` sibling then
    ``os.replace``-d, matching the registry's crash-safe write. An OS-level write
    failure surfaces as :class:`~sermon_notes.generate.NoteArtifactError` so the
    generation stage applies its existing failure posture. *Any* in-process failure
    cleans up the ``.tmp`` sibling rather than leaving it for ``git add notes/`` to
    pick up (#272) — matching :func:`~sermon_notes.registry.Registry.save` and
    :func:`~sermon_notes.transcribe._write_cache_atomic`, which catch the same breadth.
    """
    text = json.dumps(note, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        from sermon_notes.generate import NoteArtifactError

        raise NoteArtifactError(f"could not write note JSON to {path}: {exc}") from exc
    except BaseException:
        # The same cleanup for an interruption that is not an OSError — a cancelled
        # Actions job arrives as KeyboardInterrupt/SystemExit, and it is exactly the
        # kill window the sibling writers name. Re-raised untouched: only an OS-level
        # write failure is the generation stage's NoteArtifactError (#272).
        tmp.unlink(missing_ok=True)
        raise


def read_note_json(path: Path) -> dict[str, Any]:
    """Read and validate the persisted note at ``path`` (spec 0009).

    A missing sidecar raises :class:`FileNotFoundError` so the render stage can skip
    it without an LLM fallback. Malformed JSON or a note that fails the PRD §8 schema
    raises :class:`~sermon_notes.generate.SchemaValidationError`.
    """
    from sermon_notes.generate import SchemaValidationError, validate_note

    text = path.read_text(encoding="utf-8")
    try:
        note = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(f"{path} was not valid JSON: {exc}") from exc
    if not isinstance(note, dict):
        raise SchemaValidationError(f"{path} JSON was {type(note).__name__}, expected object")
    validate_note(note)
    return note
