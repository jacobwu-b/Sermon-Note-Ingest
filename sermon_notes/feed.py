"""Render the secret-free content feed the website consumes (spec 0014, ADR-0018, U3).

The feed also carries the pipeline's already-rendered PDF (ADR-0025) as a binary
asset per sermon, at `sermons/<source>/<date>_<slug>.pdf`, so the website can link
to it directly instead of generating its own (ADR-0068). The PDF render is
best-effort (ADR-0025), so its absence is not an error: the manifest and
per-sermon `pdf_path` field carry `null` for a sermon whose PDF did not render.

The feed is a *pure projection* of the existing registry + persisted note JSON
(spec 0009) into a versioned, reader-facing tree — never a new persistent schema and
never new direct ledger access (the sermons come through :mod:`sermon_notes.registry`).
The tree written to the output directory is::

    README.md                                # self-describing header for the content repo
    schema_version.json                      # {"schema_version": N} — the root marker
    churches.json                            # tenant identity per source in the feed
    manifest.json                            # reverse-chron metadata index, filterable by source
    sermons/<source>/<date>_<slug>.json      # reader-facing note content, omnibible URL per ref

Two boundary guarantees are load-bearing:

- **Allow-list projection.** Every reader-facing field is named explicitly; the
  registry's per-run telemetry (transcript hash, token counts, USD cost, model id,
  error classes, the audio CDN url) is never copied, so a future internal field cannot
  leak by default (spec 0014 strip list).
- **Determinism.** Every file is pretty-printed with sorted keys and a trailing
  newline and written atomically, so an unchanged ledger renders byte-identically and
  the content-repo diff is meaningful.

Only published sermons that carry a persisted note JSON are included; the older
``.docx``-only archive predates spec 0009 and has no reader-facing content to render.
This module makes **no external calls** — pushing the tree and firing the deploy hook
are separate boundaries (U4). It is therefore exercised entirely on disk.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sermon_notes import config, erv
from sermon_notes.artifacts import note_artifact_path, read_note_json
from sermon_notes.logging import get_logger
from sermon_notes.registry import (
    DEFAULT_REGISTRY_PATH,
    Registry,
    SermonRecord,
    church_name,
)
from sermon_notes.render import DEFAULT_NOTES_DIR

logger = get_logger()

# Bumped only on a backward-incompatible feed change so the website can detect and
# refuse a mismatched feed (spec 0014, ADR-0018). Independent of the registry schema.
# 2: sermons/ can now carry a binary .pdf beside the .json (ADR-0068) — a naive
# consumer that assumes every file under sermons/ parses as JSON must refuse rather
# than mishandle it silently.
# 3: per-sermon files carry `formation` (ADR-0071), and every feed file's key order
# now follows each file's own reading order instead of an alphabetical sort — a
# tree-wide byte-layout change even where the JSON shape is otherwise additive.
SCHEMA_VERSION = 3

# Repo root is two parents above this package; the feed renders here by default and is
# pushed to ``sermon-notes-content`` in U4. Overridable via ``CONTENT_FEED_DIR``.
_REPO_ROOT = Path(__file__).resolve().parents[2]

# Per-church provenance/attribution carried in churches.json. Church-level (not
# per-episode), so it states how the notes were produced without a sermon link.
_PROVENANCE = (
    "Study notes generated from the {church} sermon podcast, "
    "based on the published audio transcript."
)

# README rendered into the feed root so the content repo is self-describing: the
# tree is machine-published and replaced wholesale each publish, so this doc must be
# part of the render to survive. Static except the schema version, keeping the feed
# byte-deterministic. ``{schema_version}`` is the only interpolation.
_README_TEMPLATE = """\
# Sermon Note — content feed

A secret-free, versioned JSON projection of AI-generated sermon study notes. Every
file here is machine-published by the
[Sermon Note pipeline](https://github.com/jacobwu-b/Sermon-Note-Pipeline) — the
source of truth — and consumed at build time by the Sermon Note website.

**Do not hand-edit; the tree is overwritten on each publish.** The pipeline
replaces this repository's working tree wholesale on every publish, so any manual
change is discarded on the next run.

## Layout

| Path | Contents |
| --- | --- |
| `schema_version.json` | Feed contract version — the website refuses a feed whose version it does not recognize. |
| `churches.json` | Per-source identity: slug, display name, and provenance. |
| `manifest.json` | Every published sermon as a metadata entry, newest first. |
| `sermons/<source>/<date>_<slug>.json` | Reader-facing note content for one sermon. |

Feed schema version: **{schema_version}**
"""


@dataclass(frozen=True)
class FeedRenderResult:
    """Outcome of a feed render: how many sermons and which sources appear."""

    sermons: int
    sources: list[str]


def default_feed_dir() -> Path:
    """The feed output directory: ``CONTENT_FEED_DIR`` if set, else ``<repo>/feed``."""
    override = config.get("CONTENT_FEED_DIR", None)
    return Path(override) if override else _REPO_ROOT / "feed"


def sermon_slug(sermon: SermonRecord) -> str:
    """The sermon's stable slug, taken from the docx artifact stem.

    Reusing the artifact stem (rather than re-slugging the live title) keeps the feed
    aligned with the persisted ``.json`` sidecar and stable across a later title edit
    — the title can change after render, the artifact path does not. Public: it is
    also how :mod:`pipeline` builds the website note-page URL Google Chat links
    (spec 0024 amendment), so both consumers derive the exact same slug the website's
    own manifest carries — never a second, possibly-diverging derivation.
    """
    assert sermon.artifact_path is not None  # guaranteed by _publishable
    stem = Path(sermon.artifact_path).stem  # "YYYY-MM-DD_<slug>"
    return stem.removeprefix(f"{sermon.published_on}_")


def _note_path(notes_dir: Path, sermon: SermonRecord) -> Path:
    """The persisted note-JSON sidecar for ``sermon`` (spec 0009), by artifact stem.

    Only called where ``artifact_path`` is set: inside the :func:`_publishable` filter
    it is guarded by the preceding ``s.artifact_path`` truthiness check, and elsewhere
    only over already-filtered sermons.
    """
    assert sermon.artifact_path is not None
    stem = Path(sermon.artifact_path).stem
    return notes_dir / sermon.source / sermon.published_on[:4] / f"{stem}.json"


def _publishable(registry: Registry, notes_dir: Path) -> list[SermonRecord]:
    """Published sermons that have an artifact and a persisted note JSON, newest first.

    Sorted reverse-chronologically, ties broken by ``guid`` for a stable order — the
    same ordering the README index uses (spec 0006).
    """
    ready = [
        s
        for s in registry.sermons()
        if s.state == "published" and s.artifact_path and _note_path(notes_dir, s).exists()
    ]
    ready.sort(key=lambda s: (s.published_on, s.guid), reverse=True)
    return ready


def _with_url(ref: dict[str, Any], *fields: str) -> dict[str, Any]:
    """Project ``ref`` to ``reference`` + ``fields`` and embed its omnibible URL.

    A same-book cross-chapter reference gets a best-effort link to its leading
    chapter, truncated at that chapter's own last verse (spec 0016, ADR-0079).
    A reference the URL contract still cannot represent at all (unrecognized
    text) carries ``url: None`` — the same plain-text fallback the ``.docx`` uses.
    """
    projected = {"reference": ref["reference"]}
    for field in fields:
        projected[field] = ref[field]
    projected["url"] = erv.reference_url(ref["reference"])
    return projected


def _formation(note: dict[str, Any]) -> dict[str, Any] | None:
    """Project the note's ``formation`` section (ADR-0071), or ``None`` if it has none.

    Spec 0005's issue-#140 addendum validates ``formation`` forward-only: older notes
    carry no such object at all, so this returns ``None`` rather than a stub — the same
    present-key-null-value convention ``episode_url``/``pdf_path`` use, so the website
    tests one condition instead of an absent key. When present, ``one_thing``/``step``
    default to ``""`` and ``anchor`` to ``None`` for a sub-field the note itself omits,
    the same defensive ``.get()`` posture :func:`sermon_notes.render_pdf._formation`
    already takes. ``anchor`` carries its omnibible URL through :func:`_with_url`, the
    same projection ``scripture_engaged``/``cross_references`` entries use.
    """
    formation = note.get("formation")
    if not formation:
        return None
    anchor = formation.get("anchor", "")
    return {
        "one_thing": formation.get("one_thing", ""),
        "step": formation.get("step", ""),
        "anchor": _with_url({"reference": anchor}) if anchor else None,
    }


def _pdf_feed_path(sermon: SermonRecord, notes_dir: Path) -> str | None:
    """The feed-relative path to ``sermon``'s pipeline-rendered PDF, or ``None``.

    ``None`` when the ADR-0025 render did not produce a file for this sermon — it is
    best-effort, so a sermon can carry full note content and still have no PDF. Uses
    the same artifact-stem derivation as the ``.docx``/``.json`` (ADR-0068), never a
    new direct ledger access.
    """
    if not note_artifact_path(notes_dir, sermon, ".pdf").exists():
        return None
    return f"sermons/{sermon.source}/{sermon.published_on}_{sermon_slug(sermon)}.pdf"


def _project_sermon(
    sermon: SermonRecord, note: dict[str, Any], pdf_path: str | None
) -> dict[str, Any]:
    """Project a sermon + its note JSON to the reader-facing per-sermon feed shape.

    An explicit allow-list: only the at-a-glance / exposition / devotional /
    formation content and reader-facing metadata are emitted. Registry telemetry is
    never read here, so it cannot leak. Each scripture reference gains its omnibible
    URL.

    ``episode_url`` is the link back to the sermon the note was made from — the same
    value the ``.docx`` footer prints. It is ``None``, never ``""``, for a source whose
    feed publishes no episode link, so the website renders no link rather than an
    anchor pointing nowhere. The audio CDN url stays stripped either way.

    ``pdf_path`` (ADR-0068) is likewise ``None`` rather than an absent key when the
    pipeline's PDF render did not produce a file for this sermon.

    The dict is built key-by-key in the PDF's own reading order (ADR-0071): the
    masthead metadata, then ``at_a_glance``, ``exposition``, ``devotional``,
    ``formation``. ``_write_json`` no longer re-sorts this alphabetically, so the
    order below is the order the published file carries.
    """
    briefing = note["at_a_glance"]
    study = note["exposition"]
    devotional = note["devotional"]
    return {
        "source": sermon.source,
        "slug": sermon_slug(sermon),
        "date": sermon.published_on,
        "title": sermon.title,
        "speaker": sermon.speaker,
        "series": sermon.series,
        "scripture_refs": list(sermon.scripture_refs),
        "episode_url": sermon.episode_url or None,
        "pdf_path": pdf_path,
        "at_a_glance": {
            "thesis": briefing["thesis"],
            "takeaways": list(briefing["takeaways"]),
            "pull_quote": briefing["pull_quote"],
        },
        "exposition": {
            "scripture_engaged": [_with_url(entry, "text") for entry in study["scripture_engaged"]],
            "argument": [
                {
                    "heading": movement["heading"],
                    "subtitle": movement.get("subtitle", ""),
                    "body": movement["body"],
                    "insight": movement.get("insight", ""),
                }
                for movement in study["argument"]
            ],
            "cross_references": [_with_url(entry, "usage") for entry in study["cross_references"]],
        },
        "devotional": {
            "meditation": devotional["meditation"],
            "reflection_prompts": list(devotional["reflection_prompts"]),
            "closing_prayer": {"text": devotional["closing_prayer"]["text"]},
        },
        "formation": _formation(note),
    }


def _manifest_entry(sermon: SermonRecord, pdf_path: str | None) -> dict[str, Any]:
    """The reader-facing metadata entry for a sermon in ``manifest.json``.

    Carries ``episode_url`` (``None`` when the source publishes none) so a listing page
    can link each sermon to its origin without fetching every per-sermon file, and
    ``pdf_path`` (ADR-0068, ``None`` when the pipeline's PDF render did not produce a
    file) so it can offer the download without fetching the per-sermon file either.
    """
    slug = sermon_slug(sermon)
    return {
        "source": sermon.source,
        "slug": slug,
        "date": sermon.published_on,
        "title": sermon.title,
        "speaker": sermon.speaker,
        "series": sermon.series,
        "scripture_refs": list(sermon.scripture_refs),
        "episode_url": sermon.episode_url or None,
        "pdf_path": pdf_path,
        "path": f"sermons/{sermon.source}/{sermon.published_on}_{slug}.json",
    }


def _church_entry(source: str) -> dict[str, Any]:
    """The tenant-identity entry for a source in ``churches.json``."""
    display_name = church_name(source)
    return {
        "slug": source,
        "display_name": display_name,
        "provenance": _PROVENANCE.format(church=display_name),
    }


def _write_json(path: Path, payload: Any) -> None:
    """Atomically write ``payload`` as deterministic, diff-stable JSON.

    Pretty-printed with a trailing newline so equivalent renders are byte-identical;
    written to a ``.tmp`` sibling then ``os.replace``-d, matching the registry and
    note-JSON write contract (ADR-0002, spec 0009).

    Deliberately *not* ``sort_keys=True`` (ADR-0071): every ``payload`` passed here is
    a literal dict (or a list of them) built key-by-key in a fixed, meaningful order —
    the per-sermon projection's order matches the PDF's own reading order — so an
    alphabetical re-sort would only discard that order while buying nothing for
    determinism: the same code still produces the same key order on every run.
    """
    _write_text(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def _write_text(path: Path, text: str) -> None:
    """Atomically write ``text`` to ``path`` via a ``.tmp`` sibling + ``os.replace``.

    The same write contract as :func:`_write_json` (ADR-0002, spec 0009), for the
    non-JSON feed files (the README).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _copy_binary_atomic(src: Path, dest: Path) -> None:
    """Copy ``src`` to ``dest`` via a ``.tmp`` sibling + ``os.replace`` (ADR-0068).

    Same write contract as :func:`_write_json`/:func:`_write_text`, for the PDF asset
    — a partial copy must never leave a ``.pdf`` a consumer could read mid-write.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dest)


def render_feed(
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    notes_dir: Path = DEFAULT_NOTES_DIR,
    output_dir: Path | None = None,
) -> FeedRenderResult:
    """Render the content feed for every published-with-note sermon (spec 0014).

    Reads sermons through the registry module, projects each to the reader-facing
    shape, and writes the ``schema_version`` / ``churches.json`` / ``manifest.json`` /
    per-sermon tree under ``output_dir`` (default :func:`default_feed_dir`). Pure
    render to disk: no external calls, no registry mutation. Returns the count and the
    sources that appear in the feed.

    A sermon whose archived note JSON cannot be read or projected is logged and skipped
    rather than failing the whole projection (#197) — :func:`_publishable` only checks
    that the sidecar exists, so one corrupt or schema-drifted file in the committed
    archive would otherwise take down every run's feed render. The skip is per-content,
    matching how the pipeline stages isolate a poisoned sermon (ADR-0019).
    """
    output_dir = output_dir if output_dir is not None else default_feed_dir()
    registry = Registry.load(registry_path)

    # Project everything before writing anything: manifest.json and churches.json must
    # name only sermons that actually get a file, since a manifest entry the website
    # cannot resolve is a dead link (spec 0014).
    rendered: list[tuple[SermonRecord, dict[str, Any], str | None]] = []
    for candidate in _publishable(registry, notes_dir):
        try:
            note = read_note_json(_note_path(notes_dir, candidate))
            pdf_path = _pdf_feed_path(candidate, notes_dir)
            projection = _project_sermon(candidate, note, pdf_path)
        except Exception as exc:
            logger.error(
                "skipping %s in the content feed: unusable note JSON: %s: %s",
                candidate.guid,
                type(exc).__name__,
                exc,
            )
            continue
        rendered.append((candidate, projection, pdf_path))

    sermons = [sermon for sermon, _, _ in rendered]

    _write_json(output_dir / "schema_version.json", {"schema_version": SCHEMA_VERSION})
    _write_text(output_dir / "README.md", _README_TEMPLATE.format(schema_version=SCHEMA_VERSION))

    # Sources in feed order, de-duplicated, then sorted for a stable churches.json.
    sources = sorted({s.source for s in sermons})
    _write_json(
        output_dir / "churches.json",
        {"churches": [_church_entry(source) for source in sources]},
    )

    _write_json(
        output_dir / "manifest.json",
        {"sermons": [_manifest_entry(s, pdf_path) for s, _, pdf_path in rendered]},
    )

    for sermon, projection, pdf_path in rendered:
        slug = sermon_slug(sermon)
        _write_json(
            output_dir / "sermons" / sermon.source / f"{sermon.published_on}_{slug}.json",
            projection,
        )
        if pdf_path is not None:
            _copy_binary_atomic(
                note_artifact_path(notes_dir, sermon, ".pdf"), output_dir / pdf_path
            )

    logger.info(
        "rendered content feed: %d sermon(s) across %s → %s",
        len(sermons),
        ", ".join(sources) or "no sources",
        output_dir,
    )
    return FeedRenderResult(sermons=len(sermons), sources=sources)
