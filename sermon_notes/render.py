"""Render a validated note JSON into the premium three-section ``.docx`` (U7).

The RENDER half of RENDER+COMMIT (PRD §6.4). Given one generated note (the PRD §8
shape produced by U5) and its sermon metadata, this module builds the document of
PRD §7.1–7.2 — US Letter with 1" margins, serif body and sans headings, muted-blue
dividers, distinctly styled pull-quotes, per-movement insights, and scripture
blocks, a page-2+ running header, and a provenance footer — then writes it to
``notes/YYYY/YYYY-MM-DD_<slug>.docx`` (PRD §7.3). Direct quotes are embedded inline
in the argument prose (not a standalone styled block). The output is validated
programmatically; a validation failure triggers exactly one regeneration, then is
terminal (PRD §11.1).

Scope is U7 only: this module does not refresh the README index (U8), commit the
artifact, or touch the registry — the orchestrator (U10) wires those around it. The
premium fidelity itself is graded by eye in Phase 1.5; the tests assert structure
and styles, not pixels.
"""

from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any, Callable, Iterable, cast

import docx
from docx.document import Document as DocxDocument
from docx.enum.style import WD_STYLE_TYPE
from docx.opc.constants import RELATIONSHIP_TYPE as RT
from docx.enum.text import WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.oxml.xmlchemy import BaseOxmlElement
from docx.shared import Inches, Pt, RGBColor
from docx.styles.style import ParagraphStyle
from docx.text.paragraph import Paragraph

from sermon_notes import erv
from sermon_notes.logging import get_logger
from sermon_notes.note_layout import (
    GLANCE_LABEL,
    DEVOTIONAL_LABEL,
    EXPOSITION_LABEL,
    FORMATION_LABEL,
    provenance_line,
    scripture_line,
    smart_quote,
)
from sermon_notes.registry import SermonRecord, church_name

logger = get_logger()

# Output root: the repo's notes/ folder (Repository Map). Two parents above this
# package is the repo root, matching registry.DEFAULT_REGISTRY_PATH.
DEFAULT_NOTES_DIR = Path(__file__).resolve().parents[2] / "notes"

# PRD §7.2 palette and type. Fonts carry the spec's primary face; a .docx stores one
# font name per run, so the Georgia/Arial fallbacks are OS-level substitution that
# Word/previewers apply when the primary is absent — not an encodable chain.
_BODY_FONT = "Source Serif Pro"
_HEADING_FONT = "Source Sans Pro"
_MUTED_BLUE = RGBColor(0x2E, 0x75, 0xB6)
_DARK_GREY = RGBColor(0x44, 0x44, 0x44)
_TITLE_BLACK = RGBColor(0x11, 0x11, 0x11)


def _church_name(source: str) -> str:
    """The church display name for a record's ``source`` (ADR-0012).

    Resolves through the shared registry map so the header band and footer
    provenance never hardcode a church name; an unregistered source is terminal.
    """
    try:
        return church_name(source)
    except KeyError as exc:
        raise RenderError(f"no church display name registered for source {source!r}") from exc


# Custom paragraph-style names. The pull-quote / insight / scripture trio is asserted
# by the tests as the §7.2 "distinct styling" contract.
_STYLE_TITLE = "Doc Title"
_STYLE_SUBTITLE = "Doc Subtitle"
_STYLE_SECTION_LABEL = "Section Label"
_STYLE_SUBHEAD = "Subhead"
_STYLE_PULL_QUOTE = "Pull Quote"
_STYLE_INSIGHT = "Insight"
_STYLE_SCRIPTURE = "Scripture Block"

# Right-tab position for the header band / running header: content width on a US
# Letter page with 1" margins (8.5" - 2").
_CONTENT_WIDTH = Inches(6.5)


class RenderError(RuntimeError):
    """Terminal render failure: the document could not be produced and validated."""


class DocxValidationError(RenderError):
    """The written file is not a structurally valid ``.docx`` (PRD §11.1)."""


def _set_char_spacing(rpr: BaseOxmlElement, twentieths: int) -> None:
    """Apply letter-spacing (``w:spacing`` in a run-properties element)."""
    spacing = rpr.find(qn("w:spacing"))
    if spacing is None:
        spacing = OxmlElement("w:spacing")
        rpr.append(spacing)
    spacing.set(qn("w:val"), str(twentieths))


def _border(edge: str, *, sz: int, space: int) -> BaseOxmlElement:
    """Build one muted-blue border element for the named ``edge`` (PRD §7.2)."""
    element = OxmlElement(f"w:{edge}")
    element.set(qn("w:val"), "single")
    element.set(qn("w:sz"), str(sz))  # eighths of a point
    element.set(qn("w:space"), str(space))
    element.set(qn("w:color"), "2E75B6")
    return element


def _set_borders(ppr: BaseOxmlElement, borders: Iterable[BaseOxmlElement]) -> None:
    """Attach a ``w:pBdr`` carrying ``borders`` to a paragraph-properties element."""
    pbdr = ppr.find(qn("w:pBdr"))
    if pbdr is None:
        pbdr = OxmlElement("w:pBdr")
        ppr.append(pbdr)
    for border in borders:
        pbdr.append(border)


def _add_hyperlink(paragraph: Paragraph, url: str, text: str) -> None:
    """Append ``text`` to ``paragraph`` as an external hyperlink to ``url``.

    python-docx has no high-level hyperlink API, so this relates the URL on the
    paragraph's part and builds the ``w:hyperlink`` run by hand, styling it bold,
    muted-blue, and underlined to read as a link within the §7.2 palette.
    """
    r_id = paragraph.part.relate_to(url, RT.HYPERLINK, is_external=True)
    hyperlink = OxmlElement("w:hyperlink")
    hyperlink.set(qn("r:id"), r_id)

    run = OxmlElement("w:r")
    rpr = OxmlElement("w:rPr")
    rpr.append(OxmlElement("w:b"))
    color = OxmlElement("w:color")
    color.set(qn("w:val"), "2E75B6")
    rpr.append(color)
    underline = OxmlElement("w:u")
    underline.set(qn("w:val"), "single")
    rpr.append(underline)
    run.append(rpr)

    text_element = OxmlElement("w:t")
    text_element.set(qn("xml:space"), "preserve")
    text_element.text = text
    run.append(text_element)

    hyperlink.append(run)
    paragraph._p.append(hyperlink)


def _add_reference(paragraph: Paragraph, reference: str) -> None:
    """Add a scripture ``reference`` token, hyperlinked to omnibible when it maps.

    Representable references (single book + chapter, optional ascending verse
    range) become an ERV omnibible link; a same-book cross-chapter range links its
    leading chapter, truncated at that chapter's own last verse, since the
    ``?v=`` query cannot address two chapters at once (spec 0016, ADR-0079).
    Everything else (unrecognized text) renders as the plain bold reference it
    was before (spec 0010).
    """
    url = erv.reference_url(reference)
    if url is None:
        paragraph.add_run(reference).font.bold = True
        return
    _add_hyperlink(paragraph, url, reference)


def _add_scripture_segments(paragraph: Paragraph, segments: list[dict[str, str]]) -> None:
    """Add each verse ``segment``'s number as a superscript run, then its text (ADR-0070).

    ``segments`` is the ordered ``{number, text}`` list ``generate.py`` injects — one per
    ERV verse or merged block (spec 0016, ADR-0026). No space separates a number from its
    own text (the print-Bible convention); a space separates one verse's text from the
    next verse's number. A single-segment quote omits the number entirely — the citation
    above it already names the one verse being quoted (#503).
    """
    single_verse = len(segments) == 1
    for index, segment in enumerate(segments):
        if index:
            paragraph.add_run(" ")
        if not single_verse:
            paragraph.add_run(segment["number"]).font.superscript = True
        paragraph.add_run(segment["text"])


def _new_style(doc: DocxDocument, name: str) -> ParagraphStyle:
    """Add an empty paragraph style ``name`` based on Normal."""
    style = cast(ParagraphStyle, doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH))
    style.base_style = doc.styles["Normal"]
    return style


def _register_styles(doc: DocxDocument) -> None:
    """Register the PRD §7.2 paragraph styles on ``doc``.

    Normal carries the serif body at 11pt / 1.25 spacing; the rest layer on the
    section label, sub-head, and the three distinct quote/scripture treatments.
    """
    normal = doc.styles["Normal"]
    normal.font.name = _BODY_FONT
    normal.font.size = Pt(11)
    normal.paragraph_format.line_spacing = 1.25
    normal.paragraph_format.space_after = Pt(6)

    title = _new_style(doc, _STYLE_TITLE)
    title.font.name = _HEADING_FONT
    title.font.size = Pt(24)
    title.font.bold = True
    title.font.color.rgb = _TITLE_BLACK
    title.paragraph_format.space_before = Pt(4)
    title.paragraph_format.space_after = Pt(2)

    subtitle = _new_style(doc, _STYLE_SUBTITLE)
    subtitle.font.name = _HEADING_FONT
    subtitle.font.size = Pt(11)
    subtitle.font.color.rgb = _DARK_GREY
    subtitle.paragraph_format.space_after = Pt(10)

    label = _new_style(doc, _STYLE_SECTION_LABEL)
    label.font.name = _HEADING_FONT
    label.font.size = Pt(10)
    label.font.all_caps = True
    label.font.bold = True
    label.font.color.rgb = _DARK_GREY
    label.paragraph_format.space_before = Pt(16)
    label.paragraph_format.space_after = Pt(6)
    _set_char_spacing(label.element.get_or_add_rPr(), 20)  # ~1pt letter-spacing

    subhead = _new_style(doc, _STYLE_SUBHEAD)
    subhead.font.name = _HEADING_FONT
    subhead.font.size = Pt(11)
    subhead.font.bold = True
    subhead.font.color.rgb = _TITLE_BLACK
    subhead.paragraph_format.space_before = Pt(8)
    subhead.paragraph_format.space_after = Pt(2)

    pull = _new_style(doc, _STYLE_PULL_QUOTE)
    pull.font.name = _BODY_FONT
    pull.font.size = Pt(13)
    pull.font.italic = True
    pull.font.color.rgb = _MUTED_BLUE
    pull.paragraph_format.left_indent = Inches(0.4)
    pull.paragraph_format.space_before = Pt(8)
    pull.paragraph_format.space_after = Pt(8)
    _set_borders(pull.element.get_or_add_pPr(), [_border("left", sz=18, space=12)])

    insight = _new_style(doc, _STYLE_INSIGHT)
    insight.font.name = _HEADING_FONT
    insight.font.size = Pt(10)
    insight.font.italic = True
    insight.font.color.rgb = _MUTED_BLUE
    insight.paragraph_format.left_indent = Inches(0.3)
    insight.paragraph_format.space_before = Pt(2)
    insight.paragraph_format.space_after = Pt(8)

    scripture = _new_style(doc, _STYLE_SCRIPTURE)
    scripture.font.name = _HEADING_FONT
    scripture.font.size = Pt(10)
    scripture.paragraph_format.left_indent = Inches(0.3)
    scripture.paragraph_format.space_before = Pt(2)
    scripture.paragraph_format.space_after = Pt(6)


def _add_divider(doc: DocxDocument) -> None:
    """Add a thin muted-blue horizontal rule (PRD §7.2 section divider)."""
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(6)
    paragraph.paragraph_format.space_after = Pt(6)
    _set_borders(
        paragraph.paragraph_format.element.get_or_add_pPr(), [_border("bottom", sz=8, space=1)]
    )


def _add_right_tabbed(paragraph: Paragraph, left: str, right: str) -> None:
    """Lay ``left`` at the margin and ``right`` flush to the content edge via a tab."""
    paragraph.paragraph_format.tab_stops.add_tab_stop(_CONTENT_WIDTH, WD_TAB_ALIGNMENT.RIGHT)
    paragraph.add_run(left)
    paragraph.add_run("\t" + right)


def _add_section_label(doc: DocxDocument, text: str) -> None:
    """Add a section label with the §7.1 muted-blue vertical-bar marker."""
    paragraph = doc.add_paragraph(style=_STYLE_SECTION_LABEL)
    bar = paragraph.add_run("▌ ")  # ▌ left vertical bar
    bar.font.color.rgb = _MUTED_BLUE
    paragraph.add_run(text)


def _add_header_band(doc: DocxDocument, sermon: SermonRecord) -> None:
    """Render the page-1 header band: church monogram (left), date (right)."""
    paragraph = doc.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(10)
    _add_right_tabbed(paragraph, _church_name(sermon.source).upper(), sermon.published_on)
    for run in paragraph.runs:
        run.font.name = _HEADING_FONT
        run.font.size = Pt(9)
        run.font.color.rgb = _DARK_GREY
        _set_char_spacing(run.font.element.get_or_add_rPr(), 30)


def _add_glance(doc: DocxDocument, briefing: dict[str, Any]) -> None:
    """At a Glance: thesis, takeaways, and the styled pull-quote (PRD §7.1)."""
    _add_section_label(doc, GLANCE_LABEL)
    doc.add_paragraph(briefing["thesis"])
    for takeaway in briefing["takeaways"]:
        doc.add_paragraph(takeaway, style="List Bullet")
    doc.add_paragraph(smart_quote(briefing["pull_quote"]), style=_STYLE_PULL_QUOTE)


def _add_movement(doc: DocxDocument, movement: dict[str, Any]) -> None:
    """One argument movement: analytical heading (+ subtitle), body, insight line."""
    heading = doc.add_paragraph(style=_STYLE_SUBHEAD)
    heading.add_run(movement["heading"])
    subtitle = movement.get("subtitle", "")
    if subtitle:
        clause = heading.add_run(f" — {subtitle}")
        clause.font.bold = False
        clause.font.italic = True
        clause.font.color.rgb = _DARK_GREY

    doc.add_paragraph(movement["body"])

    insight = movement.get("insight", "")
    if insight:
        paragraph = doc.add_paragraph(style=_STYLE_INSIGHT)
        label = paragraph.add_run("Insight — ")
        label.font.bold = True
        paragraph.add_run(insight)


def _add_exposition(doc: DocxDocument, study: dict[str, Any]) -> None:
    """Exposition: scripture engaged, the argument (with insights), cross-references."""
    _add_section_label(doc, EXPOSITION_LABEL)

    doc.add_paragraph("Scripture engaged", style=_STYLE_SUBHEAD)
    for engaged in study["scripture_engaged"]:
        paragraph = doc.add_paragraph(style=_STYLE_SCRIPTURE)
        _add_reference(paragraph, engaged["reference"])
        text = engaged.get("text")
        # empty/absent when the ERV dataset could not resolve the reference, or the
        # citation exceeded the quotable-length cap (spec 0016, ADR-0055). A plain
        # string is an already-published note re-rendered by a backfill script
        # (ADR-0070) — it carries no per-verse boundary, so it renders as before,
        # with no superscript.
        if isinstance(text, str):
            if text:
                paragraph.add_run(f" — {text}")
        elif text:
            paragraph.add_run(" — ")
            _add_scripture_segments(paragraph, text)

    doc.add_paragraph("The argument", style=_STYLE_SUBHEAD)
    for movement in study["argument"]:
        _add_movement(doc, movement)

    if study["cross_references"]:
        doc.add_paragraph("Cross-references", style=_STYLE_SUBHEAD)
        for cross_reference in study["cross_references"]:
            paragraph = doc.add_paragraph(style="List Bullet")
            _add_reference(paragraph, cross_reference["reference"])
            paragraph.add_run(f" — {cross_reference['usage']}")


def _add_devotional(doc: DocxDocument, devotional: dict[str, Any]) -> None:
    """Devotional: meditation, reflection prompts, closing prayer (PRD §7.1)."""
    _add_section_label(doc, DEVOTIONAL_LABEL)
    doc.add_paragraph(devotional["meditation"])

    doc.add_paragraph("Reflection", style=_STYLE_SUBHEAD)
    for prompt in devotional["reflection_prompts"]:
        doc.add_paragraph(prompt, style="List Bullet")

    doc.add_paragraph("Closing prayer", style=_STYLE_SUBHEAD)
    prayer = doc.add_paragraph()
    run = prayer.add_run(devotional["closing_prayer"]["text"])
    run.font.italic = True


def _add_formation(doc: DocxDocument, formation: dict[str, Any]) -> None:
    """Formation: the takeaway to remember, the concrete step to do, its scripture anchor.

    The formation section (#140, ADR-0020) — the note's culmination after the
    Devotional. Each beat reads defensively via ``.get`` so a legacy or partial note
    degrades to fewer lines rather than raising; :func:`_build_document` only calls
    this when ``formation`` is present at all.
    """
    _add_section_label(doc, FORMATION_LABEL)

    one_thing = formation.get("one_thing", "")
    if one_thing:
        doc.add_paragraph("One thing to remember", style=_STYLE_SUBHEAD)
        doc.add_paragraph(one_thing)

    step = formation.get("step", "")
    if step:
        doc.add_paragraph("This week", style=_STYLE_SUBHEAD)
        doc.add_paragraph(step)

    anchor = formation.get("anchor", "")
    if anchor:
        paragraph = doc.add_paragraph(style=_STYLE_SCRIPTURE)
        label = paragraph.add_run("Return to ")
        label.font.italic = True
        _add_reference(paragraph, anchor)


def _fill_footer(footer: object, sermon: SermonRecord) -> None:
    """Write the provenance line (church attribution + episode link) into a footer."""
    paragraph = footer.paragraphs[0]  # type: ignore[attr-defined]
    paragraph.text = ""
    provenance = provenance_line(_church_name(sermon.source), sermon.episode_url)
    run = paragraph.add_run(provenance)
    run.font.name = _HEADING_FONT
    run.font.size = Pt(8)
    run.font.color.rgb = _DARK_GREY


def _fill_running_header(header: object, sermon: SermonRecord) -> None:
    """Write the page-2+ running header: title (left), date (right) (PRD §7.2)."""
    paragraph = header.paragraphs[0]  # type: ignore[attr-defined]
    paragraph.text = ""
    _add_right_tabbed(paragraph, sermon.title, sermon.published_on)
    for run in paragraph.runs:
        run.font.name = _HEADING_FONT
        run.font.size = Pt(9)
        run.font.color.rgb = _DARK_GREY


def _build_document(note: dict[str, Any], sermon: SermonRecord) -> DocxDocument:
    """Assemble the full three-section document for ``note`` and ``sermon``."""
    doc = docx.Document()

    section = doc.sections[0]
    section.page_width = Inches(8.5)
    section.page_height = Inches(11)
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)
    section.different_first_page_header_footer = True

    doc.core_properties.author = sermon.speaker or ""

    _register_styles(doc)

    _add_header_band(doc, sermon)
    doc.add_paragraph(sermon.title, style=_STYLE_TITLE)
    subtitle = " · ".join(
        part for part in (sermon.speaker, sermon.series, scripture_line(note, sermon)) if part
    )
    doc.add_paragraph(subtitle, style=_STYLE_SUBTITLE)
    _add_divider(doc)

    _add_glance(doc, note["at_a_glance"])
    _add_exposition(doc, note["exposition"])
    _add_devotional(doc, note["devotional"])
    formation = note.get("formation")
    if formation:
        _add_formation(doc, formation)
    _add_divider(doc)

    _fill_running_header(section.header, sermon)
    _fill_footer(section.footer, sermon)
    _fill_footer(section.first_page_footer, sermon)

    return doc


def _validate_docx(path: Path) -> None:
    """Assert ``path`` is a structurally valid ``.docx`` (PRD §11.1).

    A valid file is a ZIP archive carrying the required OOXML parts that python-docx
    can re-open into a document with body content. Any shortfall raises
    :class:`DocxValidationError`, which :func:`render_note` treats as a retryable miss.
    """
    if not zipfile.is_zipfile(path):
        raise DocxValidationError(f"{path} is not a valid .docx (not a zip archive)")
    with zipfile.ZipFile(path) as archive:
        members = set(archive.namelist())
    required = {"[Content_Types].xml", "word/document.xml"}
    missing = required - members
    if missing:
        raise DocxValidationError(f"{path} is missing required OOXML parts: {sorted(missing)}")
    try:
        reopened = docx.Document(str(path))
    except Exception as exc:  # noqa: BLE001 — any open failure is a validation failure.
        raise DocxValidationError(f"{path} could not be reopened: {exc}") from exc
    if not reopened.paragraphs:
        raise DocxValidationError(f"{path} has no body content")


def _output_path(notes_dir: Path, sermon: SermonRecord) -> Path:
    """Compute ``notes/YYYY/YYYY-MM-DD_<slug>.docx`` for ``sermon`` (PRD §7.3).

    Shares its stem with the generated-note JSON sidecar (spec 0009) by routing both
    through :func:`sermon_notes.artifacts.note_artifact_path`.
    """
    from sermon_notes.artifacts import note_artifact_path

    return note_artifact_path(notes_dir, sermon, ".docx")


def render_note(
    note: dict[str, Any],
    sermon: SermonRecord,
    *,
    notes_dir: Path = DEFAULT_NOTES_DIR,
    validate: Callable[[Path], None] = _validate_docx,
) -> Path:
    """Render ``note`` to a validated ``.docx`` and return its path (U7).

    Builds the PRD §7.1–7.2 document, writes it to ``notes/YYYY/YYYY-MM-DD_<slug>.docx``,
    and validates it. A validation failure triggers exactly one regeneration (PRD §11.1):
    the document is rebuilt from the same ``note`` and re-saved to the same path. Because
    that rebuild is deterministic, the retry recovers only *transient* write failures —
    a partial or corrupt save, a momentary filesystem lock. A structural failure rooted
    in the note content produces byte-identical output on the second attempt and fails
    identically, making the render terminal and raising :class:`RenderError`. ``validate``
    is injectable so the retry posture can be exercised.
    """
    path = _output_path(notes_dir, sermon)
    path.parent.mkdir(parents=True, exist_ok=True)

    last_error: DocxValidationError | None = None
    # Initial render + one regeneration (PRD §11.1). The rebuild is deterministic, so it
    # recovers transient write failures only — a structural content failure recurs and is terminal.
    for attempt in (1, 2):
        try:
            _build_document(note, sermon).save(str(path))
        except OSError:
            # A filesystem/infra failure (full disk, permissions) is global, not this
            # note's fault: let it propagate so the run aborts loudly and a re-run can
            # recover the batch, rather than sending every sermon terminal (#126, ADR-0019).
            raise
        except Exception as exc:  # noqa: BLE001 — a python-docx error on this note's content.
            # Content-driven and deterministic: terminal for this one sermon (#126).
            raise RenderError(f"docx build failed for {sermon.guid!r}: {exc}") from exc
        try:
            validate(path)
        except DocxValidationError as exc:
            last_error = exc
            logger.warning(
                "docx validation failed for %s on attempt %d/2: %s", sermon.guid, attempt, exc
            )
            continue
        logger.info("rendered note for %s → %s", sermon.guid, path)
        return path

    raise RenderError(
        f"docx for {sermon.guid!r} failed validation after one regeneration: {last_error}"
    ) from last_error
