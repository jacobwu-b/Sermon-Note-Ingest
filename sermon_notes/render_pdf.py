"""Render a validated note JSON into the study note's PDF (#173, ADR-0025).

The note's *second rendering*. It reads the same persisted note (spec 0009) that
:mod:`sermon_notes.render` reads and lays out the same PRD §7.1–7.2 document — the four
sections, the styled pull-quote, insight and scripture blocks, the omnibible scripture
links (spec 0010), a page-2+ running header and a provenance footer — as a PDF. It is
never a conversion of the ``.docx``: both documents descend from the note JSON, and
everything they must agree on lives in :mod:`sermon_notes.note_layout`.

The PDF exists because Discord previews it inline and cannot preview a ``.docx``
(spec 0017 amendment), so the study group reads a new note in the channel instead of
downloading it. Typography draws on the same Source Serif Pro / Source Sans Pro family
the ``.docx`` requests, embedded from ``assets/fonts/`` (ADR-0025 amendment, #313) —
falling back to reportlab's base-14 Times/Helvetica pairing if a font file is missing
or corrupt.

The PDF sets the page **serif-dominant** (ADR-0025 amendment): the serif carries the
title, section labels, subheads, scripture and body — everything the reader reads —
and the sans is reserved for apparatus, the church/date band and the running header
and footer. The ``.docx`` still sets its headings in the sans, so the two are
deliberately not identical here; a PDF is read on screen as a document in its own
right, where a reading face throughout holds together better than a display pairing.

The ``.docx`` remains the deliverable of record, so this renderer is **best-effort**:
:func:`render_note_pdf` returns ``None`` on any failure rather than raising, and the
render stage publishes on the ``.docx`` alone. A convenience rendering must never
strand a sermon that has already paid for an LLM call.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from xml.sax.saxutils import escape

from reportlab.lib.colors import HexColor
from reportlab.lib.enums import TA_RIGHT
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Flowable,
    Frame,
    HRFlowable,
    PageTemplate,
    Paragraph,
    Table,
)

from sermon_notes import erv
from sermon_notes.artifacts import note_artifact_path
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

# Output root: the repo's notes/ folder, matching the .docx renderer.
DEFAULT_NOTES_DIR = Path(__file__).resolve().parents[2] / "notes"

# PRD §7.2 palette and type. The Source Serif/Sans faces the .docx requests (ADR-0005)
# are embedded from assets/fonts/ and registered below, so the PDF sets the same family
# as the Word file rather than two unrelated generic faces (ADR-0025 amendment, #313).
# A missing or corrupt font file falls back to reportlab's base-14 Times-Roman/Helvetica
# pairing rather than losing the render.
_FONTS_DIR = Path(__file__).resolve().parents[2] / "assets" / "fonts"


def _register_brand_fonts(fonts_dir: Path) -> bool:
    """Register the embedded Source Serif/Sans faces with reportlab.

    All-or-nothing: any failure (a missing or corrupt TTF) leaves reportlab's font
    registry untouched and reports ``False``, so the caller can fall back to the base-14
    pairing rather than build a document with some styles registered and others not.
    """
    try:
        pdfmetrics.registerFont(
            TTFont("SourceSerifPro", str(fonts_dir / "SourceSerifPro-Regular.ttf"))
        )
        pdfmetrics.registerFont(
            TTFont("SourceSerifPro-Italic", str(fonts_dir / "SourceSerifPro-It.ttf"))
        )
        pdfmetrics.registerFont(
            TTFont("SourceSerifPro-Semibold", str(fonts_dir / "SourceSerifPro-Semibold.ttf"))
        )
        pdfmetrics.registerFont(
            TTFont("SourceSansPro", str(fonts_dir / "SourceSansPro-Regular.ttf"))
        )
        pdfmetrics.registerFont(
            TTFont("SourceSansPro-Bold", str(fonts_dir / "SourceSansPro-Bold.ttf"))
        )
        pdfmetrics.registerFont(
            TTFont("SourceSansPro-Italic", str(fonts_dir / "SourceSansPro-It.ttf"))
        )
    except Exception as exc:  # noqa: BLE001 — falls back to base-14; never blocks the render.
        logger.warning("pdf brand fonts unavailable (%s), falling back to base-14 fonts", exc)
        return False
    # No standalone bold-italic face is shipped: each family's boldItalic slot reuses
    # its bold face — favoring weight over slant — for the two inline emphasis spots
    # that combine them (a subtitle clause, an "Insight —" label). The serif's bold slot
    # is the Semibold: at heading sizes beside 11pt serif body, full Bold shouts where
    # Semibold carries the same hierarchy quietly.
    pdfmetrics.registerFontFamily(
        "SourceSerifPro",
        normal="SourceSerifPro",
        bold="SourceSerifPro-Semibold",
        italic="SourceSerifPro-Italic",
        boldItalic="SourceSerifPro-Semibold",
    )
    pdfmetrics.registerFontFamily(
        "SourceSansPro",
        normal="SourceSansPro",
        bold="SourceSansPro-Bold",
        italic="SourceSansPro-Italic",
        boldItalic="SourceSansPro-Bold",
    )
    return True


if _register_brand_fonts(_FONTS_DIR):
    _BODY_FONT = "SourceSerifPro"
    _BODY_ITALIC = "SourceSerifPro-Italic"
    _BODY_SEMIBOLD = "SourceSerifPro-Semibold"
    _HEADING_FONT = "SourceSansPro"
    _HEADING_BOLD = "SourceSansPro-Bold"
    _HEADING_ITALIC = "SourceSansPro-Italic"
else:
    _BODY_FONT = "Times-Roman"
    _BODY_ITALIC = "Times-Italic"
    _BODY_SEMIBOLD = "Times-Bold"
    _HEADING_FONT = "Helvetica"
    _HEADING_BOLD = "Helvetica-Bold"
    _HEADING_ITALIC = "Helvetica-Oblique"

# The page's vertical rhythm. The .docx sets ``line_spacing = 1.25``, which Word applies
# to the *font's line box* — 1.371em for Source Serif Pro, so ~1.5x the point size once
# rendered. Platypus takes an absolute leading instead, so applying 1.25 to the point
# size (as this module first did) set the whole page about a fifth tighter than the same
# note in Word. Every reading style derives its leading from this one number; the title
# is the single deliberate exception, since display type sets tighter than text type.
_LEADING = 1.5
_TITLE_LEADING = 30

# Letter-spacing for the two lines of caps, matching the .docx's ``w:spacing`` (1.5pt on
# the church/date band, 1pt on the section labels). Caps set solid read as a mistake.
_BAND_TRACKING = 1.5
_LABEL_TRACKING = 1.0

_MUTED_BLUE = HexColor("#2E75B6")
_DARK_GREY = HexColor("#444444")
_TITLE_BLACK = HexColor("#111111")

# US Letter with 1" margins (PRD §7.2): 6.5" of content between the margins.
_MARGIN = inch
_CONTENT_WIDTH = LETTER[0] - 2 * _MARGIN
# Room reserved below the text frame for the provenance footer, and above it on
# page 2+ for the running header.
_FOOTER_GAP = 0.4 * inch
_HEADER_GAP = 0.35 * inch
# The hairline that separates the running header and footer from the text block, and
# the space between that rule and the line of text it divides.
_RULE_WIDTH = 0.4
_RULE_GAP = 0.12 * inch
# The pull-quote is set in from the measure rather than standing on the margin, so its
# rule does not read as another section marker: the .docx indents the quote 0.4" and
# hangs its border 12pt to the left of that.
_QUOTE_INSET = 0.4 * inch - 12
# The provenance footer: its type size and line spacing, and the clear space kept
# between the wrapped sentence and the folio sharing its first line.
_FOOTER_SIZE = 8
_FOOTER_LEADING = 9.5
_FOOTER_GUTTER = 12


class PdfRenderError(RuntimeError):
    """The PDF could not be produced or is not a structurally valid file."""


def _style(name: str, **kwargs: Any) -> ParagraphStyle:
    """A paragraph style over the serif body default (PRD §7.2).

    Leading is derived from the style's own point size at the page rhythm
    (:data:`_LEADING`) unless a style states one, so a new style declares its size and
    inherits the rhythm rather than carrying a second number that can drift from it.
    """
    merged = {"fontName": _BODY_FONT, "fontSize": 11, "spaceAfter": 6, **kwargs}
    merged.setdefault("leading", merged["fontSize"] * _LEADING)
    return ParagraphStyle(name, **merged)


# The §7.2 style system. A study note is a reading document, so the serif carries
# everything the reader reads — title, labels, headings, scripture — and the sans is
# furniture only: the church/date band, the running header, the footer (ADR-0025
# amendment). The title holds the page on size alone, where large serif regular reads
# as editorial rather than weak; the labels and subheads take the Semibold, because at
# body size a serif-regular heading reads as slightly larger body text and the page
# loses its hierarchy.
_TITLE = _style(
    "Title",
    fontName=_BODY_FONT,
    fontSize=25,
    leading=_TITLE_LEADING,
    spaceAfter=6,
    textColor=_TITLE_BLACK,
)
_SUBTITLE = _style(
    "Subtitle", fontName=_BODY_FONT, fontSize=11, spaceAfter=10, textColor=_DARK_GREY
)
_SECTION_LABEL = _style(
    "SectionLabel", fontName=_BODY_SEMIBOLD, fontSize=10.5, textColor=_DARK_GREY
)
_SUBHEAD = _style(
    "Subhead",
    fontName=_BODY_SEMIBOLD,
    fontSize=12,
    spaceBefore=12,
    spaceAfter=2,
    textColor=_TITLE_BLACK,
)
_BODY = _style("Body")
# Platypus draws the bullet in its own default 10pt Helvetica unless told otherwise,
# which puts a sans dot beside every serif takeaway.
_BULLET = _style(
    "Bullet", leftIndent=18, bulletIndent=6, bulletFontName=_BODY_FONT, bulletFontSize=11
)
_PULL_QUOTE = _style("PullQuote", fontName=_BODY_ITALIC, fontSize=13, textColor=_MUTED_BLUE)
_INSIGHT = _style(
    "Insight",
    fontName=_BODY_ITALIC,
    fontSize=10,
    leftIndent=0.3 * inch,
    spaceBefore=2,
    spaceAfter=8,
    textColor=_MUTED_BLUE,
)
_SCRIPTURE = _style(
    "Scripture",
    fontName=_BODY_FONT,
    fontSize=10,
    leftIndent=0.3 * inch,
    spaceBefore=2,
)
_BAND = _style("Band", fontName=_HEADING_FONT, fontSize=9, textColor=_DARK_GREY)
_BAND_RIGHT = _style(
    "BandRight", fontName=_HEADING_FONT, fontSize=9, textColor=_DARK_GREY, alignment=TA_RIGHT
)


class _TrackedLine(Flowable):  # type: ignore[misc]  # reportlab is untyped (ADR-0025)
    """One line of letter-spaced text, drawn rather than laid out.

    Platypus carries no character-spacing attribute, so the two lines of caps the
    ``.docx`` tracks — the church/date band and the section labels — are drawn straight
    onto the canvas with ``setCharSpace``. The text still reaches the page as a single
    show-text operand, so it selects, copies and extracts as its own unbroken string.
    """

    def __init__(self, text: str, style: ParagraphStyle, tracking: float) -> None:
        super().__init__()
        self._text = text
        self._style = style
        self._tracking = tracking
        self._available = 0.0
        self.height = style.leading
        # A trailing character keeps its spacing in PDF, so the *visual* width is one
        # gap short of the naive count — the difference a right-aligned line hangs on.
        self.width = stringWidth(text, style.fontName, style.fontSize) + tracking * max(
            len(text) - 1, 0
        )

    def wrap(self, availWidth: float, _availHeight: float) -> tuple[float, float]:
        self._available = availWidth
        return availWidth, self.height

    def draw(self) -> None:
        # Character spacing lives on a text object, not on the canvas, so the line is
        # shown through one — still a single ``Tj``, still one selectable string.
        offset = self._available - self.width if self._style.alignment == TA_RIGHT else 0.0
        text = self.canv.beginText(offset, self.height - self._style.fontSize)
        text.setFont(self._style.fontName, self._style.fontSize)
        text.setFillColor(self._style.textColor)
        text.setCharSpace(self._tracking)
        text.textOut(self._text)
        self.canv.drawText(text)


def _ruled(flowable: Flowable, *, thickness: float, indent: float, inset: float = 0.0) -> Table:
    """Put a muted-blue vertical rule down the left of ``flowable`` (PRD §7.1–7.2).

    The PDF analog of the ``.docx``'s left paragraph border — used for the section
    label's marker bar, which stands on the margin, and for the pull-quote's rule, which
    stands ``inset`` from it. The table spans the full content width and is left-aligned
    either way: sized to anything narrower it would be *centred* in the frame, putting
    the bar on an edge of its own a few points inboard of the text it marks.
    """
    if inset:
        table = Table([["", flowable]], colWidths=[inset, _CONTENT_WIDTH - inset])
        ruled = 1
    else:
        table = Table([[flowable]], colWidths=[_CONTENT_WIDTH])
        ruled = 0
    table.hAlign = "LEFT"
    table.setStyle(
        [
            ("LINEBEFORE", (ruled, 0), (ruled, -1), thickness, _MUTED_BLUE),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("LEFTPADDING", (ruled, 0), (ruled, -1), indent),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]
    )
    return table


def _split_row(left: str, right: str) -> Table:
    """One line with ``left`` at the margin and ``right`` flush to the content edge."""
    table = Table(
        [
            [
                _TrackedLine(left, _BAND, _BAND_TRACKING),
                _TrackedLine(right, _BAND_RIGHT, _BAND_TRACKING),
            ]
        ],
        colWidths=[_CONTENT_WIDTH * 0.6, _CONTENT_WIDTH * 0.4],
    )
    table.hAlign = "LEFT"
    table.setStyle(
        [
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]
    )
    return table


def _reference_markup(reference: str) -> str:
    """A scripture ``reference``, linked to omnibible when it maps (spec 0010).

    Representable references become an ERV link in the §7.2 muted blue; a
    same-book cross-chapter range links its leading chapter, truncated at that
    chapter's own last verse (spec 0016, ADR-0079); everything else (unrecognized
    text) stays plain bold text, the same fallback the ``.docx`` makes.
    """
    text = escape(reference)
    url = erv.reference_url(reference)
    if url is None:
        return f"<b>{text}</b>"
    return f'<link href="{escape(url, {chr(34): "&quot;"})}" color="#2E75B6"><b><u>{text}</u></b></link>'


def _scripture_segments_markup(segments: list[dict[str, str]]) -> str:
    """Inline markup for each verse ``segment``'s number as a superscript (ADR-0070).

    ``segments`` is the ordered ``{number, text}`` list ``generate.py`` injects — one per
    ERV verse or merged block (spec 0016, ADR-0026). No space separates a number from its
    own text (the print-Bible convention); a space separates one verse's text from the
    next verse's number. A single-segment quote omits the number entirely — the citation
    above it already names the one verse being quoted (#503).
    """
    if len(segments) == 1:
        return escape(segments[0]["text"])
    return " ".join(
        f"<super>{escape(segment['number'])}</super>{escape(segment['text'])}"
        for segment in segments
    )


def _section_label(text: str) -> Table:
    """A section label behind the §7.1 muted-blue marker bar."""
    label = _ruled(_TrackedLine(text, _SECTION_LABEL, _LABEL_TRACKING), thickness=3, indent=6)
    # Platypus collapses adjoining space to the larger of the two, where Word adds them:
    # 20 here is the air the .docx's 16-before over a 6-after paragraph actually renders.
    label.spaceBefore = 20
    label.spaceAfter = 6
    return label


def _divider() -> HRFlowable:
    """A thin muted-blue horizontal rule (PRD §7.2 section divider)."""
    return HRFlowable(width="100%", thickness=0.5, color=_MUTED_BLUE, spaceBefore=12, spaceAfter=6)


def _glance(briefing: dict[str, Any]) -> list[Flowable]:
    """At a Glance: thesis, takeaways, and the styled pull-quote (PRD §7.1)."""
    flowables: list[Flowable] = [_section_label(GLANCE_LABEL)]
    flowables.append(Paragraph(escape(briefing["thesis"]), _BODY))
    for takeaway in briefing["takeaways"]:
        flowables.append(Paragraph(escape(takeaway), _BULLET, bulletText="•"))
    quote = _ruled(
        Paragraph(escape(smart_quote(briefing["pull_quote"])), _PULL_QUOTE),
        thickness=2.25,
        indent=12,
        inset=_QUOTE_INSET,
    )
    quote.spaceBefore = 8
    quote.spaceAfter = 8
    flowables.append(quote)
    return flowables


def _movement(movement: dict[str, Any]) -> list[Flowable]:
    """One argument movement: analytical heading (+ subtitle), body, insight line."""
    heading = escape(movement["heading"])
    subtitle = movement.get("subtitle", "")
    if subtitle:
        heading += f' <i><font color="#444444">— {escape(subtitle)}</font></i>'
    flowables: list[Flowable] = [
        Paragraph(heading, _SUBHEAD),
        Paragraph(escape(movement["body"]), _BODY),
    ]
    insight = movement.get("insight", "")
    if insight:
        flowables.append(Paragraph(f"<b>Insight — </b>{escape(insight)}", _INSIGHT))
    return flowables


def _exposition(study: dict[str, Any]) -> list[Flowable]:
    """Exposition: scripture engaged, the argument (with insights), cross-references."""
    flowables: list[Flowable] = [_section_label(EXPOSITION_LABEL)]

    flowables.append(Paragraph("Scripture engaged", _SUBHEAD))
    for engaged in study["scripture_engaged"]:
        markup = _reference_markup(engaged["reference"])
        text = engaged.get("text")
        # empty/absent when the ERV dataset could not resolve the reference, or the
        # citation exceeded the quotable-length cap (spec 0016, ADR-0055). A plain
        # string is an already-published note re-rendered by a backfill script
        # (ADR-0070) — it carries no per-verse boundary, so it renders as before,
        # with no superscript.
        if isinstance(text, str):
            if text:
                markup += f" — {escape(text)}"
        elif text:
            markup += f" — {_scripture_segments_markup(text)}"
        flowables.append(Paragraph(markup, _SCRIPTURE))

    flowables.append(Paragraph("The argument", _SUBHEAD))
    for movement in study["argument"]:
        flowables.extend(_movement(movement))

    if study["cross_references"]:
        flowables.append(Paragraph("Cross-references", _SUBHEAD))
        for cross_reference in study["cross_references"]:
            markup = _reference_markup(cross_reference["reference"])
            flowables.append(
                Paragraph(f"{markup} — {escape(cross_reference['usage'])}", _BULLET, bulletText="•")
            )
    return flowables


def _devotional(devotional: dict[str, Any]) -> list[Flowable]:
    """Devotional: meditation, reflection prompts, closing prayer (PRD §7.1)."""
    flowables: list[Flowable] = [_section_label(DEVOTIONAL_LABEL)]
    flowables.append(Paragraph(escape(devotional["meditation"]), _BODY))

    flowables.append(Paragraph("Reflection", _SUBHEAD))
    for prompt in devotional["reflection_prompts"]:
        flowables.append(Paragraph(escape(prompt), _BULLET, bulletText="•"))

    flowables.append(Paragraph("Closing prayer", _SUBHEAD))
    flowables.append(Paragraph(f"<i>{escape(devotional['closing_prayer']['text'])}</i>", _BODY))
    return flowables


def _formation(formation: dict[str, Any]) -> list[Flowable]:
    """Formation: the takeaway to remember, the step to do, its scripture anchor.

    Each beat reads defensively via ``.get`` so a partial note degrades to fewer lines
    rather than raising — the same posture the ``.docx`` renderer keeps (ADR-0020).
    """
    flowables: list[Flowable] = [_section_label(FORMATION_LABEL)]

    one_thing = formation.get("one_thing", "")
    if one_thing:
        flowables.append(Paragraph("One thing to remember", _SUBHEAD))
        flowables.append(Paragraph(escape(one_thing), _BODY))

    step = formation.get("step", "")
    if step:
        flowables.append(Paragraph("This week", _SUBHEAD))
        flowables.append(Paragraph(escape(step), _BODY))

    anchor = formation.get("anchor", "")
    if anchor:
        flowables.append(Paragraph(f"<i>Return to </i>{_reference_markup(anchor)}", _SCRIPTURE))
    return flowables


def _build_flowables(note: dict[str, Any], sermon: SermonRecord, church: str) -> list[Flowable]:
    """Assemble the whole document for ``note`` and ``sermon``, in reading order."""
    flowables: list[Flowable] = [_split_row(church.upper(), sermon.published_on)]
    flowables.append(Paragraph(escape(sermon.title), _TITLE))
    subtitle = " · ".join(
        part for part in (sermon.speaker, sermon.series, scripture_line(note, sermon)) if part
    )
    flowables.append(Paragraph(escape(subtitle), _SUBTITLE))
    flowables.append(_divider())

    flowables.extend(_glance(note["at_a_glance"]))
    flowables.extend(_exposition(note["exposition"]))
    flowables.extend(_devotional(note["devotional"]))
    formation = note.get("formation")
    if formation:
        flowables.extend(_formation(formation))
    flowables.append(_divider())
    return flowables


class _PageFurniture:
    """Draws the running header and provenance footer onto each page (PRD §7.2).

    Page 1 carries the header band as content (as the ``.docx`` does), so only pages
    2+ get the running header; every page gets the footer.

    The furniture is set in the sans face and separated from the text block by a
    hairline rule in the §7.2 muted blue, so it reads as apparatus rather than as
    content that happens to sit at the edge of the page. Every page carries its folio
    opposite the provenance line: a study note runs to several pages and is printed for
    a group, where an unnumbered page is an unorderable one.
    """

    def __init__(self, sermon: SermonRecord, church: str) -> None:
        self._sermon = sermon
        self._provenance = provenance_line(church, sermon.episode_url)

    def _rule(self, canvas: Canvas, y: float) -> None:
        """A hairline across the content width at ``y``."""
        canvas.setStrokeColor(_MUTED_BLUE)
        canvas.setLineWidth(_RULE_WIDTH)
        canvas.line(_MARGIN, y, LETTER[0] - _MARGIN, y)

    def _footer(self, canvas: Canvas) -> None:
        baseline = _MARGIN - _FOOTER_GAP
        self._rule(canvas, baseline + _RULE_GAP)
        canvas.setFont(_HEADING_FONT, _FOOTER_SIZE)
        canvas.setFillColor(_DARK_GREY)

        # The total page count is unknown while a page is being drawn, so the folio is
        # the page's own number: a plain folio, not an "n of N" that would need the
        # document built twice to know its own length.
        folio = f"Page {canvas.getPageNumber()}"
        canvas.drawRightString(LETTER[0] - _MARGIN, baseline, folio)

        # The provenance sentence carries a full episode URL and overruns the content
        # width for two of the three churches — it ran off the right edge of the page
        # before there was a folio to collide with. It wraps inside the width the folio
        # leaves, so the footer stays within its margins whatever the URL's length.
        gutter = stringWidth(folio, _HEADING_FONT, _FOOTER_SIZE) + _FOOTER_GUTTER
        lines = simpleSplit(self._provenance, _HEADING_FONT, _FOOTER_SIZE, _CONTENT_WIDTH - gutter)
        for offset, line in enumerate(lines):
            canvas.drawString(_MARGIN, baseline - offset * _FOOTER_LEADING, line)

    def first_page(self, canvas: Canvas, _doc: BaseDocTemplate) -> None:
        self._footer(canvas)

    def later_pages(self, canvas: Canvas, _doc: BaseDocTemplate) -> None:
        top = LETTER[1] - _MARGIN + _HEADER_GAP
        canvas.setFont(_HEADING_FONT, 9)
        canvas.setFillColor(_DARK_GREY)
        canvas.drawString(_MARGIN, top, self._sermon.title)
        canvas.drawRightString(LETTER[0] - _MARGIN, top, self._sermon.published_on)
        self._rule(canvas, top - _RULE_GAP)
        self._footer(canvas)


def _validate_pdf(path: Path) -> None:
    """Assert ``path`` is a structurally valid PDF: the magic header and a trailer."""
    raw = path.read_bytes()
    if not raw.startswith(b"%PDF-"):
        raise PdfRenderError(f"{path} is not a PDF (no %PDF- header)")
    if not raw.rstrip().endswith(b"%%EOF"):
        raise PdfRenderError(f"{path} is truncated (no %%EOF trailer)")


def render_note_pdf(
    note: dict[str, Any],
    sermon: SermonRecord,
    *,
    notes_dir: Path = DEFAULT_NOTES_DIR,
) -> Path | None:
    """Render ``note`` to a PDF beside its ``.docx``; return the path, or ``None``.

    Writes ``notes/<source>/YYYY/YYYY-MM-DD_<slug>.pdf``, sharing the ``.docx`` and
    ``.json`` stem through :func:`~sermon_notes.artifacts.note_artifact_path` — so the
    PDF needs no registry field of its own (ADR-0025).

    Best-effort by contract (spec 0006 addendum): every failure — an unrenderable note,
    an unregistered source, a bad write — is logged and reported as ``None`` so the
    render stage publishes on the ``.docx`` alone. That includes ``OSError``, which the
    ``.docx`` renderer deliberately re-raises as a global infrastructure failure
    (ADR-0019): by the time this runs, the ``.docx`` write has already surfaced any such
    failure loudly, so re-raising here would only convert a lost convenience copy into a
    lost sermon.
    """
    path = note_artifact_path(notes_dir, sermon, ".pdf")
    try:
        church = church_name(sermon.source)
        path.parent.mkdir(parents=True, exist_ok=True)
        furniture = _PageFurniture(sermon, church)
        doc = BaseDocTemplate(
            str(path),
            pagesize=LETTER,
            topMargin=_MARGIN,
            bottomMargin=_MARGIN,
            leftMargin=_MARGIN,
            rightMargin=_MARGIN,
            title=sermon.title,
            author=sermon.speaker or "",
        )
        # The frame is built by hand rather than by ``SimpleDocTemplate`` for its
        # padding alone: platypus pads a frame 6pt on every side, which set the text
        # block 6pt inboard of the margin the header band, rules and page furniture are
        # drawn against — and centred every full-width table on a third edge between the
        # two. The margins are the measure; nothing pads them further.
        frame = Frame(
            _MARGIN,
            _MARGIN,
            _CONTENT_WIDTH,
            LETTER[1] - 2 * _MARGIN,
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
            id="page",
        )
        doc.addPageTemplates(
            [
                PageTemplate(
                    id="first",
                    frames=[frame],
                    onPage=furniture.first_page,
                    autoNextPageTemplate="later",
                ),
                PageTemplate(id="later", frames=[frame], onPage=furniture.later_pages),
            ]
        )
        doc.build(_build_flowables(note, sermon, church))
        _validate_pdf(path)
    except Exception as exc:  # noqa: BLE001 — the PDF is best-effort (spec 0006 addendum).
        logger.warning("pdf render for %s skipped: %s", sermon.guid, exc)
        return None
    logger.info("rendered pdf note for %s → %s", sermon.guid, path)
    return path
