"""The layout decisions the ``.docx`` and ``.pdf`` renderings share (ADR-0025).

Each published note is rendered twice from one note JSON: once to ``.docx``
(:mod:`sermon_notes.render`) and once to ``.pdf`` (:mod:`sermon_notes.render_pdf`). The
two differ in typography — one lays out OOXML, the other PDF flowables — but they must
never differ in *what the note says*. Everything format-independent lives here so a
change lands in both documents at once: the PRD §7.1 section labels, the §7.2
provenance sentence and smart quotes, and the title-block scripture line.

Nothing here touches a file or a rendering library; the church display name comes from
:func:`sermon_notes.registry.church_name`, which both renderers already share.
"""

from __future__ import annotations

from typing import Any

from sermon_notes.registry import SermonRecord

# PRD §7.1 section labels, in reading order: vision → why → now-what → means.
# Renamed from EXECUTIVE BRIEFING / STUDY COMPANION / THIS WEEK per ADR-0060 (#416):
# the old names read as corporate and relationally overreaching for a church
# audience, and "THIS WEEK" duplicated the label of its own child (`step`).
GLANCE_LABEL = "AT A GLANCE"
EXPOSITION_LABEL = "EXPOSITION"
DEVOTIONAL_LABEL = "DEVOTIONAL"
FORMATION_LABEL = "FORMATION"

# PRD §7.2 provenance footer: the episode link where the sermon can be heard.
_PROVENANCE = "{link}"


def provenance_line(church: str, link: str) -> str:
    """The footer sentence attributing the note to ``church`` and its episode ``link``."""
    return _PROVENANCE.format(church=church, link=link)


def smart_quote(text: str) -> str:
    """Wrap ``text`` in typographic double quotes (PRD §7.2 smart quotes).

    Straight quotes the model may have added around its own pull-quote are stripped
    first, so the result never doubles up.
    """
    return "“" + text.strip().strip('"').strip() + "”"


def scripture_line(note: dict[str, Any], sermon: SermonRecord) -> str:
    """The scripture for the title block: engaged references, else the feed's.

    Names what the sermon actually engaged, in the order it engaged it, each reference
    once. A note that engaged nothing falls back to the references the feed advertised.
    """
    engaged = note.get("exposition", {}).get("scripture_engaged", [])
    refs: list[str] = []
    for entry in engaged:
        reference = entry.get("reference")
        if reference and reference not in refs:
            refs.append(reference)
    return "; ".join(refs or sermon.scripture_refs)
