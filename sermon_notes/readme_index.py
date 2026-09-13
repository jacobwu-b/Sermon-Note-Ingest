"""Auto-generate the README sermon index — the repo's only "UI" (PRD §7.5, U8).

The index is a reverse-chronological Markdown table regenerated from the registry,
one row per rendered sermon with a working relative link to its ``.docx``. The
rewrite touches only a region fenced by explicit HTML-comment markers, so any
hand-edited prose elsewhere in ``README.md`` survives untouched (spec 0006 risk).
Regenerating is idempotent: the same ledger always yields the same table.
"""

from __future__ import annotations

import re
from pathlib import Path

from sermon_notes.logging import get_logger
from sermon_notes.registry import (
    DEFAULT_REGISTRY_PATH,
    Registry,
    SermonRecord,
    church_name,
)

# Where the README lives inside a repo root. DEFAULT_README_PATH resolves it against the
# package's own root, two parents above this file; a caller holding some other checkout
# joins this onto that root instead (#269) — the ledger's sibling in registry.py.
README_RELPATH = Path("README.md")
DEFAULT_README_PATH = Path(__file__).resolve().parents[2] / README_RELPATH

# Fences delimiting the auto-generated region; everything outside them is human-owned.
INDEX_START = "<!-- SERMON-INDEX:START -->"
INDEX_END = "<!-- SERMON-INDEX:END -->"

_HEADER = "| Date | Church | Title | Speaker | Series | Scripture |"
_SEPARATOR = "|------|--------|-------|---------|--------|-----------|"

# Shown when an optional field (speaker, series, scripture) is absent.
_PLACEHOLDER = "—"

_logger = get_logger()

_BLOCK_RE = re.compile(re.escape(INDEX_START) + r".*?" + re.escape(INDEX_END), re.DOTALL)


# Characters that carry structural meaning in GitHub-flavored Markdown or open
# inline HTML. Feed- and LLM-derived text is untrusted (issue #85), so each is
# backslash-escaped; CommonMark renders ``\x`` as a literal ``x``, neutralizing
# emphasis, links, images, code spans, table pipes, and raw HTML (``<`` can no
# longer start a tag) without visibly altering the text. Backslash is first so
# the escapes we add are not themselves re-escaped.
_MD_UNSAFE = "\\`*_[](){}<>&|~"


def _cell(value: str) -> str:
    """Make ``value`` safe inside a Markdown table cell (escape markup, flatten lines)."""
    flattened = value.replace("\n", " ").strip()
    return "".join(f"\\{ch}" if ch in _MD_UNSAFE else ch for ch in flattened)


def _scripture(refs: list[str]) -> str:
    """Join scripture references, or the placeholder when there are none."""
    return ", ".join(refs) if refs else _PLACEHOLDER


def _row(sermon: SermonRecord) -> str:
    """Render one table row; the title links to the sermon's artifact path."""
    church = _cell(church_name(sermon.source))
    title = _cell(sermon.title)
    link = f"[{title}]({sermon.artifact_path})"
    speaker = _cell(sermon.speaker) if sermon.speaker else _PLACEHOLDER
    series = _cell(sermon.series) if sermon.series else _PLACEHOLDER
    scripture = _cell(_scripture(sermon.scripture_refs))
    return f"| {sermon.published_on} | {church} | {link} | {speaker} | {series} | {scripture} |"


def render_index_table(sermons: list[SermonRecord]) -> str:
    """Build the reverse-chronological index table (PRD §7.5).

    Only sermons with an ``artifact_path`` appear, so every row links to a file
    that exists. Rows sort newest-first, breaking ties by ``guid`` for a stable,
    idempotent ordering.
    """
    rendered = [s for s in sermons if s.artifact_path]
    rendered.sort(key=lambda s: (s.published_on, s.guid), reverse=True)
    return "\n".join([_HEADER, _SEPARATOR, *(_row(s) for s in rendered)])


def _block(table: str) -> str:
    """Wrap ``table`` in the index markers."""
    return f"{INDEX_START}\n{table}\n{INDEX_END}"


def update_readme(
    *,
    registry_path: Path = DEFAULT_REGISTRY_PATH,
    readme_path: Path = DEFAULT_README_PATH,
) -> str:
    """Regenerate the README index from the registry, returning the new README text.

    Replaces only the marked region; if the markers are absent (first ever run),
    appends a fresh index section so existing README prose is preserved either way.
    """
    sermons = Registry.load(registry_path).sermons()
    indexed = sum(1 for s in sermons if s.artifact_path)
    block = _block(render_index_table(sermons))
    text = readme_path.read_text(encoding="utf-8")

    if _BLOCK_RE.search(text):
        updated = _BLOCK_RE.sub(lambda _: block, text)
    else:
        separator = "" if text.endswith("\n") else "\n"
        updated = f"{text}{separator}\n## Sermon index\n\n{block}\n"

    readme_path.write_text(updated, encoding="utf-8")
    _logger.info("regenerated README sermon index (%d entries)", indexed)
    return updated
