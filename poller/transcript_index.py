"""Render ``transcripts/index.json``: each transcript's metadata for the web corpus (spec 0010).

A pure projection of every church's done records. Web reads only from Content, and
title, speaker, series, episode URL and blurb live only in this repo's ledgers, so
this is how they reach it. ``path`` is the record's ``content_path`` exactly as
pushed; Web loads it as written and never builds one. Metadata only, never
transcript text (ADR-0004).

Published by ``python -m poller.status`` in the same push as ``status/ingest.json``
(ADR-0020). No timestamp: an unchanged ledger renders byte-identical text.
"""

from __future__ import annotations

import json
from typing import Any

from poller import config, store

INDEX_PATH = "transcripts/index.json"


def _project(source: str, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "guid": record["guid"],
        "source": source,
        "preached_on": record.get("preached_on"),
        "title": record.get("title"),
        "speaker": record.get("speaker"),
        "series": record.get("series"),
        "episode_url": record.get("episode_url"),
        "blurb": record.get("blurb"),
        # Indexed, not .get(): a done record naming no file must fail the render, not publish null.
        "path": record["content_path"],
    }


def build_index(churches: dict[str, config.ChurchConfig]) -> dict[str, Any]:
    """Every done record of every church in the table, enabled or not."""
    return {
        "transcripts": [
            _project(name, record)
            for name in churches
            for record in store.load(name).values()
            if record.get("transcription_status") == "done"
        ]
    }


def render_index(churches: dict[str, config.ChurchConfig]) -> str:
    return json.dumps(build_index(churches), indent=2, ensure_ascii=False) + "\n"
