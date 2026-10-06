"""Publish ``status/ingest.json``: Ingest's half of the web status report (spec 0009).

A pure projection of every church's ledger plus the church table, in the shape
Sermon-Note-Pipeline's spec 0029 defines. The web app joins it by guid with
Pipeline's own ``status/pipeline.json`` and derives every reason and metric itself.

No timestamp in the file: an unchanged ledger renders byte-identical text, so the
push is a no-op on the many ticks that change nothing.

``python -m poller.status`` builds and pushes it; non-zero if the push never lands.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Callable
from typing import Any

from poller import config, content_repo, net, store

logger = logging.getLogger("poller")

STATUS_PATH = "status/ingest.json"


def _project(source: str, record: dict[str, Any]) -> dict[str, Any]:
    return {
        "guid": record["guid"],
        "source": source,
        "title": record.get("title"),
        "preached_on": record.get("preached_on"),
        "feed_published_at": record.get("feed_published_at"),
        "first_seen_at": record["first_seen_at"],
        "audio_available": net.is_fetchable_enclosure(record.get("audio_url") or ""),
        "transcription_status": record.get("transcription_status"),
        "transcribed_at": record.get("transcribed_at"),
        "transcript_adopted": record.get("transcript_adopted"),
        "last_error": record.get("transcription_last_error"),
    }


def build_status(churches: dict[str, config.ChurchConfig]) -> dict[str, Any]:
    """Every church in the table, enabled or not, and every record each one has ledgered."""
    return {
        "churches": {name: {"ingest_enabled": church.enabled} for name, church in churches.items()},
        "sermons": [_project(name, record) for name in churches for record in store.load(name).values()],
    }


def render_status(churches: dict[str, config.ChurchConfig]) -> str:
    return json.dumps(build_status(churches), indent=2, ensure_ascii=False) + "\n"


def main(
    argv: list[str] | None = None,
    *,
    push: Callable[[dict[str, str]], None] = content_repo.push_status,
) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    logging.basicConfig(
        level=config.load_log_level(),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    text = render_status(config.load_churches())
    try:
        push({STATUS_PATH: text})
    except content_repo.ContentPublishError as exc:
        logger.error("status push to Content failed: %s", exc)
        return 1
    logger.info("published %s (%d bytes)", STATUS_PATH, len(text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
