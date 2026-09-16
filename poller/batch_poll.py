"""Checks Pipeline's pending Claude batches and dispatches Pipeline early on completion.

A latency optimization on top of Sermon-Note-Pipeline's own cron, never a
correctness dependency (docs/decisions/0010): every failure here — missing
config, an unreadable registry, an unreachable Anthropic API, a rejected
dispatch — is logged and swallowed, exactly the same non-fatal contract
poller/pipeline_dispatch.py already established for the sermon_detected
dispatch (spec 0005). Pipeline's own ``_poll_pending_batches()`` re-confirms
and resolves the batch itself on the run this triggers; this module only
decides whether it's worth waking that run up early.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from poller import anthropic_client, config, pipeline_dispatch, pipeline_registry
from poller.config import PipelineConfig

logger = logging.getLogger("poller")


def poll_pending_batches(
    *,
    list_pending: Callable[
        ..., list[pipeline_registry.PendingBatch]
    ] = pipeline_registry.list_pending_batches,
    poll_batch: Callable[[str], anthropic_client.BatchStatus] = anthropic_client.poll_batch,
    dispatch: Callable[..., None] = pipeline_dispatch.dispatch_ingest_event,
) -> None:
    """Check every batch Pipeline is waiting on; dispatch Pipeline early for each newly-ended one."""
    try:
        config.load_anthropic_config()
        pipeline_cfg: PipelineConfig = config.load_pipeline_config()
    except config.ConfigError as exc:
        logger.warning("batch poll skipped: %s", exc)
        return

    try:
        pending = list_pending(config=pipeline_cfg)
    except pipeline_registry.PipelineRegistryError as exc:
        logger.warning("could not read Pipeline's registry; skipping batch poll: %s", exc)
        return

    dispatched_batch_ids: set[str] = set()
    for record in pending:
        if record.batch_id in dispatched_batch_ids:
            continue
        try:
            status = poll_batch(record.batch_id)
        except anthropic_client.LLMError as exc:
            logger.warning("%s: could not poll batch %s: %s", record.source, record.batch_id, exc)
            continue
        if not status.ended:
            continue

        event = {
            "event": "batch_ended",
            "source": record.source,
            "external_id": record.guid,
            "batch_id": record.batch_id,
        }
        try:
            dispatch(event, source=record.source, config=pipeline_cfg)
        except pipeline_dispatch.PipelineDispatchError as exc:
            logger.warning(
                "%s: failed to dispatch early for ended batch %s: %s", record.source, record.batch_id, exc
            )
            continue
        dispatched_batch_ids.add(record.batch_id)
        logger.info("%s: batch %s ended; dispatched pipeline.yml early", record.source, record.batch_id)
