from poller import anthropic_client, batch_poll, config, pipeline_registry


def _config() -> config.PipelineConfig:
    return config.PipelineConfig(repo="owner/sermon-note-pipeline", token="ghp_456")


def _pending(guid: str, source: str, batch_id: str) -> pipeline_registry.PendingBatch:
    return pipeline_registry.PendingBatch(guid=guid, source=source, batch_id=batch_id)


def _status(batch_id: str, *, ended: bool) -> anthropic_client.BatchStatus:
    return anthropic_client.BatchStatus(
        id=batch_id, ended=ended, processing_status="ended" if ended else "in_progress"
    )


def test_poll_pending_batches_dispatches_only_for_ended_batches(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-123")
    monkeypatch.setenv("PIPELINE_REPO", "owner/sermon-note-pipeline")
    monkeypatch.setenv("PIPELINE_DISPATCH_TOKEN", "ghp_456")

    pending = [_pending("g1", "menlo", "batch_1"), _pending("g2", "north_point", "batch_2")]
    statuses = {"batch_1": _status("batch_1", ended=True), "batch_2": _status("batch_2", ended=False)}
    dispatched = []

    batch_poll.poll_pending_batches(
        list_pending=lambda *, config: pending,
        poll_batch=lambda batch_id: statuses[batch_id],
        dispatch=lambda event, *, source, config: dispatched.append((event, source)),
    )

    assert len(dispatched) == 1
    event, source = dispatched[0]
    assert source == "menlo"
    assert event["source"] == "menlo"
    assert event["external_id"] == "g1"
    assert event["batch_id"] == "batch_1"


def test_poll_pending_batches_dedupes_the_same_batch_id_within_one_run(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-123")
    monkeypatch.setenv("PIPELINE_REPO", "owner/sermon-note-pipeline")
    monkeypatch.setenv("PIPELINE_DISPATCH_TOKEN", "ghp_456")

    pending = [_pending("g1", "menlo", "batch_1"), _pending("g2", "menlo", "batch_1")]
    dispatched = []

    batch_poll.poll_pending_batches(
        list_pending=lambda *, config: pending,
        poll_batch=lambda batch_id: _status(batch_id, ended=True),
        dispatch=lambda event, *, source, config: dispatched.append(event),
    )

    assert len(dispatched) == 1


def test_poll_pending_batches_is_a_noop_with_no_pending_batches(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-123")
    monkeypatch.setenv("PIPELINE_REPO", "owner/sermon-note-pipeline")
    monkeypatch.setenv("PIPELINE_DISPATCH_TOKEN", "ghp_456")

    def fail_poll(batch_id):
        raise AssertionError("must not poll Anthropic when nothing is pending")

    batch_poll.poll_pending_batches(
        list_pending=lambda *, config: [],
        poll_batch=fail_poll,
        dispatch=lambda event, *, source, config: (_ for _ in ()).throw(AssertionError("must not dispatch")),
    )


def test_poll_pending_batches_skips_config_gracefully_when_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("PIPELINE_REPO", raising=False)
    monkeypatch.delenv("PIPELINE_DISPATCH_TOKEN", raising=False)

    def fail_list_pending(*, config):
        raise AssertionError("must not read the registry without config")

    # Must not raise.
    batch_poll.poll_pending_batches(list_pending=fail_list_pending)


def test_poll_pending_batches_continues_past_a_registry_read_failure(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-123")
    monkeypatch.setenv("PIPELINE_REPO", "owner/sermon-note-pipeline")
    monkeypatch.setenv("PIPELINE_DISPATCH_TOKEN", "ghp_456")

    def fail_list_pending(*, config):
        raise pipeline_registry.PipelineRegistryError("boom")

    # Must not raise.
    batch_poll.poll_pending_batches(list_pending=fail_list_pending)


def test_poll_pending_batches_continues_past_one_batchs_anthropic_failure(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-123")
    monkeypatch.setenv("PIPELINE_REPO", "owner/sermon-note-pipeline")
    monkeypatch.setenv("PIPELINE_DISPATCH_TOKEN", "ghp_456")

    pending = [_pending("g1", "menlo", "batch_1"), _pending("g2", "north_point", "batch_2")]
    dispatched = []

    def flaky_poll(batch_id):
        if batch_id == "batch_1":
            raise anthropic_client.TransientLLMError("rate limited")
        return _status(batch_id, ended=True)

    batch_poll.poll_pending_batches(
        list_pending=lambda *, config: pending,
        poll_batch=flaky_poll,
        dispatch=lambda event, *, source, config: dispatched.append(event["batch_id"]),
    )

    assert dispatched == ["batch_2"]


def test_poll_pending_batches_continues_past_a_dispatch_failure(monkeypatch):
    from poller import pipeline_dispatch

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-123")
    monkeypatch.setenv("PIPELINE_REPO", "owner/sermon-note-pipeline")
    monkeypatch.setenv("PIPELINE_DISPATCH_TOKEN", "ghp_456")

    pending = [_pending("g1", "menlo", "batch_1"), _pending("g2", "north_point", "batch_2")]
    dispatched = []

    def flaky_dispatch(event, *, source, config):
        if event["batch_id"] == "batch_1":
            raise pipeline_dispatch.PipelineDispatchError("rejected")
        dispatched.append(event["batch_id"])

    batch_poll.poll_pending_batches(
        list_pending=lambda *, config: pending,
        poll_batch=lambda batch_id: _status(batch_id, ended=True),
        dispatch=flaky_dispatch,
    )

    assert dispatched == ["batch_2"]
