import pytest

from poller import anthropic_client


def test_poll_batch_returns_ended_status_via_injected_fn():
    def fake_poll(batch_id: str) -> anthropic_client.BatchStatus:
        assert batch_id == "batch_123"
        return anthropic_client.BatchStatus(id=batch_id, ended=True, processing_status="ended")

    status = anthropic_client.poll_batch("batch_123", poll_fn=fake_poll)
    assert status.ended is True
    assert status.processing_status == "ended"


def test_poll_batch_returns_not_ended_status_via_injected_fn():
    def fake_poll(batch_id: str) -> anthropic_client.BatchStatus:
        return anthropic_client.BatchStatus(id=batch_id, ended=False, processing_status="in_progress")

    status = anthropic_client.poll_batch("batch_123", poll_fn=fake_poll)
    assert status.ended is False


def test_poll_batch_propagates_transient_error_from_injected_fn():
    def fake_poll(batch_id: str) -> anthropic_client.BatchStatus:
        raise anthropic_client.TransientLLMError("rate limited")

    with pytest.raises(anthropic_client.TransientLLMError):
        anthropic_client.poll_batch("batch_123", poll_fn=fake_poll)


def test_poll_batch_propagates_permanent_error_from_injected_fn():
    def fake_poll(batch_id: str) -> anthropic_client.BatchStatus:
        raise anthropic_client.PermanentLLMError("bad request")

    with pytest.raises(anthropic_client.PermanentLLMError):
        anthropic_client.poll_batch("batch_123", poll_fn=fake_poll)


def test_transient_and_permanent_errors_are_llm_errors():
    assert issubclass(anthropic_client.TransientLLMError, anthropic_client.LLMError)
    assert issubclass(anthropic_client.PermanentLLMError, anthropic_client.LLMError)
