import httpx
import pytest

from poller import anthropic_client, config


@pytest.fixture(autouse=True)
def _reset_client_singleton():
    anthropic_client._client = None
    yield
    anthropic_client._client = None


def _clear_anthropic_env(monkeypatch):
    for var in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_FEDERATION_RULE_ID",
        "ANTHROPIC_ORGANIZATION_ID",
        "ANTHROPIC_SERVICE_ACCOUNT_ID",
        "ANTHROPIC_WORKSPACE_ID",
        "ACTIONS_ID_TOKEN_REQUEST_URL",
        "ACTIONS_ID_TOKEN_REQUEST_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)


def test_get_client_uses_api_key_when_set(monkeypatch):
    _clear_anthropic_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-123")
    client = anthropic_client._get_client()
    assert client.api_key == "sk-ant-123"


def test_get_client_uses_federation_credentials_when_no_api_key(monkeypatch):
    _clear_anthropic_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_FEDERATION_RULE_ID", "fdrl_123")
    monkeypatch.setenv("ANTHROPIC_ORGANIZATION_ID", "org_123")
    monkeypatch.setenv("ANTHROPIC_SERVICE_ACCOUNT_ID", "svac_123")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_123")
    client = anthropic_client._get_client()
    assert client.api_key is None


def test_get_client_raises_config_error_when_neither_auth_path_set(monkeypatch):
    _clear_anthropic_env(monkeypatch)
    with pytest.raises(config.ConfigError):
        anthropic_client._get_client()


def test_fetch_github_oidc_token_exchanges_the_actions_request_token(monkeypatch):
    _clear_anthropic_env(monkeypatch)
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_URL", "https://actions.example/token")
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "actions-req-token")

    captured = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["headers"] = headers
        return httpx.Response(200, json={"value": "oidc-jwt"}, request=httpx.Request("GET", url))

    monkeypatch.setattr(httpx, "get", fake_get)
    token = anthropic_client._fetch_github_oidc_token()
    assert token == "oidc-jwt"
    assert captured["params"] == {"audience": "https://api.anthropic.com"}
    assert captured["headers"] == {"Authorization": "Bearer actions-req-token"}


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
