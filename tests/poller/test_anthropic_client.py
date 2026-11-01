import io
import json
import types
import urllib.error
import urllib.parse

import anthropic
import pytest

from poller import anthropic_client, config, net


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
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_URL", "https://pipelines.actions.githubusercontent.com/oidc")
    monkeypatch.setenv("ACTIONS_ID_TOKEN_REQUEST_TOKEN", "actions-req-token")
    client = anthropic_client._get_client()
    assert client.api_key is None


def test_get_client_raises_config_error_when_neither_auth_path_set(monkeypatch):
    _clear_anthropic_env(monkeypatch)
    with pytest.raises(config.ConfigError):
        anthropic_client._get_client()


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_fetch_github_oidc_token_exchanges_the_actions_request_token(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout=None):
        captured["request"] = request
        return _FakeResponse(json.dumps({"count": 1, "value": "oidc-jwt"}).encode())

    monkeypatch.setattr(anthropic_client.urllib.request, "urlopen", fake_urlopen)
    token = anthropic_client._fetch_github_oidc_token(
        "https://pipelines.actions.githubusercontent.com/oidc?api-version=2.0", "actions-req-token"
    )

    request = captured["request"]
    query = urllib.parse.parse_qs(urllib.parse.urlsplit(request.full_url).query)
    assert token == "oidc-jwt"
    assert query == {"api-version": ["2.0"], "audience": ["https://api.anthropic.com"]}
    assert request.get_header("Authorization") == "Bearer actions-req-token"
    assert request.get_header("User-agent") == net.DEFAULT_USER_AGENT


def test_fetch_github_oidc_token_raises_when_github_rejects_the_request(monkeypatch):
    def fake_urlopen(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, io.BytesIO(b""))

    monkeypatch.setattr(anthropic_client.urllib.request, "urlopen", fake_urlopen)
    with pytest.raises(urllib.error.HTTPError):
        anthropic_client._fetch_github_oidc_token("https://pipelines.actions.githubusercontent.com/oidc", "t")


def _status_error(cls, status_code: int) -> anthropic.APIStatusError:
    # APIStatusError reads only these three attributes off the response.
    response = types.SimpleNamespace(request=None, status_code=status_code, headers={})
    return cls(f"HTTP {status_code}", response=response, body=None)


class _FakeClient:
    """Stands in for the SDK client at the boundary: ``messages.batches.retrieve``."""

    def __init__(self, outcomes):
        self.messages = types.SimpleNamespace(batches=types.SimpleNamespace(retrieve=self._retrieve))
        self._outcomes = outcomes

    def _retrieve(self, batch_id):
        outcome = self._outcomes[batch_id]
        if isinstance(outcome, BaseException):
            raise outcome
        return types.SimpleNamespace(id=batch_id, processing_status=outcome)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (anthropic.APIConnectionError(request=None), anthropic_client.TransientLLMError),
        (anthropic.APITimeoutError(request=None), anthropic_client.TransientLLMError),
        (_status_error(anthropic.RateLimitError, 429), anthropic_client.TransientLLMError),
        (_status_error(anthropic.InternalServerError, 500), anthropic_client.TransientLLMError),
        (_status_error(anthropic.OverloadedError, 529), anthropic_client.TransientLLMError),
        (_status_error(anthropic.BadRequestError, 400), anthropic_client.PermanentLLMError),
        (_status_error(anthropic.AuthenticationError, 401), anthropic_client.PermanentLLMError),
        (_status_error(anthropic.PermissionDeniedError, 403), anthropic_client.PermanentLLMError),
        (_status_error(anthropic.NotFoundError, 404), anthropic_client.PermanentLLMError),
        (_status_error(anthropic.UnprocessableEntityError, 422), anthropic_client.PermanentLLMError),
    ],
    ids=lambda v: type(v).__name__ if isinstance(v, BaseException) else v.__name__,
)
def test_default_poll_batch_maps_every_sdk_error_to_an_llm_error(error, expected):
    anthropic_client._client = _FakeClient({"msgbatch_01": error})
    with pytest.raises(expected) as raised:
        anthropic_client._default_poll_batch("msgbatch_01")
    assert raised.value.__cause__ is error


def test_default_poll_batch_reports_an_ended_batch():
    anthropic_client._client = _FakeClient({"msgbatch_01": "ended"})
    status = anthropic_client._default_poll_batch("msgbatch_01")
    assert status == anthropic_client.BatchStatus(id="msgbatch_01", ended=True, processing_status="ended")


def test_a_failing_oidc_fetch_surfaces_as_a_transient_error_through_the_real_sdk():
    # The SDK wraps whatever the identity-token provider raises as APIConnectionError;
    # this pins that, since _default_poll_batch's mapping depends on it.
    def failing_provider() -> str:
        raise urllib.error.URLError("pipelines.actions.githubusercontent.com unreachable")

    anthropic_client._client = anthropic.Anthropic(
        max_retries=0,
        credentials=anthropic.WorkloadIdentityCredentials(
            identity_token_provider=failing_provider,
            federation_rule_id="fdrl_123",
            organization_id="org_123",
        ),
    )
    with pytest.raises(anthropic_client.TransientLLMError):
        anthropic_client._default_poll_batch("msgbatch_01")


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
