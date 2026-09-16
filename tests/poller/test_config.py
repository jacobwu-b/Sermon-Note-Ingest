import pytest

from poller import config


def test_load_churches_parses_a_valid_object():
    raw = '{"menlo": {"rss": "https://example.org/feed.xml", "enabled": true, "notify": false}}'
    churches = config.load_churches(raw)
    assert churches["menlo"] == config.ChurchConfig(
        name="menlo", rss="https://example.org/feed.xml", enabled=True, notify=False
    )


def test_load_churches_defaults_missing_flags_to_false():
    churches = config.load_churches('{"menlo": {"rss": "https://example.org/feed.xml"}}')
    assert churches["menlo"].enabled is False
    assert churches["menlo"].notify is False


def test_load_churches_rejects_empty_value():
    with pytest.raises(config.ConfigError):
        config.load_churches("")


def test_load_churches_rejects_invalid_json():
    with pytest.raises(config.ConfigError):
        config.load_churches("{not json")


def test_load_churches_rejects_non_object_top_level():
    with pytest.raises(config.ConfigError):
        config.load_churches("[]")


def test_load_churches_rejects_non_object_entry():
    with pytest.raises(config.ConfigError):
        config.load_churches('{"menlo": "not an object"}')


def test_load_notify_config_reads_all_three_secrets(monkeypatch):
    monkeypatch.setenv("NOTIFY_EMAIL_FROM", "alerts@example.org")
    monkeypatch.setenv("NOTIFY_EMAIL_TO", "team@example.org")
    monkeypatch.setenv("RESEND_API_KEY", "re_123")
    cfg = config.load_notify_config()
    assert cfg == config.NotifyConfig(
        from_addr="alerts@example.org", to_addr="team@example.org", api_key="re_123"
    )


def test_load_notify_config_raises_when_any_secret_missing(monkeypatch):
    monkeypatch.delenv("NOTIFY_EMAIL_FROM", raising=False)
    monkeypatch.setenv("NOTIFY_EMAIL_TO", "team@example.org")
    monkeypatch.setenv("RESEND_API_KEY", "re_123")
    with pytest.raises(config.ConfigError, match="NOTIFY_EMAIL_FROM"):
        config.load_notify_config()


def test_load_whisper_config_defaults_to_small_and_int8(monkeypatch):
    monkeypatch.delenv("WHISPER_MODEL", raising=False)
    monkeypatch.delenv("WHISPER_COMPUTE_TYPE", raising=False)
    monkeypatch.delenv("WHISPER_CPU_THREADS", raising=False)
    assert config.load_whisper_config() == config.WhisperConfig(
        model="small", compute_type="int8", cpu_threads=0
    )


def test_load_whisper_config_reads_overrides(monkeypatch):
    monkeypatch.setenv("WHISPER_MODEL", "medium")
    monkeypatch.setenv("WHISPER_COMPUTE_TYPE", "float32")
    monkeypatch.setenv("WHISPER_CPU_THREADS", "3")
    assert config.load_whisper_config() == config.WhisperConfig(
        model="medium", compute_type="float32", cpu_threads=3
    )


def test_load_content_repo_config_reads_all_three(monkeypatch):
    monkeypatch.setenv("CONTENT_REPO", "owner/sermon-note-content")
    monkeypatch.setenv("CONTENT_REPO_TOKEN", "ghp_123")
    monkeypatch.setenv("CONTENT_REPO_BRANCH", "prod")
    cfg = config.load_content_repo_config()
    assert cfg == config.ContentRepoConfig(repo="owner/sermon-note-content", token="ghp_123", branch="prod")


def test_load_content_repo_config_defaults_branch_to_main(monkeypatch):
    monkeypatch.setenv("CONTENT_REPO", "owner/sermon-note-content")
    monkeypatch.setenv("CONTENT_REPO_TOKEN", "ghp_123")
    monkeypatch.delenv("CONTENT_REPO_BRANCH", raising=False)
    assert config.load_content_repo_config().branch == "main"


def test_load_content_repo_config_raises_when_required_vars_missing(monkeypatch):
    monkeypatch.delenv("CONTENT_REPO", raising=False)
    monkeypatch.delenv("CONTENT_REPO_TOKEN", raising=False)
    with pytest.raises(config.ConfigError, match="CONTENT_REPO"):
        config.load_content_repo_config()


def test_load_pipeline_config_reads_both_required_vars(monkeypatch):
    monkeypatch.setenv("PIPELINE_REPO", "owner/sermon-note-pipeline")
    monkeypatch.setenv("PIPELINE_DISPATCH_TOKEN", "ghp_456")
    cfg = config.load_pipeline_config()
    assert cfg == config.PipelineConfig(repo="owner/sermon-note-pipeline", token="ghp_456")


def test_load_pipeline_config_raises_when_required_vars_missing(monkeypatch):
    monkeypatch.delenv("PIPELINE_REPO", raising=False)
    monkeypatch.delenv("PIPELINE_DISPATCH_TOKEN", raising=False)
    with pytest.raises(config.ConfigError, match="PIPELINE_REPO"):
        config.load_pipeline_config()


def _clear_anthropic_env(monkeypatch):
    for var in (
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_FEDERATION_RULE_ID",
        "ANTHROPIC_ORGANIZATION_ID",
        "ANTHROPIC_SERVICE_ACCOUNT_ID",
        "ANTHROPIC_WORKSPACE_ID",
    ):
        monkeypatch.delenv(var, raising=False)


def test_load_anthropic_config_reads_the_api_key(monkeypatch):
    _clear_anthropic_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-123")
    assert config.load_anthropic_config() == config.AnthropicConfig(
        api_key="sk-ant-123",
        federation_rule_id=None,
        organization_id=None,
        service_account_id=None,
        workspace_id=None,
    )


def test_load_anthropic_config_reads_federation_vars_when_no_api_key(monkeypatch):
    _clear_anthropic_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_FEDERATION_RULE_ID", "fdrl_123")
    monkeypatch.setenv("ANTHROPIC_ORGANIZATION_ID", "org_123")
    monkeypatch.setenv("ANTHROPIC_SERVICE_ACCOUNT_ID", "svac_123")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_123")
    assert config.load_anthropic_config() == config.AnthropicConfig(
        api_key=None,
        federation_rule_id="fdrl_123",
        organization_id="org_123",
        service_account_id="svac_123",
        workspace_id="wrkspc_123",
    )


def test_load_anthropic_config_api_key_wins_over_federation_vars(monkeypatch):
    _clear_anthropic_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-123")
    monkeypatch.setenv("ANTHROPIC_FEDERATION_RULE_ID", "fdrl_123")
    monkeypatch.setenv("ANTHROPIC_ORGANIZATION_ID", "org_123")
    cfg = config.load_anthropic_config()
    assert cfg.api_key == "sk-ant-123"


def test_load_anthropic_config_raises_when_neither_auth_path_set(monkeypatch):
    _clear_anthropic_env(monkeypatch)
    with pytest.raises(config.ConfigError, match="ANTHROPIC_API_KEY"):
        config.load_anthropic_config()


def test_load_anthropic_config_raises_when_federation_rule_missing_organization_id(monkeypatch):
    _clear_anthropic_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_FEDERATION_RULE_ID", "fdrl_123")
    with pytest.raises(config.ConfigError, match="ANTHROPIC_ORGANIZATION_ID"):
        config.load_anthropic_config()


def test_load_log_level_defaults_to_info(monkeypatch):
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    assert config.load_log_level() == "INFO"


def test_load_log_level_reads_override(monkeypatch):
    monkeypatch.setenv("LOG_LEVEL", "debug")
    assert config.load_log_level() == "DEBUG"
