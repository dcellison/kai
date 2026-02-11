"""Tests for config.py load_config()."""

import pytest

from kai.config import load_config

# All env vars that load_config reads
_CONFIG_ENV_VARS = [
    "TELEGRAM_BOT_TOKEN",
    "ALLOWED_USER_IDS",
    "CLAUDE_MODEL",
    "CLAUDE_TIMEOUT_SECONDS",
    "CLAUDE_MAX_BUDGET_USD",
    "WEBHOOK_PORT",
    "WEBHOOK_SECRET",
    "VOICE_ENABLED",
    "TTS_ENABLED",
    "WORKSPACE_BASE",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Prevent load_dotenv from reading real .env and clear all config vars."""
    monkeypatch.setattr("kai.config.load_dotenv", lambda *a, **kw: None)
    for var in _CONFIG_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _set_required(monkeypatch, token="fake-token", user_ids="123"):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", token)
    monkeypatch.setenv("ALLOWED_USER_IDS", user_ids)


# ── Happy path ───────────────────────────────────────────────────────


class TestLoadConfigDefaults:
    def test_returns_valid_config(self, monkeypatch):
        _set_required(monkeypatch, user_ids="123,456")
        config = load_config()
        assert config.telegram_bot_token == "fake-token"
        assert config.allowed_user_ids == {123, 456}

    def test_defaults(self, monkeypatch):
        _set_required(monkeypatch)
        config = load_config()
        assert config.claude_model == "sonnet"
        assert config.claude_timeout_seconds == 120
        assert config.claude_max_budget_usd == 10.0
        assert config.webhook_port == 8080
        assert config.webhook_secret == ""
        assert config.voice_enabled is False
        assert config.tts_enabled is False
        assert config.workspace_base is None


# ── Error cases ──────────────────────────────────────────────────────


class TestLoadConfigErrors:
    def test_missing_token(self):
        with pytest.raises(SystemExit, match="TELEGRAM_BOT_TOKEN"):
            load_config()

    def test_missing_user_ids(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
        with pytest.raises(SystemExit, match="ALLOWED_USER_IDS"):
            load_config()

    def test_non_numeric_user_ids(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
        monkeypatch.setenv("ALLOWED_USER_IDS", "notanumber")
        with pytest.raises(SystemExit, match="numeric"):
            load_config()

    def test_workspace_base_nonexistent(self, monkeypatch, tmp_path):
        _set_required(monkeypatch)
        monkeypatch.setenv("WORKSPACE_BASE", str(tmp_path / "nope"))
        with pytest.raises(SystemExit, match="not an existing directory"):
            load_config()


# ── Optional fields ──────────────────────────────────────────────────


class TestLoadConfigOptional:
    def test_voice_enabled_true(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("VOICE_ENABLED", "true")
        assert load_config().voice_enabled is True

    def test_voice_enabled_false(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("VOICE_ENABLED", "false")
        assert load_config().voice_enabled is False

    def test_tts_enabled(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("TTS_ENABLED", "1")
        assert load_config().tts_enabled is True

    def test_workspace_base_valid(self, monkeypatch, tmp_path):
        _set_required(monkeypatch)
        monkeypatch.setenv("WORKSPACE_BASE", str(tmp_path))
        config = load_config()
        assert config.workspace_base == tmp_path

    def test_webhook_secret(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("WEBHOOK_SECRET", "s3cret")
        assert load_config().webhook_secret == "s3cret"
