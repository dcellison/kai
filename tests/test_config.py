"""Tests for config.py load_config()."""

import pytest

from kai.config import load_config

# All env vars that load_config reads
_CONFIG_ENV_VARS = [
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_WEBHOOK_URL",
    "TELEGRAM_WEBHOOK_SECRET",
    "ALLOWED_USER_IDS",
    "CLAUDE_MODEL",
    "CLAUDE_TIMEOUT_SECONDS",
    "CLAUDE_MAX_BUDGET_USD",
    "WEBHOOK_PORT",
    "WEBHOOK_SECRET",
    "VOICE_ENABLED",
    "TTS_ENABLED",
    "WORKSPACE_BASE",
    "ALLOWED_WORKSPACES",
    "CLAUDE_USER",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Prevent load_dotenv from reading real .env and clear all config vars."""
    monkeypatch.setattr("kai.config.load_dotenv", lambda *a, **kw: None)
    for var in _CONFIG_ENV_VARS:
        monkeypatch.delenv(var, raising=False)


def _set_required(monkeypatch, token="fake-token", user_ids="123"):
    """Set only the truly required env vars (token + user IDs).

    TELEGRAM_WEBHOOK_URL is no longer required - omitting it selects polling mode.
    Tests that need webhook mode should set it explicitly.
    """
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
        # Without TELEGRAM_WEBHOOK_URL, defaults to polling mode
        assert config.telegram_webhook_url is None
        assert config.telegram_webhook_secret is None
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

    def test_allowed_workspaces_default_empty(self, monkeypatch):
        _set_required(monkeypatch)
        assert load_config().allowed_workspaces == []

    def test_allowed_workspaces_single(self, monkeypatch, tmp_path):
        _set_required(monkeypatch)
        monkeypatch.setenv("ALLOWED_WORKSPACES", str(tmp_path))
        config = load_config()
        assert config.allowed_workspaces == [tmp_path]

    def test_allowed_workspaces_multiple(self, monkeypatch, tmp_path):
        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()
        _set_required(monkeypatch)
        monkeypatch.setenv("ALLOWED_WORKSPACES", f"{dir_a},{dir_b}")
        config = load_config()
        assert config.allowed_workspaces == [dir_a, dir_b]

    def test_allowed_workspaces_skips_nonexistent(self, monkeypatch, tmp_path):
        # Non-existent paths are skipped with a warning, not a crash
        real_dir = tmp_path / "real"
        real_dir.mkdir()
        fake_dir = tmp_path / "nope"
        _set_required(monkeypatch)
        monkeypatch.setenv("ALLOWED_WORKSPACES", f"{real_dir},{fake_dir}")
        config = load_config()
        assert config.allowed_workspaces == [real_dir]

    def test_allowed_workspaces_all_nonexistent_returns_empty(self, monkeypatch, tmp_path):
        _set_required(monkeypatch)
        monkeypatch.setenv("ALLOWED_WORKSPACES", str(tmp_path / "nope"))
        assert load_config().allowed_workspaces == []

    def test_allowed_workspaces_deduplicates(self, monkeypatch, tmp_path):
        # Same path listed twice should appear only once
        dir_a = tmp_path / "a"
        dir_a.mkdir()
        _set_required(monkeypatch)
        monkeypatch.setenv("ALLOWED_WORKSPACES", f"{dir_a},{dir_a}")
        config = load_config()
        assert len(config.allowed_workspaces) == 1
        assert config.allowed_workspaces[0] == dir_a

    def test_allowed_workspaces_deduplicates_canonical_paths(self, monkeypatch, tmp_path):
        # /a/b and /a/../a/b resolve to the same path - only one entry
        dir_a = tmp_path / "a"
        dir_a.mkdir()
        non_canonical = tmp_path / "." / "a"
        _set_required(monkeypatch)
        monkeypatch.setenv("ALLOWED_WORKSPACES", f"{dir_a},{non_canonical}")
        config = load_config()
        assert len(config.allowed_workspaces) == 1

    def test_claude_user_default_none(self, monkeypatch):
        _set_required(monkeypatch)
        assert load_config().claude_user is None

    def test_claude_user_from_env(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("CLAUDE_USER", "daniel")
        assert load_config().claude_user == "daniel"

    def test_claude_user_empty_string_becomes_none(self, monkeypatch):
        # Empty CLAUDE_USER is treated as unset (the `or None` coercion)
        _set_required(monkeypatch)
        monkeypatch.setenv("CLAUDE_USER", "")
        assert load_config().claude_user is None


# ── Telegram webhook config ─────────────────────────────────────────


class TestTelegramWebhookConfig:
    def test_missing_webhook_url_selects_polling(self, monkeypatch):
        """Without TELEGRAM_WEBHOOK_URL, config defaults to polling mode."""
        _set_required(monkeypatch)
        config = load_config()
        assert config.telegram_webhook_url is None
        assert config.telegram_webhook_secret is None

    def test_webhook_url_set_selects_webhook_mode(self, monkeypatch):
        """With TELEGRAM_WEBHOOK_URL set, config selects webhook mode."""
        _set_required(monkeypatch)
        monkeypatch.setenv("TELEGRAM_WEBHOOK_URL", "https://example.com/webhook/telegram")
        monkeypatch.setenv("WEBHOOK_SECRET", "shared-secret")
        config = load_config()
        assert config.telegram_webhook_url == "https://example.com/webhook/telegram"

    def test_secret_defaults_to_webhook_secret(self, monkeypatch):
        """TELEGRAM_WEBHOOK_SECRET falls back to WEBHOOK_SECRET when unset."""
        _set_required(monkeypatch)
        monkeypatch.setenv("TELEGRAM_WEBHOOK_URL", "https://example.com/webhook/telegram")
        monkeypatch.setenv("WEBHOOK_SECRET", "shared-secret")
        # TELEGRAM_WEBHOOK_SECRET deliberately not set
        config = load_config()
        assert config.telegram_webhook_secret == "shared-secret"

    def test_explicit_secret_overrides_fallback(self, monkeypatch):
        """TELEGRAM_WEBHOOK_SECRET uses its own value when explicitly set."""
        _set_required(monkeypatch)
        monkeypatch.setenv("TELEGRAM_WEBHOOK_URL", "https://example.com/webhook/telegram")
        monkeypatch.setenv("WEBHOOK_SECRET", "shared-secret")
        monkeypatch.setenv("TELEGRAM_WEBHOOK_SECRET", "tg-only-secret")
        config = load_config()
        assert config.telegram_webhook_secret == "tg-only-secret"

    def test_webhook_url_without_secret_raises(self, monkeypatch):
        """Webhook mode with no secret is rejected to prevent open endpoint."""
        _set_required(monkeypatch)
        monkeypatch.setenv("TELEGRAM_WEBHOOK_URL", "https://example.com/webhook/telegram")
        # Neither TELEGRAM_WEBHOOK_SECRET nor WEBHOOK_SECRET set
        with pytest.raises(SystemExit, match="TELEGRAM_WEBHOOK_SECRET"):
            load_config()

    def test_polling_mode_ignores_missing_secret(self, monkeypatch):
        """In polling mode, missing secrets are fine (no webhook to protect)."""
        _set_required(monkeypatch)
        # No TELEGRAM_WEBHOOK_URL, no secrets
        config = load_config()
        assert config.telegram_webhook_url is None
        assert config.telegram_webhook_secret is None
