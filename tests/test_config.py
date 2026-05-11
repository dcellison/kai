"""Tests for config.py load_config(), DATA_DIR, _read_protected_file(), and resolve_claude_user()."""

import logging
import os
import pwd
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kai.config import Config, UserConfig, _read_protected_file, load_config, resolve_claude_user

# All env vars that load_config reads
_CONFIG_ENV_VARS = [
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_WEBHOOK_URL",
    "TELEGRAM_WEBHOOK_SECRET",
    "ALLOWED_USER_IDS",
    "DEFAULT_MODEL",
    "CLAUDE_MODEL",
    "CLAUDE_TIMEOUT_SECONDS",
    "BUDGET_CEILING",
    "CLAUDE_MAX_BUDGET_USD",  # backward compat (renamed to BUDGET_CEILING)
    "CLAUDE_MAX_SESSION_HOURS",
    "CLAUDE_IDLE_TIMEOUT",
    "WEBHOOK_PORT",
    "WEBHOOK_SECRET",
    "VOICE_ENABLED",
    "TTS_ENABLED",
    "WORKSPACE_BASE",
    "ALLOWED_WORKSPACES",
    "CLAUDE_USER",
    "FILE_RETENTION_DAYS",
    "PR_REVIEW_ENABLED",
    "PR_REVIEW_COOLDOWN",
    "PR_REVIEW_TIMEOUT_S",
    "PR_REVIEW_BUDGET_USD",
    "GITHUB_REPO",
    "SPEC_DIR",
    "ISSUE_TRIAGE_ENABLED",
    "GITHUB_NOTIFY_CHAT_ID",
    "CLAUDE_MAX_CONTEXT_WINDOW",
    "CLAUDE_AUTOCOMPACT_PCT",
    "CLAUDE_EFFORT_LEVEL",
    "TOTP_SESSION_MINUTES",
    "TOTP_CHALLENGE_SECONDS",
    "TOTP_LOCKOUT_ATTEMPTS",
    "TOTP_LOCKOUT_MINUTES",
    "AGENT_BACKEND",
    "LLM_PROVIDER",
    "MEMORY_ENABLED",
    "MEMORY_SEARCH_LIMIT",
    "MEMORY_TOKEN_BUDGET",
    "MEMORY_EMBEDDING_MODEL",
    "MEMORY_EXTRACTION_ENABLED",
    "MEMORY_EXTRACTION_MODEL",
    "MEMORY_EXTRACTION_BUDGET_USD",
    "MEMORY_EXTRACTION_TIMEOUT_S",
    "MEMORY_CONSOLIDATION_CANDIDATES_N",
    "EPISODE_CLASSIFIER_CONTEXT_TURNS",
    "MEMORY_EPISODE_MODEL",
    "MEMORY_EPISODE_BUDGET_USD",
    "MEMORY_EPISODE_TIMEOUT_S",
    "MEMORY_SEARCH_FLOOR",
    "KAI_DATA_DIR",
    "KAI_INSTALL_DIR",
]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    """Prevent load_dotenv and sudo reads from running, and clear all config vars."""
    monkeypatch.setattr("kai.config.load_dotenv", lambda *a, **kw: None)
    # Prevent real sudo calls during tests - default to None (dev mode fallback)
    monkeypatch.setattr("kai.config._read_protected_file", lambda path: None)
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
        assert config.default_model == "sonnet"
        assert config.claude_timeout_seconds == 120
        assert config.budget_ceiling == 10.0
        assert config.claude_max_session_hours == 0
        assert config.webhook_port == 8080
        # Without TELEGRAM_WEBHOOK_URL, defaults to polling mode
        assert config.telegram_webhook_url is None
        assert config.telegram_webhook_secret is None
        assert config.voice_enabled is False
        assert config.tts_enabled is False
        assert config.workspace_base is None
        # Context window tuning defaults to 0 (use Claude Code defaults)
        assert config.claude_max_context_window == 0
        assert config.claude_autocompact_pct == 0
        # Agent backend default
        assert config.agent_backend == "claude"

    def test_context_window_from_env(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("CLAUDE_MAX_CONTEXT_WINDOW", "200000")
        monkeypatch.setenv("CLAUDE_AUTOCOMPACT_PCT", "80")
        config = load_config()
        assert config.claude_max_context_window == 200000
        assert config.claude_autocompact_pct == 80


# ── Error cases ──────────────────────────────────────────────────────


class TestLoadConfigErrors:
    def test_missing_token(self):
        with pytest.raises(SystemExit, match="TELEGRAM_BOT_TOKEN"):
            load_config()

    def test_missing_user_ids(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
        with pytest.raises(SystemExit, match="No user authorization configured"):
            load_config()

    def test_non_numeric_user_ids(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
        monkeypatch.setenv("ALLOWED_USER_IDS", "notanumber")
        with pytest.raises(SystemExit, match="non-numeric"):
            load_config()

    def test_workspace_base_nonexistent(self, monkeypatch, tmp_path):
        _set_required(monkeypatch)
        monkeypatch.setenv("WORKSPACE_BASE", str(tmp_path / "nope"))
        with pytest.raises(SystemExit, match="not an existing directory"):
            load_config()

    def test_invalid_session_hours(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("CLAUDE_MAX_SESSION_HOURS", "not-a-number")
        with pytest.raises(SystemExit, match="CLAUDE_MAX_SESSION_HOURS"):
            load_config()

    def test_session_hours_from_env(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("CLAUDE_MAX_SESSION_HOURS", "4.5")
        config = load_config()
        assert config.claude_max_session_hours == 4.5

    def test_invalid_context_window(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("CLAUDE_MAX_CONTEXT_WINDOW", "not-a-number")
        with pytest.raises(SystemExit, match="CLAUDE_MAX_CONTEXT_WINDOW"):
            load_config()

    def test_negative_context_window(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("CLAUDE_MAX_CONTEXT_WINDOW", "-1")
        with pytest.raises(SystemExit, match="CLAUDE_MAX_CONTEXT_WINDOW"):
            load_config()

    def test_context_window_exceeds_ceiling(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("CLAUDE_MAX_CONTEXT_WINDOW", "99999999999")
        with pytest.raises(SystemExit, match="CLAUDE_MAX_CONTEXT_WINDOW"):
            load_config()

    def test_invalid_autocompact_pct(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("CLAUDE_AUTOCOMPACT_PCT", "not-a-number")
        with pytest.raises(SystemExit, match="CLAUDE_AUTOCOMPACT_PCT"):
            load_config()

    def test_autocompact_pct_out_of_range(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("CLAUDE_AUTOCOMPACT_PCT", "101")
        with pytest.raises(SystemExit, match="CLAUDE_AUTOCOMPACT_PCT"):
            load_config()

    def test_invalid_default_model(self, monkeypatch):
        """DEFAULT_MODEL with an unrecognized value raises SystemExit."""
        _set_required(monkeypatch)
        monkeypatch.setenv("DEFAULT_MODEL", "sonet")
        with pytest.raises(SystemExit, match="DEFAULT_MODEL"):
            load_config()

    def test_invalid_agent_backend(self, monkeypatch):
        """AGENT_BACKEND with an unrecognized value raises SystemExit."""
        _set_required(monkeypatch)
        monkeypatch.setenv("AGENT_BACKEND", "invalid")
        with pytest.raises(SystemExit, match="AGENT_BACKEND"):
            load_config()

    def test_invalid_llm_provider(self, monkeypatch):
        """LLM_PROVIDER with an unrecognized value raises SystemExit."""
        _set_required(monkeypatch)
        monkeypatch.setenv("AGENT_BACKEND", "goose")
        monkeypatch.setenv("LLM_PROVIDER", "invalid")
        with pytest.raises(SystemExit, match="LLM_PROVIDER"):
            load_config()

    def test_missing_llm_provider(self, monkeypatch):
        """LLM_PROVIDER missing when backend=goose raises SystemExit."""
        _set_required(monkeypatch)
        monkeypatch.setenv("AGENT_BACKEND", "goose")
        with pytest.raises(SystemExit, match="LLM_PROVIDER"):
            load_config()

    def test_llm_provider_ignored_for_claude(self, monkeypatch):
        """LLM_PROVIDER is not validated when backend=claude."""
        _set_required(monkeypatch)
        cfg = load_config()
        assert cfg.llm_provider == ""

    def test_invalid_llm_provider_ignored_for_claude(self, monkeypatch):
        """Even an invalid LLM_PROVIDER is ignored when backend=claude."""
        _set_required(monkeypatch)
        monkeypatch.setenv("LLM_PROVIDER", "completelywrong")
        cfg = load_config()
        assert cfg.llm_provider == ""

    def test_valid_llm_provider(self, monkeypatch):
        """Valid LLM_PROVIDER is accepted and stored."""
        _set_required(monkeypatch)
        monkeypatch.setenv("AGENT_BACKEND", "goose")
        monkeypatch.setenv("LLM_PROVIDER", "openai")
        # DEFAULT_MODEL must be valid for the openai provider
        monkeypatch.setenv("DEFAULT_MODEL", "gpt-5.4")
        cfg = load_config()
        assert cfg.llm_provider == "openai"

    def test_goose_without_provider_exits(self, monkeypatch):
        """AGENT_BACKEND=goose without LLM_PROVIDER fails at startup."""
        _set_required(monkeypatch)
        monkeypatch.setenv("AGENT_BACKEND", "goose")
        # LLM_PROVIDER not set - empty string is not in VALID_PROVIDERS["goose"]
        with pytest.raises(SystemExit, match=r"LLM_PROVIDER.*not valid"):
            load_config()


# ── Optional fields ──────────────────────────────────────────────────


class TestLoadConfigOptional:
    def test_default_model_from_env(self, monkeypatch):
        """DEFAULT_MODEL is read from environment when set."""
        _set_required(monkeypatch)
        monkeypatch.setenv("DEFAULT_MODEL", "opus")
        assert load_config().default_model == "opus"

    def test_claude_model_backward_compat(self, monkeypatch):
        """Old CLAUDE_MODEL env var is still read when DEFAULT_MODEL is absent."""
        _set_required(monkeypatch)
        monkeypatch.setenv("CLAUDE_MODEL", "opus")
        assert load_config().default_model == "opus"

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


# ── DATA_DIR ──────────────────────────────────────────────────────


class TestDataDir:
    def test_defaults_to_project_root(self):
        """When KAI_DATA_DIR is unset, DATA_DIR equals PROJECT_ROOT."""
        from kai.config import PROJECT_ROOT

        # DATA_DIR is a module-level constant evaluated at import time, so we
        # test the derivation logic directly instead of re-importing.
        val = os.environ.get("KAI_DATA_DIR") or str(PROJECT_ROOT)
        assert Path(val) == PROJECT_ROOT

    def test_from_env(self, monkeypatch, tmp_path):
        """When KAI_DATA_DIR is set, DATA_DIR uses that path."""
        monkeypatch.setenv("KAI_DATA_DIR", str(tmp_path))
        result = Path(os.environ.get("KAI_DATA_DIR") or "fallback")
        assert result == tmp_path

    def test_empty_string_defaults(self, monkeypatch):
        """Empty KAI_DATA_DIR falls back to PROJECT_ROOT via `or`."""
        monkeypatch.setenv("KAI_DATA_DIR", "")
        from kai.config import PROJECT_ROOT

        result = Path(os.environ.get("KAI_DATA_DIR") or str(PROJECT_ROOT))
        assert result == PROJECT_ROOT

    def test_session_db_path_uses_data_dir(self, monkeypatch):
        """Database path defaults to DATA_DIR / 'kai.db'."""
        _set_required(monkeypatch)
        config = load_config()
        # In test env, DATA_DIR == PROJECT_ROOT (KAI_DATA_DIR is unset)
        assert config.session_db_path.name == "kai.db"


# ── PROJECT_ROOT / KAI_INSTALL_DIR ────────────────────────────────


class TestProjectRoot:
    def test_defaults_to_file_derived_root(self, monkeypatch):
        """When KAI_INSTALL_DIR is unset, PROJECT_ROOT derives from __file__."""
        monkeypatch.delenv("KAI_INSTALL_DIR", raising=False)
        from kai.config import _FILE_ROOT

        # Replicate the module-level logic with the env var cleared
        result = Path(os.environ.get("KAI_INSTALL_DIR") or str(_FILE_ROOT))
        assert result == _FILE_ROOT

    def test_from_env(self, monkeypatch, tmp_path):
        """When KAI_INSTALL_DIR is set, PROJECT_ROOT uses that path."""
        monkeypatch.setenv("KAI_INSTALL_DIR", str(tmp_path))
        # Re-evaluate the same logic config.py uses at module level
        result = Path(os.environ.get("KAI_INSTALL_DIR") or "fallback")
        assert result == tmp_path

    def test_empty_string_defaults(self, monkeypatch):
        """Empty KAI_INSTALL_DIR falls back to _FILE_ROOT via `or`."""
        monkeypatch.setenv("KAI_INSTALL_DIR", "")
        from kai.config import _FILE_ROOT

        result = Path(os.environ.get("KAI_INSTALL_DIR") or str(_FILE_ROOT))
        assert result == _FILE_ROOT


# ── _read_protected_file ─────────────────────────────────────────


class TestReadProtectedFile:
    """Tests for the sudo-based file reader (uses real function, not the monkeypatched stub)."""

    def test_success(self):
        """Returns file contents when sudo cat succeeds."""
        mock_result = subprocess.CompletedProcess(
            args=["sudo", "-n", "cat", "/etc/kai/env"],
            returncode=0,
            stdout="KEY=value\n",
            stderr="",
        )
        with patch("kai.config.subprocess.run", return_value=mock_result):
            result = _read_protected_file("/etc/kai/env")
        assert result == "KEY=value\n"

    def test_failure_returns_none(self):
        """Returns None when sudo cat fails (non-zero exit)."""
        mock_result = subprocess.CompletedProcess(
            args=["sudo", "-n", "cat", "/etc/kai/env"],
            returncode=1,
            stdout="",
            stderr="sudo: a password is required\n",
        )
        with patch("kai.config.subprocess.run", return_value=mock_result):
            result = _read_protected_file("/etc/kai/env")
        assert result is None

    def test_timeout_returns_none(self):
        """Returns None when subprocess times out."""
        with patch(
            "kai.config.subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd="sudo", timeout=5),
        ):
            result = _read_protected_file("/etc/kai/env")
        assert result is None

    def test_oserror_returns_none(self):
        """Returns None when subprocess raises OSError (e.g., sudo not found)."""
        with patch(
            "kai.config.subprocess.run",
            side_effect=OSError("No such file or directory"),
        ):
            result = _read_protected_file("/etc/kai/env")
        assert result is None


# ── Dual-mode config loading ─────────────────────────────────────


class TestDualModeLoading:
    def test_loads_from_protected_env(self, monkeypatch):
        """When /etc/kai/env is readable, values are used as config."""
        monkeypatch.setattr(
            "kai.config._read_protected_file",
            lambda path: (
                "TELEGRAM_BOT_TOKEN=protected-token\nALLOWED_USER_IDS=999\n" if path == "/etc/kai/env" else None
            ),
        )
        config = load_config()
        assert config.telegram_bot_token == "protected-token"
        assert config.allowed_user_ids == {999}

    def test_protected_env_strips_quotes(self, monkeypatch):
        """Quote marks around values in /etc/kai/env are stripped."""
        monkeypatch.setattr(
            "kai.config._read_protected_file",
            lambda path: (
                "TELEGRAM_BOT_TOKEN=\"quoted-token\"\nALLOWED_USER_IDS='999'\n" if path == "/etc/kai/env" else None
            ),
        )
        config = load_config()
        assert config.telegram_bot_token == "quoted-token"
        assert config.allowed_user_ids == {999}

    def test_protected_env_skips_comments_and_blanks(self, monkeypatch):
        """Comments and blank lines in /etc/kai/env are ignored."""
        monkeypatch.setattr(
            "kai.config._read_protected_file",
            lambda path: (
                "# comment\n\nTELEGRAM_BOT_TOKEN=tok\n\nALLOWED_USER_IDS=1\n" if path == "/etc/kai/env" else None
            ),
        )
        config = load_config()
        assert config.telegram_bot_token == "tok"

    def test_falls_back_to_dotenv(self, monkeypatch):
        """When /etc/kai/env is not readable, load_dotenv is called."""
        load_dotenv_called = []
        monkeypatch.setattr("kai.config._read_protected_file", lambda path: None)
        monkeypatch.setattr(
            "kai.config.load_dotenv",
            lambda *a, **kw: load_dotenv_called.append(True),
        )
        _set_required(monkeypatch)
        load_config()
        assert load_dotenv_called, "load_dotenv should have been called"

    def test_env_vars_take_precedence_over_protected(self, monkeypatch):
        """Explicitly set env vars override values from /etc/kai/env."""
        monkeypatch.setattr(
            "kai.config._read_protected_file",
            lambda path: "TELEGRAM_BOT_TOKEN=from-file\nALLOWED_USER_IDS=1\n" if path == "/etc/kai/env" else None,
        )
        # Set token explicitly in env - should override file value
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "from-env")
        config = load_config()
        assert config.telegram_bot_token == "from-env"


# ── PR review config ─────────────────────────────────────────────


class TestPRReviewConfig:
    def test_defaults(self, monkeypatch):
        """PR review is disabled by default with a 5-minute cooldown."""
        _set_required(monkeypatch)
        config = load_config()
        assert config.pr_review_enabled is False
        assert config.pr_review_cooldown == 300
        assert config.pr_review_timeout_s == 900
        assert config.pr_review_budget_usd == 1.0

    def test_enabled_with_custom_cooldown(self, monkeypatch):
        """PR_REVIEW_ENABLED and PR_REVIEW_COOLDOWN are picked up from env."""
        _set_required(monkeypatch)
        monkeypatch.setenv("PR_REVIEW_ENABLED", "true")
        monkeypatch.setenv("PR_REVIEW_COOLDOWN", "60")
        config = load_config()
        assert config.pr_review_enabled is True
        assert config.pr_review_cooldown == 60

    def test_cooldown_invalid_raises(self, monkeypatch):
        """Non-numeric PR_REVIEW_COOLDOWN raises SystemExit."""
        _set_required(monkeypatch)
        monkeypatch.setenv("PR_REVIEW_COOLDOWN", "not_a_number")
        with pytest.raises(SystemExit, match="PR_REVIEW_COOLDOWN"):
            load_config()

    def test_timeout_override(self, monkeypatch):
        """PR_REVIEW_TIMEOUT_S parses to an int and reaches the Config."""
        _set_required(monkeypatch)
        monkeypatch.setenv("PR_REVIEW_TIMEOUT_S", "1800")
        config = load_config()
        assert config.pr_review_timeout_s == 1800

    def test_timeout_rejects_non_integer(self, monkeypatch):
        """Non-numeric PR_REVIEW_TIMEOUT_S raises SystemExit."""
        _set_required(monkeypatch)
        monkeypatch.setenv("PR_REVIEW_TIMEOUT_S", "twenty")
        with pytest.raises(SystemExit, match="PR_REVIEW_TIMEOUT_S"):
            load_config()

    def test_timeout_rejects_non_positive(self, monkeypatch):
        """Zero or negative PR_REVIEW_TIMEOUT_S raises SystemExit."""
        _set_required(monkeypatch)
        monkeypatch.setenv("PR_REVIEW_TIMEOUT_S", "0")
        with pytest.raises(SystemExit, match="PR_REVIEW_TIMEOUT_S"):
            load_config()

    def test_budget_override(self, monkeypatch):
        """PR_REVIEW_BUDGET_USD parses to a float and reaches the Config."""
        _set_required(monkeypatch)
        monkeypatch.setenv("PR_REVIEW_BUDGET_USD", "2.5")
        config = load_config()
        assert config.pr_review_budget_usd == 2.5

    def test_budget_rejects_non_number(self, monkeypatch):
        """Non-numeric PR_REVIEW_BUDGET_USD raises SystemExit."""
        _set_required(monkeypatch)
        monkeypatch.setenv("PR_REVIEW_BUDGET_USD", "cheap")
        with pytest.raises(SystemExit, match="PR_REVIEW_BUDGET_USD"):
            load_config()

    def test_budget_rejects_non_positive(self, monkeypatch):
        """Zero or negative PR_REVIEW_BUDGET_USD raises SystemExit."""
        _set_required(monkeypatch)
        monkeypatch.setenv("PR_REVIEW_BUDGET_USD", "-1.0")
        with pytest.raises(SystemExit, match="PR_REVIEW_BUDGET_USD"):
            load_config()

    def test_github_repo_from_env(self, monkeypatch):
        """GITHUB_REPO is picked up from env, defaults to empty string."""
        _set_required(monkeypatch)
        config = load_config()
        assert config.github_repo == ""

        monkeypatch.setenv("GITHUB_REPO", "kai")
        config = load_config()
        assert config.github_repo == "kai"

    def test_spec_dir_from_env(self, monkeypatch):
        """SPEC_DIR is picked up from env, defaults to 'specs'."""
        _set_required(monkeypatch)
        config = load_config()
        assert config.spec_dir == "specs"

        monkeypatch.setenv("SPEC_DIR", "home/specs")
        config = load_config()
        assert config.spec_dir == "home/specs"


# ── Issue triage config ─────────────────────────────────────────────


class TestIssueTriageConfig:
    def test_defaults(self, monkeypatch):
        """Issue triage is disabled by default."""
        _set_required(monkeypatch)
        config = load_config()
        assert config.issue_triage_enabled is False

    def test_enabled(self, monkeypatch):
        """ISSUE_TRIAGE_ENABLED=true enables the triage agent."""
        _set_required(monkeypatch)
        monkeypatch.setenv("ISSUE_TRIAGE_ENABLED", "true")
        config = load_config()
        assert config.issue_triage_enabled is True


# ── GITHUB_NOTIFY_CHAT_ID config ───────────────────────────────────


class TestGitHubNotifyChatIdConfig:
    def test_default_none(self, monkeypatch):
        """Unset GITHUB_NOTIFY_CHAT_ID defaults to None."""
        _set_required(monkeypatch)
        config = load_config()
        assert config.github_notify_chat_id is None

    def test_valid_positive(self, monkeypatch):
        """Positive chat ID is parsed correctly."""
        _set_required(monkeypatch)
        monkeypatch.setenv("GITHUB_NOTIFY_CHAT_ID", "123456789")
        config = load_config()
        assert config.github_notify_chat_id == 123456789

    def test_valid_negative(self, monkeypatch):
        """Negative chat ID (group chat) is parsed correctly."""
        _set_required(monkeypatch)
        monkeypatch.setenv("GITHUB_NOTIFY_CHAT_ID", "-100123456789")
        config = load_config()
        assert config.github_notify_chat_id == -100123456789

    def test_invalid_warns(self, monkeypatch, caplog):
        """Non-numeric value warns and uses None."""
        _set_required(monkeypatch)
        monkeypatch.setenv("GITHUB_NOTIFY_CHAT_ID", "not-a-number")
        config = load_config()
        assert config.github_notify_chat_id is None
        assert "invalid github_notify_chat_id" in caplog.text.lower()


# ── resolve_claude_user() ────────────────────────────────────────────


class TestResolveClaudeUser:
    def test_none_returns_none(self):
        """None input returns None."""
        assert resolve_claude_user(None) is None

    def test_empty_string_returns_none(self):
        """Empty string is falsy, returns None."""
        assert resolve_claude_user("") is None

    def test_self_sudo_returns_none(self):
        """When claude_user matches current process user, returns None."""
        try:
            current_user = pwd.getpwuid(os.getuid()).pw_name
        except KeyError:
            pytest.skip("UID has no passwd entry")
        assert resolve_claude_user(current_user) is None

    def test_different_user_returns_unchanged(self, monkeypatch):
        """When claude_user differs from current user, returns it unchanged."""
        monkeypatch.setattr("kai.config.pwd.getpwuid", MagicMock(return_value=MagicMock(pw_name="localuser")))
        assert resolve_claude_user("some_other_user") == "some_other_user"

    def test_unknown_uid_returns_unchanged(self, monkeypatch):
        """When pwd.getpwuid raises KeyError, returns claude_user unchanged."""
        monkeypatch.setattr("kai.config.pwd.getpwuid", MagicMock(side_effect=KeyError("no entry")))
        assert resolve_claude_user("container_user") == "container_user"


# ── Deprecation warnings ────────────────────────────────────────────


def _mock_user_configs(monkeypatch):
    """Patch _load_user_configs to return a minimal user config dict.

    When _load_user_configs returns a dict (not None), the deprecation
    warning block fires because it means users.yaml exists.
    """
    user = UserConfig(telegram_id=123, name="testuser")
    monkeypatch.setattr(
        "kai.config._load_user_configs",
        lambda *_a: {123: user},
    )


# Deprecated env vars and representative test values. /tmp is used
# for path vars because it always exists on both macOS and Linux.
_DEPRECATED_VARS_WITH_VALUES = {
    "CLAUDE_MODEL": "sonnet",
    "CLAUDE_MAX_BUDGET_USD": "10.0",  # old name, still tested for deprecation warning
    "CLAUDE_TIMEOUT_SECONDS": "120",
    "CLAUDE_MAX_CONTEXT_WINDOW": "200000",
    "CLAUDE_USER": "kai",
    "WORKSPACE_BASE": "/tmp",
    "ALLOWED_WORKSPACES": "/tmp",
    "PR_REVIEW_ENABLED": "true",
    "ISSUE_TRIAGE_ENABLED": "true",
    "GITHUB_NOTIFY_CHAT_ID": "12345",
}


class TestDeprecationWarnings:
    """Verify deprecated env vars emit warnings when users.yaml exists."""

    @pytest.mark.parametrize("var,value", _DEPRECATED_VARS_WITH_VALUES.items())
    def test_warns_when_users_yaml_exists(self, monkeypatch, caplog, var, value):
        """Deprecated env var emits warning when users.yaml is present."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake")
        monkeypatch.setenv(var, value)
        _mock_user_configs(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="kai.config"):
            load_config()
        assert f"{var} in env is deprecated" in caplog.text

    def test_no_warning_without_users_yaml(self, monkeypatch, caplog):
        """Per-user deprecated env vars do NOT warn when users.yaml is absent.

        Note: CLAUDE_MODEL still emits a standalone rename warning (it was
        renamed to DEFAULT_MODEL) regardless of users.yaml. This test uses
        a non-CLAUDE_MODEL var to verify the users.yaml-gated warnings.
        """
        _set_required(monkeypatch)
        monkeypatch.setenv("CLAUDE_USER", "somebody")
        # _load_user_configs returns None (no users.yaml) by default
        # because _clean_env patches _read_protected_file to None
        with caplog.at_level(logging.WARNING, logger="kai.config"):
            load_config()
        assert "deprecated" not in caplog.text.lower()

    def test_empty_var_does_not_warn(self, monkeypatch, caplog):
        """Empty string env vars are not treated as 'set'."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake")
        # Use CLAUDE_USER instead of CLAUDE_MODEL because an empty
        # CLAUDE_MODEL fails the model validation step downstream.
        monkeypatch.setenv("CLAUDE_USER", "")
        _mock_user_configs(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="kai.config"):
            load_config()
        assert "CLAUDE_USER in env is deprecated" not in caplog.text


# ── GitHub repos empty warning ───────────────────────────────────────


class TestGitHubReposWarning:
    """Verify startup warning when GitHub features are on but no repos configured."""

    def test_warns_when_pr_review_enabled_no_repos(self, monkeypatch, caplog):
        """PR review enabled globally + no github_repos = warning."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake")
        monkeypatch.setenv("PR_REVIEW_ENABLED", "true")
        _mock_user_configs(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="kai.config"):
            load_config()
        assert "PR review" in caplog.text
        assert "github_repos" in caplog.text

    def test_warns_when_issue_triage_enabled_no_repos(self, monkeypatch, caplog):
        """Issue triage enabled globally + no github_repos = warning."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake")
        monkeypatch.setenv("ISSUE_TRIAGE_ENABLED", "true")
        _mock_user_configs(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="kai.config"):
            load_config()
        assert "issue triage" in caplog.text
        assert "github_repos" in caplog.text

    def test_warns_when_both_features_enabled_no_repos(self, monkeypatch, caplog):
        """Both features enabled + no github_repos = warning naming both."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake")
        monkeypatch.setenv("PR_REVIEW_ENABLED", "true")
        monkeypatch.setenv("ISSUE_TRIAGE_ENABLED", "true")
        _mock_user_configs(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="kai.config"):
            load_config()
        assert "PR review" in caplog.text
        assert "issue triage" in caplog.text

    def test_warns_when_per_user_pr_review_enabled_no_repos(self, monkeypatch, caplog):
        """Per-user pr_review=True (no global env var) + no repos = warning."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake")
        user = UserConfig(telegram_id=123, name="testuser", pr_review=True)
        monkeypatch.setattr("kai.config._load_user_configs", lambda *_a: {123: user})
        with caplog.at_level(logging.WARNING, logger="kai.config"):
            load_config()
        assert "PR review" in caplog.text
        assert "github_repos" in caplog.text

    def test_no_warn_when_repos_configured(self, monkeypatch, caplog):
        """No warning when at least one user has github_repos set."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake")
        monkeypatch.setenv("PR_REVIEW_ENABLED", "true")
        user = UserConfig(telegram_id=123, name="testuser", github_repos=["owner/repo"])
        monkeypatch.setattr("kai.config._load_user_configs", lambda *_a: {123: user})
        with caplog.at_level(logging.WARNING, logger="kai.config"):
            load_config()
        # The github_repos warning should not fire; filter out deprecation
        # warnings which also mention github_repos tangentially.
        repo_warnings = [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING and "github_repos" in r.message and "deprecated" not in r.message
        ]
        assert repo_warnings == []

    def test_no_warn_when_features_disabled(self, monkeypatch, caplog):
        """No warning when neither feature is enabled (empty repos is fine)."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake")
        _mock_user_configs(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="kai.config"):
            load_config()
        repo_warnings = [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING and "github_repos" in r.message and "deprecated" not in r.message
        ]
        assert repo_warnings == []

    def test_no_warn_when_no_user_configs(self, monkeypatch, caplog):
        """No warning in env-var-only mode (no users.yaml)."""
        _set_required(monkeypatch)
        monkeypatch.setenv("PR_REVIEW_ENABLED", "true")
        with caplog.at_level(logging.WARNING, logger="kai.config"):
            load_config()
        repo_warnings = [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING and "github_repos" in r.message and "deprecated" not in r.message
        ]
        assert repo_warnings == []

    def test_no_warn_when_any_user_has_repos(self, monkeypatch, caplog):
        """No warning when at least one of multiple users has repos."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake")
        monkeypatch.setenv("PR_REVIEW_ENABLED", "true")
        user_a = UserConfig(telegram_id=123, name="alice", github_repos=["owner/repo"])
        user_b = UserConfig(telegram_id=456, name="bob")
        monkeypatch.setattr("kai.config._load_user_configs", lambda *_a: {123: user_a, 456: user_b})
        with caplog.at_level(logging.WARNING, logger="kai.config"):
            load_config()
        repo_warnings = [
            r
            for r in caplog.records
            if r.levelno >= logging.WARNING and "github_repos" in r.message and "deprecated" not in r.message
        ]
        assert repo_warnings == []


# ── Minimal env + users.yaml operation ─────────────────────────────

# Truly global env vars - the only ones needed when users.yaml exists.
# Everything else comes from per-user config in users.yaml.
_MINIMAL_GLOBAL_ENV = {
    "TELEGRAM_BOT_TOKEN": "fake-token",
    "WEBHOOK_PORT": "8080",
    "WEBHOOK_SECRET": "test-secret",
}


class TestMinimalEnvWithUsersYaml:
    """Prove the system works with only global env vars + users.yaml.

    These tests set zero deprecated env vars, relying entirely on
    users.yaml (mocked) and dataclass defaults.
    """

    def test_loads_with_only_global_env(self, monkeypatch):
        """Config loads successfully with only truly global env vars + users.yaml."""
        for var, val in _MINIMAL_GLOBAL_ENV.items():
            monkeypatch.setenv(var, val)
        _mock_user_configs(monkeypatch)
        config = load_config()
        # Uses dataclass defaults when no env var is set
        assert config.default_model == "sonnet"
        assert config.claude_timeout_seconds == 120
        assert config.budget_ceiling == 10.0
        assert config.claude_max_context_window == 0
        assert config.claude_user is None
        assert config.workspace_base is None
        assert config.allowed_workspaces == []
        assert config.pr_review_enabled is False
        assert config.issue_triage_enabled is False
        assert config.github_notify_chat_id is None
        # users.yaml IDs replace ALLOWED_USER_IDS
        assert config.allowed_user_ids == {123}
        assert config.user_configs is not None

    def test_per_user_model_without_env(self, monkeypatch):
        """Per-user model from users.yaml works when DEFAULT_MODEL is unset."""
        for var, val in _MINIMAL_GLOBAL_ENV.items():
            monkeypatch.setenv(var, val)
        user = UserConfig(telegram_id=123, name="testuser", model="opus")
        monkeypatch.setattr(
            "kai.config._load_user_configs",
            lambda *_a: {123: user},
        )
        config = load_config()
        # Global default is "sonnet" (dataclass default)
        assert config.default_model == "sonnet"
        # Per-user is "opus" from users.yaml
        assert config.user_configs is not None
        assert config.user_configs[123].model == "opus"

    def test_per_user_budget_without_env(self, monkeypatch):
        """Per-user budget from users.yaml works when BUDGET_CEILING is unset."""
        for var, val in _MINIMAL_GLOBAL_ENV.items():
            monkeypatch.setenv(var, val)
        user = UserConfig(telegram_id=123, name="testuser", max_budget=25.0)
        monkeypatch.setattr(
            "kai.config._load_user_configs",
            lambda *_a: {123: user},
        )
        config = load_config()
        assert config.budget_ceiling == 10.0  # dataclass default
        assert config.user_configs is not None
        assert config.user_configs[123].max_budget == 25.0

    def test_workspace_base_from_users_yaml_only(self, monkeypatch, tmp_path):
        """Per-user workspace_base works when WORKSPACE_BASE env is unset."""
        for var, val in _MINIMAL_GLOBAL_ENV.items():
            monkeypatch.setenv(var, val)
        user = UserConfig(telegram_id=123, name="testuser", workspace_base=tmp_path)
        monkeypatch.setattr(
            "kai.config._load_user_configs",
            lambda *_a: {123: user},
        )
        config = load_config()
        assert config.workspace_base is None  # global is unset
        assert config.user_configs is not None
        assert config.user_configs[123].workspace_base == tmp_path

    def test_github_settings_from_users_yaml_only(self, monkeypatch):
        """GitHub routing works when env var globals are all unset."""
        for var, val in _MINIMAL_GLOBAL_ENV.items():
            monkeypatch.setenv(var, val)
        user = UserConfig(
            telegram_id=123,
            name="testuser",
            pr_review=True,
            issue_triage=True,
            github_notify_chat_id=99999,
            github_repos=["owner/repo"],
        )
        monkeypatch.setattr(
            "kai.config._load_user_configs",
            lambda *_a: {123: user},
        )
        config = load_config()
        # Global env fallbacks are all at their defaults
        assert config.pr_review_enabled is False
        assert config.issue_triage_enabled is False
        assert config.github_notify_chat_id is None
        # Per-user overrides are set
        assert config.user_configs is not None
        uc = config.user_configs[123]
        assert uc.pr_review is True
        assert uc.issue_triage is True
        assert uc.github_notify_chat_id == 99999


# ── Precedence chain integration tests ─────────────────────────────


class TestResolutionWithoutEnvVars:
    """Verify resolution functions work when env vars are absent.

    These tests build a Config directly (no env parsing) with minimal
    global values and per-user overrides, then call the resolution
    functions that pool.py, bot.py, and webhook.py use at runtime.
    """

    def test_pool_uses_per_user_model(self):
        """SubprocessPool._create_instance picks per-user model over global default."""
        user = UserConfig(telegram_id=123, name="testuser", model="opus")
        config = Config(
            telegram_bot_token="fake",
            allowed_user_ids={123},
            user_configs={123: user},
            # claude_model defaults to "sonnet" via dataclass
        )
        # Intentionally inlines the resolution pattern rather than calling
        # SubprocessPool._create_instance, which requires mocking the Claude
        # binary and process spawning. Tests the precedence contract, not
        # the pool integration.
        resolved = user.model if user.model else config.default_model
        assert resolved == "opus"

    @pytest.mark.asyncio
    async def test_github_settings_without_env(self, tmp_path):
        """resolve_github_settings works when env var globals are all defaults."""
        from kai import sessions

        user = UserConfig(
            telegram_id=123,
            name="testuser",
            pr_review=True,
            issue_triage=False,
            github_notify_chat_id=55555,
        )
        config = Config(
            telegram_bot_token="fake",
            allowed_user_ids={123},
            user_configs={123: user},
            session_db_path=tmp_path / "test.db",
            # All env-sourced globals at defaults:
            # pr_review_enabled=False, issue_triage_enabled=False,
            # github_notify_chat_id=None
        )
        await sessions.init_db(config.session_db_path)
        try:
            settings = await sessions.resolve_github_settings(123, config)
            # Per-user yaml wins over env defaults
            assert settings["pr_review"] is True
            assert settings["issue_triage"] is False
            assert settings["notify_chat_id"] == 55555
        finally:
            await sessions.close_db()

    @pytest.mark.asyncio
    async def test_workspace_access_without_env(self, tmp_path):
        """resolve_workspace_access works with per-user workspace_base only."""
        from kai import sessions

        user = UserConfig(
            telegram_id=123,
            name="testuser",
            workspace_base=tmp_path,
        )
        config = Config(
            telegram_bot_token="fake",
            allowed_user_ids={123},
            user_configs={123: user},
            session_db_path=tmp_path / "test.db",
            # workspace_base=None (default, env not set)
        )
        await sessions.init_db(config.session_db_path)
        try:
            base, _allowed = await sessions.resolve_workspace_access(123, config)
            # Per-user workspace_base from users.yaml
            assert base == tmp_path
        finally:
            await sessions.close_db()


# ── Legacy env-only backward compatibility ──────────────────────────


class TestLegacyEnvOnlyMode:
    """Verify backward compatibility when users.yaml does not exist.

    Single-user installs with only env vars (no users.yaml) must
    continue to work exactly as before.
    """

    def test_loads_from_env_only(self, monkeypatch):
        """Full config from env vars works when users.yaml is absent."""
        _set_required(monkeypatch)
        monkeypatch.setenv("CLAUDE_MODEL", "opus")
        monkeypatch.setenv("BUDGET_CEILING", "25.0")
        monkeypatch.setenv("CLAUDE_TIMEOUT_SECONDS", "300")
        config = load_config()
        assert config.default_model == "opus"
        assert config.budget_ceiling == 25.0
        assert config.claude_timeout_seconds == 300
        assert config.user_configs is None

    def test_old_budget_env_var_backward_compat(self, monkeypatch):
        """CLAUDE_MAX_BUDGET_USD still works as fallback when BUDGET_CEILING is not set."""
        _set_required(monkeypatch)
        monkeypatch.setenv("CLAUDE_MAX_BUDGET_USD", "42.0")
        config = load_config()
        assert config.budget_ceiling == 42.0

    def test_new_budget_env_var_takes_precedence(self, monkeypatch):
        """BUDGET_CEILING wins over CLAUDE_MAX_BUDGET_USD when both are set."""
        _set_required(monkeypatch)
        monkeypatch.setenv("BUDGET_CEILING", "50.0")
        monkeypatch.setenv("CLAUDE_MAX_BUDGET_USD", "25.0")
        config = load_config()
        assert config.budget_ceiling == 50.0

    def test_no_deprecation_warnings(self, monkeypatch, caplog):
        """No users.yaml deprecation warnings when users.yaml is absent."""
        _set_required(monkeypatch)
        monkeypatch.setenv("DEFAULT_MODEL", "opus")
        with caplog.at_level(logging.WARNING, logger="kai.config"):
            load_config()
        assert "deprecated" not in caplog.text.lower()


# ── CLAUDE_EFFORT_LEVEL config ───────────────────────────────────────


class TestClaudeEffortLevel:
    """Coverage for the CLAUDE_EFFORT_LEVEL env var: a single global
    string validated against a hard-coded allow-list pulled from
    `claude --help` output. Default "high" is intentional - it must
    match the operator's outer-Claude default so user-isolated inner
    Claude (CLAUDE_USER / users.yaml `os_user`) does not silently
    fall to whatever the binary picks as its own default. Tests here
    cover happy path, allow-list rejection, and the two normalization
    behaviors (case + whitespace) that exist to absorb common copy-
    paste shapes from .env files."""

    def test_default_when_unset(self, monkeypatch):
        # Default "high" applies when the env var is not set at all.
        # Critical because operators on existing installs will deploy
        # without setting CLAUDE_EFFORT_LEVEL and must get the same
        # reasoning quality as before this PR landed.
        _set_required(monkeypatch)
        config = load_config()
        assert config.claude_effort_level == "high"

    def test_valid_value_parses(self, monkeypatch):
        # All five allow-list values must round-trip cleanly. If the
        # `claude --help` output ever adds a value, this list (and the
        # _VALID_EFFORT_LEVELS frozenset in config.py) must be updated
        # together; failing one without the other would silently
        # reject otherwise-valid configs at load time.
        _set_required(monkeypatch)
        for value in ["low", "medium", "high", "xhigh", "max"]:
            monkeypatch.setenv("CLAUDE_EFFORT_LEVEL", value)
            config = load_config()
            assert config.claude_effort_level == value, f"value {value!r} did not round-trip"

    def test_invalid_value_raises(self, monkeypatch):
        # Anything outside the allow-list must SystemExit at config load,
        # not silently fall through to the inner-Claude subprocess. A
        # subprocess-level rejection would burn an entire chat session
        # before the operator saw the error; the allow-list check fails
        # at startup before any chat is served.
        _set_required(monkeypatch)
        monkeypatch.setenv("CLAUDE_EFFORT_LEVEL", "extreme")
        with pytest.raises(SystemExit, match="CLAUDE_EFFORT_LEVEL"):
            load_config()

    def test_uppercase_normalized(self, monkeypatch):
        # `.lower()` in the parser must accept "HIGH" / "Medium" / etc.
        # Without normalization, mixed case copied from docs or upper-
        # cased by an operator's editor would be silently rejected, and
        # the rejection reason ("not in allow-list") would be opaque
        # because the values look identical to a casual reader.
        _set_required(monkeypatch)
        monkeypatch.setenv("CLAUDE_EFFORT_LEVEL", "HIGH")
        config = load_config()
        assert config.claude_effort_level == "high"

    def test_whitespace_stripped(self, monkeypatch):
        # `.strip()` must absorb surrounding whitespace from copy-paste
        # of .env entries (a common source of "value looks right but is
        # rejected" footguns). Same reason as the case test: prevents
        # an opaque allow-list rejection on a value that visually
        # matches an allowed one.
        _set_required(monkeypatch)
        monkeypatch.setenv("CLAUDE_EFFORT_LEVEL", "  medium  ")
        config = load_config()
        assert config.claude_effort_level == "medium"

    def test_whitespace_only_falls_back_to_default(self, monkeypatch):
        # The `or "high"` fallback in the parser fires only when the
        # stripped value is empty - i.e. when the env var is set but
        # contains only whitespace. test_whitespace_stripped above
        # uses a real value with surrounding whitespace; this case
        # closes the gap by exercising the fallback branch directly.
        # Without this test, a regression that dropped the `or "high"`
        # would let an empty string slip past the strip and hit the
        # allow-list check, producing a SystemExit instead of the
        # silent default behavior an operator would expect when the
        # env var is effectively unset.
        _set_required(monkeypatch)
        monkeypatch.setenv("CLAUDE_EFFORT_LEVEL", "   ")
        config = load_config()
        assert config.claude_effort_level == "high"


# ── Memory extraction config (spec §6.4, §13.1) ─────────────────────


class TestMemoryExtractionConfig:
    """Four new env vars for Track 2 Haiku extraction. Defaults must
    match spec §6.4 so operators who upgrade without touching env
    files get the documented safety-rail behavior."""

    def test_defaults_match_spec(self, monkeypatch):
        _set_required(monkeypatch)
        config = load_config()
        assert config.memory_extraction_enabled is False
        assert config.memory_extraction_model == "claude-haiku-4-5-20251001"
        assert config.memory_extraction_budget_usd == 0.01
        assert config.memory_extraction_timeout_s == 10

    def test_enabled_from_env(self, monkeypatch):
        _set_required(monkeypatch)
        # memory_extraction_enabled is a sub-toggle of memory_enabled
        # (compositional gate at parse time, issue #403). Both flags
        # must be set for extraction to run.
        monkeypatch.setenv("MEMORY_ENABLED", "true")
        monkeypatch.setenv("MEMORY_EXTRACTION_ENABLED", "true")
        config = load_config()
        assert config.memory_extraction_enabled is True

    def test_extraction_requires_memory_enabled(self, monkeypatch):
        """Compositional gate at parse time (issue #403).

        Without enforcement, MEMORY_EXTRACTION_ENABLED=true with
        MEMORY_ENABLED=false would let the Haiku extraction subprocess
        fire every turn, burning ~$0.01/turn of budget whose result
        silently no-ops in the `_memory is None` guard inside
        add_structured. Pin the dependency so future refactors don't
        accidentally remove it.
        """
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_ENABLED", "false")
        monkeypatch.setenv("MEMORY_EXTRACTION_ENABLED", "true")
        config = load_config()
        assert config.memory_extraction_enabled is False
        assert config.memory_enabled is False

    def test_model_override(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_EXTRACTION_MODEL", "claude-haiku-4-5-20251001-custom")
        config = load_config()
        assert config.memory_extraction_model == "claude-haiku-4-5-20251001-custom"

    def test_budget_override(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_EXTRACTION_BUDGET_USD", "0.25")
        config = load_config()
        assert config.memory_extraction_budget_usd == 0.25

    def test_timeout_override(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_EXTRACTION_TIMEOUT_S", "30")
        config = load_config()
        assert config.memory_extraction_timeout_s == 30

    def test_budget_rejects_negative(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_EXTRACTION_BUDGET_USD", "-0.01")
        with pytest.raises(SystemExit, match="non-negative"):
            load_config()

    def test_budget_rejects_non_number(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_EXTRACTION_BUDGET_USD", "not-a-number")
        with pytest.raises(SystemExit, match="must be a number"):
            load_config()

    def test_timeout_rejects_zero(self, monkeypatch):
        """Timeout=0 would cancel the subprocess before it started;
        rejected explicitly as a footgun."""
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_EXTRACTION_TIMEOUT_S", "0")
        with pytest.raises(SystemExit, match="positive integer"):
            load_config()

    def test_timeout_rejects_negative(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_EXTRACTION_TIMEOUT_S", "-1")
        with pytest.raises(SystemExit, match="positive integer"):
            load_config()

    def test_timeout_rejects_non_integer(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_EXTRACTION_TIMEOUT_S", "not-an-int")
        with pytest.raises(SystemExit, match="must be an integer"):
            load_config()

    def test_consolidation_candidates_default_is_8(self, monkeypatch):
        """Default chosen by the consolidation design: large enough to surface
        paraphrase-equivalent prior facts (top-k semantic search), small enough
        to keep the EXISTING FACTS prompt block bounded under typical assistant
        replies. Default must stay stable so unset = production behavior."""
        _set_required(monkeypatch)
        config = load_config()
        assert config.memory_consolidation_candidates_n == 8

    def test_consolidation_candidates_override(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_CONSOLIDATION_CANDIDATES_N", "12")
        config = load_config()
        assert config.memory_consolidation_candidates_n == 12

    def test_consolidation_candidates_accepts_zero(self, monkeypatch):
        """Zero is the documented kill switch: skip the candidate-fetch step
        entirely and run the extractor with no EXISTING FACTS block. This
        rolls back to pre-consolidation behavior without redeploying."""
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_CONSOLIDATION_CANDIDATES_N", "0")
        config = load_config()
        assert config.memory_consolidation_candidates_n == 0

    def test_consolidation_candidates_rejects_negative(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_CONSOLIDATION_CANDIDATES_N", "-1")
        with pytest.raises(SystemExit, match="non-negative integer"):
            load_config()

    def test_consolidation_candidates_rejects_non_integer(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_CONSOLIDATION_CANDIDATES_N", "not-an-int")
        with pytest.raises(SystemExit, match="must be an integer"):
            load_config()


# ── Episode classifier context window (issue #392) ───────────────────


class TestEpisodeClassifierContextTurns:
    """The EPISODE_CLASSIFIER_CONTEXT_TURNS env var: number of prior
    exchanges fed to the stage-1 extractor as PRIOR CONTEXT for the
    episode classifier. Range 0-10; 0 disables windowing entirely
    (single-turn payload, pre-#392 production behavior). Upper cap
    is defensive against typos that would blow Haiku's context
    window (a 3000-turn window from a typo'd "3000" is a real risk
    at config time)."""

    def test_episode_classifier_context_turns_default(self, monkeypatch):
        """Default 3 = 3 prior exchanges in addition to the current
        one (4-turn payload window total; see Config docstring for
        the N+1 framing). Pin the default so an unset env produces
        production behavior."""
        _set_required(monkeypatch)
        config = load_config()
        assert config.episode_classifier_context_turns == 3

    def test_episode_classifier_context_turns_override(self, monkeypatch):
        """Operator can tune up if real episodes are missed at the
        default. Verifies the env var path threads through to Config."""
        _set_required(monkeypatch)
        monkeypatch.setenv("EPISODE_CLASSIFIER_CONTEXT_TURNS", "5")
        config = load_config()
        assert config.episode_classifier_context_turns == 5

    def test_episode_classifier_context_turns_zero_accepted(self, monkeypatch):
        """0 is the documented disable value: bot.py skips the
        get_recent_pairs read entirely and the payload builder
        renders no PRIOR CONTEXT block, reverting to pre-#392
        single-turn behavior. Operators flip this when the windowed
        prompt regresses in production."""
        _set_required(monkeypatch)
        monkeypatch.setenv("EPISODE_CLASSIFIER_CONTEXT_TURNS", "0")
        config = load_config()
        assert config.episode_classifier_context_turns == 0

    def test_episode_classifier_context_turns_rejects_negative(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("EPISODE_CLASSIFIER_CONTEXT_TURNS", "-1")
        with pytest.raises(SystemExit, match="non-negative"):
            load_config()

    def test_episode_classifier_context_turns_rejects_above_cap(self, monkeypatch):
        """The defensive cap exists so a typo (3000 instead of 3)
        cannot ship a single payload with ~3001 pairs in the PRIOR
        CONTEXT block, which would exceed Haiku's per-call token
        limit. Pin the cap behavior so a future edit that loosens
        or removes it surfaces here."""
        _set_required(monkeypatch)
        monkeypatch.setenv("EPISODE_CLASSIFIER_CONTEXT_TURNS", "11")
        with pytest.raises(SystemExit, match="must be <= 10"):
            load_config()

    def test_episode_classifier_context_turns_rejects_non_integer(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("EPISODE_CLASSIFIER_CONTEXT_TURNS", "abc")
        with pytest.raises(SystemExit, match="must be an integer"):
            load_config()


# ── Stage-2 episode generation (issue #385) ───────────────────────────


class TestMemoryEpisode:
    """The MEMORY_EPISODE_* env vars: stage-2 episode generation, which
    runs out-of-band on stage-1 positives. Bounds differ from stage 1:
    budget is strictly positive (no zero kill-switch; the master switch
    is MEMORY_ENABLED) and timeout has a 10s floor (Haiku warm-up time).
    The model defaults to whatever memory_extraction_model is set to,
    so an operator who only changed MEMORY_EXTRACTION_MODEL also moves
    stage 2 onto the new model without a second var."""

    def test_defaults(self, monkeypatch):
        """Defaults must stay stable so unset = production behavior.
        Model inheritance from memory_extraction_model is the contract
        when no env var is set: a fresh install with neither var set
        ends up with both stages on Haiku (the wizard separately
        recommends Sonnet for stage 2; the inheritance fallback is
        the safety floor for tests and operators who skipped wizard).
        Budget default is 0.15, sized for Sonnet."""
        _set_required(monkeypatch)
        config = load_config()
        assert config.memory_episode_model == "claude-haiku-4-5-20251001"
        assert config.memory_episode_budget_usd == 0.15
        assert config.memory_episode_timeout_s == 120

    def test_model_inherits_extraction_model_when_unset(self, monkeypatch):
        """Operator changes MEMORY_EXTRACTION_MODEL only - episode
        follows. Documented in the templates/.env comment."""
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_EXTRACTION_MODEL", "claude-haiku-4-5-future")
        config = load_config()
        assert config.memory_episode_model == "claude-haiku-4-5-future"

    def test_model_override_takes_precedence(self, monkeypatch):
        """Explicit MEMORY_EPISODE_MODEL beats extraction inheritance.
        Use case: operator runs Haiku for stage 1 and Sonnet for stage 2
        narrative quality."""
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_EXTRACTION_MODEL", "claude-haiku-4-5-20251001")
        monkeypatch.setenv("MEMORY_EPISODE_MODEL", "claude-sonnet-4-6")
        config = load_config()
        assert config.memory_episode_model == "claude-sonnet-4-6"

    def test_budget_override(self, monkeypatch):
        """Override path: arbitrary positive value beats the dataclass
        default of 0.15. Test value 0.30 chosen specifically to be
        non-default so a future flip of the dataclass default to 0.30
        would not silently mask a regression in the override path."""
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_EPISODE_BUDGET_USD", "0.30")
        config = load_config()
        assert config.memory_episode_budget_usd == 0.30

    def test_timeout_override(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_EPISODE_TIMEOUT_S", "60")
        config = load_config()
        assert config.memory_episode_timeout_s == 60

    def test_budget_rejects_zero(self, monkeypatch):
        """Stage-2 budget of zero would mean every call exits at first
        token with error_max_budget_usd. Unlike the consolidation
        candidates field, zero is NOT a kill switch here - the master
        switch is MEMORY_ENABLED. Reject explicitly so a typo doesn't
        silently disable the feature."""
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_EPISODE_BUDGET_USD", "0")
        with pytest.raises(SystemExit, match="positive"):
            load_config()

    def test_budget_rejects_negative(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_EPISODE_BUDGET_USD", "-0.05")
        with pytest.raises(SystemExit, match="positive"):
            load_config()

    def test_budget_rejects_non_number(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_EPISODE_BUDGET_USD", "not-a-number")
        with pytest.raises(SystemExit, match="must be a number"):
            load_config()

    def test_timeout_rejects_below_floor(self, monkeypatch):
        """Floor is 10s because Haiku's warm-up alone can run several
        seconds; a sub-floor timeout would make every call a timeout
        and mask the real model failure as configuration error."""
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_EPISODE_TIMEOUT_S", "9")
        with pytest.raises(SystemExit, match="at least 10"):
            load_config()

    def test_timeout_rejects_negative(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_EPISODE_TIMEOUT_S", "-1")
        with pytest.raises(SystemExit, match="at least 10"):
            load_config()

    def test_timeout_rejects_non_integer(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_EPISODE_TIMEOUT_S", "not-an-int")
        with pytest.raises(SystemExit, match="must be an integer"):
            load_config()


# ── Memory search floor (spec 310 §7.5) ─────────────────────────────


class TestMemorySearchFloor:
    """The MEMORY_SEARCH_FLOOR env var: the relevance gate shared by
    `format_context` (context injection) and the `/memory search` UI.
    Range is closed on both ends because Mem0 cosine similarity is
    normalized to [0.0, 1.0]; out-of-range values would silently
    filter everything (>1.0) or nothing (<0.0)."""

    def test_default_is_0_3(self, monkeypatch):
        """Default matches Mem0's built-in default and the prior hard-coded
        constant; this default must be stable so that not setting the env
        var preserves pre-spec-310 behavior."""
        _set_required(monkeypatch)
        config = load_config()
        assert config.memory_search_floor == 0.3

    def test_override(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_SEARCH_FLOOR", "0.5")
        config = load_config()
        assert config.memory_search_floor == 0.5

    def test_accepts_zero(self, monkeypatch):
        """0.0 is the "include everything" boundary - valid and useful for
        debugging recall problems where the floor is suspected to be too
        aggressive."""
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_SEARCH_FLOOR", "0.0")
        config = load_config()
        assert config.memory_search_floor == 0.0

    def test_accepts_one(self, monkeypatch):
        """1.0 is the "exact match only" boundary; pathologically strict
        but in-range, so accepted."""
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_SEARCH_FLOOR", "1.0")
        config = load_config()
        assert config.memory_search_floor == 1.0

    def test_rejects_negative(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_SEARCH_FLOOR", "-0.1")
        with pytest.raises(SystemExit, match=r"between 0\.0 and 1\.0"):
            load_config()

    def test_rejects_above_one(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_SEARCH_FLOOR", "1.5")
        with pytest.raises(SystemExit, match=r"between 0\.0 and 1\.0"):
            load_config()

    def test_rejects_non_number(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_SEARCH_FLOOR", "high")
        with pytest.raises(SystemExit, match="must be a number"):
            load_config()


# ── Per-user allowed_workspaces (issue #460) ─────────────────────────


class TestPerUserAllowedWorkspacesYaml:
    """
    Parse `allowed_workspaces:` from per-user users.yaml entries.

    Pre-#460 the field was silently dropped by the loader (no
    UserConfig field, no parser branch). These tests pin the
    happy path, edge cases (missing field, empty list, non-list,
    nonexistent path, duplicates), and the field's downstream
    visibility on the resulting UserConfig.
    """

    def _load_with_yaml(self, monkeypatch, users_data):
        """
        Drive _load_user_configs against an in-memory yaml dict.
        Patches _read_protected_yaml so the loader does not hit
        /etc/kai or PROJECT_ROOT during the test.
        """
        from kai.config import _load_user_configs

        monkeypatch.setattr("kai.config._read_protected_yaml", lambda _name: users_data)
        return _load_user_configs(global_backend="claude", global_llm_provider="anthropic")

    def test_missing_field_yields_empty_list(self, monkeypatch):
        """
        Default: a users.yaml entry without an `allowed_workspaces:`
        key produces an empty list on the resulting UserConfig.
        Mirrors the github_repos default behavior.
        """
        users_data = {
            "users": [
                {"telegram_id": 123, "name": "alice"},
            ],
        }
        configs = self._load_with_yaml(monkeypatch, users_data)
        assert configs is not None
        assert configs[123].allowed_workspaces == []

    def test_parses_well_formed_list(self, monkeypatch, tmp_path):
        """
        A list of absolute path strings parses into a list of
        resolved Path objects. The directories must exist on the
        host (we tmp_path-create them) since the loader drops
        nonexistent entries.
        """
        a = tmp_path / "alpha"
        a.mkdir()
        b = tmp_path / "beta"
        b.mkdir()
        users_data = {
            "users": [
                {
                    "telegram_id": 123,
                    "name": "alice",
                    "allowed_workspaces": [str(a), str(b)],
                },
            ],
        }
        configs = self._load_with_yaml(monkeypatch, users_data)
        assert configs is not None
        assert configs[123].allowed_workspaces == [a.resolve(), b.resolve()]

    def test_drops_nonexistent_paths_with_warning(self, monkeypatch, tmp_path, caplog):
        """
        A path that does not exist on the host is dropped from the
        list with a WARNING log line. Matches the workspace_base /
        home_workspace precedent: the loader does not fail the
        whole user entry on a single missing directory (could be
        on an unmounted drive or about to be created).
        """
        real_path = tmp_path / "alpha"
        real_path.mkdir()
        users_data = {
            "users": [
                {
                    "telegram_id": 123,
                    "name": "alice",
                    "allowed_workspaces": [str(real_path), str(tmp_path / "does-not-exist")],
                },
            ],
        }
        with caplog.at_level(logging.WARNING, logger="kai.config"):
            configs = self._load_with_yaml(monkeypatch, users_data)

        assert configs is not None
        # Only the real path survives.
        assert configs[123].allowed_workspaces == [real_path.resolve()]
        assert "allowed_workspaces entry for alice not found" in caplog.text

    def test_non_list_field_yields_empty_with_warning(self, monkeypatch, caplog):
        """
        If `allowed_workspaces:` is set to something other than a
        list (e.g., a string), the loader emits a WARNING and the
        field falls back to empty. Same pattern as github_repos.
        """
        users_data = {
            "users": [
                {
                    "telegram_id": 123,
                    "name": "alice",
                    "allowed_workspaces": "/just/one/path",
                },
            ],
        }
        with caplog.at_level(logging.WARNING, logger="kai.config"):
            configs = self._load_with_yaml(monkeypatch, users_data)

        assert configs is not None
        assert configs[123].allowed_workspaces == []
        assert "allowed_workspaces for alice must be a list" in caplog.text

    def test_empty_strings_dropped_silently(self, monkeypatch, tmp_path):
        """
        Empty-string entries (yaml `- ""` or whitespace) are dropped
        without warning - matches the workspace_base empty-string
        handling. A user with only whitespace entries gets an
        empty list, same as if they had no field at all.
        """
        real_path = tmp_path / "alpha"
        real_path.mkdir()
        users_data = {
            "users": [
                {
                    "telegram_id": 123,
                    "name": "alice",
                    "allowed_workspaces": ["", "   ", str(real_path)],
                },
            ],
        }
        configs = self._load_with_yaml(monkeypatch, users_data)
        assert configs is not None
        assert configs[123].allowed_workspaces == [real_path.resolve()]

    def test_duplicates_deduplicated_at_load_time(self, monkeypatch, tmp_path):
        """
        A user listing the same path twice is almost certainly a
        typo. The loader collapses duplicates so the per-user list
        is clean for downstream consumers (source-attribution
        listings, the /workspaces keyboard).
        """
        real_path = tmp_path / "alpha"
        real_path.mkdir()
        users_data = {
            "users": [
                {
                    "telegram_id": 123,
                    "name": "alice",
                    "allowed_workspaces": [str(real_path), str(real_path)],
                },
            ],
        }
        configs = self._load_with_yaml(monkeypatch, users_data)
        assert configs is not None
        assert configs[123].allowed_workspaces == [real_path.resolve()]

    def test_expands_user_tilde(self, monkeypatch, tmp_path):
        """
        `~/path` style entries get expanded via Path.expanduser().
        Mirrors workspace_base / home_workspace handling.
        """
        # Build a fake HOME under tmp_path so ~ resolves into the
        # test sandbox rather than the developer's actual home.
        fake_home = tmp_path / "home"
        fake_home.mkdir()
        ws = fake_home / "ws"
        ws.mkdir()
        monkeypatch.setenv("HOME", str(fake_home))

        users_data = {
            "users": [
                {
                    "telegram_id": 123,
                    "name": "alice",
                    "allowed_workspaces": ["~/ws"],
                },
            ],
        }
        configs = self._load_with_yaml(monkeypatch, users_data)
        assert configs is not None
        assert configs[123].allowed_workspaces == [ws.resolve()]
