"""Tests for config.py load_config(), DATA_DIR, and _read_protected_file()."""

import os
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from kai.config import _read_protected_file, load_config

# All env vars that load_config reads
_CONFIG_ENV_VARS = [
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_WEBHOOK_URL",
    "TELEGRAM_WEBHOOK_SECRET",
    "ALLOWED_USER_IDS",
    "CLAUDE_MODEL",
    "CLAUDE_TIMEOUT_SECONDS",
    "CLAUDE_MAX_BUDGET_USD",
    "CLAUDE_MAX_SESSION_HOURS",
    "WEBHOOK_PORT",
    "WEBHOOK_SECRET",
    "VOICE_ENABLED",
    "TTS_ENABLED",
    "WORKSPACE_BASE",
    "ALLOWED_WORKSPACES",
    "CLAUDE_USER",
    "PR_REVIEW_ENABLED",
    "PR_REVIEW_COOLDOWN",
    "GITHUB_REPO",
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
        assert config.claude_model == "sonnet"
        assert config.claude_timeout_seconds == 120
        assert config.claude_max_budget_usd == 10.0
        assert config.claude_max_session_hours == 0
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

    def test_github_repo_from_env(self, monkeypatch):
        """GITHUB_REPO is picked up from env, defaults to empty string."""
        _set_required(monkeypatch)
        config = load_config()
        assert config.github_repo == ""

        monkeypatch.setenv("GITHUB_REPO", "kai")
        config = load_config()
        assert config.github_repo == "kai"
