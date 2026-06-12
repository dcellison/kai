"""Tests for config.py load_config(), DATA_DIR, _read_protected_file(), and resolve_claude_user()."""

import logging
import os
import pwd
import subprocess
import textwrap
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kai.config import (
    MODEL_REGISTRY,
    ONESHOT_REASONER_BACKENDS,
    Config,
    ModelRole,
    UserConfig,
    _check_model_registry_complete,
    _load_memory_project_configs,
    _read_protected_file,
    get_model_for,
    load_config,
    resolve_claude_user,
)

# All env vars that load_config reads
_CONFIG_ENV_VARS = [
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_WEBHOOK_URL",
    "TELEGRAM_WEBHOOK_SECRET",
    "ALLOWED_USER_IDS",
    "DEFAULT_MODEL",
    "CLAUDE_MODEL",
    "CLAUDE_TIMEOUT_SECONDS",
    "BUDGET_CEILING",  # retired (budgets are no longer tracked)
    "CLAUDE_MAX_BUDGET_USD",  # retired (budgets are no longer tracked)
    "AGENT_MAX_SESSION_HOURS",
    "CLAUDE_MAX_SESSION_HOURS",  # backward compat (renamed to AGENT_MAX_SESSION_HOURS)
    "AGENT_IDLE_TIMEOUT",
    "CLAUDE_IDLE_TIMEOUT",  # backward compat (renamed to AGENT_IDLE_TIMEOUT)
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
    "CODEX_EFFORT_LEVEL",
    "TOTP_SESSION_MINUTES",
    "TOTP_CHALLENGE_SECONDS",
    "TOTP_LOCKOUT_ATTEMPTS",
    "TOTP_LOCKOUT_MINUTES",
    "AGENT_BACKEND",
    "LLM_PROVIDER",
    "MEMORY_ENABLED",
    "MEMORY_SEARCH_LIMIT",
    "MEMORY_SCOPED_RECALL_ENABLED",
    "MEMORY_TOKEN_BUDGET",
    "MEMORY_EMBEDDING_MODEL",
    "MEMORY_REASONER_BACKEND",
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
    "MEMORY_DUPLICATE_THRESHOLD",
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


@pytest.fixture(autouse=True)
def _mock_binary_resolver(monkeypatch):
    """Pin the binary resolver so config-load cross-binary validation
    passes on any host. Tests in this file exercise the config parser
    in isolation and do not care whether the local machine actually
    has claude/codex installed; cross-binary validation tests
    explicitly override this fixture with their own setUp.

    Without this, tests that set MEMORY_ENABLED=true +
    MEMORY_EXTRACTION_ENABLED=true would fail on hosts where the
    configured reasoner's binary is not on PATH."""
    monkeypatch.setattr(
        "kai.oneshot_binary.resolve_oneshot_binary",
        lambda backend: f"/fake/{backend}",
    )


def _set_required(monkeypatch, token="fake-token", user_ids="123"):
    """Set the truly required env vars and the mandatory users.yaml.

    Post-#565 tranche A, users.yaml is mandatory: load_config raises
    SystemExit when it cannot resolve the file. Most config tests
    exercise env-driven behavior orthogonal to auth, so this helper
    auto-patches a minimal users.yaml derived from `user_ids` so the
    auth contract is satisfied without each test caller wiring it.

    Tests that specifically exercise the mandatory-users contract or
    a custom users.yaml shape should call `_patch_protected_users_yaml`
    directly and skip this helper (or override its protected-file
    patch afterwards).

    TELEGRAM_WEBHOOK_URL is no longer required - omitting it selects polling mode.
    Tests that need webhook mode should set it explicitly.
    """
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", token)
    monkeypatch.setenv("ALLOWED_USER_IDS", user_ids)
    # Build a minimal valid users.yaml from `user_ids` so the
    # mandatory-users contract is met. Names are auto-generated;
    # role=admin so the no-admin-warning does not fire.
    user_entries = "\n".join(
        f"  - telegram_id: {uid.strip()}\n    name: test-user-{uid.strip()}\n    role: admin"
        for uid in user_ids.split(",")
        if uid.strip()
    )
    _patch_protected_users_yaml(monkeypatch, f"users:\n{user_entries}\n")


def _patch_protected_users_yaml(monkeypatch, content: str) -> None:
    """Simulate a protected install with the given users.yaml content.

    Patches `_read_protected_file` so:
      - `/etc/kai/env` returns a sentinel string, which makes
        `_resolve_users_yaml_path(protected_env_was_loaded=True)` route
        the loader at `/etc/kai/users.yaml` instead of XDG.
      - `/etc/kai/users.yaml` returns `content`, which the protected-
        path branch of `_read_users_yaml` consumes via the existing
        `_read_protected_yaml` sudo-cat shim.

    Any other protected-file lookup returns None (the default shape
    on dev hosts). Tests that specifically need the XDG / single-user
    path should use `_patch_xdg_users_yaml` (or set `KAI_USERS_YAML`).
    """

    def _fake_read(path):
        if path == "/etc/kai/users.yaml":
            return content
        if path == "/etc/kai/env":
            # Non-empty content satisfies the protected-mode predicate
            # without contributing any env vars (the comment line is
            # skipped by the parser).
            return "# test sentinel: protected install\n"
        return None

    monkeypatch.setattr("kai.config._read_protected_file", _fake_read)


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
        assert config.agent_timeout_seconds == 120
        assert config.agent_max_session_hours == 0
        assert config.webhook_port == 8080
        # Without TELEGRAM_WEBHOOK_URL, defaults to polling mode
        assert config.telegram_webhook_url is None
        assert config.telegram_webhook_secret is None
        assert config.voice_enabled is False
        assert config.tts_enabled is False
        assert config.workspace_base is None
        # Autocompact tuning defaults to 0 (use Claude Code default)
        assert config.claude_autocompact_pct == 0
        # Agent backend default
        assert config.agent_backend == "claude"

    def test_autocompact_from_env(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("CLAUDE_AUTOCOMPACT_PCT", "80")
        config = load_config()
        assert config.claude_autocompact_pct == 80

    def test_retired_context_window_env_warns_but_loads(self, monkeypatch, caplog):
        """A lingering CLAUDE_MAX_CONTEXT_WINDOW (the setting was
        removed) does not block startup; load_config warns that the
        key has no effect."""
        _set_required(monkeypatch)
        monkeypatch.setenv("CLAUDE_MAX_CONTEXT_WINDOW", "200000")
        with caplog.at_level(logging.WARNING, logger="kai.config"):
            config = load_config()
        assert config is not None
        assert any("CLAUDE_MAX_CONTEXT_WINDOW is no longer supported" in r.getMessage() for r in caplog.records)


# ── Error cases ──────────────────────────────────────────────────────


class TestLoadConfigErrors:
    def test_missing_token(self):
        with pytest.raises(SystemExit, match="TELEGRAM_BOT_TOKEN"):
            load_config()

    def test_missing_users_yaml(self, monkeypatch):
        """Missing /etc/kai/users.yaml is a hard startup failure
        (#565 tranche A). ALLOWED_USER_IDS no longer authorizes the
        daemon to start; it only contributes a migration hint.
        """
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
        monkeypatch.delenv("ALLOWED_USER_IDS", raising=False)
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            pytest.raises(SystemExit, match=r"users\.yaml is required"),
        ):
            load_config()

    def test_missing_users_yaml_with_allowed_user_ids_includes_migration_hint(self, monkeypatch):
        """Operator UX: env-only legacy installs need to see the
        ALLOWED_USER_IDS migration hint in the error message.
        """
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "tok")
        monkeypatch.setenv("ALLOWED_USER_IDS", "12345")
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            pytest.raises(SystemExit, match="ALLOWED_USER_IDS is set in env but is no longer honored"),
        ):
            load_config()

    def test_workspace_base_nonexistent(self, monkeypatch, tmp_path):
        _set_required(monkeypatch)
        monkeypatch.setenv("WORKSPACE_BASE", str(tmp_path / "nope"))
        with pytest.raises(SystemExit, match="not an existing directory"):
            load_config()

    def test_invalid_session_hours(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("AGENT_MAX_SESSION_HOURS", "not-a-number")
        with pytest.raises(SystemExit, match="AGENT_MAX_SESSION_HOURS"):
            load_config()

    def test_session_hours_from_env(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("AGENT_MAX_SESSION_HOURS", "4.5")
        config = load_config()
        assert config.agent_max_session_hours == 4.5

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
    @staticmethod
    def _protected_reader(env_content: str):
        """Lambda that responds to both /etc/kai/env and /etc/kai/users.yaml.

        users.yaml is mandatory post-#565 tranche A, so the dual-mode
        loader tests must provide a minimal admin entry to satisfy
        the auth contract; otherwise load_config raises before
        reaching the env-file assertions these tests exist for.
        """
        users_yaml = "users:\n  - telegram_id: 999\n    name: test\n    role: admin\n"

        def _read(path):
            if path == "/etc/kai/env":
                return env_content
            if path == "/etc/kai/users.yaml":
                return users_yaml
            return None

        return _read

    def test_loads_from_protected_env(self, monkeypatch):
        """When /etc/kai/env is readable, values are used as config."""
        monkeypatch.setattr(
            "kai.config._read_protected_file",
            self._protected_reader("TELEGRAM_BOT_TOKEN=protected-token\nALLOWED_USER_IDS=999\n"),
        )
        config = load_config()
        assert config.telegram_bot_token == "protected-token"
        assert config.allowed_user_ids == {999}

    def test_protected_env_strips_quotes(self, monkeypatch):
        """Quote marks around values in /etc/kai/env are stripped."""
        monkeypatch.setattr(
            "kai.config._read_protected_file",
            self._protected_reader("TELEGRAM_BOT_TOKEN=\"quoted-token\"\nALLOWED_USER_IDS='999'\n"),
        )
        config = load_config()
        assert config.telegram_bot_token == "quoted-token"
        assert config.allowed_user_ids == {999}

    def test_protected_env_skips_comments_and_blanks(self, monkeypatch):
        """Comments and blank lines in /etc/kai/env are ignored."""
        monkeypatch.setattr(
            "kai.config._read_protected_file",
            self._protected_reader("# comment\n\nTELEGRAM_BOT_TOKEN=tok\n\nALLOWED_USER_IDS=999\n"),
        )
        config = load_config()
        assert config.telegram_bot_token == "tok"

    def test_falls_back_to_dotenv(self, monkeypatch, tmp_path):
        """When /etc/kai/env is not readable, load_dotenv is called.

        Single-user mode (no protected env): the resolver picks the
        XDG users.yaml path. We pin that path via `KAI_USERS_YAML` so
        the loader's mandatory-users contract is satisfied from a
        tmp file the test owns.
        """
        load_dotenv_called = []
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake-token")
        monkeypatch.setenv("ALLOWED_USER_IDS", "123")
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users:\n  - telegram_id: 123\n    name: test\n    role: admin\n")
        monkeypatch.setenv("KAI_USERS_YAML", str(users_yaml))
        monkeypatch.setattr("kai.config._read_protected_file", lambda path: None)
        monkeypatch.setattr(
            "kai.config.load_dotenv",
            lambda *a, **kw: load_dotenv_called.append(True),
        )
        load_config()
        assert load_dotenv_called, "load_dotenv should have been called"

    def test_env_vars_take_precedence_over_protected(self, monkeypatch):
        """Explicitly set env vars override values from /etc/kai/env."""
        monkeypatch.setattr(
            "kai.config._read_protected_file",
            self._protected_reader("TELEGRAM_BOT_TOKEN=from-file\nALLOWED_USER_IDS=999\n"),
        )
        # Set token explicitly in env - should override file value
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "from-env")
        config = load_config()
        assert config.telegram_bot_token == "from-env"


# ── PR review config ─────────────────────────────────────────────


class TestPRReviewConfig:
    def test_defaults(self, monkeypatch):
        """PR review resource controls take dataclass defaults when env is empty."""
        _set_required(monkeypatch)
        config = load_config()
        assert config.pr_review_cooldown == 300
        assert config.pr_review_timeout_s == 900

    def test_custom_cooldown(self, monkeypatch):
        """PR_REVIEW_COOLDOWN is picked up from env."""
        _set_required(monkeypatch)
        monkeypatch.setenv("PR_REVIEW_COOLDOWN", "60")
        assert load_config().pr_review_cooldown == 60

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


# Renamed env vars whose legacy name still triggers a one-line
# operator-facing migration warning. Class A mirrors (CLAUDE_MODEL,
# CLAUDE_USER, PR_REVIEW_ENABLED, ISSUE_TRIAGE_ENABLED,
# GITHUB_NOTIFY_CHAT_ID) are no longer read at runtime and have no
# deprecation handling here — the loader silently ignores them.
_RENAMED_VARS_WITH_VALUES = {
    "CLAUDE_TIMEOUT_SECONDS": "120",
}


class TestRenamedEnvWarnings:
    """Renamed env vars surface a single migration warning."""

    @pytest.mark.parametrize("var,value", _RENAMED_VARS_WITH_VALUES.items())
    def test_warns_on_legacy_name(self, monkeypatch, caplog, var, value):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake")
        monkeypatch.setenv(var, value)
        _mock_user_configs(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="kai.config"):
            load_config()
        assert f"{var} in env is deprecated" in caplog.text

    def test_empty_var_does_not_warn(self, monkeypatch, caplog):
        """Empty string env vars are not treated as 'set'."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake")
        monkeypatch.setenv("CLAUDE_TIMEOUT_SECONDS", "")
        _mock_user_configs(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="kai.config"):
            load_config()
        assert "CLAUDE_TIMEOUT_SECONDS in env is deprecated" not in caplog.text


# Retired env vars: the setting no longer exists, so the loader warns
# that the lingering value has no effect (no replacement key to point
# at) and continues without error.
_RETIRED_VARS_WITH_VALUES = {
    "CLAUDE_MAX_CONTEXT_WINDOW": "100000",
    "BUDGET_CEILING": "10.0",
    "CLAUDE_MAX_BUDGET_USD": "10.0",
    "PR_REVIEW_BUDGET_USD": "1.0",
    "MEMORY_EXTRACTION_BUDGET_USD": "0.01",
    "MEMORY_EPISODE_BUDGET_USD": "0.15",
}


class TestRetiredEnvWarnings:
    """Retired env vars surface a single no-longer-supported warning
    and never abort startup."""

    @pytest.mark.parametrize("var,value", _RETIRED_VARS_WITH_VALUES.items())
    def test_warns_on_retired_key(self, monkeypatch, caplog, var, value):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake")
        monkeypatch.setenv(var, value)
        _mock_user_configs(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="kai.config"):
            load_config()
        assert f"{var} is no longer supported" in caplog.text

    @pytest.mark.parametrize("var,value", _RETIRED_VARS_WITH_VALUES.items())
    def test_retired_key_does_not_abort(self, monkeypatch, var, value):
        """A lingering retired key has no effect beyond the warning;
        load_config still returns a usable Config."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake")
        monkeypatch.setenv(var, value)
        _mock_user_configs(monkeypatch)
        config = load_config()
        assert config.telegram_bot_token == "fake"

    def test_empty_retired_key_does_not_warn(self, monkeypatch, caplog):
        """Empty string env vars are not treated as 'set'."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake")
        monkeypatch.setenv("BUDGET_CEILING", "")
        _mock_user_configs(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="kai.config"):
            load_config()
        assert "BUDGET_CEILING is no longer supported" not in caplog.text


# ── GitHub repos empty warning ───────────────────────────────────────


class TestGitHubReposWarning:
    """Startup warning when GitHub features are on but no repos configured.

    With Class A globals retired, the warning's gating signal is
    per-user (`pr_review`/`issue_triage` in users.yaml). The check
    runs once across all users.
    """

    def test_warns_when_per_user_pr_review_no_repos(self, monkeypatch, caplog):
        """A user with pr_review=True but no github_repos triggers the warning."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake")
        user = UserConfig(telegram_id=123, name="testuser", pr_review=True)
        monkeypatch.setattr("kai.config._load_user_configs", lambda *_a: {123: user})
        with caplog.at_level(logging.WARNING, logger="kai.config"):
            load_config()
        assert "PR review" in caplog.text
        assert "github_repos" in caplog.text

    def test_warns_when_per_user_triage_no_repos(self, monkeypatch, caplog):
        """A user with issue_triage=True but no github_repos triggers the warning."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake")
        user = UserConfig(telegram_id=123, name="testuser", issue_triage=True)
        monkeypatch.setattr("kai.config._load_user_configs", lambda *_a: {123: user})
        with caplog.at_level(logging.WARNING, logger="kai.config"):
            load_config()
        assert "issue triage" in caplog.text
        assert "github_repos" in caplog.text

    def test_warns_when_both_features_no_repos(self, monkeypatch, caplog):
        """A user opted into both features without repos triggers a joint warning."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake")
        user = UserConfig(telegram_id=123, name="testuser", pr_review=True, issue_triage=True)
        monkeypatch.setattr("kai.config._load_user_configs", lambda *_a: {123: user})
        with caplog.at_level(logging.WARNING, logger="kai.config"):
            load_config()
        assert "PR review" in caplog.text
        assert "issue triage" in caplog.text

    def test_no_warn_when_repos_configured(self, monkeypatch, caplog):
        """No warning when the opted-in user has github_repos set."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake")
        user = UserConfig(telegram_id=123, name="testuser", pr_review=True, github_repos=["owner/repo"])
        monkeypatch.setattr("kai.config._load_user_configs", lambda *_a: {123: user})
        with caplog.at_level(logging.WARNING, logger="kai.config"):
            load_config()
        repo_warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and "github_repos" in r.message]
        assert repo_warnings == []

    def test_no_warn_when_features_disabled(self, monkeypatch, caplog):
        """No warning when no user opts into either feature."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake")
        _mock_user_configs(monkeypatch)
        with caplog.at_level(logging.WARNING, logger="kai.config"):
            load_config()
        repo_warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and "github_repos" in r.message]
        assert repo_warnings == []

    def test_no_warn_when_any_user_has_repos(self, monkeypatch, caplog):
        """Across users, any single user with repos suppresses the warning."""
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "fake")
        user_a = UserConfig(telegram_id=123, name="alice", pr_review=True, github_repos=["owner/repo"])
        user_b = UserConfig(telegram_id=456, name="bob")
        monkeypatch.setattr("kai.config._load_user_configs", lambda *_a: {123: user_a, 456: user_b})
        with caplog.at_level(logging.WARNING, logger="kai.config"):
            load_config()
        repo_warnings = [r for r in caplog.records if r.levelno >= logging.WARNING and "github_repos" in r.message]
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
        assert config.agent_timeout_seconds == 120
        assert config.workspace_base is None
        assert config.allowed_workspaces == []
        # users.yaml IDs are the authorization surface
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
        """GitHub routing reads from users.yaml only; no global env fallbacks remain."""
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


class TestLegacyEnvBackwardCompat:
    """Backward-compat coverage for env-var renames that still apply
    even though users.yaml is now mandatory (#565 tranche A).

    Pre-tranche-A this class covered the "single-user install via
    env vars only" path; that path is gone (the loader raises
    SystemExit on absent users.yaml). What remains is the env-var
    rename surface (AGENT_TIMEOUT_SECONDS <- CLAUDE_TIMEOUT_SECONDS).
    """

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
    Claude (users.yaml `os_user`) does not silently fall to whatever
    the binary picks as its own default. Tests here
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


# ── CODEX_EFFORT_LEVEL config ────────────────────────────────────────


class TestCodexEffortLevel:
    """Coverage for the CODEX_EFFORT_LEVEL env var: same strip/lower +
    allow-list shape as CLAUDE_EFFORT_LEVEL with one contract
    difference, the set-or-absent default. Empty / unset stays empty,
    meaning CodexBackend passes no `-c model_reasoning_effort`
    override and codex falls back to the per-OS-user config.toml or
    the model default."""

    def test_default_empty_when_unset(self, monkeypatch):
        # Unset must stay empty, NOT fall back to a tier. An invented
        # default would silently override every codex user's own
        # config.toml effort setting from a knob the operator never
        # touched.
        _set_required(monkeypatch)
        config = load_config()
        assert config.codex_effort_level == ""

    def test_valid_value_parses(self, monkeypatch):
        # All five upstream values must round-trip. If codex adds a
        # tier, CODEX_EFFORT_LEVELS in config.py and this list must
        # move together.
        _set_required(monkeypatch)
        for value in ["minimal", "low", "medium", "high", "xhigh"]:
            monkeypatch.setenv("CODEX_EFFORT_LEVEL", value)
            config = load_config()
            assert config.codex_effort_level == value, f"value {value!r} did not round-trip"

    def test_invalid_value_raises(self, monkeypatch):
        # Outside the allow-list must SystemExit at config load; a
        # bad value reaching the spawn argv would fail at the codex
        # subprocess instead, burning a chat session to discover.
        _set_required(monkeypatch)
        monkeypatch.setenv("CODEX_EFFORT_LEVEL", "max")
        with pytest.raises(SystemExit, match="CODEX_EFFORT_LEVEL"):
            load_config()

    def test_uppercase_and_whitespace_normalized(self, monkeypatch):
        # Same copy-paste absorption as the claude parser: case and
        # surrounding whitespace must not cause an opaque rejection.
        _set_required(monkeypatch)
        monkeypatch.setenv("CODEX_EFFORT_LEVEL", "  HIGH ")
        config = load_config()
        assert config.codex_effort_level == "high"

    def test_whitespace_only_stays_empty(self, monkeypatch):
        # Whitespace-only is effectively unset: it must collapse to
        # empty (no override) rather than hit the allow-list check.
        _set_required(monkeypatch)
        monkeypatch.setenv("CODEX_EFFORT_LEVEL", "   ")
        config = load_config()
        assert config.codex_effort_level == ""


# ── Memory extraction config (spec §6.4, §13.1) ─────────────────────


class TestMemoryExtractionConfig:
    """Four new env vars for Track 2 Haiku extraction. Defaults must
    match spec §6.4 so operators who upgrade without touching env
    files get the documented safety-rail behavior."""

    def test_defaults_match_spec(self, monkeypatch):
        _set_required(monkeypatch)
        config = load_config()
        assert config.memory_extraction_enabled is False
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
        fire every turn, spawning a per-turn call whose result
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

    def test_model_override_logs_deprecation(self, monkeypatch, caplog):
        """The legacy MEMORY_EXTRACTION_MODEL env var is no longer
        load-bearing (issue #515): memory models resolve per-user
        from the registry via get_model_for(role, effective_backend).
        Setting the env var still parses without error, but emits a
        single deprecation warning at load_config time."""
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_EXTRACTION_MODEL", "claude-haiku-4-5-20251001-custom")
        with caplog.at_level("WARNING", logger="kai.config"):
            config = load_config()
        # Field is gone; verify no AttributeError on access shape.
        assert not hasattr(config, "memory_extraction_model")
        deprecation_msgs = [r.message for r in caplog.records if "MEMORY_EXTRACTION_MODEL is deprecated" in r.message]
        assert len(deprecation_msgs) == 1, deprecation_msgs

    def test_timeout_override(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_EXTRACTION_TIMEOUT_S", "30")
        config = load_config()
        assert config.memory_extraction_timeout_s == 30

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


# ── Memory reasoner backend selection ───────────────────────────────


class TestMemoryReasonerBackendDeprecation:
    """The MEMORY_REASONER_BACKEND env var was retired in issue #515.
    Memory reasoner selection is now per-user, derived from each
    user's effective `agent_backend`. Legacy installs that still
    carry the env var get a deprecation warning but load_config
    completes normally."""

    def test_env_var_logs_deprecation_warning(self, monkeypatch, caplog):
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_REASONER_BACKEND", "codex")
        with caplog.at_level("WARNING", logger="kai.config"):
            config = load_config()
        # No SystemExit; load_config returns normally.
        deprecation_msgs = [r.message for r in caplog.records if "MEMORY_REASONER_BACKEND is deprecated" in r.message]
        assert len(deprecation_msgs) == 1, deprecation_msgs
        # The field is gone.
        assert not hasattr(config, "memory_reasoner_backend")

    def test_unknown_value_still_only_warns(self, monkeypatch, caplog):
        """Even a typo like `gpt5` does NOT SystemExit any more; the
        value is ignored after the deprecation warning. Operators
        who used to see config-load failures will now see one warning
        and a working install."""
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_REASONER_BACKEND", "gpt5")
        with caplog.at_level("WARNING", logger="kai.config"):
            load_config()
        deprecation_msgs = [r.message for r in caplog.records if "MEMORY_REASONER_BACKEND is deprecated" in r.message]
        assert len(deprecation_msgs) == 1


class TestMemoryReasonerBinaryValidation:
    """Cross-binary validation: when memory extraction is enabled, each
    extraction-eligible backend's binary must be reachable at startup.
    Per-user dispatch (issue #515) means the eligible set is computed
    from each user's effective `agent_backend`, so the check iterates
    that set rather than reading a single global reasoner toggle.
    Retrieval-only memory does NOT require a reachable agent binary."""

    def test_claude_binary_missing_raises_systemexit(self, monkeypatch):
        """Default-claude install with extraction enabled and no claude
        on PATH must fail at config-load. The composed
        `memory_extraction_enabled` plus the eligible-set membership
        triggers the check."""
        from kai.oneshot_binary import BinaryResolutionError

        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_ENABLED", "true")
        monkeypatch.setenv("MEMORY_EXTRACTION_ENABLED", "true")

        def boom(_backend: str) -> str:
            raise BinaryResolutionError("could not resolve claude binary: `claude` not on PATH")

        monkeypatch.setattr("kai.oneshot_binary.resolve_oneshot_binary", boom)
        with pytest.raises(SystemExit) as exc:
            load_config()
        # New error wording: names the offending backend and the
        # extraction-eligible-user link, no longer cites a global env
        # var since per-user dispatch made it irrelevant.
        assert "'claude'" in str(exc.value)
        assert "binary" in str(exc.value)

    def test_codex_binary_missing_raises_systemexit(self, tmp_path, monkeypatch):
        """Codex via per-user routing: a users.yaml entry pinned to
        codex makes codex extraction-eligible. Missing codex binary
        must fail at config-load with the same per-backend error
        shape as claude."""
        from kai.oneshot_binary import BinaryResolutionError

        _set_required(monkeypatch)
        _patch_protected_users_yaml(
            monkeypatch,
            "users:\n  - telegram_id: 12345\n    name: alice\n    role: admin\n    agent_backend: codex\n    os_user: alice_os\n",
        )
        monkeypatch.setenv("MEMORY_ENABLED", "true")
        monkeypatch.setenv("MEMORY_EXTRACTION_ENABLED", "true")

        def boom(_backend: str) -> str:
            if _backend == "codex":
                raise BinaryResolutionError("could not resolve codex binary: CODEX_BIN unset, `codex` not on PATH")
            return f"/fake/{_backend}"

        monkeypatch.setattr("kai.oneshot_binary.resolve_oneshot_binary", boom)
        with pytest.raises(SystemExit) as exc:
            load_config()
        assert "'codex'" in str(exc.value)
        assert "binary" in str(exc.value)

    def test_retrieval_only_skips_binary_check(self, monkeypatch):
        """MEMORY_ENABLED=true with MEMORY_EXTRACTION_ENABLED=false
        (retrieval-only) must NOT require a reachable agent binary.
        The check is gated on the composed extraction-enabled value,
        not on memory_enabled alone, and the eligible set is empty
        when extraction is disabled."""
        from kai.oneshot_binary import BinaryResolutionError

        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_ENABLED", "true")
        # Extraction explicitly disabled; binary should NEVER be looked up.
        called = []

        def boom(backend: str) -> str:
            called.append(backend)
            raise BinaryResolutionError("should not be called")

        monkeypatch.setattr("kai.oneshot_binary.resolve_oneshot_binary", boom)
        # No SystemExit expected; the load completes successfully.
        config = load_config()
        assert config.memory_enabled is True
        assert config.memory_extraction_enabled is False
        assert called == [], f"resolver must not run on retrieval-only; got {called}"

    def test_extraction_disabled_without_memory_enabled_skips_check(self, monkeypatch):
        """The compositional gate also covers `MEMORY_EXTRACTION_ENABLED=true`
        with `MEMORY_ENABLED=false`. Composed extraction_enabled is
        False, eligible set is empty, and the binary check does not
        fire."""
        from kai.oneshot_binary import BinaryResolutionError

        _set_required(monkeypatch)
        # MEMORY_ENABLED unset / false, but EXTRACTION_ENABLED true.
        # The composed value is False; binary check skipped.
        monkeypatch.setenv("MEMORY_EXTRACTION_ENABLED", "true")
        called = []

        def boom(backend: str) -> str:
            called.append(backend)
            raise BinaryResolutionError("should not be called")

        monkeypatch.setattr("kai.oneshot_binary.resolve_oneshot_binary", boom)
        config = load_config()
        assert config.memory_extraction_enabled is False
        assert called == []


class TestCodexMemorySameUserSymmetry:
    """Codex memory follows claude's `resolve_claude_user` symmetry
    (issue #522): `os_user` is optional and same-user spawn is a
    supported deployment shape. config-load does NOT refuse any of:
    AGENT_BACKEND=codex without users.yaml, codex-effective user
    without os_user, or codex-effective user with os_user matching
    the bot user. Pinning these as starts-cleanly cases guards
    against a future change that re-introduces the pre-#522
    deployment-shape assumption."""

    def test_codex_no_users_yaml_starts_cleanly(self, monkeypatch):
        """AGENT_BACKEND=codex with extraction enabled and no
        users.yaml loads successfully. Per-user dispatch falls back
        to the global agent_backend; the spawn target is the bot
        user via the self-sudo-skip path."""
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_ENABLED", "true")
        monkeypatch.setenv("MEMORY_EXTRACTION_ENABLED", "true")
        monkeypatch.setenv("AGENT_BACKEND", "codex")
        monkeypatch.setenv("DEFAULT_MODEL", "gpt-5.4-mini")
        config = load_config()
        assert config.agent_backend == "codex"
        assert config.memory_extraction_enabled is True

    def test_codex_users_yaml_missing_os_user_starts_cleanly(self, monkeypatch):
        """A codex-effective users.yaml entry without `os_user`
        loads successfully. The runtime treats missing os_user as
        in-process spawn (claude's existing pattern), so config-load
        does not refuse the shape."""
        _set_required(monkeypatch)
        _patch_protected_users_yaml(
            monkeypatch,
            "users:\n  - telegram_id: 67890\n    name: bob\n    role: user\n",
        )
        monkeypatch.setenv("MEMORY_ENABLED", "true")
        monkeypatch.setenv("MEMORY_EXTRACTION_ENABLED", "true")
        monkeypatch.setenv("AGENT_BACKEND", "codex")
        monkeypatch.setenv("DEFAULT_MODEL", "gpt-5.4-mini")
        config = load_config()
        assert config.user_configs is not None
        assert config.user_configs[67890].os_user is None

    def test_codex_users_yaml_same_user_as_bot_starts_cleanly(self, monkeypatch):
        """A codex-effective users.yaml entry with `os_user`
        matching the bot user loads successfully. The runtime
        detects same-user via `resolve_claude_user` and spawns
        codex in-process - the same path claude has always used
        for same-user."""
        bot_user = pwd.getpwuid(os.getuid()).pw_name
        _set_required(monkeypatch)
        _patch_protected_users_yaml(
            monkeypatch,
            f"users:\n  - telegram_id: 12345\n    name: alice\n    role: admin\n    os_user: {bot_user}\n",
        )
        monkeypatch.setenv("MEMORY_ENABLED", "true")
        monkeypatch.setenv("MEMORY_EXTRACTION_ENABLED", "true")
        monkeypatch.setenv("AGENT_BACKEND", "codex")
        monkeypatch.setenv("DEFAULT_MODEL", "gpt-5.4-mini")
        config = load_config()
        assert config.user_configs[12345].os_user == bot_user

    def test_codex_users_yaml_cross_user_os_user_passes(self, monkeypatch):
        """The cross-user deployment shape still works: an `os_user`
        set to a non-bot account loads cleanly and is preserved on
        the UserConfig entry. Regression guard for the multi-user
        case."""
        _set_required(monkeypatch)
        _patch_protected_users_yaml(
            monkeypatch,
            "users:\n  - telegram_id: 12345\n    name: alice\n    role: admin\n    os_user: alice_os\n",
        )
        monkeypatch.setenv("MEMORY_ENABLED", "true")
        monkeypatch.setenv("MEMORY_EXTRACTION_ENABLED", "true")
        monkeypatch.setenv("AGENT_BACKEND", "codex")
        monkeypatch.setenv("DEFAULT_MODEL", "gpt-5.4-mini")
        config = load_config()
        assert config.user_configs[12345].os_user == "alice_os"


class TestMemoryReasonerModelResolution:
    """Memory model defaults resolve through the per-backend role
    registry via `get_model_for(role, effective_backend)`. There is no
    longer an env-var override surface for memory models (issue #515
    retired MEMORY_EXTRACTION_MODEL and MEMORY_EPISODE_MODEL); the
    registry is the only source. Claude rows must match the prior
    literal so installs without users.yaml see byte-identical model
    selection. Codex rows pick a codex-CLI-valid SKU."""

    def test_claude_default_model_unchanged(self, monkeypatch):
        """Default (claude) install: registry resolves to the same
        Haiku SKU that the retired MEMORY_EXTRACTION_MODEL default
        produced, so production behavior is byte-identical."""
        _set_required(monkeypatch)
        # load_config completes; the model selection happens at the
        # call site (memory_extraction.py), so the test reads the
        # registry directly to pin the per-backend default.
        load_config()
        assert get_model_for(ModelRole.MEMORY_EXTRACTION, "claude", "anthropic") == "claude-haiku-4-5-20251001"
        assert get_model_for(ModelRole.MEMORY_EPISODE, "claude", "anthropic") == "claude-haiku-4-5-20251001"

    def test_codex_default_resolves_to_codex_model(self, monkeypatch):
        """Codex install: registry resolves to a codex-CLI-valid SKU.
        Test path is the per-user dispatch surface that production
        will follow: AGENT_BACKEND=codex + users.yaml with os_user."""
        _set_required(monkeypatch)
        _patch_protected_users_yaml(
            monkeypatch,
            "users:\n  - telegram_id: 12345\n    name: alice\n    role: admin\n    os_user: alice_os\n",
        )
        monkeypatch.setenv("MEMORY_ENABLED", "true")
        monkeypatch.setenv("MEMORY_EXTRACTION_ENABLED", "true")
        monkeypatch.setenv("AGENT_BACKEND", "codex")
        monkeypatch.setenv("DEFAULT_MODEL", "gpt-5.4-mini")
        load_config()
        assert get_model_for(ModelRole.MEMORY_EXTRACTION, "codex", "openai") == "gpt-5.4-mini"
        assert get_model_for(ModelRole.MEMORY_EPISODE, "codex", "openai") == "gpt-5.4-mini"

    def test_legacy_extraction_model_env_var_is_ignored(self, monkeypatch, caplog):
        """The retired MEMORY_EXTRACTION_MODEL env var no longer has
        a load-bearing effect. Setting it logs a deprecation warning
        and does not change the registry-resolved model; nor does it
        raise on a Claude SKU sent to a codex install (since the value
        is dropped before any validation)."""
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_EXTRACTION_MODEL", "claude-haiku-4-5-20251001-pinned")
        with caplog.at_level("WARNING", logger="kai.config"):
            config = load_config()
        # Field is gone; the value is dropped after the warning.
        assert not hasattr(config, "memory_extraction_model")
        # Registry resolution is untouched.
        assert get_model_for(ModelRole.MEMORY_EXTRACTION, "claude", "anthropic") == "claude-haiku-4-5-20251001"
        assert any("MEMORY_EXTRACTION_MODEL is deprecated" in r.message for r in caplog.records)

    def test_legacy_episode_model_env_var_is_ignored(self, monkeypatch, caplog):
        """Same shape for the retired MEMORY_EPISODE_MODEL env var."""
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_EPISODE_MODEL", "claude-sonnet-4-6")
        with caplog.at_level("WARNING", logger="kai.config"):
            config = load_config()
        assert not hasattr(config, "memory_episode_model")
        assert get_model_for(ModelRole.MEMORY_EPISODE, "claude", "anthropic") == "claude-haiku-4-5-20251001"
        assert any("MEMORY_EPISODE_MODEL is deprecated" in r.message for r in caplog.records)


class TestExtractionEligibleBackendsHelper:
    """`_compute_extraction_eligible_backends` is the per-user
    dispatch cascade in one helper. It feeds the codex
    precondition checks at config-load and the per-backend binary
    resolution loop in memory.init_memory."""

    def test_returns_empty_when_memory_extraction_disabled(self):
        """Retrieval-only and memory-disabled installs do not run any
        reasoner, so the eligible set is empty regardless of
        agent_backend and users.yaml content."""
        from kai.config import _compute_extraction_eligible_backends

        assert _compute_extraction_eligible_backends("claude", {}, False) == set()
        assert _compute_extraction_eligible_backends("codex", {}, False) == set()

    def test_mixed_users_yaml_contributes_each_effective_backend(self):
        """users.yaml with both claude and codex users: the eligible
        set is the union of effective backends, in the same shape
        bot.py's extraction gate uses."""
        from kai.config import UserConfig, _compute_extraction_eligible_backends

        configs = {
            1: UserConfig(telegram_id=1, name="alice", os_user="a"),
            2: UserConfig(telegram_id=2, name="bob", os_user="b", agent_backend="codex"),
        }
        eligible = _compute_extraction_eligible_backends("claude", configs, True)
        assert eligible == {"claude", "codex"}

    def test_goose_users_contribute_to_eligible_set(self):
        """A users.yaml entry with `agent_backend: goose` contributes
        to the eligible set now that goose ships a OneShotReasoner;
        the config-load precondition / binary checks must fire for a
        goose user the same way they do for the other backends."""
        from kai.config import UserConfig, _compute_extraction_eligible_backends

        configs = {
            1: UserConfig(telegram_id=1, name="alice", os_user="a"),
            2: UserConfig(telegram_id=2, name="bob", os_user="b", agent_backend="goose", llm_provider="openai"),
        }
        eligible = _compute_extraction_eligible_backends("claude", configs, True)
        assert eligible == {"claude", "goose"}

    def test_non_reasoner_backends_filtered_out(self, monkeypatch):
        """The membership gate itself: a backend outside the patched
        constant does not contribute, even when extraction is enabled
        globally. Patched rather than exemplified by a real backend
        because every real backend is currently a member."""
        import kai.config as config_module
        from kai.config import UserConfig, _compute_extraction_eligible_backends

        monkeypatch.setattr(config_module, "ONESHOT_REASONER_BACKENDS", frozenset({"claude", "codex"}))
        configs = {
            1: UserConfig(telegram_id=1, name="alice", os_user="a"),
            2: UserConfig(telegram_id=2, name="bob", os_user="b", agent_backend="goose", llm_provider="openai"),
        }
        eligible = _compute_extraction_eligible_backends("claude", configs, True)
        assert eligible == {"claude"}

    def test_opencode_users_contribute_to_eligible_set(self):
        """An opencode user with extraction enabled appears in the
        eligible set. Mirrors the runtime gate at bot.py that now
        admits opencode users to the extraction path. Without this
        widening the precondition / binary-resolution loop at
        config-load would skip the opencode validation and the
        per-turn extractor would race against an unvalidated binary."""
        from kai.config import UserConfig, _compute_extraction_eligible_backends

        configs = {
            1: UserConfig(telegram_id=1, name="alice", os_user="a"),
            2: UserConfig(telegram_id=2, name="bob", os_user="b", agent_backend="opencode"),
        }
        eligible = _compute_extraction_eligible_backends("claude", configs, True)
        assert eligible == {"claude", "opencode"}

    def test_global_opencode_with_no_users_yields_opencode_only(self):
        """Single-user opencode install (global AGENT_BACKEND=opencode,
        empty users dict) still produces an empty eligible set because
        the cascade only counts users present in the dict. This is the
        same shape the claude / codex paths produce for an empty
        users dict and is intentional: the precondition check fires
        per actual user, not per global default."""
        from kai.config import _compute_extraction_eligible_backends

        assert _compute_extraction_eligible_backends("opencode", {}, True) == set()


class TestConfigNoMemoryModelFields:
    """Regression guard for issue #515 field removal. A future
    change that adds back any of these fields (e.g., as part of a
    bigger refactor that misses the spec rationale) must surface
    here rather than at runtime."""

    def test_config_has_no_memory_extraction_model_field(self):
        # `Config()` cannot be called without required fields, so
        # inspect the dataclass fields directly.
        from dataclasses import fields

        from kai.config import Config

        field_names = {f.name for f in fields(Config)}
        assert "memory_extraction_model" not in field_names
        assert "memory_episode_model" not in field_names
        assert "memory_reasoner_backend" not in field_names


class TestRegistryValidationPerEligibleBackend:
    """Per-eligible-backend `_check_model_registry_complete`
    catches missing memory-role rows at config-load. A per-user
    codex override on a global-claude install must not reach
    runtime without its codex memory-role rows being validated."""

    def test_missing_codex_memory_extraction_row_systemexits(self, monkeypatch):
        """Mixed install: AGENT_BACKEND=claude with one users.yaml
        entry pinned to `agent_backend: codex`. Patch the registry
        so the codex MEMORY_EXTRACTION row is missing; load_config
        SystemExits at startup."""
        import kai.config as config_mod
        from kai.config import ModelRole

        _set_required(monkeypatch)
        _patch_protected_users_yaml(
            monkeypatch,
            "users:\n"
            "  - telegram_id: 1\n    name: alice\n    role: admin\n"
            "    agent_backend: codex\n    os_user: alice_os\n",
        )
        monkeypatch.setenv("MEMORY_ENABLED", "true")
        monkeypatch.setenv("MEMORY_EXTRACTION_ENABLED", "true")
        # Patch the registry to drop the codex MEMORY_EXTRACTION row.
        patched_registry = dict(config_mod.MODEL_REGISTRY)
        patched_registry.pop(("codex", "openai", ModelRole.MEMORY_EXTRACTION), None)
        monkeypatch.setattr("kai.config.MODEL_REGISTRY", patched_registry)
        with pytest.raises(SystemExit):
            load_config()


# ── Stage-2 episode generation (issue #385) ───────────────────────────


class TestMemoryEpisode:
    """The MEMORY_EPISODE_* env vars: stage-2 episode generation, which
    runs out-of-band on stage-1 positives. Timeout has a 10s floor
    (Haiku warm-up time). The model is not configurable here (issue
    #515 retired MEMORY_EPISODE_MODEL);
    `get_model_for(ModelRole.MEMORY_EPISODE, effective_backend)` is the
    only source."""

    def test_defaults(self, monkeypatch):
        """Defaults must stay stable so unset = production behavior."""
        _set_required(monkeypatch)
        config = load_config()
        assert config.memory_episode_timeout_s == 120

    def test_timeout_override(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_EPISODE_TIMEOUT_S", "60")
        config = load_config()
        assert config.memory_episode_timeout_s == 60

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


# ── Memory duplicate threshold ───────────────────────────────────────


class TestMemoryDuplicateThreshold:
    """The MEMORY_DUPLICATE_THRESHOLD env var: cosine threshold for the
    write-time paraphrase-dedup gate in `_store_facts`. Range [0.3,
    1.01]: the lower bound matches `memory_search_floor`'s operator
    floor (cosine below 0.3 is near-everything-is-a-dup territory),
    and 1.01 is the unambiguous-disable sentinel since a literal
    `score == 1.0` can rarely fire on non-identical text under the
    embedding model."""

    def test_default_is_0_9(self, monkeypatch):
        """Default preserves the prior hard-coded constant from the
        boolean `_is_duplicate` era. The dataclass default IS the
        operator-validated production value; upgraders not setting
        this env var inherit the production default."""
        _set_required(monkeypatch)
        config = load_config()
        assert config.memory_duplicate_threshold == 0.9

    def test_override(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_DUPLICATE_THRESHOLD", "0.85")
        config = load_config()
        assert config.memory_duplicate_threshold == 0.85

    def test_accepts_lower_bound(self, monkeypatch):
        """0.3 is the floor (matches memory_search_floor's operator
        recommendation). Accepted but pathologically aggressive."""
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_DUPLICATE_THRESHOLD", "0.3")
        config = load_config()
        assert config.memory_duplicate_threshold == 0.3

    def test_accepts_upper_bound(self, monkeypatch):
        """1.01 is the unambiguous-disable sentinel: at this value
        even a perfect-cosine 1.0 neighbor fails the strict-ge check,
        guaranteeing no fires regardless of the embedding model."""
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_DUPLICATE_THRESHOLD", "1.01")
        config = load_config()
        assert config.memory_duplicate_threshold == 1.01

    def test_rejects_below_lower_bound(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_DUPLICATE_THRESHOLD", "0.2")
        with pytest.raises(SystemExit, match=r"between 0\.3 and 1\.01"):
            load_config()

    def test_rejects_above_upper_bound(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_DUPLICATE_THRESHOLD", "1.5")
        with pytest.raises(SystemExit, match=r"between 0\.3 and 1\.01"):
            load_config()

    def test_rejects_non_number(self, monkeypatch):
        _set_required(monkeypatch)
        monkeypatch.setenv("MEMORY_DUPLICATE_THRESHOLD", "tight")
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


# ── Per-role model registry ─────────────────────────────────────────


class TestModelRegistry:
    """
    Verify get_model_for and the MODEL_REGISTRY shape.

    The registry centralizes per-function model defaults so codex (and
    future backends) can declare their mapping in one place. These tests
    cover the lookup contract (override precedence, missing-row error,
    startup completeness check) and the claude-row-equals-prior-constant
    invariant that makes the Phase 1 migration no-behavior-change.
    """

    def test_lookup_returns_registry_value_when_no_override(self):
        """Default path: empty override returns the registry value."""
        result = get_model_for(ModelRole.PR_REVIEW, "claude", "anthropic")
        assert result == "sonnet"

    def test_override_wins_over_registry(self):
        """A truthy override short-circuits the registry lookup."""
        result = get_model_for(ModelRole.PR_REVIEW, "claude", "anthropic", override="opus")
        assert result == "opus"

    def test_empty_override_falls_through(self):
        """
        Empty-string override is treated as no override.

        This is the contract eval/behavioral.py relies on:
        `args.judge_model or ""` yields "" when the flag is unset, and
        the registry takes over. If empty short-circuited the registry,
        the codex behavioral path would never reach the gpt-5.4-nano row.
        """
        result = get_model_for(ModelRole.PR_REVIEW, "claude", "anthropic", override="")
        assert result == "sonnet"

    def test_missing_row_raises_lookup_error(self):
        """
        Per-request handler safety: a missing registry row raises
        LookupError (5xx-shape) rather than SystemExit (process-wide
        teardown). The startup completeness check is supposed to
        prevent this from ever firing in production; the runtime guard
        exists so a packaging bug surfaces as a contained per-request
        failure if it does.
        """
        with pytest.raises(LookupError, match="No registry entry"):
            get_model_for(ModelRole.PR_REVIEW, "no-such-backend", "anthropic")

    def test_completeness_check_passes_for_claude(self):
        """Claude has rows for every role; no exception."""
        _check_model_registry_complete()  # no raise

    def test_completeness_check_passes_for_codex(self):
        """Codex has rows for every role; no exception."""
        _check_model_registry_complete()  # no raise

    def test_completeness_check_skips_goose(self):
        """
        Goose model resolution lives in _GOOSE_AGENT_MODELS dicts inside
        triage.py / review.py, not in MODEL_REGISTRY. The completeness
        check skips goose explicitly so a goose-backed install does
        not fail at startup just because the registry has no goose rows.
        """
        _check_model_registry_complete()  # no raise

    def test_completeness_check_raises_on_missing_row(self, monkeypatch):
        """
        Synthetic missing-row case: remove a registry entry for the
        active backend and assert load_config-time SystemExit fires
        with a clear message. Uses monkeypatch.delitem so the change
        is reverted between tests.
        """
        monkeypatch.delitem(MODEL_REGISTRY, ("claude", "anthropic", ModelRole.PR_REVIEW))
        with pytest.raises(SystemExit) as excinfo:
            _check_model_registry_complete()
        assert "pr_review" in str(excinfo.value)
        assert "claude" in str(excinfo.value)

    def test_codex_row_must_be_in_codex_models(self, monkeypatch):
        """
        A codex registry row pointing at a model the codex CLI does not
        expose (e.g. gpt-5.4-nano, kept in PROVIDER_MODELS["openai"] for
        goose) must fail load_config-time. Without this guard the
        per-user fix was silent: a future operator could re-introduce a
        nano row in MODEL_REGISTRY because PROVIDER_MODELS still
        accepts it for goose, and the codex behavioral path would fall
        over at first invocation.
        """
        monkeypatch.setitem(MODEL_REGISTRY, ("codex", "openai", ModelRole.BEHAVIORAL_JUDGE), "gpt-5.4-nano")
        with pytest.raises(SystemExit) as excinfo:
            _check_model_registry_complete()
        msg = str(excinfo.value)
        assert "gpt-5.4-nano" in msg
        assert "behavioral_judge" in msg

    def test_codex_behavioral_judge_is_valid_codex_model(self):
        """
        Lock the runtime invariant: whatever model the codex behavioral
        judge resolves to MUST be a member of CODEX_MODELS. This is a
        belt-and-braces check sitting next to the synthetic-corruption
        test above; the registry row itself is the variable that
        operators touch when calibrating, and a regression here is the
        exact failure the synthetic test simulates.
        """
        from kai.config import CODEX_MODELS

        assert MODEL_REGISTRY[("codex", "openai", ModelRole.BEHAVIORAL_JUDGE)] in CODEX_MODELS

    def test_completeness_check_passes_for_opencode(self):
        """OpenCode has rows for every role; no exception. Mirrors
        the claude and codex completeness gates so a future role
        addition without an opencode row fails fast at startup."""
        _check_model_registry_complete()  # no raise

    def test_opencode_row_must_be_provider_slash_model_shape(self, monkeypatch):
        """
        An opencode registry row pointing at a bare name like "sonnet"
        (correct for claude / goose / codex but wrong for opencode)
        must fail at config-load. Without this guard the value would
        persist through model resolution and reach
        OPENCODE_CONFIG_CONTENT='{"model": "sonnet"}' where the
        opencode handshake rejects it as an unknown provider/model,
        with no Kai-side pointer back to the registry typo.
        """
        monkeypatch.setitem(MODEL_REGISTRY, ("opencode", "anthropic", ModelRole.BEHAVIORAL_JUDGE), "sonnet")
        with pytest.raises(SystemExit) as excinfo:
            _check_model_registry_complete()
        msg = str(excinfo.value)
        assert "provider/model" in msg
        assert "behavioral_judge" in msg

    def test_opencode_all_registry_rows_are_provider_slash_model_shape(self):
        """
        Lock the runtime invariant: every opencode registry row
        passes is_opencode_model_shape. Belt-and-braces companion
        to the synthetic-corruption test above; the registry rows
        are what operators edit when calibrating, and a regression
        here is the failure the synthetic test simulates.
        """
        from kai.config import BACKEND_PROVIDERS, is_opencode_model_shape

        for provider in BACKEND_PROVIDERS["opencode"]:
            for role in ModelRole:
                value = MODEL_REGISTRY[("opencode", provider, role)]
                assert is_opencode_model_shape(value), (
                    f"opencode/{provider} {role.value}={value!r} is not provider/model"
                )

    def test_completeness_check_raises_on_missing_opencode_row(self, monkeypatch):
        """
        Synthetic missing-row case for opencode: drop a registry
        entry and assert SystemExit with a clear message. Mirrors
        the claude test above so the completeness gate's per-backend
        symmetry stays pinned.
        """
        monkeypatch.delitem(MODEL_REGISTRY, ("opencode", "anthropic", ModelRole.PR_REVIEW))
        with pytest.raises(SystemExit) as excinfo:
            _check_model_registry_complete()
        assert "pr_review" in str(excinfo.value)
        assert "opencode" in str(excinfo.value)

    # ── Claude row equals prior constant (Phase 1 invariant) ──────

    def test_claude_pr_review_row_matches_prior_constant(self):
        """review.py used _REVIEW_MODEL = "sonnet" pre-Phase-1."""
        assert get_model_for(ModelRole.PR_REVIEW, "claude", "anthropic") == "sonnet"

    def test_claude_issue_triage_row_matches_prior_constant(self):
        """triage.py used _TRIAGE_MODEL = "sonnet" pre-Phase-1."""
        assert get_model_for(ModelRole.ISSUE_TRIAGE, "claude", "anthropic") == "sonnet"

    def test_claude_behavioral_judge_row_matches_prior_constant(self):
        """
        eval/behavioral.py used _DEFAULT_JUDGE_MODEL =
        "claude-haiku-4-5-20251001" pre-Phase-1. Locks the byte-identical
        invariant the behavioral codex path depends on: the claude
        BEHAVIORAL_JUDGE row must equal _DEFAULT_JUDGE_MODEL so an
        unset --judge-model on claude resolves to the same string the
        pre-Phase-1 argparse default emitted.
        """
        from kai.eval.behavioral import _DEFAULT_JUDGE_MODEL

        assert get_model_for(ModelRole.BEHAVIORAL_JUDGE, "claude", "anthropic") == _DEFAULT_JUDGE_MODEL

    def test_claude_behavioral_gen_row_matches_prior_constant(self):
        """eval/behavioral.py used _DEFAULT_GEN_MODEL = "sonnet" pre-Phase-1."""
        from kai.eval.behavioral import _DEFAULT_GEN_MODEL

        assert get_model_for(ModelRole.BEHAVIORAL_GEN, "claude", "anthropic") == _DEFAULT_GEN_MODEL


# ── Codex wizard hardening: backend-aware model validation ───────────


class TestCodexModelsSurface:
    """Lock the codex-vs-goose separation at the data layer.

    PROVIDER_MODELS["openai"] is goose's openai-API surface;
    CODEX_MODELS is codex CLI's separate surface. They are independent
    constants with no overlap requirement and no fallback path.
    """

    def test_codex_models_includes_gpt55_and_codex_variants(self):
        from kai.config import CODEX_MODELS

        assert "gpt-5.5" in CODEX_MODELS
        assert "gpt-5.3-codex" in CODEX_MODELS
        assert "gpt-5.3-codex-spark" in CODEX_MODELS
        assert "gpt-5.2" in CODEX_MODELS

    def test_codex_models_excludes_nano(self):
        from kai.config import CODEX_MODELS

        assert "gpt-5.4-nano" not in CODEX_MODELS

    def test_provider_models_openai_keeps_nano_for_goose(self):
        from kai.config import PROVIDER_MODELS

        assert "gpt-5.4-nano" in PROVIDER_MODELS["openai"]

    def test_codex_default_model_is_gpt55(self):
        from kai.config import CODEX_DEFAULT_MODEL

        assert CODEX_DEFAULT_MODEL == "gpt-5.5"

    def test_provider_defaults_openai_is_strongest_available(self):
        """PROVIDER_DEFAULTS["openai"] points at the OpenAI API's
        single strongest text model (per the agent-role-strongest
        rule). Verified 2026-06-09 against developers.openai.com."""
        from kai.config import PROVIDER_DEFAULTS

        assert PROVIDER_DEFAULTS["openai"] == "gpt-5.5-pro"


class TestValidateModelForBackend:
    """Backend-aware validator."""

    def test_codex_accepts_codex_models(self):
        from kai.config import validate_model_for_backend

        assert validate_model_for_backend("gpt-5.5", "codex", "openai") is True
        assert validate_model_for_backend("gpt-5.4-mini", "codex", "openai") is True

    def test_codex_rejects_nano_even_though_in_openai_provider_list(self):
        from kai.config import PROVIDER_MODELS, validate_model_for_backend

        assert "gpt-5.4-nano" in PROVIDER_MODELS["openai"]
        assert validate_model_for_backend("gpt-5.4-nano", "codex", "openai") is False

    def test_codex_rejects_claude_models(self):
        from kai.config import validate_model_for_backend

        assert validate_model_for_backend("opus", "codex", "openai") is False

    def test_goose_openai_accepts_nano(self):
        from kai.config import validate_model_for_backend

        assert validate_model_for_backend("gpt-5.4-nano", "goose", "openai") is True

    def test_claude_accepts_anthropic_models(self):
        from kai.config import validate_model_for_backend

        assert validate_model_for_backend("opus", "claude", "anthropic") is True

    def test_goose_anthropic_accepts_full_claude_ids(self):
        """Goose hands GOOSE_MODEL verbatim to the Anthropic API, so any
        claude-* ID passes structurally; a stale alias map must never be
        a ceiling on which SKUs a goose user can reach."""
        from kai.config import validate_model_for_backend

        assert validate_model_for_backend("claude-opus-4-8", "goose", "anthropic") is True
        assert validate_model_for_backend("claude-opus-4-7", "goose", "anthropic") is True
        assert validate_model_for_backend("claude-haiku-4-5-20251001", "goose", "anthropic") is True
        # The curated alias trio keeps working alongside the passthrough.
        assert validate_model_for_backend("opus", "goose", "anthropic") is True
        assert validate_model_for_backend("sonnet", "goose", "anthropic") is True

    def test_goose_anthropic_rejects_non_claude_garbage(self):
        """The structural passthrough is claude-* scoped; other strings
        still validate against the curated provider surface."""
        from kai.config import validate_model_for_backend

        assert validate_model_for_backend("gpt-5.5", "goose", "anthropic") is False
        assert validate_model_for_backend("clearly-bogus", "goose", "anthropic") is False

    def test_claude_accepts_full_claude_ids(self):
        """The claude CLI's --model flag resolves full model IDs, so
        any claude-* string passes structurally alongside the curated
        alias trio; pinning a previous generation must not require a
        backend switch."""
        from kai.config import validate_model_for_backend

        assert validate_model_for_backend("claude-opus-4-8", "claude", "anthropic") is True
        assert validate_model_for_backend("claude-opus-4-7", "claude", "anthropic") is True
        assert validate_model_for_backend("claude-haiku-4-5-20251001", "claude", "anthropic") is True
        # The curated alias trio keeps working alongside the passthrough.
        assert validate_model_for_backend("sonnet", "claude", "anthropic") is True

    def test_claude_rejects_non_claude_garbage(self):
        """The structural passthrough is claude-* scoped; other strings
        still validate against the curated provider surface."""
        from kai.config import validate_model_for_backend

        assert validate_model_for_backend("gpt-5.5", "claude", "anthropic") is False
        assert validate_model_for_backend("clearly-bogus", "claude", "anthropic") is False

    def test_goose_non_anthropic_providers_unaffected_by_passthrough(self):
        """A claude-* string on goose-with-another-provider is not a
        valid model for that provider's API; the passthrough must not
        leak across providers."""
        from kai.config import validate_model_for_backend

        assert validate_model_for_backend("claude-opus-4-8", "goose", "openai") is False

    def test_opencode_accepts_provider_slash_model_shape(self):
        """OpenCode requires the `provider/model` structural shape."""
        from kai.config import validate_model_for_backend

        assert validate_model_for_backend("anthropic/claude-sonnet-4-6", "opencode", "") is True
        assert validate_model_for_backend("openai/gpt-5.5", "opencode", "") is True
        assert validate_model_for_backend("opencode/big-pickle", "opencode", "") is True

    def test_opencode_rejects_bare_anthropic_names(self):
        """`opus` / `sonnet` are valid on claude/goose/codex but typos on opencode.

        The operator footgun the structural check guards against: a
        bare Anthropic name typed into /model on an opencode install
        would otherwise persist as OPENCODE_CONFIG_CONTENT='{"model":
        "opus"}' and fail at handshake without pointing back to the
        Kai-side typo.
        """
        from kai.config import validate_model_for_backend

        assert validate_model_for_backend("opus", "opencode", "anthropic") is False
        assert validate_model_for_backend("sonnet", "opencode", "anthropic") is False
        assert validate_model_for_backend("haiku", "opencode", "anthropic") is False

    def test_opencode_accepts_multi_slash_and_rejects_empty_segments(self):
        """Empty segments and missing separators fail; multi-slash
        nesting (openrouter-style `openrouter/anthropic/claude-...`)
        is accepted because opencode's provider layer parses any
        non-empty path-prefix shape."""
        from kai.config import validate_model_for_backend

        assert validate_model_for_backend("", "opencode", "") is False
        assert validate_model_for_backend("/foo", "opencode", "") is False
        assert validate_model_for_backend("foo/", "opencode", "") is False
        assert validate_model_for_backend("foo//bar", "opencode", "") is False
        # Multi-segment shape with all non-empty segments is valid;
        # opencode parses provider/<rest>.
        assert validate_model_for_backend("foo/bar/baz", "opencode", "") is True
        assert validate_model_for_backend("openrouter/anthropic/claude-sonnet-4-5", "opencode", "") is True
        assert validate_model_for_backend("/", "opencode", "") is False


class TestModelsForBackend:
    """Wizard/runtime model-keyboard list helper."""

    def test_codex_returns_codex_models(self):
        from kai.config import CODEX_MODELS, models_for_backend

        assert models_for_backend("codex", "openai") is CODEX_MODELS

    def test_goose_openai_returns_openai_provider_models(self):
        from kai.config import PROVIDER_MODELS, models_for_backend

        assert models_for_backend("goose", "openai") is PROVIDER_MODELS["openai"]

    def test_claude_returns_anthropic_provider_models(self):
        from kai.config import PROVIDER_MODELS, models_for_backend

        assert models_for_backend("claude", "anthropic") is PROVIDER_MODELS["anthropic"]

    def test_open_ended_provider_returns_none(self):
        from kai.config import models_for_backend

        assert models_for_backend("goose", "openrouter") is None

    def test_opencode_returns_none(self):
        """OpenCode has no curated keyboard; bot.py /model falls back to free text."""
        from kai.config import models_for_backend

        assert models_for_backend("opencode", "") is None
        assert models_for_backend("opencode", "anthropic") is None


class TestValidBackends:
    """Pin the VALID_BACKENDS set membership."""

    def test_all_four_backends_listed(self):
        """claude, goose, codex, opencode."""
        from kai.config import VALID_BACKENDS

        assert sorted(VALID_BACKENDS) == ["claude", "codex", "goose", "opencode"]

    def test_opencode_provider_prompt_gate(self):
        """OpenCode joins BACKENDS_NEEDING_PROVIDER_PROMPT (it talks
        to multiple providers and needs the operator to name one for
        the (backend, provider, role) registry lookup). The API-key
        sub-prompt is suppressed for opencode in install.py because
        opencode auth is managed by `opencode auth login`, not by
        Kai."""
        from kai.config import BACKEND_PROVIDERS, BACKENDS_NEEDING_PROVIDER_PROMPT

        assert "opencode" in BACKEND_PROVIDERS
        assert "opencode" in BACKENDS_NEEDING_PROVIDER_PROMPT
        # Single-provider backends are absent from the prompt gate so
        # their provider stays implicit.
        assert "claude" not in BACKENDS_NEEDING_PROVIDER_PROMPT
        assert "codex" not in BACKENDS_NEEDING_PROVIDER_PROMPT


class TestGetUserBackendAndProvider:
    """Per-user cascade resolver."""

    def _config(self, agent_backend="claude", llm_provider=""):
        cfg = MagicMock()
        cfg.agent_backend = agent_backend
        cfg.llm_provider = llm_provider
        return cfg

    def _user_config(self, agent_backend=None, llm_provider=None):
        uc = MagicMock()
        uc.agent_backend = agent_backend
        uc.llm_provider = llm_provider
        return uc

    def test_no_user_config_returns_global(self):
        from kai.config import get_user_backend_and_provider

        cfg = self._config(agent_backend="claude")
        backend, provider = get_user_backend_and_provider(None, cfg)
        assert backend == "claude"
        assert provider == "anthropic"

    def test_user_backend_overrides_global(self):
        from kai.config import get_user_backend_and_provider

        cfg = self._config(agent_backend="claude")
        uc = self._user_config(agent_backend="codex")
        backend, provider = get_user_backend_and_provider(uc, cfg)
        assert backend == "codex"
        assert provider == "openai"

    def test_global_codex_user_goose_returns_goose(self):
        from kai.config import get_user_backend_and_provider

        cfg = self._config(agent_backend="codex")
        uc = self._user_config(agent_backend="goose", llm_provider="openai")
        backend, provider = get_user_backend_and_provider(uc, cfg)
        assert backend == "goose"
        assert provider == "openai"

    def test_codex_provider_is_openai_regardless_of_llm_provider(self):
        from kai.config import get_user_backend_and_provider

        cfg = self._config(agent_backend="codex")
        uc = self._user_config(agent_backend="codex", llm_provider="anthropic")
        backend, provider = get_user_backend_and_provider(uc, cfg)
        assert backend == "codex"
        assert provider == "openai"


class TestAgentTimeoutSecondsRename:
    """CLAUDE_TIMEOUT_SECONDS -> AGENT_TIMEOUT_SECONDS rename with legacy alias."""

    def _required_env(self, monkeypatch, **overrides):
        for var in _CONFIG_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        base = {
            "TELEGRAM_BOT_TOKEN": "test-token",
            "ALLOWED_USER_IDS": "12345",
            "DEFAULT_MODEL": "sonnet",
            "WEBHOOK_PORT": "8080",
            "WEBHOOK_SECRET": "test-secret",
        }
        base.update(overrides)
        for k, v in base.items():
            monkeypatch.setenv(k, v)
        # users.yaml is mandatory post-#565 tranche A; patch a minimal
        # admin entry so load_config completes for these env-driven tests.
        _patch_protected_users_yaml(
            monkeypatch,
            "users:\n  - telegram_id: 12345\n    name: test\n    role: admin\n",
        )

    def test_agent_timeout_seconds_preferred(self, monkeypatch):
        self._required_env(
            monkeypatch,
            AGENT_TIMEOUT_SECONDS="180",
            CLAUDE_TIMEOUT_SECONDS="60",
        )
        cfg = load_config()
        assert cfg.agent_timeout_seconds == 180

    def test_legacy_only_falls_back_with_warning(self, monkeypatch, caplog):
        self._required_env(monkeypatch, CLAUDE_TIMEOUT_SECONDS="240")
        with caplog.at_level(logging.WARNING, logger="kai.config"):
            cfg = load_config()
        assert cfg.agent_timeout_seconds == 240
        assert any("CLAUDE_TIMEOUT_SECONDS is deprecated" in r.message for r in caplog.records)

    def test_neither_set_defaults_to_120(self, monkeypatch):
        self._required_env(monkeypatch)
        cfg = load_config()
        assert cfg.agent_timeout_seconds == 120


class TestAgentSessionLifecycleRename:
    """CLAUDE_MAX_SESSION_HOURS / CLAUDE_IDLE_TIMEOUT renamed to the
    AGENT_-prefixed forms with legacy aliases. The deprecation warning
    comes from the _renamed_env_vars map, which fires whenever a legacy
    key is present in the environment."""

    def _required_env(self, monkeypatch, **overrides):
        for var in _CONFIG_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        base = {
            "TELEGRAM_BOT_TOKEN": "test-token",
            "ALLOWED_USER_IDS": "12345",
            "DEFAULT_MODEL": "sonnet",
            "WEBHOOK_PORT": "8080",
            "WEBHOOK_SECRET": "test-secret",
        }
        base.update(overrides)
        for k, v in base.items():
            monkeypatch.setenv(k, v)
        _patch_protected_users_yaml(
            monkeypatch,
            "users:\n  - telegram_id: 12345\n    name: test\n    role: admin\n",
        )

    def test_agent_keys_preferred_over_legacy(self, monkeypatch):
        self._required_env(
            monkeypatch,
            AGENT_MAX_SESSION_HOURS="6",
            CLAUDE_MAX_SESSION_HOURS="2",
            AGENT_IDLE_TIMEOUT="900",
            CLAUDE_IDLE_TIMEOUT="300",
        )
        cfg = load_config()
        assert cfg.agent_max_session_hours == 6
        assert cfg.agent_idle_timeout == 900

    def test_legacy_only_falls_back_with_warning(self, monkeypatch, caplog):
        self._required_env(
            monkeypatch,
            CLAUDE_MAX_SESSION_HOURS="4.5",
            CLAUDE_IDLE_TIMEOUT="600",
        )
        with caplog.at_level(logging.WARNING, logger="kai.config"):
            cfg = load_config()
        assert cfg.agent_max_session_hours == 4.5
        assert cfg.agent_idle_timeout == 600
        messages = [r.getMessage() for r in caplog.records]
        assert any("CLAUDE_MAX_SESSION_HOURS in env is deprecated" in m for m in messages)
        assert any("CLAUDE_IDLE_TIMEOUT in env is deprecated" in m for m in messages)

    def test_neither_set_uses_defaults(self, monkeypatch):
        self._required_env(monkeypatch)
        cfg = load_config()
        assert cfg.agent_max_session_hours == 0
        assert cfg.agent_idle_timeout == 1800

    def test_invalid_idle_timeout_names_canonical_key(self, monkeypatch):
        """A bad value fails fast naming the canonical key, including
        when the bad value arrives through the legacy alias."""
        self._required_env(monkeypatch, CLAUDE_IDLE_TIMEOUT="soon")
        with pytest.raises(SystemExit, match="AGENT_IDLE_TIMEOUT"):
            load_config()


class TestLoadConfigBackendAwareModelValidation:
    """load_config DEFAULT_MODEL validation is backend-aware."""

    def _env(self, monkeypatch, **overrides):
        for var in _CONFIG_ENV_VARS:
            monkeypatch.delenv(var, raising=False)
        base = {
            "TELEGRAM_BOT_TOKEN": "test-token",
            "ALLOWED_USER_IDS": "12345",
            "WEBHOOK_PORT": "8080",
            "WEBHOOK_SECRET": "test-secret",
        }
        base.update(overrides)
        for k, v in base.items():
            monkeypatch.setenv(k, v)
        # users.yaml is mandatory post-#565 tranche A; patch a minimal
        # admin entry so the validation tests reach the model check.
        _patch_protected_users_yaml(
            monkeypatch,
            "users:\n  - telegram_id: 12345\n    name: test\n    role: admin\n",
        )

    def test_codex_rejects_goose_only_model(self, monkeypatch):
        self._env(monkeypatch, AGENT_BACKEND="codex", DEFAULT_MODEL="gpt-5.4-nano")
        with pytest.raises(SystemExit, match="not valid for codex"):
            load_config()

    def test_codex_rejects_claude_model(self, monkeypatch):
        self._env(monkeypatch, AGENT_BACKEND="codex", DEFAULT_MODEL="opus")
        with pytest.raises(SystemExit, match="not valid for codex"):
            load_config()

    def test_codex_accepts_gpt55(self, monkeypatch):
        self._env(monkeypatch, AGENT_BACKEND="codex", DEFAULT_MODEL="gpt-5.5")
        cfg = load_config()
        assert cfg.default_model == "gpt-5.5"


# ── Memory project registry loader (memory-projects.yaml) ────────────


class TestLoadMemoryProjects:
    """Tests for `_load_memory_project_configs()` and the
    surrounding integration with `load_config()`.

    The loader is fail-closed by design: malformed entries are
    skipped (logged) so detection later returns no project rather
    than producing accidental project-scoped recall. These tests
    pin that posture across every validation rule plus the
    interaction with `Config.allowed_workspaces`.
    """

    def _write_yaml(self, tmp_path: Path, content: str) -> Path:
        """Write a memory-projects.yaml under tmp_path; PROJECT_ROOT
        is patched to tmp_path so the loader's local-fallback branch
        picks it up without touching real config files."""
        yaml_file = tmp_path / "memory-projects.yaml"
        yaml_file.write_text(textwrap.dedent(content))
        return yaml_file

    def test_load_memory_projects_from_local_yaml(self, tmp_path):
        """Parses a valid two-project file and returns canonical
        resolved roots, the strict bool memory_enabled value, and
        the optional default-scope policy."""
        root_a = tmp_path / "project_a"
        root_a.mkdir()
        root_b = tmp_path / "project_b"
        root_b.mkdir()
        self._write_yaml(
            tmp_path,
            f"""\
            projects:
              - project_id: alpha
                display_name: Alpha
                workspace_roots:
                  - {root_a}
                memory_enabled: true
                default_scope_for_new_facts: project
              - project_id: beta
                display_name: Beta
                workspace_roots:
                  - {root_b}
                memory_enabled: false
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_memory_project_configs()

        assert set(configs.keys()) == {"alpha", "beta"}
        assert configs["alpha"].display_name == "Alpha"
        assert configs["alpha"].workspace_roots == (root_a.resolve(),)
        assert configs["alpha"].memory_enabled is True
        assert configs["alpha"].default_scope_for_new_facts == "project"
        assert configs["beta"].memory_enabled is False
        assert configs["beta"].default_scope_for_new_facts is None

    def test_memory_projects_absent_defaults_empty(self, tmp_path):
        """Neither protected nor local file exists -> empty dict."""
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_memory_project_configs()
        assert configs == {}

    def test_malformed_protected_memory_projects_returns_empty(self, tmp_path, caplog):
        """A malformed protected file fails closed at empty rather
        than silently falling through to the local-dev file. Pinning
        this stops a future refactor from re-introducing the
        dev-config-on-prod failure mode."""
        # _YAML_MALFORMED is the sentinel _read_protected_yaml returns
        # when YAML parsing fails. We import it from kai.config to
        # avoid leaking the sentinel value into the test surface.
        from kai.config import _YAML_MALFORMED

        # Even with a perfectly valid local file present, malformed
        # protected stops loading. Write the local file too so the
        # test would fail loudly if the fallthrough regressed.
        root = tmp_path / "p"
        root.mkdir()
        self._write_yaml(
            tmp_path,
            f"""\
            projects:
              - project_id: should_not_load
                display_name: Nope
                workspace_roots:
                  - {root}
                memory_enabled: true
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=_YAML_MALFORMED),
            patch("kai.config.PROJECT_ROOT", tmp_path),
            caplog.at_level("WARNING", logger="kai.config"),
        ):
            configs = _load_memory_project_configs()

        assert configs == {}
        assert "malformed" in caplog.text.lower() or "malformed" in caplog.text

    def test_invalid_memory_project_entry_is_skipped(self, tmp_path, caplog):
        """A missing required field on one entry skips only that
        entry; subsequent valid entries still load."""
        good_root = tmp_path / "good"
        good_root.mkdir()
        self._write_yaml(
            tmp_path,
            f"""\
            projects:
              - display_name: Missing Project Id
                workspace_roots:
                  - {good_root}
                memory_enabled: true
              - project_id: good
                display_name: Good
                workspace_roots:
                  - {good_root}
                memory_enabled: true
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
            caplog.at_level("WARNING", logger="kai.config"),
        ):
            configs = _load_memory_project_configs()

        assert list(configs.keys()) == ["good"]

    def test_memory_project_requires_boolean_memory_enabled(self, tmp_path):
        """memory_enabled set to a string is rejected outright. The
        loader does not coerce; the operator must use a real YAML
        bool."""
        root = tmp_path / "p"
        root.mkdir()
        self._write_yaml(
            tmp_path,
            f"""\
            projects:
              - project_id: p
                display_name: P
                workspace_roots:
                  - {root}
                memory_enabled: "true"
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_memory_project_configs()
        assert configs == {}

    def test_memory_project_requires_boolean_memory_enabled_rejects_yaml_truthy_non_bools(self, tmp_path):
        """Beyond the "true" string, YAML's other truthy non-bools
        (int 1/0 and "yes"/"no" strings) must also be rejected. Each
        variant gets its own project so a single passing variant
        cannot mask the others."""
        root = tmp_path / "p"
        root.mkdir()
        # YAML 1.1 would parse "yes" as bool True; PyYAML's default
        # safe_load is YAML 1.1, so "yes" already comes through as a
        # real bool. To genuinely test the "string yes" rejection we
        # quote it, forcing it to remain a string after parsing.
        self._write_yaml(
            tmp_path,
            f"""\
            projects:
              - project_id: int_one
                display_name: One
                workspace_roots:
                  - {root}
                memory_enabled: 1
              - project_id: int_zero
                display_name: Zero
                workspace_roots:
                  - {root}
                memory_enabled: 0
              - project_id: str_yes
                display_name: Yes
                workspace_roots:
                  - {root}
                memory_enabled: "yes"
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_memory_project_configs()
        # None of the three should have made it through; coercion of
        # YAML truthy non-bools is the failure mode we are gating.
        assert configs == {}

    def test_memory_project_rejects_invalid_default_scope(self, tmp_path):
        """default_scope_for_new_facts is gated to SCOPE_GLOBAL or
        SCOPE_PROJECT. SCOPE_TASK is not a write target in this
        issue and must be rejected; arbitrary strings likewise."""
        root = tmp_path / "p"
        root.mkdir()
        self._write_yaml(
            tmp_path,
            f"""\
            projects:
              - project_id: task_scope
                display_name: T
                workspace_roots:
                  - {root}
                memory_enabled: true
                default_scope_for_new_facts: task
              - project_id: nonsense
                display_name: N
                workspace_roots:
                  - {root}
                memory_enabled: true
                default_scope_for_new_facts: not_a_scope
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_memory_project_configs()
        assert configs == {}

    def test_memory_project_allows_nonexistent_roots(self, tmp_path):
        """Roots are NOT required to exist at load time. Registry
        may be authored before checkout exists or while a mount is
        unavailable; detection's longest-prefix match handles the
        absent-root case by failing to match (no false positive)."""
        nonexistent = tmp_path / "does_not_exist_yet"
        self._write_yaml(
            tmp_path,
            f"""\
            projects:
              - project_id: ghost
                display_name: Ghost
                workspace_roots:
                  - {nonexistent}
                memory_enabled: true
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_memory_project_configs()
        assert "ghost" in configs
        # Root is still resolved canonically even though it does not
        # exist. resolve() on a non-existing path returns the
        # absolute form without raising on macOS/Linux.
        assert configs["ghost"].workspace_roots == (nonexistent.resolve(),)

    def test_duplicate_memory_project_id_uses_first(self, tmp_path, caplog):
        """When the same project_id appears twice, the first entry
        wins and the later entry is logged + skipped. Pins the
        documented duplicate-id behavior."""
        root_a = tmp_path / "first"
        root_a.mkdir()
        root_b = tmp_path / "second"
        root_b.mkdir()
        self._write_yaml(
            tmp_path,
            f"""\
            projects:
              - project_id: dup
                display_name: First
                workspace_roots:
                  - {root_a}
                memory_enabled: true
              - project_id: dup
                display_name: Second
                workspace_roots:
                  - {root_b}
                memory_enabled: false
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
            caplog.at_level("WARNING", logger="kai.config"),
        ):
            configs = _load_memory_project_configs()
        assert configs["dup"].display_name == "First"
        assert configs["dup"].memory_enabled is True
        assert "duplicate project_id" in caplog.text

    def test_duplicate_memory_project_root_drops_later_duplicate_root(self, tmp_path, caplog):
        """When two distinct projects list the same root, the root
        is dropped from the LATER project only. The later project
        survives with its remaining unique roots."""
        shared_root = tmp_path / "shared"
        shared_root.mkdir()
        unique_root = tmp_path / "unique"
        unique_root.mkdir()
        self._write_yaml(
            tmp_path,
            f"""\
            projects:
              - project_id: first_owner
                display_name: First
                workspace_roots:
                  - {shared_root}
                memory_enabled: true
              - project_id: second_owner
                display_name: Second
                workspace_roots:
                  - {shared_root}
                  - {unique_root}
                memory_enabled: true
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
            caplog.at_level("WARNING", logger="kai.config"),
        ):
            configs = _load_memory_project_configs()
        assert "first_owner" in configs
        assert "second_owner" in configs
        # The shared root went to first_owner; second_owner keeps
        # only its unique root.
        assert configs["first_owner"].workspace_roots == (shared_root.resolve(),)
        assert configs["second_owner"].workspace_roots == (unique_root.resolve(),)
        assert "already owned" in caplog.text

    def test_duplicate_memory_project_root_drops_project_when_no_roots_remain(self, tmp_path, caplog):
        """When ALL of a later project's roots are duplicates of an
        earlier project's roots, the entire later project is dropped
        so detection never returns a project with no roots to match
        against."""
        shared_root = tmp_path / "shared"
        shared_root.mkdir()
        self._write_yaml(
            tmp_path,
            f"""\
            projects:
              - project_id: first_owner
                display_name: First
                workspace_roots:
                  - {shared_root}
                memory_enabled: true
              - project_id: orphaned
                display_name: Orphaned
                workspace_roots:
                  - {shared_root}
                memory_enabled: true
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
            caplog.at_level("WARNING", logger="kai.config"),
        ):
            configs = _load_memory_project_configs()
        assert "first_owner" in configs
        assert "orphaned" not in configs
        assert "no valid workspace_roots remain" in caplog.text

    def test_memory_project_roots_do_not_extend_allowed_workspaces(self, monkeypatch, tmp_path):
        """End-to-end load_config check: registry roots must NOT
        sneak into Config.allowed_workspaces. Workspace access is
        still owned by ALLOWED_WORKSPACES / workspaces.yaml; the
        memory registry describes scope, not access."""
        # Start with a fully minimal env to keep this independent of
        # other env-based test pollution.
        for v in _CONFIG_ENV_VARS:
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
        monkeypatch.setenv("ALLOWED_USER_IDS", "12345")
        monkeypatch.setenv("WEBHOOK_SECRET", "test-secret")
        _patch_protected_users_yaml(
            monkeypatch,
            "users:\n  - telegram_id: 12345\n    name: test\n    role: admin\n",
        )

        registry_root = tmp_path / "registry_root"
        registry_root.mkdir()
        # Write only the memory-projects.yaml; do NOT add the root to
        # workspaces.yaml or ALLOWED_WORKSPACES. The assertion is that
        # the registry root does not pull itself into allowed access.
        yaml_file = tmp_path / "memory-projects.yaml"
        yaml_file.write_text(
            textwrap.dedent(
                f"""\
                projects:
                  - project_id: scope_only
                    display_name: Scope Only
                    workspace_roots:
                      - {registry_root}
                    memory_enabled: true
                """
            )
        )

        with patch("kai.config.PROJECT_ROOT", tmp_path):
            cfg = load_config()

        # Registry loaded the project as expected.
        assert "scope_only" in cfg.memory_projects
        assert cfg.memory_projects["scope_only"].workspace_roots == (registry_root.resolve(),)
        # ...but did NOT promote its root into allowed_workspaces.
        assert registry_root.resolve() not in cfg.allowed_workspaces
        assert registry_root not in cfg.allowed_workspaces


# ── Shadow-mode toggle (#546) ────────────────────────────────────────


class TestMemoryRecallShadowConfig:
    """Tests for `Config.memory_recall_shadow_enabled` and its
    `MEMORY_RECALL_SHADOW_ENABLED` env-var parse.

    The toggle is default-on (inverted convention compared to the
    usual memory_* env vars) because the point of #546 is to
    collect real turn evidence before the read-path switch. A
    default-off flag would defeat that. The off-switch is the
    rollback path if shadow code misbehaves under load."""

    def _base_env(self, monkeypatch):
        """Minimal env so load_config builds cleanly. Caller adds
        MEMORY_RECALL_SHADOW_ENABLED on top to drive the test."""
        for v in _CONFIG_ENV_VARS:
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
        monkeypatch.setenv("ALLOWED_USER_IDS", "12345")
        monkeypatch.setenv("WEBHOOK_SECRET", "test-secret")
        monkeypatch.setenv("MEMORY_ENABLED", "true")  # gate for shadow to be on
        _patch_protected_users_yaml(
            monkeypatch,
            "users:\n  - telegram_id: 12345\n    name: test\n    role: admin\n",
        )

    def test_memory_recall_shadow_config_defaults_enabled(self, monkeypatch):
        """Unset env var → shadow enabled when memory is on. The
        default-on posture is the load-bearing decision in D10;
        flipping it to default-off would erase evidence collection."""
        self._base_env(monkeypatch)
        monkeypatch.delenv("MEMORY_RECALL_SHADOW_ENABLED", raising=False)
        cfg = load_config()
        assert cfg.memory_recall_shadow_enabled is True

    def test_memory_recall_shadow_config_can_disable(self, monkeypatch):
        """Each of the three disable strings turns the toggle off,
        case-insensitively. The disable set is intentionally
        conservative; truthy alternatives keep shadow on."""
        for disable_value in ("0", "false", "no", "False", "NO", "FALSE"):
            self._base_env(monkeypatch)
            monkeypatch.setenv("MEMORY_RECALL_SHADOW_ENABLED", disable_value)
            cfg = load_config()
            assert cfg.memory_recall_shadow_enabled is False, f"failed to disable with {disable_value!r}"

    def test_memory_recall_shadow_config_disabled_when_memory_off(self, monkeypatch):
        """Sub-toggle composition: shadow is gated on memory_enabled
        even when MEMORY_RECALL_SHADOW_ENABLED is unset/truthy. No
        legacy recall = no baseline to compare against = pointless
        shadow runs."""
        for v in _CONFIG_ENV_VARS:
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
        monkeypatch.setenv("ALLOWED_USER_IDS", "12345")
        monkeypatch.setenv("WEBHOOK_SECRET", "test-secret")
        _patch_protected_users_yaml(
            monkeypatch,
            "users:\n  - telegram_id: 12345\n    name: test\n    role: admin\n",
        )
        # MEMORY_ENABLED unset / off; shadow must also fall to off
        # regardless of the shadow env var.
        monkeypatch.setenv("MEMORY_RECALL_SHADOW_ENABLED", "true")
        cfg = load_config()
        assert cfg.memory_enabled is False
        assert cfg.memory_recall_shadow_enabled is False


class TestMemoryScopedRecallConfig:
    """Tests for `Config.memory_scoped_recall_enabled` and its
    `MEMORY_SCOPED_RECALL_ENABLED` env-var parse.

    Default-off, the OPPOSITE polarity of the shadow toggle above:
    shadow only observes, so it defaults on to collect evidence; the
    cutover changes live prompt content, so it stays off until the
    operator flips it deliberately."""

    def _base_env(self, monkeypatch):
        """Minimal env so load_config builds cleanly. Caller adds
        MEMORY_SCOPED_RECALL_ENABLED on top to drive the test."""
        for v in _CONFIG_ENV_VARS:
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
        monkeypatch.setenv("ALLOWED_USER_IDS", "12345")
        monkeypatch.setenv("WEBHOOK_SECRET", "test-secret")
        monkeypatch.setenv("MEMORY_ENABLED", "true")  # gate for the sub-toggle
        _patch_protected_users_yaml(
            monkeypatch,
            "users:\n  - telegram_id: 12345\n    name: test\n    role: admin\n",
        )

    def test_scoped_recall_defaults_off(self, monkeypatch):
        """Unset env var keeps the cutover off even with memory on;
        the default-off posture is what makes shipping the switch
        behavior-neutral."""
        self._base_env(monkeypatch)
        monkeypatch.delenv("MEMORY_SCOPED_RECALL_ENABLED", raising=False)
        cfg = load_config()
        assert cfg.memory_scoped_recall_enabled is False

    def test_scoped_recall_enable_values(self, monkeypatch):
        """Each explicit enable string turns the cutover on,
        case-insensitively."""
        for enable_value in ("1", "true", "yes", "TRUE", "Yes"):
            self._base_env(monkeypatch)
            monkeypatch.setenv("MEMORY_SCOPED_RECALL_ENABLED", enable_value)
            cfg = load_config()
            assert cfg.memory_scoped_recall_enabled is True, f"failed to enable with {enable_value!r}"

    def test_scoped_recall_non_enable_values_stay_off(self, monkeypatch):
        """Anything outside the enable set stays off; a typo must
        not flip live prompt content."""
        for off_value in ("0", "false", "no", "on", "enabled", "y"):
            self._base_env(monkeypatch)
            monkeypatch.setenv("MEMORY_SCOPED_RECALL_ENABLED", off_value)
            cfg = load_config()
            assert cfg.memory_scoped_recall_enabled is False, f"unexpectedly enabled with {off_value!r}"

    def test_scoped_recall_disabled_when_memory_off(self, monkeypatch):
        """Sub-toggle composition: scoped recall is gated on
        memory_enabled even when its own env var is set."""
        for v in _CONFIG_ENV_VARS:
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
        monkeypatch.setenv("ALLOWED_USER_IDS", "12345")
        monkeypatch.setenv("WEBHOOK_SECRET", "test-secret")
        _patch_protected_users_yaml(
            monkeypatch,
            "users:\n  - telegram_id: 12345\n    name: test\n    role: admin\n",
        )
        monkeypatch.setenv("MEMORY_SCOPED_RECALL_ENABLED", "true")
        cfg = load_config()
        assert cfg.memory_enabled is False
        assert cfg.memory_scoped_recall_enabled is False


class TestOneShotReasonerBackendsConstant:
    """Pin the contract of `ONESHOT_REASONER_BACKENDS`. This constant
    drives every site that gates on "does this backend support
    one-shot agent dispatch" (memory extraction, PR review, triage,
    smoke, behavioral eval, install-time MEMORY_* persistence). The
    tests below pin both its membership (the set of backends with a
    OneShotReasoner in `src/kai/oneshot.py`) and its type
    (`frozenset`, communicating immutability and membership-only
    intent). A type or membership drift would silently break the
    one-shot-eligible call sites; pin both invariants here so any
    future change is intentional.
    """

    def test_membership(self):
        """Every backend with a OneShotReasoner implementation in
        `src/kai/oneshot.py` is a member: claude, codex, opencode,
        and goose all ship one today.
        """
        assert "claude" in ONESHOT_REASONER_BACKENDS
        assert "codex" in ONESHOT_REASONER_BACKENDS
        assert "opencode" in ONESHOT_REASONER_BACKENDS
        assert "goose" in ONESHOT_REASONER_BACKENDS
        # Pin the exact contents so an accidental addition is caught
        # at test time; intentional additions update this assertion
        # in lockstep with the constant's definition.
        assert frozenset({"claude", "codex", "goose", "opencode"}) == ONESHOT_REASONER_BACKENDS

    def test_is_frozenset(self):
        """The constant must be a `frozenset` so callers only do
        membership checks; a mutable container would invite per-site
        mutation (`.add(...)`) that breaks the single-source-of-
        truth invariant the constant is designed to enforce.
        """
        assert isinstance(ONESHOT_REASONER_BACKENDS, frozenset)


class TestNoAdHocOneShotBackendTuples:
    """Regression guard for the "no ad-hoc literal tuples" invariant.

    Every site that gates on "is this backend one of the backends
    with a OneShotReasoner" must read from `ONESHOT_REASONER_BACKENDS`
    rather than typing a fresh literal tuple. Without this guard, a
    future contributor who adds a new gate can silently re-introduce
    the ad-hoc-tuple pattern that this constant was created to
    eliminate (and that cost the opencode rollout two follow-up
    fix-up PRs to correct).
    """

    def test_no_ad_hoc_oneshot_backend_tuples_in_source(self):
        """Production source files must use ONESHOT_REASONER_BACKENDS
        instead of literal claude/codex tuples for the one-shot-
        eligibility check. This pins the invariant the constant is
        meant to enforce: a future contributor who adds a new gate
        must read from the constant, not type a new literal.

        Allowed sites for the literal patterns:
        - The constant's own definition in `config.py`.
        - Module / function docstrings naming the set (informational).
        - String literals in error messages naming the set explicitly.

        Disallowed: a bare tuple `("claude", "codex")` or
        `("claude", "codex", "opencode")` used as a membership-check
        source.

        KNOWN BLIND SPOTS the pattern does NOT catch (acceptable
        best-effort scope; the constant's existence and the wider
        code-review discipline catch what regex cannot):

        - Reversed-order tuples (`("codex", "claude")`). No site uses
          this shape today; if a contributor introduces one, the next
          maintainer's grep catches it during review.
        - Multi-line tuples (`("claude",\\n    "codex",\\n    ...)`).
          The pattern is single-line. A contributor splitting the
          tuple across lines bypasses the test; black/ruff formatting
          keeps short tuples on one line in practice.
        - Tuples with extra elements (`("claude", "codex", "opencode",
          "goose")` or any other 4+ form). The pattern matches exactly
          the 2-element and 3-element forms that were the historical
          ad-hoc shape; a future widening to a literal 4-tuple would
          not trip the test.
        - Set / list / dict-keys forms of the same membership
          (`{"claude", "codex"}` or `["claude", "codex"]`). The
          operator-visible bug class this test guards against (the
          install.py extraction-keys cleanup gate) used tuples; sets
          and lists are rarer in this codebase but a future use
          would slip past.
        """
        import re
        from pathlib import Path

        pattern = re.compile(r'\(\s*"claude"\s*,\s*"codex"(\s*,\s*"opencode")?\s*\)')
        repo_root = Path(__file__).resolve().parent.parent
        src = repo_root / "src" / "kai"
        offenders: list[tuple[str, int, str]] = []
        for path in src.rglob("*.py"):
            # Allowed: the constant's own definition file.
            if path.name == "config.py":
                continue
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if pattern.search(line):
                    offenders.append((str(path.relative_to(repo_root)), lineno, line.strip()))
        assert not offenders, (
            f"Found ad-hoc claude/codex tuples; use ONESHOT_REASONER_BACKENDS instead. Offending sites: {offenders}"
        )


class TestProviderDefaultsStrongestModel:
    """`PROVIDER_DEFAULTS` drives the wizard's conversational-model
    prompt suggestion. Per the agent-role-strongest rule, each entry
    must name the strongest curated model the provider offers; the
    non-agent roles use the tier scheme elsewhere where balanced /
    cheap tiers split per role.
    """

    def test_anthropic_default_is_opus(self):
        """Opus is the strongest tier in PROVIDER_MODELS['anthropic']."""
        from kai.config import PROVIDER_DEFAULTS

        assert PROVIDER_DEFAULTS["anthropic"] == "opus"

    def test_openai_default_is_strongest(self):
        """gpt-5.5-pro is the single strongest text model on the
        OpenAI API per developers.openai.com (verified 2026-06-09).
        Above the gpt-5.5 / 5.4-pro / 5.4 / 5.4-mini / 5.4-nano line."""
        from kai.config import PROVIDER_DEFAULTS, PROVIDER_MODELS

        assert PROVIDER_DEFAULTS["openai"] in PROVIDER_MODELS["openai"]
        assert PROVIDER_DEFAULTS["openai"] == "gpt-5.5-pro"

    def test_google_default_is_pro(self):
        """gemini-2.5-pro is the most advanced Gemini for complex
        tasks per ai.google.dev/gemini-api/docs/models (verified
        2026-06-09). The 3.x family is mostly Preview / specialized."""
        from kai.config import PROVIDER_DEFAULTS

        assert PROVIDER_DEFAULTS["google"] == "gemini-2.5-pro"

    def test_deepseek_default_is_v4_pro(self):
        """V4 Pro is the strongest first-class OpenCode-on-DeepSeek
        model. V4 Flash is the speed tier. The legacy `deepseek-chat`
        alias is deprecated and not used as a default."""
        from kai.config import PROVIDER_DEFAULTS

        assert PROVIDER_DEFAULTS["deepseek"] == "deepseek-v4-pro"

    def test_every_default_is_in_provider_models(self):
        """Every default must be a key in PROVIDER_MODELS for the
        same provider; without this invariant the wizard's
        _prompt_choice would fail when offered the default as the
        pre-selected entry."""
        from kai.config import PROVIDER_DEFAULTS, PROVIDER_MODELS

        for provider, model in PROVIDER_DEFAULTS.items():
            assert provider in PROVIDER_MODELS, f"PROVIDER_MODELS missing entry for {provider!r}"
            assert model in PROVIDER_MODELS[provider], (
                f"PROVIDER_DEFAULTS[{provider!r}]={model!r} is not in PROVIDER_MODELS[{provider!r}]"
            )


class TestDeepSeekRegistryRowsAvoidDeprecatedAlias:
    """The `deepseek-chat` and `deepseek-reasoner` aliases are
    deprecated by DeepSeek and retire 2026-07-24. The registry
    must use the canonical V4 SKUs instead."""

    _DEPRECATED_NAMES = ("deepseek-chat", "deepseek-reasoner")

    def test_opencode_deepseek_rows_use_v4_skus(self):
        for role in ModelRole:
            value = MODEL_REGISTRY[("opencode", "deepseek", role)]
            for deprecated in self._DEPRECATED_NAMES:
                assert deprecated not in value, (
                    f"opencode/deepseek/{role.value}={value!r} contains deprecated alias {deprecated!r}"
                )
            assert value.startswith("deepseek/deepseek-v4-"), (
                f"opencode/deepseek/{role.value}={value!r} should resolve to a V4 SKU"
            )

    def test_goose_deepseek_rows_use_v4_skus(self):
        for role in ModelRole:
            value = MODEL_REGISTRY[("goose", "deepseek", role)]
            for deprecated in self._DEPRECATED_NAMES:
                assert deprecated not in value, (
                    f"goose/deepseek/{role.value}={value!r} contains deprecated alias {deprecated!r}"
                )
            assert value.startswith("deepseek-v4-"), f"goose/deepseek/{role.value}={value!r} should resolve to a V4 SKU"

    def test_balanced_and_cheap_tiers_differ_on_deepseek(self):
        """Both tiers collapsed onto `deepseek-chat` before this fix;
        re-pinning the tier distinction prevents a future regression."""
        opencode_balanced = MODEL_REGISTRY[("opencode", "deepseek", ModelRole.PR_REVIEW)]
        opencode_cheap = MODEL_REGISTRY[("opencode", "deepseek", ModelRole.MEMORY_EXTRACTION)]
        assert opencode_balanced != opencode_cheap, (
            "opencode/deepseek balanced and cheap tiers must resolve to different models"
        )

        goose_balanced = MODEL_REGISTRY[("goose", "deepseek", ModelRole.PR_REVIEW)]
        goose_cheap = MODEL_REGISTRY[("goose", "deepseek", ModelRole.MEMORY_EXTRACTION)]
        assert goose_balanced != goose_cheap, "goose/deepseek balanced and cheap tiers must resolve to different models"


class TestBackendProviders:
    """`BACKEND_PROVIDERS` is the single authoritative (backend, provider)
    allowlist. `BACKENDS_NEEDING_PROVIDER_PROMPT` is derived from it
    via the multiplicity filter so single-provider backends (claude,
    codex) bypass the "requires provider" gates while multi-provider
    backends (opencode, goose) exercise them.
    """

    def test_membership_contents(self):
        from kai.config import BACKEND_PROVIDERS

        assert set(BACKEND_PROVIDERS.keys()) == {"claude", "codex", "opencode", "goose"}
        assert BACKEND_PROVIDERS["claude"] == ("anthropic",)
        assert BACKEND_PROVIDERS["codex"] == ("openai",)
        # Multi-provider backends include deepseek (new); openrouter and ollama
        # are open-ended providers and live here too.
        assert "deepseek" in BACKEND_PROVIDERS["opencode"]
        assert "deepseek" in BACKEND_PROVIDERS["goose"]
        assert "openrouter" in BACKEND_PROVIDERS["opencode"]

    def test_strict_subset_of_valid_backends(self):
        """Every backend listed in BACKEND_PROVIDERS must also be in
        VALID_BACKENDS so the wizard's backend prompt accepts it."""
        from kai.config import BACKEND_PROVIDERS, VALID_BACKENDS

        assert set(BACKEND_PROVIDERS.keys()).issubset(VALID_BACKENDS)

    def test_backends_needing_provider_prompt_is_derived(self):
        """Derived set: single-provider backends absent; multi-provider
        backends present. Pins the multiplicity-driven shape."""
        from kai.config import BACKEND_PROVIDERS, BACKENDS_NEEDING_PROVIDER_PROMPT

        for backend, providers in BACKEND_PROVIDERS.items():
            if len(providers) > 1:
                assert backend in BACKENDS_NEEDING_PROVIDER_PROMPT
            else:
                assert backend not in BACKENDS_NEEDING_PROVIDER_PROMPT


class TestModelRegistryTripleKey:
    """The (backend, provider, role) triple-key registry is built
    mechanically from `_TIER_BY_ROLE` and `_BACKEND_PROVIDER_TIER_MODELS`.
    Every (backend, provider) pair in BACKEND_PROVIDERS has a row for
    every ModelRole; the completeness check runs at load_config time.
    """

    def test_keys_are_three_tuples(self):
        """Every key is a 3-tuple (backend, provider, ModelRole)."""
        for key in MODEL_REGISTRY:
            assert isinstance(key, tuple)
            assert len(key) == 3
            backend, provider, role = key
            assert isinstance(backend, str)
            assert isinstance(provider, str)
            assert isinstance(role, ModelRole)

    def test_every_backend_provider_pair_has_every_role(self):
        """Completeness invariant: BACKEND_PROVIDERS x ModelRole all
        present in MODEL_REGISTRY. A missing triple at runtime would
        surface as a per-request LookupError; the startup check ahead
        of dispatch is what makes the runtime invariant safe."""
        from kai.config import BACKEND_PROVIDERS

        for backend, providers in BACKEND_PROVIDERS.items():
            for provider in providers:
                for role in ModelRole:
                    assert (backend, provider, role) in MODEL_REGISTRY, (
                        f"Missing registry row for ({backend}, {provider}, {role.value})"
                    )

    def test_canonical_rows_per_backend(self):
        """Pin one representative row per backend so a future
        _BACKEND_PROVIDER_TIER_MODELS tweak surfaces here in addition
        to the larger acceptance loop above."""
        assert MODEL_REGISTRY[("claude", "anthropic", ModelRole.PR_REVIEW)] == "sonnet"
        # codex balanced tier picks the current frontier (gpt-5.5);
        # cheap tier stays at gpt-5.4-mini for high-volume roles.
        assert MODEL_REGISTRY[("codex", "openai", ModelRole.PR_REVIEW)] == "gpt-5.5"
        assert MODEL_REGISTRY[("codex", "openai", ModelRole.MEMORY_EXTRACTION)] == "gpt-5.4-mini"
        # opencode-on-deepseek balanced tier resolves to V4 Pro for
        # the reasoning-heavy PR review role; V4 Flash covers the
        # cheap tier elsewhere. The legacy `deepseek-chat` alias is
        # deprecated and not used here.
        assert MODEL_REGISTRY[("opencode", "deepseek", ModelRole.PR_REVIEW)] == "deepseek/deepseek-v4-pro"
        assert MODEL_REGISTRY[("opencode", "deepseek", ModelRole.MEMORY_EXTRACTION)] == "deepseek/deepseek-v4-flash"
        assert MODEL_REGISTRY[("goose", "anthropic", ModelRole.PR_REVIEW)] == "claude-sonnet-4-6"

    def test_codex_rows_are_in_codex_models(self):
        """The completeness check enforces this at startup; pinning
        it independently catches drift between CODEX_MODELS and the
        tier map without needing to trigger load_config."""
        from kai.config import CODEX_MODELS

        for role in ModelRole:
            value = MODEL_REGISTRY[("codex", "openai", role)]
            assert value in CODEX_MODELS, f"codex {role.value}={value} is not in CODEX_MODELS"


class TestUserConfigModelsField:
    """`UserConfig.models` is the per-user per-role override map; back-
    compat synthesizes `models["agent"]` from the legacy `model:` field
    so existing users.yaml files load unchanged."""

    def test_field_default_is_none(self):
        from kai.config import UserConfig

        uc = UserConfig(telegram_id=1, name="test")
        assert uc.models is None

    def test_models_field_accepts_per_role_dict(self):
        from kai.config import UserConfig

        uc = UserConfig(
            telegram_id=1,
            name="test",
            models={"pr_review": "deepseek/deepseek-coder", "agent": "sonnet"},
        )
        assert uc.models == {"pr_review": "deepseek/deepseek-coder", "agent": "sonnet"}


class TestResolveUserModel:
    """`resolve_user_model` enforces the per-role precedence chain:
    user_config.models > config.default_models > MODEL_REGISTRY."""

    def _config(self, default_models=None):
        cfg = MagicMock()
        cfg.agent_backend = "claude"
        cfg.llm_provider = ""
        cfg.default_models = default_models or {}
        return cfg

    def test_falls_back_to_registry_when_no_overrides(self):
        from kai.config import resolve_user_model

        result = resolve_user_model(ModelRole.PR_REVIEW, None, self._config())
        assert result == "sonnet"

    def test_user_override_wins(self):
        from kai.config import UserConfig, resolve_user_model

        uc = UserConfig(telegram_id=1, name="test", models={"pr_review": "opus"})
        result = resolve_user_model(ModelRole.PR_REVIEW, uc, self._config())
        assert result == "opus"

    def test_global_default_models_wins_over_registry(self):
        from kai.config import resolve_user_model

        cfg = self._config(default_models={"pr_review": "claude-haiku-4-5-20251001"})
        result = resolve_user_model(ModelRole.PR_REVIEW, None, cfg)
        assert result == "claude-haiku-4-5-20251001"

    def test_user_override_wins_over_global_default(self):
        from kai.config import UserConfig, resolve_user_model

        uc = UserConfig(telegram_id=1, name="test", models={"pr_review": "opus"})
        cfg = self._config(default_models={"pr_review": "claude-haiku-4-5-20251001"})
        result = resolve_user_model(ModelRole.PR_REVIEW, uc, cfg)
        assert result == "opus"


class TestLegacyEnvOverrideSeeding:
    """`_apply_legacy_model_env_overrides` reads
    PR_REVIEW_MODEL_<BACKEND> / ISSUE_TRIAGE_MODEL_<BACKEND> from the
    process env at load_config time and seeds UserConfig.models. The
    user's own `models:` map wins over the env-var seed."""

    def test_seeds_pr_review_from_env(self, monkeypatch):
        from kai.config import UserConfig, _apply_legacy_model_env_overrides

        monkeypatch.setenv("PR_REVIEW_MODEL_CLAUDE", "opus")
        uc = UserConfig(telegram_id=1, name="test")
        out = _apply_legacy_model_env_overrides({1: uc}, "claude")
        assert out[1].models == {"pr_review": "opus"}

    def test_user_own_map_wins_over_env(self, monkeypatch):
        from kai.config import UserConfig, _apply_legacy_model_env_overrides

        monkeypatch.setenv("PR_REVIEW_MODEL_CLAUDE", "opus")
        uc = UserConfig(telegram_id=1, name="test", models={"pr_review": "haiku"})
        out = _apply_legacy_model_env_overrides({1: uc}, "claude")
        # User's explicit value untouched.
        assert out[1].models == {"pr_review": "haiku"}

    def test_seeds_only_matching_backend_suffix(self, monkeypatch):
        """A user on the codex backend should NOT pick up
        PR_REVIEW_MODEL_CLAUDE; the suffix gates per-user scope."""
        from kai.config import UserConfig, _apply_legacy_model_env_overrides

        monkeypatch.setenv("PR_REVIEW_MODEL_CLAUDE", "opus")
        uc = UserConfig(telegram_id=1, name="test", agent_backend="codex")
        out = _apply_legacy_model_env_overrides({1: uc}, "claude")
        assert out[1].models is None

    def test_no_env_no_change(self, monkeypatch):
        """When no env override applies, the dict shape is preserved."""
        from kai.config import UserConfig, _apply_legacy_model_env_overrides

        monkeypatch.delenv("PR_REVIEW_MODEL_CLAUDE", raising=False)
        monkeypatch.delenv("ISSUE_TRIAGE_MODEL_CLAUDE", raising=False)
        uc = UserConfig(telegram_id=1, name="test")
        out = _apply_legacy_model_env_overrides({1: uc}, "claude")
        assert out[1].models is None
