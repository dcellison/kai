"""Tests for the protected installation module (install.py)."""

import contextlib
import json
import os
import pwd
import shutil
import signal
import sqlite3
import stat
import subprocess
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

import kai.install
from kai.install import (
    _LAUNCHD_LABEL,
    ServiceStartError,
    _apply_backend_registry,
    _apply_directories,
    _apply_goose_config,
    _apply_migrate,
    _apply_models,
    _apply_runtime_policy,
    _apply_secrets,
    _apply_service,
    _apply_source,
    _apply_sudoers,
    _apply_venv,
    _backend_command_trust_issues,
    _build_migrated_runtime_profiles,
    _check_path,
    _check_service_status,
    _check_traversal,
    _cmd_apply,
    _cmd_config,
    _cmd_status,
    _collect_backends_from_yaml,
    _collect_os_users_from_yaml,
    _collect_user_memory_owners,
    _copy_managed_home_tree,
    _copy_tree,
    _deployed_webhook_secret_migration_status,
    _file_checksum,
    _generate_env_file,
    _generate_launchd_plist,
    _generate_launcher_script,
    _generate_sudoers,
    _generate_systemd_unit,
    _generate_users_yaml,
    _migrate_identity_to_claude_md,
    _migrate_managed_home_database_paths,
    _optional_file_checksum,
    _prompt_choice,
    _prompt_optional_choice,
    _read_users_yaml_text,
    _resolve_codex_bin_prompt_default,
    _retire_install_home_claude,
    _retire_install_home_dir,
    _runtime_policy_apply_plan,
    _runtime_policy_status,
    _runtime_storage_status,
    _runtime_storage_targets,
    _secure_codex_turn_image_staging,
    _secure_history_directories,
    _secure_upload_directories,
    _set_ownership,
    _set_static_install_tree_modes,
    _src_checksum,
    _start_service,
    _stop_service,
    _user_home,
    _users_yaml_agent_backends,
    _users_yaml_goose_providers,
    _validate_chat_id,
    _validate_display_name,
    _validate_os_user,
    _validate_port,
    _validate_positive_int,
    _validate_telegram_id,
    _validate_user_ids,
    _webhook_secret_migration_status,
    cli,
)
from kai.workshop.bootstrap import (
    BootstrapHuman,
    bootstrap_default_workshop,
    bootstrap_human_principal_id,
)
from kai.workshop.domain import RuntimeProfileId, WorkshopId
from kai.workshop.runtime_profiles import ProtectedRuntimeProfile, WorkshopRuntimeProfileRegistry
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id


@pytest.fixture(autouse=True)
def _isolate_installed_backend_discovery(monkeypatch, tmp_path):
    """Keep install tests independent of host-installed backend CLIs.

    CI runners have none of claude/codex/goose/opencode/pi installed,
    while developer machines may have a real /etc/kai/backends.yaml.
    Most install tests are not about command discovery, so give them a
    deterministic registry-shaped discovery result. Tests that exercise
    missing/discovered backend behavior override this fixture locally.
    """
    monkeypatch.setattr("kai.install.BACKENDS_YAML", tmp_path / "absent-backends.yaml")
    monkeypatch.setattr("kai.install.RUNTIME_PROFILES_YAML", tmp_path / "absent-runtime-profiles.yaml")
    monkeypatch.setattr("kai.install._DEPLOYED_ENV_FILE", tmp_path / "absent-env")
    monkeypatch.setattr(
        "kai.install._discover_backend_commands",
        lambda service_user: {
            "claude": "/test/bin/claude",
            "codex": "/test/bin/codex",
            "goose": "/test/bin/goose",
            "opencode": "/test/bin/opencode",
            "pi": "/test/bin/pi",
        },
    )
    monkeypatch.setattr("kai.install.replace_named_read_access", MagicMock())


# ── Validation helpers ───────────────────────────────────────────────


class TestValidateUserIds:
    def test_single_id(self):
        assert _validate_user_ids("123") is True

    def test_multiple_ids(self):
        assert _validate_user_ids("123,456,789") is True

    def test_with_spaces(self):
        assert _validate_user_ids("123, 456") is True

    def test_empty_string(self):
        assert _validate_user_ids("") is False

    def test_non_numeric(self):
        assert _validate_user_ids("abc") is False

    def test_negative(self):
        assert _validate_user_ids("-1") is False

    def test_zero(self):
        assert _validate_user_ids("0") is False


class TestValidateTelegramId:
    def test_positive_integer(self):
        assert _validate_telegram_id("123456789") is True

    def test_zero(self):
        assert _validate_telegram_id("0") is False

    def test_negative(self):
        assert _validate_telegram_id("-1") is False

    def test_non_numeric(self):
        assert _validate_telegram_id("abc") is False

    def test_empty_string(self):
        assert _validate_telegram_id("") is False

    def test_strips_whitespace(self):
        # int() strips whitespace naturally
        assert _validate_telegram_id(" 123 ") is True


class TestValidateDisplayName:
    def test_simple_name(self):
        assert _validate_display_name("alice") is True

    def test_with_spaces(self):
        assert _validate_display_name("Alice Smith") is True

    def test_with_hyphens_underscores(self):
        assert _validate_display_name("alice-smith_01") is True

    def test_yaml_special_colon(self):
        assert _validate_display_name("alice: admin") is False

    def test_yaml_special_hash(self):
        assert _validate_display_name("bob # test") is False

    def test_empty_string(self):
        assert _validate_display_name("") is False

    def test_whitespace_only(self):
        assert _validate_display_name("   ") is False


class TestValidateOsUser:
    def test_simple_name(self):
        assert _validate_os_user("kai") is True

    def test_with_dot(self):
        assert _validate_os_user("kai.user") is True

    def test_with_hyphen_underscore(self):
        assert _validate_os_user("kai-user_01") is True

    def test_yaml_special_colon(self):
        assert _validate_os_user("kai: admin") is False

    def test_space(self):
        assert _validate_os_user("kai user") is False

    def test_empty_string(self):
        assert _validate_os_user("") is False


class TestValidatePort:
    def test_valid_port(self):
        assert _validate_port("8080") is True

    def test_port_1(self):
        assert _validate_port("1") is True

    def test_port_65535(self):
        assert _validate_port("65535") is True

    def test_port_0(self):
        assert _validate_port("0") is False

    def test_port_too_high(self):
        assert _validate_port("65536") is False

    def test_port_non_numeric(self):
        assert _validate_port("abc") is False


class TestValidatePositiveInt:
    def test_valid(self):
        assert _validate_positive_int("120") is True

    def test_zero(self):
        assert _validate_positive_int("0") is False

    def test_negative(self):
        assert _validate_positive_int("-5") is False

    def test_float(self):
        assert _validate_positive_int("1.5") is False


# ── File checksum ────────────────────────────────────────────────────


class TestFileChecksum:
    def test_returns_sha256(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = _file_checksum(f)
        assert len(result) == 64  # SHA-256 hex digest length
        assert result.isalnum()

    def test_missing_file_returns_empty(self, tmp_path):
        assert _file_checksum(tmp_path / "nope.txt") == ""

    def test_same_content_same_hash(self, tmp_path):
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("same")
        b.write_text("same")
        assert _file_checksum(a) == _file_checksum(b)

    def test_different_content_different_hash(self, tmp_path):
        a = tmp_path / "a.txt"
        b = tmp_path / "b.txt"
        a.write_text("one")
        b.write_text("two")
        assert _file_checksum(a) != _file_checksum(b)


# ── Generation functions ─────────────────────────────────────────────


class TestGenerateEnvFile:
    def test_produces_key_value_lines(self):
        env = {"TOKEN": "abc123", "PORT": "8080"}
        result = _generate_env_file(env)
        assert 'PORT="8080"' in result
        assert 'TOKEN="abc123"' in result

    def test_sorted_keys(self):
        env = {"Z_KEY": "z", "A_KEY": "a"}
        result = _generate_env_file(env)
        lines = [line for line in result.splitlines() if "=" in line and not line.startswith("#")]
        assert lines[0].startswith("A_KEY=")
        assert lines[1].startswith("Z_KEY=")

    def test_includes_header_comment(self):
        result = _generate_env_file({"K": "V"})
        assert result.startswith("#")


class TestGenerateUsersYaml:
    def test_minimal(self):
        """Minimal entry has telegram_id, name, and role."""
        content = _generate_users_yaml("123456789", "alice")
        data = yaml.safe_load(content)
        assert isinstance(data["users"], list)
        assert len(data["users"]) == 1
        entry = data["users"][0]
        assert entry["telegram_id"] == 123456789  # int, not string
        assert entry["name"] == "alice"
        assert entry["role"] == "admin"
        assert entry["allowed_services"] == []
        assert "os_user" not in entry
        assert "home_workspace" not in entry

    def test_with_optional_fields(self):
        """Optional os_user and home_workspace are included when set."""
        content = _generate_users_yaml(
            "123456789",
            "alice",
            os_user="kai",
            home_workspace="/opt/kai/home",
        )
        data = yaml.safe_load(content)
        entry = data["users"][0]
        assert entry["os_user"] == "kai"
        assert entry["home_workspace"] == "/opt/kai/home"

    def test_roundtrip_with_loader(self, monkeypatch):
        """Generated YAML can be parsed by _load_user_configs("claude", "")."""
        from kai.config import _load_user_configs

        content = _generate_users_yaml("123456789", "alice", os_user="kai")
        parsed = yaml.safe_load(content)
        # The runtime loader consumes the parsed dict returned by
        # `_read_protected_yaml`. Patch that surface directly rather
        # than writing a file the loader no longer reads.
        monkeypatch.setattr("kai.config._read_protected_yaml", lambda _: parsed)
        configs = _load_user_configs("claude", "")
        assert configs is not None
        assert 123456789 in configs
        assert configs[123456789].name == "alice"
        assert configs[123456789].role == "admin"
        assert configs[123456789].os_user == "kai"

    def test_includes_header_comment(self):
        """Generated file starts with a header comment."""
        content = _generate_users_yaml("123", "test")
        assert content.startswith("# Kai user configuration")

    def test_trailing_newline(self):
        """Generated file ends with a trailing newline."""
        content = _generate_users_yaml("123", "test")
        assert content.endswith("\n")

    def test_yaml_boolean_keywords_roundtrip(self):
        """YAML 1.1 boolean keywords in name/os_user survive roundtrip."""
        content = _generate_users_yaml("123", "yes", os_user="no")
        data = yaml.safe_load(content)
        entry = data["users"][0]
        # Must be strings, not booleans
        assert entry["name"] == "yes"
        assert isinstance(entry["name"], str)
        assert entry["os_user"] == "no"
        assert isinstance(entry["os_user"], str)

    def test_os_user_with_trailing_dots_not_corrupted(self):
        """os_user ending in '...' must not be truncated by document end stripping."""
        content = _generate_users_yaml("123", "alice", os_user="test...")
        data = yaml.safe_load(content)
        assert data["users"][0]["os_user"] == "test..."


class TestGenerateSudoers:
    def test_contains_user(self):
        result = _generate_sudoers("kai")
        assert "kai ALL=" in result

    def test_contains_cat_rules(self):
        """Sudoers uses the resolved cat path (may be /bin/cat or /usr/bin/cat)."""
        result = _generate_sudoers("testuser")
        cat_path = shutil.which("cat") or "/bin/cat"
        assert f"{cat_path} /etc/kai/env" in result
        assert f"{cat_path} /etc/kai/services.yaml" in result
        assert f"{cat_path} /etc/kai/users.yaml" in result
        assert f"{cat_path} /etc/kai/workspaces.yaml" in result
        assert f"{cat_path} /etc/kai/runtime-profiles.yaml" in result
        assert f"{cat_path} /etc/kai/memory-projects.yaml" in result
        assert f"{cat_path} /etc/kai/backends.yaml" in result
        assert f"{cat_path} /etc/kai/totp.secret" in result
        assert f"{cat_path} /etc/kai/totp.attempts" in result

    def test_contains_tee_rule(self):
        """Sudoers uses the resolved tee path (may be /usr/bin/tee)."""
        result = _generate_sudoers("kai")
        tee_path = shutil.which("tee") or "/usr/bin/tee"
        assert f"{tee_path} /etc/kai/totp.attempts" in result

    def test_nopasswd(self):
        result = _generate_sudoers("kai")
        assert "NOPASSWD" in result

    def test_no_per_user_rule_without_os_users(self):
        """No claude binary rule when os_users is empty."""
        result = _generate_sudoers("kai")
        assert "claude" not in result

    def test_os_user_rule_anchored_to_service_user_home(self, monkeypatch):
        """
        The rule's claude binary path is anchored to the SERVICE user's
        home (~/.local/bin/claude under the service user, NOT the target
        user), because the bot's runtime spawn is `sudo -u <target> --
        claude` and sudo resolves the bare `claude` against the caller's
        (service user's) PATH. The rule path must match what sudo will
        actually try to execute.
        """
        monkeypatch.setattr("kai.install._user_home", lambda u: f"/home/{u}")
        result = _generate_sudoers("kai", os_users=["alice"])
        assert "kai ALL=(alice) SETENV: NOPASSWD: /home/kai/.local/bin/claude" in result

    def test_claude_bin_ignores_caller_path(self, monkeypatch):
        """
        Regression for issue #454-adjacent install bug: the rule's claude
        binary path must NOT depend on whatever PATH the install caller
        happens to have. Pre-fix, `shutil.which("claude")` resolved
        against root's PATH at install time and silently baked a
        non-service-user binary path into the rule (e.g. when `sudo make
        install` was launched from a shell with another user's
        `~/.local/bin` on PATH), breaking the bot's sudo dispatch on
        every subsequent message.
        """
        monkeypatch.setattr("kai.install._user_home", lambda u: f"/home/{u}")
        # Caller's PATH would resolve `claude` to a completely different
        # location; the generator must ignore this and anchor on the
        # service user's home.
        real_which = shutil.which
        monkeypatch.setattr(
            shutil,
            "which",
            lambda n: "/some/other/users/local/bin/claude" if n == "claude" else real_which(n),
        )
        result = _generate_sudoers("kai", os_users=["alice"])
        assert "/home/kai/.local/bin/claude" in result
        assert "/some/other/users/local/bin/claude" not in result

    def test_claude_bin_homebrew_fallback_when_service_native_missing(self, monkeypatch):
        """On macOS, a missing service-user native Claude install falls
        back to the Homebrew cask path so runtime backend switches can
        use Claude when the direct formatter fallback is exercised."""
        monkeypatch.setattr("kai.install._user_home", lambda u: "/Users/kai")
        monkeypatch.setattr(
            "kai.install._validate_claude_bin",
            lambda p: p == "/opt/homebrew/bin/claude",
        )

        result = _generate_sudoers("kai", os_users=["alice"])

        assert "kai ALL=(alice) SETENV: NOPASSWD: /opt/homebrew/bin/claude" in result
        assert "/Users/kai/.local/bin/claude" not in result

    def test_claude_bin_arg_overrides_default(self, monkeypatch):
        """An explicit claude_bin replaces the service-home fallback
        in the per-user rule. Protected install callers pass this from
        the backend registry."""
        monkeypatch.setattr("kai.install._user_home", lambda u: f"/home/{u}")
        result = _generate_sudoers("kai", os_users=["alice"], claude_bin="/opt/homebrew/bin/claude")
        assert "kai ALL=(alice) SETENV: NOPASSWD: /opt/homebrew/bin/claude" in result
        assert "/home/kai/.local/bin/claude" not in result

    # -- Issue #341: per-user rules from users.yaml -----------------------

    def test_no_per_user_rule_when_os_users_empty(self, monkeypatch):
        """Acceptance (a): zero os_user entries → no per-user rules emitted."""
        monkeypatch.setattr("kai.install._user_home", lambda u: f"/home/{u}")
        result = _generate_sudoers("kai", os_users=[])
        assert "SETENV: NOPASSWD" not in result
        assert "claude" not in result

    def test_one_os_user_emits_one_rule(self, monkeypatch):
        """Acceptance (b): one os_user → one matching rule."""
        monkeypatch.setattr("kai.install._user_home", lambda u: f"/home/{u}")
        result = _generate_sudoers("kai", os_users=["sellison"])
        assert result.count("SETENV: NOPASSWD: /home/kai/.local/bin/claude") == 1
        assert "kai ALL=(sellison) SETENV: NOPASSWD: /home/kai/.local/bin/claude" in result

    def test_multiple_os_users_emit_one_rule_each(self, monkeypatch):
        """Acceptance (c): multiple distinct os_users → one rule each."""
        monkeypatch.setattr("kai.install._user_home", lambda u: f"/home/{u}")
        result = _generate_sudoers("kai", os_users=["sellison", "bob", "carol"])
        bin_path = "/home/kai/.local/bin/claude"
        assert f"kai ALL=(sellison) SETENV: NOPASSWD: {bin_path}" in result
        assert f"kai ALL=(bob) SETENV: NOPASSWD: {bin_path}" in result
        assert f"kai ALL=(carol) SETENV: NOPASSWD: {bin_path}" in result
        assert result.count(f"SETENV: NOPASSWD: {bin_path}") == 3

    def test_duplicate_os_users_deduped(self, monkeypatch):
        """Acceptance (c, dedupe): repeated os_user values produce one ruleset each."""
        monkeypatch.setattr("kai.install._user_home", lambda u: f"/home/{u}")
        result = _generate_sudoers("kai", os_users=["sellison", "sellison", "bob"])
        # Each target gets three rules: claude SETENV (#456), codex
        # SETENV (per-user codex OAuth isolation), and kill. The
        # claude rule's binary path uniquely identifies one rule per
        # target, so counting on the full prefix verifies dedup
        # without the kill-rule or codex-rule noise inflating the
        # count.
        claude_bin = "/home/kai/.local/bin/claude"
        assert result.count(f"kai ALL=(sellison) SETENV: NOPASSWD: {claude_bin}") == 1
        assert result.count(f"kai ALL=(bob) SETENV: NOPASSWD: {claude_bin}") == 1

    def test_os_user_matching_service_user_skipped(self, monkeypatch):
        """Acceptance (d): os_user == service_user → no rule (self-sudo path)."""
        monkeypatch.setattr("kai.install._user_home", lambda u: f"/home/{u}")
        result = _generate_sudoers("kai", os_users=["kai", "sellison"])
        # No self-sudo rule: kai ALL=(kai) would be a dead rule, since
        # resolve_claude_user() short-circuits the sudo wrapper at runtime.
        assert "kai ALL=(kai)" not in result
        bin_path = "/home/kai/.local/bin/claude"
        assert f"kai ALL=(sellison) SETENV: NOPASSWD: {bin_path}" in result
        assert result.count(f"SETENV: NOPASSWD: {bin_path}") == 1

    # -- Codex per-target rule (per-user OAuth isolation) ----------------

    def test_codex_rule_emitted_per_target(self, monkeypatch):
        """
        Each per-target ruleset includes a NOPASSWD rule for the codex
        binary alongside the claude rule. The bot spawns
        `sudo -H -u <target> -- codex app-server` so codex reads
        ~<target>/.codex/auth.json instead of the service user's home;
        without this sudoers rule, the spawn fails with "a password is
        required" and the codex backend cannot start for any user
        whose os_user differs from the service user.
        """
        monkeypatch.setattr("kai.install._user_home", lambda u: f"/home/{u}")
        # shutil.which("codex") may not find codex in the test env; the
        # function falls back to /opt/homebrew/bin/codex in that case.
        # Either path is a valid codex location.
        result = _generate_sudoers("kai", os_users=["alice", "bob"])
        # Each target gets exactly one codex SETENV rule. The binary
        # path is not pinned in the assertion (shutil.which is
        # environment-dependent); match on the rule prefix instead.
        assert "kai ALL=(alice) SETENV: NOPASSWD: " in result
        assert "kai ALL=(bob) SETENV: NOPASSWD: " in result
        # Specifically: a line referencing "codex" appears under each target.
        alice_lines = [line for line in result.splitlines() if "(alice)" in line and "codex" in line]
        bob_lines = [line for line in result.splitlines() if "(bob)" in line and "codex" in line]
        assert len(alice_lines) == 1, alice_lines
        assert len(bob_lines) == 1, bob_lines

    def test_pi_rule_emitted_per_target_from_registry_path(self):
        result = _generate_sudoers(
            "kai",
            os_users=["alice", "bob"],
            pi_bin="/opt/homebrew/bin/pi",
        )

        assert "kai ALL=(alice) SETENV: NOPASSWD: /opt/homebrew/bin/pi" in result
        assert "kai ALL=(bob) SETENV: NOPASSWD: /opt/homebrew/bin/pi" in result

    # -- Issue #456: per-target kill rule for cross-user signal escalation ----

    def test_kill_rule_emitted_per_target(self, monkeypatch):
        """
        Each per-target ruleset includes a NOPASSWD rule for the kill
        binary alongside the claude binary rule. The bot uses this to
        escalate signals to the inner claude grandchild via
        `sudo -n -u <target> /bin/kill -<sig> <pid>` in cross-user
        mode, because POSIX signal permissions prevent the service
        user from signaling a target-user process directly.
        """
        monkeypatch.setattr("kai.install._user_home", lambda u: f"/home/{u}")
        result = _generate_sudoers("kai", os_users=["alice", "bob"])
        assert "kai ALL=(alice) NOPASSWD: /bin/kill" in result
        assert "kai ALL=(bob) NOPASSWD: /bin/kill" in result

    def test_kill_rule_omitted_when_no_targets(self, monkeypatch):
        """No targets → no kill rules (no cross-user spawn to escalate to)."""
        monkeypatch.setattr("kai.install._user_home", lambda u: f"/home/{u}")
        result = _generate_sudoers("kai", os_users=[])
        assert "kill" not in result

    def test_kill_rule_no_setenv(self, monkeypatch):
        """
        The kill rule must NOT carry SETENV: - kill ignores env, and
        a SETENV grant on the kill path would broaden the env-var
        exposure surface beyond what the claude rule needs. Only the
        claude rule keeps SETENV (for KAI_WEBHOOK_SECRET, TMPDIR).
        """
        monkeypatch.setattr("kai.install._user_home", lambda u: f"/home/{u}")
        result = _generate_sudoers("kai", os_users=["alice"])
        # Filter to actual sudoers rule lines (skip the comment
        # block above the per-target rules - it mentions "kill" in
        # English explaining the scope tradeoff).
        rule_lines = [line for line in result.splitlines() if line.startswith("kai ALL=")]
        kill_lines = [line for line in rule_lines if "kill" in line]
        assert len(kill_lines) == 1
        assert "SETENV" not in kill_lines[0]
        # The claude rule still has SETENV (unchanged by #456).
        claude_lines = [line for line in rule_lines if "claude" in line]
        assert len(claude_lines) == 1
        assert "SETENV" in claude_lines[0]

    def test_kill_rule_path_is_hardcoded(self, monkeypatch):
        """
        The kill rule must always emit /bin/kill regardless of what
        shutil.which("kill") would resolve to on the install host.
        Pre-fix the generator used `shutil.which("kill") or
        "/bin/kill"` which on Linux could bake `/usr/bin/kill` into
        the sudoers rule; the runtime invokes `/bin/kill` literally,
        and a path mismatch causes sudo to silently fail the
        escalation (recreating the orphan-leak bug this PR aims to
        close). Regression guard: even when shutil.which would have
        returned a different path, the rule still says /bin/kill.
        """
        monkeypatch.setattr("kai.install._user_home", lambda u: f"/home/{u}")
        # Guard: the production code no longer calls
        # shutil.which("kill") - kill_bin is the literal
        # "/bin/kill". This monkeypatch is intentionally a no-op
        # against the current implementation; it exists so that
        # if a future change re-introduces the `shutil.which("kill")
        # or "/bin/kill"` pattern that PR #458 removed, this
        # assertion would catch the resulting path mismatch
        # (which would silently break cross-user kill escalation
        # on Linux hosts where shutil.which returns /usr/bin/kill).
        real_which = shutil.which
        monkeypatch.setattr(
            shutil,
            "which",
            lambda n: "/usr/bin/kill" if n == "kill" else real_which(n),
        )
        result = _generate_sudoers("kai", os_users=["alice"])
        assert "kai ALL=(alice) NOPASSWD: /bin/kill" in result
        assert "/usr/bin/kill" not in result


class TestCollectOsUsersFromYaml:
    """Loader for issue #341 — extract distinct os_user values from users.yaml."""

    def test_missing_file_returns_empty(self, tmp_path):
        """First-install path: users.yaml does not exist yet."""
        result = _collect_os_users_from_yaml(tmp_path / "nope.yaml")
        assert result == []

    def test_empty_file_returns_empty(self, tmp_path):
        path = tmp_path / "users.yaml"
        path.write_text("", encoding="utf-8")
        assert _collect_os_users_from_yaml(path) == []

    def test_whitespace_only_returns_empty(self, tmp_path):
        path = tmp_path / "users.yaml"
        path.write_text("   \n\t\n", encoding="utf-8")
        assert _collect_os_users_from_yaml(path) == []

    def test_no_users_key_returns_empty(self, tmp_path):
        path = tmp_path / "users.yaml"
        path.write_text("other_key: 1\n", encoding="utf-8")
        assert _collect_os_users_from_yaml(path) == []

    def test_users_not_list_returns_empty(self, tmp_path):
        path = tmp_path / "users.yaml"
        path.write_text("users: not_a_list\n", encoding="utf-8")
        assert _collect_os_users_from_yaml(path) == []

    def test_no_os_user_field_returns_empty(self, tmp_path):
        """Users without os_user (the default — bot's service user) are skipped."""
        path = tmp_path / "users.yaml"
        path.write_text(
            yaml.safe_dump({"users": [{"telegram_id": 1, "name": "alice"}]}),
            encoding="utf-8",
        )
        assert _collect_os_users_from_yaml(path) == []

    def test_one_os_user_returned(self, tmp_path):
        path = tmp_path / "users.yaml"
        path.write_text(
            yaml.safe_dump({"users": [{"telegram_id": 1, "os_user": "sellison"}]}),
            encoding="utf-8",
        )
        assert _collect_os_users_from_yaml(path) == ["sellison"]

    def test_multiple_distinct_os_users_returned(self, tmp_path):
        path = tmp_path / "users.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "users": [
                        {"telegram_id": 1, "os_user": "sellison"},
                        {"telegram_id": 2, "os_user": "bob"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        assert _collect_os_users_from_yaml(path) == ["sellison", "bob"]

    def test_duplicates_deduped_preserving_first_seen_order(self, tmp_path):
        path = tmp_path / "users.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "users": [
                        {"telegram_id": 1, "os_user": "sellison"},
                        {"telegram_id": 2, "os_user": "bob"},
                        {"telegram_id": 3, "os_user": "sellison"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        assert _collect_os_users_from_yaml(path) == ["sellison", "bob"]

    def test_empty_string_os_user_skipped(self, tmp_path):
        path = tmp_path / "users.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "users": [
                        {"telegram_id": 1, "os_user": ""},
                        {"telegram_id": 2, "os_user": "  "},
                        {"telegram_id": 3, "os_user": "sellison"},
                    ]
                }
            ),
            encoding="utf-8",
        )
        assert _collect_os_users_from_yaml(path) == ["sellison"]

    def test_non_string_os_user_skipped(self, tmp_path):
        """PyYAML may parse "yes"/"no"/numbers as bool/int; filter those out."""
        path = tmp_path / "users.yaml"
        # Hand-craft YAML to force bool/int values for os_user.
        path.write_text(
            "users:\n"
            "  - telegram_id: 1\n"
            "    os_user: true\n"
            "  - telegram_id: 2\n"
            "    os_user: 42\n"
            "  - telegram_id: 3\n"
            "    os_user: sellison\n",
            encoding="utf-8",
        )
        assert _collect_os_users_from_yaml(path) == ["sellison"]

    def test_os_user_is_stripped(self, tmp_path):
        path = tmp_path / "users.yaml"
        path.write_text(
            yaml.safe_dump({"users": [{"telegram_id": 1, "os_user": "  sellison  "}]}),
            encoding="utf-8",
        )
        assert _collect_os_users_from_yaml(path) == ["sellison"]

    def test_malformed_yaml_raises(self, tmp_path):
        """Corrupt YAML must surface so install fails loudly, not silently."""
        path = tmp_path / "users.yaml"
        path.write_text("users:\n  - [unclosed\n", encoding="utf-8")
        with pytest.raises(yaml.YAMLError):
            _collect_os_users_from_yaml(path)

    # -- Security: validate os_user before sudoers write (PR #342 review) -

    @pytest.mark.parametrize(
        "bad_value",
        [
            "alice) NOPASSWD: ALL",  # closing-paren injection
            "alice\nroot ALL=(ALL) NOPASSWD: ALL",  # newline injection
            "alice ALL",  # whitespace
            "alice;bob",  # semicolon
            "alice=bob",  # equals
            "alice/bob",  # slash
            "alice@host",  # at-sign
        ],
    )
    def test_invalid_os_user_raises(self, tmp_path, bad_value):
        """Crafted os_user values that could inject sudoers directives must raise."""
        path = tmp_path / "users.yaml"
        path.write_text(
            yaml.safe_dump({"users": [{"telegram_id": 1, "os_user": bad_value}]}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Invalid os_user"):
            _collect_os_users_from_yaml(path)

    def test_invalid_os_user_error_names_path(self, tmp_path):
        """ValueError message includes the offending path so the operator can fix it."""
        path = tmp_path / "users.yaml"
        path.write_text(
            yaml.safe_dump({"users": [{"telegram_id": 1, "os_user": "bad)user"}]}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=str(path)):
            _collect_os_users_from_yaml(path)


class TestReadUsersYamlText:
    """The wizard reads users.yaml as the operator account. The
    deployed `/etc/kai/users.yaml` is mode 0600 root-owned, so a
    direct `read_text()` raises `PermissionError`; `_read_users_yaml_text`
    falls back to `sudo cat`. Both branches plus the missing-file
    and unreadable-via-sudo cases are pinned here so a future
    refactor of the helper does not regress the protected path."""

    def test_local_readable_path_returns_content(self, tmp_path):
        """Direct read works when the file is operator-readable."""
        path = tmp_path / "users.yaml"
        path.write_text("users: []\n", encoding="utf-8")
        assert _read_users_yaml_text(path) == "users: []\n"

    def test_missing_file_returns_none(self, tmp_path):
        assert _read_users_yaml_text(tmp_path / "absent.yaml") is None

    def test_permission_error_falls_back_to_sudo_cat(self, monkeypatch, tmp_path):
        """The protected `/etc/kai/users.yaml` path: direct read
        raises PermissionError, the fallback shells out to `sudo cat`
        and returns its stdout."""
        path = tmp_path / "protected.yaml"

        def _raise_perm(self, encoding="utf-8"):
            raise PermissionError(13, "Permission denied", str(self))

        monkeypatch.setattr(Path, "read_text", _raise_perm)

        captured: dict = {}

        def _fake_subprocess_run(argv, capture_output, text, timeout):
            captured["argv"] = argv

            class _R:
                returncode = 0
                stdout = "users:\n  - telegram_id: 1\n    name: alice\n    role: admin\n"

            return _R()

        monkeypatch.setattr("kai.install.subprocess.run", _fake_subprocess_run)
        out = _read_users_yaml_text(path)
        assert captured["argv"] == ["sudo", "cat", str(path)]
        assert "alice" in out

    def test_sudo_failure_returns_none(self, monkeypatch, tmp_path):
        """When sudo refuses (no password, no rule, timeout, etc.)
        the helper returns None so the caller's missing-content
        branch fires."""
        path = tmp_path / "protected.yaml"

        def _raise_perm(self, encoding="utf-8"):
            raise PermissionError(13, "Permission denied", str(self))

        monkeypatch.setattr(Path, "read_text", _raise_perm)

        def _fake_subprocess_run(argv, capture_output, text, timeout):
            class _R:
                returncode = 1
                stdout = ""

            return _R()

        monkeypatch.setattr("kai.install.subprocess.run", _fake_subprocess_run)
        assert _read_users_yaml_text(path) is None


class TestCollectUserMemoryOwners:
    """
    Tests for the (telegram_id, os_user) loader used by the per-user
    MEMORY.md migration in #347. Mirrors the structure of
    TestCollectOsUsersFromYaml: same trust boundary, same validation
    semantics, but returns tuples instead of just usernames.
    """

    def test_missing_file_returns_empty(self, tmp_path):
        assert _collect_user_memory_owners(tmp_path / "nope.yaml") == []

    def test_empty_file_returns_empty(self, tmp_path):
        path = tmp_path / "users.yaml"
        path.write_text("", encoding="utf-8")
        assert _collect_user_memory_owners(path) == []

    def test_no_users_key_returns_empty(self, tmp_path):
        path = tmp_path / "users.yaml"
        path.write_text("other_key: 1\n", encoding="utf-8")
        assert _collect_user_memory_owners(path) == []

    def test_user_without_telegram_id_skipped(self, tmp_path):
        """A user entry without an int telegram_id cannot map to a chat dir."""
        path = tmp_path / "users.yaml"
        path.write_text(
            yaml.safe_dump({"users": [{"name": "anon"}, {"telegram_id": 42}]}),
            encoding="utf-8",
        )
        assert _collect_user_memory_owners(path) == [(42, None)]

    def test_user_without_os_user_returns_none(self, tmp_path):
        """No os_user means the inner Claude runs as the service user."""
        path = tmp_path / "users.yaml"
        path.write_text(
            yaml.safe_dump({"users": [{"telegram_id": 11, "name": "primary"}]}),
            encoding="utf-8",
        )
        assert _collect_user_memory_owners(path) == [(11, None)]

    def test_user_with_os_user_returns_tuple(self, tmp_path):
        path = tmp_path / "users.yaml"
        path.write_text(
            yaml.safe_dump({"users": [{"telegram_id": 11, "os_user": "alpha"}]}),
            encoding="utf-8",
        )
        assert _collect_user_memory_owners(path) == [(11, "alpha")]

    def test_first_seen_order_preserved(self, tmp_path):
        """The migration depends on tuples[0] being the primary operator."""
        path = tmp_path / "users.yaml"
        path.write_text(
            yaml.safe_dump(
                {
                    "users": [
                        {"telegram_id": 100, "os_user": "alpha"},
                        {"telegram_id": 200, "os_user": "beta"},
                        {"telegram_id": 300},
                    ]
                }
            ),
            encoding="utf-8",
        )
        assert _collect_user_memory_owners(path) == [
            (100, "alpha"),
            (200, "beta"),
            (300, None),
        ]

    def test_invalid_os_user_raises(self, tmp_path):
        """Same hard-fail as _collect_os_users_from_yaml: never silently skip."""
        path = tmp_path / "users.yaml"
        path.write_text(
            yaml.safe_dump({"users": [{"telegram_id": 1, "os_user": "bad)user"}]}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match=str(path)):
            _collect_user_memory_owners(path)


class TestGenerateSudoersValidation:
    """Defense-in-depth: _generate_sudoers itself must reject bad targets."""

    def test_invalid_os_user_in_os_users_raises(self):
        """Bad os_users values must raise even if they bypass the loader."""
        with pytest.raises(ValueError, match="Invalid sudoers target user"):
            _generate_sudoers("kai", os_users=["alice) NOPASSWD: ALL"])


class TestGenerateLaunchdPlist:
    def test_contains_label(self):
        result = _generate_launchd_plist("/opt/kai", "/var/lib/kai", "kai")
        assert _LAUNCHD_LABEL in result

    def test_contains_install_dir(self):
        result = _generate_launchd_plist("/opt/kai", "/var/lib/kai", "kai")
        # Plist uses launcher script, not python directly
        assert "/opt/kai/run.sh" in result
        assert "<string>/opt/kai</string>" in result

    def test_contains_data_dir(self):
        result = _generate_launchd_plist("/opt/kai", "/var/lib/kai", "kai")
        assert "/var/lib/kai" in result

    def test_contains_username(self):
        result = _generate_launchd_plist("/opt/kai", "/var/lib/kai", "myuser")
        assert "myuser" in result

    def test_sets_kai_data_dir_env(self):
        result = _generate_launchd_plist("/opt/kai", "/var/lib/kai", "kai")
        assert "KAI_DATA_DIR" in result

    def test_valid_xml_structure(self):
        result = _generate_launchd_plist("/opt/kai", "/var/lib/kai", "kai")
        assert result.startswith("<?xml")
        assert "</plist>" in result

    def test_captures_stderr_and_stdout_to_log_files(self):
        """Without StandardErrorPath / StandardOutPath, launchd routes
        Python's stderr and stdout to /dev/null. An early-init crash
        (missing env var, SystemExit before logging is configured)
        then leaves no trail while the bash wrapper's tracked PID
        keeps `launchctl print` reporting `state = running`. Pin both
        keys plus the {data_dir}/logs/ derivation so a future refactor
        cannot silently drop the visibility surface."""
        result = _generate_launchd_plist("/opt/kai", "/var/lib/kai", "kai")
        assert "<key>StandardErrorPath</key>" in result
        assert "<string>/var/lib/kai/logs/kai.stderr.log</string>" in result
        assert "<key>StandardOutPath</key>" in result
        assert "<string>/var/lib/kai/logs/kai.stdout.log</string>" in result

    def test_log_paths_derive_from_data_dir(self):
        """A non-default data_dir lands the log files under that
        directory, not under a hardcoded /var/lib/kai/. Catches a
        regression where the paths were ever inlined as literal
        strings instead of f-string interpolations of `data_dir`."""
        result = _generate_launchd_plist("/opt/kai", "/srv/kai", "kai")
        assert "<string>/srv/kai/logs/kai.stderr.log</string>" in result
        assert "<string>/srv/kai/logs/kai.stdout.log</string>" in result


class TestGenerateSystemdUnit:
    def test_contains_user(self):
        result = _generate_systemd_unit("/opt/kai", "/var/lib/kai", "kai")
        assert "User=kai" in result

    def test_contains_exec_start(self):
        result = _generate_systemd_unit("/opt/kai", "/var/lib/kai", "kai")
        assert "ExecStart=/opt/kai/venv/bin/python -m kai" in result

    def test_contains_data_dir_env(self):
        result = _generate_systemd_unit("/opt/kai", "/var/lib/kai", "kai")
        assert "KAI_DATA_DIR=/var/lib/kai" in result

    def test_network_dependency(self):
        result = _generate_systemd_unit("/opt/kai", "/var/lib/kai", "kai")
        assert "network-online.target" in result


# ── Prompt helpers ───────────────────────────────────────────────────


class TestPromptChoice:
    """
    `_prompt_choice` defends its "returns a value in `choices`" contract.

    The empty-input branch previously returned the caller's `default`
    without checking membership; an out-of-list default could leak back
    to the caller on Enter. The helper now treats an out-of-list default
    as if no default were supplied: the suffix is omitted and empty input
    re-prompts.
    """

    def test_returns_default_when_in_choices(self, monkeypatch):
        """Regression guard for the no-behavior-change case."""
        # Empty input simulates the operator pressing Enter at the prompt.
        inputs = iter([""])
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
        result = _prompt_choice("Pick", ["a", "b"], default="a")
        assert result == "a"

    def test_rejects_empty_input_when_default_not_in_choices(self, monkeypatch, capsys):
        """
        With a default that is not in choices, pressing Enter re-prompts
        rather than returning the invalid value. The "Please choose one
        of" notice is printed to stdout (via print(), captured by capsys).
        """
        # First Enter triggers the re-prompt; the second response ("a")
        # is in choices so the function returns it. Pre-fix behavior
        # would have returned "xyz" on the first Enter without re-prompting.
        inputs = iter(["", "a"])
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
        result = _prompt_choice("Pick", ["a", "b"], default="xyz")
        assert result == "a"
        captured = capsys.readouterr().out
        assert "Please choose one of: a/b" in captured

    def test_suffix_omitted_when_default_not_in_choices(self, monkeypatch):
        """
        The `[default]` suffix in the prompt string is dropped when the
        default is out-of-list, so the function does not advertise a
        value it will not accept on Enter. Couples the suffix display to
        the empty-input branch via the shared `effective_default` variable.
        """
        # Record the exact string passed to input() so we can assert on
        # the suffix. The bare lambda used in test_returns_default does
        # not preserve the prompt argument; capsys also will not catch
        # it because the mock replaces input() entirely and never writes
        # to stdout. A side-effecting list is the structural analog of
        # MagicMock.call_args_list for this case.
        recorded_prompts: list[str] = []
        inputs = iter(["a"])

        def mock_input(prompt: str) -> str:
            recorded_prompts.append(prompt)
            return next(inputs)

        monkeypatch.setattr("builtins.input", mock_input)
        _prompt_choice("Pick", ["a", "b"], default="xyz")
        assert "[xyz]" not in recorded_prompts[0]

    def test_suffix_present_when_default_in_choices(self, monkeypatch):
        """
        Pairs with `test_suffix_omitted_when_default_not_in_choices` to
        lock both directions of the suffix display.
        """
        recorded_prompts: list[str] = []
        inputs = iter([""])

        def mock_input(prompt: str) -> str:
            recorded_prompts.append(prompt)
            return next(inputs)

        monkeypatch.setattr("builtins.input", mock_input)
        _prompt_choice("Pick", ["a", "b"], default="a")
        assert "[a]" in recorded_prompts[0]

    def test_empty_default_with_invalid_input_reprompts(self, monkeypatch):
        """
        Regression guard for the empty-default case: invalid typed input
        re-prompts (not the empty-input path), valid typed input is
        accepted on the second iteration.
        """
        # "bad" is not in choices; "a" is. The function should loop until
        # it gets a value in choices.
        inputs = iter(["bad", "a"])
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
        result = _prompt_choice("Pick", ["a", "b"], default="")
        assert result == "a"

    def test_empty_default_with_empty_input_reprompts(self, monkeypatch):
        """
        Locks the unset-default row of the behavior matrix: empty input
        with no default re-prompts rather than returning a falsy value.
        The single-Enter iterator runs the loop once; the second input()
        call exhausts the iterator and raises StopIteration, which proves
        the loop is iterating (it would return without a second read if
        the empty-input branch were incorrectly returning the empty
        default).
        """
        inputs = iter([""])
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
        with pytest.raises(StopIteration):
            _prompt_choice("Pick", ["a", "b"], default="")

    def test_lowercases_and_strips_input(self, monkeypatch):
        """
        Existing behavior, kept as a load-bearing regression guard since
        callers depend on the .strip().lower() normalization to accept
        operator input in any case.
        """
        inputs = iter([" A "])
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
        result = _prompt_choice("Pick", ["a", "b"], default="")
        assert result == "a"

    def test_rejects_case_mismatched_default(self, monkeypatch):
        """
        Direct regression test for the case-sensitivity claim in the
        docstring. "A" is not byte-identical to any entry in ["a", "b"],
        so the default is treated as out-of-list; the typed "a" is
        accepted on the next iteration. If a future change relaxes the
        membership check to be case-insensitive, this test will fail and
        the docstring claim should be revisited.
        """
        inputs = iter(["", "a"])
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
        result = _prompt_choice("Pick", ["a", "b"], default="A")
        assert result == "a"


class TestPromptOptionalChoice:
    """
    `_prompt_optional_choice` covers the set-or-absent prompt shape:
    empty input is a first-class valid answer, the prefill is normalized
    via `.strip().lower()` before the `in choices` check, and the inline
    empty-default hint is shown only when there is no usable prefill.

    The Codex effort prompt is the seed call site; these tests pin the
    helper's contract independently of any specific caller.
    """

    @staticmethod
    def _record(monkeypatch, inputs: list[str]) -> list[str]:
        """
        Install an `input` stub that records every prompt string and
        returns the next scripted answer. Returned list is the live
        prompts buffer; assertions read it after the helper returns.
        """
        recorded_prompts: list[str] = []
        it = iter(inputs)

        def mock_input(prompt: str) -> str:
            recorded_prompts.append(prompt)
            return next(it)

        monkeypatch.setattr("builtins.input", mock_input)
        return recorded_prompts

    def test_empty_input_returns_empty_when_default_empty(self, monkeypatch):
        """
        With no prefill, pressing Enter is the operator's "use the
        downstream default" signal and the helper returns "". The
        inline hint advertising the empty-default path is also shown
        in the prompt so a first-time operator sees it.
        """
        prompts = self._record(monkeypatch, [""])
        result = _prompt_optional_choice("Pick", ["a", "b", "c"], default="", empty_hint="empty = pick default")
        assert result == ""
        assert "a/b/c" in prompts[0]
        assert "empty = pick default" in prompts[0]

    def test_empty_input_returns_default_when_default_in_choices(self, monkeypatch):
        """
        A valid prefill round-trips on Enter. The empty-default hint is
        suppressed in the prompt so the operator does not see two
        different "what happens on Enter" answers at once.
        """
        prompts = self._record(monkeypatch, [""])
        result = _prompt_optional_choice("Pick", ["a", "b", "c"], default="b", empty_hint="empty = pick default")
        assert result == "b"
        assert "[b]" in prompts[0]
        assert "empty = pick default" not in prompts[0]

    def test_empty_input_returns_normalized_default_when_default_case_or_whitespace_differs(self, monkeypatch):
        """
        A copy-pasted prefill like "  HIGH " in /etc/kai/env round-trips
        to its canonical "high" form on Enter, matching the runtime
        config parser's tolerance for case and whitespace. The displayed
        suffix shows the normalized value, never the raw input.

        Regression guard: a raw `in choices` check would treat
        "  HIGH " as out-of-list and erase the env var during
        reconfiguration. The normalization happens inside the helper
        so the operator's existing setting is preserved.
        """
        prompts = self._record(monkeypatch, [""])
        result = _prompt_optional_choice("Pick", ["low", "medium", "high"], default="  HIGH ")
        assert result == "high"
        assert "[high]" in prompts[0]
        assert "[  HIGH ]" not in prompts[0]

    def test_empty_input_returns_empty_when_default_not_in_choices(self, monkeypatch):
        """
        A prefill that does not match any choice after normalization is
        treated as no prefill: the suffix reverts to the inline empty-
        default hint and Enter returns "". `"max"` is deliberately
        chosen as a value that does not normalize into any allowed
        entry so this test exercises only the out-of-list branch, not
        the case-and-whitespace normalization branch.
        """
        prompts = self._record(monkeypatch, [""])
        result = _prompt_optional_choice(
            "Pick",
            ["low", "medium", "high"],
            default="max",
            empty_hint="empty = pick default",
        )
        assert result == ""
        assert "empty = pick default" in prompts[0]
        assert "[max]" not in prompts[0]

    def test_in_list_input_returned_lowercased(self, monkeypatch):
        """
        Typed input is normalized via `.strip().lower()` before the
        membership check, so the operator can type "HIGH" or " high "
        and the function returns the canonical "high". Mirrors the
        same normalization `_prompt_choice` applies to its input.
        """
        self._record(monkeypatch, ["HIGH"])
        result = _prompt_optional_choice("Pick", ["low", "medium", "high"], default="")
        assert result == "high"

    def test_invalid_input_reprompts_with_choices_and_empty_hint(self, monkeypatch, capsys):
        """
        On an out-of-list typed answer the helper re-prompts and the
        notice on stdout includes both the allowed choices and the
        empty-default hint, so the operator can recover by retyping
        OR by pressing Enter.
        """
        self._record(monkeypatch, ["bogus", "low"])
        result = _prompt_optional_choice(
            "Pick",
            ["low", "medium", "high"],
            default="",
            empty_hint="empty = pick default",
        )
        assert result == "low"
        captured = capsys.readouterr().out
        assert "low/medium/high" in captured
        assert "empty = pick default" in captured

    def test_invalid_input_reprompt_does_not_advertise_empty_default_when_prefill_present(self, monkeypatch, capsys):
        """
        With a usable prefill, Enter returns the prefill, not "". The
        recovery message must match that: advertise that empty keeps
        the prefill, NOT that empty means the downstream default. The
        latter would mislead the operator into thinking they can clear
        the override by pressing Enter at the re-prompt.

        Inputs are `["bogus", ""]`: the first answer triggers the
        recovery message, the second answer (empty) takes the prefill-
        round-trip branch. Asserts the recovery text omits the
        `empty_hint` and the final return is the prefill.
        """
        self._record(monkeypatch, ["bogus", ""])
        result = _prompt_optional_choice(
            "Pick",
            ["low", "medium", "high"],
            default="high",
            empty_hint="empty = pick default",
        )
        assert result == "high"
        captured = capsys.readouterr().out
        assert "low/medium/high" in captured
        assert "high" in captured
        assert "empty = pick default" not in captured

    def test_whitespace_only_input_treated_as_empty(self, monkeypatch):
        """
        `.strip()` on typed input collapses a whitespace-only answer to
        "", which then takes the empty-input branch. With no prefill
        the helper returns "". Pins parity with how the runtime config
        parser treats `CODEX_EFFORT_LEVEL="   "` as effectively unset.
        """
        self._record(monkeypatch, ["   "])
        result = _prompt_optional_choice("Pick", ["a", "b", "c"], default="")
        assert result == ""

    def test_custom_empty_hint_appears_in_prompt(self, monkeypatch):
        """
        The Codex call site passes `empty_hint="empty = codex default"`
        so the operator sees the semantics in their own terms. This
        test pins that the custom hint reaches the prompt verbatim in
        the no-prefill display path.

        Uses CODEX_EFFORT_LEVELS rather than a hand-written copy of the
        tuple so the canonical effort vocabulary stays single-sourced
        in `kai.config`.
        """
        from kai.config import CODEX_EFFORT_LEVELS

        prompts = self._record(monkeypatch, [""])
        _prompt_optional_choice(
            "Codex reasoning effort",
            list(CODEX_EFFORT_LEVELS),
            default="",
            empty_hint="empty = codex default",
        )
        assert "empty = codex default" in prompts[0]


# ── Config subcommand ────────────────────────────────────────────────


class TestCmdConfig:
    @staticmethod
    def _redirect_staging(monkeypatch, tmp_path):
        """Redirect the staging-path helper so the wizard writes under tmp_path.

        Tests that exercise the first-time wizard branch read the
        generated file back from `tmp_path / "users.yaml"`; without this
        redirect, the wizard would write to the operator's actual
        `~/.cache/kai-install/users.yaml` and leak across test runs.
        """
        monkeypatch.setattr(
            "kai.install._install_staging_path",
            lambda filename: tmp_path / filename,
        )
        monkeypatch.setattr(
            "kai.install._backend_choices_for_config",
            lambda service_user: ["claude", "codex", "goose", "opencode"],
        )

    @staticmethod
    def _simulate_existing_etc_users_yaml(monkeypatch, content):
        """Simulate the canonical users.yaml already existing with the given content.

        Used by tests that want the wizard to take the existing-canonical
        path (skip the user-creation prompts, leave the file untouched)
        without touching the real `/etc/kai/`. Patches the path's
        existence check AND the sudo-cat reader the wizard goes through
        (`Path.exists` for the existence gate; `_read_users_yaml_text`
        for the body the wizard's later per-user scans parse, e.g.
        `_users_yaml_goose_providers` / `_users_yaml_agent_backends`).
        The existence patch compares against the live
        `kai.install.USERS_YAML` attribute rather than a hardcoded path,
        so it stacks on top of the autouse `_isolate_users_yaml` redirect
        from conftest.
        """
        _real_exists = Path.exists

        def _exists_with_canonical(self):
            if self == kai.install.USERS_YAML:
                return True
            return _real_exists(self)

        monkeypatch.setattr(Path, "exists", _exists_with_canonical)
        parsed = yaml.safe_load(content)
        if isinstance(parsed, dict) and isinstance(parsed.get("users"), list):
            for index, entry in enumerate(parsed["users"]):
                if isinstance(entry, dict) and not entry.get("os_user"):
                    entry["os_user"] = f"test-os-user-{index}"
            content = yaml.safe_dump(parsed, sort_keys=False)
        monkeypatch.setattr("kai.install._read_users_yaml_text", lambda path: content)

    def test_backend_choices_read_existing_registry(self, tmp_path, monkeypatch):
        registry = tmp_path / "backends.yaml"
        registry.write_text(
            "version: 1\n"
            "backends:\n"
            "  codex:\n"
            "    command: /usr/local/bin/codex\n"
            "  goose:\n"
            "    command: /opt/homebrew/bin/goose\n"
            "  unknown:\n"
            "    command: /tmp/unknown\n"
        )
        monkeypatch.setattr("kai.install.BACKENDS_YAML", registry)

        assert kai.install._backend_choices_from_existing_registry() == ["codex", "goose"]

    def test_backend_choices_fallback_to_global_discovery(self, monkeypatch):
        monkeypatch.setattr("kai.install._backend_choices_from_existing_registry", lambda: [])
        monkeypatch.setattr(
            "kai.install._discover_backend_commands",
            lambda service_user: {"opencode": "/usr/local/bin/opencode", "codex": "/usr/local/bin/codex"},
        )

        assert kai.install._backend_choices_for_config("kai") == ["codex", "opencode"]

    def test_config_fails_when_no_backend_is_installed(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        monkeypatch.setattr("kai.install._install_staging_path", lambda filename: tmp_path / filename)
        monkeypatch.setattr("kai.install._backend_choices_for_config", lambda service_user: [])
        inputs = iter(
            [
                "protected",
                "/opt/kai",
                "/var/lib/kai",
                "kai",
                "darwin",
                "fake-token",
                "12345",
                "admin",
                "testuser",
                "polling",
            ]
        )
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        with pytest.raises(SystemExit, match="No installed Kai backends were found"):
            _cmd_config()

    def test_writes_install_conf(self, tmp_path, monkeypatch):
        """Config subcommand writes valid JSON to install.conf."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._redirect_staging(monkeypatch, tmp_path)

        # Simulate user inputs for each prompt (in order)
        inputs = iter(
            [
                "protected",  # deployment mode
                "/opt/kai",  # install dir
                "/var/lib/kai",  # data dir
                "kai",  # service user
                "darwin",  # platform
                "fake-token",  # bot token
                "12345",  # admin telegram ID
                "admin",  # admin display name
                "testuser",  # required protected os_user
                "polling",  # transport
                "claude",  # agent backend
                "sonnet",  # model
                "false",  # customize per-role models (decline; use registry defaults)
                "120",  # timeout
                "0",  # max session age hours (0 = no limit)
                "1800",  # idle eviction timeout seconds
                "80",  # autocompact pct
                "",  # claude effort level (take default "high")
                "8080",  # port
                "10.0.0.36",  # Workshop LAN address
                "test-secret",  # webhook secret
                "~/Projects",  # workspace base
                "",  # allowed workspaces (empty)
                "false",  # pr review enabled
                "300",  # pr review cooldown (global resource control)
                "900",  # pr review timeout (seconds)
                "false",  # issue triage enabled
                "",  # github notify chat id (empty)
                "false",  # voice
                "false",  # tts
                "",  # claude user (legacy single-user mode)
                "false",  # memory enabled
                "",  # perplexity key (empty)
            ]
        )
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        _cmd_config()

        conf_path = tmp_path / "install.conf"
        assert conf_path.exists()
        conf = json.loads(conf_path.read_text())
        assert conf["version"] == 1
        assert conf["install_dir"] == "/opt/kai"
        assert conf["env"]["TELEGRAM_BOT_TOKEN"] == "fake-token"
        assert conf["env"]["GITHUB_WEBHOOK_SECRET"] == "test-secret"
        assert conf["env"]["GENERIC_WEBHOOK_SECRET"] != "test-secret"
        assert conf["env"]["WORKSHOP_LAN_HOST"] == "10.0.0.36"
        assert "WEBHOOK_SECRET" not in conf["env"]
        # The context window setting was removed; the wizard never
        # emits the retired key.
        assert "CLAUDE_MAX_CONTEXT_WINDOW" not in conf["env"]
        assert conf["env"]["CLAUDE_AUTOCOMPACT_PCT"] == "80"
        # ALLOWED_USER_IDS should not be in the env dict
        assert "ALLOWED_USER_IDS" not in conf["env"]
        # Default backend is emitted explicitly.
        assert conf["env"]["DEFAULT_BACKEND"] == "claude"
        # users.yaml should have been generated
        yaml_path = tmp_path / "users.yaml"
        assert yaml_path.exists()
        data = yaml.safe_load(yaml_path.read_text())
        assert data["users"][0]["telegram_id"] == 12345
        assert data["users"][0]["role"] == "admin"

    def test_config_does_not_emit_backend_binary_paths(self, tmp_path, monkeypatch):
        """Backend command paths are install registry facts, not
        install.conf values collected from the operator."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._redirect_staging(monkeypatch, tmp_path)

        inputs = iter(self._base_inputs(["false"]))
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        _cmd_config()

        conf = json.loads((tmp_path / "install.conf").read_text())
        for key in ("CLAUDE_BIN", "CODEX_BIN", "GOOSE_BIN", "OPENCODE_BIN"):
            assert key not in conf["env"]

    def test_advanced_user_options(self, tmp_path, monkeypatch):
        """
        Advanced path writes os_user and skips CLAUDE_USER.

        Post-#353: the wizard no longer prompts for home_workspace and
        the admin's users.yaml entry does not carry that field - the
        admin lands in DATA_DIR/home/<chat_id>/ like any other user.
        See test_wizard_defaults_do_not_reintroduce_shared_home for the
        regression guard.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._redirect_staging(monkeypatch, tmp_path)

        inputs = iter(
            [
                "protected",  # deployment mode
                "/opt/kai",  # install dir
                "/var/lib/kai",  # data dir
                "kai",  # service user
                "darwin",  # platform
                "fake-token",  # bot token
                "12345",  # admin telegram ID
                "admin",  # admin display name
                "testuser",  # os_user
                # no home_workspace prompt post-#353
                "polling",  # transport
                "claude",  # agent backend
                "sonnet",  # model
                "false",  # customize per-role models (decline; use registry defaults)
                "120",  # timeout
                "0",  # max session age hours (0 = no limit)
                "1800",  # idle eviction timeout seconds
                "80",  # autocompact pct
                "",  # claude effort level (take default "high")
                "8080",  # port
                "",  # Workshop LAN address (disabled)
                "test-secret",  # webhook secret
                "~/Projects",  # workspace base
                "",  # allowed workspaces (empty)
                "false",  # pr review enabled
                "300",  # pr review cooldown (global resource control)
                "900",  # pr review timeout (seconds)
                "false",  # issue triage enabled
                "",  # github notify chat id (empty)
                "false",  # voice
                "false",  # tts
                # no claude user prompt (skipped by advanced mode)
                "false",  # memory enabled
                "",  # perplexity key (empty)
            ]
        )
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        _cmd_config()

        # Verify users.yaml has os_user and (post-#353) NO home_workspace.
        # The admin lands in DATA_DIR/home/<chat_id>/ like any other user;
        # the field is absent from the wizard output by design.
        yaml_path = tmp_path / "users.yaml"
        assert yaml_path.exists()
        data = yaml.safe_load(yaml_path.read_text())
        entry = data["users"][0]
        assert entry["os_user"] == "testuser"
        assert "home_workspace" not in entry, (
            "Wizard regression: admin entry should not carry home_workspace post-#353. See _cmd_config in install.py."
        )

        # CLAUDE_USER should not be in the env (skipped because os_user was set)
        conf = json.loads((tmp_path / "install.conf").read_text())
        assert "CLAUDE_USER" not in conf["env"]

    def test_wizard_defaults_do_not_reintroduce_shared_home(self, tmp_path, monkeypatch):
        """
        Regression guard for spec #353.

        Prior to #353, the wizard defaulted home_workspace to a shared
        PROJECT_ROOT/home directory. That shared default was the source
        of the multi-user privacy hazard the spec exists to fix. This
        test pins the post-#353 behavior across BOTH wizard paths:

        1. The non-advanced path (advanced=false) must not emit a
           home_workspace key into the admin's users.yaml entry.
        2. The advanced path (advanced=true, no explicit home_workspace
           prompt) must also not emit one.
        3. The generated env must not carry a CLAUDE_WORKSPACE override
           (the env var that pre-#353 wired the shared global home).

        If any future change re-introduces a global default home, one
        of these three assertions will fail.
        """
        # ---- Path 1: non-advanced wizard ----
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._redirect_staging(monkeypatch, tmp_path)

        inputs_basic = iter(
            [
                "protected",  # deployment mode
                "/opt/kai",  # install dir
                "/var/lib/kai",  # data dir
                "kai",  # service user
                "darwin",  # platform
                "fake-token",  # bot token
                "12345",  # admin telegram ID
                "admin",  # admin display name
                "testuser",  # required protected os_user
                "polling",  # transport
                "claude",  # agent backend
                "sonnet",  # model
                "false",  # customize per-role models (decline; use registry defaults)
                "120",  # timeout
                "0",  # max session age hours (0 = no limit)
                "1800",  # idle eviction timeout seconds
                "80",  # autocompact pct
                "",  # claude effort level (take default "high")
                "8080",  # port
                "",  # Workshop LAN address (disabled)
                "test-secret",  # webhook secret
                "~/Projects",  # workspace base
                "",  # allowed workspaces
                "false",  # pr review enabled
                "300",  # pr review cooldown (global resource control)
                "900",  # pr review timeout
                "false",  # issue triage enabled
                "",  # github notify chat id
                "false",  # voice
                "false",  # tts
                "",  # claude user (legacy single-user mode)
                "false",  # memory enabled
                "",  # perplexity key (empty)
            ]
        )
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs_basic))
        _cmd_config()

        yaml_path = tmp_path / "users.yaml"
        data = yaml.safe_load(yaml_path.read_text())
        basic_entry = data["users"][0]
        assert "home_workspace" not in basic_entry, (
            "Regression: non-advanced wizard re-introduced a shared home_workspace default. "
            "Spec #353 requires per-user homes under DATA_DIR/home/<chat_id>/."
        )

        conf = json.loads((tmp_path / "install.conf").read_text())
        # CLAUDE_WORKSPACE was the legacy env that wired a global home;
        # pinning it absent ensures the wizard never re-emits the var.
        assert "CLAUDE_WORKSPACE" not in conf["env"], (
            "Regression: wizard wrote CLAUDE_WORKSPACE env. Spec #353 removed the global home field; "
            "pool.py + bot.py now resolve home per chat_id."
        )

        # ---- Path 2: advanced wizard (admin chooses os_user) ----
        # Re-run the wizard from a clean tmp dir to verify the advanced
        # path also stays clean. We use a sibling dir so monkeypatched
        # PROJECT_ROOT and INSTALL_CONF point somewhere fresh.
        adv_root = tmp_path / "adv"
        adv_root.mkdir()
        monkeypatch.chdir(adv_root)
        monkeypatch.setattr("kai.install.INSTALL_CONF", adv_root / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", adv_root)
        # Redirect staging again because the leg above changed PROJECT_ROOT
        # but the staging helper is anchored to its own filename.
        self._redirect_staging(monkeypatch, adv_root)

        inputs_advanced = iter(
            [
                "protected",
                "/opt/kai",
                "/var/lib/kai",
                "kai",
                "darwin",
                "fake-token",
                "12345",
                "admin",
                "testuser",  # os_user (no home_workspace prompt should follow)
                "polling",
                "claude",
                "sonnet",
                "false",  # customize per-role models (decline; use registry defaults)
                "120",
                "0",  # max session age hours (0 = no limit)
                "1800",  # idle eviction timeout seconds
                "10.0",
                "80",
                "",  # claude effort level (take default "high")
                "8080",
                "",  # Workshop LAN address (disabled)
                "test-secret",
                "~/Projects",
                "",
                "false",
                "300",  # pr review cooldown (global resource control)
                "900",
                "1.0",
                "false",
                "",
                "false",
                "false",
                "false",  # memory enabled
                "",  # perplexity key
            ]
        )
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs_advanced))
        _cmd_config()

        adv_data = yaml.safe_load((adv_root / "users.yaml").read_text())
        adv_entry = adv_data["users"][0]
        assert "home_workspace" not in adv_entry, (
            "Regression: advanced wizard re-introduced home_workspace. The wizard removed the prompt "
            "in spec #353; an admin who needs a custom path must hand-edit users.yaml."
        )

    def test_goose_backend_writes_env(self, tmp_path, monkeypatch, capsys):
        """Selecting goose backend writes DEFAULT_BACKEND to env."""
        monkeypatch.chdir(tmp_path)
        conf_path = tmp_path / "install.conf"
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._redirect_staging(monkeypatch, tmp_path)

        # Pre-seed existing config with goose backend so the prompt
        # appears (it is gated behind an existing non-claude value).
        existing = {"version": 1, "env": {"DEFAULT_BACKEND": "goose"}}
        conf_path.write_text(json.dumps(existing))

        monkeypatch.setattr("kai.install._validate_goose_bin", lambda p: bool(p))
        inputs = iter(
            [
                "protected",  # deployment mode
                "/opt/kai",  # install dir
                "/var/lib/kai",  # data dir
                "kai",  # service user
                "darwin",  # platform
                "fake-token",  # bot token
                "12345",  # admin telegram ID
                "admin",  # admin display name
                "testuser",  # required protected os_user
                "polling",  # transport
                "goose",  # agent backend (prompt shown because existing config has goose)
                "anthropic",  # goose provider
                "sk-ant-test-key",  # ANTHROPIC_API_KEY
                "sonnet",  # model
                "false",  # customize per-role models (decline; use registry defaults)
                "120",  # timeout
                "0",  # max session age hours (0 = no limit)
                "1800",  # idle eviction timeout seconds
                "8080",  # port
                "",  # Workshop LAN address (disabled)
                "test-secret",  # webhook secret
                "~/Projects",  # workspace base
                "",  # allowed workspaces (empty)
                "false",  # pr review enabled
                "300",  # pr review cooldown (global resource control)
                "900",  # pr review timeout (seconds)
                "false",  # issue triage enabled
                "",  # github notify chat id (empty)
                "false",  # voice
                "false",  # tts
                "false",  # memory enabled
                "",  # perplexity key (empty)
            ]
        )
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        _cmd_config()

        conf = json.loads((tmp_path / "install.conf").read_text())
        assert conf["env"]["DEFAULT_BACKEND"] == "goose"
        assert conf["env"]["DEFAULT_PROVIDER"] == "anthropic"
        assert conf["env"]["ANTHROPIC_API_KEY"] == "sk-ant-test-key"
        # Memory was declined, so the retrieval-only note must not
        # print; it belongs only to the memory-enabled flow.
        out = capsys.readouterr().out
        assert "retrieval-only" not in out

    def test_goose_memory_enabled_walks_extraction_prompts(self, tmp_path, monkeypatch, capsys):
        """Goose plus memory now walks the extraction prompts: goose
        ships a OneShotReasoner, so the wizard treats it like the
        other reasoner backends and the retrieval-only note does not
        print."""
        monkeypatch.chdir(tmp_path)
        conf_path = tmp_path / "install.conf"
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._redirect_staging(monkeypatch, tmp_path)

        existing = {"version": 1, "env": {"DEFAULT_BACKEND": "goose"}}
        conf_path.write_text(json.dumps(existing))

        memory_block = [
            "true",  # memory enabled
            "true",  # memory extraction enabled (now prompted on goose)
            "60",  # per-extraction timeout (non-default so it persists)
            "8",  # consolidation candidates
            "3",  # episode classifier context turns
            "120",  # per-episode timeout
            "0.9",  # paraphrase-dedup threshold
            "2000",  # memory token budget
            "10",  # memory search limit
        ]
        monkeypatch.setattr("kai.install._validate_goose_bin", lambda p: bool(p))
        inputs = iter(self._base_inputs(memory_block, agent_backend="goose"))
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        _cmd_config()

        out = capsys.readouterr().out
        assert "retrieval-only" not in out
        conf = json.loads((tmp_path / "install.conf").read_text())
        assert conf["env"]["MEMORY_ENABLED"] == "true"
        assert conf["env"]["MEMORY_EXTRACTION_ENABLED"] == "true"
        assert conf["env"]["MEMORY_EXTRACTION_TIMEOUT_S"] == "60"

    def test_non_reasoner_backend_memory_prints_retrieval_only_note(self, tmp_path, monkeypatch, capsys):
        """The retrieval-only note still prints for a backend outside
        ONESHOT_REASONER_BACKENDS. Every real backend is a member
        today, so the else branch is exercised by patching the
        constant down; goose stands in as the excluded backend the
        same way it did before it grew a reasoner."""
        import kai.install as install_module

        monkeypatch.chdir(tmp_path)
        conf_path = tmp_path / "install.conf"
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(install_module, "ONESHOT_REASONER_BACKENDS", frozenset({"claude", "codex"}))
        self._redirect_staging(monkeypatch, tmp_path)

        existing = {"version": 1, "env": {"DEFAULT_BACKEND": "goose"}}
        conf_path.write_text(json.dumps(existing))

        memory_block = [
            "true",
            "2000",
            "10",
        ]  # memory enabled; extraction prompts gated out
        monkeypatch.setattr("kai.install._validate_goose_bin", lambda p: bool(p))
        inputs = iter(self._base_inputs(memory_block, agent_backend="goose"))
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        _cmd_config()

        out = capsys.readouterr().out
        assert "memory extraction is not available on the goose backend" in out
        assert "retrieval-only" in out
        conf = json.loads((tmp_path / "install.conf").read_text())
        assert conf["env"]["MEMORY_ENABLED"] == "true"

    def test_goose_ollama_no_api_key(self, tmp_path, monkeypatch):
        """Selecting ollama provider skips the API key prompt."""
        monkeypatch.chdir(tmp_path)
        conf_path = tmp_path / "install.conf"
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._redirect_staging(monkeypatch, tmp_path)

        existing = {"version": 1, "env": {"DEFAULT_BACKEND": "goose"}}
        conf_path.write_text(json.dumps(existing))

        # No API key input after "ollama" - the prompt is skipped.
        monkeypatch.setattr("kai.install._validate_goose_bin", lambda p: bool(p))
        inputs = iter(
            [
                "protected",  # deployment mode
                "/opt/kai",  # install dir
                "/var/lib/kai",  # data dir
                "kai",  # service user
                "darwin",  # platform
                "fake-token",  # bot token
                "12345",  # admin telegram ID
                "admin",  # admin display name
                "testuser",  # required protected os_user
                "polling",  # transport
                "goose",  # agent backend
                "ollama",  # goose provider (no key needed)
                "sonnet",  # model
                "false",  # customize per-role models (decline; use registry defaults)
                "120",  # timeout
                "0",  # max session age hours (0 = no limit)
                "1800",  # idle eviction timeout seconds
                "8080",  # port
                "",  # Workshop LAN address (disabled)
                "test-secret",  # webhook secret
                "~/Projects",  # workspace base
                "",  # allowed workspaces (empty)
                "false",  # pr review enabled
                "300",  # pr review cooldown (global resource control)
                "900",  # pr review timeout (seconds)
                "false",  # issue triage enabled
                "",  # github notify chat id (empty)
                "false",  # voice
                "false",  # tts
                "false",  # memory enabled
                "",  # perplexity key (empty)
            ]
        )
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        _cmd_config()

        conf = json.loads((tmp_path / "install.conf").read_text())
        assert conf["env"]["DEFAULT_PROVIDER"] == "ollama"
        # Ollama is local inference - no API key should be present
        for key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GOOGLE_API_KEY", "OPENROUTER_API_KEY"):
            assert key not in conf["env"]

    def test_reads_existing_defaults(self, tmp_path, monkeypatch, capsys):
        """Config subcommand uses existing install.conf values as defaults."""
        monkeypatch.chdir(tmp_path)
        conf_path = tmp_path / "install.conf"
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)

        # Write existing config
        existing = {
            "version": 1,
            "install_dir": "/custom/path",
            "data_dir": "/custom/data",
            "service_user": "myuser",
            "platform": "linux",
            "env": {
                "TELEGRAM_BOT_TOKEN": "existing-token",
                "WEBHOOK_SECRET": "existing-secret",
                "DEFAULT_BACKEND": "claude",
            },
        }
        conf_path.write_text(json.dumps(existing))

        # Pretend /etc/kai/users.yaml is already in place; the wizard
        # then skips the user-creation prompts and leaves it untouched.
        existing_users_yaml = "users:\n  - telegram_id: 999\n    name: existing\n    role: admin\n"
        self._simulate_existing_etc_users_yaml(monkeypatch, existing_users_yaml)

        monkeypatch.setattr("builtins.input", lambda prompt: "")

        _cmd_config()

        conf = json.loads(conf_path.read_text())
        # Should preserve existing values when user accepts defaults
        assert conf["install_dir"] == "/custom/path"
        assert conf["env"]["TELEGRAM_BOT_TOKEN"] == "existing-token"
        # Named credentials are generated independently and the unsupported
        # legacy value is not carried forward into the regenerated artifact.
        assert conf["env"]["GITHUB_WEBHOOK_SECRET"] != "existing-secret"
        assert conf["env"]["GENERIC_WEBHOOK_SECRET"] != "existing-secret"
        assert conf["env"]["GITHUB_WEBHOOK_SECRET"] != conf["env"]["GENERIC_WEBHOOK_SECRET"]
        assert "WEBHOOK_SECRET" not in conf["env"]
        # With a canonical users.yaml already present the wizard skips the
        # user-creation prompts entirely and never stages a new file, so
        # no admin prompt is shown and no top-level staging key is written.
        # The user-setup section also stays fully silent: no bare header
        # and no trailing blank separator (both gated on the same
        # "something to show" condition), so it adds nothing between the
        # bot-token and transport prompts.
        output = capsys.readouterr().out
        assert "Admin Telegram ID" not in output
        assert "-- User setup --" not in output
        assert "WEBHOOK_SECRET is no longer supported and will be omitted" in output
        assert "users_yaml_staging_path" not in conf

    def test_regeneration_automatically_removes_legacy_webhook_secret(self, tmp_path, monkeypatch, capsys):
        """Regeneration omits the unsupported key and keeps named secrets."""
        monkeypatch.chdir(tmp_path)
        conf_path = tmp_path / "install.conf"
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._simulate_existing_etc_users_yaml(
            monkeypatch,
            "users:\n  - telegram_id: 999\n    name: existing\n    role: admin\n",
        )
        existing = {
            "version": 1,
            "install_dir": "/opt/kai",
            "data_dir": "/var/lib/kai",
            "service_user": "kai",
            "platform": "darwin",
            "env": {
                "TELEGRAM_BOT_TOKEN": "existing-token",
                "WEBHOOK_SECRET": "legacy-secret",
                "GITHUB_WEBHOOK_SECRET": "github-secret",
                "GENERIC_WEBHOOK_SECRET": "generic-secret",
                "DEFAULT_BACKEND": "claude",
            },
        }
        conf_path.write_text(json.dumps(existing))

        monkeypatch.setattr("builtins.input", lambda prompt: "")

        _cmd_config()

        generated = json.loads(conf_path.read_text())
        assert "WEBHOOK_SECRET" not in generated["env"]
        assert generated["env"]["GITHUB_WEBHOOK_SECRET"] == "github-secret"
        assert generated["env"]["GENERIC_WEBHOOK_SECRET"] == "generic-secret"
        assert generated["env"]["TELEGRAM_BOT_TOKEN"] == "existing-token"
        output = capsys.readouterr().out
        assert "WEBHOOK_SECRET is no longer supported and will be omitted" in output

    def test_validates_required_fields(self):
        """Required-field validation rejects empty input."""
        # _prompt with required=True rejects empty input. We test the
        # underlying validator directly since testing the full interactive
        # flow with required fields is fragile with mocked input.
        assert _validate_user_ids("") is False
        assert _validate_user_ids("abc") is False
        assert _validate_port("0") is False
        assert _validate_port("99999") is False
        # Chat ID accepts any non-zero integer (group IDs are negative)
        assert _validate_chat_id("-1001234567890") is True
        assert _validate_chat_id("12345") is True
        assert _validate_chat_id("0") is False
        assert _validate_chat_id("abc") is False

    # ── Memory env var prompts (#343) ─────────────────────────────────

    @staticmethod
    def _base_inputs(
        memory_block: list[str],
        effort: str = "",
        agent_backend: str = "claude",
        llm_provider: str = "anthropic",
        llm_api_key: str = "sk-ant-test-key",
    ) -> list[str]:
        """Default wizard inputs with swappable memory block, effort, and backend.

        `effort` lets a test exercise a non-default CLAUDE_EFFORT_LEVEL
        without rebuilding the whole input list. Empty string accepts
        the wizard default ("high"); pass any allow-list value (low,
        medium, high, xhigh, max) to drive the non-default emission
        branch in install.py. Only meaningful when agent_backend is
        "claude" - passing effort with another backend raises
        ValueError to prevent silent no-ops.

        `agent_backend` lets a test exercise the goose path. When set
        to anything other than "claude", the helper:
          - Inserts an llm_provider prompt answer (and the API key for
            non-ollama providers) immediately after the agent_backend
            slot, matching the wizard's flow for non-claude backends.
          - Omits the autocompact and effort entries that the wizard
            now skips for non-claude backends per issue #380.
        Defaults (anthropic + sk-ant-test-key) are sufficient for the
        gating tests. For ollama, the API key prompt is skipped by the
        wizard, so the llm_api_key default is harmless and unused (no
        need to override it).
        """
        # Guard against silent no-ops: a caller passing effort="xhigh"
        # with a non-claude backend would expect the value to land in
        # the wizard's effort prompt, but that prompt does not fire for
        # non-claude backends (gated by issue #380). Raise loudly here
        # rather than silently dropping the value, which would be a
        # confusing failure mode for a future test author.
        if agent_backend != "claude" and effort:
            raise ValueError(
                f"_base_inputs: effort={effort!r} is ignored for non-claude "
                f"backend ({agent_backend!r}). The wizard does not prompt for "
                f"effort under non-claude backends. Pass effort only with "
                f"agent_backend='claude'."
            )
        # Wizard prompts for provider + API key only for non-claude
        # backends. The API key prompt itself is skipped when provider
        # is "ollama" (local model, no auth). Backend command paths
        # now come from the installed backend registry/discovery, not
        # from config wizard answers.
        backend_block: list[str] = []
        if agent_backend != "claude":
            backend_block.append(llm_provider)
            if llm_provider != "ollama":
                backend_block.append(llm_api_key)

        # Autocompact + effort prompts are gated on agent_backend ==
        # "claude" by issue #380. Their fixture entries are conditional
        # to match the wizard's runtime conditional.
        claude_only_pre_webhook: list[str] = []
        if agent_backend == "claude":
            claude_only_pre_webhook = [
                "80",  # autocompact pct
                effort,  # claude effort level ("" = default "high")
            ]

        return [
            "protected",  # deployment mode
            "/opt/kai",  # install dir
            "/var/lib/kai",  # data dir
            "kai",  # service user
            "darwin",  # platform
            "fake-token",  # bot token
            "12345",  # admin telegram ID
            "admin",  # admin display name
            "testuser",  # required protected os_user
            "polling",  # transport
            agent_backend,  # agent backend
            *backend_block,  # llm_provider + api_key (non-claude only)
            "sonnet",  # model
            "false",  # customize per-role models (decline; use registry defaults)
            "120",  # timeout
            "0",  # max session age hours (0 = no limit)
            "1800",  # idle eviction timeout seconds
            *claude_only_pre_webhook,  # autocompact + effort (claude only)
            "8080",  # port
            "",  # Workshop LAN address (disabled)
            "test-secret",  # webhook secret
            "~/Projects",  # workspace base
            "",  # allowed workspaces
            "300",  # pr review cooldown (global resource control)
            "900",  # pr review timeout
            "false",  # voice
            "false",  # tts
            *memory_block,
            "",  # perplexity key
        ]

    def test_memory_disabled_omits_env_keys(self, tmp_path, monkeypatch):
        """MEMORY_ENABLED=false produces no MEMORY_* env entries."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._redirect_staging(monkeypatch, tmp_path)

        inputs = iter(self._base_inputs(["false"]))  # memory disabled
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        _cmd_config()

        conf = json.loads((tmp_path / "install.conf").read_text())
        for key in conf["env"]:
            assert not key.startswith("MEMORY_"), f"unexpected memory key: {key}"

    def test_effort_level_default_omits_env_key(self, tmp_path, monkeypatch):
        """The CLAUDE_EFFORT_LEVEL env key is omitted when the operator
        accepts the wizard default. install.conf is meant to be a delta
        from defaults, so the absence of the key is the positive signal
        that nothing was overridden. Pairs with the non-default test
        below to pin both sides of the emission branch in install.py."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._redirect_staging(monkeypatch, tmp_path)

        inputs = iter(self._base_inputs(["false"], effort=""))
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        _cmd_config()

        conf = json.loads((tmp_path / "install.conf").read_text())
        assert "CLAUDE_EFFORT_LEVEL" not in conf["env"]

    def test_effort_level_non_default_writes_env_key(self, tmp_path, monkeypatch):
        """A non-default CLAUDE_EFFORT_LEVEL must round-trip from the
        wizard prompt into install.conf. Closes the gap noted in PR
        review: previously every test took the default, leaving the
        emission branch (`if claude_effort_level != "high": env[...] =`)
        untested as a positive case. xhigh is chosen because it is
        unambiguously non-default and is also the value used by the
        unit test in tests/test_claude.py for the matching subprocess
        side, keeping a single non-default value across both layers."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._redirect_staging(monkeypatch, tmp_path)

        inputs = iter(self._base_inputs(["false"], effort="xhigh"))
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        _cmd_config()

        conf = json.loads((tmp_path / "install.conf").read_text())
        assert conf["env"].get("CLAUDE_EFFORT_LEVEL") == "xhigh"

    def test_goose_backend_skips_claude_only_prompts(self, tmp_path, monkeypatch):
        """When agent_backend is goose, the Claude-only wizard prompts
        (autocompact, effort, legacy claude_user) must not fire and the
        corresponding env keys must not appear in install.conf.

        Pins the gating from issue #380. Without this test, a future
        refactor that re-introduces an unconditional prompt would slip
        through silently because every other fixture passes 'claude'.

        Note: the admin_os_user prompt is NOT gated in this PR. It
        lives inside the `if advanced:` block in install.py, and
        agent_backend is not yet defined at that point in the wizard;
        honoring the gate would require structural reordering. Deferred
        to a separate cleanup once the broader multi-backend rework
        revisits the wizard end-to-end."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._redirect_staging(monkeypatch, tmp_path)

        monkeypatch.setattr("kai.install._validate_goose_bin", lambda p: bool(p))
        inputs = iter(self._base_inputs(["false"], agent_backend="goose"))
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        _cmd_config()

        conf = json.loads((tmp_path / "install.conf").read_text())
        env = conf["env"]
        # The three Claude-only env keys gated by issue #380. Their
        # absence is the positive signal that the prompts were skipped:
        # the wizard would have written them on a non-default value,
        # and the value cannot be set to non-default without the
        # prompt firing.
        assert "CLAUDE_AUTOCOMPACT_PCT" not in env
        assert "CLAUDE_EFFORT_LEVEL" not in env
        assert "CLAUDE_USER" not in env

        # Sanity: DEFAULT_BACKEND is emitted explicitly.
        assert env.get("DEFAULT_BACKEND") == "goose"

    def test_goose_backend_prunes_existing_claude_only_keys(self, tmp_path, monkeypatch):
        """When the operator switches an existing claude install to
        goose and re-runs the wizard, the previously-set Claude-only
        env keys must disappear from install.conf.

        This is an implicit side-effect of the gating in issue #380:
        the prompt is skipped, the variable stays at its dataclass
        default, and the existing `if value != default` emission check
        correctly drops the write. Pinning the auto-prune so a future
        refactor that adds an `else: keep existing value` fallback to
        the gate cannot silently regress it."""
        monkeypatch.chdir(tmp_path)
        conf_path = tmp_path / "install.conf"
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._redirect_staging(monkeypatch, tmp_path)

        # Pre-seed: an install.conf as if the operator had previously
        # configured the wizard under claude with non-default values
        # for all three of the env keys gated by this PR.
        pre_existing = {
            "version": 1,
            "env": {
                "DEFAULT_BACKEND": "claude",
                "CLAUDE_AUTOCOMPACT_PCT": "50",
                "CLAUDE_EFFORT_LEVEL": "xhigh",
                "CLAUDE_USER": "kai",
            },
        }
        conf_path.write_text(json.dumps(pre_existing))

        # Re-run wizard, switch to goose this time.
        monkeypatch.setattr("kai.install._validate_goose_bin", lambda p: bool(p))
        inputs = iter(self._base_inputs(["false"], agent_backend="goose"))
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        _cmd_config()

        conf = json.loads(conf_path.read_text())
        env = conf["env"]
        # All three previously-set Claude-only keys must be absent.
        # DEFAULT_BACKEND must reflect the new selection.
        assert "CLAUDE_AUTOCOMPACT_PCT" not in env
        assert "CLAUDE_EFFORT_LEVEL" not in env
        assert "CLAUDE_USER" not in env
        assert env.get("DEFAULT_BACKEND") == "goose"

    def test_memory_enabled_writes_tunables(self, tmp_path, monkeypatch):
        """MEMORY_ENABLED=true with extraction writes the chosen tunables."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._redirect_staging(monkeypatch, tmp_path)

        # Memory on, extraction on, custom timeout + consolidation candidates + episode tunables + token budget + search limit.
        # The memory reasoner backend and episode model prompts were
        # retired in issue #515 (per-user dispatch via agent_backend).
        memory_block = [
            "true",  # memory enabled
            "true",  # extraction enabled (claude backend)
            "60",  # extraction timeout seconds (#345)
            "5",  # consolidation candidates (non-default, exercises emission branch)
            "5",  # episode classifier context turns (#392, non-default exercises emission)
            "60",  # episode timeout seconds (non-default)
            "0.85",  # paraphrase-dedup threshold (non-default, exercises emission)
            "3000",  # token budget
            "20",  # search limit (#345)
        ]
        inputs = iter(self._base_inputs(memory_block))
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        _cmd_config()

        conf = json.loads((tmp_path / "install.conf").read_text())
        env = conf["env"]
        assert env["MEMORY_ENABLED"] == "true"
        assert env["MEMORY_EXTRACTION_ENABLED"] == "true"
        # The retired memory-reasoner + model env vars are no longer
        # collected by the wizard; the wizard never writes them.
        assert "MEMORY_REASONER_BACKEND" not in env
        assert "MEMORY_EXTRACTION_MODEL" not in env
        assert "MEMORY_EPISODE_MODEL" not in env
        # Retired budget keys are never written.
        assert "MEMORY_EXTRACTION_BUDGET_USD" not in env
        assert env["MEMORY_EXTRACTION_TIMEOUT_S"] == "60"
        assert env["MEMORY_CONSOLIDATION_CANDIDATES_N"] == "5"
        # Episode-classifier context window (#392): operator picked
        # non-default 5, so the emission gate fires and the env entry
        # is written.
        assert env["EPISODE_CLASSIFIER_CONTEXT_TURNS"] == "5"
        assert "MEMORY_EPISODE_BUDGET_USD" not in env
        assert env["MEMORY_EPISODE_TIMEOUT_S"] == "60"
        # Paraphrase-dedup threshold: operator picked 0.85 (non-default),
        # so the emission gate fires and the env entry is written.
        assert env["MEMORY_DUPLICATE_THRESHOLD"] == "0.85"
        assert env["MEMORY_TOKEN_BUDGET"] == "3000"
        assert env["MEMORY_SEARCH_LIMIT"] == "20"

    def test_memory_episode_defaults_suppress_emission(self, tmp_path, monkeypatch):
        """Operator who accepts every wizard default in the episode
        block produces no episode-specific env entries (the emission
        gates suppress equal-to-dataclass-default values). The memory
        reasoner backend and episode model prompts were retired in
        issue #515; per-user dispatch resolves both at runtime."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._redirect_staging(monkeypatch, tmp_path)

        # Every input matches the corresponding dataclass default so
        # the emission gates suppress every key in the memory block.
        memory_block = [
            "true",  # memory enabled
            "true",  # extraction enabled
            "10",  # extraction timeout (dataclass default; suppressed)
            "8",  # consolidation candidates (dataclass default; suppressed)
            "3",  # episode classifier context turns (#392, dataclass default; suppressed)
            "120",  # episode timeout (dataclass default; suppressed)
            "0.9",  # paraphrase-dedup threshold (dataclass default; suppressed)
            "2000",  # token budget (dataclass default; suppressed)
            "10",  # search limit (dataclass default; suppressed)
        ]
        inputs = iter(self._base_inputs(memory_block))
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        _cmd_config()

        env = json.loads((tmp_path / "install.conf").read_text())["env"]
        # Retired vars never appear, regardless of input.
        assert "MEMORY_REASONER_BACKEND" not in env
        assert "MEMORY_EXTRACTION_MODEL" not in env
        assert "MEMORY_EPISODE_MODEL" not in env
        # Retired budget keys are never written; timeout matches the
        # dataclass default → no emission.
        assert "MEMORY_EPISODE_BUDGET_USD" not in env
        assert "MEMORY_EPISODE_TIMEOUT_S" not in env
        # Stage-1 keys also suppressed because their inputs match
        # the dataclass defaults too.
        assert "MEMORY_EXTRACTION_BUDGET_USD" not in env
        assert "MEMORY_EXTRACTION_TIMEOUT_S" not in env
        assert "MEMORY_CONSOLIDATION_CANDIDATES_N" not in env
        assert "EPISODE_CLASSIFIER_CONTEXT_TURNS" not in env
        assert "MEMORY_DUPLICATE_THRESHOLD" not in env

    def test_memory_round_trip_through_env_file(self, tmp_path, monkeypatch):
        """Wizard-captured memory vars survive _generate_env_file()."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._redirect_staging(monkeypatch, tmp_path)

        # Inputs in wizard order: enabled, ext enabled, timeout, consolidation, classifier-window, episode timeout, dedup threshold, token budget, search limit.
        # The memory reasoner backend and episode model prompts were
        # retired in issue #515.
        memory_block = [
            "true",
            "true",
            "45",
            "4",
            "7",  # episode classifier context turns (#392, non-default)
            "90",
            "0.8",  # paraphrase-dedup threshold (non-default)
            "2500",
            "15",
        ]
        inputs = iter(self._base_inputs(memory_block))
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        _cmd_config()

        conf = json.loads((tmp_path / "install.conf").read_text())
        rendered = _generate_env_file(conf["env"])
        assert 'MEMORY_ENABLED="true"' in rendered
        assert 'MEMORY_EXTRACTION_ENABLED="true"' in rendered
        # Retired budget keys never reach the env file - asserted
        # absent here as the round-trip equivalent of
        # test_memory_enabled_writes_tunables.
        assert "MEMORY_EXTRACTION_BUDGET_USD" not in rendered
        assert 'MEMORY_EXTRACTION_TIMEOUT_S="45"' in rendered
        assert 'MEMORY_CONSOLIDATION_CANDIDATES_N="4"' in rendered
        # Episode-classifier context window (#392) round-trip parity.
        assert 'EPISODE_CLASSIFIER_CONTEXT_TURNS="7"' in rendered
        # Retired keys (per issue #515 per-user dispatch) must NOT
        # round-trip; the wizard no longer collects them, so they
        # cannot appear in the env file.
        assert "MEMORY_REASONER_BACKEND" not in rendered
        assert "MEMORY_EXTRACTION_MODEL" not in rendered
        assert "MEMORY_EPISODE_MODEL" not in rendered
        # Episode tunables: non-default timeout survives. Round-trip
        # parity with the extraction tunables above; the retired
        # budget key stays absent.
        assert "MEMORY_EPISODE_BUDGET_USD" not in rendered
        assert 'MEMORY_EPISODE_TIMEOUT_S="90"' in rendered
        assert 'MEMORY_TOKEN_BUDGET="2500"' in rendered
        assert 'MEMORY_SEARCH_LIMIT="15"' in rendered

    def test_memory_defaults_from_existing_install_conf(self, tmp_path, monkeypatch):
        """Re-running the wizard offers prior memory choices as defaults."""
        monkeypatch.chdir(tmp_path)
        conf_path = tmp_path / "install.conf"
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._simulate_existing_etc_users_yaml(
            monkeypatch,
            "users:\n  - telegram_id: 999\n    name: existing\n    role: admin\n",
        )

        # DEFAULT_BACKEND seeded explicitly so the extraction-keys cleanup
        # has a concrete backend selection.
        existing = {
            "version": 1,
            "install_dir": "/opt/kai",
            "data_dir": "/var/lib/kai",
            "service_user": "kai",
            "platform": "darwin",
            "env": {
                "TELEGRAM_BOT_TOKEN": "existing-token",
                "WEBHOOK_SECRET": "existing-secret",
                "DEFAULT_BACKEND": "claude",
                "MEMORY_ENABLED": "true",
                "MEMORY_EXTRACTION_ENABLED": "true",
                "MEMORY_EXTRACTION_BUDGET_USD": "0.08",
                "MEMORY_EXTRACTION_TIMEOUT_S": "75",
                "MEMORY_TOKEN_BUDGET": "4000",
                "MEMORY_SEARCH_LIMIT": "25",
            },
        }
        conf_path.write_text(json.dumps(existing))
        # users.yaml is simulated above via _simulate_existing_etc_users_yaml,
        # which keeps the wizard on the existing-config branch and skips
        # the per-user prompts that lack defaults.
        monkeypatch.setattr("builtins.input", lambda prompt: "")

        _cmd_config()

        conf = json.loads(conf_path.read_text())
        env = conf["env"]
        assert env["MEMORY_ENABLED"] == "true"
        assert env["MEMORY_EXTRACTION_ENABLED"] == "true"
        # The retired budget key is dropped on regenerate: a
        # previously-set value on an upgrade path is intentionally
        # cleared rather than round-tripped.
        assert "MEMORY_EXTRACTION_BUDGET_USD" not in env
        assert env["MEMORY_EXTRACTION_TIMEOUT_S"] == "75"
        assert env["MEMORY_TOKEN_BUDGET"] == "4000"
        assert env["MEMORY_SEARCH_LIMIT"] == "25"

    def test_memory_toggle_off_drops_existing_keys(self, tmp_path, monkeypatch):
        """Switching MEMORY_ENABLED true -> false strips stale MEMORY_* keys."""
        monkeypatch.chdir(tmp_path)
        conf_path = tmp_path / "install.conf"
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._redirect_staging(monkeypatch, tmp_path)

        # Disabling memory must strip stale keys; otherwise the daemon keeps memory live after the operator opts out.
        existing = {
            "version": 1,
            "env": {
                "MEMORY_ENABLED": "true",
                "MEMORY_EXTRACTION_ENABLED": "true",
                "MEMORY_EXTRACTION_BUDGET_USD": "0.05",
                "MEMORY_EXTRACTION_TIMEOUT_S": "60",
                "MEMORY_TOKEN_BUDGET": "3000",
                "MEMORY_SEARCH_LIMIT": "20",
            },
        }
        conf_path.write_text(json.dumps(existing))

        inputs = iter(self._base_inputs(["false"]))  # memory toggled off
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        _cmd_config()

        env = json.loads(conf_path.read_text())["env"]
        for key in env:
            assert not key.startswith("MEMORY_"), f"stale memory key: {key}"

    def test_non_reasoner_backend_drops_stale_extraction_keys(self, tmp_path, monkeypatch):
        """Switching to a backend outside ONESHOT_REASONER_BACKENDS
        strips MEMORY_EXTRACTION_* keys. Every real backend is a
        member today, so the cleanup branch is exercised by patching
        the constant down with goose standing in as the excluded
        backend."""
        import kai.install as install_module

        monkeypatch.chdir(tmp_path)
        conf_path = tmp_path / "install.conf"
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(install_module, "ONESHOT_REASONER_BACKENDS", frozenset({"claude", "codex"}))
        self._redirect_staging(monkeypatch, tmp_path)

        # Switching claude -> goose must drop extraction keys: bot.py:3609 silently ignores them on non-claude.
        # MEMORY_EXTRACTION_TIMEOUT_S seeded so the cleanup pop is exercised, not just defaulted away.
        existing = {
            "version": 1,
            "env": {
                "DEFAULT_BACKEND": "goose",
                "MEMORY_ENABLED": "true",
                "MEMORY_EXTRACTION_ENABLED": "true",
                "MEMORY_EXTRACTION_BUDGET_USD": "0.05",
                "MEMORY_EXTRACTION_TIMEOUT_S": "60",
            },
        }
        conf_path.write_text(json.dumps(existing))

        monkeypatch.setattr("kai.install._validate_goose_bin", lambda p: bool(p))
        inputs = iter(
            [
                "protected",  # deployment mode
                "/opt/kai",  # install dir
                "/var/lib/kai",  # data dir
                "kai",  # service user
                "darwin",  # platform
                "fake-token",  # bot token
                "12345",  # admin telegram ID
                "admin",  # admin display name
                "testuser",  # required protected os_user
                "polling",  # transport
                "goose",  # agent backend (was claude)
                "anthropic",  # goose provider
                "sk-ant-test-key",  # API key
                "sonnet",  # model
                "false",  # customize per-role models (decline; use registry defaults)
                "120",  # timeout
                "0",  # max session age hours (0 = no limit)
                "1800",  # idle eviction timeout seconds
                "8080",  # port
                "",  # Workshop LAN address (disabled)
                "test-secret",  # webhook secret
                "~/Projects",  # workspace base
                "",  # allowed workspaces
                "300",  # pr review cooldown (global resource control)
                "900",  # pr review timeout
                "false",  # voice
                "false",  # tts
                "true",  # memory enabled (extraction + timeout prompts skipped: non-claude)
                "2000",  # token budget
                "10",  # search limit
                "",  # perplexity key
            ]
        )
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        _cmd_config()

        env = json.loads(conf_path.read_text())["env"]
        assert env["MEMORY_ENABLED"] == "true"
        assert "MEMORY_EXTRACTION_ENABLED" not in env
        assert "MEMORY_EXTRACTION_BUDGET_USD" not in env
        assert "MEMORY_EXTRACTION_TIMEOUT_S" not in env

    def test_extraction_disabled_skips_timeout_prompt(self, tmp_path, monkeypatch):
        """Extraction off must skip the timeout, consolidation, classifier-window, AND episode prompts; search limit still asked.

        Regression guard for the off-by-one trap: timeout, consolidation,
        the episode-classifier context window (#392), and the entire
        episode block sit inside the extraction-enabled branch, so
        disabling extraction must consume strictly fewer prompts than
        enabling it. (The extraction-enabled branch drives only
        timeout, consolidation, classifier-window, and episode
        timeout.) If the gating drifts on any of these prompts - notably
        if a future edit accidentally hoists the classifier-window
        prompt out of the extraction-enabled branch - the input iterator
        desynchronises and the wizard reads search limit as if it were
        one of the extraction-only prompts. The assertion below catches
        that.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._redirect_staging(monkeypatch, tmp_path)

        memory_block = [
            "true",  # memory enabled
            "false",  # extraction disabled (skips timeout + consolidation + episode block)
            "3000",  # token budget
            "20",  # search limit
        ]
        inputs = iter(self._base_inputs(memory_block))
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        _cmd_config()

        env = json.loads((tmp_path / "install.conf").read_text())["env"]
        assert env["MEMORY_ENABLED"] == "true"
        assert "MEMORY_EXTRACTION_ENABLED" not in env
        assert "MEMORY_EXTRACTION_BUDGET_USD" not in env
        assert "MEMORY_EXTRACTION_TIMEOUT_S" not in env
        assert env["MEMORY_TOKEN_BUDGET"] == "3000"
        assert env["MEMORY_SEARCH_LIMIT"] == "20"
        # Classifier-window key (#392) must also be absent: with
        # extraction off, the prompt is gated out and the dataclass
        # default never gets emitted. Pin this so a future edit that
        # hoists the prompt out of the extraction-enabled branch
        # surfaces here.
        assert "EPISODE_CLASSIFIER_CONTEXT_TURNS" not in env

    # Spec 392: episode-classifier context window. The new wizard
    # prompt fires alongside MEMORY_CONSOLIDATION_CANDIDATES_N inside
    # the extraction-enabled branch on the claude backend. These three
    # tests pin the emission contract - non-default writes the env
    # entry, default suppresses it, non-claude skips the prompt
    # entirely (the dataclass default applies at startup).

    def test_episode_classifier_context_turns_writes_env_when_non_default(self, tmp_path, monkeypatch):
        """Operator picks 5 (non-default). The emission gate fires
        and EPISODE_CLASSIFIER_CONTEXT_TURNS=5 lands in the env file."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._redirect_staging(monkeypatch, tmp_path)

        # All other fields at default; only the classifier window is
        # non-default so the test isolates the new emission path from
        # surrounding noise.
        memory_block = [
            "true",  # memory enabled
            "true",  # extraction enabled
            "10",  # extraction timeout (default; suppressed)
            "8",  # consolidation candidates (default; suppressed)
            "5",  # episode classifier context turns (#392, non-default)
            "120",  # episode timeout (default; suppressed)
            "0.9",  # paraphrase-dedup threshold (default; suppressed)
            "2000",  # token budget (default; suppressed)
            "10",  # search limit (default; suppressed)
        ]
        inputs = iter(self._base_inputs(memory_block))
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        _cmd_config()

        env = json.loads((tmp_path / "install.conf").read_text())["env"]
        assert env["EPISODE_CLASSIFIER_CONTEXT_TURNS"] == "5"

    def test_episode_classifier_context_turns_dataclass_default_suppressed(self, tmp_path, monkeypatch):
        """Operator picks 3 (the dataclass default). The
        delta-from-default emission gate suppresses the env entry so
        install.conf stays a delta from defaults rather than a
        snapshot of every available knob."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._redirect_staging(monkeypatch, tmp_path)

        memory_block = [
            "true",  # memory enabled
            "true",  # extraction enabled
            "10",  # extraction timeout (default)
            "8",  # consolidation candidates (default)
            "3",  # episode classifier context turns (#392, dataclass default)
            "120",  # episode timeout (default)
            "0.9",  # paraphrase-dedup threshold (default)
            "2000",  # token budget (default)
            "10",  # search limit (default)
        ]
        inputs = iter(self._base_inputs(memory_block))
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        _cmd_config()

        env = json.loads((tmp_path / "install.conf").read_text())["env"]
        # All defaults → key is absent.
        assert "EPISODE_CLASSIFIER_CONTEXT_TURNS" not in env

    def test_episode_classifier_context_turns_skipped_on_goose_extraction_declined(self, tmp_path, monkeypatch):
        """On a goose install that declines extraction, the
        extraction-enabled branch (including the classifier-window
        prompt) does not fire: the dataclass default applies at
        startup and the env file does not carry the key."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._redirect_staging(monkeypatch, tmp_path)

        # Goose now reaches the extraction-enabled prompt; declining it
        # keeps the rest of the extraction branch (including the new
        # classifier-window prompt) gated out.
        memory_block = [
            "true",
            "false",
            "2000",
            "10",
        ]  # memory on, extraction declined, budget, limit
        monkeypatch.setattr("kai.install._validate_goose_bin", lambda p: bool(p))
        inputs = iter(self._base_inputs(memory_block, agent_backend="goose"))
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        _cmd_config()

        env = json.loads((tmp_path / "install.conf").read_text())["env"]
        assert "EPISODE_CLASSIFIER_CONTEXT_TURNS" not in env

    def test_goose_retrieval_only_does_not_persist_invalid_reasoner_backend(self, tmp_path, monkeypatch):
        """A goose install accepting memory_enabled=true and declining
        extraction must NOT write MEMORY_REASONER_BACKEND or any
        extraction config to the generated env: the retired reasoner
        prompt never fires and extraction-off is the dataclass
        default, so the delta-from-default emission suppresses both."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._redirect_staging(monkeypatch, tmp_path)

        # Goose now reaches the extraction prompt; declining keeps the
        # run retrieval-only.
        memory_block = ["true", "false", "2000", "10", "false"]
        monkeypatch.setattr("kai.install._validate_goose_bin", lambda p: bool(p))
        inputs = iter(self._base_inputs(memory_block, agent_backend="goose"))
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        _cmd_config()

        env = json.loads((tmp_path / "install.conf").read_text())["env"]
        # The critical assertion: no invalid reasoner-backend value
        # in the env.
        assert "MEMORY_REASONER_BACKEND" not in env
        # Extraction config is also absent on goose retrieval-only.
        assert "MEMORY_EXTRACTION_ENABLED" not in env

    def test_codex_install_accept_all_defaults_yields_load_config_compatible_env(self, tmp_path, monkeypatch):
        """v6 catch-all regression: a codex install that accepts every
        wizard default for memory must produce an env block that
        load_config accepts without raising. Pins the entire 'wizard
        default invalid under selected reasoner' bug class so a
        future default flip lands as a single-test failure rather
        than a runtime config-load SystemExit.

        Test shape: rather than re-driving the full wizard (covered
        by surrounding tests), this test pins the integration
        contract directly: assemble the env block the wizard emits
        on codex accept-all-defaults and feed it into load_config.
        The wizard's emission rules are simple enough that the env
        contents are mechanical: MEMORY_ENABLED=true,
        MEMORY_EXTRACTION_ENABLED=true, MEMORY_REASONER_BACKEND=codex,
        and no other memory keys (every other knob accepted the
        dataclass default and the wizard's delta-from-default
        emission suppresses them)."""
        from kai.config import load_config

        # Pin the resolver so this test does not depend on having
        # codex actually installed on the host.
        monkeypatch.setattr(
            "kai.oneshot_binary.resolve_oneshot_binary",
            lambda backend: f"/fake/{backend}",
        )
        # users.yaml is required for codex memory (config-load
        # validates the per-user os_user precondition). The wizard
        # builds this alongside the env block; here we set up a
        # minimal users.yaml inline so the test focuses on the
        # memory env contract.
        users_yaml_text = "users:\n  - telegram_id: 1\n    name: tester\n    role: admin\n    os_user: tester_os\n"
        monkeypatch.setattr("kai.config.PROJECT_ROOT", tmp_path)
        # Pin protected-env / dotenv to be inert. Without this,
        # load_config reads /etc/kai/env (when present on the dev
        # box) and `os.environ.setdefault` populates the process env
        # with values that leak into subsequent tests. The /etc/kai/env
        # on a dev install typically carries `CODEX_BIN=/Users/...`,
        # which would make later test_review codex tests assert the
        # absolute path instead of the literal "codex". The autouse
        # fixture in test_config.py applies the same neutering for
        # its scope; this test is in test_install.py and has to do
        # it explicitly.
        monkeypatch.setattr("kai.config.load_dotenv", lambda *a, **kw: None)

        def _fake_read(path):
            # users.yaml is mandatory post-#565 tranche A; serve the
            # inline content so the loader's auth contract is met.
            if path == "/etc/kai/users.yaml":
                return users_yaml_text
            if path == "/etc/kai/env":
                # Non-empty content satisfies the protected-mode
                # predicate so the resolver picks /etc/kai/users.yaml
                # instead of XDG. The comment line is skipped by
                # the parser.
                return "# test sentinel: protected install\n"
            return None

        monkeypatch.setattr("kai.config._read_protected_file", _fake_read)
        monkeypatch.setattr(
            "kai.config.validate_protected_file_metadata",
            lambda path, **kw: path == "/etc/kai/users.yaml",
        )
        monkeypatch.setattr(
            "kai.config.pwd.getpwnam",
            lambda name: MagicMock(pw_uid=os.geteuid() + 100_000),
        )
        # Seed only the env keys the wizard would emit. Everything
        # else falls back to dataclass defaults via load_config.
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
        monkeypatch.setenv("ALLOWED_USER_IDS", "1")
        monkeypatch.setenv("DEFAULT_BACKEND", "codex")
        # DEFAULT_MODEL validates against the agent backend's model
        # set; codex requires a CODEX_MODELS entry. The wizard would
        # have prompted for this; here we set it explicitly to focus
        # the test on the memory wiring contract.
        monkeypatch.setenv("DEFAULT_MODEL", "gpt-5.4")
        monkeypatch.setenv("MEMORY_ENABLED", "true")
        monkeypatch.setenv("MEMORY_EXTRACTION_ENABLED", "true")
        # MEMORY_REASONER_BACKEND retired (issue #515); memory reasoner
        # derives from DEFAULT_BACKEND per-user. The wizard no longer
        # emits the key.
        # MEMORY_EPISODE_MODEL is intentionally NOT set; the codex
        # branch defaults the wizard prompt to blank, which means
        # load_config inherits from MEMORY_EXTRACTION_MODEL, which
        # itself defaults to the codex registry SKU. The contract
        # under test is "blank episode_model under codex resolves to
        # a CODEX_MODELS-valid SKU."
        config = load_config()
        assert config.memory_enabled is True
        assert config.memory_extraction_enabled is True
        assert config.default_backend == "codex"
        # Per-user dispatch (issue #515) resolves the memory model
        # from the registry at extraction time; verify both stage roles
        # have a codex-valid row.
        from kai.config import CODEX_MODELS, ModelRole, get_model_for

        assert get_model_for(ModelRole.MEMORY_EXTRACTION, "codex", "openai") in CODEX_MODELS
        assert get_model_for(ModelRole.MEMORY_EPISODE, "codex", "openai") in CODEX_MODELS

    def test_codex_fresh_install_with_memory_extraction_yields_valid_users_yaml(self, tmp_path, monkeypatch):
        """Fresh protected Codex+memory config carries OS isolation.

        Issue #522's same-user Codex symmetry remains supported in
        single-user mode. Protected mode now requires an explicit target
        account, and the generated env + users.yaml must still pass the
        complete load_config integration path.
        """
        from unittest.mock import MagicMock

        from kai.config import load_config

        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._redirect_staging(monkeypatch, tmp_path)
        # Pin the model + codex-binary helpers so the test does not
        # depend on the host having codex installed or a curated
        # model registry on disk. The mock value below is a real
        # CODEX_MODELS entry so load_config's model-vs-backend
        # validation passes.
        monkeypatch.setattr(
            "kai.install._prompt_default_model",
            MagicMock(return_value="gpt-5.4"),
        )
        monkeypatch.setattr("kai.install._validate_codex_bin", lambda p: bool(p))

        inputs = iter(
            [
                "protected",  # deployment mode
                "/opt/kai",  # install dir
                "/var/lib/kai",  # data dir
                "kai",  # service user
                "darwin",  # platform
                "fake-token",  # bot token
                "12345",  # admin telegram ID
                "admin",  # admin display name
                "testuser",  # required protected os_user
                "polling",  # transport
                "codex",  # agent backend
                "subscription",  # codex auth mode
                # model: handled by _prompt_default_model mock
                "false",  # customize per-role models (decline; use registry defaults)
                "120",  # agent timeout
                "0",  # max session age hours (0 = no limit)
                "1800",  # idle eviction timeout seconds
                # autocompact + claude effort skipped on non-claude backend
                "",  # codex reasoning effort (empty = codex default)
                "8080",  # webhook port
                "",  # Workshop LAN address (disabled)
                "test-secret",  # webhook secret
                "",  # workspace base
                "",  # allowed workspaces
                "300",  # pr review cooldown (global resource control)
                "900",  # pr review timeout
                "false",  # voice
                "false",  # tts
                "true",  # memory enabled
                "true",  # memory extraction enabled
                # No codex-memory os_user reprompt (#522 removed it).
                "10",  # extraction timeout
                "8",  # consolidation candidates
                "3",  # episode classifier context turns
                "120",  # episode timeout
                "0.9",  # paraphrase-dedup threshold
                "2000",  # token budget
                "10",  # search limit
                "",  # perplexity key
            ]
        )
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        _cmd_config()

        # Protected users.yaml carries the explicit non-service target.
        users_yaml_path = tmp_path / "users.yaml"
        assert users_yaml_path.exists()
        data = yaml.safe_load(users_yaml_path.read_text())
        entry = data["users"][0]
        assert entry["telegram_id"] == 12345
        assert entry.get("os_user") == "testuser"

        # Integration pin: feed the produced env + users.yaml into
        # load_config and assert it accepts the shape. The wizard
        # staged users.yaml under tmp_path; loaded by load_config via
        # the sudo-cat shim that we redirect here so it returns the
        # staged content for /etc/kai/users.yaml. The oneshot-binary
        # resolver is mocked because the test host has no codex
        # binary; load_config validates the resolved path on the
        # codex extraction-eligible branch.
        env_block = json.loads((tmp_path / "install.conf").read_text())["env"]
        staged_users_yaml_text = users_yaml_path.read_text()
        monkeypatch.setattr("kai.config.PROJECT_ROOT", tmp_path)
        monkeypatch.setattr("kai.config.load_dotenv", lambda *a, **kw: None)

        def _fake_read(path):
            if path == "/etc/kai/users.yaml":
                return staged_users_yaml_text
            if path == "/etc/kai/env":
                # Mark the load as protected-mode so the resolver
                # routes at /etc/kai/users.yaml. The comment line is
                # skipped by the env-file parser.
                return "# test sentinel: protected install\n"
            return None

        monkeypatch.setattr("kai.config._read_protected_file", _fake_read)
        monkeypatch.setattr(
            "kai.config.validate_protected_file_metadata",
            lambda path, **kw: path == "/etc/kai/users.yaml",
        )
        monkeypatch.setattr(
            "kai.config.pwd.getpwnam",
            lambda name: MagicMock(pw_uid=os.geteuid() + 100_000),
        )
        monkeypatch.setattr(
            "kai.oneshot_binary.resolve_oneshot_binary",
            lambda backend: f"/fake/{backend}",
        )
        for key, value in env_block.items():
            monkeypatch.setenv(key, value)

        config = load_config()
        assert config.memory_enabled is True
        assert config.memory_extraction_enabled is True
        assert config.default_backend == "codex"
        assert config.user_configs is not None
        assert 12345 in config.user_configs
        assert config.user_configs[12345].os_user == "testuser"


# ── Apply subcommand ─────────────────────────────────────────────────


class TestCmdApply:
    def test_exits_if_not_root(self, monkeypatch):
        """Apply exits with code 1 if not running as root."""
        monkeypatch.setattr("os.geteuid", lambda: 1000)
        with pytest.raises(SystemExit):
            _cmd_apply()

    def test_exits_if_no_install_conf(self, tmp_path, monkeypatch):
        """Apply exits with code 1 if install.conf is missing."""
        monkeypatch.setattr("os.geteuid", lambda: 0)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "nope.conf")
        with pytest.raises(SystemExit):
            _cmd_apply()

    def test_exits_if_user_not_found(self, tmp_path, monkeypatch):
        """Apply exits if the service user doesn't exist on the system."""
        monkeypatch.setattr("os.geteuid", lambda: 0)
        conf_path = tmp_path / "install.conf"
        conf_path.write_text(
            json.dumps(
                {
                    "install_dir": "/opt/kai",
                    "data_dir": "/var/lib/kai",
                    "service_user": "nonexistent_user_abc123",
                    "platform": "darwin",
                    "env": {},
                }
            )
        )
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)

        with pytest.raises(SystemExit):
            _cmd_apply()

    def test_dry_run_makes_no_changes(self, tmp_path, monkeypatch, capsys):
        """DRY_RUN=1 prints actions without executing them."""
        monkeypatch.setattr("os.geteuid", lambda: 0)
        monkeypatch.setenv("DRY_RUN", "1")
        kai.install.USERS_YAML.write_text(
            "users:\n  - telegram_id: 1\n    name: test\n    role: admin\n    os_user: root\n"
        )

        conf_path = tmp_path / "install.conf"
        conf_path.write_text(
            json.dumps(
                {
                    "install_dir": str(tmp_path / "opt" / "kai"),
                    "data_dir": str(tmp_path / "var" / "lib" / "kai"),
                    "service_user": "nobody",
                    "platform": "darwin",
                    "env": {"TELEGRAM_BOT_TOKEN": "tok", "DEFAULT_BACKEND": "claude"},
                }
            )
        )
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)

        _cmd_apply()

        output = capsys.readouterr().out
        assert "[DRY RUN]" in output
        assert "[DRY RUN] Workshop bootstrap: pending" in output
        assert "1 human principal(s)" in output
        # Verify nothing was actually created
        assert not (tmp_path / "opt" / "kai").exists()
        # Secrets reminder should NOT appear during dry run
        assert "contains secrets" not in output

    def test_dry_run_flag_reaches_every_apply_helper(self, tmp_path, monkeypatch):
        """The apply orchestrator passes True to every mutation boundary.

        This protects the end-to-end contract independently of each helper's
        own no-mutation tests. A future positional-argument regression at any
        call site fails here before a privileged dry run can reach that helper
        with dry_run=False.
        """
        monkeypatch.setattr("os.geteuid", lambda: 0)
        monkeypatch.setenv("DRY_RUN", "1")
        kai.install.USERS_YAML.write_text(
            "users:\n  - telegram_id: 1\n    name: test\n    role: admin\n    os_user: root\n"
        )
        conf_path = tmp_path / "install.conf"
        conf_path.write_text(
            json.dumps(
                {
                    "install_dir": str(tmp_path / "opt" / "kai"),
                    "data_dir": str(tmp_path / "var" / "lib" / "kai"),
                    "service_user": "nobody",
                    "platform": "darwin",
                    "env": {"TELEGRAM_BOT_TOKEN": "tok", "DEFAULT_BACKEND": "claude"},
                }
            )
        )
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)

        dry_run_positions = {
            "_stop_service": 1,
            "_apply_directories": 4,
            "_apply_source": 3,
            "_apply_venv": 2,
            "_apply_models": 1,
            "_apply_secrets": 1,
            "_apply_backend_registry": 2,
            "_apply_goose_config": 4,
            "_apply_sudoers": 1,
            "_apply_migrate": 4,
            "_apply_service": 4,
            "_start_service": 1,
        }
        observed: dict[str, bool] = {}

        def recorder(name: str, position: int):
            def record(*args, **kwargs):
                value = kwargs.get("dry_run", args[position] if len(args) > position else None)
                observed[name] = value

            return record

        for name, position in dry_run_positions.items():
            monkeypatch.setattr(f"kai.install.{name}", recorder(name, position))

        _cmd_apply()

        assert observed == {name: True for name in dry_run_positions}

    def test_generates_env_file_content(self):
        """The generated env file contains all provided values."""
        env = {
            "TELEGRAM_BOT_TOKEN": "test-token",
            "WEBHOOK_PORT": "8080",
        }
        content = _generate_env_file(env)
        assert 'TELEGRAM_BOT_TOKEN="test-token"' in content
        assert 'WEBHOOK_PORT="8080"' in content

    def test_generates_launchd_plist_for_darwin(self):
        """macOS platform generates a valid launchd plist."""
        plist = _generate_launchd_plist("/opt/kai", "/var/lib/kai", "kai")
        assert "<?xml" in plist
        assert "com.syrinx.kai" in plist
        assert "KAI_DATA_DIR" in plist

    def test_generates_systemd_unit_for_linux(self):
        """Linux platform generates a valid systemd unit."""
        unit = _generate_systemd_unit("/opt/kai", "/var/lib/kai", "kai")
        assert "[Unit]" in unit
        assert "[Service]" in unit
        assert "KAI_DATA_DIR=/var/lib/kai" in unit

    def _minimal_install_conf(self, tmp_path):
        """Write the smallest valid install.conf that lets _cmd_apply
        reach its try/finally block; the rest of the apply path is
        either short-circuited by dry-run or monkey-patched per test.
        Returns the conf path."""
        kai.install.USERS_YAML.write_text(
            "users:\n  - telegram_id: 1\n    name: test\n    role: admin\n    os_user: root\n"
        )
        conf_path = tmp_path / "install.conf"
        conf_path.write_text(
            json.dumps(
                {
                    "install_dir": str(tmp_path / "opt" / "kai"),
                    "data_dir": str(tmp_path / "var" / "lib" / "kai"),
                    "service_user": "nobody",
                    "platform": "darwin",
                    "env": {"TELEGRAM_BOT_TOKEN": "tok", "DEFAULT_BACKEND": "claude"},
                }
            )
        )
        return conf_path

    def test_apply_propagates_service_start_error_when_apply_succeeds(self, monkeypatch, tmp_path, capsys):
        """The originating-issue propagation contract: when the apply
        path completes cleanly but _start_service raises
        ServiceStartError (verify exhausted), _cmd_apply re-raises so
        the install exits non-zero rather than reporting success with
        a dead daemon.

        Uses DRY_RUN=1 so the apply helpers no-op cleanly, then
        patches _start_service to simulate the verify-exhaustion
        path that the dry-run branch would otherwise skip."""
        monkeypatch.setattr("os.geteuid", lambda: 0)
        monkeypatch.setenv("DRY_RUN", "1")

        def fake_start(platform, dry_run, **kw):
            raise ServiceStartError("simulated verify exhaustion")

        monkeypatch.setattr("kai.install._start_service", fake_start)
        monkeypatch.setattr("kai.install.INSTALL_CONF", self._minimal_install_conf(tmp_path))

        with pytest.raises(ServiceStartError):
            _cmd_apply()

        # Recovery hint should reach the operator regardless of
        # propagation; pinning it here documents that the message
        # is part of the contract, not a side effect.
        out = capsys.readouterr().out
        assert "Manual recovery" in out

    def test_apply_swallows_service_start_error_when_apply_raised_systemexit(self, monkeypatch, tmp_path):
        """SystemExit is a BaseException, not an Exception, so it
        bypasses the apply try block's `except Exception` handler.
        An earlier implementation only initialized apply_succeeded
        inside the try body or the except clause; a SystemExit apply
        failure (`_apply_venv` Python-version gate, missing Goose
        template, visudo validation failure) left apply_succeeded
        unbound, and the finally block then raised UnboundLocalError,
        replacing the actionable apply failure with an internal
        control-flow error.

        Pre-initializing apply_succeeded = False before the try
        block is the fix. This test patches an apply step to raise
        SystemExit and asserts the SystemExit propagates intact, not
        UnboundLocalError, even when _start_service also raises.
        Regression for the same failure class the rest of the PR
        is closing."""
        monkeypatch.setattr("os.geteuid", lambda: 0)
        monkeypatch.setenv("DRY_RUN", "1")

        def fake_apply_secrets(env, dry_run, users_yaml_staging_path=None, protected_yaml_staging_paths=None):
            raise SystemExit("simulated apply SystemExit (e.g. venv version gate)")

        def fake_start(platform, dry_run, **kw):
            raise ServiceStartError("would otherwise replace the SystemExit")

        monkeypatch.setattr("kai.install._apply_secrets", fake_apply_secrets)
        monkeypatch.setattr("kai.install._start_service", fake_start)
        monkeypatch.setattr("kai.install.INSTALL_CONF", self._minimal_install_conf(tmp_path))

        with pytest.raises(SystemExit) as excinfo:
            _cmd_apply()
        assert "simulated apply SystemExit" in str(excinfo.value)

    def test_apply_swallows_service_start_error_when_apply_failed(self, monkeypatch, tmp_path, capsys):
        """The mask-prevention contract: when an apply step has
        already raised, a subsequent _start_service failure must NOT
        replace the original exception. Python normally swaps the
        propagating exception when a finally block raises; the
        finally guard explicitly checks apply_succeeded and skips
        the raise so the original failure is what the operator
        sees.

        Constructs an apply failure by patching _apply_secrets to
        raise; that step runs deep enough in the try block that the
        apply_succeeded flag has not been set yet."""
        monkeypatch.setattr("os.geteuid", lambda: 0)
        monkeypatch.setenv("DRY_RUN", "1")

        class ApplyBlewUp(RuntimeError):
            pass

        def fake_apply_secrets(env, dry_run, users_yaml_staging_path=None, protected_yaml_staging_paths=None):
            raise ApplyBlewUp("simulated apply step failure")

        def fake_start(platform, dry_run, **kw):
            raise ServiceStartError("would otherwise replace ApplyBlewUp")

        monkeypatch.setattr("kai.install._apply_secrets", fake_apply_secrets)
        monkeypatch.setattr("kai.install._start_service", fake_start)
        monkeypatch.setattr("kai.install.INSTALL_CONF", self._minimal_install_conf(tmp_path))

        # The ORIGINAL apply exception propagates, not the
        # ServiceStartError raised inside the finally.
        with pytest.raises(ApplyBlewUp):
            _cmd_apply()

        # The recovery hint is still printed; the swallowing only
        # affects the propagating exception type, not the operator's
        # visibility into both failure modes.
        out = capsys.readouterr().out
        assert "Manual recovery" in out


class TestProtectedUserIsolationPreflight:
    @staticmethod
    def _write_conf(tmp_path: Path, *, staging: Path | None = None) -> Path:
        conf = {
            "version": 1,
            "deployment_mode": "protected",
            "install_dir": str(tmp_path / "opt-kai"),
            "data_dir": str(tmp_path / "data"),
            "service_user": "kai-service",
            "platform": "darwin",
            "env": {"TELEGRAM_BOT_TOKEN": "tok", "DEFAULT_BACKEND": "claude", "DEFAULT_MODEL": "sonnet"},
        }
        if staging is not None:
            conf["users_yaml_staging_path"] = str(staging)
        path = tmp_path / "install.conf"
        path.write_text(json.dumps(conf))
        return path

    @staticmethod
    def _fake_accounts(monkeypatch, *, missing: str | None = None) -> None:
        class _FakePwd:
            pw_gid = 4242

            def __init__(self, uid):
                self.pw_uid = uid

        def _lookup(name):
            if name == missing:
                raise KeyError(name)
            return _FakePwd(4242 if name == "kai-service" else 5252)

        monkeypatch.setattr("kai.install.pwd.getpwnam", _lookup)

    def test_missing_os_user_aborts_before_service_stop(self, tmp_path, monkeypatch):
        kai.install.USERS_YAML.write_text("users:\n  - telegram_id: 1\n    name: alice\n    role: admin\n")
        monkeypatch.setattr("kai.install.INSTALL_CONF", self._write_conf(tmp_path))
        monkeypatch.setattr("os.geteuid", lambda: 0)
        self._fake_accounts(monkeypatch)
        stop = MagicMock()
        monkeypatch.setattr("kai.install._stop_service", stop)

        with pytest.raises(SystemExit, match="missing required os_user"):
            _cmd_apply()
        stop.assert_not_called()
        assert not (tmp_path / "opt-kai").exists()

    def test_service_account_mapping_aborts_before_service_stop(self, tmp_path, monkeypatch):
        kai.install.USERS_YAML.write_text(
            "users:\n  - telegram_id: 1\n    name: alice\n    role: admin\n    os_user: kai-service\n"
        )
        monkeypatch.setattr("kai.install.INSTALL_CONF", self._write_conf(tmp_path))
        monkeypatch.setattr("os.geteuid", lambda: 0)
        self._fake_accounts(monkeypatch)
        stop = MagicMock()
        monkeypatch.setattr("kai.install._stop_service", stop)

        with pytest.raises(SystemExit, match="maps to service account"):
            _cmd_apply()
        stop.assert_not_called()

    def test_nonexistent_target_aborts_before_service_stop(self, tmp_path, monkeypatch):
        kai.install.USERS_YAML.write_text(
            "users:\n  - telegram_id: 1\n    name: alice\n    role: admin\n    os_user: absent-user\n"
        )
        monkeypatch.setattr("kai.install.INSTALL_CONF", self._write_conf(tmp_path))
        monkeypatch.setattr("os.geteuid", lambda: 0)
        self._fake_accounts(monkeypatch, missing="absent-user")
        stop = MagicMock()
        monkeypatch.setattr("kai.install._stop_service", stop)

        with pytest.raises(SystemExit, match="nonexistent OS account"):
            _cmd_apply()
        stop.assert_not_called()

    def test_staged_users_yaml_is_the_preflight_authority(self, tmp_path, monkeypatch):
        kai.install.USERS_YAML.write_text(
            "users:\n  - telegram_id: 1\n    name: safe\n    role: admin\n    os_user: safe-user\n"
        )
        staging = tmp_path / "staged-users.yaml"
        staging.write_text("users:\n  - telegram_id: 2\n    name: unsafe\n    role: admin\n")
        monkeypatch.setattr(
            "kai.install.INSTALL_CONF",
            self._write_conf(tmp_path, staging=staging),
        )
        monkeypatch.setattr("os.geteuid", lambda: 0)
        self._fake_accounts(monkeypatch)
        stop = MagicMock()
        monkeypatch.setattr("kai.install._stop_service", stop)

        with pytest.raises(SystemExit, match="missing required os_user"):
            _cmd_apply()
        stop.assert_not_called()


class TestCmdConfigDefaultModelDispatch:
    """
    Wizard handling for retired DEFAULT_MODEL prompting.

    The conversational default now comes from MODEL_REGISTRY's
    ModelRole.AGENT row for the effective backend/provider. Re-running
    the wizard does not call _prompt_default_model and does not emit
    DEFAULT_MODEL, even when an older install.conf still carries one.

    The wizard has many prompts that fire before and after the model
    dispatch. The helper below mocks _prompt_default_model to capture
    its call args and return a fixed value, leaving the other prompts
    to be driven via an input chain.
    """

    @staticmethod
    def _setup(monkeypatch, tmp_path, existing_env: dict[str, str] | None = None) -> None:
        """
        Configure the wizard sandbox so users_yaml_exists=True.

        Simulates `/etc/kai/users.yaml` already in place (canonical
        path) so the wizard takes the existing-config branch and skips
        the per-user prompts. Optionally seeds install.conf with the
        given env so existing_env reflects the desired pre-existing state.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        # Pretend the canonical users.yaml exists with one isolated user;
        # only the presence flag matters for this dispatch test. Protected
        # mode now validates the OS-user boundary before later prompts.
        # Compares against the live USERS_YAML attribute so it
        # stacks on the autouse `_isolate_users_yaml` redirect.
        _real_exists = Path.exists

        def _exists_with_canonical(self):
            if self == kai.install.USERS_YAML:
                return True
            return _real_exists(self)

        monkeypatch.setattr(Path, "exists", _exists_with_canonical)
        monkeypatch.setattr(
            "kai.install._read_users_yaml_text",
            lambda path: "users:\n  - telegram_id: 1\n    name: test\n    role: admin\n    os_user: test-os-user\n",
        )

        if existing_env is not None:
            (tmp_path / "install.conf").write_text(json.dumps({"env": existing_env, "version": 1}))

    @staticmethod
    def _inputs_for_claude_backend() -> list[str]:
        """Input chain for the users_yaml_exists=True + claude path.

        Post-tranche-B, the global-default prompts fire unconditionally
        even when users.yaml exists, so the chain feeds timeout,
        workspace_base, and pr_review_cooldown
        values that the pre-tranche-B users-yaml-exists branch had
        skipped silently.
        """
        return [
            "protected",  # deployment mode
            "/opt/kai",  # install dir
            "/var/lib/kai",  # data dir
            "kai",  # service user
            "darwin",  # platform
            "fake-token",  # bot token
            "polling",  # transport
            "claude",  # agent backend
            # no DEFAULT_MODEL prompt; agent default comes from MODEL_REGISTRY
            "false",  # customize per-role models (decline; use registry defaults)
            "120",  # agent timeout (global default)
            "0",  # max session age hours (0 = no limit)
            "1800",  # idle eviction timeout seconds
            "80",  # autocompact pct
            "",  # effort level (default)
            "8080",  # webhook port
            "",  # Workshop LAN address (disabled)
            "test-secret",  # webhook secret
            "~/Projects",  # workspace base (global default)
            "",  # allowed workspaces (global default, empty)
            "300",  # pr review cooldown (global resource control)
            "900",  # pr review subprocess timeout
            "false",  # voice
            "false",  # tts
            "false",  # memory enabled
            "",  # perplexity key
        ]

    @staticmethod
    def _inputs_for_codex_subscription(memory_enabled: str = "false") -> list[str]:
        """
        Input chain for the users_yaml_exists=True + codex+subscription path.

        Codex bypasses the legacy provider/key block (VALID_PROVIDERS["codex"]
        is intentionally absent), so no provider or API key prompts fire.
        autocompact_pct and effort_level are claude-only and suppressed.
        The codex auth-mode prompt is the new line vs the goose chain.
        """
        return [
            "protected",  # deployment mode
            "/opt/kai",  # install dir
            "/var/lib/kai",  # data dir
            "kai",  # service user
            "darwin",  # platform
            "fake-token",  # bot token
            "polling",  # transport
            "codex",  # agent backend
            "subscription",  # codex auth mode
            # No OPENAI_API_KEY prompt in subscription mode
            # No provider prompt (codex is single-provider; absent from BACKENDS_NEEDING_PROVIDER_PROMPT)
            # no DEFAULT_MODEL prompt; agent default comes from MODEL_REGISTRY
            "false",  # customize per-role models (decline; use registry defaults)
            "120",  # agent timeout (global default)
            "0",  # max session age hours (0 = no limit)
            "1800",  # idle eviction timeout seconds
            # No autocompact_pct / claude effort prompts (claude-only)
            "",  # codex reasoning effort (empty = codex default)
            "8080",  # webhook port
            "",  # Workshop LAN address (disabled)
            "test-secret",  # webhook secret
            "~/Projects",  # workspace base (global default)
            "",  # allowed workspaces (global default, empty)
            "300",  # pr review cooldown (global resource control)
            "900",  # pr review subprocess timeout
            "false",  # voice
            "false",  # tts
            memory_enabled,  # memory enabled
            "",  # perplexity key
        ]

    @staticmethod
    def _inputs_for_goose_openai() -> list[str]:
        """Input chain for the users_yaml_exists=True + goose+openai path."""
        return [
            "protected",  # deployment mode
            "/opt/kai",  # install dir
            "/var/lib/kai",  # data dir
            "kai",  # service user
            "darwin",  # platform
            "fake-token",  # bot token
            "polling",  # transport
            "goose",  # agent backend
            "openai",  # llm provider
            "openai-key",  # OPENAI_API_KEY
            # no DEFAULT_MODEL prompt; agent default comes from MODEL_REGISTRY
            "false",  # customize per-role models (decline; use registry defaults)
            "120",  # agent timeout (global default)
            "0",  # max session age hours (0 = no limit)
            "1800",  # idle eviction timeout seconds
            # autocompact_pct / effort prompts are backend-gated
            # (claude tunables and the codex effort prompt all skip
            # on goose)
            "8080",  # webhook port
            "",  # Workshop LAN address (disabled)
            "test-secret",  # webhook secret
            "~/Projects",  # workspace base (global default)
            "",  # allowed workspaces (global default, empty)
            "300",  # pr review cooldown (global resource control)
            "900",  # pr review subprocess timeout
            "false",  # voice
            "false",  # tts
            "false",  # memory enabled
            "",  # perplexity key
        ]

    def _run(self, monkeypatch, tmp_path, inputs: list[str], helper_return: str):
        """
        Run _cmd_config with _prompt_default_model mocked.

        Returns (mock_object, written_env). The mock should remain
        uncalled; it is installed to catch accidental reintroduction of
        the retired DEFAULT_MODEL prompt.
        """
        from unittest.mock import MagicMock

        helper_mock = MagicMock(return_value=helper_return)
        monkeypatch.setattr("kai.install._prompt_default_model", helper_mock)
        # Patch the codex-bin existence check so tests can pass any
        # path string without needing a real executable on disk.
        # Tests that specifically exercise the validator should patch
        # this in their own scope.
        monkeypatch.setattr("kai.install._validate_codex_bin", lambda p: bool(p))
        inputs_iter = iter(inputs)
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs_iter))

        _cmd_config()
        conf = json.loads((tmp_path / "install.conf").read_text())
        return helper_mock, conf["env"]

    def test_codex_effort_default_omits_env_key(self, tmp_path, monkeypatch):
        """Accepting the empty default leaves CODEX_EFFORT_LEVEL out
        of install.conf entirely: the set-or-absent contract means
        absence IS the 'use codex's own config' signal. Pairs with
        the non-default test below to pin both emission branches."""
        self._setup(monkeypatch, tmp_path)
        _, env = self._run(
            monkeypatch,
            tmp_path,
            self._inputs_for_codex_subscription(),
            helper_return="gpt-5.5",
        )
        assert "CODEX_EFFORT_LEVEL" not in env

    def test_codex_effort_non_default_writes_env_key(self, tmp_path, monkeypatch):
        """A chosen effort tier round-trips from the wizard prompt
        into install.conf, lowercased by the prompt's normalization
        (mixed case pins the .strip().lower() behavior)."""
        self._setup(monkeypatch, tmp_path)
        base = list(self._inputs_for_codex_subscription())
        idx = base.index("8080")
        # The slot immediately before the webhook port is the codex
        # effort answer in this chain.
        assert base[idx - 1] == ""
        base[idx - 1] = "HIGH"
        _, env = self._run(monkeypatch, tmp_path, base, helper_return="gpt-5.5")
        assert env.get("CODEX_EFFORT_LEVEL") == "high"

    def test_install_conf_legacy_AGENT_BACKEND_migrates_on_resave(self, tmp_path, monkeypatch):
        """Re-running the wizard against a prior install.conf that
        carries the deprecated AGENT_BACKEND key writes DEFAULT_BACKEND
        only: the wizard never re-emits the legacy key, so the resaved
        install.conf is clean. The legacy value is still honored as the
        prefill (read through the resolver)."""
        self._setup(monkeypatch, tmp_path, existing_env={"AGENT_BACKEND": "codex"})
        _, env = self._run(
            monkeypatch,
            tmp_path,
            self._inputs_for_codex_subscription(),
            helper_return="gpt-5.5",
        )
        assert env.get("DEFAULT_BACKEND") == "codex"
        assert "AGENT_BACKEND" not in env

    def test_codex_effort_prompt_displays_choices(self, tmp_path, monkeypatch):
        """
        The Codex effort prompt advertises the allowed values and the
        empty-default semantics BEFORE accepting input, so the operator
        does not have to guess the vocabulary.

        Replaces the default `lambda prompt: next(inputs_iter)` stub
        with a recording variant so the test can inspect every prompt
        string the wizard issues, then asserts that the prompt issued
        for the codex effort slot contains every value in
        CODEX_EFFORT_LEVELS plus the `empty = codex default` hint.
        """
        from unittest.mock import MagicMock

        from kai.config import CODEX_EFFORT_LEVELS

        self._setup(monkeypatch, tmp_path)
        helper_mock = MagicMock(return_value="gpt-5.5")
        monkeypatch.setattr("kai.install._prompt_default_model", helper_mock)
        monkeypatch.setattr("kai.install._validate_codex_bin", lambda p: bool(p))

        recorded_prompts: list[str] = []
        inputs_iter = iter(self._inputs_for_codex_subscription())

        def mock_input(prompt: str) -> str:
            recorded_prompts.append(prompt)
            return next(inputs_iter)

        monkeypatch.setattr("builtins.input", mock_input)
        _cmd_config()

        # Find the prompt issued for the Codex effort slot. The label
        # "Codex reasoning effort" is unique in the wizard chain so a
        # substring match identifies the right prompt without coupling
        # to its position in the input list.
        codex_prompts = [p for p in recorded_prompts if "Codex reasoning effort" in p]
        assert len(codex_prompts) == 1, f"expected exactly one Codex effort prompt, got {codex_prompts!r}"
        prompt = codex_prompts[0]
        for level in CODEX_EFFORT_LEVELS:
            assert level in prompt, f"codex effort prompt missing level {level!r}; prompt={prompt!r}"
        assert "empty = codex default" in prompt

    def test_provider_flip_drops_existing_default_model(self, tmp_path, monkeypatch):
        """
        With users.yaml present and DEFAULT_MODEL=sonnet, flipping to
        goose+openai does not prompt for a replacement. The regenerated
        install.conf drops DEFAULT_MODEL so runtime uses the registry
        agent default for goose/openai.
        """
        self._setup(monkeypatch, tmp_path, existing_env={"DEFAULT_MODEL": "sonnet"})
        monkeypatch.setattr("kai.install._validate_goose_bin", lambda p: bool(p))
        helper, env = self._run(
            monkeypatch,
            tmp_path,
            self._inputs_for_goose_openai(),
            helper_return="gpt-5.4-mini",
        )
        helper.assert_not_called()
        assert env["DEFAULT_BACKEND"] == "goose"
        assert env["DEFAULT_PROVIDER"] == "openai"
        assert "DEFAULT_MODEL" not in env

    def test_valid_existing_default_model_is_not_reemitted(self, tmp_path, monkeypatch):
        """
        With users.yaml present, DEFAULT_MODEL=sonnet, and the wizard
        kept on claude, the model prompt does not fire and the
        regenerated install.conf omits DEFAULT_MODEL.
        """
        self._setup(monkeypatch, tmp_path, existing_env={"DEFAULT_MODEL": "sonnet"})
        helper, env = self._run(
            monkeypatch,
            tmp_path,
            self._inputs_for_claude_backend(),
            helper_return="haiku",
        )
        helper.assert_not_called()
        assert "DEFAULT_MODEL" not in env

    def test_empty_existing_model_uses_registry_default_claude(self, tmp_path, monkeypatch):
        """
        users.yaml present, no DEFAULT_MODEL in existing env, backend
        stays claude. The wizard does not prompt; runtime falls through
        to MODEL_REGISTRY's claude/anthropic agent default.
        """
        self._setup(monkeypatch, tmp_path, existing_env={})
        helper, env = self._run(
            monkeypatch,
            tmp_path,
            self._inputs_for_claude_backend(),
            helper_return="sonnet",
        )
        helper.assert_not_called()
        assert "DEFAULT_MODEL" not in env

    def test_empty_existing_model_uses_registry_default_openai(self, tmp_path, monkeypatch):
        """
        Empty-existing-model path on a non-anthropic provider: flip to
        goose+openai with no DEFAULT_MODEL in existing env. The wizard
        does not prompt; runtime falls through to MODEL_REGISTRY's
        goose/openai agent default.
        """
        self._setup(monkeypatch, tmp_path, existing_env={})
        monkeypatch.setattr("kai.install._validate_goose_bin", lambda p: bool(p))
        helper, env = self._run(
            monkeypatch,
            tmp_path,
            self._inputs_for_goose_openai(),
            helper_return="gpt-5.4",
        )
        helper.assert_not_called()
        assert env["DEFAULT_BACKEND"] == "goose"
        assert env["DEFAULT_PROVIDER"] == "openai"
        assert "DEFAULT_MODEL" not in env

    def test_invalid_existing_default_model_is_dropped(self, tmp_path, monkeypatch):
        """
        When an older install.conf carries a DEFAULT_MODEL that is
        invalid for the newly selected backend/provider, the wizard
        drops it instead of prompting for a replacement. Runtime then
        uses MODEL_REGISTRY's agent default.
        """
        self._setup(monkeypatch, tmp_path, existing_env={"DEFAULT_MODEL": "opus"})
        monkeypatch.setattr("kai.install._validate_goose_bin", lambda p: bool(p))
        helper, env = self._run(
            monkeypatch,
            tmp_path,
            self._inputs_for_goose_openai(),
            helper_return="gpt-5.4-mini",
        )
        helper.assert_not_called()
        assert env["DEFAULT_BACKEND"] == "goose"
        assert env["DEFAULT_PROVIDER"] == "openai"
        assert "DEFAULT_MODEL" not in env

    def test_codex_install_enables_memory_with_codex_reasoner(self, tmp_path, monkeypatch):
        """
        Codex installs CAN enable semantic memory. The wizard no
        longer prompts for MEMORY_REASONER_BACKEND or the episode
        model (issue #515 retired both); the memory reasoner derives
        from each user's effective `agent_backend` at extraction time.
        The extraction-enabled flag persists and the stale-key cleanup
        at wizard end no longer strips the codex extraction config.

        Regression for the prior wizard behavior that forced both
        memory flags off on codex installs and printed a "currently
        requires the claude backend" message.
        """
        self._setup(
            monkeypatch,
            tmp_path,
            existing_env={"DEFAULT_MODEL": "gpt-5.4"},
        )
        memory_inputs = [
            "true",  # MEMORY_ENABLED
            "true",  # MEMORY_EXTRACTION_ENABLED
            "10",  # extraction timeout (dataclass default)
            "8",  # consolidation candidates (dataclass default)
            "3",  # episode classifier context turns (dataclass default)
            "120",  # episode timeout (dataclass default)
            "0.9",  # paraphrase-dedup threshold (dataclass default)
            "2000",  # token budget (dataclass default)
            "10",  # search limit (dataclass default)
        ]
        # Codex-subscription inputs sequence inserts memory_enabled
        # as the second-to-last entry (before the perplexity key).
        # Replace memory_enabled with "true" and append the rest of
        # the memory block in order; the helper currently emits
        # memory_enabled as the final memory-flag entry, so we splice
        # the new prompt sequence into the rest of the chain.
        base_inputs = self._inputs_for_codex_subscription(memory_enabled="true")
        # Find the memory_enabled slot ("true") and insert the rest
        # of the memory block immediately after it; the trailing
        # perplexity key entry stays at the end.
        idx = len(base_inputs) - 2  # memory_enabled position
        inputs = base_inputs[: idx + 1] + memory_inputs[1:] + base_inputs[idx + 1 :]
        _, env = self._run(monkeypatch, tmp_path, inputs, helper_return="gpt-5.4")
        # Codex install persists memory extraction config; the global
        # DEFAULT_BACKEND=codex is what makes codex extraction-eligible.
        assert env.get("MEMORY_ENABLED") == "true"
        assert env.get("MEMORY_EXTRACTION_ENABLED") == "true"
        assert env.get("DEFAULT_BACKEND") == "codex"
        # MEMORY_REASONER_BACKEND retired; the wizard must not emit it.
        assert "MEMORY_REASONER_BACKEND" not in env


class TestCmdApplyDefaultModelGate:
    """
    Defensive validation in _cmd_apply for the (DEFAULT_MODEL, provider) pair.

    The gate sits immediately after the service-user check and before
    the DRY_RUN read, so all side-effect entry points (_stop_service,
    _apply_directories, _apply_secrets, etc.) are unreachable on a failed
    validation; each test asserts the gate fires before any of them runs.
    """

    @staticmethod
    def _write_install_conf(tmp_path, env: dict[str, str]) -> Path:
        """Write a minimal install.conf with the given env dict."""
        if "DEFAULT_BACKEND" not in env and "AGENT_BACKEND" not in env:
            env = {"DEFAULT_BACKEND": "claude", **env}
        kai.install.USERS_YAML.write_text(
            "users:\n  - telegram_id: 1\n    name: test\n    role: admin\n    os_user: root\n"
        )
        conf_path = tmp_path / "install.conf"
        conf_path.write_text(
            json.dumps(
                {
                    "install_dir": str(tmp_path / "opt" / "kai"),
                    "data_dir": str(tmp_path / "var" / "lib" / "kai"),
                    "service_user": "nobody",
                    "platform": "darwin",
                    "env": env,
                }
            )
        )
        return conf_path

    @staticmethod
    def _patch_side_effects(monkeypatch):
        """
        Replace every apply-time side-effect entry point with a tracking mock.

        Returns the mock for _stop_service since that is the canonical
        "did we reach the body of apply" probe; the other mocks are silenced
        so tests do not need to know which helpers run on the happy path.
        """
        from unittest.mock import MagicMock

        stop_service_mock = MagicMock()
        for fn_name in (
            "_stop_service",
            "_apply_directories",
            "_apply_source",
            "_apply_venv",
            "_apply_models",
            "_apply_secrets",
            "_apply_backend_registry",
            "_apply_runtime_policy",
            "_apply_goose_config",
            "_apply_sudoers",
            "_apply_migrate",
            "_apply_service",
            "_start_service",
        ):
            mock = stop_service_mock if fn_name == "_stop_service" else MagicMock()
            monkeypatch.setattr(f"kai.install.{fn_name}", mock, raising=False)
        return stop_service_mock

    def test_rejects_incompatible_default_model(self, tmp_path, monkeypatch):
        """
        Model + provider mismatch raises SystemExit before any side effect.
        Error names both the bad value and the provider.
        """
        monkeypatch.setattr("os.geteuid", lambda: 0)
        conf_path = self._write_install_conf(
            tmp_path,
            {"DEFAULT_MODEL": "sonnet", "DEFAULT_BACKEND": "goose", "DEFAULT_PROVIDER": "openai"},
        )
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        stop_service = self._patch_side_effects(monkeypatch)

        with pytest.raises(SystemExit) as excinfo:
            _cmd_apply()
        msg = str(excinfo.value)
        assert "sonnet" in msg
        assert "openai" in msg
        assert "registry default" in msg
        stop_service.assert_not_called()

    def test_accepts_missing_default_model_on_non_anthropic(self, tmp_path, monkeypatch):
        """
        Missing DEFAULT_MODEL on a non-anthropic provider is valid now:
        load_config uses MODEL_REGISTRY's agent default for the active
        backend/provider.
        """
        monkeypatch.setattr("os.geteuid", lambda: 0)
        conf_path = self._write_install_conf(
            tmp_path,
            {"DEFAULT_BACKEND": "goose", "DEFAULT_PROVIDER": "openai"},
        )
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        stop_service = self._patch_side_effects(monkeypatch)

        _cmd_apply()
        stop_service.assert_called_once()

    def test_rejects_invalid_registry_agent_default(self, tmp_path, monkeypatch):
        """
        Missing DEFAULT_MODEL does not mean "install anything": apply
        validates MODEL_REGISTRY's agent default for the selected
        backend/provider before any service side effects.
        """
        monkeypatch.setattr("os.geteuid", lambda: 0)
        conf_path = self._write_install_conf(
            tmp_path,
            {"DEFAULT_BACKEND": "goose", "DEFAULT_PROVIDER": "openai"},
        )
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        monkeypatch.setattr("kai.install.get_default_model_for_backend", lambda *_args: "sonnet")
        stop_service = self._patch_side_effects(monkeypatch)

        with pytest.raises(SystemExit) as excinfo:
            _cmd_apply()
        msg = str(excinfo.value)
        assert "MODEL_REGISTRY agent default" in msg
        assert "no usable default model is installed" in msg
        assert "openai" in msg
        stop_service.assert_not_called()

    def test_accepts_missing_default_model_on_anthropic(self, tmp_path, monkeypatch):
        """
        Missing DEFAULT_MODEL on anthropic resolves through
        MODEL_REGISTRY's agent default, so the gate passes and apply
        proceeds past _stop_service.
        """
        monkeypatch.setattr("os.geteuid", lambda: 0)
        conf_path = self._write_install_conf(
            tmp_path,
            {"DEFAULT_BACKEND": "claude"},
        )
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        stop_service = self._patch_side_effects(monkeypatch)
        # DRY_RUN suppresses real filesystem operations on the happy path;
        # the gate runs before the DRY_RUN read, so this affects only
        # what happens AFTER the gate accepts the config.
        monkeypatch.setenv("DRY_RUN", "1")

        _cmd_apply()
        stop_service.assert_called_once()

    def test_accepts_open_ended_provider(self, tmp_path, monkeypatch):
        """
        Open-ended providers (openrouter, ollama) accept any non-empty
        model. validate_model_for_provider returns True for them, so the
        gate does not block regardless of the model string.
        """
        monkeypatch.setattr("os.geteuid", lambda: 0)
        conf_path = self._write_install_conf(
            tmp_path,
            {
                "DEFAULT_MODEL": "anthropic/claude-sonnet-4-6",
                "DEFAULT_BACKEND": "goose",
                "DEFAULT_PROVIDER": "openrouter",
            },
        )
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        stop_service = self._patch_side_effects(monkeypatch)
        monkeypatch.setenv("DRY_RUN", "1")

        _cmd_apply()
        stop_service.assert_called_once()

    def test_dry_run_still_validates(self, tmp_path, monkeypatch):
        """
        DRY_RUN=1 does not bypass the gate. Validation runs before the
        dry-run flag is even read, so a bad config exits with the same
        SystemExit it would emit on a non-dry-run apply.
        """
        monkeypatch.setattr("os.geteuid", lambda: 0)
        monkeypatch.setenv("DRY_RUN", "1")
        conf_path = self._write_install_conf(
            tmp_path,
            {"DEFAULT_MODEL": "sonnet", "DEFAULT_BACKEND": "goose", "DEFAULT_PROVIDER": "openai"},
        )
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        stop_service = self._patch_side_effects(monkeypatch)

        with pytest.raises(SystemExit) as excinfo:
            _cmd_apply()
        msg = str(excinfo.value)
        assert "sonnet" in msg
        assert "openai" in msg
        stop_service.assert_not_called()

    def test_normalizes_llm_provider_case_and_whitespace(self, tmp_path, monkeypatch):
        """
        Mirrors load_config's .strip().lower() normalization on
        DEFAULT_BACKEND and DEFAULT_PROVIDER. The test relies on a code path
        where normalized and un-normalized lookups produce DIFFERENT
        gate outcomes, so that the test fails if either transformation
        is dropped from the gate.

        Setup: DEFAULT_BACKEND="goose", DEFAULT_PROVIDER=" OpenAI ", DEFAULT_MODEL="sonnet".
        - With .strip().lower() applied: eff_provider="openai",
          validate("sonnet", "openai") returns False (sonnet is not in
          PROVIDER_MODELS["openai"]) -> SystemExit.
        - Without .strip().lower(): eff_provider=" OpenAI ", which is
          not in PROVIDER_MODELS or OPEN_ENDED_PROVIDERS. The validator
          falls through the unknown-provider branch (which returns True
          with a warning) and the gate silently accepts.

        Asserting SystemExit therefore captures BOTH transformations:
        dropping .strip() leaves " OpenAI " unknown; dropping .lower()
        leaves "OpenAI" unknown; either way the gate would silently
        accept and the test would fail.
        """
        monkeypatch.setattr("os.geteuid", lambda: 0)
        conf_path = self._write_install_conf(
            tmp_path,
            {
                "DEFAULT_MODEL": "sonnet",
                "DEFAULT_BACKEND": "goose",
                "DEFAULT_PROVIDER": " OpenAI ",
            },
        )
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        stop_service = self._patch_side_effects(monkeypatch)

        with pytest.raises(SystemExit) as excinfo:
            _cmd_apply()
        # The error message should name the normalized provider, not the
        # raw input, so operator-facing output stays consistent with the
        # load_config error wording.
        assert "openai" in str(excinfo.value).lower()
        stop_service.assert_not_called()

    def test_apply_resolves_legacy_AGENT_BACKEND(self, tmp_path, monkeypatch):
        """A legacy install.conf carrying AGENT_BACKEND=codex must
        validate the default model against the CODEX surface (not
        anthropic) and migrate the env dict so /etc/kai/env is written
        with DEFAULT_BACKEND and no AGENT_BACKEND.

        Apply consumes the install.conf env dict directly, never through
        load_config, so the migration has to happen inside _cmd_apply.
        gpt-5.5 is codex-valid but NOT anthropic-valid: if the apply
        path defaulted the backend to claude, this would SystemExit.
        """
        from unittest.mock import MagicMock

        monkeypatch.setattr("os.geteuid", lambda: 0)
        conf_path = self._write_install_conf(
            tmp_path,
            {"DEFAULT_MODEL": "gpt-5.5", "AGENT_BACKEND": "codex"},
        )
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        self._patch_side_effects(monkeypatch)
        # Capture the env dict handed to _apply_secrets to confirm the
        # in-memory migration ran before the write.
        secrets_mock = MagicMock()
        monkeypatch.setattr("kai.install._apply_secrets", secrets_mock)
        monkeypatch.setenv("DRY_RUN", "1")

        _cmd_apply()

        secrets_mock.assert_called_once()
        written_env = secrets_mock.call_args.args[0]
        assert written_env.get("DEFAULT_BACKEND") == "codex"
        assert "AGENT_BACKEND" not in written_env

    def test_apply_canonicalizes_legacy_codex_gpt56_model(self, tmp_path, monkeypatch):
        """Apply rewrites the rejected family shorthand before writing /etc/kai/env."""
        from unittest.mock import MagicMock

        monkeypatch.setattr("os.geteuid", lambda: 0)
        conf_path = self._write_install_conf(
            tmp_path,
            {"DEFAULT_MODEL": "gpt-5.6", "DEFAULT_BACKEND": "codex"},
        )
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        self._patch_side_effects(monkeypatch)
        secrets_mock = MagicMock()
        monkeypatch.setattr("kai.install._apply_secrets", secrets_mock)
        monkeypatch.setenv("DRY_RUN", "1")

        _cmd_apply()

        written_env = secrets_mock.call_args.args[0]
        assert written_env["DEFAULT_MODEL"] == "gpt-5.6-sol"

    def test_apply_resolves_new_DEFAULT_BACKEND(self, tmp_path, monkeypatch):
        """A new-name install.conf passes through cleanly: goose-path
        setup is reached (the gate accepts a goose-valid model and apply
        proceeds past _stop_service)."""
        monkeypatch.setattr("os.geteuid", lambda: 0)
        conf_path = self._write_install_conf(
            tmp_path,
            {"DEFAULT_MODEL": "gpt-5.5", "DEFAULT_BACKEND": "goose", "DEFAULT_PROVIDER": "openai"},
        )
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        stop_service = self._patch_side_effects(monkeypatch)
        monkeypatch.setenv("DRY_RUN", "1")

        _cmd_apply()
        stop_service.assert_called_once()

    def test_apply_normalizes_mixed_case_global_backend(self, tmp_path, monkeypatch):
        """A hand-edited install.conf with `DEFAULT_BACKEND: "Goose"`
        normalizes to lowercase in the written env so the downstream
        goose-config / sudoers gates (which read the raw value) and
        /etc/kai/env all see the canonical form. Without the write-back,
        validation lowercases for the gate but `_apply_goose_config`
        would see "Goose" and skip the deploy."""
        from unittest.mock import MagicMock

        monkeypatch.setattr("os.geteuid", lambda: 0)
        conf_path = self._write_install_conf(
            tmp_path,
            {"DEFAULT_MODEL": "gpt-5.5", "DEFAULT_BACKEND": "Goose", "DEFAULT_PROVIDER": "openai"},
        )
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        self._patch_side_effects(monkeypatch)
        secrets_mock = MagicMock()
        monkeypatch.setattr("kai.install._apply_secrets", secrets_mock)
        goose_mock = MagicMock()
        monkeypatch.setattr("kai.install._apply_goose_config", goose_mock)
        monkeypatch.setenv("DRY_RUN", "1")

        _cmd_apply()

        written_env = secrets_mock.call_args.args[0]
        assert written_env.get("DEFAULT_BACKEND") == "goose"
        # The goose-config deploy gate receives the normalized value.
        assert goose_mock.call_args.kwargs.get("agent_backend") == "goose"

    def test_apply_migrates_legacy_LLM_PROVIDER(self, tmp_path, monkeypatch):
        """A legacy install.conf carrying LLM_PROVIDER migrates to
        DEFAULT_PROVIDER in the env dict so /etc/kai/env is written with
        the new name and no LLM_PROVIDER. Mirrors the AGENT_BACKEND
        migration."""
        from unittest.mock import MagicMock

        monkeypatch.setattr("os.geteuid", lambda: 0)
        conf_path = self._write_install_conf(
            tmp_path,
            {"DEFAULT_MODEL": "gpt-5.5", "DEFAULT_BACKEND": "goose", "LLM_PROVIDER": "openai"},
        )
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        self._patch_side_effects(monkeypatch)
        secrets_mock = MagicMock()
        monkeypatch.setattr("kai.install._apply_secrets", secrets_mock)
        monkeypatch.setenv("DRY_RUN", "1")

        _cmd_apply()

        written_env = secrets_mock.call_args.args[0]
        assert written_env.get("DEFAULT_PROVIDER") == "openai"
        assert "LLM_PROVIDER" not in written_env

    def test_apply_drops_unsupported_WEBHOOK_SECRET(self, tmp_path, monkeypatch):
        """An old install.conf cannot redeploy the retired credential."""
        from unittest.mock import MagicMock

        monkeypatch.setattr("os.geteuid", lambda: 0)
        conf_path = self._write_install_conf(
            tmp_path,
            {
                "DEFAULT_MODEL": "sonnet",
                "DEFAULT_BACKEND": "claude",
                "WEBHOOK_SECRET": "legacy-secret",
                "GITHUB_WEBHOOK_SECRET": "github-secret",
                "GENERIC_WEBHOOK_SECRET": "generic-secret",
            },
        )
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        self._patch_side_effects(monkeypatch)
        secrets_mock = MagicMock()
        monkeypatch.setattr("kai.install._apply_secrets", secrets_mock)
        monkeypatch.setenv("DRY_RUN", "1")

        _cmd_apply()

        written_env = secrets_mock.call_args.args[0]
        assert "WEBHOOK_SECRET" not in written_env
        assert written_env["GITHUB_WEBHOOK_SECRET"] == "github-secret"
        assert written_env["GENERIC_WEBHOOK_SECRET"] == "generic-secret"

    @pytest.mark.parametrize(
        "named_secrets",
        [
            {},
            {"GITHUB_WEBHOOK_SECRET": "github-secret"},
            {"GENERIC_WEBHOOK_SECRET": "generic-secret"},
        ],
    )
    def test_apply_refuses_legacy_secret_without_all_named_replacements(
        self,
        tmp_path,
        monkeypatch,
        named_secrets,
    ):
        """Retirement cannot silently disable a formerly shared route."""
        monkeypatch.setattr("os.geteuid", lambda: 0)
        conf_path = self._write_install_conf(
            tmp_path,
            {
                "DEFAULT_MODEL": "sonnet",
                "DEFAULT_BACKEND": "claude",
                "WEBHOOK_SECRET": "legacy-secret",
                **named_secrets,
            },
        )
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        stop_service = self._patch_side_effects(monkeypatch)

        with pytest.raises(SystemExit, match="Run 'make config' once before 'make install'"):
            _cmd_apply()

        stop_service.assert_not_called()

    def test_apply_migrates_legacy_AGENT_TIMEOUT_SECONDS(self, tmp_path, monkeypatch):
        """A legacy install.conf carrying AGENT_TIMEOUT_SECONDS migrates
        to DEFAULT_TIMEOUT in the env dict; /etc/kai/env is written with
        the new name and no AGENT_TIMEOUT_SECONDS."""
        from unittest.mock import MagicMock

        monkeypatch.setattr("os.geteuid", lambda: 0)
        conf_path = self._write_install_conf(
            tmp_path,
            {"DEFAULT_MODEL": "sonnet", "AGENT_TIMEOUT_SECONDS": "180"},
        )
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        self._patch_side_effects(monkeypatch)
        secrets_mock = MagicMock()
        monkeypatch.setattr("kai.install._apply_secrets", secrets_mock)
        monkeypatch.setenv("DRY_RUN", "1")

        _cmd_apply()

        written_env = secrets_mock.call_args.args[0]
        assert written_env.get("DEFAULT_TIMEOUT") == "180"
        assert "AGENT_TIMEOUT_SECONDS" not in written_env

    def test_apply_DEFAULT_TIMEOUT_wins_over_legacy(self, tmp_path, monkeypatch):
        """Both DEFAULT_TIMEOUT and the legacy AGENT_TIMEOUT_SECONDS
        present: the new key wins and the legacy key is dropped."""
        from unittest.mock import MagicMock

        monkeypatch.setattr("os.geteuid", lambda: 0)
        conf_path = self._write_install_conf(
            tmp_path,
            {"DEFAULT_MODEL": "sonnet", "DEFAULT_TIMEOUT": "300", "AGENT_TIMEOUT_SECONDS": "180"},
        )
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        self._patch_side_effects(monkeypatch)
        secrets_mock = MagicMock()
        monkeypatch.setattr("kai.install._apply_secrets", secrets_mock)
        monkeypatch.setenv("DRY_RUN", "1")

        _cmd_apply()

        written_env = secrets_mock.call_args.args[0]
        assert written_env.get("DEFAULT_TIMEOUT") == "300"
        assert "AGENT_TIMEOUT_SECONDS" not in written_env


# ── Directory creation ───────────────────────────────────────────────


class TestApplyDirectories:
    """Tests for _apply_directories(), which creates the install layout."""

    @pytest.fixture(autouse=True)
    def _stub_os_calls(self, monkeypatch):
        """Stub os.chown/chmod and patch out /etc/kai (needs root on CI)."""
        monkeypatch.setattr("os.chown", lambda path, uid, gid: None)
        monkeypatch.setattr("os.chmod", lambda path, mode: None)
        # The dirs list includes hardcoded Path("/etc/kai") which cannot be
        # created without root. Patch Path.mkdir to silently skip /etc paths.
        original_mkdir = Path.mkdir

        def safe_mkdir(self, *args, **kwargs):
            if str(self).startswith("/etc"):
                return
            return original_mkdir(self, *args, **kwargs)

        monkeypatch.setattr(Path, "mkdir", safe_mkdir)

    def test_creates_workspace_base(self, tmp_path):
        """WORKSPACE_BASE is created when passed to _apply_directories."""
        install = tmp_path / "opt" / "kai"
        data = tmp_path / "var" / "lib" / "kai"
        ws_base = tmp_path / "home" / "kai" / "workspaces"

        _apply_directories(install, data, 503, 20, dry_run=False, workspace_base=ws_base)

        assert ws_base.exists()
        assert ws_base.is_dir()

    def test_skips_workspace_base_when_none(self, tmp_path):
        """No extra directory is created when workspace_base is None."""
        install = tmp_path / "opt" / "kai"
        data = tmp_path / "var" / "lib" / "kai"
        ws_base = tmp_path / "home" / "kai" / "workspaces"

        _apply_directories(install, data, 503, 20, dry_run=False, workspace_base=None)

        assert not ws_base.exists()

    def test_workspace_base_dry_run(self, tmp_path, capsys):
        """Dry run prints the workspace base without creating it."""
        install = tmp_path / "opt" / "kai"
        data = tmp_path / "var" / "lib" / "kai"
        ws_base = tmp_path / "home" / "kai" / "workspaces"

        _apply_directories(install, data, 503, 20, dry_run=True, workspace_base=ws_base)

        assert not ws_base.exists()
        output = capsys.readouterr().out
        assert str(ws_base) in output

    def test_workspace_base_already_exists(self, tmp_path):
        """Existing workspace base is left alone (no error)."""
        install = tmp_path / "opt" / "kai"
        data = tmp_path / "var" / "lib" / "kai"
        ws_base = tmp_path / "home" / "kai" / "workspaces"
        ws_base.mkdir(parents=True)

        # Should not raise
        _apply_directories(install, data, 503, 20, dry_run=False, workspace_base=ws_base)

        assert ws_base.exists()

    def test_user_data_roots_are_traversal_only(self, tmp_path, monkeypatch):
        """Per-user data roots are not listable by sibling OS users."""
        chmods: list[tuple[str, int]] = []
        monkeypatch.setattr("os.chmod", lambda path, mode: chmods.append((str(path), mode)))

        install = tmp_path / "opt" / "kai"
        data = tmp_path / "var" / "lib" / "kai"

        _apply_directories(install, data, 503, 20, dry_run=False, workspace_base=None)

        mode_by_path = dict(chmods)
        assert mode_by_path[str(data / "files")] == 0o711
        assert mode_by_path[str(data / "memory")] == 0o711
        assert mode_by_path[str(data / "preferences")] == 0o711
        assert mode_by_path[str(data / "home")] == 0o711


# ── Status subcommand ────────────────────────────────────────────────


class TestCheckTraversal:
    """Tests for _check_traversal(), which checks directory execute permissions."""

    def _mock_user(self, monkeypatch, uid=1001, gid=1001, groups=None):
        """Set up a fake service user for traversal checks."""
        import types

        user_info = types.SimpleNamespace(pw_uid=uid, pw_gid=gid, pw_dir="/home/testuser")
        monkeypatch.setattr("pwd.getpwnam", lambda name: user_info)
        monkeypatch.setattr("os.getgrouplist", lambda name, gid: groups or [gid])

    def test_fully_traversable(self, tmp_path, monkeypatch):
        """Returns None when all parents are traversable by the user."""
        # Use the real uid/gid so the check passes on all intermediate dirs
        uid = os.getuid()
        gid = os.getgid()
        self._mock_user(monkeypatch, uid=uid, gid=gid)
        target = tmp_path / "a" / "b" / "c"
        target.mkdir(parents=True)

        result = _check_traversal(target, "testuser")
        assert result is None

    def test_blocked_by_parent(self, tmp_path, monkeypatch):
        """Returns warning naming the directory that lacks execute permission."""
        # Use the real uid/gid so traversal passes system dirs, then block
        # on the directory we explicitly restrict.
        uid = os.getuid()
        gid = os.getgid()
        self._mock_user(monkeypatch, uid=uid, gid=gid)

        blocker = tmp_path / "restricted"
        target = blocker / "child"
        target.mkdir(parents=True)
        # Remove execute for owner (our uid owns these dirs)
        blocker.chmod(0o600)

        try:
            result = _check_traversal(target, "testuser")
            assert result is not None
            assert str(blocker) in result
            assert "chmod u+x" in result
        finally:
            # Restore so pytest can clean up tmp_path
            blocker.chmod(0o755)

    def test_owner_traverses_via_ux(self, tmp_path, monkeypatch):
        """Owner with u+x on a parent can traverse even without g+x or o+x."""
        uid = os.getuid()
        gid = os.getgid()
        self._mock_user(monkeypatch, uid=uid, gid=gid)

        parent = tmp_path / "owner_only"
        target = parent / "child"
        target.mkdir(parents=True)
        # Parent: owner has rwx, group/other have nothing
        parent.chmod(0o700)

        result = _check_traversal(target, "testuser")
        assert result is None

    def test_nonexistent_user(self, monkeypatch):
        """Returns warning when service user does not exist."""
        monkeypatch.setattr("pwd.getpwnam", lambda name: (_ for _ in ()).throw(KeyError(name)))
        result = _check_traversal(Path("/tmp"), "nobody99")
        assert result is not None
        assert "does not exist" in result


class TestCheckPath:
    def test_existing_path(self, tmp_path):
        result = _check_path(tmp_path, "Test")
        assert "exists" in result
        assert str(tmp_path) in result

    def test_missing_path(self, tmp_path):
        result = _check_path(tmp_path / "nope", "Test")
        assert "not found" in result


class TestCheckServiceStatus:
    def test_darwin_loaded(self, monkeypatch):
        """Reports 'loaded' when launchctl finds the service."""
        monkeypatch.setattr(
            "kai.install.subprocess.run",
            lambda *a, **kw: subprocess.CompletedProcess(args=[], returncode=0, stdout=""),
        )
        result = _check_service_status("darwin")
        assert "loaded" in result

    def test_darwin_not_loaded(self, monkeypatch):
        """Reports 'not loaded' when launchctl doesn't find the service."""
        monkeypatch.setattr(
            "kai.install.subprocess.run",
            lambda *a, **kw: subprocess.CompletedProcess(args=[], returncode=1, stdout=""),
        )
        result = _check_service_status("darwin")
        assert "not loaded" in result

    def test_linux_active(self, monkeypatch):
        """Reports status from systemctl on Linux."""
        monkeypatch.setattr(
            "kai.install.subprocess.run",
            lambda *a, **kw: subprocess.CompletedProcess(args=[], returncode=0, stdout="active\n"),
        )
        result = _check_service_status("linux")
        assert "active" in result


class TestCmdStatus:
    def test_runs_without_error(self, tmp_path, monkeypatch, capsys):
        """Status subcommand runs without crashing (no install present)."""
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        # Mock subprocess.run for service status check
        monkeypatch.setattr(
            "kai.install.subprocess.run",
            lambda *a, **kw: subprocess.CompletedProcess(args=[], returncode=1, stdout=""),
        )
        _cmd_status()
        output = capsys.readouterr().out
        assert "Installation Status" in output
        assert "Workshop bootstrap:" in output
        assert "Workshop delivery authority:" in output
        assert "Workshop message parity:" in output

    def test_reports_unsupported_webhook_secret(self, tmp_path, monkeypatch, capsys):
        conf_path = tmp_path / "install.conf"
        conf_path.write_text(
            json.dumps(
                {
                    "platform": "darwin",
                    "install_dir": str(tmp_path / "opt-kai"),
                    "data_dir": str(tmp_path / "var-lib-kai"),
                    "env": {
                        "WEBHOOK_SECRET": "legacy-secret",
                        "GITHUB_WEBHOOK_SECRET": "github-secret",
                        "GENERIC_WEBHOOK_SECRET": "generic-secret",
                    },
                }
            )
        )
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        monkeypatch.setattr(
            "kai.install.subprocess.run",
            lambda *a, **kw: subprocess.CompletedProcess(args=[], returncode=1, stdout=""),
        )

        _cmd_status()

        output = capsys.readouterr().out
        assert "unsupported WEBHOOK_SECRET present" in output
        assert "ignored by runtime" in output
        assert "legacy-secret" not in output
        assert "github-secret" not in output
        assert "generic-secret" not in output

    def test_reports_named_webhook_secrets_without_legacy_fallback(self, tmp_path, monkeypatch, capsys):
        conf_path = tmp_path / "install.conf"
        conf_path.write_text(
            json.dumps(
                {
                    "platform": "darwin",
                    "install_dir": str(tmp_path / "opt-kai"),
                    "data_dir": str(tmp_path / "var-lib-kai"),
                    "env": {
                        "GITHUB_WEBHOOK_SECRET": "github-secret",
                        "GENERIC_WEBHOOK_SECRET": "generic-secret",
                    },
                }
            )
        )
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        monkeypatch.setattr(
            "kai.install.subprocess.run",
            lambda *a, **kw: subprocess.CompletedProcess(args=[], returncode=1, stdout=""),
        )

        _cmd_status()

        output = capsys.readouterr().out
        assert "named GitHub/generic secrets configured" in output
        assert "unsupported WEBHOOK_SECRET absent" in output
        assert "github-secret" not in output
        assert "generic-secret" not in output

    def test_reports_deployed_state_separately_from_install_conf(self, tmp_path, monkeypatch, capsys):
        deployed_env = tmp_path / "deployed-env"
        deployed_env.write_text(
            'GITHUB_WEBHOOK_SECRET="deployed-github-secret"\nGENERIC_WEBHOOK_SECRET="deployed-generic-secret"\n'
        )
        conf_path = tmp_path / "install.conf"
        conf_path.write_text(
            json.dumps(
                {
                    "platform": "darwin",
                    "install_dir": str(tmp_path / "opt-kai"),
                    "data_dir": str(tmp_path / "var-lib-kai"),
                    "env": {"WEBHOOK_SECRET": "artifact-legacy-secret"},
                }
            )
        )
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        monkeypatch.setattr("kai.install._DEPLOYED_ENV_FILE", deployed_env)
        monkeypatch.setattr("kai.install.validate_protected_file_metadata", lambda *args, **kwargs: True)
        monkeypatch.setattr(
            "kai.install.subprocess.run",
            lambda *a, **kw: subprocess.CompletedProcess(args=[], returncode=1, stdout=""),
        )

        _cmd_status()

        output = capsys.readouterr().out
        assert f"Webhook secret migration (deployed {deployed_env}): named GitHub/generic secrets configured" in output
        assert "Webhook secret migration (install.conf artifact): unsupported WEBHOOK_SECRET present" in output
        assert "deployed-github-secret" not in output
        assert "deployed-generic-secret" not in output
        assert "artifact-legacy-secret" not in output

    def test_webhook_secret_migration_status_is_non_secret(self):
        assert "unsupported WEBHOOK_SECRET present" in _webhook_secret_migration_status(
            {
                "WEBHOOK_SECRET": "legacy-secret",
                "GITHUB_WEBHOOK_SECRET": "github-secret",
                "GENERIC_WEBHOOK_SECRET": "generic-secret",
            }
        )
        assert "legacy-secret" not in _webhook_secret_migration_status({"WEBHOOK_SECRET": "legacy-secret"})

    def test_deployed_status_reports_named_secrets_without_values(self, tmp_path, monkeypatch):
        env_path = tmp_path / "env"
        env_path.write_text(
            "# protected configuration\n"
            'GITHUB_WEBHOOK_SECRET="github-deployed-value"\n'
            'GENERIC_WEBHOOK_SECRET="generic-deployed-value"\n'
        )
        monkeypatch.setattr("kai.install.validate_protected_file_metadata", lambda *args, **kwargs: True)

        status = _deployed_webhook_secret_migration_status(env_path)

        assert f"deployed {env_path}" in status
        assert "named GitHub/generic secrets configured" in status
        assert "unsupported WEBHOOK_SECRET absent" in status
        assert "github-deployed-value" not in status
        assert "generic-deployed-value" not in status

    def test_deployed_status_reports_unsupported_secret_without_value(self, tmp_path, monkeypatch):
        env_path = tmp_path / "env"
        env_path.write_text('WEBHOOK_SECRET="legacy-deployed-value"\n')
        monkeypatch.setattr("kai.install.validate_protected_file_metadata", lambda *args, **kwargs: True)

        status = _deployed_webhook_secret_migration_status(env_path)

        assert "unsupported WEBHOOK_SECRET present" in status
        assert "ignored by runtime" in status
        assert "legacy-deployed-value" not in status

    def test_deployed_status_is_explicit_when_permission_denied(self, tmp_path, monkeypatch):
        env_path = tmp_path / "env"
        env_path.write_text('WEBHOOK_SECRET="must-not-appear"\n')
        monkeypatch.setattr("kai.install.validate_protected_file_metadata", lambda *args, **kwargs: True)
        real_open = Path.open

        def deny_deployed_env(self, *args, **kwargs):
            if self == env_path:
                raise PermissionError("denied")
            return real_open(self, *args, **kwargs)

        monkeypatch.setattr(Path, "open", deny_deployed_env)

        status = _deployed_webhook_secret_migration_status(env_path)

        assert "NOT VERIFIED" in status
        assert "permission denied" in status
        assert "make install-status" in status
        assert "must-not-appear" not in status

    def test_deployed_status_rejects_untrusted_metadata(self, tmp_path, monkeypatch):
        env_path = tmp_path / "env"
        env_path.write_text('WEBHOOK_SECRET="must-not-appear"\n')

        def reject_metadata(*args, **kwargs):
            raise kai.install.ProtectedConfigError("unsafe permissions")

        monkeypatch.setattr("kai.install.validate_protected_file_metadata", reject_metadata)

        status = _deployed_webhook_secret_migration_status(env_path)

        assert "NOT VERIFIED" in status
        assert "unsafe permissions" in status
        assert "must-not-appear" not in status


# ── Venv creation ────────────────────────────────────────────────────


class TestApplyVenv:
    """Tests for _apply_venv(), which creates the virtual environment."""

    def test_base_python_falls_back_to_running_interpreter(self, monkeypatch):
        """A sudo-reset PATH still uses the installer's valid Python 3.13."""
        monkeypatch.setattr(shutil, "which", lambda name: None)
        commands = []

        def fake_run(cmd, **kwargs):
            commands.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="3.13\n", stderr="")

        monkeypatch.setattr(subprocess, "run", fake_run)

        assert kai.install._resolve_venv_base_python() == kai.install.sys.executable
        assert commands[0][0] == kai.install.sys.executable

    def test_rejects_old_python(self, tmp_path, monkeypatch):
        """Exits with a clear error if the resolved Python is below 3.13."""
        install = tmp_path / "opt" / "kai"
        install.mkdir(parents=True)
        # Write a dummy pyproject.toml so the checksum logic has something
        (install / "pyproject.toml").write_text("[project]\nname = 'kai'\n")

        # Mock shutil.which to return a fake python path
        monkeypatch.setattr(shutil, "which", lambda name: "/usr/bin/python3")

        # Mock subprocess.run to return version "3.12" for the version check
        original_run = subprocess.run

        def fake_run(cmd, **kwargs):
            # Intercept the version-check command
            if isinstance(cmd, list) and "-c" in cmd:
                return subprocess.CompletedProcess(cmd, 0, stdout="3.12\n", stderr="")
            return original_run(cmd, **kwargs)

        monkeypatch.setattr(subprocess, "run", fake_run)

        with pytest.raises(SystemExit, match=r"Python >= 3\.13 required"):
            _apply_venv(install, is_update=False, dry_run=False)

    def test_skips_when_checksums_match(self, tmp_path, capsys):
        """Skips reinstall when both pyproject.toml and source checksums match."""
        install = tmp_path / "opt" / "kai"
        install.mkdir(parents=True)
        (install / "venv" / "bin").mkdir(parents=True)
        (install / "venv" / "bin" / "python").touch(mode=0o755)

        # Write pyproject.toml and source files
        (install / "pyproject.toml").write_text("[project]\nname = 'kai'\n")
        src = install / "src" / "kai"
        src.mkdir(parents=True)
        (src / "__init__.py").write_text("# init")
        (src / "main.py").write_text("print('hello')")

        # Pre-populate checksum files as if a previous install wrote them
        (install / ".pyproject.sha256").write_text(_file_checksum(install / "pyproject.toml") + "\n")
        (install / ".src.sha256").write_text(_src_checksum(install / "src") + "\n")

        _apply_venv(install, is_update=True, dry_run=False)

        output = capsys.readouterr().out
        assert "Venv unchanged" in output
        assert "checksums match" in output

    def test_matching_checksums_repair_restrictive_venv_modes(self, tmp_path, capsys):
        """The no-change fast path repairs modes inherited from umask 077."""
        install = tmp_path / "opt" / "kai"
        venv = install / "venv"
        venv_bin = venv / "bin"
        venv_bin.mkdir(parents=True)
        venv_python = venv_bin / "python"
        venv_python.touch(mode=0o755)
        package = venv / "lib" / "python3.13" / "site-packages" / "kai"
        package.mkdir(parents=True)
        module = package / "__init__.py"
        module.write_text("# init")

        (install / "pyproject.toml").write_text("[project]\nname = 'kai'\n")
        src = install / "src" / "kai"
        src.mkdir(parents=True)
        (src / "__init__.py").write_text("# init")
        (install / ".pyproject.sha256").write_text(_file_checksum(install / "pyproject.toml") + "\n")
        (install / ".src.sha256").write_text(_src_checksum(install / "src") + "\n")

        for directory in (venv, venv_bin, package.parent, package):
            directory.chmod(0o700)
        module.chmod(0o600)

        _apply_venv(install, is_update=True, dry_run=False)

        output = capsys.readouterr().out
        assert "Restored service-readable venv modes" in output
        assert "Venv unchanged" in output
        assert stat.S_IMODE(venv.stat().st_mode) == 0o755
        assert stat.S_IMODE(package.stat().st_mode) == 0o755
        assert stat.S_IMODE(module.stat().st_mode) == 0o644
        assert stat.S_IMODE(venv_python.stat().st_mode) == 0o755

    def test_reinstalls_on_source_change(self, tmp_path, monkeypatch, capsys):
        """Triggers reinstall when source files change but pyproject.toml does not."""
        install = tmp_path / "opt" / "kai"
        install.mkdir(parents=True)
        (install / "venv" / "bin").mkdir(parents=True)
        (install / "venv" / "bin" / "python").touch(mode=0o755)
        stale_build = install / "build" / "lib" / "kai"
        stale_build.mkdir(parents=True)
        (stale_build / "obsolete.py").write_text("# removed source module")

        # Write pyproject.toml and initial source
        (install / "pyproject.toml").write_text("[project]\nname = 'kai'\n")
        src = install / "src" / "kai"
        src.mkdir(parents=True)
        (src / "__init__.py").write_text("# init")

        # Save checksums for the initial state
        (install / ".pyproject.sha256").write_text(_file_checksum(install / "pyproject.toml") + "\n")
        (install / ".src.sha256").write_text(_src_checksum(install / "src") + "\n")

        # Modify a source file (simulates _apply_source copying new code)
        (src / "__init__.py").write_text("# init v2 - changed")

        # Mock subprocess.run so pip install doesn't actually run
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: subprocess.CompletedProcess(args=[], returncode=0),
        )
        # Mock _set_ownership so chown doesn't need root
        monkeypatch.setattr("kai.install._set_ownership", lambda *a, **kw: None)

        _apply_venv(install, is_update=True, dry_run=False)

        output = capsys.readouterr().out
        assert "source changed" in output
        assert "Removed stale package build artifacts" in output
        assert not (install / "build").exists()
        assert "Installed package into venv" in output

    def test_reinstalls_on_packaged_static_asset_change(self, tmp_path, monkeypatch, capsys):
        """A Workshop bundle change refreshes its copied site-packages resource."""
        install = tmp_path / "opt" / "kai"
        (install / "venv" / "bin").mkdir(parents=True)
        (install / "venv" / "bin" / "python").touch(mode=0o755)
        (install / "pyproject.toml").write_text("[project]\nname = 'kai'\n")
        static = install / "src" / "kai" / "workshop" / "static"
        static.mkdir(parents=True)
        bundle = static / "app.js"
        bundle.write_text("const version = 1;")
        (install / ".pyproject.sha256").write_text(_file_checksum(install / "pyproject.toml") + "\n")
        (install / ".src.sha256").write_text(_src_checksum(install / "src") + "\n")

        bundle.write_text("const version = 2;")
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: subprocess.CompletedProcess(args=[], returncode=0),
        )
        monkeypatch.setattr("kai.install._set_ownership", lambda *a, **kw: None)

        _apply_venv(install, is_update=True, dry_run=False)

        output = capsys.readouterr().out
        assert "source changed" in output
        assert "Installed package into venv" in output

    def test_normalizes_source_metadata_created_by_pip(self, tmp_path, monkeypatch):
        """Packaging metadata created after the source copy cannot retain 0700."""
        install = tmp_path / "opt" / "kai"
        (install / "venv" / "bin").mkdir(parents=True)
        (install / "venv" / "bin" / "python").touch(mode=0o755)
        (install / "pyproject.toml").write_text("[project]\nname = 'kai'\n")
        src = install / "src" / "kai"
        src.mkdir(parents=True)
        (src / "__init__.py").write_text("# init")

        metadata = install / "src" / "kai.egg-info"
        metadata_file = metadata / "PKG-INFO"

        def fake_run(*args, **kwargs):
            metadata.mkdir(mode=0o700)
            metadata_file.write_text("Metadata-Version: 2.4\n")
            metadata_file.chmod(0o600)
            return subprocess.CompletedProcess(args=[], returncode=0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr("kai.install._set_ownership", lambda *a, **kw: None)

        _apply_venv(install, is_update=True, dry_run=False)

        assert stat.S_IMODE(metadata.stat().st_mode) == 0o755
        assert stat.S_IMODE(metadata_file.stat().st_mode) == 0o644

    def test_reinstalls_on_source_change_dry_run(self, tmp_path, monkeypatch, capsys):
        """Dry run compares incoming source even though the copy is skipped."""
        install = tmp_path / "opt" / "kai"
        install.mkdir(parents=True)
        (install / "venv" / "bin").mkdir(parents=True)
        (install / "venv" / "bin" / "python").touch(mode=0o755)
        stale_build = install / "build" / "lib" / "kai"
        stale_build.mkdir(parents=True)
        (stale_build / "obsolete.py").write_text("# removed source module")

        (install / "pyproject.toml").write_text("[project]\nname = 'kai'\n")
        src = install / "src" / "kai"
        src.mkdir(parents=True)
        (src / "bot.py").write_text("# bot v1")

        # Save checksums for the currently installed source.
        (install / ".pyproject.sha256").write_text(_file_checksum(install / "pyproject.toml") + "\n")
        (install / ".src.sha256").write_text(_src_checksum(install / "src") + "\n")

        # The incoming repository has changed, but dry-run must not copy it to
        # the install tree merely to compute the preview.
        project = tmp_path / "project"
        (project / "src" / "kai").mkdir(parents=True)
        (project / "pyproject.toml").write_text("[project]\nname = 'kai'\n")
        (project / "src" / "kai" / "bot.py").write_text("# bot v2 - new feature")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", project)

        _apply_venv(install, is_update=True, dry_run=True)

        output = capsys.readouterr().out
        assert "DRY RUN" in output
        assert "source changed" in output
        assert "Would remove stale package build artifacts" in output
        assert (install / "build" / "lib" / "kai" / "obsolete.py").exists()
        assert (install / "src" / "kai" / "bot.py").read_text() == "# bot v1"

    def test_saves_both_checksums_after_install(self, tmp_path, monkeypatch):
        """Both .pyproject.sha256 and .src.sha256 are written after a successful install."""
        install = tmp_path / "opt" / "kai"
        install.mkdir(parents=True)
        (install / "venv" / "bin").mkdir(parents=True)
        (install / "venv" / "bin" / "python").touch(mode=0o755)

        (install / "pyproject.toml").write_text("[project]\nname = 'kai'\n")
        src = install / "src" / "kai"
        src.mkdir(parents=True)
        (src / "__init__.py").write_text("# init")

        # Fresh update with no previous checksums - should trigger reinstall
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: subprocess.CompletedProcess(args=[], returncode=0),
        )
        monkeypatch.setattr("kai.install._set_ownership", lambda *a, **kw: None)

        _apply_venv(install, is_update=True, dry_run=False)

        # Both checksum files should exist now
        assert (install / ".pyproject.sha256").exists()
        assert (install / ".constraints.sha256").exists()
        assert (install / ".src.sha256").exists()

        # And they should contain the correct checksums
        assert (install / ".pyproject.sha256").read_text().strip() == _file_checksum(install / "pyproject.toml")
        assert (install / ".constraints.sha256").read_text().strip() == ""
        assert (install / ".src.sha256").read_text().strip() == _src_checksum(install / "src")

    def test_reinstalls_on_constraints_change(self, tmp_path, monkeypatch, capsys):
        """Constraint file changes trigger a venv reinstall even when source is unchanged."""
        install = tmp_path / "opt" / "kai"
        install.mkdir(parents=True)
        (install / "venv" / "bin").mkdir(parents=True)
        (install / "venv" / "bin" / "python").touch(mode=0o755)

        (install / "pyproject.toml").write_text("[project]\nname = 'kai'\n")
        constraints = install / "requirements" / "constraints.txt"
        constraints.parent.mkdir(parents=True)
        constraints.write_text("aiohttp==3.13.4\n")
        src = install / "src" / "kai"
        src.mkdir(parents=True)
        (src / "__init__.py").write_text("# init")

        (install / ".pyproject.sha256").write_text(_file_checksum(install / "pyproject.toml") + "\n")
        (install / ".constraints.sha256").write_text(_file_checksum(constraints) + "\n")
        (install / ".src.sha256").write_text(_src_checksum(install / "src") + "\n")

        constraints.write_text("aiohttp==3.14.0\n")

        commands = []

        def fake_run(cmd, **kwargs):
            commands.append(cmd)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr("kai.install._set_ownership", lambda *a, **kw: None)

        _apply_venv(install, is_update=True, dry_run=False)

        output = capsys.readouterr().out
        assert "constraints changed" in output
        assert [
            str(install / "venv" / "bin" / "python"),
            "-m",
            "pip",
            "install",
            "--constraint",
            str(constraints),
            f"{install}[memory,totp,tts]",
        ] in commands
        assert (install / ".constraints.sha256").read_text().strip() == _file_checksum(constraints)

    def test_dry_run_reports_constraints_change(self, tmp_path, monkeypatch, capsys):
        """Dry-run compares incoming constraints without copying them."""
        install = tmp_path / "opt" / "kai"
        install.mkdir(parents=True)
        (install / "venv" / "bin").mkdir(parents=True)
        (install / "venv" / "bin" / "python").touch(mode=0o755)

        (install / "pyproject.toml").write_text("[project]\nname = 'kai'\n")
        src = install / "src" / "kai"
        src.mkdir(parents=True)
        (src / "__init__.py").write_text("# init")

        (install / ".pyproject.sha256").write_text(_file_checksum(install / "pyproject.toml") + "\n")
        (install / ".src.sha256").write_text(_src_checksum(install / "src") + "\n")

        project = tmp_path / "project"
        (project / "src" / "kai").mkdir(parents=True)
        (project / "pyproject.toml").write_text("[project]\nname = 'kai'\n")
        (project / "src" / "kai" / "__init__.py").write_text("# init")
        incoming_constraints = project / "requirements" / "constraints.txt"
        incoming_constraints.parent.mkdir(parents=True)
        incoming_constraints.write_text("aiohttp==3.13.4\n")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", project)

        _apply_venv(install, is_update=True, dry_run=True)

        output = capsys.readouterr().out
        assert "DRY RUN" in output
        assert "constraints changed" in output
        assert not (install / "requirements" / "constraints.txt").exists()

    def test_first_update_without_src_checksum(self, tmp_path, monkeypatch, capsys):
        """First update after this fix triggers reinstall (no .src.sha256 from old install)."""
        install = tmp_path / "opt" / "kai"
        install.mkdir(parents=True)
        (install / "venv" / "bin").mkdir(parents=True)
        (install / "venv" / "bin" / "python").touch(mode=0o755)

        (install / "pyproject.toml").write_text("[project]\nname = 'kai'\n")
        src = install / "src" / "kai"
        src.mkdir(parents=True)
        (src / "__init__.py").write_text("# init")

        # Only the old-style pyproject checksum exists (no .src.sha256)
        (install / ".pyproject.sha256").write_text(_file_checksum(install / "pyproject.toml") + "\n")

        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *a, **kw: subprocess.CompletedProcess(args=[], returncode=0),
        )
        monkeypatch.setattr("kai.install._set_ownership", lambda *a, **kw: None)

        _apply_venv(install, is_update=True, dry_run=False)

        output = capsys.readouterr().out
        # Should reinstall because .src.sha256 is missing (old_src == "" != new_src)
        assert "source changed" in output
        assert "Installed package into venv" in output

    def test_repairs_dangling_venv_python_before_install(self, tmp_path, monkeypatch, capsys):
        """A Homebrew upgrade leaving a dangling venv symlink is repaired in place."""
        install = tmp_path / "opt" / "kai"
        venv_bin = install / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        venv_python = venv_bin / "python"
        venv_python.symlink_to("python3.13")
        (venv_bin / "python3").symlink_to("python3.13")
        (venv_bin / "python3.13").symlink_to("/removed/homebrew/python3.13")

        (install / "pyproject.toml").write_text("[project]\nname = 'kai'\n")
        src = install / "src" / "kai"
        src.mkdir(parents=True)
        (src / "__init__.py").write_text("# init")
        (install / ".pyproject.sha256").write_text(_file_checksum(install / "pyproject.toml") + "\n")
        (install / ".src.sha256").write_text(_src_checksum(install / "src") + "\n")

        base_python = "/opt/homebrew/bin/python3.13"
        monkeypatch.setattr(shutil, "which", lambda name: base_python)
        commands = []

        def fake_run(cmd, **kwargs):
            commands.append(cmd)
            if "-c" in cmd:
                return subprocess.CompletedProcess(cmd, 0, stdout="3.13\n", stderr="")
            if cmd[:4] == [base_python, "-m", "venv", "--upgrade"]:
                venv_python.touch(mode=0o755)
            return subprocess.CompletedProcess(cmd, 0)

        monkeypatch.setattr(subprocess, "run", fake_run)
        monkeypatch.setattr("kai.install._set_ownership", lambda *a, **kw: None)

        _apply_venv(install, is_update=True, dry_run=False)

        output = capsys.readouterr().out
        assert "Repairing venv" in output
        assert "Removed 3 dangling venv interpreter symlink" in output
        assert "Repaired venv interpreter" in output
        assert [base_python, "-m", "venv", "--upgrade", str(install / "venv")] in commands
        assert [str(venv_python), "-m", "pip", "install", f"{install}[memory,totp,tts]"] in commands

    def test_dry_run_reports_dangling_venv_python_without_repair(self, tmp_path, monkeypatch, capsys):
        """Dry-run detects a broken venv but never invokes a repair command."""
        install = tmp_path / "opt" / "kai"
        venv_bin = install / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        (venv_bin / "python").symlink_to("/removed/homebrew/python3.13")

        (install / "pyproject.toml").write_text("[project]\nname = 'kai'\n")
        src = install / "src" / "kai"
        src.mkdir(parents=True)
        (src / "__init__.py").write_text("# init")
        (install / ".pyproject.sha256").write_text(_file_checksum(install / "pyproject.toml") + "\n")
        (install / ".src.sha256").write_text(_src_checksum(install / "src") + "\n")

        def fail_run(*args, **kwargs):
            raise AssertionError("dry-run must not invoke subprocess.run")

        monkeypatch.setattr(subprocess, "run", fail_run)

        _apply_venv(install, is_update=True, dry_run=True)

        output = capsys.readouterr().out
        assert "[DRY RUN] Would repair venv" in output


class TestSrcChecksum:
    """Tests for _src_checksum(), the directory content hasher."""

    def test_empty_dir(self, tmp_path):
        """Empty source directory returns a hash of zero inputs."""
        d = tmp_path / "src"
        d.mkdir()
        result = _src_checksum(d)
        # A hash of nothing is still a valid hex digest
        assert len(result) == 64

    def test_missing_dir(self, tmp_path):
        """Non-existent directory returns empty string."""
        assert _src_checksum(tmp_path / "nonexistent") == ""

    def test_deterministic(self, tmp_path):
        """Same files produce the same hash across calls."""
        d = tmp_path / "src"
        d.mkdir()
        (d / "a.py").write_text("hello")
        (d / "b.py").write_text("world")

        assert _src_checksum(d) == _src_checksum(d)

    def test_content_change_changes_hash(self, tmp_path):
        """Modifying a file changes the hash."""
        d = tmp_path / "src"
        d.mkdir()
        (d / "a.py").write_text("version 1")
        h1 = _src_checksum(d)

        (d / "a.py").write_text("version 2")
        h2 = _src_checksum(d)

        assert h1 != h2

    def test_new_file_changes_hash(self, tmp_path):
        """Adding a new .py file changes the hash."""
        d = tmp_path / "src"
        d.mkdir()
        (d / "a.py").write_text("hello")
        h1 = _src_checksum(d)

        (d / "b.py").write_text("world")
        h2 = _src_checksum(d)

        assert h1 != h2

    def test_package_data_change_changes_hash(self, tmp_path):
        """Changing packaged static data invalidates the non-editable venv."""
        d = tmp_path / "src"
        d.mkdir()
        (d / "a.py").write_text("hello")
        h1 = _src_checksum(d)

        static = d / "kai" / "workshop" / "static"
        static.mkdir(parents=True)
        (static / "app.js").write_text("console.log('new client')")
        h2 = _src_checksum(d)

        assert h1 != h2

    def test_ignores_generated_source_artifacts(self, tmp_path):
        """Bytecode and packaging metadata cannot cause reinstall loops."""
        d = tmp_path / "src"
        package = d / "kai"
        package.mkdir(parents=True)
        (package / "a.py").write_text("hello")
        h1 = _src_checksum(d)

        cache = package / "__pycache__"
        cache.mkdir()
        (cache / "a.cpython-313.pyc").write_bytes(b"bytecode")
        metadata = d / "kai.egg-info"
        metadata.mkdir()
        (metadata / "PKG-INFO").write_text("generated metadata")

        assert _src_checksum(d) == h1

    def test_rename_changes_hash(self, tmp_path):
        """Renaming a file changes the hash (path is included in the digest)."""
        d = tmp_path / "src"
        d.mkdir()
        (d / "old_name.py").write_text("content")
        h1 = _src_checksum(d)

        (d / "old_name.py").rename(d / "new_name.py")
        h2 = _src_checksum(d)

        assert h1 != h2

    def test_nested_files(self, tmp_path):
        """Recurses into subdirectories."""
        d = tmp_path / "src"
        sub = d / "kai" / "sub"
        sub.mkdir(parents=True)
        (sub / "module.py").write_text("nested")
        h1 = _src_checksum(d)

        # Hash should be non-empty and a valid hex digest
        assert len(h1) == 64

        # Changing the nested file should change the hash
        (sub / "module.py").write_text("nested v2")
        h2 = _src_checksum(d)
        assert h1 != h2


# ── Migration ────────────────────────────────────────────────────────


class TestSecureCodexTurnImageStaging:
    def test_removes_only_legacy_files_and_makes_root_non_listable(self, tmp_path, capsys):
        staging_root = tmp_path / "data" / "files" / "codex_turn_images"
        staging_root.mkdir(parents=True, mode=0o755)
        staging_root.chmod(0o755)
        legacy = staging_root / "turn-image-old.jpeg"
        legacy.write_bytes(b"private image")
        legacy.chmod(0o644)
        unrelated = staging_root / "operator-note.txt"
        unrelated.write_text("keep")
        principal_dir = staging_root / "12345"
        principal_dir.mkdir()

        _secure_codex_turn_image_staging(tmp_path / "data", dry_run=False)

        assert not legacy.exists()
        assert unrelated.read_text() == "keep"
        assert principal_dir.is_dir()
        assert staging_root.stat().st_mode & 0o777 == 0o711
        output = capsys.readouterr().out
        assert "Removed 1 legacy Codex turn image file" in output
        assert "mode 0711" in output

    def test_dry_run_reports_without_mutating(self, tmp_path, capsys):
        staging_root = tmp_path / "data" / "files" / "codex_turn_images"
        staging_root.mkdir(parents=True, mode=0o755)
        staging_root.chmod(0o755)
        legacy = staging_root / "turn-image-crash.png"
        legacy.write_bytes(b"private image")

        _secure_codex_turn_image_staging(tmp_path / "data", dry_run=True)

        assert legacy.exists()
        assert staging_root.stat().st_mode & 0o777 == 0o755
        output = capsys.readouterr().out
        assert "[DRY RUN] Would remove 1 legacy Codex turn image file" in output
        assert "[DRY RUN] Would set mode 0711" in output

    def test_refuses_symlink_staging_root(self, tmp_path):
        data_path = tmp_path / "data"
        files_dir = data_path / "files"
        files_dir.mkdir(parents=True)
        target = tmp_path / "attacker-controlled"
        target.mkdir()
        (files_dir / "codex_turn_images").symlink_to(target, target_is_directory=True)

        with pytest.raises(RuntimeError, match="Refusing unsafe Codex image staging path"):
            _secure_codex_turn_image_staging(data_path, dry_run=False)


class TestSecureUploadDirectories:
    def test_repairs_managed_directories_and_secures_existing_files(self, tmp_path, monkeypatch, capsys):
        files_root = tmp_path / "data" / "files"
        user_dir = files_root / "12345"
        user_dir.mkdir(parents=True)
        files_root.chmod(0o755)
        user_dir.chmod(0o775)
        historical = user_dir / "photo.jpg"
        historical.write_bytes(b"keep")
        historical.chmod(0o644)
        chowned: list[tuple[Path, int, int]] = []
        monkeypatch.setattr(
            "kai.install.os.chown",
            lambda path, uid, gid: chowned.append((Path(path), uid, gid)),
        )

        _secure_upload_directories(
            tmp_path / "data",
            {"12345"},
            svc_uid=9876,
            svc_gid=9877,
            dry_run=False,
        )

        assert chowned == [
            (files_root, 9876, 9877),
            (user_dir, 9876, 9877),
            (historical, 9876, 9877),
        ]
        assert stat.S_IMODE(files_root.stat().st_mode) == 0o711
        assert stat.S_IMODE(user_dir.stat().st_mode) == 0o711
        assert historical.read_bytes() == b"keep"
        assert stat.S_IMODE(historical.stat().st_mode) == 0o600
        assert "Secured upload directory" in capsys.readouterr().out

    def test_dry_run_reports_repairs_without_mutating(self, tmp_path, monkeypatch, capsys):
        user_dir = tmp_path / "data" / "files" / "12345"
        user_dir.mkdir(parents=True)
        files_root = user_dir.parent
        files_root.chmod(0o755)
        user_dir.chmod(0o775)
        chown = MagicMock()
        monkeypatch.setattr("kai.install.os.chown", chown)

        _secure_upload_directories(
            tmp_path / "data",
            {"12345"},
            svc_uid=9876,
            svc_gid=9877,
            dry_run=True,
        )

        assert stat.S_IMODE(files_root.stat().st_mode) == 0o755
        assert stat.S_IMODE(user_dir.stat().st_mode) == 0o775
        chown.assert_not_called()
        output = capsys.readouterr().out
        assert f"[DRY RUN] Would secure upload directory: {files_root}" in output
        assert f"[DRY RUN] Would secure upload directory: {user_dir}" in output

    def test_replaces_file_acl_with_only_configured_reader(self, tmp_path, monkeypatch):
        user_dir = tmp_path / "data" / "files" / "12345"
        user_dir.mkdir(parents=True)
        historical = user_dir / "photo.jpg"
        historical.write_bytes(b"keep")
        monkeypatch.setattr("kai.install.os.chown", MagicMock())
        replace = MagicMock()
        monkeypatch.setattr("kai.install.replace_named_read_access", replace)

        _secure_upload_directories(
            tmp_path / "data",
            {"12345"},
            svc_uid=9876,
            svc_gid=9877,
            dry_run=False,
            reader_users={"12345": "daniel"},
        )

        replace.assert_any_call(historical, "daniel", directory=False)

    def test_refuses_file_symlink_before_any_repair(self, tmp_path, monkeypatch):
        user_dir = tmp_path / "data" / "files" / "12345"
        user_dir.mkdir(parents=True)
        target = tmp_path / "attacker-controlled"
        target.write_text("unchanged")
        (user_dir / "photo.jpg").symlink_to(target)
        chown = MagicMock()
        monkeypatch.setattr("kai.install.os.chown", chown)

        with pytest.raises(RuntimeError, match="Refusing unsafe upload path"):
            _secure_upload_directories(
                tmp_path / "data",
                {"12345"},
                svc_uid=9876,
                svc_gid=9877,
                dry_run=False,
                reader_users={"12345": "daniel"},
            )

        assert target.read_text() == "unchanged"
        chown.assert_not_called()

    def test_refuses_managed_symlink_before_any_repair(self, tmp_path, monkeypatch):
        files_root = tmp_path / "data" / "files"
        files_root.mkdir(parents=True)
        files_root.chmod(0o755)
        target = tmp_path / "attacker-controlled"
        target.mkdir()
        (files_root / "12345").symlink_to(target, target_is_directory=True)
        chown = MagicMock()
        monkeypatch.setattr("kai.install.os.chown", chown)

        with pytest.raises(RuntimeError, match="Refusing unsafe upload path"):
            _secure_upload_directories(
                tmp_path / "data",
                {"12345"},
                svc_uid=9876,
                svc_gid=9877,
                dry_run=False,
            )

        assert stat.S_IMODE(files_root.stat().st_mode) == 0o755
        chown.assert_not_called()

    def test_leaves_unknown_directories_untouched(self, tmp_path, monkeypatch):
        files_root = tmp_path / "data" / "files"
        unknown = files_root / "not-configured"
        unknown.mkdir(parents=True)
        files_root.chmod(0o711)
        unknown.chmod(0o775)
        chowned: list[Path] = []
        monkeypatch.setattr(
            "kai.install.os.chown",
            lambda path, _uid, _gid: chowned.append(Path(path)),
        )
        current = files_root.stat()

        _secure_upload_directories(
            tmp_path / "data",
            set(),
            svc_uid=current.st_uid,
            svc_gid=current.st_gid,
            dry_run=False,
        )

        assert chowned == []
        assert stat.S_IMODE(unknown.stat().st_mode) == 0o775


class TestSecureHistoryDirectories:
    def test_secures_configured_and_unknown_history(self, tmp_path, monkeypatch):
        history_root = tmp_path / "data" / "history"
        configured = history_root / "12345"
        unknown = history_root / "-100999"
        configured.mkdir(parents=True)
        unknown.mkdir()
        configured_file = configured / "2026-08-11.jsonl"
        unknown_file = unknown / "2026-08-11.jsonl"
        legacy_file = history_root / "2025-01-01.jsonl"
        for path in (configured_file, unknown_file, legacy_file):
            path.write_text("{}\n")
            path.chmod(0o644)
        history_root.chmod(0o755)
        configured.chmod(0o755)
        unknown.chmod(0o755)
        monkeypatch.setattr("kai.install.os.chown", MagicMock())
        replace = MagicMock()
        monkeypatch.setattr("kai.install.replace_named_read_access", replace)

        _secure_history_directories(
            tmp_path / "data",
            {"12345": "daniel"},
            svc_uid=9876,
            svc_gid=9877,
            dry_run=False,
        )

        assert stat.S_IMODE(history_root.stat().st_mode) == 0o711
        assert stat.S_IMODE(configured.stat().st_mode) == 0o700
        assert stat.S_IMODE(unknown.stat().st_mode) == 0o700
        assert stat.S_IMODE(configured_file.stat().st_mode) == 0o600
        assert stat.S_IMODE(unknown_file.stat().st_mode) == 0o600
        assert stat.S_IMODE(legacy_file.stat().st_mode) == 0o600
        replace.assert_any_call(configured, "daniel", directory=True)
        replace.assert_any_call(configured_file, "daniel", directory=False)
        replace.assert_any_call(unknown, None, directory=True)
        replace.assert_any_call(unknown_file, None, directory=False)
        replace.assert_any_call(legacy_file, None, directory=False)

    def test_dry_run_reports_without_mutation(self, tmp_path, monkeypatch, capsys):
        user_dir = tmp_path / "data" / "history" / "12345"
        user_dir.mkdir(parents=True)
        history_root = user_dir.parent
        transcript = user_dir / "2026-08-11.jsonl"
        transcript.write_text("{}\n")
        history_root.chmod(0o755)
        user_dir.chmod(0o755)
        transcript.chmod(0o644)
        chown = MagicMock()
        replace = MagicMock()
        monkeypatch.setattr("kai.install.os.chown", chown)
        monkeypatch.setattr("kai.install.replace_named_read_access", replace)

        _secure_history_directories(
            tmp_path / "data",
            {"12345": "daniel"},
            svc_uid=9876,
            svc_gid=9877,
            dry_run=True,
        )

        assert stat.S_IMODE(history_root.stat().st_mode) == 0o755
        assert stat.S_IMODE(user_dir.stat().st_mode) == 0o755
        assert stat.S_IMODE(transcript.stat().st_mode) == 0o644
        chown.assert_not_called()
        replace.assert_not_called()
        assert "[DRY RUN] Would secure history tree" in capsys.readouterr().out

    def test_refuses_symlink_before_any_mutation(self, tmp_path, monkeypatch):
        user_dir = tmp_path / "data" / "history" / "12345"
        user_dir.mkdir(parents=True)
        target = tmp_path / "attacker-controlled"
        target.write_text("unchanged")
        (user_dir / "2026-08-11.jsonl").symlink_to(target)
        chown = MagicMock()
        monkeypatch.setattr("kai.install.os.chown", chown)

        with pytest.raises(RuntimeError, match="Refusing unsafe history path"):
            _secure_history_directories(
                tmp_path / "data",
                {"12345": "daniel"},
                svc_uid=9876,
                svc_gid=9877,
                dry_run=False,
            )

        assert target.read_text() == "unchanged"
        chown.assert_not_called()


class TestApplyMigrate:
    def test_copies_database(self, tmp_path, monkeypatch):
        """Copies kai.db from PROJECT_ROOT to data_path when destination doesn't exist."""
        # Set up source database
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path / "src")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "kai.db").write_text("fake-db-content")

        data_path = tmp_path / "data"
        data_path.mkdir()
        (data_path / "logs").mkdir()

        # Mock subprocess (sqlite3 integrity check) and os.chown
        monkeypatch.setattr(
            "kai.install.subprocess.run",
            lambda *a, **kw: subprocess.CompletedProcess(args=[], returncode=0, stdout="ok\n"),
        )
        monkeypatch.setattr("kai.install.os.chown", lambda *a: None)

        _apply_migrate(data_path, tmp_path / "install", svc_uid=501, svc_gid=20, dry_run=False)

        assert (data_path / "kai.db").exists()
        assert (data_path / "kai.db").read_text() == "fake-db-content"
        assert stat.S_IMODE((data_path / "kai.db").stat().st_mode) == 0o600

    def test_verifies_integrity(self, tmp_path, monkeypatch):
        """Runs PRAGMA integrity_check on the copied database."""
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path / "src")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "kai.db").write_text("fake-db")

        data_path = tmp_path / "data"
        data_path.mkdir()
        (data_path / "logs").mkdir()

        # Capture the subprocess call to verify the integrity check command
        calls: list[list[str]] = []

        def mock_run(*args, **kwargs):
            if args:
                calls.append(list(args[0]))
            return subprocess.CompletedProcess(args=[], returncode=0, stdout="ok\n")

        monkeypatch.setattr("kai.install.subprocess.run", mock_run)
        monkeypatch.setattr("kai.install.os.chown", lambda *a: None)

        _apply_migrate(data_path, tmp_path / "install", svc_uid=501, svc_gid=20, dry_run=False)

        # Find the sqlite3 call
        sqlite_calls = [c for c in calls if "sqlite3" in c[0]]
        assert len(sqlite_calls) == 1
        assert "PRAGMA integrity_check;" in sqlite_calls[0][2]

    def test_skips_if_target_exists(self, tmp_path, monkeypatch, capsys):
        """Does not overwrite an existing database at the destination."""
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path / "src")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "kai.db").write_text("source-content")

        data_path = tmp_path / "data"
        data_path.mkdir()
        (data_path / "logs").mkdir()
        (data_path / "kai.db").write_text("existing-content")
        (data_path / "kai.db").chmod(0o644)
        chowned: list[tuple[str, int, int]] = []
        monkeypatch.setattr("kai.install.os.chown", lambda path, uid, gid: chowned.append((str(path), uid, gid)))

        _apply_migrate(data_path, tmp_path / "install", svc_uid=501, svc_gid=20, dry_run=False)

        # Destination should be unchanged
        assert (data_path / "kai.db").read_text() == "existing-content"
        assert stat.S_IMODE((data_path / "kai.db").stat().st_mode) == 0o600
        assert (str(data_path / "kai.db"), 501, 20) in chowned
        assert "already exists" in capsys.readouterr().out

    def test_copies_logs(self, tmp_path, monkeypatch):
        """Copies log files from PROJECT_ROOT/logs to data_path/logs."""
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path / "src")
        (tmp_path / "src").mkdir()
        logs_src = tmp_path / "src" / "logs"
        logs_src.mkdir()
        (logs_src / "kai.log").write_text("log1")
        (logs_src / "kai.log.1").write_text("log2")

        data_path = tmp_path / "data"
        data_path.mkdir()
        logs_dst = data_path / "logs"
        logs_dst.mkdir()

        # Mock os.chown for ownership setting
        monkeypatch.setattr("kai.install.os.chown", lambda *a: None)

        _apply_migrate(data_path, tmp_path / "install", svc_uid=501, svc_gid=20, dry_run=False)

        assert (logs_dst / "kai.log").read_text() == "log1"
        assert (logs_dst / "kai.log.1").read_text() == "log2"

    def test_preserves_original(self, tmp_path, monkeypatch):
        """Source files are never deleted during migration."""
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path / "src")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "kai.db").write_text("original")
        logs_src = tmp_path / "src" / "logs"
        logs_src.mkdir()
        (logs_src / "kai.log").write_text("original-log")

        data_path = tmp_path / "data"
        data_path.mkdir()
        (data_path / "logs").mkdir()

        monkeypatch.setattr(
            "kai.install.subprocess.run",
            lambda *a, **kw: subprocess.CompletedProcess(args=[], returncode=0, stdout="ok\n"),
        )
        monkeypatch.setattr("kai.install.os.chown", lambda *a: None)

        _apply_migrate(data_path, tmp_path / "install", svc_uid=501, svc_gid=20, dry_run=False)

        # Source files must still exist
        assert (tmp_path / "src" / "kai.db").exists()
        assert (logs_src / "kai.log").exists()

    def test_dry_run(self, tmp_path, monkeypatch, capsys):
        """Dry run prints actions without copying anything."""
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path / "src")
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "kai.db").write_text("fake-db")
        logs_src = tmp_path / "src" / "logs"
        logs_src.mkdir()
        (logs_src / "kai.log").write_text("log-content")

        data_path = tmp_path / "data"
        data_path.mkdir()
        (data_path / "logs").mkdir()

        _apply_migrate(data_path, tmp_path / "install", svc_uid=501, svc_gid=20, dry_run=True)

        output = capsys.readouterr().out
        assert "[DRY RUN]" in output
        # Nothing should have been copied
        assert not (data_path / "kai.db").exists()
        assert not (data_path / "logs" / "kai.log").exists()

    def test_copies_uploaded_files(self, tmp_path, monkeypatch):
        """Copies uploaded files from home/files/ to data_path/files/."""
        install_path = tmp_path / "install"
        files_src = install_path / "home" / "files" / "123"
        files_src.mkdir(parents=True)
        (files_src / "photo.jpg").write_bytes(b"image data")
        (files_src / "doc.pdf").write_bytes(b"pdf data")

        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path / "src")
        (tmp_path / "src").mkdir()

        data_path = tmp_path / "data"
        data_path.mkdir()
        (data_path / "logs").mkdir()
        (data_path / "files").mkdir()

        monkeypatch.setattr("kai.install.os.chown", lambda *a: None)

        _apply_migrate(data_path, install_path, svc_uid=501, svc_gid=20, dry_run=False)

        assert (data_path / "files" / "123" / "photo.jpg").read_bytes() == b"image data"
        assert (data_path / "files" / "123" / "doc.pdf").read_bytes() == b"pdf data"
        # Source files preserved
        assert (files_src / "photo.jpg").exists()

    def test_uploaded_files_skip_existing(self, tmp_path, monkeypatch):
        """Does not overwrite uploaded files that already exist at the destination."""
        install_path = tmp_path / "install"
        files_src = install_path / "home" / "files"
        files_src.mkdir(parents=True)
        (files_src / "photo.jpg").write_bytes(b"source")

        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path / "src")
        (tmp_path / "src").mkdir()

        data_path = tmp_path / "data"
        data_path.mkdir()
        (data_path / "logs").mkdir()
        files_dst = data_path / "files"
        files_dst.mkdir()
        (files_dst / "photo.jpg").write_bytes(b"existing")

        monkeypatch.setattr("kai.install.os.chown", lambda *a: None)

        _apply_migrate(data_path, install_path, svc_uid=501, svc_gid=20, dry_run=False)

        assert (files_dst / "photo.jpg").read_bytes() == b"existing"

    def test_uploaded_files_dry_run(self, tmp_path, monkeypatch, capsys):
        """Dry run prints file migration actions without copying."""
        install_path = tmp_path / "install"
        files_src = install_path / "home" / "files"
        files_src.mkdir(parents=True)
        (files_src / "photo.jpg").write_bytes(b"image data")

        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path / "src")
        (tmp_path / "src").mkdir()

        data_path = tmp_path / "data"
        data_path.mkdir()
        (data_path / "logs").mkdir()
        (data_path / "files").mkdir()

        _apply_migrate(data_path, install_path, svc_uid=501, svc_gid=20, dry_run=True)

        output = capsys.readouterr().out
        assert "[DRY RUN] Would copy file:" in output
        assert "Would migrate 1 uploaded file(s)" in output
        # Nothing should have been copied
        assert not (data_path / "files" / "photo.jpg").exists()

    def test_memory_tree_ownership(self, tmp_path, monkeypatch):
        """Chowns the entire memory/ tree so runtime-created subdirs get fixed."""
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path / "src")
        (tmp_path / "src").mkdir()

        data_path = tmp_path / "data"
        data_path.mkdir()
        (data_path / "logs").mkdir()

        # Simulate a memory tree with a qdrant subdirectory and files
        # (as would exist after init_memory() has run at least once).
        memory_dir = data_path / "memory"
        memory_dir.mkdir()
        qdrant_dir = memory_dir / "qdrant"
        qdrant_dir.mkdir()
        (qdrant_dir / "meta.json").write_text("{}")
        (memory_dir / "MEMORY.md").write_text("# Memory")

        chowned: list[tuple[str, int, int]] = []
        monkeypatch.setattr(
            "kai.install.os.chown",
            lambda path, uid, gid: chowned.append((str(path), uid, gid)),
        )

        _apply_migrate(data_path, tmp_path / "install", svc_uid=501, svc_gid=20, dry_run=False)

        # Every directory and file in the memory tree should be chowned.
        chowned_paths = {entry[0] for entry in chowned}
        assert str(memory_dir) in chowned_paths
        assert str(qdrant_dir) in chowned_paths
        assert str(qdrant_dir / "meta.json") in chowned_paths
        assert str(memory_dir / "MEMORY.md") in chowned_paths
        # All entries should use the service user's uid/gid.
        assert all(uid == 501 and gid == 20 for _, uid, gid in chowned)


# ── Per-user MEMORY.md migration ─────────────────────────────────────
#
# These tests exercise the per-user migration path end-to-end with the
# real _collect_user_memory_owners. Each test passes `users_yaml_path=`
# explicitly, isolated under tmp_path, so nothing escapes the sandbox
# (the conftest `_isolate_users_yaml` redirect only covers the
# default-path case).


class TestApplyMigratePerUserMemory:
    def test_moves_legacy_global_memory_to_primary(self, tmp_path, monkeypatch, capsys):
        """
        A pre-#347 install that already migrated to the global
        memory/MEMORY.md gets that file MOVED into the primary user's
        subdirectory. Move (not copy) is required: a stale global file
        would shadow the per-user read once additional users join.
        """
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path / "src")
        (tmp_path / "src").mkdir()

        data_path = tmp_path / "data"
        data_path.mkdir()
        (data_path / "logs").mkdir()
        memory_dir = data_path / "memory"
        memory_dir.mkdir()
        legacy_global = memory_dir / "MEMORY.md"
        legacy_global.write_text("operator notes from before #347")

        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users:\n  - telegram_id: 8888\n    name: primary\n    role: admin\n")

        monkeypatch.setattr("kai.install.os.chown", lambda *a: None)

        _apply_migrate(
            data_path,
            tmp_path / "install",
            svc_uid=501,
            svc_gid=20,
            dry_run=False,
            users_yaml_path=users_yaml,
        )

        primary_dst = memory_dir / "8888" / "MEMORY.md"
        assert primary_dst.exists()
        assert primary_dst.read_text() == "operator notes from before #347"
        assert stat.S_IMODE((memory_dir / "8888").stat().st_mode) == 0o700
        assert stat.S_IMODE(primary_dst.stat().st_mode) == 0o600
        # Critical: legacy global MUST be gone (move, not copy) so a
        # later read at memory/MEMORY.md cannot leak this content.
        assert not legacy_global.exists()
        assert "Migrated MEMORY.md" in capsys.readouterr().out

    def test_memory_skips_existing_per_user(self, tmp_path, monkeypatch):
        """
        Does not overwrite a per-user MEMORY.md that already exists at
        the destination. Idempotent across reinstalls.
        """
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path / "src")
        claude_dir = tmp_path / "src" / "home" / ".claude"
        claude_dir.mkdir(parents=True)
        (claude_dir / "MEMORY.md").write_text("legacy content from source tree")

        data_path = tmp_path / "data"
        data_path.mkdir()
        (data_path / "logs").mkdir()
        primary_dir = data_path / "memory" / "9999"
        primary_dir.mkdir(parents=True)
        existing = primary_dir / "MEMORY.md"
        existing.write_text("existing personalized content")

        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users:\n  - telegram_id: 9999\n    name: primary\n    role: admin\n")

        monkeypatch.setattr("kai.install.os.chown", lambda *a: None)

        _apply_migrate(
            data_path,
            tmp_path / "install",
            svc_uid=501,
            svc_gid=20,
            dry_run=False,
            users_yaml_path=users_yaml,
        )

        assert existing.read_text() == "existing personalized content"

    def test_seeds_all_users_from_template(self, tmp_path, monkeypatch):
        """
        Every user in users.yaml (primary and additional) gets a fresh
        MEMORY.md seeded from the templates/.claude/ template when no
        DATA_DIR-side legacy file exists. One operator's notes must not
        become another's starting state.
        """
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path / "src")
        template_dir = tmp_path / "src" / "templates" / ".claude"
        template_dir.mkdir(parents=True)
        (template_dir / "MEMORY.md").write_text("# Memory\n\n## About the User\n")

        data_path = tmp_path / "data"
        data_path.mkdir()
        (data_path / "logs").mkdir()
        (data_path / "memory").mkdir()

        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text(
            "users:\n"
            "  - telegram_id: 100\n"
            "    name: primary\n"
            "    role: admin\n"
            "  - telegram_id: 200\n"
            "    name: secondary\n"
            "    role: user\n"
        )

        monkeypatch.setattr("kai.install.os.chown", lambda *a: None)

        _apply_migrate(
            data_path,
            tmp_path / "install",
            svc_uid=501,
            svc_gid=20,
            dry_run=False,
            users_yaml_path=users_yaml,
        )

        primary = (data_path / "memory" / "100" / "MEMORY.md").read_text()
        secondary = (data_path / "memory" / "200" / "MEMORY.md").read_text()

        # Both users get the template content; the deleted legacy_src_tree
        # branch (which once gave the primary special "PRIMARY_PRIVATE_NOTES"
        # treatment from PROJECT_ROOT/home/.claude/MEMORY.md) is gone.
        assert "About the User" in primary
        assert "About the User" in secondary

    def test_memory_no_users_yaml_is_noop(self, tmp_path, monkeypatch):
        """
        With no users.yaml (first-ever install / single-user dev), the
        memory migration is a no-op. Runtime falls back to the legacy
        global path when chat_id is None, so leaving the legacy file
        in place keeps `python -m kai` working unchanged.
        """
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path / "src")
        claude_dir = tmp_path / "src" / "home" / ".claude"
        claude_dir.mkdir(parents=True)
        (claude_dir / "MEMORY.md").write_text("dev content")

        data_path = tmp_path / "data"
        data_path.mkdir()
        (data_path / "logs").mkdir()
        (data_path / "memory").mkdir()

        monkeypatch.setattr("kai.install.os.chown", lambda *a: None)

        _apply_migrate(
            data_path,
            tmp_path / "install",
            svc_uid=501,
            svc_gid=20,
            dry_run=False,
            users_yaml_path=tmp_path / "does-not-exist.yaml",
        )

        # No per-user dirs created, no global file written.
        assert list((data_path / "memory").iterdir()) == []

    def test_unknown_os_user_aborts_before_disk_mutation(self, tmp_path, monkeypatch):
        """
        A users.yaml entry naming an os_user that does not exist on
        the host must raise ValueError BEFORE the migration block
        creates, copies, or moves any MEMORY.md file. The earlier
        implementation let pwd.getpwnam raise a bare KeyError after
        the migration had already touched disk, leaving operators to
        diagnose a half-applied state with no chat_id or path in the
        traceback. The validation block now hoisted to the top of
        _apply_migrate is the fix; this test guards against
        regressions where someone moves the validation back below
        the migration.
        """
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path / "src")
        claude_dir = tmp_path / "src" / "home" / ".claude"
        claude_dir.mkdir(parents=True)
        # Source-tree MEMORY.md is what the legacy-copy branch would
        # try to land at the primary user's destination if validation
        # did not abort first.
        (claude_dir / "MEMORY.md").write_text("would-be-leaked content")

        data_path = tmp_path / "data"
        data_path.mkdir()
        (data_path / "logs").mkdir()
        (data_path / "memory").mkdir()

        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text(
            "users:\n  - telegram_id: 4242\n    name: ghost\n    role: admin\n    os_user: nobody-by-this-name-12345\n"
        )

        # Force getpwnam to behave as it would for a missing user on
        # any host: raise KeyError(name). pwd.getpwnam itself raises
        # bare KeyError, no PermissionError, no ValueError.
        monkeypatch.setattr(
            "kai.install.pwd.getpwnam",
            lambda name: (_ for _ in ()).throw(KeyError(name)),
        )
        # No-op chown: if validation did not abort and the test got
        # this far, the ownership block would otherwise try to chown
        # outside tmp_path on a CI box.
        monkeypatch.setattr("kai.install.os.chown", lambda *a: None)

        with pytest.raises(ValueError) as exc_info:
            _apply_migrate(
                data_path,
                tmp_path / "install",
                svc_uid=501,
                svc_gid=20,
                dry_run=False,
                users_yaml_path=users_yaml,
            )

        # Error message must include the chat_id, the bad username,
        # and the source path so the operator knows where to fix it.
        msg = str(exc_info.value)
        assert "4242" in msg
        assert "nobody-by-this-name-12345" in msg
        assert str(users_yaml) in msg

        # Critical: no MEMORY.md was written under the primary's dir,
        # because validation aborted before the migration block ran.
        # Half-applied state (some files moved, some not, ownership
        # still pending) is exactly the scenario this test guards.
        assert not (data_path / "memory" / "4242").exists()
        assert not (data_path / "memory" / "MEMORY.md").exists()


class TestApplyMigratePerUserHome:
    """
    Tests for the per-user home workspace provisioning in _apply_migrate
    (#353). Mirrors TestApplyMigratePerUserMemory shape; the install-time
    block is a pure file-system effect (mkdir + chown + chmod), so these
    tests exercise the directory layout that gets produced.
    """

    def _write_users_yaml(self, path: Path, body: str) -> None:
        path.write_text(body)

    def _setup(self, tmp_path, monkeypatch) -> Path:
        """Common scaffolding: project root, data dirs, no-op chown."""
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path / "src")
        (tmp_path / "src").mkdir()
        data_path = tmp_path / "data"
        data_path.mkdir()
        (data_path / "logs").mkdir()
        (data_path / "memory").mkdir()
        # chown is a no-op in tests: we are not root, and the test is
        # only checking which paths got created, not their ownership.
        monkeypatch.setattr("kai.install.os.chown", lambda *a: None)
        return data_path

    def test_atomic_copy_removes_partial_staging_tree(self, tmp_path, monkeypatch):
        source = tmp_path / "legacy"
        source.mkdir()
        destination = tmp_path / ("prn_" + "a" * 32)

        def fail_after_partial_copy(_source, temporary, *, symlinks):
            assert symlinks is True
            temporary.mkdir()
            (temporary / "partial.txt").write_text("partial")
            raise OSError("simulated copy failure")

        monkeypatch.setattr("kai.install.shutil.copytree", fail_after_partial_copy)

        with pytest.raises(OSError, match="simulated copy failure"):
            _copy_managed_home_tree(source, destination)

        assert not destination.exists()
        assert not list(tmp_path.glob(f".{destination.name}.migrating-*"))

    def test_database_path_migration_handles_missing_optional_tables(self, tmp_path):
        data_path = tmp_path / "data"
        data_path.mkdir()
        db_path = data_path / "kai.db"
        connection = sqlite3.connect(db_path)
        connection.execute("CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        legacy_home = data_path / "home" / "42"
        canonical_home = data_path / "home" / ("prn_" + "d" * 32)
        connection.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?)",
            ("workspace:42", str(legacy_home / "project")),
        )
        connection.commit()
        connection.close()

        _migrate_managed_home_database_paths(
            data_path,
            {42: (legacy_home, canonical_home)},
            dry_run=False,
        )

        connection = sqlite3.connect(db_path)
        assert connection.execute("SELECT value FROM settings WHERE key = 'workspace:42'").fetchone() == (
            str(canonical_home / "project"),
        )
        connection.close()

    def test_no_override_creates_home_chat_id_dir(self, tmp_path, monkeypatch):
        """
        Case 1 (default): a user with no users.yaml home_workspace lands
        in DATA_DIR/home/<chat_id>/. This is the spec #353 default.
        """
        data_path = self._setup(tmp_path, monkeypatch)
        users_yaml = tmp_path / "users.yaml"
        self._write_users_yaml(
            users_yaml,
            "users:\n  - telegram_id: 5555\n    name: u\n    role: admin\n",
        )

        _apply_migrate(
            data_path,
            tmp_path / "install",
            svc_uid=501,
            svc_gid=20,
            dry_run=False,
            users_yaml_path=users_yaml,
        )

        assert (data_path / "home" / "5555").is_dir()

    def test_override_under_data_dir_provisions_override_path(self, tmp_path, monkeypatch):
        """
        Case 2 (W2 review fix): when home_workspace is set to a path
        INSIDE DATA_DIR, the installer must create THAT path (the one
        runtime resolve_home_workspace returns), NOT the default
        DATA_DIR/home/<chat_id>/ slot.

        Pre-fix bug: installer always created home/<chat_id>/, leaving
        the actual override path un-provisioned. First runtime write to
        the override would crash. This test pins the corrected behavior.
        """
        data_path = self._setup(tmp_path, monkeypatch)
        # Override path lives under data_path/custom_workspace/.
        override = data_path / "custom_workspace"
        users_yaml = tmp_path / "users.yaml"
        self._write_users_yaml(
            users_yaml,
            (f"users:\n  - telegram_id: 6666\n    name: u\n    role: admin\n    home_workspace: {override}\n"),
        )

        _apply_migrate(
            data_path,
            tmp_path / "install",
            svc_uid=501,
            svc_gid=20,
            dry_run=False,
            users_yaml_path=users_yaml,
        )

        # The override path itself was created.
        assert override.is_dir(), f"Override path {override} not provisioned"
        # The default slot must NOT be created - it is never used at
        # runtime when an override is set, so creating it would leave a
        # dead directory under DATA_DIR.
        assert not (data_path / "home" / "6666").exists(), (
            "Default per-user slot should not be created when override is set"
        )

    def test_override_outside_data_dir_skips_provisioning(self, tmp_path, monkeypatch):
        """
        Case 3: when home_workspace points OUTSIDE DATA_DIR, the
        installer must skip the entry entirely. The override is
        operator-managed (a clone of a dev tree, a synced volume, etc.)
        and we have no business chowning it.
        """
        data_path = self._setup(tmp_path, monkeypatch)
        # Path outside data_path: a sibling under tmp_path.
        external = tmp_path / "external_home"
        external.mkdir()
        users_yaml = tmp_path / "users.yaml"
        self._write_users_yaml(
            users_yaml,
            (f"users:\n  - telegram_id: 7777\n    name: u\n    role: admin\n    home_workspace: {external}\n"),
        )

        _apply_migrate(
            data_path,
            tmp_path / "install",
            svc_uid=501,
            svc_gid=20,
            dry_run=False,
            users_yaml_path=users_yaml,
        )

        # Default slot must NOT be created (we honored the override).
        assert not (data_path / "home" / "7777").exists()
        # External path must be left strictly alone (we did not chmod
        # or otherwise touch it - verify it is still empty).
        assert external.is_dir()
        assert list(external.iterdir()) == []

    def test_creates_home_root_when_missing(self, tmp_path, monkeypatch):
        """
        W3 review fix: if data_path/home/ does not exist when the
        per-user block runs (e.g., a future refactor splits
        _apply_directories from _apply_migrate), the block must still
        provision per-user dirs rather than silently no-op'ing. The
        defensive home_root.mkdir guarantees this.

        Pre-fix: the block was guarded by `if home_root.is_dir():`
        which silently skipped everything when the parent was absent.

        Current security contract: home_root must end up at exactly
        0o711, not whatever the umask leaves behind. mkdir(mode=0o711) is
        masked by the process umask; under the production service
        umask of 0o027 this would leave home_root at 0o710. We set a
        hostile umask inside the test to prove the explicit chmod both
        preserves traversal and prevents directory listing by siblings.
        """
        import os as _os
        import stat

        data_path = self._setup(tmp_path, monkeypatch)
        # Note: _setup does NOT create data_path/home. The defensive
        # mkdir in _apply_migrate must create it.
        assert not (data_path / "home").exists()

        users_yaml = tmp_path / "users.yaml"
        self._write_users_yaml(
            users_yaml,
            "users:\n  - telegram_id: 8888\n    name: u\n    role: admin\n",
        )

        # Hostile umask matches the production service launchd config.
        # Without the explicit os.chmod after mkdir, home_root would
        # come out as 0o710 here.
        prev_umask = _os.umask(0o027)
        try:
            _apply_migrate(
                data_path,
                tmp_path / "install",
                svc_uid=501,
                svc_gid=20,
                dry_run=False,
                users_yaml_path=users_yaml,
            )
        finally:
            _os.umask(prev_umask)

        # Both the parent home/ and the per-user slot exist.
        home_root = data_path / "home"
        assert home_root.is_dir()
        assert (home_root / "8888").is_dir()
        # Root is traversal-only for siblings; the per-user slot is private.
        assert stat.S_IMODE(home_root.stat().st_mode) == 0o711, (
            f"home_root mode {oct(stat.S_IMODE(home_root.stat().st_mode))} - "
            "umask masked the mkdir mode and explicit chmod did not run"
        )
        assert stat.S_IMODE((home_root / "8888").stat().st_mode) == 0o700

    def test_override_resolves_through_data_path_symlink(self, tmp_path, monkeypatch):
        """
        Round 3 review fix: when DATA_DIR traverses a symlink (the
        macOS case where /var/lib is a symlink to /private/var/lib),
        the override-containment check must resolve both sides.

        Pre-fix: `_collect_user_home_overrides` calls `Path.resolve()`
        on every override but `_apply_migrate` compared the resolved
        override against an unresolved data_path. An operator who
        wrote `home_workspace: /private/var/lib/kai/custom` against a
        `data_path` of `/var/lib/kai` would get is_relative_to=False
        and the entry would silently fall through to Case 3 (skip).
        First-write under a distinct os_user would then crash.

        This test pins the symlink case by creating a symlink that
        points at the real data_path and passing the symlink as the
        installer's data_path argument. Without `data_path.resolve()`
        the override (already resolved through the symlink) compares
        as external and the test fails.
        """
        # Real data_path lives under tmp_path/real_data; the installer
        # is invoked with a symlink at tmp_path/data_link that points
        # at it. This mirrors the macOS /var/lib -> /private/var/lib
        # situation that triggered the round-3 review finding.
        real_data = tmp_path / "real_data"
        real_data.mkdir()
        (real_data / "logs").mkdir()
        (real_data / "memory").mkdir()
        data_link = tmp_path / "data_link"
        data_link.symlink_to(real_data)

        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path / "src")
        (tmp_path / "src").mkdir()
        monkeypatch.setattr("kai.install.os.chown", lambda *a: None)

        # Override path written against the REAL location (post-resolve
        # form), simulating an operator who already canonicalized in
        # users.yaml. _collect_user_home_overrides will resolve() it
        # again to the same value.
        override_real = real_data / "custom_workspace"
        users_yaml = tmp_path / "users.yaml"
        self._write_users_yaml(
            users_yaml,
            (f"users:\n  - telegram_id: 9999\n    name: u\n    role: admin\n    home_workspace: {override_real}\n"),
        )

        # Pass the symlinked path as data_path - this is what the
        # round-3 bug needed to surface.
        _apply_migrate(
            data_link,
            tmp_path / "install",
            svc_uid=501,
            svc_gid=20,
            dry_run=False,
            users_yaml_path=users_yaml,
        )

        # The override path under the REAL data dir was provisioned -
        # proves the symlink-aware containment check classified it as
        # internal (Case 2) instead of external (Case 3).
        assert override_real.is_dir(), (
            "Override under symlinked DATA_DIR was not provisioned - "
            "is_relative_to comparison did not resolve data_path"
        )

    async def test_copies_complete_home_and_rewrites_kai_database_paths(
        self,
        tmp_path,
        monkeypatch,
    ):
        data_path = self._setup(tmp_path, monkeypatch)
        users_yaml = tmp_path / "users.yaml"
        self._write_users_yaml(
            users_yaml,
            "users:\n  - telegram_id: 7777\n    name: primary\n    role: admin\n",
        )
        legacy_home = data_path / "home" / "7777"
        project = legacy_home / "project"
        project.mkdir(parents=True)
        (legacy_home / "AGENTS.md").write_text("# Operator identity\n")
        executable = project / "run.sh"
        executable.write_text("#!/bin/sh\n")
        executable.chmod(0o755)
        legacy_memory = data_path / "memory" / "7777" / "MEMORY.md"
        legacy_memory.parent.mkdir(parents=True)
        legacy_memory.write_text("operator memory")

        store = await WorkshopEventStore.open(data_path / "kai.db")
        await bootstrap_default_workshop(
            store,
            (
                BootstrapHuman(
                    "Primary",
                    "admin",
                    "telegram",
                    "7777",
                    "7777",
                    profile_id(7777),
                ),
            ),
        )
        async with store.connection.execute(
            "SELECT principal_id FROM external_identities WHERE provider = 'telegram' AND external_subject = '7777'"
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        principal_id = str(row[0])
        await store.close()

        connection = sqlite3.connect(data_path / "kai.db")
        connection.executescript(
            "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
            "CREATE TABLE IF NOT EXISTS allowed_workspaces ("
            "chat_id INTEGER NOT NULL, path TEXT NOT NULL, PRIMARY KEY (chat_id, path));"
            "CREATE TABLE IF NOT EXISTS workspace_history ("
            "path TEXT NOT NULL, chat_id INTEGER NOT NULL, last_used_at TIMESTAMP, "
            "PRIMARY KEY (path, chat_id));"
            "CREATE TABLE IF NOT EXISTS memory_projects ("
            "project_id TEXT PRIMARY KEY, display_name TEXT NOT NULL, "
            "workspace_root TEXT NOT NULL UNIQUE, memory_enabled INTEGER NOT NULL, "
            "default_scope_for_new_facts TEXT, created_by INTEGER NOT NULL, "
            "created_at TIMESTAMP);"
        )
        legacy_project = str(project)
        connection.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?)",
            ("workspace:7777", legacy_project),
        )
        connection.execute(
            "INSERT INTO settings (key, value) VALUES (?, ?)",
            (f"ws_config:7777:{legacy_project}:model", "gpt-test"),
        )
        connection.execute(
            "INSERT INTO allowed_workspaces (chat_id, path) VALUES (?, ?)",
            (7777, legacy_project),
        )
        connection.execute(
            "INSERT INTO workspace_history (path, chat_id, last_used_at) VALUES (?, ?, ?)",
            (legacy_project, 7777, "2026-08-13 05:00:00"),
        )
        connection.execute(
            "INSERT INTO memory_projects "
            "(project_id, display_name, workspace_root, memory_enabled, "
            "default_scope_for_new_facts, created_by, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("project", "Project", legacy_project, 1, None, 7777, "2026-08-13 05:00:00"),
        )
        connection.commit()
        connection.close()

        _apply_migrate(
            data_path,
            tmp_path / "install",
            svc_uid=501,
            svc_gid=20,
            dry_run=True,
            users_yaml_path=users_yaml,
        )
        canonical_home = data_path / "home" / principal_id
        assert not canonical_home.exists()
        connection = sqlite3.connect(data_path / "kai.db")
        assert connection.execute("SELECT value FROM settings WHERE key = 'workspace:7777'").fetchone() == (
            legacy_project,
        )
        connection.close()

        _apply_migrate(
            data_path,
            tmp_path / "install",
            svc_uid=501,
            svc_gid=20,
            dry_run=False,
            users_yaml_path=users_yaml,
        )

        canonical_project = canonical_home / "project"
        assert (canonical_home / "AGENTS.md").read_text() == "# Operator identity\n"
        assert (canonical_project / "run.sh").stat().st_mode & 0o777 == 0o755
        assert (legacy_home / "project" / "run.sh").is_file()
        canonical_memory = data_path / "memory" / principal_id / "MEMORY.md"
        assert canonical_memory.read_text() == "operator memory"
        assert legacy_memory.read_text() == "operator memory"

        connection = sqlite3.connect(data_path / "kai.db")
        canonical_project_text = str(canonical_project)
        assert connection.execute("SELECT value FROM settings WHERE key = 'workspace:7777'").fetchone() == (
            canonical_project_text,
        )
        assert connection.execute(
            "SELECT value FROM settings WHERE key = ?",
            (f"ws_config:7777:{canonical_project_text}:model",),
        ).fetchone() == ("gpt-test",)
        assert connection.execute("SELECT path FROM allowed_workspaces WHERE chat_id = 7777").fetchone() == (
            canonical_project_text,
        )
        assert connection.execute("SELECT path FROM workspace_history WHERE chat_id = 7777").fetchone() == (
            canonical_project_text,
        )
        assert connection.execute(
            "SELECT workspace_root FROM memory_projects WHERE project_id = 'project'"
        ).fetchone() == (canonical_project_text,)
        connection.close()

        (canonical_home / "canonical-only.txt").write_text("new state")
        _apply_migrate(
            data_path,
            tmp_path / "install",
            svc_uid=501,
            svc_gid=20,
            dry_run=False,
            users_yaml_path=users_yaml,
        )
        assert (canonical_home / "canonical-only.txt").read_text() == "new state"


# ── Per-user PREFERENCES.md migration (#400) ─────────────────────────


class TestApplyMigratePerUserPreferences:
    """
    Tests for the per-user PREFERENCES.md install-time work in
    _apply_migrate (#400). Mirrors TestApplyMigratePerUserMemory shape:
    seed block creates DATA_DIR/preferences/<chat_id>/PREFERENCES.md
    from the example template; ownership pass re-chowns the tree on
    every install so os_user changes propagate to existing files.
    """

    def _setup(self, tmp_path, monkeypatch, with_template: bool = True) -> Path:
        """Common scaffolding: project root, data dirs, optional template."""
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path / "src")
        template_dir = tmp_path / "src" / "templates" / ".claude"
        template_dir.mkdir(parents=True)
        if with_template:
            (template_dir / "PREFERENCES.md").write_text("# Preferences\n\n## Style\n\n## Working Discipline\n")
        data_path = tmp_path / "data"
        data_path.mkdir()
        (data_path / "logs").mkdir()
        (data_path / "memory").mkdir()
        return data_path

    def test_seeds_per_user_preferences_from_template(self, tmp_path, monkeypatch):
        """Single-user case: PREFERENCES.md is seeded from the templates/ template."""
        data_path = self._setup(tmp_path, monkeypatch)
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users:\n  - telegram_id: 7777\n    name: primary\n    role: admin\n")
        monkeypatch.setattr("kai.install.os.chown", lambda *a: None)

        _apply_migrate(
            data_path,
            tmp_path / "install",
            svc_uid=501,
            svc_gid=20,
            dry_run=False,
            users_yaml_path=users_yaml,
        )

        target = data_path / "preferences" / "7777" / "PREFERENCES.md"
        assert target.is_file()
        assert "Style" in target.read_text()
        assert target.stat().st_mode & 0o777 == 0o600
        assert (data_path / "preferences" / "7777").stat().st_mode & 0o777 == 0o700

    def test_seeds_each_user_in_multi_user_yaml(self, tmp_path, monkeypatch):
        """Every users.yaml entry gets its own PREFERENCES.md seeded from the template."""
        data_path = self._setup(tmp_path, monkeypatch)
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text(
            "users:\n"
            "  - telegram_id: 1001\n    name: alpha\n    role: admin\n"
            "  - telegram_id: 1002\n    name: beta\n    role: user\n"
        )
        monkeypatch.setattr("kai.install.os.chown", lambda *a: None)

        _apply_migrate(
            data_path,
            tmp_path / "install",
            svc_uid=501,
            svc_gid=20,
            dry_run=False,
            users_yaml_path=users_yaml,
        )

        assert (data_path / "preferences" / "1001" / "PREFERENCES.md").is_file()
        assert (data_path / "preferences" / "1002" / "PREFERENCES.md").is_file()

    def test_idempotent_does_not_overwrite_existing(self, tmp_path, monkeypatch):
        """Pre-existing per-user PREFERENCES.md content is preserved across reinstalls."""
        data_path = self._setup(tmp_path, monkeypatch)
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users:\n  - telegram_id: 8888\n    name: keeper\n    role: admin\n")

        # Operator already has customized PREFERENCES.md; re-running install
        # must not overwrite it.
        existing = data_path / "preferences" / "8888"
        existing.mkdir(parents=True)
        target = existing / "PREFERENCES.md"
        target.write_text("operator-customized rules")
        monkeypatch.setattr("kai.install.os.chown", lambda *a: None)

        _apply_migrate(
            data_path,
            tmp_path / "install",
            svc_uid=501,
            svc_gid=20,
            dry_run=False,
            users_yaml_path=users_yaml,
        )

        assert target.read_text() == "operator-customized rules"

    async def test_copies_legacy_preferences_to_canonical_principal_without_deleting(
        self,
        tmp_path,
        monkeypatch,
    ):
        data_path = self._setup(tmp_path, monkeypatch)
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users:\n  - telegram_id: 7777\n    name: primary\n    role: admin\n")
        legacy = data_path / "preferences" / "7777" / "PREFERENCES.md"
        legacy.parent.mkdir(parents=True)
        legacy.write_text("operator-customized rules")
        store = await WorkshopEventStore.open(data_path / "kai.db")
        await bootstrap_default_workshop(
            store,
            (
                BootstrapHuman(
                    "Primary",
                    "admin",
                    "telegram",
                    "7777",
                    "7777",
                    profile_id(7777),
                ),
            ),
        )
        async with store.connection.execute(
            "SELECT principal_id FROM external_identities WHERE provider = 'telegram' AND external_subject = '7777'"
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        principal_id = str(row[0])
        await store.close()
        monkeypatch.setattr("kai.install.os.chown", lambda *args: None)

        _apply_migrate(
            data_path,
            tmp_path / "install",
            svc_uid=501,
            svc_gid=20,
            dry_run=False,
            users_yaml_path=users_yaml,
        )

        canonical = data_path / "preferences" / principal_id / "PREFERENCES.md"
        assert canonical.read_text() == "operator-customized rules"
        assert legacy.read_text() == "operator-customized rules"

    def test_template_missing_writes_placeholder(self, tmp_path, monkeypatch, capsys):
        """When the template is absent, write '# Preferences\\n' placeholder + warn."""
        data_path = self._setup(tmp_path, monkeypatch, with_template=False)
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users:\n  - telegram_id: 9999\n    name: u\n    role: admin\n")
        monkeypatch.setattr("kai.install.os.chown", lambda *a: None)

        _apply_migrate(
            data_path,
            tmp_path / "install",
            svc_uid=501,
            svc_gid=20,
            dry_run=False,
            users_yaml_path=users_yaml,
        )

        target = data_path / "preferences" / "9999" / "PREFERENCES.md"
        assert target.is_file()
        assert target.read_text() == "# Preferences\n"
        # Warn so the operator notices an incomplete install tree.
        out = capsys.readouterr().out
        assert "WARNING" in out
        # The warning names the missing template path under templates/.claude/.
        assert "templates/.claude/PREFERENCES.md" in out

    def test_dry_run_makes_no_filesystem_changes(self, tmp_path, monkeypatch, capsys):
        """Dry run prints intended actions but does not create any file or directory."""
        data_path = self._setup(tmp_path, monkeypatch)
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users:\n  - telegram_id: 4242\n    name: u\n    role: admin\n")

        _apply_migrate(
            data_path,
            tmp_path / "install",
            svc_uid=501,
            svc_gid=20,
            dry_run=True,
            users_yaml_path=users_yaml,
        )

        out = capsys.readouterr().out
        assert "Would seed" in out
        assert "PREFERENCES.md" in out
        # Nothing on disk under preferences/.
        assert not (data_path / "preferences").exists()

    def test_ownership_pass_chowns_tree(self, tmp_path, monkeypatch):
        """
        The recursive ownership pass walks DATA_DIR/preferences/ and
        chowns every entry. Without this, an os_user change in
        users.yaml between installs would leave existing PREFERENCES.md
        files owned by the previous user, reproducing the #347
        regression that motivated this layer.
        """
        data_path = self._setup(tmp_path, monkeypatch)
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users:\n  - telegram_id: 5555\n    name: u\n    role: admin\n")

        # Pre-existing tree with a stale subdir/file from a prior install.
        pref_root = data_path / "preferences"
        pref_root.mkdir()
        user_dir = pref_root / "5555"
        user_dir.mkdir()
        (user_dir / "PREFERENCES.md").write_text("existing")

        chowned: list[tuple[str, int, int]] = []
        monkeypatch.setattr(
            "kai.install.os.chown",
            lambda path, uid, gid: chowned.append((str(path), uid, gid)),
        )
        # _set_ownership uses os.lchown for symlinks; pin both so the
        # test does not require root.
        monkeypatch.setattr("kai.install.os.lchown", lambda path, uid, gid: chowned.append((str(path), uid, gid)))

        _apply_migrate(
            data_path,
            tmp_path / "install",
            svc_uid=501,
            svc_gid=20,
            dry_run=False,
            users_yaml_path=users_yaml,
        )

        chowned_paths = {entry[0] for entry in chowned}
        assert str(pref_root) in chowned_paths
        assert str(user_dir) in chowned_paths
        assert str(user_dir / "PREFERENCES.md") in chowned_paths

    def test_stray_subdir_falls_through_to_service_user(self, tmp_path, monkeypatch):
        """
        A subdir whose chat_id is not in users.yaml gets service-user
        ownership (fallback path). Mirrors the memory ownership pass's
        same fallback.
        """
        data_path = self._setup(tmp_path, monkeypatch)
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users:\n  - telegram_id: 100\n    name: u\n    role: admin\n")

        # 999 is NOT in users.yaml; it should be chowned to service user.
        pref_root = data_path / "preferences"
        pref_root.mkdir()
        stray = pref_root / "999"
        stray.mkdir()
        (stray / "PREFERENCES.md").write_text("stray content")

        chowned: list[tuple[str, int, int]] = []
        monkeypatch.setattr(
            "kai.install.os.chown",
            lambda path, uid, gid: chowned.append((str(path), uid, gid)),
        )
        monkeypatch.setattr("kai.install.os.lchown", lambda path, uid, gid: chowned.append((str(path), uid, gid)))

        _apply_migrate(
            data_path,
            tmp_path / "install",
            svc_uid=501,
            svc_gid=20,
            dry_run=False,
            users_yaml_path=users_yaml,
        )

        # The stray dir should be chowned to (501, 20), the service user.
        stray_calls = [c for c in chowned if c[0] == str(stray)]
        assert stray_calls, f"Expected chown call for stray dir {stray}"
        assert all(uid == 501 and gid == 20 for _, uid, gid in stray_calls)


class TestGitignoreRuntimePreferencesDir:
    """
    The runtime preferences/ directory must stay gitignored under
    "# Runtime artifacts" alongside memory/ and history/ so dev-mode
    runs never leak per-user PREFERENCES.md content into the working
    tree.
    """

    def _gitignore_path(self) -> Path:
        # PROJECT_ROOT lives at src/kai/install.py - go up two parents
        # to reach the repo root, then read .gitignore from there.
        from kai import install

        return Path(install.__file__).resolve().parents[2] / ".gitignore"

    def test_runtime_preferences_dir_is_ignored(self):
        """`preferences/` is listed under Runtime artifacts so dev-mode data does not leak."""
        body = self._gitignore_path().read_text()
        # Match the bare `preferences/` line, not a substring of a
        # longer path. The exact line appears under the "# Runtime
        # artifacts" block, between `memory/` and `files/`.
        assert "\npreferences/\n" in body


# ── Service lifecycle ────────────────────────────────────────────────


class TestStopService:
    def test_darwin(self, monkeypatch, tmp_path):
        """Calls launchctl bootout on macOS with system domain."""
        calls: list[list[str]] = []

        def mock_run(cmd, **kwargs):
            calls.append(list(cmd))
            return subprocess.CompletedProcess(args=cmd, returncode=0)

        monkeypatch.setattr("kai.install.subprocess.run", mock_run)

        _stop_service("darwin", svc_uid=501, service_user="kai", dry_run=False)

        assert len(calls) == 1
        assert calls[0][0] == "launchctl"
        assert calls[0][1] == "bootout"
        assert calls[0][2] == "system/com.syrinx.kai"

    def test_linux(self, monkeypatch):
        """Calls systemctl stop on Linux."""
        calls: list[list[str]] = []

        def mock_run(cmd, **kwargs):
            calls.append(list(cmd))
            return subprocess.CompletedProcess(args=cmd, returncode=0)

        monkeypatch.setattr("kai.install.subprocess.run", mock_run)

        _stop_service("linux", svc_uid=1000, service_user="kai", dry_run=False)

        assert calls == [["systemctl", "stop", "kai"]]

    def test_dry_run(self, monkeypatch, tmp_path, capsys):
        """Dry run prints the command without executing."""
        calls: list = []
        monkeypatch.setattr(
            "kai.install.subprocess.run",
            lambda *a, **kw: calls.append(True),
        )

        _stop_service("darwin", svc_uid=501, service_user="kai", dry_run=True)

        output = capsys.readouterr().out
        assert "[DRY RUN]" in output
        assert len(calls) == 0


class TestStartService:
    def test_darwin(self, monkeypatch):
        """Calls launchctl bootstrap then launchctl print to verify on
        macOS. Two calls per attempt because the bootstrap exit code is
        treated as advisory; verify is the authoritative check."""
        calls: list[list[str]] = []

        def mock_run(cmd, **kwargs):
            calls.append(list(cmd))
            return subprocess.CompletedProcess(args=cmd, returncode=0)

        monkeypatch.setattr("kai.install.subprocess.run", mock_run)
        monkeypatch.setattr("kai.install.time.sleep", lambda _s: None)

        _start_service("darwin", svc_uid=501, service_user="kai", dry_run=False)

        assert len(calls) == 2
        assert calls[0][:3] == ["launchctl", "bootstrap", "system"]
        assert calls[1] == ["launchctl", "print", "system/com.syrinx.kai"]

    def test_linux(self, monkeypatch):
        """Calls systemctl start then systemctl is-active to verify on
        Linux. Mirrors the macOS verify-after-start shape so an
        operator does not see different success contracts per platform."""
        calls: list[list[str]] = []

        def mock_run(cmd, **kwargs):
            calls.append(list(cmd))
            return subprocess.CompletedProcess(args=cmd, returncode=0)

        monkeypatch.setattr("kai.install.subprocess.run", mock_run)
        monkeypatch.setattr("kai.install.time.sleep", lambda _s: None)

        _start_service("linux", svc_uid=1000, service_user="kai", dry_run=False)

        assert calls == [
            ["systemctl", "start", "kai"],
            ["systemctl", "is-active", "kai"],
        ]

    def test_dry_run(self, monkeypatch, capsys):
        """Dry run prints the command without executing."""
        calls: list = []
        monkeypatch.setattr(
            "kai.install.subprocess.run",
            lambda *a, **kw: calls.append(True),
        )

        _start_service("linux", svc_uid=1000, service_user="kai", dry_run=True)

        output = capsys.readouterr().out
        assert "[DRY RUN]" in output
        assert len(calls) == 0

    def test_succeeds_when_bootstrap_returns_nonzero_but_verify_passes(self, monkeypatch):
        """The originating-issue pattern: launchctl bootstrap returns
        exit code 5 ("Input/output error") but the daemon is actually
        registered. The previous implementation trusted the bootstrap
        exit code and reported failure; the new contract checks the
        verify post-condition and returns success.

        Pinning this prevents a regression that re-introduces the
        `if start.returncode == 0: return` short-circuit."""

        def mock_run(cmd, **kwargs):
            if cmd[:2] == ["launchctl", "bootstrap"]:
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=5,
                    stderr=b"Bootstrap failed: 5: Input/output error",
                )
            return subprocess.CompletedProcess(args=cmd, returncode=0)

        monkeypatch.setattr("kai.install.subprocess.run", mock_run)
        monkeypatch.setattr("kai.install.time.sleep", lambda _s: None)

        # No exception means the verify was treated as authoritative.
        _start_service("darwin", svc_uid=501, service_user="kai", dry_run=False)

    def test_retries_then_succeeds_when_first_verify_fails(self, monkeypatch):
        """Two settling attempts to absorb the transient launchd-
        domain-not-yet-released window. The retry cycle should succeed
        when an early verify fails but a later one confirms
        registration; budget exhaustion is a separate test below."""
        verify_calls = 0

        def mock_run(cmd, **kwargs):
            nonlocal verify_calls
            if cmd[:2] == ["launchctl", "print"]:
                verify_calls += 1
                rc = 0 if verify_calls >= 2 else 1
                return subprocess.CompletedProcess(args=cmd, returncode=rc)
            return subprocess.CompletedProcess(args=cmd, returncode=0)

        monkeypatch.setattr("kai.install.subprocess.run", mock_run)
        monkeypatch.setattr("kai.install.time.sleep", lambda _s: None)

        # Returns cleanly on the second attempt; no exception raised.
        _start_service("darwin", svc_uid=501, service_user="kai", dry_run=False)
        assert verify_calls == 2

    def test_raises_service_start_error_when_verify_never_passes(self, monkeypatch):
        """The verify-failure exhaustion path: every retry attempt
        sees a passing-looking bootstrap and a failing verify. The
        contract is that this raises ServiceStartError rather than
        warning and returning, so the caller in _cmd_apply can
        propagate the failure and the install does not exit 0 with
        the daemon unregistered.

        The error message names the verify command so the operator
        sees the authoritative failure rather than the misleading
        bootstrap exit code."""

        def mock_run(cmd, **kwargs):
            if cmd[:2] == ["launchctl", "print"]:
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=1,
                    stderr=b'Could not find service "com.syrinx.kai" in domain for system',
                )
            return subprocess.CompletedProcess(args=cmd, returncode=0)

        monkeypatch.setattr("kai.install.subprocess.run", mock_run)
        monkeypatch.setattr("kai.install.time.sleep", lambda _s: None)

        with pytest.raises(ServiceStartError) as excinfo:
            _start_service("darwin", svc_uid=501, service_user="kai", dry_run=False)

        msg = str(excinfo.value)
        assert "launchctl print" in msg
        assert "Could not find service" in msg

    def test_raises_service_start_error_on_linux_when_is_active_fails(self, monkeypatch):
        """Same exhaustion contract on Linux: a passing systemctl
        start with a failing systemctl is-active still raises so the
        platform contracts stay symmetric."""

        def mock_run(cmd, **kwargs):
            if cmd[:2] == ["systemctl", "is-active"]:
                return subprocess.CompletedProcess(
                    args=cmd,
                    returncode=3,
                    stdout=b"inactive\n",
                )
            return subprocess.CompletedProcess(args=cmd, returncode=0)

        monkeypatch.setattr("kai.install.subprocess.run", mock_run)
        monkeypatch.setattr("kai.install.time.sleep", lambda _s: None)

        with pytest.raises(ServiceStartError) as excinfo:
            _start_service("linux", svc_uid=1000, service_user="kai", dry_run=False)

        assert "systemctl is-active" in str(excinfo.value)


# ── CLI dispatch ─────────────────────────────────────────────────────


class TestCli:
    def test_unknown_subcommand_exits(self):
        with pytest.raises(SystemExit):
            cli(["unknown"])

    def test_no_args_exits(self):
        with pytest.raises(SystemExit):
            cli([])

    def test_dispatches_status(self, monkeypatch):
        """CLI dispatches 'status' to _cmd_status."""
        called = []
        monkeypatch.setattr("kai.install._cmd_status", lambda: called.append(True))
        cli(["status"])
        assert called

    def test_dispatches_config(self, monkeypatch):
        """CLI dispatches 'config' to _cmd_config."""
        called = []
        monkeypatch.setattr("kai.install._cmd_config", lambda: called.append(True))
        cli(["config"])
        assert called

    def test_dispatches_apply(self, monkeypatch):
        """CLI dispatches 'apply' to _cmd_apply."""
        called = []
        monkeypatch.setattr("kai.install._cmd_apply", lambda: called.append(True))
        cli(["apply"])
        assert called

    def test_dry_run_flag_sets_env(self, monkeypatch):
        """--dry-run flag sets DRY_RUN=1 in the environment before calling apply."""
        import os

        captured_env = {}
        monkeypatch.delenv("DRY_RUN", raising=False)

        def mock_apply():
            # Capture the env var at call time
            captured_env["DRY_RUN"] = os.environ.get("DRY_RUN")

        monkeypatch.setattr("kai.install._cmd_apply", mock_apply)
        cli(["apply", "--dry-run"])
        assert captured_env.get("DRY_RUN") == "1"


class TestMakeInstallDryRun:
    """The Make layer must carry dry-run intent across sudo explicitly."""

    @staticmethod
    def _clean_env() -> dict[str, str]:
        env = os.environ.copy()
        env.pop("DRY_RUN", None)
        return env

    @pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
    def test_nonempty_dry_run_adds_explicit_cli_flag(self):
        result = subprocess.run(
            ["make", "-n", "DRY_RUN=1", "install"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=True,
            env=self._clean_env(),
        )

        assert "install apply --dry-run" in result.stdout

    @pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
    def test_normal_install_does_not_add_dry_run_flag(self):
        result = subprocess.run(
            ["make", "-n", "install"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=True,
            env=self._clean_env(),
        )

        assert "install apply --dry-run" not in result.stdout


class TestMakeInstallStatus:
    """The Make status target must have access to deployed protected state."""

    @pytest.mark.skipif(shutil.which("make") is None, reason="make is not installed")
    def test_status_runs_privileged_installer_command(self):
        result = subprocess.run(
            ["make", "-n", "install-status"],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=True,
        )

        command_lines = [line for line in result.stdout.splitlines() if line.startswith("sudo ")]
        assert len(command_lines) == 1
        assert command_lines[0].endswith("python -m kai install status")


# ── _set_ownership ───────────────────────────────────────────────────


class TestSetOwnership:
    def test_single_file(self, tmp_path):
        """Sets ownership on a single file."""
        f = tmp_path / "file.txt"
        f.touch()
        with patch("os.chown") as mock_chown:
            _set_ownership(f, 1000, 1000)
        mock_chown.assert_called_once_with(f, 1000, 1000)

    def test_recursive(self, tmp_path):
        """Recursive: sets ownership on directory and all children."""
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "a.txt").touch()
        (sub / "b.txt").touch()
        with patch("os.chown") as mock_chown:
            _set_ownership(tmp_path, 0, 0, recursive=True)
        # Should chown the root, sub dir, and both files
        chowned_paths = {call[0][0] for call in mock_chown.call_args_list}
        assert tmp_path in chowned_paths
        assert sub in chowned_paths
        assert sub / "a.txt" in chowned_paths
        assert sub / "b.txt" in chowned_paths

    def test_symlink_uses_lchown(self, tmp_path):
        """Symlinks are chowned via lchown, not following to target."""
        target = tmp_path / "target.txt"
        target.write_text("x")
        link = tmp_path / "link.txt"
        link.symlink_to(target)

        with patch("os.lchown") as mock_lchown, patch("os.chown") as mock_chown:
            _set_ownership(link, 1000, 1000)

        mock_lchown.assert_called_once_with(link, 1000, 1000)
        mock_chown.assert_not_called()

    def test_recursive_symlink_uses_lchown(self, tmp_path):
        """Recursive chown uses lchown for symlinks to avoid following them."""
        sub = tmp_path / "sub"
        sub.mkdir()
        (sub / "file.txt").touch()
        (sub / "link.txt").symlink_to("file.txt")

        with patch("os.lchown") as mock_lchown, patch("os.chown") as mock_chown:
            _set_ownership(tmp_path, 0, 0, recursive=True)

        lchowned = {call[0][0] for call in mock_lchown.call_args_list}
        chowned = {call[0][0] for call in mock_chown.call_args_list}
        assert sub / "link.txt" in lchowned
        assert sub / "link.txt" not in chowned
        assert sub / "file.txt" in chowned


# ── _copy_tree ───────────────────────────────────────────────────────


class TestCopyTree:
    def test_copies_tree(self, tmp_path):
        """Copies source tree to destination."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "file.py").write_text("code")
        dst = tmp_path / "dst"
        _copy_tree(src, dst)
        assert (dst / "file.py").read_text() == "code"

    def test_excludes_patterns(self, tmp_path):
        """Excluded patterns are not copied."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "file.py").write_text("code")
        cache = src / "__pycache__"
        cache.mkdir()
        (cache / "file.pyc").write_bytes(b"\x00")
        dst = tmp_path / "dst"
        _copy_tree(src, dst, excludes={"__pycache__"})
        assert (dst / "file.py").exists()
        assert not (dst / "__pycache__").exists()

    def test_preserves_destination_only_files(self, tmp_path: Path) -> None:
        """Files at destination that don't exist in source survive the copy."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "new.py").write_text("new")

        dst = tmp_path / "dst"
        dst.mkdir()
        (dst / "runtime_data.txt").write_text("must survive")

        _copy_tree(src, dst)

        assert (dst / "new.py").read_text() == "new"
        assert (dst / "runtime_data.txt").read_text() == "must survive"

    def test_replace_removes_destination_only_files(self, tmp_path: Path) -> None:
        """Generated trees can opt into exact replacement to remove stale files."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "current.py").write_text("current")

        dst = tmp_path / "dst"
        dst.mkdir()
        (dst / "obsolete.py").write_text("obsolete")

        _copy_tree(src, dst, replace=True)

        assert (dst / "current.py").read_text() == "current"
        assert not (dst / "obsolete.py").exists()

    def test_overwrites_matching_files(self, tmp_path: Path) -> None:
        """Source files overwrite same-named destination files."""
        src = tmp_path / "src"
        src.mkdir()
        (src / "config.py").write_text("updated")

        dst = tmp_path / "dst"
        dst.mkdir()
        (dst / "config.py").write_text("old version")

        _copy_tree(src, dst)

        assert (dst / "config.py").read_text() == "updated"

    def test_excludes_nested_directories(self, tmp_path: Path) -> None:
        """Excluded directories are not descended into or copied."""
        src = tmp_path / "src"
        (src / "keep").mkdir(parents=True)
        (src / "keep" / "file.txt").write_text("kept")
        (src / "skip" / "sub").mkdir(parents=True)
        (src / "skip" / "sub" / "deep.txt").write_text("should not appear")

        dst = tmp_path / "dst"
        _copy_tree(src, dst, excludes={"skip"})

        assert (dst / "keep" / "file.txt").read_text() == "kept"
        assert not (dst / "skip").exists()


# ── _user_home ───────────────────────────────────────────────────────


class TestUserHome:
    def test_known_user(self):
        """Known user returns their actual home dir from pwd."""
        import getpass

        current = getpass.getuser()
        result = _user_home(current)
        assert Path(result).is_dir()

    def test_unknown_user_darwin(self, monkeypatch):
        """Unknown user on Darwin: returns /Users/<username>."""
        monkeypatch.setattr("kai.install.pwd.getpwnam", MagicMock(side_effect=KeyError))
        monkeypatch.setattr("kai.install.sys.platform", "darwin")
        assert _user_home("testuser") == "/Users/testuser"

    def test_unknown_user_linux(self, monkeypatch):
        """Unknown user on Linux: returns /home/<username>."""
        monkeypatch.setattr("kai.install.pwd.getpwnam", MagicMock(side_effect=KeyError))
        monkeypatch.setattr("kai.install.sys.platform", "linux")
        assert _user_home("testuser") == "/home/testuser"


# ── _generate_launcher_script ────────────────────────────────────────


class TestGenerateLauncherScript:
    def test_contains_install_dir(self):
        script = _generate_launcher_script("/opt/kai")
        assert "/opt/kai" in script

    def test_contains_webhook_port(self):
        script = _generate_launcher_script("/opt/kai", webhook_port=9090)
        assert "9090" in script

    def test_starts_with_shebang(self):
        script = _generate_launcher_script("/opt/kai")
        assert script.startswith("#!/bin/bash")

    def test_contains_signal_forwarding(self):
        script = _generate_launcher_script("/opt/kai")
        assert "trap" in script
        assert "TERM" in script

    def test_no_listener_exits_nonzero_instead_of_sleeping(self):
        """Python dying before it binds the webhook port must end the
        launcher with a non-zero exit so launchd's KeepAlive restarts
        the service (visible, throttled). Any unconditional-sleep
        fallback would leave launchd reporting state=running with no
        agent behind it and the TERM trap with nothing to signal."""
        script = _generate_launcher_script("/opt/kai")
        assert "exit 1" in script
        assert "sleep 86400" not in script

    def test_bind_poll_window_covers_slow_startups(self):
        """The port poll must outlast a healthy startup's bind time
        (15-25s on this stack; the memory subsystem loads its
        embedding model before the webhook server starts). 60
        iterations at 2s gives the 120s window the script comment
        promises; a window shorter than the bind time expires on
        every healthy boot and leaves the launcher supervising
        nothing."""
        script = _generate_launcher_script("/opt/kai")
        assert "seq 1 60" in script
        assert "sleep 2" in script

    def test_trap_installed_before_bind_poll(self):
        """The TERM trap must be armed before the launcher enters the
        bind poll: a service stop during the startup window otherwise
        exits bash alone with the default signal action and orphans
        the starting python."""
        script = _generate_launcher_script("/opt/kai")
        assert script.index("trap cleanup TERM INT") < script.index("seq 1 60")

    def test_sigterm_during_bind_poll_tears_down_the_child(self, tmp_path):
        """Behavioral check on the generated script: SIGTERM arriving
        while the launcher is still polling for the listener must
        tear the spawned python down, not exit the wrapper alone. The
        trap signals the launcher's process group, so the agent dies
        even in the window where it has no resolvable pid; a wrapper
        that exits alone leaves an orphan holding the webhook port,
        which blocks the next boot's bind. Uses a real subprocess and
        short real waits because the contract under test is bash
        signal delivery, which cannot be mocked."""
        venv_bin = tmp_path / "venv" / "bin"
        venv_bin.mkdir(parents=True)
        pid_file = tmp_path / "fake.pid"
        fake = venv_bin / "python3"
        # The fake agent records its pid, then execs a sleep (same
        # pid) without ever binding the port, pinning the launcher
        # inside its bind poll.
        fake.write_text(f"#!/bin/bash\necho $$ > {pid_file}\nexec sleep 30\n")
        fake.chmod(0o755)

        script_path = tmp_path / "run.sh"
        # Port 1 is never listened on, so the poll cannot succeed.
        script_path.write_text(_generate_launcher_script(str(tmp_path), webhook_port=1))
        script_path.chmod(0o755)

        # New session mirrors launchd: the wrapper is its own process
        # group leader, the same topology the group signal relies on.
        proc = subprocess.Popen(["/bin/bash", str(script_path)], start_new_session=True)
        try:
            deadline = time.time() + 5
            while time.time() < deadline:
                if pid_file.exists() and pid_file.read_text().strip():
                    break
                time.sleep(0.1)
            fake_pid = int(pid_file.read_text().strip())

            proc.send_signal(signal.SIGTERM)
            # The trap fires once the in-flight 2s poll sleep returns;
            # allow margin beyond that for teardown.
            deadline = time.time() + 8
            while time.time() < deadline:
                try:
                    os.kill(fake_pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.2)
            else:
                pytest.fail("fake python survived SIGTERM to the launcher")
        finally:
            # EPERM joins ESRCH here: once the wrapper has exited and
            # awaits reaping, macOS refuses killpg on the zombie-led
            # group rather than reporting it gone.
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=5)


# ── _apply_source ────────────────────────────────────────────────────


class TestApplySource:
    """
    _apply_source copies src/, pyproject.toml, optional install
    constraints, and templates/config/ into the install tree, and
    retires the dead <install>/home/.claude/ subtree (and any legacy
    IDENTITY.md) via _retire_install_home_claude. Post-#447 the install
    tree carries no CLAUDE.md; the per-user runtime
    <DATA_DIR>/home/<chat_id>/.claude/CLAUDE.md is seeded by
    _apply_migrate (eager, install-time) and backend.ensure_user_home
    (lazy, first-message fallback). The retirement helper has its own
    pinned contracts in TestRetireInstallHomeClaude below; the migration
    helper that ran before #447 is no longer called from _apply_source
    but is unit-tested directly in TestMigrateIdentityToClaudeMd.
    """

    def test_dry_run_does_not_mention_retired_template_paths(self, tmp_path, capsys):
        """Dry-run output never names the retired template-copy or seed steps."""
        src = tmp_path / "source"
        ws_claude = src / "templates" / ".claude"
        ws_claude.mkdir(parents=True)
        # Even if a stray template CLAUDE.md were to exist (it does post-
        # this PR's redo: the template is the source-of-truth for per-user
        # seeding via _apply_migrate, NOT for any install-tree copy),
        # the dry-run path must not preview an install-tree copy or seed
        # from it. The negative-shape pin guards against a regression
        # that reintroduces the retired blocks.
        (ws_claude / "CLAUDE.md").write_text("identity")
        # Pre-existing dead subtree at the install destination. The
        # cleanup helper should emit a "Would remove dead ..." line.
        install_path = tmp_path / "install"
        (install_path / "home" / ".claude").mkdir(parents=True)
        (install_path / "home" / ".claude" / "stale").write_text("stale")
        with patch("kai.install.PROJECT_ROOT", src):
            _apply_source(install_path, svc_uid=1000, svc_gid=1000, dry_run=True)
        output = capsys.readouterr().out
        # Negative shape: no install-tree copy preview of the
        # templates/.claude source directory and no "Would seed" line
        # for an install-tree destination. The retired blocks would
        # have emitted both; their absence is the contract.
        assert f"Would copy: {ws_claude}" not in output
        assert "Would seed" not in output
        # Negative shape: no IDENTITY.md migration line (the migration
        # helper is not called from _apply_source post-#447).
        assert "IDENTITY.md" not in output
        # Positive shape: the cleanup helper announces the pre-existing
        # dead subtree.
        assert "[DRY RUN] Would remove dead" in output
        # Dry run never touches disk.
        assert (install_path / "home" / ".claude").exists()

    def test_dry_run_no_templates_or_config_dir_skips_both(self, tmp_path, capsys):
        """Dry run with empty source: no template-copy lines for either subtree."""
        with patch("kai.install.PROJECT_ROOT", tmp_path):
            _apply_source(tmp_path / "install", svc_uid=1000, svc_gid=1000, dry_run=True)
        output = capsys.readouterr().out
        assert "DRY RUN" in output
        # Both subtree preview lines must be absent. The templates/.claude
        # half is structural post-#447 (no install-tree copy step exists
        # at all, so the preview cannot fire). The templates/config half
        # is conditional on the source dir existing; with tmp_path as
        # PROJECT_ROOT, neither subtree exists.
        assert "templates/.claude" not in output
        assert "templates/config" not in output

    def test_copies_templates_config(self, tmp_path):
        """templates/config/ is copied to <install>/config/."""
        src = tmp_path / "source"
        (src / "src").mkdir(parents=True)
        (src / "src" / "module.py").write_text("code")
        (src / "pyproject.toml").write_text("[project]")
        # Create the config template directory. Note: no templates/.claude/
        # in this fixture - we're isolating the config copy behavior.
        config_dir = src / "templates" / "config"
        config_dir.mkdir(parents=True)
        (config_dir / "goose-config.yaml").write_text("extensions: []")
        install = tmp_path / "install"
        install.mkdir()

        with (
            patch("kai.install.PROJECT_ROOT", src),
            patch("kai.install._copy_tree") as mock_copy,
            patch("kai.install._set_ownership") as mock_own,
            patch("shutil.copy2"),
            patch("os.chown"),
        ):
            _apply_source(install, svc_uid=1000, svc_gid=1000, dry_run=False)

        # Verify the specific templates/config/ copy call rather than relying
        # on total call count (which depends on fixture state).
        config_dst = install / "config"
        config_calls = [c for c in mock_copy.call_args_list if c[0][0] == config_dir and c[0][1] == config_dst]
        assert len(config_calls) == 1

        # Installed Python source is generated state and must be replaced
        # exactly so repository deletions remove obsolete installed modules.
        source_dir = src / "src"
        source_dst = install / "src"
        source_calls = [c for c in mock_copy.call_args_list if c[0][0] == source_dir and c[0][1] == source_dst]
        assert len(source_calls) == 1
        assert source_calls[0].kwargs["replace"] is True

        # <install>/config/ should be root-owned (static template, not runtime data)
        own_calls = [c for c in mock_own.call_args_list if c[0] == (config_dst, 0, 0) and c[1].get("recursive") is True]
        assert len(own_calls) == 1

    def test_normalizes_static_tree_modes_under_restrictive_umask(self, tmp_path):
        """Copied code remains readable when the installer caller uses umask 077."""
        project = tmp_path / "source"
        package = project / "src" / "kai" / "nested"
        package.mkdir(parents=True)
        (package / "module.py").write_text("value = 1\n")
        (project / "pyproject.toml").write_text("[project]\nname = 'kai'\n")
        config = project / "templates" / "config"
        config.mkdir(parents=True)
        (config / "goose-config.yaml").write_text("extensions: []\n")
        install = tmp_path / "install"
        install.mkdir()

        previous_umask = os.umask(0o077)
        try:
            with (
                patch("kai.install.PROJECT_ROOT", project),
                patch("kai.install._set_ownership"),
                patch("os.chown"),
            ):
                _apply_source(install, svc_uid=1000, svc_gid=1000, dry_run=False)
        finally:
            os.umask(previous_umask)

        assert stat.S_IMODE((install / "src").stat().st_mode) == 0o755
        assert stat.S_IMODE((install / "src" / "kai" / "nested").stat().st_mode) == 0o755
        assert stat.S_IMODE((install / "src" / "kai" / "nested" / "module.py").stat().st_mode) == 0o644
        assert stat.S_IMODE((install / "config").stat().st_mode) == 0o755
        assert stat.S_IMODE((install / "config" / "goose-config.yaml").stat().st_mode) == 0o644

    def test_preserves_executable_intent_and_skips_symlinks(self, tmp_path):
        tree = tmp_path / "tree"
        tree.mkdir(mode=0o700)
        executable = tree / "tool"
        executable.write_text("#!/bin/sh\n")
        executable.chmod(0o700)
        regular = tree / "data"
        regular.write_text("data\n")
        regular.chmod(0o600)
        target = tmp_path / "outside"
        target.write_text("outside\n")
        target.chmod(0o600)
        (tree / "link").symlink_to(target)

        assert _set_static_install_tree_modes(tree) is True

        assert stat.S_IMODE(tree.stat().st_mode) == 0o755
        assert stat.S_IMODE(executable.stat().st_mode) == 0o755
        assert stat.S_IMODE(regular.stat().st_mode) == 0o644
        assert stat.S_IMODE(target.stat().st_mode) == 0o600
        assert _set_static_install_tree_modes(tree) is False

    def test_dry_run_includes_templates_config(self, tmp_path, capsys):
        """Dry run names templates/config/ when it exists."""
        src = tmp_path / "source"
        config_dir = src / "templates" / "config"
        config_dir.mkdir(parents=True)
        (config_dir / "goose-config.yaml").write_text("extensions: []")
        with patch("kai.install.PROJECT_ROOT", src):
            _apply_source(tmp_path / "install", svc_uid=1000, svc_gid=1000, dry_run=True)
        output = capsys.readouterr().out
        assert "templates/config" in output
        # Should not create the destination during dry run.
        assert not (tmp_path / "install" / "config").exists()
        assert not (tmp_path / "install" / "home" / "config").exists()

    def test_copies_install_constraints(self, tmp_path):
        """requirements/constraints.txt is copied to the install tree when present."""
        src = tmp_path / "source"
        (src / "src").mkdir(parents=True)
        (src / "src" / "module.py").write_text("code")
        (src / "pyproject.toml").write_text("[project]")
        constraints_src = src / "requirements" / "constraints.txt"
        constraints_src.parent.mkdir(parents=True)
        constraints_src.write_text("aiohttp==3.13.4\n")
        install = tmp_path / "install"
        install.mkdir()

        with (
            patch("kai.install.PROJECT_ROOT", src),
            patch("kai.install._set_ownership"),
            patch("os.chown"),
        ):
            _apply_source(install, svc_uid=1000, svc_gid=1000, dry_run=False)

        constraints_dst = install / "requirements" / "constraints.txt"
        assert constraints_dst.read_text() == "aiohttp==3.13.4\n"

    def test_removes_stale_install_constraints(self, tmp_path):
        """A removed source constraints file removes the stale install copy."""
        src = tmp_path / "source"
        (src / "src").mkdir(parents=True)
        (src / "src" / "module.py").write_text("code")
        (src / "pyproject.toml").write_text("[project]")
        install = tmp_path / "install"
        install.mkdir()
        constraints_dst = install / "requirements" / "constraints.txt"
        constraints_dst.parent.mkdir(parents=True)
        constraints_dst.write_text("aiohttp==3.13.4\n")

        with (
            patch("kai.install.PROJECT_ROOT", src),
            patch("kai.install._set_ownership"),
            patch("os.chown"),
        ):
            _apply_source(install, svc_uid=1000, svc_gid=1000, dry_run=False)

        assert not constraints_dst.exists()


class TestOptionalFileChecksum:
    def test_missing_file_returns_empty_string(self, tmp_path):
        assert _optional_file_checksum(tmp_path / "missing.txt") == ""

    def test_existing_file_returns_file_checksum(self, tmp_path):
        path = tmp_path / "constraints.txt"
        path.write_text("aiohttp==3.13.4\n")
        assert _optional_file_checksum(path) == _file_checksum(path)


# ── _retire_install_home_claude ──────────────────────────────────────


class TestRetireInstallHomeClaude:
    """
    Pins the wholesale-directory cleanup of <install>/home/.claude/ and
    the legacy <install>/home/IDENTITY.md. Both paths predate the
    per-user home_workspace migration in #353; nothing in the runtime
    reads either path. Issue #447 retires both.
    """

    def test_fresh_install_no_install_home_claude_dir(self, tmp_path):
        """Fresh install: no pre-existing dirs, helper is a no-op."""
        install = tmp_path / "install"
        install.mkdir()
        _retire_install_home_claude(install, dry_run=False)
        assert not (install / "home" / ".claude").exists()
        assert not (install / "home" / "IDENTITY.md").exists()

    def test_existing_install_home_claude_dir_is_removed(self, tmp_path, capsys):
        """Existing <install>/home/.claude/ with content is removed wholesale."""
        install = tmp_path / "install"
        ws_claude = install / "home" / ".claude"
        ws_claude.mkdir(parents=True)
        (ws_claude / "CLAUDE.md").write_text("# Operator content\n")
        (ws_claude / "MEMORY.md").write_text("# Memory\n")
        (ws_claude / "skills").mkdir()
        (ws_claude / "skills" / "example.md").write_text("# skill")

        _retire_install_home_claude(install, dry_run=False)

        assert not ws_claude.exists()
        output = capsys.readouterr().out
        assert "CLAUDE.md" in output
        assert "MEMORY.md" in output
        assert "example.md" in output
        assert "Removed dead" in output
        assert "nothing reads this path post-#447" in output

    def test_legacy_identity_md_regular_file_is_removed(self, tmp_path, capsys):
        """A bare regular-file <install>/home/IDENTITY.md is removed."""
        install = tmp_path / "install"
        (install / "home").mkdir(parents=True)
        identity = install / "home" / "IDENTITY.md"
        identity.write_text("# Operator content\n")

        _retire_install_home_claude(install, dry_run=False)

        assert not identity.exists()
        output = capsys.readouterr().out
        assert "Removed legacy" in output
        assert "IDENTITY.md" in output

    def test_symlink_at_identity_md_is_removed(self, tmp_path):
        """A broken or valid symlink at IDENTITY.md is removed too."""
        install = tmp_path / "install"
        (install / "home").mkdir(parents=True)
        identity = install / "home" / "IDENTITY.md"
        identity.symlink_to("/nonexistent/broken/target")

        _retire_install_home_claude(install, dry_run=False)

        assert not identity.is_symlink()
        assert not identity.exists()

    def test_dry_run_predicts_cleanup_matches_live(self, tmp_path, capsys):
        """Dry-run / live parity: each removal line corresponds 1:1."""

        def seed_install_state(install_root: Path) -> None:
            ws_claude = install_root / "home" / ".claude"
            ws_claude.mkdir(parents=True)
            (ws_claude / "CLAUDE.md").write_text("# content\n")
            (ws_claude / "MEMORY.md").write_text("# memory\n")
            (install_root / "home" / "IDENTITY.md").write_text("# operator\n")

        install_dry = tmp_path / "install_dry"
        install_live = tmp_path / "install_live"
        seed_install_state(install_dry)
        seed_install_state(install_live)

        _retire_install_home_claude(install_dry, dry_run=True)
        dry_output = capsys.readouterr().out

        _retire_install_home_claude(install_live, dry_run=False)
        live_output = capsys.readouterr().out

        # Filter to the removal-related lines and strip the tense-marker
        # prefix so the comparison runs on the action body. Live uses
        # "Removing " for per-file (before rmtree, so a rmtree OSError
        # does not leave a past-tense lie in stdout) and "Removed " for
        # the post-rmtree summary and post-unlink IDENTITY.md line.
        # Dry-run uses a single "[DRY RUN] Would remove " prefix for
        # all three phases.
        def removal_lines(text: str, prefixes: tuple[str, ...], install_path: Path) -> list[str]:
            placeholder = "<install>"
            stripped: list[str] = []
            for line in text.splitlines():
                line = line.strip()
                for prefix in prefixes:
                    if line.startswith(prefix):
                        body = line[len(prefix) :].lstrip()
                        body = body.replace(str(install_path), placeholder)
                        stripped.append(body)
                        break
            return stripped

        dry_lines = removal_lines(dry_output, ("[DRY RUN] Would remove ",), install_dry)
        live_lines = removal_lines(live_output, ("Removing ", "Removed "), install_live)
        assert dry_lines, f"Dry-run produced no removal previews; got: {dry_output!r}"
        assert dry_lines == live_lines, f"Dry-run / live parity broken.\n  Dry:  {dry_lines}\n  Live: {live_lines}"

    def test_idempotent_on_repeated_install(self, tmp_path, capsys):
        """Second call over the cleaned state is a silent no-op."""
        install = tmp_path / "install"
        install.mkdir()

        _retire_install_home_claude(install, dry_run=False)
        capsys.readouterr()  # discard first run
        _retire_install_home_claude(install, dry_run=False)

        second_output = capsys.readouterr().out
        assert not (install / "home" / ".claude").exists()
        assert "Removed dead" not in second_output
        assert "Removed legacy" not in second_output


# ── _retire_install_home_dir ─────────────────────────────────────────


class TestRetireInstallHomeDir:
    """
    Pins the wholesale retirement of `<install>/home/` after the
    `<install>/home/config/` relocation to `<install>/config/`. Three
    cases: directory missing (no-op), empty parent (clean removal),
    and the existing-install upgrade where only the orphaned
    `home/config/goose-config.yaml` remains.
    """

    def test_no_op_when_home_dir_missing(self, tmp_path):
        """Fresh install: no <install>/home/, helper is a silent no-op."""
        install = tmp_path / "install"
        install.mkdir()
        _retire_install_home_dir(install, dry_run=False)
        assert not (install / "home").exists()

    def test_removes_empty_home_dir(self, tmp_path, capsys):
        """Empty <install>/home/ is removed cleanly with a summary line."""
        install = tmp_path / "install"
        (install / "home").mkdir(parents=True)
        _retire_install_home_dir(install, dry_run=False)
        assert not (install / "home").exists()
        output = capsys.readouterr().out
        assert "Removed retired" in output
        assert str(install / "home") in output

    def test_removes_orphan_goose_config(self, tmp_path, capsys):
        """
        Existing-install upgrade: `<install>/home/config/goose-config.yaml`
        still exists (from before the relocation). The helper removes
        the orphan file and the empty home/ parent, logging the file's
        byte size before deletion.
        """
        install = tmp_path / "install"
        old_config = install / "home" / "config" / "goose-config.yaml"
        old_config.parent.mkdir(parents=True)
        old_config.write_text("extensions: {}\n")

        _retire_install_home_dir(install, dry_run=False)

        assert not (install / "home").exists()
        output = capsys.readouterr().out
        assert "goose-config.yaml" in output
        assert "Removing" in output  # per-file pre-rmtree line
        assert "Removed retired" in output  # post-rmtree summary

    def test_dry_run_predicts_orphan_cleanup(self, tmp_path, capsys):
        """Dry-run preview names the file and the directory it would remove."""
        install = tmp_path / "install"
        old_config = install / "home" / "config" / "goose-config.yaml"
        old_config.parent.mkdir(parents=True)
        old_config.write_text("extensions: {}\n")

        _retire_install_home_dir(install, dry_run=True)

        # Dry run does not touch disk.
        assert old_config.exists()
        output = capsys.readouterr().out
        assert "[DRY RUN] Would remove" in output
        assert "goose-config.yaml" in output
        assert "[DRY RUN] Would remove retired" in output

    def test_silent_on_second_call(self, tmp_path, capsys):
        """After a successful removal, the second call's no-op path is silent."""
        install = tmp_path / "install"
        (install / "home").mkdir(parents=True)
        _retire_install_home_dir(install, dry_run=False)
        capsys.readouterr()  # discard first run
        _retire_install_home_dir(install, dry_run=False)
        assert not (install / "home").exists()
        second_output = capsys.readouterr().out
        assert "Removed retired" not in second_output

    def test_refuses_when_files_subdir_present(self, tmp_path, capsys):
        """
        Existing-install upgrade with an un-migrated `home/files/` backup
        tree: the helper must refuse to remove (the pre-DATA_DIR
        uploaded-files migration in `_apply_migrate` still reads from
        this path; destroying it before that migration runs would be
        data loss). The directory is preserved and the operator sees
        a clear refusal log naming the unexpected path.
        """
        install = tmp_path / "install"
        files_dir = install / "home" / "files"
        files_dir.mkdir(parents=True)
        (files_dir / "uploaded.png").write_bytes(b"\x89PNG")
        # Coexisting orphan goose-config does not change the outcome -
        # the unexpected files/ subdir is the blocker.
        (install / "home" / "config").mkdir()
        (install / "home" / "config" / "goose-config.yaml").write_text("extensions: {}\n")

        _retire_install_home_dir(install, dry_run=False)

        # Directory and its contents are intact.
        assert files_dir.is_dir()
        assert (files_dir / "uploaded.png").is_file()
        output = capsys.readouterr().out
        assert "Refusing to remove" in output
        assert "files" in output

    def test_refuses_when_unexpected_top_level_entry(self, tmp_path, capsys):
        """
        Any top-level entry other than `config/` blocks removal. Covers
        the future-code-change case where a new install step deposits
        something under `<install>/home/`.
        """
        install = tmp_path / "install"
        home_dir = install / "home"
        home_dir.mkdir(parents=True)
        (home_dir / "skills").mkdir()
        (home_dir / "skills" / "example.md").write_text("# skill\n")

        _retire_install_home_dir(install, dry_run=False)

        assert (home_dir / "skills").is_dir()
        output = capsys.readouterr().out
        assert "Refusing to remove" in output
        assert "skills" in output

    def test_refuses_when_unexpected_file_in_config(self, tmp_path, capsys):
        """
        `home/config/` is allowed to contain only `goose-config.yaml`.
        An extra file here also triggers refusal (an operator-placed
        config override, a future template that should land under
        `<install>/config/` instead, etc.).
        """
        install = tmp_path / "install"
        config_dir = install / "home" / "config"
        config_dir.mkdir(parents=True)
        (config_dir / "goose-config.yaml").write_text("extensions: {}\n")
        (config_dir / "claude-config.yaml").write_text("model: opus\n")

        _retire_install_home_dir(install, dry_run=False)

        assert (config_dir / "goose-config.yaml").is_file()
        assert (config_dir / "claude-config.yaml").is_file()
        output = capsys.readouterr().out
        assert "Refusing to remove" in output
        assert "claude-config.yaml" in output

    def test_dry_run_predicts_refusal(self, tmp_path, capsys):
        """Dry-run on a refusal pre-state emits the same message; no disk mutation."""
        install = tmp_path / "install"
        files_dir = install / "home" / "files"
        files_dir.mkdir(parents=True)
        (files_dir / "uploaded.png").write_bytes(b"\x89PNG")

        _retire_install_home_dir(install, dry_run=True)

        assert files_dir.is_dir()
        output = capsys.readouterr().out
        assert "[DRY RUN] Refusing to remove" in output


# ── Per-user CLAUDE.md seed (in _apply_migrate's home block) ─────────


class TestApplyMigrateManagedIdentity:
    """Install-time canonical identity migration and backend adapters."""

    def _write_users_yaml(self, path: Path, entries: list[dict]) -> None:
        path.write_text(yaml.safe_dump({"users": entries}))

    def _source(self, tmp_path: Path, *, identity: str | None = "# Kai template\n") -> Path:
        src = tmp_path / "source"
        templates = src / "templates"
        claude_templates = templates / ".claude"
        claude_templates.mkdir(parents=True)
        if identity is not None:
            (templates / "AGENTS.md").write_text(identity)
        (claude_templates / "MEMORY.md").write_text("# Memory\n")
        (claude_templates / "PREFERENCES.md").write_text("# Preferences\n")
        return src

    @staticmethod
    def _pwd():
        class _Pw:
            pw_uid = 1234
            pw_gid = 1234

        return _Pw()

    def _apply(
        self,
        tmp_path: Path,
        *,
        backend: str,
        entry_backend: str | None = None,
        dry_run: bool = False,
        source_identity: str | None = "# Kai template\n",
    ) -> tuple[Path, Path, Path]:
        src = self._source(tmp_path, identity=source_identity)
        users_yaml = tmp_path / "users.yaml"
        entry = {"telegram_id": 12345, "os_user": "alice"}
        if entry_backend is not None:
            entry["backend"] = entry_backend
        self._write_users_yaml(users_yaml, [entry])
        data_path = tmp_path / "data"
        install_path = tmp_path / "install"
        install_path.mkdir(exist_ok=True)

        with (
            patch("kai.install.PROJECT_ROOT", src),
            patch("kai.install.pwd.getpwnam", return_value=self._pwd()),
            patch("kai.install._set_ownership"),
            patch("os.chown"),
        ):
            _apply_migrate(
                data_path,
                install_path,
                svc_uid=0,
                svc_gid=0,
                dry_run=dry_run,
                users_yaml_path=users_yaml,
                default_backend=backend,
            )
        return data_path / "home" / "12345", users_yaml, src

    def test_fresh_claude_user_gets_agents_and_thin_adapter(self, tmp_path):
        home, _users, _src = self._apply(tmp_path, backend="claude")
        assert (home / "AGENTS.md").read_text() == "# Kai template\n"
        assert (home / ".claude" / "CLAUDE.md").read_text() == "@../AGENTS.md\n"
        assert stat.S_IMODE((home / "AGENTS.md").stat().st_mode) == 0o600
        assert stat.S_IMODE((home / ".claude" / "CLAUDE.md").stat().st_mode) == 0o600

    @pytest.mark.parametrize("backend", ["codex", "goose", "opencode", "pi"])
    def test_fresh_non_claude_user_gets_only_agents(self, tmp_path, backend):
        home, _users, _src = self._apply(tmp_path, backend=backend)
        assert (home / "AGENTS.md").read_text() == "# Kai template\n"
        assert not (home / ".claude" / "CLAUDE.md").exists()

    def test_explicit_user_backend_overrides_install_default(self, tmp_path):
        home, _users, _src = self._apply(
            tmp_path,
            backend="codex",
            entry_backend="claude",
        )
        assert (home / ".claude" / "CLAUDE.md").read_text() == "@../AGENTS.md\n"

    def test_unknown_user_backend_does_not_fall_back(self, tmp_path):
        with pytest.raises(ValueError, match="unknown backend"):
            self._apply(
                tmp_path,
                backend="codex",
                entry_backend="not-a-backend",
            )

        assert not (tmp_path / "data" / "home" / "12345" / "AGENTS.md").exists()

    def test_existing_claude_content_migrates_before_adapter(self, tmp_path):
        home = tmp_path / "data" / "home" / "12345"
        (home / ".claude").mkdir(parents=True)
        old_content = "# OPERATOR CUSTOMIZED\nKeep this exact text.\n"
        (home / ".claude" / "CLAUDE.md").write_text(old_content)

        migrated_home, _users, _src = self._apply(
            tmp_path,
            backend="claude",
            source_identity="# Template without migration section\n",
        )

        assert migrated_home == home
        assert (home / "AGENTS.md").read_text() == old_content
        assert (home / ".claude" / "CLAUDE.md").read_text() == "@../AGENTS.md\n"

    def test_non_claude_migration_removes_retired_copy(self, tmp_path):
        home = tmp_path / "data" / "home" / "12345"
        (home / ".claude").mkdir(parents=True)
        old_content = "# OPERATOR CUSTOMIZED\n"
        (home / ".claude" / "CLAUDE.md").write_text(old_content)

        self._apply(
            tmp_path,
            backend="pi",
            source_identity="# Template without migration section\n",
        )

        assert (home / "AGENTS.md").read_text() == old_content
        assert not (home / ".claude" / "CLAUDE.md").exists()

    def test_conflicting_customized_files_abort_without_changes(self, tmp_path):
        home = tmp_path / "data" / "home" / "12345"
        (home / ".claude").mkdir(parents=True)
        (home / "AGENTS.md").write_text("# canonical\n")
        (home / ".claude" / "CLAUDE.md").write_text("# different\n")

        with pytest.raises(RuntimeError, match="Conflicting customized identity files"):
            self._apply(tmp_path, backend="claude")

        assert (home / "AGENTS.md").read_text() == "# canonical\n"
        assert (home / ".claude" / "CLAUDE.md").read_text() == "# different\n"

    def test_symlinked_identity_surface_is_refused(self, tmp_path):
        home = tmp_path / "data" / "home" / "12345"
        home.mkdir(parents=True)
        target = tmp_path / "outside"
        target.write_text("outside\n")
        (home / "AGENTS.md").symlink_to(target)

        with pytest.raises(RuntimeError, match="Refusing symlinked canonical identity"):
            self._apply(tmp_path, backend="codex")

        assert target.read_text() == "outside\n"

    def test_symlinked_claude_directory_is_refused(self, tmp_path):
        home = tmp_path / "data" / "home" / "12345"
        home.mkdir(parents=True)
        outside = tmp_path / "outside-claude"
        outside.mkdir()
        (home / ".claude").symlink_to(outside, target_is_directory=True)

        with pytest.raises(RuntimeError, match="symlinked managed identity directory"):
            self._apply(tmp_path, backend="claude")

        assert list(outside.iterdir()) == []

    def test_dry_run_reports_migration_and_changes_nothing(self, tmp_path, capsys):
        home = tmp_path / "data" / "home" / "12345"
        (home / ".claude").mkdir(parents=True)
        old_content = "# customized\n"
        claude_path = home / ".claude" / "CLAUDE.md"
        claude_path.write_text(old_content)

        self._apply(
            tmp_path,
            backend="claude",
            dry_run=True,
            source_identity="# Template without migration section\n",
        )

        output = capsys.readouterr().out
        assert f"Would migrate {claude_path} -> {home / 'AGENTS.md'}" in output
        assert f"Would write Claude identity adapter: {claude_path}" in output
        assert not (home / "AGENTS.md").exists()
        assert claude_path.read_text() == old_content

    def test_missing_template_writes_agents_placeholder(self, tmp_path):
        home, _users, _src = self._apply(
            tmp_path,
            backend="codex",
            source_identity=None,
        )
        assert (home / "AGENTS.md").read_text() == "# Identity\n"
        assert not (home / ".claude" / "CLAUDE.md").exists()


# ── _migrate_identity_to_claude_md ───────────────────────────────────


class TestMigrateIdentityToClaudeMd:
    """
    One-shot migration helper that converts the legacy IDENTITY.md +
    CLAUDE.md symlink layout into a single regular CLAUDE.md file at
    home/.claude/CLAUDE.md. Six rows in the §3.3 migration table; one
    test per row.
    """

    def test_symlink_with_identity_md_preserves_content(self, tmp_path, capsys):
        """
        Row 1: IDENTITY.md regular file plus CLAUDE.md symlink to
        ../IDENTITY.md. The migration replaces the symlink with a
        regular file holding the IDENTITY.md content (and the same
        mode bits) and removes IDENTITY.md.
        """
        install = tmp_path / "install"
        ws_claude = install / "home" / ".claude"
        ws_claude.mkdir(parents=True)
        identity = install / "home" / "IDENTITY.md"
        identity.write_text("# Operator-customized identity\n")
        # Set a non-default mode so we can assert it survives the
        # migration. This guards against a future "drop copy2 for
        # write_bytes" regression that would silently reset to
        # 0o644 minus umask. Row 2 preserves mode via Path.replace;
        # Row 1 must match for cross-row consistency.
        identity.chmod(0o600)
        claude_md = ws_claude / "CLAUDE.md"
        claude_md.symlink_to("../IDENTITY.md")

        with patch("os.chown"):
            _migrate_identity_to_claude_md(install, svc_uid=1000, svc_gid=1000, dry_run=False)

        # Content survived; layout converted.
        assert claude_md.is_file()
        assert not claude_md.is_symlink()
        assert claude_md.read_text() == "# Operator-customized identity\n"
        assert not identity.exists()
        # Mode bits preserved (low 9 bits to ignore the file-type bits).
        assert (claude_md.stat().st_mode & 0o777) == 0o600
        # Single confirmation log line.
        assert "Migrated" in capsys.readouterr().out

    def test_identity_md_no_claude_md_moves_file(self, tmp_path, capsys):
        """
        Row 2: IDENTITY.md regular file, no CLAUDE.md sibling. The
        migration renames IDENTITY.md to CLAUDE.md (atomic on the
        same filesystem) and chowns the result.
        """
        install = tmp_path / "install"
        (install / "home").mkdir(parents=True)
        identity = install / "home" / "IDENTITY.md"
        identity.write_text("# Standalone IDENTITY.md\n")
        claude_md = install / "home" / ".claude" / "CLAUDE.md"

        with patch("os.chown"):
            _migrate_identity_to_claude_md(install, svc_uid=1000, svc_gid=1000, dry_run=False)

        assert claude_md.is_file()
        assert not claude_md.is_symlink()
        assert claude_md.read_text() == "# Standalone IDENTITY.md\n"
        assert not identity.exists()
        assert "Moved" in capsys.readouterr().out

    def test_inconsistent_keeps_claude_md_warns(self, tmp_path, capsys):
        """
        Row 3: both IDENTITY.md and CLAUDE.md exist as regular files.
        The migration keeps the CLAUDE.md content (canonical going
        forward), deletes IDENTITY.md, and emits a WARNING so the
        operator can sanity-check.
        """
        install = tmp_path / "install"
        ws_claude = install / "home" / ".claude"
        ws_claude.mkdir(parents=True)
        identity = install / "home" / "IDENTITY.md"
        identity.write_text("# Stale identity\n")
        claude_md = ws_claude / "CLAUDE.md"
        claude_md.write_text("# Canonical CLAUDE.md\n")

        with patch("os.chown"):
            _migrate_identity_to_claude_md(install, svc_uid=1000, svc_gid=1000, dry_run=False)

        # CLAUDE.md content is unchanged; IDENTITY.md is gone.
        assert claude_md.read_text() == "# Canonical CLAUDE.md\n"
        assert not identity.exists()
        assert "WARNING" in capsys.readouterr().out

    def test_broken_symlink_unlinks(self, tmp_path, capsys):
        """
        Row 4 broken-target subcase: CLAUDE.md is a symlink to a missing
        target. The migration unlinks the symlink so the seed step in
        _apply_source can populate a regular file. CLAUDE.md must not
        exist after the migration step (the seed is the integration
        test's concern).
        """
        install = tmp_path / "install"
        ws_claude = install / "home" / ".claude"
        ws_claude.mkdir(parents=True)
        claude_md = ws_claude / "CLAUDE.md"
        claude_md.symlink_to("../IDENTITY.md")  # target missing

        with patch("os.chown"):
            _migrate_identity_to_claude_md(install, svc_uid=1000, svc_gid=1000, dry_run=False)

        # is_symlink() catches a removed-target symlink that exists()
        # would miss; pin both branches.
        assert not claude_md.is_symlink()
        assert not claude_md.exists()
        assert "Removed symlink" in capsys.readouterr().out

    def test_valid_non_identity_symlink_unlinks(self, tmp_path, capsys):
        """
        Row 4 valid-target subcase: CLAUDE.md is a symlink to some
        valid path that is NOT IDENTITY.md (an exotic post-merge
        tarball-restore state). Path.exists() returns True for a valid
        symlink, so without this branch the seed step would skip and
        the install would keep a symlink pointing at unrelated content
        as the operator's identity. Migration unlinks the symlink so
        the seed step produces a clean regular file.
        """
        install = tmp_path / "install"
        ws_claude = install / "home" / ".claude"
        ws_claude.mkdir(parents=True)
        # A valid file at an unrelated path inside install/.
        decoy = install / "home" / "DECOY.md"
        decoy.write_text("# Unrelated content\n")
        claude_md = ws_claude / "CLAUDE.md"
        claude_md.symlink_to("../DECOY.md")

        with patch("os.chown"):
            _migrate_identity_to_claude_md(install, svc_uid=1000, svc_gid=1000, dry_run=False)

        assert not claude_md.is_symlink()
        assert not claude_md.exists()
        # Decoy survives; only the symlink was unlinked.
        assert decoy.is_file()
        assert "Removed symlink" in capsys.readouterr().out

    def test_already_migrated_logs_and_noops(self, tmp_path, capsys):
        """
        Row 5: CLAUDE.md regular file, no IDENTITY.md. Already
        migrated; the helper makes no changes and emits the "already
        migrated; no action" log line exactly once. This guards the
        §6 acceptance "exactly one log line" contract: re-emission
        from a sibling step would silently double the count.
        """
        install = tmp_path / "install"
        ws_claude = install / "home" / ".claude"
        ws_claude.mkdir(parents=True)
        claude_md = ws_claude / "CLAUDE.md"
        claude_md.write_text("# Already migrated content\n")

        with patch("os.chown"):
            _migrate_identity_to_claude_md(install, svc_uid=1000, svc_gid=1000, dry_run=False)

        # File unchanged.
        assert claude_md.read_text() == "# Already migrated content\n"
        # Exactly one occurrence of the expected log.
        output = capsys.readouterr().out
        assert output.count("Identity surface already migrated; no action") == 1

    def test_fresh_install_silent(self, tmp_path, capsys):
        """
        Row 6: neither IDENTITY.md nor CLAUDE.md exists. Genuine fresh
        install; this step is silent (the seed step in _apply_source
        emits its own log when it copies the template). Negative-
        presence assertion guards against accidental re-emission of the
        already-migrated log from the fresh-install path.
        """
        install = tmp_path / "install"
        # No files at all.

        with patch("os.chown"):
            _migrate_identity_to_claude_md(install, svc_uid=1000, svc_gid=1000, dry_run=False)

        output = capsys.readouterr().out
        assert "Identity surface already migrated" not in output
        assert "Migrated" not in output
        assert "Moved" not in output
        assert "Removed broken symlink" not in output

    def test_dry_run_makes_no_filesystem_changes(self, tmp_path):
        """
        Dry-run logs the intended action without touching disk. Verified
        by running the row-1 (most invasive) pre-state under dry_run and
        confirming both IDENTITY.md and the symlink survive byte-identical.
        """
        install = tmp_path / "install"
        ws_claude = install / "home" / ".claude"
        ws_claude.mkdir(parents=True)
        identity = install / "home" / "IDENTITY.md"
        identity.write_text("# Operator content\n")
        claude_md = ws_claude / "CLAUDE.md"
        claude_md.symlink_to("../IDENTITY.md")

        _migrate_identity_to_claude_md(install, svc_uid=1000, svc_gid=1000, dry_run=True)

        # Layout still original; no migration applied.
        assert identity.is_file()
        assert identity.read_text() == "# Operator content\n"
        assert claude_md.is_symlink()
        assert os.readlink(claude_md) == "../IDENTITY.md"


# ── _apply_models ────────────────────────────────────────────────────


class TestApplyModels:
    def test_no_models_dir(self, tmp_path):
        """No models directory: returns early."""
        with patch("kai.install.PROJECT_ROOT", tmp_path):
            _apply_models(tmp_path / "install", dry_run=False)
        # No exception, no output

    def test_empty_models_dir(self, tmp_path):
        """Empty models directory: returns early."""
        (tmp_path / "models").mkdir()
        with patch("kai.install.PROJECT_ROOT", tmp_path):
            _apply_models(tmp_path / "install", dry_run=False)

    def test_dry_run(self, tmp_path, capsys):
        """Dry run with models: prints message."""
        models = tmp_path / "models"
        models.mkdir()
        (models / "model.bin").touch()
        with patch("kai.install.PROJECT_ROOT", tmp_path):
            _apply_models(tmp_path / "install", dry_run=True)
        assert "DRY RUN" in capsys.readouterr().out

    def test_actual(self, tmp_path):
        """Actual: calls _copy_tree and _set_ownership."""
        models = tmp_path / "models"
        models.mkdir()
        (models / "model.bin").touch()
        with (
            patch("kai.install.PROJECT_ROOT", tmp_path),
            patch("kai.install._copy_tree") as mock_copy,
            patch("kai.install._set_ownership") as mock_own,
        ):
            _apply_models(tmp_path / "install", dry_run=False)
        mock_copy.assert_called_once()
        mock_own.assert_called_once()


# ── _apply_secrets dry run ───────────────────────────────────────────


class TestApplySecretsDryRun:
    def test_dry_run(self, capsys):
        """Dry run: prints message, doesn't write files."""
        _apply_secrets({"TELEGRAM_BOT_TOKEN": "test"}, dry_run=True)
        output = capsys.readouterr().out
        assert "DRY RUN" in output
        assert "env" in output

    def test_dry_run_users_yaml_staging_path_previewed(self, tmp_path, capsys):
        """Dry run with a staged users.yaml previews the copy step.

        Pins the dry-run contract for the canonical-users-yaml flow:
        when the wizard recorded `users_yaml_staging_path` and the
        file is present, apply prints the source -> destination line
        without touching the filesystem.
        """
        staging = tmp_path / "users.yaml"
        staging.write_text("users: []\n")
        _apply_secrets(
            {"TELEGRAM_BOT_TOKEN": "test"},
            dry_run=True,
            users_yaml_staging_path=str(staging),
        )
        output = capsys.readouterr().out
        assert str(staging) in output
        assert str(kai.install.USERS_YAML) in output

    def test_dry_run_previews_retained_users_yaml_metadata_repair(self, tmp_path, monkeypatch, capsys):
        """An update without staging still previews the mandatory mode/owner repair."""
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users: []\n")
        monkeypatch.setattr("kai.install.USERS_YAML", users_yaml)

        _apply_secrets({"TELEGRAM_BOT_TOKEN": "test"}, dry_run=True)

        output = capsys.readouterr().out
        assert f"Would secure existing: {users_yaml}" in output
        assert "mode 0600, root-owned" in output

    def test_dry_run_previews_optional_protected_yaml_configs(self, tmp_path, monkeypatch, capsys):
        """Dry run previews every optional YAML file runtime can load from /etc/kai."""
        for name in ("services.yaml", "workspaces.yaml", "memory-projects.yaml"):
            (tmp_path / name).write_text("{}\n")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)

        _apply_secrets({"TELEGRAM_BOT_TOKEN": "test"}, dry_run=True)

        output = capsys.readouterr().out
        assert "/etc/kai/services.yaml" in output
        assert "/etc/kai/workspaces.yaml" in output
        assert "/etc/kai/memory-projects.yaml" in output

    def test_dry_run_prefers_staged_optional_protected_yaml_configs(self, tmp_path, monkeypatch, capsys):
        """Dry run shows staged optional YAML sources when install.conf recorded them."""
        project = tmp_path / "project"
        stage = tmp_path / "stage"
        project.mkdir()
        stage.mkdir()
        (project / "services.yaml").write_text("project: true\n")
        staged_services = stage / "services.yaml"
        staged_services.write_text("staged: true\n")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", project)

        _apply_secrets(
            {"TELEGRAM_BOT_TOKEN": "test"},
            dry_run=True,
            protected_yaml_staging_paths={"services.yaml": str(staged_services)},
        )

        output = capsys.readouterr().out
        assert f"{staged_services} -> /etc/kai/services.yaml" in output
        assert f"{project / 'services.yaml'} -> /etc/kai/services.yaml" not in output


class TestIndependentRuntimePolicy:
    def test_migration_preserves_effective_workspace_policy(self, tmp_path):
        home = tmp_path / "home"
        base = tmp_path / "projects"
        first = tmp_path / "first"
        second = tmp_path / "second"
        for path in (home, base, first, second):
            path.mkdir()
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text(
            yaml.safe_dump(
                {
                    "users": [
                        {
                            "telegram_id": 101,
                            "name": "Daniel",
                            "role": "admin",
                            "backend": "codex",
                            "home_workspace": str(home),
                            "workspace_base": str(base),
                            "allowed_workspaces": [str(first), str(second), str(first)],
                        }
                    ]
                },
                sort_keys=False,
            )
        )

        rendered = _build_migrated_runtime_profiles(
            users_yaml,
            registry_entries={"codex": {"allowed_models": ["gpt-5.5"]}},
            defaults=kai.install._RuntimePolicyDefaults(
                backend="codex",
                provider="openai",
                model="gpt-5.5",
                timeout_seconds=120,
            ),
        )

        profile = next(iter(yaml.safe_load(rendered)["runtime_profiles"].values()))
        assert profile["home_workspace"] == str(home.resolve())
        assert profile["workspace_base"] == str(base.resolve())
        assert profile["allowed_workspaces"] == [str(first.resolve()), str(second.resolve())]

    def test_migration_preserves_unavailable_absolute_paths(self, tmp_path):
        unavailable = (tmp_path / "later-mounted").resolve()
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text(
            yaml.safe_dump(
                {
                    "users": [
                        {
                            "telegram_id": 101,
                            "name": "Daniel",
                            "role": "admin",
                            "backend": "codex",
                            "workspace_base": str(unavailable),
                            "allowed_workspaces": [str(unavailable)],
                        }
                    ]
                },
                sort_keys=False,
            )
        )

        rendered = _build_migrated_runtime_profiles(
            users_yaml,
            registry_entries={"codex": {"allowed_models": ["gpt-5.5"]}},
            defaults=kai.install._RuntimePolicyDefaults(
                backend="codex",
                provider="openai",
                model="gpt-5.5",
                timeout_seconds=120,
            ),
        )

        profile = next(iter(yaml.safe_load(rendered)["runtime_profiles"].values()))
        assert profile["workspace_base"] == str(unavailable)
        assert profile["allowed_workspaces"] == [str(unavailable)]

    def test_migration_rejects_relative_workspace_paths(self, tmp_path):
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text(
            """users:
  - telegram_id: 101
    name: Daniel
    role: admin
    backend: codex
    workspace_base: relative/projects
"""
        )

        with pytest.raises(ValueError, match="workspace_base must be an absolute path"):
            _build_migrated_runtime_profiles(
                users_yaml,
                registry_entries={"codex": {"allowed_models": ["gpt-5.5"]}},
                defaults=kai.install._RuntimePolicyDefaults(
                    backend="codex",
                    provider="openai",
                    model="gpt-5.5",
                    timeout_seconds=120,
                ),
            )

    def test_migration_preserves_existing_profile_ids_and_backend_choices(self, tmp_path, monkeypatch):
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text(
            """users:
  - telegram_id: 101
    name: Daniel
    role: admin
    os_user: daniel
    backend: codex
    allowed_services:
      - perplexity
      - ""
      - "*"
      - invalid/path
      - perplexity
      - 7
  - telegram_id: 202
    name: Scott
    role: user
    os_user: sellison
    backend: claude
"""
        )
        monkeypatch.setattr(
            "kai.install._discover_backend_commands",
            lambda _service_user: {"codex": "/usr/local/bin/codex", "claude": "/usr/local/bin/claude"},
        )

        rendered = _build_migrated_runtime_profiles(
            users_yaml,
            registry_entries={
                "codex": {"allowed_models": sorted(kai.install.CODEX_MODELS)},
                "claude": {"allowed_models": ["haiku", "opus", "sonnet", "claude-*"]},
            },
            defaults=kai.install._RuntimePolicyDefaults(
                backend="codex",
                provider="openai",
                model="gpt-5.5",
                timeout_seconds=120,
            ),
        )
        document = yaml.safe_load(rendered)

        daniel_id = str(kai.install.runtime_profile_id_for_config_id(101))
        scott_id = str(kai.install.runtime_profile_id_for_config_id(202))
        assert document["version"] == 1
        assert document["runtime_profiles"][daniel_id] == {
            "display_name": "Daniel",
            "compatibility_runtime_config_id": 101,
            "backend": "codex",
            "provider": "openai",
            "model": "gpt-5.5",
            "timeout_seconds": 120,
            "allowed_services": ["perplexity"],
            "home_workspace": None,
            "workspace_base": None,
            "allowed_workspaces": [],
            "os_user": "daniel",
        }
        assert document["runtime_profiles"][scott_id]["backend"] == "claude"
        assert document["runtime_profiles"][scott_id]["provider"] == "anthropic"
        assert document["runtime_profiles"][scott_id]["model"] == "sonnet"
        assert document["runtime_profiles"][scott_id]["timeout_seconds"] == 120
        assert document["runtime_profiles"][scott_id]["allowed_services"] == []
        assert document["runtime_profiles"][scott_id]["home_workspace"] is None
        assert document["runtime_profiles"][scott_id]["workspace_base"] is None
        assert document["runtime_profiles"][scott_id]["allowed_workspaces"] == []

    def test_migration_validates_against_registry_being_installed(self, tmp_path, monkeypatch):
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text(
            """users:
  - telegram_id: 101
    name: Daniel
    role: admin
    backend: codex
    model: gpt-5.6-sol
"""
        )
        old_registry = tmp_path / "backends.yaml"
        old_registry.write_text(
            """version: 1
backends:
  codex:
    driver: codex
    runtime: local_process
    command: /usr/local/bin/codex
    allowed_models:
      - gpt-5.5
"""
        )
        monkeypatch.setenv("KAI_BACKENDS_YAML", str(old_registry))

        rendered = _build_migrated_runtime_profiles(
            users_yaml,
            registry_entries={
                "codex": {
                    "command": "/usr/local/bin/codex",
                    "allowed_models": ["gpt-5.5", "gpt-5.6-sol"],
                }
            },
            defaults=kai.install._RuntimePolicyDefaults(
                backend="codex",
                provider="openai",
                model="gpt-5.5",
                timeout_seconds=120,
            ),
        )

        profile = next(iter(yaml.safe_load(rendered)["runtime_profiles"].values()))
        assert profile["model"] == "gpt-5.6-sol"

    def test_status_reports_only_profile_count_and_backend_ids(self, tmp_path):
        profile_id = str(kai.install.runtime_profile_id_for_config_id(101))
        policy = tmp_path / "runtime-profiles.yaml"
        policy.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "runtime_profiles": {
                        profile_id: {
                            "display_name": "Secret display name",
                            "compatibility_runtime_config_id": 101,
                            "backend": "codex",
                            "provider": "openai",
                            "model": "gpt-5.5",
                            "timeout_seconds": 120,
                            "allowed_services": [],
                            "home_workspace": None,
                            "workspace_base": None,
                            "allowed_workspaces": [],
                        }
                    },
                }
            )
        )
        backends = tmp_path / "backends.yaml"
        backends.write_text("version: 1\nbackends:\n  codex: {}\n")

        status = _runtime_policy_status(policy, backends)

        assert status == "Workshop runtime policy: initialized; profiles=1, backends=codex"
        assert "Secret display name" not in status
        assert "101" not in status

    def test_status_fails_closed_when_policy_references_unregistered_backend(self, tmp_path):
        profile_id = str(kai.install.runtime_profile_id_for_config_id(101))
        policy = tmp_path / "runtime-profiles.yaml"
        policy.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "runtime_profiles": {
                        profile_id: {
                            "display_name": "Daniel",
                            "compatibility_runtime_config_id": 101,
                            "backend": "codex",
                            "provider": "openai",
                        }
                    },
                }
            )
        )
        backends = tmp_path / "backends.yaml"
        backends.write_text("version: 1\nbackends:\n  pi: {}\n")

        assert _runtime_policy_status(policy, backends).startswith("Workshop runtime policy: INVALID")

    def test_apply_initializes_root_private_policy(self, tmp_path, monkeypatch):
        policy = tmp_path / "runtime-profiles.yaml"
        monkeypatch.setattr("kai.install.RUNTIME_PROFILES_YAML", policy)
        chowned: list[tuple[Path, int, int]] = []
        monkeypatch.setattr("kai.install.os.chown", lambda path, uid, gid: chowned.append((path, uid, gid)))

        _apply_runtime_policy("initialize", "version: 1\nruntime_profiles: {}\n", dry_run=False)

        assert policy.read_text() == "version: 1\nruntime_profiles: {}\n"
        assert stat.S_IMODE(policy.stat().st_mode) == 0o600
        assert chowned == [(policy, 0, 0)]

    def test_apply_preserves_existing_policy_content(self, tmp_path, monkeypatch):
        policy = tmp_path / "runtime-profiles.yaml"
        policy.write_text("operator-authored\n")
        monkeypatch.setattr("kai.install.RUNTIME_PROFILES_YAML", policy)
        monkeypatch.setattr("kai.install.os.chown", lambda *args: None)

        _apply_runtime_policy("preserve", "replacement-must-not-land\n", dry_run=False)

        assert policy.read_text() == "operator-authored\n"

    def test_apply_writes_backward_compatible_policy_enrichment(self, tmp_path, monkeypatch):
        policy = tmp_path / "runtime-profiles.yaml"
        policy.write_text("old-policy\n")
        monkeypatch.setattr("kai.install.RUNTIME_PROFILES_YAML", policy)
        monkeypatch.setattr("kai.install.os.chown", lambda *args: None)

        _apply_runtime_policy("upgrade", "enriched-policy\n", dry_run=False)

        assert policy.read_text() == "enriched-policy\n"

    def test_apply_plan_enriches_existing_policy_without_replacing_profile_identity(
        self,
        tmp_path,
        monkeypatch,
    ):
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text(
            """users:
  - telegram_id: 101
    name: Daniel
    role: admin
    os_user: daniel
    backend: codex
    model: gpt-5.6-sol
    timeout: 345
    allowed_services:
      - perplexity
"""
        )
        profile_id = str(kai.install.runtime_profile_id_for_config_id(101))
        policy = tmp_path / "runtime-profiles.yaml"
        policy.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "runtime_profiles": {
                        profile_id: {
                            "display_name": "Operator display name",
                            "compatibility_runtime_config_id": 101,
                            "os_user": "daniel",
                            "backend": "codex",
                            "provider": "openai",
                        }
                    },
                },
                sort_keys=False,
            )
        )
        monkeypatch.setattr("kai.install.RUNTIME_PROFILES_YAML", policy)
        monkeypatch.setattr(
            "kai.install._discover_backend_commands",
            lambda _service_user: {"codex": "/usr/local/bin/codex"},
        )
        monkeypatch.setattr(
            "kai.install._backend_registry_entries",
            lambda *args: {
                "codex": {
                    "allowed_models": ["gpt-5.5", "gpt-5.6-sol"],
                }
            },
        )

        action, content, profiles = _runtime_policy_apply_plan(
            "kai",
            {
                "DEFAULT_PROVIDER": "openai",
                "DEFAULT_TIMEOUT": "120",
            },
            users_yaml,
        )

        document = yaml.safe_load(content)
        assert action == "upgrade"
        assert list(document["runtime_profiles"]) == [profile_id]
        assert document["runtime_profiles"][profile_id]["display_name"] == "Operator display name"
        assert document["runtime_profiles"][profile_id]["model"] == "gpt-5.6-sol"
        assert document["runtime_profiles"][profile_id]["timeout_seconds"] == 345
        assert document["runtime_profiles"][profile_id]["allowed_services"] == ["perplexity"]
        assert profiles.resolve(profile_id).runtime_config_id == 101

        policy.write_text(content)
        next_action, next_content, next_profiles = _runtime_policy_apply_plan(
            "kai",
            {
                "DEFAULT_PROVIDER": "openai",
                "DEFAULT_TIMEOUT": "120",
            },
            users_yaml,
        )

        assert next_action == "preserve"
        assert next_content == content
        assert next_profiles.resolve(profile_id).allowed_services == ("perplexity",)

    def test_existing_policy_must_cover_every_migrated_profile(self, tmp_path, monkeypatch):
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users:\n  - telegram_id: 101\n    name: Daniel\n    role: admin\n")
        policy = tmp_path / "runtime-profiles.yaml"
        other_id = str(kai.install.runtime_profile_id_for_config_id(202))
        policy.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "runtime_profiles": {
                        other_id: {
                            "display_name": "Other",
                            "compatibility_runtime_config_id": 202,
                            "backend": "codex",
                            "provider": "openai",
                        }
                    },
                }
            )
        )
        monkeypatch.setattr("kai.install.RUNTIME_PROFILES_YAML", policy)
        monkeypatch.setattr(
            "kai.install._backend_registry_entries",
            lambda *args: {"codex": {}},
        )

        with pytest.raises(ValueError, match="no preserved profile"):
            _runtime_policy_apply_plan(
                "kai",
                {"DEFAULT_BACKEND": "codex", "DEFAULT_PROVIDER": "openai"},
                users_yaml,
            )

    def test_existing_policy_accepts_migrated_service_scope_reordering(self, tmp_path, monkeypatch):
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text(
            """users:
  - telegram_id: 101
    name: Daniel
    role: admin
    backend: codex
    model: gpt-5.5
    allowed_services:
      - perplexity
      - weather
"""
        )
        profile_id = str(kai.install.runtime_profile_id_for_config_id(101))
        policy = tmp_path / "runtime-profiles.yaml"
        policy.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "runtime_profiles": {
                        profile_id: {
                            "display_name": "Daniel",
                            "compatibility_runtime_config_id": 101,
                            "backend": "codex",
                            "provider": "openai",
                            "model": "gpt-5.5",
                            "timeout_seconds": 120,
                            "allowed_services": ["weather", "perplexity"],
                            "home_workspace": None,
                            "workspace_base": None,
                            "allowed_workspaces": [],
                        }
                    },
                },
                sort_keys=False,
            )
        )
        monkeypatch.setattr("kai.install.RUNTIME_PROFILES_YAML", policy)
        monkeypatch.setattr(
            "kai.install._backend_registry_entries",
            lambda *args: {"codex": {"allowed_models": ["gpt-5.5"]}},
        )

        action, content, profiles = _runtime_policy_apply_plan(
            "kai",
            {
                "DEFAULT_BACKEND": "codex",
                "DEFAULT_PROVIDER": "openai",
                "DEFAULT_MODEL": "gpt-5.5",
                "DEFAULT_TIMEOUT": "120",
            },
            users_yaml,
        )

        assert action == "preserve"
        assert content == policy.read_text()
        assert profiles.resolve(profile_id).allowed_services == ("weather", "perplexity")


class TestApplyBackendRegistry:
    def test_build_registry_uses_discovered_global_commands(self, monkeypatch):
        monkeypatch.setattr(
            "kai.install._discover_backend_commands",
            lambda service_user: {
                "claude": "/global/claude",
                "codex": "/global/codex",
                "opencode": "/global/opencode",
                "goose": "/global/goose",
                "pi": "/global/pi",
            },
        )
        rendered = kai.install._build_backend_registry(
            "kai",
            {
                "DEFAULT_BACKEND": "codex",
                "CLAUDE_BIN": "/custom/claude",
                "CODEX_BIN": "/custom/codex",
                "OPENCODE_BIN": "/custom/opencode",
                "GOOSE_BIN": "/custom/goose",
            },
        )
        data = yaml.safe_load(rendered)

        assert data["backends"]["claude"]["command"] == "/global/claude"
        assert data["backends"]["codex"]["command"] == "/global/codex"
        assert data["backends"]["opencode"]["command"] == "/global/opencode"
        assert data["backends"]["goose"]["command"] == "/global/goose"
        assert data["backends"]["pi"]["command"] == "/global/pi"
        assert "allowed_models" not in data["backends"]["pi"]
        assert data["default_backend"] == "codex"
        assert "gpt-5.6-sol" in data["backends"]["codex"]["allowed_models"]
        assert "fable" in data["backends"]["claude"]["allowed_models"]
        assert "claude-*" in data["backends"]["claude"]["allowed_models"]

    def test_registry_and_sudoers_use_same_discovered_backend_paths(self, monkeypatch):
        monkeypatch.setattr(
            "kai.install._discover_backend_commands",
            lambda service_user: {
                "claude": "/registry/claude",
                "codex": "/registry/codex",
                "opencode": "/registry/opencode",
                "goose": "/registry/goose",
                "pi": "/registry/pi",
            },
        )
        env = {
            "DEFAULT_BACKEND": "codex",
            "CLAUDE_BIN": "/ignored/claude",
            "CODEX_BIN": "/ignored/codex",
            "OPENCODE_BIN": "/ignored/opencode",
            "GOOSE_BIN": "/ignored/goose",
        }

        registry = yaml.safe_load(kai.install._build_backend_registry("kai", env))
        commands = {backend: entry["command"] for backend, entry in registry["backends"].items()}
        sudoers = _generate_sudoers(
            "kai",
            os_users=["alice"],
            claude_bin=commands["claude"],
            codex_bin=commands["codex"],
            opencode_bin=commands["opencode"],
            goose_bin=commands["goose"],
            pi_bin=commands["pi"],
        )

        for backend in ("claude", "codex", "opencode", "goose", "pi"):
            command = registry["backends"][backend]["command"]
            assert f"kai ALL=(alice) SETENV: NOPASSWD: {command}" in sudoers

    def test_dry_run_prints_registry_write(self, capsys, monkeypatch):
        monkeypatch.setattr(
            "kai.install._discover_backend_commands",
            lambda service_user: {"claude": "/global/claude"},
        )
        _apply_backend_registry("kai", {}, dry_run=True)
        output = capsys.readouterr().out
        assert "backends.yaml" in output
        assert "0644" in output

    def test_command_trust_check_reports_agent_owned_executable(self, tmp_path):
        executable = tmp_path / "codex"
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o555)
        username = pwd.getpwuid(os.getuid()).pw_name

        issues = _backend_command_trust_issues(str(executable), username)

        assert any(f"resolved executable {executable} is owned by {username}" in issue for issue in issues)

    def test_command_trust_check_reports_replaceable_registered_symlink(self, tmp_path):
        target_dir = tmp_path / "target"
        target_dir.mkdir()
        executable = target_dir / "codex"
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o555)
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        registered = bin_dir / "codex"
        registered.symlink_to(executable)
        username = pwd.getpwuid(os.getuid()).pw_name

        issues = _backend_command_trust_issues(str(registered), username)

        assert f"registered path can be replaced through {bin_dir}" in issues

    def test_command_trust_check_accepts_path_unwritable_to_identity(self, tmp_path, monkeypatch):
        import types

        executable = tmp_path / "codex"
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o555)
        fake_uid = os.getuid() + 100_000
        fake_gid = os.getgid() + 100_000
        monkeypatch.setattr(
            "kai.install.pwd.getpwnam",
            lambda username: types.SimpleNamespace(pw_uid=fake_uid, pw_gid=fake_gid),
        )
        monkeypatch.setattr("kai.install.os.getgrouplist", lambda username, gid: [gid])

        assert _backend_command_trust_issues(str(executable), "isolated-agent") == ()

    def test_dry_run_warns_for_agent_writable_backend_command(self, tmp_path, capsys, monkeypatch):
        executable = tmp_path / "codex"
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)
        username = pwd.getpwuid(os.getuid()).pw_name
        monkeypatch.setattr(
            "kai.install._discover_backend_commands",
            lambda service_user: {"codex": str(executable)},
        )

        _apply_backend_registry(
            username,
            {"DEFAULT_BACKEND": "codex"},
            dry_run=True,
            users_yaml_path=tmp_path / "absent-users.yaml",
        )

        captured = capsys.readouterr()
        assert "Would write" in captured.out
        assert "local-process backend executable trust is limited" in captured.err
        assert f"codex: {executable} can be modified by agent OS user {username}" in captured.err
        assert "trusted-host compatibility runtime" in captured.err

    def test_configured_backend_must_be_discovered(self, monkeypatch):
        monkeypatch.setattr("kai.install._discover_backend_commands", lambda service_user: {"codex": "/global/codex"})

        with pytest.raises(SystemExit, match="Configured backend\\(s\\) are not installed globally: claude"):
            kai.install._build_backend_registry("kai", {"DEFAULT_BACKEND": "claude"})


# ── _apply_secrets staging copy precedence ───────────────────────────


class TestApplySecretsUsersYamlStaging:
    """Precedence rules for the new `users_yaml_staging_path` parameter.

    The presence of a non-empty path plus an existing file on disk is
    the entire signal: anything else (None, empty string, missing file)
    silently skips the copy.
    """

    @staticmethod
    def _no_other_yamls(monkeypatch, tmp_path):
        """Make optional protected YAML configs absent from PROJECT_ROOT.

        Keeps the test focused on the users.yaml staging path; the
        legacy project-tree copy for the other config files is out of
        scope here.
        """
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)

    @staticmethod
    def _intercept_filesystem(monkeypatch):
        """Capture copy2 / chmod / chown calls; suppress real env file write.

        Returns three call-log lists the test can assert against, plus
        a `Path.write_text` no-op that lets `_apply_secrets` skip the
        actual `/etc/kai/env` write (the test process is unprivileged).
        """
        copied: list[tuple[str, str]] = []
        chmodded: list[tuple[str, int]] = []
        chowned: list[tuple[str, int, int]] = []
        monkeypatch.setattr("shutil.copy2", lambda src, dst: copied.append((str(src), str(dst))))
        monkeypatch.setattr("os.chmod", lambda path, mode: chmodded.append((str(path), mode)))
        monkeypatch.setattr("os.chown", lambda path, uid, gid: chowned.append((str(path), uid, gid)))
        real_write_text = Path.write_text

        def _maybe_write(self, *args, **kwargs):
            if str(self).startswith("/etc/kai/"):
                return None
            return real_write_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", _maybe_write)
        return copied, chmodded, chowned

    def test_copies_when_path_set_and_file_exists(self, tmp_path, monkeypatch):
        """Non-empty path + existing file -> copy + 0600 + root-owned."""
        self._no_other_yamls(monkeypatch, tmp_path)
        copied, chmodded, chowned = self._intercept_filesystem(monkeypatch)
        staging = tmp_path / "staged-users.yaml"
        staging.write_text("users:\n  - telegram_id: 1\n    name: alice\n")

        _apply_secrets(
            {"TELEGRAM_BOT_TOKEN": "test"},
            dry_run=False,
            users_yaml_staging_path=str(staging),
        )

        users_yaml_dst = str(kai.install.USERS_YAML)
        users_copies = [(src, dst) for src, dst in copied if dst == users_yaml_dst]
        assert users_copies == [(str(staging), users_yaml_dst)]
        users_modes = [m for p, m in chmodded if p == users_yaml_dst]
        assert 0o600 in users_modes
        users_owners = [(uid, gid) for p, uid, gid in chowned if p == users_yaml_dst]
        assert (0, 0) in users_owners

    def test_retained_users_yaml_is_secured_without_copy(self, tmp_path, monkeypatch, capsys):
        """An existing canonical file is root-owned 0600 even without staging."""
        self._no_other_yamls(monkeypatch, tmp_path)
        users_yaml = tmp_path / "installed-users.yaml"
        users_yaml.write_text("users: []\n")
        monkeypatch.setattr("kai.install.USERS_YAML", users_yaml)
        copied, chmodded, chowned = self._intercept_filesystem(monkeypatch)

        _apply_secrets({"TELEGRAM_BOT_TOKEN": "test"}, dry_run=False, users_yaml_staging_path=None)

        assert not any(dst == str(users_yaml) for _src, dst in copied)
        assert (str(users_yaml), 0o600) in chmodded
        assert (str(users_yaml), 0, 0) in chowned
        assert f"Secured existing {users_yaml}" in capsys.readouterr().out

    def test_copies_optional_protected_yaml_configs(self, tmp_path, monkeypatch):
        """services/workspaces/memory-projects are installed as root-owned 0600 config."""
        for name in ("services.yaml", "workspaces.yaml", "memory-projects.yaml"):
            (tmp_path / name).write_text("{}\n")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        copied, chmodded, chowned = self._intercept_filesystem(monkeypatch)

        _apply_secrets({"TELEGRAM_BOT_TOKEN": "test"}, dry_run=False, users_yaml_staging_path=None)

        for name in ("services.yaml", "workspaces.yaml", "memory-projects.yaml"):
            dst = f"/etc/kai/{name}"
            assert (str(tmp_path / name), dst) in copied
            assert (dst, 0o600) in chmodded
            assert (dst, 0, 0) in chowned

    def test_staged_optional_protected_yaml_configs_win_over_project_tree(self, tmp_path, monkeypatch):
        """Recorded staging paths are the privileged-apply source of truth."""
        project = tmp_path / "project"
        stage = tmp_path / "stage"
        project.mkdir()
        stage.mkdir()
        (project / "services.yaml").write_text("project: true\n")
        staged_services = stage / "services.yaml"
        staged_services.write_text("staged: true\n")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", project)
        copied, chmodded, chowned = self._intercept_filesystem(monkeypatch)

        _apply_secrets(
            {"TELEGRAM_BOT_TOKEN": "test"},
            dry_run=False,
            protected_yaml_staging_paths={"services.yaml": str(staged_services)},
        )

        assert (str(staged_services), "/etc/kai/services.yaml") in copied
        assert (str(project / "services.yaml"), "/etc/kai/services.yaml") not in copied
        assert ("/etc/kai/services.yaml", 0o600) in chmodded
        assert ("/etc/kai/services.yaml", 0, 0) in chowned

    def test_skips_when_path_is_none(self, tmp_path, monkeypatch):
        """No staging path -> no users.yaml copy."""
        self._no_other_yamls(monkeypatch, tmp_path)
        copied, _, _ = self._intercept_filesystem(monkeypatch)

        _apply_secrets({"TELEGRAM_BOT_TOKEN": "test"}, dry_run=False, users_yaml_staging_path=None)

        assert not any(dst.endswith("users.yaml") for _src, dst in copied)

    def test_skips_when_path_is_empty_string(self, tmp_path, monkeypatch):
        """Empty string -> treated as None; no copy."""
        self._no_other_yamls(monkeypatch, tmp_path)
        copied, _, _ = self._intercept_filesystem(monkeypatch)

        _apply_secrets({"TELEGRAM_BOT_TOKEN": "test"}, dry_run=False, users_yaml_staging_path="")

        assert not any(dst.endswith("users.yaml") for _src, dst in copied)

    def test_skips_when_path_set_but_file_missing(self, tmp_path, monkeypatch):
        """Non-empty path that points at a missing file -> no copy, no error.

        A hand-edited install.conf with a stale staging path silently
        skips rather than failing the apply. The defensive check
        protects the operator from a partial-install failure mode.
        """
        self._no_other_yamls(monkeypatch, tmp_path)
        copied, _, _ = self._intercept_filesystem(monkeypatch)

        _apply_secrets(
            {"TELEGRAM_BOT_TOKEN": "test"},
            dry_run=False,
            users_yaml_staging_path=str(tmp_path / "does-not-exist.yaml"),
        )

        assert not any(dst.endswith("users.yaml") for _src, dst in copied)


# ── _strip_install_conf_keys helper ──────────────────────────────────


class TestStripInstallConfKeys:
    """Unit coverage for the install.conf top-level key remover used
    by `_cmd_apply` to drop the one-shot `users_yaml_staging_path`
    after a successful apply.
    """

    def test_strips_named_key_only(self, tmp_path, monkeypatch):
        """Targeted key removed; siblings preserved; mode stays 0600."""
        from kai.install import _strip_install_conf_keys

        conf_path = tmp_path / "install.conf"
        original = {
            "version": 1,
            "install_dir": "/opt/kai",
            "users_yaml_staging_path": "/tmp/staged",
            "env": {"TELEGRAM_BOT_TOKEN": "tok", "DEFAULT_BACKEND": "claude"},
        }
        conf_path.write_text(json.dumps(original, indent=2) + "\n")
        os.chmod(conf_path, 0o600)
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)

        _strip_install_conf_keys("users_yaml_staging_path")

        rewritten = json.loads(conf_path.read_text())
        assert "users_yaml_staging_path" not in rewritten
        assert rewritten["version"] == 1
        assert rewritten["install_dir"] == "/opt/kai"
        assert rewritten["env"] == original["env"]
        assert stat.S_IMODE(conf_path.stat().st_mode) == 0o600

    def test_idempotent_when_key_absent(self, tmp_path, monkeypatch):
        """A second call (or a call against a conf that never had the
        key) is a no-op rather than an error.
        """
        from kai.install import _strip_install_conf_keys

        conf_path = tmp_path / "install.conf"
        conf_path.write_text(json.dumps({"version": 1, "env": {}}, indent=2) + "\n")
        os.chmod(conf_path, 0o600)
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)

        _strip_install_conf_keys("users_yaml_staging_path")
        _strip_install_conf_keys("users_yaml_staging_path")

        rewritten = json.loads(conf_path.read_text())
        assert rewritten == {"version": 1, "env": {}}

    def test_missing_conf_is_noop(self, tmp_path, monkeypatch):
        """No install.conf on disk -> silent no-op, not a SystemExit."""
        from kai.install import _strip_install_conf_keys

        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "nope.conf")
        _strip_install_conf_keys("users_yaml_staging_path")


# ── _cmd_config: global-default prompts fire regardless of users.yaml ──


class TestCmdConfigGlobalDefaultsRegardlessOfUsersYaml:
    """Pins the contract that installation-wide defaults (DEFAULT_MODEL,
    DEFAULT_TIMEOUT, WORKSPACE_BASE, PR_REVIEW_COOLDOWN) prompt
    on every wizard run and land in install.conf's env regardless of
    users.yaml presence.

    The pre-fix wizard gated those prompts on `not users_yaml_exists`
    so a re-run on an existing install silently kept stale globals - the
    operator-visible bug was a backend switch leaving DEFAULT_MODEL set
    to a model from the prior backend's surface. Each test below
    exercises one global-default key on the users-yaml-exists branch
    and asserts the new value lands.
    """

    @staticmethod
    def _existing_etc_users(monkeypatch):
        TestCmdConfig._simulate_existing_etc_users_yaml(
            monkeypatch,
            "users:\n  - telegram_id: 1\n    name: alice\n    role: admin\n",
        )

    def test_agent_timeout_lands_with_users_yaml(self, tmp_path, monkeypatch):
        """DEFAULT_TIMEOUT reaches env when users.yaml is present."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._existing_etc_users(monkeypatch)

        inputs = iter(TestCmdConfigDefaultModelDispatch._inputs_for_claude_backend())
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
        monkeypatch.setattr(
            "kai.install._prompt_default_model",
            lambda backend, prov, default: "sonnet",
        )
        _cmd_config()

        env = json.loads((tmp_path / "install.conf").read_text())["env"]
        assert env["DEFAULT_TIMEOUT"] == "120"

    def test_retired_context_window_key_dropped_on_regenerate(self, tmp_path, monkeypatch):
        """A regenerate over an install.conf carrying the retired
        CLAUDE_MAX_CONTEXT_WINDOW key drops it: the setting was
        removed, so the wizard neither prompts for it nor carries the
        stale key into the regenerated env."""
        prior_conf = {
            "install_dir": "/opt/kai",
            "data_dir": "/var/lib/kai",
            "service_user": "kai",
            "platform": "darwin",
            "env": {
                "TELEGRAM_BOT_TOKEN": "fake-token",
                "CLAUDE_MAX_CONTEXT_WINDOW": "200000",
            },
        }
        (tmp_path / "install.conf").write_text(json.dumps(prior_conf))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._existing_etc_users(monkeypatch)

        inputs = iter(TestCmdConfigDefaultModelDispatch._inputs_for_claude_backend())
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
        monkeypatch.setattr(
            "kai.install._prompt_default_model",
            lambda backend, prov, default: "sonnet",
        )
        _cmd_config()

        env = json.loads((tmp_path / "install.conf").read_text())["env"]
        assert "CLAUDE_MAX_CONTEXT_WINDOW" not in env

    def test_retired_scoped_recall_keys_dropped_on_regenerate(self, tmp_path, monkeypatch):
        """A regenerate over an install.conf carrying the retired
        scoped-recall and shadow flags drops them: the runtime no
        longer reads them, and the wizard no longer prompts for them,
        so the regenerated env must not carry them forward."""
        prior_conf = {
            "install_dir": "/opt/kai",
            "data_dir": "/var/lib/kai",
            "service_user": "kai",
            "platform": "darwin",
            "env": {
                "TELEGRAM_BOT_TOKEN": "fake-token",
                "MEMORY_SCOPED_RECALL_ENABLED": "true",
                "MEMORY_RECALL_SHADOW_ENABLED": "false",
            },
        }
        (tmp_path / "install.conf").write_text(json.dumps(prior_conf))
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._existing_etc_users(monkeypatch)

        inputs = iter(TestCmdConfigDefaultModelDispatch._inputs_for_claude_backend())
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
        monkeypatch.setattr(
            "kai.install._prompt_default_model",
            lambda backend, prov, default: "sonnet",
        )
        _cmd_config()

        env = json.loads((tmp_path / "install.conf").read_text())["env"]
        assert "MEMORY_SCOPED_RECALL_ENABLED" not in env
        assert "MEMORY_RECALL_SHADOW_ENABLED" not in env

    def test_workspace_base_lands_with_users_yaml(self, tmp_path, monkeypatch):
        """WORKSPACE_BASE reaches env when users.yaml is present."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._existing_etc_users(monkeypatch)

        inputs = iter(TestCmdConfigDefaultModelDispatch._inputs_for_claude_backend())
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
        monkeypatch.setattr(
            "kai.install._prompt_default_model",
            lambda backend, prov, default: "sonnet",
        )
        _cmd_config()

        env = json.loads((tmp_path / "install.conf").read_text())["env"]
        assert env["WORKSPACE_BASE"] == "~/Projects"

    def test_pr_review_cooldown_always_prompts(self, tmp_path, monkeypatch):
        """PR_REVIEW_COOLDOWN is a global resource control: the prompt
        fires whether users.yaml exists or pr_review_enabled is true,
        and a non-default value lands in env.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._existing_etc_users(monkeypatch)

        # Override the cooldown slot in the claude chain to a non-default
        # value so the env-emission assertion is meaningful (the default
        # "300" is suppressed by the delta-from-default check).
        base = list(TestCmdConfigDefaultModelDispatch._inputs_for_claude_backend())
        cooldown_idx = base.index("300")
        base[cooldown_idx] = "450"
        inputs = iter(base)
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
        monkeypatch.setattr(
            "kai.install._prompt_default_model",
            lambda backend, prov, default: "sonnet",
        )
        _cmd_config()

        env = json.loads((tmp_path / "install.conf").read_text())["env"]
        assert env["PR_REVIEW_COOLDOWN"] == "450"

    def test_allowed_workspaces_lands_with_users_yaml(self, tmp_path, monkeypatch):
        """ALLOWED_WORKSPACES reaches env when users.yaml is present.

        Pre-fix: the prompt was gated on `not users_yaml_exists`, so a
        re-run on an existing install silently carried forward the
        previous env value (or stayed empty), with no operator-facing
        path to add or change a machine-wide allowed workspace.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._existing_etc_users(monkeypatch)

        # Real, existing paths under tmp_path because the env emission
        # block only writes ALLOWED_WORKSPACES when non-empty, and
        # downstream loaders skip non-existent entries on apply.
        ws_a = tmp_path / "ws_a"
        ws_a.mkdir()
        ws_b = tmp_path / "ws_b"
        ws_b.mkdir()

        base = list(TestCmdConfigDefaultModelDispatch._inputs_for_claude_backend())
        # ALLOWED_WORKSPACES is the slot immediately after WORKSPACE_BASE
        # (~/Projects). Locate it by neighbor anchor rather than `.index("")`
        # because other slots (effort level, perplexity key) are also empty.
        allowed_idx = base.index("~/Projects") + 1
        base[allowed_idx] = f"{ws_a},{ws_b}"
        inputs = iter(base)
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
        monkeypatch.setattr(
            "kai.install._prompt_default_model",
            lambda backend, prov, default: "sonnet",
        )
        _cmd_config()

        env = json.loads((tmp_path / "install.conf").read_text())["env"]
        assert env["ALLOWED_WORKSPACES"] == f"{ws_a},{ws_b}"


# ── _cmd_config: users.yaml canonicalization ─────────────────────────


class TestCmdConfigCanonicalUsersYaml:
    """The wizard reads /etc/kai/users.yaml only and writes any new
    file to a per-operator staging path recorded as a top-level
    install.conf key, never inside the env dict.
    """

    @staticmethod
    def _base_inputs() -> list[str]:
        """Inputs that drive the wizard through a minimal first-install
        flow (no advanced options, claude backend, no memory, no
        external services).
        """
        return [
            "protected",
            "/opt/kai",
            "/var/lib/kai",
            "kai",
            "darwin",
            "fake-token",
            "12345",  # admin telegram id
            "admin",  # admin name
            "testuser",  # required protected os_user
            "polling",
            "claude",
            "sonnet",
            "false",  # customize per-role models (decline)
            "120",
            "0",  # max session age hours (0 = no limit)
            "1800",  # idle eviction timeout seconds
            "10.0",
            "80",
            "",  # effort level (default)
            "8080",
            "",  # Workshop LAN address (disabled)
            "test-secret",
            "~/Projects",
            "",
            "false",
            "300",  # pr review cooldown (global resource control)
            "900",
            "1.0",
            "false",
            "",
            "false",
            "false",
            "",  # claude_user
            "false",  # memory enabled
            "",  # perplexity key
        ]

    def test_stray_project_root_users_yaml_warns_and_is_ignored(self, tmp_path, monkeypatch, capsys):
        """Stray PROJECT_ROOT/users.yaml triggers a deprecation notice
        and the wizard still treats the install as first-time.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        TestCmdConfig._redirect_staging(monkeypatch, tmp_path)

        stray = tmp_path / "users.yaml"
        stray.write_text("users:\n  - telegram_id: 999\n    name: stale\n    role: admin\n")

        inputs = iter(self._base_inputs())
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
        _cmd_config()

        output = capsys.readouterr().out
        assert str(stray) in output
        assert "no longer used" in output
        # Wizard ignored the stray file: it prompted for a new admin
        # and produced a fresh staging file at the redirected location.
        # The redirect maps "users.yaml" -> tmp_path/users.yaml, which
        # IS the stray path we wrote above. So the generated content
        # overwrote the stray. Verify by parsing for the wizard-supplied
        # telegram_id (12345), not the stray (999).
        staged = yaml.safe_load(stray.read_text())
        assert staged["users"][0]["telegram_id"] == 12345

    def test_first_time_install_records_top_level_staging_path(self, tmp_path, monkeypatch):
        """install.conf carries `users_yaml_staging_path` at the top
        level, NOT inside the env dict.
        """
        monkeypatch.chdir(tmp_path)
        conf_path = tmp_path / "install.conf"
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        # Use a sibling staging location so we can distinguish "top-level
        # key matches the actual staging path" from "test stuffed the
        # value via redirect at tmp_path/users.yaml".
        staging_dir = tmp_path / "stage"
        staging_dir.mkdir()
        monkeypatch.setattr(
            "kai.install._install_staging_path",
            lambda filename: staging_dir / filename,
        )

        inputs = iter(self._base_inputs())
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
        _cmd_config()

        conf = json.loads(conf_path.read_text())
        assert "users_yaml_staging_path" in conf
        assert conf["users_yaml_staging_path"] == str(staging_dir / "users.yaml")
        # MUST NOT leak into env (would otherwise reach /etc/kai/env as
        # runtime daemon configuration).
        assert "users_yaml_staging_path" not in conf["env"]
        assert "USERS_YAML_STAGING_PATH" not in conf["env"]

    def test_protected_install_records_optional_yaml_staging_paths(self, tmp_path, monkeypatch):
        """Project-root optional YAML configs are staged as installer metadata."""
        monkeypatch.chdir(tmp_path)
        conf_path = tmp_path / "install.conf"
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        staging_dir = tmp_path / "stage"
        staging_dir.mkdir()
        monkeypatch.setattr(
            "kai.install._install_staging_path",
            lambda filename: staging_dir / filename,
        )
        (tmp_path / "services.yaml").write_text("services: {}\n")
        (tmp_path / "workspaces.yaml").write_text("workspaces: []\n")

        inputs = iter(self._base_inputs())
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
        _cmd_config()

        conf = json.loads(conf_path.read_text())
        staged = conf["protected_yaml_staging_paths"]
        assert staged == {
            "services.yaml": str(staging_dir / "services.yaml"),
            "workspaces.yaml": str(staging_dir / "workspaces.yaml"),
        }
        assert "protected_yaml_staging_paths" not in conf["env"]
        assert (staging_dir / "services.yaml").read_text() == "services: {}\n"
        assert stat.S_IMODE((staging_dir / "services.yaml").stat().st_mode) == 0o600

    def test_env_file_does_not_carry_staging_path(self, tmp_path, monkeypatch):
        """Regression guard: `_generate_env_file(env)` never emits a
        `USERS_YAML_STAGING_PATH` line. Pins the schema discipline that
        keeps installer metadata out of `/etc/kai/env`.
        """
        monkeypatch.chdir(tmp_path)
        conf_path = tmp_path / "install.conf"
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        TestCmdConfig._redirect_staging(monkeypatch, tmp_path)

        inputs = iter(self._base_inputs())
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
        _cmd_config()

        env = json.loads(conf_path.read_text())["env"]
        rendered = _generate_env_file(env)
        assert "USERS_YAML_STAGING_PATH" not in rendered


# ── _cmd_apply: staging handoff ──────────────────────────────────────


class TestCmdApplyStagingHandoff:
    """`_cmd_apply` consumes the top-level `users_yaml_staging_path`
    key from install.conf, threads it through to `_apply_secrets`, and
    cleans up the staging file plus the conf key after the apply
    succeeds (real run only; dry-run preserves both for a retry).
    """

    @staticmethod
    def _minimal_conf(
        tmp_path,
        staging_path: str | None,
        protected_yaml_staging_paths: dict[str, object] | None = None,
    ) -> Path:
        """Write a minimal install.conf the apply path can validate.

        Mirrors what `_cmd_config` would have produced for a claude
        backend install. When `staging_path` is non-empty, persists it
        as the top-level key the apply expects.
        """
        conf = {
            "version": 1,
            "install_dir": str(tmp_path / "opt-kai"),
            "data_dir": str(tmp_path / "var-lib-kai"),
            "service_user": "kai",
            "platform": "darwin",
            "env": {
                "TELEGRAM_BOT_TOKEN": "tok",
                "GITHUB_WEBHOOK_SECRET": "github-secret",
                "GENERIC_WEBHOOK_SECRET": "generic-secret",
                "DEFAULT_MODEL": "sonnet",
                "DEFAULT_BACKEND": "claude",
            },
        }
        if staging_path is not None:
            conf["users_yaml_staging_path"] = staging_path
        if protected_yaml_staging_paths is not None:
            conf["protected_yaml_staging_paths"] = protected_yaml_staging_paths
        path = tmp_path / "install.conf"
        path.write_text(json.dumps(conf, indent=2) + "\n")
        os.chmod(path, 0o600)
        return path

    @staticmethod
    def _stub_apply_internals(monkeypatch, captured: dict) -> None:
        """Stub everything except staging-handoff cleanup.

        The test runs as a non-root user so the real apply steps
        cannot mutate `/etc/`. We replace each step with a stub and
        capture the `users_yaml_staging_path` kwarg that flowed into
        `_apply_secrets`.
        """
        monkeypatch.setattr("os.geteuid", lambda: 0)

        class _FakePwd:
            pw_gid = 4242

            def __init__(self, uid):
                self.pw_uid = uid

        monkeypatch.setattr("pwd.getpwnam", lambda name: _FakePwd(4242 if name == "kai" else 5252))
        kai.install.USERS_YAML.write_text(
            "users:\n  - telegram_id: 1\n    name: test\n    role: admin\n    os_user: testuser\n"
        )

        def _fake_apply_secrets(env, dry_run, users_yaml_staging_path=None, protected_yaml_staging_paths=None):
            captured["users_yaml_staging_path"] = users_yaml_staging_path
            captured["protected_yaml_staging_paths"] = protected_yaml_staging_paths

        monkeypatch.setattr("kai.install._stop_service", lambda *a, **kw: None)
        monkeypatch.setattr("kai.install._apply_directories", lambda *a, **kw: None)
        monkeypatch.setattr("kai.install._apply_source", lambda *a, **kw: None)
        monkeypatch.setattr("kai.install._apply_venv", lambda *a, **kw: None)
        monkeypatch.setattr("kai.install._apply_models", lambda *a, **kw: None)
        monkeypatch.setattr("kai.install._apply_secrets", _fake_apply_secrets)
        monkeypatch.setattr("kai.install._apply_backend_registry", lambda *a, **kw: None)
        monkeypatch.setattr("kai.install._apply_runtime_policy", lambda *a, **kw: None)
        monkeypatch.setattr("kai.install._apply_sudoers", lambda *a, **kw: None)
        monkeypatch.setattr("kai.install._apply_migrate", lambda *a, **kw: None)
        monkeypatch.setattr("kai.install._apply_service", lambda *a, **kw: None)
        monkeypatch.setattr("kai.install._start_service", lambda *a, **kw: None)
        monkeypatch.setattr("kai.install._check_traversal", lambda *a, **kw: None)

    def test_real_apply_unlinks_and_strips(self, tmp_path, monkeypatch):
        """Successful real apply removes the staging file AND drops the
        top-level conf key, preserving env and other top-level keys.
        """
        staging = tmp_path / "users.yaml"
        staging.write_text("users:\n  - telegram_id: 1\n    name: test\n    role: admin\n    os_user: testuser\n")
        conf_path = self._minimal_conf(tmp_path, str(staging))
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        monkeypatch.delenv("DRY_RUN", raising=False)
        captured: dict = {}
        self._stub_apply_internals(monkeypatch, captured)

        _cmd_apply()

        # _apply_secrets received the staging path from the top-level key.
        assert captured["users_yaml_staging_path"] == str(staging)
        # Cleanup removed the staging file and the conf key.
        assert not staging.exists()
        rewritten = json.loads(conf_path.read_text())
        assert "users_yaml_staging_path" not in rewritten
        assert "protected_yaml_staging_paths" not in rewritten
        assert rewritten["env"]["TELEGRAM_BOT_TOKEN"] == "tok"
        assert rewritten["install_dir"] == str(tmp_path / "opt-kai")
        # Mode preservation: install.conf still carries secrets.
        assert stat.S_IMODE(conf_path.stat().st_mode) == 0o600

    def test_dry_run_preserves_staging_and_conf_key(self, tmp_path, monkeypatch, capsys):
        """DRY_RUN=1 must leave the staging file in place AND keep the
        top-level conf key so a subsequent real apply completes the
        handoff exactly as if the dry run had never happened.
        """
        staging = tmp_path / "users.yaml"
        staging.write_text("users:\n  - telegram_id: 1\n    name: test\n    role: admin\n    os_user: testuser\n")
        conf_path = self._minimal_conf(tmp_path, str(staging))
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        monkeypatch.setenv("DRY_RUN", "1")
        captured: dict = {}
        self._stub_apply_internals(monkeypatch, captured)

        _cmd_apply()

        assert staging.exists()
        rewritten = json.loads(conf_path.read_text())
        assert rewritten.get("users_yaml_staging_path") == str(staging)
        # Operator-visible notice that the dry run skipped the cleanup.
        out = capsys.readouterr().out
        assert "[DRY RUN] Would unlink staging file" in out
        assert "[DRY RUN] Would strip users_yaml_staging_path" in out

    def test_no_staging_path_means_no_cleanup(self, tmp_path, monkeypatch):
        """install.conf without the top-level key -> apply does not
        invent a path or attempt cleanup; `_apply_secrets` receives None.
        """
        conf_path = self._minimal_conf(tmp_path, staging_path=None)
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        monkeypatch.delenv("DRY_RUN", raising=False)
        captured: dict = {}
        self._stub_apply_internals(monkeypatch, captured)

        _cmd_apply()

        assert captured["users_yaml_staging_path"] is None
        rewritten = json.loads(conf_path.read_text())
        assert "users_yaml_staging_path" not in rewritten

    def test_real_apply_unlinks_and_strips_optional_yaml_staging(self, tmp_path, monkeypatch):
        """Successful real apply cleans optional protected-YAML staging metadata."""
        staging = tmp_path / "services.yaml"
        staging.write_text("services: {}\n")
        conf_path = self._minimal_conf(
            tmp_path,
            staging_path=None,
            protected_yaml_staging_paths={"services.yaml": str(staging)},
        )
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        monkeypatch.delenv("DRY_RUN", raising=False)
        captured: dict = {}
        self._stub_apply_internals(monkeypatch, captured)

        _cmd_apply()

        assert captured["protected_yaml_staging_paths"] == {"services.yaml": str(staging)}
        assert not staging.exists()
        rewritten = json.loads(conf_path.read_text())
        assert "protected_yaml_staging_paths" not in rewritten

    def test_dry_run_preserves_optional_yaml_staging(self, tmp_path, monkeypatch, capsys):
        """Dry run does not consume optional protected-YAML staging files."""
        staging = tmp_path / "services.yaml"
        staging.write_text("services: {}\n")
        conf_path = self._minimal_conf(
            tmp_path,
            staging_path=None,
            protected_yaml_staging_paths={"services.yaml": str(staging)},
        )
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        monkeypatch.setenv("DRY_RUN", "1")
        captured: dict = {}
        self._stub_apply_internals(monkeypatch, captured)

        _cmd_apply()

        assert staging.exists()
        rewritten = json.loads(conf_path.read_text())
        assert rewritten.get("protected_yaml_staging_paths") == {"services.yaml": str(staging)}
        out = capsys.readouterr().out
        assert "[DRY RUN] Would unlink staging file" in out
        assert "[DRY RUN] Would strip protected_yaml_staging_paths" in out

    def test_optional_yaml_staging_path_must_be_string(self, tmp_path, monkeypatch):
        """Hand-edited staging maps fail closed before install mutations."""
        hand_edited_map = {"services.yaml": 123}
        conf_path = self._minimal_conf(
            tmp_path,
            staging_path=None,
            protected_yaml_staging_paths=hand_edited_map,
        )
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        monkeypatch.setattr("os.geteuid", lambda: 0)

        with pytest.raises(SystemExit, match="must be an absolute path string"):
            _cmd_apply()

    def test_optional_yaml_staging_path_must_be_absolute(self, tmp_path, monkeypatch):
        """Relative staging paths are refused before root copies config."""
        conf_path = self._minimal_conf(
            tmp_path,
            staging_path=None,
            protected_yaml_staging_paths={"services.yaml": "relative/services.yaml"},
        )
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        monkeypatch.setattr("os.geteuid", lambda: 0)

        with pytest.raises(SystemExit, match="must be an absolute path"):
            _cmd_apply()

    def test_home_mismatch_resolved_via_conf_key(self, tmp_path, monkeypatch):
        """Apply locates the staging file from install.conf even when
        HOME differs from the HOME under which `make config` ran.
        Pins the cross-account contract: the wizard runs as the
        operator; apply runs as root. They share the recorded path,
        not `Path.home()`.
        """
        operator_cache = tmp_path / "operator-cache"
        operator_cache.mkdir()
        staging = operator_cache / "users.yaml"
        staging.write_text("users:\n  - telegram_id: 1\n    name: test\n    role: admin\n    os_user: testuser\n")
        conf_path = self._minimal_conf(tmp_path, str(staging))
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        monkeypatch.delenv("DRY_RUN", raising=False)
        # Apply-side HOME points somewhere unrelated to the operator
        # cache; the only signal that survives is the recorded path.
        monkeypatch.setenv("HOME", str(tmp_path / "root-home"))
        captured: dict = {}
        self._stub_apply_internals(monkeypatch, captured)

        _cmd_apply()

        assert captured["users_yaml_staging_path"] == str(staging)
        assert not staging.exists()


# ── Runtime users.yaml path resolution ────────────────────────────────


class TestResolveUsersYamlPath:
    """Pins the runtime predicate for choosing /etc/kai/users.yaml vs
    the XDG single-user path. The spec carves three rules:

      1. KAI_USERS_YAML wins outright when set (test / development override).
      2. Otherwise, protected_env_was_loaded -> /etc/kai/users.yaml.
      3. Otherwise -> ${XDG_CONFIG_HOME:-$HOME/.config}/kai/users.yaml.

    `KAI_INSTALL_DIR` and `KAI_DATA_DIR` deliberately do NOT participate
    in the predicate; they are pure path overrides for data and install
    layout and must not make runtime select the protected path.
    """

    def test_protected_env_loaded_routes_protected(self, monkeypatch):
        from kai.config import _resolve_users_yaml_path

        monkeypatch.delenv("KAI_USERS_YAML", raising=False)
        assert _resolve_users_yaml_path(True) == Path("/etc/kai/users.yaml")

    def test_no_protected_env_routes_xdg(self, monkeypatch, tmp_path):
        from kai.config import _resolve_users_yaml_path

        monkeypatch.delenv("KAI_USERS_YAML", raising=False)
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        assert _resolve_users_yaml_path(False) == tmp_path / ".config" / "kai" / "users.yaml"

    def test_xdg_config_home_overrides_home(self, monkeypatch, tmp_path):
        from kai.config import _resolve_users_yaml_path

        monkeypatch.delenv("KAI_USERS_YAML", raising=False)
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
        monkeypatch.setenv("HOME", str(tmp_path / "should-not-be-used"))
        assert _resolve_users_yaml_path(False) == tmp_path / "xdg" / "kai" / "users.yaml"

    def test_explicit_override_wins_in_both_modes(self, monkeypatch, tmp_path):
        from kai.config import _resolve_users_yaml_path

        override = tmp_path / "explicit-users.yaml"
        monkeypatch.setenv("KAI_USERS_YAML", str(override))
        assert _resolve_users_yaml_path(True) == override
        assert _resolve_users_yaml_path(False) == override

    def test_kai_install_dir_does_not_route_protected(self, monkeypatch, tmp_path):
        """Pins the spec's blast-radius rule: a single-user host with
        `KAI_INSTALL_DIR` set for unrelated reasons (custom install
        layout) must not be misclassified as a protected install.
        """
        from kai.config import _resolve_users_yaml_path

        monkeypatch.delenv("KAI_USERS_YAML", raising=False)
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("KAI_INSTALL_DIR", "/opt/kai")
        monkeypatch.setenv("KAI_DATA_DIR", "/var/lib/kai")
        # protected_env_was_loaded is False (no /etc/kai/env content);
        # the env vars above must NOT override that decision.
        assert _resolve_users_yaml_path(False) == tmp_path / ".config" / "kai" / "users.yaml"


# ── Single-user wizard branch ─────────────────────────────────────────


class TestCmdConfigSingleUserMode:
    """The wizard writes XDG users.yaml + local .env directly when the
    operator selects `single_user`; no staging handoff, no apply step,
    no /etc/kai/ writes.
    """

    @staticmethod
    def _single_user_inputs() -> list[str]:
        """Minimal single-user input chain: claude backend, no advanced
        options, no memory, no external services.
        """
        return [
            "single_user",  # deployment mode
            # install_dir / data_dir / service_user / platform are skipped
            "fake-token",  # bot token
            "12345",  # admin telegram id
            "admin",  # admin display name
            "false",  # advanced
            "polling",  # transport
            "claude",  # backend
            "sonnet",  # model
            "false",  # customize per-role models (decline; use registry defaults)
            "120",  # timeout
            "0",  # max session age hours (0 = no limit)
            "1800",  # idle eviction timeout seconds
            "80",  # autocompact
            "",  # effort level
            "8080",  # webhook port
            "",  # Workshop LAN address (disabled)
            "test-secret",  # webhook secret
            "~/Projects",  # workspace_base
            "",  # allowed_workspaces
            "false",  # pr_review_enabled
            "300",  # pr_review_cooldown
            "900",  # pr_review_timeout_s
            "false",  # issue_triage
            "",  # github_notify_chat_id
            "false",  # voice
            "false",  # tts
            "",  # claude_user
            "false",  # memory enabled
            "",  # perplexity
        ]

    @staticmethod
    def _redirect_xdg(monkeypatch, tmp_path):
        """Pin XDG_CONFIG_HOME at tmp_path so the wizard writes the
        XDG users.yaml under the test root rather than the operator's
        real `~/.config/kai/`.
        """
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))

    @staticmethod
    def _no_protected_artifacts(monkeypatch):
        """Pretend the host has no readable /etc/kai/env.

        The wizard's single-user refusal check probes the protected env
        file via sudo-cat; tests run against the live operator machine
        would otherwise trip the refusal because the production install
        leaves /etc/kai/env readable through the sudoers rule. Stubbing
        the reader to None simulates a clean host with no protected
        leftovers.
        """
        monkeypatch.setattr("kai.install._read_protected_file", lambda path: None)

    def test_writes_xdg_users_yaml_and_local_env(self, tmp_path, monkeypatch):
        """Single-user fresh install: users.yaml lands under XDG,
        runtime env lands at PROJECT_ROOT/.env, install.conf records
        deployment_mode=single_user with no staging key.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._redirect_xdg(monkeypatch, tmp_path)
        self._no_protected_artifacts(monkeypatch)

        inputs = iter(self._single_user_inputs())
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
        _cmd_config()

        users_yaml = tmp_path / "xdg" / "kai" / "users.yaml"
        assert users_yaml.exists()
        assert stat.S_IMODE(users_yaml.stat().st_mode) == 0o600
        parsed = yaml.safe_load(users_yaml.read_text())
        assert parsed["users"][0]["telegram_id"] == 12345

        env_file = tmp_path / ".env"
        assert env_file.exists()
        assert stat.S_IMODE(env_file.stat().st_mode) == 0o600
        env_text = env_file.read_text()
        assert "TELEGRAM_BOT_TOKEN" in env_text
        assert "fake-token" in env_text

        conf = json.loads((tmp_path / "install.conf").read_text())
        assert conf["deployment_mode"] == "single_user"
        # No staging handoff in single-user mode; apply is refused
        # outright so a stale key would point at a path apply has no
        # business consuming.
        assert "users_yaml_staging_path" not in conf

    def test_refuses_single_user_when_protected_env_is_readable(self, tmp_path, monkeypatch):
        """Migration guard: a host with a previous protected install
        leaves /etc/kai/env readable through the sudoers rule. The
        runtime's resolver would still see that as authoritative and
        boot from the protected artifacts, silently ignoring the
        single-user files the wizard is about to write. The wizard
        refuses up front with an actionable removal recipe.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._redirect_xdg(monkeypatch, tmp_path)
        # Simulate the leftover sudoers rule: /etc/kai/env reads back
        # non-empty content via sudo -n cat.
        monkeypatch.setattr(
            "kai.install._read_protected_file",
            lambda path: "TELEGRAM_BOT_TOKEN=leftover\n" if path == "/etc/kai/env" else None,
        )

        # Only the deployment_mode prompt is reached before the refusal
        # fires; the rest of the chain is unused but kept here so the
        # test does not depend on prompt-count specifics.
        inputs = iter(["single_user"] + ["x"] * 50)
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
        with pytest.raises(SystemExit, match="single_user mode was selected"):
            _cmd_config()

    def test_protected_mode_unaffected_by_protected_env_check(self, tmp_path, monkeypatch):
        """Protected mode does not consult the single-user refusal
        check. Re-running `make config` on a working protected install
        must continue to work even though /etc/kai/env is readable
        (which is the normal protected-install state).
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        TestCmdConfig._simulate_existing_etc_users_yaml(
            monkeypatch,
            "users:\n  - telegram_id: 12345\n    name: test\n    role: admin\n",
        )
        # /etc/kai/env reads back content (sudoers rule from a real
        # protected install). The refusal must NOT fire for protected mode.
        monkeypatch.setattr(
            "kai.install._read_protected_file",
            lambda path: "TELEGRAM_BOT_TOKEN=current\n" if path == "/etc/kai/env" else None,
        )

        inputs = iter(TestCmdConfig._base_inputs(memory_block=["false"]))
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
        _cmd_config()
        # install.conf written with deployment_mode=protected; no
        # SystemExit means the refusal did not fire.
        conf = json.loads((tmp_path / "install.conf").read_text())
        assert conf["deployment_mode"] == "protected"

    def test_writes_users_yaml_parent_mode_0700(self, tmp_path, monkeypatch):
        """Operator-private parent directory: the XDG kai/ subdir is
        created mode 0700 so a freshly minted users.yaml inherits a
        restrictive enclosing scope even before its own 0600 chmod.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._redirect_xdg(monkeypatch, tmp_path)
        self._no_protected_artifacts(monkeypatch)

        inputs = iter(self._single_user_inputs())
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
        _cmd_config()

        parent = tmp_path / "xdg" / "kai"
        assert parent.is_dir()
        assert stat.S_IMODE(parent.stat().st_mode) == 0o700


# ── _cmd_apply: single-user refusal ───────────────────────────────────


class TestCmdApplySingleUserRefuses:
    """`make install` (the apply subcommand) is meaningful only for
    protected mode. Single-user installs are fully configured by
    `make config`; running apply against a single-user install would
    silently leave the operator confused about what changed. The
    wizard's success message points at `make run`; the apply gate
    enforces the same contract from the install command side.
    """

    @staticmethod
    def _write_single_user_conf(tmp_path: Path) -> Path:
        conf = {
            "version": 1,
            "deployment_mode": "single_user",
            "install_dir": str(tmp_path),
            "data_dir": str(tmp_path),
            "service_user": "kai",
            "platform": "darwin",
            "env": {"TELEGRAM_BOT_TOKEN": "tok", "DEFAULT_BACKEND": "claude"},
        }
        path = tmp_path / "install.conf"
        path.write_text(json.dumps(conf, indent=2) + "\n")
        os.chmod(path, 0o600)
        return path

    def test_apply_refuses_single_user_with_no_op_message(self, tmp_path, monkeypatch):
        """Apply on a single-user install exits with a clear message
        pointing the operator at `make run` instead.
        """
        conf_path = self._write_single_user_conf(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        monkeypatch.setattr("os.geteuid", lambda: 0)
        with pytest.raises(SystemExit, match="single_user mode is already applied"):
            _cmd_apply()

    def test_apply_protected_default_when_mode_missing(self, tmp_path, monkeypatch):
        """Legacy install.conf written before the deployment_mode key
        existed defaults to protected so the existing apply flow keeps
        running. Pins the migration shape: nothing forces operators to
        re-run `make config` purely to gain the key.
        """
        conf = {
            "version": 1,
            "install_dir": str(tmp_path / "opt-kai"),
            "data_dir": str(tmp_path / "var-lib-kai"),
            "service_user": "kai",
            "platform": "darwin",
            "env": {
                "TELEGRAM_BOT_TOKEN": "tok",
                "WEBHOOK_SECRET": "secret",
                "GITHUB_WEBHOOK_SECRET": "github-secret",
                "GENERIC_WEBHOOK_SECRET": "generic-secret",
                "DEFAULT_MODEL": "sonnet",
                "DEFAULT_BACKEND": "claude",
            },
        }
        conf_path = tmp_path / "install.conf"
        conf_path.write_text(json.dumps(conf, indent=2) + "\n")
        os.chmod(conf_path, 0o600)
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        monkeypatch.setattr("os.geteuid", lambda: 0)

        # The legacy-shape apply runs through all the real apply steps,
        # which is more than this regression cares about. Stub them
        # out and verify only that the single-user refusal does NOT
        # fire when deployment_mode is missing from the conf.

        class _FakePwd:
            pw_gid = 4242

            def __init__(self, uid):
                self.pw_uid = uid

        monkeypatch.setattr("pwd.getpwnam", lambda name: _FakePwd(4242 if name == "kai" else 5252))
        kai.install.USERS_YAML.write_text(
            "users:\n  - telegram_id: 1\n    name: test\n    role: admin\n    os_user: testuser\n"
        )
        monkeypatch.setattr("kai.install._stop_service", lambda *a, **kw: None)
        monkeypatch.setattr("kai.install._apply_directories", lambda *a, **kw: None)
        monkeypatch.setattr("kai.install._apply_source", lambda *a, **kw: None)
        monkeypatch.setattr("kai.install._apply_venv", lambda *a, **kw: None)
        monkeypatch.setattr("kai.install._apply_models", lambda *a, **kw: None)
        monkeypatch.setattr("kai.install._apply_secrets", lambda *a, **kw: None)
        monkeypatch.setattr("kai.install._apply_backend_registry", lambda *a, **kw: None)
        monkeypatch.setattr("kai.install._apply_runtime_policy", lambda *a, **kw: None)
        monkeypatch.setattr("kai.install._apply_sudoers", lambda *a, **kw: None)
        monkeypatch.setattr("kai.install._apply_migrate", lambda *a, **kw: None)
        monkeypatch.setattr("kai.install._apply_service", lambda *a, **kw: None)
        monkeypatch.setattr("kai.install._start_service", lambda *a, **kw: None)
        monkeypatch.setattr("kai.install._check_traversal", lambda *a, **kw: None)
        monkeypatch.delenv("DRY_RUN", raising=False)

        # No SystemExit expected; the legacy-shape apply completes.
        _cmd_apply()


# ── _apply_sudoers dry run ───────────────────────────────────────────


class TestApplySudoersDryRun:
    def test_dry_run(self, capsys):
        """Dry run: prints expected messages."""
        _apply_sudoers("kai", dry_run=True, agent_backend="claude")
        output = capsys.readouterr().out
        assert "DRY RUN" in output
        assert "sudoers" in output.lower() or "visudo" in output.lower()

    def test_warns_when_claude_bin_missing(self, tmp_path, capsys, monkeypatch):
        """Warning printed when the rule's claude binary path doesn't exist."""
        svc_home = tmp_path / "home" / "kai"
        svc_home.mkdir(parents=True)
        # Intentionally do NOT create svc_home/.local/bin/claude.
        monkeypatch.setattr("kai.install._user_home", lambda u: str(svc_home))
        monkeypatch.setattr(
            "kai.install._discover_backend_commands",
            lambda service_user: {"claude": str(svc_home / ".local" / "bin" / "claude")},
        )
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users:\n  - telegram_id: 1\n    os_user: alice\n")

        _apply_sudoers("kai", dry_run=True, users_yaml_path=users_yaml, agent_backend="claude")

        captured = capsys.readouterr()
        assert "Warning" in captured.err
        assert str(svc_home / ".local" / "bin" / "claude") in captured.err

    def test_no_warning_when_no_target_users(self, tmp_path, capsys, monkeypatch):
        """No warning when there are no per-user rules to emit."""
        svc_home = tmp_path / "home" / "kai"
        svc_home.mkdir(parents=True)
        monkeypatch.setattr("kai.install._user_home", lambda u: str(svc_home))
        monkeypatch.setattr(
            "kai.install._discover_backend_commands",
            lambda service_user: {"claude": str(svc_home / ".local" / "bin" / "claude")},
        )
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users: []\n")

        _apply_sudoers("kai", dry_run=True, users_yaml_path=users_yaml, agent_backend="claude")

        captured = capsys.readouterr()
        assert "Warning" not in captured.err

    def test_no_warning_when_claude_bin_exists(self, tmp_path, capsys, monkeypatch):
        """Path exists -> silent."""
        svc_home = tmp_path / "home" / "kai"
        bin_dir = svc_home / ".local" / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "claude").write_text("#!/bin/sh\n")
        monkeypatch.setattr("kai.install._user_home", lambda u: str(svc_home))
        monkeypatch.setattr(
            "kai.install._discover_backend_commands",
            lambda service_user: {"claude": str(bin_dir / "claude")},
        )
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users:\n  - telegram_id: 1\n    os_user: alice\n")

        _apply_sudoers("kai", dry_run=True, users_yaml_path=users_yaml, agent_backend="claude")

        captured = capsys.readouterr()
        assert "Warning" not in captured.err

    def test_opencode_only_install_gets_no_claude_warning(self, tmp_path, capsys, monkeypatch):
        """An install whose only backend is opencode is never told to
        install claude; the backstop checks the opencode binary instead."""
        svc_home = tmp_path / "home" / "kai"
        svc_home.mkdir(parents=True)
        # Neither the claude binary nor the opencode binary exists.
        monkeypatch.setattr("kai.install._user_home", lambda u: str(svc_home))
        monkeypatch.setattr(
            "kai.install._discover_backend_commands",
            lambda service_user: {"opencode": str(tmp_path / "nope" / "opencode")},
        )
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users:\n  - telegram_id: 1\n    os_user: alice\n")

        _apply_sudoers(
            "kai",
            dry_run=True,
            users_yaml_path=users_yaml,
            agent_backend="opencode",
        )

        captured = capsys.readouterr()
        assert "claude" not in captured.err
        assert "opencode sudoers" in captured.err
        assert "/etc/kai/backends.yaml" in captured.err

    def test_opencode_only_install_silent_when_binary_exists(self, tmp_path, capsys, monkeypatch):
        """An opencode-only install with its binary in place warns about
        nothing, even though the claude binary is absent."""
        svc_home = tmp_path / "home" / "kai"
        svc_home.mkdir(parents=True)
        opencode = tmp_path / "opencode"
        opencode.write_text("#!/bin/sh\n")
        monkeypatch.setattr("kai.install._user_home", lambda u: str(svc_home))
        monkeypatch.setattr(
            "kai.install._discover_backend_commands",
            lambda service_user: {"opencode": str(opencode)},
        )
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users:\n  - telegram_id: 1\n    os_user: alice\n")

        _apply_sudoers(
            "kai",
            dry_run=True,
            users_yaml_path=users_yaml,
            agent_backend="opencode",
        )

        captured = capsys.readouterr()
        assert "Warning" not in captured.err

    def test_codex_backend_missing_binary_names_registry(self, tmp_path, capsys, monkeypatch):
        """A codex install with a missing binary gets the codex warning
        pointing at backend registry regeneration."""
        svc_home = tmp_path / "home" / "kai"
        svc_home.mkdir(parents=True)
        monkeypatch.setattr("kai.install._user_home", lambda u: str(svc_home))
        monkeypatch.setattr(
            "kai.install._discover_backend_commands",
            lambda service_user: {"codex": str(tmp_path / "nope" / "codex")},
        )
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users:\n  - telegram_id: 1\n    os_user: alice\n")

        _apply_sudoers(
            "kai",
            dry_run=True,
            users_yaml_path=users_yaml,
            agent_backend="codex",
        )

        captured = capsys.readouterr()
        assert "codex sudoers" in captured.err
        assert "/etc/kai/backends.yaml" in captured.err
        assert "claude" not in captured.err

    def test_per_user_backend_override_widens_the_check(self, tmp_path, capsys, monkeypatch):
        """A mixed install (global opencode, one per-user claude
        override) checks both binaries: the claude warning fires for
        the per-user claude even though the global backend is not
        claude."""
        svc_home = tmp_path / "home" / "kai"
        svc_home.mkdir(parents=True)
        opencode = tmp_path / "opencode"
        opencode.write_text("#!/bin/sh\n")
        monkeypatch.setattr("kai.install._user_home", lambda u: str(svc_home))
        monkeypatch.setattr(
            "kai.install._discover_backend_commands",
            lambda service_user: {
                "claude": str(svc_home / ".local" / "bin" / "claude"),
                "opencode": str(opencode),
            },
        )
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text(
            "users:\n"
            "  - telegram_id: 1\n"
            "    os_user: alice\n"
            "  - telegram_id: 2\n"
            "    os_user: bob\n"
            "    backend: claude\n"
        )

        _apply_sudoers(
            "kai",
            dry_run=True,
            users_yaml_path=users_yaml,
            agent_backend="opencode",
        )

        captured = capsys.readouterr()
        # The per-user override adds claude to the check, so its
        # warning fires; the opencode binary exists, so no opencode
        # warning appears.
        assert "the claude sudoers rule" in captured.err
        assert "the opencode sudoers rule" not in captured.err

    def test_goose_only_install_missing_binary_names_registry(self, tmp_path, capsys, monkeypatch):
        """A goose-only install with os_users warns when the goose
        binary path does not exist, and the remedy names the registry:
        goose now has a per-user sudoers rule with a pinned path, so
        the backstop covers it the same way it covers the other
        backends. The path is passed explicitly (never the host
        fallback) so the assertion cannot flip on whether the test
        host happens to have goose installed."""
        svc_home = tmp_path / "home" / "kai"
        svc_home.mkdir(parents=True)
        monkeypatch.setattr("kai.install._user_home", lambda u: str(svc_home))
        monkeypatch.setattr(
            "kai.install._discover_backend_commands",
            lambda service_user: {"goose": str(tmp_path / "nope" / "goose")},
        )
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users:\n  - telegram_id: 1\n    os_user: alice\n")

        _apply_sudoers(
            "kai",
            dry_run=True,
            users_yaml_path=users_yaml,
            agent_backend="goose",
        )

        captured = capsys.readouterr()
        assert "the goose sudoers rule" in captured.err
        assert "/etc/kai/backends.yaml" in captured.err
        # Scoping: a goose-only install must not be told about the
        # other backends' binaries.
        assert "the claude sudoers rule" not in captured.err
        assert "the codex sudoers rule" not in captured.err

    def test_goose_only_install_with_present_binary_warns_about_nothing(self, tmp_path, capsys, monkeypatch):
        """A goose-only install whose pinned goose path exists gets no
        missing-binary warning. Deterministic counterpart to the
        missing-binary case above: the binary is a real executable
        under tmp_path, so the check cannot depend on host state."""
        svc_home = tmp_path / "home" / "kai"
        svc_home.mkdir(parents=True)
        monkeypatch.setattr("kai.install._user_home", lambda u: str(svc_home))
        monkeypatch.setattr(
            "kai.install._discover_backend_commands",
            lambda service_user: {"goose": str(goose)},
        )
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users:\n  - telegram_id: 1\n    os_user: alice\n")
        goose = tmp_path / "goose"
        goose.write_text("#!/bin/sh\necho hi\n")
        goose.chmod(0o755)

        _apply_sudoers(
            "kai",
            dry_run=True,
            users_yaml_path=users_yaml,
            agent_backend="goose",
        )

        captured = capsys.readouterr()
        assert "Warning" not in captured.err


class TestCollectBackendsFromYaml:
    """The lightweight per-user backend reader that scopes the
    sudoers missing-binary backstop."""

    def test_missing_file_returns_empty_set(self, tmp_path):
        assert _collect_backends_from_yaml(tmp_path / "nope.yaml") == set()

    def test_collects_distinct_backends(self, tmp_path):
        path = tmp_path / "users.yaml"
        path.write_text(
            "users:\n"
            "  - telegram_id: 1\n"
            "    backend: codex\n"
            "  - telegram_id: 2\n"
            "    backend: opencode\n"
            "  - telegram_id: 3\n"
            "    backend: codex\n"
            "  - telegram_id: 4\n"
        )
        assert _collect_backends_from_yaml(path) == {"codex", "opencode"}

    def test_non_string_and_blank_values_skipped(self, tmp_path):
        path = tmp_path / "users.yaml"
        path.write_text("users:\n  - telegram_id: 1\n    backend: 42\n  - telegram_id: 2\n    backend: '  '\n")
        assert _collect_backends_from_yaml(path) == set()


class TestEntryBackendLegacyKey:
    """Every installer-side users.yaml scanner reads a per-user entry's
    backend through `_entry_backend`, which prefers the new `backend`
    key and falls back to the deprecated `default_backend` then
    `agent_backend` keys for one release (renamed twice: agent_backend
    -> default_backend -> backend). Without this, a per-user
    `backend: goose` entry would be invisible to wizard/apply setup
    (provider keys, binary collection, goose config deployment,
    sudoers scoping)."""

    @pytest.mark.parametrize("key", ["backend", "default_backend", "agent_backend"])
    def test_collect_backends_reads_all_keys(self, tmp_path, key):
        path = tmp_path / "users.yaml"
        path.write_text(f"users:\n  - telegram_id: 1\n    {key}: goose\n")
        assert _collect_backends_from_yaml(path) == {"goose"}

    @pytest.mark.parametrize("key", ["backend", "default_backend", "agent_backend"])
    def test_users_yaml_agent_backends_reads_all_keys(self, tmp_path, key):
        path = tmp_path / "users.yaml"
        path.write_text(f"users:\n  - telegram_id: 1\n    {key}: codex\n")
        assert _users_yaml_agent_backends(path) == {"codex"}

    @pytest.mark.parametrize("key", ["backend", "default_backend", "agent_backend"])
    def test_goose_providers_reads_all_backend_keys(self, tmp_path, key):
        path = tmp_path / "users.yaml"
        path.write_text(f"users:\n  - telegram_id: 1\n    {key}: goose\n    provider: openai\n")
        assert _users_yaml_goose_providers(path, "anthropic") == ["openai"]

    @pytest.mark.parametrize("provider_key", ["provider", "llm_provider"])
    def test_goose_providers_reads_new_provider_key(self, tmp_path, provider_key):
        """The goose-provider scanner reads the new `provider` key and the
        deprecated `llm_provider` key through `_entry_provider`, so a user
        on the new key still gets their provider API key collected."""
        path = tmp_path / "users.yaml"
        path.write_text(f"users:\n  - telegram_id: 1\n    backend: goose\n    {provider_key}: deepseek\n")
        assert _users_yaml_goose_providers(path, "anthropic") == ["deepseek"]

    @pytest.mark.parametrize("key", ["backend", "default_backend", "agent_backend"])
    def test_goose_os_users_reads_all_keys(self, tmp_path, key):
        from kai.install import _collect_goose_os_users_from_yaml

        path = tmp_path / "users.yaml"
        path.write_text(f"users:\n  - telegram_id: 1\n    {key}: goose\n    os_user: alice\n")
        assert _collect_goose_os_users_from_yaml(path, "claude") == ["alice"]

    def test_collect_backends_normalizes_mixed_case(self, tmp_path):
        """A mixed-case per-user value is valid at runtime (the loader
        lowercases); the scanner must normalize too so the goose-config
        membership check `"goose" in <set>` matches."""
        path = tmp_path / "users.yaml"
        path.write_text("users:\n  - telegram_id: 1\n    backend: Goose\n")
        assert _collect_backends_from_yaml(path) == {"goose"}

    def test_goose_os_users_normalizes_mixed_case(self, tmp_path):
        """A mixed-case `backend: Goose` user's os_user must
        still be collected for the per-user goose config deploy."""
        from kai.install import _collect_goose_os_users_from_yaml

        path = tmp_path / "users.yaml"
        path.write_text("users:\n  - telegram_id: 1\n    backend: Goose\n    os_user: alice\n")
        assert _collect_goose_os_users_from_yaml(path, "claude") == ["alice"]


# ── _apply_service dry run ───────────────────────────────────────────


class TestApplyServiceDryRun:
    def test_dry_run_darwin(self, capsys):
        """Dry run on Darwin: prints launcher and plist messages."""
        _apply_service("/opt/kai", "/var/lib/kai", "kai", "darwin", dry_run=True)
        output = capsys.readouterr().out
        assert "DRY RUN" in output
        assert "launcher" in output.lower() or "run.sh" in output.lower()
        assert "plist" in output.lower() or "LaunchDaemon" in output

    def test_dry_run_linux(self, capsys):
        """Dry run on Linux: prints unit file message."""
        _apply_service("/opt/kai", "/var/lib/kai", "kai", "linux", dry_run=True)
        output = capsys.readouterr().out
        assert "DRY RUN" in output


class TestApplyGooseConfig:
    """Tests for _apply_goose_config() goose binary check.

    Every call passes an explicit (absent) users_yaml_path so the
    assertions cannot flip on the host's real /etc/kai/users.yaml;
    an absent file means no goose-backed os_users, i.e. the
    service-user-only deploy these tests pin.
    """

    def _setup(self, tmp_path):
        """Create a minimal install tree with the goose config template."""
        install_path = tmp_path / "opt" / "kai"
        config_dir = install_path / "config"
        config_dir.mkdir(parents=True)
        (config_dir / "goose-config.yaml").write_text("extensions: {}\n")
        return install_path

    def test_warns_when_goose_not_on_path(self, tmp_path, capsys, monkeypatch):
        """Warning printed when goose binary is not found on PATH."""
        install_path = self._setup(tmp_path)
        svc_home = tmp_path / "home" / "kai"
        svc_home.mkdir(parents=True)
        monkeypatch.setattr("kai.install._user_home", lambda u: str(svc_home))

        # Ensure shutil.which returns None (goose not installed)
        monkeypatch.setattr("shutil.which", lambda cmd: None)

        uid = os.getuid()
        gid = os.getgid()
        _apply_goose_config(
            "kai",
            install_path,
            uid,
            gid,
            dry_run=False,
            users_yaml_path=tmp_path / "absent-users.yaml",
            agent_backend="goose",
        )

        output = capsys.readouterr().out
        assert "WARNING" in output
        assert "goose" in output.lower()

    def test_no_warning_in_dry_run(self, tmp_path, capsys, monkeypatch):
        """Dry run does not warn about missing goose binary."""
        install_path = self._setup(tmp_path)
        svc_home = tmp_path / "home" / "kai"
        svc_home.mkdir(parents=True)
        monkeypatch.setattr("kai.install._user_home", lambda u: str(svc_home))
        monkeypatch.setattr("shutil.which", lambda cmd: None)

        uid = os.getuid()
        gid = os.getgid()
        _apply_goose_config(
            "kai",
            install_path,
            uid,
            gid,
            dry_run=True,
            users_yaml_path=tmp_path / "absent-users.yaml",
            agent_backend="goose",
        )

        output = capsys.readouterr().out
        assert "WARNING" not in output
        assert "DRY RUN" in output

    def test_no_warning_when_goose_on_path(self, tmp_path, capsys, monkeypatch):
        """No warning printed when goose binary exists on PATH."""
        install_path = self._setup(tmp_path)
        svc_home = tmp_path / "home" / "kai"
        svc_home.mkdir(parents=True)
        monkeypatch.setattr("kai.install._user_home", lambda u: str(svc_home))

        # Simulate goose being on PATH
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/local/bin/goose")

        uid = os.getuid()
        gid = os.getgid()
        _apply_goose_config(
            "kai",
            install_path,
            uid,
            gid,
            dry_run=False,
            users_yaml_path=tmp_path / "absent-users.yaml",
            agent_backend="goose",
        )

        output = capsys.readouterr().out
        assert "WARNING" not in output

    def test_deploys_to_each_goose_backed_os_user(self, tmp_path, monkeypatch):
        """The config lands in the service user's home AND in each
        distinct goose-backed os_user's home, each file owned by its
        home's user (the per-user `goose acp` spawn runs under
        `sudo -H` and resolves config beneath the target home)."""
        import types

        install_path = self._setup(tmp_path)
        svc_home = tmp_path / "home" / "kai"
        svc_home.mkdir(parents=True)
        alice_home = tmp_path / "home" / "alice"
        alice_home.mkdir(parents=True)
        monkeypatch.setattr("kai.install._user_home", lambda u: str(svc_home))
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/local/bin/goose")

        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text(
            "users:\n"
            "  - telegram_id: 1\n"
            "    name: alice\n"
            "    backend: goose\n"
            "    os_user: alice\n"
            # Duplicate os_user entry: the deploy must dedupe.
            "  - telegram_id: 2\n"
            "    name: alice2\n"
            "    backend: goose\n"
            "    os_user: alice\n"
            # Non-goose user with an os_user: not a deploy target.
            "  - telegram_id: 3\n"
            "    name: bob\n"
            "    backend: codex\n"
            "    os_user: bob\n"
        )

        # Fake passwd database so the os_user lookup does not depend
        # on accounts existing on the test host.
        def _fake_getpwnam(name):
            if name == "alice":
                return types.SimpleNamespace(pw_dir=str(alice_home), pw_uid=2001, pw_gid=2001)
            raise KeyError(name)

        monkeypatch.setattr("kai.install.pwd", types.SimpleNamespace(getpwnam=_fake_getpwnam))

        # Record chowns instead of performing them: the test runner
        # cannot chown to arbitrary uids, and the recorded calls ARE
        # the ownership contract under test.
        chowns: list[tuple[str, int, int]] = []
        monkeypatch.setattr(
            "kai.install._set_ownership",
            lambda path, uid, gid, recursive=False: chowns.append((str(path), uid, gid)),
        )

        _apply_goose_config(
            "kai",
            install_path,
            1001,
            1001,
            dry_run=False,
            users_yaml_path=users_yaml,
            agent_backend="claude",
        )

        # Both homes got the template.
        assert (svc_home / ".config" / "goose" / "config.yaml").exists()
        assert (alice_home / ".config" / "goose" / "config.yaml").exists()
        # Exactly one per-os_user deploy (deduped), none for bob.
        assert not (tmp_path / "home" / "bob").exists()
        # Ownership followed each home's user.
        svc_cfg = str(svc_home / ".config" / "goose" / "config.yaml")
        alice_cfg = str(alice_home / ".config" / "goose" / "config.yaml")
        assert (svc_cfg, 1001, 1001) in chowns
        assert (alice_cfg, 2001, 2001) in chowns

    def test_global_goose_install_deploys_to_os_users_without_override(self, tmp_path, monkeypatch):
        """On a global goose install, users.yaml entries with no
        per-user backend inherit goose and their os_user homes
        are deploy targets."""
        import types

        install_path = self._setup(tmp_path)
        svc_home = tmp_path / "home" / "kai"
        svc_home.mkdir(parents=True)
        carol_home = tmp_path / "home" / "carol"
        carol_home.mkdir(parents=True)
        monkeypatch.setattr("kai.install._user_home", lambda u: str(svc_home))
        monkeypatch.setattr("shutil.which", lambda cmd: "/usr/local/bin/goose")

        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users:\n  - telegram_id: 1\n    name: carol\n    os_user: carol\n")

        monkeypatch.setattr(
            "kai.install.pwd",
            types.SimpleNamespace(
                getpwnam=lambda name: types.SimpleNamespace(pw_dir=str(carol_home), pw_uid=2002, pw_gid=2002)
            ),
        )
        monkeypatch.setattr(
            "kai.install._set_ownership",
            lambda path, uid, gid, recursive=False: None,
        )

        _apply_goose_config(
            "kai",
            install_path,
            1001,
            1001,
            dry_run=False,
            users_yaml_path=users_yaml,
            agent_backend="goose",
        )

        assert (carol_home / ".config" / "goose" / "config.yaml").exists()

    def test_dry_run_lists_every_target(self, tmp_path, capsys, monkeypatch):
        """Dry run prints the create/copy pair for the service user
        and each goose-backed os_user, mutating nothing."""
        import types

        install_path = self._setup(tmp_path)
        svc_home = tmp_path / "home" / "kai"
        svc_home.mkdir(parents=True)
        alice_home = tmp_path / "home" / "alice"
        monkeypatch.setattr("kai.install._user_home", lambda u: str(svc_home))

        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users:\n  - telegram_id: 1\n    name: alice\n    backend: goose\n    os_user: alice\n")
        monkeypatch.setattr(
            "kai.install.pwd",
            types.SimpleNamespace(
                getpwnam=lambda name: types.SimpleNamespace(pw_dir=str(alice_home), pw_uid=2001, pw_gid=2001)
            ),
        )

        _apply_goose_config(
            "kai",
            install_path,
            1001,
            1001,
            dry_run=True,
            users_yaml_path=users_yaml,
            agent_backend="claude",
        )

        output = capsys.readouterr().out
        assert f"[DRY RUN] Would create: {svc_home / '.config' / 'goose'}" in output
        assert f"[DRY RUN] Would create: {alice_home / '.config' / 'goose'}" in output
        # Nothing was written.
        assert not (svc_home / ".config").exists()
        assert not alice_home.exists()

    def test_noop_when_nothing_goose_backed(self, tmp_path, capsys):
        """A claude-only install (no global goose, no per-user goose
        override) deploys nothing and is never blocked on the goose
        template - the apply pipeline calls this unconditionally."""
        # Deliberately NO template in the install tree: the gate must
        # return before the template existence check.
        install_path = tmp_path / "opt" / "kai"
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users:\n  - {telegram_id: 1, name: a, backend: codex, os_user: bob}\n")

        _apply_goose_config(
            "kai",
            install_path,
            1001,
            1001,
            dry_run=False,
            users_yaml_path=users_yaml,
            agent_backend="claude",
        )

        assert capsys.readouterr().out == ""

    def test_unknown_os_user_fails_before_any_deploy(self, tmp_path, monkeypatch):
        """An os_user missing from the passwd database aborts the
        whole step BEFORE the service-user copy, so a typo cannot
        leave a half-deployed set of homes."""
        import types

        install_path = self._setup(tmp_path)
        svc_home = tmp_path / "home" / "kai"
        svc_home.mkdir(parents=True)
        monkeypatch.setattr("kai.install._user_home", lambda u: str(svc_home))

        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users:\n  - telegram_id: 1\n    name: ghost\n    backend: goose\n    os_user: ghost\n")

        def _raise_keyerror(name):
            raise KeyError(name)

        monkeypatch.setattr("kai.install.pwd", types.SimpleNamespace(getpwnam=_raise_keyerror))

        with pytest.raises(ValueError, match="does not exist"):
            _apply_goose_config(
                "kai",
                install_path,
                1001,
                1001,
                dry_run=False,
                users_yaml_path=users_yaml,
                agent_backend="claude",
            )

        assert not (svc_home / ".config").exists()


class TestCollectGooseOsUsersFromYaml:
    """The goose-backed os_user collector behind _apply_goose_config:
    per-user overrides and global-backend inheritance select entries;
    empty/absent os_users are skipped (service-user deploy covers
    them); validation failures raise before any path is built."""

    def test_missing_file_returns_empty(self, tmp_path):
        from kai.install import _collect_goose_os_users_from_yaml

        assert _collect_goose_os_users_from_yaml(tmp_path / "nope.yaml", "goose") == []

    def test_per_user_override_selected_on_non_goose_global(self, tmp_path):
        from kai.install import _collect_goose_os_users_from_yaml

        path = tmp_path / "users.yaml"
        path.write_text(
            "users:\n"
            "  - {telegram_id: 1, name: a, backend: goose, os_user: alice}\n"
            "  - {telegram_id: 2, name: b, backend: codex, os_user: bob}\n"
            "  - {telegram_id: 3, name: c, os_user: carol}\n"
        )
        assert _collect_goose_os_users_from_yaml(path, "claude") == ["alice"]

    def test_global_goose_inherited_by_unset_entries(self, tmp_path):
        from kai.install import _collect_goose_os_users_from_yaml

        path = tmp_path / "users.yaml"
        path.write_text(
            "users:\n"
            "  - {telegram_id: 1, name: a, os_user: alice}\n"
            "  - {telegram_id: 2, name: b, backend: claude, os_user: bob}\n"
        )
        assert _collect_goose_os_users_from_yaml(path, "goose") == ["alice"]

    def test_entries_without_os_user_skipped(self, tmp_path):
        from kai.install import _collect_goose_os_users_from_yaml

        path = tmp_path / "users.yaml"
        path.write_text("users:\n  - {telegram_id: 1, name: a, backend: goose}\n")
        assert _collect_goose_os_users_from_yaml(path, "claude") == []

    def test_duplicates_deduped_preserving_order(self, tmp_path):
        from kai.install import _collect_goose_os_users_from_yaml

        path = tmp_path / "users.yaml"
        path.write_text(
            "users:\n"
            "  - {telegram_id: 1, name: a, backend: goose, os_user: alice}\n"
            "  - {telegram_id: 2, name: b, backend: goose, os_user: dana}\n"
            "  - {telegram_id: 3, name: c, backend: goose, os_user: alice}\n"
        )
        assert _collect_goose_os_users_from_yaml(path, "claude") == ["alice", "dana"]

    def test_invalid_os_user_raises(self, tmp_path):
        from kai.install import _collect_goose_os_users_from_yaml

        path = tmp_path / "users.yaml"
        path.write_text('users:\n  - {telegram_id: 1, name: a, backend: goose, os_user: "bad)user"}\n')
        with pytest.raises(ValueError, match="Invalid os_user"):
            _collect_goose_os_users_from_yaml(path, "claude")


# ── _apply_migrate per-os-user tmp dir (issue #454) ────────────────────


class TestApplyMigratePerOsUserTmpdir:
    """
    _apply_migrate creates <DATA_DIR>/tmp/<os_user>/ for every distinct
    os_user in users.yaml. The runtime (claude.py _ensure_started)
    points the inner Claude subprocess's TMPDIR at that path so each
    os_user has its own temp namespace; the shared /tmp default
    otherwise causes content-hash collisions on the claude-settings
    cache file between two os_users with the same --settings JSON.
    See issue #454.
    """

    def _write_users_yaml(self, path: Path, entries: list[dict]) -> None:
        path.write_text(yaml.safe_dump({"users": entries}))

    def _stub_pwd_getpwnam(self):
        # Map any os_user to fixed uid/gid. Real OS-account lookups
        # would fail on the test host for names like "alice".
        # chown calls are stubbed separately so the values are
        # placeholders.
        class _Pw:
            pw_uid = 1234
            pw_gid = 1234

        return _Pw()

    def _setup_templates(self, src: Path) -> None:
        """Seed the templates the rest of _apply_migrate consumes."""
        ws_claude = src / "templates" / ".claude"
        ws_claude.mkdir(parents=True)
        (ws_claude / "CLAUDE.md").write_text("# Kai\n")
        (ws_claude / "MEMORY.md").write_text("# Memory\n")
        (ws_claude / "PREFERENCES.md").write_text("# Preferences\n")

    def test_creates_per_os_user_dir(self, tmp_path):
        """Each distinct os_user gets a <DATA_DIR>/tmp/<os_user>/ directory."""
        src = tmp_path / "source"
        self._setup_templates(src)
        users_yaml = tmp_path / "users.yaml"
        self._write_users_yaml(
            users_yaml,
            [
                {"telegram_id": 100, "os_user": "alice"},
                {"telegram_id": 200, "os_user": "bob"},
            ],
        )
        data_path = tmp_path / "data"
        install_path = tmp_path / "install"
        install_path.mkdir()

        with (
            patch("kai.install.PROJECT_ROOT", src),
            patch("kai.install.pwd.getpwnam", return_value=self._stub_pwd_getpwnam()),
            patch("kai.install._set_ownership"),
            patch("os.chown"),
        ):
            _apply_migrate(
                data_path,
                install_path,
                svc_uid=0,
                svc_gid=0,
                dry_run=False,
                users_yaml_path=Path(users_yaml),
            )

        assert (data_path / "tmp" / "alice").is_dir()
        assert (data_path / "tmp" / "bob").is_dir()

    def test_per_os_user_dir_mode_is_0o700(self, tmp_path):
        """
        Per-os-user tmp dir must end up at exactly 0o700. The cache
        files inside hold the target user's claude.ai session state;
        no other identity should read them.

        Two assertions, paired deliberately. The end-state check is
        the user-facing contract (mode is 0o700 after install). The
        chmod-call check is the regression guard: it pins that the
        production code makes an EXPLICIT chmod with the target
        mode, so a future change to the `mode=` arg of mkdir (or
        the removal of the chmod under the assumption that mkdir's
        own arg is sufficient) is caught. Without the explicit
        chmod, a hardened service umask that masks any bit in
        0o700 would silently drop those bits and break dir
        traversal for the inner subprocess.
        """
        src = tmp_path / "source"
        self._setup_templates(src)
        users_yaml = tmp_path / "users.yaml"
        self._write_users_yaml(users_yaml, [{"telegram_id": 100, "os_user": "alice"}])
        data_path = tmp_path / "data"
        install_path = tmp_path / "install"
        install_path.mkdir()

        # Spy on os.chmod calls inside install.py so we can assert
        # the per-user tmp dir gets an explicit chmod(0o700) (the
        # load-bearing call) without having to perturb the process
        # umask and break the unrelated mkdir paths in
        # _apply_migrate.
        chmod_calls: list[tuple[str, int]] = []
        real_chmod = os.chmod

        def spy_chmod(path, mode, *args, **kwargs):
            chmod_calls.append((str(path), mode))
            return real_chmod(path, mode, *args, **kwargs)

        with (
            patch("kai.install.PROJECT_ROOT", src),
            patch("kai.install.pwd.getpwnam", return_value=self._stub_pwd_getpwnam()),
            patch("kai.install._set_ownership"),
            patch("os.chown"),
            patch("kai.install.os.chmod", side_effect=spy_chmod),
        ):
            _apply_migrate(
                data_path,
                install_path,
                svc_uid=0,
                svc_gid=0,
                dry_run=False,
                users_yaml_path=Path(users_yaml),
            )

        user_tmp = data_path / "tmp" / "alice"
        # End-state contract: dir ends up at exactly 0o700.
        assert stat.S_IMODE(user_tmp.stat().st_mode) == 0o700
        # Regression guard: production code makes an explicit
        # chmod(user_tmp, 0o700). Pinning the (path, mode) pair
        # means a future refactor cannot satisfy the end-state
        # assertion by accident (e.g. by changing only the
        # mkdir mode= and dropping the chmod).
        assert (str(user_tmp), 0o700) in chmod_calls

    def test_no_dir_when_no_os_user(self, tmp_path):
        """A users.yaml without any os_user entries creates no tmp dirs."""
        src = tmp_path / "source"
        self._setup_templates(src)
        users_yaml = tmp_path / "users.yaml"
        # No os_user fields - same-as-service-user mode for all users.
        self._write_users_yaml(users_yaml, [{"telegram_id": 100}])
        data_path = tmp_path / "data"
        install_path = tmp_path / "install"
        install_path.mkdir()

        with (
            patch("kai.install.PROJECT_ROOT", src),
            patch("kai.install._set_ownership"),
            patch("os.chown"),
        ):
            _apply_migrate(
                data_path,
                install_path,
                svc_uid=0,
                svc_gid=0,
                dry_run=False,
                users_yaml_path=Path(users_yaml),
            )

        assert not (data_path / "tmp").exists()

    def test_duplicate_os_users_create_one_dir_each(self, tmp_path):
        """Multiple chats with the same os_user collapse to one tmp dir."""
        src = tmp_path / "source"
        self._setup_templates(src)
        users_yaml = tmp_path / "users.yaml"
        self._write_users_yaml(
            users_yaml,
            [
                {"telegram_id": 100, "os_user": "alice"},
                {"telegram_id": 200, "os_user": "alice"},
                {"telegram_id": 300, "os_user": "bob"},
            ],
        )
        data_path = tmp_path / "data"
        install_path = tmp_path / "install"
        install_path.mkdir()

        with (
            patch("kai.install.PROJECT_ROOT", src),
            patch("kai.install.pwd.getpwnam", return_value=self._stub_pwd_getpwnam()),
            patch("kai.install._set_ownership"),
            patch("os.chown"),
        ):
            _apply_migrate(
                data_path,
                install_path,
                svc_uid=0,
                svc_gid=0,
                dry_run=False,
                users_yaml_path=Path(users_yaml),
            )

        # Exactly the two distinct os_users, nothing else.
        assert sorted(p.name for p in (data_path / "tmp").iterdir()) == ["alice", "bob"]

    def test_idempotent_reinstall(self, tmp_path, capsys):
        """Second install does not re-print 'Created' for existing dirs."""
        src = tmp_path / "source"
        self._setup_templates(src)
        users_yaml = tmp_path / "users.yaml"
        self._write_users_yaml(users_yaml, [{"telegram_id": 100, "os_user": "alice"}])
        data_path = tmp_path / "data"
        install_path = tmp_path / "install"
        install_path.mkdir()

        with (
            patch("kai.install.PROJECT_ROOT", src),
            patch("kai.install.pwd.getpwnam", return_value=self._stub_pwd_getpwnam()),
            patch("kai.install._set_ownership"),
            patch("os.chown"),
        ):
            _apply_migrate(
                data_path,
                install_path,
                svc_uid=0,
                svc_gid=0,
                dry_run=False,
                users_yaml_path=Path(users_yaml),
            )
            first_output = capsys.readouterr().out
            assert "Created" in first_output
            assert "tmp/alice" in first_output

            _apply_migrate(
                data_path,
                install_path,
                svc_uid=0,
                svc_gid=0,
                dry_run=False,
                users_yaml_path=Path(users_yaml),
            )
            second_output = capsys.readouterr().out

        # Second pass touches ownership/mode but does not print
        # "Created tmp/alice" again - the dir already exists.
        assert "tmp/alice" not in second_output

    def test_dry_run_prints_without_creating(self, tmp_path, capsys):
        """Dry run lists what would be created and does not touch disk."""
        src = tmp_path / "source"
        self._setup_templates(src)
        users_yaml = tmp_path / "users.yaml"
        self._write_users_yaml(users_yaml, [{"telegram_id": 100, "os_user": "alice"}])
        data_path = tmp_path / "data"
        install_path = tmp_path / "install"
        install_path.mkdir()

        with (
            patch("kai.install.PROJECT_ROOT", src),
            patch("kai.install.pwd.getpwnam", return_value=self._stub_pwd_getpwnam()),
            patch("kai.install._set_ownership"),
            patch("os.chown"),
        ):
            _apply_migrate(
                data_path,
                install_path,
                svc_uid=0,
                svc_gid=0,
                dry_run=True,
                users_yaml_path=Path(users_yaml),
            )

        output = capsys.readouterr().out
        assert "[DRY RUN]" in output
        assert "tmp/alice" in output
        # Dry run must not have created anything.
        assert not (data_path / "tmp").exists()


# ── Codex wizard hardening: validators and apply-time checks ─────────


class TestValidateClaudeBin:
    """claude binary path existence validator. Mirrors
    `TestValidateCodexBin` exactly: same six edge cases (empty,
    nonexistent, directory, non-executable file, relative path,
    executable file)."""

    def test_empty_value_rejected(self):
        from kai.install import _validate_claude_bin

        assert _validate_claude_bin("") is False

    def test_nonexistent_path_rejected(self, tmp_path):
        from kai.install import _validate_claude_bin

        assert _validate_claude_bin(str(tmp_path / "does_not_exist")) is False

    def test_directory_rejected(self, tmp_path):
        from kai.install import _validate_claude_bin

        assert _validate_claude_bin(str(tmp_path)) is False

    def test_non_executable_file_rejected(self, tmp_path):
        from kai.install import _validate_claude_bin

        f = tmp_path / "fake_claude"
        f.write_text("#!/bin/sh\necho hi\n")
        assert _validate_claude_bin(str(f)) is False

    def test_relative_path_rejected(self, tmp_path, monkeypatch):
        """A relative path is rejected even when it points at a real
        executable from the current working directory; the value feeds
        /etc/kai/env and sudoers, where only absolute paths work."""
        from kai.install import _validate_claude_bin

        f = tmp_path / "fake_claude"
        f.write_text("#!/bin/sh\necho hi\n")
        f.chmod(0o755)
        monkeypatch.chdir(tmp_path)
        assert _validate_claude_bin("fake_claude") is False

    def test_executable_file_accepted(self, tmp_path):
        from kai.install import _validate_claude_bin

        f = tmp_path / "fake_claude"
        f.write_text("#!/bin/sh\necho hi\n")
        f.chmod(0o755)
        assert _validate_claude_bin(str(f)) is True


class TestValidateCodexBin:
    """codex binary path existence validator."""

    def test_empty_value_rejected(self):
        from kai.install import _validate_codex_bin

        assert _validate_codex_bin("") is False

    def test_nonexistent_path_rejected(self, tmp_path):
        from kai.install import _validate_codex_bin

        assert _validate_codex_bin(str(tmp_path / "does_not_exist")) is False

    def test_directory_rejected(self, tmp_path):
        from kai.install import _validate_codex_bin

        assert _validate_codex_bin(str(tmp_path)) is False

    def test_non_executable_file_rejected(self, tmp_path):
        from kai.install import _validate_codex_bin

        f = tmp_path / "fake_codex"
        f.write_text("#!/bin/sh\necho hi\n")
        assert _validate_codex_bin(str(f)) is False

    def test_relative_path_rejected(self, tmp_path, monkeypatch):
        """A relative path is rejected even when it points at a real
        executable from the current working directory; the value feeds
        /etc/kai/env and sudoers, where only absolute paths work."""
        from kai.install import _validate_codex_bin

        f = tmp_path / "fake_codex"
        f.write_text("#!/bin/sh\necho hi\n")
        f.chmod(0o755)
        monkeypatch.chdir(tmp_path)
        assert _validate_codex_bin("fake_codex") is False

    def test_executable_file_accepted(self, tmp_path):
        from kai.install import _validate_codex_bin

        f = tmp_path / "fake_codex"
        f.write_text("#!/bin/sh\necho hi\n")
        f.chmod(0o755)
        assert _validate_codex_bin(str(f)) is True


class TestResolveCodexBinPromptDefault:
    """Upgrade-safe default selection for the Codex wizard prompt."""

    def test_valid_saved_path_remains_authoritative(self, tmp_path, monkeypatch):
        saved = tmp_path / "saved-codex"
        saved.write_text("#!/bin/sh\n")
        saved.chmod(0o755)
        detected = tmp_path / "detected-codex"
        detected.write_text("#!/bin/sh\n")
        detected.chmod(0o755)
        monkeypatch.setattr("kai.install.shutil.which", lambda command: str(detected))

        result = _resolve_codex_bin_prompt_default({"CODEX_BIN": str(saved)})

        assert result == str(saved)

    def test_stale_saved_path_recovers_to_detected_executable(self, tmp_path, monkeypatch):
        stale = tmp_path / "moved-codex"
        detected = tmp_path / "detected-codex"
        detected.write_text("#!/bin/sh\n")
        detected.chmod(0o755)
        monkeypatch.setattr("kai.install._validate_codex_bin", lambda p: p == str(detected))
        monkeypatch.setattr("kai.install.shutil.which", lambda command: str(detected))

        result = _resolve_codex_bin_prompt_default({"CODEX_BIN": str(stale)})

        assert result == str(detected)


class TestValidateOpenCodeBin:
    """opencode binary path existence validator. Mirrors
    `TestValidateCodexBin` exactly: same six edge cases (empty,
    nonexistent, directory, non-executable file, relative path,
    executable file). The shared validator shape (`is_absolute()` AND
    `is_file()` AND `os.access(_, X_OK)`) is the same one codex uses;
    pinning all three legs protects against a future refactor that
    drops one and silently weakens the binary-path check for one
    backend but not the other."""

    def test_empty_value_rejected(self):
        from kai.install import _validate_opencode_bin

        assert _validate_opencode_bin("") is False

    def test_nonexistent_path_rejected(self, tmp_path):
        from kai.install import _validate_opencode_bin

        assert _validate_opencode_bin(str(tmp_path / "does_not_exist")) is False

    def test_directory_rejected(self, tmp_path):
        from kai.install import _validate_opencode_bin

        assert _validate_opencode_bin(str(tmp_path)) is False

    def test_non_executable_file_rejected(self, tmp_path):
        from kai.install import _validate_opencode_bin

        f = tmp_path / "fake_opencode"
        f.write_text("#!/bin/sh\necho hi\n")
        assert _validate_opencode_bin(str(f)) is False

    def test_relative_path_rejected(self, tmp_path, monkeypatch):
        """A relative path is rejected even when it points at a real
        executable from the current working directory; the value feeds
        /etc/kai/env and sudoers, where only absolute paths work."""
        from kai.install import _validate_opencode_bin

        f = tmp_path / "fake_opencode"
        f.write_text("#!/bin/sh\necho hi\n")
        f.chmod(0o755)
        monkeypatch.chdir(tmp_path)
        assert _validate_opencode_bin("fake_opencode") is False

    def test_executable_file_accepted(self, tmp_path):
        from kai.install import _validate_opencode_bin

        f = tmp_path / "fake_opencode"
        f.write_text("#!/bin/sh\necho hi\n")
        f.chmod(0o755)
        assert _validate_opencode_bin(str(f)) is True


class TestValidateGooseBin:
    """goose binary path existence validator. Mirrors
    `TestValidateCodexBin` and `TestValidateOpenCodeBin` exactly: same
    six edge cases (empty, nonexistent, directory, non-executable
    file, relative path, executable file), pinning the shared
    validator shape for the third backend the same way it is pinned
    for the other two."""

    def test_empty_value_rejected(self):
        from kai.install import _validate_goose_bin

        assert _validate_goose_bin("") is False

    def test_nonexistent_path_rejected(self, tmp_path):
        from kai.install import _validate_goose_bin

        assert _validate_goose_bin(str(tmp_path / "does_not_exist")) is False

    def test_directory_rejected(self, tmp_path):
        from kai.install import _validate_goose_bin

        assert _validate_goose_bin(str(tmp_path)) is False

    def test_non_executable_file_rejected(self, tmp_path):
        from kai.install import _validate_goose_bin

        f = tmp_path / "fake_goose"
        f.write_text("#!/bin/sh\necho hi\n")
        assert _validate_goose_bin(str(f)) is False

    def test_relative_path_rejected(self, tmp_path, monkeypatch):
        """A relative path is rejected even when it points at a real
        executable from the current working directory; the value feeds
        /etc/kai/env and sudoers, where only absolute paths work."""
        from kai.install import _validate_goose_bin

        f = tmp_path / "fake_goose"
        f.write_text("#!/bin/sh\necho hi\n")
        f.chmod(0o755)
        monkeypatch.chdir(tmp_path)
        assert _validate_goose_bin("fake_goose") is False

    def test_executable_file_accepted(self, tmp_path):
        from kai.install import _validate_goose_bin

        f = tmp_path / "fake_goose"
        f.write_text("#!/bin/sh\necho hi\n")
        f.chmod(0o755)
        assert _validate_goose_bin(str(f)) is True


class TestOpenCodeConfigWizard:
    """OpenCode config writes backend/provider settings, not binary paths."""

    def _redirect_staging(self, monkeypatch, tmp_path):
        """Redirect the staging-path helper so the wizard writes under tmp_path."""
        monkeypatch.setattr(
            "kai.install._install_staging_path",
            lambda filename: tmp_path / filename,
        )

    def _opencode_inputs(self) -> list[str]:
        """Build the input sequence for an opencode-backend wizard run.

        OpenCode auth and command installation live outside
        `make config`; the installed backend registry records command
        paths during `make install`.
        """
        return [
            "protected",  # deployment mode
            "/opt/kai",  # install dir
            "/var/lib/kai",  # data dir
            "kai",  # service user
            "darwin",  # platform
            "fake-token",  # bot token
            "12345",  # admin telegram ID
            "admin",  # admin display name
            "testuser",  # required protected os_user
            "polling",  # transport
            "opencode",  # agent backend
            "anthropic",  # provider (opencode joined BACKENDS_NEEDING_PROVIDER_PROMPT)
            # API key prompt skipped for opencode (auth managed by `opencode auth login`).
            # model: handled by _prompt_default_model mock
            "false",  # customize per-role models (decline; use registry defaults)
            "120",  # agent timeout
            "0",  # max session age hours (0 = no limit)
            "1800",  # idle eviction timeout seconds
            "8080",  # webhook port
            "",  # Workshop LAN address (disabled)
            "test-secret",  # webhook secret
            "",  # workspace base
            "",  # allowed workspaces
            "300",  # pr review cooldown (global resource control)
            "900",  # pr review timeout
            "false",  # voice
            "false",  # tts
            "true",  # memory enabled
            "true",  # memory extraction enabled
            "10",  # extraction timeout
            "8",  # consolidation candidates
            "3",  # episode classifier context turns
            "120",  # episode timeout
            "0.9",  # paraphrase-dedup threshold
            "2000",  # token budget
            "10",  # search limit
            "",  # perplexity key
        ]

    def test_wizard_persists_opencode_backend_without_bin(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._redirect_staging(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "kai.install._backend_choices_for_config",
            lambda service_user: ["opencode"],
        )
        # Mock the model prompt so the test doesn't drive the
        # provider/model free-text branch directly.
        monkeypatch.setattr(
            "kai.install._prompt_default_model",
            MagicMock(return_value="anthropic/claude-sonnet-4-5"),
        )

        inputs = iter(self._opencode_inputs())
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        _cmd_config()

        env = json.loads((tmp_path / "install.conf").read_text())["env"]
        assert env["DEFAULT_BACKEND"] == "opencode"
        assert "OPENCODE_BIN" not in env

    def test_memory_extraction_enabled_persists_for_opencode_install(self, tmp_path, monkeypatch):
        """An opencode operator who answers `true` to the memory
        extraction prompt must see the value persist all the way to
        install.conf's env dict (and from there to /etc/kai/env and
        the runtime Config). The install-time cleanup gate that drops
        stale MEMORY_EXTRACTION_* keys for backends without a
        OneShotReasoner must NOT strip them for opencode (which DOES
        have an OpenCodeOneShotReasoner).

        This is the operator-visible regression test for the
        install.py cleanup gate that previously read
        `("claude", "codex")` and silently dropped the operator's
        `MEMORY_EXTRACTION_ENABLED=true` answer on opencode installs.
        Once the gate reads ONESHOT_REASONER_BACKENDS, opencode is
        admitted and the value persists.

        The base `_opencode_inputs` helper drives the wizard through
        the memory-extraction prompt with answer `true`; this test
        also overrides two tunable values to non-defaults so the
        wizard's delta-from-defaults emission rule writes those keys
        into install.conf, giving the cleanup gate's matching `.pop`
        sites real keys to (not) strip.
        """
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._redirect_staging(monkeypatch, tmp_path)
        monkeypatch.setattr(
            "kai.install._backend_choices_for_config",
            lambda service_user: ["opencode"],
        )
        monkeypatch.setattr(
            "kai.install._prompt_default_model",
            MagicMock(return_value="anthropic/claude-sonnet-4-5"),
        )

        # Build the input sequence; override two extraction tunables
        # to non-default values so the wizard's delta-from-defaults
        # emission rule writes them into install.conf and the
        # cleanup gate has matching keys to (not) strip. The "120"
        # value appears twice (agent timeout, then episode timeout);
        # take the second occurrence for the episode-timeout slot.
        inputs_list = self._opencode_inputs()
        inputs_list[inputs_list.index("10")] = "20"  # MEMORY_EXTRACTION_TIMEOUT_S
        first_120 = inputs_list.index("120")
        inputs_list[inputs_list.index("120", first_120 + 1)] = "180"  # MEMORY_EPISODE_TIMEOUT_S
        inputs = iter(inputs_list)
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        _cmd_config()

        env = json.loads((tmp_path / "install.conf").read_text())["env"]
        # DEFAULT_BACKEND pins which gate branch fired; opencode here
        # exercises the previously-broken cleanup path.
        assert env["DEFAULT_BACKEND"] == "opencode"
        # MEMORY_EXTRACTION_ENABLED was always emitted by the wizard,
        # then silently stripped by the cleanup gate before the gate
        # read from ONESHOT_REASONER_BACKENDS. Load-bearing assertion
        # for the operator-visible bug.
        assert env["MEMORY_EXTRACTION_ENABLED"] == "true"
        # The two non-default extraction tunables must also survive
        # the cleanup. Wizard's emission rule wrote them because the
        # operator chose values different from the dataclass defaults.
        assert env["MEMORY_EXTRACTION_TIMEOUT_S"] == "20"
        assert env["MEMORY_EPISODE_TIMEOUT_S"] == "180"
        assert "OPENCODE_BIN" not in env


class TestGenerateSudoersCodexBinArg:
    """_generate_sudoers takes codex_bin as an argument."""

    def test_codex_bin_arg_pins_sudoers_path(self):
        from kai.install import _generate_sudoers

        out = _generate_sudoers(
            service_user="kai",
            os_users=["daniel"],
            codex_bin="/Users/daniel/.npm-global/bin/codex",
        )
        assert "kai ALL=(daniel) SETENV: NOPASSWD: /Users/daniel/.npm-global/bin/codex" in out
        assert "/opt/homebrew/bin/codex" not in out

    def test_codex_bin_arg_none_prefers_common_absolute_path(self, monkeypatch):
        from kai.install import _generate_sudoers

        monkeypatch.setattr("kai.install._validate_codex_bin", lambda p: p == "/usr/local/bin/codex")
        out = _generate_sudoers(
            service_user="kai",
            os_users=["daniel"],
            codex_bin=None,
        )
        assert "kai ALL=(daniel) SETENV: NOPASSWD: /usr/local/bin/codex" in out
        assert "/opt/homebrew/bin/codex" not in out

    def test_codex_bin_does_not_read_os_environ(self, monkeypatch):
        """Setting CODEX_BIN in os.environ should have NO effect; only
        the explicit argument matters."""
        from kai.install import _generate_sudoers

        monkeypatch.setenv("CODEX_BIN", "/should/not/be/used")
        out = _generate_sudoers(
            service_user="kai",
            os_users=["daniel"],
            codex_bin="/correct/path/codex",
        )
        assert "/correct/path/codex" in out
        assert "/should/not/be/used" not in out


class TestGenerateSudoersBackendPathArgs:
    """Backend sudoers paths come from explicit install state, not process env."""

    def test_all_backend_path_args_pin_sudoers_paths(self):
        out = _generate_sudoers(
            service_user="kai",
            os_users=["daniel"],
            claude_bin="/pins/claude",
            codex_bin="/pins/codex",
            opencode_bin="/pins/opencode",
            goose_bin="/pins/goose",
        )

        assert "kai ALL=(daniel) SETENV: NOPASSWD: /pins/claude" in out
        assert "kai ALL=(daniel) SETENV: NOPASSWD: /pins/codex" in out
        assert "kai ALL=(daniel) SETENV: NOPASSWD: /pins/opencode" in out
        assert "kai ALL=(daniel) SETENV: NOPASSWD: /pins/goose" in out

    def test_backend_path_args_do_not_read_os_environ(self, monkeypatch):
        monkeypatch.setenv("CLAUDE_BIN", "/env/claude")
        monkeypatch.setenv("CODEX_BIN", "/env/codex")
        monkeypatch.setenv("OPENCODE_BIN", "/env/opencode")
        monkeypatch.setenv("GOOSE_BIN", "/env/goose")

        out = _generate_sudoers(
            service_user="kai",
            os_users=["daniel"],
            claude_bin="/arg/claude",
            codex_bin="/arg/codex",
            opencode_bin="/arg/opencode",
            goose_bin="/arg/goose",
        )

        assert "/arg/claude" in out
        assert "/arg/codex" in out
        assert "/arg/opencode" in out
        assert "/arg/goose" in out
        assert "/env/claude" not in out
        assert "/env/codex" not in out
        assert "/env/opencode" not in out
        assert "/env/goose" not in out


class TestApplyBackendAwareModelValidation:
    """_cmd_apply rejects codex installs with goose-only or claude models."""

    def _conf(self, tmp_path, **env_overrides):
        kai.install.USERS_YAML.write_text(
            "users:\n  - telegram_id: 1\n    name: test\n    role: admin\n    os_user: root\n"
        )
        env = {
            "TELEGRAM_BOT_TOKEN": "tok",
            "DEFAULT_BACKEND": "codex",
            "DEFAULT_MODEL": "gpt-5.5",
        }
        env.update(env_overrides)
        conf_path = tmp_path / "install.conf"
        conf_path.write_text(
            json.dumps(
                {
                    "install_dir": str(tmp_path / "opt" / "kai"),
                    "data_dir": str(tmp_path / "var" / "lib" / "kai"),
                    "service_user": "nobody",
                    "platform": "darwin",
                    "env": env,
                }
            )
        )
        return conf_path

    def test_codex_with_claude_model_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr("os.geteuid", lambda: 0)
        monkeypatch.setattr("kai.install.INSTALL_CONF", self._conf(tmp_path, DEFAULT_MODEL="opus"))
        with pytest.raises(SystemExit, match="not valid for codex"):
            _cmd_apply()

    def test_codex_with_goose_only_model_rejected(self, tmp_path, monkeypatch):
        """nano is valid for goose-openai but not codex; must be rejected."""
        monkeypatch.setattr("os.geteuid", lambda: 0)
        monkeypatch.setattr(
            "kai.install.INSTALL_CONF",
            self._conf(tmp_path, DEFAULT_MODEL="gpt-5.4-nano"),
        )
        with pytest.raises(SystemExit, match="not valid for codex"):
            _cmd_apply()


class TestCmdConfigSessionLifecycleKeys:
    """Wizard emission and migration for AGENT_MAX_SESSION_HOURS /
    AGENT_IDLE_TIMEOUT. Both are delta-from-default keys: a
    default-accepting run writes neither, a non-default answer writes
    the canonical AGENT_-prefixed key, and legacy CLAUDE_-prefixed
    keys carried in an existing install.conf prefill the prompts and
    are popped from the regenerated env."""

    def _run_claude_chain(self, tmp_path, monkeypatch, mutate=None):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        TestCmdConfig._simulate_existing_etc_users_yaml(
            monkeypatch,
            "users:\n  - telegram_id: 1\n    name: alice\n    role: admin\n",
        )
        base = list(TestCmdConfigDefaultModelDispatch._inputs_for_claude_backend())
        if mutate:
            mutate(base)
        inputs = iter(base)
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))
        monkeypatch.setattr(
            "kai.install._prompt_default_model",
            lambda backend, prov, default: "sonnet",
        )
        _cmd_config()
        return json.loads((tmp_path / "install.conf").read_text())["env"]

    def test_defaults_suppress_both_keys(self, tmp_path, monkeypatch):
        env = self._run_claude_chain(tmp_path, monkeypatch)
        assert "AGENT_MAX_SESSION_HOURS" not in env
        assert "AGENT_IDLE_TIMEOUT" not in env

    def test_non_default_values_land_with_canonical_names(self, tmp_path, monkeypatch):
        def mutate(base):
            # First "0" in the chain is the session-hours slot; "1800"
            # appears only as the idle-timeout slot.
            base[base.index("0")] = "6"
            base[base.index("1800")] = "900"

        env = self._run_claude_chain(tmp_path, monkeypatch, mutate)
        assert env["AGENT_MAX_SESSION_HOURS"] == "6"
        assert env["AGENT_IDLE_TIMEOUT"] == "900"

    def test_legacy_keys_prefill_then_migrate_to_canonical(self, tmp_path, monkeypatch):
        """A regenerate over an install.conf carrying the legacy
        CLAUDE_-prefixed keys prefills the prompts with the legacy
        values; accepting the prefills (empty input) lands the values
        under the canonical names and the legacy keys do not survive
        into the regenerated env."""
        prior_conf = {
            "install_dir": "/opt/kai",
            "data_dir": "/var/lib/kai",
            "service_user": "kai",
            "platform": "darwin",
            "env": {
                "TELEGRAM_BOT_TOKEN": "fake-token",
                "CLAUDE_MAX_SESSION_HOURS": "4",
                "CLAUDE_IDLE_TIMEOUT": "600",
            },
        }
        (tmp_path / "install.conf").write_text(json.dumps(prior_conf))

        def mutate(base):
            # Empty input accepts the prompt prefill, which reads the
            # legacy keys as fallback defaults.
            base[base.index("0")] = ""
            base[base.index("1800")] = ""

        env = self._run_claude_chain(tmp_path, monkeypatch, mutate)
        assert env["AGENT_MAX_SESSION_HOURS"] == "4"
        assert env["AGENT_IDLE_TIMEOUT"] == "600"
        assert "CLAUDE_MAX_SESSION_HOURS" not in env
        assert "CLAUDE_IDLE_TIMEOUT" not in env


class TestBuildCodexLoginReminder:
    """Post-install codex subscription-auth reminder policy.

    The reminder is global-only: it fires when DEFAULT_BACKEND=codex AND
    auth mode is subscription. Per-user `backend: codex` entries
    in users.yaml DO NOT trigger the reminder; mixed-backend installs
    are operator-managed and the reminder would be wizard noise for
    operators who chose a non-codex global backend.

    The reminder text is generic ("log in as the target os_user") and
    does NOT enumerate per-user os_users. Operators read users.yaml
    themselves to know which accounts need codex login.
    """

    @staticmethod
    def _write_users_yaml(tmp_path, entries: list[dict]) -> Path:
        path = tmp_path / "users.yaml"
        path.write_text(yaml.safe_dump({"users": entries}))
        return path

    def test_global_codex_subscription_emits_reminder(self, tmp_path):
        """The one case the reminder should fire."""
        from kai.install import _build_codex_login_reminder

        users_yaml = self._write_users_yaml(
            tmp_path,
            [{"telegram_id": 1, "name": "alice", "role": "admin", "os_user": "alice"}],
        )
        env = {"DEFAULT_BACKEND": "codex", "CODEX_AUTH_MODE": "subscription"}
        text = _build_codex_login_reminder(env, "kai", users_yaml_path=users_yaml)
        assert text is not None
        assert "Codex subscription auth required:" in text
        assert "codex login" in text

    def test_default_auth_mode_is_subscription(self, tmp_path):
        """Missing CODEX_AUTH_MODE defaults to subscription; reminder fires."""
        from kai.install import _build_codex_login_reminder

        users_yaml = self._write_users_yaml(tmp_path, [])
        env = {"DEFAULT_BACKEND": "codex"}
        text = _build_codex_login_reminder(env, "kai", users_yaml_path=users_yaml)
        assert text is not None

    def test_reminder_is_generic_no_per_user_enumeration(self, tmp_path):
        """The reminder text does NOT enumerate users.yaml os_users.

        Pre-#556 the reminder listed `    alice ~$ codex login`,
        `    bob ~$ codex login`, etc. Post-#556 it's a single generic
        line. The wizard does not read users.yaml for this reminder
        any more; mixed-backend installs are operator-managed.
        """
        from kai.install import _build_codex_login_reminder

        users_yaml = self._write_users_yaml(
            tmp_path,
            [
                {"telegram_id": 1, "name": "alice", "role": "admin", "os_user": "alice"},
                {"telegram_id": 2, "name": "bob", "role": "user", "os_user": "bob"},
            ],
        )
        env = {"DEFAULT_BACKEND": "codex"}
        text = _build_codex_login_reminder(env, "kai", users_yaml_path=users_yaml)
        assert text is not None
        # No per-user enumeration. Specific os_user lines must NOT appear.
        assert "alice ~$ codex login" not in text
        assert "bob ~$ codex login" not in text
        # Generic <os_user> placeholder DOES appear.
        assert "<os_user> ~$ codex login" in text

    def test_claude_global_with_per_user_codex_returns_none(self, tmp_path):
        """Mixed-backend installs receive no reminder.

        Pre-#556 the reminder fired when users.yaml had any codex-
        effective entry. Post-#556 the gate is global-only: a per-user
        codex override on a non-codex global install means the operator
        owns codex setup out-of-band; the wizard does not preflight it.
        """
        from kai.install import _build_codex_login_reminder

        users_yaml = self._write_users_yaml(
            tmp_path,
            [
                {
                    "telegram_id": 1,
                    "name": "alice",
                    "role": "admin",
                    "backend": "codex",
                    "os_user": "alice",
                }
            ],
        )
        env = {"DEFAULT_BACKEND": "claude"}
        assert _build_codex_login_reminder(env, "kai", users_yaml_path=users_yaml) is None

    def test_pure_claude_returns_none(self, tmp_path):
        """No codex surface, no reminder."""
        from kai.install import _build_codex_login_reminder

        users_yaml = self._write_users_yaml(
            tmp_path,
            [{"telegram_id": 1, "name": "alice", "role": "admin", "os_user": "alice"}],
        )
        env = {"DEFAULT_BACKEND": "claude"}
        assert _build_codex_login_reminder(env, "kai", users_yaml_path=users_yaml) is None

    def test_codex_api_key_mode_returns_none(self, tmp_path):
        """API-key auth needs no per-user login; the env var ships the token."""
        from kai.install import _build_codex_login_reminder

        users_yaml = self._write_users_yaml(
            tmp_path,
            [{"telegram_id": 1, "name": "alice", "role": "admin", "os_user": "alice"}],
        )
        env = {"DEFAULT_BACKEND": "codex", "CODEX_AUTH_MODE": "api_key"}
        assert _build_codex_login_reminder(env, "kai", users_yaml_path=users_yaml) is None

    def test_missing_agent_backend_returns_none(self, tmp_path):
        """No explicit codex default, no reminder."""
        from kai.install import _build_codex_login_reminder

        users_yaml = self._write_users_yaml(tmp_path, [])
        assert _build_codex_login_reminder({}, "kai", users_yaml_path=users_yaml) is None


# ── Per-user goose provider keys ─────────────────────────────────────


class TestUsersYamlGooseProviders:
    """Direct tests for the wizard's per-user goose provider scan.

    The helper reads the canonical users.yaml and returns the distinct
    providers goose entries need API keys for; everything that is not
    a well-formed goose entry degrades to an empty result so the
    wizard never crashes mid-flow on user-owned YAML."""

    @staticmethod
    def _write(tmp_path, content: str) -> Path:
        p = tmp_path / "users.yaml"
        p.write_text(content, encoding="utf-8")
        return p

    def test_goose_entry_with_explicit_provider(self, tmp_path):
        p = self._write(
            tmp_path,
            "users:\n  - telegram_id: 1\n    name: a\n    backend: goose\n    provider: deepseek\n",
        )
        assert _users_yaml_goose_providers(p, "") == ["deepseek"]

    def test_mixed_case_normalized_like_runtime(self, tmp_path):
        """Mixed-case `Goose` / `DeepSeek` are valid at runtime (the
        loader lowercases both fields before validation), so the scan
        must match the backend case-insensitively and emit the
        canonical provider form PROVIDER_KEY_VARS is keyed on."""
        p = self._write(
            tmp_path,
            "users:\n  - telegram_id: 1\n    name: a\n    backend: Goose\n    provider: DeepSeek\n",
        )
        assert _users_yaml_goose_providers(p, "") == ["deepseek"]

    def test_falls_back_to_global_provider(self, tmp_path):
        """An entry that omits provider inherits the global
        provider, mirroring the runtime cascade."""
        p = self._write(tmp_path, "users:\n  - telegram_id: 1\n    name: a\n    backend: goose\n")
        assert _users_yaml_goose_providers(p, "deepseek") == ["deepseek"]

    def test_no_global_fallback_yields_nothing(self, tmp_path):
        """No provider anywhere: the scan returns nothing and the
        runtime's users.yaml validation owns the error."""
        p = self._write(tmp_path, "users:\n  - telegram_id: 1\n    name: a\n    backend: goose\n")
        assert _users_yaml_goose_providers(p, "") == []

    def test_distinct_providers_deduplicated_and_sorted(self, tmp_path):
        p = self._write(
            tmp_path,
            "users:\n"
            "  - telegram_id: 1\n    name: a\n    backend: goose\n    provider: openai\n"
            "  - telegram_id: 2\n    name: b\n    backend: goose\n    provider: deepseek\n"
            "  - telegram_id: 3\n    name: c\n    backend: goose\n    provider: deepseek\n",
        )
        assert _users_yaml_goose_providers(p, "") == ["deepseek", "openai"]

    def test_non_goose_entries_contribute_nothing(self, tmp_path):
        """claude / codex / opencode per-user auth is per-OS-user state
        the wizard does not manage; only goose rides the daemon env."""
        p = self._write(
            tmp_path,
            "users:\n"
            "  - telegram_id: 1\n    name: a\n    backend: opencode\n    provider: deepseek\n"
            "  - telegram_id: 2\n    name: b\n    backend: codex\n"
            "  - telegram_id: 3\n    name: c\n",
        )
        assert _users_yaml_goose_providers(p, "deepseek") == []

    def test_missing_file_degrades_to_empty(self, tmp_path):
        assert _users_yaml_goose_providers(tmp_path / "absent.yaml", "deepseek") == []

    def test_malformed_yaml_degrades_to_empty(self, tmp_path):
        p = self._write(tmp_path, "users: [unclosed\n")
        assert _users_yaml_goose_providers(p, "deepseek") == []

    def test_non_dict_entries_skipped(self, tmp_path):
        p = self._write(tmp_path, "users:\n  - 42\n  - goose\n")
        assert _users_yaml_goose_providers(p, "deepseek") == []


class TestWizardPerUserGooseProviderKeys:
    """Wizard collection of provider API keys for per-user goose
    entries, the deepseek PROVIDER_KEY_VARS row, and the provider-key
    preservation pass on env regeneration."""

    PLAIN_USERS_YAML = "users:\n  - telegram_id: 1\n    name: alice\n    role: admin\n"
    GOOSE_DEEPSEEK_USERS_YAML = (
        "users:\n"
        "  - telegram_id: 1\n"
        "    name: alice\n"
        "    role: admin\n"
        "  - telegram_id: 2\n"
        "    name: bob\n"
        "    backend: goose\n"
        "    provider: deepseek\n"
    )

    @staticmethod
    def _setup(monkeypatch, tmp_path, users_yaml: str, existing_env: dict | None = None) -> None:
        """Wizard sandbox: users.yaml exists with the given content,
        install.conf optionally seeded, model prompt mocked."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(
            "kai.install._backend_choices_for_config",
            lambda service_user: ["claude", "codex", "goose", "opencode"],
        )
        TestCmdConfig._simulate_existing_etc_users_yaml(monkeypatch, users_yaml)
        monkeypatch.setattr(
            "kai.install._prompt_default_model",
            lambda backend, prov, default: "sonnet",
        )
        if existing_env is not None:
            (tmp_path / "install.conf").write_text(json.dumps({"env": existing_env, "version": 1}))

    @staticmethod
    def _run(monkeypatch, tmp_path, inputs: list[str]) -> dict:
        feed = iter(inputs)
        monkeypatch.setattr("builtins.input", lambda prompt: next(feed))
        _cmd_config()
        return json.loads((tmp_path / "install.conf").read_text())["env"]

    def test_deepseek_has_a_provider_key_var(self):
        """The map row the whole goose-on-deepseek flow keys on: the
        wizard prompt, the env emission, and refresh_models all
        resolve the var through PROVIDER_KEY_VARS."""
        from kai.config import PROVIDER_KEY_VARS

        assert PROVIDER_KEY_VARS["deepseek"] == "DEEPSEEK_API_KEY"

    def test_peruser_goose_entry_prompts_for_key(self, tmp_path, monkeypatch, capsys):
        """Global claude install with a per-user goose+deepseek entry:
        the wizard prompts for DEEPSEEK_API_KEY but does not collect a
        goose binary path; backend commands come from the registry."""
        self._setup(monkeypatch, tmp_path, self.GOOSE_DEEPSEEK_USERS_YAML)
        inputs = TestCmdConfigDefaultModelDispatch._inputs_for_claude_backend()
        inputs.insert(inputs.index("claude") + 1, "sk-ds-peruser")

        env = self._run(monkeypatch, tmp_path, inputs)

        assert env["DEEPSEEK_API_KEY"] == "sk-ds-peruser"
        assert "GOOSE_BIN" not in env
        out = capsys.readouterr().out
        assert "goose entry on deepseek" in out
        assert "DEEPSEEK_API_KEY" in out

    def test_no_goose_entries_means_no_prompt(self, tmp_path, monkeypatch):
        """Plain users.yaml: the unmodified claude input chain must be
        consumed exactly (an unexpected key prompt would raise
        StopIteration), and no key lands in env."""
        self._setup(monkeypatch, tmp_path, self.PLAIN_USERS_YAML)

        env = self._run(monkeypatch, tmp_path, TestCmdConfigDefaultModelDispatch._inputs_for_claude_backend())

        assert "DEEPSEEK_API_KEY" not in env

    def test_global_goose_deepseek_prompts_for_key(self, tmp_path, monkeypatch, capsys):
        """Global goose+deepseek: the key prompt fires through the
        global block now that deepseek has a PROVIDER_KEY_VARS row;
        the auth-less fallback message must not appear."""
        self._setup(monkeypatch, tmp_path, self.PLAIN_USERS_YAML)
        monkeypatch.setattr("kai.install._validate_goose_bin", lambda p: bool(p))
        inputs = list(TestCmdConfigDefaultModelDispatch._inputs_for_goose_openai())
        inputs[inputs.index("openai")] = "deepseek"
        inputs[inputs.index("openai-key")] = "sk-ds-global"

        env = self._run(monkeypatch, tmp_path, inputs)

        assert env["DEFAULT_PROVIDER"] == "deepseek"
        assert env["DEEPSEEK_API_KEY"] == "sk-ds-global"
        assert "does not require an API key" not in capsys.readouterr().out

    def test_peruser_key_already_collected_globally_not_reprompted(self, tmp_path, monkeypatch):
        """Global goose+deepseek AND a per-user goose+deepseek entry:
        the global block collects DEEPSEEK_API_KEY, so the per-user
        scan must not prompt again (the unmodified goose chain is
        consumed exactly)."""
        self._setup(monkeypatch, tmp_path, self.GOOSE_DEEPSEEK_USERS_YAML)
        monkeypatch.setattr("kai.install._validate_goose_bin", lambda p: bool(p))
        inputs = list(TestCmdConfigDefaultModelDispatch._inputs_for_goose_openai())
        inputs[inputs.index("openai")] = "deepseek"
        inputs[inputs.index("openai-key")] = "sk-ds-once"

        env = self._run(monkeypatch, tmp_path, inputs)

        assert env["DEEPSEEK_API_KEY"] == "sk-ds-once"

    def test_stored_key_survives_unrelated_regeneration(self, tmp_path, monkeypatch):
        """A provider key already in the env survives a wizard re-run
        whose prompts never fire (the env dict is rebuilt fresh, so
        without the preservation pass the key would silently vanish)."""
        self._setup(
            monkeypatch,
            tmp_path,
            self.PLAIN_USERS_YAML,
            existing_env={"TELEGRAM_BOT_TOKEN": "fake-token", "DEEPSEEK_API_KEY": "sk-stored"},
        )

        env = self._run(monkeypatch, tmp_path, TestCmdConfigDefaultModelDispatch._inputs_for_claude_backend())

        assert env["DEEPSEEK_API_KEY"] == "sk-stored"

    def test_codex_subscription_does_not_resurrect_openai_key(self, tmp_path, monkeypatch):
        """The preservation pass defers to the codex auth flow's
        ownership of OPENAI_API_KEY: switching a codex install to
        subscription mode sheds the stored key, and preservation must
        not carry it back in."""
        self._setup(
            monkeypatch,
            tmp_path,
            self.PLAIN_USERS_YAML,
            existing_env={"TELEGRAM_BOT_TOKEN": "fake-token", "OPENAI_API_KEY": "sk-old"},
        )
        monkeypatch.setattr("kai.install._validate_codex_bin", lambda p: bool(p))

        env = self._run(monkeypatch, tmp_path, TestCmdConfigDefaultModelDispatch._inputs_for_codex_subscription())

        assert "OPENAI_API_KEY" not in env


# ── Per-user backend binary collection ───────────────────────────────


class TestUsersYamlAgentBackends:
    """Direct tests for the wizard's per-user backend scan; parsing
    and degrade behavior are shared with `_users_yaml_entries`."""

    @staticmethod
    def _write(tmp_path, content: str) -> Path:
        p = tmp_path / "users.yaml"
        p.write_text(content, encoding="utf-8")
        return p

    def test_collects_distinct_backends(self, tmp_path):
        p = self._write(
            tmp_path,
            "users:\n"
            "  - telegram_id: 1\n    name: a\n    backend: goose\n"
            "  - telegram_id: 2\n    name: b\n    backend: codex\n"
            "  - telegram_id: 3\n    name: c\n    backend: codex\n"
            "  - telegram_id: 4\n    name: d\n",
        )
        assert _users_yaml_agent_backends(p) == {"goose", "codex"}

    def test_strips_whitespace_skips_non_strings(self, tmp_path):
        p = self._write(
            tmp_path,
            "users:\n  - telegram_id: 1\n    name: a\n    backend: ' codex '\n  - telegram_id: 2\n    name: b\n    backend: 7\n",
        )
        assert _users_yaml_agent_backends(p) == {"codex"}

    def test_mixed_case_normalized_like_runtime(self, tmp_path):
        """The runtime loader accepts `backend` case-insensitively
        (str.strip().lower() before validation), so mixed-case entries
        route users at runtime; the scan must produce the canonical
        lower-case form or the prompt gates silently miss them."""
        p = self._write(
            tmp_path,
            "users:\n  - telegram_id: 1\n    name: a\n    backend: Codex\n  - telegram_id: 2\n    name: b\n    backend: ' GOOSE '\n",
        )
        assert _users_yaml_agent_backends(p) == {"codex", "goose"}

    def test_missing_file_degrades_to_empty(self, tmp_path):
        assert _users_yaml_agent_backends(tmp_path / "absent.yaml") == set()

    def test_malformed_yaml_degrades_to_empty(self, tmp_path):
        p = self._write(tmp_path, "users: [unclosed\n")
        assert _users_yaml_agent_backends(p) == set()


class TestWizardPerUserBackendsUseRegistry:
    """Per-user backend entries do not authorize config-time binary paths."""

    CODEX_USERS_YAML = (
        "users:\n"
        "  - telegram_id: 1\n"
        "    name: alice\n"
        "    role: admin\n"
        "  - telegram_id: 2\n"
        "    name: bob\n"
        "    backend: codex\n"
    )
    PLAIN_USERS_YAML = "users:\n  - telegram_id: 1\n    name: alice\n    role: admin\n"

    @staticmethod
    def _setup(monkeypatch, tmp_path, users_yaml: str, existing_env: dict | None = None) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        monkeypatch.setattr(
            "kai.install._backend_choices_for_config",
            lambda service_user: ["claude", "codex", "goose", "opencode"],
        )
        TestCmdConfig._simulate_existing_etc_users_yaml(monkeypatch, users_yaml)
        monkeypatch.setattr(
            "kai.install._prompt_default_model",
            lambda backend, prov, default: "sonnet",
        )
        if existing_env is not None:
            (tmp_path / "install.conf").write_text(json.dumps({"env": existing_env, "version": 1}))

    @staticmethod
    def _run(monkeypatch, tmp_path, inputs: list[str]) -> dict:
        feed = iter(inputs)
        monkeypatch.setattr("builtins.input", lambda prompt: next(feed))
        _cmd_config()
        return json.loads((tmp_path / "install.conf").read_text())["env"]

    def test_peruser_codex_entry_does_not_prompt_for_binary(self, tmp_path, monkeypatch):
        self._setup(monkeypatch, tmp_path, self.CODEX_USERS_YAML)
        inputs = TestCmdConfigDefaultModelDispatch._inputs_for_claude_backend()

        env = self._run(monkeypatch, tmp_path, inputs)

        assert "CODEX_BIN" not in env

    def test_global_codex_does_not_reprompt(self, tmp_path, monkeypatch):
        self._setup(monkeypatch, tmp_path, self.CODEX_USERS_YAML)

        env = self._run(monkeypatch, tmp_path, TestCmdConfigDefaultModelDispatch._inputs_for_codex_subscription())

        assert env["DEFAULT_BACKEND"] == "codex"
        assert "CODEX_BIN" not in env

    def test_plain_users_yaml_no_binary_prompt(self, tmp_path, monkeypatch):
        """No per-user backend entries: the unmodified claude chain is
        consumed exactly and no binary key lands in env."""
        self._setup(monkeypatch, tmp_path, self.PLAIN_USERS_YAML)

        env = self._run(monkeypatch, tmp_path, TestCmdConfigDefaultModelDispatch._inputs_for_claude_backend())

        assert "CODEX_BIN" not in env
        assert "GOOSE_BIN" not in env
        assert "OPENCODE_BIN" not in env

    def test_stored_binary_offered_as_default(self, tmp_path, monkeypatch):
        """Re-run with a stored CODEX_BIN strips it on resave."""
        self._setup(
            monkeypatch,
            tmp_path,
            self.CODEX_USERS_YAML,
            existing_env={"TELEGRAM_BOT_TOKEN": "fake-token", "CODEX_BIN": "/stored/bin/codex"},
        )
        inputs = TestCmdConfigDefaultModelDispatch._inputs_for_claude_backend()

        env = self._run(monkeypatch, tmp_path, inputs)

        assert "CODEX_BIN" not in env

    def test_mixed_case_codex_entry_still_does_not_prompt(self, tmp_path, monkeypatch):
        """`backend: Codex` is valid at runtime (the loader
        lowercases before validation) and routes the user, so the
        scan must normalize the same way; a casing difference must
        not skip the binary prompt and reintroduce the startup-gate
        failure."""
        mixed_yaml = (
            "users:\n"
            "  - telegram_id: 1\n"
            "    name: alice\n"
            "    role: admin\n"
            "  - telegram_id: 2\n"
            "    name: bob\n"
            "    backend: Codex\n"
        )
        self._setup(monkeypatch, tmp_path, mixed_yaml)
        inputs = TestCmdConfigDefaultModelDispatch._inputs_for_claude_backend()

        env = self._run(monkeypatch, tmp_path, inputs)

        assert "CODEX_BIN" not in env

    def test_peruser_opencode_entry_does_not_prompt_for_binary(self, tmp_path, monkeypatch):
        opencode_yaml = (
            "users:\n"
            "  - telegram_id: 1\n"
            "    name: alice\n"
            "    role: admin\n"
            "  - telegram_id: 2\n"
            "    name: bob\n"
            "    backend: opencode\n"
        )
        self._setup(monkeypatch, tmp_path, opencode_yaml)
        inputs = TestCmdConfigDefaultModelDispatch._inputs_for_claude_backend()

        env = self._run(monkeypatch, tmp_path, inputs)

        assert "OPENCODE_BIN" not in env


class TestProtectedRuntimeStorageProvisioning:
    @staticmethod
    def _profile(backend: str, *, os_user: str | None = None) -> ProtectedRuntimeProfile:
        provider = {
            "claude": "anthropic",
            "codex": "openai",
            "goose": "openai",
            "opencode": "openai",
            "pi": "openai",
        }[backend]
        model = "sonnet" if backend == "claude" else "gpt-5.5"
        return ProtectedRuntimeProfile(
            profile_id=RuntimeProfileId("rtp_" + "f" * 32),
            runtime_config_id=987654321,
            display_name="Browser-only human",
            os_user=os_user,
            backend=backend,
            provider=provider,
            model=model,
            timeout_seconds=120,
            allowed_services=(),
            home_workspace=None,
            workspace_base=None,
            allowed_workspaces=(),
        )

    @staticmethod
    def _setup_project(tmp_path: Path, monkeypatch) -> tuple[Path, Path]:
        project = tmp_path / "project"
        templates = project / "templates"
        (templates / ".claude").mkdir(parents=True)
        (templates / "AGENTS.md").write_text("# Canonical identity\n")
        (templates / ".claude" / "MEMORY.md").write_text("# Memory\n")
        (templates / ".claude" / "PREFERENCES.md").write_text("# Preferences\n")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", project)
        monkeypatch.setattr("kai.install.os.chown", lambda *_args: None)

        data_path = tmp_path / "data"
        for name in ("logs", "memory", "files", "history"):
            (data_path / name).mkdir(parents=True, exist_ok=True)
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users: []\n")
        return data_path, users_yaml

    @pytest.mark.asyncio
    @pytest.mark.parametrize("backend", ["claude", "codex", "goose", "opencode", "pi"])
    async def test_profile_without_telegram_user_gets_canonical_managed_storage(
        self,
        backend,
        tmp_path,
        monkeypatch,
    ):
        data_path, users_yaml = self._setup_project(tmp_path, monkeypatch)
        profile = self._profile(backend)
        registry = WorkshopRuntimeProfileRegistry((profile,))
        store = await WorkshopEventStore.open(data_path / "kai.db")
        await bootstrap_default_workshop(
            store,
            (
                BootstrapHuman(
                    "Browser human",
                    "admin",
                    "workshop",
                    "browser-human",
                    "browser-direct",
                    profile.profile_id,
                ),
            ),
        )
        await store.close()

        targets = _runtime_storage_targets(data_path, registry, users_yaml)
        assert len(targets) == 1
        assert targets[0].storage_name.startswith("prn_")

        _apply_migrate(
            data_path,
            tmp_path / "install",
            svc_uid=os.getuid(),
            svc_gid=os.getgid(),
            dry_run=False,
            users_yaml_path=users_yaml,
            runtime_storage_targets=targets,
        )

        principal_name = targets[0].storage_name
        home = data_path / "home" / principal_name
        assert (home / "AGENTS.md").read_text() == "# Canonical identity\n"
        assert (data_path / "memory" / principal_name / "MEMORY.md").is_file()
        assert (data_path / "preferences" / principal_name / "PREFERENCES.md").is_file()
        claude_adapter = home / ".claude" / "CLAUDE.md"
        if backend == "claude":
            assert claude_adapter.read_text() == "@../AGENTS.md\n"
        else:
            assert not claude_adapter.exists()
        assert not (data_path / "home" / str(profile.runtime_config_id)).exists()

    @pytest.mark.asyncio
    async def test_initialized_assignments_fail_closed_for_unmapped_profile(self, tmp_path):
        data_path = tmp_path / "data"
        data_path.mkdir()
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users: []\n")
        profile = self._profile("codex")
        store = await WorkshopEventStore.open(data_path / "kai.db")
        await bootstrap_default_workshop(
            store,
            (BootstrapHuman("Other", "admin", "workshop", "other", "other-direct", None),),
        )
        await store.close()

        with pytest.raises(RuntimeError, match="exactly one canonical human owner"):
            _runtime_storage_targets(
                data_path,
                WorkshopRuntimeProfileRegistry((profile,)),
                users_yaml,
            )

    @pytest.mark.asyncio
    async def test_new_compatibility_user_uses_future_bootstrap_principal(self, tmp_path):
        data_path = tmp_path / "data"
        data_path.mkdir()
        users_yaml = tmp_path / "users.yaml"
        profile = self._profile("codex")
        users_yaml.write_text(
            f"users:\n  - telegram_id: {profile.runtime_config_id}\n    name: New Telegram human\n    role: member\n"
        )
        store = await WorkshopEventStore.open(data_path / "kai.db")
        await bootstrap_default_workshop(
            store,
            (BootstrapHuman("Existing", "admin", "telegram", "1", "1", None),),
        )
        async with store.connection.execute("SELECT id FROM workshops") as cursor:
            row = await cursor.fetchone()
        assert row is not None
        workshop_id = WorkshopId(str(row[0]))
        await store.close()

        targets = _runtime_storage_targets(
            data_path,
            WorkshopRuntimeProfileRegistry((profile,)),
            users_yaml,
        )

        assert targets[0].storage_name == str(
            bootstrap_human_principal_id(
                workshop_id,
                "telegram",
                str(profile.runtime_config_id),
            )
        )
        assert targets[0].storage_name != str(profile.runtime_config_id)

    def test_profiles_cannot_share_one_canonical_storage_owner(self, tmp_path, monkeypatch):
        first = self._profile("codex")
        second = replace(
            first,
            profile_id=RuntimeProfileId("rtp_" + "e" * 32),
            runtime_config_id=123456789,
        )
        shared_principal = "prn_" + "a" * 32
        monkeypatch.setattr(
            "kai.install._runtime_profile_principal_names",
            lambda _data_path: (
                True,
                None,
                {
                    str(first.profile_id): shared_principal,
                    str(second.profile_id): shared_principal,
                },
            ),
        )
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users: []\n")

        with pytest.raises(RuntimeError, match="same canonical human storage owner"):
            _runtime_storage_targets(
                tmp_path / "data",
                WorkshopRuntimeProfileRegistry((first, second)),
                users_yaml,
            )

    def test_symlinked_database_never_uses_uninitialized_fallback(self, tmp_path):
        data_path = tmp_path / "data"
        data_path.mkdir()
        real_db = tmp_path / "real.db"
        real_db.touch()
        (data_path / "kai.db").symlink_to(real_db)
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users: []\n")

        with pytest.raises(RuntimeError, match="Refusing symlinked database"):
            _runtime_storage_targets(
                data_path,
                WorkshopRuntimeProfileRegistry((self._profile("codex"),)),
                users_yaml,
            )

    def test_profile_only_missing_os_account_fails_before_provisioning(self, tmp_path, monkeypatch):
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users: []\n")
        profile = self._profile("codex", os_user="missing-user")
        monkeypatch.setattr(
            "kai.install._runtime_profile_principal_names",
            lambda _data_path: (True, None, {str(profile.profile_id): "prn_" + "a" * 32}),
        )
        monkeypatch.setattr(
            "kai.install.pwd.getpwnam",
            lambda name: (_ for _ in ()).throw(KeyError(name)),
        )

        with pytest.raises(ValueError, match="does not exist on this host"):
            _runtime_storage_targets(
                tmp_path / "data",
                WorkshopRuntimeProfileRegistry((profile,)),
                users_yaml,
            )

        assert not (tmp_path / "data").exists()

    def test_migrated_profile_conflicting_canonical_owner_fails_closed(self, tmp_path, monkeypatch):
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users:\n  - telegram_id: 987654321\n    name: Existing human\n    role: admin\n")
        profile = self._profile("codex")
        monkeypatch.setattr(
            "kai.install._runtime_profile_principal_names",
            lambda _data_path: (True, None, {str(profile.profile_id): "prn_" + "a" * 32}),
        )
        monkeypatch.setattr(
            "kai.install._canonical_principal_storage_names",
            lambda _data_path: {str(profile.runtime_config_id): "prn_" + "b" * 32},
        )

        with pytest.raises(RuntimeError, match="conflicts with its canonical compatibility owner"):
            _runtime_storage_targets(
                tmp_path / "data",
                WorkshopRuntimeProfileRegistry((profile,)),
                users_yaml,
            )

    @pytest.mark.asyncio
    async def test_profile_driven_dry_run_describes_canonical_home_without_writing(
        self,
        tmp_path,
        monkeypatch,
        capsys,
    ):
        data_path, users_yaml = self._setup_project(tmp_path, monkeypatch)
        profile = self._profile("codex")
        registry = WorkshopRuntimeProfileRegistry((profile,))
        store = await WorkshopEventStore.open(data_path / "kai.db")
        await bootstrap_default_workshop(
            store,
            (
                BootstrapHuman(
                    "Browser human",
                    "admin",
                    "workshop",
                    "browser-human",
                    "browser-direct",
                    profile.profile_id,
                ),
            ),
        )
        await store.close()
        targets = _runtime_storage_targets(data_path, registry, users_yaml)

        _apply_migrate(
            data_path,
            tmp_path / "install",
            svc_uid=os.getuid(),
            svc_gid=os.getgid(),
            dry_run=True,
            users_yaml_path=users_yaml,
            runtime_storage_targets=targets,
        )

        canonical_home = data_path / "home" / targets[0].storage_name
        assert f"Would create {canonical_home}" in capsys.readouterr().out
        assert not canonical_home.exists()
        assert not (data_path / "memory" / targets[0].storage_name).exists()

    @pytest.mark.asyncio
    async def test_status_reports_profile_storage_coverage_without_identifiers(
        self,
        tmp_path,
        monkeypatch,
    ):
        data_path, users_yaml = self._setup_project(tmp_path, monkeypatch)
        profile = self._profile("codex")
        store = await WorkshopEventStore.open(data_path / "kai.db")
        await bootstrap_default_workshop(
            store,
            (
                BootstrapHuman(
                    "Browser human",
                    "admin",
                    "workshop",
                    "browser-human",
                    "browser-direct",
                    profile.profile_id,
                ),
            ),
        )
        await store.close()
        targets = _runtime_storage_targets(
            data_path,
            WorkshopRuntimeProfileRegistry((profile,)),
            users_yaml,
        )
        _apply_migrate(
            data_path,
            tmp_path / "install",
            svc_uid=os.getuid(),
            svc_gid=os.getgid(),
            dry_run=False,
            users_yaml_path=users_yaml,
            runtime_storage_targets=targets,
        )

        policy = tmp_path / "runtime-profiles.yaml"
        policy.write_text(
            yaml.safe_dump(
                {
                    "version": 1,
                    "runtime_profiles": {
                        str(profile.profile_id): {
                            "display_name": profile.display_name,
                            "compatibility_runtime_config_id": profile.runtime_config_id,
                            "backend": profile.backend,
                            "provider": profile.provider,
                            "model": profile.model,
                            "timeout_seconds": profile.timeout_seconds,
                            "allowed_services": [],
                            "home_workspace": None,
                            "workspace_base": None,
                            "allowed_workspaces": [],
                        }
                    },
                }
            )
        )
        backends = tmp_path / "backends.yaml"
        backends.write_text("version: 1\nbackends:\n  codex: {}\n")
        service_user = pwd.getpwuid(os.getuid()).pw_name
        home = data_path / "home" / targets[0].storage_name
        service_entry = MagicMock(pw_uid=home.stat().st_uid, pw_gid=home.stat().st_gid)
        monkeypatch.setattr("kai.install.pwd.getpwnam", lambda _name: service_entry)

        status = _runtime_storage_status(
            data_path,
            service_user,
            policy,
            backends,
            users_yaml,
        )

        assert status == ("Workshop runtime storage: complete; profiles=1, managed=1, operator-managed=0, incomplete=0")
        assert str(profile.profile_id) not in status
        assert str(profile.runtime_config_id) not in status

        (home / "AGENTS.md").chmod(0o644)
        incomplete_status = _runtime_storage_status(
            data_path,
            service_user,
            policy,
            backends,
            users_yaml,
        )

        assert incomplete_status == (
            "Workshop runtime storage: INCOMPLETE; profiles=1, managed=1, "
            "operator-managed=0, incomplete=1; issues: home=0, identity=1, "
            "memory=0, preferences=0, temp=0"
        )
        assert str(profile.profile_id) not in incomplete_status
        assert str(profile.runtime_config_id) not in incomplete_status

    def test_profile_only_os_user_is_included_in_sudoers(self, tmp_path, monkeypatch):
        profile = self._profile("codex", os_user="browser-user")
        registry = WorkshopRuntimeProfileRegistry((profile,))
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users: []\n")
        observed_users: list[str] = []

        def fake_generate_sudoers(_service_user, os_users, **_kwargs):
            observed_users.extend(os_users)
            return "# test sudoers\n"

        monkeypatch.setattr("kai.install._generate_sudoers", fake_generate_sudoers)
        monkeypatch.setattr(
            "kai.install._backend_registry_entries",
            lambda *_args, **_kwargs: {"codex": {"command": str(tmp_path / "codex")}},
        )

        _apply_sudoers(
            "kai",
            dry_run=True,
            users_yaml_path=users_yaml,
            agent_backend="codex",
            runtime_profiles=registry,
        )

        assert observed_users == ["browser-user"]

    def test_profile_only_goose_os_user_receives_config(self, tmp_path, monkeypatch):
        import types

        profile = self._profile("goose", os_user="browser-user")
        registry = WorkshopRuntimeProfileRegistry((profile,))
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users: []\n")
        install_path = tmp_path / "install"
        (install_path / "config").mkdir(parents=True)
        (install_path / "config" / "goose-config.yaml").write_text("extensions: {}\n")
        service_home = tmp_path / "home" / "kai"
        browser_home = tmp_path / "home" / "browser-user"
        service_home.mkdir(parents=True)
        browser_home.mkdir(parents=True)
        monkeypatch.setattr("kai.install._user_home", lambda _name: str(service_home))
        monkeypatch.setattr("kai.install.shutil.which", lambda _name: "/usr/local/bin/goose")
        monkeypatch.setattr(
            "kai.install.pwd.getpwnam",
            lambda _name: types.SimpleNamespace(
                pw_dir=str(browser_home),
                pw_uid=os.getuid(),
                pw_gid=os.getgid(),
            ),
        )
        monkeypatch.setattr("kai.install._set_ownership", lambda *_args, **_kwargs: None)

        _apply_goose_config(
            "kai",
            install_path,
            os.getuid(),
            os.getgid(),
            dry_run=False,
            users_yaml_path=users_yaml,
            agent_backend="codex",
            runtime_profiles=registry,
        )

        assert (browser_home / ".config" / "goose" / "config.yaml").is_file()
