"""Tests for the protected installation module (install.py)."""

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from kai.install import (
    _LAUNCHD_LABEL,
    _apply_directories,
    _apply_goose_config,
    _apply_migrate,
    _apply_models,
    _apply_secrets,
    _apply_service,
    _apply_source,
    _apply_sudoers,
    _apply_venv,
    _check_path,
    _check_service_status,
    _check_traversal,
    _cmd_apply,
    _cmd_config,
    _cmd_status,
    _collect_os_users_from_yaml,
    _collect_user_memory_owners,
    _copy_tree,
    _file_checksum,
    _generate_env_file,
    _generate_launchd_plist,
    _generate_launcher_script,
    _generate_sudoers,
    _generate_systemd_unit,
    _generate_users_yaml,
    _migrate_identity_to_claude_md,
    _retire_install_home_claude,
    _retire_install_home_dir,
    _set_ownership,
    _src_checksum,
    _start_service,
    _stop_service,
    _user_home,
    _validate_chat_id,
    _validate_display_name,
    _validate_os_user,
    _validate_port,
    _validate_positive_float,
    _validate_positive_int,
    _validate_telegram_id,
    _validate_user_ids,
    cli,
)

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


class TestValidatePositiveFloat:
    def test_valid(self):
        assert _validate_positive_float("10.0") is True

    def test_zero(self):
        assert _validate_positive_float("0") is False

    def test_negative(self):
        assert _validate_positive_float("-1.5") is False

    def test_non_numeric(self):
        assert _validate_positive_float("abc") is False


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

    def test_roundtrip_with_loader(self, tmp_path, monkeypatch):
        """Generated YAML can be parsed by _load_user_configs("claude", "")."""
        from kai.config import _load_user_configs

        content = _generate_users_yaml("123456789", "alice", os_user="kai")
        yaml_path = tmp_path / "users.yaml"
        yaml_path.write_text(content)
        monkeypatch.setattr("kai.config.PROJECT_ROOT", tmp_path)
        # Skip the protected /etc/kai/ path so we read the tmp_path copy.
        monkeypatch.setattr("kai.config._read_protected_yaml", lambda _: None)
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

    def test_no_claude_user_rule_by_default(self):
        """No claude binary rule when claude_user is None."""
        result = _generate_sudoers("kai")
        assert "claude" not in result

    def test_claude_user_rule_anchored_to_service_user_home(self, monkeypatch):
        """
        The rule's claude binary path is anchored to the SERVICE user's
        home (~/.local/bin/claude under the service user, NOT the target
        user), because the bot's runtime spawn is `sudo -u <target> --
        claude` and sudo resolves the bare `claude` against the caller's
        (service user's) PATH. The rule path must match what sudo will
        actually try to execute.
        """
        monkeypatch.setattr("kai.install._user_home", lambda u: f"/home/{u}")
        result = _generate_sudoers("kai", claude_user="alice")
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
        result = _generate_sudoers("kai", claude_user="alice")
        assert "/home/kai/.local/bin/claude" in result
        assert "/some/other/users/local/bin/claude" not in result

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
        # Each target gets two rules now (claude + kill, #456). Count
        # the claude rules specifically to verify dedup at the target
        # level - the kill rules add their own occurrences of "(name)"
        # so a bare count("(name)") would double.
        assert result.count("kai ALL=(sellison) SETENV: NOPASSWD: ") == 1
        assert result.count("kai ALL=(bob) SETENV: NOPASSWD: ") == 1

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

    def test_legacy_claude_user_combined_with_os_users(self, monkeypatch):
        """Legacy CLAUDE_USER and yaml os_users coexist; deduped."""
        monkeypatch.setattr("kai.install._user_home", lambda u: f"/home/{u}")
        # claude_user and os_users overlap on "alice"; should produce
        # one ruleset (claude + kill rules, #456) for each distinct
        # target. Count claude rules specifically to skip kill-rule
        # noise on the same target name.
        result = _generate_sudoers("kai", claude_user="alice", os_users=["alice", "bob"])
        assert result.count("kai ALL=(alice) SETENV: NOPASSWD: ") == 1
        assert result.count("kai ALL=(bob) SETENV: NOPASSWD: ") == 1

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

    def test_invalid_claude_user_raises(self):
        """Legacy CLAUDE_USER env var path must also be validated."""
        with pytest.raises(ValueError, match="Invalid sudoers target user"):
            _generate_sudoers("kai", claude_user="alice\nroot ALL")


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


# ── Config subcommand ────────────────────────────────────────────────


class TestCmdConfig:
    @staticmethod
    def _block_etc_kai(monkeypatch):
        """Prevent the wizard from detecting /etc/kai/users.yaml on the host."""
        _real_exists = Path.exists

        def _exists_no_etc(self):
            if str(self) == "/etc/kai/users.yaml":
                return False
            return _real_exists(self)

        monkeypatch.setattr(Path, "exists", _exists_no_etc)

    def test_writes_install_conf(self, tmp_path, monkeypatch):
        """Config subcommand writes valid JSON to install.conf."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._block_etc_kai(monkeypatch)

        # Simulate user inputs for each prompt (in order)
        inputs = iter(
            [
                "/opt/kai",  # install dir
                "/var/lib/kai",  # data dir
                "kai",  # service user
                "darwin",  # platform
                "fake-token",  # bot token
                "12345",  # admin telegram ID
                "admin",  # admin display name
                "false",  # advanced user options
                "polling",  # transport
                "claude",  # agent backend
                "sonnet",  # model
                "120",  # timeout
                "10.0",  # budget
                "200000",  # max context window
                "80",  # autocompact pct
                "",  # claude effort level (take default "high")
                "8080",  # port
                "test-secret",  # webhook secret
                "~/Projects",  # workspace base
                "",  # allowed workspaces (empty)
                "false",  # pr review enabled
                "900",  # pr review timeout (seconds)
                "1.0",  # pr review budget (USD)
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
        assert conf["env"]["CLAUDE_MAX_CONTEXT_WINDOW"] == "200000"
        assert conf["env"]["CLAUDE_AUTOCOMPACT_PCT"] == "80"
        # ALLOWED_USER_IDS should not be in the env dict
        assert "ALLOWED_USER_IDS" not in conf["env"]
        # Default backend should not appear in env (only non-default values)
        assert "AGENT_BACKEND" not in conf["env"]
        # users.yaml should have been generated
        yaml_path = tmp_path / "users.yaml"
        assert yaml_path.exists()
        data = yaml.safe_load(yaml_path.read_text())
        assert data["users"][0]["telegram_id"] == 12345
        assert data["users"][0]["role"] == "admin"

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
        self._block_etc_kai(monkeypatch)

        inputs = iter(
            [
                "/opt/kai",  # install dir
                "/var/lib/kai",  # data dir
                "kai",  # service user
                "darwin",  # platform
                "fake-token",  # bot token
                "12345",  # admin telegram ID
                "admin",  # admin display name
                "true",  # advanced user options
                "testuser",  # os_user
                # no home_workspace prompt post-#353
                "polling",  # transport
                "claude",  # agent backend
                "sonnet",  # model
                "120",  # timeout
                "10.0",  # budget
                "200000",  # max context window
                "80",  # autocompact pct
                "",  # claude effort level (take default "high")
                "8080",  # port
                "test-secret",  # webhook secret
                "~/Projects",  # workspace base
                "",  # allowed workspaces (empty)
                "false",  # pr review enabled
                "900",  # pr review timeout (seconds)
                "1.0",  # pr review budget (USD)
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
        self._block_etc_kai(monkeypatch)

        inputs_basic = iter(
            [
                "/opt/kai",  # install dir
                "/var/lib/kai",  # data dir
                "kai",  # service user
                "darwin",  # platform
                "fake-token",  # bot token
                "12345",  # admin telegram ID
                "admin",  # admin display name
                "false",  # advanced user options -> no os_user, no home prompt
                "polling",  # transport
                "claude",  # agent backend
                "sonnet",  # model
                "120",  # timeout
                "10.0",  # budget
                "200000",  # max context window
                "80",  # autocompact pct
                "",  # claude effort level (take default "high")
                "8080",  # port
                "test-secret",  # webhook secret
                "~/Projects",  # workspace base
                "",  # allowed workspaces
                "false",  # pr review enabled
                "900",  # pr review timeout
                "1.0",  # pr review budget
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

        inputs_advanced = iter(
            [
                "/opt/kai",
                "/var/lib/kai",
                "kai",
                "darwin",
                "fake-token",
                "12345",
                "admin",
                "true",  # advanced -> os_user prompt
                "testuser",  # os_user (no home_workspace prompt should follow)
                "polling",
                "claude",
                "sonnet",
                "120",
                "10.0",
                "200000",
                "80",
                "",  # claude effort level (take default "high")
                "8080",
                "test-secret",
                "~/Projects",
                "",
                "false",
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

    def test_goose_backend_writes_env(self, tmp_path, monkeypatch):
        """Selecting goose backend writes AGENT_BACKEND to env."""
        monkeypatch.chdir(tmp_path)
        conf_path = tmp_path / "install.conf"
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._block_etc_kai(monkeypatch)

        # Pre-seed existing config with goose backend so the prompt
        # appears (it is gated behind an existing non-claude value).
        existing = {"version": 1, "env": {"AGENT_BACKEND": "goose"}}
        conf_path.write_text(json.dumps(existing))

        inputs = iter(
            [
                "/opt/kai",  # install dir
                "/var/lib/kai",  # data dir
                "kai",  # service user
                "darwin",  # platform
                "fake-token",  # bot token
                "12345",  # admin telegram ID
                "admin",  # admin display name
                "false",  # advanced user options
                "polling",  # transport
                "goose",  # agent backend (prompt shown because existing config has goose)
                "anthropic",  # goose provider
                "sk-ant-test-key",  # ANTHROPIC_API_KEY
                "sonnet",  # model
                "120",  # timeout
                "10.0",  # budget
                "200000",  # max context window
                "8080",  # port
                "test-secret",  # webhook secret
                "~/Projects",  # workspace base
                "",  # allowed workspaces (empty)
                "false",  # pr review enabled
                "900",  # pr review timeout (seconds)
                "1.0",  # pr review budget (USD)
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
        assert conf["env"]["AGENT_BACKEND"] == "goose"
        assert conf["env"]["LLM_PROVIDER"] == "anthropic"
        assert conf["env"]["ANTHROPIC_API_KEY"] == "sk-ant-test-key"

    def test_goose_ollama_no_api_key(self, tmp_path, monkeypatch):
        """Selecting ollama provider skips the API key prompt."""
        monkeypatch.chdir(tmp_path)
        conf_path = tmp_path / "install.conf"
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._block_etc_kai(monkeypatch)

        existing = {"version": 1, "env": {"AGENT_BACKEND": "goose"}}
        conf_path.write_text(json.dumps(existing))

        # No API key input after "ollama" - the prompt is skipped.
        inputs = iter(
            [
                "/opt/kai",  # install dir
                "/var/lib/kai",  # data dir
                "kai",  # service user
                "darwin",  # platform
                "fake-token",  # bot token
                "12345",  # admin telegram ID
                "admin",  # admin display name
                "false",  # advanced user options
                "polling",  # transport
                "goose",  # agent backend
                "ollama",  # goose provider (no key needed)
                "sonnet",  # model
                "120",  # timeout
                "10.0",  # budget
                "200000",  # max context window
                "8080",  # port
                "test-secret",  # webhook secret
                "~/Projects",  # workspace base
                "",  # allowed workspaces (empty)
                "false",  # pr review enabled
                "900",  # pr review timeout (seconds)
                "1.0",  # pr review budget (USD)
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
        assert conf["env"]["LLM_PROVIDER"] == "ollama"
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
            },
        }
        conf_path.write_text(json.dumps(existing))

        # Place an existing users.yaml so the wizard skips user prompts
        (tmp_path / "users.yaml").write_text("users:\n  - telegram_id: 999\n    name: existing\n    role: admin\n")

        # Press Enter for everything (accept all defaults)
        monkeypatch.setattr("builtins.input", lambda prompt: "")

        _cmd_config()

        conf = json.loads(conf_path.read_text())
        # Should preserve existing values when user accepts defaults
        assert conf["install_dir"] == "/custom/path"
        assert conf["env"]["TELEGRAM_BOT_TOKEN"] == "existing-token"
        # users.yaml should not have been overwritten
        output = capsys.readouterr().out
        assert "already configured" in output
        data = yaml.safe_load((tmp_path / "users.yaml").read_text())
        assert data["users"][0]["telegram_id"] == 999

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
          - Omits the autocompact, effort, and claude_user entries that
            the wizard now skips for non-claude backends per issue #380.
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
        # is "ollama" (local model, no auth).
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

        # Legacy CLAUDE_USER prompt in install.py is gated on
        # agent_backend == "claude" by issue #380. The
        # fixture entry is conditional so the input iterator does not
        # carry a surplus value when the prompt is skipped.
        claude_user_entry: list[str] = []
        if agent_backend == "claude":
            claude_user_entry = [""]

        # BUDGET_CEILING and PR_REVIEW_BUDGET_USD prompts are skipped
        # on the claude backend (issue #390): --max-budget-usd is no
        # longer emitted to claude --print argv, so prompting for a
        # value that is never enforced would be wizard noise. The
        # fixture entries are conditional to match the wizard's
        # runtime conditional. Same shape as claude_only_pre_webhook
        # immediately above; the inverted conditional reflects that
        # these prompts now fire ONLY on non-claude backends.
        budget_entry: list[str] = []
        pr_review_budget_entry: list[str] = []
        if agent_backend != "claude":
            budget_entry = ["10.0"]  # BUDGET_CEILING
            pr_review_budget_entry = ["1.0"]  # PR_REVIEW_BUDGET_USD

        return [
            "/opt/kai",  # install dir
            "/var/lib/kai",  # data dir
            "kai",  # service user
            "darwin",  # platform
            "fake-token",  # bot token
            "12345",  # admin telegram ID
            "admin",  # admin display name
            "false",  # advanced user options
            "polling",  # transport
            agent_backend,  # agent backend
            *backend_block,  # llm_provider + api_key (non-claude only)
            "sonnet",  # model
            "120",  # timeout
            *budget_entry,  # BUDGET_CEILING (non-claude only)
            "200000",  # max context window
            *claude_only_pre_webhook,  # autocompact + effort (claude only)
            "8080",  # port
            "test-secret",  # webhook secret
            "~/Projects",  # workspace base
            "",  # allowed workspaces
            "false",  # pr review enabled
            "900",  # pr review timeout
            *pr_review_budget_entry,  # PR_REVIEW_BUDGET_USD (non-claude only)
            "false",  # issue triage
            "",  # github notify chat id
            "false",  # voice
            "false",  # tts
            *claude_user_entry,  # claude user (claude only)
            *memory_block,
            "",  # perplexity key
        ]

    def test_memory_disabled_omits_env_keys(self, tmp_path, monkeypatch):
        """MEMORY_ENABLED=false produces no MEMORY_* env entries."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._block_etc_kai(monkeypatch)

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
        self._block_etc_kai(monkeypatch)

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
        self._block_etc_kai(monkeypatch)

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
        self._block_etc_kai(monkeypatch)

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

        # Sanity: AGENT_BACKEND was emitted (it always is for non-claude).
        assert env.get("AGENT_BACKEND") == "goose"

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
        self._block_etc_kai(monkeypatch)

        # Pre-seed: an install.conf as if the operator had previously
        # configured the wizard under claude with non-default values
        # for all three of the env keys gated by this PR.
        pre_existing = {
            "version": 1,
            "env": {
                "AGENT_BACKEND": "claude",
                "CLAUDE_AUTOCOMPACT_PCT": "50",
                "CLAUDE_EFFORT_LEVEL": "xhigh",
                "CLAUDE_USER": "kai",
            },
        }
        conf_path.write_text(json.dumps(pre_existing))

        # Re-run wizard, switch to goose this time.
        inputs = iter(self._base_inputs(["false"], agent_backend="goose"))
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        _cmd_config()

        conf = json.loads(conf_path.read_text())
        env = conf["env"]
        # All three previously-set Claude-only keys must be absent.
        # AGENT_BACKEND must reflect the new selection.
        assert "CLAUDE_AUTOCOMPACT_PCT" not in env
        assert "CLAUDE_EFFORT_LEVEL" not in env
        assert "CLAUDE_USER" not in env
        assert env.get("AGENT_BACKEND") == "goose"

    def test_memory_enabled_writes_tunables(self, tmp_path, monkeypatch):
        """MEMORY_ENABLED=true with extraction writes the chosen tunables."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._block_etc_kai(monkeypatch)

        # Memory on, extraction on, custom timeout + consolidation candidates + episode tunables + token budget + search limit.
        # Budget prompts (extraction, episode) are skipped on the claude
        # backend per issue #390 - --max-budget-usd is no longer emitted
        # to claude --print argv at either stage, so the wizard does not
        # ask for a value that would never be enforced.
        memory_block = [
            "true",  # memory enabled
            "true",  # extraction enabled (claude backend)
            "60",  # extraction timeout seconds (#345)
            "5",  # consolidation candidates (non-default, exercises emission branch)
            "5",  # episode classifier context turns (#392, non-default exercises emission)
            # Episode model: empty input now resolves to the wizard's
            # default of "claude-sonnet-4-6" rather than the empty
            # string. To exercise the explicit-override emission
            # branch, type a different model literal here.
            "claude-haiku-4-5-20251001",  # episode model (explicit override)
            "60",  # episode timeout seconds (non-default)
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
        # Budget keys absent on claude backend: prompt is skipped, value
        # stays at dataclass default, double-gated emission suppresses.
        assert "MEMORY_EXTRACTION_BUDGET_USD" not in env
        assert env["MEMORY_EXTRACTION_TIMEOUT_S"] == "60"
        assert env["MEMORY_CONSOLIDATION_CANDIDATES_N"] == "5"
        # Episode-classifier context window (#392): operator picked
        # non-default 5, so the emission gate fires and the env entry
        # is written.
        assert env["EPISODE_CLASSIFIER_CONTEXT_TURNS"] == "5"
        # Episode model: the wizard's non-blank default means an
        # operator who hits Enter at the prompt now gets Sonnet
        # written to the env. This test asserts the explicit-override
        # path: a model literal typed in the prompt survives to env.
        assert env["MEMORY_EPISODE_MODEL"] == "claude-haiku-4-5-20251001"
        assert "MEMORY_EPISODE_BUDGET_USD" not in env
        assert env["MEMORY_EPISODE_TIMEOUT_S"] == "60"
        assert env["MEMORY_TOKEN_BUDGET"] == "3000"
        assert env["MEMORY_SEARCH_LIMIT"] == "20"

    def test_memory_episode_wizard_default_writes_sonnet(self, tmp_path, monkeypatch):
        """Regression for the v1 default-flip: an operator who accepts
        every wizard default in the episode block should end up with
        MEMORY_EPISODE_MODEL=claude-sonnet-4-6 written to env, not the
        empty-inheritance fallback. Pins the recommendation contract so
        a future "default back to inheritance" change surfaces here.

        Budget default is 0.15 (matches the dataclass default), so the
        emission gate suppresses the key - asserted absent. Timeout
        default is 120, also suppressed."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._block_etc_kai(monkeypatch)

        # Every input below matches the corresponding dataclass
        # default so the emission gates suppress non-episode keys.
        # Asserted on the negative side below for the episode keys;
        # the positive assertion is on the Sonnet model entry.
        # Budget prompts (extraction, episode) are skipped on the
        # claude backend per issue #390, so they are absent from
        # the input list rather than supplying the dataclass default.
        memory_block = [
            "true",  # memory enabled
            "true",  # extraction enabled
            "10",  # extraction timeout (dataclass default; suppressed)
            "8",  # consolidation candidates (dataclass default; suppressed)
            "3",  # episode classifier context turns (#392, dataclass default; suppressed)
            "",  # episode model: accept wizard default = claude-sonnet-4-6
            "120",  # episode timeout (dataclass default; suppressed)
            "2000",  # token budget (dataclass default; suppressed)
            "10",  # search limit (dataclass default; suppressed)
        ]
        inputs = iter(self._base_inputs(memory_block))
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        _cmd_config()

        env = json.loads((tmp_path / "install.conf").read_text())["env"]
        # Sonnet is the recommended default, so empty input → Sonnet.
        assert env["MEMORY_EPISODE_MODEL"] == "claude-sonnet-4-6"
        # Budget and timeout match dataclass defaults → no emission.
        assert "MEMORY_EPISODE_BUDGET_USD" not in env
        assert "MEMORY_EPISODE_TIMEOUT_S" not in env
        # Stage-1 keys also suppressed because their inputs match
        # the dataclass defaults too. Assert the suppression so a
        # future change to the gate semantics surfaces here.
        assert "MEMORY_EXTRACTION_BUDGET_USD" not in env
        assert "MEMORY_EXTRACTION_TIMEOUT_S" not in env
        assert "MEMORY_CONSOLIDATION_CANDIDATES_N" not in env
        # Episode-classifier context window (#392) at default 3 is
        # also suppressed by the emission gate.
        assert "EPISODE_CLASSIFIER_CONTEXT_TURNS" not in env

    def test_memory_round_trip_through_env_file(self, tmp_path, monkeypatch):
        """Wizard-captured memory vars survive _generate_env_file()."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._block_etc_kai(monkeypatch)

        # Inputs in wizard order: enabled, ext enabled, timeout, consolidation, classifier-window, episode model/timeout, token budget, search limit.
        # Budget prompts (extraction, episode) are skipped on the claude
        # backend per issue #390, so they are absent from this input list.
        memory_block = [
            "true",
            "true",
            "45",
            "4",
            "7",  # episode classifier context turns (#392, non-default)
            "claude-haiku-4-5-future",
            "90",
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
        # Budget keys never reach the env file on claude (prompt skipped,
        # double-gated emission suppresses) - asserted absent here as the
        # round-trip equivalent of test_memory_enabled_writes_tunables.
        assert "MEMORY_EXTRACTION_BUDGET_USD" not in rendered
        assert 'MEMORY_EXTRACTION_TIMEOUT_S="45"' in rendered
        assert 'MEMORY_CONSOLIDATION_CANDIDATES_N="4"' in rendered
        # Episode-classifier context window (#392) round-trip parity.
        assert 'EPISODE_CLASSIFIER_CONTEXT_TURNS="7"' in rendered
        # Episode tunables: explicit model entry survives, non-default
        # timeout survives. Round-trip parity with the extraction tunables
        # above. Budget is suppressed on claude for the same reason.
        assert 'MEMORY_EPISODE_MODEL="claude-haiku-4-5-future"' in rendered
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
        self._block_etc_kai(monkeypatch)

        # AGENT_BACKEND seeded explicitly so the extraction-keys cleanup (non-claude pops them) doesn't depend on the wizard's implicit default.
        existing = {
            "version": 1,
            "install_dir": "/opt/kai",
            "data_dir": "/var/lib/kai",
            "service_user": "kai",
            "platform": "darwin",
            "env": {
                "TELEGRAM_BOT_TOKEN": "existing-token",
                "WEBHOOK_SECRET": "existing-secret",
                "AGENT_BACKEND": "claude",
                "MEMORY_ENABLED": "true",
                "MEMORY_EXTRACTION_ENABLED": "true",
                "MEMORY_EXTRACTION_BUDGET_USD": "0.08",
                "MEMORY_EXTRACTION_TIMEOUT_S": "75",
                "MEMORY_TOKEN_BUDGET": "4000",
                "MEMORY_SEARCH_LIMIT": "25",
            },
        }
        conf_path.write_text(json.dumps(existing))
        # users.yaml prevents per-user prompts that lack defaults.
        (tmp_path / "users.yaml").write_text("users:\n  - telegram_id: 999\n    name: existing\n    role: admin\n")

        monkeypatch.setattr("builtins.input", lambda prompt: "")

        _cmd_config()

        conf = json.loads(conf_path.read_text())
        env = conf["env"]
        assert env["MEMORY_ENABLED"] == "true"
        assert env["MEMORY_EXTRACTION_ENABLED"] == "true"
        # Budget key dropped on claude backend per issue #390: prompt
        # is skipped, the pre-init "0.01" stays untouched (existing env
        # is no longer consulted for this key on claude), and the
        # double-gated emission suppresses. A previously-set value on
        # an upgrade path is intentionally cleared - the field is
        # informational only on claude, so a stale operator value does
        # not need to round-trip.
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
        self._block_etc_kai(monkeypatch)

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

    def test_non_claude_backend_drops_stale_extraction_keys(self, tmp_path, monkeypatch):
        """Switching from claude to goose strips MEMORY_EXTRACTION_* keys."""
        monkeypatch.chdir(tmp_path)
        conf_path = tmp_path / "install.conf"
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._block_etc_kai(monkeypatch)

        # Switching claude -> goose must drop extraction keys: bot.py:3609 silently ignores them on non-claude.
        # MEMORY_EXTRACTION_TIMEOUT_S seeded so the cleanup pop is exercised, not just defaulted away.
        existing = {
            "version": 1,
            "env": {
                "AGENT_BACKEND": "goose",
                "MEMORY_ENABLED": "true",
                "MEMORY_EXTRACTION_ENABLED": "true",
                "MEMORY_EXTRACTION_BUDGET_USD": "0.05",
                "MEMORY_EXTRACTION_TIMEOUT_S": "60",
            },
        }
        conf_path.write_text(json.dumps(existing))

        inputs = iter(
            [
                "/opt/kai",  # install dir
                "/var/lib/kai",  # data dir
                "kai",  # service user
                "darwin",  # platform
                "fake-token",  # bot token
                "12345",  # admin telegram ID
                "admin",  # admin display name
                "false",  # advanced user options
                "polling",  # transport
                "goose",  # agent backend (was claude)
                "anthropic",  # goose provider
                "sk-ant-test-key",  # API key
                "sonnet",  # model
                "120",  # timeout
                "10.0",  # budget
                "200000",  # max context window
                "8080",  # port
                "test-secret",  # webhook secret
                "~/Projects",  # workspace base
                "",  # allowed workspaces
                "false",  # pr review enabled
                "900",  # pr review timeout
                "1.0",  # pr review budget
                "false",  # issue triage
                "",  # github notify chat id
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
        the new episode-classifier context window (#392), and the entire
        episode block sit inside the extraction-enabled branch, so
        disabling extraction must consume strictly fewer prompts than
        enabling it. (Issue #390 removed the extraction budget and
        episode budget prompts; #392 adds the classifier window prompt;
        the extraction-enabled branch now drives only timeout,
        consolidation, classifier-window, episode model, and episode
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
        self._block_etc_kai(monkeypatch)

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
        self._block_etc_kai(monkeypatch)

        # All other fields at default; only the classifier window is
        # non-default so the test isolates the new emission path from
        # surrounding noise.
        memory_block = [
            "true",  # memory enabled
            "true",  # extraction enabled
            "10",  # extraction timeout (default; suppressed)
            "8",  # consolidation candidates (default; suppressed)
            "5",  # episode classifier context turns (#392, non-default)
            "",  # episode model (accept Sonnet wizard default)
            "120",  # episode timeout (default; suppressed)
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
        self._block_etc_kai(monkeypatch)

        memory_block = [
            "true",  # memory enabled
            "true",  # extraction enabled
            "10",  # extraction timeout (default)
            "8",  # consolidation candidates (default)
            "3",  # episode classifier context turns (#392, dataclass default)
            "",  # episode model (Sonnet default)
            "120",  # episode timeout (default)
            "2000",  # token budget (default)
            "10",  # search limit (default)
        ]
        inputs = iter(self._base_inputs(memory_block))
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        _cmd_config()

        env = json.loads((tmp_path / "install.conf").read_text())["env"]
        # All defaults → key is absent.
        assert "EPISODE_CLASSIFIER_CONTEXT_TURNS" not in env

    def test_episode_classifier_context_turns_skipped_on_goose_backend(self, tmp_path, monkeypatch):
        """On agent_backend="goose", the entire memory-extraction
        branch is gated out (the classifier only runs under claude
        per bot.py's effective_backend == "claude" check). The wizard
        does not prompt for the new key, the dataclass default
        applies at startup, and the env file does not carry the key."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path)
        self._block_etc_kai(monkeypatch)

        # On goose, the extraction-enabled prompt itself is skipped, so
        # memory_block is shaped like a no-extraction run. The entire
        # extraction-enabled branch (including the new classifier-window
        # prompt) does not fire.
        memory_block = ["true", "2000", "10"]  # memory_enabled, token_budget, search_limit
        inputs = iter(self._base_inputs(memory_block, agent_backend="goose"))
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        _cmd_config()

        env = json.loads((tmp_path / "install.conf").read_text())["env"]
        # Backend-cleanup pop in install.py drops the key under non-
        # claude backends; pin the absence so a future regression
        # that leaves a stale value in /etc/kai/env after a
        # claude→goose flip surfaces here.
        assert "EPISODE_CLASSIFIER_CONTEXT_TURNS" not in env


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

        conf_path = tmp_path / "install.conf"
        conf_path.write_text(
            json.dumps(
                {
                    "install_dir": str(tmp_path / "opt" / "kai"),
                    "data_dir": str(tmp_path / "var" / "lib" / "kai"),
                    "service_user": "nobody",
                    "platform": "darwin",
                    "env": {"TELEGRAM_BOT_TOKEN": "tok"},
                }
            )
        )
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)

        _cmd_apply()

        output = capsys.readouterr().out
        assert "[DRY RUN]" in output
        # Verify nothing was actually created
        assert not (tmp_path / "opt" / "kai").exists()
        # Secrets reminder should NOT appear during dry run
        assert "contains secrets" not in output

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


# ── Venv creation ────────────────────────────────────────────────────


class TestApplyVenv:
    """Tests for _apply_venv(), which creates the virtual environment."""

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
        (install / "venv").mkdir()

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

    def test_reinstalls_on_source_change(self, tmp_path, monkeypatch, capsys):
        """Triggers reinstall when source files change but pyproject.toml does not."""
        install = tmp_path / "opt" / "kai"
        install.mkdir(parents=True)
        (install / "venv" / "bin").mkdir(parents=True)

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
        assert "Installed package into venv" in output

    def test_reinstalls_on_source_change_dry_run(self, tmp_path, capsys):
        """Dry run reports source change without actually reinstalling."""
        install = tmp_path / "opt" / "kai"
        install.mkdir(parents=True)
        (install / "venv").mkdir()

        (install / "pyproject.toml").write_text("[project]\nname = 'kai'\n")
        src = install / "src" / "kai"
        src.mkdir(parents=True)
        (src / "bot.py").write_text("# bot v1")

        # Save checksums, then modify source
        (install / ".pyproject.sha256").write_text(_file_checksum(install / "pyproject.toml") + "\n")
        (install / ".src.sha256").write_text(_src_checksum(install / "src") + "\n")
        (src / "bot.py").write_text("# bot v2 - new feature")

        _apply_venv(install, is_update=True, dry_run=True)

        output = capsys.readouterr().out
        assert "DRY RUN" in output
        assert "source changed" in output

    def test_saves_both_checksums_after_install(self, tmp_path, monkeypatch):
        """Both .pyproject.sha256 and .src.sha256 are written after a successful install."""
        install = tmp_path / "opt" / "kai"
        install.mkdir(parents=True)
        (install / "venv" / "bin").mkdir(parents=True)

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
        assert (install / ".src.sha256").exists()

        # And they should contain the correct checksums
        assert (install / ".pyproject.sha256").read_text().strip() == _file_checksum(install / "pyproject.toml")
        assert (install / ".src.sha256").read_text().strip() == _src_checksum(install / "src")

    def test_first_update_without_src_checksum(self, tmp_path, monkeypatch, capsys):
        """First update after this fix triggers reinstall (no .src.sha256 from old install)."""
        install = tmp_path / "opt" / "kai"
        install.mkdir(parents=True)
        (install / "venv" / "bin").mkdir(parents=True)

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


class TestSrcChecksum:
    """Tests for _src_checksum(), the directory content hasher."""

    def test_empty_dir(self, tmp_path):
        """Empty directory (no .py files) returns a hash (of zero inputs)."""
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

    def test_ignores_non_py_files(self, tmp_path):
        """Non-.py files do not affect the hash."""
        d = tmp_path / "src"
        d.mkdir()
        (d / "a.py").write_text("hello")
        h1 = _src_checksum(d)

        # Add non-Python files - hash should not change
        (d / "readme.md").write_text("docs")
        (d / "data.json").write_text("{}")
        h2 = _src_checksum(d)

        assert h1 == h2

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


class TestApplyMigrate:
    @pytest.fixture(autouse=True)
    def _isolate_users_yaml(self, monkeypatch):
        """
        Default the per-user MEMORY.md migration to a no-op for tests
        that do not care about it. Without this fixture, _apply_migrate
        falls through to its real default of /etc/kai/users.yaml, which
        on a developer machine is a populated file that triggers
        chown calls outside the tmp_path sandbox. Tests that DO care
        about memory migration pass `users_yaml_path=` explicitly and
        are unaffected by this stub.
        """
        monkeypatch.setattr("kai.install._collect_user_memory_owners", lambda _path: [])

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

        _apply_migrate(data_path, tmp_path / "install", svc_uid=501, svc_gid=20, dry_run=False)

        # Destination should be unchanged
        assert (data_path / "kai.db").read_text() == "existing-content"
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
# These tests intentionally live OUTSIDE TestApplyMigrate so that the
# autouse `_isolate_users_yaml` fixture (which stubs
# _collect_user_memory_owners to return []) does not apply. The tests
# below need the real function to exercise the per-user migration path
# end-to-end. Each test passes `users_yaml_path=` explicitly, isolated
# under tmp_path, so nothing escapes the sandbox.


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

        Round 2 review fix: home_root must end up at exactly 0o755,
        not whatever the umask leaves behind. mkdir(mode=0o755) is
        masked by the process umask; under the production service
        umask of 0o027 this would leave home_root at 0o750, blocking
        group traversal for distinct-os_user subprocesses. We set a
        hostile umask inside the test to prove the explicit chmod
        survives it.
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
        # come out as 0o750 here.
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
        # Both at exactly 0o755 (group/other read+traverse, no write).
        assert stat.S_IMODE(home_root.stat().st_mode) == 0o755, (
            f"home_root mode {oct(stat.S_IMODE(home_root.stat().st_mode))} - "
            "umask masked the mkdir mode and explicit chmod did not run"
        )
        assert stat.S_IMODE((home_root / "8888").stat().st_mode) == 0o755

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
    def test_darwin(self, monkeypatch, tmp_path):
        """Calls launchctl bootstrap on macOS with system domain."""
        calls: list[list[str]] = []

        def mock_run(cmd, **kwargs):
            calls.append(list(cmd))
            return subprocess.CompletedProcess(args=cmd, returncode=0)

        monkeypatch.setattr("kai.install.subprocess.run", mock_run)

        _start_service("darwin", svc_uid=501, service_user="kai", dry_run=False)

        assert len(calls) == 1
        assert calls[0][0] == "launchctl"
        assert calls[0][1] == "bootstrap"
        assert calls[0][2] == "system"

    def test_linux(self, monkeypatch):
        """Calls systemctl start on Linux."""
        calls: list[list[str]] = []

        def mock_run(cmd, **kwargs):
            calls.append(list(cmd))
            return subprocess.CompletedProcess(args=cmd, returncode=0)

        monkeypatch.setattr("kai.install.subprocess.run", mock_run)

        _start_service("linux", svc_uid=1000, service_user="kai", dry_run=False)

        assert calls == [["systemctl", "start", "kai"]]

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


# ── _apply_source ────────────────────────────────────────────────────


class TestApplySource:
    """
    _apply_source copies src/, pyproject.toml, and templates/config/
    into the install tree, and retires the dead <install>/home/.claude/
    subtree (and any legacy IDENTITY.md) via _retire_install_home_claude.
    Post-#447 the install tree carries no CLAUDE.md; the per-user
    runtime <DATA_DIR>/home/<chat_id>/.claude/CLAUDE.md is seeded by
    _apply_migrate (eager, install-time) and backend.ensure_user_home
    (lazy, first-message fallback). The retirement helper has its own
    pinned contracts in TestRetireInstallHomeClaude below; the
    migration helper that ran before #447 is no longer called from
    _apply_source but is unit-tested directly in
    TestMigrateIdentityToClaudeMd.
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

        # <install>/config/ should be root-owned (static template, not runtime data)
        own_calls = [c for c in mock_own.call_args_list if c[0] == (config_dst, 0, 0) and c[1].get("recursive") is True]
        assert len(own_calls) == 1

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


class TestApplyMigrateClaudeMdSeed:
    """
    _apply_migrate's home block seeds <DATA_DIR>/home/<chat_id>/.claude/
    CLAUDE.md from templates/.claude/CLAUDE.md for every users.yaml
    entry. Without this seed, a new user's home workspace is an empty
    directory and the bot has no baseline identity to read. The pattern
    mirrors the MEMORY.md / PREFERENCES.md seed blocks above this one
    in the same function. Lazy bootstrap (backend.ensure_user_home) is
    the fallback for chat_ids added between installs and is covered by
    its own test module.
    """

    def _write_users_yaml(self, path: Path, entries: list[dict]) -> None:
        path.write_text(yaml.safe_dump({"users": entries}))

    def _stub_pwd_getpwnam(self):
        # Map any os_user to a fixed uid/gid pair. _apply_migrate
        # validates os_user existence up front via pwd.getpwnam; in
        # tests we redirect to a stub so we do not need real OS
        # accounts on the test host. The chown calls inside the
        # function are themselves stubbed out via patch("os.chown")
        # in each test, so the uid/gid values here are placeholders.
        class _Pw:
            pw_uid = 1234
            pw_gid = 1234

        return _Pw()

    def test_fresh_install_seeds_per_user_claude_md(self, tmp_path):
        """A users.yaml entry produces a populated CLAUDE.md at the per-user path."""
        src = tmp_path / "source"
        ws_claude = src / "templates" / ".claude"
        ws_claude.mkdir(parents=True)
        (ws_claude / "CLAUDE.md").write_text("# Kai template\n")
        # The seed loop is shared with MEMORY.md / PREFERENCES.md; supply
        # those templates too so the rest of _apply_migrate does not
        # diverge into placeholder branches.
        (ws_claude / "MEMORY.md").write_text("# Memory\n")
        (ws_claude / "PREFERENCES.md").write_text("# Preferences\n")
        users_yaml = tmp_path / "users.yaml"
        self._write_users_yaml(users_yaml, [{"telegram_id": 12345, "os_user": "alice"}])

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

        claude_dst = data_path / "home" / "12345" / ".claude" / "CLAUDE.md"
        assert claude_dst.is_file(), f"Expected seed at {claude_dst}"
        assert claude_dst.read_text() == "# Kai template\n"

    def test_existing_per_user_claude_md_survives_reinstall(self, tmp_path):
        """Idempotent: an operator-customized destination is never overwritten."""
        src = tmp_path / "source"
        ws_claude = src / "templates" / ".claude"
        ws_claude.mkdir(parents=True)
        (ws_claude / "CLAUDE.md").write_text("# Kai template\n")
        (ws_claude / "MEMORY.md").write_text("# Memory\n")
        (ws_claude / "PREFERENCES.md").write_text("# Preferences\n")
        users_yaml = tmp_path / "users.yaml"
        self._write_users_yaml(users_yaml, [{"telegram_id": 12345, "os_user": "alice"}])

        data_path = tmp_path / "data"
        claude_dir = data_path / "home" / "12345" / ".claude"
        claude_dir.mkdir(parents=True)
        claude_dst = claude_dir / "CLAUDE.md"
        claude_dst.write_text("# OPERATOR CUSTOMIZED\n")

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

        # Customized content survives.
        assert claude_dst.read_text() == "# OPERATOR CUSTOMIZED\n"

    def test_missing_template_writes_placeholder(self, tmp_path):
        """Missing template: last-resort placeholder so the file is writable."""
        src = tmp_path / "source"
        ws_claude = src / "templates" / ".claude"
        ws_claude.mkdir(parents=True)
        # CLAUDE.md template intentionally absent; MEMORY.md / PREFERENCES.md
        # present so the rest of _apply_migrate runs normally.
        (ws_claude / "MEMORY.md").write_text("# Memory\n")
        (ws_claude / "PREFERENCES.md").write_text("# Preferences\n")
        users_yaml = tmp_path / "users.yaml"
        self._write_users_yaml(users_yaml, [{"telegram_id": 12345, "os_user": "alice"}])

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

        claude_dst = data_path / "home" / "12345" / ".claude" / "CLAUDE.md"
        assert claude_dst.is_file()
        assert claude_dst.read_text() == "# Identity\n"

    def test_dry_run_predicts_seed(self, tmp_path, capsys):
        """Dry-run names the source and destination for the seed it would write."""
        src = tmp_path / "source"
        ws_claude = src / "templates" / ".claude"
        ws_claude.mkdir(parents=True)
        (ws_claude / "CLAUDE.md").write_text("# Kai template\n")
        (ws_claude / "MEMORY.md").write_text("# Memory\n")
        (ws_claude / "PREFERENCES.md").write_text("# Preferences\n")
        users_yaml = tmp_path / "users.yaml"
        self._write_users_yaml(users_yaml, [{"telegram_id": 12345, "os_user": "alice"}])

        data_path = tmp_path / "data"
        install_path = tmp_path / "install"
        install_path.mkdir()

        with (
            patch("kai.install.PROJECT_ROOT", src),
            patch("kai.install.pwd.getpwnam", return_value=self._stub_pwd_getpwnam()),
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
        expected_dst = data_path / "home" / "12345" / ".claude" / "CLAUDE.md"
        # Dry-run preview names the per-user destination path. The
        # template source path is in the same line; pin one substring
        # so a path-format change does not flake the test.
        assert f"Would seed {expected_dst}" in output
        # Dry run never touches disk.
        assert not expected_dst.exists()


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


# ── _apply_sudoers dry run ───────────────────────────────────────────


class TestApplySudoersDryRun:
    def test_dry_run(self, capsys):
        """Dry run: prints expected messages."""
        _apply_sudoers("kai", dry_run=True)
        output = capsys.readouterr().out
        assert "DRY RUN" in output
        assert "sudoers" in output.lower() or "visudo" in output.lower()

    def test_warns_when_claude_bin_missing(self, tmp_path, capsys, monkeypatch):
        """Warning printed when the rule's claude binary path doesn't exist."""
        svc_home = tmp_path / "home" / "kai"
        svc_home.mkdir(parents=True)
        # Intentionally do NOT create svc_home/.local/bin/claude.
        monkeypatch.setattr("kai.install._user_home", lambda u: str(svc_home))
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users:\n  - telegram_id: 1\n    os_user: alice\n")

        _apply_sudoers("kai", dry_run=True, users_yaml_path=users_yaml)

        captured = capsys.readouterr()
        assert "Warning" in captured.err
        assert str(svc_home / ".local" / "bin" / "claude") in captured.err

    def test_no_warning_when_no_target_users(self, tmp_path, capsys, monkeypatch):
        """No warning when there are no per-user rules to emit."""
        svc_home = tmp_path / "home" / "kai"
        svc_home.mkdir(parents=True)
        monkeypatch.setattr("kai.install._user_home", lambda u: str(svc_home))
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users: []\n")

        _apply_sudoers("kai", dry_run=True, users_yaml_path=users_yaml)

        captured = capsys.readouterr()
        assert "Warning" not in captured.err

    def test_no_warning_when_claude_bin_exists(self, tmp_path, capsys, monkeypatch):
        """Path exists -> silent."""
        svc_home = tmp_path / "home" / "kai"
        bin_dir = svc_home / ".local" / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "claude").write_text("#!/bin/sh\n")
        monkeypatch.setattr("kai.install._user_home", lambda u: str(svc_home))
        users_yaml = tmp_path / "users.yaml"
        users_yaml.write_text("users:\n  - telegram_id: 1\n    os_user: alice\n")

        _apply_sudoers("kai", dry_run=True, users_yaml_path=users_yaml)

        captured = capsys.readouterr()
        assert "Warning" not in captured.err


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
    """Tests for _apply_goose_config() goose binary check."""

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
        _apply_goose_config("kai", install_path, uid, gid, dry_run=False)

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
        _apply_goose_config("kai", install_path, uid, gid, dry_run=True)

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
        _apply_goose_config("kai", install_path, uid, gid, dry_run=False)

        output = capsys.readouterr().out
        assert "WARNING" not in output


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
