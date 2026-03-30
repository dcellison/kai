"""Tests for the protected installation module (install.py)."""

import json
import os
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from kai.install import (
    _LAUNCHD_LABEL,
    _apply_directories,
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
    _copy_tree,
    _file_checksum,
    _generate_env_file,
    _generate_launchd_plist,
    _generate_launcher_script,
    _generate_sudoers,
    _generate_systemd_unit,
    _generate_users_yaml,
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
        """Generated YAML can be parsed by _load_user_configs()."""
        from kai.config import _load_user_configs

        content = _generate_users_yaml("123456789", "alice", os_user="kai")
        yaml_path = tmp_path / "users.yaml"
        yaml_path.write_text(content)
        monkeypatch.setattr("kai.config.PROJECT_ROOT", tmp_path)
        # Skip the protected /etc/kai/ path so we read the tmp_path copy.
        monkeypatch.setattr("kai.config._read_protected_yaml", lambda _: None)
        configs = _load_user_configs()
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

    def test_claude_user_rule_with_which(self, monkeypatch):
        """Uses shutil.which to resolve the claude binary location."""
        real_which = shutil.which
        monkeypatch.setattr(shutil, "which", lambda n: "/usr/local/bin/claude" if n == "claude" else real_which(n))
        result = _generate_sudoers("kai", claude_user="alice")
        assert "kai ALL=(alice) SETENV: NOPASSWD: /usr/local/bin/claude" in result

    def test_claude_user_rule_fallback(self, monkeypatch):
        """Falls back to service user home when claude is not on PATH."""
        monkeypatch.setattr("kai.install._user_home", lambda u: f"/home/{u}")
        real_which = shutil.which
        monkeypatch.setattr(shutil, "which", lambda n: None if n == "claude" else real_which(n))
        result = _generate_sudoers("kai", claude_user="alice")
        assert "kai ALL=(alice) SETENV: NOPASSWD: /home/kai/.local/bin/claude" in result


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
                "sonnet",  # model
                "120",  # timeout
                "10.0",  # budget
                "8080",  # port
                "test-secret",  # webhook secret
                "~/Projects",  # workspace base
                "",  # allowed workspaces (empty)
                "false",  # pr review enabled
                "false",  # issue triage enabled
                "",  # github notify chat id (empty)
                "false",  # voice
                "false",  # tts
                "",  # claude user (empty)
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
        # ALLOWED_USER_IDS should not be in the env dict
        assert "ALLOWED_USER_IDS" not in conf["env"]
        # users.yaml should have been generated
        yaml_path = tmp_path / "users.yaml"
        assert yaml_path.exists()
        data = yaml.safe_load(yaml_path.read_text())
        assert data["users"][0]["telegram_id"] == 12345
        assert data["users"][0]["role"] == "admin"

    def test_advanced_user_options(self, tmp_path, monkeypatch):
        """Advanced path writes os_user and home_workspace, skips CLAUDE_USER."""
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
                str(tmp_path),  # home_workspace
                "polling",  # transport
                "sonnet",  # model
                "120",  # timeout
                "10.0",  # budget
                "8080",  # port
                "test-secret",  # webhook secret
                "~/Projects",  # workspace base
                "",  # allowed workspaces (empty)
                "false",  # pr review enabled
                "false",  # issue triage enabled
                "",  # github notify chat id (empty)
                "false",  # voice
                "false",  # tts
                # no claude user prompt (skipped by advanced mode)
                "",  # perplexity key (empty)
            ]
        )
        monkeypatch.setattr("builtins.input", lambda prompt: next(inputs))

        _cmd_config()

        # Verify users.yaml has os_user and home_workspace
        yaml_path = tmp_path / "users.yaml"
        assert yaml_path.exists()
        data = yaml.safe_load(yaml_path.read_text())
        entry = data["users"][0]
        assert entry["os_user"] == "testuser"
        assert entry["home_workspace"] == str(tmp_path.resolve())

        # CLAUDE_USER should not be in the env (skipped because os_user was set)
        conf = json.loads((tmp_path / "install.conf").read_text())
        assert "CLAUDE_USER" not in conf["env"]

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

    def test_copies_history(self, tmp_path, monkeypatch):
        """Copies JSONL history files from home/.claude/history/ to data_path/history/."""
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path / "src")
        history_src = tmp_path / "src" / "home" / ".claude" / "history"
        history_src.mkdir(parents=True)
        (history_src / "2026-03-20.jsonl").write_text('{"ts":"2026-03-20","text":"hello"}')
        (history_src / "2026-03-21.jsonl").write_text('{"ts":"2026-03-21","text":"world"}')

        data_path = tmp_path / "data"
        data_path.mkdir()
        (data_path / "logs").mkdir()
        history_dst = data_path / "history"
        history_dst.mkdir()

        monkeypatch.setattr("kai.install.os.chown", lambda *a: None)

        _apply_migrate(data_path, tmp_path / "install", svc_uid=501, svc_gid=20, dry_run=False)

        assert (history_dst / "2026-03-20.jsonl").exists()
        assert (history_dst / "2026-03-21.jsonl").exists()
        # Source files preserved
        assert (history_src / "2026-03-20.jsonl").exists()

    def test_history_skips_existing(self, tmp_path, monkeypatch, capsys):
        """Does not overwrite history files that already exist at the destination."""
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path / "src")
        history_src = tmp_path / "src" / "home" / ".claude" / "history"
        history_src.mkdir(parents=True)
        (history_src / "2026-03-20.jsonl").write_text("source content")

        data_path = tmp_path / "data"
        data_path.mkdir()
        (data_path / "logs").mkdir()
        history_dst = data_path / "history"
        history_dst.mkdir()
        (history_dst / "2026-03-20.jsonl").write_text("existing content")

        _apply_migrate(data_path, tmp_path / "install", svc_uid=501, svc_gid=20, dry_run=False)

        # Destination unchanged
        assert (history_dst / "2026-03-20.jsonl").read_text() == "existing content"
        output = capsys.readouterr().out
        assert "already migrated" in output

    def test_copies_memory(self, tmp_path, monkeypatch, capsys):
        """Copies MEMORY.md from home/.claude/ to data_path/memory/."""
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path / "src")
        claude_dir = tmp_path / "src" / "home" / ".claude"
        claude_dir.mkdir(parents=True)
        (claude_dir / "MEMORY.md").write_text("User prefers dry humor.")

        data_path = tmp_path / "data"
        data_path.mkdir()
        (data_path / "logs").mkdir()
        (data_path / "memory").mkdir()

        monkeypatch.setattr("kai.install.os.chown", lambda *a: None)

        _apply_migrate(data_path, tmp_path / "install", svc_uid=501, svc_gid=20, dry_run=False)

        memory_dst = data_path / "memory" / "MEMORY.md"
        assert memory_dst.exists()
        assert memory_dst.read_text() == "User prefers dry humor."
        # Source preserved
        assert (claude_dir / "MEMORY.md").exists()
        assert "Migrated MEMORY.md" in capsys.readouterr().out

    def test_memory_skips_existing(self, tmp_path, monkeypatch):
        """Does not overwrite MEMORY.md that already exists at the destination."""
        monkeypatch.setattr("kai.install.PROJECT_ROOT", tmp_path / "src")
        claude_dir = tmp_path / "src" / "home" / ".claude"
        claude_dir.mkdir(parents=True)
        (claude_dir / "MEMORY.md").write_text("old content")

        data_path = tmp_path / "data"
        data_path.mkdir()
        (data_path / "logs").mkdir()
        memory_dir = data_path / "memory"
        memory_dir.mkdir()
        (memory_dir / "MEMORY.md").write_text("existing personalized content")

        _apply_migrate(data_path, tmp_path / "install", svc_uid=501, svc_gid=20, dry_run=False)

        assert (memory_dir / "MEMORY.md").read_text() == "existing personalized content"

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
    def test_dry_run(self, tmp_path, capsys):
        """Dry run: prints messages, doesn't copy."""
        # Create home/.claude/ so the dry-run message appears
        ws_claude = tmp_path / "home" / ".claude"
        ws_claude.mkdir(parents=True)
        (ws_claude / "CLAUDE.md").write_text("identity")
        with patch("kai.install.PROJECT_ROOT", tmp_path):
            _apply_source(tmp_path / "install", svc_uid=1000, svc_gid=1000, dry_run=True)
        output = capsys.readouterr().out
        assert "DRY RUN" in output
        assert "Would copy" in output
        # Dry run should NOT create home/.claude/ at the destination
        assert not (tmp_path / "install" / "home" / ".claude").exists()

    def test_dry_run_no_home_claude(self, tmp_path, capsys):
        """Dry run without home/.claude/: no home config copy message."""
        with patch("kai.install.PROJECT_ROOT", tmp_path):
            _apply_source(tmp_path / "install", svc_uid=1000, svc_gid=1000, dry_run=True)
        output = capsys.readouterr().out
        assert "DRY RUN" in output
        assert "home config" not in output

    def test_actual(self, tmp_path):
        """Actual: copies source, pyproject.toml, and home/.claude/."""
        # Set up source structure
        src = tmp_path / "source"
        (src / "src").mkdir(parents=True)
        (src / "src" / "module.py").write_text("code")
        (src / "pyproject.toml").write_text("[project]")
        ws_claude = src / "home" / ".claude"
        ws_claude.mkdir(parents=True)
        (ws_claude / "CLAUDE.md").write_text("identity")
        install = tmp_path / "install"
        install.mkdir()

        with (
            patch("kai.install.PROJECT_ROOT", src),
            patch("kai.install._copy_tree") as mock_copy,
            patch("kai.install._set_ownership") as mock_own,
            patch("shutil.copy2") as mock_cp,
            patch("os.chown") as mock_chown,
        ):
            _apply_source(install, svc_uid=1000, svc_gid=1000, dry_run=False)
        # Should call _copy_tree twice: once for src/, once for home/.claude/
        assert mock_copy.call_count == 2
        # Should call _set_ownership twice: once for src/, once for home/.claude/
        assert mock_own.call_count == 2
        mock_cp.assert_called_once()
        # os.chown called twice: once for pyproject.toml (root), once for
        # the .claude/ directory itself (service user)
        ws_claude_dst = install / "home" / ".claude"
        chown_calls = mock_chown.call_args_list
        assert any(c.args == (ws_claude_dst, 1000, 1000) for c in chown_calls), (
            f"Expected os.chown({ws_claude_dst}, 1000, 1000) in {chown_calls}"
        )

    def test_actual_no_home_claude(self, tmp_path):
        """Actual without home/.claude/: copies source only, no error."""
        src = tmp_path / "source"
        (src / "src").mkdir(parents=True)
        (src / "src" / "module.py").write_text("code")
        (src / "pyproject.toml").write_text("[project]")
        install = tmp_path / "install"
        install.mkdir()

        with (
            patch("kai.install.PROJECT_ROOT", src),
            patch("kai.install._copy_tree") as mock_copy,
            patch("kai.install._set_ownership") as mock_own,
            patch("shutil.copy2") as mock_cp,
            patch("os.chown"),
        ):
            _apply_source(install, svc_uid=1000, svc_gid=1000, dry_run=False)
        # Only one _copy_tree call (src/), no home/.claude/ copy
        mock_copy.assert_called_once()
        mock_own.assert_called_once()
        mock_cp.assert_called_once()

    def test_home_claude_excludes(self, tmp_path):
        """Home config copy excludes history/, MEMORY.md, skills/, __pycache__."""
        from kai.install import _HOME_CLAUDE_EXCLUDES

        src = tmp_path / "source"
        (src / "src").mkdir(parents=True)
        (src / "src" / "module.py").write_text("code")
        (src / "pyproject.toml").write_text("[project]")
        ws_claude = src / "home" / ".claude"
        ws_claude.mkdir(parents=True)
        (ws_claude / "CLAUDE.md").write_text("identity")
        install = tmp_path / "install"
        install.mkdir()

        with (
            patch("kai.install.PROJECT_ROOT", src),
            patch("kai.install._copy_tree") as mock_copy,
            patch("kai.install._set_ownership"),
            patch("shutil.copy2"),
            patch("os.chown"),
        ):
            _apply_source(install, svc_uid=1000, svc_gid=1000, dry_run=False)

        # Second _copy_tree call is for home/.claude/
        ws_call = mock_copy.call_args_list[1]
        assert ws_call[0][2] == _HOME_CLAUDE_EXCLUDES


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
