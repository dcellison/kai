"""Tests for the protected installation module (install.py)."""

import json
import shutil
import subprocess

import pytest

from kai.install import (
    _LAUNCHD_LABEL,
    _check_path,
    _check_service_status,
    _cmd_apply,
    _cmd_config,
    _cmd_status,
    _file_checksum,
    _generate_env_file,
    _generate_launchd_plist,
    _generate_sudoers,
    _generate_systemd_unit,
    _validate_port,
    _validate_positive_float,
    _validate_positive_int,
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
        assert "PORT=8080" in result
        assert "TOKEN=abc123" in result

    def test_sorted_keys(self):
        env = {"Z_KEY": "z", "A_KEY": "a"}
        result = _generate_env_file(env)
        lines = [line for line in result.splitlines() if "=" in line and not line.startswith("#")]
        assert lines[0].startswith("A_KEY=")
        assert lines[1].startswith("Z_KEY=")

    def test_includes_header_comment(self):
        result = _generate_env_file({"K": "V"})
        assert result.startswith("#")


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
        assert f"{cat_path} /etc/kai/totp.secret" in result

    def test_contains_tee_rule(self):
        """Sudoers uses the resolved tee path (may be /usr/bin/tee)."""
        result = _generate_sudoers("kai")
        tee_path = shutil.which("tee") or "/usr/bin/tee"
        assert f"{tee_path} /etc/kai/totp.attempts" in result

    def test_nopasswd(self):
        result = _generate_sudoers("kai")
        assert "NOPASSWD" in result


class TestGenerateLaunchdPlist:
    def test_contains_label(self):
        result = _generate_launchd_plist("/opt/kai", "/var/lib/kai", "kai")
        assert _LAUNCHD_LABEL in result

    def test_contains_install_dir(self):
        result = _generate_launchd_plist("/opt/kai", "/var/lib/kai", "kai")
        assert "/opt/kai/venv/bin/python" in result
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
    def test_writes_install_conf(self, tmp_path, monkeypatch):
        """Config subcommand writes valid JSON to install.conf."""
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("kai.install.INSTALL_CONF", tmp_path / "install.conf")

        # Simulate user inputs for each prompt (in order)
        inputs = iter(
            [
                "/opt/kai",  # install dir
                "/var/lib/kai",  # data dir
                "kai",  # service user
                "darwin",  # platform
                "fake-token",  # bot token
                "12345",  # user IDs
                "polling",  # transport
                "sonnet",  # model
                "120",  # timeout
                "10.0",  # budget
                "8080",  # port
                "test-secret",  # webhook secret
                "~/Projects",  # workspace base
                "",  # allowed workspaces (empty)
                "false",  # voice
                "false",  # tts
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
        assert conf["env"]["ALLOWED_USER_IDS"] == "12345"

    def test_reads_existing_defaults(self, tmp_path, monkeypatch):
        """Config subcommand uses existing install.conf values as defaults."""
        monkeypatch.chdir(tmp_path)
        conf_path = tmp_path / "install.conf"
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)

        # Write existing config
        existing = {
            "version": 1,
            "install_dir": "/custom/path",
            "data_dir": "/custom/data",
            "service_user": "myuser",
            "platform": "linux",
            "env": {
                "TELEGRAM_BOT_TOKEN": "existing-token",
                "ALLOWED_USER_IDS": "999",
                "WEBHOOK_SECRET": "existing-secret",
            },
        }
        conf_path.write_text(json.dumps(existing))

        # Press Enter for everything (accept all defaults)
        monkeypatch.setattr("builtins.input", lambda prompt: "")

        _cmd_config()

        conf = json.loads(conf_path.read_text())
        # Should preserve existing values when user accepts defaults
        assert conf["install_dir"] == "/custom/path"
        assert conf["env"]["TELEGRAM_BOT_TOKEN"] == "existing-token"

    def test_validates_required_fields(self):
        """Required-field validation rejects empty input."""
        # _prompt with required=True rejects empty input. We test the
        # underlying validator directly since testing the full interactive
        # flow with required fields is fragile with mocked input.
        assert _validate_user_ids("") is False
        assert _validate_user_ids("abc") is False
        assert _validate_port("0") is False
        assert _validate_port("99999") is False


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
                    "env": {"TELEGRAM_BOT_TOKEN": "tok", "ALLOWED_USER_IDS": "1"},
                }
            )
        )
        monkeypatch.setattr("kai.install.INSTALL_CONF", conf_path)

        _cmd_apply()

        output = capsys.readouterr().out
        assert "[DRY RUN]" in output
        # Verify nothing was actually created
        assert not (tmp_path / "opt" / "kai").exists()

    def test_generates_env_file_content(self):
        """The generated env file contains all provided values."""
        env = {
            "TELEGRAM_BOT_TOKEN": "test-token",
            "ALLOWED_USER_IDS": "123",
            "WEBHOOK_PORT": "8080",
        }
        content = _generate_env_file(env)
        assert "TELEGRAM_BOT_TOKEN=test-token" in content
        assert "ALLOWED_USER_IDS=123" in content
        assert "WEBHOOK_PORT=8080" in content

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


# ── Status subcommand ────────────────────────────────────────────────


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
