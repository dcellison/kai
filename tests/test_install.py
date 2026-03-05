"""Tests for the protected installation module (install.py)."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from kai.install import (
    _LAUNCHD_LABEL,
    _apply_directories,
    _apply_migrate,
    _apply_venv,
    _check_path,
    _check_service_status,
    _check_traversal,
    _cmd_apply,
    _cmd_config,
    _cmd_status,
    _file_checksum,
    _generate_env_file,
    _generate_launchd_plist,
    _generate_sudoers,
    _generate_systemd_unit,
    _start_service,
    _stop_service,
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
        assert "kai ALL=(alice) NOPASSWD: /usr/local/bin/claude" in result

    def test_claude_user_rule_fallback(self, monkeypatch):
        """Falls back to service user home when claude is not on PATH."""
        monkeypatch.setattr("kai.install._user_home", lambda u: f"/home/{u}")
        real_which = shutil.which
        monkeypatch.setattr(shutil, "which", lambda n: None if n == "claude" else real_which(n))
        result = _generate_sudoers("kai", claude_user="alice")
        assert "kai ALL=(alice) NOPASSWD: /home/kai/.local/bin/claude" in result


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

        _apply_migrate(data_path, svc_uid=501, svc_gid=20, dry_run=False)

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

        _apply_migrate(data_path, svc_uid=501, svc_gid=20, dry_run=False)

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

        _apply_migrate(data_path, svc_uid=501, svc_gid=20, dry_run=False)

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

        _apply_migrate(data_path, svc_uid=501, svc_gid=20, dry_run=False)

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

        _apply_migrate(data_path, svc_uid=501, svc_gid=20, dry_run=False)

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

        _apply_migrate(data_path, svc_uid=501, svc_gid=20, dry_run=True)

        output = capsys.readouterr().out
        assert "[DRY RUN]" in output
        # Nothing should have been copied
        assert not (data_path / "kai.db").exists()
        assert not (data_path / "logs" / "kai.log").exists()


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
