"""
Tests for per-user configuration (users.yaml).

Covers:
1. UserConfig dataclass construction
2. _load_user_configs() YAML parsing, validation, and edge cases
3. Config.get_user_config() lookup
4. Config.get_user_by_github() lookup (case-insensitive)
5. Config.get_admins() filtering
6. Legacy fallback via ALLOWED_USER_IDS
"""

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from kai.config import (
    Config,
    UserConfig,
    _load_user_configs,
)

# ── UserConfig dataclass ────────────────────────────────────────────


class TestUserConfig:
    def test_required_fields(self):
        """Minimal config: telegram_id and name only."""
        uc = UserConfig(telegram_id=123, name="alice")
        assert uc.telegram_id == 123
        assert uc.name == "alice"
        assert uc.role == "user"
        assert uc.github is None
        assert uc.os_user is None
        assert uc.home_workspace is None
        assert uc.max_budget is None
        assert uc.model is None
        assert uc.timeout is None
        assert uc.context_window is None
        assert uc.workspace_base is None

    def test_all_fields(self):
        """Full config with every field populated."""
        uc = UserConfig(
            telegram_id=123,
            name="alice",
            role="admin",
            github="alice-dev",
            os_user="alice",
            home_workspace=Path("/home/alice/workspace"),
            max_budget=15.0,
            model="opus",
            timeout=300,
            context_window=200_000,
            workspace_base=Path("/home/alice/projects"),
        )
        assert uc.role == "admin"
        assert uc.github == "alice-dev"
        assert uc.os_user == "alice"
        assert uc.max_budget == 15.0
        assert uc.model == "opus"
        assert uc.timeout == 300
        assert uc.context_window == 200_000
        assert uc.workspace_base == Path("/home/alice/projects")

    def test_frozen(self):
        """UserConfig is immutable."""
        uc = UserConfig(telegram_id=123, name="alice")
        with pytest.raises(AttributeError):
            uc.name = "bob"  # type: ignore[misc]


# ── _load_user_configs ──────────────────────────────────────────────


class TestLoadUserConfigs:
    def _write_yaml(self, tmp_path, content):
        """Write a users.yaml file."""
        yaml_file = tmp_path / "users.yaml"
        yaml_file.write_text(textwrap.dedent(content))
        return yaml_file

    def test_basic_loading(self, tmp_path):
        """Loads two users with correct fields."""
        ws = tmp_path / "ws"
        ws.mkdir()
        self._write_yaml(
            tmp_path,
            f"""\
            users:
              - telegram_id: 111
                name: alice
                role: admin
                github: alice-dev
                os_user: alice
                home_workspace: {ws}
                max_budget: 15.0
              - telegram_id: 222
                name: bob
                role: user
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_user_configs()

        assert configs is not None
        assert len(configs) == 2
        assert configs[111].name == "alice"
        assert configs[111].role == "admin"
        assert configs[111].github == "alice-dev"
        assert configs[111].os_user == "alice"
        assert configs[111].home_workspace == ws.resolve()
        assert configs[111].max_budget == 15.0
        assert configs[222].name == "bob"
        assert configs[222].role == "user"

    def test_missing_file(self, tmp_path):
        """Returns None when no YAML file exists."""
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_user_configs()
        assert configs is None

    def test_empty_file(self, tmp_path):
        """Returns None for an empty YAML file."""
        self._write_yaml(tmp_path, "")
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_user_configs()
        assert configs is None

    def test_invalid_yaml(self, tmp_path):
        """Returns None for malformed YAML."""
        (tmp_path / "users.yaml").write_text("{{bad yaml[")
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_user_configs()
        assert configs is None

    def test_missing_telegram_id(self, tmp_path):
        """Entry without telegram_id is skipped."""
        self._write_yaml(
            tmp_path,
            """\
            users:
              - name: alice
                role: admin
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_user_configs()
        assert configs is not None
        assert len(configs) == 0

    def test_whitespace_only_name(self, tmp_path):
        """Whitespace-only name is treated as missing."""
        self._write_yaml(
            tmp_path,
            """\
            users:
              - telegram_id: 111
                name: "   "
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_user_configs()
        assert configs is not None
        assert len(configs) == 0

    def test_missing_name(self, tmp_path):
        """Entry without name is skipped."""
        self._write_yaml(
            tmp_path,
            """\
            users:
              - telegram_id: 111
                role: admin
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_user_configs()
        assert configs is not None
        assert len(configs) == 0

    def test_invalid_role(self, tmp_path):
        """Invalid role causes the entry to be skipped."""
        self._write_yaml(
            tmp_path,
            """\
            users:
              - telegram_id: 111
                name: alice
                role: superuser
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_user_configs()
        assert configs is not None
        assert len(configs) == 0

    def test_invalid_budget(self, tmp_path):
        """Negative budget causes the entry to be skipped."""
        self._write_yaml(
            tmp_path,
            """\
            users:
              - telegram_id: 111
                name: alice
                max_budget: -5.0
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_user_configs()
        assert configs is not None
        assert len(configs) == 0

    def test_bool_budget_rejected(self, tmp_path):
        """Boolean budget is rejected (same bool guard as workspace config)."""
        self._write_yaml(
            tmp_path,
            """\
            users:
              - telegram_id: 111
                name: alice
                max_budget: true
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_user_configs()
        assert configs is not None
        assert len(configs) == 0

    def test_bool_telegram_id_rejected(self, tmp_path):
        """Boolean telegram_id is rejected."""
        self._write_yaml(
            tmp_path,
            """\
            users:
              - telegram_id: true
                name: alice
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_user_configs()
        assert configs is not None
        assert len(configs) == 0

    def test_duplicate_ids(self, tmp_path):
        """Duplicate telegram_id: first wins."""
        self._write_yaml(
            tmp_path,
            """\
            users:
              - telegram_id: 111
                name: alice
              - telegram_id: 111
                name: bob
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_user_configs()
        assert configs is not None
        assert len(configs) == 1
        assert configs[111].name == "alice"

    def test_home_workspace_empty_string(self, tmp_path):
        """Empty home_workspace string is treated as None, not CWD."""
        self._write_yaml(
            tmp_path,
            """\
            users:
              - telegram_id: 111
                name: alice
                home_workspace: ""
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_user_configs()
        assert configs is not None
        assert len(configs) == 1
        assert configs[111].home_workspace is None

    def test_home_workspace_nonexistent_warns_but_keeps_user(self, tmp_path):
        """Non-existent home_workspace warns and falls back to None, not skip."""
        self._write_yaml(
            tmp_path,
            """\
            users:
              - telegram_id: 111
                name: alice
                home_workspace: /nonexistent/path/12345
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_user_configs()
        assert configs is not None
        assert len(configs) == 1
        # home_workspace falls back to None (global default)
        assert configs[111].home_workspace is None

    def test_protected_installation_tried_first(self, tmp_path):
        """Protected file (/etc/kai/users.yaml) is tried before local."""
        protected_data = {"users": [{"telegram_id": 111, "name": "alice", "role": "admin"}]}
        with patch("kai.config._read_protected_yaml", return_value=protected_data):
            configs = _load_user_configs()
        assert configs is not None
        assert configs[111].name == "alice"

    def test_no_admin_warning(self, tmp_path, caplog):
        """All users with role 'user' logs a warning but does not fail."""
        self._write_yaml(
            tmp_path,
            """\
            users:
              - telegram_id: 111
                name: alice
                role: user
              - telegram_id: 222
                name: bob
                role: user
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_user_configs()
        assert configs is not None
        assert len(configs) == 2
        assert "no admin users defined" in caplog.text.lower()

    def test_default_role_is_user(self, tmp_path):
        """Omitting role defaults to 'user'."""
        self._write_yaml(
            tmp_path,
            """\
            users:
              - telegram_id: 111
                name: alice
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_user_configs()
        assert configs is not None
        assert configs[111].role == "user"

    def test_os_user_stored(self, tmp_path):
        """os_user is stored as a string (not validated in Phase 1)."""
        self._write_yaml(
            tmp_path,
            """\
            users:
              - telegram_id: 111
                name: alice
                os_user: alice_os
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_user_configs()
        assert configs is not None
        assert configs[111].os_user == "alice_os"

    # ── New per-user setting fields (model, timeout, context_window) ──

    def test_model_parsed(self, tmp_path):
        """Valid model name is stored, lowercased."""
        self._write_yaml(
            tmp_path,
            """\
            users:
              - telegram_id: 111
                name: alice
                model: Opus
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_user_configs()
        assert configs is not None
        assert configs[111].model == "opus"

    def test_invalid_model_ignored(self, tmp_path, caplog):
        """Invalid model name is ignored (set to None), user still loads."""
        self._write_yaml(
            tmp_path,
            """\
            users:
              - telegram_id: 111
                name: alice
                model: gpt4
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_user_configs()
        assert configs is not None
        assert configs[111].model is None
        assert "invalid model" in caplog.text.lower()

    def test_timeout_parsed(self, tmp_path):
        """Valid timeout is stored as int."""
        self._write_yaml(
            tmp_path,
            """\
            users:
              - telegram_id: 111
                name: alice
                timeout: 300
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_user_configs()
        assert configs is not None
        assert configs[111].timeout == 300

    def test_invalid_timeout_ignored(self, tmp_path, caplog):
        """Negative timeout is ignored, user still loads."""
        self._write_yaml(
            tmp_path,
            """\
            users:
              - telegram_id: 111
                name: alice
                timeout: -5
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_user_configs()
        assert configs is not None
        assert configs[111].timeout is None
        assert "invalid timeout" in caplog.text.lower()

    def test_context_window_parsed(self, tmp_path):
        """Valid context_window is stored as int."""
        self._write_yaml(
            tmp_path,
            """\
            users:
              - telegram_id: 111
                name: alice
                context_window: 200000
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_user_configs()
        assert configs is not None
        assert configs[111].context_window == 200_000

    def test_context_window_zero_allowed(self, tmp_path):
        """context_window of 0 means 'use default' and is valid."""
        self._write_yaml(
            tmp_path,
            """\
            users:
              - telegram_id: 111
                name: alice
                context_window: 0
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_user_configs()
        assert configs is not None
        assert configs[111].context_window == 0

    def test_context_window_below_minimum_ignored(self, tmp_path, caplog):
        """context_window below 50000 (and not 0) is ignored."""
        self._write_yaml(
            tmp_path,
            """\
            users:
              - telegram_id: 111
                name: alice
                context_window: 10000
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_user_configs()
        assert configs is not None
        assert configs[111].context_window is None
        assert "invalid context_window" in caplog.text.lower()

    def test_new_fields_default_none(self, tmp_path):
        """New optional fields default to None when omitted."""
        self._write_yaml(
            tmp_path,
            """\
            users:
              - telegram_id: 111
                name: alice
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_user_configs()
        assert configs is not None
        assert configs[111].model is None
        assert configs[111].timeout is None
        assert configs[111].context_window is None
        assert configs[111].workspace_base is None

    # ── workspace_base field ──

    def test_workspace_base_parsed(self, tmp_path):
        """Valid workspace_base directory is stored as resolved Path."""
        ws_base = tmp_path / "projects"
        ws_base.mkdir()
        self._write_yaml(
            tmp_path,
            f"""\
            users:
              - telegram_id: 111
                name: alice
                workspace_base: {ws_base}
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_user_configs()
        assert configs is not None
        assert configs[111].workspace_base == ws_base.resolve()

    def test_workspace_base_missing_dir_warns(self, tmp_path, caplog):
        """Non-existent workspace_base warns and falls back to None."""
        self._write_yaml(
            tmp_path,
            """\
            users:
              - telegram_id: 111
                name: alice
                workspace_base: /nonexistent/path/12345
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_user_configs()
        assert configs is not None
        assert len(configs) == 1
        assert configs[111].workspace_base is None
        assert "workspace_base not found" in caplog.text.lower()

    def test_workspace_base_empty_string(self, tmp_path):
        """Empty workspace_base string is treated as None."""
        self._write_yaml(
            tmp_path,
            """\
            users:
              - telegram_id: 111
                name: alice
                workspace_base: ""
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_user_configs()
        assert configs is not None
        assert configs[111].workspace_base is None


# ── Config convenience methods ──────────────────────────────────────


class TestConfigUserMethods:
    def _make_config(self, user_configs=None):
        return Config(
            telegram_bot_token="test",
            allowed_user_ids={1},
            user_configs=user_configs,
        )

    def test_get_user_config_found(self):
        """Returns UserConfig when telegram_id matches."""
        uc = UserConfig(telegram_id=111, name="alice")
        config = self._make_config({111: uc})
        assert config.get_user_config(111) is uc

    def test_get_user_config_not_found(self):
        """Returns None for unknown telegram_id."""
        config = self._make_config({})
        assert config.get_user_config(999) is None

    def test_get_user_config_no_yaml(self):
        """Returns None when user_configs is None (no users.yaml)."""
        config = self._make_config(None)
        assert config.get_user_config(111) is None

    def test_get_user_by_github(self):
        """Finds user by GitHub login."""
        uc = UserConfig(telegram_id=111, name="alice", github="alice-dev")
        config = self._make_config({111: uc})
        assert config.get_user_by_github("alice-dev") is uc

    def test_get_user_by_github_case_insensitive(self):
        """GitHub login match is case-insensitive."""
        uc = UserConfig(telegram_id=111, name="alice", github="Alice-Dev")
        config = self._make_config({111: uc})
        assert config.get_user_by_github("alice-dev") is uc
        assert config.get_user_by_github("ALICE-DEV") is uc

    def test_get_user_by_github_not_found(self):
        """Returns None for unknown GitHub login."""
        uc = UserConfig(telegram_id=111, name="alice", github="alice-dev")
        config = self._make_config({111: uc})
        assert config.get_user_by_github("unknown") is None

    def test_get_user_by_github_no_yaml(self):
        """Returns None when user_configs is None."""
        config = self._make_config(None)
        assert config.get_user_by_github("alice") is None

    def test_get_admins(self):
        """Returns list of admin users only."""
        admin = UserConfig(telegram_id=111, name="alice", role="admin")
        user = UserConfig(telegram_id=222, name="bob", role="user")
        config = self._make_config({111: admin, 222: user})
        admins = config.get_admins()
        assert len(admins) == 1
        assert admins[0] is admin

    def test_get_admins_none(self):
        """Returns empty list when no admins exist."""
        user = UserConfig(telegram_id=222, name="bob", role="user")
        config = self._make_config({222: user})
        assert config.get_admins() == []

    def test_get_admins_no_yaml(self):
        """Returns empty list when user_configs is None."""
        config = self._make_config(None)
        assert config.get_admins() == []
