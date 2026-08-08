"""
Tests for per-user configuration (users.yaml).

Covers:
1. UserConfig dataclass construction
2. _load_user_configs("claude", "") YAML parsing, validation, and edge cases
3. Config.get_user_config() lookup
4. Config.get_user_by_github() lookup (case-insensitive)
5. Config.get_admins() filtering
6. Legacy fallback via ALLOWED_USER_IDS
"""

import logging
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from kai.config import (
    _YAML_MALFORMED,
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
        assert uc.model is None
        assert uc.timeout is None
        assert uc.workspace_base is None
        assert uc.github_repos == []
        assert uc.github_notify_chat_id is None
        assert uc.pr_review is None
        assert uc.issue_triage is None
        assert uc.allowed_services == []
        assert uc.backend is None
        assert uc.provider is None

    def test_all_fields(self):
        """Full config with every field populated."""
        uc = UserConfig(
            telegram_id=123,
            name="alice",
            role="admin",
            github="alice-dev",
            os_user="alice",
            home_workspace=Path("/home/alice/workspace"),
            model="opus",
            timeout=300,
            workspace_base=Path("/home/alice/projects"),
            github_repos=["alice/repo-a", "alice/repo-b"],
            github_notify_chat_id=-100123456789,
            pr_review=True,
            issue_triage=False,
            allowed_services=["perplexity"],
            backend="goose",
            provider="openai",
        )
        assert uc.role == "admin"
        assert uc.github == "alice-dev"
        assert uc.os_user == "alice"
        assert uc.model == "opus"
        assert uc.timeout == 300
        assert uc.workspace_base == Path("/home/alice/projects")
        assert uc.github_repos == ["alice/repo-a", "alice/repo-b"]
        assert uc.github_notify_chat_id == -100123456789
        assert uc.pr_review is True
        assert uc.issue_triage is False
        assert uc.allowed_services == ["perplexity"]
        assert uc.backend == "goose"
        assert uc.provider == "openai"

    def test_frozen(self):
        """UserConfig is immutable."""
        uc = UserConfig(telegram_id=123, name="alice")
        with pytest.raises(AttributeError):
            uc.name = "bob"  # type: ignore[misc]


# ── _load_user_configs ──────────────────────────────────────────────


class TestLoadUserConfigs:
    def _yaml_dict(self, content):
        """Parse YAML content as the loader would see it after sudo-cat.

        The runtime path is `_read_protected_yaml('users.yaml')` which
        returns a parsed dict (or `_YAML_MALFORMED` / None). Tests patch
        that function with the dict this helper produces; no file is
        written because the loader no longer reads from PROJECT_ROOT.
        """
        result = yaml.safe_load(textwrap.dedent(content))
        return result if isinstance(result, dict) else _YAML_MALFORMED

    def test_basic_loading(self, tmp_path):
        """Loads two users with correct fields."""
        ws = tmp_path / "ws"
        ws.mkdir()
        data = self._yaml_dict(
            f"""\
            users:
              - telegram_id: 111
                name: alice
                role: admin
                github: alice-dev
                os_user: alice
                home_workspace: {ws}
              - telegram_id: 222
                name: bob
                role: user
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
        ):
            configs = _load_user_configs("claude", "")

        assert configs is not None
        assert len(configs) == 2
        assert configs[111].name == "alice"
        assert configs[111].role == "admin"
        assert configs[111].github == "alice-dev"
        assert configs[111].os_user == "alice"
        assert configs[111].home_workspace == ws.resolve()
        assert configs[222].name == "bob"
        assert configs[222].role == "user"

    def test_missing_file(self, monkeypatch):
        """Raises SystemExit when /etc/kai/users.yaml is absent.

        users.yaml is mandatory. The protected reader returns None
        for missing files; the loader raises with an operator-facing
        error naming the path and pointing at `make config`. No
        fallback to ALLOWED_USER_IDS at runtime.
        """
        monkeypatch.delenv("ALLOWED_USER_IDS", raising=False)
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            pytest.raises(SystemExit, match=r"users\.yaml is required"),
        ):
            _load_user_configs("claude", "")

    def test_missing_file_with_allowed_user_ids_appends_migration_hint(self, monkeypatch):
        """Missing-file error appends a migration hint when ALLOWED_USER_IDS is set.

        Operator UX: a legacy env-only install needs to know that the
        old auth path no longer works and that `make config` will
        migrate the existing telegram_ids into users.yaml.
        """
        monkeypatch.setenv("ALLOWED_USER_IDS", "12345")
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            pytest.raises(SystemExit, match="ALLOWED_USER_IDS is set in env but is no longer honored"),
        ):
            _load_user_configs("claude", "")

    def test_empty_file(self):
        """Raises SystemExit when /etc/kai/users.yaml is empty / non-dict top-level.

        The protected reader normalizes an empty or non-dict top-level
        to `_YAML_MALFORMED`; the loader treats that as fail-closed and
        raises rather than silently constructing an empty config.
        """
        with (
            patch("kai.config._read_protected_yaml", return_value=_YAML_MALFORMED),
            pytest.raises(SystemExit, match="malformed"),
        ):
            _load_user_configs("claude", "")

    def test_invalid_yaml(self):
        """Raises SystemExit when /etc/kai/users.yaml is malformed.

        Same code path as the empty-file case: the protected reader
        already raised on parse and returned `_YAML_MALFORMED`.
        """
        with (
            patch("kai.config._read_protected_yaml", return_value=_YAML_MALFORMED),
            pytest.raises(SystemExit, match="malformed"),
        ):
            _load_user_configs("claude", "")

    def test_missing_telegram_id(self, tmp_path):
        """Entry without telegram_id is skipped."""
        data = self._yaml_dict(
            """\
            users:
              - name: alice
                role: admin
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
            pytest.raises(SystemExit, match="no valid user entries"),
        ):
            _load_user_configs("claude", "")

    def test_whitespace_only_name(self, tmp_path):
        """Whitespace-only name is treated as missing."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: "   "
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
            pytest.raises(SystemExit, match="no valid user entries"),
        ):
            _load_user_configs("claude", "")

    def test_missing_name(self, tmp_path):
        """Entry without name is skipped."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                role: admin
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
            pytest.raises(SystemExit, match="no valid user entries"),
        ):
            _load_user_configs("claude", "")

    def test_invalid_role(self, tmp_path):
        """Invalid role causes the entry to be skipped."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                role: superuser
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
            pytest.raises(SystemExit, match="no valid user entries"),
        ):
            _load_user_configs("claude", "")

    def test_lingering_max_budget_ignored_with_warning(self, tmp_path, caplog):
        """A `max_budget` key from an older users.yaml is ignored with
        a warning; the entry itself still loads. Tolerance keeps an
        un-migrated file working after upgrade."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                max_budget: 15.0
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
            caplog.at_level(logging.WARNING, logger="kai.config"),
        ):
            configs = _load_user_configs("claude", "")

        assert configs is not None
        assert configs[111].name == "alice"
        assert "'max_budget' for alice is no longer supported; ignoring" in caplog.text

    def test_bool_telegram_id_rejected(self, tmp_path):
        """Boolean telegram_id is rejected."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: true
                name: alice
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
            pytest.raises(SystemExit, match="no valid user entries"),
        ):
            _load_user_configs("claude", "")

    def test_duplicate_ids(self, tmp_path):
        """Duplicate telegram_id: first wins."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
              - telegram_id: 111
                name: bob
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
        ):
            configs = _load_user_configs("claude", "")
        assert configs is not None
        assert len(configs) == 1
        assert configs[111].name == "alice"

    def test_home_workspace_empty_string(self, tmp_path):
        """Empty home_workspace string is treated as None, not CWD."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                home_workspace: ""
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
        ):
            configs = _load_user_configs("claude", "")
        assert configs is not None
        assert len(configs) == 1
        assert configs[111].home_workspace is None

    def test_home_workspace_nonexistent_warns_but_keeps_user(self, tmp_path):
        """Non-existent home_workspace warns and falls back to None, not skip."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                home_workspace: /nonexistent/path/12345
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
        ):
            configs = _load_user_configs("claude", "")
        assert configs is not None
        assert len(configs) == 1
        # home_workspace falls back to None (global default)
        assert configs[111].home_workspace is None

    def test_protected_path_is_the_only_path(self, tmp_path):
        """`/etc/kai/users.yaml` is canonical; the loader reads it via
        `_read_protected_yaml` and uses whatever that returns.
        """
        protected_data = {"users": [{"telegram_id": 111, "name": "alice", "role": "admin"}]}
        with patch("kai.config._read_protected_yaml", return_value=protected_data):
            configs = _load_user_configs("claude", "")
        assert configs is not None
        assert configs[111].name == "alice"

    def test_stray_project_root_users_yaml_is_ignored(self, tmp_path, monkeypatch):
        """A stray `PROJECT_ROOT/users.yaml` does not affect the loader.

        The dual-path fallback was removed in #559: when the protected
        reader returns None (canonical file absent), the loader raises
        SystemExit (mandatory users.yaml). A leftover users.yaml inside
        the source tree must not feed the daemon and must not bypass
        the mandatory-users contract.
        """
        monkeypatch.delenv("ALLOWED_USER_IDS", raising=False)
        stray = tmp_path / "users.yaml"
        stray.write_text("users:\n  - telegram_id: 999\n    name: stray\n    role: admin\n")
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
            pytest.raises(SystemExit, match=r"users\.yaml is required"),
        ):
            _load_user_configs("claude", "")

    def test_no_admin_warning(self, tmp_path, caplog):
        """All users with role 'user' logs a warning but does not fail."""
        data = self._yaml_dict(
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
            patch("kai.config._read_protected_yaml", return_value=data),
        ):
            configs = _load_user_configs("claude", "")
        assert configs is not None
        assert len(configs) == 2
        assert "no admin users defined" in caplog.text.lower()

    def test_default_role_is_user(self, tmp_path):
        """Omitting role defaults to 'user'."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
        ):
            configs = _load_user_configs("claude", "")
        assert configs is not None
        assert configs[111].role == "user"

    def test_os_user_stored(self, tmp_path):
        """os_user is stored as a string (not validated in Phase 1)."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                os_user: alice_os
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
        ):
            configs = _load_user_configs("claude", "")
        assert configs is not None
        assert configs[111].os_user == "alice_os"

    # ── New per-user setting fields (model, timeout) ──────────────────

    def test_model_parsed(self, tmp_path):
        """Valid model name is stored, lowercased."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                model: Opus
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
        ):
            configs = _load_user_configs("claude", "")
        assert configs is not None
        assert configs[111].model == "opus"

    def test_invalid_model_ignored(self, tmp_path, caplog):
        """Invalid model name is ignored (set to None), user still loads."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                model: gpt4
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
        ):
            configs = _load_user_configs("claude", "")
        assert configs is not None
        assert configs[111].model is None
        assert "invalid model" in caplog.text.lower()

    def test_timeout_parsed(self, tmp_path):
        """Valid timeout is stored as int."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                timeout: 300
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
        ):
            configs = _load_user_configs("claude", "")
        assert configs is not None
        assert configs[111].timeout == 300

    def test_invalid_timeout_ignored(self, tmp_path, caplog):
        """Negative timeout is ignored, user still loads."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                timeout: -5
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
        ):
            configs = _load_user_configs("claude", "")
        assert configs is not None
        assert configs[111].timeout is None
        assert "invalid timeout" in caplog.text.lower()

    def test_context_window_key_tolerated_with_warning(self, tmp_path, caplog):
        """A users.yaml still carrying context_window (the setting was
        removed) loads cleanly; the key is ignored with a warning so
        the operator knows it has no effect."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                context_window: 200000
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
        ):
            configs = _load_user_configs("claude", "")
        assert configs is not None
        assert 111 in configs
        assert "'context_window' is no longer supported" in caplog.text

    def test_new_fields_default_none(self, tmp_path):
        """New optional fields default to None when omitted."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
        ):
            configs = _load_user_configs("claude", "")
        assert configs is not None
        assert configs[111].model is None
        assert configs[111].timeout is None
        assert configs[111].workspace_base is None
        assert configs[111].github_repos == []
        assert configs[111].github_notify_chat_id is None
        assert configs[111].pr_review is None
        assert configs[111].issue_triage is None
        assert configs[111].allowed_services == []

    # ── workspace_base field ──

    def test_workspace_base_parsed(self, tmp_path):
        """Valid workspace_base directory is stored as resolved Path."""
        ws_base = tmp_path / "projects"
        ws_base.mkdir()
        data = self._yaml_dict(
            f"""\
            users:
              - telegram_id: 111
                name: alice
                workspace_base: {ws_base}
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
        ):
            configs = _load_user_configs("claude", "")
        assert configs is not None
        assert configs[111].workspace_base == ws_base.resolve()

    def test_workspace_base_missing_dir_warns(self, tmp_path, caplog):
        """Non-existent workspace_base warns and falls back to None."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                workspace_base: /nonexistent/path/12345
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
        ):
            configs = _load_user_configs("claude", "")
        assert configs is not None
        assert len(configs) == 1
        assert configs[111].workspace_base is None
        assert "workspace_base not found" in caplog.text.lower()

    def test_workspace_base_empty_string(self, tmp_path):
        """Empty workspace_base string is treated as None."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                workspace_base: ""
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
        ):
            configs = _load_user_configs("claude", "")
        assert configs is not None
        assert configs[111].workspace_base is None

    # ── allowed_services field ──

    def test_allowed_services_parsed_deduplicated_and_ordered(self, tmp_path):
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                allowed_services:
                  - perplexity
                  - weather
                  - perplexity
            """,
        )
        with patch("kai.config._read_protected_yaml", return_value=data):
            configs = _load_user_configs("claude", "")

        assert configs[111].allowed_services == ["perplexity", "weather"]

    def test_allowed_services_invalid_entries_fail_closed(self, tmp_path, caplog):
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                allowed_services:
                  - perplexity
                  - '*'
                  - nested/service
                  - 42
            """,
        )
        with patch("kai.config._read_protected_yaml", return_value=data):
            configs = _load_user_configs("claude", "")

        assert configs[111].allowed_services == ["perplexity"]
        assert "invalid allowed_services entry" in caplog.text

    def test_allowed_services_not_list_is_ignored(self, tmp_path, caplog):
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                allowed_services: perplexity
            """,
        )
        with patch("kai.config._read_protected_yaml", return_value=data):
            configs = _load_user_configs("claude", "")

        assert configs[111].allowed_services == []
        assert "allowed_services for alice must be a list" in caplog.text

    # ── github_repos field ──

    def test_github_repos_parsed(self, tmp_path):
        """Valid github_repos list is stored as list of strings."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                github_repos:
                  - alice/repo-a
                  - alice/repo-b
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
        ):
            configs = _load_user_configs("claude", "")
        assert configs is not None
        assert configs[111].github_repos == ["alice/repo-a", "alice/repo-b"]

    def test_github_repos_invalid_format(self, tmp_path, caplog):
        """Invalid repo format (no slash) is skipped with warning."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                github_repos:
                  - valid/repo
                  - no-slash
                  - too/many/slashes
                  - /
                  - /repo
                  - owner/
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
        ):
            configs = _load_user_configs("claude", "")
        assert configs is not None
        assert configs[111].github_repos == ["valid/repo"]
        assert "no-slash" in caplog.text
        assert "too/many/slashes" in caplog.text

    def test_github_repos_not_list(self, tmp_path, caplog):
        """Non-list github_repos is ignored with warning."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                github_repos: alice/repo
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
        ):
            configs = _load_user_configs("claude", "")
        assert configs is not None
        assert configs[111].github_repos == []
        assert "must be a list" in caplog.text

    def test_github_repos_default(self, tmp_path):
        """Omitted github_repos defaults to empty list."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
        ):
            configs = _load_user_configs("claude", "")
        assert configs is not None
        assert configs[111].github_repos == []

    # ── github_notify_chat_id field ──

    def test_github_notify_chat_id_parsed(self, tmp_path):
        """Valid github_notify_chat_id is stored as int."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                github_notify_chat_id: 999888777
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
        ):
            configs = _load_user_configs("claude", "")
        assert configs is not None
        assert configs[111].github_notify_chat_id == 999888777

    def test_github_notify_chat_id_negative(self, tmp_path):
        """Negative chat IDs (group chats) are valid."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                github_notify_chat_id: -100123456789
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
        ):
            configs = _load_user_configs("claude", "")
        assert configs is not None
        assert configs[111].github_notify_chat_id == -100123456789

    def test_github_notify_chat_id_invalid(self, tmp_path, caplog):
        """Invalid github_notify_chat_id warns and uses None."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                github_notify_chat_id: not-a-number
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
        ):
            configs = _load_user_configs("claude", "")
        assert configs is not None
        assert configs[111].github_notify_chat_id is None
        assert "invalid github_notify_chat_id" in caplog.text

    # ── pr_review / issue_triage fields ──

    def test_pr_review_bool(self, tmp_path):
        """pr_review True/False parsed from yaml."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                pr_review: true
              - telegram_id: 222
                name: bob
                pr_review: false
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
        ):
            configs = _load_user_configs("claude", "")
        assert configs is not None
        assert configs[111].pr_review is True
        assert configs[222].pr_review is False

    def test_pr_review_none_when_omitted(self, tmp_path):
        """Omitted pr_review defaults to None (use global)."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
        ):
            configs = _load_user_configs("claude", "")
        assert configs is not None
        assert configs[111].pr_review is None

    def test_issue_triage_bool(self, tmp_path):
        """issue_triage True/False parsed from yaml."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                issue_triage: true
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
        ):
            configs = _load_user_configs("claude", "")
        assert configs is not None
        assert configs[111].issue_triage is True

    def test_pr_review_non_bool_warns(self, tmp_path, caplog):
        """Non-boolean pr_review warns and uses None."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                pr_review: "yes"
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
        ):
            configs = _load_user_configs("claude", "")
        assert configs is not None
        assert configs[111].pr_review is None
        assert "pr_review" in caplog.text
        assert "must be true or false" in caplog.text

    def test_issue_triage_non_bool_warns(self, tmp_path, caplog):
        """Non-boolean issue_triage warns and uses None."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                issue_triage: 1
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
        ):
            configs = _load_user_configs("claude", "")
        assert configs is not None
        assert configs[111].issue_triage is None
        assert "issue_triage" in caplog.text

    # ── Per-user backend/provider ─────────────────────────────────

    def test_valid_agent_backend(self, tmp_path):
        """Valid backend is parsed and stored."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                backend: goose
                provider: openai
                model: gpt-5.4
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
        ):
            configs = _load_user_configs("claude", "")
        assert configs is not None
        assert configs[111].backend == "goose"
        assert configs[111].provider == "openai"
        assert configs[111].model == "gpt-5.4"

    def test_invalid_agent_backend_exits(self, tmp_path):
        """Invalid backend causes SystemExit."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                backend: invalid
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
            pytest.raises(SystemExit, match="invalid backend"),
        ):
            _load_user_configs("claude", "")

    def test_invalid_llm_provider_exits(self, tmp_path):
        """Invalid provider for the user's backend causes SystemExit."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                backend: goose
                provider: badprovider
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
            pytest.raises(SystemExit, match="invalid provider"),
        ):
            _load_user_configs("claude", "")

    def test_goose_backend_without_provider_exits(self, tmp_path):
        """Goose backend with no resolvable provider is a fatal error."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                backend: goose
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
            # Global provider is "" (empty), user has none set
            pytest.raises(SystemExit, match="no provider is configured"),
        ):
            _load_user_configs("claude", "")

    def test_legacy_agent_backend_in_users_yaml_still_resolved_with_warning(self, tmp_path, caplog):
        """A users.yaml entry using the oldest deprecated `agent_backend:`
        key still resolves to the backend field for one release, with a
        one-shot deprecation warning naming the user (the per-user key was
        renamed twice: agent_backend -> default_backend -> backend)."""
        import logging

        import kai.config as config_module

        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                agent_backend: codex
            """,
        )
        config_module._renamed_key_deprecation_warned.clear()
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
            caplog.at_level(logging.WARNING),
        ):
            configs = _load_user_configs("claude", "")
        assert configs[111].backend == "codex"
        assert any(
            "agent_backend is deprecated" in r.message and "rename to backend" in r.message for r in caplog.records
        )

    def test_legacy_default_backend_in_users_yaml_resolved_with_warning(self, tmp_path, caplog):
        """A users.yaml entry using the intermediate deprecated
        `default_backend:` key (the one #719 introduced as the per-user
        key) still resolves to the backend field, with a warning."""
        import logging

        import kai.config as config_module

        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                default_backend: codex
            """,
        )
        config_module._renamed_key_deprecation_warned.clear()
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
            caplog.at_level(logging.WARNING),
        ):
            configs = _load_user_configs("claude", "")
        assert configs[111].backend == "codex"
        assert any(
            "default_backend is deprecated" in r.message and "rename to backend" in r.message for r in caplog.records
        )

    def test_legacy_llm_provider_in_users_yaml_resolved_with_warning(self, tmp_path, caplog):
        """A users.yaml entry using the deprecated `llm_provider:` key
        still resolves to the provider field for one release, with a
        one-shot deprecation warning naming the user."""
        import logging

        import kai.config as config_module

        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                backend: goose
                llm_provider: openai
            """,
        )
        config_module._renamed_key_deprecation_warned.clear()
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
            caplog.at_level(logging.WARNING),
        ):
            configs = _load_user_configs("claude", "")
        assert configs[111].provider == "openai"
        assert any(
            "llm_provider is deprecated" in r.message and "rename to provider" in r.message for r in caplog.records
        )

    def test_user_without_backend_inherits_nonclaude_global(self, tmp_path):
        """A users.yaml entry that omits the backend key must NOT be
        pinned to claude: its stored override stays None so the
        runtime cascade (user override OR global) inherits the global
        backend. Regression guard for the resolver-default split: the
        per-user reader passes default=None, not default="claude".
        """
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
            """,
        )
        with patch("kai.config._read_protected_yaml", return_value=data):
            # Global backend is goose; the user has no per-user backend.
            configs = _load_user_configs("goose", "openai")
        # None on the stored override is what makes the runtime cascade
        # inherit the global backend rather than forcing claude.
        assert configs[111].backend is None
        # And the canonical cascade resolves the user to the global goose.
        from kai.config import Config, get_user_backend_and_provider

        global_cfg = Config(
            telegram_bot_token="test",
            allowed_user_ids={1},
            default_backend="goose",
            default_provider="openai",
        )
        backend, _provider = get_user_backend_and_provider(configs[111], global_cfg)
        assert backend == "goose"

    def test_goose_backend_inherits_global_provider(self, tmp_path):
        """User with goose backend inherits global provider."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                backend: goose
                model: gpt-5.4
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
        ):
            # Global provider is "openai"
            configs = _load_user_configs("goose", "openai")
        assert configs is not None
        # User inherits global provider, no per-user override stored
        assert configs[111].backend == "goose"
        assert configs[111].provider is None

    def test_user_without_provider_inherits_global(self, tmp_path):
        """A goose user that omits the provider key inherits the global
        provider through the cascade rather than resolving to "". Guards
        the per-user provider reader's default=None contract (mirrors the
        backend inheritance guard)."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                backend: goose
                model: gpt-5.4
            """,
        )
        with patch("kai.config._read_protected_yaml", return_value=data):
            configs = _load_user_configs("goose", "openai")
        # None stored is what lets the runtime cascade inherit the global.
        assert configs[111].provider is None
        from kai.config import Config, get_user_backend_and_provider

        global_cfg = Config(
            telegram_bot_token="test",
            allowed_user_ids={1},
            default_backend="goose",
            default_provider="openai",
        )
        _backend, provider = get_user_backend_and_provider(configs[111], global_cfg)
        assert provider == "openai"

    def test_model_validated_against_user_provider(self, tmp_path):
        """Model invalid for user's effective provider is rejected."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                backend: goose
                provider: openai
                model: opus
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=data),
        ):
            configs = _load_user_configs("claude", "")
        assert configs is not None
        # "opus" is not valid for openai - should be cleared to None
        assert configs[111].model is None

    def test_open_ended_provider_warns_no_model(self, tmp_path, caplog):
        """Open-ended provider with no model emits a warning."""
        data = self._yaml_dict(
            """\
            users:
              - telegram_id: 111
                name: alice
                backend: goose
                provider: ollama
            """,
        )
        import logging

        with (
            patch("kai.config._read_protected_yaml", return_value=data),
            caplog.at_level(logging.WARNING, logger="kai.config"),
        ):
            configs = _load_user_configs("claude", "")
        assert configs is not None
        assert "open-ended provider" in caplog.text
        assert "ollama" in caplog.text


# ── Config convenience methods ──────────────────────────────────────


class TestConfigUserMethods:
    def _make_config(self, user_configs: dict | None = None):
        # user_configs is non-optional on Config post-#565 tranche A.
        # Tests that previously passed None now default to empty dict;
        # the loader's mandatory-users contract is verified elsewhere
        # in TestLoadUserConfigs.
        return Config(
            telegram_bot_token="test",
            allowed_user_ids={1},
            user_configs=user_configs if user_configs is not None else {},
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
