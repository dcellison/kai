"""
Tests for per-workspace configuration (workspaces.yaml).

Covers:
1. WorkspaceConfig dataclass construction
2. _load_workspace_configs() YAML parsing, validation, and edge cases
3. _read_protected_yaml() protected file reading
4. parse_env_file() KEY=VALUE parsing
5. Config.get_workspace_config() lookup
6. Merge of YAML workspace paths into allowed_workspaces
"""

import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from kai.config import (
    WorkspaceConfig,
    _load_workspace_configs,
    _read_protected_yaml,
    _strip_quotes,
    parse_env_file,
)

# ── WorkspaceConfig dataclass ───────────────────────────────────────


class TestWorkspaceConfig:
    def test_required_path_only(self):
        """Minimal config: just a path, everything else defaults to None."""
        ws = WorkspaceConfig(path=Path("/tmp/ws"))
        assert ws.path == Path("/tmp/ws")
        assert ws.model is None
        assert ws.budget is None
        assert ws.timeout is None
        assert ws.env is None
        assert ws.env_file is None
        assert ws.system_prompt is None
        assert ws.system_prompt_file is None

    def test_all_fields(self):
        """Full config with every field populated."""
        ws = WorkspaceConfig(
            path=Path("/tmp/ws"),
            model="opus",
            budget=15.0,
            timeout=300,
            env={"FOO": "bar"},
            env_file=Path("/tmp/.env"),
            system_prompt="Be helpful",
        )
        assert ws.model == "opus"
        assert ws.budget == 15.0
        assert ws.timeout == 300
        assert ws.env == {"FOO": "bar"}
        assert ws.system_prompt == "Be helpful"

    def test_frozen(self):
        """WorkspaceConfig is immutable."""
        ws = WorkspaceConfig(path=Path("/tmp/ws"))
        with pytest.raises(AttributeError):
            ws.model = "haiku"  # type: ignore[misc]


# ── parse_env_file ──────────────────────────────────────────────────


class TestParseEnvFile:
    def test_basic_key_value(self, tmp_path):
        """Parses simple KEY=VALUE lines."""
        env_file = tmp_path / ".env"
        env_file.write_text("FOO=bar\nBAZ=qux\n")
        result = parse_env_file(env_file)
        assert result == {"FOO": "bar", "BAZ": "qux"}

    def test_quoted_values(self, tmp_path):
        """Strips surrounding quotes from values."""
        env_file = tmp_path / ".env"
        env_file.write_text("DOUBLE=\"hello\"\nSINGLE='world'\n")
        result = parse_env_file(env_file)
        assert result == {"DOUBLE": "hello", "SINGLE": "world"}

    def test_export_prefix(self, tmp_path):
        """Handles 'export KEY=VALUE' lines."""
        env_file = tmp_path / ".env"
        env_file.write_text("export FOO=bar\n")
        result = parse_env_file(env_file)
        assert result == {"FOO": "bar"}

    def test_comments_and_blanks(self, tmp_path):
        """Skips comments and blank lines."""
        env_file = tmp_path / ".env"
        env_file.write_text("# comment\n\nFOO=bar\n  \n# another\n")
        result = parse_env_file(env_file)
        assert result == {"FOO": "bar"}

    def test_malformed_lines(self, tmp_path):
        """Lines without '=' are skipped."""
        env_file = tmp_path / ".env"
        env_file.write_text("GOOD=value\nBADLINE\n")
        result = parse_env_file(env_file)
        assert result == {"GOOD": "value"}

    def test_missing_file(self, tmp_path):
        """Returns empty dict for non-existent file."""
        result = parse_env_file(tmp_path / "nonexistent")
        assert result == {}

    def test_value_with_equals(self, tmp_path):
        """Handles values containing '=' (only first = is the delimiter)."""
        env_file = tmp_path / ".env"
        env_file.write_text("URL=postgres://host/db?opt=1\n")
        result = parse_env_file(env_file)
        assert result == {"URL": "postgres://host/db?opt=1"}

    def test_single_quotes_containing_double(self, tmp_path):
        """Single-quoted values preserve internal double quotes."""
        env_file = tmp_path / ".env"
        env_file.write_text("""KEY='he said "hello"'\n""")
        result = parse_env_file(env_file)
        assert result == {"KEY": 'he said "hello"'}

    def test_double_quotes_containing_single(self, tmp_path):
        """Double-quoted values preserve internal single quotes."""
        env_file = tmp_path / ".env"
        env_file.write_text('KEY="it\'s fine"\n')
        result = parse_env_file(env_file)
        assert result == {"KEY": "it's fine"}

    def test_mismatched_quotes_not_stripped(self, tmp_path):
        """Mismatched outer quotes are left as-is."""
        env_file = tmp_path / ".env"
        env_file.write_text("""KEY='value"\n""")
        result = parse_env_file(env_file)
        assert result == {"KEY": "'value\""}

    def test_unquoted_value_unchanged(self, tmp_path):
        """Values without surrounding quotes pass through unchanged."""
        env_file = tmp_path / ".env"
        env_file.write_text("KEY=no_quotes_here\n")
        result = parse_env_file(env_file)
        assert result == {"KEY": "no_quotes_here"}

    def test_single_quote_char_unchanged(self, tmp_path):
        """A value that is just a single quote character is not corrupted."""
        env_file = tmp_path / ".env"
        env_file.write_text("KEY='\n")
        result = parse_env_file(env_file)
        assert result == {"KEY": "'"}


# ── _strip_quotes ──────────────────────────────────────────────────


class TestStripQuotes:
    def test_double_quoted(self):
        assert _strip_quotes('"hello"') == "hello"

    def test_single_quoted(self):
        assert _strip_quotes("'hello'") == "hello"

    def test_mismatched(self):
        assert _strip_quotes("'hello\"") == "'hello\""

    def test_no_quotes(self):
        assert _strip_quotes("hello") == "hello"

    def test_empty_string(self):
        assert _strip_quotes("") == ""

    def test_single_char_quote(self):
        assert _strip_quotes("'") == "'"

    def test_empty_quoted_string(self):
        assert _strip_quotes('""') == ""

    def test_inner_quotes_preserved(self):
        assert _strip_quotes("""'he said "hello"'""") == 'he said "hello"'


# ── _read_protected_yaml ────────────────────────────────────────────


class TestReadProtectedYaml:
    def test_returns_parsed_yaml(self):
        """Returns parsed dict when sudo cat succeeds."""
        with patch("kai.config._read_protected_file", return_value="workspaces:\n  - path: /tmp\n"):
            result = _read_protected_yaml("workspaces.yaml")
        assert result == {"workspaces": [{"path": "/tmp"}]}

    def test_returns_none_on_missing_file(self):
        """Returns None when the protected file doesn't exist."""
        with patch("kai.config._read_protected_file", return_value=None):
            result = _read_protected_yaml("workspaces.yaml")
        assert result is None

    def test_returns_malformed_on_non_dict(self):
        """Returns _YAML_MALFORMED when YAML parses to a non-dict."""
        from kai.config import _YAML_MALFORMED

        with patch("kai.config._read_protected_file", return_value="- item1\n- item2\n"):
            result = _read_protected_yaml("workspaces.yaml")
        assert result is _YAML_MALFORMED

    def test_returns_malformed_on_invalid_yaml(self):
        """Returns _YAML_MALFORMED (not None) when YAML is invalid."""
        from kai.config import _YAML_MALFORMED

        with patch("kai.config._read_protected_file", return_value="{{bad["):
            result = _read_protected_yaml("workspaces.yaml")
        assert result is _YAML_MALFORMED


# ── _load_workspace_configs ─────────────────────────────────────────


class TestLoadWorkspaceConfigs:
    def _write_yaml(self, tmp_path, content):
        """Write a workspaces.yaml file and patch PROJECT_ROOT to find it."""
        yaml_file = tmp_path / "workspaces.yaml"
        yaml_file.write_text(textwrap.dedent(content))
        return yaml_file

    def test_basic_loading(self, tmp_path):
        """Loads two workspaces with correct fields."""
        ws1 = tmp_path / "ws1"
        ws2 = tmp_path / "ws2"
        ws1.mkdir()
        ws2.mkdir()
        self._write_yaml(
            tmp_path,
            f"""\
            workspaces:
              - path: {ws1}
                claude:
                  model: opus
                  budget: 15.0
                  timeout: 300
              - path: {ws2}
                claude:
                  model: haiku
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_workspace_configs()

        assert len(configs) == 2
        assert configs[ws1.resolve()].model == "opus"
        assert configs[ws1.resolve()].budget == 15.0
        assert configs[ws1.resolve()].timeout == 300
        assert configs[ws2.resolve()].model == "haiku"

    def test_missing_file(self, tmp_path):
        """Returns empty dict when no YAML file exists."""
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_workspace_configs()
        assert configs == {}

    def test_empty_file(self, tmp_path):
        """Returns empty dict for an empty YAML file."""
        self._write_yaml(tmp_path, "")
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_workspace_configs()
        assert configs == {}

    def test_invalid_yaml(self, tmp_path):
        """Returns empty dict for malformed YAML."""
        (tmp_path / "workspaces.yaml").write_text("{{bad yaml[")
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_workspace_configs()
        assert configs == {}

    def test_path_resolution(self, tmp_path):
        """Paths with ~ are expanded and resolved."""
        ws = tmp_path / "myproject"
        ws.mkdir()
        self._write_yaml(tmp_path, f"workspaces:\n  - path: {ws}\n")
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_workspace_configs()
        # Key is the resolved path
        assert ws.resolve() in configs

    def test_nonexistent_path_skipped(self, tmp_path):
        """Workspace with non-existent path is skipped."""
        self._write_yaml(tmp_path, "workspaces:\n  - path: /nonexistent/path/12345\n")
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_workspace_configs()
        assert configs == {}

    def test_duplicate_paths(self, tmp_path):
        """Duplicate paths: first wins."""
        ws = tmp_path / "ws"
        ws.mkdir()
        self._write_yaml(
            tmp_path,
            f"""\
            workspaces:
              - path: {ws}
                claude:
                  model: opus
              - path: {ws}
                claude:
                  model: haiku
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_workspace_configs()
        assert len(configs) == 1
        assert configs[ws.resolve()].model == "opus"

    def test_invalid_model(self, tmp_path):
        """Invalid model name causes the entry to be skipped."""
        ws = tmp_path / "ws"
        ws.mkdir()
        self._write_yaml(
            tmp_path,
            f"""\
            workspaces:
              - path: {ws}
                claude:
                  model: gpt-4
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_workspace_configs()
        assert configs == {}

    def test_bool_budget_rejected(self, tmp_path):
        """Boolean budget (e.g. true) is rejected, not silently cast to $1.00."""
        ws = tmp_path / "ws"
        ws.mkdir()
        self._write_yaml(
            tmp_path,
            f"""\
            workspaces:
              - path: {ws}
                claude:
                  budget: true
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_workspace_configs()
        assert configs == {}

    def test_negative_budget(self, tmp_path):
        """Negative budget causes the entry to be skipped."""
        ws = tmp_path / "ws"
        ws.mkdir()
        self._write_yaml(
            tmp_path,
            f"""\
            workspaces:
              - path: {ws}
                claude:
                  budget: -5.0
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_workspace_configs()
        assert configs == {}

    def test_mutual_exclusion_system_prompt(self, tmp_path):
        """Both system_prompt and system_prompt_file causes skip."""
        ws = tmp_path / "ws"
        ws.mkdir()
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("hello")
        self._write_yaml(
            tmp_path,
            f"""\
            workspaces:
              - path: {ws}
                claude:
                  system_prompt: "inline"
                  system_prompt_file: {prompt_file}
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_workspace_configs()
        assert configs == {}

    def test_env_file_missing(self, tmp_path):
        """Missing env_file causes the entry to be skipped."""
        ws = tmp_path / "ws"
        ws.mkdir()
        self._write_yaml(
            tmp_path,
            f"""\
            workspaces:
              - path: {ws}
                claude:
                  env_file: /nonexistent/file.env
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_workspace_configs()
        assert configs == {}

    def test_env_file_valid(self, tmp_path):
        """Valid env_file path is stored."""
        ws = tmp_path / "ws"
        ws.mkdir()
        env_file = tmp_path / ".env.kai"
        env_file.write_text("FOO=bar\n")
        self._write_yaml(
            tmp_path,
            f"""\
            workspaces:
              - path: {ws}
                claude:
                  env_file: {env_file}
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_workspace_configs()
        assert configs[ws.resolve()].env_file == env_file.resolve()

    def test_system_prompt_file_missing(self, tmp_path):
        """Missing system_prompt_file causes the entry to be skipped."""
        ws = tmp_path / "ws"
        ws.mkdir()
        self._write_yaml(
            tmp_path,
            f"""\
            workspaces:
              - path: {ws}
                claude:
                  system_prompt_file: /nonexistent/prompt.txt
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_workspace_configs()
        assert configs == {}

    def test_system_prompt_file_valid(self, tmp_path):
        """Valid system_prompt_file is stored and content is readable."""
        ws = tmp_path / "ws"
        ws.mkdir()
        prompt_file = tmp_path / "prompt.txt"
        prompt_file.write_text("Be concise.")
        self._write_yaml(
            tmp_path,
            f"""\
            workspaces:
              - path: {ws}
                claude:
                  system_prompt_file: {prompt_file}
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_workspace_configs()
        cfg = configs[ws.resolve()]
        assert cfg.system_prompt_file == prompt_file.resolve()
        assert cfg.system_prompt_file.read_text() == "Be concise."

    def test_malformed_protected_yaml_does_not_fallthrough(self, tmp_path):
        """Malformed /etc/kai/workspaces.yaml stops loading, doesn't fall to local."""
        from kai.config import _YAML_MALFORMED

        ws = tmp_path / "ws"
        ws.mkdir()
        # Local file exists with valid config
        self._write_yaml(tmp_path, f"workspaces:\n  - path: {ws}\n")
        # But the protected file is malformed
        with (
            patch("kai.config._read_protected_yaml", return_value=_YAML_MALFORMED),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_workspace_configs()
        # Should NOT have loaded the local file
        assert configs == {}

    def test_float_timeout_rejected(self, tmp_path):
        """Non-integer float timeout (e.g. 3.7) is rejected."""
        ws = tmp_path / "ws"
        ws.mkdir()
        self._write_yaml(
            tmp_path,
            f"""\
            workspaces:
              - path: {ws}
                claude:
                  timeout: 3.7
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_workspace_configs()
        assert configs == {}

    def test_bool_timeout_rejected(self, tmp_path):
        """Boolean timeout (e.g. true) is rejected, not silently cast to 1."""
        ws = tmp_path / "ws"
        ws.mkdir()
        self._write_yaml(
            tmp_path,
            f"""\
            workspaces:
              - path: {ws}
                claude:
                  timeout: true
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_workspace_configs()
        assert configs == {}

    def test_integer_like_float_timeout_accepted(self, tmp_path):
        """Integer-like float (e.g. 300.0 from YAML) is accepted."""
        ws = tmp_path / "ws"
        ws.mkdir()
        # Use 300.0 (not 300) to exercise the is_integer() path.
        # YAML parses 300 as int, but 300.0 as float.
        self._write_yaml(
            tmp_path,
            f"""\
            workspaces:
              - path: {ws}
                claude:
                  timeout: 300.0
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_workspace_configs()
        assert configs[ws.resolve()].timeout == 300

    def test_protected_installation_tried_first(self, tmp_path):
        """Protected file (/etc/kai/workspaces.yaml) is tried before local."""
        ws = tmp_path / "ws"
        ws.mkdir()
        # The protected yaml returns a valid config
        protected_data = {"workspaces": [{"path": str(ws), "claude": {"model": "opus"}}]}
        with patch("kai.config._read_protected_yaml", return_value=protected_data):
            configs = _load_workspace_configs()
        assert configs[ws.resolve()].model == "opus"

    def test_minimal_entry_no_claude(self, tmp_path):
        """Entry with only path (no claude section) uses all defaults."""
        ws = tmp_path / "ws"
        ws.mkdir()
        self._write_yaml(tmp_path, f"workspaces:\n  - path: {ws}\n")
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_workspace_configs()
        cfg = configs[ws.resolve()]
        assert cfg.model is None
        assert cfg.budget is None
        assert cfg.timeout is None

    def test_env_dict_values_coerced_to_strings(self, tmp_path):
        """Env dict values (e.g., numbers in YAML) are coerced to strings."""
        ws = tmp_path / "ws"
        ws.mkdir()
        self._write_yaml(
            tmp_path,
            f"""\
            workspaces:
              - path: {ws}
                claude:
                  env:
                    PORT: 5432
                    DEBUG: true
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_workspace_configs()
        env = configs[ws.resolve()].env
        # YAML parses `true` as Python bool; coercion emits lowercase
        # to match .env file conventions (avoids "True" vs "true" bugs)
        assert env == {"PORT": "5432", "DEBUG": "true"}

    def test_env_null_values_become_empty_string(self, tmp_path):
        """YAML null/~/empty env values become empty strings, not 'None'."""
        ws = tmp_path / "ws"
        ws.mkdir()
        self._write_yaml(
            tmp_path,
            f"""\
            workspaces:
              - path: {ws}
                claude:
                  env:
                    EMPTY_VAR:
                    NULL_VAR: null
                    TILDE_VAR: ~
                    REAL_VAR: hello
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_workspace_configs()
        env = configs[ws.resolve()].env
        assert env["EMPTY_VAR"] == ""
        assert env["NULL_VAR"] == ""
        assert env["TILDE_VAR"] == ""
        assert env["REAL_VAR"] == "hello"

    def test_inline_system_prompt(self, tmp_path):
        """Inline system_prompt is stored as a string."""
        ws = tmp_path / "ws"
        ws.mkdir()
        self._write_yaml(
            tmp_path,
            f"""\
            workspaces:
              - path: {ws}
                claude:
                  system_prompt: |
                    This is a Rails app.
                    Use RSpec for tests.
            """,
        )
        with (
            patch("kai.config._read_protected_yaml", return_value=None),
            patch("kai.config.PROJECT_ROOT", tmp_path),
        ):
            configs = _load_workspace_configs()
        assert "Rails app" in configs[ws.resolve()].system_prompt


# ── Config.get_workspace_config ─────────────────────────────────────


class TestGetWorkspaceConfig:
    def test_found(self, tmp_path):
        """Returns WorkspaceConfig when path matches."""
        from kai.config import Config

        ws = tmp_path / "ws"
        ws.mkdir()
        ws_config = WorkspaceConfig(path=ws.resolve(), model="opus")
        config = Config(
            telegram_bot_token="test",
            allowed_user_ids={1},
            workspace_configs={ws.resolve(): ws_config},
        )
        assert config.get_workspace_config(ws) is ws_config

    def test_not_found(self, tmp_path):
        """Returns None when path has no config."""
        from kai.config import Config

        config = Config(
            telegram_bot_token="test",
            allowed_user_ids={1},
        )
        assert config.get_workspace_config(tmp_path / "unknown") is None
