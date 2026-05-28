"""
Tests for opencode.py OpenCode ACP subprocess backend.

Covers OpenCode-specific hook implementations on top of the shared
AcpBackend layer:

1. backend_name / backend_label class attributes
2. Static argv `opencode acp` (no --model flag; flag is not accepted)
3. build_env injects OPENCODE_CONFIG_CONTENT with the active model
4. Integer protocolVersion in build_initialize_params (NOT string "v1")
5. session/new params: cwd + empty mcpServers (same as Goose)
6. extract_text_delta parses agent_message_chunk (skips agent_thought_chunk,
   available_commands_update, usage_update, tool_call, tool_call_update)
7. handle_server_request auto-approves session/request_permission with
   allow_always (falls back through allow_once and the first option;
   returns None when no usable option is found)
8. Other server-request methods return None (no auto-response)
"""

import json
from pathlib import Path

from kai.opencode import OpenCodeBackend


def _make_opencode(**kwargs) -> OpenCodeBackend:
    """Create an OpenCodeBackend with sensible defaults for testing."""
    defaults = {
        "model": "anthropic/claude-sonnet-4-6",
        "workspace": Path("/tmp/test-workspace"),
        "timeout_seconds": 30,
    }
    defaults.update(kwargs)
    return OpenCodeBackend(**defaults)


# ── Class attributes ────────────────────────────────────────────────


class TestClassAttributes:
    """Pin backend_name and backend_label."""

    def test_backend_name(self):
        """backend_name is 'opencode' (config-key match in VALID_BACKENDS)."""
        assert OpenCodeBackend.backend_name == "opencode"

    def test_backend_label(self):
        """backend_label is the human-readable 'OpenCode' string."""
        assert OpenCodeBackend.backend_label == "OpenCode"


# ── build_argv ──────────────────────────────────────────────────────


class TestBuildArgv:
    """The CLI does not accept --model on argv; model flows through env."""

    def test_static_argv_no_model_flag(self):
        """argv is exactly `opencode acp` regardless of self.model."""
        b = _make_opencode(model="anthropic/claude-sonnet-4-6")
        assert b.build_argv() == ["opencode", "acp"]

    def test_argv_unchanged_with_empty_model(self):
        """Empty model still produces the same argv (model goes via env)."""
        b = _make_opencode(model="")
        assert b.build_argv() == ["opencode", "acp"]


# ── build_env ───────────────────────────────────────────────────────


class TestBuildEnv:
    """Verify OPENCODE_CONFIG_CONTENT injection."""

    def test_model_written_into_inline_config(self):
        """Active model lands in OPENCODE_CONFIG_CONTENT as inline JSON."""
        b = _make_opencode(model="anthropic/claude-sonnet-4-6")
        env = b.build_env({})
        assert "OPENCODE_CONFIG_CONTENT" in env
        parsed = json.loads(env["OPENCODE_CONFIG_CONTENT"])
        assert parsed == {"model": "anthropic/claude-sonnet-4-6"}

    def test_empty_model_omits_inline_config(self):
        """Empty model = no OPENCODE_CONFIG_CONTENT; OpenCode falls back to its own config files."""
        b = _make_opencode(model="")
        env = b.build_env({})
        assert "OPENCODE_CONFIG_CONTENT" not in env

    def test_base_env_preserved(self):
        """Caller-supplied env keys flow through unchanged."""
        b = _make_opencode(model="x/y")
        env = b.build_env({"FOO": "bar", "BAZ": "qux"})
        assert env["FOO"] == "bar"
        assert env["BAZ"] == "qux"
        assert "OPENCODE_CONFIG_CONTENT" in env


# ── build_initialize_params ─────────────────────────────────────────


class TestBuildInitializeParams:
    """Verify the integer protocolVersion override (OpenCode rejects string)."""

    def test_protocol_version_is_integer(self):
        """OpenCode's Zod schema requires a number; string 'v1' is rejected."""
        b = _make_opencode()
        params = b.build_initialize_params()
        assert params["protocolVersion"] == 1
        assert isinstance(params["protocolVersion"], int)

    def test_client_info_identifies_kai(self):
        """clientInfo.name is 'kai' so OpenCode logs identify the caller."""
        b = _make_opencode()
        params = b.build_initialize_params()
        assert params["clientInfo"]["name"] == "kai"
        assert "version" in params["clientInfo"]


# ── build_session_new_params ────────────────────────────────────────


class TestBuildSessionNewParams:
    """Verify session/new payload shape (cwd + empty mcpServers)."""

    def test_session_new_params(self):
        """OpenCode accepts the same shape Goose uses; cwd is workspace, mcpServers is []."""
        ws = Path("/tmp/some-workspace")
        b = _make_opencode(workspace=ws)
        params = b.build_session_new_params()
        assert params == {"cwd": str(ws), "mcpServers": []}


# ── extract_text_delta ──────────────────────────────────────────────


class TestExtractTextDelta:
    """
    Verify only `agent_message_chunk` text reaches the user.

    Smoke run on opencode 1.15.11 observed these notification shapes:
    agent_message_chunk (keep), agent_thought_chunk (skip),
    available_commands_update (skip), usage_update (skip),
    tool_call (skip), tool_call_update (skip). Pin each one so a
    future OpenCode change can't quietly leak thoughts or tool noise.
    """

    def _notification(self, session_update: str, content_text: str = "x") -> dict:
        return {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "sess-1",
                "update": {
                    "sessionUpdate": session_update,
                    "content": {"type": "text", "text": content_text},
                },
            },
        }

    def test_agent_message_chunk_returns_text(self):
        """agent_message_chunk text is user-visible; hook returns it."""
        b = _make_opencode()
        assert b.extract_text_delta(self._notification("agent_message_chunk", "hello")) == "hello"

    def test_agent_thought_chunk_returns_none(self):
        """agent_thought_chunk is internal reasoning; hook returns None."""
        b = _make_opencode()
        assert b.extract_text_delta(self._notification("agent_thought_chunk", "thinking...")) is None

    def test_available_commands_update_returns_none(self):
        """available_commands_update advertises OpenCode slash commands; not for the user."""
        b = _make_opencode()
        msg = {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "sess-1",
                "update": {"sessionUpdate": "available_commands_update", "availableCommands": []},
            },
        }
        assert b.extract_text_delta(msg) is None

    def test_usage_update_returns_none(self):
        """usage_update carries token telemetry, not user text."""
        b = _make_opencode()
        msg = {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "sess-1",
                "update": {"sessionUpdate": "usage_update", "used": 100, "size": 200000},
            },
        }
        assert b.extract_text_delta(msg) is None

    def test_tool_call_returns_none(self):
        """tool_call notifications announce tool invocations; not user-visible."""
        b = _make_opencode()
        msg = {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "sess-1",
                "update": {"sessionUpdate": "tool_call", "toolCallId": "tc-1", "title": "read"},
            },
        }
        assert b.extract_text_delta(msg) is None

    def test_tool_call_update_returns_none(self):
        """tool_call_update reports tool progress; not user-visible."""
        b = _make_opencode()
        msg = {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": "sess-1",
                "update": {"sessionUpdate": "tool_call_update", "toolCallId": "tc-1", "status": "in_progress"},
            },
        }
        assert b.extract_text_delta(msg) is None

    def test_empty_text_returns_none(self):
        """An agent_message_chunk with empty text is treated as no-op."""
        b = _make_opencode()
        assert b.extract_text_delta(self._notification("agent_message_chunk", "")) is None

    def test_non_session_update_method_returns_none(self):
        """Methods other than session/update are not text deltas."""
        b = _make_opencode()
        msg = {"jsonrpc": "2.0", "method": "session/something_else", "params": {}}
        assert b.extract_text_delta(msg) is None


# ── handle_server_request ───────────────────────────────────────────


class TestHandleServerRequest:
    """
    Verify session/request_permission auto-approval.

    OpenCode emits server-initiated session/request_permission with
    options[*].kind in {allow_once, allow_always, reject_once}. The
    adapter auto-approves (returns the allow_always optionId) so tool
    calls do not hang waiting for an interactive answer.
    """

    def _permission_request(self, options: list[dict]) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": 99,
            "method": "session/request_permission",
            "params": {"sessionId": "sess-1", "options": options},
        }

    def test_auto_approve_always_when_available(self):
        """Returns the allow_always optionId."""
        b = _make_opencode()
        msg = self._permission_request(
            [
                {"optionId": "once", "kind": "allow_once"},
                {"optionId": "always", "kind": "allow_always"},
                {"optionId": "reject", "kind": "reject_once"},
            ]
        )
        assert b.handle_server_request(msg) == {"optionId": "always"}

    def test_falls_back_to_allow_once_when_no_allow_always(self):
        """If allow_always is missing, return the allow_once option."""
        b = _make_opencode()
        msg = self._permission_request(
            [
                {"optionId": "once", "kind": "allow_once"},
                {"optionId": "reject", "kind": "reject_once"},
            ]
        )
        assert b.handle_server_request(msg) == {"optionId": "once"}

    def test_falls_back_to_first_option_when_no_allow_kinds(self):
        """If neither allow_always nor allow_once exist, take the first option as a last resort."""
        b = _make_opencode()
        msg = self._permission_request(
            [
                {"optionId": "weird", "kind": "ask_user"},
                {"optionId": "reject", "kind": "reject_once"},
            ]
        )
        assert b.handle_server_request(msg) == {"optionId": "weird"}

    def test_empty_options_returns_none(self):
        """No options at all = nothing to pick; return None so the loop logs and skips."""
        b = _make_opencode()
        msg = self._permission_request([])
        assert b.handle_server_request(msg) is None

    def test_option_without_optionId_returns_none(self):
        """Malformed option (missing optionId) = bail; never send a half-formed response."""
        b = _make_opencode()
        msg = self._permission_request([{"kind": "allow_always"}])
        # First (and only) candidate is allow_always but has no optionId.
        assert b.handle_server_request(msg) is None

    def test_non_permission_method_returns_none(self):
        """Other server-request methods (if any) are not auto-handled."""
        b = _make_opencode()
        msg = {"jsonrpc": "2.0", "id": 7, "method": "session/something_new", "params": {}}
        assert b.handle_server_request(msg) is None


# ── Constructor (smoke) ─────────────────────────────────────────────


class TestConstructor:
    """OpenCodeBackend inherits all AcpBackend init logic; pin the smoke."""

    def test_attributes_set_from_kwargs(self):
        """ABC-required attributes flow through unchanged."""
        b = _make_opencode(
            model="openai/gpt-5.5",
            workspace=Path("/tmp/abc"),
            timeout_seconds=120,
        )
        assert b.model == "openai/gpt-5.5"
        assert b.workspace == Path("/tmp/abc")
        assert b.timeout_seconds == 120
        # Lifecycle defaults inherited from AcpBackend.
        assert b._proc is None
        assert b._session_id is None
        assert b._fresh_session is True
