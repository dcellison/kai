"""
Tests for opencode.py OpenCode ACP subprocess backend.

Covers OpenCode-specific hook implementations on top of the shared
AcpBackend layer:

1. backend_name / backend_label class attributes
2. Argv `opencode acp` with OPENCODE_BIN binary pinning (no --model
   flag; flag is not accepted)
3. build_env injects OPENCODE_CONFIG_CONTENT with the active model
4. Integer protocolVersion in build_initialize_params (NOT string "v1")
5. session/new params: cwd + empty mcpServers (same as Goose)
6. extract_text_delta parses agent_message_chunk (skips agent_thought_chunk,
   available_commands_update, usage_update, tool_call, tool_call_update)
7. handle_server_request auto-approves session/request_permission with
   allow_always (falls back through allow_once and the first option;
   returns None when no usable option is found)
8. Other server-request methods return None (no auto-response)
9. preserved_env_vars carries OPENCODE_CONFIG_CONTENT plus the
   provider auth keys through the cross-user sudo wrap
"""

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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


def _json_line(obj: dict) -> bytes:
    """Encode a dict as a JSON line (bytes with trailing newline)."""
    return json.dumps(obj).encode() + b"\n"


def _initialize_result() -> bytes:
    """Build the server's response to an initialize request."""
    return _json_line(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {
                "protocolVersion": 1,
                "agentCapabilities": {
                    "loadSession": True,
                    "promptCapabilities": {"image": True, "audio": False},
                },
            },
        }
    )


def _session_new_result(session_id: str = "ses_test01") -> bytes:
    """Build the server's response to a session/new request."""
    return _json_line(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {"sessionId": session_id},
        }
    )


def _handshake_lines(session_id: str = "ses_test01") -> list[bytes]:
    """Return the two stdout lines for a successful handshake."""
    return [_initialize_result(), _session_new_result(session_id)]


def _make_mock_proc(stdout_lines: list[bytes]) -> MagicMock:
    """
    Build a mock subprocess that yields predefined stdout lines.

    stdout_lines should be a list of bytes, each ending with b"\\n".
    Once the list is exhausted, every further readline call returns
    b"" (EOF), so tests do not need to count exactly how many reads
    the caller performs. Same shape as the test_acp.py / test_goose.py
    helpers; per-file duplication is the deliberate pattern there.
    """
    proc = MagicMock()
    proc.returncode = None
    proc.pid = 12345
    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.drain = AsyncMock()
    proc.stdout = MagicMock()
    queue = list(stdout_lines)

    async def _readline() -> bytes:
        if not queue:
            return b""
        return queue.pop(0)

    proc.stdout.readline = _readline
    proc.stderr = MagicMock()
    proc.stderr.readline = AsyncMock(return_value=b"")
    proc.wait = AsyncMock()
    proc.kill = MagicMock()
    proc.terminate = MagicMock()
    return proc


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
    """The CLI does not accept --model on argv; model flows through
    env. OPENCODE_BIN pins the binary path so a sudo-wrapped per-user
    spawn matches the absolute path the sudoers rule names (mirrors
    codex's CODEX_BIN and goose's GOOSE_BIN contract); bare "opencode"
    remains the default for installs with opencode on PATH."""

    def test_static_argv_no_model_flag(self, monkeypatch):
        """argv is exactly `opencode acp` regardless of self.model."""
        monkeypatch.delenv("OPENCODE_BIN", raising=False)
        b = _make_opencode(model="anthropic/claude-sonnet-4-6")
        assert b.build_argv() == ["opencode", "acp"]

    def test_argv_unchanged_with_empty_model(self, monkeypatch):
        """Empty model still produces the same argv (model goes via env)."""
        monkeypatch.delenv("OPENCODE_BIN", raising=False)
        b = _make_opencode(model="")
        assert b.build_argv() == ["opencode", "acp"]

    def test_opencode_bin_override_pins_absolute_path(self, monkeypatch):
        monkeypatch.setenv("OPENCODE_BIN", "/Users/svc/.local/bin/opencode")
        b = _make_opencode()
        assert b.build_argv() == ["/Users/svc/.local/bin/opencode", "acp"]

    def test_empty_opencode_bin_falls_back_to_bare_name(self, monkeypatch):
        """An empty-string OPENCODE_BIN (unset-but-present in /etc/kai/env)
        must not produce an empty argv head."""
        monkeypatch.setenv("OPENCODE_BIN", "")
        b = _make_opencode()
        assert b.build_argv()[0] == "opencode"


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

    Response payload follows the ACP v1 `RequestPermissionResult`
    contract: result body is `{"outcome": {"outcome": "selected",
    "optionId": <id>}}`. The shared `_send_server_response` wraps
    that under the JSON-RPC `result` key. A bare `{"optionId": ...}`
    payload (the pre-fix shape) is structurally invalid by ACP v1
    and produces silent stalls or `OpenCode process EOF` on the live
    install.
    """

    def _permission_request(self, options: list[dict]) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": 99,
            "method": "session/request_permission",
            "params": {"sessionId": "sess-1", "options": options},
        }

    @staticmethod
    def _selected(option_id: str) -> dict:
        """Build the ACP v1 selected-outcome result body for an option."""
        return {"outcome": {"outcome": "selected", "optionId": option_id}}

    def test_auto_approve_always_when_available(self):
        """Returns the ACP v1 selected-outcome for allow_always."""
        b = _make_opencode()
        msg = self._permission_request(
            [
                {"optionId": "once", "kind": "allow_once"},
                {"optionId": "always", "kind": "allow_always"},
                {"optionId": "reject", "kind": "reject_once"},
            ]
        )
        assert b.handle_server_request(msg) == self._selected("always")

    def test_falls_back_to_allow_once_when_no_allow_always(self):
        """If allow_always is missing, return the allow_once option."""
        b = _make_opencode()
        msg = self._permission_request(
            [
                {"optionId": "once", "kind": "allow_once"},
                {"optionId": "reject", "kind": "reject_once"},
            ]
        )
        assert b.handle_server_request(msg) == self._selected("once")

    def test_falls_back_to_first_option_when_no_allow_kinds(self):
        """If neither allow_always nor allow_once exist, take the first option as a last resort."""
        b = _make_opencode()
        msg = self._permission_request(
            [
                {"optionId": "weird", "kind": "ask_user"},
                {"optionId": "reject", "kind": "reject_once"},
            ]
        )
        assert b.handle_server_request(msg) == self._selected("weird")

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


class TestHandleServerRequestLoggingAndDiagnostics:
    """
    Pin the per-request INFO diagnostic.

    Every handled session/request_permission emits one INFO line
    naming the chosen optionId and a rationale tag so production
    logs surface what the auto-approver decided. The rationale tag
    is the load-bearing signal for the next time the ACP protocol
    drifts; without it, an option-discrimination break would only
    manifest as silent stalls.
    """

    @staticmethod
    def _msg(options: list[dict]) -> dict:
        return {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "session/request_permission",
            "params": {"sessionId": "sess-1", "options": options},
        }

    def test_logs_allow_always_rationale(self, caplog):
        b = _make_opencode()
        with caplog.at_level(logging.INFO, logger="kai.opencode"):
            b.handle_server_request(
                self._msg(
                    [
                        {"optionId": "once", "kind": "allow_once"},
                        {"optionId": "always", "kind": "allow_always"},
                    ]
                )
            )
        assert any("optionId=always" in r.message and "rationale=allow_always" in r.message for r in caplog.records)

    def test_logs_allow_once_rationale(self, caplog):
        b = _make_opencode()
        with caplog.at_level(logging.INFO, logger="kai.opencode"):
            b.handle_server_request(self._msg([{"optionId": "once", "kind": "allow_once"}]))
        assert any("optionId=once" in r.message and "rationale=allow_once" in r.message for r in caplog.records)

    def test_logs_fallback_first_option_rationale(self, caplog):
        b = _make_opencode()
        with caplog.at_level(logging.INFO, logger="kai.opencode"):
            b.handle_server_request(self._msg([{"optionId": "weird", "kind": "ask_user"}]))
        assert any(
            "optionId=weird" in r.message and "rationale=fallback_first_option" in r.message for r in caplog.records
        )

    def test_logs_no_usable_option_rationale_as_warning(self, caplog):
        """Empty options array emits the no_usable_option rationale at WARNING."""
        b = _make_opencode()
        with caplog.at_level(logging.WARNING, logger="kai.opencode"):
            b.handle_server_request(self._msg([]))
        assert any("rationale=no_usable_option" in r.message for r in caplog.records)


class TestHandleServerRequestRejectedShapes:
    """
    Regression guard against silent reverts to broken response shapes.

    The pre-fix payload was `{"optionId": <id>}` (bare optionId under
    result). Other plausible-but-wrong shapes from the round-2 smoke
    candidate list include the discriminator-at-top-level variant
    `{"outcome": "selected", "optionId": <id>}`. Both produced silent
    stalls or `OpenCode process EOF` on the live install. These
    tests pin that the current implementation does NOT regress to
    either shape.
    """

    @staticmethod
    def _selected(option_id: str) -> dict:
        return {"outcome": {"outcome": "selected", "optionId": option_id}}

    def test_response_is_not_bare_optionId(self):
        """Pre-fix payload shape MUST NOT come back."""
        b = _make_opencode()
        msg = {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "session/request_permission",
            "params": {
                "sessionId": "sess-1",
                "options": [{"optionId": "always", "kind": "allow_always"}],
            },
        }
        result = b.handle_server_request(msg)
        assert result != {"optionId": "always"}
        assert result == self._selected("always")

    def test_response_is_not_flat_discriminator(self):
        """Discriminator-at-top-level variant MUST NOT come back either."""
        b = _make_opencode()
        msg = {
            "jsonrpc": "2.0",
            "id": 12,
            "method": "session/request_permission",
            "params": {
                "sessionId": "sess-1",
                "options": [{"optionId": "once", "kind": "allow_once"}],
            },
        }
        result = b.handle_server_request(msg)
        assert result != {"outcome": "selected", "optionId": "once"}
        assert result == self._selected("once")


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


# ── Startup-failure surface ──────────────────────────────────────────


class TestStartupFailureSurface:
    """
    Spec #556 acceptance criterion: a per-user `default_backend: opencode`
    entry on a non-opencode install must produce a chat-visible
    startup-failure StreamEvent (not silent failure, not fall-through
    to claude) when `opencode` is missing from PATH.

    `AcpBackend._send_locked` wraps `_ensure_started` in a try/except
    that catches OSError / RuntimeError / TimeoutError and yields a
    done StreamEvent with `error="<backend_label> startup failed: <exc>"`.
    This is the runtime safety net for the wizard-decoupling change
    that stopped preflighting per-user opencode tooling at install time.
    """

    @pytest.mark.asyncio
    async def test_missing_opencode_binary_yields_startup_failure_event(self):
        """FileNotFoundError from subprocess spawn becomes a done event with chat-visible error."""
        b = _make_opencode()

        async def _raise_filenotfound(*args, **kwargs):
            raise FileNotFoundError(2, "No such file or directory", "opencode")

        events = []
        with patch("kai.acp.asyncio.create_subprocess_exec", side_effect=_raise_filenotfound):
            async for event in b._send_locked("hello"):
                events.append(event)

        assert events, "expected at least one StreamEvent"
        final = events[-1]
        assert final.done is True
        assert final.response is not None
        assert final.response.success is False
        # Error message identifies OpenCode specifically (via backend_label)
        # so the chat reply tells the operator which backend failed.
        assert final.response.error is not None
        assert "OpenCode startup failed:" in final.response.error


# ── Shared free-function tests (used by one-shot reasoner) ───────


class TestExtractOpencodeTextDeltaFreeFunction:
    """`extract_opencode_text_delta` is the module-level free function
    the conversational backend and the one-shot reasoner both call.
    The discriminator-filter rules must stay consistent across both."""

    def test_returns_agent_message_chunk_text(self):
        from kai.opencode import extract_opencode_text_delta

        msg = {
            "method": "session/update",
            "params": {
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": "hello"},
                }
            },
        }
        assert extract_opencode_text_delta(msg) == "hello"

    @pytest.mark.parametrize(
        "kind",
        [
            "agent_thought_chunk",
            "tool_call",
            "tool_call_update",
            "usage_update",
            "available_commands_update",
        ],
    )
    def test_returns_none_for_non_text_shapes(self, kind):
        from kai.opencode import extract_opencode_text_delta

        msg = {
            "method": "session/update",
            "params": {
                "update": {
                    "sessionUpdate": kind,
                    "content": {"type": "text", "text": "MUST-NOT-APPEAR"},
                }
            },
        }
        assert extract_opencode_text_delta(msg) is None

    def test_returns_none_for_non_session_update_method(self):
        from kai.opencode import extract_opencode_text_delta

        msg = {"method": "something/else", "params": {}}
        assert extract_opencode_text_delta(msg) is None


class TestHandleOpencodePermissionRequestFreeFunction:
    """`handle_opencode_permission_request(msg, policy=...)` picks the
    first option matching the policy's priority list and wraps it in
    the ACP v1 nested-outcome shape. The conversational backend uses
    `allow_always`; the one-shot reasoner uses `reject_once`."""

    def test_allow_always_policy_selects_allow_always(self):
        from kai.opencode import handle_opencode_permission_request

        msg = {
            "method": "session/request_permission",
            "params": {
                "options": [
                    {"optionId": "ok-id", "kind": "allow_always"},
                    {"optionId": "no-id", "kind": "reject_once"},
                ]
            },
        }
        result = handle_opencode_permission_request(msg, policy="allow_always")
        assert result == {"outcome": {"outcome": "selected", "optionId": "ok-id"}}

    def test_reject_once_policy_selects_reject_once(self):
        from kai.opencode import handle_opencode_permission_request

        msg = {
            "method": "session/request_permission",
            "params": {
                "options": [
                    {"optionId": "ok-id", "kind": "allow_always"},
                    {"optionId": "no-id", "kind": "reject_once"},
                ]
            },
        }
        result = handle_opencode_permission_request(msg, policy="reject_once")
        assert result == {"outcome": {"outcome": "selected", "optionId": "no-id"}}

    def test_returns_none_for_wrong_method(self):
        from kai.opencode import handle_opencode_permission_request

        msg = {"method": "something/else", "params": {}}
        assert handle_opencode_permission_request(msg, policy="allow_always") is None

    def test_empty_options_returns_none(self):
        from kai.opencode import handle_opencode_permission_request

        msg = {"method": "session/request_permission", "params": {"options": []}}
        assert handle_opencode_permission_request(msg, policy="reject_once") is None

    def test_falls_back_to_first_option_when_no_kind_matches(self):
        """Defensive fallback: if neither the primary nor secondary
        policy kind is present, take the first option regardless. The
        rationale tag (in the INFO log) names this case as
        `fallback_first_option`. Tests do not assert log shape here;
        the behavioral contract is "do not crash, pick something."""
        from kai.opencode import handle_opencode_permission_request

        msg = {
            "method": "session/request_permission",
            "params": {"options": [{"optionId": "x-id", "kind": "unknown_kind"}]},
        }
        result = handle_opencode_permission_request(msg, policy="allow_always")
        assert result == {"outcome": {"outcome": "selected", "optionId": "x-id"}}


class TestConcatOpencodeText:
    """The sentence-boundary whitespace heuristic. Injects exactly
    one space when prev ends in `.`, `!`, or `?` and new begins with
    an uppercase letter; verbatim concat otherwise. Empty inputs
    round-trip so the accumulator's first-chunk join does not
    inject a spurious leading space."""

    def test_sentence_boundary_injects_space(self):
        from kai.opencode import concat_opencode_text

        # The operator-observed example that motivated this fix.
        assert (
            concat_opencode_text("Let me read the rest of the diff.", "Now let me read the test files")
            == "Let me read the rest of the diff. Now let me read the test files"
        )

    def test_question_and_exclamation_also_inject(self):
        from kai.opencode import concat_opencode_text

        assert concat_opencode_text("Ready?", "Then proceed") == "Ready? Then proceed"
        assert concat_opencode_text("Done!", "Next step") == "Done! Next step"

    def test_lowercase_start_does_not_inject(self):
        """Lowercase sentence starts are a known miss; the false-
        positive cost of widening the trigger is higher than the
        rare miss. Pins the heuristic so a future widening is a
        deliberate decision, not a drift."""
        from kai.opencode import concat_opencode_text

        assert concat_opencode_text("First.", "second sentence") == "First.second sentence"

    def test_word_continuation_does_not_inject(self):
        """A chunk split mid-word must NOT inject a space. Without
        this guard the heuristic would damage the operator's view of
        the streamed text in a worse way than the original gap."""
        from kai.opencode import concat_opencode_text

        # No trailing punctuation -> no space injected.
        assert concat_opencode_text("hel", "lo world") == "hello world"
        # Word continuation across a chunk that ends in a letter.
        assert concat_opencode_text("write the implement", "ation") == "write the implementation"

    def test_code_and_numeric_boundaries_do_not_inject(self):
        """The heuristic must preserve verbatim joins for code (open
        paren after a function name), version strings, URLs, and
        markdown headers. Narrow trigger keeps these safe."""
        from kai.opencode import concat_opencode_text

        # Code: function call.
        assert concat_opencode_text("function", "(arg)") == "function(arg)"
        # Numeric: version string fragments. "1." + "2.3" is the
        # canonical case where the dot is NOT a sentence terminator.
        assert concat_opencode_text("1.", "2.3") == "1.2.3"
        # URL: `://` shape.
        assert concat_opencode_text("https", "://example.com") == "https://example.com"
        # Markdown: header start without a space. `#` is not
        # alphabetic, so `isupper()` returns False and the heuristic
        # leaves the join verbatim. The rendered markdown is still
        # correct because the parser treats `##` as a header regardless
        # of the preceding punctuation, so verbatim is the right
        # default here too.
        assert concat_opencode_text("intro.", "## Section") == "intro.## Section"

    def test_empty_inputs_round_trip(self):
        """Empty `prev` (the accumulator's first-chunk state) must
        return `new` verbatim; otherwise the first user-visible
        chunk would gain a leading space. Symmetric for empty `new`."""
        from kai.opencode import concat_opencode_text

        assert concat_opencode_text("", "Hello") == "Hello"
        assert concat_opencode_text("Hello", "") == "Hello"
        assert concat_opencode_text("", "") == ""


class TestCombineTextChunksHook:
    """`combine_text_chunks` is the AcpBackend hook the conversational
    backend's read loop calls. OpenCode overrides it to inject
    sentence-boundary whitespace; Goose (and any other future ACP
    harness) keeps the default verbatim concat so its behavior is
    unaffected by this change."""

    def test_opencode_backend_combines_with_smart_concat(self):
        backend = _make_opencode()
        assert backend.combine_text_chunks("First sentence.", "Second sentence") == "First sentence. Second sentence"

    def test_opencode_backend_verbatim_for_non_sentence_boundary(self):
        backend = _make_opencode()
        assert backend.combine_text_chunks("partial", "continuation") == "partialcontinuation"

    def test_acp_default_is_verbatim_concat(self):
        """The default on `AcpBackend.combine_text_chunks` is
        `prev + new`. Goose inherits this and stays unchanged; this
        guards against an accidental widening of the new hook to
        every backend."""
        from kai.acp import AcpBackend
        from kai.backend import AgentBackend

        # Direct call via the class to bypass any subclass override.
        # Mirrors the pattern existing _FakeAcp tests use to assert
        # default hook behavior without instantiating a concrete
        # AcpBackend subclass.
        class _DefaultAcp(AcpBackend):
            backend_name = "default_test"
            backend_label = "DefaultTest"

            def build_argv(self) -> list[str]:
                return []

            def build_env(self, base_env: dict[str, str]) -> dict[str, str]:
                return base_env

            def build_session_new_params(self) -> dict:
                return {}

            def extract_text_delta(self, msg: dict) -> str | None:
                return None

        assert issubclass(_DefaultAcp, AgentBackend)
        backend = _DefaultAcp(model="m", workspace=Path("/tmp/x"), timeout_seconds=10)
        # Sentence boundary that OpenCode would treat as a join site;
        # the default backend leaves it verbatim.
        assert backend.combine_text_chunks("End.", "Start") == "End.Start"


# ── Cross-user preserve list ────────────────────────────────────────


class TestPreservedEnvVars:
    """The sudo wrap's --preserve-env list. OpenCode delivers model
    selection via OPENCODE_CONFIG_CONTENT (build_env), so the override
    must carry it through env_reset or the wrapped opencode silently
    loses its model selection; the five provider keys cover env-key
    auth, and KAI_WEBHOOK_SECRET / TMPDIR mirror the AcpBackend
    default. The primary auth store (~/.local/share/opencode/auth.json)
    rides the `sudo -H` HOME rewrite and needs no preservation."""

    def test_content_exact(self):
        b = _make_opencode()
        assert b.preserved_env_vars() == (
            "OPENCODE_CONFIG_CONTENT",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "OPENROUTER_API_KEY",
            "DEEPSEEK_API_KEY",
            "KAI_WEBHOOK_SECRET",
            "TMPDIR",
        )

    @pytest.mark.asyncio
    async def test_wrap_argv_carries_opencode_preserve_list(self, monkeypatch):
        """End-to-end through _ensure_started: the opencode wrap's
        --preserve-env CSV is the hook's list, not the base default."""
        monkeypatch.delenv("OPENCODE_BIN", raising=False)
        b = _make_opencode(os_user="oc-user")
        captured: dict = {}

        async def _fake_spawn(*args, **kwargs):
            captured["argv"] = args
            return _make_mock_proc(_handshake_lines())

        with (
            patch("kai.acp.resolve_claude_user", return_value="oc-user"),
            patch("kai.acp.asyncio.create_subprocess_exec", side_effect=_fake_spawn),
        ):
            await b._ensure_started()

        argv = list(captured["argv"])
        assert argv[:4] == ["sudo", "-H", "-u", "oc-user"]
        assert argv[4] == (
            "--preserve-env=OPENCODE_CONFIG_CONTENT,ANTHROPIC_API_KEY,"
            "OPENAI_API_KEY,GOOGLE_API_KEY,OPENROUTER_API_KEY,DEEPSEEK_API_KEY,"
            "KAI_WEBHOOK_SECRET,TMPDIR"
        )
        assert argv[5] == "--"
        assert argv[6:] == ["opencode", "acp"]
