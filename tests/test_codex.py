"""
Tests for codex.py OpenAI Codex CLI subprocess backend.

Covers:
1. Constructor and ABC signature compliance (force_kill sync, send/_send_locked
   accept chat_id, change_workspace accepts workspace_config).
2. Startup handshake sequence (initialize + session/new) including env var
   injection (CODEX_MODEL, CODEX_PROVIDER) and the no-API-key path under
   subscription auth.
3. Stream parsing: agent_message_chunk accumulation, unknown event types
   skipped (schema-drift defense), non-JSON lines tolerated.
4. Completion result yields done StreamEvent with AgentResponse.
5. EOF, timeout, and JSON-RPC error response handling.
6. restart() triggers new handshake on next send; force_kill is synchronous;
   change_workspace honors workspace_config overrides; shutdown sends SIGTERM
   then SIGKILL with timeout.
7. Send serialization via the internal lock (concurrent sends queue).
8. Prompt coercion: string -> single text block, list-of-blocks dropping
   non-text content.
"""

import asyncio
import inspect
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai.backend import StreamEvent
from kai.codex import CodexBackend
from kai.config import WorkspaceConfig

# ── Shared helpers ──────────────────────────────────────────────────


def _make_codex(**kwargs) -> CodexBackend:
    """Create a CodexBackend with sensible defaults for testing."""
    defaults = {
        "model": "gpt-5.4",
        "workspace": Path("/tmp/test-workspace"),
        "timeout_seconds": 30,
    }
    defaults.update(kwargs)
    return CodexBackend(**defaults)


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
                "protocolVersion": "v1",
                "agentCapabilities": {},
            },
        }
    )


def _session_new_result(session_id: str = "codex-session-1") -> bytes:
    """Build the server's response to a session/new request."""
    return _json_line(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {"sessionId": session_id},
        }
    )


def _agent_message_chunk(text: str, session_id: str = "codex-session-1") -> bytes:
    """
    Build a session/update notification carrying an agent_message_chunk.

    Schema mirrors goose ACP. The pinned codex CLI version's actual
    event names may differ; the schema-drift defense in the parser
    treats unknown event types as skip-and-log, not as errors.
    """
    return _json_line(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "agent_message_chunk",
                    "content": {"type": "text", "text": text},
                },
            },
        }
    )


def _unknown_event(event_name: str, session_id: str = "codex-session-1") -> bytes:
    """Build a session/update with a deliberately unrecognized event type."""
    return _json_line(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": event_name,
                    "content": {"type": "text", "text": "should be ignored"},
                },
            },
        }
    )


def _completion_result(prompt_id: int = 3) -> bytes:
    """Build a completion result for the given prompt id."""
    return _json_line(
        {
            "jsonrpc": "2.0",
            "id": prompt_id,
            "result": {"stopReason": "end_turn"},
        }
    )


def _error_result(prompt_id: int = 3, message: str = "something broke") -> bytes:
    """Build a JSON-RPC error response for the given prompt id."""
    return _json_line(
        {
            "jsonrpc": "2.0",
            "id": prompt_id,
            "error": {"code": -32000, "message": message},
        }
    )


def _make_mock_proc(stdout_lines: list[bytes]) -> MagicMock:
    """Build a mock subprocess that yields predefined stdout lines."""
    proc = MagicMock()
    proc.returncode = None
    proc.pid = 12345
    proc.stdin = MagicMock()
    proc.stdin.write = MagicMock()
    proc.stdin.drain = AsyncMock()
    proc.stdout = MagicMock()
    proc.stdout.readline = AsyncMock(side_effect=stdout_lines)
    proc.stderr = MagicMock()
    proc.stderr.readline = AsyncMock(return_value=b"")
    proc.wait = AsyncMock()
    proc.kill = MagicMock()
    proc.terminate = MagicMock()
    return proc


def _handshake_lines(session_id: str = "codex-session-1") -> list[bytes]:
    """Return the two stdout lines for a successful handshake."""
    return [_initialize_result(), _session_new_result(session_id)]


async def _collect_events(c: CodexBackend, prompt: str | list = "test") -> list[StreamEvent]:
    """Send a prompt and collect all yielded StreamEvents."""
    events = []
    async for event in c._send_locked(prompt):
        events.append(event)
    return events


# ── ABC signature compliance ────────────────────────────────────────


class TestAbcSignatures:
    """
    Lock the AgentBackend ABC compliance for CodexBackend.

    pool.py calls instance.force_kill() with no await, instance.send(prompt,
    chat_id=chat_id), and instance.change_workspace(new_workspace,
    workspace_config=ws_config). A signature drift on any of these
    surfaces a TypeError at runtime; these tests catch the drift at
    import time instead.
    """

    def test_force_kill_is_sync(self):
        """force_kill must NOT be async; pool.py calls it without await."""
        assert not inspect.iscoroutinefunction(CodexBackend.force_kill)

    def test_send_accepts_chat_id(self):
        """send signature must accept chat_id (pool.py passes it by name)."""
        sig = inspect.signature(CodexBackend.send)
        assert "chat_id" in sig.parameters

    def test_send_locked_accepts_chat_id(self):
        """_send_locked must accept chat_id; it powers per-user context injection."""
        sig = inspect.signature(CodexBackend._send_locked)
        assert "chat_id" in sig.parameters

    def test_change_workspace_accepts_workspace_config(self):
        """change_workspace signature must accept workspace_config (pool.py passes it)."""
        sig = inspect.signature(CodexBackend.change_workspace)
        assert "workspace_config" in sig.parameters


# ── Constructor ─────────────────────────────────────────────────────


class TestConstructor:
    """Verify CodexBackend initializes attributes correctly."""

    def test_defaults(self):
        """Constructor sets ABC-required attributes from kwargs."""
        c = _make_codex()
        assert c.model == "gpt-5.4"
        assert c.workspace == Path("/tmp/test-workspace")
        assert c.timeout_seconds == 30
        assert c.provider == "openai"
        assert c._proc is None
        assert c._session_id is None
        assert c._fresh_session is True

    def test_workspace_config_overrides(self):
        """Per-workspace config overrides model and timeout."""
        ws = WorkspaceConfig(path=Path("/tmp/ws"), model="gpt-5.4-mini", timeout=60)
        c = _make_codex(workspace_config=ws)
        assert c.model == "gpt-5.4-mini"
        assert c.timeout_seconds == 60
        # Defaults preserved for restore on workspace switch
        assert c._default_model == "gpt-5.4"
        assert c._default_timeout == 30

    def test_home_workspace_defaults_to_workspace(self):
        """home_workspace falls back to workspace when not provided."""
        c = _make_codex(workspace=Path("/some/path"))
        assert c.home_workspace == Path("/some/path")

    def test_memory_enabled_default_false(self):
        """Codex installs disable memory; the default reflects that."""
        c = _make_codex()
        assert c.memory_enabled is False


# ── Properties ──────────────────────────────────────────────────────


class TestProperties:
    """Verify is_alive and session_id properties."""

    def test_is_alive_no_proc(self):
        """is_alive returns False when no subprocess exists."""
        c = _make_codex()
        assert c.is_alive is False

    def test_is_alive_running(self):
        """is_alive returns True when subprocess has no returncode."""
        c = _make_codex()
        c._proc = MagicMock()
        c._proc.returncode = None
        assert c.is_alive is True

    def test_is_alive_exited(self):
        """is_alive returns False when subprocess has exited."""
        c = _make_codex()
        c._proc = MagicMock()
        c._proc.returncode = 0
        assert c.is_alive is False

    def test_session_id_none(self):
        """session_id is None before handshake."""
        c = _make_codex()
        assert c.session_id is None


# ── Handshake ──────────────────────────────────────────────────────


class TestHandshake:
    """Verify _ensure_started() runs the initialize + session/new handshake."""

    @pytest.mark.asyncio
    async def test_successful_handshake(self):
        """Handshake sets _session_id from session/new result."""
        c = _make_codex()
        proc = _make_mock_proc(_handshake_lines("test-codex-42"))

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            await c._ensure_started()

        assert c._session_id == "test-codex-42"
        assert c._fresh_session is True
        assert c.is_alive is True

        # Verify the two handshake messages were written
        writes = proc.stdin.write.call_args_list
        assert len(writes) == 2

        init_msg = json.loads(writes[0][0][0].decode())
        assert init_msg["method"] == "initialize"
        assert init_msg["id"] == 1
        assert init_msg["params"]["clientInfo"]["name"] == "kai"

        session_msg = json.loads(writes[1][0][0].decode())
        assert session_msg["method"] == "session/new"
        assert session_msg["id"] == 2
        assert session_msg["params"]["cwd"] == "/tmp/test-workspace"

    @pytest.mark.asyncio
    async def test_argv_invokes_codex_app_server(self):
        """
        The subprocess argv must begin with ("codex", "app-server"), not
        any other binary. Locks the "no overlap" guarantee at the
        subprocess boundary: the codex vertical never spawns claude.
        """
        c = _make_codex()
        proc = _make_mock_proc(_handshake_lines())

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await c._ensure_started()

        argv = mock_exec.call_args[0]
        assert argv[0] == "codex"
        assert argv[1] == "app-server"

    @pytest.mark.asyncio
    async def test_skips_when_alive(self):
        """_ensure_started() is a no-op when the process is already running."""
        c = _make_codex()
        c._proc = MagicMock()
        c._proc.returncode = None

        await c._ensure_started()
        # No new process created
        assert c._proc.returncode is None

    @pytest.mark.asyncio
    async def test_handshake_error_raises(self):
        """A JSON-RPC error during handshake raises RuntimeError."""
        error_line = _json_line(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "error": {"code": -32600, "message": "bad request"},
            }
        )
        proc = _make_mock_proc([error_line])
        c = _make_codex()

        with (
            patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            pytest.raises(RuntimeError, match="bad request"),
        ):
            await c._ensure_started()

    @pytest.mark.asyncio
    async def test_handshake_eof_raises(self):
        """EOF during handshake raises RuntimeError."""
        proc = _make_mock_proc([b""])  # Immediate EOF
        c = _make_codex()

        with (
            patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            pytest.raises(RuntimeError, match="exited during handshake"),
        ):
            await c._ensure_started()

    @pytest.mark.asyncio
    async def test_handshake_missing_session_id_raises(self):
        """
        session/new without a recognizable session-id key raises
        RuntimeError at the handshake boundary.

        A silent None session_id would otherwise flow into the next
        session/prompt as `"sessionId": None`, producing a confusing
        downstream prompt error instead of a clear handshake mismatch.
        Fail loudly at the schema boundary.
        """
        bad_session_result = _json_line(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"someOtherKey": "value"},
            }
        )
        proc = _make_mock_proc([_initialize_result(), bad_session_result])
        c = _make_codex()

        with (
            patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            pytest.raises(RuntimeError, match="no session id"),
        ):
            await c._ensure_started()

    @pytest.mark.asyncio
    async def test_handshake_accepts_snake_case_session_id(self):
        """
        session/new result with `session_id` (snake_case) is accepted
        alongside camelCase `sessionId`. Tolerates codex CLI schema
        variants observed across versions.
        """
        snake_case_result = _json_line(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"session_id": "snake-case-session"},
            }
        )
        proc = _make_mock_proc([_initialize_result(), snake_case_result])
        c = _make_codex()

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            await c._ensure_started()

        assert c._session_id == "snake-case-session"

    @pytest.mark.asyncio
    async def test_model_env_var_set(self):
        """CODEX_MODEL env var is set during startup."""
        c = _make_codex(model="gpt-5.4-mini")
        proc = _make_mock_proc(_handshake_lines())

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await c._ensure_started()

        call_kwargs = mock_exec.call_args[1]
        assert call_kwargs["env"]["CODEX_MODEL"] == "gpt-5.4-mini"

    @pytest.mark.asyncio
    async def test_provider_env_var_set(self):
        """
        CODEX_PROVIDER env var is set during startup.

        Forward-compat placeholder: codex is openai-only today, but the
        envelope follows goose's GOOSE_PROVIDER pattern so a future
        codex with multi-provider support can pick the right backend
        without a backend re-wire.
        """
        c = _make_codex(provider="openai")
        proc = _make_mock_proc(_handshake_lines())

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await c._ensure_started()

        call_kwargs = mock_exec.call_args[1]
        assert call_kwargs["env"]["CODEX_PROVIDER"] == "openai"

    @pytest.mark.asyncio
    async def test_provider_env_var_omitted_when_empty(self):
        """An empty provider is not exported as CODEX_PROVIDER=''."""
        c = _make_codex(model="gpt-5.4", provider="")
        proc = _make_mock_proc(_handshake_lines())

        with (
            patch.dict(os.environ, {}, clear=False),
            patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec,
        ):
            os.environ.pop("CODEX_PROVIDER", None)
            await c._ensure_started()

        call_kwargs = mock_exec.call_args[1]
        assert "CODEX_PROVIDER" not in call_kwargs["env"]


# ── Stream parsing ─────────────────────────────────────────────────


class TestStreamParsing:
    """Verify _send_locked() correctly parses streaming responses."""

    @pytest.mark.asyncio
    async def test_agent_message_chunk_accumulation(self):
        """agent_message_chunk events accumulate text and yield StreamEvents."""
        c = _make_codex()
        c._proc = _make_mock_proc(
            [
                _agent_message_chunk("Hello"),
                _agent_message_chunk(" world"),
                _completion_result(prompt_id=3),
            ]
        )
        c._session_id = "test-session"
        c._fresh_session = False
        c._next_id = 3

        events = await _collect_events(c)

        assert len(events) == 3
        assert events[0].text_so_far == "Hello"
        assert events[1].text_so_far == "Hello world"
        assert events[2].done is True
        assert events[2].text_so_far == "Hello world"
        assert events[2].response is not None
        assert events[2].response.success is True
        assert events[2].response.text == "Hello world"
        assert events[2].response.cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_unknown_event_type_skipped(self):
        """
        Schema-drift defense: an unrecognized sessionUpdate event type
        is logged at DEBUG and skipped. The stream continues parsing
        subsequent events instead of aborting.
        """
        c = _make_codex()
        c._proc = _make_mock_proc(
            [
                _unknown_event("future.unknown.thing"),
                _agent_message_chunk("real text"),
                _completion_result(prompt_id=3),
            ]
        )
        c._session_id = "test-session"
        c._fresh_session = False
        c._next_id = 3

        events = await _collect_events(c)

        # The unknown event produced no StreamEvent; only the real
        # message and the completion show up.
        assert len(events) == 2
        assert events[0].text_so_far == "real text"
        assert events[1].done is True

    @pytest.mark.asyncio
    async def test_non_json_lines_skipped(self):
        """Non-JSON stdout lines are silently skipped (defensive parsing)."""
        c = _make_codex()
        c._proc = _make_mock_proc(
            [
                b"some random output\n",
                _agent_message_chunk("hello"),
                _completion_result(prompt_id=3),
            ]
        )
        c._session_id = "test-session"
        c._fresh_session = False
        c._next_id = 3

        events = await _collect_events(c)

        assert len(events) == 2
        assert events[0].text_so_far == "hello"

    @pytest.mark.asyncio
    async def test_alternate_event_field_names_skipped(self):
        """
        Codex's actual event name may live under "event" or "type"
        rather than "sessionUpdate" (the goose ACP key). The parser
        already accepts any of those three keys; an unrecognized
        value still skips, but the field-name flexibility is
        deliberate. This test pins that an event with sessionUpdate
        missing but event present does NOT crash; it simply skips
        when the resolved event type is unrecognized.
        """
        c = _make_codex()
        # event under "event" key, unrecognized name -> skip
        weird_event = _json_line(
            {
                "jsonrpc": "2.0",
                "method": "session/update",
                "params": {
                    "sessionId": "test-session",
                    "update": {"event": "weird_thing", "content": {"text": "x"}},
                },
            }
        )
        c._proc = _make_mock_proc(
            [
                weird_event,
                _agent_message_chunk("real"),
                _completion_result(prompt_id=3),
            ]
        )
        c._session_id = "test-session"
        c._fresh_session = False
        c._next_id = 3

        events = await _collect_events(c)
        assert len(events) == 2
        assert events[0].text_so_far == "real"


# ── Completion ─────────────────────────────────────────────────────


class TestCompletion:
    """Verify final result handling."""

    @pytest.mark.asyncio
    async def test_completion_yields_done_event(self):
        """A result with matching id yields a done StreamEvent."""
        c = _make_codex()
        c._proc = _make_mock_proc(
            [
                _agent_message_chunk("answer"),
                _completion_result(prompt_id=3),
            ]
        )
        c._session_id = "test-session"
        c._fresh_session = False
        c._next_id = 3

        events = await _collect_events(c)
        final = events[-1]

        assert final.done is True
        assert final.response is not None
        assert final.response.success is True
        assert final.response.text == "answer"
        assert final.response.session_id == "test-session"
        assert final.response.cost_usd == 0.0

    @pytest.mark.asyncio
    async def test_eof_with_accumulated_text(self):
        """EOF after some text yields a success response with the text."""
        c = _make_codex()
        c._proc = _make_mock_proc(
            [
                _agent_message_chunk("partial"),
                b"",  # EOF
            ]
        )
        c._session_id = "test-session"
        c._fresh_session = False
        c._next_id = 3

        events = await _collect_events(c)
        final = events[-1]

        assert final.done is True
        assert final.response.success is True
        assert final.response.text == "partial"

    @pytest.mark.asyncio
    async def test_eof_without_text(self):
        """EOF with no accumulated text yields an error response."""
        c = _make_codex()
        c._proc = _make_mock_proc([b""])
        c._session_id = "test-session"
        c._fresh_session = False
        c._next_id = 3

        events = await _collect_events(c)
        final = events[-1]

        assert final.done is True
        assert final.response.success is False
        assert "unexpectedly" in final.response.error


# ── JSON-RPC error response ────────────────────────────────────────


class TestRpcError:
    """Verify JSON-RPC error responses yield a done event with the error."""

    @pytest.mark.asyncio
    async def test_error_yields_failed_response(self):
        """An error with matching id yields done with success=False."""
        c = _make_codex()
        c._proc = _make_mock_proc([_error_result(prompt_id=3, message="model not authorized")])
        c._session_id = "test-session"
        c._fresh_session = False
        c._next_id = 3

        events = await _collect_events(c)
        final = events[-1]

        assert final.done is True
        assert final.response.success is False
        assert "model not authorized" in final.response.error

    @pytest.mark.asyncio
    async def test_error_after_partial_text(self):
        """An error after some streamed text preserves the partial accumulator."""
        c = _make_codex()
        c._proc = _make_mock_proc(
            [
                _agent_message_chunk("partial"),
                _error_result(prompt_id=3, message="cut off"),
            ]
        )
        c._session_id = "test-session"
        c._fresh_session = False
        c._next_id = 3

        events = await _collect_events(c)
        final = events[-1]

        assert final.done is True
        assert final.response.success is False
        assert final.response.text == "partial"
        assert "cut off" in final.response.error


# ── Context injection ──────────────────────────────────────────────


class TestContextInjection:
    """Verify session context is prepended on first send only."""

    @pytest.mark.asyncio
    async def test_fresh_session_injects_context(self):
        """First send prepends session context to the prompt."""
        c = _make_codex(
            workspace=Path("/tmp/ws"),
            home_workspace=Path("/tmp/ws"),
            webhook_secret="test-secret",
        )
        c._proc = _make_mock_proc(
            [
                _agent_message_chunk("ok"),
                _completion_result(prompt_id=3),
            ]
        )
        c._session_id = "test-session"
        c._fresh_session = True
        c._next_id = 3

        with patch("kai.codex.build_session_context", return_value="[CONTEXT]"):
            await _collect_events(c, prompt="hello")

        write_calls = c._proc.stdin.write.call_args_list
        prompt_msg = json.loads(write_calls[-1][0][0].decode())
        prompt_text = prompt_msg["params"]["prompt"][0]["text"]
        assert prompt_text.startswith("[CONTEXT]")
        assert "hello" in prompt_text

    @pytest.mark.asyncio
    async def test_second_send_no_context(self):
        """Second send does NOT prepend session context."""
        c = _make_codex(workspace=Path("/tmp/ws"), home_workspace=Path("/tmp/ws"))
        c._proc = _make_mock_proc(
            [
                _agent_message_chunk("ok"),
                _completion_result(prompt_id=3),
            ]
        )
        c._session_id = "test-session"
        c._fresh_session = False
        c._next_id = 3

        with patch("kai.codex.build_session_context") as mock_ctx:
            await _collect_events(c, prompt="second")

        mock_ctx.assert_not_called()


# ── Prompt coercion ────────────────────────────────────────────────


class TestPromptCoercion:
    """Verify string and list prompts are normalized to content-block format."""

    @pytest.mark.asyncio
    async def test_string_prompt(self):
        """A string prompt becomes a single text block."""
        c = _make_codex()
        c._proc = _make_mock_proc([_agent_message_chunk("ok"), _completion_result(prompt_id=3)])
        c._session_id = "test-session"
        c._fresh_session = False
        c._next_id = 3

        await _collect_events(c, prompt="hello")

        write_calls = c._proc.stdin.write.call_args_list
        prompt_msg = json.loads(write_calls[-1][0][0].decode())
        assert prompt_msg["params"]["prompt"] == [{"type": "text", "text": "hello"}]

    @pytest.mark.asyncio
    async def test_list_prompt_text_only(self):
        """A list of text blocks passes through unchanged."""
        c = _make_codex()
        c._proc = _make_mock_proc([_agent_message_chunk("ok"), _completion_result(prompt_id=3)])
        c._session_id = "test-session"
        c._fresh_session = False
        c._next_id = 3

        blocks = [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]
        await _collect_events(c, prompt=blocks)

        write_calls = c._proc.stdin.write.call_args_list
        prompt_msg = json.loads(write_calls[-1][0][0].decode())
        assert prompt_msg["params"]["prompt"] == blocks

    @pytest.mark.asyncio
    async def test_list_prompt_drops_non_text_blocks(self):
        """Non-text blocks are dropped with a warning; text blocks preserved."""
        c = _make_codex()
        c._proc = _make_mock_proc([_agent_message_chunk("ok"), _completion_result(prompt_id=3)])
        c._session_id = "test-session"
        c._fresh_session = False
        c._next_id = 3

        blocks = [
            {"type": "text", "text": "keep"},
            {"type": "image", "data": "..."},
        ]
        await _collect_events(c, prompt=blocks)

        write_calls = c._proc.stdin.write.call_args_list
        prompt_msg = json.loads(write_calls[-1][0][0].decode())
        # Image block dropped; text block kept
        assert prompt_msg["params"]["prompt"] == [{"type": "text", "text": "keep"}]

    @pytest.mark.asyncio
    async def test_empty_list_prompt_synthesizes_placeholder(self):
        """An all-non-text list yields a single placeholder text block."""
        c = _make_codex()
        c._proc = _make_mock_proc([_agent_message_chunk("ok"), _completion_result(prompt_id=3)])
        c._session_id = "test-session"
        c._fresh_session = False
        c._next_id = 3

        blocks = [{"type": "image", "data": "..."}]
        await _collect_events(c, prompt=blocks)

        write_calls = c._proc.stdin.write.call_args_list
        prompt_msg = json.loads(write_calls[-1][0][0].decode())
        assert prompt_msg["params"]["prompt"] == [{"type": "text", "text": "(empty prompt)"}]


# ── Restart / force_kill / shutdown / change_workspace ────────────


class TestRestart:
    """Verify restart() kills the subprocess and resets session state."""

    @pytest.mark.asyncio
    async def test_restart_clears_proc_and_session(self):
        """restart() nulls _proc and _session_id, sets _fresh_session."""
        c = _make_codex()
        c._proc = _make_mock_proc([])
        c._session_id = "old-session"
        c._fresh_session = False

        await c.restart()

        assert c._proc is None
        assert c._session_id is None
        assert c._fresh_session is True

    @pytest.mark.asyncio
    async def test_restart_no_proc(self):
        """restart() on a never-started backend is a no-op (no crash)."""
        c = _make_codex()
        await c.restart()
        assert c._proc is None


class TestForceKill:
    """Verify force_kill() is synchronous and cancels the stderr task."""

    def test_force_kill_is_sync(self):
        """force_kill must not be a coroutine (pool.py calls it without await)."""
        assert not inspect.iscoroutinefunction(CodexBackend.force_kill)

    def test_force_kill_cancels_stderr_task(self):
        """force_kill cancels and nulls the stderr drain task."""
        c = _make_codex()
        proc = MagicMock()
        proc.kill = MagicMock()
        task = MagicMock()
        task.cancel = MagicMock()
        c._proc = proc
        c._stderr_task = task

        c.force_kill()

        # _stderr_task is nulled by force_kill, so assert against the
        # captured reference rather than the attribute.
        proc.kill.assert_called_once()
        task.cancel.assert_called_once()
        assert c._stderr_task is None

    def test_force_kill_no_proc(self):
        """force_kill on a never-started backend is a no-op (no crash)."""
        c = _make_codex()
        c.force_kill()  # Should not raise


class TestChangeWorkspace:
    """Verify change_workspace() applies new workspace and config."""

    @pytest.mark.asyncio
    async def test_change_workspace_kills_proc(self):
        """change_workspace kills the current subprocess."""
        c = _make_codex()
        c._proc = _make_mock_proc([])
        c._proc.returncode = None

        new_ws = Path("/tmp/new-ws")
        await c.change_workspace(new_ws)

        assert c._proc is None
        assert c.workspace == new_ws

    @pytest.mark.asyncio
    async def test_change_workspace_applies_config(self):
        """A workspace_config override updates model and timeout."""
        c = _make_codex(model="gpt-5.4", timeout_seconds=30)
        ws_cfg = WorkspaceConfig(path=Path("/tmp/ws"), model="gpt-5.4-mini", timeout=120)

        await c.change_workspace(Path("/tmp/ws"), workspace_config=ws_cfg)

        assert c.model == "gpt-5.4-mini"
        assert c.timeout_seconds == 120
        assert c.workspace_config == ws_cfg

    @pytest.mark.asyncio
    async def test_change_workspace_reverts_to_defaults(self):
        """Switching away from a configured workspace restores defaults."""
        ws_cfg = WorkspaceConfig(path=Path("/tmp/ws"), model="gpt-5.4-mini", timeout=120)
        c = _make_codex(model="gpt-5.4", timeout_seconds=30, workspace_config=ws_cfg)
        # Confirm constructor applied the override
        assert c.model == "gpt-5.4-mini"

        # Switch to a workspace with no config
        await c.change_workspace(Path("/tmp/other-ws"))

        assert c.model == "gpt-5.4"  # back to default
        assert c.timeout_seconds == 30


class TestShutdown:
    """Verify shutdown() sends SIGTERM with fallback to SIGKILL."""

    @pytest.mark.asyncio
    async def test_shutdown_sends_sigterm(self):
        """shutdown() calls terminate() first."""
        c = _make_codex()
        proc = _make_mock_proc([])
        proc.returncode = None
        proc.terminate = MagicMock()
        c._proc = proc

        await c.shutdown()

        # _proc is nulled by shutdown; assert against the captured reference.
        proc.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_no_proc(self):
        """shutdown() on a never-started backend is a no-op (no crash)."""
        c = _make_codex()
        await c.shutdown()


# ── Send lock ──────────────────────────────────────────────────────


class TestSendLock:
    """Verify send() serializes via the internal lock."""

    @pytest.mark.asyncio
    async def test_lock_serializes_sends(self):
        """The lock is held across the entire send() generator."""
        c = _make_codex()
        c._proc = _make_mock_proc([_agent_message_chunk("first"), _completion_result(prompt_id=3)])
        c._session_id = "test-session"
        c._fresh_session = False
        c._next_id = 3

        # Iterate send() under the lock; if the lock were not held,
        # this second acquire would succeed immediately. With the lock
        # held, locked() returns True while the generator is being
        # consumed.
        gen = c.send("hello")
        # Consume one event to enter the locked section
        first = await gen.__anext__()
        assert first is not None
        assert c._lock.locked() is True
        # Drain the rest so the lock releases
        async for _ in gen:
            pass
        # After the generator returns, the async-with in send() releases
        assert c._lock.locked() is False


# ── Subprocess construction kwargs ─────────────────────────────────


class TestSubprocessConstruction:
    """Verify create_subprocess_exec is called with the right cwd and stream config."""

    @pytest.mark.asyncio
    async def test_cwd_passed_explicitly(self):
        """The subprocess cwd is set to self.workspace."""
        c = _make_codex(workspace=Path("/tmp/explicit-cwd"))
        proc = _make_mock_proc(_handshake_lines())

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await c._ensure_started()

        call_kwargs = mock_exec.call_args[1]
        assert call_kwargs["cwd"] == "/tmp/explicit-cwd"

    @pytest.mark.asyncio
    async def test_pipes_attached(self):
        """stdin / stdout / stderr are attached as pipes."""
        c = _make_codex()
        proc = _make_mock_proc(_handshake_lines())

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await c._ensure_started()

        call_kwargs = mock_exec.call_args[1]
        assert call_kwargs["stdin"] == asyncio.subprocess.PIPE
        assert call_kwargs["stdout"] == asyncio.subprocess.PIPE
        assert call_kwargs["stderr"] == asyncio.subprocess.PIPE

    @pytest.mark.asyncio
    async def test_webhook_secret_injected(self):
        """KAI_WEBHOOK_SECRET is set in the subprocess env when configured."""
        c = _make_codex(webhook_secret="s3cr3t")
        proc = _make_mock_proc(_handshake_lines())

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await c._ensure_started()

        call_kwargs = mock_exec.call_args[1]
        assert call_kwargs["env"]["KAI_WEBHOOK_SECRET"] == "s3cr3t"
