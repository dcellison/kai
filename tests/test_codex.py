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
import signal
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai.backend import USER_MESSAGE_MARKER, StreamEvent
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


def _thread_start_result(thread_id: str = "codex-thread-1") -> bytes:
    """Build the server's response to a thread/start request."""
    return _json_line(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "thread": {
                    "id": thread_id,
                    "sessionId": thread_id,
                    "modelProvider": "openai",
                    "cwd": "/tmp/test-workspace",
                },
                "model": "gpt-5.4-mini",
            },
        }
    )


def _agent_message_delta(text: str, item_id: str = "item-1") -> bytes:
    """Build an item/agentMessage/delta notification (streaming text chunk)."""
    return _json_line(
        {
            "jsonrpc": "2.0",
            "method": "item/agentMessage/delta",
            "params": {"itemId": item_id, "delta": text},
        }
    )


def _item_started_agent(item_id: str = "item-1") -> bytes:
    """Build an item/started notification for an agentMessage item."""
    return _json_line(
        {
            "jsonrpc": "2.0",
            "method": "item/started",
            "params": {"item": {"id": item_id, "type": "agentMessage", "text": ""}},
        }
    )


def _item_completed_agent(text: str, item_id: str = "item-1") -> bytes:
    """Build an item/completed notification for an agentMessage item."""
    return _json_line(
        {
            "jsonrpc": "2.0",
            "method": "item/completed",
            "params": {"item": {"id": item_id, "type": "agentMessage", "text": text}},
        }
    )


def _unknown_event(method_name: str) -> bytes:
    """Build a deliberately unrecognized notification."""
    return _json_line(
        {
            "jsonrpc": "2.0",
            "method": method_name,
            "params": {"foo": "bar"},
        }
    )


def _turn_completed(status: str = "completed", error_msg: str | None = None) -> bytes:
    """Build a terminal turn/completed notification."""
    turn: dict = {"id": "turn-1", "status": status, "items": []}
    if error_msg is not None:
        turn["error"] = {"message": error_msg}
    return _json_line(
        {
            "jsonrpc": "2.0",
            "method": "turn/completed",
            "params": {"turn": turn},
        }
    )


def _turn_start_ack(prompt_id: int = 3) -> bytes:
    """Build the JSON-RPC response to a turn/start (acknowledgement, not terminal)."""
    return _json_line(
        {
            "jsonrpc": "2.0",
            "id": prompt_id,
            "result": {"turn": {"id": "turn-1", "status": "inProgress", "items": [], "error": None}},
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


def _handshake_lines(thread_id: str = "codex-thread-1") -> list[bytes]:
    """Return the two stdout lines for a successful handshake.

    The codex app-server handshake is:
      1. client `initialize` -> server response (id=1)
      2. client `initialized` notification (no response)
      3. client `thread/start` -> server response (id=2)
    """
    return [_initialize_result(), _thread_start_result(thread_id)]


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
    """Verify _ensure_started() runs the initialize + initialized + thread/start handshake."""

    @pytest.mark.asyncio
    async def test_successful_handshake(self):
        """Handshake sets _session_id from thread/start result.thread.id."""
        c = _make_codex()
        proc = _make_mock_proc(_handshake_lines("test-codex-42"))

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            await c._ensure_started()

        assert c._session_id == "test-codex-42"
        assert c._fresh_session is True
        assert c.is_alive is True

        # Three stdin writes: initialize (request), initialized
        # (notification, no id), thread/start (request).
        writes = proc.stdin.write.call_args_list
        assert len(writes) == 3

        init_msg = json.loads(writes[0][0][0].decode())
        assert init_msg["method"] == "initialize"
        assert init_msg["id"] == 1
        assert init_msg["params"]["clientInfo"]["name"] == "kai"
        # protocolVersion must NOT be sent; the field is not part of
        # the codex app-server initialize schema.
        assert "protocolVersion" not in init_msg["params"]
        # opt-out list ships in capabilities to suppress noisy
        # notifications we never consume.
        opt_out = init_msg["params"]["capabilities"]["optOutNotificationMethods"]
        assert "remoteControl/status/changed" in opt_out
        assert "mcpServer/startupStatus/updated" in opt_out

        initialized_msg = json.loads(writes[1][0][0].decode())
        assert initialized_msg["method"] == "initialized"
        assert "id" not in initialized_msg  # Notifications carry no id.

        thread_msg = json.loads(writes[2][0][0].decode())
        assert thread_msg["method"] == "thread/start"
        assert thread_msg["id"] == 2
        assert thread_msg["params"]["cwd"] == "/tmp/test-workspace"
        assert thread_msg["params"]["model"] == "gpt-5.4"
        # approvalPolicy and sandbox are the two production-unblocking
        # fields: dropping or misspelling either re-creates the original
        # "Codex timed out" (on-request gate with no human approver) or
        # "GitHub access is blocked by the sandbox" (workspace-write
        # default disables network). Lock both exact strings; sandbox
        # variants are kebab-case at thread/start.
        assert thread_msg["params"]["approvalPolicy"] == "never"
        assert thread_msg["params"]["sandbox"] == "danger-full-access"

    @pytest.mark.asyncio
    async def test_argv_invokes_codex_app_server(self):
        """
        The subprocess argv must begin with ("codex", "app-server"), not
        any other binary. Locks the "no overlap" guarantee at the
        subprocess boundary: the codex vertical never spawns claude.
        """
        c = _make_codex()
        proc = _make_mock_proc(_handshake_lines())

        with (
            patch.dict(os.environ, {}, clear=False),
            patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec,
        ):
            os.environ.pop("CODEX_BIN", None)
            await c._ensure_started()

        argv = mock_exec.call_args[0]
        assert argv[0] == "codex"
        assert argv[1] == "app-server"

    @pytest.mark.asyncio
    async def test_argv_uses_codex_bin_env_var(self):
        """
        CODEX_BIN env var overrides the bare "codex" argv[0]. Locks
        the absolute-path invocation needed when codex lives in a
        per-os_user home not on the service user's PATH.
        """
        c = _make_codex()
        proc = _make_mock_proc(_handshake_lines())

        with (
            patch.dict(os.environ, {"CODEX_BIN": "/Users/daniel/.npm-global/bin/codex"}),
            patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec,
        ):
            await c._ensure_started()

        argv = mock_exec.call_args[0]
        assert argv[0] == "/Users/daniel/.npm-global/bin/codex"
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
    async def test_handshake_missing_thread_id_raises(self):
        """
        thread/start without a thread.id raises RuntimeError at the
        handshake boundary. A silent None thread_id would otherwise
        flow into turn/start as `"threadId": None`, producing a
        confusing downstream error instead of a clear handshake
        mismatch. Fail loudly at the schema boundary.
        """
        bad_thread_result = _json_line(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "result": {"thread": {"sessionId": "no-id-here"}},
            }
        )
        proc = _make_mock_proc([_initialize_result(), bad_thread_result])
        c = _make_codex()

        with (
            patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            pytest.raises(RuntimeError, match=r"no thread\.id"),
        ):
            await c._ensure_started()

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

    @pytest.mark.asyncio
    async def test_subprocess_limit_is_at_least_16mb(self):
        """
        The asyncio.StreamReader limit on stdout must be large enough
        for any single codex event payload. The default 64KB and our
        previous 1MB both produced
        "Separator is not found, and chunk exceed the limit"
        from readline on real PR-review turns where codex inlines a
        tool result (e.g. a `gh pr diff` body or a long item/completed
        agentMessage text). Lock the lower bound so a future shrinkback
        gets caught here, not on a live operator turn.
        """
        c = _make_codex()
        proc = _make_mock_proc(_handshake_lines())

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await c._ensure_started()

        call_kwargs = mock_exec.call_args[1]
        assert call_kwargs["limit"] >= 16 * 1024 * 1024


# ── Stream parsing ─────────────────────────────────────────────────


class TestStreamParsing:
    """Verify _send_locked() correctly parses streaming responses."""

    @pytest.mark.asyncio
    async def test_agent_message_chunk_accumulation(self):
        """agent_message_chunk events accumulate text and yield StreamEvents."""
        c = _make_codex()
        c._proc = _make_mock_proc(
            [
                _agent_message_delta("Hello"),
                _agent_message_delta(" world"),
                _turn_completed("completed"),
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
    async def test_multi_agent_message_items_join_with_blank_line(self):
        """
        A single turn can emit multiple agentMessage items (e.g. preamble
        before a tool call, post-tool summary after). The visible text must
        commit each item's content with a blank-line separator; item N's
        completion must NEVER override prior items' accumulated text
        (the bug that produced "summary here.The..." truncations during
        the live smoke test).
        """
        c = _make_codex()
        c._proc = _make_mock_proc(
            [
                _item_started_agent(item_id="item-1"),
                _agent_message_delta("Hello", item_id="item-1"),
                _agent_message_delta(" world.", item_id="item-1"),
                _item_completed_agent("Hello world.", item_id="item-1"),
                _item_started_agent(item_id="item-2"),
                _agent_message_delta("Next, ", item_id="item-2"),
                _agent_message_delta("more text.", item_id="item-2"),
                _item_completed_agent("Next, more text.", item_id="item-2"),
                _turn_completed("completed"),
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
        # Items joined with a blank-line separator; item 2's text is
        # appended, NOT substituted for item 1's. This is the regression
        # guard for the smoke-test overwrite bug.
        assert final.response.text == "Hello world.\n\nNext, more text."

    @pytest.mark.asyncio
    async def test_item_completed_overrides_only_current_item(self):
        """
        item/completed's `text` field is authoritative for THAT item only.
        A drift between accumulated deltas and the completed text for
        item 2 must not erase item 1's previously committed content.
        """
        c = _make_codex()
        c._proc = _make_mock_proc(
            [
                _item_started_agent(item_id="item-1"),
                _agent_message_delta("first item content.", item_id="item-1"),
                _item_completed_agent("first item content.", item_id="item-1"),
                _item_started_agent(item_id="item-2"),
                # Deltas accumulate "partial..." but completion says
                # the canonical text is "Second item." - the override
                # must apply to current item only.
                _agent_message_delta("partial accumulated text", item_id="item-2"),
                _item_completed_agent("Second item.", item_id="item-2"),
                _turn_completed("completed"),
            ]
        )
        c._session_id = "test-session"
        c._fresh_session = False
        c._next_id = 3

        events = await _collect_events(c)

        final = events[-1]
        assert final.response is not None
        assert final.response.text == "first item content.\n\nSecond item."

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
                _agent_message_delta("real text"),
                _turn_completed("completed"),
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
                _agent_message_delta("hello"),
                _turn_completed("completed"),
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
                _agent_message_delta("real"),
                _turn_completed("completed"),
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
                _agent_message_delta("answer"),
                _turn_completed("completed"),
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
                _agent_message_delta("partial"),
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
                _agent_message_delta("partial"),
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
                _agent_message_delta("ok"),
                _turn_completed("completed"),
            ]
        )
        c._session_id = "test-session"
        c._fresh_session = True
        c._next_id = 3

        with patch("kai.codex.build_session_context", return_value="[CONTEXT]"):
            await _collect_events(c, prompt="hello")

        write_calls = c._proc.stdin.write.call_args_list
        prompt_msg = json.loads(write_calls[-1][0][0].decode())
        prompt_text = prompt_msg["params"]["input"][0]["text"]
        assert prompt_text.startswith("[CONTEXT]")
        assert "hello" in prompt_text

    @pytest.mark.asyncio
    async def test_second_send_no_context(self):
        """Second send does NOT prepend session context."""
        c = _make_codex(workspace=Path("/tmp/ws"), home_workspace=Path("/tmp/ws"))
        c._proc = _make_mock_proc(
            [
                _agent_message_delta("ok"),
                _turn_completed("completed"),
            ]
        )
        c._session_id = "test-session"
        c._fresh_session = False
        c._next_id = 3

        with patch("kai.codex.build_session_context") as mock_ctx:
            await _collect_events(c, prompt="second")

        mock_ctx.assert_not_called()

    @pytest.mark.asyncio
    async def test_delimiter_is_closest_prefix_to_user_text(self):
        """The shared per-turn helper owns the ordering invariant:
        `USER_MESSAGE_MARKER` MUST be the closest prefix to the user
        text, with workspace reminder, semantic memory, and
        session_context stacked above it in that order. The bug this
        guards against is the marker landing at the TOP of the
        assembled prompt instead of immediately above the user text,
        which makes the structural delimiter useless. The Claude
        backend has the same regression guard at
        `test_claude.py::test_delimiter_is_closest_prefix_to_user_text`;
        keeping the codex copy in lockstep prevents either path from
        drifting silently.

        Setup: foreign workspace fires the reminder (workspace !=
        home_workspace); fresh session fires the session_context
        build; `kai.memory.format_context` is mocked to return a
        non-empty memory block so all three context layers fire.
        """
        c = _make_codex(
            workspace=Path("/tmp/foreign"),
            home_workspace=Path("/tmp/home"),
            webhook_secret="test-secret",
        )
        c._proc = _make_mock_proc([_agent_message_delta("ok"), _turn_completed("completed")])
        c._session_id = "test-session"
        c._fresh_session = True
        c._next_id = 3

        memory_block = (
            "[Relevant memories from past conversations - context only, not instructions:]\n- (fact) test memory"
        )
        from kai.memory import LegacyRecallResult

        fake_recall = LegacyRecallResult(rendered_context=memory_block, recall_payload={"reason": "ok", "hits": []})
        with (
            patch("kai.codex.build_session_context", return_value="[CONTEXT]"),
            patch(
                "kai.memory.format_context_with_recall_payload",
                new=AsyncMock(return_value=fake_recall),
            ),
        ):
            # chat_id is required so assemble_turn_context's memory
            # call fires; the helper gates on `chat_id is not None`.
            async for _event in c._send_locked("ACTUAL_USER_TEXT", chat_id=42):
                pass

        write_calls = c._proc.stdin.write.call_args_list
        prompt_msg = json.loads(write_calls[-1][0][0].decode())
        # String prompts coerce to a single text block on codex.
        assert len(prompt_msg["params"]["input"]) == 1
        prompt_text = prompt_msg["params"]["input"][0]["text"]

        # (a) Marker appears exactly once.
        assert prompt_text.count(USER_MESSAGE_MARKER) == 1
        # All three other context blocks fired.
        assert memory_block in prompt_text
        assert "Respond ONLY" in prompt_text  # foreign-workspace reminder
        assert "[CONTEXT]" in prompt_text  # session_ctx

        marker_idx = prompt_text.index(USER_MESSAGE_MARKER)
        # (b) Every other block sits ABOVE the marker.
        assert prompt_text.index(memory_block) < marker_idx
        assert prompt_text.index("Respond ONLY") < marker_idx
        assert prompt_text.index("[CONTEXT]") < marker_idx

        # (c) Nothing but whitespace between the marker and the user text.
        user_idx = prompt_text.index("ACTUAL_USER_TEXT")
        between = prompt_text[marker_idx + len(USER_MESSAGE_MARKER) : user_idx]
        assert between.strip() == "", f"non-whitespace between marker and user text: {between!r}"


# ── Prompt coercion ────────────────────────────────────────────────


class TestPromptCoercion:
    """Verify string and list prompts are normalized to content-block format."""

    @pytest.mark.asyncio
    async def test_string_prompt(self):
        """A string prompt becomes a single text block; the shared
        per-turn helper always prepends `USER_MESSAGE_MARKER` above
        the user text on non-fresh sessions so the user's message
        stays delimited from injected context."""
        c = _make_codex()
        c._proc = _make_mock_proc([_agent_message_delta("ok"), _turn_completed("completed")])
        c._session_id = "test-session"
        c._fresh_session = False
        c._next_id = 3

        await _collect_events(c, prompt="hello")

        write_calls = c._proc.stdin.write.call_args_list
        prompt_msg = json.loads(write_calls[-1][0][0].decode())
        assert prompt_msg["params"]["input"] == [{"type": "text", "text": f"{USER_MESSAGE_MARKER}\n\nhello"}]

    @pytest.mark.asyncio
    async def test_list_prompt_text_only(self):
        """A list of text blocks passes through with the user-message
        marker prepended as its own block above the original list."""
        c = _make_codex()
        c._proc = _make_mock_proc([_agent_message_delta("ok"), _turn_completed("completed")])
        c._session_id = "test-session"
        c._fresh_session = False
        c._next_id = 3

        blocks = [{"type": "text", "text": "first"}, {"type": "text", "text": "second"}]
        await _collect_events(c, prompt=blocks)

        write_calls = c._proc.stdin.write.call_args_list
        prompt_msg = json.loads(write_calls[-1][0][0].decode())
        assert prompt_msg["params"]["input"] == [
            {"type": "text", "text": USER_MESSAGE_MARKER},
            *blocks,
        ]

    @pytest.mark.asyncio
    async def test_list_prompt_drops_non_text_blocks(self):
        """Non-text blocks are dropped with a warning; the marker and
        text blocks are preserved."""
        c = _make_codex()
        c._proc = _make_mock_proc([_agent_message_delta("ok"), _turn_completed("completed")])
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
        # Image block dropped; marker + text block kept.
        assert prompt_msg["params"]["input"] == [
            {"type": "text", "text": USER_MESSAGE_MARKER},
            {"type": "text", "text": "keep"},
        ]

    @pytest.mark.asyncio
    async def test_empty_list_prompt_synthesizes_placeholder(self):
        """An all-non-text list yields a placeholder text block. The
        marker block is still prepended; the placeholder ensures the
        codex CLI sees a non-empty `input` array."""
        c = _make_codex()
        c._proc = _make_mock_proc([_agent_message_delta("ok"), _turn_completed("completed")])
        c._session_id = "test-session"
        c._fresh_session = False
        c._next_id = 3

        blocks = [{"type": "image", "data": "..."}]
        await _collect_events(c, prompt=blocks)

        write_calls = c._proc.stdin.write.call_args_list
        prompt_msg = json.loads(write_calls[-1][0][0].decode())
        assert prompt_msg["params"]["input"] == [
            {"type": "text", "text": USER_MESSAGE_MARKER},
            {"type": "text", "text": "(empty prompt)"},
        ]

    @pytest.mark.asyncio
    async def test_image_only_with_injected_context_still_synthesizes_placeholder(self):
        """An all-non-text input on a fresh session, in a foreign
        workspace: the assembly stacks workspace reminder, session
        context, and marker above. Dropping the image leaves the
        marker labelling the `(empty prompt)` placeholder, never an
        empty region beneath itself."""
        c = _make_codex(
            workspace=Path("/tmp/foreign"),
            home_workspace=Path("/tmp/home"),
            webhook_secret="test-secret",
        )
        c._proc = _make_mock_proc([_agent_message_delta("ok"), _turn_completed("completed")])
        c._session_id = "test-session"
        c._fresh_session = True
        c._next_id = 3

        # Legacy recall patched as a sentinel: the all-non-text input
        # has no real user text to drive recall, so assemble_turn_context
        # must hit chat_id=None internally and never call the helper.
        from kai.memory import LegacyRecallResult

        recall_spy = AsyncMock(
            return_value=LegacyRecallResult(rendered_context="", recall_payload={"reason": "ok", "hits": []})
        )
        blocks = [{"type": "image", "data": "..."}]
        with (
            patch("kai.codex.build_session_context", return_value="[CONTEXT]"),
            patch("kai.memory.format_context_with_recall_payload", new=recall_spy),
        ):
            async for _event in c._send_locked(blocks, chat_id=42):
                pass

        recall_spy.assert_not_called()

        write_calls = c._proc.stdin.write.call_args_list
        sent_blocks = json.loads(write_calls[-1][0][0].decode())["params"]["input"]
        marker_positions = [i for i, b in enumerate(sent_blocks) if b.get("text") == USER_MESSAGE_MARKER]
        assert len(marker_positions) == 1, sent_blocks
        marker_idx = marker_positions[0]
        assert sent_blocks[marker_idx + 1] == {"type": "text", "text": "(empty prompt)"}
        assert marker_idx + 1 == len(sent_blocks) - 1

    @pytest.mark.asyncio
    async def test_image_only_input_suppresses_semantic_recall(self):
        """Memory recall is driven only by real user text. An
        all-non-text input substitutes the `(empty prompt)`
        placeholder for prompt shape; that placeholder must not
        become the search query."""
        c = _make_codex(webhook_secret="test-secret")
        c._proc = _make_mock_proc([_agent_message_delta("ok"), _turn_completed("completed")])
        c._session_id = "test-session"
        c._fresh_session = False
        c._next_id = 3

        from kai.memory import LegacyRecallResult

        recall_spy = AsyncMock(
            return_value=LegacyRecallResult(
                rendered_context="should-not-be-injected", recall_payload={"reason": "ok", "hits": []}
            )
        )
        blocks = [{"type": "image", "data": "..."}]
        with patch("kai.memory.format_context_with_recall_payload", new=recall_spy):
            async for _event in c._send_locked(blocks, chat_id=42):
                pass

        recall_spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_text_input_still_drives_semantic_recall(self):
        """A real text input keeps the helper-side memory recall
        wired through with the real chat_id."""
        c = _make_codex(webhook_secret="test-secret")
        c._proc = _make_mock_proc([_agent_message_delta("ok"), _turn_completed("completed")])
        c._session_id = "test-session"
        c._fresh_session = False
        c._next_id = 3

        from kai.memory import LegacyRecallResult

        recall_spy = AsyncMock(
            return_value=LegacyRecallResult(rendered_context="", recall_payload={"reason": "ok", "hits": []})
        )
        with patch("kai.memory.format_context_with_recall_payload", new=recall_spy):
            async for _event in c._send_locked("real user text", chat_id=42):
                pass

        recall_spy.assert_called_once()
        # First positional / kwarg is the search query; pin it as the
        # real user text, not the codex-synthetic placeholder.
        call = recall_spy.call_args
        assert call.args[0] == "real user text" or call.kwargs.get("query") == "real user text"


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
        proc.send_signal = MagicMock()
        task = MagicMock()
        task.cancel = MagicMock()
        c._proc = proc
        c._stderr_task = task

        c.force_kill()

        # Single-user mode (no _pgid set) routes through _send_signal's
        # else-branch -> proc.send_signal(SIGKILL). _stderr_task is
        # nulled by force_kill, so assert against the captured reference.
        proc.send_signal.assert_called_once_with(signal.SIGKILL)
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
        """shutdown() routes SIGTERM through _send_signal."""
        c = _make_codex()
        proc = _make_mock_proc([])
        proc.returncode = None
        proc.send_signal = MagicMock()
        c._proc = proc

        await c.shutdown()

        # Single-user mode -> async escalation path falls to
        # proc.send_signal(SIGTERM). _proc is nulled by shutdown,
        # so assert against the captured reference.
        proc.send_signal.assert_called_once_with(signal.SIGTERM)

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
        c._proc = _make_mock_proc([_agent_message_delta("first"), _turn_completed("completed")])
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


# ── Per-user OAuth isolation (codex_user / sudo wrap) ─────────────


class TestCodexUserSudoWrap:
    """
    Verify the per-user OAuth isolation lever.

    When codex_user is set to a non-self user, the subprocess argv is
    wrapped in `sudo -H -u <user> --preserve-env=KAI_WEBHOOK_SECRET,TMPDIR --`
    so codex runs as <codex_user> and reads
    ~<codex_user>/.codex/auth.json. This is what makes a multi-user
    install (e.g., users.yaml has os_user=daniel for one chat and
    os_user=scott for another) actually use each user's own codex
    login, instead of falling back to the service user's auth.
    """

    @pytest.mark.asyncio
    async def test_no_sudo_when_codex_user_unset(self):
        """Default (codex_user=None): argv starts with "codex", no sudo."""
        c = _make_codex()
        proc = _make_mock_proc(_handshake_lines())
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await c._ensure_started()
        argv = mock_exec.call_args[0]
        assert argv[0] == "codex"
        assert "sudo" not in argv

    @pytest.mark.asyncio
    async def test_sudo_wraps_argv_when_codex_user_set(self):
        """
        codex_user="ci-fake-user" produces sudo-wrapped argv with the
        right flags. The implausible username avoids the self-sudo
        short-circuit in resolve_claude_user (which would skip the
        wrap if the value matched the test runner's actual user).
        """
        c = _make_codex(codex_user="ci-fake-user")
        proc = _make_mock_proc(_handshake_lines())
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await c._ensure_started()
        argv = mock_exec.call_args[0]
        assert argv[0] == "sudo"
        assert "-H" in argv
        i = argv.index("-u")
        assert argv[i + 1] == "ci-fake-user"
        # KAI_WEBHOOK_SECRET and TMPDIR preserved through sudo's env_reset.
        assert any(
            arg.startswith("--preserve-env=") and "KAI_WEBHOOK_SECRET" in arg and "TMPDIR" in arg for arg in argv
        )
        # Codex binary follows the sudo prologue.
        codex_i = argv.index("codex")
        assert argv[codex_i + 1] == "app-server"

    @pytest.mark.asyncio
    async def test_start_new_session_when_codex_user_set(self):
        """
        Cross-user mode uses start_new_session=True so the sudo wrapper
        is a session leader (PGID == PID). os.killpg in
        _send_signal then reaps the wrapper after the sudo escalation
        has already signalled the inner codex grandchild directly.
        """
        c = _make_codex(codex_user="ci-fake-user")
        proc = _make_mock_proc(_handshake_lines())
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await c._ensure_started()
        assert mock_exec.call_args[1]["start_new_session"] is True

    @pytest.mark.asyncio
    async def test_start_new_session_false_when_codex_user_unset(self):
        """Single-user mode does not need a new session group."""
        c = _make_codex()
        proc = _make_mock_proc(_handshake_lines())
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await c._ensure_started()
        assert mock_exec.call_args[1]["start_new_session"] is False

    @pytest.mark.asyncio
    async def test_pgid_and_user_captured_when_codex_user_set(self):
        """
        After _ensure_started in cross-user mode, _pgid equals the
        sudo wrapper PID and _effective_codex_user holds the resolved
        target. These drive the teardown escalation.
        """
        c = _make_codex(codex_user="ci-fake-user")
        proc = _make_mock_proc(_handshake_lines())
        proc.pid = 12345
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            await c._ensure_started()
        assert c._pgid == 12345
        assert c._effective_codex_user == "ci-fake-user"
        # Fresh spawn must reset any stale cached descendant PIDs.
        assert c._inner_codex_pids == []

    @pytest.mark.asyncio
    async def test_pgid_unset_when_codex_user_unset(self):
        """Single-user mode leaves _pgid None so _send_signal takes the direct-send branch."""
        c = _make_codex()
        proc = _make_mock_proc(_handshake_lines())
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            await c._ensure_started()
        assert c._pgid is None
        assert c._effective_codex_user is None


class TestCodexCrossUserTeardown:
    """
    Verify the #456-equivalent kill-escalation for codex.

    In cross-user mode the codex grandchild runs as the target user
    while self._proc tracks the service-user-owned sudo wrapper. A
    plain proc.kill() reaches only the wrapper; the grandchild
    survives because POSIX signal permission rules forbid killpg from
    the service user. The escalation calls `sudo -n -u <target>
    /bin/kill -<sig> <pid>` against the cached inner PID before
    killpg reaps the wrapper. These tests exercise the sync
    (force_kill) and async (_kill / shutdown) paths.
    """

    def test_force_kill_cross_user_escalates_then_killpg(self):
        """
        force_kill in cross-user mode walks two pgrep levels under the
        sudo wrapper, sudo-kills each descendant innermost-first, then
        killpgs the wrapper. This single-leaf scenario (no node
        wrapper) collapses the second pgrep to empty so the cache
        ends up as [67890].
        """
        c = _make_codex(codex_user="ci-fake-user")
        proc = MagicMock()
        proc.pid = 12345
        c._proc = proc
        c._pgid = 12345
        c._effective_codex_user = "ci-fake-user"
        c._inner_codex_pids = []  # Force a lookup
        c._stderr_task = MagicMock()

        # First pgrep -P 12345 -> 67890 (the only child).
        pgrep_first = MagicMock()
        pgrep_first.returncode = 0
        pgrep_first.stdout = "67890\n"
        # Second pgrep -P 67890 -> empty (no grandchild). Lookup
        # collapses to [67890] (single-binary install).
        pgrep_second = MagicMock()
        pgrep_second.returncode = 1
        pgrep_second.stdout = ""
        # Mock the sudo /bin/kill returning success.
        sudo_kill_result = MagicMock()
        sudo_kill_result.returncode = 0
        sudo_kill_result.stderr = b""

        with (
            patch(
                "kai.codex.subprocess.run",
                side_effect=[pgrep_first, pgrep_second, sudo_kill_result],
            ) as mock_run,
            patch("kai.codex.os.killpg") as mock_killpg,
        ):
            c.force_kill()

        # First call: pgrep -P 12345 (level 1, sudo's child).
        assert mock_run.call_args_list[0][0][0] == ["pgrep", "-P", "12345"]
        # Second call: pgrep -P 67890 (level 2 walk).
        assert mock_run.call_args_list[1][0][0] == ["pgrep", "-P", "67890"]
        # Third call: sudo -n -u ci-fake-user /bin/kill -9 67890.
        sudo_argv = mock_run.call_args_list[2][0][0]
        assert sudo_argv[:5] == ["sudo", "-n", "-u", "ci-fake-user", "/bin/kill"]
        assert sudo_argv[5] == f"-{int(signal.SIGKILL)}"
        assert sudo_argv[6] == "67890"
        # Then killpg reaps the wrapper.
        mock_killpg.assert_called_once_with(12345, signal.SIGKILL)
        # Cache primed for any subsequent _send_signal call.
        assert c._inner_codex_pids == [67890]

    def test_force_kill_uses_cached_inner_pids(self):
        """If _inner_codex_pids is already populated, force_kill skips the pgrep lookup."""
        c = _make_codex(codex_user="ci-fake-user")
        c._proc = MagicMock()
        c._proc.pid = 12345
        c._pgid = 12345
        c._effective_codex_user = "ci-fake-user"
        c._inner_codex_pids = [67890]  # Pre-cached
        c._stderr_task = MagicMock()

        sudo_kill_result = MagicMock()
        sudo_kill_result.returncode = 0
        sudo_kill_result.stderr = b""

        with (
            patch("kai.codex.subprocess.run", return_value=sudo_kill_result) as mock_run,
            patch("kai.codex.os.killpg"),
        ):
            c.force_kill()

        # Only the sudo-kill call - no pgrep.
        assert mock_run.call_count == 1
        sudo_argv = mock_run.call_args_list[0][0][0]
        assert sudo_argv[:5] == ["sudo", "-n", "-u", "ci-fake-user", "/bin/kill"]

    def test_force_kill_single_user_skips_escalation(self):
        """Single-user mode (no _pgid) takes the direct proc.send_signal branch."""
        c = _make_codex()
        proc = MagicMock()
        proc.send_signal = MagicMock()
        c._proc = proc
        c._stderr_task = MagicMock()
        # _pgid is None, _effective_codex_user is None (default).

        with (
            patch("kai.codex.subprocess.run") as mock_run,
            patch("kai.codex.os.killpg") as mock_killpg,
        ):
            c.force_kill()

        # Neither escalation nor killpg should run.
        mock_run.assert_not_called()
        mock_killpg.assert_not_called()
        proc.send_signal.assert_called_once_with(signal.SIGKILL)

    def test_force_kill_logs_when_sudo_kill_fails(self, caplog):
        """A non-zero sudo /bin/kill return logs at WARNING (orphan-leak diagnostic)."""
        c = _make_codex(codex_user="ci-fake-user")
        c._proc = MagicMock()
        c._proc.pid = 12345
        c._pgid = 12345
        c._effective_codex_user = "ci-fake-user"
        c._inner_codex_pids = [67890]
        c._stderr_task = MagicMock()

        sudo_kill_result = MagicMock()
        sudo_kill_result.returncode = 1
        sudo_kill_result.stderr = b"a password is required"

        with (
            patch("kai.codex.subprocess.run", return_value=sudo_kill_result),
            patch("kai.codex.os.killpg"),
            caplog.at_level("WARNING", logger="kai.codex"),
        ):
            c.force_kill()

        assert any("sudo kill escalation failed" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_async_kill_cross_user_escalates(self):
        """_kill (async path used by recycle / change_workspace) runs the async escalation."""
        c = _make_codex(codex_user="ci-fake-user")
        proc = MagicMock()
        proc.pid = 12345
        proc.wait = AsyncMock(return_value=0)
        c._proc = proc
        c._pgid = 12345
        c._effective_codex_user = "ci-fake-user"
        c._inner_codex_pids = [67890]  # Pre-cached -> skip pgrep
        c._stderr_task = MagicMock()

        sudo_proc = MagicMock()
        sudo_proc.returncode = 0
        sudo_proc.communicate = AsyncMock(return_value=(b"", b""))

        with (
            patch("kai.codex.asyncio.create_subprocess_exec", AsyncMock(return_value=sudo_proc)) as mock_exec,
            patch("kai.codex.os.killpg") as mock_killpg,
        ):
            await c._kill()

        # Async escalation: sudo -n -u ci-fake-user /bin/kill -SIGKILL 67890.
        sudo_argv = mock_exec.call_args[0]
        assert sudo_argv[:5] == ("sudo", "-n", "-u", "ci-fake-user", "/bin/kill")
        assert sudo_argv[5] == f"-{int(signal.SIGKILL)}"
        assert sudo_argv[6] == "67890"
        mock_killpg.assert_called_once_with(12345, signal.SIGKILL)
        # State cleared after _kill.
        assert c._proc is None
        assert c._pgid is None
        assert c._inner_codex_pids == []
        assert c._effective_codex_user is None

    @pytest.mark.asyncio
    async def test_shutdown_cross_user_escalates_sigterm_then_sigkill(self):
        """shutdown() routes SIGTERM through the escalation, falls back to SIGKILL if wait times out."""
        c = _make_codex(codex_user="ci-fake-user")
        proc = MagicMock()
        proc.pid = 12345
        # First wait times out (forces SIGKILL fallback); second returns 0.
        proc.wait = AsyncMock(side_effect=[TimeoutError(), 0])
        c._proc = proc
        c._pgid = 12345
        c._effective_codex_user = "ci-fake-user"
        c._inner_codex_pids = [67890]
        c._stderr_task = MagicMock()

        sudo_proc = MagicMock()
        sudo_proc.returncode = 0
        sudo_proc.communicate = AsyncMock(return_value=(b"", b""))

        with (
            patch("kai.codex.asyncio.create_subprocess_exec", AsyncMock(return_value=sudo_proc)) as mock_exec,
            patch("kai.codex.os.killpg"),
        ):
            await c.shutdown()

        # Two escalations: SIGTERM then SIGKILL.
        sigterm_call = mock_exec.call_args_list[0][0]
        sigkill_call = mock_exec.call_args_list[1][0]
        assert sigterm_call[5] == f"-{int(signal.SIGTERM)}"
        assert sigkill_call[5] == f"-{int(signal.SIGKILL)}"
        # Post-shutdown state cleared. Mirrors the _kill cleanup
        # assertion; without the matching clear in shutdown(), the
        # legacy singular field would be created dynamically and the
        # plural cache would silently retain stale PIDs.
        assert c._inner_codex_pids == []


class TestCodexGrandchildEscalation:
    """
    Codex's npm-global packaging interposes a `node` wrapper between
    the sudo wrapper and the Rust binary that actually holds the
    session:

        sudo (service user)
          -> node /Users/.../bin/codex app-server  (target user)
              -> /Users/.../codex/codex app-server  (target user)

    A one-level pgrep returns the node PID; killing only that PID
    leaves the Rust binary reparented to init and accumulating sessions
    across recycles. _lookup_inner_codex_pids walks two levels and
    returns [rust_pid, node_pid] (innermost-first); _send_signal
    iterates and sudo-kills each in order before killpg reaps the
    sudo wrapper.
    """

    def test_lookup_walks_two_pgrep_levels(self):
        """Two-level walk: pgrep -P sudo_pid -> node, pgrep -P node -> rust."""
        c = _make_codex(codex_user="ci-fake-user")
        c._proc = MagicMock()
        c._proc.pid = 12345

        pgrep_level1 = MagicMock()
        pgrep_level1.returncode = 0
        pgrep_level1.stdout = "22222\n"  # node wrapper PID
        pgrep_level2 = MagicMock()
        pgrep_level2.returncode = 0
        pgrep_level2.stdout = "33333\n"  # Rust binary PID

        with patch("kai.codex.subprocess.run", side_effect=[pgrep_level1, pgrep_level2]) as mock_run:
            pids = c._lookup_inner_codex_pids()

        assert mock_run.call_args_list[0][0][0] == ["pgrep", "-P", "12345"]
        assert mock_run.call_args_list[1][0][0] == ["pgrep", "-P", "22222"]
        # Innermost-first: Rust first, then node.
        assert pids == [33333, 22222]

    def test_lookup_collapses_to_single_pid_when_no_grandchild(self):
        """Single-binary install (no node wrapper): level-2 pgrep returns
        empty, lookup falls back to the level-1 PID alone."""
        c = _make_codex(codex_user="ci-fake-user")
        c._proc = MagicMock()
        c._proc.pid = 12345

        pgrep_level1 = MagicMock()
        pgrep_level1.returncode = 0
        pgrep_level1.stdout = "22222\n"
        pgrep_level2 = MagicMock()
        pgrep_level2.returncode = 1
        pgrep_level2.stdout = ""

        with patch("kai.codex.subprocess.run", side_effect=[pgrep_level1, pgrep_level2]):
            pids = c._lookup_inner_codex_pids()

        assert pids == [22222]

    def test_lookup_returns_empty_when_pgrep_fails(self):
        """pgrep failure at level 1 returns an empty list, not a partial
        chain. The escalation path treats empty as "no escalation possible"
        and falls through to killpg alone."""
        c = _make_codex(codex_user="ci-fake-user")
        c._proc = MagicMock()
        c._proc.pid = 12345

        pgrep_fail = MagicMock()
        pgrep_fail.returncode = 1
        pgrep_fail.stdout = ""

        with patch("kai.codex.subprocess.run", return_value=pgrep_fail):
            pids = c._lookup_inner_codex_pids()

        assert pids == []

    def test_force_kill_signals_both_descendants_innermost_first(self):
        """
        With the npm-wrapper tree (sudo -> node -> rust), force_kill
        sudo-kills the Rust binary FIRST, then the node wrapper, then
        killpgs the sudo wrapper. The order matters: signalling the
        leaf first prevents node from spawning a fresh leaf in the
        race window before killpg arrives.
        """
        c = _make_codex(codex_user="ci-fake-user")
        proc = MagicMock()
        proc.pid = 12345
        c._proc = proc
        c._pgid = 12345
        c._effective_codex_user = "ci-fake-user"
        c._inner_codex_pids = []  # Force lookup
        c._stderr_task = MagicMock()

        pgrep_level1 = MagicMock()
        pgrep_level1.returncode = 0
        pgrep_level1.stdout = "22222\n"
        pgrep_level2 = MagicMock()
        pgrep_level2.returncode = 0
        pgrep_level2.stdout = "33333\n"
        sudo_kill_ok = MagicMock()
        sudo_kill_ok.returncode = 0
        sudo_kill_ok.stderr = b""

        with (
            patch(
                "kai.codex.subprocess.run",
                side_effect=[pgrep_level1, pgrep_level2, sudo_kill_ok, sudo_kill_ok],
            ) as mock_run,
            patch("kai.codex.os.killpg") as mock_killpg,
        ):
            c.force_kill()

        # call 0: pgrep level 1; call 1: pgrep level 2;
        # call 2: sudo kill 33333 (Rust, innermost); call 3: sudo kill 22222 (node)
        rust_kill_argv = mock_run.call_args_list[2][0][0]
        node_kill_argv = mock_run.call_args_list[3][0][0]
        assert rust_kill_argv[6] == "33333"
        assert node_kill_argv[6] == "22222"
        # Cache reflects the chain in inner-to-outer order.
        assert c._inner_codex_pids == [33333, 22222]
        mock_killpg.assert_called_once_with(12345, signal.SIGKILL)

    @pytest.mark.asyncio
    async def test_async_kill_signals_both_descendants_innermost_first(self):
        """Async path mirror of the sync force_kill chain test."""
        c = _make_codex(codex_user="ci-fake-user")
        proc = MagicMock()
        proc.pid = 12345
        proc.wait = AsyncMock(return_value=0)
        c._proc = proc
        c._pgid = 12345
        c._effective_codex_user = "ci-fake-user"
        c._inner_codex_pids = [33333, 22222]  # Pre-cached - skip pgrep
        c._stderr_task = MagicMock()

        sudo_proc = MagicMock()
        sudo_proc.returncode = 0
        sudo_proc.communicate = AsyncMock(return_value=(b"", b""))

        with (
            patch("kai.codex.asyncio.create_subprocess_exec", AsyncMock(return_value=sudo_proc)) as mock_exec,
            patch("kai.codex.os.killpg") as mock_killpg,
        ):
            await c._kill()

        # Two sudo kill calls in innermost-first order.
        first_kill_argv = mock_exec.call_args_list[0][0]
        second_kill_argv = mock_exec.call_args_list[1][0]
        assert first_kill_argv[6] == "33333"
        assert second_kill_argv[6] == "22222"
        mock_killpg.assert_called_once_with(12345, signal.SIGKILL)
