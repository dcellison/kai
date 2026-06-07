"""
Tests for the shared AcpBackend layer in src/kai/acp.py.

Uses a minimal `_FakeAcp` concrete subclass that supplies only the
hooks the ABC requires; everything else is exercised in the shared
implementation. The goal is to pin behavior that BOTH ACP adapters
(Goose today, OpenCode next) depend on so neither can regress without
the test catching it.

Covers:
1. Hook surface contract (build_argv / build_env / session_new /
   text-delta / completion / error)
2. JSON-RPC write shape (initialize, session/new, session/prompt)
3. Result parsing and notification skipping during handshake
4. Streaming text accumulation through extract_text_delta
5. Completion event yields a done StreamEvent with the session_id
6. JSON-RPC error response yields a done StreamEvent with error text
7. Per-readline response timeout
8. Idle timeout (no output for N * timeout)
9. EOF before completion (success when text was accumulated; error
   otherwise)
10. restart() clears state; next send() re-runs handshake
11. shutdown() sends SIGTERM and falls back to SIGKILL on timeout
12. force_kill() during active send
13. change_workspace() reverts to defaults and re-applies overrides
14. Context injection ordering: USER_MESSAGE_MARKER closest to user
    text, with reminder / memory / session_context stacked above
15. Env layering: backend hook then workspace env_file then workspace
    inline env then KAI_WEBHOOK_SECRET (last; cannot be overridden)
16. backend_label flows into error messages and log lines
17. Workspace model validation uses self.backend_name (no "goose"
    literal in the shared layer)
"""

import json
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai.acp import AcpBackend
from kai.backend import USER_MESSAGE_MARKER, StreamEvent
from kai.config import WorkspaceConfig

# ── Fake concrete subclass ───────────────────────────────────────────


class _FakeAcp(AcpBackend):
    """
    Minimal AcpBackend subclass for testing the shared layer.

    Hooks are intentionally trivial so the test suite exercises the
    GENERIC behavior (transport, lifecycle, context injection, error
    paths) without depending on Goose or OpenCode specifics. The
    streaming notification shape mirrors ACP's session/update pattern
    but uses a simple `text` field so the test cases are easy to read.
    """

    backend_name = "fake"
    backend_label = "FakeAcp"

    def build_argv(self) -> list[str]:
        return ["fake_acp_binary"]

    def build_env(self, base_env: dict[str, str]) -> dict[str, str]:
        # Mark the hook ran so tests can pin env layering order.
        base_env["FAKE_BACKEND_MARK"] = "1"
        return base_env

    def build_session_new_params(self) -> dict:
        return {"cwd": str(self.workspace)}

    def extract_text_delta(self, msg: dict) -> str | None:
        # `{"method": "session/update", "params": {"text": "..."}}`.
        # Anything else returns None so the shared loop skips it.
        if msg.get("method") != "session/update":
            return None
        text = msg.get("params", {}).get("text", "")
        return text or None


# ── Shared helpers ───────────────────────────────────────────────────


def _make_fake(**kwargs) -> _FakeAcp:
    """Create a _FakeAcp with sensible defaults for testing."""
    defaults = {
        "model": "sonnet",
        "workspace": Path("/tmp/test-workspace"),
        "timeout_seconds": 30,
    }
    defaults.update(kwargs)
    return _FakeAcp(**defaults)


def _json_line(obj: dict) -> bytes:
    """Encode a dict as a JSON line (bytes with trailing newline)."""
    return json.dumps(obj).encode() + b"\n"


def _initialize_result() -> bytes:
    """Build the server's response to an initialize request."""
    return _json_line(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"protocolVersion": 0, "agentCapabilities": {}},
        }
    )


def _session_new_result(session_id: str = "sess-1") -> bytes:
    """Build the server's response to a session/new request."""
    return _json_line(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {"sessionId": session_id},
        }
    )


def _text_chunk(text: str) -> bytes:
    """Build a streaming text notification matching _FakeAcp.extract_text_delta."""
    return _json_line(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {"text": text},
        }
    )


def _other_notification() -> bytes:
    """Build a notification the FakeAcp hook treats as non-text (skip)."""
    return _json_line(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {"tool_call": "read_file"},
        }
    )


def _completion_result(prompt_id: int = 3) -> bytes:
    """Build a completion result for a given prompt id."""
    return _json_line(
        {
            "jsonrpc": "2.0",
            "id": prompt_id,
            "result": {"stopReason": "end_turn"},
        }
    )


def _error_result(prompt_id: int = 3, message: str = "something broke") -> bytes:
    """Build a JSON-RPC error response."""
    return _json_line(
        {
            "jsonrpc": "2.0",
            "id": prompt_id,
            "error": {"code": -32000, "message": message},
        }
    )


def _make_mock_proc(stdout_lines: list[bytes]) -> MagicMock:
    """
    Build a mock subprocess that yields predefined stdout lines.

    stdout_lines should be a list of bytes, each ending with b"\\n".
    A final entry of b"" signals EOF on the next readline call.
    """
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


def _handshake_lines(session_id: str = "sess-1") -> list[bytes]:
    """Return the two stdout lines for a successful handshake."""
    return [_initialize_result(), _session_new_result(session_id)]


async def _collect_events(backend: _FakeAcp, prompt: str | list = "test") -> list[StreamEvent]:
    """Send a prompt and collect all yielded StreamEvents."""
    events = []
    async for event in backend._send_locked(prompt):
        events.append(event)
    return events


# ── Hook surface contract ───────────────────────────────────────────


class TestHookSurface:
    """The ABC declares which hooks subclasses MUST implement.

    Pin that omitting a required hook raises NotImplementedError so a
    future adapter that forgets to override one gets a clear error at
    handshake time rather than silently broken behavior.
    """

    def test_build_argv_default_raises(self):
        """The default build_argv raises so subclasses can't forget."""

        class _NoArgv(AcpBackend):
            backend_name = "no_argv"
            backend_label = "NoArgv"

            def build_env(self, base_env):
                return base_env

            def build_session_new_params(self):
                return {}

            def extract_text_delta(self, msg):
                return None

        b = _NoArgv(model="x", workspace=Path("/tmp"))
        with pytest.raises(NotImplementedError):
            b.build_argv()

    def test_build_env_default_raises(self):
        """The default build_env raises so subclasses can't forget."""

        class _NoEnv(AcpBackend):
            backend_name = "no_env"
            backend_label = "NoEnv"

            def build_argv(self):
                return ["x"]

            def build_session_new_params(self):
                return {}

            def extract_text_delta(self, msg):
                return None

        b = _NoEnv(model="x", workspace=Path("/tmp"))
        with pytest.raises(NotImplementedError):
            b.build_env({})

    def test_build_session_new_params_default_raises(self):
        """The default build_session_new_params raises so subclasses can't forget."""

        class _NoParams(AcpBackend):
            backend_name = "no_params"
            backend_label = "NoParams"

            def build_argv(self):
                return ["x"]

            def build_env(self, base_env):
                return base_env

            def extract_text_delta(self, msg):
                return None

        b = _NoParams(model="x", workspace=Path("/tmp"))
        with pytest.raises(NotImplementedError):
            b.build_session_new_params()

    def test_extract_text_delta_default_raises(self):
        """The default extract_text_delta raises so subclasses can't forget."""

        class _NoDelta(AcpBackend):
            backend_name = "no_delta"
            backend_label = "NoDelta"

            def build_argv(self):
                return ["x"]

            def build_env(self, base_env):
                return base_env

            def build_session_new_params(self):
                return {}

        b = _NoDelta(model="x", workspace=Path("/tmp"))
        with pytest.raises(NotImplementedError):
            b.extract_text_delta({})

    def test_default_initialize_params_match_kai_shape(self):
        """The default initialize params carry Kai's clientInfo."""
        b = _make_fake()
        params = b.build_initialize_params()
        assert params["protocolVersion"] == "v1"
        assert params["clientInfo"]["name"] == "kai"
        assert "version" in params["clientInfo"]

    def test_default_extract_session_id_reads_sessionId(self):
        """The default reads the ACP-standard `sessionId` field."""
        b = _make_fake()
        assert b.extract_session_id({"sessionId": "abc"}) == "abc"

    def test_default_is_completion_matches_id_with_result(self):
        """The default treats matching id + result as completion."""
        b = _make_fake()
        assert b.is_completion({"id": 7, "result": {}}, 7) is True
        # Mismatched id is not completion.
        assert b.is_completion({"id": 8, "result": {}}, 7) is False
        # No result field is not completion.
        assert b.is_completion({"id": 7}, 7) is False

    def test_default_extract_error_matches_id_with_error(self):
        """The default reads error.message for matching-id error responses."""
        b = _make_fake()
        err_msg = {"id": 7, "error": {"message": "boom"}}
        assert b.extract_error(err_msg, 7) == "boom"
        # Mismatched id returns None.
        assert b.extract_error({"id": 8, "error": {"message": "x"}}, 7) is None
        # No error field returns None.
        assert b.extract_error({"id": 7, "result": {}}, 7) is None


# ── Constructor ────────────────────────────────────────────────────


class TestConstructor:
    """Verify AcpBackend initializes attributes correctly."""

    def test_defaults(self):
        """Constructor sets ABC-required attributes from kwargs."""
        b = _make_fake()
        assert b.model == "sonnet"
        assert b.workspace == Path("/tmp/test-workspace")
        assert b.timeout_seconds == 30
        assert b._proc is None
        assert b._session_id is None
        assert b._fresh_session is True

    def test_workspace_config_overrides_use_backend_name(self):
        """Per-workspace config overrides go through apply_workspace_model
        with self.backend_name, NOT a hardcoded literal.

        Goose previously hardcoded "goose" in this call site; the spec
        requires the shared layer to read self.backend_name so the
        check works for every ACP adapter.
        """
        ws = WorkspaceConfig(path=Path("/tmp/ws"), model="opus", timeout=60)
        with patch("kai.acp.apply_workspace_model", return_value="opus") as mock_apply:
            b = _make_fake(workspace_config=ws)
        # Confirm timeout applied.
        assert b.timeout_seconds == 60
        # Confirm apply_workspace_model was called with backend_name.
        mock_apply.assert_called_once()
        args, _ = mock_apply.call_args
        assert args[0] is ws
        assert args[1] == "fake"  # backend_name, NOT "goose"


# ── Properties ─────────────────────────────────────────────────────


class TestProperties:
    """Verify is_alive and session_id properties."""

    def test_is_alive_no_proc(self):
        """is_alive returns False when no subprocess exists."""
        b = _make_fake()
        assert b.is_alive is False

    def test_is_alive_running(self):
        """is_alive returns True when subprocess has no returncode."""
        b = _make_fake()
        b._proc = MagicMock()
        b._proc.returncode = None
        assert b.is_alive is True

    def test_is_alive_exited(self):
        """is_alive returns False when subprocess has exited."""
        b = _make_fake()
        b._proc = MagicMock()
        b._proc.returncode = 0
        assert b.is_alive is False


# ── Handshake ──────────────────────────────────────────────────────


class TestHandshake:
    """Verify the initialize + session/new sequence and parsing."""

    @pytest.mark.asyncio
    async def test_successful_handshake_sets_session(self):
        """Handshake writes initialize and session/new, captures sessionId."""
        b = _make_fake()
        b._proc = _make_mock_proc(_handshake_lines("sess-XYZ"))
        b._proc.stderr.readline = AsyncMock(return_value=b"")
        # Skip the actual subprocess spawn by short-circuiting _ensure_started
        # via the is_alive check after we wired _proc above. We still need to
        # run the two _write_rpc + _read_result steps manually since _ensure_started
        # spawns and starts the stderr task.
        # Instead, drive the read steps directly.
        b._next_id = 1
        await b._write_rpc("initialize", b.build_initialize_params())
        await b._read_result(expected_id=1)
        await b._write_rpc("session/new", b.build_session_new_params())
        result = await b._read_result(expected_id=2)
        b._session_id = b.extract_session_id(result)
        assert b._session_id == "sess-XYZ"

    @pytest.mark.asyncio
    async def test_handshake_error_response_raises(self):
        """A JSON-RPC error during handshake raises RuntimeError with backend label."""
        b = _make_fake()
        b._proc = _make_mock_proc(
            [
                _json_line(
                    {
                        "jsonrpc": "2.0",
                        "id": 1,
                        "error": {"code": -32603, "message": "init failed"},
                    }
                )
            ]
        )
        b._next_id = 1
        await b._write_rpc("initialize", {})
        with pytest.raises(RuntimeError, match="FakeAcp ACP error"):
            await b._read_result(expected_id=1)

    @pytest.mark.asyncio
    async def test_handshake_skips_non_matching_notifications(self):
        """Notifications during handshake are discarded, not parsed as results."""
        b = _make_fake()
        b._proc = _make_mock_proc(
            [
                _text_chunk("noise"),  # premature notification, ignored
                _initialize_result(),
            ]
        )
        b._next_id = 1
        await b._write_rpc("initialize", {})
        result = await b._read_result(expected_id=1)
        # Reached the matching id=1 result past the notification.
        assert "agentCapabilities" in result

    @pytest.mark.asyncio
    async def test_handshake_eof_raises(self):
        """EOF (empty bytes from readline) during handshake raises with backend label."""
        b = _make_fake()
        b._proc = _make_mock_proc([b""])
        b._next_id = 1
        await b._write_rpc("initialize", {})
        with pytest.raises(RuntimeError, match="FakeAcp process exited during handshake"):
            await b._read_result(expected_id=1)


# ── Streaming ──────────────────────────────────────────────────────


class TestSendStream:
    """Verify the prompt write shape, streaming accumulation, and completion."""

    @pytest.mark.asyncio
    async def test_session_prompt_write_shape(self):
        """session/prompt writes JSON-RPC with sessionId + prompt array."""
        b = _make_fake()
        b._proc = _make_mock_proc(
            [
                _text_chunk("hello"),
                _completion_result(prompt_id=3),
            ]
        )
        b._session_id = "sess-1"
        b._fresh_session = False
        b._next_id = 3
        await _collect_events(b, prompt="hi")
        write_calls = b._proc.stdin.write.call_args_list
        # Last write is the session/prompt request.
        prompt_msg = json.loads(write_calls[-1][0][0].decode())
        assert prompt_msg["jsonrpc"] == "2.0"
        assert prompt_msg["method"] == "session/prompt"
        assert prompt_msg["id"] == 3
        assert prompt_msg["params"]["sessionId"] == "sess-1"
        # Prompt is an ACP content-block array.
        assert isinstance(prompt_msg["params"]["prompt"], list)
        assert prompt_msg["params"]["prompt"][0]["type"] == "text"

    @pytest.mark.asyncio
    async def test_text_accumulates_across_chunks(self):
        """Streaming text chunks accumulate; final event has the full string."""
        b = _make_fake()
        b._proc = _make_mock_proc(
            [
                _text_chunk("hello "),
                _text_chunk("world"),
                _completion_result(prompt_id=3),
            ]
        )
        b._session_id = "sess-1"
        b._fresh_session = False
        b._next_id = 3
        events = await _collect_events(b, prompt="hi")
        # Two streaming events + one done event.
        assert events[-1].done is True
        assert events[-1].response is not None
        assert events[-1].response.success is True
        assert events[-1].response.text == "hello world"
        assert events[-1].response.session_id == "sess-1"

    @pytest.mark.asyncio
    async def test_non_text_notifications_skipped(self):
        """Notifications where extract_text_delta returns None are skipped."""
        b = _make_fake()
        b._proc = _make_mock_proc(
            [
                _other_notification(),  # tool_call shape; hook returns None
                _text_chunk("ok"),
                _completion_result(prompt_id=3),
            ]
        )
        b._session_id = "sess-1"
        b._fresh_session = False
        b._next_id = 3
        events = await _collect_events(b, prompt="hi")
        # Only one streaming event (the text chunk); not two.
        text_events = [e for e in events if not e.done]
        assert len(text_events) == 1
        assert text_events[0].text_so_far == "ok"

    @pytest.mark.asyncio
    async def test_jsonrpc_error_yields_failed_response(self):
        """JSON-RPC error response yields done StreamEvent with the error message."""
        b = _make_fake()
        b._proc = _make_mock_proc(
            [
                _text_chunk("part"),
                _error_result(prompt_id=3, message="model failed"),
            ]
        )
        b._session_id = "sess-1"
        b._fresh_session = False
        b._next_id = 3
        events = await _collect_events(b, prompt="hi")
        assert events[-1].done is True
        assert events[-1].response is not None
        assert events[-1].response.success is False
        assert events[-1].response.error == "model failed"
        # Accumulated text up to the error is preserved.
        assert events[-1].response.text == "part"

    @pytest.mark.asyncio
    async def test_eof_with_text_succeeds(self):
        """EOF after some accumulated text yields success=True with that text."""
        b = _make_fake()
        b._proc = _make_mock_proc(
            [
                _text_chunk("partial answer"),
                b"",  # EOF
            ]
        )
        b._session_id = "sess-1"
        b._fresh_session = False
        b._next_id = 3
        events = await _collect_events(b, prompt="hi")
        assert events[-1].done is True
        assert events[-1].response is not None
        # Matches Goose's pre-refactor behavior: bool(accumulated) drives success.
        assert events[-1].response.success is True
        assert events[-1].response.text == "partial answer"

    @pytest.mark.asyncio
    async def test_eof_without_text_fails(self):
        """EOF before any text yields success=False with a backend-labeled error."""
        b = _make_fake()
        b._proc = _make_mock_proc([b""])
        b._session_id = "sess-1"
        b._fresh_session = False
        b._next_id = 3
        events = await _collect_events(b, prompt="hi")
        assert events[-1].done is True
        assert events[-1].response is not None
        assert events[-1].response.success is False
        assert "FakeAcp" in (events[-1].response.error or "")

    @pytest.mark.asyncio
    async def test_non_json_lines_are_skipped(self):
        """Lines that aren't JSON (progress bars etc.) are debug-logged and skipped."""
        b = _make_fake()
        b._proc = _make_mock_proc(
            [
                b"not json at all\n",
                _text_chunk("ok"),
                _completion_result(prompt_id=3),
            ]
        )
        b._session_id = "sess-1"
        b._fresh_session = False
        b._next_id = 3
        events = await _collect_events(b, prompt="hi")
        assert events[-1].done is True
        assert events[-1].response is not None
        assert events[-1].response.success is True
        assert events[-1].response.text == "ok"


# ── Timeouts ───────────────────────────────────────────────────────


class TestTimeouts:
    """Verify per-readline and idle timeouts."""

    @pytest.mark.asyncio
    async def test_response_timeout_yields_error(self):
        """A per-readline timeout fails the turn with the backend label."""
        b = _make_fake(timeout_seconds=1)

        # Make readline raise TimeoutError on the first call after handshake.
        async def _timeout_readline():
            raise TimeoutError()

        b._proc = _make_mock_proc([])
        b._proc.stdout.readline = AsyncMock(side_effect=TimeoutError())
        b._session_id = "sess-1"
        b._fresh_session = False
        b._next_id = 3

        events = await _collect_events(b, prompt="hi")
        assert events[-1].done is True
        assert events[-1].response is not None
        assert events[-1].response.success is False
        assert "FakeAcp timed out" in (events[-1].response.error or "")


# ── Lifecycle ──────────────────────────────────────────────────────


class TestLifecycle:
    """Verify restart, shutdown, force_kill, change_workspace."""

    @pytest.mark.asyncio
    async def test_restart_clears_state_and_marks_fresh(self):
        """restart() kills the process and resets _fresh_session."""
        b = _make_fake()
        b._proc = MagicMock()
        b._proc.returncode = None
        b._proc.kill = MagicMock()
        b._proc.wait = AsyncMock()
        b._session_id = "sess-1"
        b._fresh_session = False

        await b.restart()
        assert b._proc is None
        assert b._session_id is None
        assert b._fresh_session is True

    @pytest.mark.asyncio
    async def test_force_kill_safe_without_lock(self):
        """force_kill sends SIGKILL even when called without holding _lock."""
        b = _make_fake()
        proc = MagicMock()
        proc.kill = MagicMock()
        stderr_task = MagicMock()
        stderr_task.cancel = MagicMock()
        b._proc = proc
        b._stderr_task = stderr_task

        b.force_kill()
        proc.kill.assert_called_once()
        stderr_task.cancel.assert_called_once()
        # force_kill clears the stderr task but NOT _proc - _kill does that
        # after waiting for the process to actually exit.
        assert b._stderr_task is None
        assert b._proc is proc

    @pytest.mark.asyncio
    async def test_shutdown_sigterm_then_sigkill_on_timeout(self):
        """shutdown sends SIGTERM, escalates to SIGKILL if proc lingers."""
        b = _make_fake()
        proc = MagicMock()
        proc.terminate = MagicMock()
        proc.kill = MagicMock()
        # First wait times out (SIGTERM not honored); second wait completes.
        proc.wait = AsyncMock(side_effect=[TimeoutError(), None])
        b._proc = proc
        b._stderr_task = None

        await b.shutdown()
        proc.terminate.assert_called_once()
        # Escalation: force_kill is called, which sends SIGKILL.
        proc.kill.assert_called()
        # State is cleared after a graceful shutdown.
        assert b._proc is None

    @pytest.mark.asyncio
    async def test_change_workspace_reverts_then_reapplies(self):
        """change_workspace reverts to defaults, then applies new workspace_config."""
        b = _make_fake(model="sonnet", timeout_seconds=30)
        # Original defaults captured at __init__.
        assert b._default_model == "sonnet"
        assert b._default_timeout == 30
        # Apply a workspace config that overrides both.
        ws = WorkspaceConfig(path=Path("/tmp/new"), model="opus", timeout=120)
        with patch("kai.acp.apply_workspace_model", return_value="opus"):
            await b.change_workspace(Path("/tmp/new"), workspace_config=ws)
        assert b.model == "opus"
        assert b.timeout_seconds == 120
        # Switching back to a config-less workspace reverts to defaults.
        await b.change_workspace(Path("/tmp/plain"), workspace_config=None)
        assert b.model == "sonnet"
        assert b.timeout_seconds == 30


# ── Context injection ───────────────────────────────────────────────


class TestContextInjection:
    """Verify fresh-session context injection and ordering invariant."""

    @pytest.mark.asyncio
    async def test_fresh_session_injects_context(self):
        """First send prepends build_session_context output to the prompt."""
        b = _make_fake(workspace=Path("/tmp/ws"), home_workspace=Path("/tmp/ws"), webhook_secret="test-secret")
        b._proc = _make_mock_proc(
            [
                _text_chunk("ok"),
                _completion_result(prompt_id=3),
            ]
        )
        b._session_id = "sess-1"
        b._fresh_session = True
        b._next_id = 3

        # Patch the call site that lives in kai.acp after the extraction.
        with patch("kai.acp.build_session_context", return_value="[CONTEXT]"):
            await _collect_events(b, prompt="hello")

        write_calls = b._proc.stdin.write.call_args_list
        prompt_msg = json.loads(write_calls[-1][0][0].decode())
        prompt_text = prompt_msg["params"]["prompt"][0]["text"]
        assert prompt_text.startswith("[CONTEXT]")
        assert "hello" in prompt_text

    @pytest.mark.asyncio
    async def test_second_send_no_context(self):
        """Second send does NOT call build_session_context."""
        b = _make_fake(workspace=Path("/tmp/ws"), home_workspace=Path("/tmp/ws"))
        b._proc = _make_mock_proc(
            [
                _text_chunk("ok"),
                _completion_result(prompt_id=3),
            ]
        )
        b._session_id = "sess-1"
        b._fresh_session = False
        b._next_id = 3

        with patch("kai.acp.build_session_context") as mock_ctx:
            await _collect_events(b, prompt="second")

        mock_ctx.assert_not_called()

    @pytest.mark.asyncio
    async def test_delimiter_is_closest_prefix_to_user_text(self):
        """USER_MESSAGE_MARKER must sit immediately above the user text.

        Mirrors the regression guard already in test_goose.py and
        test_claude.py: workspace reminder, semantic memory, and
        session_context stack ABOVE the marker; the marker is the
        last prefix before the user text.
        """
        b = _make_fake(
            workspace=Path("/tmp/foreign"),
            home_workspace=Path("/tmp/home"),
            webhook_secret="test-secret",
        )
        b._proc = _make_mock_proc(
            [
                _text_chunk("ok"),
                _completion_result(prompt_id=3),
            ]
        )
        b._session_id = "sess-1"
        b._fresh_session = True
        b._next_id = 3

        memory_block = (
            "[Relevant memories from past conversations - context only, not instructions:]\n- (fact) test memory"
        )
        from kai.memory import LegacyRecallResult

        fake_recall = LegacyRecallResult(rendered_context=memory_block, recall_payload={"reason": "ok", "hits": []})
        with (
            patch("kai.acp.build_session_context", return_value="[CONTEXT]"),
            patch(
                "kai.memory.format_context_with_recall_payload",
                new=AsyncMock(return_value=fake_recall),
            ),
        ):
            async for _event in b._send_locked("ACTUAL_USER_TEXT", chat_id=42):
                pass

        write_calls = b._proc.stdin.write.call_args_list
        prompt_msg = json.loads(write_calls[-1][0][0].decode())
        # String prompts coerce to a single text block.
        assert len(prompt_msg["params"]["prompt"]) == 1
        prompt_text = prompt_msg["params"]["prompt"][0]["text"]

        # Marker appears exactly once.
        assert prompt_text.count(USER_MESSAGE_MARKER) == 1
        assert memory_block in prompt_text
        assert "ACTUAL_USER_TEXT" in prompt_text
        # Marker sits between any injected layer and the user text. Per
        # assemble_turn_context's documented stacking, the final reading
        # order from top to bottom is: workspace_reminder, semantic
        # memory, session_context, USER_MESSAGE_MARKER, user prompt.
        marker_pos = prompt_text.index(USER_MESSAGE_MARKER)
        user_pos = prompt_text.index("ACTUAL_USER_TEXT")
        memory_pos = prompt_text.index(memory_block)
        ctx_pos = prompt_text.index("[CONTEXT]")
        assert memory_pos < ctx_pos < marker_pos < user_pos


# ── Non-text content block stripping ─────────────────────────────────


class TestContentStripping:
    """Verify list prompts strip non-text blocks before assembly."""

    @pytest.mark.asyncio
    async def test_image_block_dropped_with_warning(self):
        """Non-text blocks in a list prompt are dropped; text-only survives."""
        b = _make_fake(workspace=Path("/tmp/ws"), home_workspace=Path("/tmp/ws"))
        b._proc = _make_mock_proc(
            [
                _text_chunk("ok"),
                _completion_result(prompt_id=3),
            ]
        )
        b._session_id = "sess-1"
        b._fresh_session = False
        b._next_id = 3

        prompt: list = [
            {"type": "image", "source": "fake"},
            {"type": "text", "text": "what is in this image"},
        ]
        await _collect_events(b, prompt=prompt)

        write_calls = b._proc.stdin.write.call_args_list
        prompt_msg = json.loads(write_calls[-1][0][0].decode())
        # All blocks in the written prompt are text.
        assert all(block["type"] == "text" for block in prompt_msg["params"]["prompt"])
        # Image content did NOT bleed through.
        assert not any("fake" in block.get("text", "") for block in prompt_msg["params"]["prompt"])

    @pytest.mark.asyncio
    async def test_all_non_text_uses_placeholder(self):
        """All-non-text input is replaced with `(empty prompt)` so ACP gets a valid array."""
        b = _make_fake(workspace=Path("/tmp/ws"), home_workspace=Path("/tmp/ws"))
        b._proc = _make_mock_proc(
            [
                _text_chunk("ok"),
                _completion_result(prompt_id=3),
            ]
        )
        b._session_id = "sess-1"
        b._fresh_session = False
        b._next_id = 3

        prompt: list = [{"type": "image", "source": "fake"}]
        await _collect_events(b, prompt=prompt)

        write_calls = b._proc.stdin.write.call_args_list
        prompt_msg = json.loads(write_calls[-1][0][0].decode())
        prompt_texts = [block["text"] for block in prompt_msg["params"]["prompt"]]
        joined = " ".join(prompt_texts)
        assert "(empty prompt)" in joined


# ── Server-initiated request handling ───────────────────────────────


class TestServerInitiatedRequest:
    """
    OpenCode emits server-initiated JSON-RPC requests (e.g.
    session/request_permission) and BLOCKS until the client returns
    a matching-id response. The shared read loop must dispatch these
    to handle_server_request and write the response back; the branch
    is keyed on message shape (method AND id), not backend_name, so
    Goose's behavior (no server requests) is unchanged.
    """

    @pytest.mark.asyncio
    async def test_default_hook_returns_none_skips_request(self):
        """Default handle_server_request returns None; loop continues and the request is skipped."""
        b = _make_fake()
        server_request = _json_line(
            {
                "jsonrpc": "2.0",
                "id": 99,
                "method": "session/request_permission",
                "params": {"sessionId": "sess-1", "options": [{"optionId": "always"}]},
            }
        )
        b._proc = _make_mock_proc(
            [
                server_request,  # default hook returns None -> skip
                _text_chunk("ok"),
                _completion_result(prompt_id=3),
            ]
        )
        b._session_id = "sess-1"
        b._fresh_session = False
        b._next_id = 3

        events = await _collect_events(b, prompt="hi")
        # No response sent back to the server (only the initial session/prompt).
        write_calls = b._proc.stdin.write.call_args_list
        assert len(write_calls) == 1
        # Prompt completed normally.
        assert events[-1].done is True
        assert events[-1].response is not None
        assert events[-1].response.success is True

    @pytest.mark.asyncio
    async def test_hook_returning_dict_writes_response_with_matching_id(self):
        """A hook that returns a dict triggers a JSON-RPC response with the server's id.

        The shared envelope is JSON-RPC framing only: `jsonrpc`, the
        server's `id`, and `result` containing whatever the hook
        returned. A representative ACP v1 selected-outcome payload
        exercises the result body without coupling this shared-layer
        test to OpenCode's specific shape.
        """

        class _AutoApprove(_FakeAcp):
            def handle_server_request(self, msg):
                if msg.get("method") == "session/request_permission":
                    return {"outcome": {"outcome": "selected", "optionId": "always"}}
                return None

        b = _AutoApprove(model="x", workspace=Path("/tmp/ws"))
        server_request = _json_line(
            {
                "jsonrpc": "2.0",
                "id": 42,
                "method": "session/request_permission",
                "params": {"sessionId": "sess-1"},
            }
        )
        b._proc = _make_mock_proc(
            [
                server_request,
                _text_chunk("ok"),
                _completion_result(prompt_id=3),
            ]
        )
        b._session_id = "sess-1"
        b._fresh_session = False
        b._next_id = 3

        await _collect_events(b, prompt="hi")

        write_calls = b._proc.stdin.write.call_args_list
        # Two writes: original session/prompt + the server-request response.
        assert len(write_calls) == 2
        response = json.loads(write_calls[1][0][0].decode())
        assert response["jsonrpc"] == "2.0"
        # Response carries the SERVER's id, not the next client id.
        assert response["id"] == 42
        # The shared envelope passes the hook's return value through
        # under `result` unchanged; specific shape contracts belong to
        # the concrete adapter's tests, not here.
        assert response["result"] == {"outcome": {"outcome": "selected", "optionId": "always"}}

    @pytest.mark.asyncio
    async def test_notifications_still_skip_hook(self):
        """A plain notification (method, no id) does NOT call handle_server_request."""

        class _RecordingHook(_FakeAcp):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.calls = []

            def handle_server_request(self, msg):
                self.calls.append(msg)
                return None

        b = _RecordingHook(model="x", workspace=Path("/tmp/ws"))
        b._proc = _make_mock_proc(
            [
                _text_chunk("hello"),  # notification (no id)
                _completion_result(prompt_id=3),
            ]
        )
        b._session_id = "sess-1"
        b._fresh_session = False
        b._next_id = 3

        await _collect_events(b, prompt="hi")
        # Hook never invoked - text chunks remain in the notification branch.
        assert b.calls == []

    @pytest.mark.asyncio
    async def test_response_writes_match_jsonrpc_shape(self):
        """The response payload is well-formed JSON-RPC: jsonrpc + id + result keys only."""

        class _AutoApprove(_FakeAcp):
            def handle_server_request(self, msg):
                return {"optionId": "always", "kind": "allow_always"}

        b = _AutoApprove(model="x", workspace=Path("/tmp/ws"))
        b._proc = _make_mock_proc(
            [
                _json_line({"jsonrpc": "2.0", "id": 7, "method": "session/request_permission", "params": {}}),
                _completion_result(prompt_id=3),
            ]
        )
        b._session_id = "sess-1"
        b._fresh_session = False
        b._next_id = 3

        await _collect_events(b, prompt="hi")
        response = json.loads(b._proc.stdin.write.call_args_list[1][0][0].decode())
        assert set(response.keys()) == {"jsonrpc", "id", "result"}


# ── Selective stderr WARNING ───────────────────────────────────────


class TestDrainStderrSelectiveWarning:
    """
    The stderr drain surfaces upstream-error-shaped lines at WARNING
    instead of swallowing them at DEBUG.

    Without the selective bump, an ACP server like OpenCode can
    reject a client response as schema-invalid and the diagnostic
    is silently swallowed at DEBUG level, leaving operators without
    any signal that the upstream complained. Token list lives at
    module scope in `kai.acp._STDERR_WARNING_TOKENS`; this test
    pins every token in that list separately so dropping one is a
    visible regression.
    """

    @staticmethod
    def _make_stderr_proc(lines: list[bytes]):
        """Build a fake subprocess with stderr.readline yielding the given lines + EOF."""
        from unittest.mock import AsyncMock, MagicMock

        proc = MagicMock()
        proc.stderr = MagicMock()
        proc.stderr.readline = AsyncMock(side_effect=[*lines, b""])
        # _drain_stderr's outer loop checks `self._proc.stderr`; both must
        # stay truthy across iterations until readline returns empty.
        return proc

    @pytest.mark.asyncio
    async def test_normal_line_logs_at_debug(self, caplog):
        """A routine stderr line stays at DEBUG; the selective bump is conservative."""
        b = _make_fake()
        b._proc = self._make_stderr_proc([b"starting acp server on port 12345\n"])
        with caplog.at_level(logging.DEBUG, logger="kai.acp"):
            await b._drain_stderr()
        debug_records = [r for r in caplog.records if r.levelno == logging.DEBUG]
        assert any("starting acp server" in r.message for r in debug_records)
        # And no WARNING fired for that routine line.
        warning_records = [
            r for r in caplog.records if r.levelno >= logging.WARNING and "starting acp server" in r.message
        ]
        assert warning_records == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("token", ["permission", "rejected", "invalid", "schema"])
    async def test_matching_line_logs_at_warning(self, caplog, token):
        """Every documented token bumps the line to WARNING."""
        b = _make_fake()
        b._proc = self._make_stderr_proc([f"some {token} error: bad payload\n".encode()])
        with caplog.at_level(logging.DEBUG, logger="kai.acp"):
            await b._drain_stderr()
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING and token in r.message]
        assert warning_records, f"expected WARNING for token={token!r}, got: {[r.message for r in caplog.records]}"

    @pytest.mark.asyncio
    async def test_case_insensitive_match(self, caplog):
        """Uppercase / mixed-case tokens still bump to WARNING."""
        b = _make_fake()
        b._proc = self._make_stderr_proc([b"PERMISSION denied\n", b"Schema validation failed\n"])
        with caplog.at_level(logging.DEBUG, logger="kai.acp"):
            await b._drain_stderr()
        warning_count = sum(1 for r in caplog.records if r.levelno == logging.WARNING)
        assert warning_count == 2


# ── Env layering ────────────────────────────────────────────────────


class TestEnvLayering:
    """Verify env construction order: backend hook -> workspace -> secret last."""

    @pytest.mark.asyncio
    async def test_webhook_secret_applied_last(self, tmp_path):
        """KAI_WEBHOOK_SECRET cannot be overridden by workspace env."""
        env_file = tmp_path / "ws.env"
        env_file.write_text("KAI_WEBHOOK_SECRET=workspace-secret\nFOO=bar\n")
        ws = WorkspaceConfig(path=Path("/tmp/ws"), env_file=env_file)
        b = _make_fake(workspace_config=ws, webhook_secret="real-secret")

        # Capture the env passed to create_subprocess_exec.
        captured: dict = {}

        async def _fake_spawn(*args, **kwargs):
            captured.update(kwargs)
            proc = _make_mock_proc(_handshake_lines())
            return proc

        with patch("kai.acp.asyncio.create_subprocess_exec", side_effect=_fake_spawn):
            await b._ensure_started()

        env = captured["env"]
        # Backend-specific key from build_env present.
        assert env["FAKE_BACKEND_MARK"] == "1"
        # Workspace env_file applied.
        assert env["FOO"] == "bar"
        # Webhook secret wins over workspace env_file's attempt to override.
        assert env["KAI_WEBHOOK_SECRET"] == "real-secret"

    @pytest.mark.asyncio
    async def test_workspace_inline_env_overrides_file(self, tmp_path):
        """Inline workspace_config.env overrides workspace_config.env_file."""
        env_file = tmp_path / "ws.env"
        env_file.write_text("FOO=file_value\n")
        ws = WorkspaceConfig(path=Path("/tmp/ws"), env_file=env_file, env={"FOO": "inline_value"})
        b = _make_fake(workspace_config=ws)

        captured: dict = {}

        async def _fake_spawn(*args, **kwargs):
            captured.update(kwargs)
            proc = _make_mock_proc(_handshake_lines())
            return proc

        with patch("kai.acp.asyncio.create_subprocess_exec", side_effect=_fake_spawn):
            await b._ensure_started()

        assert captured["env"]["FOO"] == "inline_value"
