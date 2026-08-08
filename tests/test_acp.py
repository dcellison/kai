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
18. Image content blocks: promptCapabilities.image capture from the
    initialize result, Anthropic-to-ACP block conversion, forwarding
    when supported, drop notice in the reply text when not, and the
    EOF-after-drop path staying an error
19. os_user routing: direct spawn vs sudo -H wrap, preserve-env CSV
    from the preserved_env_vars() hook, per-os-user TMPDIR anchor
20. Cross-user kill escalation on force_kill / _kill / shutdown /
    in-stream timeout when wrapped, and its absence when direct
21. The shared _kill_target_user_tree helpers (argv shape, ESRCH
    demotion, timeout bound)
"""

import asyncio
import json
import logging
import signal
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai.acp import AcpBackend, convert_image_block, drain_late_text
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


def _initialize_result(prompt_capabilities: dict | None = None) -> bytes:
    """Build the server's response to an initialize request.

    `prompt_capabilities` lands under `agentCapabilities.
    promptCapabilities` (the ACP location AcpBackend reads the image
    capability from); None omits the key entirely, matching an agent
    that does not advertise prompt capabilities.
    """
    caps: dict = {}
    if prompt_capabilities is not None:
        caps["promptCapabilities"] = prompt_capabilities
    return _json_line(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"protocolVersion": 0, "agentCapabilities": caps},
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
    Once the list is exhausted, every further readline call returns
    b"" (EOF), so tests do not need to count exactly how many reads
    the send loop performs; the post-completion drain in particular
    issues reads past the completion result.
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


def _handshake_lines(session_id: str = "sess-1", prompt_capabilities: dict | None = None) -> list[bytes]:
    """Return the two stdout lines for a successful handshake."""
    return [_initialize_result(prompt_capabilities), _session_new_result(session_id)]


async def _collect_events(backend: _FakeAcp, prompt: str | list = "test") -> list[StreamEvent]:
    """Send a prompt and collect all yielded StreamEvents."""
    events = []
    async for event in backend._send_locked(prompt):
        events.append(event)
    return events


def _anthropic_image_block(data: str = "QkFTRTY0", media_type: str = "image/jpeg") -> dict:
    """Anthropic-style base64 image block, the shape the bot layer builds."""
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": data}}


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


# ── Post-completion drain ──────────────────────────────────────────


class TestCompletionDrain:
    """Text chunks can land on stdout AFTER the session/prompt
    response: the server resolves the prompt when the session goes
    idle while its event forwarding still has chunks in flight. The
    send loop drains those trailing chunks before yielding the done
    event, so the final text is complete and nothing stays buffered
    in the pipe to surface at the start of the next turn."""

    @pytest.mark.asyncio
    async def test_late_chunk_after_completion_included_in_final_text(self):
        b = _make_fake()
        b._proc = _make_mock_proc(
            [
                _text_chunk("hello wor"),
                _completion_result(prompt_id=3),
                _text_chunk("ld"),
            ]
        )
        b._session_id = "sess-1"
        b._fresh_session = False
        b._next_id = 3
        events = await _collect_events(b, prompt="hi")
        assert events[-1].done is True
        assert events[-1].response is not None
        assert events[-1].response.success is True
        assert events[-1].response.text == "hello world"

    @pytest.mark.asyncio
    async def test_drain_skips_non_text_and_id_bearing_messages(self):
        """During the drain, non-JSON garbage, non-text notifications,
        and id-bearing messages (server requests, stray responses) are
        ignored; only text chunks accumulate."""
        b = _make_fake()
        b._proc = _make_mock_proc(
            [
                _text_chunk("body"),
                _completion_result(prompt_id=3),
                b"not json\n",
                _other_notification(),
                _json_line({"jsonrpc": "2.0", "id": 99, "method": "session/request_permission", "params": {}}),
                _json_line({"jsonrpc": "2.0", "id": 98, "result": {}}),
                _text_chunk(" tail"),
            ]
        )
        b._session_id = "sess-1"
        b._fresh_session = False
        b._next_id = 3
        events = await _collect_events(b, prompt="hi")
        assert events[-1].response is not None
        assert events[-1].response.success is True
        assert events[-1].response.text == "body tail"

    @pytest.mark.asyncio
    async def test_drain_quiet_window_closes_with_pipe_open(self, monkeypatch):
        """When the pipe stays open but nothing arrives within the
        drain window, the drain stops and the done event carries the
        text accumulated so far. The window constant lives in kai.acp
        and is resolved at call time, so patching it there covers
        every drain call site."""
        monkeypatch.setattr("kai.acp.COMPLETION_DRAIN_WINDOW_S", 0.05)
        b = _make_fake()
        b._proc = _make_mock_proc([])
        queue = [_text_chunk("done body"), _completion_result(prompt_id=3)]

        async def _readline() -> bytes:
            if queue:
                return queue.pop(0)
            # Pipe open, no data: block until the drain window
            # cancels the read.
            await asyncio.Event().wait()
            return b""

        b._proc.stdout.readline = _readline
        b._session_id = "sess-1"
        b._fresh_session = False
        b._next_id = 3
        events = await _collect_events(b, prompt="hi")
        assert events[-1].done is True
        assert events[-1].response is not None
        assert events[-1].response.success is True
        assert events[-1].response.text == "done body"


class TestDrainLateTextFreeFunction:
    """drain_late_text contract: accumulate text deltas through the
    caller's hooks until the quiet window passes or the pipe reaches
    EOF; everything that is not a text notification is ignored."""

    @pytest.mark.asyncio
    async def test_eof_ends_drain_and_returns_accumulated(self):
        proc = _make_mock_proc([_text_chunk("Hello"), _text_chunk(" world")])
        out = await drain_late_text(
            proc=proc,
            accumulated="say: ",
            extract_delta=_make_fake().extract_text_delta,
            combine=lambda prev, new: prev + new,
        )
        assert out == "say: Hello world"

    @pytest.mark.asyncio
    async def test_quiet_window_returns_accumulated_unchanged(self):
        proc = _make_mock_proc([])

        async def _readline() -> bytes:
            await asyncio.Event().wait()
            return b""

        proc.stdout.readline = _readline
        out = await drain_late_text(
            proc=proc,
            accumulated="full answer",
            extract_delta=_make_fake().extract_text_delta,
            combine=lambda prev, new: prev + new,
            window_s=0.05,
        )
        assert out == "full answer"

    @pytest.mark.asyncio
    async def test_combine_hook_applies_to_drained_chunks(self):
        """The caller's combine hook runs on drained chunks exactly as
        it does on in-turn chunks (opencode's sentence-boundary space
        injection must not be bypassed for the tail)."""
        proc = _make_mock_proc([_text_chunk("Tail starts here")])
        out = await drain_late_text(
            proc=proc,
            accumulated="Sentence ends.",
            extract_delta=_make_fake().extract_text_delta,
            combine=lambda prev, new: prev + " " + new,
        )
        assert out == "Sentence ends. Tail starts here"


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
        from kai.memory import ScopedRecallResult

        fake_recall = ScopedRecallResult(rendered_context=memory_block, recall_payload={"reason": "ok", "hits": []})
        with (
            patch("kai.acp.build_session_context", return_value="[CONTEXT]"),
            patch(
                "kai.memory.format_scoped_context_with_recall_payload",
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

    @pytest.mark.asyncio
    async def test_image_forwarded_when_agent_supports_it(self):
        """With `promptCapabilities.image` captured as True, an
        Anthropic-style image block is converted to the ACP shape and
        written on session/prompt; no drop notice appears."""
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
        b._supports_image_input = True

        prompt: list = [
            {"type": "text", "text": "what is in this image"},
            _anthropic_image_block(),
        ]
        events = await _collect_events(b, prompt=prompt)

        write_calls = b._proc.stdin.write.call_args_list
        prompt_msg = json.loads(write_calls[-1][0][0].decode())
        sent_blocks = prompt_msg["params"]["prompt"]
        assert {"type": "image", "mimeType": "image/jpeg", "data": "QkFTRTY0"} in sent_blocks
        final = events[-1].response
        assert final is not None and final.success
        assert "[Note:" not in final.text

    @pytest.mark.asyncio
    async def test_dropped_image_seeds_user_notice(self):
        """Without image capability the block is dropped AND the reply
        text carries a notice; a log-only drop leaves the user
        believing the model saw the image."""
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
            {"type": "text", "text": "what is in this image"},
            _anthropic_image_block(),
        ]
        events = await _collect_events(b, prompt=prompt)

        write_calls = b._proc.stdin.write.call_args_list
        prompt_msg = json.loads(write_calls[-1][0][0].decode())
        assert all(block["type"] == "text" for block in prompt_msg["params"]["prompt"])
        final = events[-1].response
        assert final is not None and final.success
        assert final.text.startswith("[Note: 1 attached image could not be passed to FakeAcp")
        assert "ok" in final.text

    @pytest.mark.asyncio
    async def test_two_dropped_images_pluralize_notice(self):
        """The notice counts every dropped image in the turn."""
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
            {"type": "text", "text": "compare these"},
            _anthropic_image_block(),
            _anthropic_image_block(data="QUJD"),
        ]
        events = await _collect_events(b, prompt=prompt)

        final = events[-1].response
        assert final is not None
        assert final.text.startswith("[Note: 2 attached images could not be passed to FakeAcp")

    @pytest.mark.asyncio
    async def test_malformed_image_dropped_despite_capability(self):
        """A block that is not the Anthropic base64 shape is dropped
        (with the notice) even when the agent supports images, rather
        than sending a malformed ACP block."""
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
        b._supports_image_input = True

        prompt: list = [
            {"type": "text", "text": "what is in this image"},
            {"type": "image", "source": "fake"},
        ]
        events = await _collect_events(b, prompt=prompt)

        write_calls = b._proc.stdin.write.call_args_list
        prompt_msg = json.loads(write_calls[-1][0][0].decode())
        assert all(block["type"] == "text" for block in prompt_msg["params"]["prompt"])
        final = events[-1].response
        assert final is not None
        assert final.text.startswith("[Note: 1 attached image")

    @pytest.mark.asyncio
    async def test_eof_after_drop_is_error_not_notice_success(self):
        """A process that dies without output after an image drop must
        surface as an error; the notice seed alone is not a reply."""
        b = _make_fake(workspace=Path("/tmp/ws"), home_workspace=Path("/tmp/ws"))
        b._proc = _make_mock_proc([])  # immediate EOF
        b._session_id = "sess-1"
        b._fresh_session = False
        b._next_id = 3

        prompt: list = [
            {"type": "text", "text": "what is in this image"},
            _anthropic_image_block(),
        ]
        events = await _collect_events(b, prompt=prompt)

        final = events[-1].response
        assert final is not None
        assert not final.success
        assert final.error is not None and "ended unexpectedly" in final.error


# ── Image block conversion ───────────────────────────────────────────


class TestConvertImageBlock:
    """`convert_image_block` translates the Anthropic base64 image
    shape (built by the bot layer) into ACP's image content block, and
    returns None on any other shape so the caller drops it."""

    def test_converts_anthropic_base64_block(self):
        assert convert_image_block(_anthropic_image_block()) == {
            "type": "image",
            "mimeType": "image/jpeg",
            "data": "QkFTRTY0",
        }

    def test_non_dict_source_returns_none(self):
        assert convert_image_block({"type": "image", "source": "fake"}) is None

    def test_non_base64_source_type_returns_none(self):
        block = {"type": "image", "source": {"type": "url", "url": "https://x.test/i.png"}}
        assert convert_image_block(block) is None

    def test_missing_data_returns_none(self):
        block = {"type": "image", "source": {"type": "base64", "media_type": "image/png"}}
        assert convert_image_block(block) is None

    def test_missing_media_type_returns_none(self):
        block = {"type": "image", "source": {"type": "base64", "data": "QkFTRTY0"}}
        assert convert_image_block(block) is None


# ── Image capability capture ─────────────────────────────────────────


class TestImageCapabilityCapture:
    """The handshake reads `agentCapabilities.promptCapabilities.image`
    from the initialize result; missing or malformed structures mean
    no image support."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("prompt_capabilities", "expected"),
        [
            ({"image": True}, True),
            ({"image": False}, False),
            ({"embeddedContext": True}, False),
            (None, False),
        ],
    )
    async def test_capability_captured_from_initialize(self, prompt_capabilities, expected):
        b = _make_fake()

        async def _fake_spawn(*args, **kwargs):
            return _make_mock_proc(_handshake_lines(prompt_capabilities=prompt_capabilities))

        with patch("kai.acp.asyncio.create_subprocess_exec", side_effect=_fake_spawn):
            await b._ensure_started()

        assert b._supports_image_input is expected


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
        """Only the principal credential survives workspace env layering."""
        env_file = tmp_path / "ws.env"
        env_file.write_text(
            "KAI_WEBHOOK_SECRET=workspace-secret\n"
            "WEBHOOK_SECRET=legacy-generic-secret\n"
            "GENERIC_WEBHOOK_SECRET=generic-signing-secret\n"
            "GITHUB_WEBHOOK_SECRET=github-signing-secret\n"
            "TELEGRAM_WEBHOOK_SECRET=telegram-signing-secret\n"
            "TELEGRAM_BOT_TOKEN=bot-token\n"
            "GH_TOKEN=outer-github-token\n"
            "FOO=bar\n"
        )
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
        assert "WEBHOOK_SECRET" not in env
        assert "GENERIC_WEBHOOK_SECRET" not in env
        assert "GITHUB_WEBHOOK_SECRET" not in env
        assert "TELEGRAM_WEBHOOK_SECRET" not in env
        assert "TELEGRAM_BOT_TOKEN" not in env
        assert "GH_TOKEN" not in env

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


# ── Shared JSON-RPC free-function tests ──────────────────────────


class TestWriteRpcFreeFunction:
    """`write_rpc` is the wire primitive shared between AcpBackend
    (its _write_rpc wraps it with self._next_id bookkeeping) and the
    one-shot OpenCode reasoner (which holds its own counter). The
    returned value MUST be the incremented id so both callers can
    thread it back into their state."""

    @pytest.mark.asyncio
    async def test_writes_jsonrpc_request_with_supplied_id(self):
        from kai.acp import write_rpc

        proc = MagicMock()
        proc.stdin = MagicMock()
        writes: list[bytes] = []
        proc.stdin.write = writes.append

        async def _drain() -> None:
            return None

        proc.stdin.drain = _drain

        next_id = await write_rpc(
            proc=proc,
            next_id=7,
            method="session/prompt",
            params={"sessionId": "s1", "prompt": []},
        )

        assert next_id == 8
        sent = json.loads(writes[0].decode().rstrip("\n"))
        assert sent == {
            "jsonrpc": "2.0",
            "method": "session/prompt",
            "id": 7,
            "params": {"sessionId": "s1", "prompt": []},
        }


class TestReadResultFreeFunction:
    """`read_result` reads stdout until a matching-id response lands;
    discards notifications, surfaces JSON-RPC errors and process exit
    as RuntimeError, times out via TimeoutError on the supplied bound.
    Shared between AcpBackend and the one-shot reasoner so handshake
    parsing rules cannot drift between callers."""

    @pytest.mark.asyncio
    async def test_returns_result_for_matching_id(self):
        from kai.acp import read_result

        proc = MagicMock()
        proc.stdout = MagicMock()
        lines = [
            (json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"sessionId": "abc"}}) + "\n").encode(),
        ]

        async def _readline() -> bytes:
            return lines.pop(0) if lines else b""

        proc.stdout.readline = _readline

        result = await read_result(
            proc=proc,
            expected_id=1,
            timeout_seconds=5,
            backend_label="OpenCode",
        )

        assert result == {"sessionId": "abc"}

    @pytest.mark.asyncio
    async def test_skips_notifications_until_matching_id(self):
        from kai.acp import read_result

        proc = MagicMock()
        proc.stdout = MagicMock()
        lines = [
            (
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "method": "session/update",
                        "params": {"update": {"sessionUpdate": "agent_thought_chunk"}},
                    }
                )
                + "\n"
            ).encode(),
            (json.dumps({"jsonrpc": "2.0", "id": 99, "result": {}}) + "\n").encode(),
            (json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"final": True}}) + "\n").encode(),
        ]

        async def _readline() -> bytes:
            return lines.pop(0) if lines else b""

        proc.stdout.readline = _readline

        result = await read_result(
            proc=proc,
            expected_id=1,
            timeout_seconds=5,
            backend_label="OpenCode",
        )
        assert result == {"final": True}

    @pytest.mark.asyncio
    async def test_jsonrpc_error_raises_runtime_error_with_label(self):
        from kai.acp import read_result

        proc = MagicMock()
        proc.stdout = MagicMock()
        lines = [
            (json.dumps({"jsonrpc": "2.0", "id": 1, "error": {"code": -1, "message": "rejected"}}) + "\n").encode(),
        ]

        async def _readline() -> bytes:
            return lines.pop(0) if lines else b""

        proc.stdout.readline = _readline

        with pytest.raises(RuntimeError) as excinfo:
            await read_result(
                proc=proc,
                expected_id=1,
                timeout_seconds=5,
                backend_label="TestLabel",
            )
        assert "TestLabel" in str(excinfo.value)
        assert "rejected" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_eof_raises_runtime_error(self):
        from kai.acp import read_result

        proc = MagicMock()
        proc.stdout = MagicMock()

        async def _readline() -> bytes:
            return b""

        proc.stdout.readline = _readline

        with pytest.raises(RuntimeError) as excinfo:
            await read_result(
                proc=proc,
                expected_id=1,
                timeout_seconds=5,
                backend_label="TestLabel",
            )
        assert "exited during handshake" in str(excinfo.value)

    @pytest.mark.asyncio
    async def test_skips_non_json_lines(self):
        from kai.acp import read_result

        proc = MagicMock()
        proc.stdout = MagicMock()
        lines = [
            b"not-json progress bar\n",
            (json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"ok": True}}) + "\n").encode(),
        ]

        async def _readline() -> bytes:
            return lines.pop(0) if lines else b""

        proc.stdout.readline = _readline

        result = await read_result(
            proc=proc,
            expected_id=1,
            timeout_seconds=5,
            backend_label="OpenCode",
        )
        assert result == {"ok": True}


# ── Cross-user os_user routing ──────────────────────────────────────


class TestOsUserRouting:
    """`os_user` controls the sudo wrap in _ensure_started, mirroring
    the claude/codex routing contract: an unset (or bot-matching)
    target spawns the agent directly; a non-bot target wraps the argv
    in `sudo -H -u <target> --preserve-env=<csv> --`, runs the spawn
    in a new session group, and records the escalation target + pgid
    the kill paths need."""

    @pytest.mark.asyncio
    async def test_direct_spawn_when_os_user_none(self):
        """No os_user: argv is the bare build_argv() vector, no new
        session group, no escalation state recorded."""
        b = _make_fake()
        captured: dict = {}

        async def _fake_spawn(*args, **kwargs):
            captured["argv"] = args
            captured.update(kwargs)
            return _make_mock_proc(_handshake_lines())

        with patch("kai.acp.asyncio.create_subprocess_exec", side_effect=_fake_spawn):
            await b._ensure_started()

        assert captured["argv"][0] == "fake_acp_binary"
        assert "sudo" not in captured["argv"]
        assert captured["start_new_session"] is False
        assert b._pgid is None
        assert b._effective_os_user is None

    @pytest.mark.asyncio
    async def test_self_sudo_skip_spawns_direct(self):
        """An os_user that resolve_claude_user collapses to None (the
        bot's own user) takes the byte-identical direct path. The
        resolver is patched so the assertion does not depend on which
        OS user the test runner happens to be."""
        b = _make_fake(os_user="bot-user-itself")
        captured: dict = {}

        async def _fake_spawn(*args, **kwargs):
            captured["argv"] = args
            captured.update(kwargs)
            return _make_mock_proc(_handshake_lines())

        with (
            patch("kai.acp.resolve_claude_user", return_value=None) as mock_resolve,
            patch("kai.acp.asyncio.create_subprocess_exec", side_effect=_fake_spawn),
        ):
            await b._ensure_started()

        mock_resolve.assert_called_once_with("bot-user-itself")
        assert captured["argv"][0] == "fake_acp_binary"
        assert captured["start_new_session"] is False
        assert b._pgid is None
        assert b._effective_os_user is None

    @pytest.mark.asyncio
    async def test_wrap_argv_preserve_env_and_state(self):
        """A non-bot os_user wraps the argv with -H and the default
        preserve list, spawns a new session group, and records pgid
        (== wrapper pid for session leaders) plus the resolved target
        for the kill paths."""
        b = _make_fake(os_user="other-user")
        captured: dict = {}

        async def _fake_spawn(*args, **kwargs):
            captured["argv"] = args
            captured.update(kwargs)
            return _make_mock_proc(_handshake_lines())

        with (
            patch("kai.acp.resolve_claude_user", return_value="other-user"),
            patch("kai.acp.asyncio.create_subprocess_exec", side_effect=_fake_spawn),
        ):
            await b._ensure_started()

        argv = list(captured["argv"])
        assert argv[:4] == ["sudo", "-H", "-u", "other-user"]
        # Preserve-env CSV exact contract: the AcpBackend default
        # (webhook callback auth + per-os-user temp anchor). Backends
        # that need more override preserved_env_vars(); see the
        # hook-driven test below.
        assert argv[4] == "--preserve-env=KAI_WEBHOOK_SECRET,TMPDIR"
        assert argv[5] == "--"
        assert argv[6] == "fake_acp_binary"
        assert captured["start_new_session"] is True
        # _make_mock_proc pins pid=12345; PGID == PID for session
        # leaders, recorded at spawn time.
        assert b._pgid == 12345
        assert b._effective_os_user == "other-user"

    @pytest.mark.asyncio
    async def test_preserve_env_csv_comes_from_hook(self):
        """The CSV is whatever preserved_env_vars() returns, so a
        concrete adapter (GooseBackend) can extend the list without
        the shared layer hardcoding backend specifics."""

        class _CustomPreserve(_FakeAcp):
            def preserved_env_vars(self) -> tuple[str, ...]:
                return ("ALPHA", "BETA")

        b = _CustomPreserve(model="sonnet", workspace=Path("/tmp/test-workspace"), os_user="other-user")
        captured: dict = {}

        async def _fake_spawn(*args, **kwargs):
            captured["argv"] = args
            return _make_mock_proc(_handshake_lines())

        with (
            patch("kai.acp.resolve_claude_user", return_value="other-user"),
            patch("kai.acp.asyncio.create_subprocess_exec", side_effect=_fake_spawn),
        ):
            await b._ensure_started()

        assert captured["argv"][4] == "--preserve-env=ALPHA,BETA"

    @pytest.mark.asyncio
    async def test_tmpdir_anchored_per_os_user_on_wrap(self):
        """Cross-user mode anchors TMPDIR under <DATA_DIR>/tmp/<user>
        so each os_user has its own temp namespace; the anchor is
        applied AFTER workspace env so a workspace cannot point one
        user's temp writes at another's."""
        from kai.config import DATA_DIR

        b = _make_fake(os_user="other-user")
        captured: dict = {}

        async def _fake_spawn(*args, **kwargs):
            captured.update(kwargs)
            return _make_mock_proc(_handshake_lines())

        with (
            patch("kai.acp.resolve_claude_user", return_value="other-user"),
            patch("kai.acp.asyncio.create_subprocess_exec", side_effect=_fake_spawn),
        ):
            await b._ensure_started()

        assert captured["env"]["TMPDIR"] == str(DATA_DIR / "tmp" / "other-user")

    @pytest.mark.asyncio
    async def test_tmpdir_not_anchored_on_direct_spawn(self):
        """Direct mode keeps the inherited TMPDIR (or its absence):
        no cross-user collision is possible when everything runs as
        the bot user, matching the claude backend's contract."""
        import os as _os

        b = _make_fake()
        captured: dict = {}

        async def _fake_spawn(*args, **kwargs):
            captured.update(kwargs)
            return _make_mock_proc(_handshake_lines())

        with patch("kai.acp.asyncio.create_subprocess_exec", side_effect=_fake_spawn):
            await b._ensure_started()

        assert captured["env"].get("TMPDIR") == _os.environ.get("TMPDIR")


# ── Cross-user kill escalation ──────────────────────────────────────


class TestCrossUserKillEscalation:
    """When the spawn is wrapped, _proc is the service-user-owned sudo
    wrapper and the agent underneath is target-user-owned; plain
    SIGKILL on the wrapper would orphan the agent. force_kill, _kill,
    and shutdown therefore escalate a group kill through the target
    user first - and never do so on the direct path, where _proc IS
    the agent."""

    def _wrapped_backend(self) -> tuple[_FakeAcp, MagicMock]:
        """Backend in post-spawn cross-user state, without a real
        spawn: _proc is a live-looking wrapper mock and the routing
        state is what _ensure_started records on the wrap path."""
        b = _make_fake()
        proc = MagicMock()
        proc.returncode = None
        proc.pid = 4242
        proc.kill = MagicMock()
        proc.terminate = MagicMock()
        proc.wait = AsyncMock()
        b._proc = proc
        b._effective_os_user = "other-user"
        b._pgid = 4242
        b._stderr_task = None
        return b, proc

    def test_force_kill_escalates_sync_then_kills_wrapper(self):
        """force_kill (sync, the pool's last resort) runs the sync
        group-kill variant before SIGKILLing the wrapper, and nulls
        _pgid so the async paths that follow (read loop EOF -> _kill)
        do not re-kill the already-dead group."""
        b, proc = self._wrapped_backend()

        with patch("kai.acp._kill_target_user_tree_sync") as mock_sync:
            b.force_kill()

        mock_sync.assert_called_once_with(
            target_user="other-user",
            pgid=4242,
            purpose="chat",
            backend="fake",
        )
        proc.kill.assert_called_once()
        assert b._pgid is None
        # force_kill does NOT null _proc; _kill owns full cleanup.
        assert b._proc is proc

    def test_force_kill_direct_no_escalation(self):
        """Direct mode: no escalation subprocess, just the SIGKILL."""
        b = _make_fake()
        proc = MagicMock()
        proc.kill = MagicMock()
        b._proc = proc
        b._stderr_task = None

        with patch("kai.acp._kill_target_user_tree_sync") as mock_sync:
            b.force_kill()

        mock_sync.assert_not_called()
        proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_kill_escalates_async_once(self):
        """_kill escalates through the canonical async helper, then
        falls through to force_kill - which must NOT re-escalate
        (the _pgid nulling between the two is the dedup)."""
        b, proc = self._wrapped_backend()

        with (
            patch("kai.acp._kill_target_user_tree", new=AsyncMock()) as mock_async,
            patch("kai.acp._kill_target_user_tree_sync") as mock_sync,
        ):
            await b._kill()

        mock_async.assert_awaited_once_with(
            target_user="other-user",
            pgid=4242,
            purpose="chat",
            backend="fake",
        )
        mock_sync.assert_not_called()
        proc.kill.assert_called_once()
        # Full cleanup: a recycled instance must not carry stale
        # routing state into a fresh spawn.
        assert b._proc is None
        assert b._pgid is None
        assert b._effective_os_user is None

    @pytest.mark.asyncio
    async def test_kill_direct_no_escalation(self):
        """Direct mode _kill: SIGKILL + reap only."""
        b = _make_fake()
        proc = MagicMock()
        proc.returncode = None
        proc.kill = MagicMock()
        proc.wait = AsyncMock()
        b._proc = proc
        b._stderr_task = None

        with patch("kai.acp._kill_target_user_tree", new=AsyncMock()) as mock_async:
            await b._kill()

        mock_async.assert_not_awaited()
        proc.kill.assert_called_once()
        assert b._proc is None

    @pytest.mark.asyncio
    async def test_response_timeout_escalates_when_wrapped(self):
        """The in-stream timeout path (response/idle timeouts both
        funnel into _kill) escalates for a wrapped backend, so an
        eviction or hung turn cannot orphan the target-user agent."""
        b, proc = self._wrapped_backend()
        b.timeout_seconds = 1
        proc.stdout = MagicMock()
        proc.stdout.readline = AsyncMock(side_effect=TimeoutError())
        proc.stdin = MagicMock()
        proc.stdin.write = MagicMock()
        proc.stdin.drain = AsyncMock()
        b._session_id = "sess-1"
        b._fresh_session = False
        b._next_id = 3

        with patch("kai.acp._kill_target_user_tree", new=AsyncMock()) as mock_async:
            events = await _collect_events(b, prompt="hi")

        assert events[-1].done is True
        assert events[-1].response is not None
        assert events[-1].response.success is False
        mock_async.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_graceful_path_no_escalation(self):
        """SIGTERM lands on the sudo wrapper, which relays catchable
        signals to the agent; when the wait succeeds there is nothing
        left to escalate."""
        b, proc = self._wrapped_backend()
        proc.wait = AsyncMock(return_value=None)

        with patch("kai.acp._kill_target_user_tree", new=AsyncMock()) as mock_async:
            await b.shutdown()

        mock_async.assert_not_awaited()
        proc.terminate.assert_called_once()
        proc.kill.assert_not_called()
        assert b._proc is None
        assert b._pgid is None
        assert b._effective_os_user is None

    @pytest.mark.asyncio
    async def test_shutdown_timeout_escalates_before_sigkill(self):
        """When SIGTERM is ignored, the group escalation runs BEFORE
        force_kill reaps the wrapper, mirroring the one-shot
        reasoners' kill ordering (target tree first, wrapper reap
        after). The sync variant must not fire: the _pgid nulling
        between escalation and force_kill is the dedup."""
        b, proc = self._wrapped_backend()
        proc.wait = AsyncMock(side_effect=[TimeoutError(), None])
        order: list[str] = []

        async def _fake_tree_kill(**kwargs) -> None:
            order.append("escalate")

        proc.kill = MagicMock(side_effect=lambda: order.append("sigkill"))

        with (
            patch("kai.acp._kill_target_user_tree", side_effect=_fake_tree_kill),
            patch("kai.acp._kill_target_user_tree_sync") as mock_sync,
        ):
            await b.shutdown()

        assert order == ["escalate", "sigkill"]
        mock_sync.assert_not_called()
        assert b._proc is None


# ── Cross-user kill helper free functions ───────────────────────────


class TestKillTargetUserTreeFreeFunctions:
    """The canonical cross-user group-kill helpers shared by the
    one-shot reasoners and the conversational AcpBackend. Argv shape,
    bounded wait, and the rc/ESRCH log classification (benign
    already-gone race demotes to DEBUG; real failures keep WARNING)."""

    @pytest.mark.asyncio
    async def test_async_kill_argv_shape(self):
        from kai.acp import _kill_target_user_tree

        kill_proc = MagicMock()
        kill_proc.communicate = AsyncMock(return_value=(b"", b""))
        kill_proc.returncode = 0
        captured: dict = {}

        async def _fake_exec(*args, **kwargs):
            captured["argv"] = args
            return kill_proc

        with patch("kai.acp.asyncio.create_subprocess_exec", side_effect=_fake_exec):
            await _kill_target_user_tree(
                target_user="other-user",
                pgid=777,
                purpose="chat",
                backend="goose",
            )

        # Negative-PGID group kill through the target user's /bin/kill
        # sudoers permission; -n so a missing rule fails fast instead
        # of prompting.
        assert list(captured["argv"]) == ["sudo", "-n", "-u", "other-user", "/bin/kill", "-KILL", "-777"]

    @pytest.mark.asyncio
    async def test_async_kill_esrch_demotes_to_debug(self, caplog):
        """rc=1 with the POSIX ESRCH diagnostic is the benign race
        (tree already exited); it must not pollute the WARNING
        stream the operator watches for real sudoers failures."""
        from kai.acp import _kill_target_user_tree

        kill_proc = MagicMock()
        kill_proc.communicate = AsyncMock(return_value=(b"", b"kill: -777: No such process"))
        kill_proc.returncode = 1

        with (
            patch("kai.acp.asyncio.create_subprocess_exec", AsyncMock(return_value=kill_proc)),
            caplog.at_level(logging.DEBUG, logger="kai.acp"),
        ):
            await _kill_target_user_tree(
                target_user="other-user",
                pgid=777,
                purpose="chat",
                backend="goose",
            )

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warnings
        debugs = [r for r in caplog.records if "benign race" in r.getMessage()]
        assert debugs

    @pytest.mark.asyncio
    async def test_async_kill_real_failure_keeps_warning(self, caplog):
        """A non-ESRCH rc!=0 (sudoers misconfiguration, permission
        denied) keeps the WARNING so the orphan risk is visible."""
        from kai.acp import _kill_target_user_tree

        kill_proc = MagicMock()
        kill_proc.communicate = AsyncMock(return_value=(b"", b"sudo: a password is required"))
        kill_proc.returncode = 1

        with (
            patch("kai.acp.asyncio.create_subprocess_exec", AsyncMock(return_value=kill_proc)),
            caplog.at_level(logging.WARNING, logger="kai.acp"),
        ):
            await _kill_target_user_tree(
                target_user="other-user",
                pgid=777,
                purpose="chat",
                backend="goose",
            )

        warnings = [r for r in caplog.records if "cross-user kill returned rc=1" in r.getMessage()]
        assert warnings

    def test_sync_kill_argv_and_esrch_classification(self, caplog):
        """The sync companion (force_kill path) shares the argv shape
        and the ESRCH demotion with the async canonical helper."""
        from kai.acp import _kill_target_user_tree_sync

        captured: dict = {}

        def _fake_run(cmd, **kwargs):
            captured["argv"] = cmd
            result = MagicMock()
            result.returncode = 1
            result.stderr = b"kill: -888: No such process"
            return result

        with (
            patch("kai.acp.subprocess.run", side_effect=_fake_run),
            caplog.at_level(logging.DEBUG, logger="kai.acp"),
        ):
            _kill_target_user_tree_sync(
                target_user="other-user",
                pgid=888,
                purpose="chat",
                backend="goose",
            )

        assert captured["argv"] == ["sudo", "-n", "-u", "other-user", "/bin/kill", "-KILL", "-888"]
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warnings
        debugs = [r for r in caplog.records if "benign race" in r.getMessage()]
        assert debugs

    def test_sync_kill_timeout_logs_warning(self, caplog):
        """A hung sudo is bounded and logged; the caller's cleanup
        must not stall behind the escalation."""
        import subprocess as _subprocess

        from kai.acp import _kill_target_user_tree_sync

        with (
            patch(
                "kai.acp.subprocess.run",
                side_effect=_subprocess.TimeoutExpired(cmd="sudo", timeout=5.0),
            ),
            caplog.at_level(logging.WARNING, logger="kai.acp"),
        ):
            _kill_target_user_tree_sync(
                target_user="other-user",
                pgid=999,
                purpose="chat",
                backend="goose",
            )

        warnings = [r for r in caplog.records if "cross-user kill timed out" in r.getMessage()]
        assert warnings


class TestSignalTargetUserPidFreeFunctions:
    """The per-PID cross-user signal helpers the claude and codex
    backends delegate to for inner-agent PID targeting. Same core as
    the group-kill helpers (argv shape, bounded wait, rc/ESRCH
    classification); the kill target is `-<sig> <pid>` instead of
    `-KILL -<pgid>`."""

    def test_stderr_is_esrch_classifier(self):
        """Pure unit test of the shared discriminator. Matching
        substring is the POSIX `strerror(ESRCH)` text; everything
        else returns False, including `None` and empty bytes (a
        missing-stderr rc!=0 is itself unusual and should keep the
        WARNING)."""
        from kai.acp import _stderr_is_esrch

        # Matching: the exact macOS/Linux /bin/kill diagnostic, and
        # the bare substring regardless of prefix shape.
        assert _stderr_is_esrch(b"kill: 12345: No such process\n") is True
        assert _stderr_is_esrch(b"No such process") is True
        # Non-matching: real failure modes that must keep the WARNING.
        assert _stderr_is_esrch(b"a password is required") is False
        assert _stderr_is_esrch(b"kill: Operation not permitted") is False
        assert _stderr_is_esrch(b"") is False
        assert _stderr_is_esrch(None) is False

    @pytest.mark.asyncio
    async def test_async_signal_argv_shape_with_int_sig_cast(self):
        """Argv anchors /bin/kill (matching the sudoers rule) and
        renders the signal as a bare integer: IntEnum __format__
        produced "Signals.SIGTERM" on pre-3.11 Pythons, which
        /bin/kill rejects silently, so the int() cast is the
        contract regardless of version. Passing the enum itself
        exercises the cast."""
        from kai.acp import _signal_target_user_pid

        kill_proc = MagicMock()
        kill_proc.communicate = AsyncMock(return_value=(b"", b""))
        kill_proc.returncode = 0
        captured: dict = {}

        async def _fake_exec(*args, **kwargs):
            captured["argv"] = args
            return kill_proc

        with patch("kai.acp.asyncio.create_subprocess_exec", side_effect=_fake_exec):
            await _signal_target_user_pid(
                target_user="other-user",
                pid=4242,
                sig=signal.SIGTERM,
                purpose="chat",
                backend="claude_code",
            )

        assert list(captured["argv"]) == [
            "sudo",
            "-n",
            "-u",
            "other-user",
            "/bin/kill",
            f"-{int(signal.SIGTERM)}",
            "4242",
        ]

    @pytest.mark.asyncio
    async def test_async_signal_esrch_demotes_to_debug(self, caplog):
        """rc=1 with the POSIX ESRCH diagnostic is the benign race
        (inner PID already exited); it must not pollute the WARNING
        stream the operator watches for real sudoers failures."""
        from kai.acp import _signal_target_user_pid

        kill_proc = MagicMock()
        kill_proc.communicate = AsyncMock(return_value=(b"", b"kill: 4242: No such process"))
        kill_proc.returncode = 1

        with (
            patch("kai.acp.asyncio.create_subprocess_exec", AsyncMock(return_value=kill_proc)),
            caplog.at_level(logging.DEBUG, logger="kai.acp"),
        ):
            await _signal_target_user_pid(
                target_user="other-user",
                pid=4242,
                sig=int(signal.SIGKILL),
                purpose="chat",
                backend="claude_code",
            )

        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warnings
        debugs = [r for r in caplog.records if "benign race" in r.getMessage()]
        assert debugs

    def test_sync_signal_argv_and_esrch_classification(self, caplog):
        """The sync companion (force_kill path) shares the argv shape
        and the ESRCH demotion with the async canonical helper."""
        from kai.acp import _signal_target_user_pid_sync

        captured: dict = {}

        def _fake_run(cmd, **kwargs):
            captured["argv"] = cmd
            result = MagicMock()
            result.returncode = 1
            result.stderr = b"kill: 4242: No such process"
            return result

        with (
            patch("kai.acp.subprocess.run", side_effect=_fake_run),
            caplog.at_level(logging.DEBUG, logger="kai.acp"),
        ):
            _signal_target_user_pid_sync(
                target_user="other-user",
                pid=4242,
                sig=int(signal.SIGKILL),
                purpose="chat",
                backend="codex",
            )

        assert captured["argv"] == [
            "sudo",
            "-n",
            "-u",
            "other-user",
            "/bin/kill",
            f"-{int(signal.SIGKILL)}",
            "4242",
        ]
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert not warnings
        debugs = [r for r in caplog.records if "benign race" in r.getMessage()]
        assert debugs


class TestSessionAgeRecycling:
    """Session-age recycling on the shared ACP layer. The age helpers
    (_session_age_hours / _should_recycle) are inherited from
    AgentBackend; these tests pin that AcpBackend wires the surface
    (max_session_hours kwarg, _session_started_at stamp on handshake,
    null on kill) and that _send_locked kills an expired subprocess
    before _ensure_started respawns it."""

    def test_should_recycle_expired_session(self):
        b = _make_fake(max_session_hours=4)
        proc = MagicMock()
        proc.returncode = None
        b._proc = proc
        b._session_started_at = time.monotonic() - 18000  # 5 hours
        assert b._should_recycle() is True

    def test_should_recycle_disabled_by_default(self):
        """max_session_hours defaults to 0 (no limit); an old live
        session is never recycled unless the pool passes a limit."""
        b = _make_fake()
        proc = MagicMock()
        proc.returncode = None
        b._proc = proc
        b._session_started_at = time.monotonic() - 999999
        assert b._should_recycle() is False

    @pytest.mark.asyncio
    async def test_handshake_stamps_session_start_and_kill_nulls_it(self):
        """_ensure_started records time.monotonic() so the age helpers
        have a base; _kill nulls it so a recycled instance does not
        inherit a stale age."""
        b = _make_fake()
        proc = _make_mock_proc(_handshake_lines())
        with patch("kai.acp.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            await b._ensure_started()
        assert b._session_started_at is not None
        await b._kill()
        assert b._session_started_at is None

    @pytest.mark.asyncio
    async def test_send_recycles_expired_before_ensure_started(self):
        """_send_locked() kills the expired process before
        _ensure_started() respawns it (the claude backend's recycle
        contract, minus the save-prompt ACP has no equivalent of)."""
        b = _make_fake(max_session_hours=1)

        async def fake_ensure_started():
            mock_proc = MagicMock()
            mock_proc.returncode = None
            mock_proc.stdin = MagicMock()
            mock_proc.stdin.write = MagicMock()
            mock_proc.stdin.drain = AsyncMock()
            mock_proc.stdout = MagicMock()
            mock_proc.stdout.readline = AsyncMock(return_value=b"")  # EOF
            b._proc = mock_proc
            b._session_id = "sess-recycled"
            b._fresh_session = False

        with (
            patch.object(b, "_should_recycle", return_value=True),
            patch.object(b, "_kill", new_callable=AsyncMock) as mock_kill,
            patch.object(b, "_ensure_started", side_effect=fake_ensure_started),
            patch.object(b, "_session_age_hours", return_value=2.5),
        ):
            events = []
            async for event in b._send_locked("test"):
                events.append(event)

        # _kill fires at least once for the recycle (and again from
        # the streaming loop's EOF handler, which is expected).
        assert mock_kill.await_count >= 1


class TestStreamLineLimit:
    """The asyncio.StreamReader limit on the ACP subprocess stdout."""

    @pytest.mark.asyncio
    async def test_subprocess_limit_is_at_least_16mb(self):
        """
        A single notification line must fit any plausible payload; a
        1MB ceiling is too tight for PR-review-sized tool results
        (readline raises "Separator is not found, and chunk exceed
        the limit", killing the turn). Lock the lower bound, matching
        the codex backend's pin, so a future shrinkback gets caught
        here rather than on a live operator turn.
        """
        b = _make_fake()
        proc = _make_mock_proc(_handshake_lines())

        with patch("kai.acp.asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await b._ensure_started()

        call_kwargs = mock_exec.call_args[1]
        assert call_kwargs["limit"] >= 16 * 1024 * 1024
