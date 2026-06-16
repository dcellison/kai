"""
Tests for goose.py Goose ACP subprocess backend.

Covers:
1. Startup handshake sequence (initialize + session/new)
2. Stream parsing: agent_message_chunk accumulation, other events skipped
3. Completion event yields done StreamEvent
4. Fresh session context injection (first send only)
5. restart() triggers new handshake on next send
6. force_kill() during active send
7. JSON-RPC error response handling
8. Model name mapping
"""

import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from kai.backend import USER_MESSAGE_MARKER, StreamEvent
from kai.config import WorkspaceConfig
from kai.goose import _ANTHROPIC_MODEL_MAP, GooseBackend, goose_provider_id

# ── Shared helpers ───────────────────────────────────────────────────


def _make_goose(**kwargs) -> GooseBackend:
    """Create a GooseBackend with sensible defaults for testing."""
    defaults = {
        "model": "sonnet",
        "workspace": Path("/tmp/test-workspace"),
        "timeout_seconds": 30,
    }
    defaults.update(kwargs)
    return GooseBackend(**defaults)


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
                "protocolVersion": 0,
                "agentCapabilities": {
                    "loadSession": True,
                    "promptCapabilities": {"image": True, "audio": False},
                },
            },
        }
    )


def _session_new_result(session_id: str = "20260406_01") -> bytes:
    """Build the server's response to a session/new request."""
    return _json_line(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "result": {
                "sessionId": session_id,
                "modes": {},
                "models": {},
            },
        }
    )


def _agent_message_chunk(text: str, session_id: str = "20260406_01") -> bytes:
    """Build a session/update notification with agent_message_chunk."""
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


def _agent_thought_chunk(text: str, session_id: str = "20260406_01") -> bytes:
    """Build a session/update notification with agent_thought_chunk (should be skipped)."""
    return _json_line(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "agent_thought_chunk",
                    "content": {"type": "text", "text": text},
                },
            },
        }
    )


def _tool_call_event(session_id: str = "20260406_01") -> bytes:
    """Build a session/update notification with tool_call (should be skipped)."""
    return _json_line(
        {
            "jsonrpc": "2.0",
            "method": "session/update",
            "params": {
                "sessionId": session_id,
                "update": {
                    "sessionUpdate": "tool_call",
                    "toolCallId": "tc-1",
                    "title": "Read file",
                },
            },
        }
    )


def _completion_result(prompt_id: int = 3) -> bytes:
    """Build a completion result (end_turn) for a given prompt id."""
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


def _handshake_lines(session_id: str = "20260406_01") -> list[bytes]:
    """Return the two stdout lines for a successful handshake."""
    return [_initialize_result(), _session_new_result(session_id)]


async def _collect_events(goose: GooseBackend, prompt: str | list = "test") -> list[StreamEvent]:
    """Send a prompt and collect all yielded StreamEvents."""
    events = []
    async for event in goose._send_locked(prompt):
        events.append(event)
    return events


# ── Model mapping ──────────────────────────────────────────────────


class TestModelMapping:
    """Verify _ANTHROPIC_MODEL_MAP translates logical names to provider IDs."""

    def test_known_models(self):
        """Logical names map to the current Anthropic API aliases (the
        same SKUs the claude CLI resolves its short aliases to)."""
        assert _ANTHROPIC_MODEL_MAP["sonnet"] == "claude-sonnet-4-6"
        assert _ANTHROPIC_MODEL_MAP["opus"] == "claude-opus-4-8"
        assert _ANTHROPIC_MODEL_MAP["haiku"] == "claude-haiku-4-5"

    def test_passthrough_for_unknown(self):
        """Unrecognized values pass through via .get(key, key) fallback,
        so full model IDs can pin a SKU the map does not list (e.g. the
        previous-generation opus)."""
        model_id = "claude-opus-4-7"
        assert _ANTHROPIC_MODEL_MAP.get(model_id, model_id) == model_id


class TestProviderTranslation:
    """`goose_provider_id` maps Kai provider keys to goose's wire-level
    provider names; everything without a mapping passes through."""

    def test_deepseek_maps_to_custom_deepseek(self):
        assert goose_provider_id("deepseek") == "custom_deepseek"

    @pytest.mark.parametrize("provider", ["anthropic", "openai", "google", "openrouter", "ollama"])
    def test_other_providers_pass_through(self, provider):
        assert goose_provider_id(provider) == provider


# ── Constructor ────────────────────────────────────────────────────


class TestConstructor:
    """Verify GooseBackend initializes attributes correctly."""

    def test_defaults(self):
        """Constructor sets ABC-required attributes from kwargs."""
        g = _make_goose()
        assert g.model == "sonnet"
        assert g.workspace == Path("/tmp/test-workspace")
        assert g.timeout_seconds == 30
        assert g._proc is None
        assert g._session_id is None
        assert g._fresh_session is True

    def test_workspace_config_overrides(self):
        """Per-workspace config overrides model and timeout."""
        ws = WorkspaceConfig(path=Path("/tmp/ws"), model="opus", timeout=60)
        g = _make_goose(workspace_config=ws)
        assert g.model == "opus"
        assert g.timeout_seconds == 60
        # Defaults preserved for restore on workspace switch
        assert g._default_model == "sonnet"
        assert g._default_timeout == 30

    def test_home_workspace_defaults_to_workspace(self):
        """home_workspace falls back to workspace when not provided."""
        g = _make_goose(workspace=Path("/some/path"))
        assert g.home_workspace == Path("/some/path")


# ── Properties ─────────────────────────────────────────────────────


class TestProperties:
    """Verify is_alive and session_id properties."""

    def test_is_alive_no_proc(self):
        """is_alive returns False when no subprocess exists."""
        g = _make_goose()
        assert g.is_alive is False

    def test_is_alive_running(self):
        """is_alive returns True when subprocess has no returncode."""
        g = _make_goose()
        g._proc = MagicMock()
        g._proc.returncode = None
        assert g.is_alive is True

    def test_is_alive_exited(self):
        """is_alive returns False when subprocess has exited."""
        g = _make_goose()
        g._proc = MagicMock()
        g._proc.returncode = 0
        assert g.is_alive is False

    def test_session_id_none(self):
        """session_id is None before handshake."""
        g = _make_goose()
        assert g.session_id is None


# ── Handshake ──────────────────────────────────────────────────────


class TestHandshake:
    """Verify _ensure_started() runs the initialize + session/new handshake."""

    @pytest.mark.asyncio
    async def test_successful_handshake(self):
        """Handshake sets _session_id from session/new result."""
        g = _make_goose()
        proc = _make_mock_proc(_handshake_lines("test-session-42"))

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)):
            await g._ensure_started()

        assert g._session_id == "test-session-42"
        assert g._fresh_session is True
        assert g.is_alive is True

        # Verify the two handshake messages were written
        writes = proc.stdin.write.call_args_list
        assert len(writes) == 2

        # First write: initialize
        init_msg = json.loads(writes[0][0][0].decode())
        assert init_msg["method"] == "initialize"
        assert init_msg["id"] == 1
        assert init_msg["params"]["clientInfo"]["name"] == "kai"

        # Second write: session/new
        session_msg = json.loads(writes[1][0][0].decode())
        assert session_msg["method"] == "session/new"
        assert session_msg["id"] == 2
        assert session_msg["params"]["mcpServers"] == []

    @pytest.mark.asyncio
    async def test_skips_when_alive(self):
        """_ensure_started() is a no-op when the process is already running."""
        g = _make_goose()
        g._proc = MagicMock()
        g._proc.returncode = None

        # Should return immediately without spawning
        await g._ensure_started()
        # No new process created - _session_id unchanged
        assert g._proc.returncode is None

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
        g = _make_goose()

        with (
            patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            pytest.raises(RuntimeError, match="bad request"),
        ):
            await g._ensure_started()

    @pytest.mark.asyncio
    async def test_handshake_eof_raises(self):
        """EOF during handshake raises RuntimeError."""
        proc = _make_mock_proc([b""])  # Immediate EOF
        g = _make_goose()

        with (
            patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)),
            pytest.raises(RuntimeError, match="exited during handshake"),
        ):
            await g._ensure_started()

    @pytest.mark.asyncio
    async def test_model_env_var_set(self):
        """GOOSE_MODEL env var is set during startup."""
        g = _make_goose(model="opus", provider="anthropic")
        proc = _make_mock_proc(_handshake_lines())

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await g._ensure_started()

        call_kwargs = mock_exec.call_args[1]
        assert call_kwargs["env"]["GOOSE_MODEL"] == "claude-opus-4-8"

    @pytest.mark.asyncio
    async def test_model_passthrough_for_non_anthropic(self):
        """Non-Anthropic providers pass model value through unchanged."""
        g = _make_goose(model="sonnet", provider="openai")
        proc = _make_mock_proc(_handshake_lines())

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await g._ensure_started()

        call_kwargs = mock_exec.call_args[1]
        # "sonnet" passed through as-is, not mapped to "claude-sonnet-4-6"
        assert call_kwargs["env"]["GOOSE_MODEL"] == "sonnet"

    @pytest.mark.asyncio
    async def test_provider_env_var_set(self):
        """
        GOOSE_PROVIDER env var is set during startup.

        The goose binary reads GOOSE_PROVIDER to pick which provider API
        to call; without it, session/new fails with "Internal error" on
        any fresh install that has no provider configured in
        ~/.config/goose/config.yaml. This test guards the translation
        from Kai's self.provider field to the goose-prefixed env var.
        """
        g = _make_goose(model="gpt-5.4-mini", provider="openai")
        proc = _make_mock_proc(_handshake_lines())

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await g._ensure_started()

        call_kwargs = mock_exec.call_args[1]
        assert call_kwargs["env"]["GOOSE_PROVIDER"] == "openai"

    @pytest.mark.asyncio
    async def test_deepseek_provider_translated_to_wire_name(self):
        """
        Kai's "deepseek" key becomes goose's "custom_deepseek" wire
        name in GOOSE_PROVIDER. Goose ships DeepSeek as a declarative
        provider under that name and rejects the bare key with
        "Unknown provider: deepseek" before any API call.
        """
        g = _make_goose(model="deepseek-v4-pro", provider="deepseek")
        proc = _make_mock_proc(_handshake_lines())

        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec:
            await g._ensure_started()

        call_kwargs = mock_exec.call_args[1]
        assert call_kwargs["env"]["GOOSE_PROVIDER"] == "custom_deepseek"
        # Model names are NOT translated; only the provider name is.
        assert call_kwargs["env"]["GOOSE_MODEL"] == "deepseek-v4-pro"

    @pytest.mark.asyncio
    async def test_provider_env_var_omitted_when_empty(self):
        """
        GOOSE_PROVIDER is not exported when self.provider is empty.

        Exporting GOOSE_PROVIDER="" would confuse the goose binary more
        than omitting it; the empty default exists for the claude backend
        pre-wiring path and any test fixtures that do not specify a
        provider. The guard mirrors GOOSE_MODEL's truthiness check.
        """
        g = _make_goose(model="sonnet", provider="")
        proc = _make_mock_proc(_handshake_lines())

        # Scrub GOOSE_PROVIDER from the inherited parent environment so the
        # absence is attributable to the truthiness guard, not to the host
        # shell. Without this scrub the test would silently pass on hosts
        # that happen to export GOOSE_PROVIDER and silently fail on hosts
        # that do not, making the regression invisible to local runs.
        with (
            patch.dict(os.environ, {}, clear=False) as _env,
            patch("asyncio.create_subprocess_exec", AsyncMock(return_value=proc)) as mock_exec,
        ):
            os.environ.pop("GOOSE_PROVIDER", None)
            await g._ensure_started()

        call_kwargs = mock_exec.call_args[1]
        assert "GOOSE_PROVIDER" not in call_kwargs["env"]


# ── Stream parsing ─────────────────────────────────────────────────


class TestStreamParsing:
    """Verify _send_locked() correctly parses streaming responses."""

    @pytest.mark.asyncio
    async def test_agent_message_chunk_accumulation(self):
        """agent_message_chunk events accumulate text and yield StreamEvents."""
        g = _make_goose()
        g._proc = _make_mock_proc(
            [
                _agent_message_chunk("Hello"),
                _agent_message_chunk(" world"),
                _completion_result(prompt_id=3),
            ]
        )
        g._session_id = "test-session"
        g._fresh_session = False
        g._next_id = 3  # Next prompt will use id=3

        events = await _collect_events(g)

        # Two streaming events + one done event
        assert len(events) == 3
        assert events[0].text_so_far == "Hello"
        assert events[1].text_so_far == "Hello world"
        assert events[2].done is True
        assert events[2].text_so_far == "Hello world"
        assert events[2].response is not None
        assert events[2].response.success is True
        assert events[2].response.text == "Hello world"

    @pytest.mark.asyncio
    async def test_thought_chunks_skipped(self):
        """agent_thought_chunk events are silently skipped."""
        g = _make_goose()
        g._proc = _make_mock_proc(
            [
                _agent_thought_chunk("thinking..."),
                _agent_message_chunk("result"),
                _completion_result(prompt_id=3),
            ]
        )
        g._session_id = "test-session"
        g._fresh_session = False
        g._next_id = 3

        events = await _collect_events(g)

        # Only the message chunk and completion, not the thought
        assert len(events) == 2
        assert events[0].text_so_far == "result"
        assert events[1].done is True

    @pytest.mark.asyncio
    async def test_tool_call_events_skipped(self):
        """tool_call events are silently skipped."""
        g = _make_goose()
        g._proc = _make_mock_proc(
            [
                _tool_call_event(),
                _agent_message_chunk("done"),
                _completion_result(prompt_id=3),
            ]
        )
        g._session_id = "test-session"
        g._fresh_session = False
        g._next_id = 3

        events = await _collect_events(g)

        assert len(events) == 2
        assert events[0].text_so_far == "done"

    @pytest.mark.asyncio
    async def test_non_json_lines_skipped(self):
        """Non-JSON stdout lines are silently skipped."""
        g = _make_goose()
        g._proc = _make_mock_proc(
            [
                b"some random output\n",
                _agent_message_chunk("hello"),
                _completion_result(prompt_id=3),
            ]
        )
        g._session_id = "test-session"
        g._fresh_session = False
        g._next_id = 3

        events = await _collect_events(g)

        assert len(events) == 2
        assert events[0].text_so_far == "hello"


# ── Completion ─────────────────────────────────────────────────────


class TestCompletion:
    """Verify final result handling."""

    @pytest.mark.asyncio
    async def test_completion_yields_done_event(self):
        """A result with matching id yields a done StreamEvent with AgentResponse."""
        g = _make_goose()
        g._proc = _make_mock_proc(
            [
                _agent_message_chunk("answer"),
                _completion_result(prompt_id=3),
            ]
        )
        g._session_id = "test-session"
        g._fresh_session = False
        g._next_id = 3

        events = await _collect_events(g)
        final = events[-1]

        assert final.done is True
        assert final.response is not None
        assert final.response.success is True
        assert final.response.text == "answer"
        assert final.response.session_id == "test-session"
        assert final.response.duration_ms == 0

    @pytest.mark.asyncio
    async def test_eof_with_accumulated_text(self):
        """EOF after some text yields a success response with the accumulated text."""
        g = _make_goose()
        g._proc = _make_mock_proc(
            [
                _agent_message_chunk("partial"),
                b"",  # EOF
            ]
        )
        g._session_id = "test-session"
        g._fresh_session = False
        g._next_id = 3

        events = await _collect_events(g)
        final = events[-1]

        assert final.done is True
        assert final.response is not None
        # EOF with accumulated text is treated as success (same as CC)
        assert final.response.success is True
        assert final.response.text == "partial"

    @pytest.mark.asyncio
    async def test_eof_without_text(self):
        """EOF with no accumulated text yields an error response."""
        g = _make_goose()
        g._proc = _make_mock_proc([b""])  # Immediate EOF
        g._session_id = "test-session"
        g._fresh_session = False
        g._next_id = 3

        events = await _collect_events(g)
        final = events[-1]

        assert final.done is True
        assert final.response is not None
        assert final.response.success is False
        assert "unexpectedly" in final.response.error


# ── Context injection ──────────────────────────────────────────────


class TestContextInjection:
    """Verify session context is prepended on first send only."""

    @pytest.mark.asyncio
    async def test_fresh_session_injects_context(self):
        """First send prepends session context to the prompt."""
        g = _make_goose(
            workspace=Path("/tmp/ws"),
            home_workspace=Path("/tmp/ws"),
            webhook_secret="test-secret",
        )
        g._proc = _make_mock_proc(
            [
                _agent_message_chunk("ok"),
                _completion_result(prompt_id=3),
            ]
        )
        g._session_id = "test-session"
        g._fresh_session = True
        g._next_id = 3

        # Patch build_session_context to return a known prefix. After the
        # ACP extraction, the call site lives in kai.acp, not kai.goose.
        with patch("kai.acp.build_session_context", return_value="[CONTEXT]"):
            await _collect_events(g, prompt="hello")

        # Verify the written prompt includes the context prefix
        write_calls = g._proc.stdin.write.call_args_list
        # The session/prompt write is the last one
        prompt_msg = json.loads(write_calls[-1][0][0].decode())
        prompt_text = prompt_msg["params"]["prompt"][0]["text"]
        assert prompt_text.startswith("[CONTEXT]")
        assert "hello" in prompt_text

    @pytest.mark.asyncio
    async def test_second_send_no_context(self):
        """Second send does NOT prepend session context."""
        g = _make_goose(
            workspace=Path("/tmp/ws"),
            home_workspace=Path("/tmp/ws"),
        )
        g._proc = _make_mock_proc(
            [
                _agent_message_chunk("ok"),
                _completion_result(prompt_id=3),
            ]
        )
        g._session_id = "test-session"
        g._fresh_session = False  # Already sent first message
        g._next_id = 3

        with patch("kai.acp.build_session_context") as mock_ctx:
            await _collect_events(g, prompt="second")

        # build_session_context should NOT be called
        mock_ctx.assert_not_called()

    @pytest.mark.asyncio
    async def test_foreign_workspace_reminder(self):
        """Foreign workspace adds per-message reminder."""
        g = _make_goose(
            workspace=Path("/tmp/foreign"),
            home_workspace=Path("/tmp/home"),
        )
        g._proc = _make_mock_proc(
            [
                _agent_message_chunk("ok"),
                _completion_result(prompt_id=3),
            ]
        )
        g._session_id = "test-session"
        g._fresh_session = False
        g._next_id = 3

        await _collect_events(g, prompt="hello")

        # The prompt should include the foreign workspace reminder
        write_calls = g._proc.stdin.write.call_args_list
        prompt_msg = json.loads(write_calls[-1][0][0].decode())
        # With a foreign workspace, prompt is a list with reminder
        # block prepended
        prompt_texts = [b["text"] for b in prompt_msg["params"]["prompt"]]
        combined = " ".join(prompt_texts)
        assert "IMPORTANT" in combined
        assert "hello" in combined

    @pytest.mark.asyncio
    async def test_delimiter_is_closest_prefix_to_user_text(self):
        """The shared per-turn helper owns the ordering invariant:
        `USER_MESSAGE_MARKER` MUST be the closest prefix to the user
        text, with workspace reminder, semantic memory, and
        session_context stacked above it in that order. The bug this
        guards against is the marker landing at the TOP of the
        assembled prompt instead of immediately above the user text.
        Claude and codex have the same regression guard; keeping the
        goose copy in lockstep prevents any path from drifting
        silently.
        """
        g = _make_goose(
            workspace=Path("/tmp/foreign"),
            home_workspace=Path("/tmp/home"),
            webhook_secret="test-secret",
        )
        g._proc = _make_mock_proc(
            [
                _agent_message_chunk("ok"),
                _completion_result(prompt_id=3),
            ]
        )
        g._session_id = "test-session"
        g._fresh_session = True
        g._next_id = 3

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
            async for _event in g._send_locked("ACTUAL_USER_TEXT", chat_id=42):
                pass

        write_calls = g._proc.stdin.write.call_args_list
        prompt_msg = json.loads(write_calls[-1][0][0].decode())
        # String prompts coerce to a single text block on goose.
        assert len(prompt_msg["params"]["prompt"]) == 1
        prompt_text = prompt_msg["params"]["prompt"][0]["text"]

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

    @pytest.mark.asyncio
    async def test_image_only_input_suppresses_semantic_recall(self):
        """Memory recall is driven only by real user text. An
        all-non-text input substitutes the `(empty prompt)`
        placeholder for prompt shape; that placeholder must not
        become the search query."""
        g = _make_goose(webhook_secret="test-secret")
        g._proc = _make_mock_proc([_completion_result(prompt_id=3)])
        g._session_id = "test-session"
        g._fresh_session = False
        g._next_id = 3

        from kai.memory import ScopedRecallResult

        recall_spy = AsyncMock(
            return_value=ScopedRecallResult(
                rendered_context="should-not-be-injected", recall_payload={"reason": "ok", "hits": []}
            )
        )
        blocks = [{"type": "image", "source": {"type": "base64", "data": "..."}}]
        with patch("kai.memory.format_scoped_context_with_recall_payload", new=recall_spy):
            async for _event in g._send_locked(blocks, chat_id=42):
                pass

        recall_spy.assert_not_called()

    @pytest.mark.asyncio
    async def test_text_input_still_drives_semantic_recall(self):
        """A real text input threads the user's text through to the
        helper-side memory recall as the search query."""
        g = _make_goose(webhook_secret="test-secret")
        g._proc = _make_mock_proc([_completion_result(prompt_id=3)])
        g._session_id = "test-session"
        g._fresh_session = False
        g._next_id = 3

        from kai.memory import ScopedRecallResult

        recall_spy = AsyncMock(
            return_value=ScopedRecallResult(rendered_context="", recall_payload={"reason": "ok", "hits": []})
        )
        with patch("kai.memory.format_scoped_context_with_recall_payload", new=recall_spy):
            async for _event in g._send_locked("real user text", chat_id=42):
                pass

        recall_spy.assert_called_once()
        call = recall_spy.call_args
        assert call.args[0] == "real user text" or call.kwargs.get("query") == "real user text"


# ── Restart ────────────────────────────────────────────────────────


class TestRestart:
    """Verify restart() kills process and triggers fresh handshake."""

    @pytest.mark.asyncio
    async def test_restart_kills_and_resets(self):
        """restart() kills the process and sets fresh_session."""
        g = _make_goose()
        proc = MagicMock()
        proc.returncode = None
        proc.kill = MagicMock()
        proc.wait = AsyncMock()
        g._proc = proc
        g._session_id = "old-session"
        g._fresh_session = False

        await g.restart()

        assert g._proc is None
        assert g._session_id is None
        assert g._fresh_session is True

    @pytest.mark.asyncio
    async def test_restart_then_send_triggers_handshake(self):
        """After restart, the next _ensure_started() runs a fresh handshake."""
        g = _make_goose()
        # Start with an existing process
        old_proc = MagicMock()
        old_proc.returncode = None
        old_proc.kill = MagicMock()
        old_proc.wait = AsyncMock()
        g._proc = old_proc
        g._session_id = "old-session"

        await g.restart()
        assert g._proc is None

        # Now _ensure_started() should spawn a new process
        new_proc = _make_mock_proc(_handshake_lines("new-session"))
        with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=new_proc)):
            await g._ensure_started()

        assert g._session_id == "new-session"


# ── force_kill ─────────────────────────────────────────────────────


class TestForceKill:
    """Verify force_kill() sends SIGKILL to the process."""

    def test_force_kill_sends_sigkill(self):
        """force_kill() calls proc.kill() (SIGKILL)."""
        g = _make_goose()
        proc = MagicMock()
        proc.kill = MagicMock()
        g._proc = proc

        g.force_kill()

        proc.kill.assert_called_once()

    def test_force_kill_cancels_stderr_task(self):
        """force_kill() cancels the stderr drain task."""
        g = _make_goose()
        proc = MagicMock()
        proc.kill = MagicMock()
        g._proc = proc
        stderr_task = MagicMock()
        g._stderr_task = stderr_task

        g.force_kill()

        stderr_task.cancel.assert_called_once()
        assert g._stderr_task is None

    def test_force_kill_no_proc(self):
        """force_kill() is safe to call with no process."""
        g = _make_goose()
        g.force_kill()  # Should not raise

    def test_force_kill_oserror(self):
        """force_kill() swallows OSError from kill()."""
        g = _make_goose()
        proc = MagicMock()
        proc.kill = MagicMock(side_effect=OSError("already dead"))
        g._proc = proc

        g.force_kill()  # Should not raise


# ── JSON-RPC error handling ────────────────────────────────────────


class TestRpcError:
    """Verify JSON-RPC error responses yield error StreamEvents."""

    @pytest.mark.asyncio
    async def test_rpc_error_yields_error_event(self):
        """A JSON-RPC error response yields a done event with error."""
        g = _make_goose()
        g._proc = _make_mock_proc(
            [
                _agent_message_chunk("partial"),
                _error_result(prompt_id=3, message="rate limited"),
            ]
        )
        g._session_id = "test-session"
        g._fresh_session = False
        g._next_id = 3

        events = await _collect_events(g)
        final = events[-1]

        assert final.done is True
        assert final.response is not None
        assert final.response.success is False
        assert final.response.error == "rate limited"
        # Accumulated text is preserved even on error
        assert final.response.text == "partial"

    @pytest.mark.asyncio
    async def test_rpc_error_no_accumulated_text(self):
        """An immediate error with no prior text yields empty text."""
        g = _make_goose()
        g._proc = _make_mock_proc(
            [
                _error_result(prompt_id=3, message="invalid session"),
            ]
        )
        g._session_id = "test-session"
        g._fresh_session = False
        g._next_id = 3

        events = await _collect_events(g)
        final = events[-1]

        assert final.done is True
        assert final.response.success is False
        assert final.response.text == ""
        assert final.response.error == "invalid session"


# ── Prompt coercion ────────────────────────────────────────────────


class TestPromptCoercion:
    """Verify prompt format handling for ACP."""

    @pytest.mark.asyncio
    async def test_string_prompt_wrapped(self):
        """A string prompt is wrapped in a text content block; the
        shared per-turn helper prepends `USER_MESSAGE_MARKER` so the
        user's text stays delimited from injected context."""
        g = _make_goose()
        g._proc = _make_mock_proc(
            [
                _completion_result(prompt_id=3),
            ]
        )
        g._session_id = "test-session"
        g._fresh_session = False
        g._next_id = 3

        await _collect_events(g, prompt="hello world")

        write_calls = g._proc.stdin.write.call_args_list
        prompt_msg = json.loads(write_calls[-1][0][0].decode())
        assert prompt_msg["params"]["prompt"] == [{"type": "text", "text": f"{USER_MESSAGE_MARKER}\n\nhello world"}]

    @pytest.mark.asyncio
    async def test_list_prompt_text_blocks(self):
        """A list of text blocks passes through with the marker
        prepended as its own block above the original list."""
        g = _make_goose()
        g._proc = _make_mock_proc(
            [
                _completion_result(prompt_id=3),
            ]
        )
        g._session_id = "test-session"
        g._fresh_session = False
        g._next_id = 3

        blocks = [
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]
        await _collect_events(g, prompt=blocks)

        write_calls = g._proc.stdin.write.call_args_list
        prompt_msg = json.loads(write_calls[-1][0][0].decode())
        assert prompt_msg["params"]["prompt"] == [
            {"type": "text", "text": USER_MESSAGE_MARKER},
            {"type": "text", "text": "first"},
            {"type": "text", "text": "second"},
        ]

    @pytest.mark.asyncio
    async def test_non_text_blocks_dropped(self):
        """Non-text content blocks are dropped with a warning; the
        marker and remaining text blocks survive."""
        g = _make_goose()
        g._proc = _make_mock_proc(
            [
                _completion_result(prompt_id=3),
            ]
        )
        g._session_id = "test-session"
        g._fresh_session = False
        g._next_id = 3

        blocks = [
            {"type": "image", "source": {"type": "base64", "data": "..."}},
            {"type": "text", "text": "caption"},
        ]
        await _collect_events(g, prompt=blocks)

        write_calls = g._proc.stdin.write.call_args_list
        prompt_msg = json.loads(write_calls[-1][0][0].decode())
        assert prompt_msg["params"]["prompt"] == [
            {"type": "text", "text": USER_MESSAGE_MARKER},
            {"type": "text", "text": "caption"},
        ]

    @pytest.mark.asyncio
    async def test_all_non_text_blocks_becomes_empty_prompt(self):
        """If all blocks are non-text, the placeholder appears
        beneath the marker so the marker has a user region to label."""
        g = _make_goose()
        g._proc = _make_mock_proc(
            [
                _completion_result(prompt_id=3),
            ]
        )
        g._session_id = "test-session"
        g._fresh_session = False
        g._next_id = 3

        blocks = [{"type": "image", "source": {"type": "base64", "data": "..."}}]
        await _collect_events(g, prompt=blocks)

        write_calls = g._proc.stdin.write.call_args_list
        prompt_msg = json.loads(write_calls[-1][0][0].decode())
        assert prompt_msg["params"]["prompt"] == [
            {"type": "text", "text": USER_MESSAGE_MARKER},
            {"type": "text", "text": "(empty prompt)"},
        ]


# ── change_workspace ───────────────────────────────────────────────


class TestChangeWorkspace:
    """Verify workspace switching kills process and applies config."""

    @pytest.mark.asyncio
    async def test_change_workspace_resets_to_defaults(self):
        """Switching workspace reverts model/timeout to defaults."""
        ws1 = WorkspaceConfig(path=Path("/tmp/ws"), model="opus", timeout=60)
        g = _make_goose(workspace_config=ws1)
        assert g.model == "opus"
        assert g.timeout_seconds == 60

        # Simulate a running process
        proc = MagicMock()
        proc.returncode = None
        proc.kill = MagicMock()
        proc.wait = AsyncMock()
        g._proc = proc

        await g.change_workspace(Path("/tmp/new-ws"))

        # Should revert to defaults since no workspace config
        assert g.model == "sonnet"
        assert g.timeout_seconds == 30
        assert g.workspace == Path("/tmp/new-ws")
        assert g._proc is None

    @pytest.mark.asyncio
    async def test_change_workspace_applies_new_config(self):
        """Switching workspace applies the new workspace config."""
        g = _make_goose()
        proc = MagicMock()
        proc.returncode = None
        proc.kill = MagicMock()
        proc.wait = AsyncMock()
        g._proc = proc

        ws2 = WorkspaceConfig(path=Path("/tmp/new-ws"), model="haiku", timeout=15)
        await g.change_workspace(Path("/tmp/new-ws"), workspace_config=ws2)

        assert g.model == "haiku"
        assert g.timeout_seconds == 15


# ── shutdown ───────────────────────────────────────────────────────


class TestShutdown:
    """Verify graceful shutdown behavior."""

    @pytest.mark.asyncio
    async def test_shutdown_sends_sigterm(self):
        """shutdown() sends SIGTERM first."""
        g = _make_goose()
        proc = MagicMock()
        proc.returncode = None
        proc.terminate = MagicMock()
        proc.wait = AsyncMock()
        proc.kill = MagicMock()
        g._proc = proc

        await g.shutdown()

        proc.terminate.assert_called_once()
        assert g._proc is None
        assert g._session_id is None

    @pytest.mark.asyncio
    async def test_shutdown_no_proc(self):
        """shutdown() is safe to call with no process."""
        g = _make_goose()
        await g.shutdown()  # Should not raise

    @pytest.mark.asyncio
    async def test_shutdown_terminate_oserror(self):
        """shutdown() handles OSError from terminate() when process already exited."""
        g = _make_goose()
        proc = MagicMock()
        proc.returncode = None
        proc.terminate = MagicMock(side_effect=OSError("already dead"))
        proc.wait = AsyncMock()
        g._proc = proc

        await g.shutdown()

        assert g._proc is None
        assert g._session_id is None

    @pytest.mark.asyncio
    async def test_shutdown_sigkill_fallback(self):
        """shutdown() falls back to SIGKILL on SIGTERM timeout."""
        g = _make_goose()
        proc = MagicMock()
        proc.returncode = None
        proc.terminate = MagicMock()
        # First wait (after SIGTERM) times out, second (after SIGKILL) succeeds
        proc.wait = AsyncMock(side_effect=[TimeoutError(), None])
        proc.kill = MagicMock()
        g._proc = proc

        await g.shutdown()

        proc.terminate.assert_called_once()
        proc.kill.assert_called_once()  # SIGKILL fallback
        assert g._proc is None


# ── send() lock ────────────────────────────────────────────────────


class TestSendLock:
    """Verify send() acquires the lock."""

    @pytest.mark.asyncio
    async def test_send_acquires_lock(self):
        """send() serializes through self._lock."""
        g = _make_goose()
        g._proc = _make_mock_proc(
            [
                _agent_message_chunk("ok"),
                _completion_result(prompt_id=3),
            ]
        )
        g._session_id = "test-session"
        g._fresh_session = False
        g._next_id = 3

        # Verify the lock is acquired during send
        assert not g._lock.locked()
        events = []
        async for event in g.send("test"):
            # Lock should be held during iteration
            assert g._lock.locked()
            events.append(event)
        assert not g._lock.locked()
        assert len(events) > 0


# ── Argv binary resolution ──────────────────────────────────────────


class TestBuildArgv:
    """GOOSE_BIN pins the binary path so a sudo-wrapped per-user spawn
    matches the absolute path the sudoers rule names (mirrors codex's
    CODEX_BIN contract); bare "goose" remains the default for installs
    with goose on PATH."""

    def test_bare_goose_without_override(self, monkeypatch):
        monkeypatch.delenv("GOOSE_BIN", raising=False)
        g = _make_goose()
        assert g.build_argv() == ["goose", "acp", "--with-builtin", "developer"]

    def test_goose_bin_override_pins_absolute_path(self, monkeypatch):
        monkeypatch.setenv("GOOSE_BIN", "/opt/homebrew/bin/goose")
        g = _make_goose()
        assert g.build_argv() == ["/opt/homebrew/bin/goose", "acp", "--with-builtin", "developer"]

    def test_empty_goose_bin_falls_back_to_bare_name(self, monkeypatch):
        """An empty-string GOOSE_BIN (unset-but-present in /etc/kai/env)
        must not produce an empty argv head."""
        monkeypatch.setenv("GOOSE_BIN", "")
        g = _make_goose()
        assert g.build_argv()[0] == "goose"


# ── Cross-user preserve list ────────────────────────────────────────


class TestPreservedEnvVars:
    """The sudo wrap's --preserve-env list. Goose delivers model and
    provider selection via env vars (GOOSE_MODEL / GOOSE_PROVIDER in
    build_env), so the override must carry them through env_reset or
    the wrapped goose silently loses its model selection; the five
    provider keys cover env-key auth, the endpoint-override vars
    (ANTHROPIC_HOST, OPENAI_HOST, OPENAI_BASE_PATH, OLLAMA_HOST) keep
    a custom-endpoint install routed at its gateway under the wrap,
    and KAI_WEBHOOK_SECRET / TMPDIR mirror the AcpBackend default."""

    def test_content_exact(self):
        g = _make_goose()
        assert g.preserved_env_vars() == (
            "GOOSE_MODEL",
            "GOOSE_PROVIDER",
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "GOOGLE_API_KEY",
            "OPENROUTER_API_KEY",
            "DEEPSEEK_API_KEY",
            "ANTHROPIC_HOST",
            "OPENAI_HOST",
            "OPENAI_BASE_PATH",
            "OLLAMA_HOST",
            "KAI_WEBHOOK_SECRET",
            "TMPDIR",
        )

    @pytest.mark.asyncio
    async def test_wrap_argv_carries_goose_preserve_list(self, monkeypatch):
        """End-to-end through _ensure_started: the goose wrap's
        --preserve-env CSV is the hook's list, not the base default."""
        monkeypatch.delenv("GOOSE_BIN", raising=False)
        g = _make_goose(os_user="goose-user", provider="anthropic")
        captured: dict = {}

        async def _fake_spawn(*args, **kwargs):
            captured["argv"] = args
            return _make_mock_proc(_handshake_lines())

        with (
            patch("kai.acp.resolve_claude_user", return_value="goose-user"),
            patch("kai.acp.asyncio.create_subprocess_exec", side_effect=_fake_spawn),
        ):
            await g._ensure_started()

        argv = list(captured["argv"])
        assert argv[:4] == ["sudo", "-H", "-u", "goose-user"]
        assert argv[4] == (
            "--preserve-env=GOOSE_MODEL,GOOSE_PROVIDER,ANTHROPIC_API_KEY,"
            "OPENAI_API_KEY,GOOGLE_API_KEY,OPENROUTER_API_KEY,DEEPSEEK_API_KEY,"
            "ANTHROPIC_HOST,OPENAI_HOST,OPENAI_BASE_PATH,OLLAMA_HOST,"
            "KAI_WEBHOOK_SECRET,TMPDIR"
        )
        assert argv[5] == "--"
        assert argv[6:] == ["goose", "acp", "--with-builtin", "developer"]
