"""Tests for the persistent conversational Pi backend."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from kai.pi import PiBackend, _assistant_message_text, _split_pi_model
from kai.pi_rpc import PiRpcEOFError


class FakeTransport:
    def __init__(self, records=()):
        self.records = list(records)
        self.sent: list[dict] = []

    async def send(self, command):
        self.sent.append(dict(command))

    async def receive(self, *, timeout_seconds=None):
        if not self.records:
            raise PiRpcEOFError("test stream exhausted")
        value = self.records.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class FakeProcess:
    def __init__(self):
        self.stdin = MagicMock()
        self.stdout = MagicMock()
        self.stderr = MagicMock()
        self.stderr.readline = AsyncMock(return_value=b"")
        self.pid = 4321
        self.returncode = None
        self.killed = False
        self.terminated = False

    def kill(self):
        self.killed = True
        self.returncode = -9

    def terminate(self):
        self.terminated = True
        self.returncode = 0

    async def wait(self):
        return self.returncode or 0


def make_backend(tmp_path: Path, **overrides) -> PiBackend:
    values = {
        "model": "anthropic/claude-sonnet-4-6",
        "provider": "anthropic",
        "workspace": tmp_path,
        "home_workspace": tmp_path,
        "timeout_seconds": 5,
    }
    values.update(overrides)
    return PiBackend(**values)


async def collect(backend: PiBackend, prompt="hello", chat_id=123):
    return [event async for event in backend.send(prompt, chat_id=chat_id)]


class TestPiModelParsing:
    def test_provider_model_and_thinking_suffix(self):
        assert _split_pi_model("openai/gpt-5.6-sol:xhigh") == ("openai", "gpt-5.6-sol", "xhigh")

    def test_ollama_tag_is_part_of_model_id(self):
        assert _split_pi_model("ollama/llama4:70b") == ("ollama", "llama4:70b", None)

    def test_bare_model_is_rejected(self):
        with pytest.raises(ValueError, match="provider/model"):
            _split_pi_model("sonnet")

    def test_bare_model_uses_explicit_provider(self):
        assert _split_pi_model("claude-sonnet-4-6:high", "anthropic") == (
            "anthropic",
            "claude-sonnet-4-6",
            "high",
        )

    def test_prefixed_model_must_match_explicit_provider(self):
        with pytest.raises(ValueError, match="does not match"):
            _split_pi_model("openai/gpt-5.5", "anthropic")

    def test_authoritative_message_text_ignores_thinking_and_tools(self):
        message = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "private"},
                {"type": "text", "text": "hello "},
                {"type": "toolCall", "name": "bash"},
                {"type": "text", "text": "world"},
            ],
        }
        assert _assistant_message_text(message) == "hello world"


class TestPiStartup:
    def test_argv_uses_registry_and_disables_ambient_resources(self, tmp_path, monkeypatch):
        backend = make_backend(tmp_path)
        monkeypatch.setattr("kai.pi.resolve_backend_command", lambda *a, **kw: "/opt/homebrew/bin/pi")

        argv = backend._build_argv()

        assert argv[:5] == ["/opt/homebrew/bin/pi", "--mode", "rpc", "--provider", "anthropic"]
        assert argv[5:7] == ["--model", "anthropic/claude-sonnet-4-6"]
        for flag in (
            "--no-approve",
            "--no-extensions",
            "--no-skills",
            "--no-prompt-templates",
            "--no-context-files",
        ):
            assert flag in argv
        assert "--approve" not in argv

    def test_env_removes_control_plane_secrets_and_restores_principal(self, tmp_path, monkeypatch):
        backend = make_backend(tmp_path, webhook_secret="principal-token")
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "must-not-leak")
        monkeypatch.setenv("GITHUB_WEBHOOK_SECRET", "must-not-leak")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "provider-key")
        monkeypatch.setenv("OPENAI_API_KEY", "unrelated-provider-key")

        env = backend._build_env("daniel")

        assert "TELEGRAM_BOT_TOKEN" not in env
        assert "GITHUB_WEBHOOK_SECRET" not in env
        assert env["KAI_WEBHOOK_SECRET"] == "principal-token"
        assert env["ANTHROPIC_API_KEY"] == "provider-key"
        assert "OPENAI_API_KEY" not in env
        assert env["TMPDIR"].endswith("/tmp/daniel")

    @pytest.mark.asyncio
    async def test_cross_user_start_uses_sudo_and_validates_state(self, tmp_path, monkeypatch):
        backend = make_backend(tmp_path, os_user="daniel")
        proc = FakeProcess()
        transport = FakeTransport(
            [
                {
                    "id": "kai-get_state-1",
                    "type": "response",
                    "command": "get_state",
                    "success": True,
                    "data": {
                        "sessionId": "session-1",
                        "thinkingLevel": "medium",
                        "model": {
                            "provider": "anthropic",
                            "id": "claude-sonnet-4-6",
                            "input": ["text", "image"],
                        },
                    },
                }
            ]
        )
        spawn = AsyncMock(return_value=proc)
        monkeypatch.setattr("kai.pi.resolve_claude_user", lambda value: value)
        monkeypatch.setattr("kai.pi.resolve_backend_command", lambda *a, **kw: "/opt/homebrew/bin/pi")
        monkeypatch.setattr("kai.pi.asyncio.create_subprocess_exec", spawn)
        monkeypatch.setattr("kai.pi.PiRpcTransport", lambda stdin, stdout: transport)

        await backend._ensure_started()

        argv = spawn.await_args.args
        assert argv[:4] == ("sudo", "-H", "-u", "daniel")
        assert argv[4].startswith("--preserve-env=")
        assert "ANTHROPIC_API_KEY" in argv[4]
        assert argv[5:7] == ("--", "/opt/homebrew/bin/pi")
        assert backend.session_id == "session-1"
        assert backend._supports_image_input is True
        assert transport.sent == [{"id": "kai-get_state-1", "type": "get_state"}]

    @pytest.mark.asyncio
    async def test_startup_rejects_silent_model_substitution(self, tmp_path, monkeypatch):
        backend = make_backend(tmp_path)
        proc = FakeProcess()
        transport = FakeTransport(
            [
                {
                    "id": "kai-get_state-1",
                    "type": "response",
                    "command": "get_state",
                    "success": True,
                    "data": {
                        "sessionId": "session-1",
                        "model": {"provider": "anthropic", "id": "claude-haiku-4-5", "input": ["text"]},
                    },
                }
            ]
        )
        monkeypatch.setattr("kai.pi.resolve_claude_user", lambda value: None)
        monkeypatch.setattr("kai.pi.asyncio.create_subprocess_exec", AsyncMock(return_value=proc))
        monkeypatch.setattr("kai.pi.PiRpcTransport", lambda stdin, stdout: transport)

        with pytest.raises(Exception, match="different model"):
            await backend._ensure_started()


class TestPiTurns:
    @pytest.fixture(autouse=True)
    def no_context_io(self, monkeypatch):
        monkeypatch.setattr("kai.pi.ensure_user_context_files", lambda *a, **kw: None)
        monkeypatch.setattr("kai.pi.build_session_context", lambda **kw: "")
        monkeypatch.setattr("kai.pi.build_foreign_workspace_reminder", lambda *a, **kw: None)

        async def identity(prompt, **kwargs):
            return prompt

        monkeypatch.setattr("kai.pi.assemble_turn_context", identity)

    @pytest.mark.asyncio
    async def test_streams_deltas_but_finishes_with_authoritative_message(self, tmp_path, monkeypatch):
        backend = make_backend(tmp_path)
        backend._proc = FakeProcess()
        backend._session_id = "session-1"
        backend._supports_image_input = True
        backend._fresh_session = False
        transport = FakeTransport(
            [
                {"id": "kai-prompt-1", "type": "response", "command": "prompt", "success": True},
                {"type": "agent_start"},
                {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "hel"}},
                {"type": "agent_end", "willRetry": False},
                {
                    "type": "message_end",
                    "message": {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
                },
                {"type": "agent_settled"},
            ]
        )
        backend._transport = transport
        monkeypatch.setattr(backend, "_ensure_started", AsyncMock())

        events = await collect(backend)

        assert events[0].text_so_far == "hel"
        assert events[-1].done is True
        assert events[-1].response is not None
        assert events[-1].response.success is True
        assert events[-1].response.text == "hello"
        assert events[-1].response.session_id == "session-1"
        assert transport.sent[0]["type"] == "prompt"

    @pytest.mark.asyncio
    async def test_image_is_forwarded_in_documented_shape(self, tmp_path, monkeypatch):
        backend = make_backend(tmp_path)
        backend._proc = FakeProcess()
        backend._session_id = "session-1"
        backend._supports_image_input = True
        backend._fresh_session = False
        backend._transport = FakeTransport(
            [
                {"id": "kai-prompt-1", "type": "response", "command": "prompt", "success": True},
                {"type": "agent_settled"},
            ]
        )
        monkeypatch.setattr(backend, "_ensure_started", AsyncMock())
        prompt = [
            {"type": "text", "text": "look"},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "YWJj"}},
        ]

        await collect(backend, prompt)

        assert backend._transport.sent[0]["images"] == [{"type": "image", "data": "YWJj", "mimeType": "image/png"}]

    @pytest.mark.asyncio
    async def test_unsupported_image_gets_visible_notice(self, tmp_path, monkeypatch):
        backend = make_backend(tmp_path)
        backend._proc = FakeProcess()
        backend._session_id = "session-1"
        backend._supports_image_input = False
        backend._fresh_session = False
        backend._transport = FakeTransport(
            [
                {"id": "kai-prompt-1", "type": "response", "command": "prompt", "success": True},
                {"type": "agent_settled"},
            ]
        )
        monkeypatch.setattr(backend, "_ensure_started", AsyncMock())

        events = await collect(
            backend,
            [{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "YWJj"}}],
        )

        assert "could not be passed to Pi" in events[-1].text_so_far
        assert "images" not in backend._transport.sent[0]

    @pytest.mark.asyncio
    async def test_agent_end_is_not_terminal_and_retry_can_finish(self, tmp_path, monkeypatch):
        backend = make_backend(tmp_path)
        backend._proc = FakeProcess()
        backend._session_id = "session-1"
        backend._fresh_session = False
        backend._transport = FakeTransport(
            [
                {"id": "kai-prompt-1", "type": "response", "command": "prompt", "success": True},
                {"type": "agent_end", "willRetry": True},
                {"type": "auto_retry_start"},
                {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "recovered"}},
                {"type": "agent_settled"},
            ]
        )
        monkeypatch.setattr(backend, "_ensure_started", AsyncMock())

        events = await collect(backend)

        assert events[-1].response is not None
        assert events[-1].response.success is True
        assert events[-1].response.text == "recovered"

    @pytest.mark.asyncio
    async def test_tool_events_are_not_mixed_into_assistant_text(self, tmp_path, monkeypatch):
        backend = make_backend(tmp_path)
        backend._proc = FakeProcess()
        backend._session_id = "session-1"
        backend._fresh_session = False
        backend._transport = FakeTransport(
            [
                {"id": "kai-prompt-1", "type": "response", "command": "prompt", "success": True},
                {
                    "type": "tool_execution_start",
                    "toolCallId": "tool-1",
                    "toolName": "read",
                    "args": {"path": "README.md"},
                },
                {
                    "type": "tool_execution_end",
                    "toolCallId": "tool-1",
                    "toolName": "read",
                    "result": {"content": "secret tool output"},
                    "isError": False,
                },
                {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "answer"}},
                {"type": "agent_settled"},
            ]
        )
        monkeypatch.setattr(backend, "_ensure_started", AsyncMock())

        events = await collect(backend)

        assert events[-1].response is not None
        assert events[-1].response.text == "answer"
        assert "secret tool output" not in events[-1].response.text

    @pytest.mark.asyncio
    async def test_final_retry_failure_is_reported(self, tmp_path, monkeypatch):
        backend = make_backend(tmp_path)
        backend._proc = FakeProcess()
        backend._session_id = "session-1"
        backend._fresh_session = False
        backend._transport = FakeTransport(
            [
                {"id": "kai-prompt-1", "type": "response", "command": "prompt", "success": True},
                {"type": "auto_retry_end", "success": False, "finalError": "quota exhausted"},
                {"type": "agent_settled"},
            ]
        )
        monkeypatch.setattr(backend, "_ensure_started", AsyncMock())

        events = await collect(backend)

        assert events[-1].response is not None
        assert events[-1].response.success is False
        assert events[-1].response.error == "quota exhausted"

    @pytest.mark.asyncio
    async def test_protocol_failure_preserves_partial_text_and_recycles(self, tmp_path, monkeypatch):
        backend = make_backend(tmp_path)
        backend._proc = FakeProcess()
        backend._session_id = "session-1"
        backend._fresh_session = False
        backend._transport = FakeTransport(
            [
                {"id": "kai-prompt-1", "type": "response", "command": "prompt", "success": True},
                {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "partial"}},
                PiRpcEOFError("Pi exited"),
            ]
        )
        monkeypatch.setattr(backend, "_ensure_started", AsyncMock())
        kill = AsyncMock()
        monkeypatch.setattr(backend, "_kill", kill)

        events = await collect(backend)

        assert events[-1].text_so_far == "partial"
        assert events[-1].response is not None
        assert events[-1].response.success is False
        assert "Pi exited" in events[-1].response.error
        kill.assert_awaited_once()


class TestPiLifecycle:
    @pytest.mark.asyncio
    async def test_change_workspace_resets_overrides_and_session(self, tmp_path, monkeypatch):
        old = tmp_path / "old"
        new = tmp_path / "new"
        old.mkdir()
        new.mkdir()
        backend = make_backend(old)
        backend.model = "anthropic/temporary"
        backend.timeout_seconds = 99
        backend._fresh_session = False
        monkeypatch.setattr(backend, "_kill", AsyncMock())

        await backend.change_workspace(new)

        assert backend.workspace == new
        assert backend.model == "anthropic/claude-sonnet-4-6"
        assert backend.timeout_seconds == 5
        assert backend._fresh_session is True

    @pytest.mark.asyncio
    async def test_restart_kills_and_marks_session_fresh(self, tmp_path, monkeypatch):
        backend = make_backend(tmp_path)
        backend._fresh_session = False
        kill = AsyncMock()
        monkeypatch.setattr(backend, "_kill", kill)

        await backend.restart()

        kill.assert_awaited_once()
        assert backend._fresh_session is True

    @pytest.mark.asyncio
    async def test_shutdown_uses_cross_user_safe_teardown(self, tmp_path, monkeypatch):
        backend = make_backend(tmp_path, os_user="daniel")
        kill = AsyncMock()
        monkeypatch.setattr(backend, "_kill", kill)

        await backend.shutdown()

        kill.assert_awaited_once()
