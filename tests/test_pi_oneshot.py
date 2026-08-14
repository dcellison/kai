"""Contract tests for Pi's bounded one-shot reasoner."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from kai.oneshot import OneShotOutputError, OneShotTimeout, PiOneShotReasoner


class FakeWriter:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def write(self, data: bytes) -> None:
        for line in data.splitlines():
            self.records.append(json.loads(line))

    async def drain(self) -> None:
        return None


class FakeProcess:
    def __init__(self, records: list[dict] | None = None) -> None:
        self.stdin = FakeWriter()
        self.stdout = asyncio.StreamReader()
        self.stderr = asyncio.StreamReader()
        for record in records or []:
            self.stdout.feed_data(json.dumps(record).encode() + b"\n")
        if records is not None:
            self.stdout.feed_eof()
        self.stderr.feed_eof()
        self.pid = 4321
        self.returncode: int | None = None
        self.killed = False

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9

    async def wait(self) -> int:
        return self.returncode or 0


def state_response(model: str = "claude-sonnet-4-6") -> dict:
    return {
        "id": "kai-oneshot-state",
        "type": "response",
        "command": "get_state",
        "success": True,
        "data": {
            "sessionId": "ephemeral",
            "model": {"provider": "anthropic", "id": model},
        },
    }


def prompt_response() -> dict:
    return {
        "id": "kai-oneshot-prompt",
        "type": "response",
        "command": "prompt",
        "success": True,
    }


def completed_message(text: str) -> dict:
    return {
        "type": "message_end",
        "message": {"role": "assistant", "content": [{"type": "text", "text": text}]},
    }


async def run_with_process(tmp_path: Path, monkeypatch, process: FakeProcess, **kwargs):
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr("kai.oneshot.resolve_oneshot_binary", lambda backend: "/opt/homebrew/bin/pi")
    monkeypatch.setattr("kai.oneshot.resolve_claude_user", lambda user: user)
    monkeypatch.setattr("kai.oneshot.asyncio.create_subprocess_exec", spawn)
    reasoner = PiOneShotReasoner(cwd=tmp_path, os_user=kwargs.pop("os_user", None), provider="anthropic")
    result = await reasoner.run(
        prompt=kwargs.pop("prompt", "hello"),
        system_prompt=kwargs.pop("system_prompt", None),
        model=kwargs.pop("model", "anthropic/claude-sonnet-4-6"),
        timeout=kwargs.pop("timeout", 5),
        purpose=kwargs.pop("purpose", "test"),
        json_schema=kwargs.pop("json_schema", None),
    )
    assert not kwargs
    return result, spawn


@pytest.mark.asyncio
async def test_pi_oneshot_uses_hardened_rpc_flags_and_authoritative_message(tmp_path, monkeypatch):
    process = FakeProcess(
        [
            state_response(),
            prompt_response(),
            {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "delta": "draft"}},
            {"type": "agent_end", "willRetry": False},
            completed_message("final answer"),
            {"type": "agent_settled"},
        ]
    )

    result, spawn = await run_with_process(tmp_path, monkeypatch, process)

    assert result.text == "final answer"
    assert result.backend == "pi"
    argv = spawn.await_args_list[0].args
    assert argv[:3] == ("/opt/homebrew/bin/pi", "--mode", "rpc")
    for flag in (
        "--no-session",
        "--no-tools",
        "--no-approve",
        "--no-extensions",
        "--no-skills",
        "--no-prompt-templates",
        "--no-context-files",
    ):
        assert flag in argv
    assert "--approve" not in argv
    assert process.stdin.records == [
        {"id": "kai-oneshot-state", "type": "get_state"},
        {"id": "kai-oneshot-prompt", "type": "prompt", "message": "hello"},
    ]
    assert process.killed is True


@pytest.mark.asyncio
async def test_pi_oneshot_schema_normalizes_to_shared_envelope(tmp_path, monkeypatch):
    process = FakeProcess(
        [
            state_response(),
            prompt_response(),
            completed_message('```json\n{"facts": []}\n```'),
            {"type": "agent_settled"},
        ]
    )
    result, _ = await run_with_process(
        tmp_path,
        monkeypatch,
        process,
        json_schema={"type": "object", "required": ["facts"]},
    )
    assert json.loads(result.text) == {"is_error": False, "structured_output": {"facts": []}}


@pytest.mark.asyncio
async def test_pi_oneshot_system_prompt_is_boundary_delimited(tmp_path, monkeypatch):
    process = FakeProcess([state_response(), prompt_response(), completed_message("ok"), {"type": "agent_settled"}])
    await run_with_process(tmp_path, monkeypatch, process, system_prompt="system rules", prompt="user input")
    rendered = process.stdin.records[1]["message"]
    assert "system rules" in rendered
    assert rendered.endswith("\n\nuser input")
    assert "BEGIN SYSTEM" in rendered and "END SYSTEM" in rendered


@pytest.mark.asyncio
async def test_pi_oneshot_rejects_silent_model_substitution(tmp_path, monkeypatch):
    process = FakeProcess([state_response("claude-haiku-4-5")])
    with pytest.raises(OneShotOutputError, match="different model"):
        await run_with_process(tmp_path, monkeypatch, process)
    assert process.killed is True


@pytest.mark.asyncio
async def test_pi_oneshot_surfaces_terminal_retry_failure(tmp_path, monkeypatch):
    process = FakeProcess(
        [
            state_response(),
            prompt_response(),
            {"type": "auto_retry_end", "success": False, "finalError": "quota exhausted"},
            {"type": "agent_settled"},
        ]
    )
    with pytest.raises(OneShotOutputError, match="quota exhausted"):
        await run_with_process(tmp_path, monkeypatch, process)


@pytest.mark.asyncio
async def test_pi_oneshot_fails_closed_if_tool_event_appears(tmp_path, monkeypatch):
    process = FakeProcess(
        [
            state_response(),
            prompt_response(),
            {"type": "tool_execution_start", "toolCallId": "tool-1", "toolName": "bash", "args": {}},
        ]
    )
    with pytest.raises(OneShotOutputError, match="tools being disabled"):
        await run_with_process(tmp_path, monkeypatch, process)


@pytest.mark.asyncio
async def test_pi_oneshot_cross_user_wrap_preserves_only_pi_auth(tmp_path, monkeypatch):
    process = FakeProcess([state_response(), prompt_response(), completed_message("ok"), {"type": "agent_settled"}])
    _, spawn = await run_with_process(tmp_path, monkeypatch, process, os_user="daniel")
    argv = spawn.await_args_list[0].args
    assert argv[:6] == ("sudo", "-H", "-D", str(tmp_path), "-u", "daniel")
    assert "ANTHROPIC_API_KEY" in argv[6]
    assert "OPENAI_API_KEY" not in argv[6]
    assert "GEMINI_API_KEY" not in argv[6]
    assert "KAI_WEBHOOK_SECRET" not in argv[6]
    assert spawn.await_args_list[0].kwargs["cwd"] is None
    assert spawn.await_args_list[0].kwargs["start_new_session"] is True


@pytest.mark.asyncio
async def test_pi_subscription_wrap_needs_no_environment_credentials(tmp_path, monkeypatch):
    process = FakeProcess(
        [
            {
                **state_response("gpt-5.5"),
                "data": {
                    "sessionId": "ephemeral",
                    "model": {"provider": "openai-codex", "id": "gpt-5.5"},
                },
            },
            prompt_response(),
            completed_message("ok"),
            {"type": "agent_settled"},
        ]
    )
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr("kai.oneshot.resolve_oneshot_binary", lambda backend: "/opt/homebrew/bin/pi")
    monkeypatch.setattr("kai.oneshot.resolve_claude_user", lambda user: user)
    monkeypatch.setattr("kai.oneshot.asyncio.create_subprocess_exec", spawn)
    reasoner = PiOneShotReasoner(cwd=tmp_path, os_user="daniel", provider="openai-codex")

    await reasoner.run(
        prompt="hello",
        model="openai-codex/gpt-5.5",
        timeout=5,
        purpose="test",
    )

    argv = spawn.await_args_list[0].args
    assert argv[:6] == ("sudo", "-H", "-D", str(tmp_path), "-u", "daniel")
    assert argv[6:8] == ("--", "/opt/homebrew/bin/pi")
    assert not any(str(part).startswith("--preserve-env=") for part in argv)
    assert spawn.await_args_list[0].kwargs["cwd"] is None


@pytest.mark.asyncio
async def test_pi_oneshot_timeout_kills_process(tmp_path, monkeypatch):
    process = FakeProcess(records=None)
    with pytest.raises(OneShotTimeout):
        await run_with_process(tmp_path, monkeypatch, process, timeout=0.01)
    assert process.killed is True
