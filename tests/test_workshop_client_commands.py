"""Compatibility contracts for authorized Workshop client execution."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, call

from kai.workshop.client_commands import WorkshopClientCommandExecutor
from kai.workshop.conversation_commands import ConversationCommandDisposition
from kai.workshop.domain import ChannelId, PrincipalId, RunId
from kai.workshop.execution_coordinator import (
    CanonicalExecutionDisposition,
    CanonicalExecutionResult,
)
from kai.workshop.inbound import ClientInboundMessage


def _message() -> ClientInboundMessage:
    return ClientInboundMessage(
        principal_id=PrincipalId.new(),
        channel_id=ChannelId.new(),
        client_message_id="browser-command-1",
        body="Hello from Workshop",
        occurred_at=datetime(2026, 8, 13, 12, 0, tzinfo=UTC),
    )


async def test_completed_client_command_preserves_history_session_and_memory(
    monkeypatch,
):
    message = _message()
    run_id = RunId.new()
    accepted = SimpleNamespace(
        compatibility_chat_id=101,
        command=SimpleNamespace(disposition=ConversationCommandDisposition.NEWLY_ACCEPTED),
        run=SimpleNamespace(run_id=run_id),
    )
    terminal = SimpleNamespace(body="Completed through the protected lane")
    run = SimpleNamespace(status="completed")
    execution_result = CanonicalExecutionResult(
        CanonicalExecutionDisposition.COMPLETED,
        run,
        terminal,
        session_id="session-1",
        workspace="/workspace/project",
        selection=SimpleNamespace(model="gpt-5.6-sol"),
    )
    execution = SimpleNamespace(
        accept_client=AsyncMock(return_value=accepted),
        execute=AsyncMock(return_value=execution_result),
    )
    config = SimpleNamespace(
        get_user_config=lambda chat_id: SimpleNamespace(os_user="daniel") if chat_id == 101 else None
    )
    log = Mock(side_effect=["user-log", "assistant-log"])
    save_session = AsyncMock()
    schedule = Mock()
    monkeypatch.setattr("kai.workshop.client_commands.log_message", log)
    monkeypatch.setattr("kai.workshop.client_commands.sessions.save_session", save_session)
    monkeypatch.setattr("kai.workshop.client_commands.schedule_memory_ingestion", schedule)

    result = await WorkshopClientCommandExecutor(execution, config).submit(message)

    assert result.acceptance is accepted
    assert result.execution is execution_result
    execution.accept_client.assert_awaited_once_with(message)
    execution.execute.assert_awaited_once_with(run_id)
    assert log.call_args_list == [
        call(
            direction="user",
            chat_id=101,
            text="Hello from Workshop",
            reader_user="daniel",
        ),
        call(
            direction="assistant",
            chat_id=101,
            text="Completed through the protected lane",
            reader_user="daniel",
        ),
    ]
    save_session.assert_awaited_once_with(101, "session-1", "gpt-5.6-sol")
    schedule.assert_called_once_with(
        prompt="Hello from Workshop",
        assistant_text="Completed through the protected lane",
        chat_id=101,
        session_id="session-1",
        config=config,
        workspace="/workspace/project",
        user_log="user-log",
        assistant_log="assistant-log",
    )


async def test_terminal_replay_does_not_duplicate_compatibility_writes(monkeypatch):
    message = _message()
    run_id = RunId.new()
    accepted = SimpleNamespace(
        compatibility_chat_id=101,
        command=SimpleNamespace(disposition=ConversationCommandDisposition.TERMINAL_REPLAY),
        run=SimpleNamespace(run_id=run_id),
    )
    execution = SimpleNamespace(
        accept_client=AsyncMock(return_value=accepted),
        execute=AsyncMock(
            return_value=CanonicalExecutionResult(
                CanonicalExecutionDisposition.TERMINAL_REPLAY,
                SimpleNamespace(status="completed"),
            )
        ),
    )
    config = SimpleNamespace(get_user_config=lambda _chat_id: None)
    log = Mock()
    save_session = AsyncMock()
    schedule = Mock()
    monkeypatch.setattr("kai.workshop.client_commands.log_message", log)
    monkeypatch.setattr("kai.workshop.client_commands.sessions.save_session", save_session)
    monkeypatch.setattr("kai.workshop.client_commands.schedule_memory_ingestion", schedule)

    await WorkshopClientCommandExecutor(execution, config).submit(message)

    log.assert_not_called()
    save_session.assert_not_awaited()
    schedule.assert_not_called()
