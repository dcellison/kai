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
from tests.workshop_profiles import profile_id


def _message() -> ClientInboundMessage:
    return ClientInboundMessage(
        principal_id=PrincipalId.new(),
        channel_id=ChannelId.new(),
        client_message_id="browser-command-1",
        body="Hello from Workshop",
        occurred_at=datetime(2026, 8, 13, 12, 0, tzinfo=UTC),
    )


async def test_completed_client_command_preserves_history_session_and_memory():
    message = _message()
    run_id = RunId.new()
    accepted = SimpleNamespace(
        runtime_profile_id=profile_id(101),
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
    profile_state = SimpleNamespace(
        append_history=Mock(side_effect=["user-log", "assistant-log"]),
        save_session=AsyncMock(),
        schedule_memory_ingestion=Mock(),
    )
    compatibility_state = SimpleNamespace(for_profile=Mock(return_value=profile_state))

    result = await WorkshopClientCommandExecutor(execution, compatibility_state).submit(message)

    assert result.acceptance is accepted
    assert result.execution is execution_result
    execution.accept_client.assert_awaited_once_with(message)
    execution.execute.assert_awaited_once_with(run_id)
    compatibility_state.for_profile.assert_called_once_with(profile_id(101))
    assert profile_state.append_history.call_args_list == [
        call(
            direction="user",
            text="Hello from Workshop",
        ),
        call(
            direction="assistant",
            text="Completed through the protected lane",
        ),
    ]
    profile_state.save_session.assert_awaited_once_with("session-1", "gpt-5.6-sol")
    profile_state.schedule_memory_ingestion.assert_called_once_with(
        prompt="Hello from Workshop",
        assistant_text="Completed through the protected lane",
        session_id="session-1",
        workspace="/workspace/project",
        user_log="user-log",
        assistant_log="assistant-log",
    )


async def test_terminal_replay_does_not_duplicate_compatibility_writes():
    message = _message()
    run_id = RunId.new()
    accepted = SimpleNamespace(
        runtime_profile_id=profile_id(101),
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
    profile_state = SimpleNamespace(
        append_history=Mock(),
        save_session=AsyncMock(),
        schedule_memory_ingestion=Mock(),
    )
    compatibility_state = SimpleNamespace(for_profile=Mock(return_value=profile_state))

    await WorkshopClientCommandExecutor(execution, compatibility_state).submit(message)

    compatibility_state.for_profile.assert_called_once_with(profile_id(101))
    profile_state.append_history.assert_not_called()
    profile_state.save_session.assert_not_awaited()
    profile_state.schedule_memory_ingestion.assert_not_called()
