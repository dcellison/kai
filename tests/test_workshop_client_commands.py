"""Compatibility contracts for asynchronous Workshop client execution."""

from __future__ import annotations

import asyncio
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
from kai.workshop.private_text_execution import RecoverableClientRun
from tests.workshop_profiles import profile_id


def _message() -> ClientInboundMessage:
    return ClientInboundMessage(
        principal_id=PrincipalId.new(),
        channel_id=ChannelId.new(),
        client_message_id="browser-command-1",
        body="Hello from Workshop",
        occurred_at=datetime(2026, 8, 13, 12, 0, tzinfo=UTC),
    )


async def test_submission_returns_before_execution_and_preserves_compatibility_state():
    message = _message()
    run_id = RunId.new()
    run = SimpleNamespace(run_id=run_id, status="accepted")
    accepted = SimpleNamespace(
        runtime_profile_id=profile_id(101),
        command=SimpleNamespace(disposition=ConversationCommandDisposition.NEWLY_ACCEPTED),
        run=run,
    )
    release = asyncio.Event()
    terminal = SimpleNamespace(body="Completed through the protected lane")
    execution_result = CanonicalExecutionResult(
        CanonicalExecutionDisposition.COMPLETED,
        SimpleNamespace(status="completed"),
        terminal,
        session_id="session-1",
        workspace="/workspace/project",
        selection=SimpleNamespace(model="gpt-5.6-sol"),
    )

    async def execute(_run_id):
        await release.wait()
        return execution_result

    execution = SimpleNamespace(
        accept_client=AsyncMock(return_value=accepted),
        execute=AsyncMock(side_effect=execute),
        run_state=AsyncMock(return_value=run),
        recoverable_client_runs=AsyncMock(return_value=()),
        request_run_cancellation=AsyncMock(),
    )
    profile_state = SimpleNamespace(
        append_history=Mock(side_effect=["user-log", "assistant-log"]),
        save_session=AsyncMock(),
        schedule_memory_ingestion=Mock(),
    )
    compatibility_state = SimpleNamespace(for_profile=Mock(return_value=profile_state))
    executor = WorkshopClientCommandExecutor(execution, compatibility_state)
    await executor.start()
    try:
        submission = await executor.submit(message)
        await asyncio.sleep(0)

        assert submission.acceptance is accepted
        assert submission.run is run
        assert not release.is_set()
        execution.execute.assert_awaited_once_with(run_id)

    finally:
        release.set()
        await executor.stop()

    assert profile_state.append_history.call_args_list == [
        call(direction="user", text="Hello from Workshop"),
        call(direction="assistant", text="Completed through the protected lane"),
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


async def test_terminal_replay_does_not_schedule_or_duplicate_compatibility_writes():
    message = _message()
    run_id = RunId.new()
    run = SimpleNamespace(run_id=run_id, status="completed")
    accepted = SimpleNamespace(
        runtime_profile_id=profile_id(101),
        command=SimpleNamespace(disposition=ConversationCommandDisposition.TERMINAL_REPLAY),
        run=run,
    )
    execution = SimpleNamespace(
        accept_client=AsyncMock(return_value=accepted),
        execute=AsyncMock(),
        run_state=AsyncMock(return_value=run),
        recoverable_client_runs=AsyncMock(return_value=()),
        request_run_cancellation=AsyncMock(),
    )
    profile_state = SimpleNamespace(
        append_history=Mock(),
        save_session=AsyncMock(),
        schedule_memory_ingestion=Mock(),
    )
    compatibility_state = SimpleNamespace(for_profile=Mock(return_value=profile_state))
    executor = WorkshopClientCommandExecutor(execution, compatibility_state)
    await executor.start()
    try:
        await executor.submit(message)
    finally:
        await executor.stop()

    execution.execute.assert_not_awaited()
    compatibility_state.for_profile.assert_not_called()


async def test_start_reconciles_durably_accepted_browser_run():
    run_id = RunId.new()
    execution = SimpleNamespace(
        execute=AsyncMock(
            return_value=CanonicalExecutionResult(
                CanonicalExecutionDisposition.PREPARATION_DEFERRED,
                SimpleNamespace(status="accepted"),
            )
        ),
        recoverable_client_runs=AsyncMock(
            return_value=(
                RecoverableClientRun(
                    run_id,
                    profile_id(101),
                    "Recovered browser prompt",
                ),
            )
        ),
        request_run_cancellation=AsyncMock(),
    )
    executor = WorkshopClientCommandExecutor(execution, SimpleNamespace())

    await executor.start()
    await asyncio.sleep(0)
    await executor.stop()

    execution.execute.assert_awaited_once_with(run_id)
