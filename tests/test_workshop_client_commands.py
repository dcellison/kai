"""Compatibility contracts for asynchronous Workshop client execution."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

from kai.workshop.client_commands import WorkshopClientCommandExecutor
from kai.workshop.conversation_commands import ConversationCommandDisposition
from kai.workshop.domain import ChannelId, MessageId, PrincipalId, RunId
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


async def test_submission_returns_before_execution_without_owning_post_run_effects():
    message = _message()
    run_id = RunId.new()
    inbound_message_id = MessageId.new()
    result_message_id = MessageId.new()
    run = SimpleNamespace(run_id=run_id, status="accepted")
    accepted = SimpleNamespace(
        runtime_profile_id=profile_id(101),
        command=SimpleNamespace(
            disposition=ConversationCommandDisposition.NEWLY_ACCEPTED,
            message=SimpleNamespace(event=SimpleNamespace(envelope=SimpleNamespace(aggregate_id=inbound_message_id))),
        ),
        run=run,
    )
    release = asyncio.Event()
    terminal = SimpleNamespace(
        body="Completed through the protected lane",
        finalization=SimpleNamespace(
            message=SimpleNamespace(event=SimpleNamespace(envelope=SimpleNamespace(aggregate_id=result_message_id)))
        ),
    )
    execution_result = CanonicalExecutionResult(
        CanonicalExecutionDisposition.COMPLETED,
        SimpleNamespace(status="completed"),
        terminal,
        session_id="session-1",
        workspace="/workspace/project",
        selection=SimpleNamespace(model="gpt-5.6-sol"),
    )

    async def execute(_run_id, *, stream_observer=None):
        del stream_observer
        await release.wait()
        return execution_result

    execution = SimpleNamespace(
        accept_client=AsyncMock(return_value=accepted),
        execute=AsyncMock(side_effect=execute),
        run_state=AsyncMock(return_value=run),
        recoverable_client_runs=AsyncMock(return_value=()),
        request_run_cancellation=AsyncMock(),
        prior_conversation_pairs=AsyncMock(return_value=(("Earlier", "Exchange"),)),
    )
    executor = WorkshopClientCommandExecutor(execution)
    await executor.start()
    try:
        submission = await executor.submit(message)
        await asyncio.sleep(0)

        assert submission.acceptance is accepted
        assert submission.run is run
        assert not release.is_set()
        execution.execute.assert_awaited_once_with(run_id, stream_observer=None)

    finally:
        release.set()
        await executor.stop()


async def test_terminal_replay_does_not_schedule_execution():
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
    executor = WorkshopClientCommandExecutor(execution)
    await executor.start()
    try:
        await executor.submit(message)
    finally:
        await executor.stop()

    execution.execute.assert_not_awaited()


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
                    MessageId.new(),
                    "Recovered browser prompt",
                ),
            )
        ),
        request_run_cancellation=AsyncMock(),
    )
    executor = WorkshopClientCommandExecutor(execution)

    await executor.start()
    await asyncio.sleep(0)
    await executor.stop()

    execution.execute.assert_awaited_once_with(run_id, stream_observer=None)
