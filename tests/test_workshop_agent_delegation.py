"""Bounded canonical agent-to-agent delegation contracts."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kai.workshop.agent_delegation import (
    AgentDelegationAuthority,
    AgentDelegationDenied,
    WorkshopAgentDelegationService,
)
from kai.workshop.conversation_commands import WorkshopConversationCommandService
from kai.workshop.domain import (
    MessageId,
    PrincipalId,
    RunExecutionOwnerId,
    RunId,
    RuntimeProfileId,
)
from kai.workshop.execution_coordinator import (
    CanonicalCancellationDisposition,
    CanonicalExecutionDisposition,
    CanonicalExecutionResult,
)
from kai.workshop.outbound import OutboundMessage, record_outbound_message
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.run_execution_authority import (
    RunExecutionSelection,
    WorkshopRunExecutionAuthority,
)
from kai.workshop.run_lifecycle import DurableRun, WorkshopRunLifecycle
from tests.test_workshop_wake_policy import _message, _open_group_store


class _CompletingExecution:
    def __init__(self, authority: WorkshopRunExecutionAuthority) -> None:
        self._authority = authority
        self._store = authority.event_store
        self.executed: list[RunId] = []

    async def execute(self, run_id: RunId) -> CanonicalExecutionResult:
        self.executed.append(run_id)
        now = datetime.now(UTC)
        granted = await self._authority.grant(
            run_id,
            owner_id=RunExecutionOwnerId.new(),
            occurred_at=now,
            lease_expires_at=now + timedelta(minutes=1),
        )
        started = await self._authority.start(
            granted.claim,
            occurred_at=now + timedelta(milliseconds=1),
        )
        outbound = await record_outbound_message(
            self._store,
            OutboundMessage(
                in_reply_to_message_id=started.run.inbound_message_id,
                body="Delegated result from Nova.",
                occurred_at=now + timedelta(milliseconds=2),
                agent_id=started.run.agent_id,
            ),
        )
        result_message_id = outbound.event.envelope.aggregate_id
        assert isinstance(result_message_id, MessageId)
        completed = await self._authority.complete(
            started.claim,
            result_message_id=result_message_id,
            occurred_at=now + timedelta(milliseconds=3),
        )
        return CanonicalExecutionResult(
            disposition=CanonicalExecutionDisposition.COMPLETED,
            run=completed.run,
        )

    async def run_state(self, run_id: RunId) -> DurableRun:
        return await WorkshopRunLifecycle(self._store).state(run_id)

    async def request_run_cancellation(
        self,
        run_id: RunId,
    ) -> CanonicalCancellationDisposition:
        return CanonicalCancellationDisposition.ALREADY_TERMINAL


async def _running_parent(path: Path):
    store, human_id, channel_id, agent_ids = await _open_group_store(path)
    caller_agent_id, target_agent_id = agent_ids
    await store.connection.execute(
        "UPDATE agent_definition_revisions SET capabilities_json = ? "
        "WHERE id = (SELECT active_revision_id FROM agent_definitions WHERE agent_id = ?)",
        (json.dumps(["agent_delegation", "text_generation"]), caller_agent_id),
    )
    await store.connection.commit()
    accepted = await WorkshopConversationCommandService(store).accept_client(
        _message(
            human_id,
            channel_id,
            "delegation-parent",
            "@Kai coordinate this task",
            datetime.now(UTC),
        )
    )
    assert len(accepted.command.lifecycles) == 1
    parent = accepted.command.lifecycles[0].run
    authority = WorkshopRunExecutionAuthority(
        store,
        selection_resolver=lambda _run: RunExecutionSelection("codex", "gpt-5.6-sol"),
        registered_backend_ids=frozenset({"codex"}),
    )
    now = datetime.now(UTC)
    granted = await authority.grant(
        parent.run_id,
        owner_id=RunExecutionOwnerId.new(),
        occurred_at=now,
        lease_expires_at=now + timedelta(minutes=1),
    )
    await authority.start(granted.claim, occurred_at=now + timedelta(milliseconds=1))
    async with store.connection.execute(
        "SELECT sponsor_principal_id, runtime_profile_id FROM runs WHERE id = ?",
        (parent.run_id,),
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None and row[0] is not None and row[1] is not None
    return (
        store,
        authority,
        AgentDelegationAuthority(
            sponsor_principal_id=PrincipalId(str(row[0])),
            channel_id=channel_id,
            caller_agent_id=caller_agent_id,
            runtime_profile_id=RuntimeProfileId(str(row[1])),
        ),
        parent,
        target_agent_id,
    )


async def test_explicit_delegation_is_visible_durable_bounded_and_idempotent(
    tmp_path: Path,
) -> None:
    store, authority, caller, parent, target_agent_id = await _running_parent(tmp_path / "kai.db")
    execution = _CompletingExecution(authority)
    service = WorkshopAgentDelegationService(store, execution)  # type: ignore[arg-type]
    await service.start()
    try:
        async with store.connection.execute("SELECT MAX(position) FROM event_log") as cursor:
            before_row = await cursor.fetchone()
        assert before_row is not None
        before_position = int(before_row[0])
        result = await service.delegate(
            caller,
            target_handle="nova",
            task="Return a bounded qualification result.",
            context={"summary": "Only shared channel context."},
            idempotency_key="delegation-one",
        )
        replay = await service.delegate(
            caller,
            target_handle="NOVA",
            task="Return a bounded qualification result.",
            context={"summary": "Only shared channel context."},
            idempotency_key="delegation-one",
        )

        assert result.response == "Delegated result from Nova."
        assert replay == result
        assert execution.executed == [result.delegation.child_run_id]
        assert result.delegation.parent_run_id == parent.run_id
        assert result.delegation.target_agent_id == target_agent_id
        assert result.delegation.status == "completed"
        child = await WorkshopRunLifecycle(store).state(result.delegation.child_run_id)
        assert child.parent_run_id == parent.run_id
        assert child.delegation_id == result.delegation.delegation_id
        async with store.connection.execute(
            "SELECT p.kind, m.body FROM messages m JOIN principals p "
            "ON p.id = m.author_principal_id WHERE m.id IN (?, ?) ORDER BY m.created_event_position",
            (
                result.delegation.request_message_id,
                result.delegation.response_message_id,
            ),
        ) as cursor:
            messages = [tuple(row) for row in await cursor.fetchall()]
        assert len(messages) == 2
        assert messages[0][0] == "agent"
        assert "@nova" in messages[0][1]
        assert messages[1] == ("agent", "Delegated result from Nova.")

        replay_events = await store.read_events(after_position=before_position)
        await store.connection.execute(
            "DELETE FROM agent_delegations WHERE id = ?",
            (result.delegation.delegation_id,),
        )
        await store.connection.execute(
            "DELETE FROM run_attempts WHERE run_id = ?",
            (result.delegation.child_run_id,),
        )
        await store.connection.execute(
            "DELETE FROM runs WHERE id = ?",
            (result.delegation.child_run_id,),
        )
        await store.connection.execute(
            "DELETE FROM messages WHERE id = ?",
            (result.delegation.response_message_id,),
        )
        await store.connection.execute(
            "DELETE FROM messages WHERE id = ?",
            (result.delegation.request_message_id,),
        )
        projection = CanonicalConversationProjection()
        for event in replay_events:
            await projection.apply(store.connection, event)
        await store.connection.commit()

        assert await service.snapshot(result.delegation.delegation_id) == result.delegation
    finally:
        await service.stop()
        await store.close()


async def test_delegation_rejects_cycles_before_creating_any_child_state(
    tmp_path: Path,
) -> None:
    store, authority, caller, _parent, _target_agent_id = await _running_parent(tmp_path / "kai.db")
    service = WorkshopAgentDelegationService(
        store,
        _CompletingExecution(authority),  # type: ignore[arg-type]
    )
    await service.start()
    try:
        with pytest.raises(AgentDelegationDenied) as denied:
            await service.delegate(
                caller,
                target_handle="kai",
                task="Delegate back to the caller.",
                idempotency_key="cycle",
            )
        assert denied.value.code == "cycle"
        async with store.connection.execute("SELECT COUNT(*) FROM agent_delegations") as cursor:
            assert int((await cursor.fetchone())[0]) == 0
    finally:
        await service.stop()
        await store.close()
