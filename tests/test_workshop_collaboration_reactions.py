"""Contracts for proof-bound agent-authored Workshop reactions."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from kai.workshop.collaboration_authority import (
    CollaborationDenied,
    CollaborationHostPolicy,
    CollaborationOperation,
    CollaborationOwnerPolicy,
    WorkshopCollaborationAuthority,
)
from kai.workshop.collaboration_reactions import (
    CollaborationReactionDenied,
    WorkshopCollaborationReactionService,
)
from kai.workshop.domain import WorkshopEventType
from kai.workshop.inbound import InboundMessage, record_inbound_message
from kai.workshop.message_reactions import load_message_reactors
from tests.test_workshop_collaboration_authority import (
    _NOW,
    _base_identity,
    _running_attempt,
)


class _Execution:
    def __init__(self, authority: WorkshopCollaborationAuthority) -> None:
        self.authority = authority

    async def authorize_collaboration(self, *args, **kwargs):
        return await self.authority.authorize(*args, **kwargs)


async def _reaction_context(path: Path, *, quota: int = 64):
    store, _execution, started = await _running_attempt(path, suffix="reaction")
    revision_id = started.run.agent_definition_revision_id
    assert revision_id is not None
    await store.connection.execute(
        "UPDATE agent_definition_revisions SET collaboration_operations_json = ? WHERE id = ?",
        ('["reaction"]', revision_id),
    )
    await store.connection.commit()
    authority = WorkshopCollaborationAuthority(
        store,
        host_policy=CollaborationHostPolicy(
            version=1,
            allowed_operations=frozenset({CollaborationOperation.REACTION}),
            quotas={CollaborationOperation.REACTION: quota},
        ),
        owner_policy_resolver=lambda _revision: CollaborationOwnerPolicy(
            version=1,
            allowed_operations=frozenset({CollaborationOperation.REACTION}),
        ),
        token_factory=lambda: "reaction-proof-0000000000000000000000000001",
    )
    grant, invocation = await authority.issue(
        started.claim,
        occurred_at=_NOW + timedelta(seconds=3),
    )
    return (
        store,
        started,
        grant,
        invocation,
        WorkshopCollaborationReactionService(
            store,
            _Execution(authority),
            clock=lambda: _NOW + timedelta(seconds=5),
        ),
    )


async def test_agent_reaction_is_attributed_idempotent_removable_and_rebuild_safe(
    tmp_path: Path,
) -> None:
    store, started, grant, invocation, service = await _reaction_context(tmp_path / "kai.db")
    target = started.run.inbound_message_id
    identity = _base_identity(started)
    try:
        added = await service.react(
            identity,
            proof=invocation.token,
            message_id=target,
            reaction="eyes",
            active=True,
            idempotency_key="agent-reaction-add",
        )
        assert added.active is True
        assert added.changed is True
        assert added.event_position is not None
        assert added.replayed is False

        async with store.connection.execute(
            "SELECT principal_id, agent_definition_revision_id, run_id, run_attempt_id, "
            "collaboration_grant_id FROM message_reactions WHERE message_id = ? AND reaction = 'eyes'",
            (target,),
        ) as cursor:
            reaction_row = await cursor.fetchone()
        assert reaction_row is not None
        assert tuple(str(value) for value in reaction_row) == (
            str(grant.agent_principal_id),
            str(grant.agent_definition_revision_id),
            str(grant.run_id),
            str(grant.attempt_id),
            str(grant.grant_id),
        )
        reactors = await load_message_reactors(
            store,
            viewer_principal_id=started.run.requested_by_principal_id,
            channel_id=started.run.channel_id,
            message_id=target,
            reaction="eyes",
        )
        assert reactors.total == 1
        assert reactors.truncated is False
        assert len(reactors.reactors) == 1
        assert reactors.reactors[0].principal_id == grant.agent_principal_id
        assert reactors.reactors[0].kind == "agent"
        assert reactors.reactors[0].handle == "kai"

        replay = await service.react(
            identity,
            proof=invocation.token,
            message_id=target,
            reaction="eyes",
            active=True,
            idempotency_key="agent-reaction-add",
        )
        assert replay == type(replay)(target, "eyes", True, True, added.event_position, True)

        unchanged = await service.react(
            identity,
            proof=invocation.token,
            message_id=target,
            reaction="eyes",
            active=True,
            idempotency_key="agent-reaction-add-again",
        )
        assert unchanged.changed is False
        assert unchanged.event_position is None

        removed = await service.react(
            identity,
            proof=invocation.token,
            message_id=target,
            reaction="eyes",
            active=False,
            idempotency_key="agent-reaction-remove",
        )
        assert removed.active is False
        assert removed.changed is True
        assert removed.event_position is not None

        async with store.connection.execute(
            "SELECT COUNT(*) FROM message_reactions WHERE message_id = ? AND principal_id = ?",
            (target, grant.agent_principal_id),
        ) as cursor:
            assert int((await cursor.fetchone())[0]) == 0
        async with store.connection.execute(
            "SELECT COUNT(*), SUM(is_error) FROM run_traces WHERE run_id = ?",
            (started.run.run_id,),
        ) as cursor:
            trace_count, trace_errors = await cursor.fetchone()
        assert (int(trace_count), int(trace_errors)) == (3, 0)
        async with store.connection.execute(
            "SELECT COUNT(*) FROM runs",
        ) as cursor:
            assert int((await cursor.fetchone())[0]) == 1
    finally:
        await store.close()


async def test_agent_reaction_denies_messages_newer_than_attempt_snapshot_and_records_denial(
    tmp_path: Path,
) -> None:
    store, started, grant, invocation, service = await _reaction_context(tmp_path / "kai.db")
    identity = _base_identity(started)
    try:
        future = await record_inbound_message(
            store,
            InboundMessage(
                transport="telegram",
                update_id="reaction-future",
                message_id="reaction-future-message",
                sender_subject="101",
                channel_subject="101",
                body="This message was not visible when the attempt began.",
                occurred_at=_NOW + timedelta(seconds=4),
            ),
        )
        with pytest.raises(CollaborationReactionDenied) as denied:
            await service.react(
                identity,
                proof=invocation.token,
                message_id=future.event.envelope.aggregate_id,
                reaction="question",
                active=True,
                idempotency_key="agent-reaction-future",
            )
        assert denied.value.code == "target_not_visible"

        async with store.connection.execute(
            "SELECT outcome, denial_code, agent_principal_id, run_id FROM "
            "collaboration_reaction_receipts WHERE grant_id = ? AND idempotency_key = ?",
            (grant.grant_id, "agent-reaction-future"),
        ) as cursor:
            receipt = await cursor.fetchone()
        assert receipt is not None
        assert tuple(str(value) for value in receipt) == (
            "denied",
            "target_not_visible",
            str(grant.agent_principal_id),
            str(grant.run_id),
        )
        async with store.connection.execute(
            "SELECT summary, is_error FROM run_traces WHERE run_id = ?",
            (started.run.run_id,),
        ) as cursor:
            trace = await cursor.fetchone()
        assert trace is not None
        assert str(trace[0]) == "Denied question reaction: target_not_visible"
        assert bool(trace[1]) is True

        with pytest.raises(CollaborationReactionDenied) as replayed_denial:
            await service.react(
                identity,
                proof=invocation.token,
                message_id=future.event.envelope.aggregate_id,
                reaction="question",
                active=True,
                idempotency_key="agent-reaction-future",
            )
        assert replayed_denial.value.code == "target_not_visible"
        async with store.connection.execute(
            "SELECT COUNT(*) FROM event_log WHERE event_type = ? AND aggregate_id = ?",
            (WorkshopEventType.MESSAGE_REACTION_ADDED.value, future.event.envelope.aggregate_id),
        ) as cursor:
            assert int((await cursor.fetchone())[0]) == 0
    finally:
        await store.close()


async def test_agent_reaction_quota_is_enforced_before_a_second_mutation(tmp_path: Path) -> None:
    store, started, grant, invocation, service = await _reaction_context(
        tmp_path / "kai.db",
        quota=1,
    )
    target = started.run.inbound_message_id
    identity = _base_identity(started)
    try:
        await service.react(
            identity,
            proof=invocation.token,
            message_id=target,
            reaction="check",
            active=True,
            idempotency_key="agent-reaction-quota-one",
        )
        with pytest.raises(CollaborationDenied) as denied:
            await service.react(
                identity,
                proof=invocation.token,
                message_id=target,
                reaction="check",
                active=False,
                idempotency_key="agent-reaction-quota-two",
            )
        assert denied.value.code == "quota_exhausted"
        async with store.connection.execute(
            "SELECT 1 FROM message_reactions WHERE message_id = ? AND principal_id = ? AND reaction = 'check'",
            (target, grant.agent_principal_id),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None and bool(row[0]) is True
        async with store.connection.execute(
            "SELECT decision, denial_code FROM collaboration_operation_decisions "
            "WHERE grant_id = ? AND idempotency_key = 'agent-reaction-quota-two'",
            (grant.grant_id,),
        ) as cursor:
            decision = await cursor.fetchone()
        assert decision is not None and tuple(decision) == ("denied", "quota_exhausted")
    finally:
        await store.close()
