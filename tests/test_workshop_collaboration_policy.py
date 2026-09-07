"""Contracts for owner-managed Workshop collaboration policy."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.collaboration_authority import (
    CollaborationBaseIdentity,
    CollaborationOperation,
    CollaborationProofError,
    WorkshopCollaborationAuthority,
)
from kai.workshop.collaboration_policy import (
    WorkshopCollaborationPolicyAccessDenied,
    WorkshopCollaborationPolicyConflict,
    WorkshopCollaborationPolicyService,
)
from kai.workshop.conversation_commands import WorkshopConversationCommandService
from kai.workshop.delivery_authority import WorkshopConversationDeliveryAuthority
from kai.workshop.domain import AgentDefinitionId, PrincipalId, RunExecutionOwnerId, RuntimeProfileId
from kai.workshop.inbound import InboundMessage
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.run_execution_authority import RunExecutionSelection, WorkshopRunExecutionAuthority
from kai.workshop.store import WorkshopEventStore

_NOW = datetime(2026, 9, 7, 12, 0, tzinfo=UTC)
_ALICE_RUNTIME = RuntimeProfileId.new()
_BOB_RUNTIME = RuntimeProfileId.new()


class _PrivateExecution:
    def __init__(self, authority: WorkshopCollaborationAuthority) -> None:
        self.collaboration_authority = authority

    async def revoke_collaboration_for_definition(
        self,
        definition_id: AgentDefinitionId,
        *,
        occurred_at: datetime,
    ) -> int:
        return await self.collaboration_authority.revoke_definition(
            definition_id,
            occurred_at=occurred_at,
        )


async def _identity(store: WorkshopEventStore, subject: str) -> PrincipalId:
    async with store.connection.execute(
        "SELECT principal_id FROM external_identities WHERE provider = 'telegram' AND external_subject = ?",
        (subject,),
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return PrincipalId(str(row[0]))


async def _started_attempt(path: Path):
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman("Alice", "admin", "telegram", "101", "101", _ALICE_RUNTIME),
            BootstrapHuman("Bob", "member", "telegram", "202", "202", _BOB_RUNTIME),
        ),
    )
    alice_id = await _identity(store, "101")
    bob_id = await _identity(store, "202")
    command_service = WorkshopConversationCommandService(store)
    accepted = await command_service.accept(
        InboundMessage(
            transport="telegram",
            update_id="policy-command-1",
            message_id="policy-message-1",
            sender_subject="101",
            channel_subject="101",
            body="Perform policy qualification",
            occurred_at=_NOW,
        )
    )
    await WorkshopConversationDeliveryAuthority(store).activate()
    execution = WorkshopRunExecutionAuthority(
        store,
        selection_resolver=lambda _run: RunExecutionSelection("codex", "gpt-5.6-sol"),
        registered_backend_ids=frozenset({"codex"}),
    )
    granted = await execution.grant(
        accepted.run.run_id,
        owner_id=RunExecutionOwnerId.new(),
        occurred_at=_NOW + timedelta(seconds=1),
        lease_expires_at=_NOW + timedelta(minutes=1),
    )
    started = await execution.start(granted.claim, occurred_at=_NOW + timedelta(seconds=2))
    async with store.connection.execute(
        "SELECT agent_definition_id FROM agent_definition_revisions WHERE id = ?",
        (started.run.agent_definition_revision_id,),
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return (
        store,
        alice_id,
        bob_id,
        command_service,
        execution,
        started,
        AgentDefinitionId(str(row[0])),
    )


def _base_identity(started: Any) -> CollaborationBaseIdentity:
    assert started.run.runtime_profile_id is not None
    return CollaborationBaseIdentity(
        principal_id=started.run.requested_by_principal_id,
        channel_id=started.run.channel_id,
        agent_id=started.run.agent_id,
        runtime_profile_id=started.run.runtime_profile_id,
    )


async def test_owner_policy_is_canonical_versioned_replayable_and_read_only_to_members(
    tmp_path: Path,
) -> None:
    store, alice_id, bob_id, _commands, _execution, _started, definition_id = await _started_attempt(
        tmp_path / "kai.db"
    )
    try:
        authority = WorkshopCollaborationAuthority(store)
        service = WorkshopCollaborationPolicyService(
            store,
            cast(Any, _PrivateExecution(authority)),
        )

        migrated = await service.inspect(alice_id, definition_id)
        delegation = next(
            item for item in migrated.operations if item.operation == CollaborationOperation.AGENT_DELEGATION
        )
        assert migrated.can_manage is True
        assert migrated.policy_version == 0
        assert delegation.requested is True
        assert delegation.owner_allowed is True
        assert delegation.effective_for_new_attempt is True

        member_view = await service.inspect(bob_id, definition_id)
        assert member_view.can_manage is False
        assert member_view.active_grants is None
        assert all(item.owner_allowed is None for item in member_view.operations)
        with pytest.raises(WorkshopCollaborationPolicyAccessDenied):
            await service.set_allowed(
                bob_id,
                definition_id,
                allowed_operations=[],
                expected_policy_version=0,
                client_operation_id="member-policy",
            )

        first = await service.set_allowed(
            alice_id,
            definition_id,
            allowed_operations=[],
            expected_policy_version=0,
            client_operation_id="owner-policy-1",
        )
        assert first.changed is True
        assert first.replayed is False
        assert first.snapshot.policy_version == 1
        assert not any(item.owner_allowed for item in first.snapshot.operations)
        member_blocked = await service.inspect(bob_id, definition_id)
        member_delegation = next(
            item for item in member_blocked.operations if item.operation == CollaborationOperation.AGENT_DELEGATION
        )
        assert member_delegation.owner_allowed is None
        assert member_delegation.effective_for_new_attempt is False
        assert member_delegation.unavailable_reason == "Unavailable under current agent policy"

        replay = await service.set_allowed(
            alice_id,
            definition_id,
            allowed_operations=[],
            expected_policy_version=0,
            client_operation_id="owner-policy-1",
        )
        assert replay.replayed is True
        assert replay.changed is False
        assert replay.snapshot == first.snapshot
        with pytest.raises(WorkshopCollaborationPolicyConflict):
            await service.set_allowed(
                alice_id,
                definition_id,
                allowed_operations=[CollaborationOperation.AGENT_DELEGATION.value],
                expected_policy_version=0,
                client_operation_id="owner-policy-1",
            )
        with pytest.raises(WorkshopCollaborationPolicyConflict):
            await service.set_allowed(
                alice_id,
                definition_id,
                allowed_operations=[],
                expected_policy_version=0,
                client_operation_id="owner-policy-stale",
            )

        await store.rebuild_projection(CanonicalConversationProjection())
        assert await service.inspect(alice_id, definition_id) == first.snapshot
    finally:
        await store.close()


async def test_policy_reports_detached_agent_without_exposing_owner_runtime_details(
    tmp_path: Path,
) -> None:
    store, alice_id, bob_id, _commands, _execution, _started, definition_id = await _started_attempt(
        tmp_path / "kai.db"
    )
    try:
        authority = WorkshopCollaborationAuthority(store)
        service = WorkshopCollaborationPolicyService(store, cast(Any, _PrivateExecution(authority)))
        await store.connection.execute(
            "UPDATE channel_agents SET detached_at = ? WHERE agent_id = "
            "(SELECT agent_id FROM agent_definitions WHERE id = ?)",
            (_NOW.isoformat(), definition_id),
        )
        await store.connection.commit()

        owner = await service.inspect(alice_id, definition_id)
        owner_delegation = next(
            item for item in owner.operations if item.operation == CollaborationOperation.AGENT_DELEGATION
        )
        assert owner_delegation.effective_for_new_attempt is False
        assert owner_delegation.unavailable_reason == "Agent is not attached to an active conversation"

        member = await service.inspect(bob_id, definition_id)
        member_delegation = next(
            item for item in member.operations if item.operation == CollaborationOperation.AGENT_DELEGATION
        )
        assert member_delegation.owner_allowed is None
        assert member_delegation.unavailable_reason == "Unavailable under current agent policy"
    finally:
        await store.close()


async def test_policy_changes_only_future_grants_and_emergency_revoke_fences_live_proof(
    tmp_path: Path,
) -> None:
    store, alice_id, _bob_id, commands, execution, started, definition_id = await _started_attempt(tmp_path / "kai.db")
    try:
        tokens = iter(
            (
                "policy-proof-000000000000000000000000000001",
                "policy-proof-000000000000000000000000000002",
            )
        )
        authority = WorkshopCollaborationAuthority(
            store,
            token_factory=lambda: next(tokens),
        )
        service = WorkshopCollaborationPolicyService(
            store,
            cast(Any, _PrivateExecution(authority)),
        )
        initial_grant, invocation = await authority.issue(
            started.claim,
            occurred_at=_NOW + timedelta(seconds=3),
        )
        assert CollaborationOperation.AGENT_DELEGATION in initial_grant.effective_operations

        updated = await service.set_allowed(
            alice_id,
            definition_id,
            allowed_operations=[],
            expected_policy_version=0,
            client_operation_id="future-policy",
        )
        assert updated.snapshot.policy_version == 1
        authorized = await authority.authorize(
            invocation.token,
            CollaborationOperation.AGENT_DELEGATION,
            base_identity=_base_identity(started),
            idempotency_key="existing-attempt-operation",
            request_hash="a" * 64,
            occurred_at=_NOW + timedelta(seconds=4),
        )
        assert authorized.replayed is False

        revoked_snapshot, revoked_count = await service.revoke_active(alice_id, definition_id)
        assert revoked_count == 1
        assert revoked_snapshot.active_grants == 0
        with pytest.raises(CollaborationProofError):
            await authority.authenticate(
                invocation.token,
                CollaborationOperation.AGENT_DELEGATION,
                occurred_at=_NOW + timedelta(seconds=5),
            )
        activity = await service.activity(alice_id, started.run.run_id)
        assert [(item.kind, item.outcome, item.detail) for item in activity] == [
            ("operation", "authorized", "quota use 1"),
            ("revocation", "revoked", "owner_emergency_revoke"),
        ]

        await execution.fail(
            started.claim,
            failure_code="qualification_complete",
            occurred_at=_NOW + timedelta(seconds=6),
        )
        accepted = await commands.accept(
            InboundMessage(
                transport="telegram",
                update_id="policy-command-2",
                message_id="policy-message-2",
                sender_subject="101",
                channel_subject="101",
                body="Perform later policy qualification",
                occurred_at=_NOW + timedelta(seconds=7),
            )
        )
        granted = await execution.grant(
            accepted.run.run_id,
            owner_id=RunExecutionOwnerId.new(),
            occurred_at=_NOW + timedelta(seconds=8),
            lease_expires_at=_NOW + timedelta(minutes=2),
        )
        later = await execution.start(granted.claim, occurred_at=_NOW + timedelta(seconds=9))
        later_grant, _later_invocation = await authority.issue(
            later.claim,
            occurred_at=_NOW + timedelta(seconds=10),
        )
        assert later_grant.owner_policy_version == 1
        assert later_grant.effective_operations == frozenset()
    finally:
        await store.close()
