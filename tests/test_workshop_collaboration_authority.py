"""Contracts for exact-attempt Workshop collaboration authority."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kai.internal_api_auth import InternalAPIAuth
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.collaboration_authority import (
    CollaborationDenied,
    CollaborationHostPolicy,
    CollaborationOperation,
    CollaborationOwnerPolicy,
    CollaborationProofError,
    WorkshopCollaborationAuthority,
)
from kai.workshop.conversation_commands import WorkshopConversationCommandService
from kai.workshop.delivery_authority import WorkshopConversationDeliveryAuthority
from kai.workshop.domain import RunExecutionOwnerId, RuntimeProfileId
from kai.workshop.inbound import InboundMessage
from kai.workshop.internal_api_contexts import WorkshopInternalAPIExecutionContext
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.run_execution_authority import (
    RunExecutionSelection,
    WorkshopRunExecutionAuthority,
)
from kai.workshop.store import WorkshopEventStore

_NOW = datetime(2026, 9, 5, 12, 0, tzinfo=UTC)
_RUNTIME_PROFILE_ID = RuntimeProfileId.new()


async def _running_attempt(path: Path, *, suffix: str = "1"):
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman(
                display_name="Workshop Human",
                role="admin",
                transport="telegram",
                external_subject="101",
                external_channel_id="101",
                runtime_profile_id=_RUNTIME_PROFILE_ID,
            ),
        ),
    )
    accepted = await WorkshopConversationCommandService(store).accept(
        InboundMessage(
            transport="telegram",
            update_id=f"collaboration-command-{suffix}",
            message_id=f"collaboration-message-{suffix}",
            sender_subject="101",
            channel_subject="101",
            body=f"Perform collaboration qualification {suffix}",
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
    return store, execution, started


async def test_grant_snapshots_revision_owner_host_context_and_limits(tmp_path: Path) -> None:
    store, _execution, started = await _running_attempt(tmp_path / "kai.db")
    try:
        authority = WorkshopCollaborationAuthority(
            store,
            owner_policy_resolver=lambda _revision: CollaborationOwnerPolicy(
                version=7,
                allowed_operations=frozenset(
                    {
                        CollaborationOperation.AGENT_DELEGATION,
                        CollaborationOperation.CONTEXT_READ,
                    }
                ),
            ),
            token_factory=lambda: "attempt-proof-000000000000000000000000000001",
        )

        grant, invocation = await authority.issue(
            started.claim,
            occurred_at=_NOW + timedelta(seconds=3),
        )

        assert grant.attempt_id == started.claim.attempt_id
        assert grant.run_id == started.run.run_id
        assert grant.fence_token == started.claim.fence_token
        assert grant.agent_definition_revision_id == started.run.agent_definition_revision_id
        assert grant.requested_operations == frozenset({CollaborationOperation.AGENT_DELEGATION})
        assert grant.owner_allowed_operations == frozenset(
            {
                CollaborationOperation.AGENT_DELEGATION,
                CollaborationOperation.CONTEXT_READ,
            }
        )
        assert grant.effective_operations == frozenset({CollaborationOperation.AGENT_DELEGATION})
        assert grant.owner_policy_version == 7
        assert grant.host_policy_version == 1
        assert grant.quotas[CollaborationOperation.AGENT_DELEGATION] == 12
        assert grant.proof_fingerprint != invocation.token
        assert invocation.token not in repr(invocation)
        async with store.connection.execute(
            "SELECT payload_json, metadata_json FROM event_log WHERE aggregate_id = ? ORDER BY position",
            (grant.grant_id,),
        ) as cursor:
            serialized_events = "".join(str(value) for row in await cursor.fetchall() for value in row)
        assert invocation.token not in serialized_events
    finally:
        await store.close()


async def test_persistent_internal_credential_cannot_authenticate_collaboration(tmp_path: Path) -> None:
    store, _execution, started = await _running_attempt(tmp_path / "kai.db")
    try:
        authority = WorkshopCollaborationAuthority(store)
        await authority.issue(started.claim, occurred_at=_NOW + timedelta(seconds=3))
        runtime_profile_id = started.run.runtime_profile_id
        assert runtime_profile_id is not None
        context = WorkshopInternalAPIExecutionContext(
            principal_id=started.run.requested_by_principal_id,
            channel_id=started.run.channel_id,
            agent_id=started.run.agent_id,
            runtime_profile_id=runtime_profile_id,
        )
        persistent_token = InternalAPIAuth.for_execution_contexts((context,)).agent_credential_for(context)

        with pytest.raises(CollaborationProofError, match="Invalid collaboration proof"):
            await authority.authenticate(
                persistent_token,
                CollaborationOperation.AGENT_DELEGATION,
                occurred_at=_NOW + timedelta(seconds=4),
            )
    finally:
        await store.close()


async def test_exact_attempt_proof_fails_after_terminal_state_and_revocation(tmp_path: Path) -> None:
    store, execution, started = await _running_attempt(tmp_path / "kai.db")
    try:
        authority = WorkshopCollaborationAuthority(
            store,
            token_factory=lambda: "attempt-proof-000000000000000000000000000002",
        )
        grant, invocation = await authority.issue(
            started.claim,
            occurred_at=_NOW + timedelta(seconds=3),
        )
        authorized = await authority.authenticate(
            invocation.token,
            CollaborationOperation.AGENT_DELEGATION,
            occurred_at=_NOW + timedelta(seconds=4),
        )
        assert authorized.grant.grant_id == grant.grant_id

        await execution.fail(
            started.claim,
            failure_code="qualification_complete",
            occurred_at=_NOW + timedelta(seconds=5),
        )
        with pytest.raises(CollaborationDenied, match="no longer active"):
            await authority.authenticate(
                invocation.token,
                CollaborationOperation.AGENT_DELEGATION,
                occurred_at=_NOW + timedelta(seconds=6),
            )

        revoked, changed = await authority.revoke(
            invocation,
            revocation_code="attempt_terminal",
            occurred_at=_NOW + timedelta(seconds=6),
        )
        assert changed is True
        assert revoked.revocation_code == "attempt_terminal"
        with pytest.raises(CollaborationProofError, match="Invalid collaboration proof"):
            await authority.authenticate(
                invocation.token,
                CollaborationOperation.AGENT_DELEGATION,
                occurred_at=_NOW + timedelta(seconds=7),
            )
    finally:
        await store.close()


async def test_host_policy_can_deny_requested_operation_without_changing_revision(tmp_path: Path) -> None:
    store, _execution, started = await _running_attempt(tmp_path / "kai.db")
    try:
        authority = WorkshopCollaborationAuthority(
            store,
            host_policy=CollaborationHostPolicy(
                version=4,
                allowed_operations=frozenset(),
                quotas={},
            ),
            token_factory=lambda: "attempt-proof-000000000000000000000000000003",
        )
        grant, invocation = await authority.issue(
            started.claim,
            occurred_at=_NOW + timedelta(seconds=3),
        )

        assert grant.requested_operations == frozenset({CollaborationOperation.AGENT_DELEGATION})
        assert grant.effective_operations == frozenset()
        assert grant.quotas == {}
        with pytest.raises(CollaborationDenied) as denied:
            await authority.authenticate(
                invocation.token,
                CollaborationOperation.AGENT_DELEGATION,
                occurred_at=_NOW + timedelta(seconds=4),
            )
        assert denied.value.code == "operation_not_granted"
    finally:
        await store.close()


async def test_restart_does_not_recover_transient_proof_but_projection_recovers_snapshot(tmp_path: Path) -> None:
    store, _execution, started = await _running_attempt(tmp_path / "kai.db")
    try:
        first = WorkshopCollaborationAuthority(
            store,
            token_factory=lambda: "attempt-proof-000000000000000000000000000004",
        )
        grant, invocation = await first.issue(
            started.claim,
            occurred_at=_NOW + timedelta(seconds=3),
        )

        restarted = WorkshopCollaborationAuthority(store)
        with pytest.raises(CollaborationProofError, match="Invalid collaboration proof"):
            await restarted.authenticate(
                invocation.token,
                CollaborationOperation.AGENT_DELEGATION,
                occurred_at=_NOW + timedelta(seconds=4),
            )

        assert await restarted.reconcile_unbound(occurred_at=_NOW + timedelta(seconds=5)) == 1
        assert await restarted.reconcile_unbound(occurred_at=_NOW + timedelta(seconds=6)) == 0
        reconciled = await restarted.snapshot(grant.grant_id)
        assert reconciled.revocation_code == "host_restart"
        await store.rebuild_projection(CanonicalConversationProjection())
        replayed = await restarted.snapshot(grant.grant_id)
        assert replayed == reconciled
    finally:
        await store.close()


async def test_old_attempt_proof_cannot_act_during_later_attempt_on_same_runtime(
    tmp_path: Path,
) -> None:
    store, execution, first_started = await _running_attempt(tmp_path / "kai.db")
    try:
        tokens = iter(
            (
                "attempt-proof-000000000000000000000000000006",
                "attempt-proof-000000000000000000000000000007",
            )
        )
        authority = WorkshopCollaborationAuthority(store, token_factory=lambda: next(tokens))
        _first_grant, first_invocation = await authority.issue(
            first_started.claim,
            occurred_at=_NOW + timedelta(seconds=3),
        )
        await execution.fail(
            first_started.claim,
            failure_code="qualification_complete",
            occurred_at=_NOW + timedelta(seconds=4),
        )
        await authority.revoke(
            first_invocation,
            revocation_code="attempt_terminal",
            occurred_at=_NOW + timedelta(seconds=5),
        )

        accepted = await WorkshopConversationCommandService(store).accept(
            InboundMessage(
                transport="telegram",
                update_id="collaboration-command-later",
                message_id="collaboration-message-later",
                sender_subject="101",
                channel_subject="101",
                body="Perform later collaboration qualification",
                occurred_at=_NOW + timedelta(seconds=6),
            )
        )
        granted = await execution.grant(
            accepted.run.run_id,
            owner_id=RunExecutionOwnerId.new(),
            occurred_at=_NOW + timedelta(seconds=7),
            lease_expires_at=_NOW + timedelta(minutes=2),
        )
        second_started = await execution.start(
            granted.claim,
            occurred_at=_NOW + timedelta(seconds=8),
        )
        _second_grant, second_invocation = await authority.issue(
            second_started.claim,
            occurred_at=_NOW + timedelta(seconds=9),
        )

        with pytest.raises(CollaborationProofError, match="Invalid collaboration proof"):
            await authority.authenticate(
                first_invocation.token,
                CollaborationOperation.AGENT_DELEGATION,
                occurred_at=_NOW + timedelta(seconds=10),
            )
        authorized = await authority.authenticate(
            second_invocation.token,
            CollaborationOperation.AGENT_DELEGATION,
            occurred_at=_NOW + timedelta(seconds=10),
        )
        assert authorized.grant.attempt_id == second_started.claim.attempt_id
    finally:
        await store.close()


async def test_archive_and_detach_immediately_fence_live_grant(tmp_path: Path) -> None:
    store, _execution, started = await _running_attempt(tmp_path / "kai.db")
    try:
        authority = WorkshopCollaborationAuthority(
            store,
            token_factory=lambda: "attempt-proof-000000000000000000000000000005",
        )
        _grant, invocation = await authority.issue(
            started.claim,
            occurred_at=_NOW + timedelta(seconds=3),
        )
        await store.connection.execute(
            "UPDATE agent_definitions SET lifecycle_state = 'archived' WHERE agent_id = ?",
            (started.run.agent_id,),
        )
        await store.connection.commit()

        with pytest.raises(CollaborationDenied) as denied:
            await authority.authenticate(
                invocation.token,
                CollaborationOperation.AGENT_DELEGATION,
                occurred_at=_NOW + timedelta(seconds=4),
            )
        assert denied.value.code == "agent_unavailable"
    finally:
        await store.close()


async def test_version_sixty_seven_migrates_only_legacy_delegation_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kai.workshop import schema

    path = tmp_path / "kai.db"
    with monkeypatch.context() as migration_context:
        migration_context.setattr(schema, "WORKSHOP_SCHEMA_VERSION", 66)
        migration_context.setattr(schema, "_MIGRATIONS", schema._MIGRATIONS[:66])
        legacy = await WorkshopEventStore.open(path)
        try:
            await bootstrap_default_workshop(
                legacy,
                (
                    BootstrapHuman(
                        display_name="Workshop Human",
                        role="admin",
                        transport="telegram",
                        external_subject="101",
                        external_channel_id="101",
                        runtime_profile_id=_RUNTIME_PROFILE_ID,
                    ),
                ),
            )
            async with legacy.connection.execute(
                "SELECT id FROM agent_definition_revisions ORDER BY revision_number LIMIT 1"
            ) as cursor:
                row = await cursor.fetchone()
            assert row is not None
            no_delegation_revision_id = str(row[0])
            await legacy.connection.execute(
                "UPDATE agent_definition_revisions SET capabilities_json = '[]' WHERE id = ?",
                (no_delegation_revision_id,),
            )
            await legacy.connection.commit()
        finally:
            await legacy.close()

    upgraded = await WorkshopEventStore.open(path)
    try:
        assert await upgraded.schema_version() == 67
        async with upgraded.connection.execute(
            "SELECT id, capabilities_json, collaboration_operations_json "
            "FROM agent_definition_revisions ORDER BY revision_number"
        ) as cursor:
            rows = list(await cursor.fetchall())
        assert rows
        for revision_id, capabilities_json, operations_json in rows:
            capabilities = json.loads(str(capabilities_json))
            operations = json.loads(str(operations_json))
            if str(revision_id) == no_delegation_revision_id:
                assert capabilities == []
                assert operations == []
            else:
                assert operations == (["agent_delegation"] if "agent_delegation" in capabilities else [])
    finally:
        await upgraded.close()
