"""Contracts for proof-bound agent-authored Workshop publications."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest

from kai.workshop.artifacts import WorkshopArtifactService
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.collaboration_authority import (
    CollaborationDenied,
    CollaborationHostPolicy,
    CollaborationOperation,
    CollaborationOwnerPolicy,
    WorkshopCollaborationAuthority,
)
from kai.workshop.collaboration_publications import (
    WorkshopCollaborationPublicationService,
)
from kai.workshop.conversation_commands import WorkshopConversationCommandService
from kai.workshop.delivery_authority import WorkshopConversationDeliveryAuthority
from kai.workshop.delivery_policy import WorkshopDeliveryBindingPolicy
from kai.workshop.domain import (
    AgentDefinitionId,
    AgentDefinitionRevisionId,
    EventEnvelope,
    RunExecutionOwnerId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.inbound import InboundMessage
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.run_execution_authority import RunExecutionSelection, WorkshopRunExecutionAuthority
from kai.workshop.runtime_profiles import ProtectedRuntimeProfile, WorkshopRuntimeProfileRegistry
from kai.workshop.storage_namespaces import WorkshopPrincipalStorageRegistry
from kai.workshop.store import WorkshopEventStore
from tests.test_workshop_collaboration_authority import _NOW, _base_identity
from tests.test_workshop_wake_policy import (
    _accepted_message_id,
    _message,
    _open_group_store,
)
from tests.workshop_profiles import profile_id, profile_registry


class _Execution:
    def __init__(self, authority: WorkshopCollaborationAuthority) -> None:
        self.authority = authority

    async def authorize_collaboration(self, *args, **kwargs):
        return await self.authority.authorize(*args, **kwargs)


async def _publication_context(path: Path):
    runtime_profile_id = profile_id(101)
    store = await WorkshopEventStore.open(path / "kai.db")
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman(
                "Workshop Human",
                "admin",
                "telegram",
                "101",
                "101",
                runtime_profile_id,
            ),
        ),
    )
    async with store.connection.execute(
        "SELECT d.id, d.workshop_id, d.active_revision_id, r.revision_number "
        "FROM agent_definitions d JOIN agent_definition_revisions r "
        "ON r.id = d.active_revision_id WHERE d.handle = 'kai'",
    ) as cursor:
        definition = await cursor.fetchone()
    assert definition is not None
    definition_id = AgentDefinitionId(str(definition[0]))
    workshop_id = WorkshopId(str(definition[1]))
    revision_id = AgentDefinitionRevisionId.derived(definition_id, "publication-revision")
    for event in (
        EventEnvelope.create(
            event_type=WorkshopEventType.AGENT_DEFINITION_REVISION_ADDED,
            event_version=2,
            workshop_id=workshop_id,
            aggregate_type="agent_definition_revision",
            aggregate_id=revision_id,
            occurred_at=_NOW - timedelta(seconds=2),
            payload={
                "definition_id": definition_id,
                "revision_number": int(definition[3]) + 1,
                "purpose": "Publish bounded collaboration progress.",
                "instructions": "Use only explicitly granted collaboration publication tools.",
                "capabilities": ["text_generation"],
                "collaboration_operations": ["artifact_publish", "progress_publish"],
            },
        ),
        EventEnvelope.create(
            event_type=WorkshopEventType.AGENT_DEFINITION_REVISION_ACTIVATED,
            event_version=1,
            workshop_id=workshop_id,
            aggregate_type="agent_definition",
            aggregate_id=definition_id,
            occurred_at=_NOW - timedelta(seconds=1),
            payload={"revision_id": revision_id},
        ),
    ):
        await store.append(event)
        await store.project_pending(CanonicalConversationProjection())
    accepted = await WorkshopConversationCommandService(store).accept(
        InboundMessage(
            "telegram",
            "publication-command",
            "publication-message",
            "101",
            "101",
            "Perform publication qualification",
            _NOW,
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
    assert started.run.agent_definition_revision_id == revision_id
    operations = frozenset(
        {
            CollaborationOperation.PROGRESS_PUBLISH,
            CollaborationOperation.ARTIFACT_PUBLISH,
        }
    )
    authority = WorkshopCollaborationAuthority(
        store,
        host_policy=CollaborationHostPolicy(
            allowed_operations=operations,
            quotas={operation: 4 for operation in operations},
        ),
        owner_policy_resolver=lambda _revision: CollaborationOwnerPolicy(
            version=1,
            allowed_operations=operations,
        ),
        token_factory=lambda: "publication-proof-00000000000000000000000001",
    )
    grant, invocation = await authority.issue(
        started.claim,
        occurred_at=_NOW + timedelta(seconds=3),
    )
    profiles = WorkshopRuntimeProfileRegistry(
        (
            ProtectedRuntimeProfile(
                profile_id=runtime_profile_id,
                display_name="Publication owner",
                os_user=None,
                backend="codex",
                provider="openai",
                model="gpt-5.6-sol",
                timeout_seconds=120,
                allowed_services=(),
                home_workspace=None,
                workspace_base=None,
                allowed_workspaces=(),
            ),
        )
    )
    storage = await WorkshopPrincipalStorageRegistry.from_store(store, profiles)
    artifacts = WorkshopArtifactService(
        store,
        data_dir=path,
        principal_storage=storage,
        runtime_profiles=profiles,
    )
    service = WorkshopCollaborationPublicationService(
        store,
        _Execution(authority),
        artifacts,
        data_dir=path,
        delivery_policy=WorkshopDeliveryBindingPolicy(frozenset()),
        clock=lambda: _NOW + timedelta(seconds=5),
    )
    return store, started, grant, invocation, service


async def test_progress_and_artifact_are_attributed_idempotent_and_rebuild_safe(tmp_path: Path) -> None:
    store, started, grant, invocation, service = await _publication_context(tmp_path)
    identity = _base_identity(started)
    source = tmp_path / "qualification.txt"
    source.write_text("bounded artifact\n")
    try:
        progress = await service.publish_message(
            identity,
            proof=invocation.token,
            kind="progress",
            body="Bounded progress",
            idempotency_key="progress-1",
        )
        replay = await service.publish_message(
            identity,
            proof=invocation.token,
            kind="progress",
            body="Bounded progress",
            idempotency_key="progress-1",
        )
        assert replay == type(replay)(progress.message_id, None, progress.event_position, True)

        artifact = await service.publish_artifact(
            identity,
            proof=invocation.token,
            path=source,
            caption="Qualification artifact",
            idempotency_key="artifact-1",
        )
        assert artifact.artifact_id is not None
        artifact_replay = await service.publish_artifact(
            identity,
            proof=invocation.token,
            path=source,
            caption="Qualification artifact",
            idempotency_key="artifact-1",
        )
        assert artifact_replay == type(artifact_replay)(
            artifact.message_id,
            artifact.artifact_id,
            artifact.event_position,
            True,
        )
        async with store.connection.execute(
            "SELECT author_principal_id, agent_definition_revision_id, run_id, run_attempt_id, "
            "collaboration_grant_id, collaboration_operation FROM messages "
            "WHERE id IN (?, ?) ORDER BY created_event_position",
            (progress.message_id, artifact.message_id),
        ) as cursor:
            rows = tuple(await cursor.fetchall())
        expected_prefix = (
            str(grant.agent_principal_id),
            str(grant.agent_definition_revision_id),
            str(grant.run_id),
            str(grant.attempt_id),
            str(grant.grant_id),
        )
        assert tuple(str(value) for value in rows[0]) == (*expected_prefix, "progress_publish")
        assert tuple(str(value) for value in rows[1]) == (*expected_prefix, "artifact_publish")
        async with store.connection.execute(
            "SELECT created_by_principal_id, agent_definition_revision_id, run_id, run_attempt_id, "
            "collaboration_grant_id, storage_path FROM artifacts WHERE id = ?",
            (artifact.artifact_id,),
        ) as cursor:
            artifact_row = await cursor.fetchone()
        assert artifact_row is not None
        assert tuple(str(value) for value in artifact_row[:5]) == expected_prefix
        assert str(grant.sponsor_principal_id) in str(artifact_row[5])
        async with store.connection.execute("SELECT COUNT(*) FROM delivery_outbox") as cursor:
            assert int((await cursor.fetchone())[0]) == 0

        await store.rebuild_projection(CanonicalConversationProjection())
        async with store.connection.execute(
            "SELECT COUNT(*) FROM collaboration_publication_receipts WHERE grant_id = ?",
            (grant.grant_id,),
        ) as cursor:
            assert int((await cursor.fetchone())[0]) == 2
        async with store.connection.execute(
            "SELECT COUNT(*) FROM messages WHERE collaboration_grant_id = ?",
            (grant.grant_id,),
        ) as cursor:
            assert int((await cursor.fetchone())[0]) == 2
    finally:
        await store.close()


async def test_detached_attempt_fails_closed_after_authorization_and_records_no_message(tmp_path: Path) -> None:
    store, started, grant, invocation, service = await _publication_context(tmp_path)
    identity = _base_identity(started)
    try:
        await store.connection.execute(
            "UPDATE channel_agents SET detached_at = ? WHERE channel_id = ? AND agent_id = ?",
            ((_NOW + timedelta(seconds=4)).isoformat(), grant.channel_id, grant.agent_id),
        )
        await store.connection.commit()
        with pytest.raises(CollaborationDenied) as denied:
            await service.publish_message(
                identity,
                proof=invocation.token,
                kind="progress",
                body="Must not appear",
                idempotency_key="detached-progress",
            )
        assert denied.value.code == "authority_detached"
        async with store.connection.execute(
            "SELECT decision, denial_code FROM collaboration_operation_decisions "
            "WHERE grant_id = ? AND idempotency_key = 'detached-progress'",
            (grant.grant_id,),
        ) as cursor:
            assert tuple(await cursor.fetchone()) == ("denied", "authority_detached")
        async with store.connection.execute(
            "SELECT COUNT(*) FROM messages WHERE collaboration_grant_id = ?",
            (grant.grant_id,),
        ) as cursor:
            assert int((await cursor.fetchone())[0]) == 0
    finally:
        await store.close()


async def test_thread_reply_derives_root_notifies_human_and_never_wakes_mentioned_agent(
    tmp_path: Path,
) -> None:
    store, human_id, group_id, agent_ids = await _open_group_store(tmp_path / "thread.db")
    try:
        async with store.connection.execute(
            "SELECT d.id, d.workshop_id, r.revision_number FROM agent_definitions d "
            "JOIN agent_definition_revisions r ON r.id = d.active_revision_id WHERE d.agent_id = ?",
            (agent_ids[0],),
        ) as cursor:
            definition = await cursor.fetchone()
        assert definition is not None
        definition_id = AgentDefinitionId(str(definition[0]))
        workshop_id = WorkshopId(str(definition[1]))
        revision_id = AgentDefinitionRevisionId.derived(definition_id, "thread-publication")
        for event in (
            EventEnvelope.create(
                event_type=WorkshopEventType.AGENT_DEFINITION_REVISION_ADDED,
                event_version=2,
                workshop_id=workshop_id,
                aggregate_type="agent_definition_revision",
                aggregate_id=revision_id,
                occurred_at=_NOW - timedelta(seconds=2),
                payload={
                    "definition_id": definition_id,
                    "revision_number": int(definition[2]) + 1,
                    "purpose": "Publish bounded thread replies.",
                    "instructions": "Reply only in the exact granted thread.",
                    "capabilities": ["text_generation"],
                    "collaboration_operations": ["thread_reply"],
                },
            ),
            EventEnvelope.create(
                event_type=WorkshopEventType.AGENT_DEFINITION_REVISION_ACTIVATED,
                event_version=1,
                workshop_id=workshop_id,
                aggregate_type="agent_definition",
                aggregate_id=definition_id,
                occurred_at=_NOW - timedelta(seconds=1),
                payload={"revision_id": revision_id},
            ),
        ):
            await store.append(event)
            await store.project_pending(CanonicalConversationProjection())
        commands = WorkshopConversationCommandService(store)
        root = await commands.accept_client(_message(human_id, group_id, "root", "Root", _NOW))
        root_id = _accepted_message_id(root)
        accepted = await commands.accept_client(
            _message(
                human_id,
                group_id,
                "thread-command",
                "@kai work in this thread",
                _NOW + timedelta(seconds=1),
                thread_root_id=root_id,
            )
        )
        assert len(accepted.command.runs) == 1
        run = accepted.command.runs[0]
        await WorkshopConversationDeliveryAuthority(store).activate()
        execution = WorkshopRunExecutionAuthority(
            store,
            selection_resolver=lambda _run: RunExecutionSelection("codex", "gpt-5.6-sol"),
            registered_backend_ids=frozenset({"codex"}),
        )
        granted = await execution.grant(
            run.run_id,
            owner_id=RunExecutionOwnerId.new(),
            occurred_at=_NOW + timedelta(seconds=2),
            lease_expires_at=_NOW + timedelta(minutes=1),
        )
        started = await execution.start(granted.claim, occurred_at=_NOW + timedelta(seconds=3))
        operations = frozenset({CollaborationOperation.THREAD_REPLY})
        authority = WorkshopCollaborationAuthority(
            store,
            host_policy=CollaborationHostPolicy(
                allowed_operations=operations,
                quotas={CollaborationOperation.THREAD_REPLY: 2},
            ),
            owner_policy_resolver=lambda _revision: CollaborationOwnerPolicy(1, operations),
            token_factory=lambda: "thread-publication-proof-000000000000000000001",
        )
        grant, invocation = await authority.issue(
            started.claim,
            occurred_at=_NOW + timedelta(seconds=4),
        )
        profiles = profile_registry(101, 202)
        storage = await WorkshopPrincipalStorageRegistry.from_store(store, profiles)
        service = WorkshopCollaborationPublicationService(
            store,
            _Execution(authority),
            WorkshopArtifactService(
                store,
                data_dir=tmp_path,
                principal_storage=storage,
                runtime_profiles=profiles,
            ),
            data_dir=tmp_path,
            delivery_policy=WorkshopDeliveryBindingPolicy(frozenset()),
            clock=lambda: _NOW + timedelta(seconds=5),
        )
        result = await service.publish_message(
            _base_identity(started),
            proof=invocation.token,
            kind="thread_reply",
            body="@Daniel bounded reply; @Nova is not activated.",
            idempotency_key="thread-reply-1",
        )
        async with store.connection.execute(
            "SELECT reply_to_message_id, thread_root_id FROM messages WHERE id = ?",
            (result.message_id,),
        ) as cursor:
            assert tuple(await cursor.fetchone()) == (root_id, root_id)
        async with store.connection.execute(
            "SELECT COUNT(*) FROM human_notifications WHERE source_message_id = ? AND recipient_principal_id = ?",
            (result.message_id, human_id),
        ) as cursor:
            assert int((await cursor.fetchone())[0]) == 1
        from kai.workshop.wake_policy import resolve_message_wake_targets

        assert (await resolve_message_wake_targets(store, result.message_id)).agent_ids == ()
        assert grant.thread_root_id == root_id
    finally:
        await store.close()
