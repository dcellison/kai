"""Principal-agent enablement, isolation, and continuity contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kai.workshop.agent_authority import (
    WorkshopAgentAuthorityError,
    reconcile_single_owner_agent_authority,
)
from kai.workshop.agent_enablement import (
    WorkshopAgentEnablementAccessDenied,
    WorkshopAgentEnablementService,
)
from kai.workshop.agent_lifecycle import (
    WorkshopAgentLifecycleAccessDenied,
    WorkshopAgentLifecycleService,
)
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.client_api import _read_agent_lifecycle_events
from kai.workshop.conversation_runs import resolve_canonical_conversation_run
from kai.workshop.diagnostics import workshop_agent_authority_status
from kai.workshop.domain import AgentDefinitionId, AgentEnablementId, ChannelId, MessageId, PrincipalId
from kai.workshop.execution_state import WorkshopExecutionStateRegistry
from kai.workshop.inbound import InboundMessage, record_inbound_message
from kai.workshop.internal_api_contexts import (
    WorkshopInternalAPIContextRegistry,
    WorkshopInternalAPIExecutionContext,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id, profile_registry


@dataclass
class _RuntimePool:
    registered: list[WorkshopInternalAPIExecutionContext] = field(default_factory=list)
    suspended: list[WorkshopInternalAPIExecutionContext] = field(default_factory=list)
    rebound: list[tuple[WorkshopInternalAPIExecutionContext, WorkshopInternalAPIExecutionContext]] = field(
        default_factory=list
    )

    def register_canonical_lane(self, context: WorkshopInternalAPIExecutionContext) -> None:
        self.registered.append(context)

    async def suspend_canonical_lane(self, context: WorkshopInternalAPIExecutionContext) -> None:
        self.suspended.append(context)

    async def rebind_canonical_lane(
        self,
        prior: WorkshopInternalAPIExecutionContext,
        replacement: WorkshopInternalAPIExecutionContext,
    ) -> None:
        self.rebound.append((prior, replacement))


async def _principal(store: WorkshopEventStore, external_subject: str) -> PrincipalId:
    async with store.connection.execute(
        "SELECT principal_id FROM external_identities WHERE provider = 'telegram' AND external_subject = ?",
        (external_subject,),
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return PrincipalId(str(row[0]))


async def _active_specialist(
    store: WorkshopEventStore,
    principal_id: PrincipalId,
) -> AgentDefinitionId:
    lifecycle = WorkshopAgentLifecycleService(store)
    draft = await lifecycle.create_draft(
        principal_id,
        idempotency_key="specialist-create",
        handle="specialist",
        display_name="Specialist",
        description="An isolated specialist.",
        presentation={"avatar": "S"},
        purpose="Qualify reusable agents.",
        instructions="Work only in the current canonical conversation.",
        capabilities=["text_generation"],
    )
    active = await lifecycle.activate_revision(
        principal_id,
        draft.definition_id,
        revision_id=draft.revisions[0].revision_id,
        idempotency_key="specialist-activate",
        expected_version=draft.state_version,
    )
    return active.definition_id


async def _service(path: Path):
    store = await WorkshopEventStore.open(path)
    profiles = profile_registry(101, 202)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman("Daniel", "admin", "telegram", "101", "101", profile_id(101)),
            BootstrapHuman("Scott", "member", "telegram", "202", "202", profile_id(202)),
        ),
    )
    execution = await WorkshopExecutionStateRegistry.from_store(store, profiles)
    contexts = await WorkshopInternalAPIContextRegistry.from_store(store, profiles)
    runtime_pool = _RuntimePool()
    return (
        store,
        WorkshopAgentEnablementService(
            store,
            profiles,
            execution,
            contexts,
            runtime_pool,  # type: ignore[arg-type]
        ),
        execution,
        contexts,
        runtime_pool,
    )


async def test_two_principals_share_one_owner_runtime_in_isolated_direct_lanes(tmp_path: Path) -> None:
    store, service, execution, contexts, runtime_pool = await _service(tmp_path / "kai.db")
    try:
        daniel = await _principal(store, "101")
        scott = await _principal(store, "202")
        definition_id = await _active_specialist(store, daniel)

        daniel_enabled = await service.enable(
            daniel,
            definition_id,
            profile_id(101),
            idempotency_key="daniel-enable",
        )
        scott_enabled = await service.enable(
            scott,
            definition_id,
            profile_id(202),
            idempotency_key="scott-enable",
        )

        assert daniel_enabled.lifecycle_state == scott_enabled.lifecycle_state == "enabled"
        assert daniel_enabled.agent_id == scott_enabled.agent_id
        assert daniel_enabled.direct_channel_id != scott_enabled.direct_channel_id
        assert daniel_enabled.runtime_profile_id == profile_id(101)
        assert scott_enabled.runtime_profile_id == profile_id(202)
        assert len(runtime_pool.registered) == 2
        assert execution.maybe_for_principal_channel(daniel, daniel_enabled.direct_channel_id) is not None
        assert execution.maybe_for_principal_channel(scott, scott_enabled.direct_channel_id) is not None
        assert len(contexts.contexts) == 4
        async with store.connection.execute(
            "SELECT direct_channel_id, COUNT(*) FROM principal_agent_enablements "
            "WHERE agent_definition_id = ? GROUP BY direct_channel_id ORDER BY direct_channel_id",
            (definition_id,),
        ) as cursor:
            assert [int(row[1]) for row in await cursor.fetchall()] == [1, 1]

        resolutions = []
        for subject in ("101", "202"):
            recorded = await record_inbound_message(
                store,
                InboundMessage(
                    transport="telegram",
                    update_id=f"shared-owner-{subject}",
                    message_id=f"shared-owner-message-{subject}",
                    sender_subject=subject,
                    channel_subject=subject,
                    body="Use the single owner runtime.",
                    occurred_at=datetime.now(UTC),
                ),
            )
            message_id = recorded.event.envelope.aggregate_id
            assert isinstance(message_id, MessageId)
            resolutions.append(await resolve_canonical_conversation_run(store, message_id))
        assert {resolution.target.requested_by_principal_id for resolution in resolutions} == {
            daniel,
            scott,
        }
        assert {resolution.sponsor_principal_id for resolution in resolutions} == {daniel}
        assert {resolution.runtime_profile_id for resolution in resolutions} == {profile_id(101)}
    finally:
        await store.close()


async def test_runtime_authority_cannot_cross_principals(tmp_path: Path) -> None:
    store, service, _execution, _contexts, _runtime_pool = await _service(tmp_path / "kai.db")
    try:
        daniel = await _principal(store, "101")
        definition_id = await _active_specialist(store, daniel)
        with pytest.raises(WorkshopAgentEnablementAccessDenied, match="not owned"):
            await service.enable(
                daniel,
                definition_id,
                profile_id(202),
                idempotency_key="cross-principal-runtime",
            )
        async with store.connection.execute(
            "SELECT COUNT(*) FROM principal_agent_enablements WHERE agent_definition_id = ?",
            (definition_id,),
        ) as cursor:
            assert int((await cursor.fetchone())[0]) == 0
    finally:
        await store.close()


async def test_direct_conversation_requires_an_explicit_durable_start(tmp_path: Path) -> None:
    store, service, _execution, _contexts, _runtime_pool = await _service(tmp_path / "kai.db")
    try:
        daniel = await _principal(store, "101")
        definition_id = await _active_specialist(store, daniel)
        enabled = await service.enable(
            daniel,
            definition_id,
            profile_id(101),
            idempotency_key="explicit-conversation-enable",
        )

        assert enabled.conversation_started is False
        started = await service.start_conversation(
            daniel,
            definition_id,
            idempotency_key="explicit-conversation-start",
            expected_version=enabled.state_version,
        )
        replayed = await service.start_conversation(
            daniel,
            definition_id,
            idempotency_key="explicit-conversation-start",
            expected_version=enabled.state_version,
        )

        assert started.conversation_started is True
        assert replayed == started
        assert started.state_version is not None
        assert enabled.state_version is not None
        assert started.state_version > enabled.state_version
        async with store.connection.execute(
            "SELECT conversation_started_at, conversation_started_event_position "
            "FROM principal_agent_enablements WHERE agent_definition_id = ? AND principal_id = ?",
            (definition_id, daniel),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert row[0] is not None
        assert int(row[1]) == started.state_version
    finally:
        await store.close()


async def test_archival_retires_runtime_lane_but_preserves_direct_channel(tmp_path: Path) -> None:
    store, service, execution, contexts, runtime_pool = await _service(tmp_path / "kai.db")
    try:
        daniel = await _principal(store, "101")
        definition_id = await _active_specialist(store, daniel)
        enabled = await service.enable(
            daniel,
            definition_id,
            profile_id(101),
            idempotency_key="archive-runtime-enable",
        )
        assert enabled.direct_channel_id is not None
        current = await WorkshopAgentLifecycleService(store).get_visible(daniel, definition_id)

        archived = await WorkshopAgentLifecycleService(store).archive(
            daniel,
            definition_id,
            idempotency_key="archive-runtime-definition",
            expected_version=current.state_version,
        )
        await service.suspend_archived_definition(daniel, definition_id)

        assert archived.lifecycle_state == "archived"
        assert runtime_pool.suspended[-1].channel_id == enabled.direct_channel_id
        assert execution.maybe_for_principal_channel(daniel, enabled.direct_channel_id) is None
        assert all(context.channel_id != enabled.direct_channel_id for context in contexts.contexts)
        async with store.connection.execute(
            "SELECT lifecycle_state FROM principal_agent_enablements "
            "WHERE agent_definition_id = ? AND principal_id = ?",
            (definition_id, daniel),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None and str(row[0]) == "disabled"
        async with store.connection.execute(
            "SELECT COUNT(*) FROM channels WHERE id = ?",
            (enabled.direct_channel_id,),
        ) as cursor:
            assert int((await cursor.fetchone())[0]) == 1

        restarted_execution = await WorkshopExecutionStateRegistry.from_store(
            store,
            profile_registry(101, 202),
        )
        restarted_contexts = await WorkshopInternalAPIContextRegistry.from_store(
            store,
            profile_registry(101, 202),
        )
        assert restarted_execution.maybe_for_principal_channel(daniel, enabled.direct_channel_id) is None
        assert all(context.channel_id != enabled.direct_channel_id for context in restarted_contexts.contexts)
        authority_status = workshop_agent_authority_status(tmp_path / "kai.db")
        assert authority_status.startswith("Workshop agent authority: active;")
        assert "owner runtimes=0" in authority_status.split("integrity gaps=", 1)[1]
    finally:
        await store.close()


async def test_version_fifty_seven_archived_enablement_is_retired_on_upgrade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from kai.workshop import schema

    path = tmp_path / "kai.db"
    profiles = profile_registry(101, 202)
    with monkeypatch.context() as migration_context:
        migration_context.setattr(schema, "WORKSHOP_SCHEMA_VERSION", 57)
        migration_context.setattr(schema, "_MIGRATIONS", schema._MIGRATIONS[:57])
        legacy = await WorkshopEventStore.open(path)
        await bootstrap_default_workshop(
            legacy,
            (
                BootstrapHuman("Daniel", "admin", "telegram", "101", "101", profile_id(101)),
                BootstrapHuman("Scott", "member", "telegram", "202", "202", profile_id(202)),
            ),
        )
        execution = await WorkshopExecutionStateRegistry.from_store(legacy, profiles)
        contexts = await WorkshopInternalAPIContextRegistry.from_store(legacy, profiles)
        service = WorkshopAgentEnablementService(
            legacy,
            profiles,
            execution,
            contexts,
            _RuntimePool(),  # type: ignore[arg-type]
        )
        daniel = await _principal(legacy, "101")
        definition_id = await _active_specialist(legacy, daniel)
        workshop_id = await service._workshop_for(daniel)
        definition = await service._active_definition(workshop_id, definition_id)
        enablement_id = AgentEnablementId.derived(definition_id, f"principal:{daniel}")
        channel_id = ChannelId.derived(enablement_id, "direct-channel")
        for event in service._creation_events(
            workshop_id,
            daniel,
            definition_id,
            definition,
            enablement_id,
            channel_id,
            profile_id(101),
            datetime.now(UTC),
            "legacy-archived-enable",
            {"source": "test"},
        ):
            await legacy.append(event)
        await legacy.project_pending(CanonicalConversationProjection())
        await legacy.connection.execute(
            "UPDATE agent_definitions SET lifecycle_state = 'archived' WHERE id = ?",
            (definition_id,),
        )
        await legacy.connection.commit()
        await legacy.close()

    upgraded = await WorkshopEventStore.open(path)
    try:
        assert await upgraded.schema_version() == 58
        async with upgraded.connection.execute(
            "SELECT lifecycle_state, conversation_started_at "
            "FROM principal_agent_enablements WHERE agent_definition_id = ?",
            (definition_id,),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert tuple(row) == ("disabled", None)
    finally:
        await upgraded.close()


async def test_reconciliation_allows_archived_agent_without_runtime(tmp_path: Path) -> None:
    store, _service_instance, _execution, _contexts, _runtime_pool = await _service(tmp_path / "kai.db")
    try:
        daniel = await _principal(store, "101")
        lifecycle = WorkshopAgentLifecycleService(store)
        draft = await lifecycle.create_draft(
            daniel,
            idempotency_key="archived-without-runtime-create",
            handle="archived_specialist",
            display_name="Archived specialist",
            description="An archived agent with no runtime access.",
            presentation={"avatar": "A"},
            purpose="Preserve archived provenance.",
            instructions="Do not run after archival.",
            capabilities=["text_generation"],
        )
        archived = await lifecycle.archive(
            daniel,
            draft.definition_id,
            idempotency_key="archived-without-runtime-archive",
            expected_version=draft.state_version,
        )

        result = await reconcile_single_owner_agent_authority(
            store,
            profile_registry(101, 202),
        )

        assert archived.lifecycle_state == "archived"
        assert result.definitions == 2
        async with store.connection.execute(
            "SELECT owner_principal_id, owner_runtime_profile_id, owner_direct_channel_id "
            "FROM agent_definitions WHERE id = ?",
            (draft.definition_id,),
        ) as cursor:
            authority = await cursor.fetchone()
        assert authority is not None
        assert tuple(authority) == (str(daniel), None, None)
    finally:
        await store.close()


async def test_reconciliation_rejects_active_agent_without_runtime(tmp_path: Path) -> None:
    store, _service_instance, _execution, _contexts, _runtime_pool = await _service(tmp_path / "kai.db")
    try:
        daniel = await _principal(store, "101")
        definition_id = await _active_specialist(store, daniel)

        with pytest.raises(WorkshopAgentAuthorityError, match="has no enabled runtime"):
            await reconcile_single_owner_agent_authority(
                store,
                profile_registry(101, 202),
            )

        async with store.connection.execute(
            "SELECT owner_runtime_profile_id FROM agent_definitions WHERE id = ?",
            (definition_id,),
        ) as cursor:
            authority = await cursor.fetchone()
        assert authority is not None
        assert authority[0] is None
    finally:
        await store.close()


async def test_nonowner_cannot_change_an_agent_definition(tmp_path: Path) -> None:
    store, _service_instance, _execution, _contexts, _runtime_pool = await _service(tmp_path / "kai.db")
    try:
        daniel = await _principal(store, "101")
        scott = await _principal(store, "202")
        definition_id = await _active_specialist(store, daniel)
        current = await WorkshopAgentLifecycleService(store).get_visible(scott, definition_id)

        with pytest.raises(WorkshopAgentLifecycleAccessDenied, match="Only the agent owner"):
            await WorkshopAgentLifecycleService(store).add_revision(
                scott,
                definition_id,
                idempotency_key="nonowner-revision",
                expected_version=current.state_version,
                purpose="Attempted takeover",
                instructions="Do not accept this revision.",
                capabilities=["text_generation"],
            )
    finally:
        await store.close()


async def test_diagnostics_detect_cross_principal_runtime_binding(tmp_path: Path) -> None:
    store, _service_instance, _execution, _contexts, _runtime_pool = await _service(tmp_path / "kai.db")
    try:
        daniel = await _principal(store, "101")
        scott = await _principal(store, "202")
        clean = workshop_agent_authority_status(tmp_path / "kai.db")
        assert clean.startswith("Workshop agent authority: active;")
        assert "integrity gaps=0" in clean

        await store.connection.execute(
            "UPDATE principal_agent_enablements SET runtime_profile_id = ? WHERE principal_id = ?",
            (profile_id(101), scott),
        )
        await store.connection.commit()

        status = workshop_agent_authority_status(tmp_path / "kai.db")
        assert status.startswith("Workshop agent authority: INCOMPLETE;")
        assert "enablements=1" in status.split("integrity gaps=", 1)[1]
        assert "runtime bindings=1" in status
        assert str(daniel) not in status
        assert str(scott) not in status
    finally:
        await store.close()


async def test_disable_and_reenable_preserve_direct_channel(tmp_path: Path) -> None:
    store, service, _execution, _contexts, runtime_pool = await _service(tmp_path / "kai.db")
    try:
        daniel = await _principal(store, "101")
        scott = await _principal(store, "202")
        definition_id = await _active_specialist(store, daniel)
        await service.enable(
            daniel,
            definition_id,
            profile_id(101),
            idempotency_key="owner-enable",
        )
        enabled = await service.enable(
            scott,
            definition_id,
            profile_id(202),
            idempotency_key="access-enable-once",
        )
        disabled = await service.disable(
            scott,
            definition_id,
            idempotency_key="access-disable-once",
            expected_version=enabled.state_version,
        )
        restored = await service.enable(
            scott,
            definition_id,
            profile_id(202),
            idempotency_key="access-enable-again",
            expected_version=disabled.state_version,
        )

        assert disabled.lifecycle_state == "disabled"
        assert restored.lifecycle_state == "enabled"
        assert restored.direct_channel_id == enabled.direct_channel_id
        assert runtime_pool.suspended[0].channel_id == enabled.direct_channel_id
        async with store.connection.execute(
            "SELECT COUNT(*) FROM channels WHERE id = ?",
            (enabled.direct_channel_id,),
        ) as cursor:
            assert int((await cursor.fetchone())[0]) == 1
    finally:
        await store.close()


async def test_agent_event_doorbells_are_scoped_to_the_enabled_principal(
    tmp_path: Path,
) -> None:
    store, service, _execution, _contexts, _runtime_pool = await _service(tmp_path / "kai.db")
    try:
        daniel = await _principal(store, "101")
        scott = await _principal(store, "202")
        definition_id = await _active_specialist(store, daniel)
        async with store.connection.execute(
            "SELECT workshop_id FROM workshop_memberships WHERE principal_id = ?",
            (daniel,),
        ) as cursor:
            workshop_row = await cursor.fetchone()
        assert workshop_row is not None
        workshop_id = str(workshop_row[0])
        async with store.connection.execute("SELECT COALESCE(MAX(position), 0) FROM event_log") as cursor:
            before = int((await cursor.fetchone())[0])

        await service.enable(
            daniel,
            definition_id,
            profile_id(101),
            idempotency_key="daniel-doorbell",
        )
        await service.enable(
            scott,
            definition_id,
            profile_id(202),
            idempotency_key="scott-doorbell",
        )

        daniel_events, daniel_position = await _read_agent_lifecycle_events(
            store,
            workshop_id=workshop_id,
            principal_id=daniel,
            role="admin",
            after_position=before,
        )
        scott_events, scott_position = await _read_agent_lifecycle_events(
            store,
            workshop_id=workshop_id,
            principal_id=scott,
            role="member",
            after_position=before,
        )

        assert len(daniel_events) == len(scott_events) == 1
        assert b"event: agent.enablement.changed" in daniel_events[0]
        assert b"event: agent.enablement.changed" in scott_events[0]
        assert str(definition_id).encode() in daniel_events[0]
        assert str(definition_id).encode() in scott_events[0]
        assert daniel_position == scott_position
    finally:
        await store.close()
