"""Principal-agent enablement, isolation, and continuity contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from kai.workshop.agent_enablement import (
    WorkshopAgentEnablementAccessDenied,
    WorkshopAgentEnablementService,
)
from kai.workshop.agent_lifecycle import WorkshopAgentLifecycleService
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.domain import AgentDefinitionId, PrincipalId
from kai.workshop.execution_state import WorkshopExecutionStateRegistry
from kai.workshop.internal_api_contexts import (
    WorkshopInternalAPIContextRegistry,
    WorkshopInternalAPIExecutionContext,
)
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


async def test_two_principals_enable_same_definition_in_isolated_direct_lanes(tmp_path: Path) -> None:
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


async def test_disable_and_reenable_preserve_direct_channel(tmp_path: Path) -> None:
    store, service, _execution, _contexts, runtime_pool = await _service(tmp_path / "kai.db")
    try:
        daniel = await _principal(store, "101")
        definition_id = await _active_specialist(store, daniel)
        enabled = await service.enable(
            daniel,
            definition_id,
            profile_id(101),
            idempotency_key="enable-once",
        )
        disabled = await service.disable(
            daniel,
            definition_id,
            idempotency_key="disable-once",
            expected_version=enabled.state_version,
        )
        restored = await service.enable(
            daniel,
            definition_id,
            profile_id(101),
            idempotency_key="enable-again",
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
