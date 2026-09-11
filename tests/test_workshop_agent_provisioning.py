"""Replay-safe principal-owned agent provisioning contracts."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from kai.workshop.agent_creation_options import (
    AgentCreationBackendOption,
    AgentCreationModelOption,
    AgentCreationOptions,
    AgentCreationRuntimeOption,
    AgentCreationWorkspaceOption,
)
from kai.workshop.agent_enablement import WorkshopAgentEnablementService
from kai.workshop.agent_lifecycle import WorkshopAgentLifecycleService
from kai.workshop.agent_provisioning import (
    WorkshopAgentProvisioningAccessDenied,
    WorkshopAgentProvisioningConflict,
    WorkshopAgentProvisioningService,
)
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.diagnostics import workshop_agent_provisioning_status
from kai.workshop.domain import PrincipalId
from kai.workshop.execution_state import WorkshopExecutionStateRegistry
from kai.workshop.internal_api_contexts import (
    WorkshopInternalAPIContextRegistry,
    WorkshopInternalAPIExecutionContext,
)
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id, profile_registry


class _Crash(BaseException):
    pass


@dataclass
class _RuntimePool:
    registered: list[WorkshopInternalAPIExecutionContext] = field(default_factory=list)

    def register_canonical_lane(self, context: WorkshopInternalAPIExecutionContext) -> None:
        self.registered.append(context)


@dataclass
class _CreationOptions:
    principal_id: PrincipalId
    home: Path

    async def inspect(self, principal_id: PrincipalId) -> AgentCreationOptions:
        if principal_id != self.principal_id:
            return AgentCreationOptions(principal_id, False, (), ())
        backend = AgentCreationBackendOption(
            option_id="codex:openai",
            backend="codex",
            provider="openai",
            current=True,
            readiness="ready",
            default_model="gpt-5.6-sol",
            models=(
                AgentCreationModelOption(
                    "gpt-5.6-sol",
                    "GPT-5.6 Sol",
                    "available",
                    True,
                    False,
                ),
            ),
            catalogue_status="succeeded",
            catalogue_stale=False,
            blockers=(),
        )
        runtime = AgentCreationRuntimeOption(
            runtime_profile_id=profile_id(101),
            display_name="Daniel",
            current_backend_option_id=backend.option_id,
            default_workspace=str(self.home),
            default_timeout_seconds=300,
            minimum_timeout_seconds=1,
            maximum_timeout_seconds=1800,
            backends=(backend,),
            workspaces=(AgentCreationWorkspaceOption(str(self.home), "Home", True, True),),
            ready=True,
            blockers=(),
        )
        return AgentCreationOptions(principal_id, True, (runtime,), ())


@dataclass
class _Settings:
    values: dict[str, object] = field(default_factory=dict)
    fail_model_once: bool = False

    def authority_for_principal_channel(self, principal_id, channel_id):
        return principal_id, channel_id

    async def set_backend(self, _authority, value):
        self.values["backend"] = value

    async def set_model(self, _authority, value):
        if self.fail_model_once:
            self.fail_model_once = False
            raise RuntimeError("transient model failure")
        self.values["model"] = value

    async def switch_workspace(self, _authority, value):
        self.values["workspace"] = value

    async def set_timeout(self, _authority, value):
        self.values["timeout"] = value


@dataclass
class _CollaborationPolicy:
    values: tuple[str, ...] = ()

    def validate_initial_allowed_operations(self, requested, allowed):
        assert isinstance(requested, list)
        assert isinstance(allowed, list)
        assert set(allowed).issubset(requested)
        return tuple(allowed)

    async def set_allowed(
        self,
        _principal_id,
        _definition_id,
        *,
        allowed_operations,
        expected_policy_version,
        client_operation_id,
    ):
        assert expected_policy_version == 0
        assert client_operation_id.startswith("provision:apv_")
        self.values = tuple(allowed_operations)


class _InterruptingProvisioningService(WorkshopAgentProvisioningService):
    def __init__(self, *args, interrupt_after: str | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._interrupt_after = interrupt_after

    async def _after_stage(self, stage: str) -> None:
        if stage == self._interrupt_after:
            self._interrupt_after = None
            raise _Crash(stage)


async def _principal(store: WorkshopEventStore, subject: str = "101") -> PrincipalId:
    async with store.connection.execute(
        "SELECT principal_id FROM external_identities WHERE provider = 'telegram' AND external_subject = ?",
        (subject,),
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return PrincipalId(str(row[0]))


async def _services(tmp_path: Path):
    store = await WorkshopEventStore.open(tmp_path / "kai.db")
    profiles = profile_registry(101)
    await bootstrap_default_workshop(
        store,
        (BootstrapHuman("Daniel", "admin", "telegram", "101", "101", profile_id(101)),),
    )
    principal_id = await _principal(store)
    execution = await WorkshopExecutionStateRegistry.from_store(store, profiles)
    contexts = await WorkshopInternalAPIContextRegistry.from_store(store, profiles)
    runtime_pool = _RuntimePool()
    enablement = WorkshopAgentEnablementService(
        store,
        profiles,
        execution,
        contexts,
        runtime_pool,  # type: ignore[arg-type]
    )
    settings = _Settings()
    collaboration = _CollaborationPolicy()
    dependencies = (
        store,
        _CreationOptions(principal_id, tmp_path.resolve()),
        WorkshopAgentLifecycleService(store),
        enablement,
        settings,
        collaboration,
    )
    return store, principal_id, dependencies, runtime_pool, settings


def _request() -> dict[str, object]:
    return {
        "client_operation_id": "q1488-provision-1",
        "definition": {
            "handle": "builder",
            "display_name": "Builder",
            "description": "A bounded builder.",
            "presentation": {"avatar": "B"},
        },
        "revision": {
            "purpose": "Build carefully.",
            "instructions": "Use the authorized workspace.",
            "capabilities": ["text_generation"],
            "collaboration_operations": [],
        },
        "runtime": {
            "runtime_profile_id": str(profile_id(101)),
            "backend_option_id": "codex:openai",
            "model": "gpt-5.6-sol",
            "workspace": str(Path.cwd()),
            "timeout_seconds": 420,
        },
        "collaboration_policy": {"allowed_operations": []},
    }


@pytest.mark.parametrize(
    "stage",
    (
        "definition_created",
        "revision_activated",
        "enablement_created",
        "runtime_registered",
        "backend_selected",
        "model_selected",
        "workspace_selected",
        "timeout_selected",
        "collaboration_policy_set",
        "ready",
    ),
)
async def test_interruption_after_every_durable_stage_resumes_exact_lane(
    tmp_path: Path,
    stage: str,
) -> None:
    store, principal_id, dependencies, _runtime_pool, settings = await _services(tmp_path)
    request = _request()
    request["runtime"] = {**request["runtime"], "workspace": str(tmp_path.resolve())}  # type: ignore[dict-item]
    try:
        async with store.connection.execute(
            "SELECT COUNT(*) FROM event_log WHERE event_type = 'principal_agent.conversation_started'",
        ) as cursor:
            initial_conversation_starts = int((await cursor.fetchone())[0])
        interrupted = _InterruptingProvisioningService(
            *dependencies,
            interrupt_after=stage,
        )
        with pytest.raises(_Crash):
            await interrupted.provision(principal_id, **request)

        restarted = WorkshopAgentProvisioningService(*dependencies)  # type: ignore[arg-type]
        result = await restarted.provision(principal_id, **request)
        replay = await restarted.provision(principal_id, **request)

        assert result.status == "ready"
        assert result.completed_stages[-1] == "ready"
        assert replay.replayed is True
        assert replay.definition_id == result.definition_id
        assert settings.values == {
            "backend": "codex:openai",
            "model": "gpt-5.6-sol",
            "workspace": str(tmp_path.resolve()),
            "timeout": 420,
        }
        async with store.connection.execute(
            "SELECT (SELECT COUNT(*) FROM agent_definitions), "
            "(SELECT COUNT(*) FROM agent_definition_revisions WHERE agent_definition_id = ?), "
            "(SELECT COUNT(*) FROM principal_agent_enablements WHERE agent_definition_id = ?), "
            "(SELECT COUNT(*) FROM agent_provisioning_stage_receipts)",
            (result.definition_id, result.definition_id),
        ) as cursor:
            counts = await cursor.fetchone()
        assert counts is not None
        # Bootstrap contributes only the predefined Kai definition.
        assert tuple(int(value) for value in counts) == (2, 1, 1, 10)
        async with store.connection.execute(
            "SELECT COUNT(*) FROM event_log WHERE event_type = 'principal_agent.conversation_started'",
        ) as cursor:
            assert int((await cursor.fetchone())[0]) == initial_conversation_starts
        assert "Workshop agent provisioning: active; operations=1 (ready=1" in (
            workshop_agent_provisioning_status(tmp_path / "kai.db")
        )
    finally:
        await store.close()


async def test_exact_concurrency_replays_and_conflicting_identity_fails(tmp_path: Path) -> None:
    store, principal_id, dependencies, _runtime_pool, _settings = await _services(tmp_path)
    request = _request()
    request["runtime"] = {**request["runtime"], "workspace": str(tmp_path.resolve())}  # type: ignore[dict-item]
    service = WorkshopAgentProvisioningService(*dependencies)  # type: ignore[arg-type]
    try:
        first, second = await asyncio.gather(
            service.provision(principal_id, **request),
            service.provision(principal_id, **request),
        )
        assert {first.replayed, second.replayed} == {False, True}
        assert first.definition_id == second.definition_id

        changed = dict(request)
        changed["definition"] = {**request["definition"], "display_name": "Other"}  # type: ignore[dict-item]
        with pytest.raises(WorkshopAgentProvisioningConflict, match="different"):
            await service.provision(principal_id, **changed)
    finally:
        await store.close()


async def test_conflicting_concurrent_inputs_create_only_one_agent(tmp_path: Path) -> None:
    store, principal_id, dependencies, _runtime_pool, _settings = await _services(tmp_path)
    first = _request()
    first["runtime"] = {**first["runtime"], "workspace": str(tmp_path.resolve())}  # type: ignore[dict-item]
    second = dict(first)
    second["definition"] = {**first["definition"], "display_name": "Conflicting"}  # type: ignore[dict-item]
    service = WorkshopAgentProvisioningService(*dependencies)  # type: ignore[arg-type]
    try:
        outcomes = await asyncio.gather(
            service.provision(principal_id, **first),
            service.provision(principal_id, **second),
            return_exceptions=True,
        )
        assert sum(not isinstance(outcome, BaseException) for outcome in outcomes) == 1
        assert sum(isinstance(outcome, WorkshopAgentProvisioningConflict) for outcome in outcomes) == 1
        async with store.connection.execute(
            "SELECT COUNT(*) FROM agent_definitions WHERE handle = 'builder'",
        ) as cursor:
            assert int((await cursor.fetchone())[0]) == 1
    finally:
        await store.close()


async def test_committed_enablement_recovers_registration_after_store_reopen(
    tmp_path: Path,
) -> None:
    store, principal_id, dependencies, _runtime_pool, settings = await _services(tmp_path)
    request = _request()
    request["runtime"] = {**request["runtime"], "workspace": str(tmp_path.resolve())}  # type: ignore[dict-item]
    collaboration = dependencies[-1]
    interrupted = _InterruptingProvisioningService(
        *dependencies,
        interrupt_after="enablement_created",
    )
    with pytest.raises(_Crash):
        await interrupted.provision(principal_id, **request)
    await store.close()

    reopened = await WorkshopEventStore.open(tmp_path / "kai.db")
    profiles = profile_registry(101)
    execution = await WorkshopExecutionStateRegistry.from_store(reopened, profiles)
    contexts = await WorkshopInternalAPIContextRegistry.from_store(reopened, profiles)
    runtime_pool = _RuntimePool()
    enablement = WorkshopAgentEnablementService(
        reopened,
        profiles,
        execution,
        contexts,
        runtime_pool,  # type: ignore[arg-type]
    )
    restarted = WorkshopAgentProvisioningService(
        reopened,
        _CreationOptions(principal_id, tmp_path.resolve()),  # type: ignore[arg-type]
        WorkshopAgentLifecycleService(reopened),
        enablement,
        settings,  # type: ignore[arg-type]
        collaboration,  # type: ignore[arg-type]
    )
    try:
        result = await restarted.provision(principal_id, **request)
        assert result.status == "ready"
        assert len(runtime_pool.registered) == 1
        assert runtime_pool.registered[0].channel_id == result.direct_channel_id
    finally:
        await reopened.close()


async def test_late_failure_is_bounded_and_resumable(tmp_path: Path) -> None:
    store, principal_id, dependencies, _runtime_pool, settings = await _services(tmp_path)
    request = _request()
    request["runtime"] = {**request["runtime"], "workspace": str(tmp_path.resolve())}  # type: ignore[dict-item]
    settings.fail_model_once = True
    service = WorkshopAgentProvisioningService(*dependencies)  # type: ignore[arg-type]
    try:
        blocked = await service.provision(principal_id, **request)
        assert blocked.status == "needs_attention"
        assert blocked.next_stage == "model_selected"
        assert [item.code for item in blocked.blockers] == ["stage_failed"]

        completed = await service.provision(principal_id, **request)
        assert completed.status == "ready"
    finally:
        await store.close()


async def test_incomplete_setup_is_principal_scoped_and_exposes_saved_input(
    tmp_path: Path,
) -> None:
    store, principal_id, dependencies, _runtime_pool, settings = await _services(tmp_path)
    request = _request()
    request["runtime"] = {**request["runtime"], "workspace": str(tmp_path.resolve())}  # type: ignore[dict-item]
    settings.fail_model_once = True
    service = WorkshopAgentProvisioningService(*dependencies)  # type: ignore[arg-type]
    try:
        blocked = await service.provision(principal_id, **request)
        setups = await service.list_incomplete(principal_id)

        assert len(setups) == 1
        assert setups[0].provisioning.operation_id == blocked.operation_id
        assert setups[0].provisioning.status == "needs_attention"
        assert setups[0].request == request

        completed = await service.provision(principal_id, **request)
        assert completed.status == "ready"
        assert await service.list_incomplete(principal_id) == ()
    finally:
        await store.close()


async def test_revision_one_draft_continues_without_duplicate_definition(
    tmp_path: Path,
) -> None:
    store, principal_id, dependencies, _runtime_pool, settings = await _services(tmp_path)
    lifecycle = dependencies[2]
    draft = await lifecycle.create_draft(
        principal_id,
        idempotency_key="q1490-intentional-draft",
        handle="builder",
        display_name="Builder",
        description="A bounded builder.",
        presentation={"avatar": "B"},
        purpose="Build carefully.",
        instructions="Use the authorized workspace.",
        capabilities=["text_generation"],
        collaboration_operations=[],
    )
    request = _request()
    request["definition"] = {
        **request["definition"],  # type: ignore[dict-item]
        "existing_definition_id": str(draft.definition_id),
    }
    request["runtime"] = {**request["runtime"], "workspace": str(tmp_path.resolve())}  # type: ignore[dict-item]
    service = WorkshopAgentProvisioningService(*dependencies)  # type: ignore[arg-type]
    try:
        async with store.connection.execute(
            "SELECT COUNT(*) FROM event_log WHERE event_type = 'principal_agent.conversation_started'",
        ) as cursor:
            initial_conversation_starts = int((await cursor.fetchone())[0])
        settings.fail_model_once = True
        blocked = await service.provision(principal_id, **request)
        assert blocked.status == "needs_attention"
        assert blocked.definition_id == draft.definition_id

        completed = await WorkshopAgentProvisioningService(*dependencies).provision(  # type: ignore[arg-type]
            principal_id,
            **request,
        )

        assert completed.status == "ready"
        assert completed.definition_id == draft.definition_id
        async with store.connection.execute(
            "SELECT COUNT(*) FROM agent_definitions WHERE handle = 'builder'",
        ) as cursor:
            assert int((await cursor.fetchone())[0]) == 1
        async with store.connection.execute(
            "SELECT COUNT(*) FROM event_log WHERE event_type = 'principal_agent.conversation_started'",
        ) as cursor:
            assert int((await cursor.fetchone())[0]) == initial_conversation_starts
    finally:
        await store.close()


async def test_another_principal_cannot_continue_an_owned_draft(tmp_path: Path) -> None:
    store, principal_id, dependencies, _runtime_pool, _settings = await _services(tmp_path)
    lifecycle = dependencies[2]
    draft = await lifecycle.create_draft(
        principal_id,
        idempotency_key="q1490-private-draft",
        handle="builder",
        display_name="Builder",
        description="A bounded builder.",
        presentation={"avatar": "B"},
        purpose="Build carefully.",
        instructions="Use the authorized workspace.",
        capabilities=["text_generation"],
        collaboration_operations=[],
    )
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman("Daniel", "admin", "telegram", "101", "101", profile_id(101)),
            BootstrapHuman("Scott", "member", "telegram", "202", "202", profile_id(202)),
        ),
    )
    other_principal_id = await _principal(store, "202")
    request = _request()
    request["definition"] = {
        **request["definition"],  # type: ignore[dict-item]
        "existing_definition_id": str(draft.definition_id),
    }
    request["runtime"] = {**request["runtime"], "workspace": str(tmp_path.resolve())}  # type: ignore[dict-item]
    service = WorkshopAgentProvisioningService(*dependencies)  # type: ignore[arg-type]
    try:
        with pytest.raises(WorkshopAgentProvisioningAccessDenied, match="Access denied"):
            await service.provision(other_principal_id, **request)
        async with store.connection.execute(
            "SELECT COUNT(*) FROM agent_provisioning_operations",
        ) as cursor:
            assert int((await cursor.fetchone())[0]) == 0
    finally:
        await store.close()


async def test_unauthorized_runtime_fails_before_agent_state(tmp_path: Path) -> None:
    store, principal_id, dependencies, _runtime_pool, _settings = await _services(tmp_path)
    request = _request()
    runtime = dict(request["runtime"])  # type: ignore[arg-type]
    runtime["runtime_profile_id"] = str(profile_id(202))
    request["runtime"] = runtime
    service = WorkshopAgentProvisioningService(*dependencies)  # type: ignore[arg-type]
    try:
        with pytest.raises(WorkshopAgentProvisioningAccessDenied):
            await service.provision(principal_id, **request)
        async with store.connection.execute(
            "SELECT (SELECT COUNT(*) FROM agent_provisioning_operations), (SELECT COUNT(*) FROM agent_definitions)",
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        assert (int(row[0]), int(row[1])) == (0, 1)
    finally:
        await store.close()
