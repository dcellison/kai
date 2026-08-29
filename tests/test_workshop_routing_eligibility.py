from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from kai.claude import ClaudeCodeBackend
from kai.codex import CodexBackend
from kai.goose import GooseBackend
from kai.opencode import OpenCodeBackend
from kai.pi import PiBackend
from kai.workshop.domain import AgentId, ChannelId, PrincipalId, RuntimeProfileId
from kai.workshop.execution_state import (
    WorkshopExecutionStateNamespace,
    WorkshopExecutionStateRegistry,
)
from kai.workshop.model_catalogue import (
    ModelCatalogueEntry,
    ModelCatalogueEntryStatus,
    ModelCatalogueProvenance,
    ModelCatalogueRefreshState,
    ModelCatalogueRefreshStatus,
    ModelCatalogueSnapshot,
)
from kai.workshop.model_discovery_inventory import ModelDiscoveryReadiness
from kai.workshop.routing_eligibility import (
    CapabilitySupport,
    RoutingEligibilityAccessDenied,
    RoutingEligibilityError,
    RoutingTaskClass,
    RuntimeCapability,
    WorkshopRoutingEligibilityService,
    capability_assessment_for_model,
)


def _id(identifier_type, value: int):
    return identifier_type(f"{identifier_type.prefix}_{value:032x}")


def test_all_registered_backend_adapters_publish_the_routing_contract() -> None:
    expected = frozenset({"text_generation", "tool_activity", "workspace_execution"})

    assert ClaudeCodeBackend.routing_capabilities == expected
    assert CodexBackend.routing_capabilities == expected
    assert GooseBackend.routing_capabilities == expected
    assert OpenCodeBackend.routing_capabilities == expected
    assert PiBackend.routing_capabilities == expected


def _namespace(value: int) -> WorkshopExecutionStateNamespace:
    return WorkshopExecutionStateNamespace(
        principal_id=_id(PrincipalId, value),
        channel_id=_id(ChannelId, value),
        agent_id=_id(AgentId, value),
        runtime_profile_id=_id(RuntimeProfileId, value),
        legacy_runtime_key=None,
    )


def _lane(
    backend: str,
    provider: str,
    *,
    selected: bool = False,
    readiness: ModelDiscoveryReadiness = ModelDiscoveryReadiness.READY,
):
    return SimpleNamespace(
        option_id=f"{backend}:{provider}",
        backend=backend,
        provider=provider,
        selected=selected,
        readiness=readiness,
        default_model=f"{backend}-default",
    )


def _entry(model_id: str, image_input: bool | None) -> ModelCatalogueEntry:
    if image_input is None:
        capabilities: dict[str, object] = {}
    else:
        capabilities = {"input_modalities": ["text", *(["image"] if image_input else [])]}
    return ModelCatalogueEntry(
        model_id=model_id,
        display_label=model_id,
        status=ModelCatalogueEntryStatus.AVAILABLE,
        selectable=True,
        retained=False,
        provenances=(
            ModelCatalogueProvenance(
                source="discovered:test",
                status=ModelCatalogueEntryStatus.AVAILABLE,
                display_label=model_id,
                capabilities=capabilities,
            ),
        ),
    )


def _snapshot(
    namespace: WorkshopExecutionStateNamespace,
    lane,
    *,
    image_input: bool | None,
    stale: bool = False,
) -> ModelCatalogueSnapshot:
    now = datetime(2026, 8, 29, tzinfo=UTC)
    return ModelCatalogueSnapshot(
        principal_id=namespace.principal_id,
        runtime_profile_id=namespace.runtime_profile_id,
        option_id=lane.option_id,
        cache_key=f"cache-{lane.option_id}",
        entries=(_entry(lane.default_model if not lane.selected else "selected-model", image_input),),
        refresh=ModelCatalogueRefreshState(
            status=ModelCatalogueRefreshStatus.SUCCEEDED,
            generation=1,
            last_attempt_at=now,
            last_successful_refresh_at=now,
            expires_at=now + timedelta(days=1),
            error_code=None,
            error_detail=None,
        ),
        stale=stale,
        last_known_good=False,
    )


class _Inventory:
    def __init__(self, namespace: WorkshopExecutionStateNamespace, lanes: tuple[object, ...]) -> None:
        self.profile = SimpleNamespace(
            principal_id=namespace.principal_id,
            channel_id=str(namespace.channel_id),
            agent_id=str(namespace.agent_id),
            runtime_profile_id=namespace.runtime_profile_id,
            backends=lanes,
        )

    def for_principal(self, principal_id):
        return (self.profile,) if principal_id == self.profile.principal_id else ()


class _Catalogue:
    def __init__(self, snapshots: dict[str, ModelCatalogueSnapshot]) -> None:
        self.snapshots = snapshots
        self.inspected: list[tuple[PrincipalId, RuntimeProfileId, str]] = []

    def authority_for_principal(self, principal_id):
        return SimpleNamespace(principal_id=principal_id)

    async def inspect(self, authority, runtime_profile_id, option_id):
        self.inspected.append((authority.principal_id, runtime_profile_id, option_id))
        return self.snapshots[option_id]


class _RuntimePool:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.selection_reads = 0

    async def get_effective_workspace(self, _runtime_profile_id):
        return self.workspace

    async def get_effective_model(self, _runtime_profile_id):
        self.selection_reads += 1
        return "selected-model"

    def runtime_profile(self, _runtime_profile_id):
        return SimpleNamespace(allowed_services=("perplexity",))


def _service(tmp_path: Path, lanes: tuple[object, ...], snapshots: dict[str, ModelCatalogueSnapshot]):
    namespace = _namespace(1)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    pool = _RuntimePool(workspace)
    catalogue = _Catalogue(snapshots)
    service = WorkshopRoutingEligibilityService(
        execution_state=WorkshopExecutionStateRegistry((namespace,)),
        inventory=_Inventory(namespace, lanes),  # type: ignore[arg-type]
        catalogue=catalogue,  # type: ignore[arg-type]
        runtime_pool=pool,  # type: ignore[arg-type]
    )
    authority = service.authority_for_principal_channel(
        namespace.principal_id,
        namespace.channel_id,
    )
    return service, authority, pool, catalogue, namespace


@pytest.mark.asyncio
async def test_report_explains_all_five_authorized_backends_without_switching(tmp_path: Path) -> None:
    namespace = _namespace(1)
    lanes = (
        _lane("claude", "anthropic", selected=True, readiness=ModelDiscoveryReadiness.UNVERIFIED),
        _lane("codex", "openai"),
        _lane("goose", "ollama"),
        _lane("opencode", "openrouter"),
        _lane("pi", "openai-codex"),
    )
    snapshots = {lane.option_id: _snapshot(namespace, lane, image_input=True) for lane in lanes}
    ungranted = _lane("opencode", "deepseek")
    snapshots[ungranted.option_id] = _snapshot(namespace, ungranted, image_input=True)
    service, authority, pool, catalogue, _ = _service(tmp_path, lanes, snapshots)

    report = await service.inspect(authority, RoutingTaskClass.CODING)

    assert report.version == 1
    assert report.required_capabilities == (
        RuntimeCapability.TEXT_GENERATION,
        RuntimeCapability.TOOL_ACTIVITY,
        RuntimeCapability.WORKSPACE_EXECUTION,
    )
    assert [item.option_id for item in report.candidates] == [lane.option_id for lane in lanes]
    assert all(item.eligible for item in report.candidates)
    assert all(item.allowed_services == ("perplexity",) for item in report.candidates)
    assert report.candidates[0].reasons[0].code == "runtime_unverified"
    assert report.candidates[0].reasons[-1].code == "eligible"
    assert pool.selection_reads == 1
    assert len(catalogue.inspected) == 5
    assert all(option_id != ungranted.option_id for _, _, option_id in catalogue.inspected)
    assert not hasattr(pool, "select_backend")


@pytest.mark.asyncio
async def test_vision_fails_closed_for_stale_unknown_and_unsupported_evidence(tmp_path: Path) -> None:
    namespace = _namespace(1)
    supported = _lane("claude", "anthropic", selected=True)
    unsupported = _lane("codex", "openai")
    stale = _lane("goose", "openrouter")
    unknown = _lane("pi", "openai")
    lanes = (supported, unsupported, stale, unknown)
    snapshots = {
        supported.option_id: _snapshot(namespace, supported, image_input=True),
        unsupported.option_id: _snapshot(namespace, unsupported, image_input=False),
        stale.option_id: _snapshot(namespace, stale, image_input=True, stale=True),
        unknown.option_id: _snapshot(namespace, unknown, image_input=None),
    }
    service, authority, *_ = _service(tmp_path, lanes, snapshots)

    report = await service.inspect(authority, "vision")

    by_option = {item.option_id: item for item in report.candidates}
    assert by_option[supported.option_id].eligible is True
    assert by_option[unsupported.option_id].eligible is False
    assert by_option[unsupported.option_id].reasons[-1].code == "capability_unsupported"
    assert by_option[stale.option_id].eligible is False
    assert by_option[stale.option_id].reasons[-1].code == "capability_unknown"
    assert by_option[unknown.option_id].eligible is False


@pytest.mark.asyncio
async def test_agent_definition_requirements_only_constrain_runtime_authority(tmp_path: Path) -> None:
    namespace = _namespace(1)
    supported = _lane("claude", "anthropic", selected=True)
    unsupported = _lane("codex", "openai")
    lanes = (supported, unsupported)
    service, authority, *_ = _service(
        tmp_path,
        lanes,
        {
            supported.option_id: _snapshot(namespace, supported, image_input=True),
            unsupported.option_id: _snapshot(namespace, unsupported, image_input=False),
        },
    )

    report = await service.inspect(
        authority,
        RoutingTaskClass.CONVERSATION,
        additional_required=(RuntimeCapability.IMAGE_INPUT,),
    )

    assert report.required_capabilities == (
        RuntimeCapability.TEXT_GENERATION,
        RuntimeCapability.IMAGE_INPUT,
    )
    assert report.candidates[0].eligible is True
    assert report.candidates[1].eligible is False
    assert report.candidates[1].reasons[-1].code == "capability_unsupported"


@pytest.mark.asyncio
async def test_unavailable_runtime_and_workspace_are_rejected(tmp_path: Path) -> None:
    namespace = _namespace(1)
    lane = _lane("claude", "anthropic", selected=True, readiness=ModelDiscoveryReadiness.UNAVAILABLE)
    snapshots = {lane.option_id: _snapshot(namespace, lane, image_input=True)}
    service, authority, pool, *_ = _service(tmp_path, (lane,), snapshots)
    pool.workspace.rmdir()

    report = await service.inspect(authority, "coding")

    candidate = report.candidates[0]
    assert candidate.eligible is False
    assert {reason.code for reason in candidate.reasons} == {
        "runtime_unavailable",
        "capability_unsupported",
    }
    workspace = next(
        item for item in candidate.capabilities if item.capability == RuntimeCapability.WORKSPACE_EXECUTION
    )
    assert workspace.support == CapabilitySupport.UNSUPPORTED


@pytest.mark.asyncio
async def test_cross_principal_authority_and_unknown_task_fail_closed(tmp_path: Path) -> None:
    namespace = _namespace(1)
    lane = _lane("claude", "anthropic", selected=True)
    service, authority, *_ = _service(
        tmp_path,
        (lane,),
        {lane.option_id: _snapshot(namespace, lane, image_input=True)},
    )

    with pytest.raises(RoutingEligibilityAccessDenied):
        service.authority_for_principal_channel(_id(PrincipalId, 2), namespace.channel_id)
    with pytest.raises(RoutingEligibilityError, match="Unsupported task class"):
        await service.inspect(authority, "best")


def test_image_capability_normalization_accepts_only_explicit_evidence() -> None:
    assert capability_assessment_for_model(_entry("supported", True)) == CapabilitySupport.SUPPORTED
    assert capability_assessment_for_model(_entry("unsupported", False)) == CapabilitySupport.UNSUPPORTED
    assert capability_assessment_for_model(_entry("unknown", None)) == CapabilitySupport.UNKNOWN
