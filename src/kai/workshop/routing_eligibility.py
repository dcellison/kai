"""Explain capability-aware eligibility without selecting or invoking a runtime."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from kai.backend import AgentBackend
from kai.workshop.domain import AgentId, ChannelId, PrincipalId, RuntimeProfileId
from kai.workshop.execution_state import WorkshopExecutionStateRegistry
from kai.workshop.internal_api_contexts import WorkshopInternalAPIExecutionContext
from kai.workshop.model_catalogue import (
    ModelCatalogueEntry,
    ModelCatalogueEntryStatus,
    ModelCatalogueError,
    ModelCatalogueSnapshot,
    WorkshopModelCatalogueService,
)
from kai.workshop.model_discovery_inventory import (
    ModelDiscoveryBackendInventory,
    ModelDiscoveryReadiness,
    WorkshopModelDiscoveryInventoryService,
)
from kai.workshop.runtime_pool import WorkshopRuntimePool


class RoutingEligibilityError(RuntimeError):
    """A routing report cannot be resolved safely."""


class RoutingEligibilityAccessDenied(RoutingEligibilityError):
    """The caller does not own the requested canonical execution lane."""


class RoutingTaskClass(StrEnum):
    """Initial explicit task classes; automatic classification is forbidden."""

    CONVERSATION = "conversation"
    CODING = "coding"
    VISION = "vision"


class RuntimeCapability(StrEnum):
    """Small version-1 vocabulary derived from observable runtime facts."""

    TEXT_GENERATION = "text_generation"
    TOOL_ACTIVITY = "tool_activity"
    WORKSPACE_EXECUTION = "workspace_execution"
    IMAGE_INPUT = "image_input"


class CapabilitySupport(StrEnum):
    SUPPORTED = "supported"
    UNSUPPORTED = "unsupported"
    UNKNOWN = "unknown"


_TASK_REQUIREMENTS: dict[RoutingTaskClass, tuple[RuntimeCapability, ...]] = {
    RoutingTaskClass.CONVERSATION: (RuntimeCapability.TEXT_GENERATION,),
    RoutingTaskClass.CODING: (
        RuntimeCapability.TEXT_GENERATION,
        RuntimeCapability.TOOL_ACTIVITY,
        RuntimeCapability.WORKSPACE_EXECUTION,
    ),
    RoutingTaskClass.VISION: (
        RuntimeCapability.TEXT_GENERATION,
        RuntimeCapability.IMAGE_INPUT,
    ),
}


@dataclass(frozen=True, slots=True)
class RoutingEligibilityAuthority:
    principal_id: PrincipalId
    channel_id: ChannelId
    agent_id: AgentId
    runtime_profile_id: RuntimeProfileId


@dataclass(frozen=True, slots=True)
class CapabilityAssessment:
    capability: RuntimeCapability
    support: CapabilitySupport
    evidence: str


@dataclass(frozen=True, slots=True)
class EligibilityReason:
    code: str
    detail: str


@dataclass(frozen=True, slots=True)
class RuntimeEligibilityCandidate:
    option_id: str
    backend: str
    provider: str
    allowed_services: tuple[str, ...]
    model_id: str
    model_source: str
    selected: bool
    eligible: bool
    capabilities: tuple[CapabilityAssessment, ...]
    reasons: tuple[EligibilityReason, ...]


@dataclass(frozen=True, slots=True)
class RuntimeEligibilityReport:
    version: int
    task_class: RoutingTaskClass
    required_capabilities: tuple[RuntimeCapability, ...]
    principal_id: PrincipalId
    channel_id: ChannelId
    agent_id: AgentId
    runtime_profile_id: RuntimeProfileId
    workspace: str
    candidates: tuple[RuntimeEligibilityCandidate, ...]


class WorkshopRoutingEligibilityService:
    """Build reproducible reports from canonical and protected authorities."""

    def __init__(
        self,
        *,
        execution_state: WorkshopExecutionStateRegistry,
        inventory: WorkshopModelDiscoveryInventoryService,
        catalogue: WorkshopModelCatalogueService,
        runtime_pool: WorkshopRuntimePool,
    ) -> None:
        self._execution_state = execution_state
        self._inventory = inventory
        self._catalogue = catalogue
        self._runtime_pool = runtime_pool

    def authority_for_principal_channel(
        self,
        principal_id: str | PrincipalId,
        channel_id: str | ChannelId,
    ) -> RoutingEligibilityAuthority:
        namespace = self._execution_state.maybe_for_principal_channel(principal_id, channel_id)
        if namespace is None:
            raise RoutingEligibilityAccessDenied("Routing eligibility access denied")
        return RoutingEligibilityAuthority(
            namespace.principal_id,
            namespace.channel_id,
            namespace.agent_id,
            namespace.runtime_profile_id,
        )

    def authority_for_principal_runtime(
        self,
        principal_id: str | PrincipalId,
        runtime_profile_id: str | RuntimeProfileId,
    ) -> RoutingEligibilityAuthority:
        """Resolve a canonical run assignment independently of its channel."""
        namespace = self._execution_state.maybe_for_runtime_profile_id(runtime_profile_id)
        if namespace is None or namespace.principal_id != principal_id:
            raise RoutingEligibilityAccessDenied("Routing eligibility access denied")
        return RoutingEligibilityAuthority(
            namespace.principal_id,
            namespace.channel_id,
            namespace.agent_id,
            namespace.runtime_profile_id,
        )

    async def inspect(
        self,
        authority: RoutingEligibilityAuthority,
        task_class: str | RoutingTaskClass,
        *,
        additional_required: tuple[RuntimeCapability, ...] = (),
    ) -> RuntimeEligibilityReport:
        canonical_task = _task_class(task_class)
        namespace = self._execution_state.maybe_for_principal_channel(
            authority.principal_id,
            authority.channel_id,
        )
        if (
            namespace is None
            or namespace.principal_id != authority.principal_id
            or namespace.channel_id != authority.channel_id
            or namespace.agent_id != authority.agent_id
        ):
            raise RoutingEligibilityAccessDenied("Routing eligibility access denied")
        profiles = self._inventory.for_principal(authority.principal_id)
        profile = next(
            (item for item in profiles if item.runtime_profile_id == authority.runtime_profile_id),
            None,
        )
        if profile is None:
            raise RoutingEligibilityAccessDenied("Routing eligibility access denied")

        runtime_authority = WorkshopInternalAPIExecutionContext(
            authority.principal_id,
            authority.channel_id,
            authority.agent_id,
            authority.runtime_profile_id,
        )
        workspace = await self._runtime_pool.get_effective_workspace(runtime_authority)
        selected_backend, selected_provider = self._runtime_pool.get_backend_provider(runtime_authority)
        selected_option_id = f"{selected_backend}:{selected_provider}"
        selected_model = await self._runtime_pool.get_effective_model(runtime_authority)
        allowed_services = self._runtime_pool.runtime_profile(authority.runtime_profile_id).allowed_services
        catalogue_authority = self._catalogue.authority_for_principal(authority.principal_id)
        if any(not isinstance(capability, RuntimeCapability) for capability in additional_required):
            raise ValueError("additional_required must contain RuntimeCapability values")
        required = tuple(dict.fromkeys((*_TASK_REQUIREMENTS[canonical_task], *additional_required)))
        candidates: list[RuntimeEligibilityCandidate] = []
        for lane in profile.backends:
            selected = lane.option_id == selected_option_id
            model_id = selected_model if selected else lane.default_model
            model_source = "current_selection" if selected else "protected_default"
            snapshot: ModelCatalogueSnapshot | None
            try:
                snapshot = await self._catalogue.inspect(
                    catalogue_authority,
                    authority.runtime_profile_id,
                    lane.option_id,
                )
            except ModelCatalogueError:
                snapshot = None
            candidates.append(
                self._candidate(
                    lane,
                    model_id=model_id,
                    model_source=model_source,
                    selected=selected,
                    allowed_services=allowed_services,
                    workspace=workspace,
                    required=required,
                    catalogue=snapshot,
                )
            )
        return RuntimeEligibilityReport(
            version=1,
            task_class=canonical_task,
            required_capabilities=required,
            principal_id=authority.principal_id,
            channel_id=authority.channel_id,
            agent_id=authority.agent_id,
            runtime_profile_id=authority.runtime_profile_id,
            workspace=str(workspace.resolve()),
            candidates=tuple(candidates),
        )

    @staticmethod
    def _candidate(
        lane: ModelDiscoveryBackendInventory,
        *,
        model_id: str,
        model_source: str,
        selected: bool,
        allowed_services: tuple[str, ...],
        workspace: Path,
        required: tuple[RuntimeCapability, ...],
        catalogue: ModelCatalogueSnapshot | None,
    ) -> RuntimeEligibilityCandidate:
        assessments = _capability_assessments(
            workspace=workspace,
            model_id=model_id,
            catalogue=catalogue,
        )
        by_capability = {item.capability: item for item in assessments}
        reasons: list[EligibilityReason] = []
        if lane.readiness == ModelDiscoveryReadiness.UNAVAILABLE:
            reasons.append(EligibilityReason("runtime_unavailable", "The authorized runtime is unavailable."))
        elif lane.readiness == ModelDiscoveryReadiness.MISCONFIGURED:
            reasons.append(EligibilityReason("runtime_misconfigured", "The authorized runtime is misconfigured."))
        elif lane.readiness == ModelDiscoveryReadiness.UNVERIFIED:
            reasons.append(
                EligibilityReason(
                    "runtime_unverified",
                    "Runtime authentication is backend-managed and cannot be verified without execution.",
                )
            )
        for capability in required:
            assessment = by_capability[capability]
            if assessment.support == CapabilitySupport.UNSUPPORTED:
                reasons.append(
                    EligibilityReason(
                        "capability_unsupported",
                        f"Required capability {capability.value} is unsupported.",
                    )
                )
            elif assessment.support == CapabilitySupport.UNKNOWN:
                reasons.append(
                    EligibilityReason(
                        "capability_unknown",
                        f"Required capability {capability.value} has no current supporting evidence.",
                    )
                )
        blocked = any(
            reason.code
            in {
                "runtime_unavailable",
                "runtime_misconfigured",
                "capability_unsupported",
                "capability_unknown",
            }
            for reason in reasons
        )
        if not blocked:
            reasons.append(EligibilityReason("eligible", "All required capability checks passed."))
        return RuntimeEligibilityCandidate(
            option_id=lane.option_id,
            backend=lane.backend,
            provider=lane.provider,
            allowed_services=allowed_services,
            model_id=model_id,
            model_source=model_source,
            selected=selected,
            eligible=not blocked,
            capabilities=assessments,
            reasons=tuple(reasons),
        )


def _task_class(value: str | RoutingTaskClass) -> RoutingTaskClass:
    try:
        return value if isinstance(value, RoutingTaskClass) else RoutingTaskClass(value)
    except (TypeError, ValueError) as exc:
        valid = ", ".join(item.value for item in RoutingTaskClass)
        raise RoutingEligibilityError(f"Unsupported task class; expected one of: {valid}") from exc


def _capability_assessments(
    *,
    workspace: Path,
    model_id: str,
    catalogue: ModelCatalogueSnapshot | None,
) -> tuple[CapabilityAssessment, ...]:
    contract = AgentBackend.routing_capabilities
    image_support, image_evidence = _image_input_support(model_id, catalogue)
    return (
        CapabilityAssessment(
            RuntimeCapability.TEXT_GENERATION,
            _contract_support(RuntimeCapability.TEXT_GENERATION, contract),
            "agent_backend_contract_v1",
        ),
        CapabilityAssessment(
            RuntimeCapability.TOOL_ACTIVITY,
            _contract_support(RuntimeCapability.TOOL_ACTIVITY, contract),
            "agent_backend_contract_v1",
        ),
        CapabilityAssessment(
            RuntimeCapability.WORKSPACE_EXECUTION,
            CapabilitySupport.SUPPORTED if workspace.is_dir() else CapabilitySupport.UNSUPPORTED,
            "protected_workspace" if workspace.is_dir() else "workspace_unavailable",
        ),
        CapabilityAssessment(
            RuntimeCapability.IMAGE_INPUT,
            image_support,
            image_evidence,
        ),
    )


def _contract_support(
    capability: RuntimeCapability,
    contract: frozenset[str],
) -> CapabilitySupport:
    return CapabilitySupport.SUPPORTED if capability.value in contract else CapabilitySupport.UNKNOWN


def _image_input_support(
    model_id: str,
    catalogue: ModelCatalogueSnapshot | None,
) -> tuple[CapabilitySupport, str]:
    if catalogue is None:
        return CapabilitySupport.UNKNOWN, "catalogue_unavailable"
    if catalogue.stale:
        return CapabilitySupport.UNKNOWN, "catalogue_stale"
    entry = next((item for item in catalogue.entries if item.model_id == model_id), None)
    if entry is None or entry.status == ModelCatalogueEntryStatus.UNAVAILABLE:
        return CapabilitySupport.UNKNOWN, "model_not_available"
    observations = {
        observed
        for provenance in entry.provenances
        if (observed := _image_observation(provenance.capabilities)) is not None
    }
    if observations == {True}:
        return CapabilitySupport.SUPPORTED, "model_catalogue"
    if observations == {False}:
        return CapabilitySupport.UNSUPPORTED, "model_catalogue"
    if observations == {False, True}:
        return CapabilitySupport.UNKNOWN, "catalogue_conflict"
    return CapabilitySupport.UNKNOWN, "capability_not_advertised"


def _image_observation(capabilities: object) -> bool | None:
    if not isinstance(capabilities, dict):
        return None
    provider = capabilities.get("provider_capabilities")
    if isinstance(provider, dict):
        image = provider.get("image_input")
        if isinstance(image, dict) and isinstance(image.get("supported"), bool):
            return bool(image["supported"])
    for field in ("input_modalities", "input"):
        modalities = capabilities.get(field)
        if isinstance(modalities, list) and all(isinstance(item, str) for item in modalities):
            return "image" in modalities
    return None


def capability_assessment_for_model(
    model: ModelCatalogueEntry,
) -> CapabilitySupport:
    """Testable normalization helper for one fresh catalogue entry."""
    observations = {
        observed
        for provenance in model.provenances
        if (observed := _image_observation(provenance.capabilities)) is not None
    }
    if observations == {True}:
        return CapabilitySupport.SUPPORTED
    if observations == {False}:
        return CapabilitySupport.UNSUPPORTED
    return CapabilitySupport.UNKNOWN
