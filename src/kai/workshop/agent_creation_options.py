"""Principal-scoped readiness and choices for Workshop agent creation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from kai.workshop.domain import PrincipalId, RuntimeProfileId
from kai.workshop.model_catalogue import ModelCatalogueError, WorkshopModelCatalogueService
from kai.workshop.model_discovery_inventory import (
    ModelDiscoveryBackendInventory,
    ModelDiscoveryProfileInventory,
    ModelDiscoveryReadiness,
    WorkshopModelDiscoveryInventoryService,
)
from kai.workshop.runtime_pool import WorkshopRuntimePool
from kai.workshop.settings_workspaces import MIN_SELF_SERVICE_TIMEOUT_SECONDS


@dataclass(frozen=True, slots=True)
class AgentCreationBlocker:
    """One stable, actionable reason a creation choice is not ready."""

    code: str
    detail: str


@dataclass(frozen=True, slots=True)
class AgentCreationModelOption:
    """One model visible through the canonical model catalogue."""

    model_id: str
    display_name: str
    status: str
    selectable: bool
    retained: bool


@dataclass(frozen=True, slots=True)
class AgentCreationBackendOption:
    """One operator-authorized backend/provider choice."""

    option_id: str
    backend: str
    provider: str
    current: bool
    readiness: str
    default_model: str
    models: tuple[AgentCreationModelOption, ...]
    catalogue_status: str | None
    catalogue_stale: bool
    blockers: tuple[AgentCreationBlocker, ...]


@dataclass(frozen=True, slots=True)
class AgentCreationWorkspaceOption:
    """One authorized workspace path and its current availability."""

    path: str
    name: str
    default: bool
    available: bool


@dataclass(frozen=True, slots=True)
class AgentCreationRuntimeOption:
    """One principal-owned protected runtime suitable for agent creation."""

    runtime_profile_id: RuntimeProfileId
    display_name: str
    current_backend_option_id: str
    default_workspace: str | None
    default_timeout_seconds: int
    minimum_timeout_seconds: int
    maximum_timeout_seconds: int
    backends: tuple[AgentCreationBackendOption, ...]
    workspaces: tuple[AgentCreationWorkspaceOption, ...]
    ready: bool
    blockers: tuple[AgentCreationBlocker, ...]


@dataclass(frozen=True, slots=True)
class AgentCreationOptions:
    """Complete bounded creation readiness for one authenticated principal."""

    principal_id: PrincipalId
    ready: bool
    runtimes: tuple[AgentCreationRuntimeOption, ...]
    blockers: tuple[AgentCreationBlocker, ...]


class WorkshopAgentCreationOptionsService:
    """Project creation choices from existing canonical runtime authorities.

    Inspection is deliberately side-effect free: it never refreshes model
    discovery, starts a runtime, writes settings, or creates an agent draft.
    """

    def __init__(
        self,
        inventory: WorkshopModelDiscoveryInventoryService,
        model_catalogue: WorkshopModelCatalogueService,
        runtime_pool: WorkshopRuntimePool,
    ) -> None:
        self._inventory = inventory
        self._model_catalogue = model_catalogue
        self._runtime_pool = runtime_pool

    async def inspect(self, principal_id: PrincipalId) -> AgentCreationOptions:
        inventories = self._inventory.for_principal(principal_id)
        runtimes = tuple([await self._runtime(item) for item in inventories])
        blockers: tuple[AgentCreationBlocker, ...]
        if not runtimes:
            blockers = (
                AgentCreationBlocker(
                    "no_authorized_runtime_profiles",
                    "No protected runtime profile is authorized for this principal.",
                ),
            )
        elif not any(runtime.ready for runtime in runtimes):
            blockers = (
                AgentCreationBlocker(
                    "no_ready_runtime_profiles",
                    "No authorized runtime profile currently has a usable backend and default workspace.",
                ),
            )
        else:
            blockers = ()
        return AgentCreationOptions(principal_id, not blockers, runtimes, blockers)

    async def _runtime(self, inventory: ModelDiscoveryProfileInventory) -> AgentCreationRuntimeOption:
        profile = self._runtime_pool.runtime_profile(inventory.runtime_profile_id)
        backends = tuple(
            [
                await self._backend(
                    inventory.principal_id,
                    inventory.runtime_profile_id,
                    item,
                )
                for item in inventory.backends
            ]
        )
        workspaces, default_workspace = await self._workspaces(profile.profile_id, profile.workspace_base)
        blockers: list[AgentCreationBlocker] = []
        if not any(not backend.blockers for backend in backends):
            blockers.append(
                AgentCreationBlocker(
                    "no_available_backend",
                    "No authorized backend is currently available for this runtime profile.",
                )
            )
        if default_workspace is None:
            blockers.append(
                AgentCreationBlocker(
                    "default_workspace_unavailable",
                    "The runtime profile's default workspace is unavailable.",
                )
            )
        return AgentCreationRuntimeOption(
            runtime_profile_id=profile.profile_id,
            display_name=profile.display_name,
            current_backend_option_id=inventory.selected_option_id,
            default_workspace=default_workspace,
            default_timeout_seconds=profile.timeout_seconds,
            minimum_timeout_seconds=MIN_SELF_SERVICE_TIMEOUT_SECONDS,
            maximum_timeout_seconds=profile.maximum_timeout_seconds,
            backends=backends,
            workspaces=workspaces,
            ready=not blockers,
            blockers=tuple(blockers),
        )

    async def _backend(
        self,
        principal_id: PrincipalId,
        runtime_profile_id: RuntimeProfileId,
        inventory: ModelDiscoveryBackendInventory,
    ) -> AgentCreationBackendOption:
        models: tuple[AgentCreationModelOption, ...] = ()
        catalogue_status: str | None = None
        catalogue_stale = True
        try:
            snapshot = await self._model_catalogue.inspect(
                self._model_catalogue.authority_for_principal(principal_id),
                runtime_profile_id,
                inventory.option_id,
            )
        except ModelCatalogueError:
            pass
        else:
            models = tuple(
                AgentCreationModelOption(
                    item.model_id,
                    item.display_label,
                    item.status.value,
                    item.selectable,
                    item.retained,
                )
                for item in snapshot.entries
            )
            catalogue_status = snapshot.refresh.status.value if snapshot.refresh is not None else None
            catalogue_stale = snapshot.stale

        blockers: tuple[AgentCreationBlocker, ...] = ()
        if inventory.readiness == ModelDiscoveryReadiness.UNAVAILABLE:
            blockers = (
                AgentCreationBlocker(
                    "backend_unavailable",
                    "The authorized backend is not currently available.",
                ),
            )
        elif inventory.readiness == ModelDiscoveryReadiness.MISCONFIGURED:
            blockers = (
                AgentCreationBlocker(
                    "backend_misconfigured",
                    "The authorized backend requires operator attention.",
                ),
            )
        return AgentCreationBackendOption(
            option_id=inventory.option_id,
            backend=inventory.backend,
            provider=inventory.provider,
            current=inventory.selected,
            readiness=inventory.readiness.value,
            default_model=inventory.default_model,
            models=models,
            catalogue_status=catalogue_status,
            catalogue_stale=catalogue_stale,
            blockers=blockers,
        )

    async def _workspaces(
        self,
        runtime_profile_id: RuntimeProfileId,
        workspace_base: Path | None,
    ) -> tuple[tuple[AgentCreationWorkspaceOption, ...], str | None]:
        try:
            home = self._runtime_pool.get_home_workspace(runtime_profile_id).expanduser().resolve()
        except RuntimeError:
            return (), None
        _base, allowed = await self._runtime_pool.resolve_workspace_access(runtime_profile_id)
        candidates = {home, *(path.expanduser().resolve() for path in allowed)}
        options = tuple(
            AgentCreationWorkspaceOption(
                path=str(path),
                name=self._workspace_name(path, workspace_base, home),
                default=path == home,
                available=path.is_dir(),
            )
            for path in sorted(candidates, key=lambda item: (item != home, str(item)))
        )
        default_workspace = str(home) if home.is_dir() else None
        return options, default_workspace

    @staticmethod
    def _workspace_name(path: Path, base: Path | None, home: Path) -> str:
        if path == home:
            return "Home"
        if base is not None:
            try:
                return str(path.relative_to(base.resolve())) or path.name
            except ValueError:
                pass
        return path.name
