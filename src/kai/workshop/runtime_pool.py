"""Protected runtime-profile facade over Kai's canonical subprocess pool."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

from kai.backend import StreamEvent
from kai.config import ModelRole, WorkspaceConfig
from kai.workshop.domain import RuntimeProfileId
from kai.workshop.internal_api_contexts import WorkshopInternalAPIExecutionContext
from kai.workshop.runtime_profiles import (
    ProtectedRuntimeBackend,
    ProtectedRuntimeProfile,
    WorkshopRuntimeProfileRegistry,
)

# Type-only: kai.pool imports kai.sessions, which imports this package, so
# a runtime import here would close an import cycle. The pool arrives as a
# constructor argument; only its type names are needed at module scope.
if TYPE_CHECKING:
    from kai.pool import PreparedBackendExecution, SubprocessPool

type AgentPrompt = str | list[dict[str, str]]
type RuntimeAuthority = str | RuntimeProfileId | WorkshopInternalAPIExecutionContext


class WorkshopRuntimePool:
    """Address protected agent runtimes only by canonical Workshop profile."""

    def __init__(
        self,
        pool: SubprocessPool,
        profiles: WorkshopRuntimeProfileRegistry,
    ) -> None:
        self._pool = pool
        self._profiles = profiles

    def runtime_profile(
        self,
        runtime_profile_id: RuntimeAuthority,
    ) -> ProtectedRuntimeProfile:
        """Resolve the protected profile without exposing compatibility lookup."""
        return self._profiles.resolve(
            runtime_profile_id.runtime_profile_id
            if isinstance(runtime_profile_id, WorkshopInternalAPIExecutionContext)
            else runtime_profile_id
        )

    def _selector(self, authority: RuntimeAuthority):
        profile = self.runtime_profile(authority)
        return authority if isinstance(authority, WorkshopInternalAPIExecutionContext) else profile.profile_id

    def register_canonical_lane(self, context: WorkshopInternalAPIExecutionContext) -> None:
        self.runtime_profile(context)
        self._pool.register_canonical_lane(context)

    async def rebind_canonical_lane(
        self,
        prior: WorkshopInternalAPIExecutionContext,
        replacement: WorkshopInternalAPIExecutionContext,
    ) -> None:
        self.runtime_profile(replacement)
        await self._pool.rebind_canonical_lane(prior, replacement)

    async def suspend_canonical_lane(self, context: WorkshopInternalAPIExecutionContext) -> None:
        await self._pool.suspend_canonical_lane(context)

    def legacy_runtime_key(self, runtime_profile_id: str | RuntimeProfileId) -> int | None:
        """Return migration-only state for the temporary cutover coordinator."""
        return self._profiles.legacy_runtime_key(runtime_profile_id)

    async def prepare_execution(
        self,
        runtime_profile_id: RuntimeAuthority,
    ) -> PreparedBackendExecution:
        return await self._pool.prepare_execution(self._selector(runtime_profile_id))

    async def prepare_routed_execution(
        self,
        runtime_profile_id: RuntimeAuthority,
        backend_option_id: str,
        model: str,
    ) -> PreparedBackendExecution:
        """Prepare an authorized per-option runtime without changing the default."""
        return await self._pool.prepare_routed_execution(
            self._selector(runtime_profile_id),
            backend_option_id,
            model,
        )

    def get_model(self, runtime_profile_id: RuntimeAuthority) -> str:
        return self._pool.get_model(self._selector(runtime_profile_id))

    async def get_effective_model(self, runtime_profile_id: RuntimeAuthority) -> str:
        """Return canonical persisted selection without starting a backend."""
        return await self._pool.get_effective_model(self._selector(runtime_profile_id))

    def get_backend_provider(
        self,
        runtime_profile_id: RuntimeAuthority,
    ) -> tuple[str, str]:
        return self._pool.get_backend_provider(self._selector(runtime_profile_id))

    def backend_option(
        self,
        runtime_profile_id: RuntimeAuthority,
        backend: str,
    ) -> ProtectedRuntimeBackend:
        return self.runtime_profile(runtime_profile_id).backend_option(backend)

    def is_in_flight(self, runtime_profile_id: RuntimeAuthority) -> bool:
        return self._pool.is_in_flight(self._selector(runtime_profile_id))

    async def select_backend(
        self,
        runtime_profile_id: RuntimeAuthority,
        backend_option_id: str,
        *,
        commit_selection: Callable[[], Awaitable[None]] | None = None,
    ) -> bool:
        return await self._pool.select_backend(
            self._selector(runtime_profile_id),
            backend_option_id,
            commit_selection=commit_selection,
        )

    def is_running(self, runtime_profile_id: RuntimeAuthority) -> bool:
        """Return whether this protected profile currently has a live backend."""
        return self._pool.get_if_exists(self._selector(runtime_profile_id)) is not None

    def get_role_model(
        self,
        runtime_profile_id: RuntimeAuthority,
        role: ModelRole,
    ) -> str:
        """Resolve a role model through one protected canonical profile."""
        return self._pool.get_role_model(self._selector(runtime_profile_id), role)

    async def send(
        self,
        prompt: AgentPrompt,
        *,
        runtime_profile_id: RuntimeAuthority,
    ) -> AsyncIterator[StreamEvent]:
        async for event in self._pool.send(prompt, runtime=self._selector(runtime_profile_id)):
            yield event

    async def get_effective_workspace(
        self,
        runtime_profile_id: RuntimeAuthority,
    ) -> Path:
        return await self._pool.get_effective_workspace(self._selector(runtime_profile_id))

    def get_home_workspace(
        self,
        runtime_profile_id: RuntimeAuthority,
    ) -> Path:
        return self._pool.get_home_workspace(self._selector(runtime_profile_id))

    async def resolve_workspace_access(
        self,
        runtime_profile_id: RuntimeAuthority,
    ) -> tuple[Path | None, list[Path]]:
        return await self._pool.resolve_workspace_access(self._selector(runtime_profile_id))

    def set_model_if_running(
        self,
        runtime_profile_id: RuntimeAuthority,
        model: str,
    ) -> None:
        instance = self._pool.get_if_exists(self._selector(runtime_profile_id))
        if instance is not None:
            instance.model = model

    def set_timeout_if_running(
        self,
        runtime_profile_id: RuntimeAuthority,
        timeout_seconds: int,
    ) -> None:
        instance = self._pool.get_if_exists(self._selector(runtime_profile_id))
        if instance is not None:
            instance.timeout_seconds = timeout_seconds

    async def apply_workspace_config_if_running(
        self,
        runtime_profile_id: RuntimeAuthority,
        workspace: Path,
        *,
        workspace_config: WorkspaceConfig | None,
        model: str,
        timeout_seconds: int,
    ) -> None:
        """Apply effective config, restarting an existing runtime exactly once.

        Every backend's ``change_workspace`` contract kills its current
        subprocess before applying the replacement configuration.  Calling a
        second explicit restart would therefore be redundant and can add a
        second shutdown delay.
        """
        instance = self._pool.get_if_exists(self._selector(runtime_profile_id))
        if instance is None:
            return
        await instance.change_workspace(
            workspace,
            workspace_config=workspace_config,
        )
        # change_workspace resets to the backend's construction defaults and
        # applies workspace-local fields. Reapply the fully resolved values so
        # user-level overrides remain effective when the workspace has no
        # corresponding override.
        instance.model = model
        instance.timeout_seconds = timeout_seconds

    async def change_workspace(
        self,
        runtime_profile_id: RuntimeAuthority,
        workspace: Path,
        *,
        workspace_config: WorkspaceConfig | None,
    ) -> None:
        await self._pool.change_workspace(
            self._selector(runtime_profile_id),
            workspace,
            workspace_config=workspace_config,
        )

    async def restart(self, runtime_profile_id: RuntimeAuthority) -> None:
        await self._pool.restart(self._selector(runtime_profile_id))
