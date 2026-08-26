"""Protected runtime-profile facade over Kai's canonical subprocess pool."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING

from kai.backend import StreamEvent
from kai.config import ModelRole, WorkspaceConfig
from kai.workshop.domain import RuntimeProfileId
from kai.workshop.runtime_profiles import ProtectedRuntimeProfile, WorkshopRuntimeProfileRegistry

# Type-only: kai.pool imports kai.sessions, which imports this package, so
# a runtime import here would close an import cycle. The pool arrives as a
# constructor argument; only its type names are needed at module scope.
if TYPE_CHECKING:
    from kai.pool import PreparedBackendExecution, SubprocessPool

type AgentPrompt = str | list[dict[str, str]]


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
        runtime_profile_id: str | RuntimeProfileId,
    ) -> ProtectedRuntimeProfile:
        """Resolve the protected profile without exposing compatibility lookup."""
        return self._profiles.resolve(runtime_profile_id)

    def legacy_runtime_key(self, runtime_profile_id: str | RuntimeProfileId) -> int | None:
        """Return migration-only state for the temporary cutover coordinator."""
        return self._profiles.legacy_runtime_key(runtime_profile_id)

    async def prepare_execution(
        self,
        runtime_profile_id: str | RuntimeProfileId,
    ) -> PreparedBackendExecution:
        profile_id = self._profiles.resolve(runtime_profile_id).profile_id
        return await self._pool.prepare_execution(profile_id)

    def get_model(self, runtime_profile_id: str | RuntimeProfileId) -> str:
        profile_id = self._profiles.resolve(runtime_profile_id).profile_id
        return self._pool.get_model(profile_id)

    def is_running(self, runtime_profile_id: str | RuntimeProfileId) -> bool:
        """Return whether this protected profile currently has a live backend."""
        profile_id = self._profiles.resolve(runtime_profile_id).profile_id
        return self._pool.get_if_exists(profile_id) is not None

    def get_role_model(
        self,
        runtime_profile_id: str | RuntimeProfileId,
        role: ModelRole,
    ) -> str:
        """Resolve a role model through one protected canonical profile."""
        profile_id = self._profiles.resolve(runtime_profile_id).profile_id
        return self._pool.get_role_model(profile_id, role)

    async def send(
        self,
        prompt: AgentPrompt,
        *,
        runtime_profile_id: str | RuntimeProfileId,
    ) -> AsyncIterator[StreamEvent]:
        profile_id = self._profiles.resolve(runtime_profile_id).profile_id
        async for event in self._pool.send(prompt, runtime=profile_id):
            yield event

    async def get_effective_workspace(
        self,
        runtime_profile_id: str | RuntimeProfileId,
    ) -> Path:
        profile_id = self._profiles.resolve(runtime_profile_id).profile_id
        return await self._pool.get_effective_workspace(profile_id)

    def get_home_workspace(
        self,
        runtime_profile_id: str | RuntimeProfileId,
    ) -> Path:
        profile_id = self._profiles.resolve(runtime_profile_id).profile_id
        return self._pool.get_home_workspace(profile_id)

    async def resolve_workspace_access(
        self,
        runtime_profile_id: str | RuntimeProfileId,
    ) -> tuple[Path | None, list[Path]]:
        profile_id = self._profiles.resolve(runtime_profile_id).profile_id
        return await self._pool.resolve_workspace_access(profile_id)

    def set_model_if_running(
        self,
        runtime_profile_id: str | RuntimeProfileId,
        model: str,
    ) -> None:
        profile_id = self._profiles.resolve(runtime_profile_id).profile_id
        instance = self._pool.get_if_exists(profile_id)
        if instance is not None:
            instance.model = model

    def set_timeout_if_running(
        self,
        runtime_profile_id: str | RuntimeProfileId,
        timeout_seconds: int,
    ) -> None:
        profile_id = self._profiles.resolve(runtime_profile_id).profile_id
        instance = self._pool.get_if_exists(profile_id)
        if instance is not None:
            instance.timeout_seconds = timeout_seconds

    async def apply_workspace_config_if_running(
        self,
        runtime_profile_id: str | RuntimeProfileId,
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
        profile_id = self._profiles.resolve(runtime_profile_id).profile_id
        instance = self._pool.get_if_exists(profile_id)
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
        runtime_profile_id: str | RuntimeProfileId,
        workspace: Path,
        *,
        workspace_config: WorkspaceConfig | None,
    ) -> None:
        profile_id = self._profiles.resolve(runtime_profile_id).profile_id
        await self._pool.change_workspace(
            profile_id,
            workspace,
            workspace_config=workspace_config,
        )

    async def restart(self, runtime_profile_id: str | RuntimeProfileId) -> None:
        profile_id = self._profiles.resolve(runtime_profile_id).profile_id
        await self._pool.restart(profile_id)
