"""Protected runtime-profile facade over Kai's compatibility subprocess pool."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

from kai.backend import StreamEvent
from kai.pool import PreparedBackendExecution, SubprocessPool
from kai.workshop.domain import RuntimeProfileId
from kai.workshop.runtime_profiles import ProtectedRuntimeProfile, WorkshopRuntimeProfileRegistry

type AgentPrompt = str | list[dict[str, str]]


class WorkshopRuntimePool:
    """Address compatibility runtimes only by protected Workshop profile.

    The underlying host pool and compatibility stores still require the
    configured-user integer key. This facade owns that conversion so canonical
    run resolution and execution services never receive or select it.
    """

    def __init__(
        self,
        pool: SubprocessPool,
        profiles: WorkshopRuntimeProfileRegistry,
    ) -> None:
        self._pool = pool
        self._profiles = profiles

    def compatibility_runtime_config_id(
        self,
        runtime_profile_id: str | RuntimeProfileId,
    ) -> int:
        """Return the private compatibility key for non-pool migrations."""
        return self._profiles.resolve(runtime_profile_id).runtime_config_id

    def runtime_profile(
        self,
        runtime_profile_id: str | RuntimeProfileId,
    ) -> ProtectedRuntimeProfile:
        """Resolve the protected profile without exposing compatibility lookup."""
        return self._profiles.resolve(runtime_profile_id)

    async def prepare_execution(
        self,
        runtime_profile_id: str | RuntimeProfileId,
    ) -> PreparedBackendExecution:
        runtime_config_id = self.compatibility_runtime_config_id(runtime_profile_id)
        return await self._pool.prepare_execution(runtime_config_id)

    def get_model(self, runtime_profile_id: str | RuntimeProfileId) -> str:
        runtime_config_id = self.compatibility_runtime_config_id(runtime_profile_id)
        return self._pool.get_model(runtime_config_id)

    async def send(
        self,
        prompt: AgentPrompt,
        *,
        runtime_profile_id: str | RuntimeProfileId,
    ) -> AsyncIterator[StreamEvent]:
        runtime_config_id = self.compatibility_runtime_config_id(runtime_profile_id)
        async for event in self._pool.send(prompt, chat_id=runtime_config_id):
            yield event

    async def get_effective_workspace(
        self,
        runtime_profile_id: str | RuntimeProfileId,
    ) -> Path:
        runtime_config_id = self.compatibility_runtime_config_id(runtime_profile_id)
        return await self._pool.get_effective_workspace(runtime_config_id)
