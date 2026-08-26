"""Canonical mutable state addressed by protected Workshop runtime profile."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from kai import sessions
from kai.config import Config
from kai.conversation_compatibility import ingest_conversation_memory
from kai.workshop.domain import CanonicalMemoryProvenance, RuntimeProfileId
from kai.workshop.execution_state import WorkshopExecutionStateRegistry
from kai.workshop.runtime_pool import WorkshopRuntimePool
from kai.workshop.runtime_profiles import ProtectedRuntimeProfile


@dataclass(frozen=True, slots=True)
class WorkshopProfileRuntimeState:
    """Canonical state and policy for one protected runtime profile."""

    _profile: ProtectedRuntimeProfile = field(repr=False)
    _principal_id: str = field(repr=False)
    _config: Config = field(repr=False)

    @property
    def memory_context_turns(self) -> int:
        return self._config.episode_classifier_context_turns

    async def github_token(self) -> str | None:
        return await sessions.get_canonical_github_token(self._principal_id)

    @property
    def allowed_triage_projects(self) -> tuple[str, ...]:
        return self._profile.allowed_triage_projects

    async def ingest_memory(
        self,
        *,
        prompt: str | list,
        assistant_text: str,
        session_id: str | None,
        workspace: str,
        canonical_provenance: CanonicalMemoryProvenance,
        canonical_prior_pairs: tuple[tuple[str, str], ...],
    ) -> None:
        await ingest_conversation_memory(
            prompt=prompt,
            assistant_text=assistant_text,
            chat_id=None,
            canonical_user_id=self._principal_id,
            runtime_profile_id=str(self._profile.profile_id),
            session_id=session_id,
            config=self._config,
            workspace=workspace,
            user_log=None,
            assistant_log=None,
            canonical_provenance=canonical_provenance,
            canonical_prior_pairs=canonical_prior_pairs,
            effective_backend=self._profile.backend,
            effective_provider=self._profile.provider,
            os_user_override=self._profile.os_user,
        )

    async def has_memory_for_run(self, run_id: str) -> bool:
        from kai import memory

        memories = await asyncio.to_thread(
            memory.get_all,
            user_id=self._principal_id,
            runtime_profile_id=str(self._profile.profile_id),
            limit=None,
        )
        return any(str(item.metadata.get(memory.WORKSHOP_RUN_ID_KEY, "")) == run_id for item in memories)


class WorkshopRuntimeStateWriter:
    """Bind opaque runtime profiles to canonical principal-owned state."""

    def __init__(
        self,
        config: Config,
        runtime_pool: WorkshopRuntimePool,
        execution_state: WorkshopExecutionStateRegistry,
    ) -> None:
        self._config = config
        self._runtime_pool = runtime_pool
        self._execution_state = execution_state

    def for_profile(
        self,
        runtime_profile_id: str | RuntimeProfileId,
    ) -> WorkshopProfileRuntimeState:
        profile = self._runtime_pool.runtime_profile(runtime_profile_id)
        namespace = self._execution_state.resolve_profile(profile.profile_id)
        return WorkshopProfileRuntimeState(
            profile,
            str(namespace.principal_id),
            self._config,
        )
