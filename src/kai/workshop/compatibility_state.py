"""Profile-addressed writes to Kai's remaining compatibility state."""

from __future__ import annotations

from dataclasses import dataclass, field

from kai import sessions
from kai.config import Config
from kai.conversation_compatibility import CanonicalMemoryProvenance, schedule_memory_ingestion
from kai.workshop.domain import RuntimeProfileId
from kai.workshop.runtime_pool import WorkshopRuntimePool


@dataclass(frozen=True, slots=True)
class WorkshopProfileCompatibilityState:
    """Write existing state for one protected runtime profile.

    The integer key is deliberately private to this adapter. Remaining
    compatibility calls retain it; the semantic-memory boundary resolves it
    to the canonical principal and rejects legacy-owner reads. Canonical
    Workshop callers address this state only through a runtime profile.
    """

    _runtime_config_id: int = field(repr=False)
    _config: Config = field(repr=False)
    _backend: str = field(repr=False)

    @property
    def memory_context_turns(self) -> int:
        """Return the configured canonical episode-context window."""
        return self._config.episode_classifier_context_turns

    async def save_session(self, session_id: str, model: str) -> None:
        await sessions.save_session(self._runtime_config_id, session_id, model)

    def schedule_memory_ingestion(
        self,
        *,
        prompt: str | list,
        assistant_text: str,
        session_id: str | None,
        workspace: str,
        canonical_provenance: CanonicalMemoryProvenance,
        canonical_prior_pairs: tuple[tuple[str, str], ...],
    ) -> None:
        schedule_memory_ingestion(
            prompt=prompt,
            assistant_text=assistant_text,
            chat_id=self._runtime_config_id,
            session_id=session_id,
            config=self._config,
            workspace=workspace,
            user_log=None,
            assistant_log=None,
            canonical_provenance=canonical_provenance,
            canonical_prior_pairs=canonical_prior_pairs,
            effective_backend=self._backend,
        )


class WorkshopCompatibilityStateWriter:
    """Bind protected runtime profiles to unchanged compatibility stores."""

    def __init__(
        self,
        config: Config,
        runtime_pool: WorkshopRuntimePool,
    ) -> None:
        self._config = config
        self._runtime_pool = runtime_pool

    def for_profile(
        self,
        runtime_profile_id: str | RuntimeProfileId,
    ) -> WorkshopProfileCompatibilityState:
        profile = self._runtime_pool.runtime_profile(runtime_profile_id)
        return WorkshopProfileCompatibilityState(
            profile.runtime_config_id,
            self._config,
            profile.backend,
        )
