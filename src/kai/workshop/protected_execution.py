"""Production-unused protected preparation for Workshop execution."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path

from kai.backend import StreamEvent
from kai.config import VALID_BACKENDS, validate_model_for_backend
from kai.pool import PreparedBackendExecution, SubprocessPool
from kai.workshop.conversation_runs import resolve_canonical_conversation_run
from kai.workshop.domain import RunId
from kai.workshop.run_execution_authority import RunExecutionSelection
from kai.workshop.run_lifecycle import DurableRun, RunStatus, WorkshopRunLifecycle
from kai.workshop.store import WorkshopEventStore

type AgentPrompt = str | list[dict[str, str]]


class ProtectedExecutionPreparationError(RuntimeError):
    """The canonical run cannot safely bind one effective runtime."""


@dataclass(frozen=True, slots=True)
class PreparedWorkshopExecution:
    run: DurableRun
    selection: RunExecutionSelection
    workspace: Path
    _runtime: PreparedBackendExecution = field(repr=False, compare=False)

    async def stream(self, prompt: AgentPrompt) -> AsyncIterator[StreamEvent]:
        """Dispatch once through the exact runtime bound during preparation."""
        async for event in self._runtime.stream(prompt):
            yield event

    def validate_current(self) -> None:
        """Verify the exact runtime immediately before the started boundary."""
        self._runtime.validate_current()

    async def cancel(self) -> None:
        """Stop only this prepared compatibility runtime."""
        await self._runtime.cancel()


class WorkshopProtectedExecutionPreparationService:
    """Resolve canonical run authority into one protected compatibility runtime."""

    def __init__(
        self,
        store: WorkshopEventStore,
        pool: SubprocessPool,
        *,
        registered_backend_ids: frozenset[str],
    ) -> None:
        if not registered_backend_ids:
            raise ValueError("registered_backend_ids must not be empty")
        unknown = registered_backend_ids - VALID_BACKENDS
        if unknown:
            raise ValueError(f"registered_backend_ids contains unsupported backends: {', '.join(sorted(unknown))}")
        self._store = store
        self._pool = pool
        self._registered_backend_ids = registered_backend_ids

    async def prepare(self, run_id: RunId) -> PreparedWorkshopExecution:
        if not isinstance(run_id, RunId):
            raise ValueError("run_id must be a RunId")
        run = await WorkshopRunLifecycle(self._store).state(run_id)
        if run.status != RunStatus.ACCEPTED or run.cancellation_requested_at is not None:
            raise ProtectedExecutionPreparationError("Only an uncancelled accepted run can prepare execution")

        resolution = await resolve_canonical_conversation_run(self._store, run.inbound_message_id)
        target = resolution.target
        if (
            target.workshop_id != run.workshop_id
            or target.channel_id != run.channel_id
            or target.requested_by_principal_id != run.requested_by_principal_id
            or target.agent_id != run.agent_id
        ):
            raise ProtectedExecutionPreparationError("Canonical run authority changed after acceptance")

        runtime = await self._pool.prepare_execution(resolution._legacy_pool_key)
        prepared = runtime.selection
        if prepared.backend not in self._registered_backend_ids:
            raise ProtectedExecutionPreparationError("Effective backend is not present in the protected registry")
        if not validate_model_for_backend(prepared.model, prepared.backend, prepared.provider):
            raise ProtectedExecutionPreparationError("Effective model is not valid for the protected backend selection")
        selection = RunExecutionSelection(
            backend=prepared.backend,
            provider=prepared.provider or None,
            model=prepared.model,
        )
        return PreparedWorkshopExecution(
            run=run,
            selection=selection,
            workspace=runtime.workspace,
            _runtime=runtime,
        )
