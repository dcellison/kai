"""Protected preparation for Workshop execution."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from kai.backend import StreamEvent
from kai.config import VALID_BACKENDS, validate_model_for_backend
from kai.workshop.domain import ChannelId, RunId, RuntimeProfileId
from kai.workshop.internal_api_contexts import WorkshopInternalAPIExecutionContext
from kai.workshop.routing_policy import RunRoutingDecision, WorkshopRoutingPolicyService
from kai.workshop.run_execution_authority import RunExecutionSelection
from kai.workshop.run_lifecycle import DurableRun, RunStatus, WorkshopRunLifecycle
from kai.workshop.runtime_pool import WorkshopRuntimePool
from kai.workshop.store import WorkshopEventStore

# Type-only: kai.pool imports kai.sessions, which imports this package, so
# a runtime import here would close an import cycle. The prepared execution
# arrives from the runtime pool; only its type name is needed here.
if TYPE_CHECKING:
    from kai.pool import PreparedBackendExecution

type AgentPrompt = str | list[dict[str, str]]


class ProtectedExecutionPreparationError(RuntimeError):
    """The canonical run cannot safely bind one effective runtime."""


class ProtectedExecutionRoutingRejected(ProtectedExecutionPreparationError):
    """An explicit routed task failed its configured conservative policy."""

    def __init__(self, run: DurableRun, decision: RunRoutingDecision) -> None:
        super().__init__(f"Explicit task route rejected: {decision.reason_code}")
        self.run = run
        self.decision = decision


@dataclass(frozen=True, slots=True)
class PreparedWorkshopExecution:
    run: DurableRun
    runtime_profile_id: RuntimeProfileId
    selection: RunExecutionSelection
    workspace: Path
    history_reader_user: str | None
    routing_decision: RunRoutingDecision
    _runtime: PreparedBackendExecution = field(repr=False, compare=False)

    async def stream(self, prompt: AgentPrompt) -> AsyncIterator[StreamEvent]:
        """Dispatch once through the exact runtime bound during preparation."""
        async for event in self._runtime.stream(prompt):
            yield event

    def stage_canonical_history(self, history: str) -> None:
        self._runtime.stage_canonical_history(history)

    def stage_agent_definition_context(self, context: str) -> None:
        self._runtime.stage_canonical_agent_context(context)

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
        pool: WorkshopRuntimePool,
        routing_policy: WorkshopRoutingPolicyService,
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
        self._routing_policy = routing_policy
        self._registered_backend_ids = registered_backend_ids

    async def prepare(self, run_id: RunId) -> PreparedWorkshopExecution:
        if not isinstance(run_id, RunId):
            raise ValueError("run_id must be a RunId")
        run = await WorkshopRunLifecycle(self._store).state(run_id)
        if run.status != RunStatus.ACCEPTED or run.cancellation_requested_at is not None:
            raise ProtectedExecutionPreparationError("Only an uncancelled accepted run can prepare execution")

        if run.runtime_profile_id is None or run.sponsor_principal_id is None:
            raise ProtectedExecutionPreparationError("Canonical run is missing its runtime sponsorship snapshot")
        async with self._store.connection.execute(
            "SELECT kind FROM channels WHERE id = ? AND workshop_id = ?",
            (run.channel_id, run.workshop_id),
        ) as cursor:
            channel_row = await cursor.fetchone()
        if channel_row is None or str(channel_row[0]) not in {"direct", "group"}:
            raise ProtectedExecutionPreparationError("Canonical run channel is unavailable")
        private_context = str(channel_row[0]) == "direct"
        async with self._store.connection.execute(
            "SELECT owner_direct_channel_id FROM agent_definitions "
            "WHERE agent_id = ? AND owner_principal_id = ? "
            "AND owner_runtime_profile_id = ? AND lifecycle_state = 'active'",
            (run.agent_id, run.sponsor_principal_id, run.runtime_profile_id),
        ) as cursor:
            authority_row = await cursor.fetchone()
        settings_channel_id = (
            ChannelId(str(authority_row[0]))
            if authority_row is not None and authority_row[0] is not None
            else run.channel_id
        )

        decision = await self._routing_policy.decide_for_run(
            run,
            run.runtime_profile_id,
        )
        if decision.rejected or decision.selected_backend_option_id is None:
            raise ProtectedExecutionRoutingRejected(run, decision)
        runtime_authority = WorkshopInternalAPIExecutionContext(
            principal_id=run.requested_by_principal_id,
            channel_id=run.channel_id,
            agent_id=run.agent_id,
            runtime_profile_id=run.runtime_profile_id,
            private_context=private_context,
            sponsor_principal_id=run.sponsor_principal_id,
            settings_channel_id=settings_channel_id,
        )
        runtime = await self._pool.prepare_routed_execution(
            runtime_authority,
            decision.selected_backend_option_id,
            decision.selection.model,
        )
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
        if selection != decision.selection:
            await runtime.cancel()
            raise ProtectedExecutionPreparationError("Prepared runtime does not match the durable routing decision")
        profile = self._pool.runtime_profile(run.runtime_profile_id)
        return PreparedWorkshopExecution(
            run=run,
            runtime_profile_id=run.runtime_profile_id,
            selection=selection,
            workspace=runtime.workspace,
            history_reader_user=profile.os_user,
            routing_decision=decision,
            _runtime=runtime,
        )
