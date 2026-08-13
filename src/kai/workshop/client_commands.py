"""Authorized asynchronous Workshop client command execution."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from kai.history import LogEntry
from kai.workshop.compatibility_state import WorkshopCompatibilityStateWriter
from kai.workshop.conversation_commands import (
    ClientConversationCommandAcceptance,
    ConversationCommandDisposition,
)
from kai.workshop.domain import RunId, RuntimeProfileId
from kai.workshop.execution_coordinator import (
    CanonicalCancellationDisposition,
    CanonicalExecutionDisposition,
)
from kai.workshop.inbound import ClientInboundMessage
from kai.workshop.private_text_execution import (
    RecoverableClientRun,
    WorkshopPrivateTextExecutionService,
)
from kai.workshop.run_lifecycle import DurableRun

log = logging.getLogger(__name__)
_RECONCILE_INTERVAL_SECONDS = 5.0


class ClientCommandExecutorUnavailableError(RuntimeError):
    """The process-level owner cannot accept another browser command."""


@dataclass(frozen=True, slots=True)
class ClientCommandSubmission:
    """Durable acceptance returned before protected execution completes."""

    acceptance: ClientConversationCommandAcceptance
    run: DurableRun


@dataclass(frozen=True, slots=True)
class _ClientRunContext:
    run_id: RunId
    runtime_profile_id: RuntimeProfileId
    body: str
    user_log: LogEntry | None = None


class WorkshopClientCommandExecutor:
    """Own browser-originated run tasks independently of HTTP requests."""

    def __init__(
        self,
        execution: WorkshopPrivateTextExecutionService,
        compatibility_state: WorkshopCompatibilityStateWriter,
    ) -> None:
        self._execution = execution
        self._compatibility_state = compatibility_state
        self._tasks: dict[RunId, asyncio.Task[None]] = {}
        self._task_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._reconcile_task: asyncio.Task[None] | None = None
        self._closed = False

    @property
    def ready(self) -> bool:
        task = self._reconcile_task
        return not self._closed and task is not None and not task.done()

    async def start(self) -> None:
        if self._closed:
            raise RuntimeError("Workshop client command executor is closed")
        if self._reconcile_task is not None:
            raise RuntimeError("Workshop client command executor is already started")
        await self._reconcile()
        self._reconcile_task = asyncio.create_task(
            self._reconcile_loop(),
            name="kai-workshop-client-run-reconciliation",
        )

    async def submit(self, message: ClientInboundMessage) -> ClientCommandSubmission:
        if not self.ready:
            raise ClientCommandExecutorUnavailableError("Workshop client command executor is unavailable")
        accepted = await self._execution.accept_client(message)
        user_log: LogEntry | None = None
        if accepted.command.disposition == ConversationCommandDisposition.NEWLY_ACCEPTED:
            compatibility_state = self._compatibility_state.for_profile(accepted.runtime_profile_id)
            user_log = compatibility_state.append_history(
                direction="user",
                text=message.body,
            )
        if accepted.command.disposition in {
            ConversationCommandDisposition.NEWLY_ACCEPTED,
            ConversationCommandDisposition.READY_REPLAY,
        }:
            await self._schedule(
                _ClientRunContext(
                    accepted.run.run_id,
                    accepted.runtime_profile_id,
                    message.body,
                    user_log,
                )
            )
        return ClientCommandSubmission(
            accepted,
            await self._execution.run_state(accepted.run.run_id),
        )

    async def state(self, run_id: RunId) -> DurableRun:
        return await self._execution.run_state(run_id)

    async def cancel(self, run_id: RunId) -> CanonicalCancellationDisposition:
        return await self._execution.request_run_cancellation(run_id)

    async def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        reconcile = self._reconcile_task
        if reconcile is not None:
            await asyncio.shield(reconcile)
        async with self._task_lock:
            active = tuple(self._tasks.items())
        for run_id, _ in active:
            try:
                await self._execution.request_run_cancellation(run_id)
            except Exception:
                log.exception("Could not cancel Workshop client run %s during shutdown", run_id)
        if active:
            await asyncio.gather(*(task for _, task in active), return_exceptions=True)
        self._reconcile_task = None

    async def _schedule(self, context: _ClientRunContext) -> None:
        async with self._task_lock:
            if self._closed or context.run_id in self._tasks:
                return
            task = asyncio.create_task(
                self._execute(context),
                name=f"kai-workshop-client-run:{context.run_id}",
            )
            self._tasks[context.run_id] = task

    async def _execute(self, context: _ClientRunContext) -> None:
        task = asyncio.current_task()
        try:
            result = await self._execution.execute(context.run_id)
            if result.terminal is None:
                return
            compatibility_state = self._compatibility_state.for_profile(context.runtime_profile_id)
            assistant_log = compatibility_state.append_history(
                direction="assistant",
                text=result.terminal.body,
            )
            if result.session_id and result.selection is not None:
                await compatibility_state.save_session(
                    result.session_id,
                    result.selection.model,
                )
            if result.disposition == CanonicalExecutionDisposition.COMPLETED and result.workspace is not None:
                compatibility_state.schedule_memory_ingestion(
                    prompt=context.body,
                    assistant_text=result.terminal.body,
                    session_id=result.session_id,
                    workspace=result.workspace,
                    user_log=context.user_log,
                    assistant_log=assistant_log,
                )
        except Exception:
            log.exception("Workshop client run task failed for %s", context.run_id)
        finally:
            async with self._task_lock:
                if self._tasks.get(context.run_id) is task:
                    del self._tasks[context.run_id]

    async def _reconcile(self) -> None:
        for recoverable in await self._execution.recoverable_client_runs():
            await self._schedule(self._recovery_context(recoverable))

    async def _reconcile_loop(self) -> None:
        while True:
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=_RECONCILE_INTERVAL_SECONDS,
                )
                return
            except TimeoutError:
                pass
            try:
                await self._reconcile()
            except Exception:
                log.exception("Workshop client run reconciliation failed")

    @staticmethod
    def _recovery_context(run: RecoverableClientRun) -> _ClientRunContext:
        return _ClientRunContext(
            run.run_id,
            run.runtime_profile_id,
            run.body,
        )
