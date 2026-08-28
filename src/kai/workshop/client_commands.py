"""Authorized asynchronous Workshop client command execution."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from kai.streaming_text import stream_publishable_prefix
from kai.workshop.artifacts import StagedArtifact
from kai.workshop.conversation_commands import (
    ClientConversationCommandAcceptance,
    ConversationCommandDisposition,
)
from kai.workshop.domain import MessageId, RunId, RuntimeProfileId
from kai.workshop.execution_coordinator import (
    CanonicalCancellationDisposition,
)
from kai.workshop.inbound import ClientInboundMessage
from kai.workshop.private_text_execution import (
    RecoverableClientRun,
    WorkshopPrivateTextExecutionService,
)
from kai.workshop.run_lifecycle import DurableRun
from kai.workshop.run_previews import WorkshopRunPreviewRegistry

log = logging.getLogger(__name__)
_RECONCILE_INTERVAL_SECONDS = 5.0


class ClientCommandExecutorUnavailableError(RuntimeError):
    """The process-level owner cannot accept another browser command."""


@dataclass(frozen=True, slots=True)
class ClientCommandSubmission:
    """Durable acceptance returned before protected execution completes."""

    acceptance: ClientConversationCommandAcceptance
    runs: tuple[DurableRun, ...]

    @property
    def run(self) -> DurableRun:
        if len(self.runs) != 1:
            raise RuntimeError("Client command did not accept exactly one run")
        return self.runs[0]


@dataclass(frozen=True, slots=True)
class _ClientRunContext:
    run_id: RunId
    runtime_profile_id: RuntimeProfileId
    inbound_message_id: MessageId
    body: str


class WorkshopClientCommandExecutor:
    """Own browser-originated run tasks independently of HTTP requests."""

    def __init__(
        self,
        execution: WorkshopPrivateTextExecutionService,
        *,
        run_previews: WorkshopRunPreviewRegistry | None = None,
    ) -> None:
        self._execution = execution
        self._run_previews = run_previews
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

    async def submit(
        self,
        message: ClientInboundMessage,
        *,
        artifact: StagedArtifact | None = None,
    ) -> ClientCommandSubmission:
        if not self.ready:
            raise ClientCommandExecutorUnavailableError("Workshop client command executor is unavailable")
        accepted = await self._execution.accept_client(message, artifact=artifact)
        lifecycles = getattr(accepted.command, "lifecycles", None)
        if lifecycles is None:
            run_profiles = ((accepted.run, accepted.runtime_profile_id, accepted.command.disposition),)
        else:
            run_profiles = tuple(
                (lifecycle.run, runtime_profile_id, disposition)
                for lifecycle, runtime_profile_id, disposition in zip(
                    lifecycles,
                    accepted.runtime_profile_ids,
                    accepted.command.run_dispositions,
                    strict=True,
                )
            )
        inbound_message_id: MessageId | None = None
        for accepted_run, runtime_profile_id, disposition in run_profiles:
            if disposition in {
                ConversationCommandDisposition.NEWLY_ACCEPTED,
                ConversationCommandDisposition.READY_REPLAY,
            }:
                if inbound_message_id is None:
                    candidate = accepted.command.message.event.envelope.aggregate_id
                    if not isinstance(candidate, MessageId):
                        raise RuntimeError("Workshop client command did not identify a canonical message")
                    inbound_message_id = candidate
                await self._schedule(
                    _ClientRunContext(
                        accepted_run.run_id,
                        runtime_profile_id,
                        inbound_message_id,
                        message.body,
                    )
                )
        runs: list[DurableRun] = []
        for accepted_run, _, _ in run_profiles:
            runs.append(await self._execution.run_state(accepted_run.run_id))
        return ClientCommandSubmission(accepted, tuple(runs))

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
            await self._execution.execute(
                context.run_id,
                stream_observer=await self._preview_observer(context.run_id),
            )
        except Exception:
            log.exception("Workshop client run task failed for %s", context.run_id)
        finally:
            if self._run_previews is not None:
                self._run_previews.clear(context.run_id)
            async with self._task_lock:
                if self._tasks.get(context.run_id) is task:
                    del self._tasks[context.run_id]

    async def _preview_observer(self, run_id: RunId):
        """Build a stream observer that publishes stable partial text."""
        previews = self._run_previews
        if previews is None:
            return None
        channel_id = (await self._execution.run_state(run_id)).channel_id
        last_published: str | None = None

        async def observe(event) -> None:
            nonlocal last_published
            if not event.text_so_far:
                return
            publishable = stream_publishable_prefix(event.text_so_far)
            if publishable is None or publishable == last_published:
                return
            last_published = publishable
            previews.publish(run_id, channel_id, publishable)

        return observe

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
            run.inbound_message_id,
            run.body,
        )
