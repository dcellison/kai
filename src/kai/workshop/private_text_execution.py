"""Production owner for authenticated private-text Workshop execution."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from kai.workshop.artifacts import StagedArtifact
from kai.workshop.conversation_commands import (
    ClientConversationCommandAcceptance,
    ConversationCommandAcceptance,
    WorkshopConversationCommandService,
)
from kai.workshop.conversation_context import assemble_canonical_prior_pairs
from kai.workshop.delivery_policy import WorkshopDeliveryBindingPolicy
from kai.workshop.domain import MessageId, RunId, RuntimeProfileId
from kai.workshop.execution_coordinator import (
    CanonicalCancellationDisposition,
    CanonicalExecutionResult,
    StreamObserver,
    SuccessTransformer,
    WorkshopCanonicalExecutionCoordinator,
)
from kai.workshop.inbound import ClientInboundMessage, InboundMessage, ScheduledInboundMessage
from kai.workshop.protected_execution import WorkshopProtectedExecutionPreparationService
from kai.workshop.run_lifecycle import WorkshopRunLifecycle
from kai.workshop.runtime_pool import WorkshopRuntimePool
from kai.workshop.store import WorkshopEventStore
from kai.workshop.transcript_export import CanonicalTranscriptProjection

_RECOVERY_INTERVAL_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class RecoverableClientRun:
    """One accepted Workshop-client run that has no active attempt."""

    run_id: RunId
    runtime_profile_id: RuntimeProfileId
    inbound_message_id: MessageId
    body: str


class WorkshopPrivateTextExecutionService:
    """Own one store, coordinator, and recovery loop for private text."""

    def __init__(
        self,
        store: WorkshopEventStore,
        coordinator: WorkshopCanonicalExecutionCoordinator,
        command_service: WorkshopConversationCommandService,
        database_lock: asyncio.Lock,
        runtime_pool: WorkshopRuntimePool,
    ) -> None:
        self._store = store
        self._coordinator = coordinator
        self._command_service = command_service
        self._database_lock = database_lock
        self._runtime_pool = runtime_pool
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    @classmethod
    async def open_and_start(
        cls,
        database_path: Path,
        runtime_pool: WorkshopRuntimePool,
        *,
        registered_backend_ids: frozenset[str],
        delivery_policy: WorkshopDeliveryBindingPolicy,
    ) -> WorkshopPrivateTextExecutionService:
        store = await WorkshopEventStore.open(database_path)
        database_lock = asyncio.Lock()
        coordinator = WorkshopCanonicalExecutionCoordinator(
            store,
            WorkshopProtectedExecutionPreparationService(
                store,
                runtime_pool,
                registered_backend_ids=registered_backend_ids,
            ),
            registered_backend_ids=registered_backend_ids,
            database_lock=database_lock,
            transcript_projection=CanonicalTranscriptProjection(database_path.parent / "history"),
            artifact_storage_root=database_path.parent / "files",
            delivery_policy=delivery_policy,
        )
        service = cls(
            store,
            coordinator,
            WorkshopConversationCommandService(
                store,
                artifact_storage_root=database_path.parent / "files",
                delivery_policy=delivery_policy,
            ),
            database_lock,
            runtime_pool,
        )
        try:
            await coordinator.recover_expired()
            service._task = asyncio.create_task(
                service._recovery_loop(),
                name="kai-workshop-private-text-recovery",
            )
        except BaseException:
            await store.close()
            raise
        return service

    @property
    def ready(self) -> bool:
        return not self._closed and self._task is not None and not self._task.done()

    async def accept(
        self,
        message: InboundMessage,
        *,
        artifact: StagedArtifact | None = None,
    ) -> ConversationCommandAcceptance:
        if self._closed:
            raise RuntimeError("Workshop private-text execution service is closed")
        async with self._database_lock:
            return await self._command_service.accept(message, artifact=artifact)

    async def accept_client(
        self,
        message: ClientInboundMessage,
        *,
        artifact: StagedArtifact | None = None,
    ) -> ClientConversationCommandAcceptance:
        if self._closed:
            raise RuntimeError("Workshop private-text execution service is closed")
        async with self._database_lock:
            return await self._command_service.accept_client(message, artifact=artifact)

    async def accept_scheduled(
        self,
        message: ScheduledInboundMessage,
    ) -> ClientConversationCommandAcceptance:
        if self._closed:
            raise RuntimeError("Workshop private-text execution service is closed")
        async with self._database_lock:
            return await self._command_service.accept_scheduled(message)

    async def execute(
        self,
        run_id: RunId,
        *,
        stream_observer: StreamObserver | None = None,
        success_transformer: SuccessTransformer | None = None,
    ) -> CanonicalExecutionResult:
        if self._closed:
            raise RuntimeError("Workshop private-text execution service is closed")
        return await self._coordinator.execute(
            run_id,
            stream_observer=stream_observer,
            success_transformer=success_transformer,
        )

    async def run_state(self, run_id: RunId):
        """Return canonical state for one typed run ID."""
        if self._closed:
            raise RuntimeError("Workshop private-text execution service is closed")
        async with self._database_lock:
            return await WorkshopRunLifecycle(self._store).state(run_id)

    async def prior_conversation_pairs(
        self,
        run_id: RunId,
        *,
        limit: int,
    ) -> tuple[tuple[str, str], ...]:
        """Return canonical completed exchanges for memory extraction."""
        if self._closed:
            raise RuntimeError("Workshop private-text execution service is closed")
        async with self._database_lock:
            run = await WorkshopRunLifecycle(self._store).state(run_id)
            return await assemble_canonical_prior_pairs(self._store, run, limit=limit)

    async def recoverable_client_runs(self) -> tuple[RecoverableClientRun, ...]:
        """Find browser runs that are durably accepted and safe to dispatch."""
        if self._closed:
            raise RuntimeError("Workshop private-text execution service is closed")
        async with (
            self._database_lock,
            self._store.connection.execute(
                "SELECT r.id, ra.runtime_profile_id, r.inbound_message_id, m.body "
                "FROM runs r "
                "JOIN messages m ON m.id = r.inbound_message_id "
                "JOIN event_log e ON e.position = m.created_event_position "
                "JOIN channel_agent_runtime_assignments ra "
                "ON ra.channel_id = r.channel_id AND ra.agent_id = r.agent_id "
                "WHERE r.status = 'accepted' AND r.cancellation_requested_at IS NULL "
                "AND json_extract(e.metadata_json, '$.source') = 'workshop_client' "
                "AND NOT EXISTS (SELECT 1 FROM run_attempts a WHERE a.run_id = r.id "
                "AND a.status IN ('granted', 'started')) "
                "ORDER BY r.accepted_at, r.id"
            ) as cursor,
        ):
            rows = list(await cursor.fetchall())
        return tuple(
            RecoverableClientRun(
                RunId(str(row[0])),
                RuntimeProfileId(str(row[1])),
                MessageId(str(row[2])),
                str(row[3]),
            )
            for row in rows
        )

    async def request_run_cancellation(
        self,
        run_id: RunId,
    ) -> CanonicalCancellationDisposition:
        """Cancel one canonical run after authorization at the caller boundary."""
        if self._closed:
            raise RuntimeError("Workshop private-text execution service is closed")
        return await self._coordinator.request_cancellation(run_id)

    async def request_cancellation(
        self,
        *,
        telegram_user_id: int,
        telegram_chat_id: int,
    ) -> CanonicalCancellationDisposition:
        """Resolve transport identity to a canonical run before cancellation."""
        if telegram_user_id <= 0 or telegram_chat_id <= 0 or telegram_user_id != telegram_chat_id:
            return CanonicalCancellationDisposition.NOT_ACTIVE
        async with (
            self._database_lock,
            self._store.connection.execute(
                "SELECT r.id FROM runs r "
                "JOIN channels c ON c.id = r.channel_id AND c.kind = 'direct' "
                "JOIN external_identities ei ON ei.principal_id = r.requested_by_principal_id "
                "AND ei.provider = 'telegram' AND ei.external_subject = ? "
                "JOIN channel_bindings cb ON cb.channel_id = r.channel_id "
                "AND cb.transport = 'telegram' AND cb.external_channel_id = ? "
                "WHERE r.status IN ('accepted', 'started') "
                "ORDER BY r.accepted_at DESC, r.id",
                (str(telegram_user_id), str(telegram_chat_id)),
            ) as cursor,
        ):
            rows = list(await cursor.fetchall())
        if not rows:
            return CanonicalCancellationDisposition.NOT_ACTIVE
        if len(rows) > 1:
            raise RuntimeError("Canonical private channel has multiple nonterminal runs")
        return await self._coordinator.request_cancellation(RunId(str(rows[0][0])))

    async def wait(self) -> None:
        task = self._task
        if task is None:
            raise RuntimeError("Workshop private-text recovery loop was not started")
        await asyncio.shield(task)

    async def stop(self) -> None:
        if self._closed:
            return
        self._stop_event.set()
        task = self._task
        try:
            if task is not None:
                await asyncio.shield(task)
        finally:
            self._closed = True
            self._task = None
            await self._store.close()

    async def _recovery_loop(self) -> None:
        while True:
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=_RECOVERY_INTERVAL_SECONDS)
                return
            except TimeoutError:
                await self._coordinator.recover_expired()
