"""Production owner for authenticated private-text Workshop execution."""

from __future__ import annotations

import asyncio
from pathlib import Path

from kai.pool import SubprocessPool
from kai.workshop.conversation_commands import (
    ClientConversationCommandAcceptance,
    ConversationCommandAcceptance,
    WorkshopConversationCommandService,
)
from kai.workshop.domain import RunId
from kai.workshop.execution_coordinator import (
    CanonicalCancellationDisposition,
    CanonicalExecutionResult,
    StreamObserver,
    WorkshopCanonicalExecutionCoordinator,
)
from kai.workshop.inbound import ClientInboundMessage, InboundMessage
from kai.workshop.protected_execution import WorkshopProtectedExecutionPreparationService
from kai.workshop.store import WorkshopEventStore

_RECOVERY_INTERVAL_SECONDS = 5.0


class WorkshopPrivateTextExecutionService:
    """Own one store, coordinator, and recovery loop for private text."""

    def __init__(
        self,
        store: WorkshopEventStore,
        coordinator: WorkshopCanonicalExecutionCoordinator,
        command_service: WorkshopConversationCommandService,
        database_lock: asyncio.Lock,
    ) -> None:
        self._store = store
        self._coordinator = coordinator
        self._command_service = command_service
        self._database_lock = database_lock
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._closed = False

    @classmethod
    async def open_and_start(
        cls,
        database_path: Path,
        pool: SubprocessPool,
        *,
        registered_backend_ids: frozenset[str],
    ) -> WorkshopPrivateTextExecutionService:
        store = await WorkshopEventStore.open(database_path)
        database_lock = asyncio.Lock()
        coordinator = WorkshopCanonicalExecutionCoordinator(
            store,
            WorkshopProtectedExecutionPreparationService(
                store,
                pool,
                registered_backend_ids=registered_backend_ids,
            ),
            registered_backend_ids=registered_backend_ids,
            database_lock=database_lock,
        )
        service = cls(store, coordinator, WorkshopConversationCommandService(store), database_lock)
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

    async def accept(self, message: InboundMessage) -> ConversationCommandAcceptance:
        if self._closed:
            raise RuntimeError("Workshop private-text execution service is closed")
        async with self._database_lock:
            return await self._command_service.accept(message)

    async def accept_client(
        self,
        message: ClientInboundMessage,
    ) -> ClientConversationCommandAcceptance:
        if self._closed:
            raise RuntimeError("Workshop private-text execution service is closed")
        async with self._database_lock:
            return await self._command_service.accept_client(message)

    async def execute(
        self,
        run_id: RunId,
        *,
        stream_observer: StreamObserver | None = None,
    ) -> CanonicalExecutionResult:
        if self._closed:
            raise RuntimeError("Workshop private-text execution service is closed")
        return await self._coordinator.execute(run_id, stream_observer=stream_observer)

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
