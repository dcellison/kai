"""Lifecycle ownership for Workshop Telegram delivery."""

from __future__ import annotations

import asyncio
from enum import StrEnum
from pathlib import Path
from typing import Protocol, cast

from telegram import Bot

from kai.workshop.delivery_authority import WorkshopConversationDeliveryAuthority
from kai.workshop.delivery_fragments import WorkshopDeliveryFragments
from kai.workshop.delivery_outbox import (
    CONVERSATION_REPLY_PURPOSE,
    DeliveryRecoveryResult,
    WorkshopDeliveryOutbox,
)
from kai.workshop.store import WorkshopEventStore
from kai.workshop.telegram_delivery import (
    WORKSHOP_CLIENT_TEXT_MODE,
    TelegramTextBot,
    WorkshopTelegramDeliveryAdapter,
    WorkshopTelegramDeliveryWorker,
    WorkshopTelegramStreamingFinalizationAdapter,
    WorkshopTelegramStreamingFinalizationWorker,
)


class _DeliveryLeaseRecovery(Protocol):
    async def recover_expired_leases(self) -> DeliveryRecoveryResult: ...


class _TelegramDeliveryWorkerLoop(Protocol):
    async def run(self, stop_event: asyncio.Event) -> None: ...


class TelegramDeliveryRuntimeState(StrEnum):
    """Observable lifecycle state for the delivery runtime owner."""

    STOPPED = "stopped"
    STARTING = "starting"
    READY = "ready"
    STOPPING = "stopping"
    FAILED = "failed"


class TelegramDeliveryRuntimeStateError(RuntimeError):
    """The requested runtime transition is not valid in the current state."""


class TelegramDeliveryWorkerExitedError(RuntimeError):
    """The owned worker task exited without a shutdown request."""


class WorkshopTelegramDeliveryRuntime:
    """Own exactly one recoverable Telegram worker task when explicitly started.

    Production uses this owner for the conversation-finalization worker. The
    qualification worker remains operator-invoked and outside this runtime.
    """

    def __init__(
        self,
        recovery: _DeliveryLeaseRecovery,
        worker: _TelegramDeliveryWorkerLoop,
    ) -> None:
        self._recovery = recovery
        self._worker = worker
        self._lock = asyncio.Lock()
        self._state = TelegramDeliveryRuntimeState.STOPPED
        self._stop_event: asyncio.Event | None = None
        self._task: asyncio.Task[None] | None = None
        self._failure: BaseException | None = None
        self._recovery_result: DeliveryRecoveryResult | None = None

    @property
    def state(self) -> TelegramDeliveryRuntimeState:
        return self._state

    @property
    def ready(self) -> bool:
        return self._state == TelegramDeliveryRuntimeState.READY

    @property
    def has_worker_task(self) -> bool:
        """Whether this owner currently retains a worker task, running or done."""
        return self._task is not None

    @property
    def failure(self) -> BaseException | None:
        return self._failure

    @property
    def recovery_result(self) -> DeliveryRecoveryResult | None:
        return self._recovery_result

    async def start(self) -> DeliveryRecoveryResult:
        """Recover abandoned work, then create the single owned worker task."""
        async with self._lock:
            if self._state != TelegramDeliveryRuntimeState.STOPPED:
                raise TelegramDeliveryRuntimeStateError(
                    f"Telegram delivery runtime cannot start while {self._state.value}"
                )

            self._state = TelegramDeliveryRuntimeState.STARTING
            self._failure = None
            self._recovery_result = None
            try:
                recovered = await self._recovery.recover_expired_leases()
            except asyncio.CancelledError:
                self._state = TelegramDeliveryRuntimeState.STOPPED
                raise
            except Exception as error:
                self._failure = error
                self._state = TelegramDeliveryRuntimeState.FAILED
                raise

            stop_event = asyncio.Event()
            task = asyncio.create_task(
                self._worker.run(stop_event),
                name="kai-workshop-telegram-delivery",
            )
            self._stop_event = stop_event
            self._task = task
            self._recovery_result = recovered
            self._state = TelegramDeliveryRuntimeState.READY
            task.add_done_callback(self._worker_done)
            return recovered

    async def wait(self) -> None:
        """Wait for the worker and propagate any unexpected termination."""
        task = self._task
        if task is None:
            if self._failure is not None:
                raise self._failure
            raise TelegramDeliveryRuntimeStateError("Telegram delivery runtime has no worker task")

        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if not task.cancelled():
                raise
        except Exception:
            pass
        self._record_completion(task)
        if self._failure is not None:
            raise self._failure

    async def stop(self) -> None:
        """Cooperatively stop polling and await the current serialized iteration."""
        async with self._lock:
            if self._state == TelegramDeliveryRuntimeState.STOPPED:
                return

            task = self._task
            stop_event = self._stop_event
            if task is not None and task.done() and self._failure is None:
                # Preserve an exit that happened immediately before shutdown;
                # setting the event first would misclassify it as intentional.
                self._record_completion(task, stop_was_requested=False)
            self._state = TelegramDeliveryRuntimeState.STOPPING
            if stop_event is not None:
                stop_event.set()

        if task is not None:
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if not task.cancelled():
                    # Cancelling the stop caller must not cancel the worker.
                    raise
            except Exception:
                pass
            self._record_completion(task)

        async with self._lock:
            failure = self._failure
            if task is self._task:
                self._task = None
                self._stop_event = None
            self._state = TelegramDeliveryRuntimeState.STOPPED

        if failure is not None:
            raise failure

    def _worker_done(self, task: asyncio.Task[None]) -> None:
        self._record_completion(task)

    def _record_completion(
        self,
        task: asyncio.Task[None],
        *,
        stop_was_requested: bool | None = None,
    ) -> None:
        if task is not self._task or not task.done():
            return

        if stop_was_requested is None:
            stop_was_requested = self._stop_event is not None and self._stop_event.is_set()
        if task.cancelled():
            failure: BaseException | None = TelegramDeliveryWorkerExitedError(
                "Telegram delivery worker was cancelled outside its runtime owner"
            )
        else:
            failure = task.exception()
            if failure is None and not stop_was_requested:
                failure = TelegramDeliveryWorkerExitedError(
                    "Telegram delivery worker exited without a shutdown request"
                )

        if failure is not None:
            self._failure = failure
            self._state = TelegramDeliveryRuntimeState.FAILED


class WorkshopTelegramConversationDeliveryService:
    """Own the active authority epoch, dedicated store, and worker runtime."""

    def __init__(
        self,
        store: WorkshopEventStore,
        client_store: WorkshopEventStore,
        runtime: WorkshopTelegramDeliveryRuntime,
        client_runtime: WorkshopTelegramDeliveryRuntime,
    ) -> None:
        self._store = store
        self._client_store = client_store
        self._runtime = runtime
        self._client_runtime = client_runtime
        self._closed = False

    @classmethod
    async def open_and_start(
        cls,
        database_path: Path,
        bot: Bot,
        *,
        worker_id: str = "kai-telegram-conversation-finalization",
    ) -> WorkshopTelegramConversationDeliveryService:
        """Activate authority and recover work before returning ready."""
        store = await WorkshopEventStore.open(database_path)
        client_store: WorkshopEventStore | None = None
        client_runtime: WorkshopTelegramDeliveryRuntime | None = None
        try:
            client_store = await WorkshopEventStore.open(database_path)
            epoch = (await WorkshopConversationDeliveryAuthority(store).activate()).epoch
            outbox = WorkshopDeliveryOutbox(store)
            fragments = WorkshopDeliveryFragments(store)
            client_worker = WorkshopTelegramDeliveryWorker(
                WorkshopDeliveryOutbox(client_store),
                WorkshopDeliveryFragments(client_store),
                WorkshopTelegramDeliveryAdapter(cast(TelegramTextBot, bot)),
                worker_id=f"{worker_id}-client-message",
                purpose=CONVERSATION_REPLY_PURPOSE,
                modes=(WORKSHOP_CLIENT_TEXT_MODE,),
            )
            client_runtime = WorkshopTelegramDeliveryRuntime(client_worker, client_worker)
            await client_runtime.start()
            worker = WorkshopTelegramStreamingFinalizationWorker(
                outbox,
                fragments,
                WorkshopTelegramStreamingFinalizationAdapter(cast(TelegramTextBot, bot)),
                worker_id=worker_id,
                authority_epoch_id=epoch.epoch_id,
            )
            runtime = WorkshopTelegramDeliveryRuntime(worker, worker)
            await runtime.start()
        except BaseException:
            if client_runtime is not None:
                try:
                    await client_runtime.stop()
                except Exception:
                    pass
            if client_store is not None:
                await client_store.close()
            await store.close()
            raise
        assert client_store is not None
        assert client_runtime is not None
        return cls(store, client_store, runtime, client_runtime)

    @property
    def ready(self) -> bool:
        return not self._closed and self._runtime.ready and self._client_runtime.ready

    @property
    def runtime(self) -> WorkshopTelegramDeliveryRuntime:
        return self._runtime

    async def wait(self) -> None:
        waits = {
            asyncio.create_task(self._runtime.wait()),
            asyncio.create_task(self._client_runtime.wait()),
        }
        done, pending = await asyncio.wait(waits, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            await task

    async def stop(self) -> None:
        if self._closed:
            return
        try:
            results = await asyncio.gather(
                self._client_runtime.stop(),
                self._runtime.stop(),
                return_exceptions=True,
            )
        finally:
            self._closed = True
            await self._client_store.close()
            await self._store.close()
        for result in results:
            if isinstance(result, BaseException):
                raise result
