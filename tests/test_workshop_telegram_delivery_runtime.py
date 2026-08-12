"""Lifecycle contracts for the Workshop Telegram delivery owner."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from kai.workshop.delivery_authority import WorkshopConversationDeliveryAuthority
from kai.workshop.delivery_outbox import DeliveryRecoveryResult
from kai.workshop.store import WorkshopEventStore
from kai.workshop.telegram_delivery_runtime import (
    TelegramDeliveryRuntimeState,
    TelegramDeliveryRuntimeStateError,
    TelegramDeliveryWorkerExitedError,
    WorkshopTelegramConversationDeliveryService,
    WorkshopTelegramDeliveryRuntime,
)


class _ControlledRecovery:
    def __init__(self, *, result: DeliveryRecoveryResult | None = None) -> None:
        self.result = result or DeliveryRecoveryResult(requeued=0, failed=0)
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.release.set()
        self.calls = 0
        self.error: Exception | None = None

    async def recover_expired_leases(self) -> DeliveryRecoveryResult:
        self.calls += 1
        self.entered.set()
        await self.release.wait()
        if self.error is not None:
            raise self.error
        return self.result


class _ControlledWorker:
    def __init__(self, *, active: bool = False) -> None:
        self.active = active
        self.started = asyncio.Event()
        self.iteration_started = asyncio.Event()
        self.release_iteration = asyncio.Event()
        self.cancelled = False
        self.calls = 0

    async def run(self, stop_event: asyncio.Event) -> None:
        self.calls += 1
        self.started.set()
        try:
            if self.active:
                self.iteration_started.set()
                await self.release_iteration.wait()
            await stop_event.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class _FaultingWorker:
    def __init__(self, *, exit_normally: bool = False) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.exit_normally = exit_normally
        self.error = RuntimeError("delivery worker failed")

    async def run(self, stop_event: asyncio.Event) -> None:
        del stop_event
        self.started.set()
        await self.release.wait()
        if not self.exit_normally:
            raise self.error


async def test_owner_creates_no_task_until_explicitly_started():
    recovery = _ControlledRecovery()
    worker = _ControlledWorker()
    runtime = WorkshopTelegramDeliveryRuntime(recovery, worker)

    assert runtime.state == TelegramDeliveryRuntimeState.STOPPED
    assert not runtime.ready
    assert not runtime.has_worker_task
    assert runtime.failure is None
    assert runtime.recovery_result is None

    await runtime.stop()

    assert recovery.calls == 0
    assert worker.calls == 0
    assert not runtime.has_worker_task


async def test_start_recovers_before_ready_and_starts_exactly_one_worker():
    recovered = DeliveryRecoveryResult(requeued=2, failed=1)
    recovery = _ControlledRecovery(result=recovered)
    recovery.release.clear()
    worker = _ControlledWorker()
    runtime = WorkshopTelegramDeliveryRuntime(recovery, worker)

    start_task = asyncio.create_task(runtime.start())
    await recovery.entered.wait()
    assert runtime.state == TelegramDeliveryRuntimeState.STARTING
    assert not runtime.ready
    assert not runtime.has_worker_task
    assert worker.calls == 0

    recovery.release.set()
    assert await start_task == recovered
    assert runtime.ready
    assert runtime.recovery_result == recovered
    assert runtime.has_worker_task
    await worker.started.wait()
    assert worker.calls == 1

    with pytest.raises(TelegramDeliveryRuntimeStateError, match="cannot start while ready"):
        await runtime.start()

    await runtime.stop()
    assert runtime.state == TelegramDeliveryRuntimeState.STOPPED
    assert not runtime.has_worker_task


async def test_idle_shutdown_is_cooperative_and_repeatable():
    recovery = _ControlledRecovery()
    worker = _ControlledWorker()
    runtime = WorkshopTelegramDeliveryRuntime(recovery, worker)
    await runtime.start()
    await worker.started.wait()

    await asyncio.gather(runtime.stop(), runtime.stop())
    await runtime.stop()

    assert runtime.state == TelegramDeliveryRuntimeState.STOPPED
    assert not runtime.has_worker_task
    assert not worker.cancelled


async def test_active_shutdown_awaits_iteration_without_cancelling_worker():
    recovery = _ControlledRecovery()
    worker = _ControlledWorker(active=True)
    runtime = WorkshopTelegramDeliveryRuntime(recovery, worker)
    await runtime.start()
    await worker.iteration_started.wait()

    stop_task = asyncio.create_task(runtime.stop())
    await asyncio.sleep(0)
    assert runtime.state == TelegramDeliveryRuntimeState.STOPPING
    assert not stop_task.done()
    assert not worker.cancelled

    worker.release_iteration.set()
    await stop_task
    assert runtime.state == TelegramDeliveryRuntimeState.STOPPED
    assert not runtime.has_worker_task
    assert not worker.cancelled


async def test_worker_exception_is_visible_to_supervisor_and_stop():
    recovery = _ControlledRecovery()
    worker = _FaultingWorker()
    runtime = WorkshopTelegramDeliveryRuntime(recovery, worker)
    await runtime.start()
    await worker.started.wait()
    worker.release.set()

    with pytest.raises(RuntimeError, match="delivery worker failed") as wait_error:
        await runtime.wait()
    assert wait_error.value is worker.error
    assert runtime.state == TelegramDeliveryRuntimeState.FAILED
    assert runtime.failure is worker.error

    with pytest.raises(RuntimeError, match="delivery worker failed"):
        await runtime.stop()
    assert runtime.state == TelegramDeliveryRuntimeState.STOPPED
    assert not runtime.has_worker_task


async def test_normal_worker_exit_without_stop_is_a_visible_failure():
    recovery = _ControlledRecovery()
    worker = _FaultingWorker(exit_normally=True)
    runtime = WorkshopTelegramDeliveryRuntime(recovery, worker)
    await runtime.start()
    await worker.started.wait()
    worker.release.set()

    with pytest.raises(TelegramDeliveryWorkerExitedError, match="without a shutdown request"):
        await runtime.wait()
    assert runtime.state == TelegramDeliveryRuntimeState.FAILED

    with pytest.raises(TelegramDeliveryWorkerExitedError):
        await runtime.stop()


async def test_recovery_failure_prevents_readiness_and_worker_creation():
    recovery = _ControlledRecovery()
    recovery.error = RuntimeError("lease recovery failed")
    worker = _ControlledWorker()
    runtime = WorkshopTelegramDeliveryRuntime(recovery, worker)

    with pytest.raises(RuntimeError, match="lease recovery failed") as start_error:
        await runtime.start()
    assert start_error.value is recovery.error
    assert runtime.state == TelegramDeliveryRuntimeState.FAILED
    assert runtime.failure is recovery.error
    assert not runtime.ready
    assert not runtime.has_worker_task
    assert worker.calls == 0

    with pytest.raises(RuntimeError, match="lease recovery failed"):
        await runtime.stop()
    assert runtime.state == TelegramDeliveryRuntimeState.STOPPED


async def test_conversation_service_activates_once_and_reuses_epoch_after_restart(tmp_path: Path):
    database = tmp_path / "kai.db"
    bot = AsyncMock()

    first_service = await WorkshopTelegramConversationDeliveryService.open_and_start(database, bot)
    assert first_service.ready
    observer = await WorkshopEventStore.open(database)
    first_epoch = await WorkshopConversationDeliveryAuthority(observer).active_epoch()
    await observer.close()
    await first_service.stop()

    second_service = await WorkshopTelegramConversationDeliveryService.open_and_start(database, bot)
    assert second_service.ready
    observer = await WorkshopEventStore.open(database)
    second_epoch = await WorkshopConversationDeliveryAuthority(observer).active_epoch()
    await observer.close()
    await second_service.stop()

    assert second_epoch.epoch_id == first_epoch.epoch_id
