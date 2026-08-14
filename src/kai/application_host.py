"""Transport-neutral construction and lifecycle for Kai core services."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from kai import sessions
from kai.config import Config
from kai.pool import SubprocessPool
from kai.workshop.client_commands import WorkshopClientCommandExecutor
from kai.workshop.compatibility_state import WorkshopCompatibilityStateWriter
from kai.workshop.conversation_runs import WorkshopConversationRunService
from kai.workshop.delivery_authority import (
    DeliveryAuthorityEpoch,
    WorkshopConversationDeliveryAuthority,
)
from kai.workshop.private_text_execution import WorkshopPrivateTextExecutionService
from kai.workshop.runtime_pool import WorkshopRuntimePool
from kai.workshop.runtime_profiles import WorkshopRuntimeProfileRegistry
from kai.workshop.storage_namespaces import WorkshopPrincipalStorageRegistry
from kai.workshop.store import WorkshopEventStore


class KaiApplicationState(StrEnum):
    """Observable lifecycle state for the transport-neutral application host."""

    NEW = "new"
    STARTING = "starting"
    READY = "ready"
    DRAINING = "draining"
    STOPPED = "stopped"
    FAILED = "failed"


class KaiAdapterReadiness(Protocol):
    """JSON-safe readiness contract for one external adapter."""

    def as_dict(self) -> dict[str, object]: ...


class KaiApplicationAdapter(Protocol):
    """Lifecycle contract for one configured external adapter."""

    @property
    def readiness(self) -> KaiAdapterReadiness: ...

    async def start(self) -> None: ...

    async def wait(self) -> None: ...

    async def stop(self) -> None: ...


@dataclass(frozen=True, slots=True)
class KaiCoreReadiness:
    """Non-sensitive component readiness exposed to operators and adapters."""

    state: KaiApplicationState
    runtime: bool
    executor: bool
    client_api: bool
    store: bool

    @property
    def ready(self) -> bool:
        return self.state == KaiApplicationState.READY and all(
            (self.runtime, self.executor, self.client_api, self.store)
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.state.value,
            "ready": self.ready,
            "components": {
                "runtime": self.runtime,
                "executor": self.executor,
                "client_api": self.client_api,
                "store": self.store,
            },
        }


@dataclass(frozen=True, slots=True)
class KaiCoreServices:
    """Typed dependencies shared by client and transport adapters."""

    subprocess_pool: SubprocessPool
    runtime_profiles: WorkshopRuntimeProfileRegistry
    runtime_pool: WorkshopRuntimePool
    conversation_runs: WorkshopConversationRunService
    private_text_execution: WorkshopPrivateTextExecutionService
    client_commands: WorkshopClientCommandExecutor
    client_store: WorkshopEventStore
    principal_storage: WorkshopPrincipalStorageRegistry
    delivery_authority_epoch: DeliveryAuthorityEpoch


class KaiApplicationHost:
    """Own core service construction, supervision, readiness, and shutdown.

    This module deliberately imports no Telegram package and accepts no
    Telegram object. Adapters receive :attr:`services` after startup and cannot
    become the owner of runtime or Workshop execution lifecycle.
    """

    def __init__(
        self,
        *,
        config: Config,
        runtime_profiles: WorkshopRuntimeProfileRegistry,
        principal_storage: WorkshopPrincipalStorageRegistry,
        services_info: list[dict],
        registered_backend_ids: frozenset[str],
    ) -> None:
        self._config = config
        self._runtime_profiles = runtime_profiles
        self._principal_storage = principal_storage
        self._services_info = services_info
        self._registered_backend_ids = registered_backend_ids
        self._state = KaiApplicationState.NEW
        self._services: KaiCoreServices | None = None
        self._adapters: dict[str, KaiApplicationAdapter] = {}

    @property
    def services(self) -> KaiCoreServices:
        services = self._services
        if services is None:
            raise RuntimeError("Kai core services are not started")
        return services

    @property
    def readiness(self) -> KaiCoreReadiness:
        services = self._services
        return KaiCoreReadiness(
            state=self._state,
            runtime=services is not None and self._state == KaiApplicationState.READY,
            executor=services is not None and services.private_text_execution.ready,
            client_api=services is not None and services.client_commands.ready,
            store=services is not None,
        )

    @property
    def adapter_readiness(self) -> dict[str, object]:
        """Return non-sensitive readiness reported by attached adapters."""
        return {name: adapter.readiness.as_dict() for name, adapter in self._adapters.items()}

    async def start(self) -> KaiCoreServices:
        if self._state != KaiApplicationState.NEW:
            raise RuntimeError(f"Kai application host cannot start from {self._state.value}")
        self._state = KaiApplicationState.STARTING

        subprocess_pool: SubprocessPool | None = None
        private_execution: WorkshopPrivateTextExecutionService | None = None
        client_store: WorkshopEventStore | None = None
        client_commands: WorkshopClientCommandExecutor | None = None
        try:
            subprocess_pool = SubprocessPool(
                config=self._config,
                services_info=self._services_info,
                runtime_profiles=self._runtime_profiles,
            )
            runtime_pool = WorkshopRuntimePool(subprocess_pool, self._runtime_profiles)
            conversation_runs = WorkshopConversationRunService(
                runtime_pool,
                sessions.resolve_workshop_conversation_run,
            )

            subprocess_pool.start()
            # Terminal run recovery atomically creates delivery work stamped
            # with the active authority epoch. Resume or create that durable,
            # transport-neutral authority before either recovery owner starts.
            client_store = await WorkshopEventStore.open(Path(self._config.session_db_path))
            delivery_authority_epoch = (await WorkshopConversationDeliveryAuthority(client_store).activate()).epoch
            private_execution = await WorkshopPrivateTextExecutionService.open_and_start(
                Path(self._config.session_db_path),
                runtime_pool,
                registered_backend_ids=self._registered_backend_ids,
            )
            client_commands = WorkshopClientCommandExecutor(
                private_execution,
                WorkshopCompatibilityStateWriter(self._config, runtime_pool),
            )
            await client_commands.start()

            self._services = KaiCoreServices(
                subprocess_pool=subprocess_pool,
                runtime_profiles=self._runtime_profiles,
                runtime_pool=runtime_pool,
                conversation_runs=conversation_runs,
                private_text_execution=private_execution,
                client_commands=client_commands,
                client_store=client_store,
                principal_storage=self._principal_storage,
                delivery_authority_epoch=delivery_authority_epoch,
            )
            self._state = KaiApplicationState.READY
            return self._services
        except BaseException:
            self._state = KaiApplicationState.FAILED
            if client_commands is not None:
                await client_commands.stop()
            if client_store is not None:
                await client_store.close()
            if private_execution is not None:
                await private_execution.stop()
            if subprocess_pool is not None:
                await subprocess_pool.shutdown()
            raise

    async def attach_adapter(self, name: str, adapter: KaiApplicationAdapter) -> None:
        """Start and supervise an adapter after the core is ready."""
        if self._state != KaiApplicationState.READY:
            raise RuntimeError(f"Kai adapter cannot start while core is {self._state.value}")
        if not name:
            raise RuntimeError("Kai adapter name cannot be empty")
        if name in self._adapters:
            raise RuntimeError(f"Kai adapter {name!r} is already attached")
        try:
            await adapter.start()
        except BaseException:
            self._state = KaiApplicationState.FAILED
            try:
                await adapter.stop()
            except Exception:
                pass
            raise
        self._adapters[name] = adapter

    async def wait(self) -> None:
        """Expose failure of a required supervised core worker."""
        await asyncio.gather(
            self.services.private_text_execution.wait(),
            *(adapter.wait() for adapter in self._adapters.values()),
        )

    async def stop(self) -> None:
        if self._state in {KaiApplicationState.NEW, KaiApplicationState.STOPPED}:
            self._state = KaiApplicationState.STOPPED
            return
        services = self._services
        self._state = KaiApplicationState.DRAINING
        if services is None:
            self._state = KaiApplicationState.STOPPED
            return

        errors: list[Exception] = []
        for adapter in reversed(tuple(self._adapters.values())):
            try:
                await adapter.stop()
            except Exception as exc:
                errors.append(exc)
        self._adapters.clear()
        for operation in (
            services.client_commands.stop,
            services.client_store.close,
            services.private_text_execution.stop,
            services.subprocess_pool.shutdown,
        ):
            try:
                await operation()
            except Exception as exc:
                errors.append(exc)
        self._services = None
        self._state = KaiApplicationState.STOPPED
        if errors:
            raise ExceptionGroup("Kai core shutdown failed", errors)
