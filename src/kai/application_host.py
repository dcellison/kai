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
from kai.workshop.artifacts import WorkshopArtifactService
from kai.workshop.client_commands import WorkshopClientCommandExecutor
from kai.workshop.client_preferences import (
    ClientVoiceCapability,
    WorkshopClientPreferenceService,
)
from kai.workshop.conversation_runs import WorkshopConversationRunService
from kai.workshop.delivery_authority import (
    DeliveryAuthorityEpoch,
    WorkshopConversationDeliveryAuthority,
)
from kai.workshop.delivery_policy import WorkshopDeliveryBindingPolicy
from kai.workshop.execution_state import WorkshopExecutionStateRegistry
from kai.workshop.github_automation import WorkshopGitHubAutomationService
from kai.workshop.github_settings import WorkshopGitHubSettingsService
from kai.workshop.integration_notifications import WorkshopIntegrationNotificationService
from kai.workshop.internal_api_contexts import WorkshopInternalAPIContextRegistry
from kai.workshop.memory_queries import WorkshopMemoryQueryService
from kai.workshop.notification_preferences import WorkshopNotificationPreferenceService
from kai.workshop.post_run_effects import WorkshopPostRunEffectService
from kai.workshop.preferences import WorkshopPreferenceService
from kai.workshop.private_text_execution import WorkshopPrivateTextExecutionService
from kai.workshop.proactive_publication import (
    ProactivePublicationAuthority,
    WorkshopProactivePublicationService,
)
from kai.workshop.run_previews import WorkshopRunPreviewRegistry
from kai.workshop.runtime_pool import WorkshopRuntimePool
from kai.workshop.runtime_profiles import WorkshopRuntimeProfileRegistry
from kai.workshop.runtime_state import WorkshopRuntimeStateWriter
from kai.workshop.scheduler import WorkshopCanonicalScheduler
from kai.workshop.settings_workspaces import WorkshopSettingsWorkspaceService
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
    scheduler: bool
    github_automation: bool
    post_run_effects: bool

    @property
    def ready(self) -> bool:
        return self.state == KaiApplicationState.READY and all(
            (
                self.runtime,
                self.executor,
                self.client_api,
                self.store,
                self.scheduler,
                self.github_automation,
                self.post_run_effects,
            )
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
                "scheduler": self.scheduler,
                "github_automation": self.github_automation,
                "post_run_effects": self.post_run_effects,
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
    internal_api_contexts: WorkshopInternalAPIContextRegistry
    delivery_authority_epoch: DeliveryAuthorityEpoch
    run_previews: WorkshopRunPreviewRegistry
    scheduler: WorkshopCanonicalScheduler
    artifacts: WorkshopArtifactService
    settings_workspaces: WorkshopSettingsWorkspaceService
    memory_queries: WorkshopMemoryQueryService
    preference_documents: WorkshopPreferenceService
    github_settings: WorkshopGitHubSettingsService
    notification_preferences: WorkshopNotificationPreferenceService
    client_preferences: WorkshopClientPreferenceService
    proactive_publication: WorkshopProactivePublicationService
    integration_notifications: WorkshopIntegrationNotificationService
    github_automation: WorkshopGitHubAutomationService
    post_run_effects: WorkshopPostRunEffectService
    delivery_policy: WorkshopDeliveryBindingPolicy


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
        execution_state: WorkshopExecutionStateRegistry,
        principal_storage: WorkshopPrincipalStorageRegistry,
        internal_api_contexts: WorkshopInternalAPIContextRegistry,
        services_info: list[dict],
        registered_backend_ids: frozenset[str],
        delivery_policy: WorkshopDeliveryBindingPolicy,
        client_voice_capabilities: tuple[ClientVoiceCapability, ...] = (),
    ) -> None:
        self._config = config
        self._runtime_profiles = runtime_profiles
        self._execution_state = execution_state
        self._principal_storage = principal_storage
        self._internal_api_contexts = internal_api_contexts
        self._services_info = services_info
        self._registered_backend_ids = registered_backend_ids
        self._delivery_policy = delivery_policy
        self._client_voice_capabilities = client_voice_capabilities
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
            scheduler=services is not None and services.scheduler.readiness.ready,
            github_automation=services is not None and services.github_automation.ready,
            post_run_effects=services is not None and services.post_run_effects.ready,
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
        scheduler: WorkshopCanonicalScheduler | None = None
        integration_notifications: WorkshopIntegrationNotificationService | None = None
        github_automation: WorkshopGitHubAutomationService | None = None
        post_run_effects: WorkshopPostRunEffectService | None = None
        github_settings: WorkshopGitHubSettingsService | None = None
        notification_preferences: WorkshopNotificationPreferenceService | None = None
        client_preferences: WorkshopClientPreferenceService | None = None
        try:
            delivery_policy = self._delivery_policy
            subprocess_pool = SubprocessPool(
                config=self._config,
                services_info=self._services_info,
                runtime_profiles=self._runtime_profiles,
                internal_api_contexts=self._internal_api_contexts,
            )
            runtime_pool = WorkshopRuntimePool(subprocess_pool, self._runtime_profiles)
            await subprocess_pool.hydrate_backend_selections()
            runtime_state = WorkshopRuntimeStateWriter(
                self._config,
                runtime_pool,
                self._execution_state,
            )
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
                delivery_policy=delivery_policy,
            )
            post_run_effects = await WorkshopPostRunEffectService.open_and_start(
                Path(self._config.session_db_path),
                runtime_state,
            )
            run_previews = WorkshopRunPreviewRegistry()
            client_commands = WorkshopClientCommandExecutor(
                private_execution,
                run_previews=run_previews,
            )
            await client_commands.start()
            scheduler = await WorkshopCanonicalScheduler.open_and_start(
                Path(self._config.session_db_path),
                private_execution,
                delivery_policy,
            )
            artifacts = WorkshopArtifactService(
                client_store,
                data_dir=Path(self._config.session_db_path).parent,
                principal_storage=self._principal_storage,
                runtime_profiles=self._runtime_profiles,
            )
            settings_workspaces = WorkshopSettingsWorkspaceService(
                self._config,
                runtime_pool,
                self._execution_state,
            )
            memory_queries = WorkshopMemoryQueryService(
                self._config,
                client_store,
                runtime_pool,
                self._execution_state,
            )
            preference_documents = WorkshopPreferenceService(
                Path(self._config.session_db_path).parent,
                self._principal_storage,
            )
            github_settings = await WorkshopGitHubSettingsService.open(
                Path(self._config.session_db_path),
                self._execution_state,
                self._runtime_profiles,
            )
            notification_preferences = await WorkshopNotificationPreferenceService.open(
                Path(self._config.session_db_path),
                self._execution_state,
            )
            client_preferences = await WorkshopClientPreferenceService.open(
                Path(self._config.session_db_path),
                self._client_voice_capabilities,
            )
            proactive_publication = WorkshopProactivePublicationService(
                client_store,
                artifacts,
                artifact_storage_root=Path(self._config.session_db_path).parent / "files",
                delivery_policy=delivery_policy,
            )
            for context in self._internal_api_contexts.contexts:
                await proactive_publication.validate_authority(
                    ProactivePublicationAuthority(
                        principal_id=context.principal_id,
                        channel_id=context.channel_id,
                        agent_id=context.agent_id,
                        runtime_profile_id=context.runtime_profile_id,
                    )
                )
            integration_notifications = await WorkshopIntegrationNotificationService.open(
                Path(self._config.session_db_path),
                delivery_policy,
                notification_preferences,
            )
            github_automation = await WorkshopGitHubAutomationService.open_and_start(
                Path(self._config.session_db_path),
                runtime_pool,
                self._execution_state,
                runtime_state,
                integration_notifications,
                notification_preferences,
                spec_dir=self._config.spec_dir,
                review_timeout_seconds=self._config.pr_review_timeout_s,
            )

            self._services = KaiCoreServices(
                subprocess_pool=subprocess_pool,
                runtime_profiles=self._runtime_profiles,
                runtime_pool=runtime_pool,
                conversation_runs=conversation_runs,
                private_text_execution=private_execution,
                client_commands=client_commands,
                client_store=client_store,
                principal_storage=self._principal_storage,
                internal_api_contexts=self._internal_api_contexts,
                delivery_authority_epoch=delivery_authority_epoch,
                run_previews=run_previews,
                scheduler=scheduler,
                artifacts=artifacts,
                settings_workspaces=settings_workspaces,
                memory_queries=memory_queries,
                preference_documents=preference_documents,
                github_settings=github_settings,
                notification_preferences=notification_preferences,
                client_preferences=client_preferences,
                proactive_publication=proactive_publication,
                integration_notifications=integration_notifications,
                github_automation=github_automation,
                post_run_effects=post_run_effects,
                delivery_policy=delivery_policy,
            )
            self._state = KaiApplicationState.READY
            return self._services
        except BaseException:
            self._state = KaiApplicationState.FAILED
            if scheduler is not None:
                await scheduler.stop()
            if github_automation is not None:
                await github_automation.stop()
            if integration_notifications is not None:
                await integration_notifications.close()
            if github_settings is not None:
                await github_settings.close()
            if notification_preferences is not None:
                await notification_preferences.close()
            if client_preferences is not None:
                await client_preferences.close()
            if client_commands is not None:
                await client_commands.stop()
            if post_run_effects is not None:
                await post_run_effects.stop()
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
            self.services.scheduler.wait(),
            self.services.github_automation.wait(),
            self.services.post_run_effects.wait(),
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
            services.scheduler.stop,
            services.github_automation.stop,
            services.integration_notifications.close,
            services.github_settings.close,
            services.notification_preferences.close,
            services.client_preferences.close,
            services.client_commands.stop,
            services.post_run_effects.stop,
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
