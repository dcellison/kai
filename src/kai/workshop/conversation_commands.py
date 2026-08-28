"""Atomic acceptance for canonical Workshop commands."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path

from kai.workshop.artifacts import StagedArtifact, record_inbound_artifact_in_transaction
from kai.workshop.delivery_outbox import CONVERSATION_REPLY_PURPOSE, DeliveryRequestResult
from kai.workshop.delivery_planning import CanonicalDeliveryIntent, WorkshopDeliveryPlanner
from kai.workshop.delivery_policy import WorkshopDeliveryBindingPolicy
from kai.workshop.domain import ChannelId, MessageId, RuntimeProfileId
from kai.workshop.inbound import (
    ClientInboundMessage,
    InboundMessage,
    ScheduledInboundMessage,
    record_client_inbound_message_in_transaction,
    record_inbound_message_in_transaction,
    record_scheduled_inbound_message_in_transaction,
)
from kai.workshop.run_lifecycle import DurableRun, RunLifecycleResult, RunStatus, WorkshopRunLifecycle
from kai.workshop.runtime_assignments import (
    WorkshopRuntimeAssignmentError,
    resolve_channel_runtime_profile,
)
from kai.workshop.store import AppendResult, WorkshopEventStore
from kai.workshop.wake_policy import resolve_message_wake_targets


class ConversationCommandAcceptanceError(RuntimeError):
    """Base error for an invalid atomic conversation-command acceptance."""


class ConversationCommandStateConflictError(ConversationCommandAcceptanceError):
    """Canonical message and run acceptance do not share one prior state."""


class ConversationCommandDisposition(StrEnum):
    """Durable replay decision returned before any backend preparation."""

    NEWLY_ACCEPTED = "newly_accepted"
    READY_REPLAY = "ready_replay"
    ACTIVE_REPLAY = "active_replay"
    CANCELLATION_PENDING_REPLAY = "cancellation_pending_replay"
    TERMINAL_REPLAY = "terminal_replay"
    MESSAGE_ONLY = "message_only"


@dataclass(frozen=True, slots=True)
class ConversationCommandAcceptance:
    message: AppendResult
    lifecycles: tuple[RunLifecycleResult, ...]
    run_dispositions: tuple[ConversationCommandDisposition, ...]
    disposition: ConversationCommandDisposition

    @property
    def lifecycle(self) -> RunLifecycleResult:
        if len(self.lifecycles) != 1:
            raise ConversationCommandStateConflictError("Command did not accept exactly one run")
        return self.lifecycles[0]

    @property
    def runs(self) -> tuple[DurableRun, ...]:
        return tuple(item.run for item in self.lifecycles)

    @property
    def run(self) -> DurableRun:
        return self.lifecycle.run


@dataclass(frozen=True, slots=True)
class ClientConversationCommandAcceptance:
    """Atomic client acceptance plus independently durable adapter work."""

    command: ConversationCommandAcceptance
    deliveries: tuple[DeliveryRequestResult, ...]
    runtime_profile_ids: tuple[RuntimeProfileId, ...]

    @property
    def delivery(self) -> DeliveryRequestResult | None:
        """Compatibility view for callers that expect at most one adapter."""
        return self.deliveries[0] if len(self.deliveries) == 1 else None

    @property
    def run(self) -> DurableRun:
        return self.command.run

    @property
    def runtime_profile_id(self) -> RuntimeProfileId:
        if len(self.runtime_profile_ids) != 1:
            raise ConversationCommandStateConflictError("Command did not resolve exactly one runtime profile")
        return self.runtime_profile_ids[0]


class WorkshopConversationCommandService:
    """Atomically append canonical inbound and run-acceptance facts.

    This service starts no process. Its disposition is the durable input to
    the execution coordinator owned by the production private-text runtime.
    """

    def __init__(
        self,
        store: WorkshopEventStore,
        *,
        artifact_storage_root: Path | None = None,
        delivery_policy: WorkshopDeliveryBindingPolicy | None = None,
    ) -> None:
        self._store = store
        self._artifact_storage_root = artifact_storage_root
        effective_policy = delivery_policy or WorkshopDeliveryBindingPolicy.disabled()
        self._delivery_planner = WorkshopDeliveryPlanner(self._store, effective_policy)

    async def accept(
        self,
        message: InboundMessage,
        *,
        artifact: StagedArtifact | None = None,
    ) -> ConversationCommandAcceptance:
        if not isinstance(message, InboundMessage):
            raise ValueError("message must be an InboundMessage")
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            inbound = await record_inbound_message_in_transaction(self._store, message)
            inbound_message_id = inbound.event.envelope.aggregate_id
            if not isinstance(inbound_message_id, MessageId):
                raise ConversationCommandStateConflictError("Canonical inbound event did not identify a message")
            artifact_result = None
            if artifact is not None:
                if self._artifact_storage_root is None:
                    raise ConversationCommandStateConflictError("Canonical artifact storage is unavailable")
                artifact_result = await record_inbound_artifact_in_transaction(
                    self._store,
                    artifact.for_message(inbound_message_id),
                    storage_root=self._artifact_storage_root,
                )
            lifecycles = await self._accept_woken_runs(
                inbound_message_id,
                occurred_at=inbound.event.envelope.occurred_at,
            )
            prior_states = {inbound.inserted, *(item.changed for item in lifecycles)}
            if artifact_result is not None:
                prior_states.add(artifact_result.inserted)
            if len(prior_states) != 1:
                raise ConversationCommandStateConflictError(
                    "Canonical inbound, artifact, and run acceptance did not share one prior state"
                )
            run_dispositions = await self._run_dispositions(inbound.inserted, lifecycles)
            disposition = self._aggregate_disposition(run_dispositions)
            await connection.commit()
            return ConversationCommandAcceptance(
                message=inbound,
                lifecycles=lifecycles,
                run_dispositions=run_dispositions,
                disposition=disposition,
            )
        except Exception:
            await connection.rollback()
            raise

    async def accept_client(
        self,
        message: ClientInboundMessage,
        *,
        artifact: StagedArtifact | None = None,
    ) -> ClientConversationCommandAcceptance:
        """Accept one canonical browser command and optional adapter echoes."""
        if not isinstance(message, ClientInboundMessage):
            raise ValueError("message must be a ClientInboundMessage")
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            inbound = await record_client_inbound_message_in_transaction(self._store, message)
            inbound_message_id = inbound.event.envelope.aggregate_id
            if not isinstance(inbound_message_id, MessageId):
                raise ConversationCommandStateConflictError("Canonical inbound event did not identify a message")
            artifact_result = None
            if artifact is not None:
                if self._artifact_storage_root is None:
                    raise ConversationCommandStateConflictError("Canonical artifact storage is unavailable")
                if message.artifact_source_unique_id != artifact.source_unique_id:
                    raise ConversationCommandStateConflictError("Canonical message and artifact identity disagree")
                artifact_result = await record_inbound_artifact_in_transaction(
                    self._store,
                    artifact.for_message(inbound_message_id),
                    storage_root=self._artifact_storage_root,
                )
            lifecycles = await self._accept_woken_runs(
                inbound_message_id,
                occurred_at=inbound.event.envelope.occurred_at,
            )
            runtime_profile_ids = await self._resolve_runtime_profiles(message.channel_id, lifecycles)
            planning = await self._delivery_planner.plan_in_transaction(
                CanonicalDeliveryIntent(
                    message_id=inbound_message_id,
                    channel_id=message.channel_id,
                    mode="workshop_client_text",
                    purpose=CONVERSATION_REPLY_PURPOSE,
                    occurred_at=inbound.event.envelope.occurred_at,
                    recipient_principal_id=message.principal_id,
                )
            )
            prior_states = {inbound.inserted, *(item.changed for item in lifecycles)}
            if artifact_result is not None:
                prior_states.add(artifact_result.inserted)
            prior_states.update(delivery.inserted for delivery in planning.deliveries)
            if len(prior_states) != 1:
                raise ConversationCommandStateConflictError(
                    "Canonical inbound, run acceptance, and delivery request did not share one prior state"
                )
            run_dispositions = await self._run_dispositions(inbound.inserted, lifecycles)
            disposition = self._aggregate_disposition(run_dispositions)
            await connection.commit()
            return ClientConversationCommandAcceptance(
                command=ConversationCommandAcceptance(
                    inbound,
                    lifecycles,
                    run_dispositions,
                    disposition,
                ),
                deliveries=planning.deliveries,
                runtime_profile_ids=runtime_profile_ids,
            )
        except Exception:
            await connection.rollback()
            raise

    async def accept_scheduled(
        self,
        message: ScheduledInboundMessage,
    ) -> ClientConversationCommandAcceptance:
        """Accept core-owned scheduled work without creating an inbound echo."""
        if not isinstance(message, ScheduledInboundMessage):
            raise ValueError("message must be a ScheduledInboundMessage")
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            inbound = await record_scheduled_inbound_message_in_transaction(self._store, message)
            inbound_message_id = inbound.event.envelope.aggregate_id
            if not isinstance(inbound_message_id, MessageId):
                raise ConversationCommandStateConflictError("Scheduled inbound event did not identify a message")
            lifecycle = await WorkshopRunLifecycle(self._store).accept_in_transaction(
                inbound_message_id,
                occurred_at=inbound.event.envelope.occurred_at,
            )
            try:
                _, runtime_profile_id = await resolve_channel_runtime_profile(
                    self._store,
                    message.channel_id,
                )
            except WorkshopRuntimeAssignmentError as exc:
                raise ConversationCommandStateConflictError(str(exc)) from exc
            if inbound.inserted != lifecycle.changed:
                raise ConversationCommandStateConflictError(
                    "Scheduled inbound and run acceptance did not share one prior state"
                )
            disposition = (
                ConversationCommandDisposition.NEWLY_ACCEPTED
                if inbound.inserted
                else await self._replay_disposition(lifecycle.run)
            )
            await connection.commit()
            return ClientConversationCommandAcceptance(
                command=ConversationCommandAcceptance(
                    inbound,
                    (lifecycle,),
                    (disposition,),
                    disposition,
                ),
                deliveries=(),
                runtime_profile_ids=(runtime_profile_id,),
            )
        except Exception:
            await connection.rollback()
            raise

    async def _replay_disposition(self, run: DurableRun) -> ConversationCommandDisposition:
        if run.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
            return ConversationCommandDisposition.TERMINAL_REPLAY
        if run.cancellation_requested_at is not None:
            return ConversationCommandDisposition.CANCELLATION_PENDING_REPLAY
        async with self._store.connection.execute(
            "SELECT COUNT(*) FROM run_attempts WHERE run_id = ? AND status IN ('granted', 'started')",
            (run.run_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise ConversationCommandStateConflictError("Active-attempt query returned no row")
        active_attempts = int(row[0])
        if active_attempts > 1:
            raise ConversationCommandStateConflictError("Durable run has multiple active execution attempts")
        if active_attempts == 1:
            return ConversationCommandDisposition.ACTIVE_REPLAY
        if run.status == RunStatus.STARTED:
            raise ConversationCommandStateConflictError("Started run has no active execution attempt")
        if run.status != RunStatus.ACCEPTED:
            raise ConversationCommandStateConflictError("Nonterminal run has an unsupported durable state")
        return ConversationCommandDisposition.READY_REPLAY

    async def _accept_woken_runs(
        self,
        message_id: MessageId,
        *,
        occurred_at: datetime,
    ) -> tuple[RunLifecycleResult, ...]:
        decision = await resolve_message_wake_targets(
            self._store,
            message_id,
            scope=None,
        )
        lifecycle = WorkshopRunLifecycle(self._store)
        accepted: list[RunLifecycleResult] = []
        for agent_id in decision.agent_ids:
            accepted.append(
                await lifecycle.accept_in_transaction(
                    message_id,
                    occurred_at=occurred_at,
                    agent_id=agent_id,
                )
            )
        return tuple(accepted)

    async def _resolve_runtime_profiles(
        self,
        channel_id: ChannelId,
        lifecycles: tuple[RunLifecycleResult, ...],
    ) -> tuple[RuntimeProfileId, ...]:
        profiles: list[RuntimeProfileId] = []
        for lifecycle in lifecycles:
            try:
                _, profile_id = await resolve_channel_runtime_profile(
                    self._store,
                    channel_id,
                    lifecycle.run.agent_id,
                )
            except WorkshopRuntimeAssignmentError as exc:
                raise ConversationCommandStateConflictError(str(exc)) from exc
            profiles.append(profile_id)
        return tuple(profiles)

    async def _run_dispositions(
        self,
        inserted: bool,
        lifecycles: tuple[RunLifecycleResult, ...],
    ) -> tuple[ConversationCommandDisposition, ...]:
        if not lifecycles:
            return ()
        if inserted:
            return (ConversationCommandDisposition.NEWLY_ACCEPTED,) * len(lifecycles)
        return tuple([await self._replay_disposition(item.run) for item in lifecycles])

    @staticmethod
    def _aggregate_disposition(
        dispositions: tuple[ConversationCommandDisposition, ...],
    ) -> ConversationCommandDisposition:
        if not dispositions:
            return ConversationCommandDisposition.MESSAGE_ONLY
        if len(set(dispositions)) == 1:
            return dispositions[0]
        for candidate in (
            ConversationCommandDisposition.CANCELLATION_PENDING_REPLAY,
            ConversationCommandDisposition.ACTIVE_REPLAY,
            ConversationCommandDisposition.READY_REPLAY,
            ConversationCommandDisposition.TERMINAL_REPLAY,
        ):
            if candidate in dispositions:
                return candidate
        raise ConversationCommandStateConflictError("Woken runs have unsupported replay states")
