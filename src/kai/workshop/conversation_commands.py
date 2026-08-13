"""Atomic acceptance for canonical Workshop commands."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from kai.workshop.delivery_outbox import (
    CONVERSATION_REPLY_PURPOSE,
    DeliveryRequest,
    DeliveryRequestResult,
    WorkshopDeliveryOutbox,
)
from kai.workshop.domain import ChannelBindingId, ChannelId, MessageId, PrincipalId
from kai.workshop.inbound import (
    ClientInboundMessage,
    InboundMessage,
    record_client_inbound_message_in_transaction,
    record_inbound_message_in_transaction,
)
from kai.workshop.run_lifecycle import DurableRun, RunLifecycleResult, RunStatus, WorkshopRunLifecycle
from kai.workshop.store import AppendResult, WorkshopEventStore


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


@dataclass(frozen=True, slots=True)
class ConversationCommandAcceptance:
    message: AppendResult
    lifecycle: RunLifecycleResult
    disposition: ConversationCommandDisposition

    @property
    def run(self) -> DurableRun:
        return self.lifecycle.run


@dataclass(frozen=True, slots=True)
class ClientConversationCommandAcceptance:
    """Atomic client acceptance plus its durable Telegram bridge request."""

    command: ConversationCommandAcceptance
    delivery: DeliveryRequestResult
    compatibility_chat_id: int

    @property
    def run(self) -> DurableRun:
        return self.command.run


class WorkshopConversationCommandService:
    """Atomically append canonical inbound and run-acceptance facts.

    This service starts no process. Its disposition is the durable input to
    the execution coordinator owned by the production private-text runtime.
    """

    def __init__(self, store: WorkshopEventStore) -> None:
        self._store = store

    async def accept(self, message: InboundMessage) -> ConversationCommandAcceptance:
        if not isinstance(message, InboundMessage):
            raise ValueError("message must be an InboundMessage")
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            inbound = await record_inbound_message_in_transaction(self._store, message)
            inbound_message_id = inbound.event.envelope.aggregate_id
            if not isinstance(inbound_message_id, MessageId):
                raise ConversationCommandStateConflictError("Canonical inbound event did not identify a message")
            lifecycle = await WorkshopRunLifecycle(self._store).accept_in_transaction(
                inbound_message_id,
                occurred_at=inbound.event.envelope.occurred_at,
            )
            if inbound.inserted != lifecycle.changed:
                raise ConversationCommandStateConflictError(
                    "Canonical inbound and run acceptance did not share one prior state"
                )
            disposition = (
                ConversationCommandDisposition.NEWLY_ACCEPTED
                if inbound.inserted
                else await self._replay_disposition(lifecycle.run)
            )
            await connection.commit()
            return ConversationCommandAcceptance(
                message=inbound,
                lifecycle=lifecycle,
                disposition=disposition,
            )
        except Exception:
            await connection.rollback()
            raise

    async def accept_client(
        self,
        message: ClientInboundMessage,
    ) -> ClientConversationCommandAcceptance:
        """Accept one canonical browser command and Telegram echo atomically."""
        if not isinstance(message, ClientInboundMessage):
            raise ValueError("message must be a ClientInboundMessage")
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            inbound = await record_client_inbound_message_in_transaction(self._store, message)
            inbound_message_id = inbound.event.envelope.aggregate_id
            if not isinstance(inbound_message_id, MessageId):
                raise ConversationCommandStateConflictError("Canonical inbound event did not identify a message")
            lifecycle = await WorkshopRunLifecycle(self._store).accept_in_transaction(
                inbound_message_id,
                occurred_at=inbound.event.envelope.occurred_at,
            )
            binding_id, chat_id = await self._telegram_compatibility_target(
                message.principal_id,
                message.channel_id,
            )
            delivery = await WorkshopDeliveryOutbox(self._store).request_delivery_in_transaction(
                DeliveryRequest(
                    message_id=inbound_message_id,
                    channel_binding_id=binding_id,
                    mode="workshop_client_text",
                    purpose=CONVERSATION_REPLY_PURPOSE,
                    occurred_at=inbound.event.envelope.occurred_at,
                    max_attempts=5,
                )
            )
            prior_states = {inbound.inserted, lifecycle.changed, delivery.inserted}
            if len(prior_states) != 1:
                raise ConversationCommandStateConflictError(
                    "Canonical inbound, run acceptance, and delivery request did not share one prior state"
                )
            disposition = (
                ConversationCommandDisposition.NEWLY_ACCEPTED
                if inbound.inserted
                else await self._replay_disposition(lifecycle.run)
            )
            await connection.commit()
            return ClientConversationCommandAcceptance(
                command=ConversationCommandAcceptance(inbound, lifecycle, disposition),
                delivery=delivery,
                compatibility_chat_id=chat_id,
            )
        except Exception:
            await connection.rollback()
            raise

    async def _telegram_compatibility_target(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
    ) -> tuple[ChannelBindingId, int]:
        async with self._store.connection.execute(
            "SELECT cb.id, cb.external_channel_id, ei.external_subject "
            "FROM channel_bindings cb "
            "JOIN external_identities ei ON ei.principal_id = ? "
            "AND ei.provider = 'telegram' "
            "WHERE cb.channel_id = ? AND cb.transport = 'telegram'",
            (principal_id, channel_id),
        ) as cursor:
            rows = list(await cursor.fetchall())
        if len(rows) != 1 or str(rows[0][1]) != str(rows[0][2]):
            raise ConversationCommandStateConflictError(
                "Client command requires one matching direct Telegram compatibility binding"
            )
        try:
            chat_id = int(str(rows[0][2]))
        except ValueError as exc:
            raise ConversationCommandStateConflictError("Telegram compatibility identity is not numeric") from exc
        if chat_id <= 0:
            raise ConversationCommandStateConflictError("Telegram compatibility identity is not a positive user ID")
        return ChannelBindingId(str(rows[0][0])), chat_id

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
