"""Production-unused atomic acceptance for canonical Workshop commands."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from kai.workshop.domain import MessageId
from kai.workshop.inbound import InboundMessage, record_inbound_message_in_transaction
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


class WorkshopConversationCommandService:
    """Atomically append canonical inbound and run-acceptance facts.

    This service starts no process and is not constructed by production code.
    Its disposition is the durable input to a later execution coordinator.
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
                occurred_at=message.occurred_at,
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
