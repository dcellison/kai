"""Canonical recording for core-owned scheduled reminders."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime

from kai.workshop.delivery_outbox import NOTIFICATION_PURPOSE, DeliveryRequestResult
from kai.workshop.delivery_planning import CanonicalDeliveryIntent, WorkshopDeliveryPlanner
from kai.workshop.delivery_policy import WorkshopDeliveryBindingPolicy
from kai.workshop.domain import (
    ChannelId,
    EventEnvelope,
    EventId,
    MessageId,
    PrincipalId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import IdempotencyConflictError, WorkshopEventStore


@dataclass(frozen=True, slots=True)
class ScheduledReminder:
    job_id: int
    occurrence_id: str
    body: str
    occurred_at: datetime


@dataclass(frozen=True, slots=True)
class ScheduledReminderRecord:
    message_id: MessageId
    deliveries: tuple[DeliveryRequestResult, ...]

    @property
    def delivery(self) -> DeliveryRequestResult | None:
        return self.deliveries[0] if len(self.deliveries) == 1 else None


class WorkshopScheduledReminderRecorder:
    """Append one reminder to its canonical channel and optional adapter outbox."""

    def __init__(self, store: WorkshopEventStore, delivery_policy: WorkshopDeliveryBindingPolicy) -> None:
        self._store = store
        self._delivery_planner = WorkshopDeliveryPlanner(store, delivery_policy)
        self._lock = asyncio.Lock()

    async def record(self, reminder: ScheduledReminder) -> ScheduledReminderRecord:
        async with self._lock:
            return await self._record_locked(reminder)

    async def _record_locked(self, reminder: ScheduledReminder) -> ScheduledReminderRecord:
        async with self._store.connection.execute(
            "SELECT c.workshop_id, c.id, a.principal_id "
            "FROM workshop_scheduled_jobs j "
            "JOIN channels c ON c.id = j.channel_id "
            "JOIN agents a ON a.id = j.agent_id AND a.workshop_id = c.workshop_id "
            "WHERE j.id = ?",
            (reminder.job_id,),
        ) as cursor:
            rows = list(await cursor.fetchall())
        if len(rows) != 1:
            raise RuntimeError("Scheduled reminder does not have one canonical owner")
        workshop_id = WorkshopId(str(rows[0][0]))
        channel_id = ChannelId(str(rows[0][1]))
        agent_principal_id = PrincipalId(str(rows[0][2]))
        stable_name = f"scheduled-reminder:v1:{reminder.job_id}:{reminder.occurrence_id}"
        message_id = MessageId.derived(workshop_id, stable_name)
        idempotency_key = f"workshop-scheduled-reminder:v1:{message_id}"
        payload = {
            "channel_id": channel_id,
            "author_principal_id": agent_principal_id,
            "body": reminder.body,
        }
        metadata = {
            "source": "scheduled_job",
            "job_id": reminder.job_id,
            "occurrence_id": reminder.occurrence_id,
            "job_type": "reminder",
        }
        occurred_at = reminder.occurred_at.astimezone(UTC)
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            existing = await self._store.event_by_idempotency_key(idempotency_key)
            inserted = existing is None
            if existing is None:
                await self._store.append_in_transaction(
                    EventEnvelope.create(
                        event_id=EventId.derived(workshop_id, f"{stable_name}:event"),
                        event_type=WorkshopEventType.MESSAGE_CREATED,
                        event_version=1,
                        workshop_id=workshop_id,
                        aggregate_type="message",
                        aggregate_id=message_id,
                        actor_principal_id=agent_principal_id,
                        occurred_at=occurred_at,
                        idempotency_key=idempotency_key,
                        payload=payload,
                        metadata=metadata,
                    )
                )
            elif existing.envelope.payload != payload or existing.envelope.metadata != metadata:
                raise IdempotencyConflictError(f"Event identity {idempotency_key!r} was reused with different content")
            projection = CanonicalConversationProjection()
            await self._store.project_pending_in_transaction(projection)
            planning = await self._delivery_planner.plan_in_transaction(
                CanonicalDeliveryIntent(
                    message_id=message_id,
                    channel_id=channel_id,
                    mode="text",
                    purpose=NOTIFICATION_PURPOSE,
                    occurred_at=occurred_at,
                )
            )
            if any(delivery.inserted != inserted for delivery in planning.deliveries):
                raise RuntimeError("Scheduled reminder and deliveries do not share one prior state")
            await self._store.project_pending_in_transaction(projection)
            await connection.commit()
            return ScheduledReminderRecord(message_id, planning.deliveries)
        except Exception:
            await connection.rollback()
            raise
