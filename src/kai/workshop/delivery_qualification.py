"""Explicit installed qualification path for the Workshop delivery outbox."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from kai.workshop.delivery_outbox import (
    QUALIFICATION_PURPOSE,
    DeliveryClaim,
    DeliveryRequest,
    DeliveryRequestResult,
    DeliveryState,
    WorkshopDeliveryOutbox,
)
from kai.workshop.domain import (
    ChannelBindingId,
    ChannelId,
    DeliveryId,
    EventEnvelope,
    EventId,
    MessageId,
    PrincipalId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import IdempotencyConflictError, WorkshopEventStore

_NOTIFICATION_QUALIFICATION_BODY = "Kai Workshop notification-group delivery qualification."


class DeliveryQualificationError(RuntimeError):
    """The requested installed qualification cannot be performed safely."""


class WorkshopDeliveryQualification:
    """Select and exercise one real delivery without becoming a worker loop."""

    def __init__(self, store: WorkshopEventStore) -> None:
        self._store = store
        self._outbox = WorkshopDeliveryOutbox(store)

    async def prepare(self, telegram_user_id: int) -> DeliveryRequestResult:
        """Queue the latest canonical Kai reply in one configured direct chat."""
        if (
            not isinstance(telegram_user_id, int)
            or isinstance(telegram_user_id, bool)
            or not 1 <= telegram_user_id <= 2**63 - 1
        ):
            raise ValueError("telegram_user_id must be a positive signed 64-bit integer")

        subject = str(telegram_user_id)
        async with self._store.connection.execute(
            "SELECT m.id, cb.id FROM external_identities ei "
            "JOIN principals human ON human.id = ei.principal_id AND human.kind = 'human' "
            "JOIN channel_memberships cm ON cm.principal_id = human.id AND cm.role = 'owner' "
            "JOIN channels c ON c.id = cm.channel_id AND c.kind = 'direct' "
            "JOIN workshop_memberships wm ON wm.workshop_id = c.workshop_id "
            "AND wm.principal_id = human.id "
            "JOIN channel_bindings cb ON cb.channel_id = cm.channel_id "
            "AND cb.transport = 'telegram' AND cb.external_channel_id = ei.external_subject "
            "JOIN messages m ON m.channel_id = cm.channel_id "
            "JOIN principals author ON author.id = m.author_principal_id AND author.kind = 'agent' "
            "WHERE ei.provider = 'telegram' AND ei.external_subject = ? "
            "ORDER BY m.created_event_position DESC LIMIT 1",
            (subject,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise DeliveryQualificationError(
                "No canonical Kai reply exists for that configured Telegram user; send Kai a normal message first"
            )

        return await self._outbox.request_delivery(
            DeliveryRequest(
                message_id=MessageId(str(row[0])),
                channel_binding_id=ChannelBindingId(str(row[1])),
                mode="text",
                purpose=QUALIFICATION_PURPOSE,
                occurred_at=datetime.now(UTC),
                max_attempts=3,
            )
        )

    async def prepare_notification_group(self, telegram_chat_id: int) -> DeliveryRequestResult:
        """Atomically create and queue one recognizable outbound-only group message."""
        if (
            not isinstance(telegram_chat_id, int)
            or isinstance(telegram_chat_id, bool)
            or not -(2**63) <= telegram_chat_id < 0
        ):
            raise ValueError("telegram_chat_id must be a negative signed 64-bit integer")

        async with self._store.connection.execute(
            "SELECT c.workshop_id, c.id, cb.id, a.principal_id "
            "FROM channels c "
            "JOIN channel_bindings cb ON cb.channel_id = c.id AND cb.transport = 'telegram' "
            "JOIN channel_agents ca ON ca.channel_id = c.id "
            "JOIN agents a ON a.id = ca.agent_id AND a.workshop_id = c.workshop_id "
            "JOIN principals p ON p.id = a.principal_id AND p.kind = 'agent' "
            "WHERE c.kind = 'notification' AND cb.external_channel_id = ?",
            (str(telegram_chat_id),),
        ) as cursor:
            rows = list(await cursor.fetchall())
        if len(rows) != 1:
            raise DeliveryQualificationError(
                "Telegram notification group does not resolve to one canonical outbound-only channel; restart Kai "
                "after configuring the group"
            )

        workshop_id = WorkshopId(str(rows[0][0]))
        channel_id = ChannelId(str(rows[0][1]))
        binding_id = ChannelBindingId(str(rows[0][2]))
        agent_principal_id = PrincipalId(str(rows[0][3]))
        message_id = MessageId.derived(workshop_id, f"notification-group-qualification:{binding_id}")
        idempotency_key = f"workshop-delivery-qualification:v1:notification-group:{binding_id}"
        payload = {
            "channel_id": channel_id,
            "author_principal_id": agent_principal_id,
            "body": _NOTIFICATION_QUALIFICATION_BODY,
        }
        occurred_at = datetime.now(UTC)
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            existing = await self._store.event_by_idempotency_key(idempotency_key)
            message_inserted = existing is None
            if existing is None:
                await self._store.append_in_transaction(
                    EventEnvelope.create(
                        event_id=EventId.derived(
                            workshop_id,
                            f"notification-group-qualification-event:{binding_id}",
                        ),
                        event_type=WorkshopEventType.MESSAGE_CREATED,
                        event_version=1,
                        workshop_id=workshop_id,
                        aggregate_type="message",
                        aggregate_id=message_id,
                        actor_principal_id=agent_principal_id,
                        occurred_at=occurred_at,
                        idempotency_key=idempotency_key,
                        payload=payload,
                        metadata={"source": "delivery_qualification"},
                    )
                )
            elif (
                existing.envelope.event_type != WorkshopEventType.MESSAGE_CREATED
                or existing.envelope.aggregate_id != message_id
                or existing.envelope.actor_principal_id != agent_principal_id
                or existing.envelope.payload != payload
            ):
                raise IdempotencyConflictError(f"Event identity {idempotency_key!r} was reused with different content")

            projection = CanonicalConversationProjection()
            await self._store.project_pending_in_transaction(projection)
            delivery = await self._outbox.request_delivery_in_transaction(
                DeliveryRequest(
                    message_id=message_id,
                    channel_binding_id=binding_id,
                    mode="text",
                    purpose=QUALIFICATION_PURPOSE,
                    occurred_at=occurred_at,
                    max_attempts=3,
                )
            )
            if message_inserted != delivery.inserted:
                raise DeliveryQualificationError(
                    "Notification-group qualification message and delivery do not share one prior state"
                )
            await self._store.project_pending_in_transaction(projection)
            await connection.commit()
            return delivery
        except Exception:
            await connection.rollback()
            raise

    async def simulate_interruption(
        self,
        delivery_id: DeliveryId,
        *,
        worker_id: str,
        lease_duration: timedelta,
    ) -> DeliveryClaim:
        """Claim exact work and exit before sending to test lease recovery."""
        claim = await self._outbox.claim_next(
            worker_id,
            purposes=(QUALIFICATION_PURPOSE,),
            lease_duration=lease_duration,
            transport="telegram",
            modes=("text",),
            delivery_id=delivery_id,
        )
        if claim is None:
            raise DeliveryQualificationError("The selected delivery is not currently claimable")
        return claim

    async def status(self, delivery_id: DeliveryId) -> DeliveryState:
        return await self._outbox.state(delivery_id)
