"""Explicit installed qualification path for the Workshop delivery outbox."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from kai.workshop.delivery_outbox import (
    DeliveryClaim,
    DeliveryRequest,
    DeliveryRequestResult,
    DeliveryState,
    WorkshopDeliveryOutbox,
)
from kai.workshop.domain import ChannelBindingId, DeliveryId, MessageId
from kai.workshop.store import WorkshopEventStore


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
                occurred_at=datetime.now(UTC),
                max_attempts=3,
            )
        )

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
