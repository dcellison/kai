"""Canonical notification recording for authenticated external integrations."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kai.workshop.delivery_outbox import (
    NOTIFICATION_PURPOSE,
    DeliveryRequest,
    DeliveryRequestResult,
    WorkshopDeliveryOutbox,
)
from kai.workshop.domain import (
    ChannelBindingId,
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

_DELIVERY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SOURCE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_EVENT_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class IntegrationNotificationError(RuntimeError):
    """Canonical integration work could not be recorded safely."""


class AmbiguousIntegrationNotificationDestinationError(IntegrationNotificationError):
    """An external or default destination does not resolve uniquely."""


@dataclass(frozen=True, slots=True)
class IntegrationNotification:
    """One authenticated external delivery formatted for human presentation."""

    delivery_id: str
    source: str
    event_type: str
    body: str
    occurred_at: datetime
    repository: str = ""

    def __post_init__(self) -> None:
        if not _DELIVERY_PATTERN.fullmatch(self.delivery_id):
            raise ValueError("delivery_id must be a bounded external delivery identifier")
        if not _SOURCE_PATTERN.fullmatch(self.source):
            raise ValueError("source must be a lowercase integration identifier")
        if not _EVENT_PATTERN.fullmatch(self.event_type):
            raise ValueError("event_type must be a lowercase integration event identifier")
        if self.repository and not _REPOSITORY_PATTERN.fullmatch(self.repository):
            raise ValueError("repository must be empty or an owner/name identifier")
        if not self.body or len(self.body) > 4_096_000:
            raise ValueError("body must contain between 1 and 4096000 characters")
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class IntegrationNotificationRecord:
    """One canonical message and its durable external delivery requests."""

    message_id: MessageId
    inserted: bool
    deliveries: tuple[DeliveryRequestResult, ...]


class WorkshopIntegrationNotificationService:
    """Own canonical integration recording independently of client adapters."""

    def __init__(self, store: WorkshopEventStore) -> None:
        self._store = store
        self._outbox = WorkshopDeliveryOutbox(store)
        self._lock = asyncio.Lock()
        self._closed = False

    @classmethod
    async def open(cls, database_path: Path) -> WorkshopIntegrationNotificationService:
        return cls(await WorkshopEventStore.open(database_path))

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._store.close()

    async def record_for_binding(
        self,
        notification: IntegrationNotification,
        *,
        transport: str,
        external_channel_id: str,
    ) -> IntegrationNotificationRecord | None:
        """Resolve an adapter binding and record one canonical notification."""
        if not transport or not external_channel_id:
            raise ValueError("transport and external_channel_id must be non-empty")
        async with self._lock:
            self._ensure_open()
            async with self._store.connection.execute(
                "SELECT c.id, cb.id FROM channels c "
                "JOIN channel_bindings cb ON cb.channel_id = c.id "
                "WHERE c.kind IN ('direct', 'notification') "
                "AND cb.transport = ? AND cb.external_channel_id = ? ORDER BY c.id",
                (transport, external_channel_id),
            ) as cursor:
                rows = tuple(await cursor.fetchall())
            if not rows:
                return None
            if len(rows) != 1:
                raise AmbiguousIntegrationNotificationDestinationError(
                    "External notification destination resolves to multiple canonical channels"
                )
            return await self._record_locked(
                notification,
                ChannelId(str(rows[0][0])),
                source_binding_id=ChannelBindingId(str(rows[0][1])),
            )

    async def record_for_default_admin(
        self,
        notification: IntegrationNotification,
    ) -> IntegrationNotificationRecord:
        """Record to the unique admin human's canonical direct channel."""
        async with self._lock:
            self._ensure_open()
            async with self._store.connection.execute(
                "SELECT DISTINCT c.id FROM channels c "
                "JOIN channel_memberships cm ON cm.channel_id = c.id "
                "JOIN principals p ON p.id = cm.principal_id AND p.kind = 'human' "
                "JOIN workshop_memberships wm ON wm.workshop_id = c.workshop_id "
                "AND wm.principal_id = p.id AND wm.role = 'admin' "
                "WHERE c.kind = 'direct' ORDER BY c.id"
            ) as cursor:
                rows = tuple(await cursor.fetchall())
            if len(rows) != 1:
                raise AmbiguousIntegrationNotificationDestinationError(
                    "Generic webhook delivery requires exactly one canonical admin direct channel"
                )
            return await self._record_locked(notification, ChannelId(str(rows[0][0])))

    async def record_for_channel(
        self,
        notification: IntegrationNotification,
        channel_id: ChannelId,
    ) -> IntegrationNotificationRecord:
        """Record to one already-authorized canonical destination."""
        if not isinstance(channel_id, ChannelId):
            raise ValueError("channel_id must be a ChannelId")
        async with self._lock:
            self._ensure_open()
            async with self._store.connection.execute(
                "SELECT 1 FROM channels WHERE id = ? AND kind IN ('direct', 'notification')",
                (channel_id,),
            ) as cursor:
                rows = tuple(await cursor.fetchall())
            if len(rows) != 1:
                raise AmbiguousIntegrationNotificationDestinationError(
                    "Canonical notification destination is missing or unsupported"
                )
            return await self._record_locked(notification, channel_id)

    async def _record_locked(
        self,
        notification: IntegrationNotification,
        channel_id: ChannelId,
        *,
        source_binding_id: ChannelBindingId | None = None,
    ) -> IntegrationNotificationRecord:
        if not isinstance(notification, IntegrationNotification):
            raise ValueError("notification must be an IntegrationNotification")
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            async with connection.execute(
                "SELECT c.workshop_id, a.principal_id FROM channels c "
                "JOIN channel_agents ca ON ca.channel_id = c.id "
                "JOIN agents a ON a.id = ca.agent_id AND a.workshop_id = c.workshop_id "
                "JOIN principals p ON p.id = a.principal_id AND p.kind = 'agent' "
                "WHERE c.id = ? ORDER BY a.id",
                (channel_id,),
            ) as cursor:
                agents = tuple(await cursor.fetchall())
            if len(agents) != 1:
                raise AmbiguousIntegrationNotificationDestinationError(
                    "Canonical notification destination must have exactly one agent"
                )

            workshop_id = WorkshopId(str(agents[0][0]))
            agent_principal_id = PrincipalId(str(agents[0][1]))
            if notification.source == "github" and source_binding_id is not None:
                # Preserve the pre-core-service identity exactly so a GitHub
                # redelivery recorded before this cutover remains idempotent.
                stable_name = f"github-notification:v1:{source_binding_id}:{notification.delivery_id}"
                idempotency_key = f"workshop-github-notification:v1:{source_binding_id}:{notification.delivery_id}"
                metadata = {
                    "source": "github",
                    "github_delivery_id": notification.delivery_id,
                    "github_event": notification.event_type,
                    "repository": notification.repository,
                }
            else:
                stable_name = (
                    f"integration-notification:v1:{channel_id}:{notification.source}:{notification.delivery_id}"
                )
                idempotency_key = stable_name
                metadata = {
                    "source": notification.source,
                    "external_delivery_id": notification.delivery_id,
                    "integration_event": notification.event_type,
                    "repository": notification.repository,
                }
            message_id = MessageId.derived(workshop_id, stable_name)
            payload = {
                "channel_id": channel_id,
                "author_principal_id": agent_principal_id,
                "body": notification.body,
            }
            occurred_at = notification.occurred_at.astimezone(UTC)
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
            elif (
                existing.envelope.event_type != WorkshopEventType.MESSAGE_CREATED
                or existing.envelope.aggregate_id != message_id
                or existing.envelope.actor_principal_id != agent_principal_id
                or existing.envelope.payload != payload
                or existing.envelope.metadata != metadata
            ):
                raise IdempotencyConflictError(f"Event identity {idempotency_key!r} was reused with different content")

            projection = CanonicalConversationProjection()
            await self._store.project_pending_in_transaction(projection)
            async with connection.execute(
                "SELECT id FROM channel_bindings WHERE channel_id = ? ORDER BY id",
                (channel_id,),
            ) as cursor:
                binding_rows = tuple(await cursor.fetchall())
            deliveries = tuple(
                [
                    await self._outbox.request_delivery_in_transaction(
                        DeliveryRequest(
                            message_id=message_id,
                            channel_binding_id=ChannelBindingId(str(row[0])),
                            mode="text",
                            purpose=NOTIFICATION_PURPOSE,
                            occurred_at=occurred_at,
                            max_attempts=5,
                        )
                    )
                    for row in binding_rows
                ]
            )
            if inserted and any(not delivery.inserted for delivery in deliveries):
                raise IntegrationNotificationError(
                    "Canonical notification message and deliveries do not share one prior state"
                )
            await self._store.project_pending_in_transaction(projection)
            await connection.commit()
            return IntegrationNotificationRecord(
                message_id=message_id,
                inserted=inserted,
                deliveries=deliveries,
            )
        except Exception:
            await connection.rollback()
            raise

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Workshop integration notification service is closed")
