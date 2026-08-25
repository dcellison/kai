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
from kai.workshop.delivery_policy import WorkshopDeliveryBindingPolicy
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
_ROUTE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
GENERIC_INTEGRATION_SOURCE = "generic"
DEFAULT_INTEGRATION_ROUTE = "default"


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


@dataclass(frozen=True, slots=True)
class IntegrationRouteStatus:
    """Persisted canonical routing state for one external integration."""

    source: str
    route_name: str
    state: str
    channel_id: ChannelId | None
    detail: str


class WorkshopIntegrationNotificationService:
    """Own canonical integration recording independently of client adapters."""

    def __init__(self, store: WorkshopEventStore, delivery_policy: WorkshopDeliveryBindingPolicy) -> None:
        self._store = store
        self._delivery_policy = delivery_policy
        self._outbox = WorkshopDeliveryOutbox(store)
        self._lock = asyncio.Lock()
        self._closed = False

    @classmethod
    async def open(
        cls,
        database_path: Path,
        delivery_policy: WorkshopDeliveryBindingPolicy,
    ) -> WorkshopIntegrationNotificationService:
        service = cls(await WorkshopEventStore.open(database_path), delivery_policy)
        try:
            await service.reconcile_default_generic_route()
        except BaseException:
            await service.close()
            raise
        return service

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

    async def reconcile_default_generic_route(self) -> IntegrationRouteStatus:
        """Seed the default generic route from canonical policy when unambiguous."""
        async with self._lock:
            self._ensure_open()
            existing = await self._route_status_locked(
                GENERIC_INTEGRATION_SOURCE,
                DEFAULT_INTEGRATION_ROUTE,
            )
            if existing.state != "missing":
                return existing

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
                detail = (
                    "no canonical admin direct channel is available"
                    if not rows
                    else "multiple canonical admin direct channels are available"
                )
                return IntegrationRouteStatus(
                    source=GENERIC_INTEGRATION_SOURCE,
                    route_name=DEFAULT_INTEGRATION_ROUTE,
                    state="missing" if not rows else "ambiguous",
                    channel_id=None,
                    detail=detail,
                )

            channel_id = ChannelId(str(rows[0][0]))
            await self._validate_channel_locked(channel_id)
            try:
                await self._store.connection.execute("BEGIN IMMEDIATE")
                await self._store.connection.execute(
                    "INSERT OR IGNORE INTO workshop_integration_routes "
                    "(source, route_name, channel_id) VALUES (?, ?, ?)",
                    (GENERIC_INTEGRATION_SOURCE, DEFAULT_INTEGRATION_ROUTE, channel_id),
                )
                await self._store.connection.commit()
            except Exception:
                await self._store.connection.rollback()
                raise
            return await self._route_status_locked(
                GENERIC_INTEGRATION_SOURCE,
                DEFAULT_INTEGRATION_ROUTE,
            )

    async def route_status(self, *, source: str, route_name: str) -> IntegrationRouteStatus:
        """Inspect one canonical integration route without resolving a transport."""
        self._validate_route_identity(source, route_name)
        async with self._lock:
            self._ensure_open()
            return await self._route_status_locked(source, route_name)

    async def set_route(
        self,
        *,
        source: str,
        route_name: str,
        channel_id: ChannelId,
    ) -> IntegrationRouteStatus:
        """Explicitly assign one integration route to a canonical channel."""
        self._validate_route_identity(source, route_name)
        if not isinstance(channel_id, ChannelId):
            raise ValueError("channel_id must be a ChannelId")
        async with self._lock:
            self._ensure_open()
            await self._validate_channel_locked(channel_id)
            try:
                await self._store.connection.execute("BEGIN IMMEDIATE")
                await self._store.connection.execute(
                    "INSERT INTO workshop_integration_routes "
                    "(source, route_name, channel_id) VALUES (?, ?, ?) "
                    "ON CONFLICT(source, route_name) DO UPDATE SET "
                    "channel_id = excluded.channel_id, "
                    "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
                    (source, route_name, channel_id),
                )
                await self._store.connection.commit()
            except Exception:
                await self._store.connection.rollback()
                raise
            return await self._route_status_locked(source, route_name)

    async def record_for_route(
        self,
        notification: IntegrationNotification,
        *,
        route_name: str,
    ) -> IntegrationNotificationRecord:
        """Record using explicit canonical routing policy for the integration source."""
        if not isinstance(notification, IntegrationNotification):
            raise ValueError("notification must be an IntegrationNotification")
        self._validate_route_identity(notification.source, route_name)
        async with self._lock:
            self._ensure_open()
            status = await self._route_status_locked(notification.source, route_name)
            if status.state != "active" or status.channel_id is None:
                raise IntegrationNotificationError(
                    f"Integration route {notification.source}/{route_name} is {status.state}: {status.detail}"
                )
            return await self._record_locked(notification, status.channel_id)

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

    async def _route_status_locked(
        self,
        source: str,
        route_name: str,
    ) -> IntegrationRouteStatus:
        async with self._store.connection.execute(
            "SELECT channel_id FROM workshop_integration_routes WHERE source = ? AND route_name = ?",
            (source, route_name),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return IntegrationRouteStatus(
                source=source,
                route_name=route_name,
                state="missing",
                channel_id=None,
                detail="no canonical destination is configured",
            )
        channel_id = ChannelId(str(row[0]))
        try:
            await self._validate_channel_locked(channel_id)
        except AmbiguousIntegrationNotificationDestinationError as exc:
            return IntegrationRouteStatus(
                source=source,
                route_name=route_name,
                state="invalid",
                channel_id=channel_id,
                detail=str(exc),
            )
        return IntegrationRouteStatus(
            source=source,
            route_name=route_name,
            state="active",
            channel_id=channel_id,
            detail="canonical destination is configured",
        )

    async def _validate_channel_locked(self, channel_id: ChannelId) -> None:
        async with self._store.connection.execute(
            "SELECT c.kind, COUNT(p.id) FROM channels c "
            "LEFT JOIN channel_agents ca ON ca.channel_id = c.id "
            "LEFT JOIN agents a ON a.id = ca.agent_id AND a.workshop_id = c.workshop_id "
            "LEFT JOIN principals p ON p.id = a.principal_id AND p.kind = 'agent' "
            "WHERE c.id = ? GROUP BY c.id, c.kind",
            (channel_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None or str(row[0]) not in {"direct", "notification"}:
            raise AmbiguousIntegrationNotificationDestinationError(
                "Canonical notification destination is missing or unsupported"
            )
        if int(row[1]) != 1:
            raise AmbiguousIntegrationNotificationDestinationError(
                "Canonical notification destination must have exactly one agent"
            )

    @staticmethod
    def _validate_route_identity(source: str, route_name: str) -> None:
        if not _SOURCE_PATTERN.fullmatch(source):
            raise ValueError("source must be a lowercase integration identifier")
        if not _ROUTE_PATTERN.fullmatch(route_name):
            raise ValueError("route_name must be a bounded lowercase route identifier")

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
            binding_ids = await self._delivery_policy.binding_ids(self._store, channel_id)
            deliveries = tuple(
                [
                    await self._outbox.request_delivery_in_transaction(
                        DeliveryRequest(
                            message_id=message_id,
                            channel_binding_id=binding_id,
                            mode="text",
                            purpose=NOTIFICATION_PURPOSE,
                            occurred_at=occurred_at,
                            max_attempts=5,
                        )
                    )
                    for binding_id in binding_ids
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
