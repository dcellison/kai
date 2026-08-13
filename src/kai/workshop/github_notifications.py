"""Canonical GitHub notification recording for Kai Workshop."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass
from datetime import UTC, datetime

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

_GITHUB_DELIVERY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_GITHUB_EVENT_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_REPOSITORY_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")


class GitHubNotificationError(RuntimeError):
    """Canonical notification work could not be recorded safely."""


class AmbiguousGitHubNotificationChannelError(GitHubNotificationError):
    """A Telegram destination resolves to more than one notification channel."""


@dataclass(frozen=True, slots=True)
class GitHubNotification:
    """One authenticated GitHub delivery formatted for human presentation."""

    delivery_id: str
    event_type: str
    repository: str
    telegram_chat_id: int
    body: str
    occurred_at: datetime

    def __post_init__(self) -> None:
        if not _GITHUB_DELIVERY_PATTERN.fullmatch(self.delivery_id):
            raise ValueError("delivery_id must be a bounded GitHub delivery identifier")
        if not _GITHUB_EVENT_PATTERN.fullmatch(self.event_type):
            raise ValueError("event_type must be a lowercase GitHub event identifier")
        if self.repository and not _REPOSITORY_PATTERN.fullmatch(self.repository):
            raise ValueError("repository must be empty or an owner/name identifier")
        if (
            not isinstance(self.telegram_chat_id, int)
            or isinstance(self.telegram_chat_id, bool)
            or not -(2**63) <= self.telegram_chat_id < 0
        ):
            raise ValueError("telegram_chat_id must be a negative signed 64-bit integer")
        if not self.body or len(self.body) > 4_096_000:
            raise ValueError("body must contain between 1 and 4096000 characters")
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class GitHubNotificationRecord:
    """The canonical message and durable Telegram request for one delivery."""

    message_id: MessageId
    delivery: DeliveryRequestResult


class WorkshopGitHubNotificationRecorder:
    """Record authenticated GitHub deliveries under one serialized writer."""

    def __init__(self, store: WorkshopEventStore) -> None:
        self._store = store
        self._outbox = WorkshopDeliveryOutbox(store)
        self._lock = asyncio.Lock()

    async def record(
        self,
        notification: GitHubNotification,
    ) -> GitHubNotificationRecord | None:
        """Record and queue one notification, or return None for an unbound target."""
        if not isinstance(notification, GitHubNotification):
            raise ValueError("notification must be a GitHubNotification")
        async with self._lock:
            return await self._record_locked(notification)

    async def _record_locked(
        self,
        notification: GitHubNotification,
    ) -> GitHubNotificationRecord | None:
        async with self._store.connection.execute(
            "SELECT c.workshop_id, c.id, cb.id, a.principal_id "
            "FROM channels c "
            "JOIN channel_bindings cb ON cb.channel_id = c.id "
            "AND cb.transport = 'telegram' "
            "JOIN channel_agents ca ON ca.channel_id = c.id "
            "JOIN agents a ON a.id = ca.agent_id AND a.workshop_id = c.workshop_id "
            "JOIN principals p ON p.id = a.principal_id AND p.kind = 'agent' "
            "WHERE c.kind = 'notification' AND cb.external_channel_id = ?",
            (str(notification.telegram_chat_id),),
        ) as cursor:
            rows = list(await cursor.fetchall())
        if not rows:
            return None
        if len(rows) != 1:
            raise AmbiguousGitHubNotificationChannelError(
                "Telegram notification destination resolves to multiple canonical channels"
            )

        workshop_id = WorkshopId(str(rows[0][0]))
        channel_id = ChannelId(str(rows[0][1]))
        binding_id = ChannelBindingId(str(rows[0][2]))
        agent_principal_id = PrincipalId(str(rows[0][3]))
        stable_name = f"github-notification:v1:{binding_id}:{notification.delivery_id}"
        message_id = MessageId.derived(workshop_id, stable_name)
        idempotency_key = f"workshop-github-notification:v1:{binding_id}:{notification.delivery_id}"
        payload = {
            "channel_id": channel_id,
            "author_principal_id": agent_principal_id,
            "body": notification.body,
        }
        metadata = {
            "source": "github",
            "github_delivery_id": notification.delivery_id,
            "github_event": notification.event_type,
            "repository": notification.repository,
        }
        occurred_at = notification.occurred_at.astimezone(UTC)
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            existing = await self._store.event_by_idempotency_key(idempotency_key)
            message_inserted = existing is None
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
            delivery = await self._outbox.request_delivery_in_transaction(
                DeliveryRequest(
                    message_id=message_id,
                    channel_binding_id=binding_id,
                    mode="text",
                    purpose=NOTIFICATION_PURPOSE,
                    occurred_at=occurred_at,
                    max_attempts=5,
                )
            )
            if message_inserted != delivery.inserted:
                raise GitHubNotificationError("GitHub notification message and delivery do not share one prior state")
            await self._store.project_pending_in_transaction(projection)
            await connection.commit()
            return GitHubNotificationRecord(message_id=message_id, delivery=delivery)
        except Exception:
            await connection.rollback()
            raise
