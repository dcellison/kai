"""Authenticated inbound-message shadow records for Kai Workshop."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime

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
from kai.workshop.store import AppendResult, WorkshopEventStore

_TRANSPORT_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class InboundBindingNotFoundError(LookupError):
    """The authenticated transport identity has no canonical channel binding."""


@dataclass(frozen=True, slots=True)
class InboundMessage:
    transport: str
    update_id: str
    message_id: str
    sender_subject: str
    channel_subject: str
    body: str
    occurred_at: datetime

    def __post_init__(self) -> None:
        if not _TRANSPORT_PATTERN.fullmatch(self.transport):
            raise ValueError("transport must be a lowercase identifier")
        for field_name in ("update_id", "message_id", "sender_subject", "channel_subject"):
            value = getattr(self, field_name)
            if not value or value != value.strip():
                raise ValueError(f"{field_name} must be non-empty without surrounding whitespace")
        if not self.body:
            raise ValueError("body must be non-empty")
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class _ResolvedInboundBinding:
    workshop_id: WorkshopId
    principal_id: PrincipalId
    channel_id: ChannelId


def _stable_token(message: InboundMessage) -> str:
    identity = "\0".join(
        (
            message.transport,
            message.update_id,
            message.channel_subject,
            message.message_id,
        )
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


async def _resolve_binding(store: WorkshopEventStore, message: InboundMessage) -> _ResolvedInboundBinding:
    async with store.connection.execute(
        "SELECT c.workshop_id, e.principal_id, c.id, c.kind "
        "FROM external_identities e "
        "JOIN workshop_memberships wm ON wm.principal_id = e.principal_id "
        "JOIN channel_bindings b ON b.transport = e.provider "
        "JOIN channels c ON c.id = b.channel_id AND c.workshop_id = wm.workshop_id "
        "WHERE e.provider = ? AND e.external_subject = ? "
        "AND b.transport = ? AND b.external_channel_id = ?",
        (
            message.transport,
            message.sender_subject,
            message.transport,
            message.channel_subject,
        ),
    ) as cursor:
        rows = list(await cursor.fetchall())
    if len(rows) != 1:
        raise InboundBindingNotFoundError("Inbound transport identity and channel do not resolve uniquely")
    row = rows[0]
    channel_kind = str(row[3])
    if channel_kind not in {"direct", "group"}:
        raise InboundBindingNotFoundError("Transport channel is not configured for inbound conversation")
    if channel_kind == "direct" and message.sender_subject != message.channel_subject:
        raise InboundBindingNotFoundError("Direct transport channel is not bound to the authenticated sender")
    return _ResolvedInboundBinding(
        workshop_id=WorkshopId(str(row[0])),
        principal_id=PrincipalId(str(row[1])),
        channel_id=ChannelId(str(row[2])),
    )


async def record_inbound_message(store: WorkshopEventStore, message: InboundMessage) -> AppendResult:
    """Append and project one authenticated inbound transport message."""
    binding = await _resolve_binding(store, message)
    token = _stable_token(message)
    result = await store.append(
        EventEnvelope.create(
            event_id=EventId.derived(binding.workshop_id, f"inbound-message-event:{token}"),
            event_type=WorkshopEventType.MESSAGE_CREATED,
            event_version=1,
            workshop_id=binding.workshop_id,
            aggregate_type="message",
            aggregate_id=MessageId.derived(binding.workshop_id, f"inbound-message:{token}"),
            actor_principal_id=binding.principal_id,
            occurred_at=message.occurred_at,
            idempotency_key=f"workshop-inbound:v1:{message.transport}:{token}",
            payload={
                "channel_id": binding.channel_id,
                "author_principal_id": binding.principal_id,
                "body": message.body,
            },
            metadata={
                "source": message.transport,
                "transport_update_id": message.update_id,
                "transport_message_id": message.message_id,
            },
        )
    )
    await store.project_pending(CanonicalConversationProjection())
    return result
