"""Authenticated inbound-message shadow records for Kai Workshop."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, replace
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
_CLIENT_MESSAGE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


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
class ClientInboundMessage:
    """One browser command after session authentication and authorization."""

    principal_id: PrincipalId
    channel_id: ChannelId
    client_message_id: str
    body: str
    occurred_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.principal_id, PrincipalId):
            raise ValueError("principal_id must be a PrincipalId")
        if not isinstance(self.channel_id, ChannelId):
            raise ValueError("channel_id must be a ChannelId")
        if not _CLIENT_MESSAGE_ID_PATTERN.fullmatch(self.client_message_id):
            raise ValueError("client_message_id must be a bounded opaque identifier")
        if not isinstance(self.body, str) or not self.body.strip():
            raise ValueError("body must contain non-whitespace text")
        if len(self.body) > 50_000:
            raise ValueError("body must be at most 50000 characters")
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


def _inbound_envelope(binding: _ResolvedInboundBinding, message: InboundMessage) -> EventEnvelope:
    token = _stable_token(message)
    return EventEnvelope.create(
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


async def _resolve_client_binding(
    store: WorkshopEventStore,
    message: ClientInboundMessage,
) -> _ResolvedInboundBinding:
    async with store.connection.execute(
        "SELECT c.workshop_id FROM channels c "
        "JOIN channel_memberships cm ON cm.channel_id = c.id AND cm.principal_id = ? "
        "JOIN workshop_memberships wm ON wm.workshop_id = c.workshop_id "
        "AND wm.principal_id = cm.principal_id "
        "JOIN principals p ON p.id = cm.principal_id AND p.kind = 'human' "
        "WHERE c.id = ?",
        (message.principal_id, message.channel_id),
    ) as cursor:
        rows = list(await cursor.fetchall())
    if len(rows) != 1:
        raise InboundBindingNotFoundError("Client principal is not an authorized human channel member")
    return _ResolvedInboundBinding(
        workshop_id=WorkshopId(str(rows[0][0])),
        principal_id=message.principal_id,
        channel_id=message.channel_id,
    )


def _client_inbound_envelope(
    binding: _ResolvedInboundBinding,
    message: ClientInboundMessage,
) -> EventEnvelope:
    stable_name = f"client-message:{message.principal_id}:{message.channel_id}:{message.client_message_id}"
    message_id = MessageId.derived(binding.workshop_id, stable_name)
    return EventEnvelope.create(
        event_id=EventId.derived(binding.workshop_id, f"client-message-event:{message_id}"),
        event_type=WorkshopEventType.MESSAGE_CREATED,
        event_version=1,
        workshop_id=binding.workshop_id,
        aggregate_type="message",
        aggregate_id=message_id,
        actor_principal_id=binding.principal_id,
        occurred_at=message.occurred_at,
        idempotency_key=f"workshop-client-message:v1:{message_id}",
        payload={
            "channel_id": binding.channel_id,
            "author_principal_id": binding.principal_id,
            "body": message.body,
        },
        metadata={
            "source": "workshop_client",
            "client_message_id": message.client_message_id,
        },
    )


async def record_inbound_message_in_transaction(
    store: WorkshopEventStore,
    message: InboundMessage,
) -> AppendResult:
    """Append and project one inbound message inside a caller-owned transaction."""
    if not store.connection.in_transaction:
        raise RuntimeError("record_inbound_message_in_transaction requires an active transaction")
    binding = await _resolve_binding(store, message)
    result = await store.append_in_transaction(_inbound_envelope(binding, message))
    await store.project_pending_in_transaction(CanonicalConversationProjection())
    return result


async def record_client_inbound_message_in_transaction(
    store: WorkshopEventStore,
    message: ClientInboundMessage,
) -> AppendResult:
    """Append an authenticated client message inside a caller-owned transaction."""
    if not store.connection.in_transaction:
        raise RuntimeError("record_client_inbound_message_in_transaction requires an active transaction")
    binding = await _resolve_client_binding(store, message)
    envelope = _client_inbound_envelope(binding, message)
    if envelope.idempotency_key is None:
        raise RuntimeError("Client message envelope did not define an idempotency key")
    existing = await store.event_by_idempotency_key(envelope.idempotency_key)
    if existing is not None:
        envelope = _client_inbound_envelope(
            binding,
            replace(message, occurred_at=existing.envelope.occurred_at),
        )
    result = await store.append_in_transaction(envelope)
    await store.project_pending_in_transaction(CanonicalConversationProjection())
    return result


async def record_inbound_message(store: WorkshopEventStore, message: InboundMessage) -> AppendResult:
    """Append and project one authenticated inbound transport message."""
    binding = await _resolve_binding(store, message)
    result = await store.append(_inbound_envelope(binding, message))
    await store.project_pending(CanonicalConversationProjection())
    return result
