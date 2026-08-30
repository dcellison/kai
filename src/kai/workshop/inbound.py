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
    MessageMention,
    PrincipalId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import AppendResult, WorkshopEventStore

_TRANSPORT_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_CLIENT_MESSAGE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_ROUTING_TASK_CLASSES = frozenset({"conversation", "coding", "vision"})


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
    routing_task_class: str | None = None

    def __post_init__(self) -> None:
        if not _TRANSPORT_PATTERN.fullmatch(self.transport):
            raise ValueError("transport must be a lowercase identifier")
        for field_name in ("update_id", "message_id", "sender_subject", "channel_subject"):
            value = getattr(self, field_name)
            if not value or value != value.strip():
                raise ValueError(f"{field_name} must be non-empty without surrounding whitespace")
        if not self.body:
            raise ValueError("body must be non-empty")
        if self.routing_task_class is not None and self.routing_task_class not in _ROUTING_TASK_CLASSES:
            raise ValueError("routing_task_class must be conversation, coding, vision, or None")
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
    thread_root_id: MessageId | None = None
    artifact_source_unique_id: str | None = None
    routing_task_class: str | None = None

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
        if self.thread_root_id is not None and not isinstance(self.thread_root_id, MessageId):
            raise ValueError("thread_root_id must be a MessageId or None")
        if self.artifact_source_unique_id is not None and (
            not isinstance(self.artifact_source_unique_id, str)
            or not self.artifact_source_unique_id
            or len(self.artifact_source_unique_id) > 512
        ):
            raise ValueError("artifact_source_unique_id must be bounded or None")
        if self.routing_task_class is not None and self.routing_task_class not in _ROUTING_TASK_CLASSES:
            raise ValueError("routing_task_class must be conversation, coding, vision, or None")
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class ScheduledInboundMessage:
    """One core-owned scheduled command resolved through canonical ownership."""

    principal_id: PrincipalId
    channel_id: ChannelId
    job_id: int
    occurrence_id: str
    body: str
    occurred_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.principal_id, PrincipalId):
            raise ValueError("principal_id must be a PrincipalId")
        if not isinstance(self.channel_id, ChannelId):
            raise ValueError("channel_id must be a ChannelId")
        if isinstance(self.job_id, bool) or not isinstance(self.job_id, int) or self.job_id <= 0:
            raise ValueError("job_id must be a positive integer")
        if not _CLIENT_MESSAGE_ID_PATTERN.fullmatch(self.occurrence_id):
            raise ValueError("occurrence_id must be a bounded opaque identifier")
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


def _mention_payload(mentions: tuple[MessageMention, ...]) -> list[dict[str, object]]:
    return [
        {
            "principal_id": mention.principal_id,
            "kind": mention.kind,
            "start": mention.start,
            "length": mention.length,
        }
        for mention in mentions
    ]


async def resolve_message_mentions(
    store: WorkshopEventStore,
    channel_id: ChannelId,
    body: str,
) -> tuple[MessageMention, ...]:
    """Resolve channel-member display names against one accepted message body."""
    async with store.connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'agent_definitions'"
    ) as cursor:
        definitions_supported = await cursor.fetchone() is not None
    member_query = (
        "SELECT p.id, p.kind, COALESCE(ad.handle, p.display_name) "
        "FROM channel_memberships cm "
        "JOIN channels c ON c.id = cm.channel_id "
        "JOIN principals p ON p.id = cm.principal_id "
        "LEFT JOIN agents a ON a.principal_id = p.id "
        "LEFT JOIN agent_definitions ad ON ad.agent_id = a.id "
        "LEFT JOIN channel_agents ca ON ca.channel_id = cm.channel_id "
        "AND ca.agent_id = a.id AND ca.detached_at IS NULL "
        "LEFT JOIN principal_agent_enablements sponsored "
        "ON sponsored.principal_id = ca.sponsor_principal_id "
        "AND sponsored.agent_id = ca.agent_id "
        "AND sponsored.runtime_profile_id = ca.sponsored_runtime_profile_id "
        "AND sponsored.lifecycle_state = 'enabled' "
        "WHERE cm.channel_id = ? "
        "AND (p.kind = 'human' OR (p.kind = 'agent' AND "
        "(c.kind = 'direct' OR sponsored.id IS NOT NULL))) "
        "ORDER BY length(COALESCE(ad.handle, p.display_name)) DESC, "
        "COALESCE(ad.handle, p.display_name), p.id"
        if definitions_supported
        else "SELECT p.id, p.kind, p.display_name FROM channel_memberships cm "
        "JOIN principals p ON p.id = cm.principal_id "
        "WHERE cm.channel_id = ? AND p.kind IN ('human', 'agent') "
        "ORDER BY length(p.display_name) DESC, p.display_name, p.id"
    )
    async with store.connection.execute(
        member_query,
        (channel_id,),
    ) as cursor:
        rows = list(await cursor.fetchall())

    # A duplicated case-insensitive display name is ambiguous and therefore
    # cannot safely identify either principal.
    candidates: list[tuple[PrincipalId, str, str]] = []
    grouped: dict[str, list[tuple[PrincipalId, str, str]]] = {}
    for row in rows:
        candidate = (PrincipalId(str(row[0])), str(row[1]), str(row[2]))
        grouped.setdefault(candidate[2].casefold(), []).append(candidate)
    for matches in grouped.values():
        if len(matches) == 1:
            candidates.append(matches[0])
    candidates.sort(key=lambda item: (-len(item[2]), item[2].casefold(), item[0]))

    mentions: list[MessageMention] = []
    cursor_position = 0
    while True:
        start = body.find("@", cursor_position)
        if start < 0:
            break
        if start > 0 and (body[start - 1].isalnum() or body[start - 1] == "_"):
            cursor_position = start + 1
            continue
        matched: MessageMention | None = None
        for principal_id, kind, display_name in candidates:
            end = start + 1 + len(display_name)
            if end > len(body):
                continue
            if body[start + 1 : end].casefold() != display_name.casefold():
                continue
            if end < len(body) and (body[end].isalnum() or body[end] == "_"):
                continue
            matched = MessageMention(principal_id, kind, start, end - start)
            break
        if matched is None:
            cursor_position = start + 1
            continue
        mentions.append(matched)
        cursor_position = matched.start + matched.length
    return tuple(mentions)


def _existing_mentions(envelope: EventEnvelope) -> tuple[MessageMention, ...] | None:
    raw_mentions = envelope.payload.get("mentions")
    if raw_mentions is None:
        return None
    if not isinstance(raw_mentions, list):
        raise RuntimeError("Stored message mentions are malformed")
    mentions: list[MessageMention] = []
    for value in raw_mentions:
        if not isinstance(value, dict):
            raise RuntimeError("Stored message mention is malformed")
        try:
            mentions.append(
                MessageMention(
                    principal_id=PrincipalId(value["principal_id"]),
                    kind=value["kind"],
                    start=value["start"],
                    length=value["length"],
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Stored message mention is malformed") from exc
    return tuple(mentions)


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


def _inbound_envelope(
    binding: _ResolvedInboundBinding,
    message: InboundMessage,
    mentions: tuple[MessageMention, ...] | None,
) -> EventEnvelope:
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
            **({"mentions": _mention_payload(mentions)} if mentions is not None else {}),
        },
        metadata={
            "source": message.transport,
            "transport_update_id": message.update_id,
            "transport_message_id": message.message_id,
            **({"routing_task_class": message.routing_task_class} if message.routing_task_class is not None else {}),
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


async def _resolve_scheduled_binding(
    store: WorkshopEventStore,
    message: ScheduledInboundMessage,
) -> _ResolvedInboundBinding:
    async with store.connection.execute(
        "SELECT c.workshop_id FROM workshop_scheduled_jobs j "
        "JOIN channels c ON c.id = j.channel_id "
        "JOIN channel_memberships cm ON cm.channel_id = c.id AND cm.principal_id = j.principal_id "
        "WHERE j.id = ? AND j.principal_id = ? AND j.channel_id = ?",
        (message.job_id, message.principal_id, message.channel_id),
    ) as cursor:
        rows = list(await cursor.fetchall())
    if len(rows) != 1:
        raise InboundBindingNotFoundError("Scheduled job does not have one canonical owner and channel")
    return _ResolvedInboundBinding(
        workshop_id=WorkshopId(str(rows[0][0])),
        principal_id=message.principal_id,
        channel_id=message.channel_id,
    )


def _client_inbound_envelope(
    binding: _ResolvedInboundBinding,
    message: ClientInboundMessage,
    mentions: tuple[MessageMention, ...] | None,
) -> EventEnvelope:
    stable_name = f"client-message:{message.principal_id}:{message.channel_id}:{message.client_message_id}"
    message_id = MessageId.derived(binding.workshop_id, stable_name)
    metadata: dict[str, object] = {
        "source": "workshop_client",
        "client_message_id": message.client_message_id,
    }
    if message.artifact_source_unique_id is not None:
        metadata["artifact_source_unique_id"] = message.artifact_source_unique_id
    if message.routing_task_class is not None:
        metadata["routing_task_class"] = message.routing_task_class
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
            **({"thread_root_id": message.thread_root_id} if message.thread_root_id is not None else {}),
            **({"mentions": _mention_payload(mentions)} if mentions is not None else {}),
        },
        metadata=metadata,
    )


def _scheduled_inbound_envelope(
    binding: _ResolvedInboundBinding,
    message: ScheduledInboundMessage,
    mentions: tuple[MessageMention, ...] | None,
) -> EventEnvelope:
    stable_name = f"scheduled-job:{message.job_id}:{message.occurrence_id}"
    message_id = MessageId.derived(binding.workshop_id, stable_name)
    return EventEnvelope.create(
        event_id=EventId.derived(binding.workshop_id, f"scheduled-job-event:{message_id}"),
        event_type=WorkshopEventType.MESSAGE_CREATED,
        event_version=1,
        workshop_id=binding.workshop_id,
        aggregate_type="message",
        aggregate_id=message_id,
        actor_principal_id=binding.principal_id,
        occurred_at=message.occurred_at,
        idempotency_key=f"workshop-scheduled-job:v1:{message_id}",
        payload={
            "channel_id": binding.channel_id,
            "author_principal_id": binding.principal_id,
            "body": message.body,
            **({"mentions": _mention_payload(mentions)} if mentions is not None else {}),
        },
        metadata={
            "source": "scheduled_job",
            "job_id": message.job_id,
            "occurrence_id": message.occurrence_id,
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
    candidate = _inbound_envelope(binding, message, ())
    if candidate.idempotency_key is None:
        raise RuntimeError("Inbound message envelope did not define an idempotency key")
    existing = await store.event_by_idempotency_key(candidate.idempotency_key)
    if existing is None:
        mentions = await resolve_message_mentions(store, binding.channel_id, message.body)
    else:
        mentions = _existing_mentions(existing.envelope)
        message = replace(message, occurred_at=existing.envelope.occurred_at)
    result = await store.append_in_transaction(_inbound_envelope(binding, message, mentions))
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
    envelope = _client_inbound_envelope(binding, message, ())
    if envelope.idempotency_key is None:
        raise RuntimeError("Client message envelope did not define an idempotency key")
    existing = await store.event_by_idempotency_key(envelope.idempotency_key)
    if existing is None:
        mentions = await resolve_message_mentions(store, binding.channel_id, message.body)
    else:
        mentions = _existing_mentions(existing.envelope)
        message = replace(message, occurred_at=existing.envelope.occurred_at)
    envelope = _client_inbound_envelope(binding, message, mentions)
    result = await store.append_in_transaction(envelope)
    await store.project_pending_in_transaction(CanonicalConversationProjection())
    return result


async def record_scheduled_inbound_message_in_transaction(
    store: WorkshopEventStore,
    message: ScheduledInboundMessage,
) -> AppendResult:
    """Append one core-owned scheduled command inside a caller transaction."""
    if not store.connection.in_transaction:
        raise RuntimeError("record_scheduled_inbound_message_in_transaction requires an active transaction")
    binding = await _resolve_scheduled_binding(store, message)
    envelope = _scheduled_inbound_envelope(binding, message, ())
    if envelope.idempotency_key is None:
        raise RuntimeError("Scheduled message envelope did not define an idempotency key")
    existing = await store.event_by_idempotency_key(envelope.idempotency_key)
    if existing is None:
        mentions = await resolve_message_mentions(store, binding.channel_id, message.body)
    else:
        mentions = _existing_mentions(existing.envelope)
        message = replace(message, occurred_at=existing.envelope.occurred_at)
    envelope = _scheduled_inbound_envelope(binding, message, mentions)
    result = await store.append_in_transaction(envelope)
    await store.project_pending_in_transaction(CanonicalConversationProjection())
    return result


async def record_inbound_message(store: WorkshopEventStore, message: InboundMessage) -> AppendResult:
    """Append and project one authenticated inbound transport message."""
    binding = await _resolve_binding(store, message)
    candidate = _inbound_envelope(binding, message, ())
    if candidate.idempotency_key is None:
        raise RuntimeError("Inbound message envelope did not define an idempotency key")
    existing = await store.event_by_idempotency_key(candidate.idempotency_key)
    if existing is None:
        mentions = await resolve_message_mentions(store, binding.channel_id, message.body)
    else:
        mentions = _existing_mentions(existing.envelope)
        message = replace(message, occurred_at=existing.envelope.occurred_at)
    result = await store.append(_inbound_envelope(binding, message, mentions))
    await store.project_pending(CanonicalConversationProjection())
    return result
