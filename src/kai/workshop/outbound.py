"""Assistant results, atomic delivery requests, and delivery observations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from kai.workshop.delivery_outbox import (
    CONVERSATION_REPLY_PURPOSE,
    DeliveryRequestResult,
)
from kai.workshop.delivery_planning import CanonicalDeliveryIntent, WorkshopDeliveryPlanner
from kai.workshop.delivery_policy import WorkshopDeliveryBindingPolicy
from kai.workshop.domain import (
    AgentId,
    ChannelId,
    DeliveryId,
    EventEnvelope,
    EventId,
    MessageId,
    PrincipalId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.human_notifications import append_human_notifications_in_transaction
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import AppendResult, IdempotencyConflictError, WorkshopEventStore

_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class OutboundMessageNotFoundError(LookupError):
    """A referenced canonical message or its unique agent binding was not found."""


class OutboundDeliveryStateConflictError(RuntimeError):
    """Only one half of an atomic outbound message and delivery already exists."""


@dataclass(frozen=True, slots=True)
class OutboundMessage:
    in_reply_to_message_id: MessageId
    body: str
    occurred_at: datetime
    agent_id: AgentId | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.in_reply_to_message_id, MessageId):
            raise ValueError("in_reply_to_message_id must be a MessageId")
        if not self.body:
            raise ValueError("body must be non-empty")
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        if self.agent_id is not None and not isinstance(self.agent_id, AgentId):
            raise ValueError("agent_id must be an AgentId when provided")


@dataclass(frozen=True, slots=True)
class DeliveryObservation:
    message_id: MessageId
    transport: str
    mode: str
    succeeded: bool
    occurred_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.message_id, MessageId):
            raise ValueError("message_id must be a MessageId")
        for field_name in ("transport", "mode"):
            if not _IDENTIFIER_PATTERN.fullmatch(getattr(self, field_name)):
                raise ValueError(f"{field_name} must be a lowercase identifier")
        if not isinstance(self.succeeded, bool):
            raise ValueError("succeeded must be a boolean")
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")


@dataclass(frozen=True, slots=True)
class _ResolvedOutbound:
    workshop_id: WorkshopId
    channel_id: ChannelId
    agent_id: AgentId
    agent_principal_id: PrincipalId
    recipient_principal_id: PrincipalId
    thread_root_id: MessageId | None


@dataclass(frozen=True, slots=True)
class OutboundDeliveryResult:
    message: AppendResult
    deliveries: tuple[DeliveryRequestResult, ...]

    @property
    def delivery(self) -> DeliveryRequestResult | None:
        return self.deliveries[0] if len(self.deliveries) == 1 else None


@dataclass(frozen=True, slots=True)
class OutboundStreamingFinalizationResult:
    message: AppendResult
    deliveries: tuple[DeliveryRequestResult, ...]

    @property
    def delivery(self) -> DeliveryRequestResult | None:
        return self.deliveries[0] if len(self.deliveries) == 1 else None

    @property
    def plan(self) -> None:
        """Adapter operation plans are no longer created by core finalization."""
        return None


async def _resolve_outbound(
    store: WorkshopEventStore,
    message_id: MessageId,
    agent_id: AgentId | None = None,
) -> _ResolvedOutbound:
    async with store.connection.execute(
        "SELECT 1 FROM pragma_table_info('messages') WHERE name = 'thread_root_id'"
    ) as schema_cursor:
        has_thread_root = await schema_cursor.fetchone() is not None
    thread_root_expression = "m.thread_root_id" if has_thread_root else "NULL"
    async with store.connection.execute(
        "SELECT c.workshop_id, m.channel_id, ca.agent_id, a.principal_id, "
        "m.author_principal_id, "
        f"{thread_root_expression} "
        "FROM messages m "
        "JOIN principals author ON author.id = m.author_principal_id AND author.kind = 'human' "
        "JOIN channels c ON c.id = m.channel_id "
        "JOIN channel_agents ca ON ca.channel_id = c.id "
        "JOIN agents a ON a.id = ca.agent_id AND a.workshop_id = c.workshop_id "
        "WHERE m.id = ? AND (? IS NULL OR ca.agent_id = ?)",
        (message_id, agent_id, agent_id),
    ) as cursor:
        rows = list(await cursor.fetchall())
    if not rows and agent_id is not None:
        async with store.connection.execute(
            "SELECT c.workshop_id, m.channel_id, r.agent_id, target.principal_id, "
            "r.requested_by_principal_id, "
            f"{thread_root_expression} "
            "FROM messages m "
            "JOIN principals author ON author.id = m.author_principal_id "
            "AND author.kind = 'agent' "
            "JOIN runs r ON r.inbound_message_id = m.id AND r.agent_id = ? "
            "AND r.delegation_id IS NOT NULL "
            "JOIN agent_delegations d ON d.id = r.delegation_id "
            "AND d.request_message_id = m.id AND d.child_run_id = r.id "
            "JOIN channels c ON c.id = m.channel_id "
            "JOIN agents target ON target.id = r.agent_id "
            "AND target.workshop_id = c.workshop_id "
            "WHERE m.id = ?",
            (agent_id, message_id),
        ) as cursor:
            rows = list(await cursor.fetchall())
    if len(rows) != 1:
        raise OutboundMessageNotFoundError("Inbound message and agent binding do not resolve uniquely")
    return _ResolvedOutbound(
        workshop_id=WorkshopId(str(rows[0][0])),
        channel_id=ChannelId(str(rows[0][1])),
        agent_id=AgentId(str(rows[0][2])),
        agent_principal_id=PrincipalId(str(rows[0][3])),
        recipient_principal_id=PrincipalId(str(rows[0][4])),
        thread_root_id=(MessageId(str(rows[0][5])) if rows[0][5] is not None else None),
    )


def _outbound_key(message: OutboundMessage) -> str:
    if message.agent_id is None:
        return f"workshop-outbound:v1:{message.in_reply_to_message_id}"
    return f"workshop-outbound:v2:{message.in_reply_to_message_id}:{message.agent_id}"


def _outbound_identity(message: OutboundMessage) -> str:
    if message.agent_id is None:
        return str(message.in_reply_to_message_id)
    return f"{message.in_reply_to_message_id}:agent:{message.agent_id}"


def _outbound_payload(binding: _ResolvedOutbound, message: OutboundMessage) -> dict[str, object]:
    return {
        "channel_id": binding.channel_id,
        "author_principal_id": binding.agent_principal_id,
        "reply_to_message_id": message.in_reply_to_message_id,
        "body": message.body,
        **({"thread_root_id": binding.thread_root_id} if binding.thread_root_id is not None else {}),
    }


def _outbound_envelope(binding: _ResolvedOutbound, message: OutboundMessage) -> EventEnvelope:
    identity = _outbound_identity(message)
    return EventEnvelope.create(
        event_id=EventId.derived(binding.workshop_id, f"outbound-message-event:{identity}"),
        event_type=WorkshopEventType.MESSAGE_CREATED,
        event_version=1,
        workshop_id=binding.workshop_id,
        aggregate_type="message",
        aggregate_id=MessageId.derived(
            binding.workshop_id,
            f"outbound-message:{identity}",
        ),
        actor_principal_id=binding.agent_principal_id,
        occurred_at=message.occurred_at,
        idempotency_key=_outbound_key(message),
        payload=_outbound_payload(binding, message),
        metadata={"source": "agent"},
    )


async def _existing_outbound(
    store: WorkshopEventStore,
    binding: _ResolvedOutbound,
    message: OutboundMessage,
) -> AppendResult | None:
    key = _outbound_key(message)
    existing = await store.event_by_idempotency_key(key)
    if existing is None and message.agent_id is not None:
        legacy_key = f"workshop-outbound:v1:{message.in_reply_to_message_id}"
        legacy = await store.event_by_idempotency_key(legacy_key)
        if legacy is not None and legacy.envelope.actor_principal_id == binding.agent_principal_id:
            existing = legacy
            key = legacy_key
    if existing is None:
        return None
    if (
        existing.envelope.event_type != WorkshopEventType.MESSAGE_CREATED
        or existing.envelope.payload != _outbound_payload(binding, message)
        or existing.envelope.actor_principal_id != binding.agent_principal_id
    ):
        raise IdempotencyConflictError(f"Event identity {key!r} was reused with different content")
    return AppendResult(event=existing, inserted=False)


async def record_outbound_message(store: WorkshopEventStore, message: OutboundMessage) -> AppendResult:
    """Append one canonical assistant reply to an existing inbound message."""
    try:
        await store.connection.execute("BEGIN IMMEDIATE")
        binding = await _resolve_outbound(store, message.in_reply_to_message_id, message.agent_id)
        result = await _existing_outbound(store, binding, message)
        if result is None:
            result = await store.append_in_transaction(_outbound_envelope(binding, message))
        projection = CanonicalConversationProjection()
        await store.project_pending_in_transaction(projection)
        if result.inserted:
            await append_human_notifications_in_transaction(store, result.event)
        await store.project_pending_in_transaction(projection)
        await store.connection.commit()
        return result
    except Exception:
        await store.connection.rollback()
        raise


async def record_outbound_message_with_delivery(
    store: WorkshopEventStore,
    message: OutboundMessage,
    *,
    delivery_policy: WorkshopDeliveryBindingPolicy,
) -> OutboundDeliveryResult:
    """Atomically create one canonical reply and eligible adapter deliveries.

    The service accepts no transport or destination identity from its caller;
    enabled adapter capabilities and canonical bindings determine delivery.
    """
    connection = store.connection
    try:
        await connection.execute("BEGIN IMMEDIATE")
        binding = await _resolve_outbound(store, message.in_reply_to_message_id, message.agent_id)
        message_result = await _existing_outbound(store, binding, message)
        if message_result is None:
            message_result = await store.append_in_transaction(_outbound_envelope(binding, message))

        projection = CanonicalConversationProjection()
        await store.project_pending_in_transaction(projection)
        if message_result.inserted:
            await append_human_notifications_in_transaction(store, message_result.event)
        message_id = message_result.event.envelope.aggregate_id
        if not isinstance(message_id, MessageId):
            raise RuntimeError("Canonical outbound event did not identify a message")
        planning = await WorkshopDeliveryPlanner(store, delivery_policy).plan_in_transaction(
            CanonicalDeliveryIntent(
                message_id=message_id,
                channel_id=binding.channel_id,
                mode="text",
                purpose=CONVERSATION_REPLY_PURPOSE,
                occurred_at=message.occurred_at,
                recipient_principal_id=binding.recipient_principal_id,
            )
        )
        if any(message_result.inserted != delivery.inserted for delivery in planning.deliveries):
            raise OutboundDeliveryStateConflictError(
                "Canonical reply and delivery request did not share one prior state"
            )
        await store.project_pending_in_transaction(projection)
        await connection.commit()
        return OutboundDeliveryResult(message=message_result, deliveries=planning.deliveries)
    except Exception:
        await connection.rollback()
        raise


async def record_outbound_message_with_streaming_finalization(
    store: WorkshopEventStore,
    message: OutboundMessage,
    *,
    delivery_policy: WorkshopDeliveryBindingPolicy,
) -> OutboundStreamingFinalizationResult:
    """Atomically persist a reply and capability-selected delivery requests.

    The authenticated private-text route calls this service after agent
    completion. Its only routing input is the canonical inbound message ID.
    Formatting, chunking, preview resolution, and provider operations belong to
    the adapter that claims each independently durable request.
    """
    connection = store.connection
    try:
        await connection.execute("BEGIN IMMEDIATE")
        result = await record_outbound_message_with_streaming_finalization_in_transaction(
            store,
            message,
            delivery_policy=delivery_policy,
        )
        await connection.commit()
        return result
    except Exception:
        await connection.rollback()
        raise


async def record_outbound_message_with_streaming_finalization_in_transaction(
    store: WorkshopEventStore,
    message: OutboundMessage,
    *,
    delivery_policy: WorkshopDeliveryBindingPolicy,
    request_delivery: bool = True,
) -> OutboundStreamingFinalizationResult:
    """Persist a reply and adapter delivery intents in a caller-owned transaction."""
    if not store.connection.in_transaction:
        raise RuntimeError(
            "record_outbound_message_with_streaming_finalization_in_transaction requires an active transaction"
        )
    binding = await _resolve_outbound(store, message.in_reply_to_message_id, message.agent_id)
    message_result = await _existing_outbound(store, binding, message)
    if message_result is None:
        message_result = await store.append_in_transaction(_outbound_envelope(binding, message))

    projection = CanonicalConversationProjection()
    await store.project_pending_in_transaction(projection)
    if message_result.inserted:
        await append_human_notifications_in_transaction(store, message_result.event)
    message_id = message_result.event.envelope.aggregate_id
    if not isinstance(message_id, MessageId):
        raise RuntimeError("Canonical outbound event did not identify a message")
    planning = (
        await WorkshopDeliveryPlanner(store, delivery_policy).plan_in_transaction(
            CanonicalDeliveryIntent(
                message_id=message_id,
                channel_id=binding.channel_id,
                mode="text",
                purpose=CONVERSATION_REPLY_PURPOSE,
                occurred_at=message.occurred_at,
                recipient_principal_id=binding.recipient_principal_id,
                preview_eligible=True,
            )
        )
        if request_delivery
        else None
    )
    deliveries = planning.deliveries if planning is not None else ()
    prior_states = {message_result.inserted}
    prior_states.update(delivery.inserted for delivery in deliveries)
    if len(prior_states) != 1:
        raise OutboundDeliveryStateConflictError("Canonical reply and delivery requests did not share one prior state")
    await store.project_pending_in_transaction(projection)
    return OutboundStreamingFinalizationResult(
        message=message_result,
        deliveries=deliveries,
    )


async def _resolve_delivery(store: WorkshopEventStore, message_id: MessageId) -> tuple[WorkshopId, ChannelId]:
    async with store.connection.execute(
        "SELECT c.workshop_id, m.channel_id FROM messages m JOIN channels c ON c.id = m.channel_id WHERE m.id = ?",
        (message_id,),
    ) as cursor:
        row = await cursor.fetchone()
    if row is None:
        raise OutboundMessageNotFoundError("Delivered canonical message was not found")
    return WorkshopId(str(row[0])), ChannelId(str(row[1]))


async def record_delivery_observation(
    store: WorkshopEventStore,
    observation: DeliveryObservation,
) -> AppendResult:
    """Append a non-authoritative observation of one transport delivery outcome."""
    workshop_id, channel_id = await _resolve_delivery(store, observation.message_id)
    status = "succeeded" if observation.succeeded else "failed"
    delivery_id = DeliveryId.derived(
        workshop_id,
        f"delivery:{observation.message_id}:{observation.transport}:{observation.mode}",
    )
    key = f"workshop-delivery:v1:{observation.message_id}:{observation.transport}:{observation.mode}:{status}"
    payload = {
        "message_id": observation.message_id,
        "channel_id": channel_id,
        "transport": observation.transport,
        "mode": observation.mode,
    }
    existing = await store.event_by_idempotency_key(key)
    if existing is not None:
        expected_type = (
            WorkshopEventType.DELIVERY_SUCCEEDED if observation.succeeded else WorkshopEventType.DELIVERY_FAILED
        )
        if existing.envelope.event_type != expected_type or existing.envelope.payload != payload:
            raise IdempotencyConflictError(f"Event identity {key!r} was reused with different content")
        await store.project_pending(CanonicalConversationProjection())
        return AppendResult(event=existing, inserted=False)

    result = await store.append(
        EventEnvelope.create(
            event_id=EventId.derived(
                workshop_id,
                f"delivery-event:{observation.message_id}:{observation.transport}:{observation.mode}:{status}",
            ),
            event_type=(
                WorkshopEventType.DELIVERY_SUCCEEDED if observation.succeeded else WorkshopEventType.DELIVERY_FAILED
            ),
            event_version=1,
            workshop_id=workshop_id,
            aggregate_type="delivery",
            aggregate_id=delivery_id,
            occurred_at=observation.occurred_at,
            idempotency_key=key,
            payload=payload,
            metadata={"source": "transport_observer"},
        )
    )
    await store.project_pending(CanonicalConversationProjection())
    return result
