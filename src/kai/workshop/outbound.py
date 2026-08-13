"""Assistant results, atomic delivery requests, and delivery observations."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime

from kai.telegram_utils import chunk_text
from kai.workshop.delivery_authority import WorkshopConversationDeliveryAuthority
from kai.workshop.delivery_fragments import (
    EDIT_OPERATION,
    SEND_OPERATION,
    DeliveryFragmentOperation,
    DeliveryFragmentPlanResult,
    WorkshopDeliveryFragments,
)
from kai.workshop.delivery_outbox import (
    CONVERSATION_REPLY_PURPOSE,
    STREAMING_FINALIZATION_CONTRACT,
    DeliveryRequest,
    DeliveryRequestResult,
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
from kai.workshop.store import AppendResult, IdempotencyConflictError, WorkshopEventStore
from kai.workshop.streaming_preview import resolve_telegram_finalization_target

_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class OutboundMessageNotFoundError(LookupError):
    """A referenced canonical message or its unique agent binding was not found."""


class OutboundDeliveryBindingError(LookupError):
    """The reply channel does not have exactly one canonical Telegram binding."""


class OutboundDeliveryStateConflictError(RuntimeError):
    """Only one half of an atomic outbound message and delivery already exists."""


class OutboundStreamingPreviewConflictError(RuntimeError):
    """A persisted preview no longer matches its canonical routing identities."""


@dataclass(frozen=True, slots=True)
class OutboundMessage:
    in_reply_to_message_id: MessageId
    body: str
    occurred_at: datetime

    def __post_init__(self) -> None:
        if not isinstance(self.in_reply_to_message_id, MessageId):
            raise ValueError("in_reply_to_message_id must be a MessageId")
        if not self.body:
            raise ValueError("body must be non-empty")
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")


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
    agent_principal_id: PrincipalId


@dataclass(frozen=True, slots=True)
class OutboundDeliveryResult:
    message: AppendResult
    delivery: DeliveryRequestResult


@dataclass(frozen=True, slots=True)
class OutboundStreamingFinalizationResult:
    message: AppendResult
    delivery: DeliveryRequestResult
    plan: DeliveryFragmentPlanResult


async def _resolve_outbound(store: WorkshopEventStore, message_id: MessageId) -> _ResolvedOutbound:
    async with store.connection.execute(
        "SELECT c.workshop_id, m.channel_id, a.principal_id "
        "FROM messages m "
        "JOIN principals author ON author.id = m.author_principal_id AND author.kind = 'human' "
        "JOIN channels c ON c.id = m.channel_id "
        "JOIN channel_agents ca ON ca.channel_id = c.id "
        "JOIN agents a ON a.id = ca.agent_id AND a.workshop_id = c.workshop_id "
        "WHERE m.id = ?",
        (message_id,),
    ) as cursor:
        rows = list(await cursor.fetchall())
    if len(rows) != 1:
        raise OutboundMessageNotFoundError("Inbound message and agent binding do not resolve uniquely")
    return _ResolvedOutbound(
        workshop_id=WorkshopId(str(rows[0][0])),
        channel_id=ChannelId(str(rows[0][1])),
        agent_principal_id=PrincipalId(str(rows[0][2])),
    )


def _outbound_key(message_id: MessageId) -> str:
    return f"workshop-outbound:v1:{message_id}"


def _outbound_payload(binding: _ResolvedOutbound, message: OutboundMessage) -> dict[str, object]:
    return {
        "channel_id": binding.channel_id,
        "author_principal_id": binding.agent_principal_id,
        "reply_to_message_id": message.in_reply_to_message_id,
        "body": message.body,
    }


def _outbound_envelope(binding: _ResolvedOutbound, message: OutboundMessage) -> EventEnvelope:
    return EventEnvelope.create(
        event_id=EventId.derived(binding.workshop_id, f"outbound-message-event:{message.in_reply_to_message_id}"),
        event_type=WorkshopEventType.MESSAGE_CREATED,
        event_version=1,
        workshop_id=binding.workshop_id,
        aggregate_type="message",
        aggregate_id=MessageId.derived(
            binding.workshop_id,
            f"outbound-message:{message.in_reply_to_message_id}",
        ),
        actor_principal_id=binding.agent_principal_id,
        occurred_at=message.occurred_at,
        idempotency_key=_outbound_key(message.in_reply_to_message_id),
        payload=_outbound_payload(binding, message),
        metadata={"source": "agent"},
    )


async def _existing_outbound(
    store: WorkshopEventStore,
    binding: _ResolvedOutbound,
    message: OutboundMessage,
) -> AppendResult | None:
    key = _outbound_key(message.in_reply_to_message_id)
    existing = await store.event_by_idempotency_key(key)
    if existing is None:
        return None
    if (
        existing.envelope.event_type != WorkshopEventType.MESSAGE_CREATED
        or existing.envelope.payload != _outbound_payload(binding, message)
        or existing.envelope.actor_principal_id != binding.agent_principal_id
    ):
        raise IdempotencyConflictError(f"Event identity {key!r} was reused with different content")
    return AppendResult(event=existing, inserted=False)


async def _resolve_telegram_binding(
    store: WorkshopEventStore,
    channel_id: ChannelId,
) -> ChannelBindingId:
    async with store.connection.execute(
        "SELECT id FROM channel_bindings WHERE channel_id = ? AND transport = 'telegram' ORDER BY id",
        (channel_id,),
    ) as cursor:
        rows = list(await cursor.fetchall())
    if len(rows) != 1:
        raise OutboundDeliveryBindingError("Canonical reply channel must have exactly one Telegram binding")
    return ChannelBindingId(str(rows[0][0]))


async def _confirmed_preview_message_id(
    store: WorkshopEventStore,
    *,
    inbound_message_id: MessageId,
    workshop_id: WorkshopId,
    channel_id: ChannelId,
    channel_binding_id: ChannelBindingId,
) -> int | None:
    async with store.connection.execute(
        "SELECT workshop_id, channel_id, channel_binding_id, external_message_id, state "
        "FROM telegram_streaming_previews WHERE inbound_message_id = ?",
        (inbound_message_id,),
    ) as cursor:
        row = await cursor.fetchone()
    if row is None:
        return None
    if (
        str(row[0]) != workshop_id
        or str(row[1]) != channel_id
        or str(row[2]) != channel_binding_id
        or str(row[4]) != "confirmed_non_final"
    ):
        raise OutboundStreamingPreviewConflictError(
            "Confirmed streaming preview does not match the canonical reply target"
        )
    external_message_id = int(row[3])
    if external_message_id <= 0:
        raise OutboundStreamingPreviewConflictError("Confirmed streaming preview has an invalid message ID")
    return external_message_id


def _streaming_finalization_operations(
    body: str,
    *,
    preview_message_id: int | None,
) -> tuple[DeliveryFragmentOperation, ...]:
    bodies = tuple(chunk_text(body, max_len=4096))
    if preview_message_id is None:
        return tuple(DeliveryFragmentOperation(SEND_OPERATION, fragment) for fragment in bodies)
    return (
        DeliveryFragmentOperation(
            EDIT_OPERATION,
            bodies[0],
            target_external_message_id=preview_message_id,
        ),
        *(DeliveryFragmentOperation(SEND_OPERATION, fragment) for fragment in bodies[1:]),
    )


async def record_outbound_message(store: WorkshopEventStore, message: OutboundMessage) -> AppendResult:
    """Append one canonical assistant reply to an existing inbound message."""
    binding = await _resolve_outbound(store, message.in_reply_to_message_id)
    existing = await _existing_outbound(store, binding, message)
    if existing is not None:
        await store.project_pending(CanonicalConversationProjection())
        return existing

    result = await store.append(_outbound_envelope(binding, message))
    await store.project_pending(CanonicalConversationProjection())
    return result


async def record_outbound_message_with_delivery(
    store: WorkshopEventStore,
    message: OutboundMessage,
) -> OutboundDeliveryResult:
    """Atomically create one canonical reply and its pending Telegram text delivery.

    This service is deliberately production-unused. It accepts no transport or
    destination identity from its caller and does not send or register a worker.
    """
    connection = store.connection
    try:
        await connection.execute("BEGIN IMMEDIATE")
        binding = await _resolve_outbound(store, message.in_reply_to_message_id)
        channel_binding_id = await _resolve_telegram_binding(store, binding.channel_id)
        message_result = await _existing_outbound(store, binding, message)
        if message_result is None:
            message_result = await store.append_in_transaction(_outbound_envelope(binding, message))

        projection = CanonicalConversationProjection()
        await store.project_pending_in_transaction(projection)
        message_id = message_result.event.envelope.aggregate_id
        if not isinstance(message_id, MessageId):
            raise RuntimeError("Canonical outbound event did not identify a message")
        delivery_result = await WorkshopDeliveryOutbox(store).request_delivery_in_transaction(
            DeliveryRequest(
                message_id=message_id,
                channel_binding_id=channel_binding_id,
                mode="text",
                purpose=CONVERSATION_REPLY_PURPOSE,
                occurred_at=message.occurred_at,
                max_attempts=5,
            )
        )
        if message_result.inserted != delivery_result.inserted:
            raise OutboundDeliveryStateConflictError(
                "Canonical reply and delivery request did not share one prior state"
            )
        await store.project_pending_in_transaction(projection)
        await connection.commit()
        return OutboundDeliveryResult(message=message_result, delivery=delivery_result)
    except Exception:
        await connection.rollback()
        raise


async def record_outbound_message_with_streaming_finalization(
    store: WorkshopEventStore,
    message: OutboundMessage,
) -> OutboundStreamingFinalizationResult:
    """Atomically persist a reply, delivery request, and immutable operation plan.

    The authenticated private-text route calls this service after agent
    completion. Its only routing input is the canonical inbound message ID. It
    resolves the direct Telegram binding and any confirmed non-final preview
    internally, and it neither sends nor edits Telegram itself.
    """
    connection = store.connection
    try:
        await connection.execute("BEGIN IMMEDIATE")
        result = await record_outbound_message_with_streaming_finalization_in_transaction(store, message)
        await connection.commit()
        return result
    except Exception:
        await connection.rollback()
        raise


async def record_outbound_message_with_streaming_finalization_in_transaction(
    store: WorkshopEventStore,
    message: OutboundMessage,
) -> OutboundStreamingFinalizationResult:
    """Persist a reply and immutable delivery plan in a caller-owned transaction."""
    if not store.connection.in_transaction:
        raise RuntimeError(
            "record_outbound_message_with_streaming_finalization_in_transaction requires an active transaction"
        )
    binding = await _resolve_outbound(store, message.in_reply_to_message_id)
    streaming_target = await resolve_telegram_finalization_target(
        store,
        message.in_reply_to_message_id,
    )
    if streaming_target.workshop_id != binding.workshop_id or streaming_target.channel_id != binding.channel_id:
        raise OutboundStreamingPreviewConflictError(
            "Telegram streaming target does not match the canonical agent reply target"
        )
    preview_message_id = (
        await _confirmed_preview_message_id(
            store,
            inbound_message_id=message.in_reply_to_message_id,
            workshop_id=binding.workshop_id,
            channel_id=binding.channel_id,
            channel_binding_id=streaming_target.channel_binding_id,
        )
        if streaming_target.preview_eligible
        else None
    )
    operations = _streaming_finalization_operations(
        message.body,
        preview_message_id=preview_message_id,
    )
    authority_epoch = await WorkshopConversationDeliveryAuthority(store).active_epoch_in_transaction()
    assert authority_epoch is not None
    message_result = await _existing_outbound(store, binding, message)
    if message_result is None:
        message_result = await store.append_in_transaction(_outbound_envelope(binding, message))

    projection = CanonicalConversationProjection()
    await store.project_pending_in_transaction(projection)
    message_id = message_result.event.envelope.aggregate_id
    if not isinstance(message_id, MessageId):
        raise RuntimeError("Canonical outbound event did not identify a message")
    delivery_result = await WorkshopDeliveryOutbox(store).request_delivery_in_transaction(
        DeliveryRequest(
            message_id=message_id,
            channel_binding_id=streaming_target.channel_binding_id,
            mode="text",
            purpose=CONVERSATION_REPLY_PURPOSE,
            occurred_at=message.occurred_at,
            execution_contract=STREAMING_FINALIZATION_CONTRACT,
            authority_epoch_id=authority_epoch.epoch_id,
            max_attempts=5,
        )
    )
    plan_result = await WorkshopDeliveryFragments(store).prepare_operations_in_transaction(
        delivery_result.delivery.delivery_id,
        operations,
        occurred_at=message.occurred_at,
    )
    prior_states = {
        message_result.inserted,
        delivery_result.inserted,
        plan_result.inserted,
    }
    if len(prior_states) != 1:
        raise OutboundDeliveryStateConflictError(
            "Canonical reply, delivery request, and operation plan did not share one prior state"
        )
    await store.project_pending_in_transaction(projection)
    return OutboundStreamingFinalizationResult(
        message=message_result,
        delivery=delivery_result,
        plan=plan_result,
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
