"""Transport-neutral planning of durable delivery intents."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from kai.workshop.delivery_authority import WorkshopConversationDeliveryAuthority
from kai.workshop.delivery_outbox import (
    CONVERSATION_REPLY_PURPOSE,
    SEND_FRAGMENTS_CONTRACT,
    STREAMING_FINALIZATION_CONTRACT,
    DeliveryPurpose,
    DeliveryRequest,
    DeliveryRequestResult,
    WorkshopDeliveryOutbox,
)
from kai.workshop.delivery_policy import WorkshopDeliveryBindingPolicy
from kai.workshop.domain import ChannelId, MessageId, PrincipalId
from kai.workshop.store import WorkshopEventStore

DeliveryContentKind = Literal["text", "attachment"]


@dataclass(frozen=True, slots=True)
class CanonicalDeliveryIntent:
    """Canonical publication facts from which adapter work may be planned."""

    message_id: MessageId
    channel_id: ChannelId
    mode: str
    purpose: DeliveryPurpose
    occurred_at: datetime
    content_kind: DeliveryContentKind = "text"
    recipient_principal_id: PrincipalId | None = None
    preview_eligible: bool = False
    max_attempts: int = 5

    def __post_init__(self) -> None:
        if not isinstance(self.message_id, MessageId):
            raise ValueError("message_id must be a MessageId")
        if not isinstance(self.channel_id, ChannelId):
            raise ValueError("channel_id must be a ChannelId")
        if self.content_kind not in {"text", "attachment"}:
            raise ValueError("content_kind must be text or attachment")
        if self.recipient_principal_id is not None and not isinstance(
            self.recipient_principal_id,
            PrincipalId,
        ):
            raise ValueError("recipient_principal_id must be a PrincipalId when supplied")
        if not isinstance(self.preview_eligible, bool):
            raise ValueError("preview_eligible must be a boolean")


@dataclass(frozen=True, slots=True)
class DeliveryPlanningResult:
    """All independently durable adapter requests for one publication."""

    deliveries: tuple[DeliveryRequestResult, ...]


class WorkshopDeliveryPlanner:
    """Fan out one canonical intent according to enabled adapter capabilities."""

    def __init__(
        self,
        store: WorkshopEventStore,
        policy: WorkshopDeliveryBindingPolicy,
    ) -> None:
        self._store = store
        self._policy = policy
        self._outbox = WorkshopDeliveryOutbox(store)

    async def plan_in_transaction(
        self,
        intent: CanonicalDeliveryIntent,
    ) -> DeliveryPlanningResult:
        if not self._store.connection.in_transaction:
            raise RuntimeError("plan_in_transaction requires an active transaction")
        bindings = await self._policy.bindings(
            self._store,
            intent.channel_id,
            principal_id=intent.recipient_principal_id,
        )
        selected = tuple(
            binding
            for binding in bindings
            if (
                (intent.content_kind == "text" and binding.capabilities.final_text)
                or (intent.content_kind == "attachment" and binding.capabilities.attachments)
            )
        )
        needs_streaming_authority = any(
            intent.purpose == CONVERSATION_REPLY_PURPOSE
            and intent.preview_eligible
            and binding.capabilities.preview_streaming
            for binding in selected
        )
        authority_epoch_id = None
        if needs_streaming_authority:
            authority_epoch = await WorkshopConversationDeliveryAuthority(self._store).active_epoch_in_transaction()
            assert authority_epoch is not None
            authority_epoch_id = authority_epoch.epoch_id

        deliveries: list[DeliveryRequestResult] = []
        for binding in selected:
            streaming = (
                intent.purpose == CONVERSATION_REPLY_PURPOSE
                and intent.preview_eligible
                and binding.capabilities.preview_streaming
            )
            deliveries.append(
                await self._outbox.request_delivery_in_transaction(
                    DeliveryRequest(
                        message_id=intent.message_id,
                        channel_binding_id=binding.binding_id,
                        mode=intent.mode,
                        purpose=intent.purpose,
                        occurred_at=intent.occurred_at,
                        execution_contract=(STREAMING_FINALIZATION_CONTRACT if streaming else SEND_FRAGMENTS_CONTRACT),
                        authority_epoch_id=authority_epoch_id if streaming else None,
                        max_attempts=intent.max_attempts,
                    )
                )
            )
        return DeliveryPlanningResult(tuple(deliveries))
