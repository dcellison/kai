"""Transport-independent Kai Workshop domain and persistence foundations.

The current production service seeds this state and shadow-records accepted
inbound text, successful assistant results, and delivery observations. Existing
Telegram routing, histories, and responses remain authoritative until replay
and parity evidence supports an explicit cutover.
"""

from kai.workshop.domain import (
    AgentId,
    ChannelAgentId,
    ChannelBindingId,
    ChannelId,
    DeliveryId,
    EventEnvelope,
    EventId,
    ExternalIdentityId,
    MessageId,
    PrincipalId,
    WorkshopEventType,
    WorkshopId,
    WorkshopMembershipId,
)
from kai.workshop.store import (
    AppendResult,
    EventIntegrityError,
    IdempotencyConflictError,
    Projection,
    ProjectionCheckpoint,
    StoredEvent,
    WorkshopEventStore,
)

__all__ = [
    "AgentId",
    "AppendResult",
    "ChannelAgentId",
    "ChannelBindingId",
    "ChannelId",
    "DeliveryId",
    "EventEnvelope",
    "EventId",
    "EventIntegrityError",
    "ExternalIdentityId",
    "IdempotencyConflictError",
    "MessageId",
    "PrincipalId",
    "Projection",
    "ProjectionCheckpoint",
    "StoredEvent",
    "WorkshopEventStore",
    "WorkshopEventType",
    "WorkshopId",
    "WorkshopMembershipId",
]
