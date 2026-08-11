"""Transport-independent Kai Workshop domain and persistence foundations.

This package is intentionally not wired into Kai's production startup or
message paths yet.  The bootstrap migration will integrate it after the
domain and event-store contracts have been reviewed independently.
"""

from kai.workshop.domain import (
    AgentId,
    ChannelAgentId,
    ChannelBindingId,
    ChannelId,
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
