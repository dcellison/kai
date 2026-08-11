"""Core, transport-independent value objects for Kai Workshop."""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, ClassVar, Self

ENVELOPE_VERSION = 1

_ID_BODY_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_EVENT_TYPE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]*)+$")
_AGGREGATE_TYPE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


class WorkshopEventType(StrEnum):
    """The collaboration-only vocabulary supported by the first schema."""

    WORKSHOP_CREATED = "workshop.created"
    WORKSHOP_MEMBER_ADDED = "workshop.member_added"
    PRINCIPAL_CREATED = "principal.created"
    EXTERNAL_IDENTITY_BOUND = "external_identity.bound"
    CHANNEL_CREATED = "channel.created"
    CHANNEL_MEMBER_ADDED = "channel.member_added"
    TRANSPORT_CHANNEL_BOUND = "transport.channel_bound"
    AGENT_CREATED = "agent.created"
    CHANNEL_AGENT_ATTACHED = "channel.agent_attached"
    MESSAGE_CREATED = "message.created"
    ARTIFACT_CREATED = "artifact.created"
    DELIVERY_REQUESTED = "delivery.requested"
    DELIVERY_SUCCEEDED = "delivery.succeeded"
    DELIVERY_FAILED = "delivery.failed"


class OpaqueId(str):
    """A globally unique identifier whose prefix identifies only its type."""

    prefix: ClassVar[str]

    def __new__(cls, value: str) -> Self:
        if not isinstance(value, str):
            raise TypeError(f"{cls.__name__} must be constructed from a string")
        expected_prefix = f"{cls.prefix}_"
        body = value.removeprefix(expected_prefix)
        if not value.startswith(expected_prefix) or not _ID_BODY_PATTERN.fullmatch(body):
            raise ValueError(f"Invalid {cls.__name__}: expected {expected_prefix}<32 lowercase hex characters>")
        return str.__new__(cls, value)

    @classmethod
    def new(cls) -> Self:
        return cls(f"{cls.prefix}_{uuid.uuid4().hex}")

    @classmethod
    def derived(cls, namespace: OpaqueId, stable_name: str) -> Self:
        """Derive a stable typed ID within an already-random Workshop namespace."""
        if not stable_name:
            raise ValueError("stable_name must be non-empty")
        namespace_uuid = uuid.UUID(hex=namespace.partition("_")[2])
        return cls(f"{cls.prefix}_{uuid.uuid5(namespace_uuid, f'{cls.prefix}:{stable_name}').hex}")


class WorkshopId(OpaqueId):
    prefix = "wsp"


class PrincipalId(OpaqueId):
    prefix = "prn"


class DeviceId(OpaqueId):
    prefix = "dev"


class ClientSessionId(OpaqueId):
    prefix = "cse"


class EnrollmentGrantId(OpaqueId):
    prefix = "enr"


class ExternalIdentityId(OpaqueId):
    prefix = "xid"


class WorkshopMembershipId(OpaqueId):
    prefix = "wmb"


class ChannelId(OpaqueId):
    prefix = "chn"


class ChannelMembershipId(OpaqueId):
    prefix = "chm"


class ChannelBindingId(OpaqueId):
    prefix = "cbd"


class AgentId(OpaqueId):
    prefix = "agt"


class ChannelAgentId(OpaqueId):
    prefix = "cag"


class MessageId(OpaqueId):
    prefix = "msg"


class ArtifactId(OpaqueId):
    prefix = "art"


class DeliveryId(OpaqueId):
    prefix = "dlv"


class DeliveryAttemptId(OpaqueId):
    prefix = "dla"


class EventId(OpaqueId):
    prefix = "evt"


_ID_TYPES: dict[str, type[OpaqueId]] = {
    identifier_type.prefix: identifier_type
    for identifier_type in (
        WorkshopId,
        PrincipalId,
        DeviceId,
        ClientSessionId,
        EnrollmentGrantId,
        ExternalIdentityId,
        WorkshopMembershipId,
        ChannelId,
        ChannelMembershipId,
        ChannelBindingId,
        AgentId,
        ChannelAgentId,
        MessageId,
        ArtifactId,
        DeliveryId,
        DeliveryAttemptId,
        EventId,
    )
}


def parse_opaque_id(value: str) -> OpaqueId:
    """Restore the concrete identifier type from a serialized value."""
    prefix, separator, _ = value.partition("_")
    identifier_type = _ID_TYPES.get(prefix)
    if not separator or identifier_type is None:
        raise ValueError(f"Unknown Workshop identifier: {value!r}")
    return identifier_type(value)


def _canonical_json(value: Any, *, field_name: str) -> tuple[dict[str, Any], str]:
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    try:
        encoded = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain only valid JSON values") from exc
    return decoded, encoded


@dataclass(frozen=True, slots=True)
class EventEnvelope:
    """A versioned authoritative-fact envelope independent of any transport."""

    event_id: EventId
    event_type: str
    event_version: int
    workshop_id: WorkshopId
    aggregate_type: str
    aggregate_id: OpaqueId
    occurred_at: datetime
    payload: dict[str, Any]
    actor_principal_id: PrincipalId | None = None
    idempotency_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    envelope_version: int = ENVELOPE_VERSION
    _payload_json: str = field(init=False, repr=False, compare=False)
    _metadata_json: str = field(init=False, repr=False, compare=False)
    _content_hash: str = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.event_id, EventId):
            raise ValueError("event_id must be an EventId")
        if not _EVENT_TYPE_PATTERN.fullmatch(self.event_type):
            raise ValueError("event_type must be a lowercase dotted identifier")
        if not isinstance(self.event_version, int) or isinstance(self.event_version, bool) or self.event_version < 1:
            raise ValueError("event_version must be a positive integer")
        if self.envelope_version != ENVELOPE_VERSION:
            raise ValueError(f"Unsupported envelope_version: {self.envelope_version}")
        if not isinstance(self.workshop_id, WorkshopId):
            raise ValueError("workshop_id must be a WorkshopId")
        if not _AGGREGATE_TYPE_PATTERN.fullmatch(self.aggregate_type):
            raise ValueError("aggregate_type must be a lowercase identifier")
        if not isinstance(self.aggregate_id, OpaqueId):
            raise ValueError("aggregate_id must be a typed Workshop identifier")
        if self.actor_principal_id is not None and not isinstance(self.actor_principal_id, PrincipalId):
            raise ValueError("actor_principal_id must be a PrincipalId or None")
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        if self.idempotency_key is not None:
            if not self.idempotency_key.strip():
                raise ValueError("idempotency_key must be non-empty when supplied")
            if len(self.idempotency_key) > 512:
                raise ValueError("idempotency_key must be at most 512 characters")

        normalized_payload, payload_json = _canonical_json(self.payload, field_name="payload")
        normalized_metadata, metadata_json = _canonical_json(self.metadata, field_name="metadata")
        object.__setattr__(self, "payload", normalized_payload)
        object.__setattr__(self, "metadata", normalized_metadata)
        object.__setattr__(self, "occurred_at", self.occurred_at.astimezone(UTC))
        object.__setattr__(self, "_payload_json", payload_json)
        object.__setattr__(self, "_metadata_json", metadata_json)
        object.__setattr__(self, "_content_hash", self._calculate_content_hash())

    @classmethod
    def create(
        cls,
        *,
        event_type: str,
        event_version: int,
        workshop_id: WorkshopId,
        aggregate_type: str,
        aggregate_id: OpaqueId,
        occurred_at: datetime,
        payload: dict[str, Any],
        event_id: EventId | None = None,
        actor_principal_id: PrincipalId | None = None,
        idempotency_key: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EventEnvelope:
        return cls(
            event_id=event_id or EventId.new(),
            event_type=event_type,
            event_version=event_version,
            workshop_id=workshop_id,
            aggregate_type=aggregate_type,
            aggregate_id=aggregate_id,
            occurred_at=occurred_at,
            payload=payload,
            actor_principal_id=actor_principal_id,
            idempotency_key=idempotency_key,
            metadata=metadata or {},
        )

    @property
    def payload_json(self) -> str:
        return self._payload_json

    @property
    def metadata_json(self) -> str:
        return self._metadata_json

    @property
    def content_hash(self) -> str:
        """Hash semantic content while excluding only generated event identity."""
        return self._content_hash

    def _calculate_content_hash(self) -> str:
        content = {
            "actor_principal_id": self.actor_principal_id,
            "aggregate_id": self.aggregate_id,
            "aggregate_type": self.aggregate_type,
            "envelope_version": self.envelope_version,
            "event_type": self.event_type,
            "event_version": self.event_version,
            "metadata": self.metadata,
            "occurred_at": self.occurred_at.isoformat(),
            "payload": self.payload,
            "workshop_id": self.workshop_id,
        }
        encoded = json.dumps(content, allow_nan=False, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
