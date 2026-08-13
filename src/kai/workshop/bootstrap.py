"""Deterministic bootstrap of the first Workshop collaboration records."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from kai.workshop.domain import (
    AgentId,
    ChannelAgentId,
    ChannelBindingId,
    ChannelId,
    ChannelMembershipId,
    EventEnvelope,
    ExternalIdentityId,
    OpaqueId,
    PrincipalId,
    RuntimeAssignmentId,
    RuntimeProfileId,
    WorkshopEventType,
    WorkshopId,
    WorkshopMembershipId,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import StoredEvent, WorkshopEventStore

_BOOTSTRAP_PREFIX = "workshop-bootstrap:v1"
_TRANSPORT_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True, slots=True)
class BootstrapHuman:
    display_name: str
    role: str
    transport: str
    external_subject: str
    external_channel_id: str
    runtime_profile_id: RuntimeProfileId | None = None

    def __post_init__(self) -> None:
        if not self.display_name.strip():
            raise ValueError("display_name must be non-empty")
        if self.role not in {"admin", "member"}:
            raise ValueError("role must be 'admin' or 'member'")
        if not _TRANSPORT_PATTERN.fullmatch(self.transport):
            raise ValueError("transport must be a lowercase identifier")
        if not self.external_subject.strip():
            raise ValueError("external_subject must be non-empty")
        if not self.external_channel_id.strip():
            raise ValueError("external_channel_id must be non-empty")
        if self.runtime_profile_id is not None and not isinstance(self.runtime_profile_id, RuntimeProfileId):
            raise ValueError("runtime_profile_id must be an opaque RuntimeProfileId")


@dataclass(frozen=True, slots=True)
class BootstrapNotificationChannel:
    """One outbound-only transport destination shared by configured humans."""

    transport: str
    external_channel_id: str
    member_external_subjects: tuple[str, ...]

    def __post_init__(self) -> None:
        if not _TRANSPORT_PATTERN.fullmatch(self.transport):
            raise ValueError("transport must be a lowercase identifier")
        if not self.external_channel_id.strip():
            raise ValueError("external_channel_id must be non-empty")
        if not self.member_external_subjects:
            raise ValueError("member_external_subjects must be non-empty")
        if len(set(self.member_external_subjects)) != len(self.member_external_subjects):
            raise ValueError("member_external_subjects must be unique")
        if any(not subject.strip() for subject in self.member_external_subjects):
            raise ValueError("member_external_subjects must contain non-empty values")


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    workshop_id: WorkshopId
    agent_id: AgentId
    created_events: int
    existing_events: int
    human_count: int
    channel_count: int


def _stable_token(*parts: str) -> str:
    encoded = "\0".join(parts).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _idempotency_key(kind: str, stable_token: str) -> str:
    return f"{_BOOTSTRAP_PREFIX}:{kind}:{stable_token}"


async def _ensure_event(
    store: WorkshopEventStore,
    *,
    idempotency_key: str,
    event_type: WorkshopEventType,
    workshop_id: WorkshopId,
    aggregate_type: str,
    aggregate_id: OpaqueId,
    occurred_at: datetime,
    payload: dict[str, Any],
) -> tuple[StoredEvent, bool]:
    existing = await store.event_by_idempotency_key(idempotency_key)
    if existing is not None:
        return existing, False
    result = await store.append(
        EventEnvelope.create(
            event_type=event_type,
            event_version=1,
            workshop_id=workshop_id,
            aggregate_type=aggregate_type,
            aggregate_id=aggregate_id,
            occurred_at=occurred_at,
            idempotency_key=idempotency_key,
            payload=payload,
            metadata={"source": "bootstrap"},
        )
    )
    return result.event, result.inserted


async def bootstrap_default_workshop(
    store: WorkshopEventStore,
    humans: Iterable[BootstrapHuman],
    *,
    notification_channels: Iterable[BootstrapNotificationChannel] = (),
) -> BootstrapResult:
    """Seed one Workshop and its configured humans without changing routing."""
    ordered_humans = sorted(humans, key=lambda human: (human.transport, human.external_subject))
    ordered_notifications = sorted(
        notification_channels,
        key=lambda channel: (channel.transport, channel.external_channel_id),
    )
    seen_identities: set[tuple[str, str]] = set()
    seen_channels: set[tuple[str, str]] = set()
    for human in ordered_humans:
        identity = (human.transport, human.external_subject)
        channel = (human.transport, human.external_channel_id)
        if identity in seen_identities:
            raise ValueError(f"Duplicate bootstrap external identity for transport {human.transport!r}")
        if channel in seen_channels:
            raise ValueError(f"Duplicate bootstrap external channel for transport {human.transport!r}")
        seen_identities.add(identity)
        seen_channels.add(channel)
    for channel in ordered_notifications:
        identity = (channel.transport, channel.external_channel_id)
        if identity in seen_channels:
            raise ValueError(f"Duplicate bootstrap external channel for transport {channel.transport!r}")
        seen_channels.add(identity)
        for subject in channel.member_external_subjects:
            if (channel.transport, subject) not in seen_identities:
                raise ValueError("Notification channel member must reference a configured external identity")

    created_events = 0
    existing_events = 0
    now = datetime.now(UTC)
    workshop_key = _idempotency_key("workshop", "default")
    workshop_event = await store.event_by_idempotency_key(workshop_key)
    if workshop_event is None:
        workshop_id = WorkshopId.new()
        result = await store.append(
            EventEnvelope.create(
                event_type=WorkshopEventType.WORKSHOP_CREATED,
                event_version=1,
                workshop_id=workshop_id,
                aggregate_type="workshop",
                aggregate_id=workshop_id,
                occurred_at=now,
                idempotency_key=workshop_key,
                payload={"name": "Kai Workshop"},
                metadata={"source": "bootstrap"},
            )
        )
        workshop_event = result.event
        created_events += int(result.inserted)
        existing_events += int(not result.inserted)
    else:
        existing_events += 1
    if not isinstance(workshop_event.envelope.aggregate_id, WorkshopId):
        raise RuntimeError("Default Workshop bootstrap event has the wrong aggregate type")
    workshop_id = workshop_event.envelope.aggregate_id

    async def ensure(**kwargs) -> StoredEvent:
        nonlocal created_events, existing_events
        event, inserted = await _ensure_event(store, occurred_at=now, workshop_id=workshop_id, **kwargs)
        created_events += int(inserted)
        existing_events += int(not inserted)
        return event

    agent_principal_id = PrincipalId.derived(workshop_id, "agent-principal:kai")
    agent_membership_id = WorkshopMembershipId.derived(workshop_id, "membership:agent:kai")
    agent_id = AgentId.derived(workshop_id, "agent:kai")
    await ensure(
        idempotency_key=_idempotency_key("principal", "agent-kai"),
        event_type=WorkshopEventType.PRINCIPAL_CREATED,
        aggregate_type="principal",
        aggregate_id=agent_principal_id,
        payload={"kind": "agent", "display_name": "Kai"},
    )
    await ensure(
        idempotency_key=_idempotency_key("membership", "agent-kai"),
        event_type=WorkshopEventType.WORKSHOP_MEMBER_ADDED,
        aggregate_type="workshop_membership",
        aggregate_id=agent_membership_id,
        payload={"principal_id": agent_principal_id, "role": "agent"},
    )
    await ensure(
        idempotency_key=_idempotency_key("agent", "kai"),
        event_type=WorkshopEventType.AGENT_CREATED,
        aggregate_type="agent",
        aggregate_id=agent_id,
        payload={"principal_id": agent_principal_id, "name": "Kai"},
    )

    for human in ordered_humans:
        token = _stable_token(human.transport, human.external_subject)
        channel_token = _stable_token(human.transport, human.external_channel_id)
        principal_id = PrincipalId.derived(workshop_id, f"human:{token}")
        external_identity_id = ExternalIdentityId.derived(workshop_id, f"external-identity:{token}")
        membership_id = WorkshopMembershipId.derived(workshop_id, f"membership:human:{token}")
        channel_id = ChannelId.derived(workshop_id, f"direct-channel:{channel_token}")
        human_channel_membership_id = ChannelMembershipId.derived(
            workshop_id,
            f"channel-membership:{channel_token}:human:{token}",
        )
        agent_channel_membership_id = ChannelMembershipId.derived(
            workshop_id,
            f"channel-membership:{channel_token}:agent:kai",
        )
        binding_id = ChannelBindingId.derived(workshop_id, f"channel-binding:{channel_token}")
        channel_agent_id = ChannelAgentId.derived(workshop_id, f"channel-agent:{channel_token}:kai")
        await ensure(
            idempotency_key=_idempotency_key("principal", token),
            event_type=WorkshopEventType.PRINCIPAL_CREATED,
            aggregate_type="principal",
            aggregate_id=principal_id,
            payload={"kind": "human", "display_name": human.display_name.strip()},
        )
        await ensure(
            idempotency_key=_idempotency_key("external-identity", token),
            event_type=WorkshopEventType.EXTERNAL_IDENTITY_BOUND,
            aggregate_type="external_identity",
            aggregate_id=external_identity_id,
            payload={
                "principal_id": principal_id,
                "provider": human.transport,
                "external_subject": human.external_subject,
            },
        )
        await ensure(
            idempotency_key=_idempotency_key("membership", token),
            event_type=WorkshopEventType.WORKSHOP_MEMBER_ADDED,
            aggregate_type="workshop_membership",
            aggregate_id=membership_id,
            payload={"principal_id": principal_id, "role": human.role},
        )
        await ensure(
            idempotency_key=_idempotency_key("channel", channel_token),
            event_type=WorkshopEventType.CHANNEL_CREATED,
            aggregate_type="channel",
            aggregate_id=channel_id,
            payload={"kind": "direct", "name": "Direct"},
        )
        await ensure(
            idempotency_key=_idempotency_key("channel-membership-human", channel_token),
            event_type=WorkshopEventType.CHANNEL_MEMBER_ADDED,
            aggregate_type="channel_membership",
            aggregate_id=human_channel_membership_id,
            payload={
                "channel_id": channel_id,
                "principal_id": principal_id,
                "role": "owner",
            },
        )
        await ensure(
            idempotency_key=_idempotency_key("channel-membership-agent", channel_token),
            event_type=WorkshopEventType.CHANNEL_MEMBER_ADDED,
            aggregate_type="channel_membership",
            aggregate_id=agent_channel_membership_id,
            payload={
                "channel_id": channel_id,
                "principal_id": agent_principal_id,
                "role": "participant",
            },
        )
        await ensure(
            idempotency_key=_idempotency_key("channel-binding", channel_token),
            event_type=WorkshopEventType.TRANSPORT_CHANNEL_BOUND,
            aggregate_type="channel_binding",
            aggregate_id=binding_id,
            payload={
                "channel_id": channel_id,
                "transport": human.transport,
                "external_channel_id": human.external_channel_id,
            },
        )
        await ensure(
            idempotency_key=_idempotency_key("channel-agent", channel_token),
            event_type=WorkshopEventType.CHANNEL_AGENT_ATTACHED,
            aggregate_type="channel_agent",
            aggregate_id=channel_agent_id,
            payload={"channel_id": channel_id, "agent_id": agent_id},
        )
        if human.runtime_profile_id is not None:
            assignment_id = RuntimeAssignmentId.derived(channel_id, f"runtime-profile:{agent_id}")
            previous_key = _idempotency_key("runtime-assignment", channel_token)
            assignment_key = _idempotency_key("runtime-assignment-v2", channel_token)
            reassignment_key = _idempotency_key(
                "runtime-reassignment-v2",
                f"{channel_token}:{human.runtime_profile_id}",
            )
            if (
                await store.event_by_idempotency_key(assignment_key) is not None
                or await store.event_by_idempotency_key(reassignment_key) is not None
            ):
                existing_events += 1
            elif await store.event_by_idempotency_key(previous_key) is not None:
                await ensure(
                    idempotency_key=reassignment_key,
                    event_type=WorkshopEventType.RUNTIME_PROFILE_REASSIGNED,
                    aggregate_type="runtime_assignment",
                    aggregate_id=assignment_id,
                    payload={
                        "channel_id": channel_id,
                        "agent_id": agent_id,
                        "runtime_profile_id": human.runtime_profile_id,
                    },
                )
            else:
                await ensure(
                    idempotency_key=assignment_key,
                    event_type=WorkshopEventType.RUNTIME_PROFILE_ASSIGNED,
                    aggregate_type="runtime_assignment",
                    aggregate_id=assignment_id,
                    payload={
                        "channel_id": channel_id,
                        "agent_id": agent_id,
                        "runtime_profile_id": human.runtime_profile_id,
                    },
                )

    for notification in ordered_notifications:
        channel_token = _stable_token(notification.transport, notification.external_channel_id)
        channel_id = ChannelId.derived(workshop_id, f"notification-channel:{channel_token}")
        agent_channel_membership_id = ChannelMembershipId.derived(
            workshop_id,
            f"notification-channel-membership:{channel_token}:agent:kai",
        )
        binding_id = ChannelBindingId.derived(
            workshop_id,
            f"notification-channel-binding:{channel_token}",
        )
        channel_agent_id = ChannelAgentId.derived(
            workshop_id,
            f"notification-channel-agent:{channel_token}:kai",
        )
        await ensure(
            idempotency_key=_idempotency_key("notification-channel", channel_token),
            event_type=WorkshopEventType.CHANNEL_CREATED,
            aggregate_type="channel",
            aggregate_id=channel_id,
            payload={"kind": "notification", "name": "Notifications"},
        )
        for subject in sorted(notification.member_external_subjects):
            human_token = _stable_token(notification.transport, subject)
            principal_id = PrincipalId.derived(workshop_id, f"human:{human_token}")
            membership_id = ChannelMembershipId.derived(
                workshop_id,
                f"notification-channel-membership:{channel_token}:human:{human_token}",
            )
            await ensure(
                idempotency_key=_idempotency_key(
                    "notification-channel-membership-human",
                    f"{channel_token}:{human_token}",
                ),
                event_type=WorkshopEventType.CHANNEL_MEMBER_ADDED,
                aggregate_type="channel_membership",
                aggregate_id=membership_id,
                payload={
                    "channel_id": channel_id,
                    "principal_id": principal_id,
                    "role": "participant",
                },
            )
        await ensure(
            idempotency_key=_idempotency_key("notification-channel-membership-agent", channel_token),
            event_type=WorkshopEventType.CHANNEL_MEMBER_ADDED,
            aggregate_type="channel_membership",
            aggregate_id=agent_channel_membership_id,
            payload={
                "channel_id": channel_id,
                "principal_id": agent_principal_id,
                "role": "participant",
            },
        )
        await ensure(
            idempotency_key=_idempotency_key("notification-channel-binding", channel_token),
            event_type=WorkshopEventType.TRANSPORT_CHANNEL_BOUND,
            aggregate_type="channel_binding",
            aggregate_id=binding_id,
            payload={
                "channel_id": channel_id,
                "transport": notification.transport,
                "external_channel_id": notification.external_channel_id,
            },
        )
        await ensure(
            idempotency_key=_idempotency_key("notification-channel-agent", channel_token),
            event_type=WorkshopEventType.CHANNEL_AGENT_ATTACHED,
            aggregate_type="channel_agent",
            aggregate_id=channel_agent_id,
            payload={"channel_id": channel_id, "agent_id": agent_id},
        )

    await store.rebuild_projection(CanonicalConversationProjection())
    return BootstrapResult(
        workshop_id=workshop_id,
        agent_id=agent_id,
        created_events=created_events,
        existing_events=existing_events,
        human_count=len(ordered_humans),
        channel_count=len(ordered_humans) + len(ordered_notifications),
    )
