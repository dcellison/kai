"""Canonical, transport-independent Workshop channel creation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from kai.workshop.domain import (
    AgentId,
    ChannelAgentId,
    ChannelId,
    ChannelMembershipId,
    EventEnvelope,
    PrincipalId,
    RuntimeAssignmentId,
    RuntimeProfileId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import WorkshopEventStore


class WorkshopChannelLifecycleError(RuntimeError):
    """A canonical channel lifecycle operation could not be completed."""


class WorkshopChannelLifecycleAccessDenied(WorkshopChannelLifecycleError):
    """The principal cannot create a channel in the requested Workshop."""


class WorkshopChannelLifecycleValidationError(WorkshopChannelLifecycleError):
    """A channel creation request is malformed or references unavailable state."""


class WorkshopChannelLifecycleStorageError(WorkshopChannelLifecycleError):
    """Canonical channel creation could not be persisted atomically."""


@dataclass(frozen=True, slots=True)
class CreatedWorkshopChannel:
    """The canonical identity and immutable creation properties of a group channel."""

    channel_id: ChannelId
    workshop_id: WorkshopId
    name: str
    visibility: str
    origin_channel_id: ChannelId | None
    owner_principal_id: PrincipalId
    agent_ids: tuple[AgentId, ...]


@dataclass(frozen=True, slots=True)
class _AgentAttachment:
    agent_id: AgentId
    principal_id: PrincipalId
    runtime_profile_id: RuntimeProfileId


def _normalize_name(value: object) -> str:
    if not isinstance(value, str):
        raise WorkshopChannelLifecycleValidationError("Channel name must be text")
    normalized = value.strip()
    if not normalized or len(normalized) > 200:
        raise WorkshopChannelLifecycleValidationError("Channel name must contain 1 through 200 characters")
    return normalized


def _normalize_agents(values: object) -> tuple[AgentId, ...]:
    if not isinstance(values, list) or not values or len(values) > 16:
        raise WorkshopChannelLifecycleValidationError("agent_ids must contain 1 through 16 canonical agent IDs")
    try:
        normalized = tuple(AgentId(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise WorkshopChannelLifecycleValidationError("agent_ids must contain canonical agent IDs") from exc
    if len(set(normalized)) != len(normalized):
        raise WorkshopChannelLifecycleValidationError("agent_ids must not contain duplicates")
    return normalized


def _normalize_origin(value: object) -> ChannelId | None:
    if value is None:
        return None
    try:
        return ChannelId(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise WorkshopChannelLifecycleValidationError("origin_channel_id must be a canonical channel ID") from exc


class WorkshopChannelLifecycleService:
    """Create private group channels from canonical client authority."""

    def __init__(self, store: WorkshopEventStore) -> None:
        self._store = store

    async def create_group(
        self,
        principal_id: PrincipalId,
        *,
        name: object,
        agent_ids: object,
        origin_channel_id: object = None,
    ) -> CreatedWorkshopChannel:
        if not isinstance(principal_id, PrincipalId):
            raise WorkshopChannelLifecycleAccessDenied("Canonical human principal is required")
        normalized_name = _normalize_name(name)
        normalized_agents = _normalize_agents(agent_ids)
        normalized_origin = _normalize_origin(origin_channel_id)

        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            workshop_id = await self._resolve_workshop(principal_id, normalized_origin)
            attachments = await self._resolve_agent_attachments(
                principal_id,
                workshop_id,
                normalized_agents,
            )
            channel_id = ChannelId.new()
            now = datetime.now(UTC)
            events = self._creation_events(
                workshop_id=workshop_id,
                channel_id=channel_id,
                principal_id=principal_id,
                name=normalized_name,
                origin_channel_id=normalized_origin,
                attachments=attachments,
                occurred_at=now,
            )
            for event in events:
                result = await self._store.append_in_transaction(event)
                if not result.inserted:
                    raise WorkshopChannelLifecycleError("New channel event identity unexpectedly already exists")
            await self._store.project_pending_in_transaction(CanonicalConversationProjection())
            await connection.commit()
        except WorkshopChannelLifecycleError:
            await connection.rollback()
            raise
        except Exception as exc:
            await connection.rollback()
            raise WorkshopChannelLifecycleStorageError("Channel creation could not be persisted") from exc

        return CreatedWorkshopChannel(
            channel_id=channel_id,
            workshop_id=workshop_id,
            name=normalized_name,
            visibility="private",
            origin_channel_id=normalized_origin,
            owner_principal_id=principal_id,
            agent_ids=tuple(attachment.agent_id for attachment in attachments),
        )

    async def _resolve_workshop(
        self,
        principal_id: PrincipalId,
        origin_channel_id: ChannelId | None,
    ) -> WorkshopId:
        if origin_channel_id is not None:
            async with self._store.connection.execute(
                "SELECT c.workshop_id FROM channels c "
                "JOIN channel_memberships cm ON cm.channel_id = c.id "
                "AND cm.principal_id = ? "
                "JOIN workshop_memberships wm ON wm.workshop_id = c.workshop_id "
                "AND wm.principal_id = cm.principal_id "
                "JOIN principals p ON p.id = cm.principal_id AND p.kind = 'human' "
                "WHERE c.id = ?",
                (principal_id, origin_channel_id),
            ) as cursor:
                rows = list(await cursor.fetchall())
        else:
            async with self._store.connection.execute(
                "SELECT wm.workshop_id FROM workshop_memberships wm "
                "JOIN principals p ON p.id = wm.principal_id AND p.kind = 'human' "
                "WHERE wm.principal_id = ? ORDER BY wm.workshop_id",
                (principal_id,),
            ) as cursor:
                rows = list(await cursor.fetchall())
        if len(rows) != 1:
            raise WorkshopChannelLifecycleAccessDenied("Channel creation requires one accessible Workshop context")
        return WorkshopId(str(rows[0][0]))

    async def _resolve_agent_attachments(
        self,
        principal_id: PrincipalId,
        workshop_id: WorkshopId,
        agent_ids: tuple[AgentId, ...],
    ) -> tuple[_AgentAttachment, ...]:
        placeholders = ",".join("?" for _ in agent_ids)
        async with self._store.connection.execute(
            "SELECT a.id, a.principal_id, ra.runtime_profile_id "
            "FROM agents a "
            "JOIN workshop_memberships agent_wm ON agent_wm.workshop_id = a.workshop_id "
            "AND agent_wm.principal_id = a.principal_id AND agent_wm.role = 'agent' "
            "JOIN channel_agents ca ON ca.agent_id = a.id "
            "JOIN channels c ON c.id = ca.channel_id AND c.workshop_id = a.workshop_id "
            "AND c.kind = 'direct' "
            "JOIN channel_memberships cm ON cm.channel_id = c.id "
            "AND cm.principal_id = ? AND cm.role = 'owner' "
            "JOIN channel_agent_runtime_assignments ra ON ra.channel_id = c.id "
            "AND ra.agent_id = a.id "
            f"WHERE a.workshop_id = ? AND a.id IN ({placeholders}) "
            "ORDER BY a.id",
            (principal_id, workshop_id, *agent_ids),
        ) as cursor:
            rows = list(await cursor.fetchall())
        by_agent: dict[AgentId, list[_AgentAttachment]] = {}
        for row in rows:
            attachment = _AgentAttachment(
                AgentId(str(row[0])),
                PrincipalId(str(row[1])),
                RuntimeProfileId(str(row[2])),
            )
            by_agent.setdefault(attachment.agent_id, []).append(attachment)
        if any(len(by_agent.get(agent_id, ())) != 1 for agent_id in agent_ids):
            raise WorkshopChannelLifecycleValidationError(
                "Every requested agent must be attached to the creator's direct channel"
            )
        return tuple(by_agent[agent_id][0] for agent_id in agent_ids)

    @staticmethod
    def _creation_events(
        *,
        workshop_id: WorkshopId,
        channel_id: ChannelId,
        principal_id: PrincipalId,
        name: str,
        origin_channel_id: ChannelId | None,
        attachments: tuple[_AgentAttachment, ...],
        occurred_at: datetime,
    ) -> tuple[EventEnvelope, ...]:
        metadata = {"source": "workshop_client"}
        channel_payload: dict[str, object] = {
            "kind": "group",
            "name": name,
            "visibility": "private",
        }
        if origin_channel_id is not None:
            channel_payload["origin_channel_id"] = origin_channel_id
        events = [
            EventEnvelope.create(
                event_type=WorkshopEventType.CHANNEL_CREATED,
                event_version=1,
                workshop_id=workshop_id,
                aggregate_type="channel",
                aggregate_id=channel_id,
                actor_principal_id=principal_id,
                occurred_at=occurred_at,
                idempotency_key=f"workshop-client:channel:{channel_id}:created",
                payload=channel_payload,
                metadata=metadata,
            ),
            EventEnvelope.create(
                event_type=WorkshopEventType.CHANNEL_MEMBER_ADDED,
                event_version=1,
                workshop_id=workshop_id,
                aggregate_type="channel_membership",
                aggregate_id=ChannelMembershipId.derived(channel_id, f"principal:{principal_id}"),
                actor_principal_id=principal_id,
                occurred_at=occurred_at,
                idempotency_key=f"workshop-client:channel:{channel_id}:member:{principal_id}",
                payload={
                    "channel_id": channel_id,
                    "principal_id": principal_id,
                    "role": "owner",
                },
                metadata=metadata,
            ),
        ]
        for attachment in attachments:
            events.extend(
                (
                    EventEnvelope.create(
                        event_type=WorkshopEventType.CHANNEL_MEMBER_ADDED,
                        event_version=1,
                        workshop_id=workshop_id,
                        aggregate_type="channel_membership",
                        aggregate_id=ChannelMembershipId.derived(
                            channel_id,
                            f"principal:{attachment.principal_id}",
                        ),
                        actor_principal_id=principal_id,
                        occurred_at=occurred_at,
                        idempotency_key=(f"workshop-client:channel:{channel_id}:member:{attachment.principal_id}"),
                        payload={
                            "channel_id": channel_id,
                            "principal_id": attachment.principal_id,
                            "role": "participant",
                        },
                        metadata=metadata,
                    ),
                    EventEnvelope.create(
                        event_type=WorkshopEventType.CHANNEL_AGENT_ATTACHED,
                        event_version=1,
                        workshop_id=workshop_id,
                        aggregate_type="channel_agent",
                        aggregate_id=ChannelAgentId.derived(
                            channel_id,
                            f"agent:{attachment.agent_id}",
                        ),
                        actor_principal_id=principal_id,
                        occurred_at=occurred_at,
                        idempotency_key=(f"workshop-client:channel:{channel_id}:agent:{attachment.agent_id}"),
                        payload={"channel_id": channel_id, "agent_id": attachment.agent_id},
                        metadata=metadata,
                    ),
                    EventEnvelope.create(
                        event_type=WorkshopEventType.RUNTIME_PROFILE_ASSIGNED,
                        event_version=1,
                        workshop_id=workshop_id,
                        aggregate_type="runtime_assignment",
                        aggregate_id=RuntimeAssignmentId.derived(
                            channel_id,
                            f"runtime-profile:{attachment.agent_id}",
                        ),
                        actor_principal_id=principal_id,
                        occurred_at=occurred_at,
                        idempotency_key=(f"workshop-client:channel:{channel_id}:runtime:{attachment.agent_id}"),
                        payload={
                            "channel_id": channel_id,
                            "agent_id": attachment.agent_id,
                            "runtime_profile_id": attachment.runtime_profile_id,
                        },
                        metadata=metadata,
                    ),
                )
            )
        return tuple(events)
