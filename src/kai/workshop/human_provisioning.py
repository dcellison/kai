"""Canonical, transport-independent Workshop human provisioning."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime

from kai.workshop.domain import (
    AgentId,
    ChannelAgentId,
    ChannelId,
    ChannelMembershipId,
    EventEnvelope,
    PrincipalId,
    WorkshopEventType,
    WorkshopId,
    WorkshopMembershipId,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import IdempotencyConflictError, WorkshopEventStore


class WorkshopHumanProvisioningError(RuntimeError):
    """A canonical human could not be provisioned safely."""


@dataclass(frozen=True, slots=True)
class ProvisionedWorkshopHuman:
    workshop_id: WorkshopId
    principal_id: PrincipalId
    channel_id: ChannelId
    display_name: str
    role: str
    provisioning_key: str
    created: bool


_PROVISIONING_KEY_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


def _normalize_display_name(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value.strip()) > 200:
        raise WorkshopHumanProvisioningError("Display name must contain 1 through 200 characters")
    return value.strip()


def _normalize_role(value: object) -> str:
    if value not in {"admin", "member"}:
        raise WorkshopHumanProvisioningError("Role must be 'admin' or 'member'")
    return str(value)


def _normalize_provisioning_key(value: object) -> str:
    if not isinstance(value, str) or not _PROVISIONING_KEY_PATTERN.fullmatch(value):
        raise WorkshopHumanProvisioningError(
            "Provisioning key must be a lowercase identifier of 1 through 64 characters"
        )
    return value


class WorkshopHumanProvisioner:
    """Create collaboration identity without granting transport or runtime authority."""

    def __init__(self, store: WorkshopEventStore) -> None:
        self._store = store

    async def provision(
        self,
        provisioning_key: str,
        display_name: str,
        role: str,
        *,
        workshop_id: WorkshopId | None = None,
    ) -> ProvisionedWorkshopHuman:
        normalized_key = _normalize_provisioning_key(provisioning_key)
        normalized_name = _normalize_display_name(display_name)
        normalized_role = _normalize_role(role)
        if workshop_id is not None and not isinstance(workshop_id, WorkshopId):
            raise WorkshopHumanProvisioningError("workshop_id must be a WorkshopId when provided")

        now = datetime.now(UTC)
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            resolved_workshop_id = await self._resolve_workshop(workshop_id)
            agent_id, agent_principal_id = await self._resolve_kai_agent(resolved_workshop_id)
            stable_prefix = f"operator-human:{normalized_key}"
            principal_id = PrincipalId.derived(resolved_workshop_id, stable_prefix)
            membership_id = WorkshopMembershipId.derived(
                resolved_workshop_id,
                f"{stable_prefix}:workshop-membership",
            )
            channel_id = ChannelId.derived(
                resolved_workshop_id,
                f"{stable_prefix}:direct-channel",
            )
            human_channel_membership_id = ChannelMembershipId.derived(
                resolved_workshop_id,
                f"{stable_prefix}:human-channel-membership",
            )
            agent_channel_membership_id = ChannelMembershipId.derived(
                resolved_workshop_id,
                f"{stable_prefix}:agent-channel-membership",
            )
            channel_agent_id = ChannelAgentId.derived(
                resolved_workshop_id,
                f"{stable_prefix}:channel-agent",
            )
            principal_event_key = f"operator:human-provisioning:{principal_id}:principal"
            if await self._store.event_by_idempotency_key(principal_event_key) is not None:
                if not await self._matches_existing_provisioning(
                    resolved_workshop_id,
                    principal_id,
                    channel_id,
                    agent_id,
                    agent_principal_id,
                    normalized_name,
                    normalized_role,
                ):
                    raise WorkshopHumanProvisioningError(
                        "Provisioning key already exists with different human or channel semantics"
                    )
                await connection.commit()
                return ProvisionedWorkshopHuman(
                    resolved_workshop_id,
                    principal_id,
                    channel_id,
                    normalized_name,
                    normalized_role,
                    normalized_key,
                    False,
                )
            events = (
                EventEnvelope.create(
                    event_type=WorkshopEventType.PRINCIPAL_CREATED,
                    event_version=1,
                    workshop_id=resolved_workshop_id,
                    aggregate_type="principal",
                    aggregate_id=principal_id,
                    occurred_at=now,
                    idempotency_key=principal_event_key,
                    payload={"kind": "human", "display_name": normalized_name},
                    metadata={"source": "operator_cli"},
                ),
                EventEnvelope.create(
                    event_type=WorkshopEventType.WORKSHOP_MEMBER_ADDED,
                    event_version=1,
                    workshop_id=resolved_workshop_id,
                    aggregate_type="workshop_membership",
                    aggregate_id=membership_id,
                    occurred_at=now,
                    idempotency_key=f"operator:human-provisioning:{principal_id}:workshop-membership",
                    payload={"principal_id": principal_id, "role": normalized_role},
                    metadata={"source": "operator_cli"},
                ),
                EventEnvelope.create(
                    event_type=WorkshopEventType.CHANNEL_CREATED,
                    event_version=1,
                    workshop_id=resolved_workshop_id,
                    aggregate_type="channel",
                    aggregate_id=channel_id,
                    occurred_at=now,
                    idempotency_key=f"operator:human-provisioning:{principal_id}:direct-channel",
                    payload={"kind": "direct", "name": "Direct"},
                    metadata={"source": "operator_cli"},
                ),
                EventEnvelope.create(
                    event_type=WorkshopEventType.CHANNEL_MEMBER_ADDED,
                    event_version=1,
                    workshop_id=resolved_workshop_id,
                    aggregate_type="channel_membership",
                    aggregate_id=human_channel_membership_id,
                    occurred_at=now,
                    idempotency_key=f"operator:human-provisioning:{principal_id}:human-channel-membership",
                    payload={
                        "channel_id": channel_id,
                        "principal_id": principal_id,
                        "role": "owner",
                    },
                    metadata={"source": "operator_cli"},
                ),
                EventEnvelope.create(
                    event_type=WorkshopEventType.CHANNEL_MEMBER_ADDED,
                    event_version=1,
                    workshop_id=resolved_workshop_id,
                    aggregate_type="channel_membership",
                    aggregate_id=agent_channel_membership_id,
                    occurred_at=now,
                    idempotency_key=f"operator:human-provisioning:{principal_id}:agent-channel-membership",
                    payload={
                        "channel_id": channel_id,
                        "principal_id": agent_principal_id,
                        "role": "participant",
                    },
                    metadata={"source": "operator_cli"},
                ),
                EventEnvelope.create(
                    event_type=WorkshopEventType.CHANNEL_AGENT_ATTACHED,
                    event_version=1,
                    workshop_id=resolved_workshop_id,
                    aggregate_type="channel_agent",
                    aggregate_id=channel_agent_id,
                    occurred_at=now,
                    idempotency_key=f"operator:human-provisioning:{principal_id}:channel-agent",
                    payload={"channel_id": channel_id, "agent_id": agent_id},
                    metadata={"source": "operator_cli"},
                ),
            )
            inserted_states: set[bool] = set()
            for event in events:
                result = await self._store.append_in_transaction(event)
                inserted_states.add(result.inserted)
            if len(inserted_states) != 1:
                raise WorkshopHumanProvisioningError("Provisioning events do not share one prior state")
            await self._store.project_pending_in_transaction(CanonicalConversationProjection())
            await connection.commit()
        except IdempotencyConflictError as exc:
            await connection.rollback()
            raise WorkshopHumanProvisioningError(
                "Provisioning key already exists with different human or channel semantics"
            ) from exc
        except Exception:
            await connection.rollback()
            raise
        return ProvisionedWorkshopHuman(
            resolved_workshop_id,
            principal_id,
            channel_id,
            normalized_name,
            normalized_role,
            normalized_key,
            inserted_states == {True},
        )

    async def _matches_existing_provisioning(
        self,
        workshop_id: WorkshopId,
        principal_id: PrincipalId,
        channel_id: ChannelId,
        agent_id: AgentId,
        agent_principal_id: PrincipalId,
        display_name: str,
        role: str,
    ) -> bool:
        async with self._store.connection.execute(
            "SELECT 1 FROM principals p "
            "JOIN workshop_memberships wm ON wm.principal_id = p.id "
            "AND wm.workshop_id = ? AND wm.role = ? "
            "JOIN channels c ON c.id = ? AND c.workshop_id = wm.workshop_id "
            "AND c.kind = 'direct' "
            "JOIN channel_memberships human_cm ON human_cm.channel_id = c.id "
            "AND human_cm.principal_id = p.id AND human_cm.role = 'owner' "
            "JOIN channel_agents ca ON ca.channel_id = c.id AND ca.agent_id = ? "
            "JOIN channel_memberships agent_cm ON agent_cm.channel_id = c.id "
            "AND agent_cm.principal_id = ? AND agent_cm.role = 'participant' "
            "WHERE p.id = ? AND p.kind = 'human' AND p.display_name = ? LIMIT 1",
            (
                workshop_id,
                role,
                channel_id,
                agent_id,
                agent_principal_id,
                principal_id,
                display_name,
            ),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return False
        async with self._store.connection.execute(
            "SELECT COUNT(*) FROM event_log WHERE idempotency_key LIKE ?",
            (f"operator:human-provisioning:{principal_id}:%",),
        ) as cursor:
            count_row = await cursor.fetchone()
        return count_row is not None and int(count_row[0]) == 6

    async def _resolve_workshop(self, workshop_id: WorkshopId | None) -> WorkshopId:
        if workshop_id is None:
            async with self._store.connection.execute("SELECT id FROM workshops ORDER BY id") as cursor:
                rows = list(await cursor.fetchall())
            if len(rows) != 1:
                raise WorkshopHumanProvisioningError(
                    "Provisioning requires --workshop-id unless exactly one Workshop exists"
                )
            return WorkshopId(str(rows[0][0]))
        async with self._store.connection.execute(
            "SELECT 1 FROM workshops WHERE id = ?",
            (workshop_id,),
        ) as cursor:
            exists = await cursor.fetchone() is not None
        if not exists:
            raise WorkshopHumanProvisioningError("Workshop is unavailable")
        return workshop_id

    async def _resolve_kai_agent(self, workshop_id: WorkshopId) -> tuple[AgentId, PrincipalId]:
        async with self._store.connection.execute(
            "SELECT a.id, a.principal_id FROM agents a "
            "JOIN principals p ON p.id = a.principal_id AND p.kind = 'agent' "
            "JOIN workshop_memberships wm ON wm.workshop_id = a.workshop_id "
            "AND wm.principal_id = a.principal_id AND wm.role = 'agent' "
            "WHERE a.workshop_id = ? AND a.name = 'Kai'",
            (workshop_id,),
        ) as cursor:
            rows = list(await cursor.fetchall())
        if len(rows) != 1:
            raise WorkshopHumanProvisioningError("Workshop does not contain exactly one canonical Kai agent")
        return AgentId(str(rows[0][0])), PrincipalId(str(rows[0][1]))
