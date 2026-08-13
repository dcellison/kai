"""Explicit runtime-profile policy for canonical Workshop channel agents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime

from kai.workshop.domain import (
    AgentId,
    ChannelId,
    EventEnvelope,
    PrincipalId,
    RuntimeAssignmentId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import IdempotencyConflictError, WorkshopEventStore

_RUNTIME_PROFILE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class WorkshopRuntimeAssignmentError(RuntimeError):
    """Runtime policy is absent, ambiguous, or unauthorized."""


@dataclass(frozen=True, slots=True)
class WorkshopRuntimeAssignment:
    assignment_id: RuntimeAssignmentId
    workshop_id: WorkshopId
    principal_id: PrincipalId
    channel_id: ChannelId
    agent_id: AgentId
    runtime_profile_id: str
    created: bool


def normalize_runtime_profile_id(value: object) -> str:
    if not isinstance(value, str) or not _RUNTIME_PROFILE_PATTERN.fullmatch(value):
        raise WorkshopRuntimeAssignmentError(
            "Runtime profile ID must contain 1 through 128 bounded identifier characters"
        )
    return value


def runtime_assignment_envelope(
    *,
    workshop_id: WorkshopId,
    assignment_id: RuntimeAssignmentId,
    channel_id: ChannelId,
    agent_id: AgentId,
    runtime_profile_id: str,
    occurred_at: datetime,
    idempotency_key: str,
    source: str,
) -> EventEnvelope:
    return EventEnvelope.create(
        event_type=WorkshopEventType.RUNTIME_PROFILE_ASSIGNED,
        event_version=1,
        workshop_id=workshop_id,
        aggregate_type="runtime_assignment",
        aggregate_id=assignment_id,
        occurred_at=occurred_at,
        idempotency_key=idempotency_key,
        payload={
            "channel_id": channel_id,
            "agent_id": agent_id,
            "runtime_profile_id": normalize_runtime_profile_id(runtime_profile_id),
        },
        metadata={"source": source},
    )


async def resolve_channel_runtime_profile(
    store: WorkshopEventStore,
    channel_id: ChannelId,
    agent_id: AgentId | None = None,
) -> tuple[AgentId, str]:
    """Resolve exactly one explicit runtime assignment for a channel agent."""
    if not isinstance(channel_id, ChannelId):
        raise WorkshopRuntimeAssignmentError("channel_id must be a ChannelId")
    if agent_id is not None and not isinstance(agent_id, AgentId):
        raise WorkshopRuntimeAssignmentError("agent_id must be an AgentId when provided")
    parameters: tuple[object, ...]
    agent_clause = ""
    if agent_id is None:
        parameters = (channel_id,)
    else:
        agent_clause = " AND ra.agent_id = ?"
        parameters = (channel_id, agent_id)
    async with store.connection.execute(
        "SELECT ra.agent_id, ra.runtime_profile_id "
        "FROM channel_agent_runtime_assignments ra "
        "JOIN channel_agents ca ON ca.channel_id = ra.channel_id AND ca.agent_id = ra.agent_id "
        "WHERE ra.channel_id = ?" + agent_clause,
        parameters,
    ) as cursor:
        rows = list(await cursor.fetchall())
    if len(rows) != 1:
        raise WorkshopRuntimeAssignmentError("Channel agent requires exactly one explicit runtime profile assignment")
    return AgentId(str(rows[0][0])), normalize_runtime_profile_id(str(rows[0][1]))


def compatibility_user_id(runtime_profile_id: str) -> int:
    """Adapt a protected profile ID to the current integer-keyed runtime pool."""
    normalized = normalize_runtime_profile_id(runtime_profile_id)
    if not normalized.isascii() or not normalized.isdigit() or normalized.startswith("0"):
        raise WorkshopRuntimeAssignmentError(
            "Runtime profile is not compatible with the current integer-keyed host runtime"
        )
    value = int(normalized)
    if value <= 0:
        raise WorkshopRuntimeAssignmentError(
            "Runtime profile is not compatible with the current integer-keyed host runtime"
        )
    return value


class WorkshopRuntimeAssignmentService:
    """Grant a canonical channel agent one protected runtime profile."""

    def __init__(self, store: WorkshopEventStore) -> None:
        self._store = store

    async def assign(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
        runtime_profile_id: str,
    ) -> WorkshopRuntimeAssignment:
        if not isinstance(principal_id, PrincipalId) or not isinstance(channel_id, ChannelId):
            raise WorkshopRuntimeAssignmentError("Canonical principal and channel IDs are required")
        normalized_profile = normalize_runtime_profile_id(runtime_profile_id)
        compatibility_user_id(normalized_profile)
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            workshop_id, agent_id = await self._resolve_owned_channel_agent(
                principal_id,
                channel_id,
            )
            assignment_id = RuntimeAssignmentId.derived(
                channel_id,
                f"runtime-profile:{agent_id}",
            )
            existing = await self._existing_assignment(channel_id, agent_id)
            if existing is not None:
                if existing != (assignment_id, normalized_profile):
                    raise WorkshopRuntimeAssignmentError(
                        "Channel agent already has a different runtime profile assignment"
                    )
                await connection.commit()
                return WorkshopRuntimeAssignment(
                    assignment_id,
                    workshop_id,
                    principal_id,
                    channel_id,
                    agent_id,
                    normalized_profile,
                    False,
                )
            await self._ensure_profile_unassigned(normalized_profile)
            event = runtime_assignment_envelope(
                workshop_id=workshop_id,
                assignment_id=assignment_id,
                channel_id=channel_id,
                agent_id=agent_id,
                runtime_profile_id=normalized_profile,
                occurred_at=datetime.now(UTC),
                idempotency_key=f"operator:runtime-assignment:{channel_id}:{agent_id}",
                source="operator_cli",
            )
            result = await self._store.append_in_transaction(event)
            if not result.inserted:
                raise WorkshopRuntimeAssignmentError("Runtime assignment event exists without its canonical projection")
            await self._store.project_pending_in_transaction(CanonicalConversationProjection())
            await connection.commit()
        except IdempotencyConflictError as exc:
            await connection.rollback()
            raise WorkshopRuntimeAssignmentError(
                "Channel agent runtime assignment conflicts with an existing event"
            ) from exc
        except Exception:
            await connection.rollback()
            raise
        return WorkshopRuntimeAssignment(
            assignment_id,
            workshop_id,
            principal_id,
            channel_id,
            agent_id,
            normalized_profile,
            True,
        )

    async def _resolve_owned_channel_agent(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
    ) -> tuple[WorkshopId, AgentId]:
        async with self._store.connection.execute(
            "SELECT c.workshop_id, ca.agent_id FROM principals p "
            "JOIN workshop_memberships wm ON wm.principal_id = p.id "
            "JOIN channels c ON c.workshop_id = wm.workshop_id AND c.id = ? AND c.kind = 'direct' "
            "JOIN channel_memberships cm ON cm.channel_id = c.id AND cm.principal_id = p.id "
            "AND cm.role = 'owner' JOIN channel_agents ca ON ca.channel_id = c.id "
            "WHERE p.id = ? AND p.kind = 'human'",
            (channel_id, principal_id),
        ) as cursor:
            rows = list(await cursor.fetchall())
        if len(rows) != 1:
            raise WorkshopRuntimeAssignmentError(
                "Canonical human must own a direct channel with exactly one attached agent"
            )
        return WorkshopId(str(rows[0][0])), AgentId(str(rows[0][1]))

    async def _existing_assignment(
        self,
        channel_id: ChannelId,
        agent_id: AgentId,
    ) -> tuple[RuntimeAssignmentId, str] | None:
        async with self._store.connection.execute(
            "SELECT id, runtime_profile_id FROM channel_agent_runtime_assignments "
            "WHERE channel_id = ? AND agent_id = ?",
            (channel_id, agent_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return RuntimeAssignmentId(str(row[0])), str(row[1])

    async def _ensure_profile_unassigned(self, runtime_profile_id: str) -> None:
        async with self._store.connection.execute(
            "SELECT 1 FROM channel_agent_runtime_assignments WHERE runtime_profile_id = ?",
            (runtime_profile_id,),
        ) as cursor:
            assigned = await cursor.fetchone() is not None
        if assigned:
            raise WorkshopRuntimeAssignmentError("Runtime profile is already assigned to another channel agent")
