"""Canonical single-owner authority for Workshop agents and protected runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from kai.workshop.domain import (
    AgentDefinitionId,
    EventEnvelope,
    PrincipalId,
    RuntimeProfileId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.runtime_profiles import WorkshopRuntimeProfileRegistry
from kai.workshop.store import WorkshopEventStore


class WorkshopAgentAuthorityError(RuntimeError):
    """Agent ownership or runtime sponsorship is incomplete or ambiguous."""


@dataclass(frozen=True, slots=True)
class WorkshopAgentAuthorityReconciliation:
    profiles: int
    captured_profile_owners: int
    definitions: int
    assigned_owners: int
    assigned_runtimes: int


async def reconcile_single_owner_agent_authority(
    store: WorkshopEventStore,
    runtime_profiles: WorkshopRuntimeProfileRegistry,
) -> WorkshopAgentAuthorityReconciliation:
    """Capture profile ownership and converge every agent on one owner runtime."""
    connection = store.connection
    captured = 0
    owners = 0
    runtimes = 0
    try:
        await connection.execute("BEGIN IMMEDIATE")
        for profile in runtime_profiles.profiles:
            async with connection.execute(
                "SELECT principal_id FROM runtime_profile_owners WHERE runtime_profile_id = ?",
                (profile.profile_id,),
            ) as cursor:
                owner_row = await cursor.fetchone()
            if owner_row is None:
                async with connection.execute(
                    "SELECT DISTINCT cm.principal_id FROM channel_agent_runtime_assignments ra "
                    "JOIN channels c ON c.id = ra.channel_id AND c.kind = 'direct' "
                    "JOIN channel_memberships cm ON cm.channel_id = c.id AND cm.role = 'owner' "
                    "JOIN principals p ON p.id = cm.principal_id AND p.kind = 'human' "
                    "WHERE ra.runtime_profile_id = ? ORDER BY cm.principal_id",
                    (profile.profile_id,),
                ) as cursor:
                    owner_rows = list(await cursor.fetchall())
                if len(owner_rows) != 1:
                    raise WorkshopAgentAuthorityError(
                        f"Runtime profile {profile.profile_id} must have exactly one established human owner"
                    )
                await connection.execute(
                    "INSERT INTO runtime_profile_owners (runtime_profile_id, principal_id) VALUES (?, ?)",
                    (profile.profile_id, str(owner_rows[0][0])),
                )
                captured += 1

        async with connection.execute(
            "SELECT d.id, d.workshop_id, d.owner_principal_id, e.actor_principal_id "
            "FROM agent_definitions d JOIN event_log e ON e.position = d.created_event_position "
            "ORDER BY d.created_event_position"
        ) as cursor:
            definitions = list(await cursor.fetchall())
        for row in definitions:
            definition_id = AgentDefinitionId(str(row[0]))
            workshop_id = WorkshopId(str(row[1]))
            owner_principal_id = PrincipalId(str(row[2])) if row[2] is not None else None
            creator_id = PrincipalId(str(row[3])) if row[3] is not None else None
            if owner_principal_id is None:
                owner_principal_id = await _resolve_initial_owner(
                    store,
                    workshop_id,
                    creator_id,
                )
                owner_event = EventEnvelope.create(
                    event_type=WorkshopEventType.AGENT_DEFINITION_AUTHORITY_ASSIGNED,
                    event_version=1,
                    workshop_id=workshop_id,
                    aggregate_type="agent_definition",
                    aggregate_id=definition_id,
                    actor_principal_id=owner_principal_id,
                    occurred_at=datetime.now(UTC),
                    idempotency_key=f"agent-authority:{definition_id}:owner:{owner_principal_id}",
                    payload={"owner_principal_id": owner_principal_id},
                    metadata={"source": "single_owner_agent_authority_migration"},
                )
                await store.append_in_transaction(owner_event)
                await store.project_pending_in_transaction(CanonicalConversationProjection())
                owners += 1

            async with connection.execute(
                "SELECT e.runtime_profile_id, e.direct_channel_id, ro.principal_id "
                "FROM principal_agent_enablements e "
                "JOIN runtime_profile_owners ro ON ro.runtime_profile_id = e.runtime_profile_id "
                "WHERE e.agent_definition_id = ? AND e.principal_id = ? "
                "AND e.lifecycle_state = 'enabled'",
                (definition_id, owner_principal_id),
            ) as cursor:
                runtime_rows = list(await cursor.fetchall())
            if not runtime_rows:
                # Draft definitions may exist before their owner activates a runtime.
                async with connection.execute(
                    "SELECT lifecycle_state FROM agent_definitions WHERE id = ?",
                    (definition_id,),
                ) as cursor:
                    lifecycle_row = await cursor.fetchone()
                if lifecycle_row is not None and str(lifecycle_row[0]) == "draft":
                    continue
                raise WorkshopAgentAuthorityError(
                    f"Agent definition {definition_id} has no enabled runtime owned by its owner"
                )
            if len(runtime_rows) != 1 or str(runtime_rows[0][2]) != str(owner_principal_id):
                raise WorkshopAgentAuthorityError(
                    f"Agent definition {definition_id} has ambiguous owner runtime authority"
                )
            runtime_profile_id = RuntimeProfileId(str(runtime_rows[0][0]))
            owner_channel_id = str(runtime_rows[0][1])
            async with connection.execute(
                "SELECT owner_principal_id, owner_runtime_profile_id, owner_direct_channel_id "
                "FROM agent_definitions WHERE id = ?",
                (definition_id,),
            ) as cursor:
                current = await cursor.fetchone()
            assert current is not None
            desired = (str(owner_principal_id), str(runtime_profile_id), owner_channel_id)
            actual = tuple(None if value is None else str(value) for value in current)
            if actual == desired:
                continue
            authority_event = EventEnvelope.create(
                event_type=WorkshopEventType.AGENT_DEFINITION_AUTHORITY_ASSIGNED,
                event_version=2,
                workshop_id=workshop_id,
                aggregate_type="agent_definition",
                aggregate_id=definition_id,
                actor_principal_id=owner_principal_id,
                occurred_at=datetime.now(UTC),
                idempotency_key=(
                    f"agent-authority:{definition_id}:runtime:{runtime_profile_id}:channel:{owner_channel_id}"
                ),
                payload={
                    "owner_principal_id": owner_principal_id,
                    "runtime_profile_id": runtime_profile_id,
                    "owner_direct_channel_id": owner_channel_id,
                },
                metadata={"source": "single_owner_agent_authority_migration"},
            )
            await store.append_in_transaction(authority_event)
            await store.project_pending_in_transaction(CanonicalConversationProjection())
            runtimes += 1
        await connection.commit()
    except Exception:
        await connection.rollback()
        raise
    return WorkshopAgentAuthorityReconciliation(
        profiles=len(runtime_profiles.profiles),
        captured_profile_owners=captured,
        definitions=len(definitions),
        assigned_owners=owners,
        assigned_runtimes=runtimes,
    )


async def _resolve_initial_owner(
    store: WorkshopEventStore,
    workshop_id: WorkshopId,
    creator_id: PrincipalId | None,
) -> PrincipalId:
    if creator_id is not None:
        async with store.connection.execute(
            "SELECT 1 FROM workshop_memberships wm JOIN principals p "
            "ON p.id = wm.principal_id AND p.kind = 'human' "
            "WHERE wm.workshop_id = ? AND wm.principal_id = ?",
            (workshop_id, creator_id),
        ) as cursor:
            if await cursor.fetchone() is not None:
                return creator_id
    async with store.connection.execute(
        "SELECT wm.principal_id FROM workshop_memberships wm JOIN principals p "
        "ON p.id = wm.principal_id AND p.kind = 'human' "
        "WHERE wm.workshop_id = ? AND wm.role = 'admin' ORDER BY wm.principal_id",
        (workshop_id,),
    ) as cursor:
        rows = list(await cursor.fetchall())
    if len(rows) != 1:
        raise WorkshopAgentAuthorityError(
            f"Workshop {workshop_id} must have one administrator to own its predefined agent"
        )
    return PrincipalId(str(rows[0][0]))
