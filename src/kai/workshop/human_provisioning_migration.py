"""Retire direct channels created by the legacy human provisioner."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from kai.workshop.domain import ChannelId, EventEnvelope, PrincipalId, WorkshopEventType, WorkshopId
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import WorkshopEventStore


@dataclass(frozen=True, slots=True)
class WorkshopHumanProvisioningMigration:
    identities: int
    legacy_channels: int
    retained_channels: int
    archived_channels: int
    restored_channels: int
    unresolved_channels: int


async def reconcile_legacy_human_provisioning_channels(
    store: WorkshopEventStore,
) -> WorkshopHumanProvisioningMigration:
    """Retain operational legacy lanes and archive only truly empty ones."""
    async with store.connection.execute(
        "SELECT COUNT(*) FROM event_log WHERE event_type = 'principal.created' "
        "AND idempotency_key LIKE 'operator:human-provisioning:%:principal'"
    ) as cursor:
        identity_row = await cursor.fetchone()
    assert identity_row is not None
    identities = int(identity_row[0])
    async with store.connection.execute(
        "SELECT c.workshop_id, c.id, cm.principal_id, c.archived_at, "
        "CASE WHEN pae.id IS NULL THEN 0 ELSE 1 END, "
        "CASE WHEN EXISTS(SELECT 1 FROM channel_agent_runtime_assignments ra "
        "WHERE ra.channel_id = c.id) "
        "OR EXISTS(SELECT 1 FROM messages m WHERE m.channel_id = c.id) "
        "OR EXISTS(SELECT 1 FROM runs r WHERE r.channel_id = c.id) "
        "OR EXISTS(SELECT 1 FROM channel_bindings cb WHERE cb.channel_id = c.id) "
        "THEN 1 ELSE 0 END, "
        "CASE WHEN EXISTS(SELECT 1 FROM event_log restored "
        "WHERE restored.aggregate_id = c.id AND restored.event_type = 'channel.restored' "
        "AND json_extract(restored.metadata_json, '$.source') = 'human_provisioning_migration') "
        "THEN 1 ELSE 0 END "
        "FROM event_log e JOIN channels c ON c.id = e.aggregate_id AND c.kind = 'direct' "
        "JOIN channel_memberships cm ON cm.channel_id = c.id AND cm.role = 'owner' "
        "JOIN principals p ON p.id = cm.principal_id AND p.kind = 'human' "
        "LEFT JOIN principal_agent_enablements pae ON pae.direct_channel_id = c.id "
        "WHERE e.event_type = 'channel.created' "
        "AND e.idempotency_key LIKE 'operator:human-provisioning:%:direct-channel' "
        "ORDER BY c.id"
    ) as cursor:
        rows = list(await cursor.fetchall())

    retained = 0
    archived = 0
    restored = 0
    unresolved = 0
    connection = store.connection
    try:
        await connection.execute("BEGIN IMMEDIATE")
        for row in rows:
            workshop_id = WorkshopId(str(row[0]))
            channel_id = ChannelId(str(row[1]))
            principal_id = PrincipalId(str(row[2]))
            operational = bool(row[4]) or bool(row[5])
            if operational:
                retained += 1
                if row[3] is not None:
                    result = await store.append_in_transaction(
                        EventEnvelope.create(
                            event_type=WorkshopEventType.CHANNEL_RESTORED,
                            event_version=1,
                            workshop_id=workshop_id,
                            aggregate_type="channel",
                            aggregate_id=channel_id,
                            actor_principal_id=principal_id,
                            occurred_at=datetime.now(UTC),
                            idempotency_key=(f"human-provisioning-migration:{channel_id}:restore-operational"),
                            payload={},
                            metadata={
                                "source": "human_provisioning_migration",
                                "reason": "operational_authority",
                            },
                        )
                    )
                    restored += int(result.inserted)
                elif bool(row[6]):
                    restored += 1
                continue
            if row[3] is not None:
                archived += 1
                continue
            result = await store.append_in_transaction(
                EventEnvelope.create(
                    event_type=WorkshopEventType.CHANNEL_ARCHIVED,
                    event_version=1,
                    workshop_id=workshop_id,
                    aggregate_type="channel",
                    aggregate_id=channel_id,
                    actor_principal_id=principal_id,
                    occurred_at=datetime.now(UTC),
                    idempotency_key=f"human-provisioning-migration:{channel_id}:archive",
                    payload={},
                    metadata={"source": "human_provisioning_migration"},
                )
            )
            archived += int(result.inserted)
        await store.project_pending_in_transaction(CanonicalConversationProjection())
        await connection.commit()
    except Exception:
        await connection.rollback()
        raise
    return WorkshopHumanProvisioningMigration(
        identities,
        len(rows),
        retained,
        archived,
        restored,
        unresolved,
    )
