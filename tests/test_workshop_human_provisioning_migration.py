"""Legacy automatic-Kai channel retirement contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.diagnostics import workshop_human_provisioning_status
from kai.workshop.domain import (
    AgentId,
    ChannelAgentId,
    ChannelId,
    ChannelMembershipId,
    EventEnvelope,
    PrincipalId,
    RuntimeAssignmentId,
    WorkshopEventType,
)
from kai.workshop.human_provisioning import WorkshopHumanProvisioner
from kai.workshop.human_provisioning_migration import reconcile_legacy_human_provisioning_channels
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.storage_namespaces import WorkshopPrincipalStorageRegistry
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id, profile_registry


async def test_orphaned_legacy_provisioner_channel_is_archived_and_replay_safe(
    tmp_path: Path,
) -> None:
    path = tmp_path / "kai.db"
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (BootstrapHuman("Alice", "admin", "telegram", "101", "101", profile_id(101)),),
    )
    try:
        human = await WorkshopHumanProvisioner(store).provision("charlie", "Charlie", "member")
        async with store.connection.execute(
            "SELECT a.id, a.principal_id FROM agents a WHERE a.workshop_id = ? AND a.name = 'Kai'",
            (human.workshop_id,),
        ) as cursor:
            agent_row = await cursor.fetchone()
        assert agent_row is not None
        agent_id = AgentId(str(agent_row[0]))
        agent_principal_id = PrincipalId(str(agent_row[1]))
        stable_prefix = "operator-human:charlie"
        channel_id = ChannelId.derived(human.workshop_id, f"{stable_prefix}:direct-channel")
        now = datetime.now(UTC)
        legacy_events = (
            EventEnvelope.create(
                event_type=WorkshopEventType.CHANNEL_CREATED,
                event_version=1,
                workshop_id=human.workshop_id,
                aggregate_type="channel",
                aggregate_id=channel_id,
                occurred_at=now,
                idempotency_key=f"operator:human-provisioning:{human.principal_id}:direct-channel",
                payload={"kind": "direct", "name": "Direct"},
                metadata={"source": "operator_cli"},
            ),
            EventEnvelope.create(
                event_type=WorkshopEventType.CHANNEL_MEMBER_ADDED,
                event_version=1,
                workshop_id=human.workshop_id,
                aggregate_type="channel_membership",
                aggregate_id=ChannelMembershipId.derived(
                    human.workshop_id,
                    f"{stable_prefix}:human-channel-membership",
                ),
                occurred_at=now,
                idempotency_key=f"operator:human-provisioning:{human.principal_id}:human-channel-membership",
                payload={"channel_id": channel_id, "principal_id": human.principal_id, "role": "owner"},
                metadata={"source": "operator_cli"},
            ),
            EventEnvelope.create(
                event_type=WorkshopEventType.CHANNEL_MEMBER_ADDED,
                event_version=1,
                workshop_id=human.workshop_id,
                aggregate_type="channel_membership",
                aggregate_id=ChannelMembershipId.derived(
                    human.workshop_id,
                    f"{stable_prefix}:agent-channel-membership",
                ),
                occurred_at=now,
                idempotency_key=f"operator:human-provisioning:{human.principal_id}:agent-channel-membership",
                payload={"channel_id": channel_id, "principal_id": agent_principal_id, "role": "participant"},
                metadata={"source": "operator_cli"},
            ),
            EventEnvelope.create(
                event_type=WorkshopEventType.CHANNEL_AGENT_ATTACHED,
                event_version=1,
                workshop_id=human.workshop_id,
                aggregate_type="channel_agent",
                aggregate_id=ChannelAgentId.derived(
                    human.workshop_id,
                    f"{stable_prefix}:channel-agent",
                ),
                occurred_at=now,
                idempotency_key=f"operator:human-provisioning:{human.principal_id}:channel-agent",
                payload={"channel_id": channel_id, "agent_id": agent_id},
                metadata={"source": "operator_cli"},
            ),
        )
        for event in legacy_events:
            await store.append(event)
        await store.project_pending(CanonicalConversationProjection())

        migration = await reconcile_legacy_human_provisioning_channels(store)
        replay = await reconcile_legacy_human_provisioning_channels(store)
        retried_human = await WorkshopHumanProvisioner(store).provision("charlie", "Charlie", "member")

        assert migration.legacy_channels == 1
        assert migration.archived_channels == 1
        assert migration.restored_channels == 0
        assert migration.unresolved_channels == 0
        assert replay.archived_channels == 1
        assert retried_human.created is False
        async with store.connection.execute(
            "SELECT archived_at FROM channels WHERE id = ?",
            (channel_id,),
        ) as cursor:
            assert (await cursor.fetchone())[0] is not None
        await store.rebuild_projection(CanonicalConversationProjection())
        async with store.connection.execute(
            "SELECT archived_at FROM channels WHERE id = ?",
            (channel_id,),
        ) as cursor:
            assert (await cursor.fetchone())[0] is not None
        assert workshop_human_provisioning_status(path).startswith(
            "Workshop human provisioning: active; identities=1, legacy channels=1 "
            "(retained=0, archived=1, restored=0, unresolved=0), integrity gaps=0"
        )
    finally:
        await store.close()


async def test_operational_legacy_channel_is_restored_after_prior_retirement(
    tmp_path: Path,
) -> None:
    path = tmp_path / "kai.db"
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (BootstrapHuman("Alice", "admin", "telegram", "101", "101", profile_id(101)),),
    )
    try:
        human = await WorkshopHumanProvisioner(store).provision("charlie", "Charlie", "member")
        async with store.connection.execute(
            "SELECT a.id, a.principal_id FROM agents a WHERE a.workshop_id = ? AND a.name = 'Kai'",
            (human.workshop_id,),
        ) as cursor:
            agent_row = await cursor.fetchone()
        assert agent_row is not None
        agent_id = AgentId(str(agent_row[0]))
        agent_principal_id = PrincipalId(str(agent_row[1]))
        channel_id = ChannelId.derived(human.workshop_id, "operator-human:charlie:direct-channel")
        now = datetime.now(UTC)
        events = (
            EventEnvelope.create(
                event_type=WorkshopEventType.CHANNEL_CREATED,
                event_version=1,
                workshop_id=human.workshop_id,
                aggregate_type="channel",
                aggregate_id=channel_id,
                occurred_at=now,
                idempotency_key=f"operator:human-provisioning:{human.principal_id}:direct-channel",
                payload={"kind": "direct", "name": "Direct"},
                metadata={"source": "operator_cli"},
            ),
            EventEnvelope.create(
                event_type=WorkshopEventType.CHANNEL_MEMBER_ADDED,
                event_version=1,
                workshop_id=human.workshop_id,
                aggregate_type="channel_membership",
                aggregate_id=ChannelMembershipId.derived(channel_id, f"human:{human.principal_id}"),
                actor_principal_id=human.principal_id,
                occurred_at=now,
                idempotency_key=f"legacy-operational:{channel_id}:human",
                payload={"channel_id": channel_id, "principal_id": human.principal_id, "role": "owner"},
                metadata={"source": "operator_cli"},
            ),
            EventEnvelope.create(
                event_type=WorkshopEventType.CHANNEL_MEMBER_ADDED,
                event_version=1,
                workshop_id=human.workshop_id,
                aggregate_type="channel_membership",
                aggregate_id=ChannelMembershipId.derived(channel_id, f"agent:{agent_principal_id}"),
                actor_principal_id=human.principal_id,
                occurred_at=now,
                idempotency_key=f"legacy-operational:{channel_id}:agent",
                payload={"channel_id": channel_id, "principal_id": agent_principal_id, "role": "participant"},
                metadata={"source": "operator_cli"},
            ),
            EventEnvelope.create(
                event_type=WorkshopEventType.CHANNEL_AGENT_ATTACHED,
                event_version=1,
                workshop_id=human.workshop_id,
                aggregate_type="channel_agent",
                aggregate_id=ChannelAgentId.derived(channel_id, f"agent:{agent_id}"),
                actor_principal_id=human.principal_id,
                occurred_at=now,
                idempotency_key=f"legacy-operational:{channel_id}:attachment",
                payload={"channel_id": channel_id, "agent_id": agent_id},
                metadata={"source": "operator_cli"},
            ),
            EventEnvelope.create(
                event_type=WorkshopEventType.RUNTIME_PROFILE_ASSIGNED,
                event_version=1,
                workshop_id=human.workshop_id,
                aggregate_type="runtime_assignment",
                aggregate_id=RuntimeAssignmentId.derived(channel_id, f"runtime-profile:{agent_id}"),
                actor_principal_id=human.principal_id,
                occurred_at=now,
                idempotency_key=f"legacy-operational:{channel_id}:runtime",
                payload={
                    "channel_id": channel_id,
                    "agent_id": agent_id,
                    "runtime_profile_id": profile_id(202),
                },
                metadata={"source": "operator_cli"},
            ),
        )
        for event in events:
            await store.append(event)
        await store.project_pending(CanonicalConversationProjection())
        await store.append(
            EventEnvelope.create(
                event_type=WorkshopEventType.CHANNEL_ARCHIVED,
                event_version=1,
                workshop_id=human.workshop_id,
                aggregate_type="channel",
                aggregate_id=channel_id,
                actor_principal_id=human.principal_id,
                occurred_at=now,
                idempotency_key=f"human-provisioning-migration:{channel_id}:archive",
                payload={},
                metadata={"source": "human_provisioning_migration"},
            )
        )
        await store.project_pending(CanonicalConversationProjection())

        migration = await reconcile_legacy_human_provisioning_channels(store)
        replay = await reconcile_legacy_human_provisioning_channels(store)

        assert migration.retained_channels == 1
        assert migration.archived_channels == 0
        assert migration.restored_channels == 1
        assert replay.restored_channels == 1
        async with store.connection.execute(
            "SELECT archived_at FROM channels WHERE id = ?",
            (channel_id,),
        ) as cursor:
            assert (await cursor.fetchone())[0] is None
        storage = await WorkshopPrincipalStorageRegistry.from_store(
            store,
            profile_registry(101, 202),
        )
        assert storage.for_runtime_profile(profile_id(202)).principal_id == human.principal_id
        await store.rebuild_projection(CanonicalConversationProjection())
        async with store.connection.execute(
            "SELECT archived_at FROM channels WHERE id = ?",
            (channel_id,),
        ) as cursor:
            assert (await cursor.fetchone())[0] is None
        assert workshop_human_provisioning_status(path).startswith(
            "Workshop human provisioning: active; identities=1, legacy channels=1 "
            "(retained=1, archived=0, restored=1, unresolved=0), integrity gaps=0"
        )
    finally:
        await store.close()
