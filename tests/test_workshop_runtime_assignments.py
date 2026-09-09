"""Explicit Workshop channel-agent runtime authority contracts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.conversation_commands import WorkshopConversationCommandService
from kai.workshop.conversation_runs import resolve_canonical_conversation_run
from kai.workshop.domain import (
    ChannelAgentId,
    ChannelId,
    ChannelMembershipId,
    EventEnvelope,
    MessageId,
    PrincipalId,
    RuntimeProfileId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.human_provisioning import WorkshopHumanProvisioner
from kai.workshop.inbound import ClientInboundMessage
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.runtime_assignments import (
    WorkshopRuntimeAssignmentError,
    WorkshopRuntimeAssignmentService,
    resolve_channel_runtime_profile,
)
from kai.workshop.runtime_profiles import WorkshopRuntimeProfileError, WorkshopRuntimeProfileRegistry
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id, profile_registry


async def _store(path: Path) -> WorkshopEventStore:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (BootstrapHuman("Alice", "admin", "telegram", "101", "101", profile_id(101)),),
    )
    return store


async def _unassigned_direct_lane(store: WorkshopEventStore, principal_id: PrincipalId) -> ChannelId:
    async with store.connection.execute(
        "SELECT wm.workshop_id, a.id, a.principal_id FROM workshop_memberships wm "
        "JOIN agents a ON a.workshop_id = wm.workshop_id AND a.name = 'Kai' "
        "WHERE wm.principal_id = ?",
        (principal_id,),
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    workshop_id = WorkshopId(str(row[0]))
    agent_id = str(row[1])
    agent_principal_id = str(row[2])
    channel_id = ChannelId.derived(principal_id, "runtime-assignment-test")
    now = datetime.now(UTC)
    events = (
        EventEnvelope.create(
            event_type=WorkshopEventType.CHANNEL_CREATED,
            event_version=1,
            workshop_id=workshop_id,
            aggregate_type="channel",
            aggregate_id=channel_id,
            occurred_at=now,
            idempotency_key=f"test-runtime-assignment:{principal_id}:channel",
            payload={"kind": "direct", "name": "Direct"},
        ),
        EventEnvelope.create(
            event_type=WorkshopEventType.CHANNEL_MEMBER_ADDED,
            event_version=1,
            workshop_id=workshop_id,
            aggregate_type="channel_membership",
            aggregate_id=ChannelMembershipId.derived(channel_id, f"human:{principal_id}"),
            occurred_at=now,
            idempotency_key=f"test-runtime-assignment:{principal_id}:human",
            payload={"channel_id": channel_id, "principal_id": principal_id, "role": "owner"},
        ),
        EventEnvelope.create(
            event_type=WorkshopEventType.CHANNEL_MEMBER_ADDED,
            event_version=1,
            workshop_id=workshop_id,
            aggregate_type="channel_membership",
            aggregate_id=ChannelMembershipId.derived(channel_id, f"agent:{agent_principal_id}"),
            occurred_at=now,
            idempotency_key=f"test-runtime-assignment:{principal_id}:agent",
            payload={"channel_id": channel_id, "principal_id": agent_principal_id, "role": "participant"},
        ),
        EventEnvelope.create(
            event_type=WorkshopEventType.CHANNEL_AGENT_ATTACHED,
            event_version=1,
            workshop_id=workshop_id,
            aggregate_type="channel_agent",
            aggregate_id=ChannelAgentId.derived(channel_id, f"agent:{agent_id}"),
            occurred_at=now,
            idempotency_key=f"test-runtime-assignment:{principal_id}:attachment",
            payload={"channel_id": channel_id, "agent_id": agent_id},
        ),
    )
    for event in events:
        await store.append(event)
    await store.project_pending(CanonicalConversationProjection())
    return channel_id


class TestRuntimeAssignmentPolicy:
    async def test_non_telegram_policy_profile_can_be_assigned_to_browser_only_human(self, tmp_path: Path):
        store = await _store(tmp_path / "kai.db")
        runtime_profile_id = RuntimeProfileId("rtp_99999999999999999999999999999999")
        profiles = WorkshopRuntimeProfileRegistry.from_document(
            {
                "version": 1,
                "runtime_profiles": {
                    str(runtime_profile_id): {
                        "display_name": "Browser coding",
                        "backend": "codex",
                        "provider": "openai",
                        "model": "gpt-5.5",
                        "timeout_seconds": 120,
                        "allowed_services": [],
                        "allowed_workspaces": [],
                    }
                },
            },
            backend_registry={"codex": {}},
        )
        try:
            human = await WorkshopHumanProvisioner(store).provision(
                "browser-human",
                "Browser human",
                "member",
            )
            channel_id = await _unassigned_direct_lane(store, human.principal_id)

            assigned = await WorkshopRuntimeAssignmentService(store, profiles).assign(
                human.principal_id,
                channel_id,
                runtime_profile_id,
            )

            assert assigned.runtime_profile_id == runtime_profile_id
            async with store.connection.execute(
                "SELECT COUNT(*) FROM external_identities WHERE principal_id = ?",
                (human.principal_id,),
            ) as cursor:
                assert int((await cursor.fetchone())[0]) == 0
        finally:
            await store.close()

    async def test_provisioned_human_uses_agent_owner_runtime_after_access_assignment(
        self,
        tmp_path: Path,
    ):
        store = await _store(tmp_path / "kai.db")
        try:
            human = await WorkshopHumanProvisioner(store).provision(
                "charlie",
                "Charlie",
                "member",
            )
            channel_id = await _unassigned_direct_lane(store, human.principal_id)
            profiles = profile_registry(101, 202)
            service = WorkshopRuntimeAssignmentService(store, profiles)

            assigned = await service.assign(
                human.principal_id,
                channel_id,
                profile_id(202),
            )
            retried = await service.assign(
                human.principal_id,
                channel_id,
                profile_id(202),
            )

            assert assigned.created is True
            assert retried.created is False
            assert retried.assignment_id == assigned.assignment_id
            assert await resolve_channel_runtime_profile(store, channel_id) == (
                assigned.agent_id,
                profile_id(202),
            )
            async with store.connection.execute(
                "SELECT COUNT(*) FROM external_identities WHERE principal_id = ?",
                (human.principal_id,),
            ) as cursor:
                assert int((await cursor.fetchone())[0]) == 0

            accepted = await WorkshopConversationCommandService(store).accept_client(
                ClientInboundMessage(
                    principal_id=human.principal_id,
                    channel_id=channel_id,
                    client_message_id="charlie-command-1",
                    body="Use the agent owner's runtime",
                    occurred_at=datetime.now(UTC),
                )
            )
            resolution = await resolve_canonical_conversation_run(
                store,
                MessageId(str(accepted.command.message.event.envelope.aggregate_id)),
            )

            assert accepted.runtime_profile_id == profile_id(101)
            assert resolution.runtime_profile_id == profile_id(101)
        finally:
            await store.close()

    async def test_assignment_rejects_cross_human_authority_and_profile_reuse(
        self,
        tmp_path: Path,
    ):
        store = await _store(tmp_path / "kai.db")
        try:
            charlie = await WorkshopHumanProvisioner(store).provision(
                "charlie",
                "Charlie",
                "member",
            )
            dana = await WorkshopHumanProvisioner(store).provision(
                "dana",
                "Dana",
                "member",
            )
            charlie_channel = await _unassigned_direct_lane(store, charlie.principal_id)
            dana_channel = await _unassigned_direct_lane(store, dana.principal_id)
            service = WorkshopRuntimeAssignmentService(store, profile_registry(101, 202))

            with pytest.raises(WorkshopRuntimeAssignmentError, match="must own"):
                await service.assign(charlie.principal_id, dana_channel, profile_id(202))

            await service.assign(charlie.principal_id, charlie_channel, profile_id(202))
            with pytest.raises(WorkshopRuntimeAssignmentError, match="already assigned"):
                await service.assign(dana.principal_id, dana_channel, profile_id(202))
        finally:
            await store.close()

    async def test_projection_rebuild_restores_runtime_assignment(self, tmp_path: Path):
        store = await _store(tmp_path / "kai.db")
        try:
            human = await WorkshopHumanProvisioner(store).provision(
                "charlie",
                "Charlie",
                "member",
            )
            channel_id = await _unassigned_direct_lane(store, human.principal_id)
            assigned = await WorkshopRuntimeAssignmentService(store, profile_registry(101, 202)).assign(
                human.principal_id,
                channel_id,
                profile_id(202),
            )
            await store.connection.execute("DELETE FROM channel_agent_runtime_assignments")
            await store.connection.commit()

            checkpoint = await store.rebuild_projection(CanonicalConversationProjection())

            assert checkpoint.version == 32
            assert await resolve_channel_runtime_profile(store, channel_id) == (
                assigned.agent_id,
                profile_id(202),
            )
        finally:
            await store.close()


class TestRuntimeProfileCompatibilityBoundary:
    @pytest.mark.parametrize("value", ("profile-daniel", "0", "01"))
    def test_protected_registry_rejects_non_profile_ids(self, value: str):
        with pytest.raises(WorkshopRuntimeProfileError, match="invalid"):
            profile_registry(101).resolve(value)

    def test_opaque_profile_does_not_encode_runtime_configuration_key(self):
        profiles = profile_registry(202)
        profile = profiles.resolve(profile_id(202))

        assert profile.profile_id != str(profiles.legacy_runtime_key(profile.profile_id))
        assert "202" not in profile.profile_id
