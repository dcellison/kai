"""Contracts for canonical Workshop standing-participation authority."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from kai.workshop.agent_lifecycle import WorkshopAgentLifecycleService
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.channel_lifecycle import WorkshopChannelLifecycleService
from kai.workshop.collaboration_authority import (
    CollaborationHostPolicy,
    StandingParticipationHostPolicy,
    WorkshopCollaborationAuthority,
)
from kai.workshop.collaboration_policy import WorkshopCollaborationPolicyService
from kai.workshop.conversation_commands import WorkshopConversationCommandService
from kai.workshop.diagnostics import workshop_standing_participation_status
from kai.workshop.domain import AgentId, ChannelId, PrincipalId
from kai.workshop.inbound import ClientInboundMessage
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.standing_participation import WorkshopStandingParticipationService
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id

_NOW = datetime(2026, 9, 9, 12, 0, tzinfo=UTC)


class _PrivateExecution:
    def __init__(self, authority: WorkshopCollaborationAuthority) -> None:
        self.collaboration_authority = authority

    async def revoke_collaboration_for_definition(self, *_args: object, **_kwargs: object) -> int:
        return 0


async def _eligible_authority(
    path: Path,
) -> tuple[
    WorkshopEventStore,
    PrincipalId,
    ChannelId,
    AgentId,
    WorkshopStandingParticipationService,
]:
    store, human_id, channel_id, agent_id = await _base_group_store(path)
    lifecycle = WorkshopAgentLifecycleService(store)
    definition = next(item for item in await lifecycle.list_visible(human_id) if item.agent_id == agent_id)
    revised = await lifecycle.add_revision(
        human_id,
        definition.definition_id,
        idempotency_key="standing-revision",
        expected_version=definition.state_version,
        purpose="Participate selectively in opted-in Workshop channels.",
        instructions="Respond when mentioned and otherwise observe only when authorized.",
        capabilities=["text_generation"],
        collaboration_operations=["standing_participation"],
    )
    active = await lifecycle.activate_revision(
        human_id,
        definition.definition_id,
        revision_id=revised.revisions[-1].revision_id,
        idempotency_key="standing-activate",
        expected_version=revised.state_version,
    )
    host_policy = CollaborationHostPolicy(standing_participation=StandingParticipationHostPolicy(enabled=True))
    authority = WorkshopCollaborationAuthority(store, host_policy=host_policy)
    owner_policy = WorkshopCollaborationPolicyService(
        store,
        cast(Any, _PrivateExecution(authority)),
    )
    allowed = await owner_policy.set_allowed(
        human_id,
        active.definition_id,
        allowed_operations=["standing_participation"],
        expected_policy_version=0,
        client_operation_id="standing-owner-allow",
    )
    assert allowed.snapshot.policy_version == 1
    standing = WorkshopStandingParticipationService(store, host_policy)
    enabled = await standing.set_channel_policy(
        human_id,
        channel_id,
        enabled=True,
        expected_policy_version=0,
        client_operation_id="standing-channel-enable",
    )
    assert enabled.snapshot.policy.enabled is True
    return store, human_id, channel_id, agent_id, standing


async def _base_group_store(
    path: Path,
) -> tuple[WorkshopEventStore, PrincipalId, ChannelId, AgentId]:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman(
                display_name="Daniel",
                role="admin",
                transport="telegram",
                external_subject="101",
                external_channel_id="101",
                runtime_profile_id=profile_id(101),
            ),
        ),
    )
    async with store.connection.execute(
        "SELECT p.id, a.id FROM external_identities ei JOIN principals p ON p.id = ei.principal_id "
        "JOIN agent_definitions d ON d.owner_principal_id = p.id JOIN agents a ON a.id = d.agent_id "
        "WHERE ei.provider = 'telegram' AND ei.external_subject = '101'"
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    human_id = PrincipalId(str(row[0]))
    agent_id = AgentId(str(row[1]))
    group = await WorkshopChannelLifecycleService(store).create_group(
        human_id,
        name="Standing participation qualification",
        agent_ids=[agent_id],
    )
    return store, human_id, group.channel_id, agent_id


def _message(principal_id: PrincipalId, channel_id: ChannelId, identity: str) -> ClientInboundMessage:
    return ClientInboundMessage(
        principal_id=principal_id,
        channel_id=channel_id,
        client_message_id=identity,
        body="@kai Please reply to this qualification.",
        occurred_at=_NOW,
    )


async def test_explicit_mention_starts_one_replayable_subscription_without_replacing_response(
    tmp_path: Path,
) -> None:
    store, human_id, channel_id, agent_id, standing = await _eligible_authority(tmp_path / "kai.db")
    try:
        commands = WorkshopConversationCommandService(store, standing_participation=standing)
        command = _message(human_id, channel_id, "standing-mention-1")
        accepted = await commands.accept_client(command)
        replay = await commands.accept_client(command)

        assert len(accepted.command.runs) == 1
        assert len(replay.command.runs) == 1
        snapshot = await standing.inspect(human_id, channel_id)
        active = [item for item in snapshot.subscriptions if item.lifecycle_state == "active"]
        assert [item.agent_id for item in active] == [agent_id]
        assert active[0].owner_policy_version == 1
        assert active[0].channel_policy_version == 1
        assert active[0].host_policy_version == 2
        async with store.connection.execute("SELECT COUNT(*) FROM collaboration_grants") as cursor:
            grants = await cursor.fetchone()
        assert grants is not None and int(grants[0]) == 0
        async with store.connection.execute(
            "SELECT COUNT(*) FROM event_log WHERE event_type = 'channel.agent_standing_started'"
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None and int(row[0]) == 1
    finally:
        await store.close()


async def test_host_revocation_is_fenced_lazily_and_diagnostics_report_clean_state(
    tmp_path: Path,
) -> None:
    path = tmp_path / "kai.db"
    store, human_id, channel_id, _agent_id, standing = await _eligible_authority(path)
    try:
        await WorkshopConversationCommandService(store, standing_participation=standing).accept_client(
            _message(human_id, channel_id, "standing-host-revocation")
        )
        disabled_host = WorkshopStandingParticipationService(store, CollaborationHostPolicy())

        fenced = await disabled_host.inspect(human_id, channel_id)

        assert [item.lifecycle_state for item in fenced.subscriptions] == ["ended"]
        assert fenced.subscriptions[0].end_reason == "host_policy_revoked"
    finally:
        await store.close()

    status = workshop_standing_participation_status(path, host_enabled=False)
    assert status.startswith("Workshop standing participation: active;")
    assert "host=disabled" in status
    assert "channel policies=1 (enabled=1)" in status
    assert "subscriptions=1 (active=0, paused overflow=0, ended=1)" in status
    assert "integrity gaps=0, replay gaps=0" in status


async def test_agent_detachment_explicitly_ends_active_subscription(tmp_path: Path) -> None:
    store, human_id, channel_id, agent_id, standing = await _eligible_authority(tmp_path / "kai.db")
    try:
        await WorkshopConversationCommandService(store, standing_participation=standing).accept_client(
            _message(human_id, channel_id, "standing-channel-archive")
        )

        await WorkshopChannelLifecycleService(store).detach_agent(
            human_id,
            channel_id,
            agent_id,
            client_operation_id="standing-agent-detach-1",
        )

        snapshot = await standing.inspect(human_id, channel_id)
        assert [item.lifecycle_state for item in snapshot.subscriptions] == ["ended"]
        assert snapshot.subscriptions[0].end_reason == "detached"
    finally:
        await store.close()


async def test_channel_policy_disable_ends_subscription_and_rebuild_preserves_terminal_reason(
    tmp_path: Path,
) -> None:
    store, human_id, channel_id, _agent_id, standing = await _eligible_authority(tmp_path / "kai.db")
    try:
        await WorkshopConversationCommandService(store, standing_participation=standing).accept_client(
            _message(human_id, channel_id, "standing-mention-2")
        )
        disabled = await standing.set_channel_policy(
            human_id,
            channel_id,
            enabled=False,
            expected_policy_version=1,
            client_operation_id="standing-channel-disable",
        )
        assert disabled.snapshot.policy.enabled is False
        assert [item.lifecycle_state for item in disabled.snapshot.subscriptions] == ["ended"]
        assert disabled.snapshot.subscriptions[0].end_reason == "channel_policy_disabled"

        replay = await standing.set_channel_policy(
            human_id,
            channel_id,
            enabled=False,
            expected_policy_version=1,
            client_operation_id="standing-channel-disable",
        )
        assert replay.replayed is True
        async with store.connection.execute(
            "SELECT COUNT(*) FROM event_log WHERE event_type = 'channel.agent_standing_ended'"
        ) as cursor:
            ended_events = await cursor.fetchone()
        assert ended_events is not None and int(ended_events[0]) == 1

        await store.rebuild_projection(CanonicalConversationProjection())
        rebuilt = await standing.inspect(human_id, channel_id)
        assert rebuilt == disabled.snapshot
    finally:
        await store.close()


async def test_host_rollout_disabled_keeps_mention_response_without_subscription(tmp_path: Path) -> None:
    store, human_id, channel_id, _agent_id = await _base_group_store(tmp_path / "kai.db")
    try:
        standing = WorkshopStandingParticipationService(store, CollaborationHostPolicy())
        accepted = await WorkshopConversationCommandService(store, standing_participation=standing).accept_client(
            _message(human_id, channel_id, "standing-host-off")
        )
        assert len(accepted.command.runs) == 1
        assert (await standing.inspect(human_id, channel_id)).subscriptions == ()
    finally:
        await store.close()
