"""Contracts for canonical Workshop standing-participation authority."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
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
from kai.workshop.domain import (
    AgentId,
    ChannelId,
    EventEnvelope,
    EventId,
    MessageId,
    PrincipalId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.inbound import ClientInboundMessage
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.standing_participation import WorkshopStandingParticipationService
from kai.workshop.store import WorkshopEventStore
from kai.workshop.wake_policy import EngagementScope, dismiss_channel_agent
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


async def _start_without_run(
    store: WorkshopEventStore,
    standing: WorkshopStandingParticipationService,
    human_id: PrincipalId,
    channel_id: ChannelId,
    agent_id: AgentId,
    identity: str,
    *,
    occurred_at: datetime = _NOW,
) -> MessageId:
    """Start standing from a canonical mention without leaving a respond run."""
    async with store.connection.execute(
        "SELECT c.workshop_id, a.principal_id, d.handle FROM channels c "
        "JOIN agents a ON a.id = ? JOIN agent_definitions d ON d.agent_id = a.id "
        "WHERE c.id = ?",
        (agent_id, channel_id),
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    workshop_id = WorkshopId(str(row[0]))
    agent_principal_id = PrincipalId(str(row[1]))
    handle = str(row[2])
    body = f"@{handle} qualify standing lifecycle"
    message_id = MessageId.derived(channel_id, identity)
    event = EventEnvelope.create(
        event_id=EventId.derived(message_id, "created"),
        event_type=WorkshopEventType.MESSAGE_CREATED,
        event_version=2,
        workshop_id=workshop_id,
        aggregate_type="message",
        aggregate_id=message_id,
        actor_principal_id=human_id,
        occurred_at=occurred_at,
        idempotency_key=f"standing-qualification:{identity}",
        payload={
            "channel_id": channel_id,
            "author_principal_id": human_id,
            "body": body,
            "mentions": [
                {
                    "principal_id": agent_principal_id,
                    "kind": "agent",
                    "start": 0,
                    "length": len(handle) + 1,
                }
            ],
        },
        metadata={"source": "qualification"},
    )
    await store.connection.execute("BEGIN IMMEDIATE")
    try:
        await store.append_in_transaction(event)
        await store.project_pending_in_transaction(CanonicalConversationProjection())
        await standing.start_from_message_in_transaction(
            message_id,
            (agent_id,),
            occurred_at=occurred_at,
        )
        await store.connection.commit()
    except Exception:
        await store.connection.rollback()
        raise
    return message_id


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
        assert active[0].host_policy_version == 3
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
        disabled_host = WorkshopStandingParticipationService(
            store,
            CollaborationHostPolicy(
                standing_participation=StandingParticipationHostPolicy(enabled=False),
            ),
        )

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
        standing = WorkshopStandingParticipationService(
            store,
            CollaborationHostPolicy(
                standing_participation=StandingParticipationHostPolicy(enabled=False),
            ),
        )
        accepted = await WorkshopConversationCommandService(store, standing_participation=standing).accept_client(
            _message(human_id, channel_id, "standing-host-off")
        )
        assert len(accepted.command.runs) == 1
        assert (await standing.inspect(human_id, channel_id)).subscriptions == ()
    finally:
        await store.close()


async def test_production_host_policy_enables_standing_by_default() -> None:
    host = CollaborationHostPolicy()

    assert host.version == 3
    assert host.standing_participation.enabled is True
    assert "standing_participation" in host.effective_allowed_operations


async def test_channel_dismissal_ends_subscription_immediately_and_replays(tmp_path: Path) -> None:
    store, human_id, channel_id, agent_id, standing = await _eligible_authority(tmp_path / "dismiss.db")
    try:
        await _start_without_run(store, standing, human_id, channel_id, agent_id, "dismiss-start")
        first = await dismiss_channel_agent(
            store,
            principal_id=human_id,
            scope=EngagementScope(channel_id, None),
            agent_id=agent_id,
            client_dismissal_id="standing-dismiss",
            occurred_at=_NOW + timedelta(seconds=1),
        )
        replay = await dismiss_channel_agent(
            store,
            principal_id=human_id,
            scope=EngagementScope(channel_id, None),
            agent_id=agent_id,
            client_dismissal_id="standing-dismiss",
            occurred_at=_NOW + timedelta(seconds=1),
        )

        snapshot = await standing.inspect(human_id, channel_id)
        assert first.inserted is True
        assert replay.inserted is False
        assert snapshot.subscriptions[0].lifecycle_state == "ended"
        assert snapshot.subscriptions[0].end_reason == "dismissed"
    finally:
        await store.close()


async def test_definition_archive_ends_subscription_immediately(tmp_path: Path) -> None:
    store, human_id, channel_id, agent_id, standing = await _eligible_authority(tmp_path / "definition.db")
    try:
        await _start_without_run(store, standing, human_id, channel_id, agent_id, "definition-start")
        lifecycle = WorkshopAgentLifecycleService(store)
        definition = next(item for item in await lifecycle.list_visible(human_id) if item.agent_id == agent_id)
        await lifecycle.archive(
            human_id,
            definition.definition_id,
            idempotency_key="standing-definition-archive",
            expected_version=definition.state_version,
        )

        snapshot = await standing.inspect(human_id, channel_id)
        assert snapshot.subscriptions[0].lifecycle_state == "ended"
        assert snapshot.subscriptions[0].end_reason == "definition_archived"
    finally:
        await store.close()


async def test_channel_archive_ends_subscription_immediately(tmp_path: Path) -> None:
    store, human_id, channel_id, agent_id, standing = await _eligible_authority(tmp_path / "channel.db")
    try:
        await _start_without_run(store, standing, human_id, channel_id, agent_id, "channel-start")
        await WorkshopChannelLifecycleService(store).archive(
            human_id,
            channel_id,
            client_operation_id="standing-channel-archive",
        )

        async with store.connection.execute(
            "SELECT lifecycle_state, end_reason FROM channel_agent_standings WHERE channel_id = ?",
            (channel_id,),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None and tuple(row) == ("ended", "channel_archived")
    finally:
        await store.close()


async def test_owner_revocation_ends_subscription_immediately(tmp_path: Path) -> None:
    store, human_id, channel_id, agent_id, standing = await _eligible_authority(tmp_path / "owner.db")
    try:
        await _start_without_run(store, standing, human_id, channel_id, agent_id, "owner-start")
        lifecycle = WorkshopAgentLifecycleService(store)
        definition = next(item for item in await lifecycle.list_visible(human_id) if item.agent_id == agent_id)
        policy = WorkshopCollaborationPolicyService(
            store,
            cast(Any, _PrivateExecution(WorkshopCollaborationAuthority(store))),
        )
        await policy.set_allowed(
            human_id,
            definition.definition_id,
            allowed_operations=[],
            expected_policy_version=1,
            client_operation_id="standing-owner-revoke",
        )

        snapshot = await standing.inspect(human_id, channel_id)
        assert snapshot.subscriptions[0].lifecycle_state == "ended"
        assert snapshot.subscriptions[0].end_reason == "owner_policy_revoked"
    finally:
        await store.close()


async def test_revision_loss_and_quiet_expiry_fail_closed_lazily(tmp_path: Path) -> None:
    revision_store, human_id, channel_id, agent_id, standing = await _eligible_authority(tmp_path / "revision.db")
    try:
        await _start_without_run(revision_store, standing, human_id, channel_id, agent_id, "revision-start")
        lifecycle = WorkshopAgentLifecycleService(revision_store)
        definition = next(item for item in await lifecycle.list_visible(human_id) if item.agent_id == agent_id)
        revised = await lifecycle.add_revision(
            human_id,
            definition.definition_id,
            idempotency_key="standing-nonparticipating-revision",
            expected_version=definition.state_version,
            purpose="Respond only when explicitly mentioned.",
            instructions="Do not participate as a standing agent.",
            capabilities=["text_generation"],
            collaboration_operations=[],
        )
        await lifecycle.activate_revision(
            human_id,
            definition.definition_id,
            revision_id=revised.revisions[-1].revision_id,
            idempotency_key="standing-nonparticipating-activate",
            expected_version=revised.state_version,
        )
        snapshot = await standing.inspect(human_id, channel_id)
        assert snapshot.subscriptions[0].end_reason == "access_removed"
    finally:
        await revision_store.close()

    expiry_store, human_id, channel_id, agent_id, _standing = await _eligible_authority(tmp_path / "expiry.db")
    expiry_policy = CollaborationHostPolicy(
        standing_participation=StandingParticipationHostPolicy(enabled=True, quiet_expiry_seconds=1),
    )
    expiry = WorkshopStandingParticipationService(expiry_store, expiry_policy)
    try:
        started_at = datetime.now(UTC) - timedelta(seconds=2)
        await _start_without_run(
            expiry_store,
            expiry,
            human_id,
            channel_id,
            agent_id,
            "expiry-start",
            occurred_at=started_at,
        )
        snapshot = await expiry.inspect(human_id, channel_id)
        assert snapshot.subscriptions[0].lifecycle_state == "ended"
        assert snapshot.subscriptions[0].end_reason == "quiet_expired"
    finally:
        await expiry_store.close()
