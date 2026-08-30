"""Contracts for group mention routing and derived agent engagement."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.conversation_commands import (
    ConversationCommandAcceptance,
    ConversationCommandDisposition,
    WorkshopConversationCommandService,
)
from kai.workshop.domain import (
    AgentDefinitionId,
    AgentDefinitionRevisionId,
    AgentEnablementId,
    AgentId,
    ChannelAgentId,
    ChannelId,
    ChannelMembershipId,
    EventEnvelope,
    MessageId,
    PrincipalId,
    RuntimeAssignmentId,
    WorkshopEventType,
    WorkshopId,
    WorkshopMembershipId,
)
from kai.workshop.inbound import ClientInboundMessage
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import WorkshopEventStore
from kai.workshop.wake_policy import (
    EngagementScope,
    dismiss_channel_agent,
    resolve_agent_engagements,
    resolve_message_wake_targets,
)
from tests.workshop_profiles import profile_id

_NOW = datetime(2026, 8, 28, 12, 0, tzinfo=UTC)


async def _open_group_store(
    path: Path,
) -> tuple[WorkshopEventStore, PrincipalId, ChannelId, tuple[AgentId, AgentId]]:
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
        "SELECT w.id, human.id, a.id, a.principal_id, ra.runtime_profile_id "
        "FROM workshops w JOIN workshop_memberships hwm ON hwm.workshop_id = w.id "
        "JOIN principals human ON human.id = hwm.principal_id AND human.kind = 'human' "
        "JOIN agents a ON a.workshop_id = w.id "
        "JOIN channel_agents ca ON ca.agent_id = a.id "
        "JOIN channels c ON c.id = ca.channel_id AND c.kind = 'direct' "
        "JOIN channel_agent_runtime_assignments ra ON ra.channel_id = c.id AND ra.agent_id = a.id"
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    workshop_id = WorkshopId(str(row[0]))
    human_id = PrincipalId(str(row[1]))
    first_agent_id = AgentId(str(row[2]))
    first_agent_principal = PrincipalId(str(row[3]))
    first_profile = str(row[4])
    second_agent_id = AgentId.new()
    second_agent_principal = PrincipalId.new()
    group_id = ChannelId.new()
    created_at = _NOW.isoformat()
    await store.connection.execute(
        "INSERT INTO principals (id, kind, display_name, created_at) VALUES (?, 'agent', 'Nova', ?)",
        (second_agent_principal, created_at),
    )
    await store.connection.execute(
        "INSERT INTO workshop_memberships (id, workshop_id, principal_id, role, created_at) "
        "VALUES (?, ?, ?, 'agent', ?)",
        (
            WorkshopMembershipId.derived(workshop_id, f"principal:{second_agent_principal}"),
            workshop_id,
            second_agent_principal,
            created_at,
        ),
    )
    await store.connection.execute(
        "INSERT INTO agents (id, workshop_id, principal_id, name, created_at) VALUES (?, ?, ?, 'Nova', ?)",
        (second_agent_id, workshop_id, second_agent_principal, created_at),
    )
    await store.connection.execute(
        "INSERT INTO channels (id, workshop_id, kind, name, created_at) VALUES (?, ?, 'group', 'Wake policy', ?)",
        (group_id, workshop_id, created_at),
    )
    for principal_id, role in (
        (human_id, "owner"),
        (first_agent_principal, "participant"),
        (second_agent_principal, "participant"),
    ):
        await store.connection.execute(
            "INSERT INTO channel_memberships (id, channel_id, principal_id, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                ChannelMembershipId.derived(group_id, f"principal:{principal_id}"),
                group_id,
                principal_id,
                role,
                created_at,
            ),
        )
    for agent_id, runtime_profile in (
        (first_agent_id, first_profile),
        (second_agent_id, profile_id(202)),
    ):
        await store.connection.execute(
            "INSERT INTO channel_agents (id, channel_id, agent_id, created_at) VALUES (?, ?, ?, ?)",
            (ChannelAgentId.derived(group_id, f"agent:{agent_id}"), group_id, agent_id, created_at),
        )
        await store.connection.execute(
            "INSERT INTO channel_agent_runtime_assignments "
            "(id, channel_id, agent_id, runtime_profile_id, created_at, created_event_position) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                RuntimeAssignmentId.derived(group_id, f"runtime:{agent_id}"),
                group_id,
                agent_id,
                runtime_profile,
                created_at,
                1 if agent_id == first_agent_id else 2,
            ),
        )
    await store.connection.commit()
    nova_definition_id = AgentDefinitionId.derived(second_agent_id, "definition")
    nova_revision_id = AgentDefinitionRevisionId.derived(nova_definition_id, "revision:1")
    for event in (
        EventEnvelope.create(
            event_type=WorkshopEventType.AGENT_DEFINITION_CREATED,
            event_version=1,
            workshop_id=workshop_id,
            aggregate_type="agent_definition",
            aggregate_id=nova_definition_id,
            occurred_at=_NOW,
            payload={
                "agent_id": second_agent_id,
                "handle": "nova",
                "display_name": "Nova",
                "description": "Test agent",
                "presentation": {"avatar": "N"},
                "lifecycle_state": "active",
            },
        ),
        EventEnvelope.create(
            event_type=WorkshopEventType.AGENT_DEFINITION_REVISION_ADDED,
            event_version=1,
            workshop_id=workshop_id,
            aggregate_type="agent_definition_revision",
            aggregate_id=nova_revision_id,
            occurred_at=_NOW,
            payload={
                "definition_id": nova_definition_id,
                "revision_number": 1,
                "purpose": "Exercise multi-agent wake routing.",
                "instructions": "Respond as Nova when explicitly mentioned.",
                "capabilities": ["text_generation"],
            },
        ),
        EventEnvelope.create(
            event_type=WorkshopEventType.AGENT_DEFINITION_REVISION_ACTIVATED,
            event_version=1,
            workshop_id=workshop_id,
            aggregate_type="agent_definition",
            aggregate_id=nova_definition_id,
            occurred_at=_NOW,
            payload={"revision_id": nova_revision_id},
        ),
    ):
        await store.append(event)
        await store.project_pending(CanonicalConversationProjection())
    async with store.connection.execute("SELECT MAX(position) FROM event_log") as cursor:
        latest_event = await cursor.fetchone()
    assert latest_event is not None
    definition_event_position = int(latest_event[0])
    nova_direct_channel = ChannelId.derived(second_agent_id, f"principal:{human_id}")
    nova_direct_attachment = ChannelAgentId.derived(
        nova_direct_channel,
        f"agent:{second_agent_id}",
    )
    nova_direct_assignment = RuntimeAssignmentId.derived(
        nova_direct_channel,
        f"runtime:{second_agent_id}",
    )
    nova_enablement = AgentEnablementId.derived(
        nova_definition_id,
        f"principal:{human_id}",
    )
    nova_profile = profile_id(202)
    await store.connection.execute(
        "INSERT INTO channels (id, workshop_id, kind, name, created_at) VALUES (?, ?, 'direct', 'Nova', ?)",
        (nova_direct_channel, workshop_id, created_at),
    )
    for principal_id, role in (
        (human_id, "owner"),
        (second_agent_principal, "participant"),
    ):
        await store.connection.execute(
            "INSERT INTO channel_memberships (id, channel_id, principal_id, role, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                ChannelMembershipId.derived(
                    nova_direct_channel,
                    f"principal:{principal_id}",
                ),
                nova_direct_channel,
                principal_id,
                role,
                created_at,
            ),
        )
    await store.connection.execute(
        "INSERT INTO channel_agents "
        "(id, channel_id, agent_id, created_at, sponsor_principal_id, "
        "sponsored_runtime_profile_id, attached_event_position) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            nova_direct_attachment,
            nova_direct_channel,
            second_agent_id,
            created_at,
            human_id,
            nova_profile,
            definition_event_position,
        ),
    )
    await store.connection.execute(
        "INSERT INTO channel_agent_runtime_assignments "
        "(id, channel_id, agent_id, runtime_profile_id, created_at, created_event_position) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            nova_direct_assignment,
            nova_direct_channel,
            second_agent_id,
            nova_profile,
            created_at,
            definition_event_position,
        ),
    )
    await store.connection.execute(
        "INSERT INTO principal_agent_enablements "
        "(id, workshop_id, principal_id, agent_definition_id, agent_id, "
        "direct_channel_id, runtime_profile_id, lifecycle_state, created_at, "
        "updated_at, created_event_position, last_event_position) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, 'enabled', ?, ?, ?, ?)",
        (
            nova_enablement,
            workshop_id,
            human_id,
            nova_definition_id,
            second_agent_id,
            nova_direct_channel,
            nova_profile,
            created_at,
            created_at,
            definition_event_position,
            definition_event_position,
        ),
    )
    await store.connection.execute(
        "UPDATE channel_agents SET sponsor_principal_id = ?, "
        "sponsored_runtime_profile_id = ? WHERE channel_id = ? AND agent_id = ?",
        (human_id, first_profile, group_id, first_agent_id),
    )
    await store.connection.execute(
        "UPDATE channel_agents SET sponsor_principal_id = ?, "
        "sponsored_runtime_profile_id = ? WHERE channel_id = ? AND agent_id = ?",
        (human_id, nova_profile, group_id, second_agent_id),
    )
    await store.connection.commit()
    return store, human_id, group_id, (first_agent_id, second_agent_id)


def _message(
    principal_id: PrincipalId,
    channel_id: ChannelId,
    identity: str,
    body: str,
    occurred_at: datetime,
    *,
    thread_root_id: MessageId | None = None,
) -> ClientInboundMessage:
    return ClientInboundMessage(
        principal_id=principal_id,
        channel_id=channel_id,
        client_message_id=identity,
        body=body,
        occurred_at=occurred_at,
        thread_root_id=thread_root_id,
    )


def _accepted_message_id(acceptance: ConversationCommandAcceptance) -> MessageId:
    message_id = acceptance.command.message.event.envelope.aggregate_id
    assert isinstance(message_id, MessageId)
    return message_id


async def test_group_message_only_and_exact_replay_create_no_run(tmp_path: Path):
    store, human_id, group_id, _ = await _open_group_store(tmp_path / "kai.db")
    try:
        service = WorkshopConversationCommandService(store)
        command = _message(human_id, group_id, "plain-1", "Hello everyone", _NOW)

        accepted = await service.accept_client(command)
        replay = await service.accept_client(command)

        assert accepted.command.disposition == ConversationCommandDisposition.MESSAGE_ONLY
        assert replay.command.disposition == ConversationCommandDisposition.MESSAGE_ONLY
        assert accepted.command.lifecycles == replay.command.lifecycles == ()
        async with store.connection.execute("SELECT COUNT(*) FROM messages") as cursor:
            assert int((await cursor.fetchone())[0]) == 1
        async with store.connection.execute("SELECT COUNT(*) FROM runs") as cursor:
            assert int((await cursor.fetchone())[0]) == 0
    finally:
        await store.close()


async def test_mentions_route_exclusively_and_support_multiple_agents(tmp_path: Path):
    store, human_id, group_id, agent_ids = await _open_group_store(tmp_path / "kai.db")
    try:
        service = WorkshopConversationCommandService(store)
        first = await service.accept_client(_message(human_id, group_id, "mention-1", "@Kai respond", _NOW))
        second_only = await service.accept_client(
            _message(human_id, group_id, "mention-2", "@Nova respond", _NOW + timedelta(seconds=1))
        )
        both_message = _message(
            human_id,
            group_id,
            "mention-3",
            "@kAi and @Nova respond",
            _NOW + timedelta(seconds=2),
        )
        both = await service.accept_client(both_message)

        assert tuple(run.agent_id for run in first.command.runs) == (agent_ids[0],)
        assert tuple(run.agent_id for run in second_only.command.runs) == (agent_ids[1],)
        assert {run.agent_id for run in both.command.runs} == set(agent_ids)
        assert len(set(run.run_id for run in both.command.runs)) == 2
        assert len(both.runtime_profile_ids) == 2

        await store.connection.execute(
            "UPDATE runs SET status = 'completed', started_at = ?, terminal_at = ? WHERE id = ?",
            (
                (_NOW + timedelta(seconds=3)).isoformat(),
                (_NOW + timedelta(seconds=4)).isoformat(),
                both.command.runs[0].run_id,
            ),
        )
        await store.connection.commit()
        replay = await service.accept_client(both_message)
        assert set(replay.command.run_dispositions) == {
            ConversationCommandDisposition.TERMINAL_REPLAY,
            ConversationCommandDisposition.READY_REPLAY,
        }
    finally:
        await store.close()


async def test_unmentioned_message_wakes_only_engaged_and_dismissal_clears_it(tmp_path: Path):
    store, human_id, group_id, agent_ids = await _open_group_store(tmp_path / "kai.db")
    try:
        service = WorkshopConversationCommandService(store)
        await service.accept_client(_message(human_id, group_id, "engage", "@Kai stay", _NOW))
        engagement = await resolve_agent_engagements(
            store,
            EngagementScope(group_id),
            current_at=_NOW + timedelta(seconds=1),
        )
        assert len(engagement) == 1
        assert engagement[0].agent_id == agent_ids[0]
        assert engagement[0].engaged_at == _NOW
        assert engagement[0].expires_at == _NOW + timedelta(seconds=900)
        engaged = await service.accept_client(
            _message(human_id, group_id, "plain-engaged", "@Daniel continue", _NOW + timedelta(seconds=1))
        )
        assert tuple(run.agent_id for run in engaged.command.runs) == (agent_ids[0],)

        await dismiss_channel_agent(
            store,
            principal_id=human_id,
            scope=EngagementScope(group_id),
            agent_id=agent_ids[0],
            client_dismissal_id="dismiss-1",
            occurred_at=_NOW + timedelta(seconds=2),
        )
        assert (
            await resolve_agent_engagements(
                store,
                EngagementScope(group_id),
                current_at=_NOW + timedelta(seconds=3),
            )
            == ()
        )
        dismissed = await service.accept_client(
            _message(human_id, group_id, "plain-dismissed", "Anyone?", _NOW + timedelta(seconds=3))
        )
        assert dismissed.command.disposition == ConversationCommandDisposition.MESSAGE_ONLY
        assert dismissed.command.runs == ()
    finally:
        await store.close()


async def test_engagement_expires_and_agent_authored_message_never_wakes(tmp_path: Path):
    store, human_id, group_id, agent_ids = await _open_group_store(tmp_path / "kai.db")
    try:
        service = WorkshopConversationCommandService(store)
        await service.accept_client(_message(human_id, group_id, "engage", "@Kai stay", _NOW))
        expired = await service.accept_client(
            _message(human_id, group_id, "expired", "Still there?", _NOW + timedelta(seconds=901))
        )
        assert expired.command.disposition == ConversationCommandDisposition.MESSAGE_ONLY

        async with store.connection.execute(
            "SELECT a.principal_id, c.workshop_id FROM agents a JOIN channels c "
            "ON c.workshop_id = a.workshop_id WHERE a.id = ? AND c.id = ?",
            (agent_ids[0], group_id),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        message_id = MessageId.new()
        event = EventEnvelope.create(
            event_type=WorkshopEventType.MESSAGE_CREATED,
            event_version=1,
            workshop_id=WorkshopId(str(row[1])),
            aggregate_type="message",
            aggregate_id=message_id,
            actor_principal_id=PrincipalId(str(row[0])),
            occurred_at=_NOW + timedelta(seconds=902),
            payload={
                "channel_id": group_id,
                "author_principal_id": str(row[0]),
                "body": "Agent-authored update",
            },
            metadata={"source": "test"},
        )
        await store.append(event)
        await store.project_pending(CanonicalConversationProjection())
        assert (await resolve_message_wake_targets(store, message_id)).agent_ids == ()
    finally:
        await store.close()


async def test_thread_engagement_wakes_only_followups_in_the_same_thread(tmp_path: Path):
    store, human_id, group_id, agent_ids = await _open_group_store(tmp_path / "kai.db")
    try:
        service = WorkshopConversationCommandService(store)
        root_a = _accepted_message_id(
            await service.accept_client(_message(human_id, group_id, "root-a", "Thread A", _NOW - timedelta(seconds=2)))
        )
        root_b = _accepted_message_id(
            await service.accept_client(_message(human_id, group_id, "root-b", "Thread B", _NOW - timedelta(seconds=1)))
        )

        mentioned = await service.accept_client(
            _message(
                human_id,
                group_id,
                "thread-a-mention",
                "@Kai join thread A",
                _NOW,
                thread_root_id=root_a,
            )
        )
        same_thread = await service.accept_client(
            _message(
                human_id,
                group_id,
                "thread-a-followup",
                "Continue here",
                _NOW + timedelta(seconds=1),
                thread_root_id=root_a,
            )
        )
        other_thread = await service.accept_client(
            _message(
                human_id,
                group_id,
                "thread-b-followup",
                "Do not wake there",
                _NOW + timedelta(seconds=2),
                thread_root_id=root_b,
            )
        )
        top_level = await service.accept_client(
            _message(human_id, group_id, "top-followup", "Do not wake here", _NOW + timedelta(seconds=3))
        )

        assert tuple(run.agent_id for run in mentioned.command.runs) == (agent_ids[0],)
        assert tuple(run.agent_id for run in same_thread.command.runs) == (agent_ids[0],)
        assert other_thread.command.disposition == ConversationCommandDisposition.MESSAGE_ONLY
        assert top_level.command.disposition == ConversationCommandDisposition.MESSAGE_ONLY
        assert (
            await resolve_agent_engagements(
                store,
                EngagementScope(group_id, root_a),
                current_at=_NOW + timedelta(seconds=3),
            )
        )[0].agent_id == agent_ids[0]
        assert (
            await resolve_agent_engagements(
                store,
                EngagementScope(group_id, root_b),
                current_at=_NOW + timedelta(seconds=3),
            )
            == ()
        )
        assert (
            await resolve_agent_engagements(
                store,
                EngagementScope(group_id),
                current_at=_NOW + timedelta(seconds=3),
            )
            == ()
        )
    finally:
        await store.close()


async def test_thread_dismissal_and_quiet_window_are_scoped_per_thread(tmp_path: Path):
    store, human_id, group_id, agent_ids = await _open_group_store(tmp_path / "kai.db")
    try:
        service = WorkshopConversationCommandService(store)
        root_a = _accepted_message_id(
            await service.accept_client(_message(human_id, group_id, "root-a", "Thread A", _NOW - timedelta(seconds=2)))
        )
        root_b = _accepted_message_id(
            await service.accept_client(_message(human_id, group_id, "root-b", "Thread B", _NOW - timedelta(seconds=1)))
        )
        await service.accept_client(
            _message(human_id, group_id, "mention-a", "@Kai thread A", _NOW, thread_root_id=root_a)
        )
        await service.accept_client(
            _message(
                human_id,
                group_id,
                "mention-b",
                "@Kai thread B",
                _NOW + timedelta(seconds=1),
                thread_root_id=root_b,
            )
        )

        await dismiss_channel_agent(
            store,
            principal_id=human_id,
            scope=EngagementScope(group_id, root_a),
            agent_id=agent_ids[0],
            client_dismissal_id="dismiss-thread-a",
            occurred_at=_NOW + timedelta(seconds=2),
        )

        assert (
            await resolve_agent_engagements(
                store,
                EngagementScope(group_id, root_a),
                current_at=_NOW + timedelta(seconds=3),
            )
            == ()
        )
        thread_b = await resolve_agent_engagements(
            store,
            EngagementScope(group_id, root_b),
            current_at=_NOW + timedelta(seconds=3),
        )
        assert tuple(item.agent_id for item in thread_b) == (agent_ids[0],)
        assert (
            await resolve_agent_engagements(
                store,
                EngagementScope(group_id, root_b),
                current_at=_NOW + timedelta(seconds=902),
            )
            == ()
        )
    finally:
        await store.close()


async def test_dismissal_rejects_naive_timestamps(tmp_path: Path):
    store, human_id, group_id, agent_ids = await _open_group_store(tmp_path / "kai.db")
    try:
        with pytest.raises(ValueError, match="timezone-aware"):
            await dismiss_channel_agent(
                store,
                principal_id=human_id,
                scope=EngagementScope(group_id),
                agent_id=agent_ids[0],
                client_dismissal_id="naive-time",
                occurred_at=datetime(2026, 8, 28, 12, 0),
            )
    finally:
        await store.close()
