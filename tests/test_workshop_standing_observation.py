"""Contracts for the canonical standing-agent observation inbox."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from kai.workshop.agent_lifecycle import WorkshopAgentLifecycleService
from kai.workshop.channel_lifecycle import WorkshopChannelLifecycleService
from kai.workshop.collaboration_authority import CollaborationHostPolicy, StandingParticipationHostPolicy
from kai.workshop.conversation_commands import WorkshopConversationCommandService
from kai.workshop.diagnostics import workshop_standing_observation_status
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
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.standing_observation import WorkshopStandingObservationService
from kai.workshop.standing_participation import WorkshopStandingParticipationService
from kai.workshop.store import AppendResult, WorkshopEventStore
from kai.workshop.wake_policy import EngagementScope, dismiss_channel_agent
from tests.test_workshop_standing_participation import _NOW, _eligible_authority, _message


def _host_policy(**overrides: object) -> CollaborationHostPolicy:
    return CollaborationHostPolicy(
        standing_participation=StandingParticipationHostPolicy(enabled=True, **overrides),
    )


async def _append_message(
    store: WorkshopEventStore,
    channel_id: ChannelId,
    author_id: PrincipalId,
    identity: str,
    *,
    source: str = "workshop_client",
    offset_seconds: float = 1,
    thread_root_id: MessageId | None = None,
) -> AppendResult:
    async with store.connection.execute("SELECT workshop_id FROM channels WHERE id = ?", (channel_id,)) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    message_id = MessageId.derived(channel_id, identity)
    payload: dict[str, object] = {
        "channel_id": channel_id,
        "author_principal_id": author_id,
        "body": identity,
        "mentions": [],
    }
    if thread_root_id is not None:
        payload["reply_to_message_id"] = thread_root_id
        payload["thread_root_id"] = thread_root_id
    event = EventEnvelope.create(
        event_id=EventId.derived(message_id, "created"),
        event_type=WorkshopEventType.MESSAGE_CREATED,
        event_version=2,
        workshop_id=WorkshopId(str(row[0])),
        aggregate_type="message",
        aggregate_id=message_id,
        actor_principal_id=author_id,
        occurred_at=_NOW + timedelta(seconds=offset_seconds),
        idempotency_key=f"standing-observation:{identity}",
        payload=payload,
        metadata={"source": source},
    )
    connection = store.connection
    try:
        await connection.execute("BEGIN IMMEDIATE")
        result = await store.append_in_transaction(event)
        await store.project_pending_in_transaction(CanonicalConversationProjection())
        await connection.commit()
        return result
    except Exception:
        await connection.rollback()
        raise


async def _other_agent_principal(store: WorkshopEventStore, owner_id: PrincipalId) -> PrincipalId:
    draft = await WorkshopAgentLifecycleService(store).create_draft(
        owner_id,
        idempotency_key="standing-observation-peer",
        handle="observation_peer",
        display_name="Observation peer",
        description="Produces canonical agent messages for observation tests.",
        presentation={"avatar": "O"},
        purpose="Provide another canonical agent author.",
        instructions="Publish only test messages.",
        capabilities=["text_generation"],
        collaboration_operations=[],
    )
    async with store.connection.execute(
        "SELECT principal_id FROM agents WHERE id = ?",
        (draft.agent_id,),
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return PrincipalId(str(row[0]))


async def _observation_authority(
    path: Path,
) -> tuple[
    WorkshopEventStore,
    PrincipalId,
    ChannelId,
    AgentId,
    WorkshopStandingParticipationService,
]:
    store, human_id, channel_id, agent_id, standing = await _eligible_authority(path)
    await WorkshopConversationCommandService(store, standing_participation=standing).accept_client(
        _message(human_id, channel_id, "standing-observation-start")
    )
    return store, human_id, channel_id, agent_id, standing


async def test_every_canonical_message_source_uses_one_coalesced_observation_trigger(tmp_path: Path) -> None:
    store, human_id, channel_id, agent_id, _standing = await _observation_authority(tmp_path / "kai.db")
    observation = WorkshopStandingObservationService(store, _host_policy())
    try:
        await observation.synchronize_host_policy()
        async with store.connection.execute("SELECT COUNT(*) FROM runs") as cursor:
            initial_runs = int((await cursor.fetchone())[0])
        sources = (
            "workshop_client",
            "telegram",
            "scheduled_job",
            "github",
            "workshop_collaboration_publication",
            "internal_api",
            "future_adapter",
        )
        results = [
            await _append_message(
                store,
                channel_id,
                human_id,
                f"source-{index}",
                source=source,
                offset_seconds=1 + index / 10,
            )
            for index, source in enumerate(sources)
        ]

        states = await observation.inspect(channel_id, agent_id, current_at=_NOW + timedelta(seconds=2))
        assert len(states) == 1
        state = states[0]
        assert state.scope_kind == "channel"
        assert state.pending_message_count == len(sources)
        assert state.oldest_pending_event_position == results[0].event.position
        assert state.pending_through_event_position == results[-1].event.position
        assert state.considered_through_event_position == results[-1].event.position
        assert state.latest_human_anchor_message_id == results[-1].event.envelope.aggregate_id
        assert state.not_before == results[0].event.envelope.occurred_at + timedelta(seconds=2)
        batches = await observation.pending_batches(state)
        assert len(batches) == 1
        assert batches[0].message_ids == tuple(result.event.envelope.aggregate_id for result in results)
        async with store.connection.execute("SELECT COUNT(*) FROM runs") as cursor:
            assert int((await cursor.fetchone())[0]) == initial_runs
    finally:
        await store.close()


async def test_own_output_advances_considered_cursor_without_pending_work(tmp_path: Path) -> None:
    store, human_id, channel_id, agent_id, _standing = await _observation_authority(tmp_path / "kai.db")
    observation = WorkshopStandingObservationService(store, _host_policy())
    try:
        await observation.synchronize_host_policy()
        async with store.connection.execute("SELECT principal_id FROM agents WHERE id = ?", (agent_id,)) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        own = await _append_message(
            store,
            channel_id,
            PrincipalId(str(row[0])),
            "own-agent-output",
            source="agent",
        )
        own_state = (await observation.inspect(channel_id, agent_id, current_at=_NOW + timedelta(seconds=2)))[0]
        assert own_state.lifecycle_state == "idle"
        assert own_state.pending_message_count == 0
        assert own_state.considered_through_event_position == own.event.position
        assert own_state.latest_human_anchor_message_id is not None
        assert own_state.latest_human_anchor_message_id != own.event.envelope.aggregate_id

        human = await _append_message(store, channel_id, human_id, "human-after-own-output", offset_seconds=2)
        pending = (await observation.inspect(channel_id, agent_id, current_at=_NOW + timedelta(seconds=3)))[0]
        assert pending.pending_message_count == 1
        assert pending.latest_human_anchor_message_id == human.event.envelope.aggregate_id
        assert (await observation.pending_batches(pending))[0].message_ids == (human.event.envelope.aggregate_id,)
    finally:
        await store.close()


async def test_channel_and_thread_scopes_keep_independent_cursors_and_anchors(tmp_path: Path) -> None:
    store, human_id, channel_id, agent_id, _standing = await _observation_authority(tmp_path / "kai.db")
    observation = WorkshopStandingObservationService(store, _host_policy())
    try:
        await observation.synchronize_host_policy()
        root = await _append_message(store, channel_id, human_id, "scope-root", offset_seconds=1)
        reply = await _append_message(
            store,
            channel_id,
            human_id,
            "scope-reply",
            offset_seconds=2,
            thread_root_id=MessageId(str(root.event.envelope.aggregate_id)),
        )

        states = await observation.inspect(channel_id, agent_id, current_at=_NOW + timedelta(seconds=3))
        by_kind = {state.scope_kind: state for state in states}
        assert set(by_kind) == {"channel", "thread"}
        assert by_kind["channel"].scope_id == str(channel_id)
        assert by_kind["channel"].pending_message_count == 1
        assert by_kind["channel"].latest_human_anchor_message_id == root.event.envelope.aggregate_id
        assert by_kind["thread"].scope_id == str(root.event.envelope.aggregate_id)
        assert by_kind["thread"].pending_message_count == 1
        assert by_kind["thread"].latest_human_anchor_message_id == reply.event.envelope.aggregate_id
    finally:
        await store.close()


async def test_thread_dismissal_prevents_later_thread_observation(tmp_path: Path) -> None:
    store, human_id, channel_id, agent_id, _standing = await _observation_authority(tmp_path / "kai.db")
    observation = WorkshopStandingObservationService(store, _host_policy())
    try:
        await observation.synchronize_host_policy()
        root = await _append_message(store, channel_id, human_id, "dismissed-thread-root", offset_seconds=1)
        root_id = MessageId(str(root.event.envelope.aggregate_id))
        await dismiss_channel_agent(
            store,
            principal_id=human_id,
            scope=EngagementScope(channel_id, root_id),
            agent_id=agent_id,
            client_dismissal_id="standing-observation-thread-dismissal",
            occurred_at=_NOW + timedelta(seconds=2),
        )
        await _append_message(
            store,
            channel_id,
            human_id,
            "dismissed-thread-reply",
            offset_seconds=3,
            thread_root_id=root_id,
        )

        states = await observation.inspect(channel_id, agent_id, current_at=_NOW + timedelta(seconds=4))
        assert [state.scope_kind for state in states] == ["channel"]
        assert states[0].pending_message_count == 1
        assert states[0].latest_human_anchor_message_id == root_id
    finally:
        await store.close()


async def test_pending_range_retains_multiple_future_batches_without_cursor_gap(tmp_path: Path) -> None:
    store, human_id, channel_id, agent_id, _standing = await _observation_authority(tmp_path / "kai.db")
    policy = _host_policy(max_messages_per_observe_run=2)
    observation = WorkshopStandingObservationService(store, policy)
    try:
        await observation.synchronize_host_policy()
        results = [
            await _append_message(store, channel_id, human_id, f"batch-{index}", offset_seconds=index + 1)
            for index in range(5)
        ]
        state = (await observation.inspect(channel_id, agent_id, current_at=_NOW + timedelta(seconds=6)))[0]
        batches = await observation.pending_batches(state)

        assert [len(batch.message_ids) for batch in batches] == [2, 2, 1]
        assert tuple(message_id for batch in batches for message_id in batch.message_ids) == tuple(
            result.event.envelope.aggregate_id for result in results
        )
        assert batches[0].from_event_position == results[0].event.position
        assert batches[-1].through_event_position == results[-1].event.position
        assert state.delivered_through_event_position < batches[0].from_event_position
    finally:
        await store.close()


async def test_overflow_pauses_explicitly_and_retains_inspectable_omitted_range(tmp_path: Path) -> None:
    path = tmp_path / "kai.db"
    store, human_id, channel_id, agent_id, _standing = await _observation_authority(path)
    policy = _host_policy(max_pending_messages_per_scope=2)
    observation = WorkshopStandingObservationService(store, policy)
    try:
        await observation.synchronize_host_policy()
        results = [
            await _append_message(store, channel_id, human_id, f"overflow-{index}", offset_seconds=index + 1)
            for index in range(4)
        ]
        state = (await observation.inspect(channel_id, agent_id, current_at=_NOW + timedelta(seconds=5)))[0]

        assert state.lifecycle_state == "paused_overflow"
        assert state.pending_message_count == 2
        assert state.pending_through_event_position == results[1].event.position
        assert state.considered_through_event_position == results[3].event.position
        assert state.overflow_reason == "pending_count"
        assert state.overflow_from_event_position == results[2].event.position
        assert state.overflow_through_event_position == results[3].event.position
        assert [len(batch.message_ids) for batch in await observation.pending_batches(state)] == [2]
        assert "paused overflow=1" in workshop_standing_observation_status(path)
        assert (
            "cursors=1 (delivery boundary initialized=1, backlog beyond boundary=1)"
            in workshop_standing_observation_status(path)
        )
        assert "integrity gaps=0, replay gaps=0" in workshop_standing_observation_status(path)
    finally:
        await store.close()


async def test_member_resume_discards_overflow_backlog_idempotently_and_rebuilds_exactly(
    tmp_path: Path,
) -> None:
    path = tmp_path / "kai.db"
    store, human_id, channel_id, agent_id, _standing = await _observation_authority(path)
    policy = _host_policy(max_pending_messages_per_scope=2)
    observation = WorkshopStandingObservationService(store, policy)
    standing = WorkshopStandingParticipationService(store, policy)
    try:
        await observation.synchronize_host_policy()
        for index in range(4):
            await _append_message(
                store,
                channel_id,
                human_id,
                f"resume-overflow-{index}",
                offset_seconds=index + 1,
            )
        paused = (
            await observation.inspect(
                channel_id,
                agent_id,
                current_at=_NOW + timedelta(seconds=5),
            )
        )[0]
        assert paused.lifecycle_state == "paused_overflow"

        resumed = await standing.resume_observation(
            human_id,
            channel_id,
            agent_id,
            scope_id=paused.scope_id,
            expected_state_version=paused.state_version,
            client_operation_id="resume-overflow-1",
        )
        replay = await standing.resume_observation(
            human_id,
            channel_id,
            agent_id,
            scope_id=paused.scope_id,
            expected_state_version=paused.state_version,
            client_operation_id="resume-overflow-1",
        )
        state = next(item for item in resumed.observations if item.scope_id == paused.scope_id)
        replayed = next(item for item in replay.observations if item.scope_id == paused.scope_id)

        assert state.lifecycle_state == "idle"
        assert state.pending_message_count == 0
        assert state.delivered_through_event_position == paused.considered_through_event_position
        assert state.overflow_reason is None
        assert replayed == state

        await store.rebuild_projection(CanonicalConversationProjection())
        rebuilt = next(
            item
            for item in (await standing.inspect(human_id, channel_id)).observations
            if item.scope_id == paused.scope_id
        )
        assert rebuilt == state
    finally:
        await store.close()


async def test_pending_age_reconciliation_pauses_without_advancing_delivery(tmp_path: Path) -> None:
    store, human_id, channel_id, agent_id, _standing = await _observation_authority(tmp_path / "kai.db")
    policy = _host_policy(max_pending_age_seconds=5)
    observation = WorkshopStandingObservationService(store, policy)
    try:
        await observation.synchronize_host_policy()
        source = await _append_message(store, channel_id, human_id, "aged-pending", offset_seconds=1)
        before = (await observation.inspect(channel_id, agent_id, current_at=_NOW + timedelta(seconds=2)))[0]
        after = (await observation.inspect(channel_id, agent_id, current_at=_NOW + timedelta(seconds=7)))[0]

        assert before.lifecycle_state == "pending"
        assert after.lifecycle_state == "paused_overflow"
        assert after.overflow_reason == "pending_age"
        assert after.overflow_from_event_position == source.event.position
        assert after.delivered_through_event_position == before.delivered_through_event_position
    finally:
        await store.close()


async def test_retry_rollback_rebuild_and_cross_channel_isolation_are_exact(tmp_path: Path) -> None:
    store, human_id, channel_id, agent_id, _standing = await _observation_authority(tmp_path / "kai.db")
    observation = WorkshopStandingObservationService(store, _host_policy())
    try:
        await observation.synchronize_host_policy()
        source = await _append_message(store, channel_id, human_id, "durable-source", offset_seconds=1)
        before = await observation.inspect(channel_id, agent_id, current_at=_NOW + timedelta(seconds=2))

        await store.connection.execute("BEGIN IMMEDIATE")
        replay = await store.append_in_transaction(source.event.envelope)
        await store.project_pending_in_transaction(CanonicalConversationProjection())
        await store.connection.commit()
        assert replay.inserted is False
        assert await observation.inspect(channel_id, agent_id, current_at=_NOW + timedelta(seconds=2)) == before

        rolled_back_id = MessageId.derived(channel_id, "rolled-back-source")
        await store.connection.execute("BEGIN IMMEDIATE")
        rolled_back = await store.append_in_transaction(
            EventEnvelope.create(
                event_id=EventId.derived(rolled_back_id, "created"),
                event_type=WorkshopEventType.MESSAGE_CREATED,
                event_version=2,
                workshop_id=source.event.envelope.workshop_id,
                aggregate_type="message",
                aggregate_id=rolled_back_id,
                actor_principal_id=human_id,
                occurred_at=_NOW + timedelta(seconds=2),
                idempotency_key="standing-observation:rolled-back-source",
                payload={
                    "channel_id": channel_id,
                    "author_principal_id": human_id,
                    "body": "rolled back",
                    "mentions": [],
                },
                metadata={"source": "workshop_client"},
            )
        )
        await store.project_pending_in_transaction(CanonicalConversationProjection())
        await store.connection.rollback()
        assert await store.event_by_idempotency_key("standing-observation:rolled-back-source") is None
        assert await observation.inspect(channel_id, agent_id, current_at=_NOW + timedelta(seconds=2)) == before
        assert rolled_back.inserted is True

        other = await WorkshopChannelLifecycleService(store).create_group(
            human_id,
            name="No standing",
            agent_ids=[agent_id],
        )
        await _append_message(store, other.channel_id, human_id, "other-channel", offset_seconds=3)
        async with store.connection.execute(
            "SELECT COUNT(*) FROM channel_agent_observation_states WHERE channel_id = ?",
            (other.channel_id,),
        ) as cursor:
            assert int((await cursor.fetchone())[0]) == 0

        checkpoint = await store.rebuild_projection(CanonicalConversationProjection())
        assert checkpoint.version == 33
        rebuilt = await observation.inspect(channel_id, agent_id, current_at=_NOW + timedelta(seconds=2))
        assert rebuilt == before
    finally:
        await store.close()


async def test_other_agent_output_is_observed_but_never_creates_a_human_anchor(tmp_path: Path) -> None:
    store, human_id, channel_id, agent_id, _standing = await _observation_authority(tmp_path / "kai.db")
    observation = WorkshopStandingObservationService(store, _host_policy())
    try:
        await observation.synchronize_host_policy()
        peer_principal = await _other_agent_principal(store, human_id)
        root = await _append_message(
            store,
            channel_id,
            human_id,
            "peer-thread-root",
            offset_seconds=1,
        )
        output = await _append_message(
            store,
            channel_id,
            peer_principal,
            "peer-agent-output",
            source="agent",
            offset_seconds=2,
            thread_root_id=MessageId(str(root.event.envelope.aggregate_id)),
        )
        states = await observation.inspect(channel_id, agent_id, current_at=_NOW + timedelta(seconds=3))
        state = next(item for item in states if item.scope_kind == "thread")

        assert state.pending_message_count == 1
        assert state.pending_through_event_position == output.event.position
        assert state.latest_human_anchor_message_id is None
        assert state.latest_human_anchor_event_position is None
    finally:
        await store.close()
