"""Final adversarial qualification for Workshop standing participation."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from kai.backend import AgentResponse, StreamEvent
from kai.config import Config
from kai.pool import SubprocessPool
from kai.workshop.agent_enablement import WorkshopAgentEnablementService
from kai.workshop.agent_lifecycle import WorkshopAgentLifecycleService
from kai.workshop.channel_lifecycle import WorkshopChannelLifecycleService
from kai.workshop.collaboration_authority import (
    CollaborationOperation,
    WorkshopCollaborationAuthority,
)
from kai.workshop.collaboration_policy import WorkshopCollaborationPolicyService
from kai.workshop.conversation_commands import WorkshopConversationCommandService
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
from kai.workshop.execution_coordinator import (
    CanonicalCancellationDisposition,
    CanonicalExecutionDisposition,
    WorkshopCanonicalExecutionCoordinator,
)
from kai.workshop.execution_state import WorkshopExecutionStateRegistry
from kai.workshop.inbound import ClientInboundMessage
from kai.workshop.internal_api_contexts import (
    WorkshopInternalAPIContextRegistry,
    WorkshopInternalAPIExecutionContext,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.run_execution_authority import (
    RunExecutionOwnerId,
    RunExecutionSelection,
    WorkshopRunExecutionAuthority,
)
from kai.workshop.run_lifecycle import RunStatus, WorkshopRunLifecycle
from kai.workshop.runtime_profiles import (
    ProtectedRuntimeBackend,
    ProtectedRuntimeProfile,
    WorkshopRuntimeProfileRegistry,
)
from kai.workshop.standing_observation import WorkshopStandingObservationService
from kai.workshop.standing_participation import WorkshopStandingParticipationService
from kai.workshop.store import WorkshopEventStore
from tests.test_pool import _internal_api_contexts
from tests.test_workshop_execution_coordinator import _Preparation, _Prepared
from tests.test_workshop_standing_observation import (
    _NOW,
    _append_message,
    _host_policy,
    _observation_authority,
)
from tests.test_workshop_standing_participation import _eligible_authority
from tests.workshop_delivery import TELEGRAM_DELIVERY_POLICY
from tests.workshop_profiles import profile_id, profile_registry


@dataclass
class _RuntimePool:
    registered: list[WorkshopInternalAPIExecutionContext] = field(default_factory=list)

    def register_canonical_lane(self, context: WorkshopInternalAPIExecutionContext) -> None:
        self.registered.append(context)

    async def suspend_canonical_lane(self, _context: WorkshopInternalAPIExecutionContext) -> None:
        return None

    async def rebind_canonical_lane(
        self,
        _prior: WorkshopInternalAPIExecutionContext,
        replacement: WorkshopInternalAPIExecutionContext,
    ) -> None:
        self.registered.append(replacement)


class _PolicyExecution:
    def __init__(self, store: WorkshopEventStore) -> None:
        self.collaboration_authority = WorkshopCollaborationAuthority(store)

    async def revoke_collaboration_for_definition(self, *_args: object, **_kwargs: object) -> int:
        return 0


async def _add_second_standing_agent(
    store: WorkshopEventStore,
    human_id: PrincipalId,
    channel_id: ChannelId,
) -> AgentId:
    lifecycle = WorkshopAgentLifecycleService(store)
    draft = await lifecycle.create_draft(
        human_id,
        idempotency_key="standing-peer-create",
        handle="standing_peer",
        display_name="Standing peer",
        description="A second standing participant used for loop qualification.",
        presentation={"avatar": "S"},
        purpose="Participate selectively in the qualification channel.",
        instructions="Speak only when a concise contribution is useful.",
        capabilities=["text_generation"],
        collaboration_operations=["standing_participation"],
    )
    active = await lifecycle.activate_revision(
        human_id,
        draft.definition_id,
        revision_id=draft.revisions[0].revision_id,
        idempotency_key="standing-peer-activate",
        expected_version=draft.state_version,
    )
    profiles = profile_registry(101)
    execution = await WorkshopExecutionStateRegistry.from_store(store, profiles)
    contexts = await WorkshopInternalAPIContextRegistry.from_store(store, profiles)
    enabled = await WorkshopAgentEnablementService(
        store,
        profiles,
        execution,
        contexts,
        _RuntimePool(),  # type: ignore[arg-type]
    ).enable(
        human_id,
        active.definition_id,
        profile_id(101),
        idempotency_key="standing-peer-enable",
    )
    await WorkshopChannelLifecycleService(store).attach_agent(
        human_id,
        channel_id,
        active.agent_id,
        client_operation_id="standing-peer-attach",
    )
    policy = WorkshopCollaborationPolicyService(store, _PolicyExecution(store))  # type: ignore[arg-type]
    changed = await policy.set_allowed(
        human_id,
        active.definition_id,
        allowed_operations=["standing_participation"],
        expected_policy_version=0,
        client_operation_id="standing-peer-owner-policy",
    )
    standing_operation = next(
        item for item in changed.snapshot.operations if item.operation == CollaborationOperation.STANDING_PARTICIPATION
    )
    assert standing_operation.owner_allowed is True
    assert standing_operation.effective_for_new_attempt is True
    assert enabled.runtime_profile_id == profile_id(101)
    return active.agent_id


async def _start_together(
    store: WorkshopEventStore,
    standing: WorkshopStandingParticipationService,
    human_id: PrincipalId,
    channel_id: ChannelId,
    agent_ids: tuple[AgentId, AgentId],
) -> None:
    placeholders = ",".join("?" for _ in agent_ids)
    async with store.connection.execute(
        "SELECT a.id, a.principal_id, d.handle FROM agents a "
        "JOIN agent_definitions d ON d.agent_id = a.id "
        f"WHERE a.id IN ({placeholders}) ORDER BY a.id",
        agent_ids,
    ) as cursor:
        rows = list(await cursor.fetchall())
    assert len(rows) == 2
    body = " ".join(f"@{row[2]!s}" for row in rows) + " begin standing qualification"
    mentions: list[dict[str, object]] = []
    offset = 0
    ordered_agent_ids: list[AgentId] = []
    for row in rows:
        token = f"@{row[2]!s}"
        mentions.append(
            {
                "principal_id": PrincipalId(str(row[1])),
                "kind": "agent",
                "start": offset,
                "length": len(token),
            }
        )
        ordered_agent_ids.append(AgentId(str(row[0])))
        offset += len(token) + 1
    async with store.connection.execute("SELECT workshop_id FROM channels WHERE id = ?", (channel_id,)) as cursor:
        workshop_row = await cursor.fetchone()
    assert workshop_row is not None
    message_id = MessageId.derived(channel_id, "standing-pair-start")
    event = EventEnvelope.create(
        event_id=EventId.derived(message_id, "created"),
        event_type=WorkshopEventType.MESSAGE_CREATED,
        event_version=2,
        workshop_id=WorkshopId(str(workshop_row[0])),
        aggregate_type="message",
        aggregate_id=message_id,
        actor_principal_id=human_id,
        occurred_at=_NOW,
        idempotency_key="standing-qualification:pair-start",
        payload={
            "channel_id": channel_id,
            "author_principal_id": human_id,
            "body": body,
            "mentions": mentions,
        },
        metadata={"source": "qualification"},
    )
    await store.connection.execute("BEGIN IMMEDIATE")
    try:
        await store.append_in_transaction(event)
        await store.project_pending_in_transaction(CanonicalConversationProjection())
        await standing.start_from_message_in_transaction(
            message_id,
            tuple(ordered_agent_ids),
            occurred_at=_NOW,
        )
        await store.connection.commit()
    except Exception:
        await store.connection.rollback()
        raise


def _coordinator(
    store: WorkshopEventStore,
    prepared: _Prepared,
    *,
    offset_seconds: float,
) -> WorkshopCanonicalExecutionCoordinator:
    assert prepared.run.runtime_profile_id is not None
    prepared.runtime_profile_id = prepared.run.runtime_profile_id
    return WorkshopCanonicalExecutionCoordinator(
        store,
        _Preparation(prepared),
        registered_backend_ids=frozenset({"codex"}),
        clock=lambda: _NOW + timedelta(seconds=offset_seconds),
        delivery_policy=TELEGRAM_DELIVERY_POLICY,
        collaboration_host_policy=_host_policy(minimum_unsolicited_interval_seconds=1),
    )


async def test_two_standing_agents_reach_quiescence_after_one_contribution_each(
    tmp_path: Path,
) -> None:
    store, human_id, channel_id, first_agent_id, standing = await _eligible_authority(tmp_path / "kai.db")
    try:
        second_agent_id = await _add_second_standing_agent(store, human_id, channel_id)
        await _start_together(
            store,
            standing,
            human_id,
            channel_id,
            (first_agent_id, second_agent_id),
        )
        observation = WorkshopStandingObservationService(
            store,
            _host_policy(minimum_unsolicited_interval_seconds=1),
        )
        await observation.synchronize_host_policy()
        human_message = await _append_message(
            store,
            channel_id,
            human_id,
            "bounded-two-agent-anchor",
            offset_seconds=1,
        )

        first = await observation.accept_next_ready(occurred_at=_NOW + timedelta(seconds=4))
        assert first is not None
        first_result = await _coordinator(
            store,
            _Prepared(first.run, response=AgentResponse(success=True, text="First useful contribution")),
            offset_seconds=10,
        ).execute(first.run.run_id)
        assert first_result.run.standing_outcome == "spoke"

        second = await observation.accept_next_ready(occurred_at=_NOW + timedelta(seconds=13))
        assert second is not None and second.run.agent_id != first.run.agent_id
        second_result = await _coordinator(
            store,
            _Prepared(second.run, response=AgentResponse(success=True, text="Second useful contribution")),
            offset_seconds=20,
        ).execute(second.run.run_id)
        assert second_result.run.standing_outcome == "spoke"

        final = await observation.accept_next_ready(occurred_at=_NOW + timedelta(seconds=23))
        assert final is not None and final.run.agent_id == first.run.agent_id
        final_result = await _coordinator(
            store,
            _Prepared(final.run, response=AgentResponse(success=True, text="Loop attempt")),
            offset_seconds=30,
        ).execute(final.run.run_id)
        assert final_result.run.standing_outcome == "publication_suppressed"
        assert await observation.accept_next_ready(occurred_at=_NOW + timedelta(seconds=40)) is None

        async with store.connection.execute(
            "SELECT body FROM messages WHERE body IN "
            "('First useful contribution', 'Second useful contribution', 'Loop attempt') "
            "ORDER BY created_event_position"
        ) as cursor:
            assert [str(row[0]) for row in await cursor.fetchall()] == [
                "First useful contribution",
                "Second useful contribution",
            ]
        assert first.run.human_anchor_message_id == human_message.event.envelope.aggregate_id
        assert second.run.human_anchor_message_id == human_message.event.envelope.aggregate_id
        assert final.run.human_anchor_message_id == human_message.event.envelope.aggregate_id
    finally:
        await store.close()


async def test_observe_cancellation_before_and_during_dispatch_preserves_retryable_batch(
    tmp_path: Path,
) -> None:
    store, human_id, channel_id, agent_id, _standing = await _observation_authority(tmp_path / "kai.db")
    observation = WorkshopStandingObservationService(store, _host_policy())
    try:
        await observation.synchronize_host_policy()
        source = await _append_message(store, channel_id, human_id, "cancel-observe", offset_seconds=1)
        accepted = await observation.accept_next_ready(occurred_at=_NOW + timedelta(seconds=4))
        assert accepted is not None
        prepared = _Prepared(accepted.run, response=AgentResponse(success=True, text="Must not publish"))
        coordinator = _coordinator(store, prepared, offset_seconds=10)

        assert await coordinator.request_cancellation(accepted.run.run_id) == CanonicalCancellationDisposition.REQUESTED
        assert (await WorkshopRunLifecycle(store).state(accepted.run.run_id)).status == RunStatus.CANCELLED
        assert prepared.prompts == []

        retry = await observation.accept_next_ready(occurred_at=_NOW + timedelta(seconds=11))
        assert retry is not None
        release = asyncio.Event()
        running = _Prepared(
            retry.run,
            response=AgentResponse(success=True, text="Must also not publish"),
            wait=release,
        )
        running_coordinator = _coordinator(store, running, offset_seconds=20)
        execution = asyncio.create_task(running_coordinator.execute(retry.run.run_id))
        while not running.prompts:
            await asyncio.sleep(0)
        assert (
            await running_coordinator.request_cancellation(retry.run.run_id)
            == CanonicalCancellationDisposition.REQUESTED
        )
        result = await execution

        assert result.disposition == CanonicalExecutionDisposition.FAILED
        assert result.run.terminal_code == "standing_cancelled"
        assert running.cancelled is True
        state = (await observation.inspect(channel_id, agent_id, current_at=_NOW + timedelta(seconds=21)))[0]
        assert state.pending_message_count == 1
        assert (await observation.pending_batches(state))[0].message_ids == (source.event.envelope.aggregate_id,)
        async with store.connection.execute(
            "SELECT COUNT(*) FROM messages WHERE body IN ('Must not publish', 'Must also not publish')"
        ) as cursor:
            assert int((await cursor.fetchone())[0]) == 0
    finally:
        await store.close()


async def test_expired_observe_attempt_recovers_after_restart_without_losing_batch(
    tmp_path: Path,
) -> None:
    database = tmp_path / "kai.db"
    store, human_id, channel_id, agent_id, _standing = await _observation_authority(database)
    observation = WorkshopStandingObservationService(store, _host_policy())
    await observation.synchronize_host_policy()
    source = await _append_message(store, channel_id, human_id, "restart-observe", offset_seconds=1)
    accepted = await observation.accept_next_ready(occurred_at=_NOW + timedelta(seconds=4))
    assert accepted is not None
    selection = RunExecutionSelection("codex", "gpt-5.6-sol", "openai")
    authority = WorkshopRunExecutionAuthority(
        store,
        selection_resolver=lambda _run: selection,
        registered_backend_ids=frozenset({"codex"}),
    )
    granted = await authority.grant(
        accepted.run.run_id,
        owner_id=RunExecutionOwnerId.new(),
        occurred_at=_NOW + timedelta(seconds=5),
        lease_expires_at=_NOW + timedelta(seconds=6),
    )
    await authority.start(granted.claim, occurred_at=_NOW + timedelta(seconds=5.5))
    await store.close()

    restarted = await WorkshopEventStore.open(database)
    restarted_observation = WorkshopStandingObservationService(restarted, _host_policy())
    prepared = _Prepared(accepted.run)
    coordinator = _coordinator(restarted, prepared, offset_seconds=7)
    try:
        recovered = await coordinator.recover_expired(occurred_at=_NOW + timedelta(seconds=7))
        assert recovered.interrupted_after_dispatch == 1
        interrupted = await WorkshopRunLifecycle(restarted).state(accepted.run.run_id)
        assert interrupted.status == RunStatus.FAILED
        assert interrupted.terminal_code == "execution_interrupted"

        pending = (
            await restarted_observation.inspect(
                channel_id,
                agent_id,
                current_at=_NOW + timedelta(seconds=8),
            )
        )[0]
        assert pending.pending_message_count == 1
        assert (await restarted_observation.pending_batches(pending))[0].message_ids == (
            source.event.envelope.aggregate_id,
        )
        assert await restarted_observation.accept_next_ready(occurred_at=_NOW + timedelta(seconds=66)) is None
        retry = await restarted_observation.accept_next_ready(occurred_at=_NOW + timedelta(seconds=67))
        assert retry is not None
        assert retry.run.run_id != accepted.run.run_id
        assert retry.run.observed_message_ids == accepted.run.observed_message_ids
    finally:
        await restarted.close()


async def test_observe_prompt_contains_only_the_immutable_channel_scope(tmp_path: Path) -> None:
    store, human_id, channel_id, agent_id, _standing = await _observation_authority(tmp_path / "kai.db")
    observation = WorkshopStandingObservationService(store, _host_policy())
    try:
        await observation.synchronize_host_policy()
        await _append_message(store, channel_id, human_id, "visible-observation", offset_seconds=1)
        other = await WorkshopChannelLifecycleService(store).create_group(
            human_id,
            name="Private other scope",
            agent_ids=[agent_id],
        )
        await _append_message(
            store,
            other.channel_id,
            human_id,
            "CROSS_SCOPE_SECRET_MUST_NOT_APPEAR",
            source="telegram",
            offset_seconds=2,
        )
        accepted = await observation.accept_next_ready(occurred_at=_NOW + timedelta(seconds=4))
        assert accepted is not None

        prompt = await observation.prompt_for_run(
            accepted.run,
            grant_operations=frozenset({CollaborationOperation.STANDING_PARTICIPATION}),
            occurred_at=_NOW + timedelta(seconds=4),
        )

        assert "visible-observation" in prompt
        assert "CROSS_SCOPE_SECRET_MUST_NOT_APPEAR" not in prompt
        assert "telegram" not in prompt.casefold()
        assert "chat_id" not in prompt
        assert "credential" not in prompt.casefold()
        assert "untrusted conversation data" in prompt
    finally:
        await store.close()


async def test_direct_agent_conversation_never_creates_standing_participation(
    tmp_path: Path,
) -> None:
    store, human_id, _group_channel_id, agent_id, standing = await _eligible_authority(tmp_path / "kai.db")
    try:
        async with store.connection.execute(
            "SELECT direct_channel_id FROM principal_agent_enablements "
            "WHERE principal_id = ? AND agent_id = ? AND lifecycle_state = 'enabled'",
            (human_id, agent_id),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        direct_channel_id = ChannelId(str(row[0]))

        accepted = await WorkshopConversationCommandService(
            store,
            standing_participation=standing,
        ).accept_client(
            ClientInboundMessage(
                principal_id=human_id,
                channel_id=direct_channel_id,
                client_message_id="standing-direct-control",
                body="@kai direct conversations remain ordinary responses",
                occurred_at=_NOW,
            )
        )

        assert len(accepted.command.runs) == 1
        assert accepted.command.runs[0].kind.value == "respond"
        async with store.connection.execute(
            "SELECT COUNT(*) FROM channel_agent_standings WHERE channel_id = ?",
            (direct_channel_id,),
        ) as cursor:
            assert int((await cursor.fetchone())[0]) == 0
    finally:
        await store.close()


@pytest.mark.parametrize(
    ("backend", "provider", "model"),
    (
        ("claude", "anthropic", "sonnet"),
        ("codex", "openai", "gpt-5.6-sol"),
        ("goose", "openai", "gpt-5.5"),
        ("opencode", "openai", "openai/gpt-5.5"),
        ("pi", "openai", "openai/gpt-5.5"),
    ),
)
async def test_standing_lane_reuses_one_backend_instance_across_observation_turns(
    tmp_path: Path,
    backend: str,
    provider: str,
    model: str,
) -> None:
    runtime_profile_id = profile_id(101)
    home = tmp_path / backend
    home.mkdir()
    option = ProtectedRuntimeBackend(backend, provider, model)
    profiles = WorkshopRuntimeProfileRegistry(
        (
            ProtectedRuntimeProfile(
                profile_id=runtime_profile_id,
                display_name=f"{backend} standing qualification",
                os_user=None,
                backend=backend,
                provider=provider,
                model=model,
                timeout_seconds=120,
                allowed_services=(),
                home_workspace=home,
                workspace_base=None,
                allowed_workspaces=(),
                backend_options=(option,),
            ),
        ),
        legacy_runtime_keys={runtime_profile_id: 101},
    )
    contexts = _internal_api_contexts(101)
    context = contexts.for_runtime_profile(runtime_profile_id)
    pool = SubprocessPool(
        config=Config(
            telegram_bot_token="test",
            allowed_user_ids={101},
            session_db_path=tmp_path / "kai.db",
        ),
        services_info=[],
        runtime_profiles=profiles,
        internal_api_contexts=contexts,
    )
    instance = pool.get(context)
    instance.shutdown = AsyncMock()
    pool._pending_workspace_restore.clear()
    pool._pending_settings_restore.clear()

    async def response_events(*_args: object, **_kwargs: object):
        yield StreamEvent(
            text_so_far="",
            done=True,
            response=AgentResponse(success=True, text="<<silent>>"),
        )

    instance.send = MagicMock(side_effect=response_events)

    first = await pool.prepare_routed_execution(context, option.option_id, model)
    async for _event in first.stream("first observation"):
        pass
    second = await pool.prepare_routed_execution(context, option.option_id, model)
    async for _event in second.stream("second observation"):
        pass

    assert first._instance is instance
    assert second._instance is instance
    assert len(pool._pool) == 1
    assert instance.send.call_count == 2
    instance.shutdown.assert_not_awaited()
