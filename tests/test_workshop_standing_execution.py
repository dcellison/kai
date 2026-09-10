"""Execution contracts for bounded standing-agent observation runs."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from kai.backend import AgentResponse
from kai.workshop.conversation_commands import WorkshopConversationCommandService
from kai.workshop.diagnostics import (
    workshop_runtime_session_status,
    workshop_standing_observe_execution_status,
)
from kai.workshop.execution_coordinator import (
    CanonicalExecutionDisposition,
    WorkshopCanonicalExecutionCoordinator,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.run_lifecycle import RunKind, RunStatus
from kai.workshop.runtime_sessions import load_runtime_session
from kai.workshop.standing_observation import StandingObserveSettlement, WorkshopStandingObservationService
from tests.test_workshop_execution_coordinator import (
    _Preparation,
    _PreparationByRun,
    _Prepared,
    _RejectedPreparation,
)
from tests.test_workshop_standing_observation import (
    _NOW,
    _append_message,
    _host_policy,
    _observation_authority,
    _other_agent_principal,
)
from tests.test_workshop_standing_participation import _message
from tests.workshop_delivery import TELEGRAM_DELIVERY_POLICY


def _coordinator(store, prepared, policy):
    assert prepared.run.runtime_profile_id is not None
    prepared.runtime_profile_id = prepared.run.runtime_profile_id
    return WorkshopCanonicalExecutionCoordinator(
        store,
        _Preparation(prepared),
        registered_backend_ids=frozenset({"codex"}),
        clock=lambda: _NOW + timedelta(seconds=30),
        delivery_policy=TELEGRAM_DELIVERY_POLICY,
        collaboration_host_policy=policy,
    )


async def test_ready_batch_becomes_immutable_observe_run_and_exact_silence_advances_cursor(
    tmp_path: Path,
) -> None:
    database = tmp_path / "kai.db"
    store, human_id, channel_id, agent_id, _standing = await _observation_authority(database)
    policy = _host_policy()
    observation = WorkshopStandingObservationService(store, policy)
    try:
        await observation.synchronize_host_policy()
        first = await _append_message(store, channel_id, human_id, "observe-silent-one", offset_seconds=1)
        second = await _append_message(store, channel_id, human_id, "observe-silent-two", offset_seconds=1.1)

        accepted = await observation.accept_next_ready(occurred_at=_NOW + timedelta(seconds=4))
        assert accepted is not None
        assert accepted.run.kind == RunKind.OBSERVE
        assert accepted.run.observed_message_ids == (
            first.event.envelope.aggregate_id,
            second.event.envelope.aggregate_id,
        )
        prepared = _Prepared(
            accepted.run,
            response=AgentResponse(success=True, text="<<silent>>", session_id="standing-session"),
        )
        result = await _coordinator(store, prepared, policy).execute(accepted.run.run_id)

        assert result.disposition == CanonicalExecutionDisposition.COMPLETED
        assert result.run.status == RunStatus.COMPLETED
        assert result.run.standing_outcome == "silent"
        assert "Observed messages:" in prepared.prompts[0]
        assert "observe-silent-one" in prepared.prompts[0]
        assert "Effective authority: standing_participation=granted" in prepared.prompts[0]
        assert "Remaining limits: observe inferences=29; visible contributions=12" in prepared.prompts[0]
        assert "observe-silent-one" not in prepared.canonical_histories[0]
        assert "observe-silent-two" not in prepared.canonical_histories[0]
        async with store.connection.execute(
            "SELECT pending_message_count, delivered_through_event_position "
            "FROM channel_agent_observation_states WHERE channel_id = ? AND agent_id = ?",
            (channel_id, agent_id),
        ) as cursor:
            state = await cursor.fetchone()
        assert tuple(state) == (0, second.event.position)
        async with store.connection.execute("SELECT COUNT(*) FROM messages WHERE body = '<<silent>>'") as cursor:
            assert int((await cursor.fetchone())[0]) == 0
    finally:
        await store.close()
    status = workshop_standing_observe_execution_status(database)
    assert status.startswith("Workshop standing observe execution: active;")
    assert "runs=1 (nonterminal=0, spoke=0, silent=1, suppressed=0, failed=0)" in status
    assert "quota ledger=(inferences=1, publications=0)" in status


async def test_standing_publication_is_once_per_human_anchor_and_suppressed_text_is_protected(
    tmp_path: Path,
) -> None:
    store, human_id, channel_id, _agent_id, _standing = await _observation_authority(tmp_path / "kai.db")
    policy = _host_policy(minimum_unsolicited_interval_seconds=1)
    observation = WorkshopStandingObservationService(store, policy)
    try:
        await observation.synchronize_host_policy()
        await _append_message(store, channel_id, human_id, "observe-anchor", offset_seconds=1)
        first = await observation.accept_next_ready(occurred_at=_NOW + timedelta(seconds=4))
        assert first is not None
        spoken = _Prepared(
            first.run,
            response=AgentResponse(
                success=True,
                text="Useful contribution",
                session_id="standing-provider-session",
            ),
        )
        first_result = await _coordinator(store, spoken, policy).execute(first.run.run_id)
        assert first_result.run.standing_outcome == "spoke"
        assert isinstance(first_result.terminal, StandingObserveSettlement)
        assert first_result.terminal.runtime_session is not None
        session = await load_runtime_session(store, channel_id, first.run.agent_id)
        assert session is not None
        assert session.last_run_id == first.run.run_id
        assert session.last_result_message_id == first_result.run.result_message_id
        assert session.provider_session_id == "standing-provider-session"
        assert workshop_runtime_session_status(tmp_path / "kai.db").startswith(
            "Workshop conversation continuity: active; successful lanes=1, sessions=1"
        )

        peer = await _other_agent_principal(store, human_id)
        await _append_message(store, channel_id, peer, "observe-peer-followup", offset_seconds=12)
        second = await observation.accept_next_ready(occurred_at=_NOW + timedelta(seconds=15))
        assert second is not None
        assert second.run.human_anchor_message_id == first.run.human_anchor_message_id
        suppressed = _Prepared(second.run, response=AgentResponse(success=True, text="Second contribution"))
        second_result = await _coordinator(store, suppressed, policy).execute(second.run.run_id)

        assert second_result.run.standing_outcome == "publication_suppressed"
        async with store.connection.execute(
            "SELECT body, suppression_reason FROM standing_observation_suppressed_outputs WHERE run_id = ?",
            (second.run.run_id,),
        ) as cursor:
            assert tuple(await cursor.fetchone()) == ("Second contribution", "anchor_used")
        async with store.connection.execute(
            "SELECT body FROM messages WHERE body IN ('Useful contribution', 'Second contribution') "
            "ORDER BY created_event_position"
        ) as cursor:
            assert [str(row[0]) for row in await cursor.fetchall()] == ["Useful contribution"]
        assert "missing=0, stale=0" in workshop_runtime_session_status(tmp_path / "kai.db")
    finally:
        await store.close()


async def test_empty_observe_response_fails_silently_and_defers_retry(tmp_path: Path) -> None:
    store, human_id, channel_id, _agent_id, _standing = await _observation_authority(tmp_path / "kai.db")
    policy = _host_policy()
    observation = WorkshopStandingObservationService(store, policy)
    try:
        await observation.synchronize_host_policy()
        await _append_message(store, channel_id, human_id, "observe-empty", offset_seconds=1)
        accepted = await observation.accept_next_ready(occurred_at=_NOW + timedelta(seconds=4))
        assert accepted is not None
        empty = _Prepared(accepted.run, response=AgentResponse(success=True, text=""))
        result = await _coordinator(store, empty, policy).execute(accepted.run.run_id)

        assert result.disposition == CanonicalExecutionDisposition.FAILED
        assert result.run.terminal_code == "no_response"
        async with store.connection.execute(
            "SELECT pending_message_count, not_before FROM channel_agent_observation_states "
            "WHERE channel_id = ? AND agent_id = ?",
            (channel_id, accepted.run.agent_id),
        ) as cursor:
            state = await cursor.fetchone()
        assert int(state[0]) == 1
        assert str(state[1]) == (_NOW + timedelta(seconds=90)).isoformat()
    finally:
        await store.close()


async def test_host_disable_after_acceptance_fails_closed_before_backend_dispatch(tmp_path: Path) -> None:
    store, human_id, channel_id, _agent_id, _standing = await _observation_authority(tmp_path / "kai.db")
    policy = _host_policy()
    observation = WorkshopStandingObservationService(store, policy)
    try:
        await observation.synchronize_host_policy()
        await _append_message(store, channel_id, human_id, "observe-revoked", offset_seconds=1)
        accepted = await observation.accept_next_ready(occurred_at=_NOW + timedelta(seconds=4))
        assert accepted is not None
        await store.connection.execute("UPDATE standing_participation_host_policy SET enabled = 0")
        await store.connection.commit()
        prepared = _Prepared(accepted.run, response=AgentResponse(success=True, text="Must not publish"))

        result = await _coordinator(store, prepared, policy).execute(accepted.run.run_id)

        assert result.disposition == CanonicalExecutionDisposition.FAILED
        assert result.run.terminal_code == "standing_authority_revoked"
        assert prepared.prompts == []
        async with store.connection.execute(
            "SELECT pending_message_count, delivered_through_event_position "
            "FROM channel_agent_observation_states WHERE channel_id = ? AND agent_id = ?",
            (channel_id, accepted.run.agent_id),
        ) as cursor:
            state = await cursor.fetchone()
        assert int(state[0]) == 1
        assert int(state[1]) < accepted.batch.through_event_position
    finally:
        await store.close()


async def test_host_disable_after_backend_response_suppresses_proposed_publication(tmp_path: Path) -> None:
    store, human_id, channel_id, _agent_id, _standing = await _observation_authority(tmp_path / "kai.db")
    policy = _host_policy()
    observation = WorkshopStandingObservationService(store, policy)
    try:
        await observation.synchronize_host_policy()
        await _append_message(store, channel_id, human_id, "observe-publication-revoked", offset_seconds=1)
        accepted = await observation.accept_next_ready(occurred_at=_NOW + timedelta(seconds=4))
        assert accepted is not None

        async def disable_host() -> None:
            await store.connection.execute("UPDATE standing_participation_host_policy SET enabled = 0")
            await store.connection.commit()

        body = "Proposed after authority was revoked"
        prepared = _Prepared(
            accepted.run,
            response=AgentResponse(success=True, text=body),
            on_stream=disable_host,
        )
        result = await _coordinator(store, prepared, policy).execute(accepted.run.run_id)

        assert result.disposition == CanonicalExecutionDisposition.FAILED
        assert result.run.terminal_code == "standing_authority_revoked"
        async with store.connection.execute(
            "SELECT body, suppression_reason FROM standing_observation_suppressed_outputs WHERE run_id = ?",
            (accepted.run.run_id,),
        ) as cursor:
            assert tuple(await cursor.fetchone()) == (body, "authority_revoked")
        async with store.connection.execute("SELECT COUNT(*) FROM messages WHERE body = ?", (body,)) as cursor:
            assert int((await cursor.fetchone())[0]) == 0
    finally:
        await store.close()


async def test_mixed_silent_sentinel_is_visible_and_recorded_as_protocol_anomaly(tmp_path: Path) -> None:
    store, human_id, channel_id, _agent_id, _standing = await _observation_authority(tmp_path / "kai.db")
    policy = _host_policy()
    observation = WorkshopStandingObservationService(store, policy)
    try:
        await observation.synchronize_host_policy()
        await _append_message(store, channel_id, human_id, "observe-mixed", offset_seconds=1)
        accepted = await observation.accept_next_ready(occurred_at=_NOW + timedelta(seconds=4))
        assert accepted is not None
        body = "Useful contribution containing <<silent>> visibly"
        prepared = _Prepared(accepted.run, response=AgentResponse(success=True, text=body))

        result = await _coordinator(store, prepared, policy).execute(accepted.run.run_id)

        assert result.run.standing_outcome == "spoke"
        async with store.connection.execute(
            "SELECT body FROM messages WHERE id = ?",
            (result.run.result_message_id,),
        ) as cursor:
            assert str((await cursor.fetchone())[0]) == body
        async with store.connection.execute(
            "SELECT body FROM standing_observation_protocol_anomalies WHERE run_id = ?",
            (accepted.run.run_id,),
        ) as cursor:
            assert str((await cursor.fetchone())[0]) == body
    finally:
        await store.close()


async def test_inference_quota_is_independent_and_projection_rebuild_is_exact(tmp_path: Path) -> None:
    store, human_id, channel_id, _agent_id, _standing = await _observation_authority(tmp_path / "kai.db")
    policy = _host_policy(max_observe_runs_per_hour=1)
    observation = WorkshopStandingObservationService(store, policy)
    try:
        await observation.synchronize_host_policy()
        await _append_message(store, channel_id, human_id, "observe-quota-one", offset_seconds=1)
        accepted = await observation.accept_next_ready(occurred_at=_NOW + timedelta(seconds=4))
        assert accepted is not None
        prepared = _Prepared(accepted.run, response=AgentResponse(success=True, text="Quota contribution"))
        result = await _coordinator(store, prepared, policy).execute(accepted.run.run_id)
        assert result.run.standing_outcome == "spoke"

        await _append_message(store, channel_id, human_id, "observe-quota-two", offset_seconds=10)
        assert await observation.accept_next_ready(occurred_at=_NOW + timedelta(seconds=13)) is None

        await store.rebuild_projection(CanonicalConversationProjection())
        rebuilt = await observation.inspect(channel_id, accepted.run.agent_id, current_at=_NOW + timedelta(seconds=13))
        assert rebuilt[0].pending_message_count == 1
        async with store.connection.execute(
            "SELECT result_message_id FROM standing_observation_publications WHERE run_id = ?",
            (accepted.run.run_id,),
        ) as cursor:
            publication = await cursor.fetchone()
        assert publication is not None
        assert str(publication[0]) == str(result.run.result_message_id)
    finally:
        await store.close()


async def test_routing_rejection_is_silent_and_has_a_fresh_standing_grant(tmp_path: Path) -> None:
    store, human_id, channel_id, _agent_id, _standing = await _observation_authority(tmp_path / "kai.db")
    policy = _host_policy()
    observation = WorkshopStandingObservationService(store, policy)
    try:
        await observation.synchronize_host_policy()
        await _append_message(store, channel_id, human_id, "observe-routing-rejected", offset_seconds=1)
        accepted = await observation.accept_next_ready(occurred_at=_NOW + timedelta(seconds=4))
        assert accepted is not None
        coordinator = WorkshopCanonicalExecutionCoordinator(
            store,
            _RejectedPreparation(accepted.run),
            registered_backend_ids=frozenset({"codex"}),
            clock=lambda: _NOW + timedelta(seconds=30),
            delivery_policy=TELEGRAM_DELIVERY_POLICY,
            collaboration_host_policy=policy,
        )

        result = await coordinator.execute(accepted.run.run_id)

        assert result.disposition == CanonicalExecutionDisposition.FAILED
        assert result.run.terminal_code == "routing_ineligible"
        async with store.connection.execute(
            "SELECT COUNT(*) FROM collaboration_grants WHERE run_id = ?",
            (accepted.run.run_id,),
        ) as cursor:
            assert int((await cursor.fetchone())[0]) == 1
        async with store.connection.execute(
            "SELECT COUNT(*) FROM messages WHERE body LIKE '%routing policy%'",
        ) as cursor:
            assert int((await cursor.fetchone())[0]) == 0
    finally:
        await store.close()


async def test_respond_run_supersedes_accepted_observe_without_second_backend_call(tmp_path: Path) -> None:
    store, human_id, channel_id, _agent_id, standing = await _observation_authority(tmp_path / "kai.db")
    policy = _host_policy()
    observation = WorkshopStandingObservationService(store, policy)
    try:
        await observation.synchronize_host_policy()
        await _append_message(store, channel_id, human_id, "observe-before-respond", offset_seconds=1)
        accepted = await observation.accept_next_ready(occurred_at=_NOW + timedelta(seconds=4))
        assert accepted is not None
        command = await WorkshopConversationCommandService(
            store,
            standing_participation=standing,
        ).accept_client(_message(human_id, channel_id, "explicit-respond-priority"))
        assert len(command.command.runs) == 1
        response_run = command.command.runs[0]
        response = _Prepared(response_run, response=AgentResponse(success=True, text="Explicit response"))
        observe = _Prepared(accepted.run, response=AgentResponse(success=True, text="Must not run"))
        coordinator = WorkshopCanonicalExecutionCoordinator(
            store,
            _PreparationByRun((response, observe)),
            registered_backend_ids=frozenset({"codex"}),
            clock=lambda: _NOW + timedelta(seconds=30),
            delivery_policy=TELEGRAM_DELIVERY_POLICY,
            collaboration_host_policy=policy,
        )

        deferred = await coordinator.execute(accepted.run.run_id)
        assert deferred.disposition == CanonicalExecutionDisposition.PREPARATION_DEFERRED
        assert observe.prompts == []
        completed = await coordinator.execute(response_run.run_id)
        assert completed.disposition == CanonicalExecutionDisposition.COMPLETED
        superseded = await coordinator.execute(accepted.run.run_id)

        assert superseded.disposition == CanonicalExecutionDisposition.CANCELLED
        assert superseded.run.cancellation_code == "respond_superseded"
        assert observe.prompts == []
    finally:
        await store.close()
