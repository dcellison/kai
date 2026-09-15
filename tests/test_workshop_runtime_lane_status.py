"""Canonical runtime-lane status transition semantics."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.domain import AgentId, ChannelId, MessageId, PrincipalId, RunId
from kai.workshop.run_execution_authority import RunExecutionSelection
from kai.workshop.runtime_lane_status import (
    RuntimeLaneStatusAuthority,
    WorkshopRuntimeLaneStatusService,
)
from kai.workshop.runtime_sessions import (
    CanonicalRuntimeSession,
    ProviderSessionResetConflictError,
    load_provider_session_reset_state,
    load_runtime_session,
    reset_provider_session,
)
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id

_NOW = datetime(2026, 9, 15, 12, 0, tzinfo=UTC)


def _authority() -> RuntimeLaneStatusAuthority:
    principal_id = PrincipalId("prn_" + "1" * 32)
    return RuntimeLaneStatusAuthority(
        requester_principal_id=principal_id,
        channel_id=ChannelId("chn_" + "2" * 32),
        channel_kind="direct",
        agent_id=AgentId("agt_" + "3" * 32),
        agent_name="Kai",
        agent_handle="kai",
        sponsor_principal_id=principal_id,
        sponsor_display_name="Daniel",
        runtime_profile_id=profile_id(1),
        owns_agent=True,
        operator=True,
    )


def _session(authority: RuntimeLaneStatusAuthority) -> CanonicalRuntimeSession:
    return CanonicalRuntimeSession(
        channel_id=authority.channel_id,
        agent_id=authority.agent_id,
        runtime_profile_id=authority.runtime_profile_id,
        selection=RunExecutionSelection(
            backend="codex",
            provider="openai",
            model="gpt-5.6-sol",
        ),
        workspace="/srv/kai",
        provider_session_id="provider-session",
        retained_context_revision="1" * 64,
        last_run_id=RunId("run_" + "4" * 32),
        last_result_message_id=MessageId("msg_" + "5" * 32),
        context_through_event_position=42,
        created_at=_NOW,
        updated_at=_NOW,
    )


def _states(
    session: CanonicalRuntimeSession | None,
    authority: RuntimeLaneStatusAuthority,
) -> tuple[str, str]:
    return WorkshopRuntimeLaneStatusService._session_states(
        session,
        authority,
        backend="codex",
        provider="openai",
        model="gpt-5.6-sol",
        workspace="/srv/kai",
    )


def test_runtime_lane_session_states_cover_start_and_stateless_backends() -> None:
    authority = _authority()
    session = _session(authority)

    assert _states(None, authority) == ("not_started", "not_started")
    assert _states(session, authority) == ("active", "active")
    assert _states(replace(session, provider_session_id=None), authority) == ("stateless", "active")


def test_runtime_lane_session_states_detect_authority_and_selection_changes() -> None:
    authority = _authority()
    session = _session(authority)

    assert _states(replace(session, runtime_profile_id=profile_id(2)), authority) == ("stale", "stale")
    assert _states(
        replace(
            session,
            selection=RunExecutionSelection(
                backend="claude",
                provider="anthropic",
                model="fable",
            ),
        ),
        authority,
    ) == ("refresh_pending", "refresh_pending")
    assert _states(replace(session, workspace="/srv/other"), authority) == (
        "refresh_pending",
        "refresh_pending",
    )
    assert _states(replace(session, retained_context_revision="0" * 64), authority) == (
        "refresh_pending",
        "refresh_pending",
    )


@pytest.mark.asyncio
async def test_fresh_provider_session_is_replay_safe_and_lane_scoped(tmp_path: Path) -> None:
    store = await WorkshopEventStore.open(tmp_path / "kai.db")
    try:
        await bootstrap_default_workshop(
            store,
            (
                BootstrapHuman("Daniel", "admin", "telegram", "101", "101", profile_id(101)),
                BootstrapHuman("Scott", "member", "telegram", "202", "202", profile_id(202)),
            ),
        )
        async with store.connection.execute(
            "SELECT e.principal_id, c.id, ca.agent_id, ra.runtime_profile_id "
            "FROM external_identities e "
            "JOIN channel_bindings cb ON cb.transport = e.provider "
            "AND cb.external_channel_id = e.external_subject "
            "JOIN channels c ON c.id = cb.channel_id "
            "JOIN channel_agents ca ON ca.channel_id = c.id "
            "JOIN channel_agent_runtime_assignments ra "
            "ON ra.channel_id = c.id AND ra.agent_id = ca.agent_id "
            "WHERE e.provider = 'telegram' ORDER BY e.external_subject"
        ) as cursor:
            rows = list(await cursor.fetchall())
        assert len(rows) == 2

        for index, row in enumerate(rows, start=1):
            await store.connection.execute(
                "INSERT INTO channel_agent_runtime_sessions ("
                "channel_id, agent_id, runtime_profile_id, backend, provider, model, workspace, "
                "provider_session_id, retained_context_revision, last_run_id, last_result_message_id, "
                "context_through_event_position, created_at, updated_at"
                ") VALUES (?, ?, ?, 'codex', 'openai', 'gpt-5.6-sol', '/srv/kai', ?, ?, ?, ?, ?, ?, ?)",
                (
                    row[1],
                    row[2],
                    row[3],
                    f"provider-session-{index}",
                    str(index) * 64,
                    str(RunId.new()),
                    str(MessageId.new()),
                    index,
                    _NOW.isoformat(),
                    _NOW.isoformat(),
                ),
            )
        await store.connection.commit()

        requester = PrincipalId(str(rows[0][0]))
        channel = ChannelId(str(rows[0][1]))
        agent = AgentId(str(rows[0][2]))
        runtime_profile = profile_id(101)
        first = await reset_provider_session(
            store,
            requester_principal_id=requester,
            channel_id=channel,
            agent_id=agent,
            runtime_profile_id=runtime_profile,
            client_operation_id="fresh-session-1",
            live_process_stopped=True,
            occurred_at=_NOW,
        )
        assert first.generation == 1
        assert first.prior_session_present is True
        assert first.live_process_stopped is True
        assert first.replayed is False
        reset_state = await load_provider_session_reset_state(store, channel, agent)
        assert reset_state is not None
        assert reset_state.generation == first.generation
        assert reset_state.revision == first.revision
        assert await load_runtime_session(store, channel, agent) is None
        assert (
            await load_runtime_session(
                store,
                ChannelId(str(rows[1][1])),
                AgentId(str(rows[1][2])),
            )
            is not None
        )

        replay = await reset_provider_session(
            store,
            requester_principal_id=requester,
            channel_id=channel,
            agent_id=agent,
            runtime_profile_id=runtime_profile,
            client_operation_id="fresh-session-1",
            live_process_stopped=False,
            occurred_at=_NOW,
        )
        assert replay.replayed is True
        assert replay.revision == first.revision
        assert replay.live_process_stopped is True

        second = await reset_provider_session(
            store,
            requester_principal_id=requester,
            channel_id=channel,
            agent_id=agent,
            runtime_profile_id=runtime_profile,
            client_operation_id="fresh-session-2",
            live_process_stopped=False,
            occurred_at=_NOW,
        )
        assert second.generation == 2
        assert second.revision != first.revision
        assert second.prior_session_present is False

        with pytest.raises(ProviderSessionResetConflictError):
            await reset_provider_session(
                store,
                requester_principal_id=requester,
                channel_id=ChannelId(str(rows[1][1])),
                agent_id=AgentId(str(rows[1][2])),
                runtime_profile_id=profile_id(202),
                client_operation_id="fresh-session-1",
                live_process_stopped=False,
                occurred_at=_NOW,
            )
    finally:
        await store.close()
