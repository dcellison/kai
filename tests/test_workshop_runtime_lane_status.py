"""Canonical runtime-lane status transition semantics."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

from kai.workshop.domain import AgentId, ChannelId, MessageId, PrincipalId, RunId
from kai.workshop.run_execution_authority import RunExecutionSelection
from kai.workshop.runtime_lane_status import (
    RuntimeLaneStatusAuthority,
    WorkshopRuntimeLaneStatusService,
)
from kai.workshop.runtime_sessions import CanonicalRuntimeSession
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
