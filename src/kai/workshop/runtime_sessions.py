"""Canonical channel-agent runtime continuity state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from kai.workshop.domain import AgentId, ChannelId, MessageId, RunId, RuntimeProfileId
from kai.workshop.run_execution_authority import RunExecutionSelection
from kai.workshop.store import WorkshopEventStore


class RuntimeSessionStateError(RuntimeError):
    """Canonical runtime-session facts conflict with durable authority."""


class RuntimeSessionStateConflictError(RuntimeSessionStateError):
    """Continuity bookkeeping is stale or conflicts with newer authority."""


@dataclass(frozen=True, slots=True)
class RuntimeSessionSettlement:
    channel_id: ChannelId
    agent_id: AgentId
    runtime_profile_id: RuntimeProfileId
    selection: RunExecutionSelection
    workspace: str
    provider_session_id: str | None
    run_id: RunId


@dataclass(frozen=True, slots=True)
class CanonicalRuntimeSession:
    channel_id: ChannelId
    agent_id: AgentId
    runtime_profile_id: RuntimeProfileId
    selection: RunExecutionSelection
    workspace: str
    provider_session_id: str | None
    last_run_id: RunId
    last_result_message_id: MessageId
    context_through_event_position: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class RuntimeSessionSettlementResult:
    session: CanonicalRuntimeSession
    changed: bool


def _timestamp(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("occurred_at must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _parse_timestamp(value: object) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def _require_text(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value


def _from_row(row) -> CanonicalRuntimeSession:
    return CanonicalRuntimeSession(
        channel_id=ChannelId(str(row[0])),
        agent_id=AgentId(str(row[1])),
        runtime_profile_id=RuntimeProfileId(str(row[2])),
        selection=RunExecutionSelection(
            backend=str(row[3]),
            provider=str(row[4]) if row[4] is not None else None,
            model=str(row[5]),
        ),
        workspace=str(row[6]),
        provider_session_id=str(row[7]) if row[7] is not None else None,
        last_run_id=RunId(str(row[8])),
        last_result_message_id=MessageId(str(row[9])),
        context_through_event_position=int(row[10]),
        created_at=_parse_timestamp(row[11]),
        updated_at=_parse_timestamp(row[12]),
    )


async def load_runtime_session(
    store: WorkshopEventStore,
    channel_id: ChannelId,
    agent_id: AgentId,
) -> CanonicalRuntimeSession | None:
    """Load canonical continuity state for one conversation lane."""
    async with store.connection.execute(
        "SELECT channel_id, agent_id, runtime_profile_id, backend, provider, model, "
        "workspace, provider_session_id, last_run_id, last_result_message_id, "
        "context_through_event_position, created_at, updated_at "
        "FROM channel_agent_runtime_sessions WHERE channel_id = ? AND agent_id = ?",
        (channel_id, agent_id),
    ) as cursor:
        row = await cursor.fetchone()
    return None if row is None else _from_row(row)


async def settle_runtime_session_in_transaction(
    store: WorkshopEventStore,
    settlement: RuntimeSessionSettlement,
    *,
    result_message_id: MessageId,
    context_through_event_position: int,
    occurred_at: datetime,
) -> RuntimeSessionSettlementResult:
    """Atomically advance canonical continuity after a successful run."""
    if not store.connection.in_transaction:
        raise RuntimeError("settle_runtime_session_in_transaction requires an active transaction")
    if not isinstance(settlement, RuntimeSessionSettlement):
        raise TypeError("settlement must be a RuntimeSessionSettlement")
    if not isinstance(result_message_id, MessageId):
        raise TypeError("result_message_id must be a MessageId")
    if context_through_event_position <= 0:
        raise ValueError("context_through_event_position must be positive")
    _require_text(settlement.workspace, "workspace")
    when = _timestamp(occurred_at)

    async with store.connection.execute("PRAGMA table_info(agent_definitions)") as cursor:
        definition_columns = {str(row[1]) for row in await cursor.fetchall()}
    if "owner_runtime_profile_id" in definition_columns:
        async with store.connection.execute(
            "SELECT CASE WHEN d.lifecycle_state = 'active' THEN "
            "COALESCE(d.owner_runtime_profile_id, ca.sponsored_runtime_profile_id, ra.runtime_profile_id) "
            "END FROM channel_agents ca JOIN agent_definitions d ON d.agent_id = ca.agent_id "
            "LEFT JOIN channel_agent_runtime_assignments ra ON ra.channel_id = ca.channel_id "
            "AND ra.agent_id = ca.agent_id WHERE ca.channel_id = ? AND ca.agent_id = ? "
            "AND ca.detached_at IS NULL",
            (settlement.channel_id, settlement.agent_id),
        ) as cursor:
            authority = await cursor.fetchone()
    else:
        async with store.connection.execute(
            "SELECT runtime_profile_id FROM channel_agent_runtime_assignments WHERE channel_id = ? AND agent_id = ?",
            (settlement.channel_id, settlement.agent_id),
        ) as cursor:
            authority = await cursor.fetchone()
    if authority is None or authority[0] is None or str(authority[0]) != settlement.runtime_profile_id:
        raise RuntimeSessionStateConflictError("Runtime session does not match current canonical authority")

    existing = await load_runtime_session(store, settlement.channel_id, settlement.agent_id)
    expected = (
        settlement.runtime_profile_id,
        settlement.selection,
        settlement.workspace,
        settlement.provider_session_id,
        settlement.run_id,
        result_message_id,
        context_through_event_position,
    )
    if existing is not None and existing.last_run_id == settlement.run_id:
        actual = (
            existing.runtime_profile_id,
            existing.selection,
            existing.workspace,
            existing.provider_session_id,
            existing.last_run_id,
            existing.last_result_message_id,
            existing.context_through_event_position,
        )
        if actual != expected:
            raise RuntimeSessionStateConflictError("Runtime session replay has conflicting facts")
        return RuntimeSessionSettlementResult(existing, changed=False)
    if existing is not None and existing.context_through_event_position >= context_through_event_position:
        raise RuntimeSessionStateConflictError("Runtime session context boundary cannot move backward")

    await store.connection.execute(
        "INSERT INTO channel_agent_runtime_sessions ("
        "channel_id, agent_id, runtime_profile_id, backend, provider, model, workspace, "
        "provider_session_id, last_run_id, last_result_message_id, "
        "context_through_event_position, created_at, updated_at"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(channel_id, agent_id) DO UPDATE SET "
        "runtime_profile_id=excluded.runtime_profile_id, backend=excluded.backend, "
        "provider=excluded.provider, model=excluded.model, workspace=excluded.workspace, "
        "provider_session_id=excluded.provider_session_id, last_run_id=excluded.last_run_id, "
        "last_result_message_id=excluded.last_result_message_id, "
        "context_through_event_position=excluded.context_through_event_position, "
        "updated_at=excluded.updated_at",
        (
            settlement.channel_id,
            settlement.agent_id,
            settlement.runtime_profile_id,
            settlement.selection.backend,
            settlement.selection.provider,
            settlement.selection.model,
            settlement.workspace,
            settlement.provider_session_id,
            settlement.run_id,
            result_message_id,
            context_through_event_position,
            existing.created_at.isoformat().replace("+00:00", "Z") if existing else when,
            when,
        ),
    )
    current = await load_runtime_session(store, settlement.channel_id, settlement.agent_id)
    if current is None:
        raise RuntimeSessionStateError("Runtime session settlement was not stored")
    return RuntimeSessionSettlementResult(current, changed=True)
