"""Canonical channel-agent runtime continuity state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime

from kai.workshop.domain import AgentId, ChannelId, MessageId, PrincipalId, RunId, RuntimeProfileId
from kai.workshop.run_execution_authority import RunExecutionSelection
from kai.workshop.store import WorkshopEventStore


class RuntimeSessionStateError(RuntimeError):
    """Canonical runtime-session facts conflict with durable authority."""


class RuntimeSessionStateConflictError(RuntimeSessionStateError):
    """Continuity bookkeeping is stale or conflicts with newer authority."""


class ProviderSessionResetConflictError(RuntimeSessionStateError):
    """A reset operation ID was reused for a different runtime lane."""


@dataclass(frozen=True, slots=True)
class RuntimeSessionSettlement:
    channel_id: ChannelId
    agent_id: AgentId
    runtime_profile_id: RuntimeProfileId
    selection: RunExecutionSelection
    workspace: str
    provider_session_id: str | None
    run_id: RunId
    retained_context_revision: str = "0" * 64


@dataclass(frozen=True, slots=True)
class CanonicalRuntimeSession:
    channel_id: ChannelId
    agent_id: AgentId
    runtime_profile_id: RuntimeProfileId
    selection: RunExecutionSelection
    workspace: str
    provider_session_id: str | None
    retained_context_revision: str | None
    last_run_id: RunId
    last_result_message_id: MessageId
    context_through_event_position: int
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True, slots=True)
class RuntimeSessionSettlementResult:
    session: CanonicalRuntimeSession
    changed: bool


@dataclass(frozen=True, slots=True)
class ProviderSessionResetResult:
    requester_principal_id: PrincipalId
    channel_id: ChannelId
    agent_id: AgentId
    runtime_profile_id: RuntimeProfileId
    client_operation_id: str
    generation: int
    revision: str
    prior_session_present: bool
    live_process_stopped: bool
    created_at: datetime
    replayed: bool


@dataclass(frozen=True, slots=True)
class ProviderSessionResetState:
    channel_id: ChannelId
    agent_id: AgentId
    runtime_profile_id: RuntimeProfileId
    generation: int
    revision: str
    updated_at: datetime


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


def _from_row(row, *, has_retained_context_revision: bool = True) -> CanonicalRuntimeSession:
    offset = 1 if has_retained_context_revision else 0
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
        retained_context_revision=(str(row[8]) if row[8] is not None else None)
        if has_retained_context_revision
        else None,
        last_run_id=RunId(str(row[8 + offset])),
        last_result_message_id=MessageId(str(row[9 + offset])),
        context_through_event_position=int(row[10 + offset]),
        created_at=_parse_timestamp(row[11 + offset]),
        updated_at=_parse_timestamp(row[12 + offset]),
    )


async def _has_retained_context_revision(store: WorkshopEventStore) -> bool:
    async with store.connection.execute("PRAGMA table_info(channel_agent_runtime_sessions)") as cursor:
        return "retained_context_revision" in {str(row[1]) for row in await cursor.fetchall()}


async def load_runtime_session(
    store: WorkshopEventStore,
    channel_id: ChannelId,
    agent_id: AgentId,
) -> CanonicalRuntimeSession | None:
    """Load canonical continuity state for one conversation lane."""
    has_revision = await _has_retained_context_revision(store)
    revision_column = "retained_context_revision, " if has_revision else ""
    async with store.connection.execute(
        "SELECT channel_id, agent_id, runtime_profile_id, backend, provider, model, "
        f"workspace, provider_session_id, {revision_column}last_run_id, last_result_message_id, "
        "context_through_event_position, created_at, updated_at "
        "FROM channel_agent_runtime_sessions WHERE channel_id = ? AND agent_id = ?",
        (channel_id, agent_id),
    ) as cursor:
        row = await cursor.fetchone()
    return None if row is None else _from_row(row, has_retained_context_revision=has_revision)


async def clear_runtime_session(
    store: WorkshopEventStore,
    channel_id: ChannelId,
    agent_id: AgentId,
) -> bool:
    """Delete obsolete provider continuity for exactly one canonical lane."""
    cursor = await store.connection.execute(
        "DELETE FROM channel_agent_runtime_sessions WHERE channel_id = ? AND agent_id = ?",
        (channel_id, agent_id),
    )
    await store.connection.commit()
    return cursor.rowcount > 0


def _provider_session_reset_from_row(row, *, replayed: bool) -> ProviderSessionResetResult:
    return ProviderSessionResetResult(
        requester_principal_id=PrincipalId(str(row[0])),
        client_operation_id=str(row[1]),
        channel_id=ChannelId(str(row[2])),
        agent_id=AgentId(str(row[3])),
        runtime_profile_id=RuntimeProfileId(str(row[4])),
        generation=int(row[5]),
        revision=str(row[6]),
        prior_session_present=bool(row[7]),
        live_process_stopped=bool(row[8]),
        created_at=_parse_timestamp(row[9]),
        replayed=replayed,
    )


async def load_provider_session_reset_operation(
    store: WorkshopEventStore,
    requester_principal_id: PrincipalId,
    client_operation_id: str,
) -> ProviderSessionResetResult | None:
    """Load one replay-safe reset receipt without changing its runtime lane."""
    async with store.connection.execute(
        "SELECT requester_principal_id, client_operation_id, channel_id, agent_id, "
        "runtime_profile_id, generation, revision, prior_session_present, "
        "live_process_stopped, created_at FROM provider_session_reset_operations "
        "WHERE requester_principal_id = ? AND client_operation_id = ?",
        (requester_principal_id, client_operation_id),
    ) as cursor:
        row = await cursor.fetchone()
    return None if row is None else _provider_session_reset_from_row(row, replayed=True)


async def load_provider_session_reset_state(
    store: WorkshopEventStore,
    channel_id: ChannelId,
    agent_id: AgentId,
) -> ProviderSessionResetState | None:
    """Load the shared fresh-session boundary visible to every adapter."""
    async with store.connection.execute(
        "SELECT channel_id, agent_id, runtime_profile_id, generation, revision, updated_at "
        "FROM provider_session_reset_state WHERE channel_id = ? AND agent_id = ?",
        (channel_id, agent_id),
    ) as cursor:
        row = await cursor.fetchone()
    if row is None:
        return None
    return ProviderSessionResetState(
        channel_id=ChannelId(str(row[0])),
        agent_id=AgentId(str(row[1])),
        runtime_profile_id=RuntimeProfileId(str(row[2])),
        generation=int(row[3]),
        revision=str(row[4]),
        updated_at=_parse_timestamp(row[5]),
    )


async def reset_provider_session(
    store: WorkshopEventStore,
    *,
    requester_principal_id: PrincipalId,
    channel_id: ChannelId,
    agent_id: AgentId,
    runtime_profile_id: RuntimeProfileId,
    client_operation_id: str,
    live_process_stopped: bool,
    occurred_at: datetime,
) -> ProviderSessionResetResult:
    """Invalidate exactly one canonical lane and persist its replay-safe revision."""
    if not isinstance(client_operation_id, str) or not 1 <= len(client_operation_id) <= 128:
        raise ValueError("client_operation_id must contain between 1 and 128 characters")
    if occurred_at.tzinfo is None or occurred_at.utcoffset() is None:
        raise ValueError("occurred_at must be timezone-aware")
    when = _timestamp(occurred_at)
    await store.connection.execute("BEGIN IMMEDIATE")
    try:
        existing = await load_provider_session_reset_operation(
            store,
            requester_principal_id,
            client_operation_id,
        )
        if existing is not None:
            if (
                existing.channel_id != channel_id
                or existing.agent_id != agent_id
                or existing.runtime_profile_id != runtime_profile_id
            ):
                raise ProviderSessionResetConflictError(
                    "The fresh-session operation ID already belongs to another runtime lane"
                )
            await store.connection.commit()
            return existing

        async with store.connection.execute(
            "SELECT generation FROM provider_session_reset_state WHERE channel_id = ? AND agent_id = ?",
            (channel_id, agent_id),
        ) as cursor:
            state = await cursor.fetchone()
        generation = 1 if state is None else int(state[0]) + 1
        revision_payload = {
            "version": 1,
            "requester_principal_id": str(requester_principal_id),
            "channel_id": str(channel_id),
            "agent_id": str(agent_id),
            "runtime_profile_id": str(runtime_profile_id),
            "client_operation_id": client_operation_id,
            "generation": generation,
        }
        revision = hashlib.sha256(
            json.dumps(revision_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        async with store.connection.execute(
            "SELECT 1 FROM channel_agent_runtime_sessions WHERE channel_id = ? AND agent_id = ?",
            (channel_id, agent_id),
        ) as cursor:
            prior_session_present = await cursor.fetchone() is not None
        await store.connection.execute(
            "DELETE FROM channel_agent_runtime_sessions WHERE channel_id = ? AND agent_id = ?",
            (channel_id, agent_id),
        )
        await store.connection.execute(
            "INSERT INTO provider_session_reset_state "
            "(channel_id, agent_id, runtime_profile_id, generation, revision, "
            "last_client_operation_id, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(channel_id, agent_id) DO UPDATE SET "
            "runtime_profile_id=excluded.runtime_profile_id, generation=excluded.generation, "
            "revision=excluded.revision, last_client_operation_id=excluded.last_client_operation_id, "
            "updated_at=excluded.updated_at",
            (
                channel_id,
                agent_id,
                runtime_profile_id,
                generation,
                revision,
                client_operation_id,
                when,
            ),
        )
        await store.connection.execute(
            "INSERT INTO provider_session_reset_operations "
            "(requester_principal_id, client_operation_id, channel_id, agent_id, "
            "runtime_profile_id, generation, revision, prior_session_present, "
            "live_process_stopped, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                requester_principal_id,
                client_operation_id,
                channel_id,
                agent_id,
                runtime_profile_id,
                generation,
                revision,
                int(prior_session_present),
                int(live_process_stopped),
                when,
            ),
        )
        await store.connection.commit()
    except BaseException:
        await store.connection.rollback()
        raise
    return ProviderSessionResetResult(
        requester_principal_id=requester_principal_id,
        channel_id=channel_id,
        agent_id=agent_id,
        runtime_profile_id=runtime_profile_id,
        client_operation_id=client_operation_id,
        generation=generation,
        revision=revision,
        prior_session_present=prior_session_present,
        live_process_stopped=live_process_stopped,
        created_at=occurred_at.astimezone(UTC),
        replayed=False,
    )


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
        settlement.retained_context_revision,
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
            existing.retained_context_revision,
            existing.last_run_id,
            existing.last_result_message_id,
            existing.context_through_event_position,
        )
        if actual != expected:
            raise RuntimeSessionStateConflictError("Runtime session replay has conflicting facts")
        return RuntimeSessionSettlementResult(existing, changed=False)
    if existing is not None and existing.context_through_event_position >= context_through_event_position:
        raise RuntimeSessionStateConflictError("Runtime session context boundary cannot move backward")

    has_revision = await _has_retained_context_revision(store)
    columns = (
        "channel_id, agent_id, runtime_profile_id, backend, provider, model, workspace, "
        "provider_session_id, "
        + ("retained_context_revision, " if has_revision else "")
        + "last_run_id, last_result_message_id, context_through_event_position, created_at, updated_at"
    )
    update_revision = "retained_context_revision=excluded.retained_context_revision, " if has_revision else ""
    values: tuple[object, ...] = (
        settlement.channel_id,
        settlement.agent_id,
        settlement.runtime_profile_id,
        settlement.selection.backend,
        settlement.selection.provider,
        settlement.selection.model,
        settlement.workspace,
        settlement.provider_session_id,
        *((settlement.retained_context_revision,) if has_revision else ()),
        settlement.run_id,
        result_message_id,
        context_through_event_position,
        existing.created_at.isoformat().replace("+00:00", "Z") if existing else when,
        when,
    )
    await store.connection.execute(
        f"INSERT INTO channel_agent_runtime_sessions ({columns}) "
        f"VALUES ({', '.join('?' for _ in values)}) "
        "ON CONFLICT(channel_id,agent_id) DO UPDATE SET "
        "runtime_profile_id=excluded.runtime_profile_id, backend=excluded.backend, "
        "provider=excluded.provider, model=excluded.model, workspace=excluded.workspace, "
        f"provider_session_id=excluded.provider_session_id, {update_revision}"
        "last_run_id=excluded.last_run_id, last_result_message_id=excluded.last_result_message_id, "
        "context_through_event_position=excluded.context_through_event_position, updated_at=excluded.updated_at",
        values,
    )
    current = await load_runtime_session(store, settlement.channel_id, settlement.agent_id)
    if current is None:
        raise RuntimeSessionStateError("Runtime session settlement was not stored")
    return RuntimeSessionSettlementResult(current, changed=True)
