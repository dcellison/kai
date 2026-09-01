"""Durable lifecycle for canonical Workshop runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from kai.workshop.conversation_runs import (
    ConversationRunUnavailableError,
    resolve_canonical_conversation_run,
    resolve_canonical_conversation_target,
)
from kai.workshop.domain import (
    AgentDefinitionRevisionId,
    AgentDelegationId,
    AgentId,
    ChannelId,
    EventEnvelope,
    EventId,
    MessageId,
    PrincipalId,
    RunId,
    RuntimeProfileId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import StoredEvent, WorkshopEventStore


class RunStatus(StrEnum):
    ACCEPTED = "accepted"
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunLifecycleError(RuntimeError):
    """Base error for a rejected durable run operation."""


class RunNotFoundError(RunLifecycleError, LookupError):
    """A typed run ID does not identify a durable run."""


class RunLifecycleConflictError(RunLifecycleError):
    """A deterministic lifecycle identity was reused with different facts."""


@dataclass(frozen=True, slots=True)
class DurableRun:
    run_id: RunId
    workshop_id: WorkshopId
    channel_id: ChannelId
    requested_by_principal_id: PrincipalId
    agent_id: AgentId
    inbound_message_id: MessageId
    status: RunStatus
    accepted_at: datetime
    started_at: datetime | None
    terminal_at: datetime | None
    terminal_code: str | None
    cancellation_requested_at: datetime | None
    cancellation_code: str | None
    result_message_id: MessageId | None
    last_event_position: int
    agent_definition_revision_id: AgentDefinitionRevisionId | None = None
    runtime_profile_id: RuntimeProfileId | None = None
    sponsor_principal_id: PrincipalId | None = None
    parent_run_id: RunId | None = None
    delegation_id: AgentDelegationId | None = None


@dataclass(frozen=True, slots=True)
class RunLifecycleResult:
    run: DurableRun
    event: StoredEvent
    changed: bool


def _require_timestamp(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("occurred_at must be timezone-aware")
    return value.astimezone(UTC)


def _parse_timestamp(value: object) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def _optional_timestamp(value: object) -> datetime | None:
    return None if value is None else _parse_timestamp(value)


async def load_durable_run(store: WorkshopEventStore, run_id: RunId) -> DurableRun | None:
    """Load one projected durable run without granting any authorization."""
    async with store.connection.execute("PRAGMA table_info(runs)") as cursor:
        run_columns = {str(row[1]) for row in await cursor.fetchall()}
    revision_expression = (
        "agent_definition_revision_id"
        if "agent_definition_revision_id" in run_columns
        else "NULL AS agent_definition_revision_id"
    )
    runtime_expression = "runtime_profile_id" if "runtime_profile_id" in run_columns else "NULL AS runtime_profile_id"
    sponsor_expression = (
        "sponsor_principal_id" if "sponsor_principal_id" in run_columns else "NULL AS sponsor_principal_id"
    )
    parent_expression = "parent_run_id" if "parent_run_id" in run_columns else "NULL AS parent_run_id"
    delegation_expression = "delegation_id" if "delegation_id" in run_columns else "NULL AS delegation_id"
    async with store.connection.execute(
        "SELECT id, workshop_id, channel_id, requested_by_principal_id, agent_id, "
        f"inbound_message_id, {revision_expression}, {runtime_expression}, {sponsor_expression}, "
        f"{parent_expression}, {delegation_expression}, "
        "status, accepted_at, "
        "started_at, terminal_at, terminal_code, "
        "cancellation_requested_at, cancellation_code, result_message_id, "
        "last_event_position FROM runs WHERE id = ?",
        (run_id,),
    ) as cursor:
        row = await cursor.fetchone()
    if row is None:
        return None
    return DurableRun(
        run_id=RunId(str(row[0])),
        workshop_id=WorkshopId(str(row[1])),
        channel_id=ChannelId(str(row[2])),
        requested_by_principal_id=PrincipalId(str(row[3])),
        agent_id=AgentId(str(row[4])),
        inbound_message_id=MessageId(str(row[5])),
        agent_definition_revision_id=(AgentDefinitionRevisionId(str(row[6])) if row[6] is not None else None),
        runtime_profile_id=RuntimeProfileId(str(row[7])) if row[7] is not None else None,
        sponsor_principal_id=PrincipalId(str(row[8])) if row[8] is not None else None,
        parent_run_id=RunId(str(row[9])) if row[9] is not None else None,
        delegation_id=AgentDelegationId(str(row[10])) if row[10] is not None else None,
        status=RunStatus(str(row[11])),
        accepted_at=_parse_timestamp(row[12]),
        started_at=_optional_timestamp(row[13]),
        terminal_at=_optional_timestamp(row[14]),
        terminal_code=str(row[15]) if row[15] is not None else None,
        cancellation_requested_at=_optional_timestamp(row[16]),
        cancellation_code=str(row[17]) if row[17] is not None else None,
        result_message_id=MessageId(str(row[18])) if row[18] is not None else None,
        last_event_position=int(row[19]),
    )


async def _load_run_by_inbound_message(
    store: WorkshopEventStore,
    inbound_message_id: MessageId,
    agent_id: AgentId | None = None,
) -> DurableRun | None:
    async with store.connection.execute(
        "SELECT id FROM runs WHERE inbound_message_id = ? AND (? IS NULL OR agent_id = ?) ORDER BY id",
        (inbound_message_id, agent_id, agent_id),
    ) as cursor:
        rows = list(await cursor.fetchall())
    if len(rows) > 1:
        raise RunLifecycleConflictError("Inbound message has multiple runs; an agent_id is required")
    return None if not rows else await load_durable_run(store, RunId(str(rows[0][0])))


def _event_key(run_id: RunId, status: RunStatus) -> str:
    return f"workshop-run:v1:{run_id}:{status}"


def _event_type(status: RunStatus) -> WorkshopEventType:
    return {
        RunStatus.ACCEPTED: WorkshopEventType.RUN_ACCEPTED,
        RunStatus.STARTED: WorkshopEventType.RUN_STARTED,
        RunStatus.COMPLETED: WorkshopEventType.RUN_COMPLETED,
        RunStatus.FAILED: WorkshopEventType.RUN_FAILED,
        RunStatus.CANCELLED: WorkshopEventType.RUN_CANCELLED,
    }[status]


async def _existing_event(
    store: WorkshopEventStore,
    *,
    run: DurableRun,
    status: RunStatus,
    actor_principal_id: PrincipalId,
    payload: dict[str, object],
    event_version: int = 1,
) -> StoredEvent | None:
    key = _event_key(run.run_id, status)
    existing = await store.event_by_idempotency_key(key)
    if existing is None:
        return None
    envelope = existing.envelope
    if (
        envelope.event_type != _event_type(status)
        or envelope.event_version != event_version
        or envelope.workshop_id != run.workshop_id
        or envelope.aggregate_type != "run"
        or envelope.aggregate_id != run.run_id
        or envelope.actor_principal_id != actor_principal_id
        or envelope.payload != payload
    ):
        raise RunLifecycleConflictError(f"Durable run event identity for {status} has conflicting facts")
    return existing


class WorkshopRunLifecycle:
    """Create and transition canonical runs without invoking a backend.

    Nothing constructs this service in production yet. Its only authority is
    durable event and projection state; it owns no process, worker, or client
    endpoint.
    """

    def __init__(self, store: WorkshopEventStore) -> None:
        self._store = store

    async def state(self, run_id: RunId) -> DurableRun:
        if not isinstance(run_id, RunId):
            raise ValueError("run_id must be a RunId")
        run = await load_durable_run(self._store, run_id)
        if run is None:
            raise RunNotFoundError("Durable Workshop run was not found")
        return run

    async def accept(
        self,
        inbound_message_id: MessageId,
        *,
        occurred_at: datetime,
        agent_id: AgentId | None = None,
    ) -> RunLifecycleResult:
        if not isinstance(inbound_message_id, MessageId):
            raise ValueError("inbound_message_id must be a MessageId")
        occurred_at = _require_timestamp(occurred_at)
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            result = await self.accept_in_transaction(
                inbound_message_id,
                occurred_at=occurred_at,
                agent_id=agent_id,
            )
            await connection.commit()
            return result
        except Exception:
            await connection.rollback()
            raise

    async def accept_in_transaction(
        self,
        inbound_message_id: MessageId,
        *,
        occurred_at: datetime,
        agent_id: AgentId | None = None,
    ) -> RunLifecycleResult:
        """Accept one run inside a caller-owned transaction."""
        if not isinstance(inbound_message_id, MessageId):
            raise ValueError("inbound_message_id must be a MessageId")
        occurred_at = _require_timestamp(occurred_at)
        if not self._store.connection.in_transaction:
            raise RuntimeError("accept_in_transaction requires an active transaction")

        projection = CanonicalConversationProjection()
        await self._store.project_pending_in_transaction(projection)
        existing_run = await _load_run_by_inbound_message(
            self._store,
            inbound_message_id,
            agent_id,
        )
        if existing_run is not None:
            existing_payload: dict[str, object] = {
                "inbound_message_id": existing_run.inbound_message_id,
                "channel_id": existing_run.channel_id,
                "requested_by_principal_id": existing_run.requested_by_principal_id,
                "agent_id": existing_run.agent_id,
            }
            event_version = 1
            if existing_run.agent_definition_revision_id is not None:
                existing_payload["agent_definition_revision_id"] = existing_run.agent_definition_revision_id
                event_version = 2
            if existing_run.runtime_profile_id is not None and existing_run.sponsor_principal_id is not None:
                existing_payload["runtime_profile_id"] = existing_run.runtime_profile_id
                existing_payload["sponsor_principal_id"] = existing_run.sponsor_principal_id
                event_version = 3
            if existing_run.parent_run_id is not None and existing_run.delegation_id is not None:
                existing_payload["parent_run_id"] = existing_run.parent_run_id
                existing_payload["delegation_id"] = existing_run.delegation_id
                event_version = 4
            existing_event = await _existing_event(
                self._store,
                run=existing_run,
                status=RunStatus.ACCEPTED,
                actor_principal_id=existing_run.requested_by_principal_id,
                payload=existing_payload,
                event_version=event_version,
            )
            if existing_event is None:
                raise RunLifecycleConflictError("Durable run is missing its acceptance event")
            return RunLifecycleResult(run=existing_run, event=existing_event, changed=False)

        target = await resolve_canonical_conversation_target(
            self._store,
            inbound_message_id,
            agent_id,
        )
        resolution = None
        try:
            resolution = await resolve_canonical_conversation_run(
                self._store,
                inbound_message_id,
                agent_id,
            )
        except ConversationRunUnavailableError:
            async with self._store.connection.execute(
                "SELECT kind FROM channels WHERE id = ?",
                (target.channel_id,),
            ) as cursor:
                channel_row = await cursor.fetchone()
            if channel_row is None or str(channel_row[0]) != "direct":
                raise
        run_id = RunId.derived(
            target.workshop_id,
            f"conversation:{inbound_message_id}:{target.agent_id}",
        )
        payload: dict[str, object] = {
            "inbound_message_id": target.inbound_message_id,
            "channel_id": target.channel_id,
            "requested_by_principal_id": target.requested_by_principal_id,
            "agent_id": target.agent_id,
        }
        async with self._store.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'agent_definitions'"
        ) as cursor:
            definitions_supported = await cursor.fetchone() is not None
        event_version = 1
        if definitions_supported:
            async with self._store.connection.execute(
                "SELECT active_revision_id FROM agent_definitions WHERE agent_id = ? AND lifecycle_state = 'active'",
                (target.agent_id,),
            ) as cursor:
                definition_row = await cursor.fetchone()
            if definition_row is None or definition_row[0] is None:
                raise RunLifecycleConflictError("Agent has no active canonical definition revision")
            definition_revision_id = AgentDefinitionRevisionId(str(definition_row[0]))
            payload["agent_definition_revision_id"] = definition_revision_id
            event_version = 2
        async with self._store.connection.execute("PRAGMA table_info(runs)") as cursor:
            run_columns = {str(row[1]) for row in await cursor.fetchall()}
        if resolution is not None and definitions_supported and "runtime_profile_id" in run_columns:
            payload["runtime_profile_id"] = resolution.runtime_profile_id
            payload["sponsor_principal_id"] = resolution.sponsor_principal_id
            event_version = 3
        event = EventEnvelope.create(
            event_id=EventId.derived(run_id, "accepted"),
            event_type=WorkshopEventType.RUN_ACCEPTED,
            event_version=event_version,
            workshop_id=target.workshop_id,
            aggregate_type="run",
            aggregate_id=run_id,
            actor_principal_id=target.requested_by_principal_id,
            occurred_at=occurred_at,
            idempotency_key=_event_key(run_id, RunStatus.ACCEPTED),
            payload=payload,
            metadata={"source": "workshop_run_lifecycle"},
        )
        appended = await self._store.append_in_transaction(event)
        await self._store.project_pending_in_transaction(projection)
        run = await load_durable_run(self._store, run_id)
        if run is None:
            raise RunLifecycleConflictError("Accepted run was not projected")
        return RunLifecycleResult(run=run, event=appended.event, changed=appended.inserted)
