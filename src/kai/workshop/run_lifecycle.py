"""Production-unused durable lifecycle for canonical Workshop runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from kai.workshop.conversation_runs import resolve_canonical_conversation_target
from kai.workshop.domain import (
    AgentId,
    ChannelId,
    EventEnvelope,
    EventId,
    MessageId,
    PrincipalId,
    RunId,
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


async def _load_run(store: WorkshopEventStore, run_id: RunId) -> DurableRun | None:
    async with store.connection.execute(
        "SELECT id, workshop_id, channel_id, requested_by_principal_id, agent_id, "
        "inbound_message_id, status, accepted_at, started_at, terminal_at, terminal_code, "
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
        status=RunStatus(str(row[6])),
        accepted_at=_parse_timestamp(row[7]),
        started_at=_optional_timestamp(row[8]),
        terminal_at=_optional_timestamp(row[9]),
        terminal_code=str(row[10]) if row[10] is not None else None,
        cancellation_requested_at=_optional_timestamp(row[11]),
        cancellation_code=str(row[12]) if row[12] is not None else None,
        result_message_id=MessageId(str(row[13])) if row[13] is not None else None,
        last_event_position=int(row[14]),
    )


async def _load_run_by_inbound_message(
    store: WorkshopEventStore,
    inbound_message_id: MessageId,
) -> DurableRun | None:
    async with store.connection.execute(
        "SELECT id FROM runs WHERE inbound_message_id = ?",
        (inbound_message_id,),
    ) as cursor:
        row = await cursor.fetchone()
    return None if row is None else await _load_run(store, RunId(str(row[0])))


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
) -> StoredEvent | None:
    key = _event_key(run.run_id, status)
    existing = await store.event_by_idempotency_key(key)
    if existing is None:
        return None
    envelope = existing.envelope
    if (
        envelope.event_type != _event_type(status)
        or envelope.event_version != 1
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
        run = await _load_run(self._store, run_id)
        if run is None:
            raise RunNotFoundError("Durable Workshop run was not found")
        return run

    async def accept(self, inbound_message_id: MessageId, *, occurred_at: datetime) -> RunLifecycleResult:
        if not isinstance(inbound_message_id, MessageId):
            raise ValueError("inbound_message_id must be a MessageId")
        occurred_at = _require_timestamp(occurred_at)
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            result = await self.accept_in_transaction(inbound_message_id, occurred_at=occurred_at)
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
    ) -> RunLifecycleResult:
        """Accept one run inside a caller-owned transaction."""
        if not isinstance(inbound_message_id, MessageId):
            raise ValueError("inbound_message_id must be a MessageId")
        occurred_at = _require_timestamp(occurred_at)
        if not self._store.connection.in_transaction:
            raise RuntimeError("accept_in_transaction requires an active transaction")

        projection = CanonicalConversationProjection()
        await self._store.project_pending_in_transaction(projection)
        existing_run = await _load_run_by_inbound_message(self._store, inbound_message_id)
        if existing_run is not None:
            existing_payload: dict[str, object] = {
                "inbound_message_id": existing_run.inbound_message_id,
                "channel_id": existing_run.channel_id,
                "requested_by_principal_id": existing_run.requested_by_principal_id,
                "agent_id": existing_run.agent_id,
            }
            existing_event = await _existing_event(
                self._store,
                run=existing_run,
                status=RunStatus.ACCEPTED,
                actor_principal_id=existing_run.requested_by_principal_id,
                payload=existing_payload,
            )
            if existing_event is None:
                raise RunLifecycleConflictError("Durable run is missing its acceptance event")
            return RunLifecycleResult(run=existing_run, event=existing_event, changed=False)

        target = await resolve_canonical_conversation_target(self._store, inbound_message_id)
        run_id = RunId.derived(target.workshop_id, f"conversation:{inbound_message_id}")
        payload: dict[str, object] = {
            "inbound_message_id": target.inbound_message_id,
            "channel_id": target.channel_id,
            "requested_by_principal_id": target.requested_by_principal_id,
            "agent_id": target.agent_id,
        }
        event = EventEnvelope.create(
            event_id=EventId.derived(run_id, "accepted"),
            event_type=WorkshopEventType.RUN_ACCEPTED,
            event_version=1,
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
        run = await _load_run(self._store, run_id)
        if run is None:
            raise RunLifecycleConflictError("Accepted run was not projected")
        return RunLifecycleResult(run=run, event=appended.event, changed=appended.inserted)
