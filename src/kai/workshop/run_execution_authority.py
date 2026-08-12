"""Production-unused fenced execution authority for durable Workshop runs."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum

from kai.workshop.domain import (
    EventEnvelope,
    EventId,
    MessageId,
    PrincipalId,
    RunAttemptId,
    RunExecutionOwnerId,
    RunId,
    WorkshopEventType,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.run_lifecycle import DurableRun, RunNotFoundError, RunStatus, _load_run
from kai.workshop.store import StoredEvent, WorkshopEventStore

_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_MODEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}$")
_MAX_LEASE = timedelta(minutes=5)
_EXECUTION_CONTRACT = "trusted_host_compatibility_v1"


class RunAttemptStatus(StrEnum):
    GRANTED = "granted"
    STARTED = "started"
    EXPIRED = "expired"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class RunExecutionAuthorityError(RuntimeError):
    """Base error for a rejected run-execution authority operation."""


class RunExecutionConflictError(RunExecutionAuthorityError):
    """Current durable run state cannot accept the requested operation."""


class StaleRunExecutionAuthorityError(RunExecutionAuthorityError):
    """An owner tried to act through an expired or superseded fence."""


@dataclass(frozen=True, slots=True)
class RunExecutionSelection:
    backend: str
    model: str
    provider: str | None = None

    def __post_init__(self) -> None:
        _require_identifier(self.backend, field_name="backend")
        if not isinstance(self.model, str) or not _MODEL_PATTERN.fullmatch(self.model):
            raise ValueError("model must be a bounded model identifier")
        if self.provider is not None:
            _require_identifier(self.provider, field_name="provider")


@dataclass(frozen=True, slots=True)
class RunAttempt:
    attempt_id: RunAttemptId
    run_id: RunId
    attempt_sequence: int
    owner_id: RunExecutionOwnerId
    fence_token: int
    status: RunAttemptStatus
    selection: RunExecutionSelection
    execution_contract: str
    lease_version: int
    granted_at: datetime
    lease_expires_at: datetime
    started_at: datetime | None
    terminal_at: datetime | None
    terminal_code: str | None
    last_event_position: int


@dataclass(frozen=True, slots=True)
class RunExecutionClaim:
    attempt_id: RunAttemptId
    run_id: RunId
    owner_id: RunExecutionOwnerId
    fence_token: int
    lease_version: int

    @classmethod
    def from_attempt(cls, attempt: RunAttempt) -> RunExecutionClaim:
        return cls(
            attempt_id=attempt.attempt_id,
            run_id=attempt.run_id,
            owner_id=attempt.owner_id,
            fence_token=attempt.fence_token,
            lease_version=attempt.lease_version,
        )


@dataclass(frozen=True, slots=True)
class RunExecutionResult:
    run: DurableRun
    attempt: RunAttempt
    events: tuple[StoredEvent, ...]
    changed: bool

    @property
    def claim(self) -> RunExecutionClaim:
        return RunExecutionClaim.from_attempt(self.attempt)


@dataclass(frozen=True, slots=True)
class RunRecoveryResult:
    expired_before_dispatch: int
    interrupted_after_dispatch: int


def _timestamp(value: datetime, *, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _parse_timestamp(value: object) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def _optional_timestamp(value: object) -> datetime | None:
    return None if value is None else _parse_timestamp(value)


def _require_identifier(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not _IDENTIFIER_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase identifier")
    return value


def _require_code(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not _CODE_PATTERN.fullmatch(value):
        raise ValueError(f"{field_name} must be a lowercase identifier of at most 64 characters")
    return value


def _attempt_key(attempt_id: RunAttemptId, operation: str, *, version: int | None = None) -> str:
    suffix = f":{version}" if version is not None else ""
    return f"workshop-run-attempt:v1:{attempt_id}:{operation}{suffix}"


def _run_key(run_id: RunId, operation: str) -> str:
    return f"workshop-run:v2:{run_id}:{operation}"


async def _load_attempt(store: WorkshopEventStore, attempt_id: RunAttemptId) -> RunAttempt | None:
    async with store.connection.execute(
        "SELECT id, run_id, attempt_sequence, owner_id, fence_token, status, backend, provider, "
        "model, execution_contract, lease_version, granted_at, lease_expires_at, started_at, "
        "terminal_at, terminal_code, last_event_position FROM run_attempts WHERE id = ?",
        (attempt_id,),
    ) as cursor:
        row = await cursor.fetchone()
    if row is None:
        return None
    return RunAttempt(
        attempt_id=RunAttemptId(str(row[0])),
        run_id=RunId(str(row[1])),
        attempt_sequence=int(row[2]),
        owner_id=RunExecutionOwnerId(str(row[3])),
        fence_token=int(row[4]),
        status=RunAttemptStatus(str(row[5])),
        selection=RunExecutionSelection(
            backend=str(row[6]),
            provider=str(row[7]) if row[7] is not None else None,
            model=str(row[8]),
        ),
        execution_contract=str(row[9]),
        lease_version=int(row[10]),
        granted_at=_parse_timestamp(row[11]),
        lease_expires_at=_parse_timestamp(row[12]),
        started_at=_optional_timestamp(row[13]),
        terminal_at=_optional_timestamp(row[14]),
        terminal_code=str(row[15]) if row[15] is not None else None,
        last_event_position=int(row[16]),
    )


async def _agent_principal(store: WorkshopEventStore, run: DurableRun) -> PrincipalId:
    async with store.connection.execute(
        "SELECT principal_id FROM agents WHERE id = ? AND workshop_id = ?",
        (run.agent_id, run.workshop_id),
    ) as cursor:
        row = await cursor.fetchone()
    if row is None:
        raise RunExecutionConflictError("Durable run no longer resolves its attached agent")
    return PrincipalId(str(row[0]))


def _attempt_payload(claim: RunExecutionClaim) -> dict[str, object]:
    return {
        "run_id": claim.run_id,
        "owner_id": claim.owner_id,
        "fence_token": claim.fence_token,
        "lease_version": claim.lease_version,
    }


class WorkshopRunExecutionAuthority:
    """Grant and fence execution without starting any backend process.

    The injected resolver is the protected policy boundary: callers identify a
    run and an execution owner, but cannot substitute a command, backend, or
    model. No production object constructs this service yet.
    """

    def __init__(
        self,
        store: WorkshopEventStore,
        *,
        selection_resolver: Callable[[DurableRun], RunExecutionSelection],
        registered_backend_ids: frozenset[str],
    ) -> None:
        if not registered_backend_ids:
            raise ValueError("registered_backend_ids must not be empty")
        for backend_id in registered_backend_ids:
            _require_identifier(backend_id, field_name="registered backend ID")
        self._store = store
        self._selection_resolver = selection_resolver
        self._registered_backend_ids = registered_backend_ids

    @property
    def event_store(self) -> WorkshopEventStore:
        """Return the store used by transaction coordinators in this package."""
        return self._store

    async def attempt(self, attempt_id: RunAttemptId) -> RunAttempt:
        if not isinstance(attempt_id, RunAttemptId):
            raise ValueError("attempt_id must be a RunAttemptId")
        attempt = await _load_attempt(self._store, attempt_id)
        if attempt is None:
            raise RunExecutionConflictError("Durable Workshop run attempt was not found")
        return attempt

    async def grant(
        self,
        run_id: RunId,
        *,
        owner_id: RunExecutionOwnerId,
        occurred_at: datetime,
        lease_expires_at: datetime,
    ) -> RunExecutionResult:
        if not isinstance(run_id, RunId) or not isinstance(owner_id, RunExecutionOwnerId):
            raise ValueError("run_id and owner_id must be typed Workshop IDs")
        occurred_at = _timestamp(occurred_at, field_name="occurred_at")
        lease_expires_at = _timestamp(lease_expires_at, field_name="lease_expires_at")
        if lease_expires_at <= occurred_at or lease_expires_at - occurred_at > _MAX_LEASE:
            raise ValueError("lease must be positive and no longer than five minutes")
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            projection = CanonicalConversationProjection()
            await self._store.project_pending_in_transaction(projection)
            run = await _load_run(self._store, run_id)
            if run is None:
                raise RunNotFoundError("Durable Workshop run was not found")
            async with connection.execute(
                "SELECT id FROM run_attempts WHERE run_id = ? AND status IN ('granted', 'started')",
                (run_id,),
            ) as cursor:
                active_row = await cursor.fetchone()
            if active_row is not None:
                active = await self._require_attempt(RunAttemptId(str(active_row[0])))
                if active.owner_id != owner_id:
                    raise RunExecutionConflictError("Durable run already has an active execution owner")
                await connection.commit()
                return RunExecutionResult(run=run, attempt=active, events=(), changed=False)
            if run.status != RunStatus.ACCEPTED or run.cancellation_requested_at is not None:
                raise RunExecutionConflictError("Only an uncancelled accepted run can receive execution authority")

            selection = self._selection_resolver(run)
            if not isinstance(selection, RunExecutionSelection):
                raise TypeError("selection_resolver must return RunExecutionSelection")
            if selection.backend not in self._registered_backend_ids:
                raise RunExecutionConflictError("Resolved backend is not present in the protected registry")
            async with connection.execute(
                "SELECT COALESCE(MAX(attempt_sequence), 0) FROM run_attempts WHERE run_id = ?",
                (run_id,),
            ) as cursor:
                sequence_row = await cursor.fetchone()
            if sequence_row is None:
                raise RunExecutionConflictError("Run attempt sequence query returned no row")
            sequence = int(sequence_row[0]) + 1
            attempt_id = RunAttemptId.derived(run_id, f"attempt:{sequence}")
            actor = await _agent_principal(self._store, run)
            event = EventEnvelope.create(
                event_id=EventId.derived(attempt_id, "granted"),
                event_type=WorkshopEventType.RUN_ATTEMPT_GRANTED,
                event_version=1,
                workshop_id=run.workshop_id,
                aggregate_type="run_attempt",
                aggregate_id=attempt_id,
                actor_principal_id=actor,
                occurred_at=occurred_at,
                idempotency_key=_attempt_key(attempt_id, "granted"),
                payload={
                    "run_id": run_id,
                    "attempt_sequence": sequence,
                    "owner_id": owner_id,
                    "fence_token": sequence,
                    "backend": selection.backend,
                    "provider": selection.provider,
                    "model": selection.model,
                    "execution_contract": _EXECUTION_CONTRACT,
                    "lease_version": 1,
                    "lease_expires_at": lease_expires_at.isoformat(),
                },
                metadata={"source": "workshop_run_execution_authority"},
            )
            appended = await self._store.append_in_transaction(event)
            await self._store.project_pending_in_transaction(projection)
            attempt = await self._require_attempt(attempt_id)
            await connection.commit()
            return RunExecutionResult(run=run, attempt=attempt, events=(appended.event,), changed=True)
        except Exception:
            await connection.rollback()
            raise

    async def start(self, claim: RunExecutionClaim, *, occurred_at: datetime) -> RunExecutionResult:
        return await self._settle(
            claim,
            operation="started",
            attempt_type=WorkshopEventType.RUN_ATTEMPT_STARTED,
            run_type=WorkshopEventType.RUN_STARTED,
            occurred_at=occurred_at,
            terminal_code=None,
        )

    async def renew(
        self,
        claim: RunExecutionClaim,
        *,
        occurred_at: datetime,
        lease_expires_at: datetime,
    ) -> RunExecutionResult:
        occurred_at = _timestamp(occurred_at, field_name="occurred_at")
        lease_expires_at = _timestamp(lease_expires_at, field_name="lease_expires_at")
        if lease_expires_at <= occurred_at or lease_expires_at - occurred_at > _MAX_LEASE:
            raise ValueError("renewed lease must be positive and no longer than five minutes")
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            projection = CanonicalConversationProjection()
            await self._store.project_pending_in_transaction(projection)
            run, attempt = await self._current_claim(claim)
            next_version = claim.lease_version + 1
            key = _attempt_key(claim.attempt_id, "lease_renewed", version=next_version)
            prior = await self._store.event_by_idempotency_key(key)
            if prior is not None:
                if prior.envelope.payload.get("lease_expires_at") != lease_expires_at.isoformat():
                    raise RunExecutionConflictError("Lease renewal retry has conflicting expiry")
                await connection.commit()
                return RunExecutionResult(run=run, attempt=attempt, events=(prior,), changed=False)
            if attempt.lease_version != claim.lease_version:
                raise StaleRunExecutionAuthorityError("Execution claim has a stale lease version")
            if (
                attempt.status not in {RunAttemptStatus.GRANTED, RunAttemptStatus.STARTED}
                or occurred_at >= attempt.lease_expires_at
            ):
                raise StaleRunExecutionAuthorityError("Execution claim no longer holds active authority")
            actor = await _agent_principal(self._store, run)
            payload = _attempt_payload(claim)
            payload["lease_version"] = next_version
            payload["lease_expires_at"] = lease_expires_at.isoformat()
            event = EventEnvelope.create(
                event_id=EventId.derived(claim.attempt_id, f"lease-renewed:{next_version}"),
                event_type=WorkshopEventType.RUN_ATTEMPT_LEASE_RENEWED,
                event_version=1,
                workshop_id=run.workshop_id,
                aggregate_type="run_attempt",
                aggregate_id=claim.attempt_id,
                actor_principal_id=actor,
                occurred_at=occurred_at,
                idempotency_key=key,
                payload=payload,
                metadata={"source": "workshop_run_execution_authority"},
            )
            appended = await self._store.append_in_transaction(event)
            await self._store.project_pending_in_transaction(projection)
            updated = await self._require_attempt(claim.attempt_id)
            await connection.commit()
            return RunExecutionResult(run=run, attempt=updated, events=(appended.event,), changed=True)
        except Exception:
            await connection.rollback()
            raise

    async def request_cancellation(
        self,
        run_id: RunId,
        *,
        cancellation_code: str,
        occurred_at: datetime,
    ) -> tuple[DurableRun, StoredEvent, bool]:
        if not isinstance(run_id, RunId):
            raise ValueError("run_id must be a RunId")
        cancellation_code = _require_code(cancellation_code, field_name="cancellation_code")
        occurred_at = _timestamp(occurred_at, field_name="occurred_at")
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            projection = CanonicalConversationProjection()
            await self._store.project_pending_in_transaction(projection)
            run = await _load_run(self._store, run_id)
            if run is None:
                raise RunNotFoundError("Durable Workshop run was not found")
            key = _run_key(run_id, "cancellation_requested")
            prior = await self._store.event_by_idempotency_key(key)
            if prior is not None:
                if prior.envelope.payload != {"cancellation_code": cancellation_code}:
                    raise RunExecutionConflictError("Cancellation request retry has conflicting facts")
                await connection.commit()
                return run, prior, False
            if run.status not in {RunStatus.ACCEPTED, RunStatus.STARTED}:
                raise RunExecutionConflictError("Only a nonterminal run can request cancellation")
            event = EventEnvelope.create(
                event_id=EventId.derived(run_id, "cancellation-requested"),
                event_type=WorkshopEventType.RUN_CANCELLATION_REQUESTED,
                event_version=1,
                workshop_id=run.workshop_id,
                aggregate_type="run",
                aggregate_id=run_id,
                actor_principal_id=run.requested_by_principal_id,
                occurred_at=occurred_at,
                idempotency_key=key,
                payload={"cancellation_code": cancellation_code},
                metadata={"source": "workshop_run_execution_authority"},
            )
            appended = await self._store.append_in_transaction(event)
            await self._store.project_pending_in_transaction(projection)
            updated = await self._require_run(run_id)
            await connection.commit()
            return updated, appended.event, True
        except Exception:
            await connection.rollback()
            raise

    async def complete(
        self,
        claim: RunExecutionClaim,
        *,
        result_message_id: MessageId,
        occurred_at: datetime,
    ) -> RunExecutionResult:
        if not isinstance(result_message_id, MessageId):
            raise ValueError("result_message_id must be a MessageId")
        return await self._settle(
            claim,
            operation="completed",
            attempt_type=WorkshopEventType.RUN_ATTEMPT_COMPLETED,
            run_type=WorkshopEventType.RUN_COMPLETED,
            occurred_at=occurred_at,
            terminal_code=None,
            result_message_id=result_message_id,
        )

    async def complete_in_transaction(
        self,
        claim: RunExecutionClaim,
        *,
        result_message_id: MessageId,
        occurred_at: datetime,
    ) -> RunExecutionResult:
        """Complete a claim inside a caller-owned terminal transaction."""
        if not isinstance(result_message_id, MessageId):
            raise ValueError("result_message_id must be a MessageId")
        return await self._settle_in_transaction(
            claim,
            operation="completed",
            attempt_type=WorkshopEventType.RUN_ATTEMPT_COMPLETED,
            run_type=WorkshopEventType.RUN_COMPLETED,
            occurred_at=occurred_at,
            terminal_code=None,
            result_message_id=result_message_id,
        )

    async def fail(
        self,
        claim: RunExecutionClaim,
        *,
        failure_code: str,
        occurred_at: datetime,
    ) -> RunExecutionResult:
        return await self._settle(
            claim,
            operation="failed",
            attempt_type=WorkshopEventType.RUN_ATTEMPT_FAILED,
            run_type=WorkshopEventType.RUN_FAILED,
            occurred_at=occurred_at,
            terminal_code=_require_code(failure_code, field_name="failure_code"),
        )

    async def fail_in_transaction(
        self,
        claim: RunExecutionClaim,
        *,
        failure_code: str,
        occurred_at: datetime,
    ) -> RunExecutionResult:
        """Fail a claim inside a caller-owned terminal transaction."""
        return await self._settle_in_transaction(
            claim,
            operation="failed",
            attempt_type=WorkshopEventType.RUN_ATTEMPT_FAILED,
            run_type=WorkshopEventType.RUN_FAILED,
            occurred_at=occurred_at,
            terminal_code=_require_code(failure_code, field_name="failure_code"),
        )

    async def confirm_cancellation(
        self,
        claim: RunExecutionClaim,
        *,
        occurred_at: datetime,
    ) -> RunExecutionResult:
        return await self._settle(
            claim,
            operation="cancelled",
            attempt_type=WorkshopEventType.RUN_ATTEMPT_CANCELLED,
            run_type=WorkshopEventType.RUN_CANCELLED,
            occurred_at=occurred_at,
            terminal_code=None,
        )

    async def confirm_cancellation_in_transaction(
        self,
        claim: RunExecutionClaim,
        *,
        occurred_at: datetime,
    ) -> RunExecutionResult:
        """Confirm cancellation inside a caller-owned terminal transaction."""
        return await self._settle_in_transaction(
            claim,
            operation="cancelled",
            attempt_type=WorkshopEventType.RUN_ATTEMPT_CANCELLED,
            run_type=WorkshopEventType.RUN_CANCELLED,
            occurred_at=occurred_at,
            terminal_code=None,
        )

    async def recover_expired(self, *, occurred_at: datetime) -> RunRecoveryResult:
        occurred_at = _timestamp(occurred_at, field_name="occurred_at")
        connection = self._store.connection
        expired = interrupted = 0
        try:
            await connection.execute("BEGIN IMMEDIATE")
            projection = CanonicalConversationProjection()
            await self._store.project_pending_in_transaction(projection)
            async with connection.execute(
                "SELECT id FROM run_attempts WHERE status IN ('granted', 'started') "
                "AND lease_expires_at <= ? ORDER BY lease_expires_at, id",
                (occurred_at.isoformat(),),
            ) as cursor:
                attempt_ids = [RunAttemptId(str(row[0])) for row in await cursor.fetchall()]
            for attempt_id in attempt_ids:
                attempt = await self._require_attempt(attempt_id)
                run = await self._require_run(attempt.run_id)
                claim = RunExecutionClaim.from_attempt(attempt)
                actor = await _agent_principal(self._store, run)
                if attempt.status == RunAttemptStatus.GRANTED:
                    event = self._attempt_terminal_envelope(
                        run,
                        claim,
                        actor=actor,
                        event_type=WorkshopEventType.RUN_ATTEMPT_EXPIRED,
                        operation="expired",
                        terminal_code="lease_expired",
                        occurred_at=occurred_at,
                    )
                    await self._store.append_in_transaction(event)
                    expired += 1
                else:
                    attempt_event = self._attempt_terminal_envelope(
                        run,
                        claim,
                        actor=actor,
                        event_type=WorkshopEventType.RUN_ATTEMPT_INTERRUPTED,
                        operation="interrupted",
                        terminal_code="execution_interrupted",
                        occurred_at=occurred_at,
                    )
                    await self._store.append_in_transaction(attempt_event)
                    await self._store.append_in_transaction(
                        self._run_terminal_envelope(
                            run,
                            claim,
                            actor=actor,
                            event_type=WorkshopEventType.RUN_FAILED,
                            operation="failed",
                            terminal_code="execution_interrupted",
                            occurred_at=occurred_at,
                        )
                    )
                    interrupted += 1
            await self._store.project_pending_in_transaction(projection)
            await connection.commit()
            return RunRecoveryResult(expired_before_dispatch=expired, interrupted_after_dispatch=interrupted)
        except Exception:
            await connection.rollback()
            raise

    async def _settle(
        self,
        claim: RunExecutionClaim,
        *,
        operation: str,
        attempt_type: WorkshopEventType,
        run_type: WorkshopEventType,
        occurred_at: datetime,
        terminal_code: str | None,
        result_message_id: MessageId | None = None,
    ) -> RunExecutionResult:
        occurred_at = _timestamp(occurred_at, field_name="occurred_at")
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            result = await self._settle_in_transaction(
                claim,
                operation=operation,
                attempt_type=attempt_type,
                run_type=run_type,
                occurred_at=occurred_at,
                terminal_code=terminal_code,
                result_message_id=result_message_id,
            )
            await connection.commit()
            return result
        except Exception:
            await connection.rollback()
            raise

    async def _settle_in_transaction(
        self,
        claim: RunExecutionClaim,
        *,
        operation: str,
        attempt_type: WorkshopEventType,
        run_type: WorkshopEventType,
        occurred_at: datetime,
        terminal_code: str | None,
        result_message_id: MessageId | None = None,
    ) -> RunExecutionResult:
        if not self._store.connection.in_transaction:
            raise RuntimeError("run settlement in transaction requires an active transaction")
        occurred_at = _timestamp(occurred_at, field_name="occurred_at")
        projection = CanonicalConversationProjection()
        await self._store.project_pending_in_transaction(projection)
        run, attempt = await self._current_claim(claim)
        if operation == "cancelled":
            if run.cancellation_code is None:
                raise RunExecutionConflictError("Run has no durable cancellation request")
            terminal_code = run.cancellation_code
        attempt_key = _attempt_key(claim.attempt_id, operation)
        prior_attempt = await self._store.event_by_idempotency_key(attempt_key)
        run_key = _run_key(claim.run_id, operation)
        prior_run = await self._store.event_by_idempotency_key(run_key)
        actor = await _agent_principal(self._store, run)
        if prior_attempt is not None or prior_run is not None:
            if prior_attempt is None or prior_run is None:
                raise RunExecutionConflictError("Atomic run transition is missing one of its events")
            expected_attempt_payload = _attempt_payload(claim)
            if terminal_code is not None:
                expected_attempt_payload["terminal_code"] = terminal_code
            expected_run_payload: dict[str, object] = {"attempt_id": claim.attempt_id}
            if run_type == WorkshopEventType.RUN_COMPLETED:
                expected_run_payload["result_message_id"] = result_message_id
            elif run_type == WorkshopEventType.RUN_FAILED:
                expected_run_payload["failure_code"] = terminal_code
            elif run_type == WorkshopEventType.RUN_CANCELLED:
                expected_run_payload["cancellation_code"] = terminal_code
            if (
                prior_attempt.envelope.event_type != attempt_type
                or prior_attempt.envelope.aggregate_id != claim.attempt_id
                or prior_attempt.envelope.actor_principal_id != actor
                or prior_attempt.envelope.payload != expected_attempt_payload
                or prior_run.envelope.event_type != run_type
                or prior_run.envelope.aggregate_id != claim.run_id
                or prior_run.envelope.actor_principal_id != actor
                or prior_run.envelope.payload != expected_run_payload
            ):
                raise RunExecutionConflictError("Run transition retry has conflicting facts")
            return RunExecutionResult(
                run=run,
                attempt=attempt,
                events=(prior_attempt, prior_run),
                changed=False,
            )
        if attempt.lease_version != claim.lease_version:
            raise StaleRunExecutionAuthorityError("Execution claim has a stale lease version")
        allowed_statuses = (
            {RunAttemptStatus.GRANTED}
            if operation == "started"
            else {RunAttemptStatus.GRANTED, RunAttemptStatus.STARTED}
            if operation == "cancelled"
            else {RunAttemptStatus.STARTED}
        )
        if attempt.status not in allowed_statuses or occurred_at >= attempt.lease_expires_at:
            raise StaleRunExecutionAuthorityError("Execution claim no longer holds active authority")
        if operation == "started" and run.cancellation_requested_at is not None:
            raise RunExecutionConflictError("Execution cannot start after cancellation was requested")
        if operation == "started":
            attempt_event = self._attempt_transition_envelope(
                run,
                claim,
                actor=actor,
                event_type=attempt_type,
                operation=operation,
                occurred_at=occurred_at,
            )
            run_event = self._run_transition_envelope(
                run,
                claim,
                actor=actor,
                event_type=run_type,
                operation=operation,
                occurred_at=occurred_at,
            )
        else:
            attempt_event = self._attempt_terminal_envelope(
                run,
                claim,
                actor=actor,
                event_type=attempt_type,
                operation=operation,
                terminal_code=terminal_code,
                occurred_at=occurred_at,
            )
            run_event = self._run_terminal_envelope(
                run,
                claim,
                actor=actor,
                event_type=run_type,
                operation=operation,
                terminal_code=terminal_code,
                result_message_id=result_message_id,
                occurred_at=occurred_at,
            )
        first = await self._store.append_in_transaction(attempt_event)
        second = await self._store.append_in_transaction(run_event)
        await self._store.project_pending_in_transaction(projection)
        updated_run = await self._require_run(claim.run_id)
        updated_attempt = await self._require_attempt(claim.attempt_id)
        return RunExecutionResult(
            run=updated_run,
            attempt=updated_attempt,
            events=(first.event, second.event),
            changed=True,
        )

    async def _current_claim(self, claim: RunExecutionClaim) -> tuple[DurableRun, RunAttempt]:
        if not isinstance(claim, RunExecutionClaim):
            raise ValueError("claim must be a RunExecutionClaim")
        run = await self._require_run(claim.run_id)
        attempt = await self._require_attempt(claim.attempt_id)
        if (
            attempt.run_id != claim.run_id
            or attempt.owner_id != claim.owner_id
            or attempt.fence_token != claim.fence_token
        ):
            raise StaleRunExecutionAuthorityError("Execution claim does not match its fenced owner")
        return run, attempt

    async def _require_run(self, run_id: RunId) -> DurableRun:
        run = await _load_run(self._store, run_id)
        if run is None:
            raise RunNotFoundError("Durable Workshop run was not found")
        return run

    async def _require_attempt(self, attempt_id: RunAttemptId) -> RunAttempt:
        attempt = await _load_attempt(self._store, attempt_id)
        if attempt is None:
            raise RunExecutionConflictError("Durable Workshop run attempt was not found")
        return attempt

    @staticmethod
    def _attempt_transition_envelope(
        run: DurableRun,
        claim: RunExecutionClaim,
        *,
        actor: PrincipalId,
        event_type: WorkshopEventType,
        operation: str,
        occurred_at: datetime,
    ) -> EventEnvelope:
        return EventEnvelope.create(
            event_id=EventId.derived(claim.attempt_id, operation),
            event_type=event_type,
            event_version=1,
            workshop_id=run.workshop_id,
            aggregate_type="run_attempt",
            aggregate_id=claim.attempt_id,
            actor_principal_id=actor,
            occurred_at=occurred_at,
            idempotency_key=_attempt_key(claim.attempt_id, operation),
            payload=_attempt_payload(claim),
            metadata={"source": "workshop_run_execution_authority"},
        )

    @staticmethod
    def _attempt_terminal_envelope(
        run: DurableRun,
        claim: RunExecutionClaim,
        *,
        actor: PrincipalId,
        event_type: WorkshopEventType,
        operation: str,
        terminal_code: str | None,
        occurred_at: datetime,
    ) -> EventEnvelope:
        payload = _attempt_payload(claim)
        if terminal_code is not None:
            payload["terminal_code"] = terminal_code
        return EventEnvelope.create(
            event_id=EventId.derived(claim.attempt_id, operation),
            event_type=event_type,
            event_version=1,
            workshop_id=run.workshop_id,
            aggregate_type="run_attempt",
            aggregate_id=claim.attempt_id,
            actor_principal_id=actor,
            occurred_at=occurred_at,
            idempotency_key=_attempt_key(claim.attempt_id, operation),
            payload=payload,
            metadata={"source": "workshop_run_execution_authority"},
        )

    @staticmethod
    def _run_transition_envelope(
        run: DurableRun,
        claim: RunExecutionClaim,
        *,
        actor: PrincipalId,
        event_type: WorkshopEventType,
        operation: str,
        occurred_at: datetime,
    ) -> EventEnvelope:
        return EventEnvelope.create(
            event_id=EventId.derived(run.run_id, f"v2:{operation}"),
            event_type=event_type,
            event_version=2,
            workshop_id=run.workshop_id,
            aggregate_type="run",
            aggregate_id=run.run_id,
            actor_principal_id=actor,
            occurred_at=occurred_at,
            idempotency_key=_run_key(run.run_id, operation),
            payload={"attempt_id": claim.attempt_id},
            metadata={"source": "workshop_run_execution_authority"},
        )

    @staticmethod
    def _run_terminal_envelope(
        run: DurableRun,
        claim: RunExecutionClaim,
        *,
        actor: PrincipalId,
        event_type: WorkshopEventType,
        operation: str,
        terminal_code: str | None,
        occurred_at: datetime,
        result_message_id: MessageId | None = None,
    ) -> EventEnvelope:
        payload: dict[str, object] = {"attempt_id": claim.attempt_id}
        if event_type == WorkshopEventType.RUN_COMPLETED:
            if result_message_id is None:
                raise ValueError("Completed run requires result_message_id")
            payload["result_message_id"] = result_message_id
        elif event_type == WorkshopEventType.RUN_FAILED:
            if terminal_code is None:
                raise ValueError("Failed run requires failure_code")
            payload["failure_code"] = terminal_code
        elif event_type == WorkshopEventType.RUN_CANCELLED:
            if terminal_code is None:
                raise ValueError("Cancelled run requires cancellation_code")
            payload["cancellation_code"] = terminal_code
        return EventEnvelope.create(
            event_id=EventId.derived(run.run_id, f"v2:{operation}"),
            event_type=event_type,
            event_version=2,
            workshop_id=run.workshop_id,
            aggregate_type="run",
            aggregate_id=run.run_id,
            actor_principal_id=actor,
            occurred_at=occurred_at,
            idempotency_key=_run_key(run.run_id, operation),
            payload=payload,
            metadata={"source": "workshop_run_execution_authority"},
        )
