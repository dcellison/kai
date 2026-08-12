"""Production-unused canonical coordination for durable Workshop execution."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Protocol

from kai.agent_failure import AgentFailureKind
from kai.backend import AgentResponse
from kai.workshop.domain import AgentId, ChannelId, RunExecutionOwnerId, RunId
from kai.workshop.protected_execution import PreparedWorkshopExecution
from kai.workshop.run_execution_authority import (
    RunAttemptStatus,
    RunExecutionClaim,
    RunExecutionConflictError,
    RunExecutionSelection,
    WorkshopRunExecutionAuthority,
)
from kai.workshop.run_lifecycle import DurableRun, RunStatus, WorkshopRunLifecycle
from kai.workshop.store import WorkshopEventStore
from kai.workshop.terminal_transactions import (
    TerminalFailureCode,
    TerminalTransactionResult,
    WorkshopRunTerminalTransactionCoordinator,
)


class ProtectedPreparation(Protocol):
    async def prepare(self, run_id: RunId) -> PreparedWorkshopExecution: ...


class CanonicalExecutionDisposition(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TERMINAL_REPLAY = "terminal_replay"
    ACTIVE_REPLAY = "active_replay"
    CANCELLATION_PENDING_REPLAY = "cancellation_pending_replay"
    PREPARATION_DEFERRED = "preparation_deferred"


class CanonicalCancellationDisposition(StrEnum):
    REQUESTED = "requested"
    NOT_ACTIVE = "not_active"
    ALREADY_TERMINAL = "already_terminal"
    INTERRUPTED = "interrupted"


@dataclass(frozen=True, slots=True)
class CanonicalExecutionResult:
    disposition: CanonicalExecutionDisposition
    run: DurableRun
    terminal: TerminalTransactionResult | None = None


@dataclass(frozen=True, slots=True)
class CanonicalRecoveryResult:
    expired_before_dispatch: int
    interrupted_after_dispatch: int


@dataclass(slots=True)
class _ActiveExecution:
    run_id: RunId
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    finished: asyncio.Event = field(default_factory=asyncio.Event)
    cancellation_done: asyncio.Event = field(default_factory=asyncio.Event)
    renewal_stop: asyncio.Event = field(default_factory=asyncio.Event)
    cancel_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    claim_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    cancellation_requested: bool = False
    cancellation_error: BaseException | None = None
    prepared: PreparedWorkshopExecution | None = None
    authority: WorkshopRunExecutionAuthority | None = None
    claim: RunExecutionClaim | None = None
    started: bool = False
    settling: bool = False


_FAILURE_CODE_BY_KIND = {
    AgentFailureKind.AUTHENTICATION_EXPIRED: TerminalFailureCode.AUTHENTICATION_EXPIRED,
    AgentFailureKind.AUTHENTICATION_REQUIRED: TerminalFailureCode.AUTHENTICATION_REQUIRED,
    AgentFailureKind.QUOTA_EXHAUSTED: TerminalFailureCode.QUOTA_EXHAUSTED,
    AgentFailureKind.MODEL_UNAVAILABLE: TerminalFailureCode.MODEL_UNAVAILABLE,
    AgentFailureKind.PROVIDER_UNAVAILABLE: TerminalFailureCode.PROVIDER_UNAVAILABLE,
    AgentFailureKind.TRANSIENT: TerminalFailureCode.TRANSIENT,
    AgentFailureKind.BACKEND_CRASHED: TerminalFailureCode.BACKEND_CRASHED,
    AgentFailureKind.UNKNOWN: TerminalFailureCode.UNKNOWN,
}


class WorkshopCanonicalExecutionCoordinator:
    """Own one canonical lane from accepted run through terminal settlement.

    The public execution input is only a canonical ``RunId``. Prompt, lane,
    compatibility runtime identity, backend selection, owner, and delivery
    authority are all derived behind this boundary. Production does not yet
    construct this coordinator.
    """

    def __init__(
        self,
        store: WorkshopEventStore,
        preparation: ProtectedPreparation,
        *,
        registered_backend_ids: frozenset[str],
        clock: Callable[[], datetime] | None = None,
        lease_duration: timedelta = timedelta(minutes=5),
    ) -> None:
        if not registered_backend_ids:
            raise ValueError("registered_backend_ids must not be empty")
        if lease_duration <= timedelta(0) or lease_duration > timedelta(minutes=5):
            raise ValueError("lease_duration must be positive and no longer than five minutes")
        self._store = store
        self._preparation = preparation
        self._registered_backend_ids = registered_backend_ids
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lease_duration = lease_duration
        self._lanes: dict[tuple[ChannelId, AgentId], asyncio.Lock] = {}
        self._active: dict[RunId, _ActiveExecution] = {}
        self._map_lock = asyncio.Lock()
        self._database_lock = asyncio.Lock()

    async def execute(self, run_id: RunId) -> CanonicalExecutionResult:
        if not isinstance(run_id, RunId):
            raise ValueError("run_id must be a RunId")
        run = await self._run(run_id)
        lane = await self._lane(run.channel_id, run.agent_id)
        async with lane:
            run = await self._run(run_id)
            replay = await self._replay_disposition(run)
            if replay is not None:
                return CanonicalExecutionResult(replay, run)

            active = _ActiveExecution(run_id)
            async with self._map_lock:
                self._active[run_id] = active
            try:
                return await self._execute_owned(active, run)
            finally:
                active.finished.set()
                active.ready.set()
                active.cancellation_done.set()
                async with self._map_lock:
                    if self._active.get(run_id) is active:
                        del self._active[run_id]

    async def request_cancellation(self, run_id: RunId) -> CanonicalCancellationDisposition:
        if not isinstance(run_id, RunId):
            raise ValueError("run_id must be a RunId")
        async with self._map_lock:
            active = self._active.get(run_id)
            if active is not None:
                active.cancellation_requested = True
        if active is None:
            run = await self._run(run_id)
            return (
                CanonicalCancellationDisposition.ALREADY_TERMINAL
                if run.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
                else CanonicalCancellationDisposition.NOT_ACTIVE
            )

        await active.ready.wait()
        async with active.cancel_lock:
            if active.finished.is_set() or active.claim is None or active.authority is None or active.prepared is None:
                return CanonicalCancellationDisposition.NOT_ACTIVE
            if not active.cancellation_done.is_set():
                try:
                    async with self._database_lock:
                        await active.authority.request_cancellation(
                            run_id,
                            cancellation_code="requested_by_human",
                            occurred_at=self._now(),
                        )
                    await active.prepared.cancel()
                except RunExecutionConflictError:
                    active.cancellation_error = None
                    active.cancellation_done.set()
                    return CanonicalCancellationDisposition.ALREADY_TERMINAL
                except BaseException as exc:
                    active.cancellation_error = exc
                finally:
                    active.cancellation_done.set()
        return (
            CanonicalCancellationDisposition.REQUESTED
            if active.cancellation_error is None
            else CanonicalCancellationDisposition.INTERRUPTED
        )

    async def recover_expired(self, *, occurred_at: datetime | None = None) -> CanonicalRecoveryResult:
        now = self._timestamp(occurred_at or self._now())
        expired = interrupted = 0
        async with self._database_lock:
            probe = self._probe_authority()
            attempts = await probe.expired_attempts(occurred_at=now)
            for attempt in attempts:
                authority = self._authority(attempt.selection)
                claim = RunExecutionClaim.from_attempt(attempt)
                if attempt.status == RunAttemptStatus.GRANTED:
                    await authority.expire_grant(claim, occurred_at=now)
                    expired += 1
                else:
                    await WorkshopRunTerminalTransactionCoordinator(authority).interrupt_expired(
                        claim,
                        occurred_at=now,
                    )
                    interrupted += 1
        return CanonicalRecoveryResult(expired, interrupted)

    async def _execute_owned(self, active: _ActiveExecution, run: DurableRun) -> CanonicalExecutionResult:
        try:
            async with self._database_lock:
                prepared = await self._preparation.prepare(run.run_id)
            authority = self._authority(prepared.selection)
            now = self._now()
            async with self._database_lock:
                granted = await authority.grant(
                    run.run_id,
                    owner_id=RunExecutionOwnerId.new(),
                    occurred_at=now,
                    lease_expires_at=now + self._lease_duration,
                )
            active.prepared = prepared
            active.authority = authority
            active.claim = granted.claim
            active.ready.set()

            if active.cancellation_requested:
                return await self._settle_requested_cancellation(active)

            prepared.validate_current()
            async with self._database_lock:
                started = await authority.start(granted.claim, occurred_at=self._now())
            active.claim = started.claim
            active.started = True

            response = await self._consume_with_renewal(active, prepared)
            if active.cancellation_requested:
                return await self._settle_requested_cancellation(active)
            terminal = WorkshopRunTerminalTransactionCoordinator(authority)
            async with self._database_lock:
                active.settling = True
                if response is None or (response.success and not response.text.strip()):
                    settled = await terminal.fail(
                        active.claim,
                        failure_code=TerminalFailureCode.NO_RESPONSE,
                        occurred_at=self._now(),
                    )
                    disposition = CanonicalExecutionDisposition.FAILED
                elif response.success:
                    settled = await terminal.complete(
                        active.claim,
                        body=response.text,
                        occurred_at=self._now(),
                    )
                    disposition = CanonicalExecutionDisposition.COMPLETED
                else:
                    settled = await terminal.fail(
                        active.claim,
                        failure_code=_FAILURE_CODE_BY_KIND.get(
                            response.failure_kind or AgentFailureKind.UNKNOWN,
                            TerminalFailureCode.UNKNOWN,
                        ),
                        occurred_at=self._now(),
                    )
                    disposition = CanonicalExecutionDisposition.FAILED
            return CanonicalExecutionResult(disposition, settled.execution.run, settled)
        except Exception:
            if active.cancellation_requested and active.claim is not None:
                return await self._settle_requested_cancellation(active)
            if active.settling:
                raise
            if active.started and active.authority is not None and active.claim is not None:
                async with self._database_lock:
                    active.settling = True
                    settled = await WorkshopRunTerminalTransactionCoordinator(active.authority).fail(
                        active.claim,
                        failure_code=TerminalFailureCode.EXECUTION_INTERRUPTED,
                        occurred_at=self._now(),
                    )
                return CanonicalExecutionResult(CanonicalExecutionDisposition.FAILED, settled.execution.run, settled)
            return CanonicalExecutionResult(
                CanonicalExecutionDisposition.PREPARATION_DEFERRED, await self._run(run.run_id)
            )

    async def _settle_requested_cancellation(self, active: _ActiveExecution) -> CanonicalExecutionResult:
        await active.cancellation_done.wait()
        assert active.authority is not None and active.claim is not None
        if active.cancellation_error is not None:
            if active.started:
                async with self._database_lock:
                    active.settling = True
                    result = await WorkshopRunTerminalTransactionCoordinator(active.authority).fail(
                        active.claim,
                        failure_code=TerminalFailureCode.EXECUTION_INTERRUPTED,
                        occurred_at=self._now(),
                    )
                return CanonicalExecutionResult(CanonicalExecutionDisposition.FAILED, result.execution.run, result)
            return CanonicalExecutionResult(
                CanonicalExecutionDisposition.PREPARATION_DEFERRED,
                await self._run(active.run_id),
            )
        async with self._database_lock:
            active.settling = True
            result = await WorkshopRunTerminalTransactionCoordinator(active.authority).confirm_cancellation(
                active.claim,
                occurred_at=self._now(),
            )
        return CanonicalExecutionResult(CanonicalExecutionDisposition.CANCELLED, result.execution.run, result)

    async def _consume(self, prepared: PreparedWorkshopExecution) -> AgentResponse | None:
        prompt = await self._prompt(prepared.run)
        response: AgentResponse | None = None
        async for event in prepared.stream(prompt):
            if event.done:
                response = event.response
                break
        return response

    async def _consume_with_renewal(
        self,
        active: _ActiveExecution,
        prepared: PreparedWorkshopExecution,
    ) -> AgentResponse | None:
        renewal = asyncio.create_task(self._renew_while_running(active))
        try:
            return await self._consume(prepared)
        finally:
            active.renewal_stop.set()
            await renewal

    async def _renew_while_running(self, active: _ActiveExecution) -> None:
        interval = self._lease_duration.total_seconds() / 2
        while True:
            try:
                await asyncio.wait_for(active.renewal_stop.wait(), timeout=interval)
                return
            except TimeoutError:
                pass
            async with active.claim_lock:
                if active.settling or active.claim is None or active.authority is None:
                    return
                now = self._now()
                async with self._database_lock:
                    renewed = await active.authority.renew(
                        active.claim,
                        occurred_at=now,
                        lease_expires_at=now + self._lease_duration,
                    )
                active.claim = renewed.claim

    async def _prompt(self, run: DurableRun) -> str:
        async with (
            self._database_lock,
            self._store.connection.execute(
                "SELECT body FROM messages WHERE id = ? AND channel_id = ? AND author_principal_id = ?",
                (run.inbound_message_id, run.channel_id, run.requested_by_principal_id),
            ) as cursor,
        ):
            row = await cursor.fetchone()
        if row is None:
            raise RunExecutionConflictError("Durable run no longer resolves its canonical prompt")
        return str(row[0])

    async def _run(self, run_id: RunId) -> DurableRun:
        async with self._database_lock:
            return await WorkshopRunLifecycle(self._store).state(run_id)

    async def _replay_disposition(self, run: DurableRun) -> CanonicalExecutionDisposition | None:
        if run.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
            return CanonicalExecutionDisposition.TERMINAL_REPLAY
        if run.cancellation_requested_at is not None:
            return CanonicalExecutionDisposition.CANCELLATION_PENDING_REPLAY
        authority = self._probe_authority()
        async with self._database_lock:
            attempt = await authority.active_attempt(run.run_id)
        return CanonicalExecutionDisposition.ACTIVE_REPLAY if attempt is not None else None

    async def _lane(self, channel_id: ChannelId, agent_id: AgentId) -> asyncio.Lock:
        key = (channel_id, agent_id)
        async with self._map_lock:
            return self._lanes.setdefault(key, asyncio.Lock())

    def _authority(self, selection: RunExecutionSelection) -> WorkshopRunExecutionAuthority:
        return WorkshopRunExecutionAuthority(
            self._store,
            selection_resolver=lambda _run: selection,
            registered_backend_ids=self._registered_backend_ids,
        )

    def _probe_authority(self) -> WorkshopRunExecutionAuthority:
        backend = min(self._registered_backend_ids)
        return self._authority(RunExecutionSelection(backend, "coordinator-probe"))

    def _now(self) -> datetime:
        return self._timestamp(self._clock())

    @staticmethod
    def _timestamp(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return value.astimezone(UTC)
