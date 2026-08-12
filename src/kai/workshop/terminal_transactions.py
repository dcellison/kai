"""Atomic, production-unused terminal transactions for Workshop runs."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum

import aiosqlite

from kai.workshop.domain import MessageId
from kai.workshop.outbound import (
    OutboundMessage,
    OutboundStreamingFinalizationResult,
    record_outbound_message_with_streaming_finalization_in_transaction,
)
from kai.workshop.run_execution_authority import (
    RunExecutionClaim,
    RunExecutionResult,
    WorkshopRunExecutionAuthority,
)
from kai.workshop.run_lifecycle import WorkshopRunLifecycle


class TerminalTransactionError(RuntimeError):
    """Base error for atomic Workshop terminal settlement."""


class TerminalTransactionStateConflictError(TerminalTransactionError):
    """Visible outcome and run settlement do not share one prior state."""


class TerminalTransactionCommitUncertainError(TerminalTransactionError):
    """A deterministic retry could not resolve a possible committed outcome."""


class _PossibleCommitError(RuntimeError):
    """SQLite could not confirm whether the transaction commit completed."""


class TerminalOutcome(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TerminalFailureCode(StrEnum):
    """Bounded backend-neutral failure facts allowed into Workshop history."""

    AUTHENTICATION_EXPIRED = "authentication_expired"
    AUTHENTICATION_REQUIRED = "authentication_required"
    QUOTA_EXHAUSTED = "quota_exhausted"
    MODEL_UNAVAILABLE = "model_unavailable"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    TRANSIENT = "transient"
    BACKEND_CRASHED = "backend_crashed"
    NO_RESPONSE = "no_response"
    EXECUTION_INTERRUPTED = "execution_interrupted"
    UNKNOWN = "unknown"


_FAILURE_MESSAGES = {
    TerminalFailureCode.AUTHENTICATION_EXPIRED: (
        "Authentication for the configured agent has expired. Kai did not complete this request."
    ),
    TerminalFailureCode.AUTHENTICATION_REQUIRED: (
        "Authentication for the configured agent is required. Kai did not complete this request."
    ),
    TerminalFailureCode.QUOTA_EXHAUSTED: (
        "The configured agent reported that its usage allowance is exhausted. Kai did not complete this request."
    ),
    TerminalFailureCode.MODEL_UNAVAILABLE: ("The configured model is unavailable. Kai did not complete this request."),
    TerminalFailureCode.PROVIDER_UNAVAILABLE: (
        "The configured agent provider is unavailable. Kai did not complete this request."
    ),
    TerminalFailureCode.TRANSIENT: (
        "The configured agent reported a temporary failure. Kai did not complete this request."
    ),
    TerminalFailureCode.BACKEND_CRASHED: (
        "The configured agent stopped unexpectedly. Kai did not complete this request."
    ),
    TerminalFailureCode.NO_RESPONSE: "The configured agent returned no response. Kai did not complete this request.",
    TerminalFailureCode.EXECUTION_INTERRUPTED: (
        "Kai was interrupted while the configured agent was working. This request was not retried."
    ),
    TerminalFailureCode.UNKNOWN: "The configured agent could not complete this request.",
}

_CANCELLATION_MESSAGE = "This request was cancelled."


@dataclass(frozen=True, slots=True)
class TerminalTransactionResult:
    outcome: TerminalOutcome
    body: str
    finalization: OutboundStreamingFinalizationResult
    execution: RunExecutionResult
    changed: bool


class WorkshopRunTerminalTransactionCoordinator:
    """Commit a visible terminal outcome and fenced settlement together.

    The caller supplies a fenced execution claim and, for successful work,
    only the assistant's final text. Routing, delivery authority, operation
    plan, cancellation code, and terminal failure text are protected inputs.
    This service is not constructed by production code.
    """

    def __init__(self, authority: WorkshopRunExecutionAuthority) -> None:
        if not isinstance(authority, WorkshopRunExecutionAuthority):
            raise TypeError("authority must be a WorkshopRunExecutionAuthority")
        self._authority = authority
        self._store = authority.event_store

    async def complete(
        self,
        claim: RunExecutionClaim,
        *,
        body: str,
        occurred_at: datetime,
    ) -> TerminalTransactionResult:
        return await self._resolve_possible_commit(
            lambda: self._settle_once(
                claim,
                outcome=TerminalOutcome.COMPLETED,
                body=body,
                occurred_at=occurred_at,
            )
        )

    async def fail(
        self,
        claim: RunExecutionClaim,
        *,
        failure_code: TerminalFailureCode,
        occurred_at: datetime,
    ) -> TerminalTransactionResult:
        if not isinstance(failure_code, TerminalFailureCode):
            raise ValueError("failure_code must be a TerminalFailureCode")
        return await self._resolve_possible_commit(
            lambda: self._settle_once(
                claim,
                outcome=TerminalOutcome.FAILED,
                body=_FAILURE_MESSAGES[failure_code],
                occurred_at=occurred_at,
                failure_code=failure_code,
            )
        )

    async def confirm_cancellation(
        self,
        claim: RunExecutionClaim,
        *,
        occurred_at: datetime,
    ) -> TerminalTransactionResult:
        return await self._resolve_possible_commit(
            lambda: self._settle_once(
                claim,
                outcome=TerminalOutcome.CANCELLED,
                body=_CANCELLATION_MESSAGE,
                occurred_at=occurred_at,
            )
        )

    async def interrupt_expired(
        self,
        claim: RunExecutionClaim,
        *,
        occurred_at: datetime,
    ) -> TerminalTransactionResult:
        """Atomically expose and settle an expired post-dispatch interruption."""
        return await self._resolve_possible_commit(
            lambda: self._settle_once(
                claim,
                outcome=TerminalOutcome.FAILED,
                body=_FAILURE_MESSAGES[TerminalFailureCode.EXECUTION_INTERRUPTED],
                occurred_at=occurred_at,
                failure_code=TerminalFailureCode.EXECUTION_INTERRUPTED,
                expired_interruption=True,
            )
        )

    async def _resolve_possible_commit(
        self,
        operation: Callable[[], Awaitable[TerminalTransactionResult]],
    ) -> TerminalTransactionResult:
        try:
            return await operation()
        except _PossibleCommitError:
            try:
                return await operation()
            except Exception as resolution_error:
                raise TerminalTransactionCommitUncertainError(
                    "Could not determine whether the Workshop terminal transaction committed"
                ) from resolution_error

    async def _settle_once(
        self,
        claim: RunExecutionClaim,
        *,
        outcome: TerminalOutcome,
        body: str,
        occurred_at: datetime,
        failure_code: TerminalFailureCode | None = None,
        expired_interruption: bool = False,
    ) -> TerminalTransactionResult:
        if not isinstance(claim, RunExecutionClaim):
            raise ValueError("claim must be a RunExecutionClaim")
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            run = await WorkshopRunLifecycle(self._store).state(claim.run_id)
            finalization = await record_outbound_message_with_streaming_finalization_in_transaction(
                self._store,
                OutboundMessage(
                    in_reply_to_message_id=run.inbound_message_id,
                    body=body,
                    occurred_at=occurred_at,
                ),
            )
            message_id = finalization.message.event.envelope.aggregate_id
            if not isinstance(message_id, MessageId):
                raise TerminalTransactionStateConflictError("Canonical terminal outcome did not identify a message")
            if outcome == TerminalOutcome.COMPLETED:
                execution = await self._authority.complete_in_transaction(
                    claim,
                    result_message_id=message_id,
                    occurred_at=occurred_at,
                )
            elif outcome == TerminalOutcome.FAILED:
                assert failure_code is not None
                if expired_interruption:
                    execution = await self._authority.interrupt_expired_in_transaction(
                        claim,
                        occurred_at=occurred_at,
                    )
                else:
                    execution = await self._authority.fail_in_transaction(
                        claim,
                        failure_code=failure_code.value,
                        occurred_at=occurred_at,
                    )
            else:
                execution = await self._authority.confirm_cancellation_in_transaction(
                    claim,
                    occurred_at=occurred_at,
                )

            prior_states = {
                finalization.message.inserted,
                finalization.delivery.inserted,
                finalization.plan.inserted,
                execution.changed,
            }
            if len(prior_states) != 1:
                raise TerminalTransactionStateConflictError(
                    "Canonical outcome, delivery plan, and run settlement did not share one prior state"
                )
            changed = execution.changed
            try:
                await connection.commit()
            except aiosqlite.Error as commit_error:
                raise _PossibleCommitError("Workshop terminal commit result was unavailable") from commit_error
            return TerminalTransactionResult(
                outcome=outcome,
                body=body,
                finalization=finalization,
                execution=execution,
                changed=changed,
            )
        except Exception:
            await connection.rollback()
            raise
