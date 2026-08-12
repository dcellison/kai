"""Production-unused durable delivery work for Kai Workshop."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Literal, cast

import aiosqlite

from kai.workshop.domain import (
    ChannelBindingId,
    ChannelId,
    DeliveryAttemptId,
    DeliveryAuthorityEpochId,
    DeliveryId,
    EventEnvelope,
    EventId,
    MessageId,
    PrincipalId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.store import WorkshopEventStore

_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_WORKER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_MAX_LEASE = timedelta(minutes=5)
_MAX_ATTEMPTS = 20
_MAX_MINIMUM_RETRY_DELAY = timedelta(days=1)
_RETRY_BASE_SECONDS = 5
_RETRY_MAX_SECONDS = 300

DeliveryPurpose = Literal["conversation_reply", "qualification"]
CONVERSATION_REPLY_PURPOSE: DeliveryPurpose = "conversation_reply"
QUALIFICATION_PURPOSE: DeliveryPurpose = "qualification"
_DELIVERY_PURPOSES = frozenset({CONVERSATION_REPLY_PURPOSE, QUALIFICATION_PURPOSE})

DeliveryExecutionContract = Literal["send_fragments", "streaming_finalization"]
SEND_FRAGMENTS_CONTRACT: DeliveryExecutionContract = "send_fragments"
STREAMING_FINALIZATION_CONTRACT: DeliveryExecutionContract = "streaming_finalization"
_DELIVERY_EXECUTION_CONTRACTS = frozenset({SEND_FRAGMENTS_CONTRACT, STREAMING_FINALIZATION_CONTRACT})


class DeliveryOutboxError(RuntimeError):
    """Base class for fail-closed delivery outbox errors."""


class DeliveryTargetNotFoundError(DeliveryOutboxError):
    """A canonical message and channel binding did not resolve together."""


class DeliveryRequestConflictError(DeliveryOutboxError):
    """A delivery identity was reused with different request semantics."""


class StaleDeliveryLeaseError(DeliveryOutboxError):
    """A worker tried to complete work after losing its lease."""


class IncompleteDeliveryFragmentsError(DeliveryOutboxError):
    """A worker tried to complete delivery before every fragment was sent."""


class UnsettledDeliveryFragmentError(DeliveryOutboxError):
    """A worker tried to settle delivery while a fragment was still in flight."""


@dataclass(frozen=True, slots=True)
class DeliveryRequest:
    message_id: MessageId
    channel_binding_id: ChannelBindingId
    mode: str
    purpose: DeliveryPurpose
    occurred_at: datetime
    execution_contract: DeliveryExecutionContract = SEND_FRAGMENTS_CONTRACT
    authority_epoch_id: DeliveryAuthorityEpochId | None = None
    max_attempts: int = 5

    def __post_init__(self) -> None:
        if not isinstance(self.message_id, MessageId):
            raise ValueError("message_id must be a MessageId")
        if not isinstance(self.channel_binding_id, ChannelBindingId):
            raise ValueError("channel_binding_id must be a ChannelBindingId")
        if not _IDENTIFIER_PATTERN.fullmatch(self.mode):
            raise ValueError("mode must be a lowercase identifier")
        _validate_purpose(self.purpose)
        _validate_execution_contract(self.execution_contract)
        requires_epoch = (
            self.purpose == CONVERSATION_REPLY_PURPOSE and self.execution_contract == STREAMING_FINALIZATION_CONTRACT
        )
        if requires_epoch and not isinstance(self.authority_epoch_id, DeliveryAuthorityEpochId):
            raise ValueError("streaming conversation delivery requires a DeliveryAuthorityEpochId")
        if not requires_epoch and self.authority_epoch_id is not None:
            raise ValueError("authority_epoch_id is only valid for streaming conversation delivery")
        if self.occurred_at.tzinfo is None or self.occurred_at.utcoffset() is None:
            raise ValueError("occurred_at must be timezone-aware")
        if (
            not isinstance(self.max_attempts, int)
            or isinstance(self.max_attempts, bool)
            or not 1 <= self.max_attempts <= _MAX_ATTEMPTS
        ):
            raise ValueError(f"max_attempts must be between 1 and {_MAX_ATTEMPTS}")


@dataclass(frozen=True, slots=True)
class DeliveryState:
    delivery_id: DeliveryId
    message_id: MessageId
    channel_id: ChannelId
    channel_binding_id: ChannelBindingId
    transport: str
    mode: str
    purpose: DeliveryPurpose
    execution_contract: DeliveryExecutionContract
    authority_epoch_id: DeliveryAuthorityEpochId | None
    status: str
    max_attempts: int
    attempt_count: int
    available_at: datetime
    last_error_code: str | None
    requested_event_position: int
    completed_at: datetime | None


@dataclass(frozen=True, slots=True)
class DeliveryRequestResult:
    delivery: DeliveryState
    inserted: bool


@dataclass(frozen=True, slots=True)
class DeliveryClaim:
    delivery_id: DeliveryId
    attempt_id: DeliveryAttemptId
    attempt_number: int
    workshop_id: WorkshopId
    channel_id: ChannelId
    channel_binding_id: ChannelBindingId
    message_id: MessageId
    transport: str
    external_channel_id: str
    mode: str
    purpose: DeliveryPurpose
    execution_contract: DeliveryExecutionContract
    authority_epoch_id: DeliveryAuthorityEpochId | None
    body: str
    lease_expires_at: datetime


@dataclass(frozen=True, slots=True)
class DeliveryRecoveryResult:
    requeued: int
    failed: int


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _optional_timestamp(value: object) -> datetime | None:
    return _parse_timestamp(str(value)) if value is not None else None


def _validate_worker_id(worker_id: str) -> None:
    if not _WORKER_PATTERN.fullmatch(worker_id):
        raise ValueError("worker_id must be a bounded process identifier")


def _validate_error_code(error_code: str) -> None:
    if not _IDENTIFIER_PATTERN.fullmatch(error_code):
        raise ValueError("error_code must be a lowercase identifier")


def _validate_purpose(purpose: str) -> None:
    if purpose not in _DELIVERY_PURPOSES:
        raise ValueError("purpose must be conversation_reply or qualification")


def _validate_purposes(purposes: tuple[DeliveryPurpose, ...]) -> None:
    if not purposes or len(set(purposes)) != len(purposes):
        raise ValueError("purposes must contain unique values")
    for purpose in purposes:
        _validate_purpose(purpose)


def _validate_execution_contract(execution_contract: str) -> None:
    if execution_contract not in _DELIVERY_EXECUTION_CONTRACTS:
        raise ValueError("execution_contract must be send_fragments or streaming_finalization")


def _validate_execution_contracts(contracts: tuple[DeliveryExecutionContract, ...]) -> None:
    if not contracts or len(set(contracts)) != len(contracts):
        raise ValueError("execution_contracts must contain unique values")
    for contract in contracts:
        _validate_execution_contract(contract)


def _validate_authority_scope(
    purposes: tuple[DeliveryPurpose, ...],
    contracts: tuple[DeliveryExecutionContract, ...],
    authority_epoch_id: DeliveryAuthorityEpochId | None,
) -> None:
    requires_epoch = CONVERSATION_REPLY_PURPOSE in purposes and STREAMING_FINALIZATION_CONTRACT in contracts
    if requires_epoch and not isinstance(authority_epoch_id, DeliveryAuthorityEpochId):
        raise ValueError("streaming conversation work requires an exact authority epoch")
    if not requires_epoch and authority_epoch_id is not None:
        raise ValueError("authority_epoch_id is only valid for streaming conversation work")


def _delivery_purpose(value: object) -> DeliveryPurpose:
    purpose = str(value)
    _validate_purpose(purpose)
    return cast(DeliveryPurpose, purpose)


def _delivery_execution_contract(value: object) -> DeliveryExecutionContract:
    contract = str(value)
    _validate_execution_contract(contract)
    return cast(DeliveryExecutionContract, contract)


def _retry_delay(attempt_number: int) -> timedelta:
    seconds = min(_RETRY_BASE_SECONDS * (2 ** (attempt_number - 1)), _RETRY_MAX_SECONDS)
    return timedelta(seconds=seconds)


class WorkshopDeliveryOutbox:
    """Durable delivery state with lease-based, at-least-once work claims.

    Production private-text finalization and explicit qualification use this
    class. Other Telegram paths retain their existing delivery behavior until
    separately bounded cutovers move them.
    """

    def __init__(
        self,
        store: WorkshopEventStore,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store
        self._clock = clock or (lambda: datetime.now(UTC))

    async def request_delivery(self, request: DeliveryRequest) -> DeliveryRequestResult:
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            result = await self.request_delivery_in_transaction(request)
            await connection.commit()
            return result
        except Exception:
            await connection.rollback()
            raise

    async def request_delivery_in_transaction(self, request: DeliveryRequest) -> DeliveryRequestResult:
        """Persist one delivery request without committing an existing transaction."""
        connection = self._store.connection
        if not connection.in_transaction:
            raise RuntimeError("request_delivery_in_transaction requires an active transaction")

        resolved = await self._resolve_target(request)
        workshop_id, channel_id, author_principal_id, transport = resolved
        if request.authority_epoch_id is not None:
            async with connection.execute(
                "SELECT COUNT(*) FROM delivery_authority_epochs "
                "WHERE id = ? AND lane = 'telegram_conversation_streaming_finalization' "
                "AND status = 'active'",
                (request.authority_epoch_id,),
            ) as cursor:
                authority_row = await cursor.fetchone()
            if authority_row is None or int(authority_row[0]) != 1:
                raise DeliveryRequestConflictError("Delivery authority epoch is not active")
        delivery_id = DeliveryId.derived(
            workshop_id,
            f"delivery:{request.message_id}:{request.channel_binding_id}:{request.mode}",
        )

        existing = await self._state_for_identity(
            request.message_id,
            request.channel_binding_id,
            request.mode,
        )
        if existing is not None:
            if (
                existing.delivery_id != delivery_id
                or existing.channel_binding_id != request.channel_binding_id
                or existing.transport != transport
                or existing.max_attempts != request.max_attempts
                or existing.purpose != request.purpose
                or existing.execution_contract != request.execution_contract
                or existing.authority_epoch_id != request.authority_epoch_id
            ):
                raise DeliveryRequestConflictError("Delivery request identity has different semantics")
            return DeliveryRequestResult(delivery=existing, inserted=False)

        occurred_at = request.occurred_at.astimezone(UTC)
        streaming_finalization = request.execution_contract == STREAMING_FINALIZATION_CONTRACT
        event_version = 4 if streaming_finalization else 2
        event_identity_version = "v4" if streaming_finalization else "v2"
        payload: dict[str, object] = {
            "message_id": request.message_id,
            "channel_id": channel_id,
            "channel_binding_id": request.channel_binding_id,
            "transport": transport,
            "mode": request.mode,
            "purpose": request.purpose,
            "max_attempts": request.max_attempts,
        }
        if streaming_finalization:
            payload["execution_contract"] = request.execution_contract
            payload["authority_epoch_id"] = request.authority_epoch_id
        event = EventEnvelope.create(
            event_id=EventId.derived(
                workshop_id,
                f"delivery-request-event:{event_identity_version}:"
                f"{request.message_id}:{request.channel_binding_id}:{request.mode}",
            ),
            event_type=WorkshopEventType.DELIVERY_REQUESTED,
            event_version=event_version,
            workshop_id=workshop_id,
            aggregate_type="delivery",
            aggregate_id=delivery_id,
            actor_principal_id=author_principal_id,
            occurred_at=occurred_at,
            idempotency_key=(
                f"workshop-delivery-request:{event_identity_version}:"
                f"{request.message_id}:{request.channel_binding_id}:{request.mode}"
            ),
            payload=payload,
            metadata={"source": "delivery_outbox"},
        )
        appended = await self._store.append_in_transaction(event)
        if not appended.inserted:
            raise DeliveryRequestConflictError("Delivery request event exists without outbox state")

        timestamp = _format_timestamp(occurred_at)
        await connection.execute(
            "INSERT INTO delivery_outbox "
            "(id, workshop_id, channel_id, channel_binding_id, message_id, transport, mode, purpose, "
            "execution_contract, authority_epoch_id, "
            "status, max_attempts, attempt_count, available_at, requested_event_position, "
            "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?, 0, ?, ?, ?, ?)",
            (
                delivery_id,
                workshop_id,
                channel_id,
                request.channel_binding_id,
                request.message_id,
                transport,
                request.mode,
                request.purpose,
                request.execution_contract,
                request.authority_epoch_id,
                request.max_attempts,
                timestamp,
                appended.event.position,
                timestamp,
                timestamp,
            ),
        )
        state = await self._state_by_id(delivery_id)
        return DeliveryRequestResult(delivery=state, inserted=True)

    async def claim_next(
        self,
        worker_id: str,
        *,
        purposes: tuple[DeliveryPurpose, ...],
        execution_contracts: tuple[DeliveryExecutionContract, ...] = (SEND_FRAGMENTS_CONTRACT,),
        lease_duration: timedelta = timedelta(seconds=30),
        transport: str | None = None,
        modes: tuple[str, ...] | None = None,
        delivery_id: DeliveryId | None = None,
        authority_epoch_id: DeliveryAuthorityEpochId | None = None,
    ) -> DeliveryClaim | None:
        _validate_worker_id(worker_id)
        _validate_purposes(purposes)
        _validate_execution_contracts(execution_contracts)
        _validate_authority_scope(purposes, execution_contracts, authority_epoch_id)
        if lease_duration <= timedelta(0) or lease_duration > _MAX_LEASE:
            raise ValueError("lease_duration must be positive and at most five minutes")
        if transport is not None and not _IDENTIFIER_PATTERN.fullmatch(transport):
            raise ValueError("transport must be a lowercase identifier when supplied")
        if modes is not None:
            if not modes or len(set(modes)) != len(modes):
                raise ValueError("modes must contain unique values when supplied")
            for mode in modes:
                if not _IDENTIFIER_PATTERN.fullmatch(mode):
                    raise ValueError("modes must contain lowercase identifiers")
        if delivery_id is not None and not isinstance(delivery_id, DeliveryId):
            raise ValueError("delivery_id must be a DeliveryId when supplied")

        connection = self._store.connection
        now = self._now()
        try:
            await connection.execute("BEGIN IMMEDIATE")
            await self._recover_expired_in_transaction(
                now,
                purposes=purposes,
                execution_contracts=execution_contracts,
                delivery_id=delivery_id,
                authority_epoch_id=authority_epoch_id,
            )
            now_text = _format_timestamp(now)
            filters = [
                "o.status IN ('pending', 'retry_wait')",
                "o.available_at <= ?",
                "o.attempt_count < o.max_attempts",
                "NOT EXISTS ("
                "SELECT 1 FROM delivery_outbox predecessor "
                "WHERE predecessor.channel_binding_id = o.channel_binding_id "
                "AND predecessor.purpose = o.purpose "
                "AND predecessor.authority_epoch_id IS o.authority_epoch_id "
                "AND predecessor.requested_event_position < o.requested_event_position "
                "AND predecessor.status NOT IN ('succeeded', 'failed')"
                ")",
            ]
            parameters: list[object] = [now_text]
            purpose_placeholders = ", ".join("?" for _ in purposes)
            filters.append(f"o.purpose IN ({purpose_placeholders})")
            parameters.extend(purposes)
            contract_placeholders = ", ".join("?" for _ in execution_contracts)
            filters.append(f"o.execution_contract IN ({contract_placeholders})")
            parameters.extend(execution_contracts)
            if authority_epoch_id is not None:
                filters.append("o.authority_epoch_id = ?")
                parameters.append(authority_epoch_id)
                filters.append(
                    "EXISTS (SELECT 1 FROM delivery_authority_epochs ae "
                    "WHERE ae.id = o.authority_epoch_id "
                    "AND ae.lane = 'telegram_conversation_streaming_finalization' "
                    "AND ae.status = 'active')"
                )
            if delivery_id is not None:
                filters.append("o.id = ?")
                parameters.append(delivery_id)
            if transport is not None:
                filters.append("o.transport = ?")
                parameters.append(transport)
            if modes is not None:
                placeholders = ", ".join("?" for _ in modes)
                filters.append(f"o.mode IN ({placeholders})")
                parameters.extend(modes)
            async with connection.execute(
                "SELECT o.id, o.workshop_id, o.channel_id, o.channel_binding_id, o.message_id, "
                "o.transport, o.mode, o.purpose, o.execution_contract, o.authority_epoch_id, o.attempt_count, "
                "m.body, cb.external_channel_id "
                "FROM delivery_outbox o "
                "JOIN messages m ON m.id = o.message_id AND m.channel_id = o.channel_id "
                "JOIN channel_bindings cb ON cb.id = o.channel_binding_id "
                "AND cb.channel_id = o.channel_id AND cb.transport = o.transport "
                f"WHERE {' AND '.join(filters)} "
                "ORDER BY o.requested_event_position LIMIT 1",
                parameters,
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                await connection.commit()
                return None

            delivery_id = DeliveryId(str(row[0]))
            attempt_number = int(row[10]) + 1
            attempt_id = DeliveryAttemptId.new()
            expires_at = now + lease_duration
            expires_text = _format_timestamp(expires_at)
            cursor = await connection.execute(
                "UPDATE delivery_outbox SET status = 'leased', attempt_count = ?, lease_id = ?, "
                "lease_owner = ?, lease_expires_at = ?, updated_at = ? "
                "WHERE id = ? AND status IN ('pending', 'retry_wait') AND available_at <= ?",
                (
                    attempt_number,
                    attempt_id,
                    worker_id,
                    expires_text,
                    now_text,
                    delivery_id,
                    now_text,
                ),
            )
            if cursor.rowcount != 1:
                raise DeliveryOutboxError("Delivery claim lost its serialized update")
            await connection.execute(
                "INSERT INTO delivery_attempts "
                "(id, delivery_id, attempt_number, worker_id, started_at, lease_expires_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (attempt_id, delivery_id, attempt_number, worker_id, now_text, expires_text),
            )
            await connection.commit()
            return DeliveryClaim(
                delivery_id=delivery_id,
                attempt_id=attempt_id,
                attempt_number=attempt_number,
                workshop_id=WorkshopId(str(row[1])),
                channel_id=ChannelId(str(row[2])),
                channel_binding_id=ChannelBindingId(str(row[3])),
                message_id=MessageId(str(row[4])),
                transport=str(row[5]),
                mode=str(row[6]),
                purpose=_delivery_purpose(row[7]),
                execution_contract=_delivery_execution_contract(row[8]),
                authority_epoch_id=(DeliveryAuthorityEpochId(str(row[9])) if row[9] is not None else None),
                body=str(row[11]),
                external_channel_id=str(row[12]),
                lease_expires_at=expires_at,
            )
        except Exception:
            await connection.rollback()
            raise

    async def mark_succeeded(self, claim: DeliveryClaim) -> DeliveryState:
        return await self._complete(claim, succeeded=True, retryable=False, error_code=None)

    async def mark_failed(
        self,
        claim: DeliveryClaim,
        *,
        retryable: bool,
        error_code: str,
        minimum_retry_delay: timedelta | None = None,
    ) -> DeliveryState:
        if not isinstance(retryable, bool):
            raise ValueError("retryable must be a boolean")
        _validate_error_code(error_code)
        if minimum_retry_delay is not None and (
            minimum_retry_delay <= timedelta(0) or minimum_retry_delay > _MAX_MINIMUM_RETRY_DELAY
        ):
            raise ValueError("minimum_retry_delay must be positive and at most one day")
        return await self._complete(
            claim,
            succeeded=False,
            retryable=retryable,
            error_code=error_code,
            minimum_retry_delay=minimum_retry_delay,
        )

    async def recover_expired_leases(
        self,
        *,
        purposes: tuple[DeliveryPurpose, ...],
        execution_contracts: tuple[DeliveryExecutionContract, ...] = (SEND_FRAGMENTS_CONTRACT,),
        authority_epoch_id: DeliveryAuthorityEpochId | None = None,
    ) -> DeliveryRecoveryResult:
        _validate_purposes(purposes)
        _validate_execution_contracts(execution_contracts)
        _validate_authority_scope(purposes, execution_contracts, authority_epoch_id)
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            result = await self._recover_expired_in_transaction(
                self._now(),
                purposes=purposes,
                execution_contracts=execution_contracts,
                authority_epoch_id=authority_epoch_id,
            )
            await connection.commit()
            return result
        except Exception:
            await connection.rollback()
            raise

    async def state(self, delivery_id: DeliveryId) -> DeliveryState:
        if not isinstance(delivery_id, DeliveryId):
            raise ValueError("delivery_id must be a DeliveryId")
        return await self._state_by_id(delivery_id)

    async def _complete(
        self,
        claim: DeliveryClaim,
        *,
        succeeded: bool,
        retryable: bool,
        error_code: str | None,
        minimum_retry_delay: timedelta | None = None,
    ) -> DeliveryState:
        if not isinstance(claim, DeliveryClaim):
            raise ValueError("claim must be a DeliveryClaim")
        connection = self._store.connection
        now = self._now()
        now_text = _format_timestamp(now)
        desired_outcome = "succeeded" if succeeded else None
        try:
            await connection.execute("BEGIN IMMEDIATE")
            async with connection.execute(
                "SELECT o.status, o.attempt_count, o.max_attempts, o.lease_id, "
                "o.lease_expires_at, a.outcome, a.completed_at, a.error_code, "
                "o.workshop_id, o.channel_id, o.channel_binding_id, o.message_id, o.transport, o.mode, "
                "o.authority_epoch_id "
                "FROM delivery_outbox o JOIN delivery_attempts a "
                "ON a.delivery_id = o.id AND a.id = ? WHERE o.id = ?",
                (claim.attempt_id, claim.delivery_id),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                raise StaleDeliveryLeaseError("Delivery attempt does not exist")
            persisted_epoch = DeliveryAuthorityEpochId(str(row[14])) if row[14] is not None else None
            if persisted_epoch != claim.authority_epoch_id:
                raise StaleDeliveryLeaseError("Delivery attempt does not belong to the claimed authority epoch")
            if persisted_epoch is not None:
                async with connection.execute(
                    "SELECT COUNT(*) FROM delivery_authority_epochs "
                    "WHERE id = ? AND lane = 'telegram_conversation_streaming_finalization' "
                    "AND status = 'active'",
                    (persisted_epoch,),
                ) as cursor:
                    active_epoch_row = await cursor.fetchone()
                if active_epoch_row is None or int(active_epoch_row[0]) != 1:
                    raise StaleDeliveryLeaseError("Delivery authority epoch is not active")
            if row[5] is not None:
                if succeeded and row[5] == desired_outcome and row[0] == "succeeded":
                    if row[6] is None:
                        raise DeliveryOutboxError("Completed delivery attempt has no completion timestamp")
                    await self._append_outcome_event(
                        delivery_id=claim.delivery_id,
                        workshop_id=WorkshopId(str(row[8])),
                        channel_id=ChannelId(str(row[9])),
                        channel_binding_id=ChannelBindingId(str(row[10])),
                        message_id=MessageId(str(row[11])),
                        transport=str(row[12]),
                        mode=str(row[13]),
                        authority_epoch_id=persisted_epoch,
                        attempt_number=int(row[1]),
                        status="succeeded",
                        error_code=None,
                        occurred_at=_parse_timestamp(str(row[6])),
                    )
                    state = await self._state_by_id(claim.delivery_id)
                    await connection.commit()
                    return state
                raise StaleDeliveryLeaseError("Delivery attempt was already completed differently")
            if row[0] != "leased" or row[3] != claim.attempt_id:
                raise StaleDeliveryLeaseError("Delivery attempt no longer owns the active lease")
            if row[4] is None or _parse_timestamp(str(row[4])) <= now:
                await self._recover_expired_in_transaction(
                    now,
                    purposes=(claim.purpose,),
                    execution_contracts=(claim.execution_contract,),
                    delivery_id=claim.delivery_id,
                    authority_epoch_id=claim.authority_epoch_id,
                )
                await connection.commit()
                raise StaleDeliveryLeaseError("Delivery attempt lease has expired")

            attempt_count = int(row[1])
            max_attempts = int(row[2])
            async with connection.execute(
                "SELECT COUNT(*), "
                "SUM(CASE WHEN status = 'sent' THEN 0 ELSE 1 END), "
                "SUM(CASE WHEN status = 'sending' THEN 1 ELSE 0 END), "
                "SUM(CASE WHEN status = 'uncertain' THEN 1 ELSE 0 END) "
                "FROM delivery_fragments WHERE delivery_id = ?",
                (claim.delivery_id,),
            ) as cursor:
                fragment_row = await cursor.fetchone()
            if fragment_row is None:
                raise DeliveryOutboxError("Delivery fragment aggregate query returned no row")
            fragment_count = int(fragment_row[0])
            incomplete_fragments = int(fragment_row[1]) if fragment_count > 0 else 0
            sending_fragments = int(fragment_row[2]) if fragment_count > 0 else 0
            uncertain_fragments = int(fragment_row[3]) if fragment_count > 0 else 0
            if sending_fragments > 0 or (retryable and uncertain_fragments > 0):
                raise UnsettledDeliveryFragmentError("Delivery has an in-flight or uncertain fragment")
            if succeeded:
                if incomplete_fragments > 0:
                    raise IncompleteDeliveryFragmentsError("Delivery has incomplete or uncertain fragments")
                outcome = "succeeded"
                status = "succeeded"
                available_at = now
                completed_at: str | None = now_text
            elif retryable and attempt_count < max_attempts:
                outcome = "retry_scheduled"
                status = "retry_wait"
                retry_delay = _retry_delay(attempt_count)
                if minimum_retry_delay is not None:
                    retry_delay = max(retry_delay, minimum_retry_delay)
                available_at = now + retry_delay
                completed_at = None
            else:
                outcome = "failed"
                status = "failed"
                available_at = now
                completed_at = now_text

            await connection.execute(
                "UPDATE delivery_attempts SET completed_at = ?, outcome = ?, error_code = ? "
                "WHERE id = ? AND completed_at IS NULL",
                (now_text, outcome, error_code, claim.attempt_id),
            )
            await connection.execute(
                "UPDATE delivery_outbox SET status = ?, available_at = ?, lease_id = NULL, "
                "lease_owner = NULL, lease_expires_at = NULL, last_error_code = ?, "
                "updated_at = ?, completed_at = ? WHERE id = ? AND lease_id = ?",
                (
                    status,
                    _format_timestamp(available_at),
                    error_code,
                    now_text,
                    completed_at,
                    claim.delivery_id,
                    claim.attempt_id,
                ),
            )
            if status in {"succeeded", "failed"}:
                await self._append_outcome_event(
                    delivery_id=claim.delivery_id,
                    workshop_id=WorkshopId(str(row[8])),
                    channel_id=ChannelId(str(row[9])),
                    channel_binding_id=ChannelBindingId(str(row[10])),
                    message_id=MessageId(str(row[11])),
                    transport=str(row[12]),
                    mode=str(row[13]),
                    authority_epoch_id=persisted_epoch,
                    attempt_number=attempt_count,
                    status=status,
                    error_code=error_code,
                    occurred_at=now,
                )
            state = await self._state_by_id(claim.delivery_id)
            await connection.commit()
            return state
        except Exception:
            await connection.rollback()
            raise

    async def _recover_expired_in_transaction(
        self,
        now: datetime,
        *,
        purposes: tuple[DeliveryPurpose, ...],
        execution_contracts: tuple[DeliveryExecutionContract, ...],
        delivery_id: DeliveryId | None = None,
        authority_epoch_id: DeliveryAuthorityEpochId | None = None,
    ) -> DeliveryRecoveryResult:
        connection = self._store.connection
        now_text = _format_timestamp(now)
        purpose_placeholders = ", ".join("?" for _ in purposes)
        contract_placeholders = ", ".join("?" for _ in execution_contracts)
        identity_filter = " AND id = ?" if delivery_id is not None else ""
        parameters: tuple[object, ...] = (now_text, *purposes, *execution_contracts)
        authority_filter = " AND authority_epoch_id = ?" if authority_epoch_id is not None else ""
        active_authority_filter = (
            " AND EXISTS (SELECT 1 FROM delivery_authority_epochs ae "
            "WHERE ae.id = delivery_outbox.authority_epoch_id "
            "AND ae.lane = 'telegram_conversation_streaming_finalization' "
            "AND ae.status = 'active')"
            if authority_epoch_id is not None
            else ""
        )
        if authority_epoch_id is not None:
            parameters = (*parameters, authority_epoch_id)
        if delivery_id is not None:
            parameters = (*parameters, delivery_id)
        async with connection.execute(
            "SELECT id, lease_id, attempt_count, max_attempts, workshop_id, channel_id, "
            "channel_binding_id, message_id, transport, mode, authority_epoch_id FROM delivery_outbox "
            f"WHERE status = 'leased' AND lease_expires_at <= ? "
            f"AND purpose IN ({purpose_placeholders}) "
            f"AND execution_contract IN ({contract_placeholders}){authority_filter}"
            f"{active_authority_filter}{identity_filter} "
            "ORDER BY requested_event_position",
            parameters,
        ) as cursor:
            rows = await cursor.fetchall()
        requeued = 0
        failed = 0
        for row in rows:
            delivery_id = DeliveryId(str(row[0]))
            attempt_id = DeliveryAttemptId(str(row[1]))
            async with connection.execute(
                "SELECT operation FROM delivery_fragments "
                "WHERE delivery_id = ? AND status IN ('sending', 'uncertain') ORDER BY fragment_index",
                (delivery_id,),
            ) as cursor:
                uncertain_rows = list(await cursor.fetchall())
            external_effect_uncertain = bool(uncertain_rows)
            if external_effect_uncertain:
                await connection.execute(
                    "UPDATE delivery_fragments SET status = 'uncertain', updated_at = ? "
                    "WHERE delivery_id = ? AND status = 'sending'",
                    (now_text, delivery_id),
                )
            terminal = external_effect_uncertain or int(row[2]) >= int(row[3])
            status = "failed" if terminal else "retry_wait"
            completed_at = now_text if terminal else None
            attempt_outcome = "failed" if external_effect_uncertain else "lease_expired"
            uncertain_operations = {str(fragment_row[0]) for fragment_row in uncertain_rows}
            if uncertain_operations == {"edit"}:
                uncertainty_code = "delivery_edit_uncertain"
            elif uncertain_operations == {"send"}:
                uncertainty_code = "delivery_send_uncertain"
            else:
                uncertainty_code = "delivery_effect_uncertain"
            error_code = uncertainty_code if external_effect_uncertain else "lease_expired"
            await connection.execute(
                "UPDATE delivery_attempts SET completed_at = ?, outcome = ?, "
                "error_code = ? WHERE id = ? AND completed_at IS NULL",
                (now_text, attempt_outcome, error_code, attempt_id),
            )
            await connection.execute(
                "UPDATE delivery_outbox SET status = ?, available_at = ?, lease_id = NULL, "
                "lease_owner = NULL, lease_expires_at = NULL, last_error_code = ?, "
                "updated_at = ?, completed_at = ? WHERE id = ? AND lease_id = ?",
                (status, now_text, error_code, now_text, completed_at, delivery_id, attempt_id),
            )
            if terminal:
                await self._append_outcome_event(
                    delivery_id=delivery_id,
                    workshop_id=WorkshopId(str(row[4])),
                    channel_id=ChannelId(str(row[5])),
                    channel_binding_id=ChannelBindingId(str(row[6])),
                    message_id=MessageId(str(row[7])),
                    transport=str(row[8]),
                    mode=str(row[9]),
                    authority_epoch_id=(DeliveryAuthorityEpochId(str(row[10])) if row[10] is not None else None),
                    attempt_number=int(row[2]),
                    status="failed",
                    error_code=error_code,
                    occurred_at=now,
                )
                failed += 1
            else:
                requeued += 1
        return DeliveryRecoveryResult(requeued=requeued, failed=failed)

    async def _append_outcome_event(
        self,
        *,
        delivery_id: DeliveryId,
        workshop_id: WorkshopId,
        channel_id: ChannelId,
        channel_binding_id: ChannelBindingId,
        message_id: MessageId,
        transport: str,
        mode: str,
        authority_epoch_id: DeliveryAuthorityEpochId | None,
        attempt_number: int,
        status: str,
        error_code: str | None,
        occurred_at: datetime,
    ) -> None:
        if status not in {"succeeded", "failed"}:
            raise DeliveryOutboxError("Delivery outcome status is not terminal")
        payload: dict[str, object] = {
            "message_id": message_id,
            "channel_id": channel_id,
            "channel_binding_id": channel_binding_id,
            "transport": transport,
            "mode": mode,
            "attempt_number": attempt_number,
        }
        if error_code is not None:
            payload["error_code"] = error_code
        if authority_epoch_id is not None:
            payload["authority_epoch_id"] = authority_epoch_id
        event_identity_version = "v3" if authority_epoch_id is not None else "v2"
        event_version = 3 if authority_epoch_id is not None else 2
        event_type = (
            WorkshopEventType.DELIVERY_SUCCEEDED if status == "succeeded" else WorkshopEventType.DELIVERY_FAILED
        )
        event = EventEnvelope.create(
            event_id=EventId.derived(
                workshop_id,
                f"delivery-outcome-event:{event_identity_version}:{delivery_id}:{status}",
            ),
            event_type=event_type,
            event_version=event_version,
            workshop_id=workshop_id,
            aggregate_type="delivery",
            aggregate_id=delivery_id,
            occurred_at=occurred_at,
            idempotency_key=(f"workshop-delivery-outcome:{event_identity_version}:{delivery_id}:{status}"),
            payload=payload,
            metadata={"source": "delivery_outbox"},
        )
        await self._store.append_in_transaction(event)

    async def _resolve_target(
        self,
        request: DeliveryRequest,
    ) -> tuple[WorkshopId, ChannelId, PrincipalId, str]:
        async with self._store.connection.execute(
            "SELECT c.workshop_id, m.channel_id, m.author_principal_id, cb.transport "
            "FROM messages m JOIN channels c ON c.id = m.channel_id "
            "JOIN channel_bindings cb ON cb.channel_id = m.channel_id "
            "WHERE m.id = ? AND cb.id = ?",
            (request.message_id, request.channel_binding_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise DeliveryTargetNotFoundError("Canonical message and channel binding were not found together")
        transport = str(row[3])
        if not _IDENTIFIER_PATTERN.fullmatch(transport):
            raise DeliveryTargetNotFoundError("Canonical binding transport is not a supported identifier")
        return (
            WorkshopId(str(row[0])),
            ChannelId(str(row[1])),
            PrincipalId(str(row[2])),
            transport,
        )

    async def _state_for_identity(
        self,
        message_id: MessageId,
        channel_binding_id: ChannelBindingId,
        mode: str,
    ) -> DeliveryState | None:
        async with self._store.connection.execute(
            "SELECT * FROM delivery_outbox WHERE message_id = ? AND channel_binding_id = ? AND mode = ?",
            (message_id, channel_binding_id, mode),
        ) as cursor:
            row = await cursor.fetchone()
        return self._state_from_row(row) if row is not None else None

    async def _state_by_id(self, delivery_id: DeliveryId) -> DeliveryState:
        async with self._store.connection.execute(
            "SELECT * FROM delivery_outbox WHERE id = ?",
            (delivery_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise DeliveryTargetNotFoundError(f"Delivery {delivery_id} was not found")
        return self._state_from_row(row)

    @staticmethod
    def _state_from_row(row: aiosqlite.Row) -> DeliveryState:
        return DeliveryState(
            delivery_id=DeliveryId(str(row["id"])),
            message_id=MessageId(str(row["message_id"])),
            channel_id=ChannelId(str(row["channel_id"])),
            channel_binding_id=ChannelBindingId(str(row["channel_binding_id"])),
            transport=str(row["transport"]),
            mode=str(row["mode"]),
            purpose=_delivery_purpose(row["purpose"]),
            execution_contract=_delivery_execution_contract(row["execution_contract"]),
            authority_epoch_id=(
                DeliveryAuthorityEpochId(str(row["authority_epoch_id"]))
                if row["authority_epoch_id"] is not None
                else None
            ),
            status=str(row["status"]),
            max_attempts=int(row["max_attempts"]),
            attempt_count=int(row["attempt_count"]),
            available_at=_parse_timestamp(str(row["available_at"])),
            last_error_code=str(row["last_error_code"]) if row["last_error_code"] is not None else None,
            requested_event_position=int(row["requested_event_position"]),
            completed_at=_optional_timestamp(row["completed_at"]),
        )

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return now.astimezone(UTC)
