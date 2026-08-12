"""Production-unused durable delivery work for Kai Workshop."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import aiosqlite

from kai.workshop.domain import (
    ChannelBindingId,
    ChannelId,
    DeliveryAttemptId,
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
    occurred_at: datetime
    max_attempts: int = 5

    def __post_init__(self) -> None:
        if not isinstance(self.message_id, MessageId):
            raise ValueError("message_id must be a MessageId")
        if not isinstance(self.channel_binding_id, ChannelBindingId):
            raise ValueError("channel_binding_id must be a ChannelBindingId")
        if not _IDENTIFIER_PATTERN.fullmatch(self.mode):
            raise ValueError("mode must be a lowercase identifier")
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


def _retry_delay(attempt_number: int) -> timedelta:
    seconds = min(_RETRY_BASE_SECONDS * (2 ** (attempt_number - 1)), _RETRY_MAX_SECONDS)
    return timedelta(seconds=seconds)


class WorkshopDeliveryOutbox:
    """Durable delivery state with lease-based, at-least-once work claims.

    No production worker or transport adapter uses this class yet. Existing
    direct Telegram delivery remains authoritative until a later cutover.
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
            resolved = await self._resolve_target(request)
            workshop_id, channel_id, author_principal_id, transport = resolved
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
                ):
                    raise DeliveryRequestConflictError("Delivery request identity has different semantics")
                await connection.commit()
                return DeliveryRequestResult(delivery=existing, inserted=False)

            occurred_at = request.occurred_at.astimezone(UTC)
            event = EventEnvelope.create(
                event_id=EventId.derived(
                    workshop_id,
                    f"delivery-request-event:{request.message_id}:{request.channel_binding_id}:{request.mode}",
                ),
                event_type=WorkshopEventType.DELIVERY_REQUESTED,
                event_version=1,
                workshop_id=workshop_id,
                aggregate_type="delivery",
                aggregate_id=delivery_id,
                actor_principal_id=author_principal_id,
                occurred_at=occurred_at,
                idempotency_key=(
                    f"workshop-delivery-request:v1:{request.message_id}:{request.channel_binding_id}:{request.mode}"
                ),
                payload={
                    "message_id": request.message_id,
                    "channel_id": channel_id,
                    "channel_binding_id": request.channel_binding_id,
                    "transport": transport,
                    "mode": request.mode,
                    "max_attempts": request.max_attempts,
                },
                metadata={"source": "delivery_outbox"},
            )
            appended = await self._store.append_in_transaction(event)
            if not appended.inserted:
                raise DeliveryRequestConflictError("Delivery request event exists without outbox state")

            timestamp = _format_timestamp(occurred_at)
            await connection.execute(
                "INSERT INTO delivery_outbox "
                "(id, workshop_id, channel_id, channel_binding_id, message_id, transport, mode, "
                "status, max_attempts, attempt_count, available_at, requested_event_position, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, 0, ?, ?, ?, ?)",
                (
                    delivery_id,
                    workshop_id,
                    channel_id,
                    request.channel_binding_id,
                    request.message_id,
                    transport,
                    request.mode,
                    request.max_attempts,
                    timestamp,
                    appended.event.position,
                    timestamp,
                    timestamp,
                ),
            )
            state = await self._state_by_id(delivery_id)
            await connection.commit()
            return DeliveryRequestResult(delivery=state, inserted=True)
        except Exception:
            await connection.rollback()
            raise

    async def claim_next(
        self,
        worker_id: str,
        *,
        lease_duration: timedelta = timedelta(seconds=30),
        transport: str | None = None,
        modes: tuple[str, ...] | None = None,
    ) -> DeliveryClaim | None:
        _validate_worker_id(worker_id)
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

        connection = self._store.connection
        now = self._now()
        try:
            await connection.execute("BEGIN IMMEDIATE")
            await self._recover_expired_in_transaction(now)
            now_text = _format_timestamp(now)
            filters = [
                "o.status IN ('pending', 'retry_wait')",
                "o.available_at <= ?",
                "o.attempt_count < o.max_attempts",
            ]
            parameters: list[object] = [now_text]
            if transport is not None:
                filters.append("o.transport = ?")
                parameters.append(transport)
            if modes is not None:
                placeholders = ", ".join("?" for _ in modes)
                filters.append(f"o.mode IN ({placeholders})")
                parameters.extend(modes)
            async with connection.execute(
                "SELECT o.id, o.workshop_id, o.channel_id, o.channel_binding_id, o.message_id, "
                "o.transport, o.mode, o.attempt_count, m.body, cb.external_channel_id "
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
            attempt_number = int(row[7]) + 1
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
                body=str(row[8]),
                external_channel_id=str(row[9]),
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

    async def recover_expired_leases(self) -> DeliveryRecoveryResult:
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            result = await self._recover_expired_in_transaction(self._now())
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
                "o.workshop_id, o.channel_id, o.channel_binding_id, o.message_id, o.transport, o.mode "
                "FROM delivery_outbox o JOIN delivery_attempts a "
                "ON a.delivery_id = o.id AND a.id = ? WHERE o.id = ?",
                (claim.attempt_id, claim.delivery_id),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                raise StaleDeliveryLeaseError("Delivery attempt does not exist")
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
                await self._recover_expired_in_transaction(now)
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

    async def _recover_expired_in_transaction(self, now: datetime) -> DeliveryRecoveryResult:
        connection = self._store.connection
        now_text = _format_timestamp(now)
        async with connection.execute(
            "SELECT id, lease_id, attempt_count, max_attempts, workshop_id, channel_id, "
            "channel_binding_id, message_id, transport, mode FROM delivery_outbox "
            "WHERE status = 'leased' AND lease_expires_at <= ? ORDER BY requested_event_position",
            (now_text,),
        ) as cursor:
            rows = await cursor.fetchall()
        requeued = 0
        failed = 0
        for row in rows:
            delivery_id = DeliveryId(str(row[0]))
            attempt_id = DeliveryAttemptId(str(row[1]))
            async with connection.execute(
                "SELECT COUNT(*) FROM delivery_fragments WHERE delivery_id = ? AND status IN ('sending', 'uncertain')",
                (delivery_id,),
            ) as cursor:
                uncertain_row = await cursor.fetchone()
            send_uncertain = uncertain_row is not None and int(uncertain_row[0]) > 0
            if send_uncertain:
                await connection.execute(
                    "UPDATE delivery_fragments SET status = 'uncertain', updated_at = ? "
                    "WHERE delivery_id = ? AND status = 'sending'",
                    (now_text, delivery_id),
                )
            terminal = send_uncertain or int(row[2]) >= int(row[3])
            status = "failed" if terminal else "retry_wait"
            completed_at = now_text if terminal else None
            attempt_outcome = "failed" if send_uncertain else "lease_expired"
            error_code = "delivery_send_uncertain" if send_uncertain else "lease_expired"
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
        event_type = (
            WorkshopEventType.DELIVERY_SUCCEEDED if status == "succeeded" else WorkshopEventType.DELIVERY_FAILED
        )
        event = EventEnvelope.create(
            event_id=EventId.derived(
                workshop_id,
                f"delivery-outcome-event:v2:{delivery_id}:{status}",
            ),
            event_type=event_type,
            event_version=2,
            workshop_id=workshop_id,
            aggregate_type="delivery",
            aggregate_id=delivery_id,
            occurred_at=occurred_at,
            idempotency_key=f"workshop-delivery-outcome:v2:{delivery_id}:{status}",
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
