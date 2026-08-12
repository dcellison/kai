"""Durable authority epochs for Workshop conversation delivery."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from kai.workshop.delivery_outbox import (
    CONVERSATION_REPLY_PURPOSE,
    STREAMING_FINALIZATION_CONTRACT,
)
from kai.workshop.domain import DeliveryAuthorityEpochId
from kai.workshop.store import WorkshopEventStore

TELEGRAM_CONVERSATION_FINALIZATION_LANE = "telegram_conversation_streaming_finalization"


class DeliveryAuthorityError(RuntimeError):
    """Base class for fail-closed delivery-authority errors."""


class DeliveryAuthorityInactiveError(DeliveryAuthorityError):
    """No active authority epoch can accept or execute conversation work."""


class DeliveryAuthorityHistoricalWorkError(DeliveryAuthorityError):
    """Unclassified matching work prevents a safe first activation."""


class DeliveryAuthorityOutstandingWorkError(DeliveryAuthorityError):
    """Non-terminal work prevents a safe authority deactivation."""


class DeliveryAuthorityUnreconciledFailureError(DeliveryAuthorityError):
    """Terminal failures require explicit acknowledgement before rollback."""


@dataclass(frozen=True, slots=True)
class DeliveryAuthorityEpoch:
    epoch_id: DeliveryAuthorityEpochId
    status: str
    activated_at: datetime
    deactivated_at: datetime | None
    terminal_failures_acknowledged_at: datetime | None


@dataclass(frozen=True, slots=True)
class DeliveryAuthorityActivationResult:
    epoch: DeliveryAuthorityEpoch
    inserted: bool


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


class WorkshopConversationDeliveryAuthority:
    """Own the durable activation boundary for one future Telegram route.

    Production startup activates or resumes one epoch before Telegram ingress.
    The separate service remains usable by operator tooling and tests without
    implicitly constructing a delivery worker.
    """

    def __init__(
        self,
        store: WorkshopEventStore,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store
        self._clock = clock or (lambda: datetime.now(UTC))

    async def activate(self) -> DeliveryAuthorityActivationResult:
        connection = self._store.connection
        now = self._now()
        try:
            await connection.execute("BEGIN IMMEDIATE")
            active = await self.active_epoch_in_transaction(required=False)
            if active is not None:
                await connection.commit()
                return DeliveryAuthorityActivationResult(epoch=active, inserted=False)

            async with connection.execute(
                "SELECT COUNT(*) FROM delivery_outbox "
                "WHERE purpose = ? AND execution_contract = ? AND authority_epoch_id IS NULL",
                (CONVERSATION_REPLY_PURPOSE, STREAMING_FINALIZATION_CONTRACT),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                raise DeliveryAuthorityError("Historical delivery query returned no row")
            if int(row[0]) != 0:
                raise DeliveryAuthorityHistoricalWorkError(
                    "Unclassified streaming-finalization work requires operator reconciliation"
                )

            epoch = DeliveryAuthorityEpoch(
                epoch_id=DeliveryAuthorityEpochId.new(),
                status="active",
                activated_at=now,
                deactivated_at=None,
                terminal_failures_acknowledged_at=None,
            )
            await connection.execute(
                "INSERT INTO delivery_authority_epochs (id, lane, status, activated_at) VALUES (?, ?, 'active', ?)",
                (epoch.epoch_id, TELEGRAM_CONVERSATION_FINALIZATION_LANE, _format_timestamp(now)),
            )
            await connection.commit()
            return DeliveryAuthorityActivationResult(epoch=epoch, inserted=True)
        except Exception:
            await connection.rollback()
            raise

    async def deactivate(
        self,
        *,
        acknowledge_terminal_failures: bool = False,
    ) -> DeliveryAuthorityEpoch:
        if not isinstance(acknowledge_terminal_failures, bool):
            raise ValueError("acknowledge_terminal_failures must be a boolean")
        connection = self._store.connection
        now = self._now()
        try:
            await connection.execute("BEGIN IMMEDIATE")
            active = await self.active_epoch_in_transaction(required=True)
            assert active is not None
            async with connection.execute(
                "SELECT COUNT(*) FROM delivery_outbox "
                "WHERE authority_epoch_id = ? AND status IN ('pending', 'leased', 'retry_wait')",
                (active.epoch_id,),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                raise DeliveryAuthorityError("Outstanding delivery query returned no row")
            if int(row[0]) != 0:
                raise DeliveryAuthorityOutstandingWorkError("Active delivery authority has non-terminal work")

            async with connection.execute(
                "SELECT COUNT(*) FROM delivery_outbox WHERE authority_epoch_id = ? AND status = 'failed'",
                (active.epoch_id,),
            ) as cursor:
                failure_row = await cursor.fetchone()
            if failure_row is None:
                raise DeliveryAuthorityError("Terminal delivery query returned no row")
            terminal_failures = int(failure_row[0])
            if terminal_failures and not acknowledge_terminal_failures:
                raise DeliveryAuthorityUnreconciledFailureError(
                    "Terminal delivery failures require explicit acknowledgement"
                )

            timestamp = _format_timestamp(now)
            cursor = await connection.execute(
                "UPDATE delivery_authority_epochs SET status = 'deactivated', deactivated_at = ?, "
                "terminal_failures_acknowledged_at = ? "
                "WHERE id = ? AND status = 'active'",
                (timestamp, timestamp if terminal_failures else None, active.epoch_id),
            )
            if cursor.rowcount != 1:
                raise DeliveryAuthorityError("Active delivery authority changed during deactivation")
            await connection.commit()
            return DeliveryAuthorityEpoch(
                epoch_id=active.epoch_id,
                status="deactivated",
                activated_at=active.activated_at,
                deactivated_at=now,
                terminal_failures_acknowledged_at=(now if terminal_failures else None),
            )
        except Exception:
            await connection.rollback()
            raise

    async def active_epoch(self) -> DeliveryAuthorityEpoch:
        epoch = await self.active_epoch_in_transaction(required=True)
        assert epoch is not None
        return epoch

    async def active_epoch_in_transaction(
        self,
        *,
        required: bool = True,
    ) -> DeliveryAuthorityEpoch | None:
        async with self._store.connection.execute(
            "SELECT id, status, activated_at, deactivated_at, terminal_failures_acknowledged_at "
            "FROM delivery_authority_epochs "
            "WHERE lane = ? AND status = 'active' ORDER BY activated_at",
            (TELEGRAM_CONVERSATION_FINALIZATION_LANE,),
        ) as cursor:
            rows = list(await cursor.fetchall())
        if len(rows) > 1:
            raise DeliveryAuthorityError("Multiple active delivery-authority epochs exist")
        if not rows:
            if required:
                raise DeliveryAuthorityInactiveError("Conversation delivery authority is not active")
            return None
        row = rows[0]
        return DeliveryAuthorityEpoch(
            epoch_id=DeliveryAuthorityEpochId(str(row[0])),
            status=str(row[1]),
            activated_at=_parse_timestamp(str(row[2])),
            deactivated_at=_parse_timestamp(str(row[3])) if row[3] is not None else None,
            terminal_failures_acknowledged_at=(_parse_timestamp(str(row[4])) if row[4] is not None else None),
        )

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return now.astimezone(UTC)
