"""Durable per-fragment progress for production-unused Workshop delivery."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

import aiosqlite

from kai.workshop.delivery_outbox import DeliveryClaim, StaleDeliveryLeaseError
from kai.workshop.domain import DeliveryAttemptId, DeliveryId
from kai.workshop.store import WorkshopEventStore

_MAX_FRAGMENTS = 1000


class DeliveryFragmentError(RuntimeError):
    """Base class for fail-closed durable fragment errors."""


class DeliveryFragmentPlanConflictError(DeliveryFragmentError):
    """A delivery already has a different immutable fragment plan."""


class DeliveryFragmentStateError(DeliveryFragmentError):
    """Fragment progress is missing, non-sequential, or otherwise invalid."""


class DeliveryFragmentUncertainError(DeliveryFragmentError):
    """A fragment may have crossed the external transport boundary."""


@dataclass(frozen=True, slots=True)
class DeliveryFragment:
    delivery_id: DeliveryId
    fragment_index: int
    fragment_count: int
    body: str
    status: str
    attempt_id: DeliveryAttemptId | None
    external_message_id: str | None
    sent_at: datetime | None


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


class WorkshopDeliveryFragments:
    """Persist an immutable plan and monotonic progress for one delivery.

    A fragment moves ``pending -> sending -> sent``. The ``sending`` state is
    written before the external API call. If the call cannot be proven to have
    failed or succeeded, it moves to ``uncertain`` and must never be retried
    automatically. This deliberately prefers operator reconciliation over a
    duplicate Telegram message.
    """

    def __init__(
        self,
        store: WorkshopEventStore,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store
        self._clock = clock or (lambda: datetime.now(UTC))

    async def prepare(self, claim: DeliveryClaim, bodies: tuple[str, ...]) -> tuple[DeliveryFragment, ...]:
        self._validate_claim(claim)
        self._validate_bodies(bodies)
        connection = self._store.connection
        now = self._now()
        try:
            await connection.execute("BEGIN IMMEDIATE")
            await self._require_lease_owner(connection, claim, now=now, require_unexpired=True)
            existing = await self._rows(connection, claim.delivery_id)
            if existing:
                existing_bodies = tuple(str(row["body"]) for row in existing)
                counts = {int(row["fragment_count"]) for row in existing}
                if existing_bodies != bodies or counts != {len(bodies)}:
                    raise DeliveryFragmentPlanConflictError("Delivery fragment plan is immutable")
                fragments = tuple(self._from_row(row) for row in existing)
                await connection.commit()
                return fragments

            timestamp = _format_timestamp(now)
            await connection.executemany(
                "INSERT INTO delivery_fragments "
                "(delivery_id, fragment_index, fragment_count, body, status, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, 'pending', ?, ?)",
                (
                    (claim.delivery_id, index, len(bodies), body, timestamp, timestamp)
                    for index, body in enumerate(bodies)
                ),
            )
            fragments = tuple(self._from_row(row) for row in await self._rows(connection, claim.delivery_id))
            await connection.commit()
            return fragments
        except Exception:
            await connection.rollback()
            raise

    async def begin_next(self, claim: DeliveryClaim) -> DeliveryFragment | None:
        self._validate_claim(claim)
        connection = self._store.connection
        now = self._now()
        try:
            await connection.execute("BEGIN IMMEDIATE")
            await self._require_lease_owner(connection, claim, now=now, require_unexpired=True)
            rows = await self._rows(connection, claim.delivery_id)
            if not rows:
                raise DeliveryFragmentStateError("Delivery has no fragment plan")
            self._validate_progress(rows)
            for row in rows:
                status = str(row["status"])
                if status in {"sending", "uncertain"}:
                    raise DeliveryFragmentUncertainError("Delivery has an unresolved fragment")
                if status == "pending":
                    timestamp = _format_timestamp(now)
                    cursor = await connection.execute(
                        "UPDATE delivery_fragments SET status = 'sending', attempt_id = ?, updated_at = ? "
                        "WHERE delivery_id = ? AND fragment_index = ? AND status = 'pending'",
                        (
                            claim.attempt_id,
                            timestamp,
                            claim.delivery_id,
                            int(row["fragment_index"]),
                        ),
                    )
                    if cursor.rowcount != 1:
                        raise DeliveryFragmentStateError("Fragment lost its serialized transition")
                    updated = await self._row(connection, claim.delivery_id, int(row["fragment_index"]))
                    await connection.commit()
                    return self._from_row(updated)
            await connection.commit()
            return None
        except Exception:
            await connection.rollback()
            raise

    async def mark_sent(
        self,
        claim: DeliveryClaim,
        fragment: DeliveryFragment,
        *,
        external_message_id: int,
    ) -> DeliveryFragment:
        if (
            not isinstance(external_message_id, int)
            or isinstance(external_message_id, bool)
            or external_message_id <= 0
        ):
            raise ValueError("external_message_id must be a positive integer")
        self._validate_fragment_for_claim(claim, fragment)
        connection = self._store.connection
        now = self._now()
        message_id = str(external_message_id)
        try:
            await connection.execute("BEGIN IMMEDIATE")
            await self._require_lease_owner(connection, claim, now=now, require_unexpired=False)
            current = await self._row(connection, claim.delivery_id, fragment.fragment_index)
            if str(current["status"]) == "sent":
                if str(current["external_message_id"]) != message_id:
                    raise DeliveryFragmentStateError("Fragment was completed with another message ID")
                result = self._from_row(current)
                await connection.commit()
                return result
            if str(current["status"]) != "sending" or str(current["attempt_id"]) != claim.attempt_id:
                raise DeliveryFragmentStateError("Fragment is not owned by this delivery attempt")
            timestamp = _format_timestamp(now)
            await connection.execute(
                "UPDATE delivery_fragments SET status = 'sent', external_message_id = ?, "
                "updated_at = ?, sent_at = ? WHERE delivery_id = ? AND fragment_index = ?",
                (message_id, timestamp, timestamp, claim.delivery_id, fragment.fragment_index),
            )
            result = self._from_row(await self._row(connection, claim.delivery_id, fragment.fragment_index))
            await connection.commit()
            return result
        except Exception:
            await connection.rollback()
            raise

    async def release_after_definitive_failure(
        self,
        claim: DeliveryClaim,
        fragment: DeliveryFragment,
    ) -> DeliveryFragment:
        self._validate_fragment_for_claim(claim, fragment)
        return await self._settle_unsent(claim, fragment, uncertain=False)

    async def mark_uncertain(
        self,
        claim: DeliveryClaim,
        fragment: DeliveryFragment,
    ) -> DeliveryFragment:
        self._validate_fragment_for_claim(claim, fragment)
        return await self._settle_unsent(claim, fragment, uncertain=True)

    async def fragments(self, delivery_id: DeliveryId) -> tuple[DeliveryFragment, ...]:
        if not isinstance(delivery_id, DeliveryId):
            raise ValueError("delivery_id must be a DeliveryId")
        rows = await self._rows(self._store.connection, delivery_id)
        return tuple(self._from_row(row) for row in rows)

    async def _settle_unsent(
        self,
        claim: DeliveryClaim,
        fragment: DeliveryFragment,
        *,
        uncertain: bool,
    ) -> DeliveryFragment:
        connection = self._store.connection
        now = self._now()
        try:
            await connection.execute("BEGIN IMMEDIATE")
            await self._require_lease_owner(connection, claim, now=now, require_unexpired=False)
            current = await self._row(connection, claim.delivery_id, fragment.fragment_index)
            desired = "uncertain" if uncertain else "pending"
            if str(current["status"]) == desired:
                result = self._from_row(current)
                await connection.commit()
                return result
            if str(current["status"]) != "sending" or str(current["attempt_id"]) != claim.attempt_id:
                raise DeliveryFragmentStateError("Fragment is not owned by this delivery attempt")
            timestamp = _format_timestamp(now)
            await connection.execute(
                "UPDATE delivery_fragments SET status = ?, attempt_id = ?, updated_at = ? "
                "WHERE delivery_id = ? AND fragment_index = ?",
                (
                    desired,
                    claim.attempt_id if uncertain else None,
                    timestamp,
                    claim.delivery_id,
                    fragment.fragment_index,
                ),
            )
            result = self._from_row(await self._row(connection, claim.delivery_id, fragment.fragment_index))
            await connection.commit()
            return result
        except Exception:
            await connection.rollback()
            raise

    async def _require_lease_owner(
        self,
        connection: aiosqlite.Connection,
        claim: DeliveryClaim,
        *,
        now: datetime,
        require_unexpired: bool,
    ) -> None:
        async with connection.execute(
            "SELECT status, lease_id, lease_expires_at FROM delivery_outbox WHERE id = ?",
            (claim.delivery_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None or str(row["status"]) != "leased" or str(row["lease_id"]) != claim.attempt_id:
            raise StaleDeliveryLeaseError("Delivery attempt no longer owns the active lease")
        if require_unexpired and (
            row["lease_expires_at"] is None or _parse_timestamp(str(row["lease_expires_at"])) <= now
        ):
            raise StaleDeliveryLeaseError("Delivery attempt lease has expired")

    @staticmethod
    async def _rows(
        connection: aiosqlite.Connection,
        delivery_id: DeliveryId,
    ) -> list[aiosqlite.Row]:
        async with connection.execute(
            "SELECT * FROM delivery_fragments WHERE delivery_id = ? ORDER BY fragment_index",
            (delivery_id,),
        ) as cursor:
            return list(await cursor.fetchall())

    @staticmethod
    async def _row(
        connection: aiosqlite.Connection,
        delivery_id: DeliveryId,
        fragment_index: int,
    ) -> aiosqlite.Row:
        async with connection.execute(
            "SELECT * FROM delivery_fragments WHERE delivery_id = ? AND fragment_index = ?",
            (delivery_id, fragment_index),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise DeliveryFragmentStateError("Delivery fragment does not exist")
        return row

    @staticmethod
    def _validate_bodies(bodies: tuple[str, ...]) -> None:
        if not isinstance(bodies, tuple) or not bodies or len(bodies) > _MAX_FRAGMENTS:
            raise ValueError(f"bodies must contain between 1 and {_MAX_FRAGMENTS} fragments")
        if any(not isinstance(body, str) or not body or len(body) > 4096 for body in bodies):
            raise ValueError("each fragment body must contain between 1 and 4096 characters")

    @staticmethod
    def _validate_claim(claim: DeliveryClaim) -> None:
        if not isinstance(claim, DeliveryClaim):
            raise ValueError("claim must be a DeliveryClaim")

    @staticmethod
    def _validate_fragment_for_claim(claim: DeliveryClaim, fragment: DeliveryFragment) -> None:
        WorkshopDeliveryFragments._validate_claim(claim)
        if not isinstance(fragment, DeliveryFragment) or fragment.delivery_id != claim.delivery_id:
            raise ValueError("fragment must belong to the claimed delivery")

    @staticmethod
    def _validate_progress(rows: list[aiosqlite.Row]) -> None:
        seen_pending = False
        for expected_index, row in enumerate(rows):
            if int(row["fragment_index"]) != expected_index or int(row["fragment_count"]) != len(rows):
                raise DeliveryFragmentStateError("Delivery fragment plan is inconsistent")
            status = str(row["status"])
            if status == "sent" and seen_pending:
                raise DeliveryFragmentStateError("Delivery fragment progress is non-sequential")
            if status != "sent":
                seen_pending = True

    @staticmethod
    def _from_row(row: aiosqlite.Row) -> DeliveryFragment:
        return DeliveryFragment(
            delivery_id=DeliveryId(str(row["delivery_id"])),
            fragment_index=int(row["fragment_index"]),
            fragment_count=int(row["fragment_count"]),
            body=str(row["body"]),
            status=str(row["status"]),
            attempt_id=(DeliveryAttemptId(str(row["attempt_id"])) if row["attempt_id"] is not None else None),
            external_message_id=(str(row["external_message_id"]) if row["external_message_id"] is not None else None),
            sent_at=(_parse_timestamp(str(row["sent_at"])) if row["sent_at"] is not None else None),
        )

    def _now(self) -> datetime:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("clock must return a timezone-aware datetime")
        return now.astimezone(UTC)
