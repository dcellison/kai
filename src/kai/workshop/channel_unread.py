"""Canonical, principal-private unread authority for channel timelines."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime

from kai.workshop.domain import (
    ChannelId,
    ChannelReadPositionId,
    EventEnvelope,
    MessageId,
    PrincipalId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import WorkshopEventStore

MAX_UNREAD_COUNT = 1000
MAX_UNREAD_CHANNELS = 500


class WorkshopChannelUnreadError(RuntimeError):
    """A channel unread request could not be completed."""


class WorkshopChannelUnreadAccessDenied(WorkshopChannelUnreadError):
    """The channel unread state is unavailable to the authenticated principal."""


class WorkshopChannelUnreadConflict(WorkshopChannelUnreadError):
    """The requested read position conflicts with current canonical state."""


class WorkshopChannelUnreadValidationError(WorkshopChannelUnreadError):
    """The channel unread request is malformed."""


@dataclass(frozen=True, slots=True)
class ChannelUnreadState:
    channel_id: ChannelId
    channel_kind: str
    channel_name: str | None
    archived: bool
    membership_baseline_event_position: int
    read_through_event_position: int
    read_through_message_id: MessageId | None
    state_version: int
    last_event_position: int
    unread_count: int
    unread_count_capped: bool
    first_unread_message_id: MessageId | None
    first_unread_event_position: int | None


@dataclass(frozen=True, slots=True)
class ChannelUnreadSnapshot:
    channels: tuple[ChannelUnreadState, ...]
    total_unread: int
    total_unread_capped: bool
    through_position: int


@dataclass(frozen=True, slots=True)
class ChannelReadPositionMutation:
    state: ChannelUnreadState
    replayed: bool


@dataclass(frozen=True, slots=True)
class ChannelReadPositionEvent:
    state: ChannelUnreadState
    event_position: int


@dataclass(frozen=True, slots=True)
class ChannelReadPositionEventBatch:
    events: tuple[ChannelReadPositionEvent, ...]
    next_position: int


@dataclass(frozen=True, slots=True)
class _ReadPositionRow:
    workshop_id: WorkshopId
    channel_id: ChannelId
    channel_kind: str
    channel_name: str | None
    archived: bool
    membership_baseline_event_position: int
    read_through_event_position: int
    read_through_message_id: MessageId | None
    state_version: int
    last_event_position: int


def _operation_id(value: object) -> str:
    if not isinstance(value, str):
        raise WorkshopChannelUnreadValidationError("client_operation_id must be text")
    normalized = value.strip()
    if not normalized or len(normalized) > 200:
        raise WorkshopChannelUnreadValidationError("client_operation_id must contain 1 through 200 characters")
    return normalized


def _expected_version(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise WorkshopChannelUnreadValidationError("expected_state_version must be a non-negative integer")
    return value


def _read_position_from_row(row: object) -> _ReadPositionRow:
    values = tuple(row)  # type: ignore[arg-type]
    return _ReadPositionRow(
        workshop_id=WorkshopId(str(values[0])),
        channel_id=ChannelId(str(values[1])),
        channel_kind=str(values[2]),
        channel_name=str(values[3]) if values[3] is not None else None,
        archived=values[4] is not None,
        membership_baseline_event_position=int(values[5]),
        read_through_event_position=int(values[6]),
        read_through_message_id=MessageId(str(values[7])) if values[7] is not None else None,
        state_version=int(values[8]),
        last_event_position=int(values[9]),
    )


_READ_POSITION_COLUMNS = (
    "c.workshop_id, c.id, c.kind, c.name, c.archived_at, "
    "rp.membership_baseline_event_position, rp.read_through_event_position, "
    "rp.read_through_message_id, rp.state_version, rp.last_event_position"
)


class WorkshopChannelUnreadService:
    """Query and advance one authenticated human's canonical channel cursors."""

    def __init__(self, store: WorkshopEventStore) -> None:
        self._store = store

    async def snapshot(self, principal_id: PrincipalId) -> ChannelUnreadSnapshot:
        self._validate_principal(principal_id)
        async with self._store.connection.execute(
            f"SELECT {_READ_POSITION_COLUMNS} FROM channel_read_positions rp "
            "JOIN channel_memberships cm ON cm.channel_id = rp.channel_id "
            "AND cm.principal_id = rp.principal_id "
            "JOIN channels c ON c.id = rp.channel_id "
            "WHERE rp.principal_id = ? AND c.archived_at IS NULL "
            "ORDER BY CASE c.kind WHEN 'direct' THEN 0 WHEN 'group' THEN 1 ELSE 2 END, "
            "lower(coalesce(c.name, '')), c.id LIMIT ?",
            (principal_id, MAX_UNREAD_CHANNELS + 1),
        ) as cursor:
            rows = list(await cursor.fetchall())
        if len(rows) > MAX_UNREAD_CHANNELS:
            raise WorkshopChannelUnreadValidationError("Too many accessible channels for one unread snapshot")
        states = tuple([await self._state_from_position(principal_id, _read_position_from_row(row)) for row in rows])
        exact_total = sum(state.unread_count for state in states)
        capped = any(state.unread_count_capped for state in states) or exact_total > MAX_UNREAD_COUNT
        async with self._store.connection.execute("SELECT COALESCE(MAX(position), 0) FROM event_log") as cursor:
            tip_row = await cursor.fetchone()
        return ChannelUnreadSnapshot(
            channels=states,
            total_unread=min(exact_total, MAX_UNREAD_COUNT),
            total_unread_capped=capped,
            through_position=int(tip_row[0]) if tip_row is not None else 0,
        )

    async def channel(self, principal_id: PrincipalId, channel_id: ChannelId) -> ChannelUnreadState:
        self._validate_principal(principal_id)
        position = await self._load_accessible_position(principal_id, channel_id)
        return await self._state_from_position(principal_id, position)

    async def advance(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
        message_id: MessageId,
        *,
        expected_state_version: object,
        client_operation_id: object,
        occurred_at: datetime | None = None,
    ) -> ChannelReadPositionMutation:
        self._validate_principal(principal_id)
        if not isinstance(channel_id, ChannelId) or not isinstance(message_id, MessageId):
            raise WorkshopChannelUnreadValidationError("Invalid channel read-position boundary")
        expected = _expected_version(expected_state_version)
        operation_id = _operation_id(client_operation_id)
        now = occurred_at or datetime.now(UTC)
        if now.tzinfo is None or now.utcoffset() is None:
            raise WorkshopChannelUnreadValidationError("Channel read-position time must be timezone-aware")
        now = now.astimezone(UTC)
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            current = await self._load_accessible_position(principal_id, channel_id)
            target_position = await self._authorized_target_position(principal_id, channel_id, message_id)
            idempotency_key = f"channel-read-position:v1:{principal_id}:{channel_id}:{operation_id}"
            existing = await self._store.event_by_idempotency_key(idempotency_key)
            if existing is not None:
                if (
                    existing.envelope.event_type != WorkshopEventType.CHANNEL_READ_POSITION_ADVANCED
                    or existing.envelope.actor_principal_id != principal_id
                    or existing.envelope.payload
                    != {
                        "principal_id": principal_id,
                        "channel_id": channel_id,
                        "message_id": message_id,
                        "expected_state_version": expected,
                    }
                ):
                    raise WorkshopChannelUnreadConflict(
                        "client_operation_id is already bound to a different read-position mutation"
                    )
                state = await self._state_from_position(
                    principal_id,
                    await self._load_accessible_position(principal_id, channel_id),
                )
                await connection.rollback()
                return ChannelReadPositionMutation(state, True)
            if current.state_version != expected:
                raise WorkshopChannelUnreadConflict("Channel read position revision is stale")
            if target_position <= current.read_through_event_position:
                raise WorkshopChannelUnreadConflict("Channel read position cannot move backward")
            event = EventEnvelope.create(
                event_type=WorkshopEventType.CHANNEL_READ_POSITION_ADVANCED,
                event_version=1,
                workshop_id=current.workshop_id,
                aggregate_type="channel_read_position",
                aggregate_id=ChannelReadPositionId.derived(channel_id, f"principal:{principal_id}"),
                actor_principal_id=principal_id,
                occurred_at=now,
                idempotency_key=idempotency_key,
                payload={
                    "principal_id": principal_id,
                    "channel_id": channel_id,
                    "message_id": message_id,
                    "expected_state_version": expected,
                },
                metadata={"source": "workshop_client"},
            )
            result = await self._store.append_in_transaction(event)
            if not result.inserted:
                raise WorkshopChannelUnreadConflict("Channel read-position replay was inconsistent")
            await self._store.project_pending_in_transaction(CanonicalConversationProjection())
            state = await self._state_from_position(
                principal_id,
                await self._load_accessible_position(principal_id, channel_id),
            )
            await connection.commit()
            return ChannelReadPositionMutation(state, False)
        except Exception:
            await connection.rollback()
            raise

    async def events(
        self,
        principal_id: PrincipalId,
        *,
        after_position: int | None,
        limit: int = 100,
    ) -> ChannelReadPositionEventBatch:
        self._validate_principal(principal_id)
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise WorkshopChannelUnreadValidationError("event limit must be from 1 through 100")
        async with self._store.connection.execute("SELECT COALESCE(MAX(position), 0) FROM event_log") as cursor:
            tip_row = await cursor.fetchone()
        tip = int(tip_row[0]) if tip_row is not None else 0
        if after_position is None:
            return ChannelReadPositionEventBatch((), tip)
        if (
            not isinstance(after_position, int)
            or isinstance(after_position, bool)
            or after_position < 0
            or after_position > tip
        ):
            raise WorkshopChannelUnreadValidationError("Invalid channel unread event resume position")
        async with self._store.connection.execute(
            "SELECT e.position, json_extract(e.payload_json, '$.channel_id') "
            "FROM event_log e JOIN channel_read_positions rp "
            "ON rp.principal_id = ? AND rp.channel_id = json_extract(e.payload_json, '$.channel_id') "
            "JOIN channel_memberships cm ON cm.principal_id = rp.principal_id "
            "AND cm.channel_id = rp.channel_id JOIN channels c ON c.id = rp.channel_id "
            "WHERE c.archived_at IS NULL AND e.position > ? AND ((e.event_type = ? "
            "AND e.actor_principal_id = ?) OR (e.event_type = ? "
            "AND e.actor_principal_id != ? "
            "AND json_extract(e.payload_json, '$.thread_root_id') IS NULL "
            "AND coalesce(json_extract(e.metadata_json, '$.source'), '') != 'scheduled_job')) "
            "ORDER BY e.position LIMIT ?",
            (
                principal_id,
                after_position,
                WorkshopEventType.CHANNEL_READ_POSITION_ADVANCED.value,
                principal_id,
                WorkshopEventType.MESSAGE_CREATED.value,
                principal_id,
                limit,
            ),
        ) as cursor:
            rows = list(await cursor.fetchall())
        events = tuple(
            [
                ChannelReadPositionEvent(
                    await self.channel(principal_id, ChannelId(str(row[1]))),
                    int(row[0]),
                )
                for row in rows
            ]
        )
        return ChannelReadPositionEventBatch(events, events[-1].event_position if events else after_position)

    async def _load_accessible_position(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
    ) -> _ReadPositionRow:
        async with self._store.connection.execute(
            f"SELECT {_READ_POSITION_COLUMNS} FROM channel_read_positions rp "
            "JOIN channel_memberships cm ON cm.channel_id = rp.channel_id "
            "AND cm.principal_id = rp.principal_id JOIN channels c ON c.id = rp.channel_id "
            "WHERE rp.principal_id = ? AND rp.channel_id = ?",
            (principal_id, channel_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise WorkshopChannelUnreadAccessDenied("Channel unread state is unavailable")
        return _read_position_from_row(row)

    async def _authorized_target_position(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
        message_id: MessageId,
    ) -> int:
        async with self._store.connection.execute(
            "SELECT m.created_event_position, m.thread_root_id, p.kind, e.metadata_json "
            "FROM messages m JOIN channel_memberships cm ON cm.channel_id = m.channel_id "
            "AND cm.principal_id = ? JOIN principals p ON p.id = m.author_principal_id "
            "JOIN event_log e ON e.position = m.created_event_position "
            "WHERE m.id = ? AND m.channel_id = ?",
            (principal_id, message_id, channel_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None or row[1] is not None:
            raise WorkshopChannelUnreadAccessDenied("Channel unread state is unavailable")
        try:
            metadata = json.loads(str(row[3]))
        except json.JSONDecodeError as exc:
            raise WorkshopChannelUnreadError("Canonical message metadata is malformed") from exc
        if str(row[2]) == "human" and isinstance(metadata, dict) and metadata.get("source") == "scheduled_job":
            raise WorkshopChannelUnreadAccessDenied("Channel unread state is unavailable")
        return int(row[0])

    async def _state_from_position(
        self,
        principal_id: PrincipalId,
        position: _ReadPositionRow,
    ) -> ChannelUnreadState:
        async with self._store.connection.execute(
            "SELECT m.id, m.created_event_position FROM messages m "
            "JOIN principals p ON p.id = m.author_principal_id "
            "JOIN event_log e ON e.position = m.created_event_position "
            "WHERE m.channel_id = ? AND m.thread_root_id IS NULL "
            "AND m.author_principal_id != ? AND m.created_event_position > ? "
            "AND NOT (p.kind = 'human' AND "
            "coalesce(json_extract(e.metadata_json, '$.source'), '') = 'scheduled_job') "
            "ORDER BY m.created_event_position LIMIT ?",
            (
                position.channel_id,
                principal_id,
                position.read_through_event_position,
                MAX_UNREAD_COUNT + 1,
            ),
        ) as cursor:
            rows = list(await cursor.fetchall())
        capped = len(rows) > MAX_UNREAD_COUNT
        bounded = rows[:MAX_UNREAD_COUNT]
        first = bounded[0] if bounded else None
        return ChannelUnreadState(
            channel_id=position.channel_id,
            channel_kind=position.channel_kind,
            channel_name=position.channel_name,
            archived=position.archived,
            membership_baseline_event_position=position.membership_baseline_event_position,
            read_through_event_position=position.read_through_event_position,
            read_through_message_id=position.read_through_message_id,
            state_version=position.state_version,
            last_event_position=position.last_event_position,
            unread_count=len(bounded),
            unread_count_capped=capped,
            first_unread_message_id=MessageId(str(first[0])) if first is not None else None,
            first_unread_event_position=int(first[1]) if first is not None else None,
        )

    @staticmethod
    def _validate_principal(principal_id: PrincipalId) -> None:
        if not isinstance(principal_id, PrincipalId):
            raise WorkshopChannelUnreadValidationError("Invalid channel unread principal")
