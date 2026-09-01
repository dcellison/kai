"""Canonical, principal-private followed-thread unread authority."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from kai.workshop.domain import (
    ChannelId,
    EventEnvelope,
    MessageId,
    PrincipalId,
    ThreadReadPositionId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import WorkshopEventStore

MAX_THREAD_UNREAD_COUNT = 1000


class WorkshopThreadUnreadError(RuntimeError):
    """A followed-thread unread request could not be completed."""


class WorkshopThreadUnreadAccessDenied(WorkshopThreadUnreadError):
    """The requested thread is unavailable to the authenticated principal."""


class WorkshopThreadUnreadConflict(WorkshopThreadUnreadError):
    """The requested thread mutation conflicts with canonical state."""


class WorkshopThreadUnreadValidationError(WorkshopThreadUnreadError):
    """The requested thread mutation is malformed."""


@dataclass(frozen=True, slots=True)
class ThreadUnreadState:
    channel_id: ChannelId
    thread_root_id: MessageId
    followed: bool
    follow_baseline_event_position: int
    read_through_event_position: int
    read_through_message_id: MessageId | None
    state_version: int
    last_event_position: int
    unread_count: int
    unread_count_capped: bool
    first_unread_message_id: MessageId | None
    first_unread_event_position: int | None


@dataclass(frozen=True, slots=True)
class ThreadUnreadMutation:
    state: ThreadUnreadState
    replayed: bool


@dataclass(frozen=True, slots=True)
class ThreadUnreadEvent:
    state: ThreadUnreadState
    event_position: int
    transition: WorkshopEventType


@dataclass(frozen=True, slots=True)
class ThreadUnreadEventBatch:
    events: tuple[ThreadUnreadEvent, ...]
    next_position: int


@dataclass(frozen=True, slots=True)
class _ThreadAuthority:
    workshop_id: WorkshopId
    channel_id: ChannelId
    thread_root_id: MessageId
    current_boundary_position: int
    current_boundary_message_id: MessageId


def _operation_id(value: object) -> str:
    if not isinstance(value, str):
        raise WorkshopThreadUnreadValidationError("client_operation_id must be text")
    normalized = value.strip()
    if not normalized or len(normalized) > 200:
        raise WorkshopThreadUnreadValidationError("client_operation_id must contain 1 through 200 characters")
    return normalized


def _expected_version(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise WorkshopThreadUnreadValidationError("expected_state_version must be a non-negative integer")
    return value


class WorkshopThreadUnreadService:
    """Query and mutate one authenticated human's followed-thread cursor."""

    def __init__(self, store: WorkshopEventStore) -> None:
        self._store = store

    async def thread(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
        thread_root_id: MessageId,
    ) -> ThreadUnreadState:
        self._validate_ids(principal_id, channel_id, thread_root_id)
        authority = await self._authority(principal_id, channel_id, thread_root_id)
        return await self._state(principal_id, authority)

    async def follow(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
        thread_root_id: MessageId,
        *,
        expected_state_version: object,
        client_operation_id: object,
        occurred_at: datetime | None = None,
    ) -> ThreadUnreadMutation:
        return await self._set_followed(
            principal_id,
            channel_id,
            thread_root_id,
            followed=True,
            expected_state_version=expected_state_version,
            client_operation_id=client_operation_id,
            occurred_at=occurred_at,
        )

    async def unfollow(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
        thread_root_id: MessageId,
        *,
        expected_state_version: object,
        client_operation_id: object,
        occurred_at: datetime | None = None,
    ) -> ThreadUnreadMutation:
        return await self._set_followed(
            principal_id,
            channel_id,
            thread_root_id,
            followed=False,
            expected_state_version=expected_state_version,
            client_operation_id=client_operation_id,
            occurred_at=occurred_at,
        )

    async def advance(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
        thread_root_id: MessageId,
        message_id: MessageId,
        *,
        expected_state_version: object,
        client_operation_id: object,
        occurred_at: datetime | None = None,
    ) -> ThreadUnreadMutation:
        self._validate_ids(principal_id, channel_id, thread_root_id)
        if not isinstance(message_id, MessageId):
            raise WorkshopThreadUnreadValidationError("Invalid thread read-position boundary")
        expected = _expected_version(expected_state_version)
        operation_id = _operation_id(client_operation_id)
        now = self._time(occurred_at)
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            authority = await self._authority(principal_id, channel_id, thread_root_id)
            current = await self._state(principal_id, authority)
            async with connection.execute(
                "SELECT created_event_position FROM messages WHERE id = ? AND channel_id = ? AND thread_root_id = ?",
                (message_id, channel_id, thread_root_id),
            ) as cursor:
                target = await cursor.fetchone()
            if target is None:
                raise WorkshopThreadUnreadAccessDenied("Thread unread state is unavailable")
            target_position = int(target[0])
            payload = {
                "principal_id": principal_id,
                "channel_id": channel_id,
                "thread_root_id": thread_root_id,
                "message_id": message_id,
                "expected_state_version": expected,
            }
            replay = await self._replay(
                principal_id,
                thread_root_id,
                operation_id,
                WorkshopEventType.THREAD_READ_POSITION_ADVANCED,
                payload,
            )
            if replay:
                state = await self._state(principal_id, authority)
                await connection.rollback()
                return ThreadUnreadMutation(state, True)
            if not current.followed:
                raise WorkshopThreadUnreadConflict("Thread is not followed")
            if current.state_version != expected:
                raise WorkshopThreadUnreadConflict("Thread read position revision is stale")
            if target_position <= current.read_through_event_position:
                raise WorkshopThreadUnreadConflict("Thread read position cannot move backward")
            await self._append(
                principal_id,
                authority,
                operation_id,
                WorkshopEventType.THREAD_READ_POSITION_ADVANCED,
                payload,
                now,
            )
            state = await self._state(principal_id, authority)
            await connection.commit()
            return ThreadUnreadMutation(state, False)
        except Exception:
            await connection.rollback()
            raise

    async def events(
        self,
        principal_id: PrincipalId,
        *,
        after_position: int | None,
        limit: int = 100,
    ) -> ThreadUnreadEventBatch:
        if not isinstance(principal_id, PrincipalId):
            raise WorkshopThreadUnreadValidationError("Invalid thread unread principal")
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise WorkshopThreadUnreadValidationError("event limit must be from 1 through 100")
        async with self._store.connection.execute("SELECT COALESCE(MAX(position), 0) FROM event_log") as cursor:
            tip_row = await cursor.fetchone()
        tip = int(tip_row[0]) if tip_row is not None else 0
        if after_position is None:
            return ThreadUnreadEventBatch((), tip)
        if (
            not isinstance(after_position, int)
            or isinstance(after_position, bool)
            or after_position < 0
            or after_position > tip
        ):
            raise WorkshopThreadUnreadValidationError("Invalid thread unread event resume position")
        event_types = (
            WorkshopEventType.THREAD_FOLLOWED.value,
            WorkshopEventType.THREAD_UNFOLLOWED.value,
            WorkshopEventType.THREAD_READ_POSITION_ADVANCED.value,
        )
        async with self._store.connection.execute(
            "SELECT DISTINCT e.position, e.event_type, "
            "COALESCE(json_extract(e.payload_json, '$.channel_id'), m.channel_id), "
            "COALESCE(json_extract(e.payload_json, '$.thread_root_id'), m.thread_root_id, m.id) "
            "FROM event_log e LEFT JOIN messages m ON m.created_event_position = e.position "
            "JOIN thread_read_positions rp ON rp.principal_id = ? "
            "AND rp.channel_id = COALESCE(json_extract(e.payload_json, '$.channel_id'), m.channel_id) "
            "AND rp.thread_root_id = COALESCE(json_extract(e.payload_json, '$.thread_root_id'), m.thread_root_id, m.id) "
            "JOIN channel_memberships cm ON cm.principal_id = rp.principal_id AND cm.channel_id = rp.channel_id "
            "WHERE e.position > ? AND ((e.event_type IN (?, ?, ?) AND e.actor_principal_id = ?) "
            "OR (e.event_type = ? AND (rp.followed = 1 OR rp.last_event_position = e.position))) "
            "ORDER BY e.position LIMIT ?",
            (
                principal_id,
                after_position,
                *event_types,
                principal_id,
                WorkshopEventType.MESSAGE_CREATED.value,
                limit,
            ),
        ) as cursor:
            rows = list(await cursor.fetchall())
        events: list[ThreadUnreadEvent] = []
        for row in rows:
            channel_id = ChannelId(str(row[2]))
            root_id = MessageId(str(row[3]))
            authority = await self._authority(principal_id, channel_id, root_id)
            raw_transition = str(row[1])
            transition = (
                WorkshopEventType(raw_transition)
                if raw_transition != WorkshopEventType.MESSAGE_CREATED.value
                else WorkshopEventType.MESSAGE_CREATED
            )
            events.append(
                ThreadUnreadEvent(
                    await self._state(principal_id, authority),
                    int(row[0]),
                    transition,
                )
            )
        return ThreadUnreadEventBatch(tuple(events), events[-1].event_position if events else after_position)

    async def _set_followed(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
        thread_root_id: MessageId,
        *,
        followed: bool,
        expected_state_version: object,
        client_operation_id: object,
        occurred_at: datetime | None,
    ) -> ThreadUnreadMutation:
        self._validate_ids(principal_id, channel_id, thread_root_id)
        expected = _expected_version(expected_state_version)
        operation_id = _operation_id(client_operation_id)
        now = self._time(occurred_at)
        transition = WorkshopEventType.THREAD_FOLLOWED if followed else WorkshopEventType.THREAD_UNFOLLOWED
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            authority = await self._authority(principal_id, channel_id, thread_root_id)
            current = await self._state(principal_id, authority)
            payload = {
                "principal_id": principal_id,
                "channel_id": channel_id,
                "thread_root_id": thread_root_id,
                "expected_state_version": expected,
                "boundary_event_position": authority.current_boundary_position,
                "boundary_message_id": authority.current_boundary_message_id,
            }
            replay = await self._replay(principal_id, thread_root_id, operation_id, transition, payload)
            if replay:
                state = await self._state(principal_id, authority)
                await connection.rollback()
                return ThreadUnreadMutation(state, True)
            if current.state_version != expected:
                raise WorkshopThreadUnreadConflict("Thread follow revision is stale")
            if current.followed == followed:
                raise WorkshopThreadUnreadConflict("Thread follow state is unchanged")
            await self._append(principal_id, authority, operation_id, transition, payload, now)
            state = await self._state(principal_id, authority)
            await connection.commit()
            return ThreadUnreadMutation(state, False)
        except Exception:
            await connection.rollback()
            raise

    async def _append(
        self,
        principal_id: PrincipalId,
        authority: _ThreadAuthority,
        operation_id: str,
        transition: WorkshopEventType,
        payload: dict[str, object],
        occurred_at: datetime,
    ) -> None:
        result = await self._store.append_in_transaction(
            EventEnvelope.create(
                event_type=transition,
                event_version=1,
                workshop_id=authority.workshop_id,
                aggregate_type="thread_read_position",
                aggregate_id=ThreadReadPositionId.derived(
                    authority.thread_root_id,
                    f"principal:{principal_id}",
                ),
                actor_principal_id=principal_id,
                occurred_at=occurred_at,
                idempotency_key=f"thread-unread:v1:{principal_id}:{authority.thread_root_id}:{operation_id}",
                payload=payload,
                metadata={"source": "workshop_client"},
            )
        )
        if not result.inserted:
            raise WorkshopThreadUnreadConflict("Thread unread replay was inconsistent")
        await self._store.project_pending_in_transaction(CanonicalConversationProjection())

    async def _replay(
        self,
        principal_id: PrincipalId,
        thread_root_id: MessageId,
        operation_id: str,
        transition: WorkshopEventType,
        payload: dict[str, object],
    ) -> bool:
        existing = await self._store.event_by_idempotency_key(
            f"thread-unread:v1:{principal_id}:{thread_root_id}:{operation_id}"
        )
        if existing is None:
            return False
        if (
            existing.envelope.event_type != transition
            or existing.envelope.actor_principal_id != principal_id
            or existing.envelope.payload != payload
        ):
            raise WorkshopThreadUnreadConflict("client_operation_id is already bound to a different thread mutation")
        return True

    async def _authority(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
        thread_root_id: MessageId,
    ) -> _ThreadAuthority:
        async with self._store.connection.execute(
            "SELECT c.workshop_id, root.created_event_position, "
            "COALESCE((SELECT reply.created_event_position FROM messages reply "
            "WHERE reply.thread_root_id = root.id ORDER BY reply.created_event_position DESC LIMIT 1), "
            "root.created_event_position), "
            "COALESCE((SELECT reply.id FROM messages reply WHERE reply.thread_root_id = root.id "
            "ORDER BY reply.created_event_position DESC LIMIT 1), root.id) "
            "FROM messages root JOIN channels c ON c.id = root.channel_id "
            "JOIN channel_memberships cm ON cm.channel_id = c.id AND cm.principal_id = ? "
            "WHERE root.id = ? AND root.channel_id = ? AND root.thread_root_id IS NULL AND c.kind = 'group'",
            (principal_id, thread_root_id, channel_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise WorkshopThreadUnreadAccessDenied("Thread unread state is unavailable")
        return _ThreadAuthority(
            workshop_id=WorkshopId(str(row[0])),
            channel_id=channel_id,
            thread_root_id=thread_root_id,
            current_boundary_position=int(row[2]),
            current_boundary_message_id=MessageId(str(row[3])),
        )

    async def _state(
        self,
        principal_id: PrincipalId,
        authority: _ThreadAuthority,
    ) -> ThreadUnreadState:
        async with self._store.connection.execute(
            "SELECT followed, follow_baseline_event_position, read_through_event_position, "
            "read_through_message_id, state_version, last_event_position "
            "FROM thread_read_positions WHERE principal_id = ? AND thread_root_id = ?",
            (principal_id, authority.thread_root_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            followed = False
            baseline = authority.current_boundary_position
            read_through = authority.current_boundary_position
            read_message = authority.current_boundary_message_id
            state_version = 0
            last_position = authority.current_boundary_position
        else:
            followed = bool(row[0])
            baseline = int(row[1])
            read_through = int(row[2])
            read_message = MessageId(str(row[3])) if row[3] is not None else None
            state_version = int(row[4])
            last_position = int(row[5])
        rows: list[object] = []
        if followed:
            async with self._store.connection.execute(
                "SELECT id, created_event_position FROM messages "
                "WHERE thread_root_id = ? AND author_principal_id != ? "
                "AND created_event_position > ? ORDER BY created_event_position LIMIT ?",
                (authority.thread_root_id, principal_id, read_through, MAX_THREAD_UNREAD_COUNT + 1),
            ) as cursor:
                rows = list(await cursor.fetchall())
        capped = len(rows) > MAX_THREAD_UNREAD_COUNT
        bounded = rows[:MAX_THREAD_UNREAD_COUNT]
        first = tuple(bounded[0]) if bounded else None  # type: ignore[arg-type]
        return ThreadUnreadState(
            channel_id=authority.channel_id,
            thread_root_id=authority.thread_root_id,
            followed=followed,
            follow_baseline_event_position=baseline,
            read_through_event_position=read_through,
            read_through_message_id=read_message,
            state_version=state_version,
            last_event_position=last_position,
            unread_count=len(bounded),
            unread_count_capped=capped,
            first_unread_message_id=MessageId(str(first[0])) if first is not None else None,
            first_unread_event_position=int(first[1]) if first is not None else None,
        )

    @staticmethod
    def _time(value: datetime | None) -> datetime:
        result = value or datetime.now(UTC)
        if result.tzinfo is None or result.utcoffset() is None:
            raise WorkshopThreadUnreadValidationError("Thread unread time must be timezone-aware")
        return result.astimezone(UTC)

    @staticmethod
    def _validate_ids(
        principal_id: PrincipalId,
        channel_id: ChannelId,
        thread_root_id: MessageId,
    ) -> None:
        if (
            not isinstance(principal_id, PrincipalId)
            or not isinstance(channel_id, ChannelId)
            or not isinstance(thread_root_id, MessageId)
        ):
            raise WorkshopThreadUnreadValidationError("Invalid thread unread identity")
