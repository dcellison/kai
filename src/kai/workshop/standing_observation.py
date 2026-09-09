"""Canonical, adapter-neutral observation inbox for standing Workshop agents."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import aiosqlite

from kai.workshop.collaboration_authority import CollaborationHostPolicy, StandingParticipationHostPolicy
from kai.workshop.domain import AgentId, ChannelId, MessageId, WorkshopEventType
from kai.workshop.store import StoredEvent, WorkshopEventStore

OBSERVATION_PROJECTION_VERSION = 1


@dataclass(frozen=True, slots=True)
class StandingObservationState:
    channel_id: ChannelId
    agent_id: AgentId
    scope_kind: str
    scope_id: str
    delivered_through_event_position: int
    pending_through_event_position: int
    considered_through_event_position: int
    oldest_pending_event_position: int | None
    pending_message_count: int
    latest_human_anchor_message_id: MessageId | None
    latest_human_anchor_event_position: int | None
    not_before: datetime | None
    lifecycle_state: str
    overflowed_at: datetime | None
    overflow_reason: str | None
    overflow_from_event_position: int | None
    overflow_through_event_position: int | None
    state_version: int
    last_event_position: int


@dataclass(frozen=True, slots=True)
class StandingObservationBatch:
    channel_id: ChannelId
    agent_id: AgentId
    scope_id: str
    from_event_position: int
    through_event_position: int
    message_ids: tuple[MessageId, ...]


def _parse_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("Standing observation timestamp is malformed")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Standing observation timestamp must be timezone-aware")
    return parsed.astimezone(UTC)


async def _table_exists(connection: aiosqlite.Connection, table: str) -> bool:
    async with connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ) as cursor:
        return await cursor.fetchone() is not None


async def synchronize_standing_observation_host_policy_in_transaction(
    connection: aiosqlite.Connection,
    host_policy: CollaborationHostPolicy,
    *,
    occurred_at: datetime,
) -> None:
    """Publish the process-owned observation limits for universal projection use."""
    if not connection.in_transaction:
        raise RuntimeError("Standing observation host-policy synchronization requires a transaction")
    policy = host_policy.standing_participation
    await connection.execute(
        "INSERT INTO standing_participation_host_policy "
        "(singleton, host_policy_version, enabled, coalescing_grace_seconds, "
        "max_messages_per_observe_run, max_pending_messages_per_scope, "
        "max_pending_age_seconds, updated_at) VALUES (1, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(singleton) DO UPDATE SET host_policy_version = excluded.host_policy_version, "
        "enabled = excluded.enabled, coalescing_grace_seconds = excluded.coalescing_grace_seconds, "
        "max_messages_per_observe_run = excluded.max_messages_per_observe_run, "
        "max_pending_messages_per_scope = excluded.max_pending_messages_per_scope, "
        "max_pending_age_seconds = excluded.max_pending_age_seconds, updated_at = excluded.updated_at",
        (
            host_policy.version,
            int(policy.enabled),
            policy.coalescing_grace_seconds,
            policy.max_messages_per_observe_run,
            policy.max_pending_messages_per_scope,
            policy.max_pending_age_seconds,
            occurred_at.astimezone(UTC).isoformat(),
        ),
    )


async def _current_host_policy(
    connection: aiosqlite.Connection,
) -> tuple[int, StandingParticipationHostPolicy] | None:
    if not await _table_exists(connection, "standing_participation_host_policy"):
        return None
    async with connection.execute(
        "SELECT host_policy_version, enabled, coalescing_grace_seconds, "
        "max_messages_per_observe_run, max_pending_messages_per_scope, max_pending_age_seconds "
        "FROM standing_participation_host_policy WHERE singleton = 1"
    ) as cursor:
        row = await cursor.fetchone()
    if row is None:
        return None
    return (
        int(row[0]),
        StandingParticipationHostPolicy(
            enabled=bool(row[1]),
            coalescing_grace_seconds=int(row[2]),
            max_messages_per_observe_run=int(row[3]),
            max_pending_messages_per_scope=int(row[4]),
            max_pending_age_seconds=int(row[5]),
        ),
    )


async def apply_canonical_message_to_standing_observations(
    connection: aiosqlite.Connection,
    event: StoredEvent,
) -> None:
    """Advance every eligible standing inbox once from one canonical message."""
    if event.envelope.event_type != WorkshopEventType.MESSAGE_CREATED:
        return
    if not await _table_exists(connection, "channel_agent_observation_states"):
        return
    configured = await _current_host_policy(connection)
    if configured is None or not configured[1].enabled:
        return
    _host_version, policy = configured
    message_id = MessageId(str(event.envelope.aggregate_id))
    async with connection.execute(
        "SELECT m.channel_id, m.author_principal_id, m.thread_root_id, c.kind, c.archived_at, "
        "p.kind, a.id FROM messages m JOIN channels c ON c.id = m.channel_id "
        "JOIN principals p ON p.id = m.author_principal_id "
        "LEFT JOIN agents a ON a.principal_id = p.id WHERE m.id = ?",
        (message_id,),
    ) as cursor:
        message = await cursor.fetchone()
    if message is None or str(message[3]) != "group" or message[4] is not None:
        return
    channel_id = ChannelId(str(message[0]))
    author_kind = str(message[5])
    author_agent_id = AgentId(str(message[6])) if message[6] is not None else None
    thread_root = str(message[2]) if message[2] is not None else None
    scope_kind = "thread" if thread_root is not None else "channel"
    scope_id = thread_root or str(channel_id)
    occurred_at = event.envelope.occurred_at.astimezone(UTC)
    async with connection.execute(
        "SELECT s.agent_id, s.started_event_position, s.started_at, s.started_by_message_id, "
        "starter.created_event_position, starter.thread_root_id FROM channel_agent_standings s "
        "JOIN messages starter ON starter.id = s.started_by_message_id "
        "WHERE s.channel_id = ? AND s.lifecycle_state = 'active' AND s.started_event_position < ? "
        "ORDER BY s.started_event_position, s.agent_id",
        (channel_id, event.position),
    ) as cursor:
        subscriptions = list(await cursor.fetchall())
    for subscription in subscriptions:
        agent_id = AgentId(str(subscription[0]))
        starting_scope_id = str(subscription[5]) if subscription[5] is not None else str(channel_id)
        initial_anchor_id = MessageId(str(subscription[3])) if starting_scope_id == scope_id else None
        initial_anchor_position = int(subscription[4]) if initial_anchor_id is not None else None
        async with connection.execute(
            "SELECT 1 FROM channel_agent_dismissals WHERE channel_id = ? AND agent_id = ? "
            "AND dismissed_at >= ? AND (thread_root_message_id IS NULL OR thread_root_message_id = ?) LIMIT 1",
            (channel_id, agent_id, str(subscription[2]), thread_root),
        ) as cursor:
            if await cursor.fetchone() is not None:
                continue
        await _advance_observation_state(
            connection,
            event_position=event.position,
            occurred_at=occurred_at,
            channel_id=channel_id,
            agent_id=agent_id,
            scope_kind=scope_kind,
            scope_id=scope_id,
            subscription_start_position=int(subscription[1]),
            message_id=message_id,
            author_kind=author_kind,
            own_message=author_agent_id == agent_id,
            initial_human_anchor_message_id=initial_anchor_id,
            initial_human_anchor_event_position=initial_anchor_position,
            policy=policy,
        )


async def _advance_observation_state(
    connection: aiosqlite.Connection,
    *,
    event_position: int,
    occurred_at: datetime,
    channel_id: ChannelId,
    agent_id: AgentId,
    scope_kind: str,
    scope_id: str,
    subscription_start_position: int,
    message_id: MessageId,
    author_kind: str,
    own_message: bool,
    initial_human_anchor_message_id: MessageId | None,
    initial_human_anchor_event_position: int | None,
    policy: StandingParticipationHostPolicy,
) -> None:
    async with connection.execute(
        "SELECT delivered_through_event_position, pending_through_event_position, "
        "considered_through_event_position, oldest_pending_event_position, pending_message_count, "
        "latest_human_anchor_message_id, latest_human_anchor_event_position, not_before, "
        "lifecycle_state, overflowed_at, overflow_reason, overflow_from_event_position, "
        "overflow_through_event_position, state_version, last_event_position "
        "FROM channel_agent_observation_states WHERE channel_id = ? AND agent_id = ? AND scope_id = ?",
        (channel_id, agent_id, scope_id),
    ) as cursor:
        current = await cursor.fetchone()
    human_anchor_id = message_id if author_kind == "human" else initial_human_anchor_message_id
    human_anchor_position = event_position if author_kind == "human" else initial_human_anchor_event_position
    if current is None:
        if own_message:
            values = (
                subscription_start_position,
                subscription_start_position,
                event_position,
                None,
                0,
                human_anchor_id,
                human_anchor_position,
                None,
                "idle",
            )
        else:
            values = (
                subscription_start_position,
                event_position,
                event_position,
                event_position,
                1,
                human_anchor_id,
                human_anchor_position,
                (occurred_at + timedelta(seconds=policy.coalescing_grace_seconds)).isoformat(),
                "pending",
            )
        await connection.execute(
            "INSERT INTO channel_agent_observation_states "
            "(channel_id, agent_id, scope_kind, scope_id, delivered_through_event_position, "
            "pending_through_event_position, considered_through_event_position, "
            "oldest_pending_event_position, pending_message_count, latest_human_anchor_message_id, "
            "latest_human_anchor_event_position, not_before, lifecycle_state, overflowed_at, "
            "overflow_reason, overflow_from_event_position, overflow_through_event_position, "
            "projection_version, state_version, last_event_position) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, ?, 1, ?)",
            (
                channel_id,
                agent_id,
                scope_kind,
                scope_id,
                *values,
                OBSERVATION_PROJECTION_VERSION,
                event_position,
            ),
        )
        return
    if int(current[14]) >= event_position:
        return
    anchor_id = human_anchor_id or current[5]
    anchor_position = human_anchor_position or current[6]
    if own_message:
        await connection.execute(
            "UPDATE channel_agent_observation_states SET considered_through_event_position = ?, "
            "latest_human_anchor_message_id = ?, latest_human_anchor_event_position = ?, "
            "state_version = state_version + 1, last_event_position = ? "
            "WHERE channel_id = ? AND agent_id = ? AND scope_id = ?",
            (event_position, anchor_id, anchor_position, event_position, channel_id, agent_id, scope_id),
        )
        return
    if str(current[8]) == "paused_overflow":
        await connection.execute(
            "UPDATE channel_agent_observation_states SET considered_through_event_position = ?, "
            "latest_human_anchor_message_id = ?, latest_human_anchor_event_position = ?, "
            "overflow_through_event_position = ?, state_version = state_version + 1, "
            "last_event_position = ? WHERE channel_id = ? AND agent_id = ? AND scope_id = ?",
            (
                event_position,
                anchor_id,
                anchor_position,
                event_position,
                event_position,
                channel_id,
                agent_id,
                scope_id,
            ),
        )
        return
    pending_count = int(current[4])
    oldest_pending = int(current[3]) if current[3] is not None else event_position
    overflow_reason: str | None = None
    overflow_from = event_position
    if pending_count + 1 > policy.max_pending_messages_per_scope:
        overflow_reason = "pending_count"
    elif pending_count:
        async with connection.execute(
            "SELECT occurred_at FROM event_log WHERE position = ? AND event_type = ?",
            (oldest_pending, WorkshopEventType.MESSAGE_CREATED.value),
        ) as cursor:
            oldest = await cursor.fetchone()
        if oldest is None:
            overflow_reason = "retention"
            overflow_from = oldest_pending
        elif occurred_at - _parse_timestamp(oldest[0]) > timedelta(seconds=policy.max_pending_age_seconds):
            overflow_reason = "pending_age"
            overflow_from = oldest_pending
    if overflow_reason is not None:
        await connection.execute(
            "UPDATE channel_agent_observation_states SET considered_through_event_position = ?, "
            "latest_human_anchor_message_id = ?, latest_human_anchor_event_position = ?, "
            "lifecycle_state = 'paused_overflow', overflowed_at = ?, overflow_reason = ?, "
            "overflow_from_event_position = ?, overflow_through_event_position = ?, "
            "state_version = state_version + 1, last_event_position = ? "
            "WHERE channel_id = ? AND agent_id = ? AND scope_id = ?",
            (
                event_position,
                anchor_id,
                anchor_position,
                occurred_at.isoformat(),
                overflow_reason,
                overflow_from,
                event_position,
                event_position,
                channel_id,
                agent_id,
                scope_id,
            ),
        )
        return
    not_before = current[7]
    if pending_count == 0:
        not_before = (occurred_at + timedelta(seconds=policy.coalescing_grace_seconds)).isoformat()
    await connection.execute(
        "UPDATE channel_agent_observation_states SET pending_through_event_position = ?, "
        "considered_through_event_position = ?, oldest_pending_event_position = ?, "
        "pending_message_count = ?, latest_human_anchor_message_id = ?, "
        "latest_human_anchor_event_position = ?, not_before = ?, lifecycle_state = 'pending', "
        "state_version = state_version + 1, last_event_position = ? "
        "WHERE channel_id = ? AND agent_id = ? AND scope_id = ?",
        (
            event_position,
            event_position,
            oldest_pending,
            pending_count + 1,
            anchor_id,
            anchor_position,
            not_before,
            event_position,
            channel_id,
            agent_id,
            scope_id,
        ),
    )


class WorkshopStandingObservationService:
    """Inspect bounded observation work without accepting or dispatching runs."""

    def __init__(self, store: WorkshopEventStore, host_policy: CollaborationHostPolicy) -> None:
        self._store = store
        self._host_policy = host_policy

    async def synchronize_host_policy(self) -> None:
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            await synchronize_standing_observation_host_policy_in_transaction(
                connection,
                self._host_policy,
                occurred_at=datetime.now(UTC),
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise

    async def inspect(
        self,
        channel_id: ChannelId,
        agent_id: AgentId,
        *,
        current_at: datetime | None = None,
    ) -> tuple[StandingObservationState, ...]:
        await self._reconcile_age(channel_id, agent_id, current_at=current_at or datetime.now(UTC))
        async with self._store.connection.execute(
            "SELECT scope_kind, scope_id, delivered_through_event_position, "
            "pending_through_event_position, considered_through_event_position, "
            "oldest_pending_event_position, pending_message_count, latest_human_anchor_message_id, "
            "latest_human_anchor_event_position, not_before, lifecycle_state, overflowed_at, "
            "overflow_reason, overflow_from_event_position, overflow_through_event_position, "
            "state_version, last_event_position FROM channel_agent_observation_states "
            "WHERE channel_id = ? AND agent_id = ? ORDER BY oldest_pending_event_position, scope_id",
            (channel_id, agent_id),
        ) as cursor:
            rows = list(await cursor.fetchall())
        return tuple(self._state(channel_id, agent_id, row) for row in rows)

    async def pending_batches(self, state: StandingObservationState) -> tuple[StandingObservationBatch, ...]:
        if state.pending_message_count == 0:
            return ()
        scope_clause = "m.thread_root_id IS NULL" if state.scope_kind == "channel" else "m.thread_root_id = ?"
        parameters: list[object] = [state.agent_id, state.channel_id]
        if state.scope_kind == "thread":
            parameters.append(state.scope_id)
        parameters.extend((state.delivered_through_event_position, state.pending_through_event_position))
        async with self._store.connection.execute(
            "SELECT m.id, m.created_event_position FROM messages m "
            "WHERE m.author_principal_id != (SELECT principal_id FROM agents WHERE id = ?) "
            "AND m.channel_id = ? AND "
            + scope_clause
            + " AND m.created_event_position > ? AND m.created_event_position <= ? "
            "ORDER BY m.created_event_position, m.id",
            tuple(parameters),
        ) as cursor:
            rows = list(await cursor.fetchall())
        limit = self._host_policy.standing_participation.max_messages_per_observe_run
        batches: list[StandingObservationBatch] = []
        for offset in range(0, len(rows), limit):
            chunk = rows[offset : offset + limit]
            batches.append(
                StandingObservationBatch(
                    state.channel_id,
                    state.agent_id,
                    state.scope_id,
                    int(chunk[0][1]),
                    int(chunk[-1][1]),
                    tuple(MessageId(str(row[0])) for row in chunk),
                )
            )
        return tuple(batches)

    async def _reconcile_age(
        self,
        channel_id: ChannelId,
        agent_id: AgentId,
        *,
        current_at: datetime,
    ) -> None:
        if current_at.tzinfo is None or current_at.utcoffset() is None:
            raise ValueError("current_at must be timezone-aware")
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            async with connection.execute(
                "SELECT scope_id, oldest_pending_event_position, considered_through_event_position "
                "FROM channel_agent_observation_states WHERE channel_id = ? AND agent_id = ? "
                "AND lifecycle_state = 'pending' AND pending_message_count > 0",
                (channel_id, agent_id),
            ) as cursor:
                rows = list(await cursor.fetchall())
            maximum_age = timedelta(seconds=self._host_policy.standing_participation.max_pending_age_seconds)
            for row in rows:
                async with connection.execute(
                    "SELECT occurred_at FROM event_log WHERE position = ? AND event_type = ?",
                    (int(row[1]), WorkshopEventType.MESSAGE_CREATED.value),
                ) as cursor:
                    source = await cursor.fetchone()
                reason = "retention" if source is None else None
                if source is not None and current_at.astimezone(UTC) - _parse_timestamp(source[0]) > maximum_age:
                    reason = "pending_age"
                if reason is None:
                    continue
                await connection.execute(
                    "UPDATE channel_agent_observation_states SET lifecycle_state = 'paused_overflow', "
                    "overflowed_at = ?, overflow_reason = ?, overflow_from_event_position = ?, "
                    "overflow_through_event_position = ?, state_version = state_version + 1 "
                    "WHERE channel_id = ? AND agent_id = ? AND scope_id = ? AND lifecycle_state = 'pending'",
                    (
                        current_at.astimezone(UTC).isoformat(),
                        reason,
                        int(row[1]),
                        int(row[2]),
                        channel_id,
                        agent_id,
                        str(row[0]),
                    ),
                )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise

    @staticmethod
    def _state(channel_id: ChannelId, agent_id: AgentId, row: aiosqlite.Row) -> StandingObservationState:
        return StandingObservationState(
            channel_id=channel_id,
            agent_id=agent_id,
            scope_kind=str(row[0]),
            scope_id=str(row[1]),
            delivered_through_event_position=int(row[2]),
            pending_through_event_position=int(row[3]),
            considered_through_event_position=int(row[4]),
            oldest_pending_event_position=int(row[5]) if row[5] is not None else None,
            pending_message_count=int(row[6]),
            latest_human_anchor_message_id=MessageId(str(row[7])) if row[7] is not None else None,
            latest_human_anchor_event_position=int(row[8]) if row[8] is not None else None,
            not_before=_parse_timestamp(row[9]) if row[9] is not None else None,
            lifecycle_state=str(row[10]),
            overflowed_at=_parse_timestamp(row[11]) if row[11] is not None else None,
            overflow_reason=str(row[12]) if row[12] is not None else None,
            overflow_from_event_position=int(row[13]) if row[13] is not None else None,
            overflow_through_event_position=int(row[14]) if row[14] is not None else None,
            state_version=int(row[15]),
            last_event_position=int(row[16]),
        )
