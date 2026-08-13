"""Authorized, ordered Workshop events exposed to interactive clients."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from kai.workshop.domain import ChannelId, MessageId, PrincipalId, RunId, WorkshopEventType
from kai.workshop.run_lifecycle import DurableRun, load_durable_run
from kai.workshop.store import WorkshopEventStore
from kai.workshop.timeline import (
    ChannelTimelineAuthorizer,
    TimelineAccessDeniedError,
    TimelineMessage,
    TimelineResumeError,
)

_MAX_BATCH_SIZE = 100
_MAX_SQLITE_INTEGER = 2**63 - 1
_RUN_EVENT_TYPES = (
    WorkshopEventType.RUN_ACCEPTED,
    WorkshopEventType.RUN_STARTED,
    WorkshopEventType.RUN_CANCELLATION_REQUESTED,
    WorkshopEventType.RUN_COMPLETED,
    WorkshopEventType.RUN_FAILED,
    WorkshopEventType.RUN_CANCELLED,
)


@dataclass(frozen=True, slots=True)
class ClientTimelineMessageEvent:
    message: TimelineMessage

    @property
    def event_position(self) -> int:
        return self.message.event_position


@dataclass(frozen=True, slots=True)
class ClientRunLifecycleEvent:
    run: DurableRun
    transition: WorkshopEventType
    event_position: int
    occurred_at: datetime


type ClientChannelEvent = ClientTimelineMessageEvent | ClientRunLifecycleEvent


@dataclass(frozen=True, slots=True)
class ClientChannelEventBatch:
    events: tuple[ClientChannelEvent, ...]
    next_position: int


def _parse_timestamp(value: object) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def _validate_request(
    principal_id: PrincipalId,
    channel_id: ChannelId,
    after_position: int | None,
    limit: int,
) -> None:
    if not isinstance(principal_id, PrincipalId):
        raise ValueError("principal_id must be a PrincipalId")
    if not isinstance(channel_id, ChannelId):
        raise ValueError("channel_id must be a ChannelId")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= _MAX_BATCH_SIZE:
        raise ValueError(f"limit must be an integer from 1 through {_MAX_BATCH_SIZE}")
    if after_position is not None and (
        not isinstance(after_position, int)
        or isinstance(after_position, bool)
        or after_position < 0
        or after_position > _MAX_SQLITE_INTEGER
    ):
        raise TimelineResumeError("Invalid client event resume position")


async def _latest_event_position(store: WorkshopEventStore) -> int:
    async with store.connection.execute("SELECT COALESCE(MAX(position), 0) FROM event_log") as cursor:
        row = await cursor.fetchone()
    if row is None:
        raise RuntimeError("Event-log boundary query returned no row")
    return int(row[0])


async def _latest_relevant_position(
    store: WorkshopEventStore,
    principal_id: PrincipalId,
    channel_id: ChannelId,
) -> int:
    placeholders = ", ".join("?" for _ in _RUN_EVENT_TYPES)
    parameters = (
        channel_id,
        *tuple(event_type.value for event_type in _RUN_EVENT_TYPES),
        channel_id,
        principal_id,
    )
    async with store.connection.execute(
        "SELECT COALESCE(MAX(e.position), 0) FROM event_log e "
        "LEFT JOIN messages m ON m.created_event_position = e.position "
        "LEFT JOIN runs r ON r.id = e.aggregate_id "
        "WHERE m.channel_id = ? OR (e.aggregate_type = 'run' "
        f"AND e.event_type IN ({placeholders}) AND r.channel_id = ? "
        "AND r.requested_by_principal_id = ?)",
        parameters,
    ) as cursor:
        row = await cursor.fetchone()
    if row is None:
        raise RuntimeError("Client event boundary query returned no row")
    return int(row[0])


async def read_client_channel_events(
    store: WorkshopEventStore,
    *,
    principal_id: PrincipalId,
    channel_id: ChannelId,
    authorizer: ChannelTimelineAuthorizer,
    after_position: int | None,
    limit: int = 100,
) -> ClientChannelEventBatch:
    """Read resumable message and private run activity from one channel.

    Message events follow channel read authorization. Run lifecycle events are
    additionally restricted to the requesting human, matching the existing
    run inspection endpoint. ``None`` begins at the current relevant boundary
    so opening a stream never replays historical activity unexpectedly.
    """
    _validate_request(principal_id, channel_id, after_position, limit)
    if await authorizer.can_read_channel(principal_id, channel_id) is not True:
        raise TimelineAccessDeniedError("Timeline access denied")
    async with store.connection.execute("SELECT 1 FROM channels WHERE id = ?", (channel_id,)) as cursor:
        if await cursor.fetchone() is None:
            raise TimelineAccessDeniedError("Timeline access denied")

    if after_position is None:
        return ClientChannelEventBatch(
            (),
            await _latest_relevant_position(store, principal_id, channel_id),
        )
    if after_position > await _latest_event_position(store):
        raise TimelineResumeError("Client event resume position is ahead of the event log")

    placeholders = ", ".join("?" for _ in _RUN_EVENT_TYPES)
    parameters = (
        after_position,
        channel_id,
        *tuple(event_type.value for event_type in _RUN_EVENT_TYPES),
        channel_id,
        principal_id,
        limit,
    )
    async with store.connection.execute(
        "SELECT e.position, e.event_type, e.occurred_at, "
        "m.id AS message_id, m.channel_id AS message_channel_id, "
        "m.author_principal_id, p.kind AS author_kind, p.display_name AS author_display_name, "
        "m.reply_to_message_id, m.body, m.created_at AS message_created_at, "
        "r.id AS run_id "
        "FROM event_log e "
        "LEFT JOIN messages m ON m.created_event_position = e.position "
        "LEFT JOIN principals p ON p.id = m.author_principal_id "
        "LEFT JOIN runs r ON r.id = e.aggregate_id "
        "WHERE e.position > ? AND (m.channel_id = ? OR (e.aggregate_type = 'run' "
        f"AND e.event_type IN ({placeholders}) AND r.channel_id = ? "
        "AND r.requested_by_principal_id = ?)) "
        "ORDER BY e.position ASC LIMIT ?",
        parameters,
    ) as cursor:
        rows = list(await cursor.fetchall())

    events: list[ClientChannelEvent] = []
    for row in rows:
        position = int(row["position"])
        if row["message_id"] is not None:
            events.append(
                ClientTimelineMessageEvent(
                    TimelineMessage(
                        message_id=MessageId(str(row["message_id"])),
                        channel_id=ChannelId(str(row["message_channel_id"])),
                        author_principal_id=PrincipalId(str(row["author_principal_id"])),
                        author_kind=str(row["author_kind"]),
                        author_display_name=str(row["author_display_name"]),
                        reply_to_message_id=(
                            MessageId(str(row["reply_to_message_id"]))
                            if row["reply_to_message_id"] is not None
                            else None
                        ),
                        body=str(row["body"]),
                        event_position=position,
                        created_at=_parse_timestamp(row["message_created_at"]),
                    )
                )
            )
            continue

        transition = WorkshopEventType(str(row["event_type"]))
        run = await load_durable_run(store, RunId(str(row["run_id"])))
        if run is None:
            raise RuntimeError("Projected client run disappeared")
        events.append(
            ClientRunLifecycleEvent(
                run=run,
                transition=transition,
                event_position=position,
                occurred_at=_parse_timestamp(row["occurred_at"]),
            )
        )

    next_position = events[-1].event_position if events else after_position
    return ClientChannelEventBatch(tuple(events), next_position)
