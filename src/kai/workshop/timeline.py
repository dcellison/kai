"""Authorized, transport-independent reads of canonical Workshop timelines."""

from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import aiosqlite

from kai.workshop.domain import ChannelId, MessageId, PrincipalId
from kai.workshop.store import WorkshopEventStore

_CURSOR_PREFIX = "v1."
_MAX_CURSOR_LENGTH = 512
_MAX_PAGE_SIZE = 100
_MAX_SQLITE_INTEGER = 2**63 - 1


class TimelineAccessDeniedError(PermissionError):
    """The principal may not read the requested channel timeline."""


class TimelineCursorError(ValueError):
    """A timeline pagination cursor is invalid for the requested channel."""


class TimelineResumeError(ValueError):
    """A live timeline position cannot be resumed against this event store."""


class ChannelTimelineAuthorizer(Protocol):
    """Authorize a principal against one specific canonical channel."""

    async def can_read_channel(self, principal_id: PrincipalId, channel_id: ChannelId) -> bool: ...


@dataclass(frozen=True, slots=True)
class TimelineMessage:
    """One canonical message suitable for a transport-neutral timeline."""

    message_id: MessageId
    channel_id: ChannelId
    author_principal_id: PrincipalId
    author_kind: str
    author_display_name: str
    reply_to_message_id: MessageId | None
    body: str
    event_position: int
    created_at: datetime


@dataclass(frozen=True, slots=True)
class TimelinePage:
    """A stable page from a bounded snapshot of one channel timeline."""

    messages: tuple[TimelineMessage, ...]
    next_cursor: str | None
    through_position: int


@dataclass(frozen=True, slots=True)
class TimelineUpdateBatch:
    """Canonical messages after one resumable channel event position."""

    messages: tuple[TimelineMessage, ...]
    next_position: int


@dataclass(frozen=True, slots=True)
class _CursorState:
    channel_id: ChannelId
    after_position: int
    through_position: int


def _encode_cursor(state: _CursorState) -> str:
    payload = json.dumps(
        {
            "after_position": state.after_position,
            "channel_id": state.channel_id,
            "through_position": state.through_position,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    encoded = base64.urlsafe_b64encode(payload).rstrip(b"=").decode("ascii")
    return f"{_CURSOR_PREFIX}{encoded}"


def _decode_cursor(cursor: str) -> _CursorState:
    if not isinstance(cursor, str) or not cursor.startswith(_CURSOR_PREFIX) or len(cursor) > _MAX_CURSOR_LENGTH:
        raise TimelineCursorError("Invalid timeline cursor")
    encoded = cursor.removeprefix(_CURSOR_PREFIX)
    if not encoded:
        raise TimelineCursorError("Invalid timeline cursor")
    padding = "=" * (-len(encoded) % 4)
    try:
        raw = base64.b64decode(encoded + padding, altchars=b"-_", validate=True)
        payload = json.loads(raw.decode("utf-8"))
    except (binascii.Error, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise TimelineCursorError("Invalid timeline cursor") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "after_position",
        "channel_id",
        "through_position",
    }:
        raise TimelineCursorError("Invalid timeline cursor")
    after_position = payload["after_position"]
    through_position = payload["through_position"]
    if (
        not isinstance(after_position, int)
        or isinstance(after_position, bool)
        or not isinstance(through_position, int)
        or isinstance(through_position, bool)
        or after_position < 0
        or through_position < after_position
        or through_position > _MAX_SQLITE_INTEGER
    ):
        raise TimelineCursorError("Invalid timeline cursor")
    try:
        channel_id = ChannelId(payload["channel_id"])
    except (TypeError, ValueError) as exc:
        raise TimelineCursorError("Invalid timeline cursor") from exc
    return _CursorState(channel_id, after_position, through_position)


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _validate_request(principal_id: PrincipalId, channel_id: ChannelId, limit: int) -> None:
    if not isinstance(principal_id, PrincipalId):
        raise ValueError("principal_id must be a PrincipalId")
    if not isinstance(channel_id, ChannelId):
        raise ValueError("channel_id must be a ChannelId")
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= _MAX_PAGE_SIZE:
        raise ValueError(f"limit must be an integer from 1 through {_MAX_PAGE_SIZE}")


async def _authorize(
    authorizer: ChannelTimelineAuthorizer,
    principal_id: PrincipalId,
    channel_id: ChannelId,
) -> None:
    allowed = await authorizer.can_read_channel(principal_id, channel_id)
    if allowed is not True:
        raise TimelineAccessDeniedError("Timeline access denied")


async def _channel_exists(store: WorkshopEventStore, channel_id: ChannelId) -> bool:
    async with store.connection.execute("SELECT 1 FROM channels WHERE id = ?", (channel_id,)) as cursor:
        return await cursor.fetchone() is not None


async def _latest_message_position(store: WorkshopEventStore, channel_id: ChannelId) -> int:
    async with store.connection.execute(
        "SELECT COALESCE(MAX(created_event_position), 0) FROM messages WHERE channel_id = ?",
        (channel_id,),
    ) as cursor:
        row = await cursor.fetchone()
    if row is None:
        raise RuntimeError("Timeline boundary query returned no row")
    return int(row[0])


async def _latest_event_position(store: WorkshopEventStore) -> int:
    async with store.connection.execute("SELECT COALESCE(MAX(position), 0) FROM event_log") as cursor:
        row = await cursor.fetchone()
    if row is None:
        raise RuntimeError("Event-log boundary query returned no row")
    return int(row[0])


def _messages_from_rows(rows: list[aiosqlite.Row]) -> tuple[TimelineMessage, ...]:
    return tuple(
        TimelineMessage(
            message_id=MessageId(str(row[0])),
            channel_id=ChannelId(str(row[1])),
            author_principal_id=PrincipalId(str(row[2])),
            author_kind=str(row[3]),
            author_display_name=str(row[4]),
            reply_to_message_id=MessageId(str(row[5])) if row[5] is not None else None,
            body=str(row[6]),
            event_position=int(row[7]),
            created_at=_parse_timestamp(str(row[8])),
        )
        for row in rows
    )


async def read_channel_timeline(
    store: WorkshopEventStore,
    *,
    principal_id: PrincipalId,
    channel_id: ChannelId,
    authorizer: ChannelTimelineAuthorizer,
    cursor: str | None = None,
    limit: int = 50,
) -> TimelinePage:
    """Read an authorized, stable snapshot page from one canonical channel."""
    _validate_request(principal_id, channel_id, limit)
    await _authorize(authorizer, principal_id, channel_id)

    if not await _channel_exists(store, channel_id):
        raise TimelineAccessDeniedError("Timeline access denied")

    if cursor is None:
        state = _CursorState(channel_id, 0, await _latest_message_position(store, channel_id))
    else:
        state = _decode_cursor(cursor)
        if state.channel_id != channel_id:
            raise TimelineCursorError("Timeline cursor belongs to another channel")

    async with store.connection.execute(
        "SELECT m.id, m.channel_id, m.author_principal_id, p.kind, p.display_name, "
        "m.reply_to_message_id, m.body, m.created_event_position, m.created_at "
        "FROM messages m JOIN principals p ON p.id = m.author_principal_id "
        "WHERE m.channel_id = ? AND m.created_event_position > ? "
        "AND m.created_event_position <= ? ORDER BY m.created_event_position ASC LIMIT ?",
        (channel_id, state.after_position, state.through_position, limit + 1),
    ) as query_cursor:
        rows = list(await query_cursor.fetchall())

    has_more = len(rows) > limit
    page_rows = rows[:limit]
    messages = _messages_from_rows(page_rows)
    next_cursor = None
    if has_more:
        next_cursor = _encode_cursor(
            _CursorState(
                channel_id=channel_id,
                after_position=messages[-1].event_position,
                through_position=state.through_position,
            )
        )
    return TimelinePage(messages, next_cursor, state.through_position)


async def read_channel_timeline_updates(
    store: WorkshopEventStore,
    *,
    principal_id: PrincipalId,
    channel_id: ChannelId,
    authorizer: ChannelTimelineAuthorizer,
    after_position: int | None,
    limit: int = 100,
) -> TimelineUpdateBatch:
    """Read an authorized resumable batch of new canonical channel messages.

    ``None`` begins at the current channel boundary, which makes a new stream
    future-only. Supplying a prior SSE event position replays every later
    canonical message in order. Positions ahead of the durable event log fail
    closed so a restored or mismatched client is forced through timeline
    resynchronization instead of silently missing future messages.
    """
    _validate_request(principal_id, channel_id, limit)
    if after_position is not None and (
        not isinstance(after_position, int)
        or isinstance(after_position, bool)
        or after_position < 0
        or after_position > _MAX_SQLITE_INTEGER
    ):
        raise TimelineResumeError("Invalid timeline resume position")
    await _authorize(authorizer, principal_id, channel_id)

    if not await _channel_exists(store, channel_id):
        raise TimelineAccessDeniedError("Timeline access denied")

    if after_position is None:
        return TimelineUpdateBatch((), await _latest_message_position(store, channel_id))
    if after_position > await _latest_event_position(store):
        raise TimelineResumeError("Timeline resume position is ahead of the event log")

    async with store.connection.execute(
        "SELECT m.id, m.channel_id, m.author_principal_id, p.kind, p.display_name, "
        "m.reply_to_message_id, m.body, m.created_event_position, m.created_at "
        "FROM messages m JOIN principals p ON p.id = m.author_principal_id "
        "WHERE m.channel_id = ? AND m.created_event_position > ? "
        "ORDER BY m.created_event_position ASC LIMIT ?",
        (channel_id, after_position, limit),
    ) as query_cursor:
        rows = list(await query_cursor.fetchall())

    messages = _messages_from_rows(rows)
    next_position = messages[-1].event_position if messages else after_position
    return TimelineUpdateBatch(messages, next_position)
