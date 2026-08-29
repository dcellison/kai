"""Authorized, transport-independent reads of canonical Workshop timelines."""

from __future__ import annotations

import base64
import binascii
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from typing import Protocol

import aiosqlite

from kai.workshop.artifacts import ArtifactSummary, artifacts_for_messages
from kai.workshop.domain import ChannelId, MessageId, MessageMention, PrincipalId
from kai.workshop.store import WorkshopEventStore

_CURSOR_PREFIX = "v1."
_MAX_CURSOR_LENGTH = 512
_MAX_PAGE_SIZE = 100
_MAX_SQLITE_INTEGER = 2**63 - 1
_VISIBLE_MESSAGE_PREDICATE = (
    "NOT (p.kind = 'human' AND COALESCE(json_extract(e.metadata_json, '$.source'), '') = 'scheduled_job')"
)


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
    thread_root_id: MessageId | None
    body: str
    event_position: int
    created_at: datetime
    mentions: tuple[MessageMention, ...] = ()
    artifacts: tuple[ArtifactSummary, ...] = ()
    reply_count: int = 0
    latest_reply_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class TimelinePage:
    """A stable page from a bounded snapshot of one channel timeline.

    Exactly one pagination direction is populated: forward pages carry
    next_cursor, tail-first pages carry previous_cursor. Both are None on
    a page that exhausted its direction.
    """

    messages: tuple[TimelineMessage, ...]
    next_cursor: str | None
    through_position: int
    previous_cursor: str | None = None


@dataclass(frozen=True, slots=True)
class TimelineUpdateBatch:
    """Canonical messages after one resumable channel event position."""

    messages: tuple[TimelineMessage, ...]
    next_position: int


@dataclass(frozen=True, slots=True)
class ThreadTimelinePage:
    """One stable page of replies beneath an authorized root message."""

    root: TimelineMessage
    messages: tuple[TimelineMessage, ...]
    next_cursor: str | None
    through_position: int


@dataclass(frozen=True, slots=True)
class _CursorState:
    channel_id: ChannelId
    after_position: int
    through_position: int


@dataclass(frozen=True, slots=True)
class _TailCursorState:
    channel_id: ChannelId
    before_position: int
    through_position: int


@dataclass(frozen=True, slots=True)
class _ThreadCursorState:
    channel_id: ChannelId
    thread_root_id: MessageId
    after_position: int
    through_position: int


def _encode_payload(payload: dict[str, object]) -> str:
    raw = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    encoded = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
    return f"{_CURSOR_PREFIX}{encoded}"


def _encode_cursor(state: _CursorState) -> str:
    return _encode_payload(
        {
            "after_position": state.after_position,
            "channel_id": state.channel_id,
            "through_position": state.through_position,
        }
    )


def _encode_tail_cursor(state: _TailCursorState) -> str:
    return _encode_payload(
        {
            "before_position": state.before_position,
            "channel_id": state.channel_id,
            "through_position": state.through_position,
        }
    )


def _encode_thread_cursor(state: _ThreadCursorState) -> str:
    return _encode_payload(
        {
            "after_position": state.after_position,
            "channel_id": state.channel_id,
            "thread_root_id": state.thread_root_id,
            "through_position": state.through_position,
        }
    )


def _cursor_position(payload: dict[str, object], key: str) -> int:
    value = payload[key]
    if not isinstance(value, int) or isinstance(value, bool):
        raise TimelineCursorError("Invalid timeline cursor")
    return value


def _decode_cursor(cursor: str) -> _CursorState | _TailCursorState | _ThreadCursorState:
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
    # The key set is the direction marker: forward cursors carry
    # after_position, tail cursors before_position. Anything else fails
    # closed as malformed.
    if not isinstance(payload, dict):
        raise TimelineCursorError("Invalid timeline cursor")
    keys = set(payload)
    if keys == {"after_position", "channel_id", "through_position"} or keys == {
        "after_position",
        "channel_id",
        "thread_root_id",
        "through_position",
    }:
        boundary_key = "after_position"
    elif keys == {"before_position", "channel_id", "through_position"}:
        boundary_key = "before_position"
    else:
        raise TimelineCursorError("Invalid timeline cursor")
    boundary = _cursor_position(payload, boundary_key)
    through_position = _cursor_position(payload, "through_position")
    if boundary < 0 or through_position < boundary or through_position > _MAX_SQLITE_INTEGER:
        raise TimelineCursorError("Invalid timeline cursor")
    try:
        channel_id = ChannelId(payload["channel_id"])
    except (TypeError, ValueError) as exc:
        raise TimelineCursorError("Invalid timeline cursor") from exc
    if "thread_root_id" in payload:
        try:
            thread_root_id = MessageId(payload["thread_root_id"])
        except (TypeError, ValueError) as exc:
            raise TimelineCursorError("Invalid timeline cursor") from exc
        return _ThreadCursorState(channel_id, thread_root_id, boundary, through_position)
    if boundary_key == "after_position":
        return _CursorState(channel_id, boundary, through_position)
    # A tail cursor's boundary is the position of a message already
    # returned, so zero can never be legitimate: positions start at one.
    if boundary == 0:
        raise TimelineCursorError("Invalid timeline cursor")
    return _TailCursorState(channel_id, boundary, through_position)


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def is_internal_scheduled_invocation(metadata_json: object, author_kind: object) -> bool:
    """Return whether a canonical message is an internal scheduled command.

    Scheduled agent work is represented by a human-authored canonical message
    so the run keeps durable ownership, context, and audit provenance. That
    record is not a message the human sent from a client and must not be echoed
    into client timelines. Scheduled reminders use the same source metadata
    but are agent-authored user-visible output, so author kind is part of the
    classification.
    """
    if str(author_kind) != "human":
        return False
    try:
        metadata = json.loads(str(metadata_json))
    except json.JSONDecodeError:
        return False
    return isinstance(metadata, dict) and metadata.get("source") == "scheduled_job"


def parse_message_mentions_json(value: object) -> tuple[MessageMention, ...]:
    """Decode mentions persisted by the canonical message projection."""
    try:
        raw_mentions = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise RuntimeError("Canonical message mentions are malformed") from exc
    if not isinstance(raw_mentions, list):
        raise RuntimeError("Canonical message mentions are malformed")
    mentions: list[MessageMention] = []
    for raw in raw_mentions:
        if not isinstance(raw, dict) or set(raw) != {"principal_id", "kind", "start", "length"}:
            raise RuntimeError("Canonical message mention is malformed")
        try:
            mentions.append(
                MessageMention(
                    principal_id=PrincipalId(raw["principal_id"]),
                    kind=raw["kind"],
                    start=raw["start"],
                    length=raw["length"],
                )
            )
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Canonical message mention is malformed") from exc
    return tuple(mentions)


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
    messages: list[TimelineMessage] = []
    for row in rows:
        if is_internal_scheduled_invocation(row[11], row[3]):
            continue
        messages.append(
            TimelineMessage(
                message_id=MessageId(str(row[0])),
                channel_id=ChannelId(str(row[1])),
                author_principal_id=PrincipalId(str(row[2])),
                author_kind=str(row[3]),
                author_display_name=str(row[4]),
                reply_to_message_id=MessageId(str(row[5])) if row[5] is not None else None,
                thread_root_id=MessageId(str(row[6])) if row[6] is not None else None,
                body=str(row[7]),
                event_position=int(row[8]),
                created_at=_parse_timestamp(str(row[9])),
                mentions=parse_message_mentions_json(row[10]),
                reply_count=int(row[12]),
                latest_reply_at=(_parse_timestamp(str(row[13])) if row[13] is not None else None),
            )
        )
    return tuple(messages)


async def attach_message_artifacts(
    store: WorkshopEventStore,
    messages: tuple[TimelineMessage, ...],
) -> tuple[TimelineMessage, ...]:
    grouped = await artifacts_for_messages(
        store,
        tuple(message.message_id for message in messages),
    )
    return tuple(replace(message, artifacts=grouped.get(message.message_id, ())) for message in messages)


async def _read_forward_page(
    store: WorkshopEventStore,
    channel_id: ChannelId,
    state: _CursorState,
    limit: int,
) -> TimelinePage:
    async with store.connection.execute(
        "SELECT m.id, m.channel_id, m.author_principal_id, p.kind, p.display_name, "
        "m.reply_to_message_id, m.thread_root_id, m.body, m.created_event_position, m.created_at, "
        "m.mentions_json, e.metadata_json, "
        "(SELECT COUNT(*) FROM messages tr WHERE tr.thread_root_id = m.id "
        "AND tr.created_event_position <= ?), "
        "(SELECT MAX(tr.created_at) FROM messages tr WHERE tr.thread_root_id = m.id "
        "AND tr.created_event_position <= ?) "
        "FROM messages m JOIN principals p ON p.id = m.author_principal_id "
        "JOIN event_log e ON e.position = m.created_event_position "
        "WHERE m.channel_id = ? AND m.thread_root_id IS NULL AND m.created_event_position > ? "
        f"AND m.created_event_position <= ? AND {_VISIBLE_MESSAGE_PREDICATE} "
        "ORDER BY m.created_event_position ASC LIMIT ?",
        (
            state.through_position,
            state.through_position,
            channel_id,
            state.after_position,
            state.through_position,
            limit + 1,
        ),
    ) as query_cursor:
        rows = list(await query_cursor.fetchall())

    has_more = len(rows) > limit
    page_rows = rows[:limit]
    messages = await attach_message_artifacts(store, _messages_from_rows(page_rows))
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


async def _read_tail_page(
    store: WorkshopEventStore,
    channel_id: ChannelId,
    state: _TailCursorState,
    limit: int,
) -> TimelinePage:
    # The strict upper bound alone keeps the page inside the snapshot:
    # decoded cursors guarantee before_position <= through_position, and
    # the synthetic initial state uses through_position + 1 so the bound
    # itself is included. Rows come back newest-first to take the page
    # nearest the boundary, then flip to ascending so every page reads
    # in event order.
    async with store.connection.execute(
        "SELECT m.id, m.channel_id, m.author_principal_id, p.kind, p.display_name, "
        "m.reply_to_message_id, m.thread_root_id, m.body, m.created_event_position, m.created_at, "
        "m.mentions_json, e.metadata_json, "
        "(SELECT COUNT(*) FROM messages tr WHERE tr.thread_root_id = m.id "
        "AND tr.created_event_position <= ?), "
        "(SELECT MAX(tr.created_at) FROM messages tr WHERE tr.thread_root_id = m.id "
        "AND tr.created_event_position <= ?) "
        "FROM messages m JOIN principals p ON p.id = m.author_principal_id "
        "JOIN event_log e ON e.position = m.created_event_position "
        "WHERE m.channel_id = ? AND m.thread_root_id IS NULL AND m.created_event_position < ? "
        f"AND {_VISIBLE_MESSAGE_PREDICATE} ORDER BY m.created_event_position DESC LIMIT ?",
        (state.through_position, state.through_position, channel_id, state.before_position, limit + 1),
    ) as query_cursor:
        rows = list(await query_cursor.fetchall())

    has_more = len(rows) > limit
    page_rows = list(reversed(rows[:limit]))
    messages = await attach_message_artifacts(store, _messages_from_rows(page_rows))
    previous_cursor = None
    if has_more:
        previous_cursor = _encode_tail_cursor(
            _TailCursorState(
                channel_id=channel_id,
                before_position=messages[0].event_position,
                through_position=state.through_position,
            )
        )
    return TimelinePage(messages, None, state.through_position, previous_cursor)


async def read_channel_timeline(
    store: WorkshopEventStore,
    *,
    principal_id: PrincipalId,
    channel_id: ChannelId,
    authorizer: ChannelTimelineAuthorizer,
    cursor: str | None = None,
    limit: int = 50,
    tail: bool = False,
) -> TimelinePage:
    """Read an authorized, stable snapshot page from one canonical channel.

    Without a cursor, ``tail=False`` starts a forward walk from the start
    of the snapshot and ``tail=True`` returns the newest page, whose
    previous_cursor walks earlier history under the same snapshot bound.
    A cursor carries its own direction, so ``tail`` must stay False when
    one is supplied.
    """
    _validate_request(principal_id, channel_id, limit)
    if tail and cursor is not None:
        raise ValueError("tail requests must not carry a cursor")
    await _authorize(authorizer, principal_id, channel_id)

    if not await _channel_exists(store, channel_id):
        raise TimelineAccessDeniedError("Timeline access denied")

    if cursor is not None:
        state = _decode_cursor(cursor)
        if state.channel_id != channel_id:
            raise TimelineCursorError("Timeline cursor belongs to another channel")
        if isinstance(state, _ThreadCursorState):
            raise TimelineCursorError("Thread cursor cannot page a channel timeline")
        if isinstance(state, _TailCursorState):
            return await _read_tail_page(store, channel_id, state, limit)
        return await _read_forward_page(store, channel_id, state, limit)

    through_position = await _latest_message_position(store, channel_id)
    if tail:
        # The initial tail page has no boundary message yet; one past the
        # snapshot bound makes the strict inequality include the bound.
        return await _read_tail_page(
            store,
            channel_id,
            _TailCursorState(channel_id, through_position + 1, through_position),
            limit,
        )
    return await _read_forward_page(store, channel_id, _CursorState(channel_id, 0, through_position), limit)


async def _read_thread_root(
    store: WorkshopEventStore,
    channel_id: ChannelId,
    thread_root_id: MessageId,
    through_position: int,
) -> TimelineMessage:
    async with store.connection.execute(
        "SELECT m.id, m.channel_id, m.author_principal_id, p.kind, p.display_name, "
        "m.reply_to_message_id, m.thread_root_id, m.body, m.created_event_position, m.created_at, "
        "m.mentions_json, e.metadata_json, "
        "(SELECT COUNT(*) FROM messages tr WHERE tr.thread_root_id = m.id "
        "AND tr.created_event_position <= ?), "
        "(SELECT MAX(tr.created_at) FROM messages tr WHERE tr.thread_root_id = m.id "
        "AND tr.created_event_position <= ?) "
        "FROM messages m JOIN principals p ON p.id = m.author_principal_id "
        "JOIN event_log e ON e.position = m.created_event_position "
        "JOIN channels c ON c.id = m.channel_id "
        "WHERE m.id = ? AND m.channel_id = ? AND m.thread_root_id IS NULL "
        "AND c.kind = 'group' AND m.created_event_position <= ?",
        (through_position, through_position, thread_root_id, channel_id, through_position),
    ) as cursor:
        rows = list(await cursor.fetchall())
    messages = await attach_message_artifacts(store, _messages_from_rows(rows))
    if len(messages) != 1:
        raise TimelineAccessDeniedError("Thread access denied")
    return messages[0]


async def read_thread_timeline(
    store: WorkshopEventStore,
    *,
    principal_id: PrincipalId,
    channel_id: ChannelId,
    thread_root_id: MessageId,
    authorizer: ChannelTimelineAuthorizer,
    cursor: str | None = None,
    limit: int = 50,
) -> ThreadTimelinePage:
    """Read one authorized, stable, forward-paged group-channel thread."""
    _validate_request(principal_id, channel_id, limit)
    if not isinstance(thread_root_id, MessageId):
        raise ValueError("thread_root_id must be a MessageId")
    await _authorize(authorizer, principal_id, channel_id)
    if not await _channel_exists(store, channel_id):
        raise TimelineAccessDeniedError("Thread access denied")

    if cursor is None:
        through_position = await _latest_message_position(store, channel_id)
        state = _ThreadCursorState(channel_id, thread_root_id, 0, through_position)
    else:
        decoded = _decode_cursor(cursor)
        if not isinstance(decoded, _ThreadCursorState):
            raise TimelineCursorError("Channel cursor cannot page a thread timeline")
        if decoded.channel_id != channel_id or decoded.thread_root_id != thread_root_id:
            raise TimelineCursorError("Thread cursor belongs to another thread")
        state = decoded

    root = await _read_thread_root(store, channel_id, thread_root_id, state.through_position)
    async with store.connection.execute(
        "SELECT m.id, m.channel_id, m.author_principal_id, p.kind, p.display_name, "
        "m.reply_to_message_id, m.thread_root_id, m.body, m.created_event_position, m.created_at, "
        "m.mentions_json, e.metadata_json, 0, NULL "
        "FROM messages m JOIN principals p ON p.id = m.author_principal_id "
        "JOIN event_log e ON e.position = m.created_event_position "
        "WHERE m.channel_id = ? AND m.thread_root_id = ? "
        "AND m.created_event_position > ? AND m.created_event_position <= ? "
        f"AND {_VISIBLE_MESSAGE_PREDICATE} "
        "ORDER BY m.created_event_position ASC LIMIT ?",
        (channel_id, thread_root_id, state.after_position, state.through_position, limit + 1),
    ) as query_cursor:
        rows = list(await query_cursor.fetchall())
    has_more = len(rows) > limit
    messages = await attach_message_artifacts(store, _messages_from_rows(rows[:limit]))
    next_cursor = None
    if has_more:
        next_cursor = _encode_thread_cursor(
            _ThreadCursorState(
                channel_id,
                thread_root_id,
                messages[-1].event_position,
                state.through_position,
            )
        )
    return ThreadTimelinePage(root, messages, next_cursor, state.through_position)


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
        "m.reply_to_message_id, m.thread_root_id, m.body, m.created_event_position, m.created_at, "
        "m.mentions_json, e.metadata_json, "
        "(SELECT COUNT(*) FROM messages tr WHERE tr.thread_root_id = m.id), "
        "(SELECT MAX(tr.created_at) FROM messages tr WHERE tr.thread_root_id = m.id) "
        "FROM messages m JOIN principals p ON p.id = m.author_principal_id "
        "JOIN event_log e ON e.position = m.created_event_position "
        "WHERE m.channel_id = ? AND m.created_event_position > ? "
        "ORDER BY m.created_event_position ASC LIMIT ?",
        (channel_id, after_position, limit),
    ) as query_cursor:
        rows = list(await query_cursor.fetchall())

    messages = await attach_message_artifacts(store, _messages_from_rows(rows))
    next_position = int(rows[-1][8]) if rows else after_position
    return TimelineUpdateBatch(messages, next_position)
