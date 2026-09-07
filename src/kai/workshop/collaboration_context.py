"""Attempt-scoped, bounded reads of canonical Workshop conversation context."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from kai.backend import TraceEntry
from kai.workshop.collaboration_authority import (
    CollaborationAuthorization,
    CollaborationBaseIdentity,
    CollaborationOperation,
)
from kai.workshop.domain import ChannelId, PrincipalId
from kai.workshop.run_execution_authority import RunExecutionClaim
from kai.workshop.run_traces import WorkshopRunTraceStore
from kai.workshop.store import WorkshopEventStore
from kai.workshop.timeline import (
    ChannelTimelineAuthorizer,
    TimelineMessage,
    read_channel_timeline,
    read_thread_timeline,
)

_IDEMPOTENCY_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_MAX_CURSOR_CHARACTERS = 512
_MAX_PAGE_MESSAGES = 20
_MAX_MESSAGE_CHARACTERS = 4_000
_MAX_MENTIONS_PER_MESSAGE = 32
_MAX_ARTIFACTS_PER_MESSAGE = 8
_MAX_ROSTER_MEMBERS = 100
_READ_TIMEOUT_SECONDS = 2.0


class CollaborationContextError(RuntimeError):
    """A bounded context request could not be served."""


class CollaborationContextValidationError(CollaborationContextError):
    """The request shape or cursor is invalid."""


@dataclass(frozen=True, slots=True)
class CollaborationContextResult:
    """One bounded page whose authority and snapshot are server-derived."""

    payload: dict[str, object]


class _ExecutionService(Protocol):
    async def authorize_collaboration(
        self,
        proof: str,
        operation: CollaborationOperation,
        *,
        base_identity: CollaborationBaseIdentity,
        idempotency_key: str,
        request_hash: str,
        occurred_at: datetime,
    ) -> CollaborationAuthorization: ...


@dataclass(frozen=True, slots=True)
class _ExactChannelAuthorizer(ChannelTimelineAuthorizer):
    principal_id: PrincipalId
    channel_id: ChannelId

    async def can_read_channel(self, principal_id: PrincipalId, channel_id: ChannelId) -> bool:
        return principal_id == self.principal_id and channel_id == self.channel_id


def _normalize_cursor(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value or len(value) > _MAX_CURSOR_CHARACTERS:
        raise CollaborationContextValidationError("cursor must be a bounded opaque string")
    return value


def _normalize_limit(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= _MAX_PAGE_MESSAGES:
        raise CollaborationContextValidationError(f"limit must be an integer from 1 through {_MAX_PAGE_MESSAGES}")
    return value


def _normalize_idempotency_key(value: object) -> str:
    if not isinstance(value, str) or not _IDEMPOTENCY_KEY_PATTERN.fullmatch(value):
        raise CollaborationContextValidationError("idempotency_key must be a bounded opaque identifier")
    return value


def _request_hash(cursor: str | None, limit: int) -> str:
    encoded = json.dumps(
        {"cursor": cursor, "limit": limit},
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _bounded_message(message: TimelineMessage) -> tuple[dict[str, object], bool]:
    body_truncated = len(message.body) > _MAX_MESSAGE_CHARACTERS
    mentions_truncated = len(message.mentions) > _MAX_MENTIONS_PER_MESSAGE
    artifacts_truncated = len(message.artifacts) > _MAX_ARTIFACTS_PER_MESSAGE
    payload: dict[str, object] = {
        "message_id": str(message.message_id),
        "author": {
            "principal_id": str(message.author_principal_id),
            "kind": message.author_kind,
            "display_name": message.author_display_name,
        },
        "reply_to_message_id": (str(message.reply_to_message_id) if message.reply_to_message_id is not None else None),
        "thread_root_id": str(message.thread_root_id) if message.thread_root_id is not None else None,
        "body": message.body[:_MAX_MESSAGE_CHARACTERS],
        "body_truncated": body_truncated,
        "event_position": message.event_position,
        "created_at": _format_timestamp(message.created_at),
        "mentions": [
            {
                "principal_id": str(mention.principal_id),
                "kind": mention.kind,
                "start": mention.start,
                "length": mention.length,
            }
            for mention in message.mentions[:_MAX_MENTIONS_PER_MESSAGE]
        ],
        "artifacts": [
            {
                "artifact_id": str(artifact.artifact_id),
                "kind": artifact.kind,
                "media_type": artifact.media_type,
                "byte_size": artifact.byte_size,
                "content_sha256": artifact.content_sha256,
                "original_filename": artifact.original_filename,
                "created_at": _format_timestamp(artifact.created_at),
            }
            for artifact in message.artifacts[:_MAX_ARTIFACTS_PER_MESSAGE]
        ],
        "reactions": [
            {
                "reaction": reaction.reaction,
                "count": reaction.count,
                "reacted_by_requester": reaction.reacted_by_viewer,
            }
            for reaction in message.reactions
        ],
        "reply_count": message.reply_count,
        "latest_reply_at": (
            _format_timestamp(message.latest_reply_at) if message.latest_reply_at is not None else None
        ),
    }
    return payload, body_truncated or mentions_truncated or artifacts_truncated


class WorkshopCollaborationContextService:
    """Serve context from one exact active grant without accepting selectors."""

    def __init__(self, store: WorkshopEventStore, execution: _ExecutionService) -> None:
        self._store = store
        self._execution = execution
        self._traces = WorkshopRunTraceStore(store)

    async def read(
        self,
        base_identity: CollaborationBaseIdentity,
        *,
        proof: str,
        cursor: object,
        limit: object,
        idempotency_key: object,
    ) -> CollaborationContextResult:
        normalized_cursor = _normalize_cursor(cursor)
        normalized_limit = _normalize_limit(limit)
        normalized_key = _normalize_idempotency_key(idempotency_key)
        fingerprint = _request_hash(normalized_cursor, normalized_limit)
        now = datetime.now(UTC)
        async with asyncio.timeout(_READ_TIMEOUT_SECONDS):
            authorization = await self._execution.authorize_collaboration(
                proof,
                CollaborationOperation.CONTEXT_READ,
                base_identity=base_identity,
                idempotency_key=normalized_key,
                request_hash=fingerprint,
                occurred_at=now,
            )
            grant = authorization.grant
            snapshot_position, channel = await self._scope(grant.grant_id, grant.channel_id)
            authorizer = _ExactChannelAuthorizer(grant.requested_by_principal_id, grant.channel_id)

            root_payload: dict[str, object] | None = None
            truncated = False
            if grant.thread_root_id is not None:
                page = await read_thread_timeline(
                    self._store,
                    principal_id=grant.requested_by_principal_id,
                    channel_id=grant.channel_id,
                    thread_root_id=grant.thread_root_id,
                    authorizer=authorizer,
                    cursor=normalized_cursor,
                    limit=normalized_limit,
                    snapshot_through_position=snapshot_position,
                )
                root_payload, root_truncated = _bounded_message(page.root)
                message_payloads = []
                truncated = root_truncated
                for message in page.messages:
                    item, item_truncated = _bounded_message(message)
                    message_payloads.append(item)
                    truncated = truncated or item_truncated
                next_cursor = page.next_cursor
                context_kind = "thread"
            else:
                page = await read_channel_timeline(
                    self._store,
                    principal_id=grant.requested_by_principal_id,
                    channel_id=grant.channel_id,
                    authorizer=authorizer,
                    cursor=normalized_cursor,
                    limit=normalized_limit,
                    tail=normalized_cursor is None,
                    snapshot_through_position=snapshot_position,
                )
                message_payloads = []
                for message in page.messages:
                    item, item_truncated = _bounded_message(message)
                    message_payloads.append(item)
                    truncated = truncated or item_truncated
                next_cursor = page.previous_cursor or page.next_cursor
                context_kind = "channel"

            roster, roster_truncated = await self._roster(grant.channel_id, snapshot_position)
            truncated = truncated or roster_truncated
            # Fence the response against cancellation or supersession that raced
            # the bounded database read. An idempotent replay remains subject to
            # the current live-attempt check in the authority layer.
            await self._execution.authorize_collaboration(
                proof,
                CollaborationOperation.CONTEXT_READ,
                base_identity=base_identity,
                idempotency_key=normalized_key,
                request_hash=fingerprint,
                occurred_at=datetime.now(UTC),
            )
            payload: dict[str, object] = {
                "version": 1,
                "content_trust": "untrusted",
                "context_kind": context_kind,
                "channel": channel,
                "thread_root": root_payload,
                "roster": roster,
                "snapshot_through_event_position": snapshot_position,
                "messages": message_payloads,
                "next_cursor": next_cursor,
                "truncated": truncated,
                "limits": {
                    "max_page_messages": _MAX_PAGE_MESSAGES,
                    "max_message_characters": _MAX_MESSAGE_CHARACTERS,
                    "max_roster_members": _MAX_ROSTER_MEMBERS,
                    "timeout_seconds": _READ_TIMEOUT_SECONDS,
                },
            }
            await self._trace(
                authorization,
                normalized_key,
                requested_limit=normalized_limit,
                cursor_supplied=normalized_cursor is not None,
                payload=payload,
            )
            return CollaborationContextResult(payload)

    async def _scope(self, grant_id: object, channel_id: ChannelId) -> tuple[int, dict[str, object]]:
        async with self._store.connection.execute(
            "SELECT g.issued_event_position, c.id, c.kind, c.name "
            "FROM collaboration_grants g JOIN channels c ON c.id = g.channel_id "
            "WHERE g.id = ? AND g.channel_id = ? AND c.archived_at IS NULL",
            (grant_id, channel_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None or str(row[2]) not in {"group", "direct"}:
            raise CollaborationContextValidationError("The granted conversation context is unavailable")
        return int(row[0]), {
            "channel_id": str(row[1]),
            "kind": str(row[2]),
            "name": str(row[3]) if row[3] is not None else None,
        }

    async def _roster(self, channel_id: ChannelId, snapshot_position: int) -> tuple[list[dict[str, object]], bool]:
        async with self._store.connection.execute(
            "WITH ranked_memberships AS ("
            "SELECT json_extract(e.payload_json, '$.principal_id') AS principal_id, "
            "json_extract(e.payload_json, '$.role') AS role, e.event_type, "
            "ROW_NUMBER() OVER (PARTITION BY json_extract(e.payload_json, '$.principal_id') "
            "ORDER BY e.position DESC) AS state_rank "
            "FROM event_log e WHERE e.aggregate_type = 'channel_membership' "
            "AND json_extract(e.payload_json, '$.channel_id') = ? "
            "AND e.event_type IN ('channel.member_added', 'channel.member_removed') "
            "AND e.position <= ?"
            ") SELECT p.id, p.kind, p.display_name, hh.handle, rm.role, NULL "
            "FROM ranked_memberships rm JOIN principals p ON p.id = rm.principal_id "
            "JOIN channels c ON c.id = ? "
            "LEFT JOIN human_handles hh ON hh.workshop_id = c.workshop_id "
            "AND hh.principal_id = p.id WHERE rm.state_rank = 1 "
            "AND rm.event_type = 'channel.member_added' AND p.kind = 'human' "
            "UNION ALL "
            "SELECT p.id, p.kind, p.display_name, ad.handle, NULL, a.id "
            "FROM channel_agents ca JOIN agents a ON a.id = ca.agent_id "
            "JOIN principals p ON p.id = a.principal_id "
            "JOIN agent_definitions ad ON ad.agent_id = a.id "
            "WHERE ca.channel_id = ? AND ca.attached_event_position <= ? "
            "AND (ca.detached_event_position IS NULL OR ca.detached_event_position > ?) "
            "ORDER BY 2, 3, 1 LIMIT ?",
            (
                channel_id,
                snapshot_position,
                channel_id,
                channel_id,
                snapshot_position,
                snapshot_position,
                _MAX_ROSTER_MEMBERS + 1,
            ),
        ) as cursor:
            rows = list(await cursor.fetchall())
        truncated = len(rows) > _MAX_ROSTER_MEMBERS
        return [
            {
                "principal_id": str(row[0]),
                "kind": str(row[1]),
                "display_name": str(row[2]),
                "handle": str(row[3]) if row[3] is not None else None,
                "channel_role": str(row[4]) if row[4] is not None else None,
                "agent_id": str(row[5]) if row[5] is not None else None,
            }
            for row in rows[:_MAX_ROSTER_MEMBERS]
        ], truncated

    async def _trace(
        self,
        authorization: CollaborationAuthorization,
        idempotency_key: str,
        requested_limit: int,
        cursor_supplied: bool,
        payload: dict[str, object],
    ) -> None:
        grant = authorization.grant
        tool_use_id = f"context-read:{idempotency_key}"
        async with self._store.connection.execute(
            "SELECT 1 FROM run_traces WHERE run_id = ? AND tool_use_id = ? LIMIT 1",
            (grant.run_id, tool_use_id),
        ) as cursor:
            if await cursor.fetchone() is not None:
                return
        async with self._store.connection.execute(
            "SELECT lease_version FROM run_attempts WHERE id = ?",
            (grant.attempt_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return
        messages = payload["messages"]
        assert isinstance(messages, list)
        detail = {
            "grant_id": str(grant.grant_id),
            "context_kind": payload["context_kind"],
            "channel_id": str(grant.channel_id),
            "thread_root_id": str(grant.thread_root_id) if grant.thread_root_id is not None else None,
            "snapshot_through_event_position": payload["snapshot_through_event_position"],
            "requested_limit": requested_limit,
            "cursor_supplied": cursor_supplied,
            "result_count": len(messages),
            "has_more": payload["next_cursor"] is not None,
            "truncated": payload["truncated"],
        }
        await self._traces.append(
            RunExecutionClaim(
                grant.attempt_id,
                grant.run_id,
                grant.execution_owner_id,
                grant.fence_token,
                int(row[0]),
            ),
            TraceEntry(
                kind="tool_result",
                tool_use_id=tool_use_id,
                summary="Read bounded Workshop context",
                detail=json.dumps(detail, separators=(",", ":"), sort_keys=True),
            ),
            occurred_at=datetime.now(UTC),
        )
