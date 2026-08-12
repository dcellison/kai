"""Canonical collaboration projection for Workshop events."""

from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import aiosqlite

from kai.workshop.domain import AgentId, ChannelId, MessageId, PrincipalId, RunId, WorkshopEventType, WorkshopId
from kai.workshop.store import StoredEvent

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
_MEDIA_TYPE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9!#$&^_.+-]*/[a-z0-9][a-z0-9!#$&^_.+-]*$")
_RUN_TERMINAL_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Workshop event payload requires non-empty {key!r}")
    return value


def _parse_projection_timestamp(value: object) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def _require_exact_payload(payload: dict[str, Any], keys: set[str]) -> None:
    if set(payload) != keys:
        raise ValueError(f"Workshop event payload must contain exactly {sorted(keys)!r}")


async def _apply_run_event(connection: aiosqlite.Connection, event: StoredEvent) -> None:
    envelope = event.envelope
    if not isinstance(envelope.aggregate_id, RunId) or envelope.aggregate_type != "run":
        raise ValueError("Workshop run events require a typed run aggregate")

    occurred_at = envelope.occurred_at.isoformat()
    payload = envelope.payload
    if envelope.event_type == WorkshopEventType.RUN_ACCEPTED:
        _require_exact_payload(
            payload,
            {"inbound_message_id", "channel_id", "requested_by_principal_id", "agent_id"},
        )
        inbound_message_id = MessageId(_required_text(payload, "inbound_message_id"))
        channel_id = ChannelId(_required_text(payload, "channel_id"))
        requested_by = PrincipalId(_required_text(payload, "requested_by_principal_id"))
        agent_id = AgentId(_required_text(payload, "agent_id"))
        async with connection.execute(
            "SELECT c.workshop_id, m.channel_id, m.author_principal_id, ca.agent_id, m.created_at "
            "FROM messages m "
            "JOIN principals p ON p.id = m.author_principal_id AND p.kind = 'human' "
            "JOIN channels c ON c.id = m.channel_id "
            "JOIN channel_memberships cm ON cm.channel_id = m.channel_id "
            "AND cm.principal_id = m.author_principal_id "
            "JOIN channel_agents ca ON ca.channel_id = m.channel_id "
            "JOIN agents a ON a.id = ca.agent_id AND a.workshop_id = c.workshop_id "
            "WHERE m.id = ?",
            (inbound_message_id,),
        ) as cursor:
            rows = list(await cursor.fetchall())
        expected = (envelope.workshop_id, channel_id, requested_by, agent_id)
        if len(rows) != 1 or tuple(rows[0][:4]) != expected:
            raise ValueError("Workshop run must match one human message and attached channel agent")
        inbound_created_at = _parse_projection_timestamp(rows[0][4])
        if envelope.occurred_at < inbound_created_at:
            raise ValueError("Workshop run cannot be accepted before its inbound message")
        if envelope.actor_principal_id != requested_by:
            raise ValueError("Workshop run acceptance actor must be its requesting human")
        await connection.execute(
            "INSERT INTO runs "
            "(id, workshop_id, channel_id, requested_by_principal_id, agent_id, "
            "inbound_message_id, status, accepted_at, last_event_position) "
            "VALUES (?, ?, ?, ?, ?, ?, 'accepted', ?, ?)",
            (
                envelope.aggregate_id,
                envelope.workshop_id,
                channel_id,
                requested_by,
                agent_id,
                inbound_message_id,
                occurred_at,
                event.position,
            ),
        )
        return

    async with connection.execute(
        "SELECT r.workshop_id, r.status, r.requested_by_principal_id, a.principal_id, "
        "r.accepted_at, r.started_at "
        "FROM runs r JOIN agents a ON a.id = r.agent_id WHERE r.id = ?",
        (envelope.aggregate_id,),
    ) as cursor:
        row = await cursor.fetchone()
    if row is None or WorkshopId(str(row[0])) != envelope.workshop_id:
        raise ValueError("Workshop run transition references an unknown run")
    status = str(row[1])
    requested_by = PrincipalId(str(row[2]))
    agent_principal = PrincipalId(str(row[3]))
    accepted_at = _parse_projection_timestamp(row[4])
    started_at = None if row[5] is None else _parse_projection_timestamp(row[5])

    if envelope.event_type == WorkshopEventType.RUN_STARTED:
        _require_exact_payload(payload, set())
        if status != "accepted" or envelope.actor_principal_id != agent_principal or envelope.occurred_at < accepted_at:
            raise ValueError("Workshop run can start only once through its attached agent")
        await connection.execute(
            "UPDATE runs SET status = 'started', started_at = ?, last_event_position = ? WHERE id = ?",
            (occurred_at, event.position, envelope.aggregate_id),
        )
        return

    if envelope.event_type == WorkshopEventType.RUN_COMPLETED:
        _require_exact_payload(payload, set())
        if (
            status != "started"
            or envelope.actor_principal_id != agent_principal
            or started_at is None
            or envelope.occurred_at < started_at
        ):
            raise ValueError("Workshop run can complete only from started through its attached agent")
        await connection.execute(
            "UPDATE runs SET status = 'completed', terminal_at = ?, last_event_position = ? WHERE id = ?",
            (occurred_at, event.position, envelope.aggregate_id),
        )
        return

    if envelope.event_type == WorkshopEventType.RUN_FAILED:
        _require_exact_payload(payload, {"failure_code"})
        failure_code = _required_text(payload, "failure_code")
        if not _RUN_TERMINAL_CODE_PATTERN.fullmatch(failure_code):
            raise ValueError("Workshop run failure_code must be a lowercase identifier")
        if (
            status != "started"
            or envelope.actor_principal_id != agent_principal
            or started_at is None
            or envelope.occurred_at < started_at
        ):
            raise ValueError("Workshop run can fail only from started through its attached agent")
        await connection.execute(
            "UPDATE runs SET status = 'failed', terminal_at = ?, terminal_code = ?, "
            "last_event_position = ? WHERE id = ?",
            (occurred_at, failure_code, event.position, envelope.aggregate_id),
        )
        return

    if envelope.event_type == WorkshopEventType.RUN_CANCELLED:
        _require_exact_payload(payload, {"cancellation_code"})
        cancellation_code = _required_text(payload, "cancellation_code")
        if not _RUN_TERMINAL_CODE_PATTERN.fullmatch(cancellation_code):
            raise ValueError("Workshop run cancellation_code must be a lowercase identifier")
        lifecycle_floor = started_at if started_at is not None else accepted_at
        if (
            status not in {"accepted", "started"}
            or envelope.actor_principal_id != requested_by
            or envelope.occurred_at < lifecycle_floor
        ):
            raise ValueError("Workshop run can be cancelled only while nonterminal by its requesting human")
        await connection.execute(
            "UPDATE runs SET status = 'cancelled', terminal_at = ?, terminal_code = ?, "
            "last_event_position = ? WHERE id = ?",
            (occurred_at, cancellation_code, event.position, envelope.aggregate_id),
        )
        return

    raise ValueError(f"Unsupported Workshop run event: {envelope.event_type}")


class CanonicalConversationProjection:
    """Rebuild the initial Workshop collaboration records from events."""

    name = "canonical_conversations"
    # Artifact events were not emitted before artifact support existed, so
    # adding their first handler does not change replay of any prior event.
    # Keep the version stable to avoid an unnecessary production rebuild.
    version = 5

    async def reset(self, connection: aiosqlite.Connection) -> None:
        async with connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'") as cursor:
            existing_tables = {str(row[0]) for row in await cursor.fetchall()}
        for table in (
            "deliveries",
            "artifacts",
            "runs",
            "messages",
            "channel_agents",
            "channel_bindings",
            "channel_memberships",
            "channels",
            "agents",
            "workshop_memberships",
            "external_identities",
            "principals",
            "workshops",
        ):
            if table in existing_tables:
                await connection.execute(f"DELETE FROM {table}")

    async def apply(self, connection: aiosqlite.Connection, event: StoredEvent) -> None:
        envelope = event.envelope
        payload = envelope.payload
        occurred_at = envelope.occurred_at.isoformat()

        if envelope.event_type in {
            WorkshopEventType.RUN_ACCEPTED,
            WorkshopEventType.RUN_STARTED,
            WorkshopEventType.RUN_COMPLETED,
            WorkshopEventType.RUN_FAILED,
            WorkshopEventType.RUN_CANCELLED,
        }:
            await _apply_run_event(connection, event)
            return

        if envelope.event_type == WorkshopEventType.WORKSHOP_CREATED:
            await connection.execute(
                "INSERT INTO workshops (id, name, created_at) VALUES (?, ?, ?)",
                (envelope.aggregate_id, _required_text(payload, "name"), occurred_at),
            )
        elif envelope.event_type == WorkshopEventType.PRINCIPAL_CREATED:
            await connection.execute(
                "INSERT INTO principals (id, kind, display_name, created_at) VALUES (?, ?, ?, ?)",
                (
                    envelope.aggregate_id,
                    _required_text(payload, "kind"),
                    _required_text(payload, "display_name"),
                    occurred_at,
                ),
            )
        elif envelope.event_type == WorkshopEventType.EXTERNAL_IDENTITY_BOUND:
            await connection.execute(
                "INSERT INTO external_identities "
                "(id, principal_id, provider, external_subject, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    envelope.aggregate_id,
                    _required_text(payload, "principal_id"),
                    _required_text(payload, "provider"),
                    _required_text(payload, "external_subject"),
                    occurred_at,
                ),
            )
        elif envelope.event_type == WorkshopEventType.WORKSHOP_MEMBER_ADDED:
            await connection.execute(
                "INSERT INTO workshop_memberships "
                "(id, workshop_id, principal_id, role, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    envelope.aggregate_id,
                    envelope.workshop_id,
                    _required_text(payload, "principal_id"),
                    _required_text(payload, "role"),
                    occurred_at,
                ),
            )
        elif envelope.event_type == WorkshopEventType.CHANNEL_CREATED:
            name = payload.get("name")
            if name is not None and not isinstance(name, str):
                raise ValueError("Workshop channel name must be a string or null")
            await connection.execute(
                "INSERT INTO channels (id, workshop_id, kind, name, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    envelope.aggregate_id,
                    envelope.workshop_id,
                    _required_text(payload, "kind"),
                    name,
                    occurred_at,
                ),
            )
        elif envelope.event_type == WorkshopEventType.CHANNEL_MEMBER_ADDED:
            role = _required_text(payload, "role")
            if role not in {"owner", "participant"}:
                raise ValueError("Workshop channel member role must be 'owner' or 'participant'")
            await connection.execute(
                "INSERT INTO channel_memberships "
                "(id, channel_id, principal_id, role, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    envelope.aggregate_id,
                    _required_text(payload, "channel_id"),
                    _required_text(payload, "principal_id"),
                    role,
                    occurred_at,
                ),
            )
        elif envelope.event_type == WorkshopEventType.TRANSPORT_CHANNEL_BOUND:
            await connection.execute(
                "INSERT INTO channel_bindings "
                "(id, channel_id, transport, external_channel_id, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    envelope.aggregate_id,
                    _required_text(payload, "channel_id"),
                    _required_text(payload, "transport"),
                    _required_text(payload, "external_channel_id"),
                    occurred_at,
                ),
            )
        elif envelope.event_type == WorkshopEventType.AGENT_CREATED:
            await connection.execute(
                "INSERT INTO agents (id, workshop_id, principal_id, name, created_at) VALUES (?, ?, ?, ?, ?)",
                (
                    envelope.aggregate_id,
                    envelope.workshop_id,
                    _required_text(payload, "principal_id"),
                    _required_text(payload, "name"),
                    occurred_at,
                ),
            )
        elif envelope.event_type == WorkshopEventType.CHANNEL_AGENT_ATTACHED:
            await connection.execute(
                "INSERT INTO channel_agents (id, channel_id, agent_id, created_at) VALUES (?, ?, ?, ?)",
                (
                    envelope.aggregate_id,
                    _required_text(payload, "channel_id"),
                    _required_text(payload, "agent_id"),
                    occurred_at,
                ),
            )
        elif envelope.event_type == WorkshopEventType.MESSAGE_CREATED:
            reply_to = payload.get("reply_to_message_id")
            if reply_to is not None and not isinstance(reply_to, str):
                raise ValueError("Workshop reply_to_message_id must be a string or null")
            await connection.execute(
                "INSERT INTO messages "
                "(id, channel_id, author_principal_id, reply_to_message_id, body, "
                "created_event_position, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    envelope.aggregate_id,
                    _required_text(payload, "channel_id"),
                    _required_text(payload, "author_principal_id"),
                    reply_to,
                    _required_text(payload, "body"),
                    event.position,
                    occurred_at,
                ),
            )
        elif envelope.event_type == WorkshopEventType.ARTIFACT_CREATED:
            created_by = _required_text(payload, "created_by_principal_id")
            if envelope.actor_principal_id != created_by:
                raise ValueError("Workshop artifact actor must match created_by_principal_id")
            channel_id = _required_text(payload, "channel_id")
            message_id = _required_text(payload, "message_id")
            async with connection.execute(
                "SELECT c.workshop_id, m.channel_id, m.author_principal_id, p.kind "
                "FROM messages m JOIN channels c ON c.id = m.channel_id "
                "JOIN principals p ON p.id = m.author_principal_id WHERE m.id = ?",
                (message_id,),
            ) as cursor:
                message_row = await cursor.fetchone()
            if message_row is None or tuple(message_row) != (
                envelope.workshop_id,
                channel_id,
                created_by,
                "human",
            ):
                raise ValueError("Workshop artifact must belong to its human-authored message")
            kind = _required_text(payload, "kind")
            if kind not in {"photo", "document", "voice"}:
                raise ValueError("Workshop artifact kind is unsupported")
            byte_size = payload.get("byte_size")
            if not isinstance(byte_size, int) or isinstance(byte_size, bool) or byte_size < 0:
                raise ValueError("Workshop artifact byte_size must be a non-negative integer")
            content_sha256 = _required_text(payload, "content_sha256")
            if not _SHA256_PATTERN.fullmatch(content_sha256):
                raise ValueError("Workshop artifact content_sha256 must be lowercase SHA-256")
            original_filename = payload.get("original_filename")
            if original_filename is not None and (
                not isinstance(original_filename, str)
                or not original_filename
                or original_filename != original_filename.strip()
                or len(original_filename) > 255
                or original_filename in {".", ".."}
                or "/" in original_filename
                or "\\" in original_filename
                or "\0" in original_filename
            ):
                raise ValueError("Workshop artifact original_filename must be a bounded string or null")
            media_type = _required_text(payload, "media_type")
            if not _MEDIA_TYPE_PATTERN.fullmatch(media_type):
                raise ValueError("Workshop artifact media_type must be a lowercase MIME type")
            storage_path = _required_text(payload, "storage_path")
            if not Path(storage_path).is_absolute():
                raise ValueError("Workshop artifact storage_path must be absolute")
            source_transport = _required_text(payload, "source_transport")
            if not _IDENTIFIER_PATTERN.fullmatch(source_transport):
                raise ValueError("Workshop artifact source_transport must be a lowercase identifier")
            source_unique_id = _required_text(payload, "source_unique_id")
            if source_unique_id != source_unique_id.strip() or len(source_unique_id) > 512:
                raise ValueError("Workshop artifact source_unique_id must be bounded")
            await connection.execute(
                "INSERT INTO artifacts "
                "(id, workshop_id, channel_id, message_id, created_by_principal_id, kind, "
                "media_type, byte_size, content_sha256, original_filename, storage_path, "
                "source_transport, source_unique_id, created_event_position, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    envelope.aggregate_id,
                    envelope.workshop_id,
                    channel_id,
                    message_id,
                    created_by,
                    kind,
                    media_type,
                    byte_size,
                    content_sha256,
                    original_filename,
                    storage_path,
                    source_transport,
                    source_unique_id,
                    event.position,
                    occurred_at,
                ),
            )
        elif envelope.event_type in {
            WorkshopEventType.DELIVERY_SUCCEEDED,
            WorkshopEventType.DELIVERY_FAILED,
        }:
            status = "succeeded" if envelope.event_type == WorkshopEventType.DELIVERY_SUCCEEDED else "failed"
            message_id = _required_text(payload, "message_id")
            channel_id = _required_text(payload, "channel_id")
            transport = _required_text(payload, "transport")
            if envelope.event_version == 1:
                channel_binding_id = None
                async with connection.execute(
                    "SELECT 1 FROM messages WHERE id = ? AND channel_id = ?",
                    (message_id, channel_id),
                ) as cursor:
                    message_row = await cursor.fetchone()
                if message_row is None:
                    raise ValueError("Workshop delivery message must belong to its channel")
            elif envelope.event_version in {2, 3}:
                channel_binding_id = _required_text(payload, "channel_binding_id")
                async with connection.execute(
                    "SELECT 1 FROM messages m JOIN channel_bindings cb ON cb.channel_id = m.channel_id "
                    "WHERE m.id = ? AND m.channel_id = ? AND cb.id = ? AND cb.transport = ?",
                    (message_id, channel_id, channel_binding_id, transport),
                ) as cursor:
                    binding_row = await cursor.fetchone()
                if binding_row is None:
                    raise ValueError("Workshop delivery message and binding must belong to the same channel")
            else:
                raise ValueError("Workshop delivery event version is unsupported")
            await connection.execute(
                "INSERT INTO deliveries "
                "(id, message_id, channel_id, channel_binding_id, transport, mode, status, "
                "created_at, updated_at, last_event_position) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "status = excluded.status, updated_at = excluded.updated_at, "
                "last_event_position = excluded.last_event_position",
                (
                    envelope.aggregate_id,
                    message_id,
                    channel_id,
                    channel_binding_id,
                    transport,
                    _required_text(payload, "mode"),
                    status,
                    occurred_at,
                    occurred_at,
                    event.position,
                ),
            )
