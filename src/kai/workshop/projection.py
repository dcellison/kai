"""Canonical collaboration projection for Workshop events."""

from __future__ import annotations

from typing import Any

import aiosqlite

from kai.workshop.domain import WorkshopEventType
from kai.workshop.store import StoredEvent


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Workshop event payload requires non-empty {key!r}")
    return value


class CanonicalConversationProjection:
    """Rebuild the initial Workshop collaboration records from events."""

    name = "canonical_conversations"
    version = 2

    async def reset(self, connection: aiosqlite.Connection) -> None:
        for table in (
            "deliveries",
            "messages",
            "channel_agents",
            "channel_bindings",
            "channels",
            "agents",
            "workshop_memberships",
            "external_identities",
            "principals",
            "workshops",
        ):
            await connection.execute(f"DELETE FROM {table}")

    async def apply(self, connection: aiosqlite.Connection, event: StoredEvent) -> None:
        envelope = event.envelope
        payload = envelope.payload
        occurred_at = envelope.occurred_at.isoformat()

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
        elif envelope.event_type in {
            WorkshopEventType.DELIVERY_SUCCEEDED,
            WorkshopEventType.DELIVERY_FAILED,
        }:
            status = "succeeded" if envelope.event_type == WorkshopEventType.DELIVERY_SUCCEEDED else "failed"
            await connection.execute(
                "INSERT INTO deliveries "
                "(id, message_id, channel_id, transport, mode, status, created_at, updated_at, "
                "last_event_position) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(id) DO UPDATE SET "
                "status = excluded.status, updated_at = excluded.updated_at, "
                "last_event_position = excluded.last_event_position",
                (
                    envelope.aggregate_id,
                    _required_text(payload, "message_id"),
                    _required_text(payload, "channel_id"),
                    _required_text(payload, "transport"),
                    _required_text(payload, "mode"),
                    status,
                    occurred_at,
                    occurred_at,
                    event.position,
                ),
            )
