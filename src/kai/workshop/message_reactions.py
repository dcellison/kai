"""Canonical, transport-independent message reactions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from kai.workshop.domain import (
    ChannelId,
    EventEnvelope,
    MessageId,
    MessageReactionSummary,
    PrincipalId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import WorkshopEventStore

SUPPORTED_MESSAGE_REACTIONS = frozenset({"thumbs_up", "heart", "laugh", "celebrate", "eyes", "check"})


class MessageReactionAccessDeniedError(PermissionError):
    """The principal may not react to the requested message."""


class MessageReactionValidationError(ValueError):
    """A reaction mutation is malformed or unsupported."""


@dataclass(frozen=True, slots=True)
class MessageReactionMutation:
    """The authoritative result of setting one principal's reaction state."""

    message_id: MessageId
    reaction: str
    active: bool
    changed: bool
    event_position: int | None
    reactions: tuple[MessageReactionSummary, ...]


def validate_message_reaction(value: object) -> str:
    if not isinstance(value, str) or value not in SUPPORTED_MESSAGE_REACTIONS:
        raise MessageReactionValidationError("Unsupported message reaction")
    return value


async def load_message_reactions(
    store: WorkshopEventStore,
    *,
    message_ids: tuple[MessageId, ...],
    viewer_principal_id: PrincipalId,
) -> dict[MessageId, tuple[MessageReactionSummary, ...]]:
    """Aggregate reactions for messages, including viewer-specific state."""
    if not message_ids:
        return {}
    placeholders = ", ".join("?" for _ in message_ids)
    async with store.connection.execute(
        "SELECT message_id, reaction, COUNT(*), "
        "MAX(CASE WHEN principal_id = ? THEN 1 ELSE 0 END) "
        f"FROM message_reactions WHERE message_id IN ({placeholders}) "
        "GROUP BY message_id, reaction "
        "ORDER BY message_id, MIN(created_event_position), reaction",
        (viewer_principal_id, *message_ids),
    ) as cursor:
        rows = list(await cursor.fetchall())
    grouped: dict[MessageId, list[MessageReactionSummary]] = {}
    for row in rows:
        message_id = MessageId(str(row[0]))
        grouped.setdefault(message_id, []).append(
            MessageReactionSummary(
                reaction=str(row[1]),
                count=int(row[2]),
                reacted_by_viewer=bool(row[3]),
            )
        )
    return {message_id: tuple(reactions) for message_id, reactions in grouped.items()}


async def set_message_reaction(
    store: WorkshopEventStore,
    *,
    principal_id: PrincipalId,
    channel_id: ChannelId,
    message_id: MessageId,
    reaction: str,
    active: bool,
    occurred_at: datetime | None = None,
) -> MessageReactionMutation:
    """Set one reaction state atomically under canonical channel authority."""
    if not isinstance(principal_id, PrincipalId):
        raise MessageReactionValidationError("Invalid reaction principal")
    if not isinstance(channel_id, ChannelId) or not isinstance(message_id, MessageId):
        raise MessageReactionValidationError("Invalid reaction target")
    normalized_reaction = validate_message_reaction(reaction)
    if not isinstance(active, bool):
        raise MessageReactionValidationError("Reaction active state must be a boolean")
    now = occurred_at or datetime.now(UTC)
    if now.tzinfo is None or now.utcoffset() is None:
        raise MessageReactionValidationError("Reaction time must be timezone-aware")

    connection = store.connection
    try:
        await connection.execute("BEGIN IMMEDIATE")
        async with connection.execute(
            "SELECT c.workshop_id FROM messages m "
            "JOIN channels c ON c.id = m.channel_id "
            "JOIN channel_memberships cm ON cm.channel_id = c.id "
            "AND cm.principal_id = ? "
            "WHERE m.id = ? AND m.channel_id = ?",
            (principal_id, message_id, channel_id),
        ) as cursor:
            target = await cursor.fetchone()
        if target is None:
            raise MessageReactionAccessDeniedError("Reaction access denied")

        async with connection.execute(
            "SELECT 1 FROM message_reactions WHERE message_id = ? AND principal_id = ? AND reaction = ?",
            (message_id, principal_id, normalized_reaction),
        ) as cursor:
            current_active = await cursor.fetchone() is not None

        event_position: int | None = None
        if current_active != active:
            event = EventEnvelope.create(
                event_type=(
                    WorkshopEventType.MESSAGE_REACTION_ADDED if active else WorkshopEventType.MESSAGE_REACTION_REMOVED
                ),
                event_version=1,
                workshop_id=WorkshopId(str(target[0])),
                aggregate_type="message",
                aggregate_id=message_id,
                actor_principal_id=principal_id,
                occurred_at=now,
                payload={
                    "channel_id": str(channel_id),
                    "principal_id": str(principal_id),
                    "reaction": normalized_reaction,
                },
            )
            result = await store.append_in_transaction(event)
            if not result.inserted:
                raise RuntimeError("New message reaction event unexpectedly already exists")
            await store.project_pending_in_transaction(CanonicalConversationProjection())
            event_position = result.event.position
        await connection.commit()
    except (MessageReactionAccessDeniedError, MessageReactionValidationError):
        await connection.rollback()
        raise
    except Exception:
        await connection.rollback()
        raise

    reactions = await load_message_reactions(
        store,
        message_ids=(message_id,),
        viewer_principal_id=principal_id,
    )
    return MessageReactionMutation(
        message_id=message_id,
        reaction=normalized_reaction,
        active=active,
        changed=current_active != active,
        event_position=event_position,
        reactions=reactions.get(message_id, ()),
    )
