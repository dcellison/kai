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

SUPPORTED_MESSAGE_REACTIONS = frozenset(
    {
        "thumbs_up",
        "thumbs_down",
        "heart",
        "laugh",
        "celebrate",
        "eyes",
        "check",
        "thinking",
        "surprised",
        "sad",
        "fire",
        "question",
    }
)
MAX_MESSAGE_REACTORS = 100


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


@dataclass(frozen=True, slots=True)
class MessageReactorIdentity:
    """One bounded, user-visible human or agent reactor identity."""

    principal_id: PrincipalId
    kind: str
    display_name: str
    handle: str | None


@dataclass(frozen=True, slots=True)
class MessageReactorSnapshot:
    reactors: tuple[MessageReactorIdentity, ...]
    total: int
    truncated: bool


def validate_message_reaction(value: object) -> str:
    if not isinstance(value, str) or value not in SUPPORTED_MESSAGE_REACTIONS:
        raise MessageReactionValidationError("Unsupported message reaction")
    return value


async def load_message_reactions(
    store: WorkshopEventStore,
    *,
    message_ids: tuple[MessageId, ...],
    viewer_principal_id: PrincipalId,
    through_position: int | None = None,
) -> dict[MessageId, tuple[MessageReactionSummary, ...]]:
    """Aggregate reactions for messages, including viewer-specific state."""
    if not message_ids:
        return {}
    placeholders = ", ".join("?" for _ in message_ids)
    if through_position is None:
        async with store.connection.execute(
            "SELECT message_id, reaction, COUNT(*), "
            "MAX(CASE WHEN principal_id = ? THEN 1 ELSE 0 END) "
            f"FROM message_reactions WHERE message_id IN ({placeholders}) "
            "GROUP BY message_id, reaction "
            "ORDER BY message_id, MIN(created_event_position), reaction",
            (viewer_principal_id, *message_ids),
        ) as cursor:
            rows = list(await cursor.fetchall())
    else:
        # The current-state projection intentionally deletes removed reactions.
        # Snapshot reads therefore reconstruct the latest state at the fixed
        # event boundary from canonical events instead of leaking later edits.
        async with store.connection.execute(
            "WITH ranked AS ("
            "SELECT aggregate_id AS message_id, "
            "json_extract(payload_json, '$.principal_id') AS principal_id, "
            "json_extract(payload_json, '$.reaction') AS reaction, event_type, position, "
            "ROW_NUMBER() OVER (PARTITION BY aggregate_id, "
            "json_extract(payload_json, '$.principal_id'), "
            "json_extract(payload_json, '$.reaction') ORDER BY position DESC) AS state_rank "
            "FROM event_log WHERE aggregate_type = 'message' "
            "AND event_type IN (?, ?) AND position <= ? "
            f"AND aggregate_id IN ({placeholders})"
            ") SELECT message_id, reaction, COUNT(*), "
            "MAX(CASE WHEN principal_id = ? THEN 1 ELSE 0 END) "
            "FROM ranked WHERE state_rank = 1 AND event_type = ? "
            "GROUP BY message_id, reaction ORDER BY message_id, MIN(position), reaction",
            (
                WorkshopEventType.MESSAGE_REACTION_ADDED.value,
                WorkshopEventType.MESSAGE_REACTION_REMOVED.value,
                through_position,
                *message_ids,
                viewer_principal_id,
                WorkshopEventType.MESSAGE_REACTION_ADDED.value,
            ),
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
            "WHERE m.id = ? AND m.channel_id = ? AND c.archived_at IS NULL",
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


async def load_message_reactors(
    store: WorkshopEventStore,
    *,
    viewer_principal_id: PrincipalId,
    channel_id: ChannelId,
    message_id: MessageId,
    reaction: object,
) -> MessageReactorSnapshot:
    """Load bounded reactor identities after checking the viewer's channel access."""
    if not isinstance(viewer_principal_id, PrincipalId):
        raise MessageReactionValidationError("Invalid reaction viewer")
    if not isinstance(channel_id, ChannelId) or not isinstance(message_id, MessageId):
        raise MessageReactionValidationError("Invalid reaction target")
    normalized_reaction = validate_message_reaction(reaction)
    async with store.connection.execute(
        "SELECT c.workshop_id FROM messages m JOIN channels c ON c.id = m.channel_id "
        "JOIN channel_memberships cm ON cm.channel_id = c.id AND cm.principal_id = ? "
        "WHERE m.id = ? AND m.channel_id = ? AND c.archived_at IS NULL",
        (viewer_principal_id, message_id, channel_id),
    ) as cursor:
        access = await cursor.fetchone()
    if access is None:
        raise MessageReactionAccessDeniedError("Reaction access denied")
    async with store.connection.execute(
        "SELECT mr.principal_id, p.kind, p.display_name, "
        "CASE p.kind WHEN 'human' THEN hh.handle ELSE ad.handle END, COUNT(*) OVER() "
        "FROM message_reactions mr JOIN principals p ON p.id = mr.principal_id "
        "JOIN channels c ON c.id = ? "
        "LEFT JOIN human_handles hh ON hh.workshop_id = c.workshop_id AND hh.principal_id = p.id "
        "LEFT JOIN agents a ON a.principal_id = p.id "
        "LEFT JOIN agent_definitions ad ON ad.agent_id = a.id "
        "WHERE mr.message_id = ? AND mr.reaction = ? "
        "ORDER BY mr.created_event_position, mr.principal_id LIMIT ?",
        (channel_id, message_id, normalized_reaction, MAX_MESSAGE_REACTORS),
    ) as cursor:
        rows = list(await cursor.fetchall())
    total = int(rows[0][4]) if rows else 0
    truncated = total > MAX_MESSAGE_REACTORS
    visible = rows[:MAX_MESSAGE_REACTORS]
    return MessageReactorSnapshot(
        reactors=tuple(
            MessageReactorIdentity(
                principal_id=PrincipalId(str(row[0])),
                kind=str(row[1]),
                display_name=str(row[2]),
                handle=str(row[3]) if row[3] is not None else None,
            )
            for row in visible
        ),
        total=total,
        truncated=truncated,
    )
