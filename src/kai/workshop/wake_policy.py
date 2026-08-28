"""Channel-kind-derived wake policy for canonical Workshop messages."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from kai.workshop.domain import (
    AgentId,
    ChannelId,
    EventEnvelope,
    EventId,
    MessageId,
    PrincipalId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import AppendResult, WorkshopEventStore

GROUP_AGENT_ENGAGEMENT_WINDOW_SECONDS = 900


class WakePolicyError(RuntimeError, LookupError):
    """Canonical channel state cannot produce a safe wake decision."""


@dataclass(frozen=True, slots=True)
class EngagementScope:
    """Conversation scope, ready for thread-specific engagement later."""

    channel_id: ChannelId
    thread_root_message_id: MessageId | None = None


@dataclass(frozen=True, slots=True)
class WakeDecision:
    agent_ids: tuple[AgentId, ...]


@dataclass(frozen=True, slots=True)
class AgentEngagement:
    agent_id: AgentId
    engaged_at: datetime
    expires_at: datetime


async def resolve_agent_engagements(
    store: WorkshopEventStore,
    scope: EngagementScope,
    *,
    current_at: datetime,
    before_event_position: int | None = None,
) -> tuple[AgentEngagement, ...]:
    """Derive the agents currently engaged in one supported conversation scope."""
    if scope.thread_root_message_id is not None:
        raise WakePolicyError("Thread-scoped engagement is not available yet")
    if current_at.tzinfo is None or current_at.utcoffset() is None:
        raise ValueError("current_at must be timezone-aware")
    if before_event_position is not None and before_event_position < 1:
        raise ValueError("before_event_position must be positive")
    current_at = current_at.astimezone(UTC)
    async with store.connection.execute(
        "SELECT c.kind, ca.agent_id, a.principal_id FROM channels c "
        "LEFT JOIN channel_agents ca ON ca.channel_id = c.id "
        "LEFT JOIN agents a ON a.id = ca.agent_id WHERE c.id = ? "
        "ORDER BY ca.agent_id",
        (scope.channel_id,),
    ) as cursor:
        rows = list(await cursor.fetchall())
    if not rows:
        raise WakePolicyError("Engagement scope channel is unavailable")
    if str(rows[0][0]) != "group":
        return ()

    window_start = current_at - timedelta(seconds=GROUP_AGENT_ENGAGEMENT_WINDOW_SECONDS)
    engaged: list[AgentEngagement] = []
    for row in rows:
        if row[1] is None or row[2] is None:
            continue
        agent_id = AgentId(str(row[1]))
        agent_principal_id = PrincipalId(str(row[2]))
        position_clause = "AND m.created_event_position < ? " if before_event_position is not None else ""
        parameters: tuple[object, ...] = (
            (scope.channel_id, before_event_position) if before_event_position is not None else (scope.channel_id,)
        ) + (
            window_start.isoformat(),
            current_at.isoformat(),
            agent_principal_id,
            agent_principal_id,
        )
        async with store.connection.execute(
            "SELECT m.created_at FROM messages m WHERE m.channel_id = ? "
            + position_clause
            + "AND m.created_at >= ? AND m.created_at <= ? "
            "AND (m.author_principal_id = ? OR EXISTS ("
            "SELECT 1 FROM json_each(m.mentions_json) mention "
            "WHERE json_extract(mention.value, '$.kind') = 'agent' "
            "AND json_extract(mention.value, '$.principal_id') = ?)) "
            "ORDER BY m.created_at DESC, m.created_event_position DESC LIMIT 1",
            parameters,
        ) as cursor:
            engagement_row = await cursor.fetchone()
        if engagement_row is None:
            continue
        engaged_at = str(engagement_row[0])
        async with store.connection.execute(
            "SELECT 1 FROM channel_agent_dismissals WHERE channel_id = ? "
            "AND agent_id = ? AND thread_root_message_id IS NULL "
            "AND dismissed_at >= ? AND dismissed_at <= ? LIMIT 1",
            (scope.channel_id, agent_id, engaged_at, current_at.isoformat()),
        ) as cursor:
            dismissed = await cursor.fetchone()
        if dismissed is None:
            engaged_at_value = datetime.fromisoformat(engaged_at.replace("Z", "+00:00")).astimezone(UTC)
            engaged.append(
                AgentEngagement(
                    agent_id=agent_id,
                    engaged_at=engaged_at_value,
                    expires_at=engaged_at_value + timedelta(seconds=GROUP_AGENT_ENGAGEMENT_WINDOW_SECONDS),
                )
            )
    return tuple(engaged)


async def resolve_engaged_agents(
    store: WorkshopEventStore,
    scope: EngagementScope,
    *,
    current_at: datetime,
    before_event_position: int | None = None,
) -> tuple[AgentId, ...]:
    """Return only the engaged agent identities for wake routing."""
    return tuple(
        engagement.agent_id
        for engagement in await resolve_agent_engagements(
            store,
            scope,
            current_at=current_at,
            before_event_position=before_event_position,
        )
    )


async def resolve_message_wake_targets(
    store: WorkshopEventStore,
    message_id: MessageId,
    *,
    scope: EngagementScope | None = None,
) -> WakeDecision:
    """Resolve the exact agents a projected canonical message may wake."""
    if not isinstance(message_id, MessageId):
        raise ValueError("message_id must be a MessageId")
    async with store.connection.execute(
        "SELECT m.channel_id, c.kind, m.author_principal_id, p.kind, "
        "m.mentions_json, m.created_at, m.created_event_position "
        "FROM messages m JOIN channels c ON c.id = m.channel_id "
        "JOIN principals p ON p.id = m.author_principal_id WHERE m.id = ?",
        (message_id,),
    ) as cursor:
        row = await cursor.fetchone()
    if row is None:
        raise WakePolicyError("Wake policy message is not projected")
    channel_id = ChannelId(str(row[0]))
    kind = str(row[1])
    author_kind = str(row[3])
    if scope is None:
        scope = EngagementScope(channel_id)
    if scope.channel_id != channel_id or scope.thread_root_message_id is not None:
        raise WakePolicyError("Wake policy scope does not match the supported channel scope")
    if author_kind != "human":
        return WakeDecision(())

    async with store.connection.execute(
        "SELECT ca.agent_id, a.principal_id FROM channel_agents ca "
        "JOIN agents a ON a.id = ca.agent_id WHERE ca.channel_id = ? "
        "ORDER BY ca.agent_id",
        (channel_id,),
    ) as cursor:
        attached = [(AgentId(str(item[0])), PrincipalId(str(item[1]))) for item in await cursor.fetchall()]
    if kind == "direct":
        if len(attached) != 1:
            raise WakePolicyError("Inbound message must resolve to one human channel member and one attached agent")
        return WakeDecision((attached[0][0],))
    if kind != "group":
        return WakeDecision(())

    mentions = json.loads(str(row[4]))
    mentioned_principals = {
        PrincipalId(str(item["principal_id"]))
        for item in mentions
        if isinstance(item, dict) and item.get("kind") == "agent"
    }
    explicitly_mentioned = tuple(
        agent_id for agent_id, principal_id in attached if principal_id in mentioned_principals
    )
    if explicitly_mentioned:
        return WakeDecision(explicitly_mentioned)

    current_at = datetime.fromisoformat(str(row[5]).replace("Z", "+00:00")).astimezone(UTC)
    return WakeDecision(
        await resolve_engaged_agents(
            store,
            scope,
            current_at=current_at,
            before_event_position=int(row[6]),
        )
    )


async def dismiss_channel_agent(
    store: WorkshopEventStore,
    *,
    principal_id: PrincipalId,
    scope: EngagementScope,
    agent_id: AgentId,
    client_dismissal_id: str,
    occurred_at: datetime,
) -> AppendResult:
    """Append one idempotent, principal-authorized group dismissal fact."""
    if not client_dismissal_id or len(client_dismissal_id) > 200:
        raise ValueError("client_dismissal_id must be a bounded non-empty string")
    if occurred_at.tzinfo is None or occurred_at.utcoffset() is None:
        raise ValueError("occurred_at must be timezone-aware")
    occurred_at = occurred_at.astimezone(UTC)
    async with store.connection.execute(
        "SELECT c.workshop_id FROM channels c "
        "JOIN channel_memberships cm ON cm.channel_id = c.id AND cm.principal_id = ? "
        "JOIN principals p ON p.id = cm.principal_id AND p.kind = 'human' "
        "JOIN channel_agents ca ON ca.channel_id = c.id AND ca.agent_id = ? "
        "WHERE c.id = ? AND c.kind = 'group'",
        (principal_id, agent_id, scope.channel_id),
    ) as cursor:
        row = await cursor.fetchone()
    if row is None:
        raise WakePolicyError("Agent dismissal requires group membership and an attached agent")
    workshop_id = WorkshopId(str(row[0]))
    event_id = EventId.derived(
        scope.channel_id,
        f"agent-dismissal:{principal_id}:{agent_id}:{client_dismissal_id}",
    )
    event = EventEnvelope.create(
        event_id=event_id,
        event_type=WorkshopEventType.CHANNEL_AGENT_DISMISSED,
        event_version=1,
        workshop_id=workshop_id,
        aggregate_type="channel",
        aggregate_id=scope.channel_id,
        actor_principal_id=principal_id,
        occurred_at=occurred_at,
        idempotency_key=f"channel-agent-dismissal:v1:{event_id}",
        payload={
            "agent_id": agent_id,
            "thread_root_message_id": scope.thread_root_message_id,
        },
        metadata={"source": "workshop_client"},
    )
    connection = store.connection
    try:
        await connection.execute("BEGIN IMMEDIATE")
        result = await store.append_in_transaction(event)
        await store.project_pending_in_transaction(CanonicalConversationProjection())
        await connection.commit()
        return result
    except Exception:
        await connection.rollback()
        raise
