"""Transport-neutral preparation and execution of canonical conversation runs."""

from __future__ import annotations

from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from kai.backend import StreamEvent
from kai.workshop.domain import AgentId, ChannelId, MessageId, PrincipalId, RuntimeProfileId, WorkshopId
from kai.workshop.runtime_assignments import (
    WorkshopRuntimeAssignmentError,
    resolve_channel_runtime_profile,
)
from kai.workshop.runtime_pool import WorkshopRuntimePool
from kai.workshop.store import WorkshopEventStore

type AgentPrompt = str | list[dict[str, str]]


class ConversationRunUnavailableError(LookupError):
    """A canonical inbound message cannot resolve one authorized run target."""


@dataclass(frozen=True, slots=True)
class CanonicalConversationRunTarget:
    """Canonical identities for one inbound human message and attached agent."""

    inbound_message_id: MessageId
    workshop_id: WorkshopId
    channel_id: ChannelId
    requested_by_principal_id: PrincipalId
    agent_id: AgentId


@dataclass(frozen=True, slots=True)
class CanonicalConversationRunResolution:
    """Canonical target plus its explicitly assigned protected runtime."""

    target: CanonicalConversationRunTarget
    runtime_profile_id: RuntimeProfileId
    sponsor_principal_id: PrincipalId


class ConversationPool(Protocol):
    """The narrow profile-addressed runtime surface used by run preparation."""

    def get_model(self, runtime_profile_id: str | RuntimeProfileId) -> str: ...

    def send(
        self,
        prompt: AgentPrompt,
        *,
        runtime_profile_id: str | RuntimeProfileId,
    ) -> AsyncIterator[StreamEvent]: ...

    async def get_effective_workspace(
        self,
        runtime_profile_id: str | RuntimeProfileId,
    ) -> Path: ...


async def resolve_canonical_conversation_target(
    store: WorkshopEventStore,
    inbound_message_id: MessageId,
    agent_id: AgentId | None = None,
) -> CanonicalConversationRunTarget:
    """Resolve a human-authored message to one selected canonical channel agent."""
    if not isinstance(inbound_message_id, MessageId):
        raise ValueError("inbound_message_id must be a MessageId")

    async with store.connection.execute("PRAGMA table_info(channel_agents)") as cursor:
        channel_agent_columns = {str(row[1]) for row in await cursor.fetchall()}
    sponsorship_supported = "detached_at" in channel_agent_columns
    attachment_clause = " AND ca.detached_at IS NULL" if sponsorship_supported else ""
    sponsorship_join = (
        "LEFT JOIN agent_definitions sponsored "
        "ON sponsored.agent_id = ca.agent_id "
        "AND sponsored.lifecycle_state = 'active' "
        "AND ((sponsored.owner_principal_id = ca.sponsor_principal_id "
        "AND sponsored.owner_runtime_profile_id = ca.sponsored_runtime_profile_id) "
        "OR (sponsored.owner_principal_id IS NULL AND EXISTS ("
        "SELECT 1 FROM principal_agent_enablements legacy "
        "WHERE legacy.agent_definition_id = sponsored.id "
        "AND legacy.principal_id = ca.sponsor_principal_id "
        "AND legacy.runtime_profile_id = ca.sponsored_runtime_profile_id "
        "AND legacy.lifecycle_state = 'enabled'))) "
        if sponsorship_supported
        else ""
    )
    sponsorship_clause = " AND (c.kind = 'direct' OR sponsored.id IS NOT NULL)" if sponsorship_supported else ""
    async with store.connection.execute(
        "SELECT c.workshop_id, m.channel_id, m.author_principal_id, ca.agent_id "
        "FROM messages m "
        "JOIN channels c ON c.id = m.channel_id "
        "JOIN principals p ON p.id = m.author_principal_id AND p.kind = 'human' "
        "JOIN channel_memberships cm ON cm.channel_id = m.channel_id "
        "AND cm.principal_id = m.author_principal_id "
        "JOIN channel_agents ca ON ca.channel_id = m.channel_id"
        + attachment_clause
        + " "
        + sponsorship_join
        + "WHERE m.id = ? AND (? IS NULL OR ca.agent_id = ?)"
        + sponsorship_clause,
        (inbound_message_id, agent_id, agent_id),
    ) as cursor:
        target_rows = list(await cursor.fetchall())
    if len(target_rows) != 1:
        raise ConversationRunUnavailableError(
            "Inbound message must resolve to one human channel member and one attached agent"
        )

    return CanonicalConversationRunTarget(
        inbound_message_id=inbound_message_id,
        workshop_id=WorkshopId(str(target_rows[0][0])),
        channel_id=ChannelId(str(target_rows[0][1])),
        requested_by_principal_id=PrincipalId(str(target_rows[0][2])),
        agent_id=AgentId(str(target_rows[0][3])),
    )


async def resolve_canonical_conversation_run(
    store: WorkshopEventStore,
    inbound_message_id: MessageId,
    agent_id: AgentId | None = None,
) -> CanonicalConversationRunResolution:
    """Resolve a canonical target plus its assigned opaque runtime profile.

    The public run request contains only a canonical message ID. During the
    migration, the channel-agent's protected runtime-profile assignment is the
    only execution identity. Protected pool policy validates and resolves that
    profile later; human or transport identities are not execution authority.
    """
    target = await resolve_canonical_conversation_target(store, inbound_message_id, agent_id)
    async with store.connection.execute("PRAGMA table_info(agent_definitions)") as cursor:
        definition_columns = {str(row[1]) for row in await cursor.fetchall()}
    if "owner_principal_id" not in definition_columns:
        try:
            _, runtime_profile_id = await resolve_channel_runtime_profile(
                store,
                target.channel_id,
                target.agent_id,
            )
        except WorkshopRuntimeAssignmentError as exc:
            raise ConversationRunUnavailableError(str(exc)) from exc
        async with store.connection.execute("PRAGMA table_info(channel_agents)") as cursor:
            attachment_columns = {str(row[1]) for row in await cursor.fetchall()}
        sponsor_principal_id = target.requested_by_principal_id
        if "sponsor_principal_id" in attachment_columns:
            async with store.connection.execute(
                "SELECT sponsor_principal_id, sponsored_runtime_profile_id, c.kind "
                "FROM channel_agents ca JOIN channels c ON c.id = ca.channel_id "
                "WHERE ca.channel_id = ? AND ca.agent_id = ? AND ca.detached_at IS NULL",
                (target.channel_id, target.agent_id),
            ) as cursor:
                sponsor_row = await cursor.fetchone()
            if sponsor_row is None or sponsor_row[1] is None or str(sponsor_row[1]) != str(runtime_profile_id):
                raise ConversationRunUnavailableError("Channel agent has incomplete runtime sponsorship")
            if sponsor_row[0] is not None:
                sponsor_principal_id = PrincipalId(str(sponsor_row[0]))
            elif str(sponsor_row[2]) != "direct":
                raise ConversationRunUnavailableError("Channel agent has incomplete runtime sponsorship")
        else:
            async with store.connection.execute(
                "SELECT kind FROM channels WHERE id = ?",
                (target.channel_id,),
            ) as cursor:
                channel_row = await cursor.fetchone()
            if channel_row is None or str(channel_row[0]) != "direct":
                raise ConversationRunUnavailableError("Legacy group agent has no runtime sponsorship")
        return CanonicalConversationRunResolution(
            target=target,
            runtime_profile_id=runtime_profile_id,
            sponsor_principal_id=sponsor_principal_id,
        )
    async with store.connection.execute(
        "SELECT COALESCE(d.owner_principal_id, ca.sponsor_principal_id, "
        "CASE WHEN c.kind = 'direct' THEN target_owner.principal_id END), "
        "COALESCE(d.owner_runtime_profile_id, ca.sponsored_runtime_profile_id, ra.runtime_profile_id), "
        "COALESCE(d.owner_direct_channel_id, CASE WHEN c.kind = 'direct' THEN c.id END, "
        "legacy.direct_channel_id) "
        "FROM agent_definitions d JOIN channel_agents ca ON ca.agent_id = d.agent_id "
        "JOIN channels c ON c.id = ca.channel_id "
        "LEFT JOIN channel_memberships target_owner ON target_owner.channel_id = c.id "
        "AND target_owner.role = 'owner' "
        "LEFT JOIN channel_agent_runtime_assignments ra ON ra.channel_id = ca.channel_id "
        "AND ra.agent_id = ca.agent_id "
        "LEFT JOIN principal_agent_enablements legacy ON legacy.agent_definition_id = d.id "
        "AND legacy.principal_id = ca.sponsor_principal_id "
        "AND legacy.runtime_profile_id = ca.sponsored_runtime_profile_id "
        "AND legacy.lifecycle_state = 'enabled' "
        "WHERE ca.channel_id = ? AND ca.agent_id = ? AND ca.detached_at IS NULL "
        "AND d.lifecycle_state = 'active'",
        (target.channel_id, target.agent_id),
    ) as cursor:
        sponsor_row = await cursor.fetchone()
    if sponsor_row is None or any(value is None for value in sponsor_row):
        raise ConversationRunUnavailableError(
            "Agent has no explicit runtime profile assignment or owner runtime authority"
        )
    sponsor_principal_id = PrincipalId(str(sponsor_row[0]))
    runtime_profile_id = RuntimeProfileId(str(sponsor_row[1]))

    return CanonicalConversationRunResolution(
        target=target,
        runtime_profile_id=runtime_profile_id,
        sponsor_principal_id=sponsor_principal_id,
    )


@dataclass(frozen=True, slots=True)
class PreparedConversationRun:
    """One canonical run prepared against a protected runtime profile."""

    target: CanonicalConversationRunTarget
    model: str
    _pool: ConversationPool = field(repr=False, compare=False)
    _runtime_profile_id: RuntimeProfileId = field(repr=False, compare=False)

    async def stream(self, prompt: AgentPrompt) -> AsyncIterator[StreamEvent]:
        """Stream normalized harness events without exposing the pool key."""
        async for event in self._pool.send(
            prompt,
            runtime_profile_id=self._runtime_profile_id,
        ):
            yield event

    async def effective_workspace(self) -> Path:
        """Return the run's effective project workspace through the adapter."""
        return await self._pool.get_effective_workspace(self._runtime_profile_id)


class WorkshopConversationRunService:
    """Prepare canonical conversation runs independently of an ingress client."""

    def __init__(
        self,
        pool: WorkshopRuntimePool,
        resolver: Callable[[MessageId], Awaitable[CanonicalConversationRunResolution]],
    ) -> None:
        self._pool = pool
        self._resolver = resolver

    async def prepare(self, inbound_message_id: MessageId) -> PreparedConversationRun:
        resolution = await self._resolver(inbound_message_id)
        target = resolution.target
        if target.inbound_message_id != inbound_message_id:
            raise ConversationRunUnavailableError("Run resolver returned a different inbound message")
        return PreparedConversationRun(
            target=target,
            model=self._pool.get_model(resolution.runtime_profile_id),
            _pool=self._pool,
            _runtime_profile_id=resolution.runtime_profile_id,
        )
