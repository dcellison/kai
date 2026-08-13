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
) -> CanonicalConversationRunTarget:
    """Resolve a human-authored message to exactly one canonical channel agent."""
    if not isinstance(inbound_message_id, MessageId):
        raise ValueError("inbound_message_id must be a MessageId")

    async with store.connection.execute(
        "SELECT c.workshop_id, m.channel_id, m.author_principal_id, ca.agent_id "
        "FROM messages m "
        "JOIN channels c ON c.id = m.channel_id "
        "JOIN principals p ON p.id = m.author_principal_id AND p.kind = 'human' "
        "JOIN channel_memberships cm ON cm.channel_id = m.channel_id "
        "AND cm.principal_id = m.author_principal_id "
        "JOIN channel_agents ca ON ca.channel_id = m.channel_id "
        "WHERE m.id = ?",
        (inbound_message_id,),
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
) -> CanonicalConversationRunResolution:
    """Resolve a canonical target plus its assigned opaque runtime profile.

    The public run request contains only a canonical message ID. During the
    migration, the channel-agent's protected runtime-profile assignment is the
    only execution identity. Protected pool policy validates and resolves that
    profile later; human or transport identities are not execution authority.
    """
    target = await resolve_canonical_conversation_target(store, inbound_message_id)
    try:
        _, runtime_profile_id = await resolve_channel_runtime_profile(
            store,
            target.channel_id,
            target.agent_id,
        )
    except WorkshopRuntimeAssignmentError as exc:
        raise ConversationRunUnavailableError(str(exc)) from exc

    return CanonicalConversationRunResolution(
        target=target,
        runtime_profile_id=runtime_profile_id,
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
