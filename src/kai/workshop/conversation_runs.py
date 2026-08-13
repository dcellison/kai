"""Transport-neutral preparation and execution of canonical conversation runs."""

from __future__ import annotations

import re
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from kai.backend import StreamEvent
from kai.workshop.domain import AgentId, ChannelId, MessageId, PrincipalId, WorkshopId
from kai.workshop.store import WorkshopEventStore

type AgentPrompt = str | list[dict[str, str]]
_RUNTIME_SUBJECT_PATTERN = re.compile(r"^[1-9][0-9]*$")


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
class CompatibilityConversationRunResolution:
    """Canonical target plus the private key required by the current pool."""

    target: CanonicalConversationRunTarget
    _legacy_pool_key: int = field(repr=False, compare=False)


class ConversationPool(Protocol):
    """The narrow existing-pool surface hidden behind the run service."""

    def get_model(self, chat_id: int) -> str: ...

    def send(self, prompt: AgentPrompt, *, chat_id: int) -> AsyncIterator[StreamEvent]: ...

    async def get_effective_workspace(self, chat_id: int) -> Path: ...


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
) -> CompatibilityConversationRunResolution:
    """Resolve a canonical target plus the current private pool adapter key.

    The public run request contains only a canonical message ID. During the
    migration, the resolved human's protected Kai runtime identity supplies
    the existing pool/config key internally. A caller cannot provide or
    override that key, and transport identities are not execution authority.
    """
    target = await resolve_canonical_conversation_target(store, inbound_message_id)
    async with store.connection.execute(
        "SELECT external_subject FROM external_identities WHERE principal_id = ? AND provider = 'kai'",
        (target.requested_by_principal_id,),
    ) as cursor:
        identity_rows = list(await cursor.fetchall())
    if len(identity_rows) != 1:
        raise ConversationRunUnavailableError("Canonical human requires exactly one Kai runtime identity")
    external_subject = str(identity_rows[0][0])
    if not _RUNTIME_SUBJECT_PATTERN.fullmatch(external_subject):
        raise ConversationRunUnavailableError("Kai runtime identity is not a positive configured-user ID")

    return CompatibilityConversationRunResolution(
        target=target,
        _legacy_pool_key=int(external_subject),
    )


@dataclass(frozen=True, slots=True)
class PreparedConversationRun:
    """One canonical run prepared against the current compatibility runtime."""

    target: CanonicalConversationRunTarget
    model: str
    _pool: ConversationPool = field(repr=False, compare=False)
    _legacy_pool_key: int = field(repr=False, compare=False)

    async def stream(self, prompt: AgentPrompt) -> AsyncIterator[StreamEvent]:
        """Stream normalized harness events without exposing the pool key."""
        async for event in self._pool.send(prompt, chat_id=self._legacy_pool_key):
            yield event

    async def effective_workspace(self) -> Path:
        """Return the run's effective project workspace through the adapter."""
        return await self._pool.get_effective_workspace(self._legacy_pool_key)


class WorkshopConversationRunService:
    """Prepare canonical conversation runs independently of an ingress client."""

    def __init__(
        self,
        pool: ConversationPool,
        resolver: Callable[[MessageId], Awaitable[CompatibilityConversationRunResolution]],
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
            model=self._pool.get_model(resolution._legacy_pool_key),
            _pool=self._pool,
            _legacy_pool_key=resolution._legacy_pool_key,
        )
