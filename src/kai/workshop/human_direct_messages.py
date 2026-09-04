"""Canonical private direct conversations between two Workshop humans."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import UTC, datetime

from kai.workshop.domain import (
    ChannelId,
    ChannelMembershipId,
    EventEnvelope,
    PrincipalId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.human_avatars import human_avatar_descriptors
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import WorkshopEventStore


class WorkshopHumanDirectMessageError(RuntimeError):
    """A canonical human direct-message operation could not be completed."""


class WorkshopHumanDirectMessageAccessDenied(WorkshopHumanDirectMessageError):
    """The actor cannot discover or contact the requested Workshop human."""


class WorkshopHumanDirectMessageConflict(WorkshopHumanDirectMessageError):
    """Existing canonical state does not identify one safe human conversation."""


class WorkshopHumanDirectMessageStorageError(WorkshopHumanDirectMessageError):
    """A human direct-message mutation could not be persisted atomically."""


@dataclass(frozen=True, slots=True)
class WorkshopHumanPeer:
    """One same-Workshop human the authenticated principal may contact."""

    principal_id: PrincipalId
    display_name: str
    handle: str
    channel_id: ChannelId | None
    avatar_state_version: int = 0
    avatar_active: bool = False


@dataclass(frozen=True, slots=True)
class WorkshopHumanPeerSnapshot:
    """Principal-bounded human discovery within one Workshop."""

    workshop_id: WorkshopId
    peers: tuple[WorkshopHumanPeer, ...]


@dataclass(frozen=True, slots=True)
class WorkshopHumanDirectConversation:
    """One canonical unordered human pair and its private channel."""

    workshop_id: WorkshopId
    channel_id: ChannelId
    peer: WorkshopHumanPeer
    created: bool


async def is_canonical_human_direct_channel(
    store: WorkshopEventStore,
    channel_id: ChannelId,
) -> bool:
    """Return whether a channel has the exact non-executable human-DM shape."""
    if not isinstance(channel_id, ChannelId):
        return False
    async with store.connection.execute(
        "SELECT c.kind, c.archived_at, "
        "(SELECT COUNT(*) FROM channel_memberships cm WHERE cm.channel_id = c.id), "
        "(SELECT COUNT(*) FROM channel_memberships cm JOIN principals p "
        "ON p.id = cm.principal_id AND p.kind = 'human' WHERE cm.channel_id = c.id), "
        "(SELECT COUNT(*) FROM channel_memberships cm JOIN workshop_memberships wm "
        "ON wm.principal_id = cm.principal_id AND wm.workshop_id = c.workshop_id "
        "WHERE cm.channel_id = c.id), "
        "(SELECT COUNT(*) FROM channel_memberships cm WHERE cm.channel_id = c.id AND cm.role = 'owner'), "
        "(SELECT COUNT(*) FROM channel_agents ca WHERE ca.channel_id = c.id), "
        "(SELECT COUNT(*) FROM channel_agent_runtime_assignments cara WHERE cara.channel_id = c.id), "
        "(SELECT COUNT(*) FROM channel_bindings cb WHERE cb.channel_id = c.id) "
        "FROM channels c WHERE c.id = ?",
        (channel_id,),
    ) as cursor:
        row = await cursor.fetchone()
    return bool(
        row is not None
        and str(row[0]) == "direct"
        and row[1] is None
        and int(row[2]) == 2
        and int(row[3]) == 2
        and int(row[4]) == 2
        and int(row[5]) == 2
        and int(row[6]) == 0
        and int(row[7]) == 0
        and int(row[8]) == 0
    )


class WorkshopHumanDirectMessageService:
    """Discover peers and idempotently create one channel per human pair."""

    def __init__(self, store: WorkshopEventStore) -> None:
        self._store = store

    async def list_peers(
        self,
        principal_id: PrincipalId,
        workshop_id: WorkshopId,
    ) -> WorkshopHumanPeerSnapshot:
        await self._require_workshop_human(principal_id, workshop_id)
        async with self._store.connection.execute(
            "SELECT p.id, p.display_name, hh.handle FROM workshop_memberships wm "
            "JOIN principals p ON p.id = wm.principal_id AND p.kind = 'human' "
            "JOIN human_handles hh ON hh.workshop_id = wm.workshop_id AND hh.principal_id = p.id "
            "WHERE wm.workshop_id = ? AND wm.principal_id != ? "
            "ORDER BY lower(p.display_name), p.id",
            (workshop_id, principal_id),
        ) as cursor:
            rows = list(await cursor.fetchall())
        peers: list[WorkshopHumanPeer] = []
        for row in rows:
            peer_id = PrincipalId(str(row[0]))
            channels = await self._matching_channels(workshop_id, principal_id, peer_id)
            if len(channels) > 1:
                raise WorkshopHumanDirectMessageConflict(
                    "Multiple canonical direct conversations exist for this human pair"
                )
            peers.append(
                WorkshopHumanPeer(
                    peer_id,
                    str(row[1]),
                    str(row[2]),
                    channels[0] if channels else None,
                )
            )
        descriptors = await human_avatar_descriptors(
            self._store,
            (peer.principal_id for peer in peers),
        )
        peers = [
            replace(
                peer,
                avatar_state_version=descriptors[peer.principal_id].state_version,
                avatar_active=descriptors[peer.principal_id].active,
            )
            if peer.principal_id in descriptors
            else peer
            for peer in peers
        ]
        return WorkshopHumanPeerSnapshot(workshop_id, tuple(peers))

    async def start(
        self,
        principal_id: PrincipalId,
        workshop_id: WorkshopId,
        peer_principal_id: PrincipalId,
    ) -> WorkshopHumanDirectConversation:
        """Create or return the sole canonical conversation for an unordered pair."""
        if principal_id == peer_principal_id:
            raise WorkshopHumanDirectMessageAccessDenied("A human cannot start a direct conversation with themself")
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            await self._require_workshop_human(principal_id, workshop_id)
            peer = await self._require_workshop_human(peer_principal_id, workshop_id)
            channels = await self._matching_channels(workshop_id, principal_id, peer_principal_id)
            if len(channels) > 1:
                raise WorkshopHumanDirectMessageConflict(
                    "Multiple canonical direct conversations exist for this human pair"
                )
            if channels:
                await connection.rollback()
                return WorkshopHumanDirectConversation(
                    workshop_id,
                    channels[0],
                    await self._with_avatar(WorkshopHumanPeer(peer_principal_id, peer[0], peer[1], channels[0])),
                    False,
                )

            low, high = sorted((str(principal_id), str(peer_principal_id)))
            channel_id = ChannelId.derived(workshop_id, f"human-direct:{low}:{high}")
            async with connection.execute("SELECT 1 FROM channels WHERE id = ?", (channel_id,)) as cursor:
                if await cursor.fetchone() is not None:
                    raise WorkshopHumanDirectMessageConflict(
                        "The deterministic human conversation identity is already occupied"
                    )
            now = datetime.now(UTC)
            events = (
                EventEnvelope.create(
                    event_type=WorkshopEventType.CHANNEL_CREATED,
                    event_version=1,
                    workshop_id=workshop_id,
                    aggregate_type="channel",
                    aggregate_id=channel_id,
                    actor_principal_id=principal_id,
                    occurred_at=now,
                    idempotency_key=f"workshop:human-direct:{low}:{high}:channel",
                    payload={"kind": "direct", "name": "Direct"},
                    metadata={"source": "workshop_client", "conversation_kind": "human_direct"},
                ),
                *(
                    EventEnvelope.create(
                        event_type=WorkshopEventType.CHANNEL_MEMBER_ADDED,
                        event_version=1,
                        workshop_id=workshop_id,
                        aggregate_type="channel_membership",
                        aggregate_id=ChannelMembershipId.derived(channel_id, f"principal:{member_id}"),
                        actor_principal_id=principal_id,
                        occurred_at=now,
                        idempotency_key=f"workshop:human-direct:{low}:{high}:member:{member_id}",
                        payload={
                            "channel_id": channel_id,
                            "principal_id": member_id,
                            "role": "owner",
                        },
                        metadata={"source": "workshop_client", "conversation_kind": "human_direct"},
                    )
                    for member_id in (PrincipalId(low), PrincipalId(high))
                ),
            )
            for event in events:
                appended = await self._store.append_in_transaction(event)
                if not appended.inserted:
                    raise WorkshopHumanDirectMessageConflict(
                        "Human conversation creation conflicted with existing canonical events"
                    )
            await self._store.project_pending_in_transaction(CanonicalConversationProjection())
            if not await is_canonical_human_direct_channel(self._store, channel_id):
                raise WorkshopHumanDirectMessageConflict("Created human conversation failed canonical validation")
            await connection.commit()
            return WorkshopHumanDirectConversation(
                workshop_id,
                channel_id,
                await self._with_avatar(WorkshopHumanPeer(peer_principal_id, peer[0], peer[1], channel_id)),
                True,
            )
        except WorkshopHumanDirectMessageError:
            await connection.rollback()
            raise
        except Exception as exc:
            await connection.rollback()
            raise WorkshopHumanDirectMessageStorageError("Human direct conversation could not be persisted") from exc

    async def _require_workshop_human(
        self,
        principal_id: PrincipalId,
        workshop_id: WorkshopId,
    ) -> tuple[str, str]:
        if not isinstance(principal_id, PrincipalId) or not isinstance(workshop_id, WorkshopId):
            raise WorkshopHumanDirectMessageAccessDenied("Canonical Workshop human identity is required")
        async with self._store.connection.execute(
            "SELECT p.display_name, hh.handle FROM workshop_memberships wm "
            "JOIN principals p ON p.id = wm.principal_id AND p.kind = 'human' "
            "JOIN human_handles hh ON hh.workshop_id = wm.workshop_id AND hh.principal_id = p.id "
            "WHERE wm.workshop_id = ? AND wm.principal_id = ?",
            (workshop_id, principal_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise WorkshopHumanDirectMessageAccessDenied("Human is unavailable in this Workshop")
        return str(row[0]), str(row[1])

    async def _with_avatar(self, peer: WorkshopHumanPeer) -> WorkshopHumanPeer:
        descriptors = await human_avatar_descriptors(self._store, (peer.principal_id,))
        descriptor = descriptors.get(peer.principal_id)
        if descriptor is None:
            return peer
        return replace(
            peer,
            avatar_state_version=descriptor.state_version,
            avatar_active=descriptor.active,
        )

    async def _matching_channels(
        self,
        workshop_id: WorkshopId,
        first: PrincipalId,
        second: PrincipalId,
    ) -> tuple[ChannelId, ...]:
        async with self._store.connection.execute(
            "SELECT c.id FROM channels c WHERE c.workshop_id = ? AND c.kind = 'direct' "
            "AND c.archived_at IS NULL "
            "AND (SELECT COUNT(*) FROM channel_memberships cm WHERE cm.channel_id = c.id) = 2 "
            "AND EXISTS (SELECT 1 FROM channel_memberships cm WHERE cm.channel_id = c.id "
            "AND cm.principal_id = ? AND cm.role = 'owner') "
            "AND EXISTS (SELECT 1 FROM channel_memberships cm WHERE cm.channel_id = c.id "
            "AND cm.principal_id = ? AND cm.role = 'owner') "
            "AND NOT EXISTS (SELECT 1 FROM channel_agents ca WHERE ca.channel_id = c.id) "
            "AND NOT EXISTS (SELECT 1 FROM channel_agent_runtime_assignments cara WHERE cara.channel_id = c.id) "
            "AND NOT EXISTS (SELECT 1 FROM channel_bindings cb WHERE cb.channel_id = c.id) "
            "ORDER BY c.id",
            (workshop_id, first, second),
        ) as cursor:
            return tuple(ChannelId(str(row[0])) for row in await cursor.fetchall())
