"""Canonical principal authorization policies for Kai Workshop."""

from __future__ import annotations

from kai.workshop.domain import ChannelId, PrincipalId
from kai.workshop.store import WorkshopEventStore


class CanonicalChannelAuthorizer:
    """Authorize reads through explicit channel and workshop memberships."""

    def __init__(self, store: WorkshopEventStore) -> None:
        self._store = store

    async def can_read_channel(self, principal_id: PrincipalId, channel_id: ChannelId) -> bool:
        """Return true only for an explicit, workshop-aligned channel member."""
        if not isinstance(principal_id, PrincipalId) or not isinstance(channel_id, ChannelId):
            return False
        async with self._store.connection.execute(
            "SELECT 1 FROM channel_memberships cm "
            "JOIN channels c ON c.id = cm.channel_id "
            "JOIN workshop_memberships wm ON wm.principal_id = cm.principal_id "
            "AND wm.workshop_id = c.workshop_id "
            "WHERE cm.principal_id = ? AND cm.channel_id = ? LIMIT 1",
            (principal_id, channel_id),
        ) as cursor:
            return await cursor.fetchone() is not None

    async def can_submit_command(self, principal_id: PrincipalId, channel_id: ChannelId) -> bool:
        """Authorize a human in one executable conversational channel."""
        if not isinstance(principal_id, PrincipalId) or not isinstance(channel_id, ChannelId):
            return False
        async with self._store.connection.execute(
            "SELECT 1 FROM channel_memberships cm "
            "JOIN principals p ON p.id = cm.principal_id AND p.kind = 'human' "
            "JOIN channels c ON c.id = cm.channel_id "
            "JOIN workshop_memberships wm ON wm.principal_id = cm.principal_id "
            "AND wm.workshop_id = c.workshop_id "
            "WHERE cm.principal_id = ? AND cm.channel_id = ? "
            "AND c.kind IN ('direct', 'group') "
            "AND EXISTS (SELECT 1 FROM channel_agents ca "
            "JOIN channel_agent_runtime_assignments cara "
            "ON cara.channel_id = ca.channel_id AND cara.agent_id = ca.agent_id "
            "LEFT JOIN agent_definitions sponsored "
            "ON sponsored.agent_id = ca.agent_id "
            "AND sponsored.lifecycle_state = 'active' "
            "WHERE ca.channel_id = c.id AND ca.detached_at IS NULL "
            "AND (c.kind = 'direct' OR (sponsored.id IS NOT NULL AND ("
            "(sponsored.owner_principal_id = ca.sponsor_principal_id "
            "AND sponsored.owner_runtime_profile_id = ca.sponsored_runtime_profile_id) "
            "OR (sponsored.owner_principal_id IS NULL AND EXISTS ("
            "SELECT 1 FROM principal_agent_enablements legacy "
            "WHERE legacy.agent_definition_id = sponsored.id "
            "AND legacy.principal_id = ca.sponsor_principal_id "
            "AND legacy.runtime_profile_id = ca.sponsored_runtime_profile_id "
            "AND legacy.lifecycle_state = 'enabled'))))) "
            ") "
            "LIMIT 1",
            (principal_id, channel_id),
        ) as cursor:
            return await cursor.fetchone() is not None
