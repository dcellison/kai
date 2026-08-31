"""Trusted operator controls for Workshop human-client enrollment and revocation."""

from __future__ import annotations

import re
from dataclasses import dataclass

from kai.workshop.client_sessions import (
    IssuedWorkshopEnrollmentGrant,
    WorkshopClientEnrollmentManager,
    WorkshopClientSessionManager,
)
from kai.workshop.domain import ChannelId, DeviceId, EnrollmentGrantId, PrincipalId
from kai.workshop.store import WorkshopEventStore


class WorkshopClientAccessError(RuntimeError):
    """An operator client-access request did not resolve safely."""


@dataclass(frozen=True, slots=True)
class IssuedClientEnrollment:
    grant: IssuedWorkshopEnrollmentGrant
    channel_id: ChannelId


@dataclass(frozen=True, slots=True)
class EnrollableWorkshopHuman:
    principal_id: PrincipalId
    display_name: str
    handle: str
    direct_channels: tuple[ChannelId, ...]


_EXTERNAL_IDENTITY_KIND = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")


def _external_identity_value(kind: str, value: str) -> str:
    if not isinstance(kind, str) or _EXTERNAL_IDENTITY_KIND.fullmatch(kind) is None:
        raise WorkshopClientAccessError("External identity kind is invalid")
    if not isinstance(value, str) or not value or len(value) > 512:
        raise WorkshopClientAccessError("External identity value is invalid")
    return value


class WorkshopClientAccess:
    """Issue and revoke credentials only for an existing canonical human."""

    def __init__(self, store: WorkshopEventStore) -> None:
        self._store = store
        self._enrollment = WorkshopClientEnrollmentManager(store)
        self._sessions = WorkshopClientSessionManager(store)

    async def list_humans(self) -> tuple[EnrollableWorkshopHuman, ...]:
        """List canonical humans and their owned direct channels."""
        async with self._store.connection.execute(
            "SELECT p.id, p.display_name, hh.handle, c.id FROM principals p "
            "JOIN workshop_memberships wm ON wm.principal_id = p.id "
            "JOIN human_handles hh ON hh.workshop_id = wm.workshop_id AND hh.principal_id = p.id "
            "LEFT JOIN channel_memberships cm ON cm.principal_id = p.id AND cm.role = 'owner' "
            "LEFT JOIN channels c ON c.id = cm.channel_id AND c.workshop_id = wm.workshop_id "
            "AND c.kind = 'direct' WHERE p.kind = 'human' ORDER BY p.display_name, p.id, c.id"
        ) as cursor:
            rows = list(await cursor.fetchall())
        humans: dict[PrincipalId, tuple[str, str, list[ChannelId]]] = {}
        for row in rows:
            principal_id = PrincipalId(str(row[0]))
            entry = humans.setdefault(principal_id, (str(row[1]), str(row[2]), []))
            if row[3] is not None:
                entry[2].append(ChannelId(str(row[3])))
        return tuple(
            EnrollableWorkshopHuman(principal_id, display_name, handle, tuple(channels))
            for principal_id, (display_name, handle, channels) in humans.items()
        )

    async def _require_direct_human(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
    ) -> None:
        if not isinstance(principal_id, PrincipalId) or not isinstance(channel_id, ChannelId):
            raise WorkshopClientAccessError("Canonical principal and channel IDs are required")
        async with self._store.connection.execute(
            "SELECT 1 FROM principals p "
            "JOIN workshop_memberships wm ON wm.principal_id = p.id "
            "JOIN channels c ON c.workshop_id = wm.workshop_id AND c.id = ? AND c.kind = 'direct' "
            "JOIN channel_memberships cm ON cm.channel_id = c.id AND cm.principal_id = p.id "
            "AND cm.role = 'owner' WHERE p.id = ? AND p.kind = 'human' LIMIT 1",
            (channel_id, principal_id),
        ) as cursor:
            available = await cursor.fetchone() is not None
        if not available:
            raise WorkshopClientAccessError("Canonical human does not own that direct channel in an active Workshop")

    async def resolve_external_direct_human(
        self,
        *,
        provider: str,
        external_subject: str,
        transport: str,
        external_channel_id: str,
    ) -> tuple[PrincipalId, ChannelId]:
        """Resolve one adapter identity and channel to a canonical direct human."""
        subject = _external_identity_value(provider, external_subject)
        channel_subject = _external_identity_value(transport, external_channel_id)
        async with self._store.connection.execute(
            "SELECT ei.principal_id, c.id FROM external_identities ei "
            "JOIN principals p ON p.id = ei.principal_id AND p.kind = 'human' "
            "JOIN channel_memberships cm ON cm.principal_id = p.id AND cm.role = 'owner' "
            "JOIN channels c ON c.id = cm.channel_id AND c.kind = 'direct' "
            "JOIN workshop_memberships wm ON wm.workshop_id = c.workshop_id AND wm.principal_id = p.id "
            "JOIN channel_bindings cb ON cb.channel_id = c.id AND cb.transport = ? "
            "AND cb.external_channel_id = ? "
            "WHERE ei.provider = ? AND ei.external_subject = ?",
            (transport, channel_subject, provider, subject),
        ) as cursor:
            rows = list(await cursor.fetchall())
        if len(rows) != 1:
            raise WorkshopClientAccessError(
                "External identity does not resolve to exactly one canonical human and direct channel; restart "
                "Kai after configuring the adapter binding"
            )
        return PrincipalId(str(rows[0][0])), ChannelId(str(rows[0][1]))

    async def issue_enrollment(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
    ) -> IssuedClientEnrollment:
        await self._require_direct_human(principal_id, channel_id)
        grant = await self._enrollment.issue_grant(principal_id)
        return IssuedClientEnrollment(grant, channel_id)

    async def revoke_device(self, principal_id: PrincipalId, device_id: DeviceId) -> None:
        if not await self._sessions.revoke_device(principal_id, device_id):
            raise WorkshopClientAccessError("Client device is unavailable for that canonical human")

    async def revoke_enrollment(
        self,
        principal_id: PrincipalId,
        grant_id: EnrollmentGrantId,
    ) -> None:
        if not await self._enrollment.revoke_grant(principal_id, grant_id):
            raise WorkshopClientAccessError("Enrollment grant is unavailable for that canonical human")
