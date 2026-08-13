"""Trusted operator controls for Workshop human-client enrollment and revocation."""

from __future__ import annotations

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
    direct_channels: tuple[ChannelId, ...]


def _telegram_subject(telegram_user_id: int) -> str:
    if (
        not isinstance(telegram_user_id, int)
        or isinstance(telegram_user_id, bool)
        or not 1 <= telegram_user_id <= 2**63 - 1
    ):
        raise WorkshopClientAccessError("Telegram user ID must be a positive signed 64-bit integer")
    return str(telegram_user_id)


class WorkshopClientAccess:
    """Issue and revoke credentials only for an existing canonical human."""

    def __init__(self, store: WorkshopEventStore) -> None:
        self._store = store
        self._enrollment = WorkshopClientEnrollmentManager(store)
        self._sessions = WorkshopClientSessionManager(store)

    async def list_humans(self) -> tuple[EnrollableWorkshopHuman, ...]:
        """List canonical humans and their owned direct channels."""
        async with self._store.connection.execute(
            "SELECT p.id, p.display_name, c.id FROM principals p "
            "JOIN workshop_memberships wm ON wm.principal_id = p.id "
            "LEFT JOIN channel_memberships cm ON cm.principal_id = p.id AND cm.role = 'owner' "
            "LEFT JOIN channels c ON c.id = cm.channel_id AND c.workshop_id = wm.workshop_id "
            "AND c.kind = 'direct' WHERE p.kind = 'human' ORDER BY p.display_name, p.id, c.id"
        ) as cursor:
            rows = list(await cursor.fetchall())
        humans: dict[PrincipalId, tuple[str, list[ChannelId]]] = {}
        for row in rows:
            principal_id = PrincipalId(str(row[0]))
            entry = humans.setdefault(principal_id, (str(row[1]), []))
            if row[2] is not None:
                entry[1].append(ChannelId(str(row[2])))
        return tuple(
            EnrollableWorkshopHuman(principal_id, display_name, tuple(channels))
            for principal_id, (display_name, channels) in humans.items()
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

    async def _resolve_direct_human(self, telegram_user_id: int) -> tuple[PrincipalId, ChannelId]:
        subject = _telegram_subject(telegram_user_id)
        async with self._store.connection.execute(
            "SELECT ei.principal_id, c.id FROM external_identities ei "
            "JOIN principals p ON p.id = ei.principal_id AND p.kind = 'human' "
            "JOIN channel_memberships cm ON cm.principal_id = p.id AND cm.role = 'owner' "
            "JOIN channels c ON c.id = cm.channel_id AND c.kind = 'direct' "
            "JOIN workshop_memberships wm ON wm.workshop_id = c.workshop_id AND wm.principal_id = p.id "
            "JOIN channel_bindings cb ON cb.channel_id = c.id AND cb.transport = 'telegram' "
            "AND cb.external_channel_id = ei.external_subject "
            "WHERE ei.provider = 'telegram' AND ei.external_subject = ?",
            (subject,),
        ) as cursor:
            rows = list(await cursor.fetchall())
        if len(rows) != 1:
            raise WorkshopClientAccessError(
                "Telegram user does not resolve to exactly one canonical human and direct channel; restart Kai "
                "after configuring the user"
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

    async def issue_enrollment_for_telegram(self, telegram_user_id: int) -> IssuedClientEnrollment:
        principal_id, channel_id = await self._resolve_direct_human(telegram_user_id)
        return await self.issue_enrollment(principal_id, channel_id)

    async def revoke_device(self, principal_id: PrincipalId, device_id: DeviceId) -> None:
        if not await self._sessions.revoke_device(principal_id, device_id):
            raise WorkshopClientAccessError("Client device is unavailable for that canonical human")

    async def revoke_device_for_telegram(self, telegram_user_id: int, device_id: DeviceId) -> None:
        principal_id, _ = await self._resolve_direct_human(telegram_user_id)
        await self.revoke_device(principal_id, device_id)

    async def revoke_enrollment(
        self,
        principal_id: PrincipalId,
        grant_id: EnrollmentGrantId,
    ) -> None:
        if not await self._enrollment.revoke_grant(principal_id, grant_id):
            raise WorkshopClientAccessError("Enrollment grant is unavailable for that canonical human")

    async def revoke_enrollment_for_telegram(
        self,
        telegram_user_id: int,
        grant_id: EnrollmentGrantId,
    ) -> None:
        principal_id, _ = await self._resolve_direct_human(telegram_user_id)
        await self.revoke_enrollment(principal_id, grant_id)
