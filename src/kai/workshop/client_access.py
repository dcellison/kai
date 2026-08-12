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

    async def issue_enrollment(self, telegram_user_id: int) -> IssuedClientEnrollment:
        principal_id, channel_id = await self._resolve_direct_human(telegram_user_id)
        grant = await self._enrollment.issue_grant(principal_id)
        return IssuedClientEnrollment(grant, channel_id)

    async def revoke_device(self, telegram_user_id: int, device_id: DeviceId) -> None:
        principal_id, _ = await self._resolve_direct_human(telegram_user_id)
        if not await self._sessions.revoke_device(principal_id, device_id):
            raise WorkshopClientAccessError("Client device is unavailable for that canonical human")

    async def revoke_enrollment(self, telegram_user_id: int, grant_id: EnrollmentGrantId) -> None:
        principal_id, _ = await self._resolve_direct_human(telegram_user_id)
        if not await self._enrollment.revoke_grant(principal_id, grant_id):
            raise WorkshopClientAccessError("Enrollment grant is unavailable for that canonical human")
