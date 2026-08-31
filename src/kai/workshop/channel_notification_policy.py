"""Canonical principal-owned channel notification policy."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import aiosqlite

from kai.workshop.domain import ChannelId, PrincipalId, RuntimeProfileId
from kai.workshop.execution_state import WorkshopExecutionStateRegistry

DEFAULT_CHANNEL_LEVEL = "mentions_replies"
CHANNEL_NOTIFICATION_LEVELS = ("all", DEFAULT_CHANNEL_LEVEL, "muted")


class WorkshopChannelNotificationPolicyError(RuntimeError):
    """Base failure for canonical channel notification policy."""


class WorkshopChannelNotificationPolicyAccessDenied(WorkshopChannelNotificationPolicyError):
    """The authenticated principal does not own the requested policy."""


class WorkshopChannelNotificationPolicyValidationError(WorkshopChannelNotificationPolicyError):
    """A requested policy value is invalid."""


class WorkshopChannelNotificationPolicyConflict(WorkshopChannelNotificationPolicyError):
    """Policy changed after the caller loaded it."""


class WorkshopChannelNotificationPolicyStorageError(WorkshopChannelNotificationPolicyError):
    """Canonical policy storage is unavailable or unsafe."""


@dataclass(frozen=True, slots=True)
class ChannelNotificationPolicyAuthority:
    principal_id: PrincipalId
    runtime_profile_id: RuntimeProfileId


@dataclass(frozen=True, slots=True)
class ChannelNotificationSetting:
    channel_id: ChannelId
    channel_name: str
    level: str
    source: str


@dataclass(frozen=True, slots=True)
class DoNotDisturbSetting:
    enabled: bool
    timezone: str
    start: str
    end: str


@dataclass(frozen=True, slots=True)
class ChannelNotificationPolicyMutation:
    operation: str
    changed: bool


@dataclass(frozen=True, slots=True)
class ChannelNotificationPolicySnapshot:
    channels: tuple[ChannelNotificationSetting, ...]
    muted_mentions_notify: bool
    do_not_disturb: DoNotDisturbSetting
    revision: str
    mutation: ChannelNotificationPolicyMutation | None = None


def _format_minute(value: int) -> str:
    return f"{value // 60:02d}:{value % 60:02d}"


def _parse_time(value: object, *, field: str) -> int:
    if not isinstance(value, str):
        raise WorkshopChannelNotificationPolicyValidationError(f"{field} must use HH:MM")
    parts = value.split(":")
    if len(parts) != 2 or any(len(part) != 2 or not part.isdigit() for part in parts):
        raise WorkshopChannelNotificationPolicyValidationError(f"{field} must use HH:MM")
    hour, minute = (int(part) for part in parts)
    if hour > 23 or minute > 59:
        raise WorkshopChannelNotificationPolicyValidationError(f"{field} must use HH:MM")
    return hour * 60 + minute


def _validate_timezone(value: object) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 100:
        raise WorkshopChannelNotificationPolicyValidationError("DND timezone must be an IANA timezone")
    normalized = value.strip()
    try:
        ZoneInfo(normalized)
    except ZoneInfoNotFoundError as exc:
        raise WorkshopChannelNotificationPolicyValidationError("DND timezone must be an IANA timezone") from exc
    return normalized


class WorkshopChannelNotificationPolicyService:
    """Inspect and mutate one principal's transport-neutral notification policy."""

    def __init__(
        self,
        connection: aiosqlite.Connection,
        execution_state: WorkshopExecutionStateRegistry,
    ) -> None:
        self._connection = connection
        self._execution_state = execution_state
        self._lock = asyncio.Lock()

    @classmethod
    async def open(
        cls,
        path: Path,
        execution_state: WorkshopExecutionStateRegistry,
    ) -> WorkshopChannelNotificationPolicyService:
        if not path.is_file():
            raise WorkshopChannelNotificationPolicyStorageError("Notification policy database is unavailable")
        connection = await aiosqlite.connect(str(path))
        connection.row_factory = aiosqlite.Row
        await connection.execute("PRAGMA busy_timeout=5000")
        await connection.execute("PRAGMA foreign_keys=ON")
        return cls(connection, execution_state)

    async def close(self) -> None:
        await self._connection.close()

    def authority_for_principal(
        self,
        principal_id: str | PrincipalId,
    ) -> ChannelNotificationPolicyAuthority:
        namespace = self._execution_state.maybe_for_principal_id(str(principal_id))
        if namespace is None:
            raise WorkshopChannelNotificationPolicyAccessDenied("The principal does not own notification policy")
        return ChannelNotificationPolicyAuthority(namespace.principal_id, namespace.runtime_profile_id)

    def authority_for_principal_profile(
        self,
        principal_id: str | PrincipalId,
        runtime_profile_id: str | RuntimeProfileId,
    ) -> ChannelNotificationPolicyAuthority:
        try:
            authority = ChannelNotificationPolicyAuthority(
                principal_id if isinstance(principal_id, PrincipalId) else PrincipalId(principal_id),
                runtime_profile_id
                if isinstance(runtime_profile_id, RuntimeProfileId)
                else RuntimeProfileId(runtime_profile_id),
            )
        except (TypeError, ValueError) as exc:
            raise WorkshopChannelNotificationPolicyAccessDenied(
                "The principal does not own notification policy"
            ) from exc
        self._validate_authority(authority)
        return authority

    async def inspect(
        self,
        authority: ChannelNotificationPolicyAuthority,
    ) -> ChannelNotificationPolicySnapshot:
        async with self._lock:
            self._validate_authority(authority)
            return await self._snapshot_locked(authority)

    async def external_delivery_allowed(
        self,
        principal_id: str | PrincipalId,
        occurred_at: datetime,
    ) -> bool:
        """Return whether adapter delivery is outside the principal's DND window."""
        if occurred_at.tzinfo is None or occurred_at.utcoffset() is None:
            raise WorkshopChannelNotificationPolicyValidationError("Delivery timestamp must be timezone-aware")
        authority = self.authority_for_principal(principal_id)
        snapshot = await self.inspect(authority)
        dnd = snapshot.do_not_disturb
        if not dnd.enabled:
            return True
        local = occurred_at.astimezone(ZoneInfo(dnd.timezone))
        minute = local.hour * 60 + local.minute
        start = _parse_time(dnd.start, field="DND start")
        end = _parse_time(dnd.end, field="DND end")
        inside = start <= minute < end if start < end else minute >= start or minute < end
        return not inside

    async def set_channel_level(
        self,
        authority: ChannelNotificationPolicyAuthority,
        channel_id: str | ChannelId,
        level: str,
        *,
        expected_revision: str,
    ) -> ChannelNotificationPolicySnapshot:
        normalized_level = level.strip().lower()
        if normalized_level not in CHANNEL_NOTIFICATION_LEVELS:
            raise WorkshopChannelNotificationPolicyValidationError(
                "Channel notification level must be all, mentions_replies, or muted"
            )
        try:
            canonical_channel_id = channel_id if isinstance(channel_id, ChannelId) else ChannelId(channel_id)
        except (TypeError, ValueError) as exc:
            raise WorkshopChannelNotificationPolicyValidationError("Channel is invalid") from exc
        async with self._lock:
            current = await self._checked_snapshot_locked(authority, expected_revision)
            existing = next((item for item in current.channels if item.channel_id == canonical_channel_id), None)
            if existing is None:
                raise WorkshopChannelNotificationPolicyAccessDenied("Channel is not accessible")
            try:
                await self._connection.execute("BEGIN IMMEDIATE")
                if normalized_level == DEFAULT_CHANNEL_LEVEL:
                    await self._connection.execute(
                        "DELETE FROM principal_channel_notification_policies WHERE principal_id = ? AND channel_id = ?",
                        (authority.principal_id, canonical_channel_id),
                    )
                else:
                    await self._connection.execute(
                        "INSERT INTO principal_channel_notification_policies "
                        "(principal_id, channel_id, level) VALUES (?, ?, ?) "
                        "ON CONFLICT(principal_id, channel_id) DO UPDATE SET "
                        "level = excluded.level, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
                        (authority.principal_id, canonical_channel_id, normalized_level),
                    )
                await self._connection.commit()
            except Exception:
                await self._connection.rollback()
                raise
            snapshot = await self._snapshot_locked(authority)
            return ChannelNotificationPolicySnapshot(
                snapshot.channels,
                snapshot.muted_mentions_notify,
                snapshot.do_not_disturb,
                snapshot.revision,
                ChannelNotificationPolicyMutation("set_channel_level", existing.level != normalized_level),
            )

    async def set_muted_mentions_notify(
        self,
        authority: ChannelNotificationPolicyAuthority,
        enabled: bool,
        *,
        expected_revision: str,
    ) -> ChannelNotificationPolicySnapshot:
        if not isinstance(enabled, bool):
            raise WorkshopChannelNotificationPolicyValidationError("Muted mention override must be true or false")
        async with self._lock:
            current = await self._checked_snapshot_locked(authority, expected_revision)
            await self._write_principal_policy_locked(
                authority,
                muted_mentions_notify=enabled,
                dnd=current.do_not_disturb,
            )
            snapshot = await self._snapshot_locked(authority)
            return ChannelNotificationPolicySnapshot(
                snapshot.channels,
                snapshot.muted_mentions_notify,
                snapshot.do_not_disturb,
                snapshot.revision,
                ChannelNotificationPolicyMutation(
                    "set_muted_mentions_notify",
                    current.muted_mentions_notify != enabled,
                ),
            )

    async def set_do_not_disturb(
        self,
        authority: ChannelNotificationPolicyAuthority,
        *,
        enabled: bool,
        timezone: object,
        start: object,
        end: object,
        expected_revision: str,
    ) -> ChannelNotificationPolicySnapshot:
        if not isinstance(enabled, bool):
            raise WorkshopChannelNotificationPolicyValidationError("DND enabled must be true or false")
        normalized_timezone = _validate_timezone(timezone)
        start_minute = _parse_time(start, field="DND start")
        end_minute = _parse_time(end, field="DND end")
        if start_minute == end_minute:
            raise WorkshopChannelNotificationPolicyValidationError("DND start and end must differ")
        dnd = DoNotDisturbSetting(
            enabled,
            normalized_timezone,
            _format_minute(start_minute),
            _format_minute(end_minute),
        )
        async with self._lock:
            current = await self._checked_snapshot_locked(authority, expected_revision)
            await self._write_principal_policy_locked(
                authority,
                muted_mentions_notify=current.muted_mentions_notify,
                dnd=dnd,
            )
            snapshot = await self._snapshot_locked(authority)
            return ChannelNotificationPolicySnapshot(
                snapshot.channels,
                snapshot.muted_mentions_notify,
                snapshot.do_not_disturb,
                snapshot.revision,
                ChannelNotificationPolicyMutation("set_do_not_disturb", current.do_not_disturb != dnd),
            )

    def _validate_authority(self, authority: ChannelNotificationPolicyAuthority) -> None:
        namespace = self._execution_state.maybe_for_runtime_profile_id(authority.runtime_profile_id)
        if namespace is None or namespace.principal_id != authority.principal_id:
            raise WorkshopChannelNotificationPolicyAccessDenied("The principal does not own notification policy")

    async def _checked_snapshot_locked(
        self,
        authority: ChannelNotificationPolicyAuthority,
        expected_revision: str,
    ) -> ChannelNotificationPolicySnapshot:
        self._validate_authority(authority)
        current = await self._snapshot_locked(authority)
        if current.revision != expected_revision:
            raise WorkshopChannelNotificationPolicyConflict("Notification policy changed since it was loaded")
        return current

    async def _write_principal_policy_locked(
        self,
        authority: ChannelNotificationPolicyAuthority,
        *,
        muted_mentions_notify: bool,
        dnd: DoNotDisturbSetting,
    ) -> None:
        start_minute = _parse_time(dnd.start, field="DND start")
        end_minute = _parse_time(dnd.end, field="DND end")
        try:
            await self._connection.execute("BEGIN IMMEDIATE")
            await self._connection.execute(
                "INSERT INTO principal_human_notification_policies "
                "(principal_id, muted_mentions_notify, dnd_enabled, dnd_timezone, "
                "dnd_start_minute, dnd_end_minute) VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(principal_id) DO UPDATE SET "
                "muted_mentions_notify = excluded.muted_mentions_notify, "
                "dnd_enabled = excluded.dnd_enabled, dnd_timezone = excluded.dnd_timezone, "
                "dnd_start_minute = excluded.dnd_start_minute, "
                "dnd_end_minute = excluded.dnd_end_minute, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
                (
                    authority.principal_id,
                    int(muted_mentions_notify),
                    int(dnd.enabled),
                    dnd.timezone,
                    start_minute,
                    end_minute,
                ),
            )
            await self._connection.commit()
        except Exception:
            await self._connection.rollback()
            raise

    async def _snapshot_locked(
        self,
        authority: ChannelNotificationPolicyAuthority,
    ) -> ChannelNotificationPolicySnapshot:
        async with self._connection.execute(
            "SELECT c.id, COALESCE(c.name, ''), p.level FROM channels c "
            "JOIN channel_memberships cm ON cm.channel_id = c.id AND cm.principal_id = ? "
            "LEFT JOIN principal_channel_notification_policies p "
            "ON p.principal_id = ? AND p.channel_id = c.id "
            "WHERE c.kind = 'group' AND c.archived_at IS NULL "
            "ORDER BY lower(COALESCE(c.name, '')), c.id",
            (authority.principal_id, authority.principal_id),
        ) as cursor:
            rows = tuple(await cursor.fetchall())
        channels = tuple(
            ChannelNotificationSetting(
                ChannelId(str(row[0])),
                str(row[1]).strip() or "Channel",
                str(row[2]) if row[2] is not None else DEFAULT_CHANNEL_LEVEL,
                "personal override" if row[2] is not None else "default",
            )
            for row in rows
        )
        async with self._connection.execute(
            "SELECT muted_mentions_notify, dnd_enabled, dnd_timezone, "
            "dnd_start_minute, dnd_end_minute FROM principal_human_notification_policies "
            "WHERE principal_id = ?",
            (authority.principal_id,),
        ) as cursor:
            policy = await cursor.fetchone()
        muted_mentions_notify = True if policy is None else bool(policy[0])
        dnd = DoNotDisturbSetting(
            False if policy is None else bool(policy[1]),
            "UTC" if policy is None else str(policy[2]),
            "22:00" if policy is None else _format_minute(int(policy[3])),
            "07:00" if policy is None else _format_minute(int(policy[4])),
        )
        encoded = json.dumps(
            {
                "principal_id": str(authority.principal_id),
                "channels": [[str(item.channel_id), item.level] for item in channels],
                "muted_mentions_notify": muted_mentions_notify,
                "dnd": [dnd.enabled, dnd.timezone, dnd.start, dnd.end],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        revision = "cnp_" + hashlib.sha256(encoded.encode()).hexdigest()[:32]
        return ChannelNotificationPolicySnapshot(channels, muted_mentions_notify, dnd, revision)
