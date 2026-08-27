"""Principal-scoped canonical destinations for integration notifications."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import aiosqlite

from kai.workshop.domain import ChannelId, PrincipalId, RuntimeProfileId
from kai.workshop.execution_state import WorkshopExecutionStateRegistry

GITHUB_INTEGRATION_CLASS = "github"
GENERIC_INTEGRATION_CLASS = "generic"
_SUPPORTED_CLASSES = frozenset({GITHUB_INTEGRATION_CLASS, GENERIC_INTEGRATION_CLASS})
_GENERIC_DEFAULT_ROUTE = (GENERIC_INTEGRATION_CLASS, "default")


class WorkshopNotificationPreferenceError(RuntimeError):
    """Base failure for canonical notification-delivery preferences."""


class WorkshopNotificationPreferenceAccessDenied(WorkshopNotificationPreferenceError):
    """The authenticated principal does not own this preference record."""


class WorkshopNotificationPreferenceValidationError(WorkshopNotificationPreferenceError):
    """A requested destination or integration class is invalid."""


class WorkshopNotificationPreferenceConflict(WorkshopNotificationPreferenceError):
    """Preferences changed after the caller loaded them."""


class WorkshopNotificationPreferenceStorageError(WorkshopNotificationPreferenceError):
    """Canonical notification preference state is unavailable or unsafe."""


@dataclass(frozen=True, slots=True)
class NotificationPreferenceAuthority:
    principal_id: PrincipalId
    runtime_profile_id: RuntimeProfileId


@dataclass(frozen=True, slots=True)
class NotificationDestinationChoice:
    choice_id: str
    display_name: str
    kind: str
    supported_classes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IntegrationDeliveryPreference:
    integration_class: str
    display_name: str
    destination_choice_id: str
    destination_name: str
    destination_kind: str
    source: str
    editable: bool
    resettable: bool


@dataclass(frozen=True, slots=True)
class NotificationPreferenceMutation:
    operation: str
    changed: bool


@dataclass(frozen=True, slots=True)
class NotificationPreferenceSnapshot:
    destinations: tuple[NotificationDestinationChoice, ...]
    preferences: tuple[IntegrationDeliveryPreference, ...]
    revision: str
    mutation: NotificationPreferenceMutation | None = None


@dataclass(frozen=True, slots=True)
class _Destination:
    channel_id: ChannelId
    choice_id: str
    display_name: str
    kind: str


@dataclass(frozen=True, slots=True)
class _EffectivePreference:
    integration_class: str
    destination: _Destination
    source: str
    resettable: bool


def _choice_id(principal_id: PrincipalId, channel_id: ChannelId) -> str:
    encoded = f"kai-notification-destination:v1:{principal_id}:{channel_id}".encode()
    return "ndst_" + hashlib.sha256(encoded).hexdigest()[:32]


class WorkshopNotificationPreferenceService:
    """Authorize and resolve personal integration destinations without adapters."""

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
    ) -> WorkshopNotificationPreferenceService:
        if not path.is_file():
            raise WorkshopNotificationPreferenceStorageError("Notification preference database is unavailable")
        connection = await aiosqlite.connect(str(path))
        connection.row_factory = aiosqlite.Row
        await connection.execute("PRAGMA busy_timeout=5000")
        await connection.execute("PRAGMA foreign_keys=ON")
        return cls(connection, execution_state)

    async def close(self) -> None:
        await self._connection.close()

    async def reconcile_protected_routes(self) -> None:
        """Bind an unambiguous protected generic route to its human owner."""
        async with self._lock:
            await self._reconcile_generic_owner_locked()

    def authority_for_principal(
        self,
        principal_id: str | PrincipalId,
    ) -> NotificationPreferenceAuthority:
        namespace = self._execution_state.maybe_for_principal_id(str(principal_id))
        if namespace is None:
            raise WorkshopNotificationPreferenceAccessDenied("The principal does not own notification preferences")
        return NotificationPreferenceAuthority(
            namespace.principal_id,
            namespace.runtime_profile_id,
        )

    def authority_for_principal_profile(
        self,
        principal_id: str | PrincipalId,
        runtime_profile_id: str | RuntimeProfileId,
    ) -> NotificationPreferenceAuthority:
        try:
            canonical_principal = principal_id if isinstance(principal_id, PrincipalId) else PrincipalId(principal_id)
            canonical_profile = (
                runtime_profile_id
                if isinstance(runtime_profile_id, RuntimeProfileId)
                else RuntimeProfileId(runtime_profile_id)
            )
        except (TypeError, ValueError) as exc:
            raise WorkshopNotificationPreferenceAccessDenied(
                "The principal does not own notification preferences"
            ) from exc
        authority = NotificationPreferenceAuthority(
            canonical_principal,
            canonical_profile,
        )
        self._validate_authority(authority)
        return authority

    async def inspect(
        self,
        authority: NotificationPreferenceAuthority,
    ) -> NotificationPreferenceSnapshot:
        async with self._lock:
            self._validate_authority(authority)
            await self._reconcile_generic_owner_locked()
            return await self._snapshot_locked(authority)

    async def select(
        self,
        authority: NotificationPreferenceAuthority,
        integration_class: str,
        choice_id: str,
        *,
        expected_revision: str,
    ) -> NotificationPreferenceSnapshot:
        normalized_class = self._validate_integration_class(integration_class)
        if not choice_id.startswith("ndst_") or len(choice_id) != 37:
            raise WorkshopNotificationPreferenceValidationError("Notification destination is invalid")
        async with self._lock:
            self._validate_authority(authority)
            await self._reconcile_generic_owner_locked()
            current = await self._snapshot_locked(authority)
            if current.revision != expected_revision:
                raise WorkshopNotificationPreferenceConflict("Notification preferences changed since they were loaded")
            supported = {item.integration_class for item in current.preferences if item.editable}
            if normalized_class not in supported:
                raise WorkshopNotificationPreferenceAccessDenied(
                    "The integration class is not editable by this principal"
                )
            destinations = await self._authorized_destinations_locked(authority)
            destination = next((item for item in destinations if item.choice_id == choice_id), None)
            if destination is None:
                raise WorkshopNotificationPreferenceAccessDenied("The notification destination is not authorized")
            changed = not any(
                item.integration_class == normalized_class
                and item.destination_choice_id == destination.choice_id
                and item.source == "personal override"
                for item in current.preferences
            )
            try:
                await self._connection.execute("BEGIN IMMEDIATE")
                await self._connection.execute(
                    "INSERT INTO principal_notification_delivery_preferences "
                    "(principal_id, integration_class, channel_id) VALUES (?, ?, ?) "
                    "ON CONFLICT(principal_id, integration_class) DO UPDATE SET "
                    "channel_id = excluded.channel_id, "
                    "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')",
                    (authority.principal_id, normalized_class, destination.channel_id),
                )
                await self._connection.commit()
            except Exception:
                await self._connection.rollback()
                raise
            snapshot = await self._snapshot_locked(authority)
            return NotificationPreferenceSnapshot(
                snapshot.destinations,
                snapshot.preferences,
                snapshot.revision,
                NotificationPreferenceMutation(
                    f"select_{normalized_class}_notification_destination",
                    changed,
                ),
            )

    async def reset(
        self,
        authority: NotificationPreferenceAuthority,
        integration_class: str,
        *,
        expected_revision: str,
    ) -> NotificationPreferenceSnapshot:
        normalized_class = self._validate_integration_class(integration_class)
        async with self._lock:
            self._validate_authority(authority)
            await self._reconcile_generic_owner_locked()
            current = await self._snapshot_locked(authority)
            if current.revision != expected_revision:
                raise WorkshopNotificationPreferenceConflict("Notification preferences changed since they were loaded")
            supported = {item.integration_class for item in current.preferences if item.editable}
            if normalized_class not in supported:
                raise WorkshopNotificationPreferenceAccessDenied(
                    "The integration class is not editable by this principal"
                )
            cursor = await self._connection.execute(
                "DELETE FROM principal_notification_delivery_preferences "
                "WHERE principal_id = ? AND integration_class = ?",
                (authority.principal_id, normalized_class),
            )
            await self._connection.commit()
            snapshot = await self._snapshot_locked(authority)
            return NotificationPreferenceSnapshot(
                snapshot.destinations,
                snapshot.preferences,
                snapshot.revision,
                NotificationPreferenceMutation(
                    f"reset_{normalized_class}_notification_destination",
                    cursor.rowcount == 1,
                ),
            )

    async def effective_channel(
        self,
        principal_id: PrincipalId,
        integration_class: str,
    ) -> ChannelId:
        authority = self.authority_for_principal(principal_id)
        normalized_class = self._validate_integration_class(integration_class)
        async with self._lock:
            await self._reconcile_generic_owner_locked()
            effective = await self._effective_locked(authority, normalized_class)
            return effective.destination.channel_id

    async def effective_channel_for_route(
        self,
        *,
        source: str,
        route_name: str,
    ) -> ChannelId:
        async with self._lock:
            await self._reconcile_generic_owner_locked()
            async with self._connection.execute(
                "SELECT principal_id FROM workshop_integration_route_owners WHERE source = ? AND route_name = ?",
                (source, route_name),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                raise WorkshopNotificationPreferenceStorageError("Integration route has no unambiguous principal owner")
            authority = self.authority_for_principal(PrincipalId(str(row[0])))
            effective = await self._effective_locked(authority, source)
            return effective.destination.channel_id

    def _validate_authority(self, authority: NotificationPreferenceAuthority) -> None:
        namespace = self._execution_state.maybe_for_runtime_profile_id(authority.runtime_profile_id)
        if namespace is None or namespace.principal_id != authority.principal_id:
            raise WorkshopNotificationPreferenceAccessDenied("The principal does not own notification preferences")

    @staticmethod
    def _validate_integration_class(integration_class: str) -> str:
        normalized = integration_class.strip().lower()
        if normalized not in _SUPPORTED_CLASSES:
            raise WorkshopNotificationPreferenceValidationError("Integration class must be github or generic")
        return normalized

    async def _authorized_destinations_locked(
        self,
        authority: NotificationPreferenceAuthority,
    ) -> tuple[_Destination, ...]:
        namespace = self._execution_state.maybe_for_runtime_profile_id(authority.runtime_profile_id)
        if namespace is None or namespace.principal_id != authority.principal_id:
            raise WorkshopNotificationPreferenceAccessDenied("The principal does not own notification preferences")
        async with self._connection.execute(
            "SELECT c.id, c.kind, coalesce(c.name, ''), COUNT(DISTINCT ca.agent_id) "
            "FROM channels c "
            "JOIN channel_memberships cm ON cm.channel_id = c.id AND cm.principal_id = ? "
            "LEFT JOIN channel_agents ca ON ca.channel_id = c.id "
            "WHERE c.workshop_id = (SELECT workshop_id FROM channels WHERE id = ?) "
            "AND (c.id = ? OR c.kind = 'notification') "
            "GROUP BY c.id, c.kind, c.name ORDER BY "
            "CASE c.kind WHEN 'direct' THEN 0 ELSE 1 END, lower(coalesce(c.name, '')), c.id",
            (authority.principal_id, namespace.channel_id, namespace.channel_id),
        ) as cursor:
            rows = tuple(await cursor.fetchall())
        destinations: list[_Destination] = []
        for row in rows:
            if int(row[3]) != 1:
                continue
            channel_id = ChannelId(str(row[0]))
            kind = str(row[1])
            name = str(row[2]).strip()
            display_name = name or ("Direct messages" if kind == "direct" else "Notifications")
            destinations.append(
                _Destination(
                    channel_id,
                    _choice_id(authority.principal_id, channel_id),
                    display_name,
                    kind,
                )
            )
        if not any(item.channel_id == namespace.channel_id for item in destinations):
            raise WorkshopNotificationPreferenceStorageError("Canonical direct notification fallback is unavailable")
        return tuple(destinations)

    async def _supported_classes_locked(
        self,
        authority: NotificationPreferenceAuthority,
    ) -> tuple[str, ...]:
        supported = [GITHUB_INTEGRATION_CLASS]
        async with self._connection.execute(
            "SELECT 1 FROM workshop_integration_route_owners "
            "WHERE source = 'generic' AND route_name = 'default' AND principal_id = ?",
            (authority.principal_id,),
        ) as cursor:
            if await cursor.fetchone() is not None:
                supported.append(GENERIC_INTEGRATION_CLASS)
        return tuple(supported)

    async def _effective_locked(
        self,
        authority: NotificationPreferenceAuthority,
        integration_class: str,
    ) -> _EffectivePreference:
        supported = await self._supported_classes_locked(authority)
        if integration_class not in supported:
            raise WorkshopNotificationPreferenceAccessDenied("The integration class is not editable by this principal")
        destinations = await self._authorized_destinations_locked(authority)
        by_channel = {item.channel_id: item for item in destinations}
        async with self._connection.execute(
            "SELECT channel_id FROM principal_notification_delivery_preferences "
            "WHERE principal_id = ? AND integration_class = ?",
            (authority.principal_id, integration_class),
        ) as cursor:
            row = await cursor.fetchone()
        if row is not None:
            selected = by_channel.get(ChannelId(str(row[0])))
            if selected is not None:
                return _EffectivePreference(
                    integration_class,
                    selected,
                    "personal override",
                    True,
                )

        direct_channel_id = self._execution_state.resolve_profile(authority.runtime_profile_id).channel_id
        direct = by_channel[direct_channel_id]
        if integration_class == GITHUB_INTEGRATION_CLASS:
            notification_destinations = [item for item in destinations if item.kind == "notification"]
            if len(notification_destinations) == 1:
                return _EffectivePreference(
                    integration_class,
                    notification_destinations[0],
                    "operator policy",
                    row is not None,
                )
        elif integration_class == GENERIC_INTEGRATION_CLASS:
            async with self._connection.execute(
                "SELECT protected_channel_id FROM workshop_integration_route_owners "
                "WHERE source = 'generic' AND route_name = 'default' AND principal_id = ?",
                (authority.principal_id,),
            ) as cursor:
                owner = await cursor.fetchone()
            if owner is not None:
                protected = by_channel.get(ChannelId(str(owner[0])))
                if protected is not None:
                    return _EffectivePreference(
                        integration_class,
                        protected,
                        "operator policy",
                        row is not None,
                    )
        return _EffectivePreference(
            integration_class,
            direct,
            "canonical direct fallback",
            row is not None,
        )

    async def _snapshot_locked(
        self,
        authority: NotificationPreferenceAuthority,
    ) -> NotificationPreferenceSnapshot:
        destinations = await self._authorized_destinations_locked(authority)
        supported = await self._supported_classes_locked(authority)
        effective = tuple([await self._effective_locked(authority, item) for item in supported])
        choices = tuple(
            NotificationDestinationChoice(
                item.choice_id,
                item.display_name,
                item.kind,
                supported,
            )
            for item in destinations
        )
        preferences = tuple(
            IntegrationDeliveryPreference(
                item.integration_class,
                "GitHub" if item.integration_class == GITHUB_INTEGRATION_CLASS else "Generic webhooks",
                item.destination.choice_id,
                item.destination.display_name,
                item.destination.kind,
                item.source,
                True,
                item.resettable,
            )
            for item in effective
        )
        revision_payload = {
            "principal_id": str(authority.principal_id),
            "destinations": [
                [item.choice_id, str(item.channel_id), item.display_name, item.kind] for item in destinations
            ],
            "preferences": [
                [
                    item.integration_class,
                    item.destination.choice_id,
                    item.source,
                    item.resettable,
                ]
                for item in effective
            ],
        }
        encoded = json.dumps(revision_payload, sort_keys=True, separators=(",", ":"))
        revision = "ndp_" + hashlib.sha256(encoded.encode()).hexdigest()[:32]
        return NotificationPreferenceSnapshot(choices, preferences, revision)

    async def _reconcile_generic_owner_locked(self) -> None:
        source, route_name = _GENERIC_DEFAULT_ROUTE
        async with self._connection.execute(
            "SELECT 1 FROM workshop_integration_route_owners WHERE source = ? AND route_name = ?",
            (source, route_name),
        ) as cursor:
            if await cursor.fetchone() is not None:
                return
        async with self._connection.execute(
            "SELECT channel_id FROM workshop_integration_routes WHERE source = ? AND route_name = ?",
            (source, route_name),
        ) as cursor:
            route = await cursor.fetchone()
        if route is None:
            return
        protected_channel_id = ChannelId(str(route[0]))
        async with self._connection.execute(
            "SELECT DISTINCT p.id FROM channel_memberships cm "
            "JOIN principals p ON p.id = cm.principal_id AND p.kind = 'human' "
            "JOIN channels c ON c.id = cm.channel_id "
            "JOIN workshop_memberships wm ON wm.workshop_id = c.workshop_id "
            "AND wm.principal_id = p.id AND wm.role = 'admin' "
            "WHERE cm.channel_id = ? ORDER BY p.id",
            (protected_channel_id,),
        ) as cursor:
            owners = tuple(await cursor.fetchall())
        if len(owners) != 1:
            return
        try:
            await self._connection.execute("BEGIN IMMEDIATE")
            await self._connection.execute(
                "INSERT OR IGNORE INTO workshop_integration_route_owners "
                "(source, route_name, principal_id, protected_channel_id) "
                "VALUES (?, ?, ?, ?)",
                (source, route_name, str(owners[0][0]), protected_channel_id),
            )
            await self._connection.commit()
        except Exception:
            await self._connection.rollback()
            raise
