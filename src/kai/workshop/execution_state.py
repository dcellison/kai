"""Canonical ownership and legacy backfill for mutable execution state."""

from __future__ import annotations

from dataclasses import dataclass

import aiosqlite

from kai.workshop.domain import AgentId, ChannelId, PrincipalId, RuntimeProfileId
from kai.workshop.runtime_profiles import WorkshopRuntimeProfileRegistry
from kai.workshop.store import WorkshopEventStore


class WorkshopExecutionStateError(RuntimeError):
    """Protected execution state cannot resolve one canonical owner."""


@dataclass(frozen=True, slots=True)
class WorkshopExecutionStateNamespace:
    """Canonical identities that own one protected runtime's mutable state."""

    principal_id: PrincipalId
    channel_id: ChannelId
    agent_id: AgentId
    runtime_profile_id: RuntimeProfileId
    legacy_runtime_key: int | None

    def require_legacy_runtime_key(self) -> int:
        """Return archived migration state or fail closed."""
        if self.legacy_runtime_key is None:
            raise WorkshopExecutionStateError("Canonical runtime has no legacy-state archive")
        return self.legacy_runtime_key


@dataclass(frozen=True, slots=True)
class WorkshopExecutionStateMigration:
    """Aggregate result of one idempotent legacy-state reconciliation."""

    profiles: int
    newly_migrated: int
    settings: int
    workspace_settings: int
    history: int
    grants: int


class WorkshopExecutionStateRegistry:
    """Resolve protected compatibility keys to canonical execution owners."""

    def __init__(self, namespaces: tuple[WorkshopExecutionStateNamespace, ...]) -> None:
        by_legacy_key: dict[int, WorkshopExecutionStateNamespace] = {}
        by_profile: dict[RuntimeProfileId, WorkshopExecutionStateNamespace] = {}
        by_principal: dict[PrincipalId, WorkshopExecutionStateNamespace | None] = {}
        by_principal_channel: dict[tuple[PrincipalId, ChannelId], WorkshopExecutionStateNamespace | None] = {}
        lanes: list[WorkshopExecutionStateNamespace] = []
        for namespace in namespaces:
            if not isinstance(namespace, WorkshopExecutionStateNamespace):
                raise TypeError("namespaces must contain WorkshopExecutionStateNamespace values")
            if namespace.legacy_runtime_key is not None and namespace.legacy_runtime_key in by_legacy_key:
                raise WorkshopExecutionStateError("Duplicate archived runtime execution-state key")
            lane_key = (namespace.principal_id, namespace.channel_id)
            primary = by_profile.get(namespace.runtime_profile_id)
            if primary is not None and primary.principal_id != namespace.principal_id:
                raise WorkshopExecutionStateError("Runtime profile cannot cross canonical human owners")
            if namespace.legacy_runtime_key is not None:
                by_legacy_key[namespace.legacy_runtime_key] = namespace
            by_profile.setdefault(namespace.runtime_profile_id, namespace)
            if namespace.principal_id not in by_principal:
                by_principal[namespace.principal_id] = namespace
            else:
                principal_owner = by_principal[namespace.principal_id]
                if principal_owner is not None and principal_owner.runtime_profile_id != namespace.runtime_profile_id:
                    by_principal[namespace.principal_id] = None
            if lane_key in by_principal_channel:
                by_principal_channel[lane_key] = None
            else:
                by_principal_channel[lane_key] = namespace
            lanes.append(namespace)
        if not by_profile:
            raise WorkshopExecutionStateError("At least one execution-state namespace is required")
        self._by_legacy_key = by_legacy_key
        self._by_profile = by_profile
        self._by_principal = by_principal
        self._by_principal_channel = by_principal_channel
        self._lanes = lanes

    @classmethod
    async def from_store(
        cls,
        store: WorkshopEventStore,
        runtime_profiles: WorkshopRuntimeProfileRegistry,
    ) -> WorkshopExecutionStateRegistry:
        """Resolve every protected profile through its direct-channel owner."""
        async with store.connection.execute(
            "SELECT ra.runtime_profile_id, ra.channel_id, ra.agent_id, cm.principal_id, "
            "ra.created_event_position "
            "FROM channel_agent_runtime_assignments ra "
            "JOIN channels c ON c.id = ra.channel_id AND c.kind = 'direct' "
            "JOIN channel_memberships cm ON cm.channel_id = c.id AND cm.role = 'owner' "
            "JOIN principals p ON p.id = cm.principal_id AND p.kind = 'human' "
            "ORDER BY ra.runtime_profile_id, ra.created_event_position, cm.principal_id"
        ) as cursor:
            rows = list(await cursor.fetchall())

        resolved: dict[RuntimeProfileId, list[tuple[ChannelId, AgentId, PrincipalId]]] = {}
        for row in rows:
            try:
                profile_id = RuntimeProfileId(str(row[0]))
                identities = (ChannelId(str(row[1])), AgentId(str(row[2])), PrincipalId(str(row[3])))
            except (TypeError, ValueError) as exc:
                raise WorkshopExecutionStateError(
                    "Canonical execution-state ownership contains an invalid opaque identifier"
                ) from exc
            if identities not in resolved.setdefault(profile_id, []):
                resolved[profile_id].append(identities)

        namespaces: list[WorkshopExecutionStateNamespace] = []
        for profile in runtime_profiles.profiles:
            identities = resolved.get(profile.profile_id, [])
            if not identities or len({item[2] for item in identities}) != 1:
                raise WorkshopExecutionStateError("Protected runtime profile must map to one canonical human owner")
            for index, (channel_id, agent_id, principal_id) in enumerate(identities):
                namespaces.append(
                    WorkshopExecutionStateNamespace(
                        principal_id=principal_id,
                        channel_id=channel_id,
                        agent_id=agent_id,
                        runtime_profile_id=profile.profile_id,
                        legacy_runtime_key=(
                            runtime_profiles.legacy_runtime_key(profile.profile_id) if index == 0 else None
                        ),
                    )
                )
        return cls(tuple(namespaces))

    @property
    def namespaces(self) -> tuple[WorkshopExecutionStateNamespace, ...]:
        """Return the primary legacy-compatible owner for each protected profile."""
        return tuple(sorted(self._by_profile.values(), key=lambda item: item.runtime_profile_id))

    @property
    def lanes(self) -> tuple[WorkshopExecutionStateNamespace, ...]:
        """Return every isolated canonical channel-agent execution lane."""
        return tuple(self._lanes)

    def register_lane(self, namespace: WorkshopExecutionStateNamespace) -> None:
        """Register a newly enabled lane without replacing profile-level defaults."""
        if not isinstance(namespace, WorkshopExecutionStateNamespace):
            raise TypeError("namespace must be a WorkshopExecutionStateNamespace")
        key = (namespace.principal_id, namespace.channel_id)
        existing = self._by_principal_channel.get(key)
        if existing is not None:
            if existing != namespace:
                raise WorkshopExecutionStateError("Canonical execution-state lane conflicts")
            return
        primary = self._by_profile.get(namespace.runtime_profile_id)
        if primary is None or primary.principal_id != namespace.principal_id:
            raise WorkshopExecutionStateError("Runtime profile is not owned by the canonical principal")
        if namespace.legacy_runtime_key is not None:
            raise WorkshopExecutionStateError("New execution lanes cannot claim archived runtime state")
        self._by_principal_channel[key] = namespace
        self._lanes.append(namespace)

    def replace_lane(
        self,
        prior: WorkshopExecutionStateNamespace,
        replacement: WorkshopExecutionStateNamespace,
    ) -> None:
        """Replace a lane's protected profile without changing its identity."""
        key = (prior.principal_id, prior.channel_id)
        if (
            self._by_principal_channel.get(key) != prior
            or replacement.principal_id != prior.principal_id
            or replacement.channel_id != prior.channel_id
            or replacement.agent_id != prior.agent_id
            or replacement.legacy_runtime_key is not None
        ):
            raise WorkshopExecutionStateError("Canonical execution-state lane replacement conflicts")
        primary = self._by_profile.get(replacement.runtime_profile_id)
        if primary is None or primary.principal_id != replacement.principal_id:
            raise WorkshopExecutionStateError("Replacement runtime profile is not owned by the principal")
        self._by_principal_channel[key] = replacement
        self._lanes[self._lanes.index(prior)] = replacement

    def maybe_for_legacy_runtime_key(self, legacy_runtime_key: int) -> WorkshopExecutionStateNamespace | None:
        if isinstance(legacy_runtime_key, bool) or not isinstance(legacy_runtime_key, int):
            return None
        return self._by_legacy_key.get(legacy_runtime_key)

    def maybe_for_runtime_config_id(self, runtime_config_id: int) -> WorkshopExecutionStateNamespace | None:
        """Deprecated adapter/migration alias; protected core uses canonical lookups."""
        return self.maybe_for_legacy_runtime_key(runtime_config_id)

    def maybe_for_principal_id(self, principal_id: str) -> WorkshopExecutionStateNamespace | None:
        """Resolve the principal's one protected profile across any number of lanes."""
        try:
            canonical_id = PrincipalId(principal_id)
        except (TypeError, ValueError):
            return None
        return self._by_principal.get(canonical_id)

    def maybe_for_principal_channel(
        self,
        principal_id: str | PrincipalId,
        channel_id: str | ChannelId,
    ) -> WorkshopExecutionStateNamespace | None:
        """Resolve one exact canonical direct-execution authority."""
        try:
            canonical_principal = principal_id if isinstance(principal_id, PrincipalId) else PrincipalId(principal_id)
            canonical_channel = channel_id if isinstance(channel_id, ChannelId) else ChannelId(channel_id)
        except (TypeError, ValueError):
            return None
        return self._by_principal_channel.get((canonical_principal, canonical_channel))

    def maybe_for_runtime_profile_id(
        self,
        runtime_profile_id: str | RuntimeProfileId,
    ) -> WorkshopExecutionStateNamespace | None:
        """Resolve one exact protected runtime profile."""
        try:
            canonical_id = (
                runtime_profile_id
                if isinstance(runtime_profile_id, RuntimeProfileId)
                else RuntimeProfileId(runtime_profile_id)
            )
        except (TypeError, ValueError):
            return None
        return self._by_profile.get(canonical_id)

    def resolve_profile(
        self,
        runtime_profile_id: str | RuntimeProfileId,
    ) -> WorkshopExecutionStateNamespace:
        """Resolve an exact protected runtime profile or fail closed."""
        namespace = self.maybe_for_runtime_profile_id(runtime_profile_id)
        if namespace is None:
            raise WorkshopExecutionStateError("Protected runtime profile has no canonical execution-state owner")
        return namespace


async def reconcile_legacy_execution_state(
    connection: aiosqlite.Connection,
    registry: WorkshopExecutionStateRegistry,
) -> WorkshopExecutionStateMigration:
    """Backfill each protected owner once without overwriting canonical state."""
    totals = {"settings": 0, "workspace_settings": 0, "history": 0, "grants": 0}
    newly_migrated = 0
    try:
        await connection.execute("BEGIN IMMEDIATE")
        for namespace in registry.namespaces:
            if namespace.legacy_runtime_key is None:
                continue
            async with connection.execute(
                "SELECT runtime_config_id, principal_id, channel_id, agent_id "
                "FROM workshop_execution_state_migrations WHERE runtime_profile_id = ?",
                (namespace.runtime_profile_id,),
            ) as cursor:
                migrated = await cursor.fetchone()
                if migrated is not None:
                    migrated_owner = (
                        int(migrated["runtime_config_id"]),
                        str(migrated["principal_id"]),
                        str(migrated["channel_id"]),
                        str(migrated["agent_id"]),
                    )
                    current_owner = (
                        namespace.legacy_runtime_key,
                        str(namespace.principal_id),
                        str(namespace.channel_id),
                        str(namespace.agent_id),
                    )
                    if migrated_owner != current_owner:
                        raise WorkshopExecutionStateError(
                            "Canonical execution-state migration conflicts with current protected ownership "
                            f"for runtime profile {namespace.runtime_profile_id}; restore its recorded "
                            "canonical assignment or restore the database from backup"
                        )
                    continue

            counts = await _backfill_namespace(connection, namespace)
            for key, value in counts.items():
                totals[key] += value
            await connection.execute(
                "INSERT INTO workshop_execution_state_migrations ("
                "runtime_profile_id, runtime_config_id, principal_id, channel_id, agent_id, "
                "settings_count, workspace_settings_count, history_count, grants_count"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    namespace.runtime_profile_id,
                    namespace.legacy_runtime_key,
                    namespace.principal_id,
                    namespace.channel_id,
                    namespace.agent_id,
                    counts["settings"],
                    counts["workspace_settings"],
                    counts["history"],
                    counts["grants"],
                ),
            )
            newly_migrated += 1
        await connection.commit()
    except Exception:
        await connection.rollback()
        raise
    return WorkshopExecutionStateMigration(
        profiles=len(registry.namespaces),
        newly_migrated=newly_migrated,
        settings=totals["settings"],
        workspace_settings=totals["workspace_settings"],
        history=totals["history"],
        grants=totals["grants"],
    )


async def _backfill_namespace(
    connection: aiosqlite.Connection,
    namespace: WorkshopExecutionStateNamespace,
) -> dict[str, int]:
    legacy_runtime_key = namespace.legacy_runtime_key
    if legacy_runtime_key is None:
        return {"settings": 0, "workspace_settings": 0, "history": 0, "grants": 0}
    counts = {"settings": 0, "workspace_settings": 0, "history": 0, "grants": 0}
    for field in ("model", "timeout", "workspace"):
        async with connection.execute(
            "SELECT value FROM settings WHERE key = ?",
            (f"{field}:{legacy_runtime_key}",),
        ) as cursor:
            row = await cursor.fetchone()
        if row is not None:
            cursor = await connection.execute(
                "INSERT OR IGNORE INTO channel_agent_execution_settings "
                "(channel_id, agent_id, runtime_profile_id, field, value) VALUES (?, ?, ?, ?, ?)",
                (
                    namespace.channel_id,
                    namespace.agent_id,
                    namespace.runtime_profile_id,
                    field,
                    str(row[0]),
                ),
            )
            counts["settings"] += cursor.rowcount

    prefix = f"ws_config:{legacy_runtime_key}:"
    async with connection.execute(
        "SELECT key, value FROM settings WHERE SUBSTR(key, 1, ?) = ? ORDER BY key",
        (len(prefix), prefix),
    ) as cursor:
        rows = list(await cursor.fetchall())
    for row in rows:
        remainder = str(row[0])[len(prefix) :]
        if ":" not in remainder:
            continue
        workspace_path, field = remainder.rsplit(":", 1)
        if not workspace_path or field not in {"model", "timeout", "env", "prompt"}:
            continue
        cursor = await connection.execute(
            "INSERT OR IGNORE INTO channel_agent_workspace_settings "
            "(channel_id, agent_id, runtime_profile_id, workspace_path, field, value) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                namespace.channel_id,
                namespace.agent_id,
                namespace.runtime_profile_id,
                workspace_path,
                field,
                str(row[1]),
            ),
        )
        counts["workspace_settings"] += cursor.rowcount

    cursor = await connection.execute(
        "INSERT OR IGNORE INTO principal_workspace_history (principal_id, path, last_used_at) "
        "SELECT ?, path, last_used_at FROM workspace_history WHERE chat_id = ?",
        (namespace.principal_id, legacy_runtime_key),
    )
    counts["history"] += cursor.rowcount
    cursor = await connection.execute(
        "INSERT OR IGNORE INTO principal_workspace_grants (principal_id, path) "
        "SELECT ?, path FROM allowed_workspaces WHERE chat_id = ?",
        (namespace.principal_id, legacy_runtime_key),
    )
    counts["grants"] += cursor.rowcount
    return counts
