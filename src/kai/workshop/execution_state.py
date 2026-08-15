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
    runtime_config_id: int


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
        by_config_id: dict[int, WorkshopExecutionStateNamespace] = {}
        by_profile: dict[RuntimeProfileId, WorkshopExecutionStateNamespace] = {}
        for namespace in namespaces:
            if not isinstance(namespace, WorkshopExecutionStateNamespace):
                raise TypeError("namespaces must contain WorkshopExecutionStateNamespace values")
            if namespace.runtime_config_id in by_config_id:
                raise WorkshopExecutionStateError("Duplicate runtime configuration execution-state owner")
            if namespace.runtime_profile_id in by_profile:
                raise WorkshopExecutionStateError("Duplicate runtime profile execution-state owner")
            by_config_id[namespace.runtime_config_id] = namespace
            by_profile[namespace.runtime_profile_id] = namespace
        if not by_config_id:
            raise WorkshopExecutionStateError("At least one execution-state namespace is required")
        self._by_config_id = by_config_id
        self._by_profile = by_profile

    @classmethod
    async def from_store(
        cls,
        store: WorkshopEventStore,
        runtime_profiles: WorkshopRuntimeProfileRegistry,
    ) -> WorkshopExecutionStateRegistry:
        """Resolve every protected profile through its direct-channel owner."""
        async with store.connection.execute(
            "SELECT ra.runtime_profile_id, ra.channel_id, ra.agent_id, cm.principal_id "
            "FROM channel_agent_runtime_assignments ra "
            "JOIN channels c ON c.id = ra.channel_id AND c.kind = 'direct' "
            "JOIN channel_memberships cm ON cm.channel_id = c.id AND cm.role = 'owner' "
            "JOIN principals p ON p.id = cm.principal_id AND p.kind = 'human' "
            "ORDER BY ra.runtime_profile_id, cm.principal_id"
        ) as cursor:
            rows = list(await cursor.fetchall())

        resolved: dict[RuntimeProfileId, set[tuple[ChannelId, AgentId, PrincipalId]]] = {}
        for row in rows:
            try:
                profile_id = RuntimeProfileId(str(row[0]))
                identities = (ChannelId(str(row[1])), AgentId(str(row[2])), PrincipalId(str(row[3])))
            except (TypeError, ValueError) as exc:
                raise WorkshopExecutionStateError(
                    "Canonical execution-state ownership contains an invalid opaque identifier"
                ) from exc
            resolved.setdefault(profile_id, set()).add(identities)

        namespaces: list[WorkshopExecutionStateNamespace] = []
        for profile in runtime_profiles.profiles:
            identities = resolved.get(profile.profile_id, set())
            if len(identities) != 1:
                raise WorkshopExecutionStateError(
                    "Protected runtime profile must map to exactly one canonical execution-state owner"
                )
            channel_id, agent_id, principal_id = next(iter(identities))
            namespaces.append(
                WorkshopExecutionStateNamespace(
                    principal_id=principal_id,
                    channel_id=channel_id,
                    agent_id=agent_id,
                    runtime_profile_id=profile.profile_id,
                    runtime_config_id=profile.runtime_config_id,
                )
            )
        return cls(tuple(namespaces))

    @property
    def namespaces(self) -> tuple[WorkshopExecutionStateNamespace, ...]:
        return tuple(sorted(self._by_profile.values(), key=lambda item: item.runtime_profile_id))

    def maybe_for_runtime_config_id(self, runtime_config_id: int) -> WorkshopExecutionStateNamespace | None:
        if isinstance(runtime_config_id, bool) or not isinstance(runtime_config_id, int):
            return None
        return self._by_config_id.get(runtime_config_id)


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
                        namespace.runtime_config_id,
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
                    namespace.runtime_config_id,
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
    counts = {"settings": 0, "workspace_settings": 0, "history": 0, "grants": 0}
    for field in ("model", "timeout", "workspace"):
        async with connection.execute(
            "SELECT value FROM settings WHERE key = ?",
            (f"{field}:{namespace.runtime_config_id}",),
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

    prefix = f"ws_config:{namespace.runtime_config_id}:"
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
        (namespace.principal_id, namespace.runtime_config_id),
    )
    counts["history"] += cursor.rowcount
    cursor = await connection.execute(
        "INSERT OR IGNORE INTO principal_workspace_grants (principal_id, path) "
        "SELECT ?, path FROM allowed_workspaces WHERE chat_id = ?",
        (namespace.principal_id, namespace.runtime_config_id),
    )
    counts["grants"] += cursor.rowcount
    return counts
