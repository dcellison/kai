"""One-time inventory and receipt for retiring transport-shaped runtime keys."""

from __future__ import annotations

from dataclasses import dataclass

import aiosqlite

from kai.workshop.execution_state import WorkshopExecutionStateRegistry


class WorkshopRuntimeKeyCutoverError(RuntimeError):
    """A protected legacy-key cutover is incomplete or conflicts with authority."""


@dataclass(frozen=True, slots=True)
class WorkshopRuntimeKeyCutover:
    profiles: int
    newly_recorded: int
    archived_keys: int


async def reconcile_workshop_runtime_key_cutover(
    connection: aiosqlite.Connection,
    registry: WorkshopExecutionStateRegistry,
    *,
    memory_enabled: bool,
) -> WorkshopRuntimeKeyCutover:
    """Record immutable legacy inventories after every canonical migration."""
    newly_recorded = 0
    archived_keys = 0
    try:
        await connection.execute("BEGIN IMMEDIATE")
        for namespace in registry.namespaces:
            legacy_key = namespace.legacy_runtime_key
            if legacy_key is not None:
                archived_keys += 1
            async with connection.execute(
                "SELECT legacy_runtime_key, principal_id, channel_id, agent_id "
                "FROM workshop_runtime_key_cutovers WHERE runtime_profile_id = ?",
                (namespace.runtime_profile_id,),
            ) as cursor:
                existing = await cursor.fetchone()
            identity = (
                legacy_key,
                str(namespace.principal_id),
                str(namespace.channel_id),
                str(namespace.agent_id),
            )
            if existing is not None:
                recorded = (
                    int(existing[0]) if existing[0] is not None else None,
                    str(existing[1]),
                    str(existing[2]),
                    str(existing[3]),
                )
                if recorded != identity:
                    raise WorkshopRuntimeKeyCutoverError(
                        "Runtime-key cutover receipt conflicts with current canonical ownership"
                    )
                continue

            counts = await _inventory(connection, str(namespace.runtime_profile_id), legacy_key)
            if memory_enabled and legacy_key is not None and counts[7] is None:
                raise WorkshopRuntimeKeyCutoverError(
                    "Protected semantic-memory migration must complete before runtime-key cutover"
                )
            await connection.execute(
                "INSERT INTO workshop_runtime_key_cutovers ("
                "runtime_profile_id, legacy_runtime_key, principal_id, channel_id, agent_id, "
                "settings_rows, workspace_rows, session_rows, lock_rows, history_rows, "
                "grant_rows, github_rows, memory_rows"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    namespace.runtime_profile_id,
                    legacy_key,
                    namespace.principal_id,
                    namespace.channel_id,
                    namespace.agent_id,
                    counts[0],
                    counts[1],
                    counts[2],
                    0,
                    counts[3],
                    counts[4],
                    counts[5],
                    counts[7] or 0,
                ),
            )
            newly_recorded += 1
        await connection.commit()
    except Exception:
        await connection.rollback()
        raise
    return WorkshopRuntimeKeyCutover(
        profiles=len(registry.namespaces),
        newly_recorded=newly_recorded,
        archived_keys=archived_keys,
    )


async def _inventory(
    connection: aiosqlite.Connection,
    runtime_profile_id: str,
    legacy_key: int | None,
) -> tuple[int, int, int, int, int, int, int, int | None]:
    if legacy_key is None:
        async with connection.execute(
            "SELECT COUNT(*) FROM principal_github_subscriptions subscription "
            "JOIN channel_agent_runtime_assignments assignment "
            "ON assignment.runtime_profile_id = ? "
            "JOIN channel_memberships membership ON membership.channel_id = assignment.channel_id "
            "AND membership.role = 'owner' AND membership.principal_id = subscription.principal_id",
            (runtime_profile_id,),
        ) as cursor:
            github_row = await cursor.fetchone()
        assert github_row is not None
        github = int(github_row[0])
        return (0, 0, 0, 0, 0, github, 0, 0)

    async with connection.execute(
        "SELECT settings_count, workspace_settings_count, history_count, grants_count "
        "FROM workshop_execution_state_migrations WHERE runtime_profile_id = ?",
        (runtime_profile_id,),
    ) as cursor:
        execution = await cursor.fetchone()
    async with connection.execute(
        "SELECT github_subscription_count FROM workshop_operational_state_migrations WHERE runtime_profile_id = ?",
        (runtime_profile_id,),
    ) as cursor:
        github = await cursor.fetchone()
    async with connection.execute(
        "SELECT total_count FROM workshop_memory_authority_migrations WHERE runtime_profile_id = ?",
        (runtime_profile_id,),
    ) as cursor:
        memory = await cursor.fetchone()
    if execution is None or github is None:
        raise WorkshopRuntimeKeyCutoverError(
            "Canonical execution and operational migrations must complete before runtime-key cutover"
        )
    async with connection.execute("SELECT COUNT(*) FROM sessions WHERE chat_id = ?", (legacy_key,)) as cursor:
        session_row = await cursor.fetchone()
    assert session_row is not None
    session_rows = int(session_row[0])
    return (
        int(execution[0]),
        int(execution[1]),
        session_rows,
        int(execution[2]),
        int(execution[3]),
        int(github[0]),
        0,
        int(memory[0]) if memory is not None else None,
    )
