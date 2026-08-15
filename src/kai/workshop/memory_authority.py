"""Canonical ownership and one-way migration for semantic memory."""

from __future__ import annotations

import asyncio
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import aiosqlite

from kai import memory
from kai.workshop.domain import AgentId, ChannelId, PrincipalId, RuntimeProfileId
from kai.workshop.execution_state import (
    WorkshopExecutionStateNamespace,
    WorkshopExecutionStateRegistry,
)


class WorkshopMemoryAuthorityError(RuntimeError):
    """Protected semantic memory cannot resolve one canonical owner."""


@dataclass(frozen=True, slots=True)
class WorkshopMemoryAuthorityMigration:
    """Aggregate result of an idempotent protected-memory reconciliation."""

    profiles: int
    newly_migrated: int
    moved: int
    stamped: int
    total: int


def memory_authority_registry_from_database(
    database_path: Path,
) -> WorkshopExecutionStateRegistry | None:
    """Load canonical memory owners for an offline administrative process."""
    if database_path.is_symlink():
        raise WorkshopMemoryAuthorityError("Refusing symlinked canonical memory database")
    if not database_path.is_file():
        return None
    connection = sqlite3.connect(f"{database_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        tables = {
            str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
        if "workshop_memory_authority_migrations" not in tables:
            return None
        rows = connection.execute(
            "SELECT runtime_profile_id, runtime_config_id, principal_id, channel_id, agent_id "
            "FROM workshop_memory_authority_migrations ORDER BY runtime_profile_id"
        ).fetchall()
        protected_profiles = (
            int(connection.execute("SELECT COUNT(*) FROM workshop_execution_state_migrations").fetchone()[0])
            if "workshop_execution_state_migrations" in tables
            else 0
        )
    finally:
        connection.close()
    if len(rows) != protected_profiles:
        raise WorkshopMemoryAuthorityError(
            "Canonical semantic-memory receipts do not cover every protected runtime profile; "
            "start Kai once with memory enabled so the protected migration can finish before "
            "running offline memory tools"
        )
    if not rows:
        return None
    try:
        namespaces = tuple(
            WorkshopExecutionStateNamespace(
                principal_id=PrincipalId(str(row[2])),
                channel_id=ChannelId(str(row[3])),
                agent_id=AgentId(str(row[4])),
                runtime_profile_id=RuntimeProfileId(str(row[0])),
                runtime_config_id=int(row[1]),
            )
            for row in rows
        )
    except (TypeError, ValueError) as exc:
        raise WorkshopMemoryAuthorityError("Canonical semantic-memory receipts contain invalid ownership") from exc
    return WorkshopExecutionStateRegistry(namespaces)


async def reconcile_workshop_memory_authority(
    connection: aiosqlite.Connection,
    registry: WorkshopExecutionStateRegistry,
) -> WorkshopMemoryAuthorityMigration:
    """Move each protected Mem0 namespace to its canonical principal once.

    Qdrant and SQLite cannot share one transaction. Each vector move is
    idempotent and preserves its stable memory ID; the durable receipt is
    committed only after the old namespace is empty and the canonical owner
    metadata has been verified. A crash before the receipt therefore resumes
    safely from the union of the partially moved namespaces.
    """
    moved = stamped = total = newly_migrated = 0
    for namespace in registry.namespaces:
        async with connection.execute(
            "SELECT runtime_config_id, principal_id, channel_id, agent_id "
            "FROM workshop_memory_authority_migrations WHERE runtime_profile_id = ?",
            (namespace.runtime_profile_id,),
        ) as cursor:
            migrated = await cursor.fetchone()
        current_owner = (
            namespace.runtime_config_id,
            str(namespace.principal_id),
            str(namespace.channel_id),
            str(namespace.agent_id),
        )
        if migrated is not None:
            migrated_owner = (
                int(migrated["runtime_config_id"]),
                str(migrated["principal_id"]),
                str(migrated["channel_id"]),
                str(migrated["agent_id"]),
            )
            if migrated_owner != current_owner:
                raise WorkshopMemoryAuthorityError(
                    "Canonical semantic-memory migration conflicts with current protected ownership "
                    f"for runtime profile {namespace.runtime_profile_id}; restore its recorded "
                    "canonical assignment or restore the memory and database backups"
                )
            continue

        try:
            sibling_namespaces = tuple(
                sibling
                for sibling in registry.namespaces
                if sibling.runtime_profile_id != namespace.runtime_profile_id
                and sibling.principal_id == namespace.principal_id
            )
            result = await asyncio.to_thread(
                memory.migrate_memory_namespace,
                namespace,
                sibling_namespaces=sibling_namespaces,
            )
        except memory.CanonicalMemoryAuthorityError as exc:
            raise WorkshopMemoryAuthorityError(str(exc)) from exc

        try:
            await connection.execute("BEGIN IMMEDIATE")
            await connection.execute(
                "INSERT INTO workshop_memory_authority_migrations ("
                "runtime_profile_id, runtime_config_id, principal_id, channel_id, agent_id, "
                "moved_count, stamped_count, total_count"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    namespace.runtime_profile_id,
                    namespace.runtime_config_id,
                    namespace.principal_id,
                    namespace.channel_id,
                    namespace.agent_id,
                    result.moved,
                    result.stamped,
                    result.total,
                ),
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise
        newly_migrated += 1
        moved += result.moved
        stamped += result.stamped
        total += result.total

    return WorkshopMemoryAuthorityMigration(
        profiles=len(registry.namespaces),
        newly_migrated=newly_migrated,
        moved=moved,
        stamped=stamped,
        total=total,
    )
