"""Canonical ownership and migration for mutable workspace grants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import aiosqlite

from kai.workshop.execution_state import WorkshopExecutionStateRegistry
from kai.workshop.runtime_profiles import WorkshopRuntimeProfileRegistry


class WorkshopWorkspaceGrantError(RuntimeError):
    """Workspace grant authority is incomplete or internally inconsistent."""


@dataclass(frozen=True, slots=True)
class WorkshopWorkspaceGrantMigration:
    """Aggregate result of one replay-safe legacy grant reconciliation."""

    profiles: int
    newly_migrated: int
    legacy_rows: int
    migrated_rows: int
    invalid_rows: int


def _canonical_legacy_path(raw_path: object) -> str | None:
    if not isinstance(raw_path, str) or not raw_path.strip() or raw_path != raw_path.strip():
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        return None
    try:
        return str(path.resolve(strict=False))
    except OSError:
        return None


async def reconcile_workspace_grant_authority(
    connection: aiosqlite.Connection,
    execution_state: WorkshopExecutionStateRegistry,
    runtime_profiles: WorkshopRuntimeProfileRegistry,
) -> WorkshopWorkspaceGrantMigration:
    """Move chat-keyed and pre-provenance grants into canonical ownership."""
    totals = {"legacy_rows": 0, "migrated_rows": 0, "invalid_rows": 0}
    newly_migrated = 0
    try:
        await connection.execute("BEGIN IMMEDIATE")
        for namespace in execution_state.namespaces:
            legacy_runtime_key = namespace.legacy_runtime_key
            if legacy_runtime_key is None:
                continue
            async with connection.execute(
                "SELECT legacy_runtime_key, principal_id FROM workshop_workspace_grant_migrations "
                "WHERE runtime_profile_id = ?",
                (namespace.runtime_profile_id,),
            ) as cursor:
                receipt = await cursor.fetchone()
            if receipt is not None:
                if (int(receipt["legacy_runtime_key"]), str(receipt["principal_id"])) != (
                    legacy_runtime_key,
                    str(namespace.principal_id),
                ):
                    raise WorkshopWorkspaceGrantError(
                        "Workspace grant migration conflicts with canonical runtime ownership"
                    )
                continue

            profile = runtime_profiles.resolve(namespace.runtime_profile_id)
            workspace_base = profile.workspace_base.resolve() if profile.workspace_base is not None else None
            async with connection.execute(
                "SELECT path FROM principal_workspace_grants "
                "WHERE principal_id = ? AND runtime_profile_id IS NULL ORDER BY created_at, rowid",
                (namespace.principal_id,),
            ) as cursor:
                prior_canonical_rows = [str(row["path"]) for row in await cursor.fetchall()]
            async with connection.execute(
                "SELECT path FROM allowed_workspaces WHERE chat_id = ? ORDER BY rowid",
                (legacy_runtime_key,),
            ) as cursor:
                adapter_rows = [str(row["path"]) for row in await cursor.fetchall()]

            raw_rows = tuple(dict.fromkeys((*prior_canonical_rows, *adapter_rows)))
            migrated_paths: set[str] = set()
            invalid_rows = 0
            for raw_path in raw_rows:
                canonical_path = _canonical_legacy_path(raw_path)
                if canonical_path is None:
                    invalid_rows += 1
                    await connection.execute(
                        "DELETE FROM principal_workspace_grants WHERE principal_id = ? AND path = ? "
                        "AND runtime_profile_id IS NULL",
                        (namespace.principal_id, raw_path),
                    )
                    continue
                candidate = Path(canonical_path)
                provenance = (
                    "principal_created"
                    if workspace_base is not None and candidate.parent == workspace_base
                    else "legacy_migrated"
                )
                await connection.execute(
                    "INSERT OR IGNORE INTO principal_workspace_grants "
                    "(principal_id, path, runtime_profile_id, provenance) VALUES (?, ?, ?, ?)",
                    (
                        namespace.principal_id,
                        canonical_path,
                        namespace.runtime_profile_id,
                        provenance,
                    ),
                )
                await connection.execute(
                    "DELETE FROM principal_workspace_grants WHERE principal_id = ? AND path = ? "
                    "AND runtime_profile_id IS NULL",
                    (namespace.principal_id, raw_path),
                )
                if raw_path != canonical_path and raw_path in adapter_rows:
                    await connection.execute(
                        "INSERT OR IGNORE INTO allowed_workspaces (chat_id, path) VALUES (?, ?)",
                        (legacy_runtime_key, canonical_path),
                    )
                    await connection.execute(
                        "DELETE FROM allowed_workspaces WHERE chat_id = ? AND path = ?",
                        (legacy_runtime_key, raw_path),
                    )
                migrated_paths.add(canonical_path)

            await connection.execute(
                "INSERT INTO workshop_workspace_grant_migrations "
                "(runtime_profile_id, legacy_runtime_key, principal_id, legacy_rows, migrated_rows, invalid_rows) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    namespace.runtime_profile_id,
                    legacy_runtime_key,
                    namespace.principal_id,
                    len(raw_rows),
                    len(migrated_paths),
                    invalid_rows,
                ),
            )
            totals["legacy_rows"] += len(raw_rows)
            totals["migrated_rows"] += len(migrated_paths)
            totals["invalid_rows"] += invalid_rows
            newly_migrated += 1

        async with connection.execute(
            "SELECT COUNT(*) AS count FROM principal_workspace_grants g "
            "WHERE g.runtime_profile_id IS NULL OR g.provenance IS NULL "
            "OR NOT EXISTS (SELECT 1 FROM runtime_profile_owners o "
            "WHERE o.runtime_profile_id = g.runtime_profile_id AND o.principal_id = g.principal_id)"
        ) as cursor:
            row = await cursor.fetchone()
        if row is not None and int(row["count"]):
            raise WorkshopWorkspaceGrantError("Canonical workspace grants contain unresolved ownership")
        await connection.commit()
    except Exception:
        await connection.rollback()
        raise
    return WorkshopWorkspaceGrantMigration(
        profiles=len(execution_state.namespaces),
        newly_migrated=newly_migrated,
        legacy_rows=totals["legacy_rows"],
        migrated_rows=totals["migrated_rows"],
        invalid_rows=totals["invalid_rows"],
    )
