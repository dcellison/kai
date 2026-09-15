"""Canonical ownership and migration for the memory-project registry."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

import aiosqlite

from kai.config import MemoryProjectConfig
from kai.workshop.execution_state import WorkshopExecutionStateRegistry

_PROJECT_ID_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")
_VALID_SCOPES = frozenset({"global", "project"})


class WorkshopMemoryProjectRegistryError(RuntimeError):
    """Canonical memory-project ownership is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class WorkshopMemoryProjectRegistryMigration:
    """Aggregate result of one replay-safe legacy registry reconciliation."""

    legacy_projects: int
    newly_migrated: int
    migrated: int
    missing_owners: int
    invalid: int
    conflicting: int
    pinned_projects: int
    principal_projects: int


def _source_digest(row: aiosqlite.Row) -> str:
    payload = {
        "project_id": row["project_id"],
        "display_name": row["display_name"],
        "workspace_root": row["workspace_root"],
        "memory_enabled": row["memory_enabled"],
        "default_scope_for_new_facts": row["default_scope_for_new_facts"],
        "created_by": row["created_by"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _configuration_digest(projects: dict[str, MemoryProjectConfig]) -> str:
    payload = [
        {
            "project_id": project.project_id,
            "display_name": project.display_name,
            "roots": [str(root) for root in project.workspace_roots],
            "memory_enabled": project.memory_enabled,
            "default_scope": project.default_scope_for_new_facts,
        }
        for project in sorted(projects.values(), key=lambda item: item.project_id)
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _validate_legacy_row(row: aiosqlite.Row) -> tuple[str, str, str, bool, str | None] | None:
    project_id = row["project_id"]
    display_name = row["display_name"]
    raw_root = row["workspace_root"]
    memory_enabled = row["memory_enabled"]
    default_scope = row["default_scope_for_new_facts"]
    if (
        not isinstance(project_id, str)
        or _PROJECT_ID_RE.fullmatch(project_id) is None
        or not isinstance(display_name, str)
        or not display_name.strip()
        or len(display_name.strip()) > 128
        or not isinstance(raw_root, str)
        or not raw_root.strip()
        or raw_root != raw_root.strip()
        or not Path(raw_root).is_absolute()
        or memory_enabled not in (0, 1, False, True)
        or (default_scope is not None and default_scope not in _VALID_SCOPES)
    ):
        return None
    try:
        root = str(Path(raw_root).resolve(strict=False))
    except OSError:
        return None
    return project_id, display_name.strip(), root, bool(memory_enabled), default_scope


def _conflicts_with_pinned(
    project_id: str,
    root: Path,
    pinned_projects: dict[str, MemoryProjectConfig],
) -> bool:
    if project_id in pinned_projects:
        return True
    pinned_roots = (pinned_root for project in pinned_projects.values() for pinned_root in project.workspace_roots)
    return any(
        root == pinned_root or root.is_relative_to(pinned_root) or pinned_root.is_relative_to(root)
        for pinned_root in pinned_roots
    )


async def reconcile_memory_project_registry(
    connection: aiosqlite.Connection,
    execution_state: WorkshopExecutionStateRegistry,
    pinned_projects: dict[str, MemoryProjectConfig],
) -> WorkshopMemoryProjectRegistryMigration:
    """Migrate legacy chat-keyed rows and record current pinned authority."""
    newly_migrated = 0
    outcomes = {"migrated": 0, "missing_owner": 0, "invalid": 0, "conflicting": 0}
    try:
        await connection.execute("BEGIN IMMEDIATE")
        async with connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'memory_projects'"
        ) as cursor:
            has_legacy_registry = await cursor.fetchone() is not None
        if has_legacy_registry:
            async with connection.execute(
                "SELECT project_id, display_name, workspace_root, memory_enabled, "
                "default_scope_for_new_facts, created_by FROM memory_projects ORDER BY project_id"
            ) as cursor:
                legacy_rows = list(await cursor.fetchall())
        else:
            legacy_rows = []

        for row in legacy_rows:
            source_digest = _source_digest(row)
            async with connection.execute(
                "SELECT outcome, source_digest FROM workshop_memory_project_migrations WHERE legacy_project_id = ?",
                (str(row["project_id"]),),
            ) as cursor:
                prior_receipt = await cursor.fetchone()
            if prior_receipt is not None and str(prior_receipt["source_digest"]) == source_digest:
                outcomes[str(prior_receipt["outcome"])] += 1
                continue

            validated = _validate_legacy_row(row)
            try:
                legacy_owner_key = int(row["created_by"])
            except (TypeError, ValueError):
                legacy_owner_key = 0
            owner = execution_state.maybe_for_legacy_runtime_key(legacy_owner_key)
            outcome = "migrated"
            principal_id: str | None = None
            runtime_profile_id: str | None = None
            if prior_receipt is not None:
                # Legacy rows are retained as a non-authoritative archive after
                # cutover. A changed archived row must never mutate canonical
                # state or recreate a project intentionally unregistered later.
                outcome = "conflicting"
            elif validated is None:
                outcome = "invalid"
            elif owner is None:
                outcome = "missing_owner"
            else:
                project_id, display_name, workspace_root, memory_enabled, default_scope = validated
                principal_id = str(owner.principal_id)
                runtime_profile_id = str(owner.runtime_profile_id)
                root = Path(workspace_root)
                if _conflicts_with_pinned(project_id, root, pinned_projects):
                    outcome = "conflicting"
                else:
                    async with connection.execute(
                        "SELECT project_id, workspace_root, principal_id, runtime_profile_id "
                        "FROM principal_memory_projects WHERE project_id = ? OR workspace_root = ?",
                        (project_id, workspace_root),
                    ) as cursor:
                        existing = await cursor.fetchone()
                    if existing is not None:
                        exact = (
                            str(existing["project_id"]) == project_id
                            and str(existing["workspace_root"]) == workspace_root
                            and str(existing["principal_id"]) == principal_id
                            and str(existing["runtime_profile_id"]) == runtime_profile_id
                        )
                        if not exact:
                            outcome = "conflicting"
                    if outcome == "migrated":
                        cursor = await connection.execute(
                            "INSERT OR IGNORE INTO principal_memory_projects "
                            "(project_id, display_name, workspace_root, principal_id, runtime_profile_id, "
                            "memory_enabled, default_scope_for_new_facts, provenance) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, 'legacy_migrated')",
                            (
                                project_id,
                                display_name,
                                workspace_root,
                                principal_id,
                                runtime_profile_id,
                                1 if memory_enabled else 0,
                                default_scope,
                            ),
                        )
                        newly_migrated += int(cursor.rowcount > 0)

            outcomes[outcome] += 1
            await connection.execute(
                "INSERT INTO workshop_memory_project_migrations "
                "(legacy_project_id, legacy_created_by, principal_id, runtime_profile_id, outcome, "
                "source_digest, reconciled_at) VALUES (?, ?, ?, ?, ?, ?, "
                "strftime('%Y-%m-%dT%H:%M:%fZ', 'now')) "
                "ON CONFLICT(legacy_project_id) DO UPDATE SET "
                "legacy_created_by = excluded.legacy_created_by, principal_id = excluded.principal_id, "
                "runtime_profile_id = excluded.runtime_profile_id, outcome = excluded.outcome, "
                "source_digest = excluded.source_digest, reconciled_at = excluded.reconciled_at",
                (
                    str(row["project_id"]),
                    legacy_owner_key,
                    principal_id,
                    runtime_profile_id,
                    outcome,
                    source_digest,
                ),
            )

        await connection.execute(
            "INSERT INTO workshop_memory_project_registry_state "
            "(singleton, pinned_projects, pinned_roots, configuration_digest, reconciled_at) "
            "VALUES (1, ?, ?, ?, strftime('%Y-%m-%dT%H:%M:%fZ', 'now')) "
            "ON CONFLICT(singleton) DO UPDATE SET pinned_projects = excluded.pinned_projects, "
            "pinned_roots = excluded.pinned_roots, configuration_digest = excluded.configuration_digest, "
            "reconciled_at = excluded.reconciled_at",
            (
                len(pinned_projects),
                sum(len(project.workspace_roots) for project in pinned_projects.values()),
                _configuration_digest(pinned_projects),
            ),
        )
        await connection.commit()
    except Exception:
        await connection.rollback()
        raise

    async with connection.execute("SELECT COUNT(*) AS count FROM principal_memory_projects") as cursor:
        principal_row = await cursor.fetchone()
    return WorkshopMemoryProjectRegistryMigration(
        legacy_projects=len(legacy_rows),
        newly_migrated=newly_migrated,
        migrated=outcomes["migrated"],
        missing_owners=outcomes["missing_owner"],
        invalid=outcomes["invalid"],
        conflicting=outcomes["conflicting"],
        pinned_projects=len(pinned_projects),
        principal_projects=int(principal_row["count"]) if principal_row is not None else 0,
    )
