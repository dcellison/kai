"""Canonical principal/runtime memory-project registry tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from kai import sessions
from kai.config import MemoryProjectConfig
from kai.workshop.bootstrap import BootstrapHuman
from kai.workshop.diagnostics import workshop_memory_project_registry_status
from tests.workshop_profiles import profile_id, profile_registry


@pytest.fixture
async def database(tmp_path: Path):
    path = tmp_path / "kai.db"
    yield path
    await sessions.close_db()


async def _registry(database: Path, runtime_key: int = 101):
    await sessions.init_db(database)
    await sessions.bootstrap_workshop_foundation(
        (
            BootstrapHuman(
                "Human 101",
                "admin",
                "telegram",
                str(runtime_key),
                str(runtime_key),
                profile_id(runtime_key),
            ),
        )
    )
    registry, _migration = await sessions.initialize_workshop_execution_state(profile_registry(runtime_key))
    return registry


def _pinned(project_id: str, root: Path) -> MemoryProjectConfig:
    return MemoryProjectConfig(
        project_id=project_id,
        display_name=project_id.capitalize(),
        workspace_roots=(root,),
        memory_enabled=True,
        default_scope_for_new_facts="project",
    )


async def test_legacy_project_migrates_once_and_cannot_resurrect_after_removal(
    database: Path,
    tmp_path: Path,
) -> None:
    root = tmp_path / "research"
    root.mkdir()
    await sessions.init_db(database)
    await sessions.register_memory_project(
        project_id="research",
        display_name="Research",
        workspace_root=str(root),
        created_by=101,
    )
    registry = await _registry(database)

    first = await sessions.initialize_workshop_memory_project_registry(registry, {})
    second = await sessions.initialize_workshop_memory_project_registry(registry, {})
    rows = await sessions.get_canonical_memory_project_rows()

    assert first.newly_migrated == 1
    assert second.newly_migrated == 0
    assert [(row["project_id"], row["provenance"]) for row in rows] == [("research", "legacy_migrated")]

    namespace = registry.namespaces[0]
    removed = await sessions.unregister_canonical_memory_project(
        namespace,
        project_id="research",
        expected_state_version=0,
        workspace_digest="0" * 64,
    )
    replay = await sessions.initialize_workshop_memory_project_registry(registry, {})

    assert removed is True
    assert replay.newly_migrated == 0
    assert await sessions.get_canonical_memory_project_rows() == []
    assert workshop_memory_project_registry_status(database).startswith(
        "Workshop memory-project registry: active; pinned=0, principal=0"
    )


async def test_pinned_conflicts_and_missing_owners_are_diagnostic_not_authoritative(
    database: Path,
    tmp_path: Path,
) -> None:
    pinned_root = tmp_path / "kai"
    pinned_root.mkdir()
    nested = pinned_root / "nested"
    nested.mkdir()
    missing_owner_root = tmp_path / "orphan"
    missing_owner_root.mkdir()
    await sessions.init_db(database)
    await sessions.register_memory_project(
        project_id="nested",
        display_name="Nested",
        workspace_root=str(nested),
        created_by=101,
    )
    await sessions.register_memory_project(
        project_id="orphan",
        display_name="Orphan",
        workspace_root=str(missing_owner_root),
        created_by=999,
    )
    registry = await _registry(database)

    migration = await sessions.initialize_workshop_memory_project_registry(
        registry,
        {"kai": _pinned("kai", pinned_root)},
    )

    assert migration.conflicting == 1
    assert migration.missing_owners == 1
    assert migration.principal_projects == 0
    status = workshop_memory_project_registry_status(database)
    assert status.startswith("Workshop memory-project registry: INCOMPLETE;")
    assert "conflicting=1" in status
    assert "missing owners=1" in status


async def test_changed_legacy_archive_never_overwrites_canonical_state(
    database: Path,
    tmp_path: Path,
) -> None:
    original = tmp_path / "original"
    changed = tmp_path / "changed"
    original.mkdir()
    changed.mkdir()
    await sessions.init_db(database)
    await sessions.register_memory_project(
        project_id="research",
        display_name="Research",
        workspace_root=str(original),
        created_by=101,
    )
    registry = await _registry(database)
    await sessions.initialize_workshop_memory_project_registry(registry, {})
    await sessions._get_db().execute(
        "UPDATE memory_projects SET workspace_root = ? WHERE project_id = 'research'",
        (str(changed),),
    )
    await sessions._get_db().commit()

    migration = await sessions.initialize_workshop_memory_project_registry(registry, {})
    rows = await sessions.get_canonical_memory_project_rows()

    assert migration.conflicting == 1
    assert rows[0]["workspace_root"] == str(original)
    assert "conflicting=1" in workshop_memory_project_registry_status(database)
