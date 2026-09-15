"""Canonical principal/runtime workspace grant authority tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from kai import sessions
from kai.workshop.bootstrap import BootstrapHuman
from kai.workshop.diagnostics import workshop_workspace_grant_status
from tests.workshop_profiles import profile_id, profile_registry


@pytest.fixture
async def database(tmp_path: Path):
    path = tmp_path / "kai.db"
    yield path
    await sessions.close_db()


async def _bootstrap(database: Path, *runtime_ids: int):
    await sessions.init_db(database)
    await sessions.bootstrap_workshop_foundation(
        tuple(
            BootstrapHuman(
                display_name=f"Human {runtime_id}",
                role="admin" if index == 0 else "member",
                transport="telegram",
                external_subject=str(runtime_id),
                external_channel_id=str(runtime_id),
                runtime_profile_id=profile_id(runtime_id),
            )
            for index, runtime_id in enumerate(runtime_ids)
        )
    )
    profiles = profile_registry(*runtime_ids)
    registry, _migration = await sessions.initialize_workshop_execution_state(profiles)
    return profiles, registry


async def test_legacy_adapter_grants_migrate_once_with_canonical_ownership(database: Path) -> None:
    await sessions.init_db(database)
    await sessions.add_allowed_workspace(101, "/projects/legacy")
    await sessions.bootstrap_workshop_foundation(
        (
            BootstrapHuman(
                "Human 101",
                "admin",
                "telegram",
                "101",
                "101",
                profile_id(101),
            ),
        )
    )
    profiles = profile_registry(101)
    registry, _execution_migration = await sessions.initialize_workshop_execution_state(profiles)

    first = await sessions.initialize_workshop_workspace_grant_authority(registry, profiles)
    second = await sessions.initialize_workshop_workspace_grant_authority(registry, profiles)
    rows = await sessions.get_canonical_workspace_grant_rows(registry.namespaces[0])

    assert first.profiles == 1
    assert first.newly_migrated == 1
    assert first.migrated_rows == 1
    assert first.invalid_rows == 0
    assert second.newly_migrated == 0
    assert [(row["path"], row["provenance"]) for row in rows] == [("/projects/legacy", "legacy_migrated")]
    assert workshop_workspace_grant_status(database).startswith(
        "Workshop workspace grants: active; profiles=1, migrated=1, missing=0"
    )


async def test_version_seventy_nine_database_preserves_legacy_grants(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from kai.workshop import schema

    database = tmp_path / "pre-grant-authority.db"
    with monkeypatch.context() as migration_context:
        migration_context.setattr(schema, "WORKSHOP_SCHEMA_VERSION", 79)
        migration_context.setattr(schema, "_MIGRATIONS", schema._MIGRATIONS[:79])
        await sessions.init_db(database)
        await sessions.add_allowed_workspace(101, "/projects/from-telegram")
        await sessions.close_db()

    profiles, registry = await _bootstrap(database, 101)
    migration = await sessions.initialize_workshop_workspace_grant_authority(registry, profiles)

    assert migration.newly_migrated == 1
    assert await sessions.get_canonical_workspace_grants(registry.namespaces[0]) == [Path("/projects/from-telegram")]
    await sessions.close_db()


async def test_version_eighty_receipts_upgrade_before_projection_rebuild(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from kai.workshop import schema

    database = tmp_path / "workspace-grant-rebuild.db"
    human = BootstrapHuman(
        "Human 101",
        "admin",
        "telegram",
        "101",
        "101",
        profile_id(101),
    )
    profiles = profile_registry(101)

    with monkeypatch.context() as migration_context:
        migration_context.setattr(schema, "WORKSHOP_SCHEMA_VERSION", 80)
        migration_context.setattr(schema, "_MIGRATIONS", schema._MIGRATIONS[:80])
        await sessions.init_db(database)
        await sessions.add_allowed_workspace(101, "/projects/from-telegram")
        await sessions.bootstrap_workshop_foundation((human,))
        registry, _execution_migration = await sessions.initialize_workshop_execution_state(profiles)
        await sessions.initialize_workshop_workspace_grant_authority(registry, profiles)
        await sessions._get_db().execute(
            "ALTER TABLE workshop_workspace_grant_migrations RENAME TO workshop_workspace_grant_migrations_without_fk"
        )
        await sessions._get_db().execute(
            """
            CREATE TABLE workshop_workspace_grant_migrations (
                runtime_profile_id TEXT PRIMARY KEY CHECK (
                    length(runtime_profile_id) BETWEEN 1 AND 128
                ),
                legacy_runtime_key INTEGER NOT NULL UNIQUE CHECK (legacy_runtime_key > 0),
                principal_id TEXT NOT NULL REFERENCES principals(id) ON DELETE RESTRICT,
                legacy_rows INTEGER NOT NULL CHECK (legacy_rows >= 0),
                migrated_rows INTEGER NOT NULL CHECK (migrated_rows >= 0),
                invalid_rows INTEGER NOT NULL CHECK (invalid_rows >= 0),
                migrated_at TEXT NOT NULL
            )
            """
        )
        await sessions._get_db().execute(
            "INSERT INTO workshop_workspace_grant_migrations "
            "SELECT * FROM workshop_workspace_grant_migrations_without_fk"
        )
        await sessions._get_db().execute("DROP TABLE workshop_workspace_grant_migrations_without_fk")
        await sessions._get_db().commit()
        async with sessions._get_db().execute(
            "SELECT runtime_profile_id, legacy_runtime_key, principal_id, "
            "legacy_rows, migrated_rows, invalid_rows "
            "FROM workshop_workspace_grant_migrations"
        ) as cursor:
            receipt = await cursor.fetchone()
        assert receipt is not None
        async with sessions._get_db().execute("PRAGMA foreign_key_list(workshop_workspace_grant_migrations)") as cursor:
            assert len(await cursor.fetchall()) == 1
        await sessions.close_db()

    await sessions.init_db(database)
    result = await sessions.bootstrap_workshop_foundation((human,))

    assert result.workshop_id
    async with sessions._get_db().execute(
        "SELECT runtime_profile_id, legacy_runtime_key, principal_id, "
        "legacy_rows, migrated_rows, invalid_rows "
        "FROM workshop_workspace_grant_migrations"
    ) as cursor:
        upgraded_receipt = await cursor.fetchone()
    assert upgraded_receipt is not None
    assert tuple(upgraded_receipt) == tuple(receipt)
    async with sessions._get_db().execute("PRAGMA foreign_key_list(workshop_workspace_grant_migrations)") as cursor:
        assert await cursor.fetchall() == []
    async with sessions._get_db().execute("PRAGMA foreign_key_check") as cursor:
        assert await cursor.fetchall() == []
    await sessions.close_db()


async def test_canonical_grants_are_isolated_by_principal_and_runtime(database: Path) -> None:
    profiles, registry = await _bootstrap(database, 101, 202)
    await sessions.initialize_workshop_workspace_grant_authority(registry, profiles)
    first, second = registry.namespaces

    assert await sessions.add_canonical_workspace_grant(
        first,
        "/projects/alice",
        provenance="principal_added",
    )
    assert await sessions.add_canonical_workspace_grant(
        second,
        "/projects/bob",
        provenance="principal_created",
    )

    assert await sessions.get_canonical_workspace_grants(first) == [Path("/projects/alice")]
    assert await sessions.get_canonical_workspace_grants(second) == [Path("/projects/bob")]


async def test_invalid_legacy_grant_is_fail_closed_and_visible(database: Path) -> None:
    await sessions.init_db(database)
    await sessions.add_allowed_workspace(101, "relative/path")
    await sessions.bootstrap_workshop_foundation(
        (
            BootstrapHuman(
                "Human 101",
                "admin",
                "telegram",
                "101",
                "101",
                profile_id(101),
            ),
        )
    )
    profiles = profile_registry(101)
    registry, _execution_migration = await sessions.initialize_workshop_execution_state(profiles)

    migration = await sessions.initialize_workshop_workspace_grant_authority(registry, profiles)

    assert migration.invalid_rows == 1
    assert await sessions.get_canonical_workspace_grants(registry.namespaces[0]) == []
    status = workshop_workspace_grant_status(database)
    assert status.startswith("Workshop workspace grants: INCOMPLETE;")
    assert "invalid=1" in status
