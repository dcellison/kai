"""Canonical mutable execution-state migration and authority tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from kai import sessions
from kai.workshop.bootstrap import BootstrapHuman
from kai.workshop.diagnostics import workshop_execution_state_status
from kai.workshop.execution_state import WorkshopExecutionStateError
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id, profile_registry


async def _initialize(db_path: Path, *runtime_ids: int):
    await sessions.init_db(db_path)
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
    return await sessions.initialize_workshop_execution_state(profile_registry(*runtime_ids))


@pytest.fixture
async def database(tmp_path: Path):
    path = tmp_path / "kai.db"
    yield path
    await sessions.close_db()


class TestCanonicalExecutionStateMigration:
    async def test_replaces_execution_and_workspace_settings_in_one_canonical_transaction(
        self,
        database: Path,
    ) -> None:
        registry, _ = await _initialize(database, 101)
        namespace = registry.namespaces[0]
        await sessions.set_canonical_execution_setting(namespace, "model", "gpt-5.6-sol")
        await sessions.set_canonical_workspace_config_setting(
            namespace,
            "/projects/kai",
            "env",
            '{"SECRET":"preserved"}',
        )

        await sessions.replace_canonical_settings_state(
            namespace,
            {
                "backend": "codex",
                "timeout": "180",
                "workspace": "/projects/kai",
            },
            workspace_path="/projects/kai",
            workspace_settings={"env": '{"SECRET":"preserved"}', "prompt": "bounded prompt"},
        )

        assert await sessions.get_canonical_execution_settings(namespace) == {
            "backend": "codex",
            "timeout": "180",
            "workspace": "/projects/kai",
        }
        assert await sessions.get_canonical_workspace_config_settings(namespace, "/projects/kai") == {
            "env": '{"SECRET":"preserved"}',
            "prompt": "bounded prompt",
        }

    async def test_version_twenty_database_upgrades_additively(self, tmp_path: Path, monkeypatch):
        from kai.workshop import schema

        path = tmp_path / "preexisting.db"
        with monkeypatch.context() as migration_context:
            migration_context.setattr(schema, "WORKSHOP_SCHEMA_VERSION", 20)
            migration_context.setattr(schema, "_MIGRATIONS", schema._MIGRATIONS[:20])
            old_store = await WorkshopEventStore.open(path)
            await old_store.connection.execute(
                "INSERT INTO workshops (id, name, created_at) VALUES (?, ?, ?)",
                ("wsp_00000000000000000000000000000001", "Preserved", "2026-08-15T00:00:00Z"),
            )
            await old_store.connection.commit()
            await old_store.close()

        upgraded = await WorkshopEventStore.open(path)
        try:
            assert await upgraded.schema_version() == 43
            tables = await upgraded.schema_tables()
            assert {
                "channel_agent_execution_settings",
                "channel_agent_workspace_settings",
                "principal_workspace_history",
                "principal_workspace_grants",
                "workshop_execution_state_migrations",
            } <= tables
            async with upgraded.connection.execute("SELECT name FROM workshops") as cursor:
                assert (await cursor.fetchone())[0] == "Preserved"
        finally:
            await upgraded.close()

    async def test_version_thirty_three_preserves_settings_and_allows_backend_selection(
        self,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        from kai.workshop import schema

        path = tmp_path / "pre-backend-selection.db"
        with monkeypatch.context() as migration_context:
            migration_context.setattr(schema, "WORKSHOP_SCHEMA_VERSION", 33)
            migration_context.setattr(schema, "_MIGRATIONS", schema._MIGRATIONS[:33])
            old_store = await WorkshopEventStore.open(path)
            await old_store.connection.execute(
                "INSERT INTO channel_agent_execution_settings "
                "(channel_id, agent_id, runtime_profile_id, field, value) "
                "VALUES (?, ?, ?, 'timeout', '180')",
                ("chn_" + "1" * 32, "agt_" + "2" * 32, str(profile_id(101))),
            )
            await old_store.connection.commit()
            await old_store.close()

        upgraded = await WorkshopEventStore.open(path)
        try:
            assert await upgraded.schema_version() == 43
            async with upgraded.connection.execute(
                "SELECT field, value FROM channel_agent_execution_settings"
            ) as cursor:
                assert tuple(await cursor.fetchone()) == ("timeout", "180")
            await upgraded.connection.execute(
                "INSERT INTO channel_agent_execution_settings "
                "(channel_id, agent_id, runtime_profile_id, field, value) "
                "VALUES (?, ?, ?, 'backend', 'codex')",
                ("chn_" + "1" * 32, "agt_" + "2" * 32, str(profile_id(101))),
            )
            await upgraded.connection.commit()
        finally:
            await upgraded.close()

    async def test_backfills_all_state_once_and_reads_only_canonical_rows(self, database: Path):
        await sessions.init_db(database)
        await sessions.set_setting("model:101", "gpt-5.6-sol")
        await sessions.set_setting("timeout:101", "240")
        await sessions.set_setting("workspace:101", "/projects/current")
        await sessions.set_setting("ws_config:101:/projects/a:b:model", "gpt-5.5")
        await sessions.upsert_workspace_history("/projects/recent", 101)
        await sessions.add_allowed_workspace(101, "/projects/granted")
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

        registry, migration = await sessions.initialize_workshop_execution_state(profile_registry(101))

        assert len(registry.namespaces) == 1
        assert migration.newly_migrated == 1
        assert migration.settings == 3
        assert migration.workspace_settings == 1
        assert migration.history == 1
        assert migration.grants == 1
        assert await sessions.get_user_settings(101) == {"model": "gpt-5.6-sol", "timeout": "240"}
        assert await sessions.get_active_workspace(101) == "/projects/current"
        assert await sessions.get_workspace_config_settings(101, "/projects/a:b") == {"model": "gpt-5.5"}
        history = await sessions.get_workspace_history(101)
        assert [row["path"] for row in history] == ["/projects/recent"]
        assert history[0]["last_used_at"]
        assert await sessions.get_allowed_workspaces(101) == [Path("/projects/granted")]

        # Protected reads no longer consult the integer compatibility stores.
        await sessions._get_db().execute("UPDATE settings SET value = 'tampered' WHERE key = 'model:101'")
        await sessions._get_db().execute("DELETE FROM workspace_history WHERE chat_id = 101")
        await sessions._get_db().execute("DELETE FROM allowed_workspaces WHERE chat_id = 101")
        await sessions._get_db().commit()

        assert await sessions.get_user_settings(101) == {"model": "gpt-5.6-sol", "timeout": "240"}
        assert [row["path"] for row in await sessions.get_workspace_history(101)] == ["/projects/recent"]
        assert await sessions.get_allowed_workspaces(101) == [Path("/projects/granted")]

    async def test_restart_is_idempotent_and_never_reimports_legacy_tampering(self, database: Path):
        await sessions.init_db(database)
        await sessions.set_setting("model:101", "gpt-5.6-sol")
        await sessions.bootstrap_workshop_foundation(
            (BootstrapHuman("Human 101", "admin", "telegram", "101", "101", profile_id(101)),)
        )
        await sessions.initialize_workshop_execution_state(profile_registry(101))
        await sessions.close_db()

        await sessions.init_db(database)
        await sessions.set_setting("model:101", "legacy-change")
        await sessions.bootstrap_workshop_foundation(
            (BootstrapHuman("Human 101", "admin", "telegram", "101", "101", profile_id(101)),)
        )
        _registry, migration = await sessions.initialize_workshop_execution_state(profile_registry(101))

        assert migration.newly_migrated == 0
        assert await sessions.get_user_settings(101) == {"model": "gpt-5.6-sol"}

    async def test_changed_canonical_ownership_fails_closed(self, database: Path):
        await _initialize(database, 101)
        await sessions._get_db().execute(
            "UPDATE workshop_execution_state_migrations SET channel_id = ?",
            ("chn_00000000000000000000000000000001",),
        )
        await sessions._get_db().commit()

        with pytest.raises(
            WorkshopExecutionStateError,
            match=(
                r"conflicts with current protected ownership for runtime profile rtp_.*; "
                r"restore its recorded canonical assignment or restore the database from backup"
            ),
        ):
            await sessions.initialize_workshop_execution_state(profile_registry(101))


class TestCanonicalExecutionStateWrites:
    async def test_protected_session_calls_cannot_read_or_mutate_legacy_archive(
        self,
        database: Path,
    ):
        await _initialize(database, 101)
        await sessions._get_db().execute(
            "INSERT INTO sessions (chat_id, session_id, model) VALUES (?, ?, ?)",
            (101, "archived-session", "archived-model"),
        )
        await sessions._get_db().commit()

        assert await sessions.get_session(101) is None
        await sessions.save_session(101, "new-session", "gpt-5.6-sol")
        await sessions.clear_session(101)

        async with sessions._get_db().execute(
            "SELECT session_id, model FROM sessions WHERE chat_id = ?",
            (101,),
        ) as cursor:
            archived = await cursor.fetchone()
        assert archived is not None
        assert tuple(archived) == ("archived-session", "archived-model")

    async def test_protected_mutations_freeze_legacy_archive_and_isolate_humans(self, database: Path):
        await _initialize(database, 101, 202)

        await sessions.set_user_setting(101, "model", "gpt-5.5")
        await sessions.set_active_workspace(101, "/projects/alice")
        await sessions.set_workspace_config_setting(101, "/projects/alice", "timeout", "300")
        await sessions.upsert_workspace_history("/projects/alice", 101)
        await sessions.add_allowed_workspace(101, "/projects/alice")

        assert await sessions.get_user_settings(101) == {"model": "gpt-5.5"}
        assert await sessions.get_user_settings(202) == {}
        assert await sessions.get_workspace_history(202) == []
        assert await sessions.get_allowed_workspaces(202) == []
        for key in (
            "model:101",
            "workspace:101",
            "ws_config:101:/projects/alice:timeout",
        ):
            assert await sessions.get_setting(key) is None

        await sessions.delete_user_setting(101, "model")
        await sessions.delete_active_workspace(101)
        await sessions.delete_all_workspace_config(101, "/projects/alice")
        await sessions.delete_workspace_history("/projects/alice", 101)
        assert await sessions.remove_allowed_workspace(101, "/projects/alice") is True

        assert await sessions.get_user_settings(101) == {}
        assert await sessions.get_active_workspace(101) is None
        assert await sessions.get_workspace_config_settings(101, "/projects/alice") == {}
        assert await sessions.get_workspace_history(101) == []
        assert await sessions.get_allowed_workspaces(101) == []

    async def test_workspace_config_delete_preserves_colon_prefixed_canonical_workspace(
        self,
        database: Path,
    ):
        await _initialize(database, 101)
        shorter = "/projects/alice"
        colon_prefixed = "/projects/alice:archive"
        await sessions.set_workspace_config_setting(101, shorter, "model", "gpt-5.5")
        await sessions.set_workspace_config_setting(101, colon_prefixed, "model", "gpt-5.6-sol")

        await sessions.delete_all_workspace_config(101, shorter)

        assert await sessions.get_workspace_config_settings(101, shorter) == {}
        assert await sessions.get_workspace_config_settings(101, colon_prefixed) == {"model": "gpt-5.6-sol"}
        assert await sessions.get_setting("ws_config:101:/projects/alice:model") is None
        assert await sessions.get_setting("ws_config:101:/projects/alice:archive:model") is None

    async def test_unmapped_positive_adapter_keys_fail_closed_after_cutover(self, database: Path):
        await _initialize(database, 101)

        with pytest.raises(RuntimeError, match="no canonical execution-state owner"):
            await sessions.set_user_setting(999, "model", "development-model")
        with pytest.raises(RuntimeError, match="no canonical execution-state owner"):
            await sessions.set_active_workspace(999, "/tmp/development")
        with pytest.raises(RuntimeError, match="no canonical execution-state owner"):
            await sessions.add_allowed_workspace(999, "/tmp/development")


class TestCanonicalExecutionStateDiagnostic:
    async def test_reports_counts_without_identity_values(self, database: Path):
        await _initialize(database, 101)
        await sessions.set_user_setting(101, "model", "secret-model-name")

        status = workshop_execution_state_status(database)

        assert status.startswith("Workshop execution state: active;")
        assert "profiles=1, migrated=1, missing=0, stale=0" in status
        assert "protected legacy reads=disabled" in status
        assert "rollback dual writes=disabled" in status
        assert "secret-model-name" not in status
        assert "101" not in status

    async def test_reports_unmapped_legacy_state_without_exposing_it(self, database: Path):
        await _initialize(database, 101)
        await sessions.set_setting("model:999", "unmapped-secret")
        await sessions._get_db().execute(
            "INSERT INTO workspace_history (path, chat_id) VALUES (?, ?)",
            ("/secret/path", 999),
        )
        await sessions._get_db().commit()

        status = workshop_execution_state_status(database)

        assert status.startswith("Workshop execution state: INCOMPLETE;")
        assert "unclassified=2" in status
        assert "unmapped-secret" not in status
        assert "/secret/path" not in status
        assert "999" not in status
