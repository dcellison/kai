"""Tests for sessions.py async database CRUD."""

import sqlite3
import stat
from pathlib import Path

import aiosqlite
import pytest

from kai import sessions


@pytest.fixture
async def db(tmp_path):
    """Initialize a fresh database for each test."""
    await sessions.init_db(tmp_path / "test.db")
    yield
    await sessions.close_db()


# ── Sessions ─────────────────────────────────────────────────────────


class TestSessions:
    async def test_get_unknown_returns_none(self, db):
        assert await sessions.get_session(999) is None

    async def test_save_then_get(self, db):
        await sessions.save_session(1, "sess-abc", "sonnet")
        result = await sessions.get_session(1)
        assert result == "sess-abc"

    async def test_save_twice_updates_in_place(self, db):
        await sessions.save_session(1, "sess-1", "sonnet")
        await sessions.save_session(1, "sess-2", "opus")
        stats = await sessions.get_stats(1)
        assert stats["session_id"] == "sess-2"
        assert stats["model"] == "opus"

    async def test_clear_session(self, db):
        await sessions.save_session(1, "sess-1", "sonnet")
        await sessions.clear_session(1)
        assert await sessions.get_session(1) is None

    async def test_get_stats(self, db):
        await sessions.save_session(1, "sess-1", "opus")
        stats = await sessions.get_stats(1)
        assert stats["session_id"] == "sess-1"
        assert stats["model"] == "opus"
        assert "created_at" in stats
        assert "last_used_at" in stats

    async def test_get_stats_unknown(self, db):
        assert await sessions.get_stats(999) is None


class TestTelegramUpdateQueue:
    async def test_enqueue_is_idempotent_by_update_id(self, db):
        first_id, first_inserted = await sessions.enqueue_telegram_update(1001, '{"update_id":1001}')
        second_id, second_inserted = await sessions.enqueue_telegram_update(1001, '{"update_id":1001,"retry":true}')

        assert first_inserted is True
        assert second_inserted is False
        assert second_id == first_id

        async with sessions._get_db().execute("SELECT COUNT(*) FROM telegram_update_queue") as cursor:
            row = await cursor.fetchone()
        assert row[0] == 1

    async def test_claim_and_complete_update(self, db):
        row_id, _ = await sessions.enqueue_telegram_update(1002, '{"update_id":1002}')

        claimed = await sessions.claim_next_telegram_update()

        assert claimed is not None
        assert claimed["id"] == row_id
        assert claimed["update_id"] == 1002
        assert claimed["payload"] == '{"update_id":1002}'
        assert claimed["status"] == "processing"
        assert claimed["attempt_count"] == 1
        assert await sessions.claim_next_telegram_update() is None

        assert await sessions.complete_telegram_update(row_id) is True
        assert await sessions.complete_telegram_update(row_id) is False
        assert await sessions.claim_next_telegram_update() is None

    async def test_retry_returns_processing_update_to_pending(self, db):
        row_id, _ = await sessions.enqueue_telegram_update(1003, '{"update_id":1003}')
        first_claim = await sessions.claim_next_telegram_update()
        assert first_claim is not None

        assert await sessions.retry_telegram_update(row_id, "temporary failure") is True

        second_claim = await sessions.claim_next_telegram_update()
        assert second_claim is not None
        assert second_claim["id"] == row_id
        assert second_claim["attempt_count"] == 2
        assert second_claim["last_error"] == "temporary failure"

    async def test_discard_marks_processing_update_done_with_error(self, db):
        row_id, _ = await sessions.enqueue_telegram_update(1005, '{"update_id":1005}')
        claimed = await sessions.claim_next_telegram_update()
        assert claimed is not None

        assert await sessions.discard_telegram_update(row_id, "poison update") is True

        async with sessions._get_db().execute(
            "SELECT status, last_error FROM telegram_update_queue WHERE id = ?",
            (row_id,),
        ) as cursor:
            row = await cursor.fetchone()
        assert row["status"] == "done"
        assert row["last_error"] == "poison update"
        assert await sessions.claim_next_telegram_update() is None

    async def test_requeue_processing_updates_for_startup_recovery(self, db):
        row_id, _ = await sessions.enqueue_telegram_update(1004, '{"update_id":1004}')
        first_claim = await sessions.claim_next_telegram_update()
        assert first_claim is not None

        assert await sessions.requeue_processing_telegram_updates() == 1

        recovered_claim = await sessions.claim_next_telegram_update()
        assert recovered_claim is not None
        assert recovered_claim["id"] == row_id
        assert recovered_claim["attempt_count"] == 2


class TestMemoryProjectRows:
    """Persistence layer for chat-registered memory projects. The
    merge/validation logic lives in kai.memory_projects; these tests
    pin the CRUD contract and the DB-level uniqueness backstops."""

    async def test_register_and_list_roundtrip(self, db):
        await sessions.register_memory_project(
            project_id="phi",
            display_name="Phi",
            workspace_root="/work/phi",
            created_by=123,
        )
        rows = await sessions.get_memory_project_rows()
        assert len(rows) == 1
        row = rows[0]
        assert row["project_id"] == "phi"
        assert row["display_name"] == "Phi"
        assert row["workspace_root"] == "/work/phi"
        assert row["memory_enabled"] is True
        assert row["default_scope_for_new_facts"] == "project"
        assert row["created_by"] == 123

    async def test_unregister_returns_whether_row_existed(self, db):
        await sessions.register_memory_project(
            project_id="phi",
            display_name="Phi",
            workspace_root="/work/phi",
            created_by=123,
        )
        assert await sessions.unregister_memory_project("phi") is True
        assert await sessions.unregister_memory_project("phi") is False
        assert await sessions.get_memory_project_rows() == []

    async def test_duplicate_project_id_raises(self, db):
        """PRIMARY KEY backstop for the registration race; the
        handler checks the merged registry first, but two concurrent
        registrations must not both land."""
        await sessions.register_memory_project(
            project_id="phi",
            display_name="Phi",
            workspace_root="/work/phi",
            created_by=123,
        )
        with pytest.raises(sqlite3.IntegrityError):
            await sessions.register_memory_project(
                project_id="phi",
                display_name="Phi Again",
                workspace_root="/work/elsewhere",
                created_by=456,
            )

    async def test_duplicate_root_raises(self, db):
        """UNIQUE(workspace_root) backstop: the detector needs a
        single owner per root."""
        await sessions.register_memory_project(
            project_id="phi",
            display_name="Phi",
            workspace_root="/work/phi",
            created_by=123,
        )
        with pytest.raises(sqlite3.IntegrityError):
            await sessions.register_memory_project(
                project_id="phi2",
                display_name="Phi Two",
                workspace_root="/work/phi",
                created_by=456,
            )


# ── Jobs ─────────────────────────────────────────────────────────────


class TestJobs:
    async def test_create_returns_int_id(self, db):
        job_id = await sessions.create_job(
            chat_id=1,
            name="test",
            job_type="reminder",
            prompt="hello",
            schedule_type="once",
            schedule_data='{"run_at": "2026-12-01T00:00:00"}',
        )
        assert isinstance(job_id, int)

    async def test_get_jobs_returns_active(self, db):
        await sessions.create_job(
            chat_id=1,
            name="j1",
            job_type="reminder",
            prompt="p1",
            schedule_type="once",
            schedule_data="{}",
        )
        jobs = await sessions.get_jobs(1)
        assert len(jobs) == 1
        assert jobs[0]["name"] == "j1"

    async def test_get_jobs_filters_by_chat(self, db):
        await sessions.create_job(
            chat_id=1,
            name="j1",
            job_type="reminder",
            prompt="p",
            schedule_type="once",
            schedule_data="{}",
        )
        await sessions.create_job(
            chat_id=2,
            name="j2",
            job_type="reminder",
            prompt="p",
            schedule_type="once",
            schedule_data="{}",
        )
        assert len(await sessions.get_jobs(1)) == 1
        assert len(await sessions.get_jobs(2)) == 1

    async def test_get_job_by_id(self, db):
        job_id = await sessions.create_job(
            chat_id=1,
            name="j1",
            job_type="agent",
            prompt="analyze",
            schedule_type="daily",
            schedule_data='{"times": ["09:00"]}',
        )
        job = await sessions.get_job_by_id(job_id)
        assert job is not None
        assert job["name"] == "j1"
        assert job["job_type"] == "agent"

    async def test_legacy_agent_type_is_stored_canonically(self, db):
        job_id = await sessions.create_job(
            chat_id=1,
            name="legacy",
            job_type="claude",
            prompt="analyze",
            schedule_type="daily",
            schedule_data='{"times": ["09:00"]}',
        )
        job = await sessions.get_job_by_id(job_id)
        assert job is not None
        assert job["job_type"] == "agent"

    async def test_create_job_rejects_unknown_type(self, db):
        with pytest.raises(ValueError, match="job_type must be one of: reminder, agent"):
            await sessions.create_job(
                chat_id=1,
                name="invalid",
                job_type="unknown",
                prompt="do something",
                schedule_type="once",
                schedule_data="{}",
            )

        assert await sessions.get_jobs(1) == []

    async def test_get_job_by_id_unknown(self, db):
        assert await sessions.get_job_by_id(999) is None

    async def test_get_all_active_jobs(self, db):
        await sessions.create_job(
            chat_id=1,
            name="j1",
            job_type="reminder",
            prompt="p",
            schedule_type="once",
            schedule_data="{}",
        )
        await sessions.create_job(
            chat_id=2,
            name="j2",
            job_type="reminder",
            prompt="p",
            schedule_type="once",
            schedule_data="{}",
        )
        all_jobs = await sessions.get_all_active_jobs()
        assert len(all_jobs) == 2

    async def test_deactivate_job(self, db):
        job_id = await sessions.create_job(
            chat_id=1,
            name="j1",
            job_type="reminder",
            prompt="p",
            schedule_type="once",
            schedule_data="{}",
        )
        await sessions.deactivate_job(job_id)
        assert len(await sessions.get_jobs(1)) == 0
        assert len(await sessions.get_all_active_jobs()) == 0

    async def test_delete_job(self, db):
        job_id = await sessions.create_job(
            chat_id=1,
            name="j1",
            job_type="reminder",
            prompt="p",
            schedule_type="once",
            schedule_data="{}",
        )
        assert await sessions.delete_job(job_id) is True
        assert await sessions.get_job_by_id(job_id) is None

    async def test_delete_job_nonexistent(self, db):
        assert await sessions.delete_job(999) is False

    async def test_update_job_single_field(self, db):
        job_id = await sessions.create_job(
            chat_id=1,
            name="original",
            job_type="reminder",
            prompt="original prompt",
            schedule_type="once",
            schedule_data='{"run_at": "2026-02-20T10:00:00+00:00"}',
        )
        updated = await sessions.update_job(job_id, name="updated")
        assert updated is True
        job = await sessions.get_job_by_id(job_id)
        assert job is not None
        assert job["name"] == "updated"
        assert job["prompt"] == "original prompt"

    async def test_update_job_multiple_fields(self, db):
        job_id = await sessions.create_job(
            chat_id=1,
            name="j1",
            job_type="agent",
            prompt="old prompt",
            schedule_type="interval",
            schedule_data='{"seconds": 3600}',
            auto_remove=False,
        )
        updated = await sessions.update_job(
            job_id,
            prompt="new prompt",
            schedule_data='{"seconds": 7200}',
            auto_remove=True,
        )
        assert updated is True
        job = await sessions.get_job_by_id(job_id)
        assert job is not None
        assert job["prompt"] == "new prompt"
        assert job["schedule_data"] == '{"seconds": 7200}'
        assert job["auto_remove"] is True

    async def test_update_job_inactive_returns_false(self, db):
        job_id = await sessions.create_job(
            chat_id=1,
            name="j1",
            job_type="reminder",
            prompt="p",
            schedule_type="once",
            schedule_data="{}",
        )
        await sessions.deactivate_job(job_id)
        updated = await sessions.update_job(job_id, name="new name")
        assert updated is False

    async def test_update_job_nonexistent_returns_false(self, db):
        updated = await sessions.update_job(999, name="new name")
        assert updated is False

    async def test_update_job_no_fields_returns_false(self, db):
        job_id = await sessions.create_job(
            chat_id=1,
            name="j1",
            job_type="reminder",
            prompt="p",
            schedule_type="once",
            schedule_data="{}",
        )
        updated = await sessions.update_job(job_id)
        assert updated is False

    async def test_auto_remove_stored_as_bool(self, db):
        job_id = await sessions.create_job(
            chat_id=1,
            name="j1",
            job_type="agent",
            prompt="check",
            schedule_type="interval",
            schedule_data='{"seconds": 3600}',
            auto_remove=True,
        )
        job = await sessions.get_job_by_id(job_id)
        assert job["auto_remove"] is True

        job_id2 = await sessions.create_job(
            chat_id=1,
            name="j2",
            job_type="reminder",
            prompt="hi",
            schedule_type="once",
            schedule_data="{}",
            auto_remove=False,
        )
        job2 = await sessions.get_job_by_id(job_id2)
        assert job2["auto_remove"] is False


# ── Settings ─────────────────────────────────────────────────────────


class TestSettings:
    async def test_get_unknown_returns_none(self, db):
        assert await sessions.get_setting("nonexistent") is None

    async def test_set_then_get(self, db):
        await sessions.set_setting("theme", "dark")
        assert await sessions.get_setting("theme") == "dark"

    async def test_set_overwrites(self, db):
        await sessions.set_setting("theme", "dark")
        await sessions.set_setting("theme", "light")
        assert await sessions.get_setting("theme") == "light"

    async def test_delete_setting(self, db):
        await sessions.set_setting("key", "val")
        await sessions.delete_setting("key")
        assert await sessions.get_setting("key") is None

    async def test_delete_settings_by_prefix(self, db):
        """Removes all keys matching the prefix, leaves others intact."""
        await sessions.set_setting("memory_seeded:111", "1")
        await sessions.set_setting("memory_seeded:222", "1")
        await sessions.set_setting("workspace:111", "/home")
        await sessions.delete_settings_by_prefix("memory_seeded:")
        # Prefixed keys gone
        assert await sessions.get_setting("memory_seeded:111") is None
        assert await sessions.get_setting("memory_seeded:222") is None
        # Unrelated key untouched
        assert await sessions.get_setting("workspace:111") == "/home"

    async def test_delete_settings_by_prefix_noop_when_empty(self, db):
        """No error when no keys match the prefix."""
        await sessions.delete_settings_by_prefix("nonexistent:")
        # Should not raise


# ── Workspace config overrides ─────────────────────────────────────


class TestWorkspaceConfigSettings:
    """Tests for per-user-per-workspace config stored in the settings table."""

    async def test_empty_for_unconfigured(self, db):
        """Returns empty dict when no overrides exist."""
        result = await sessions.get_workspace_config_settings(111, "/some/path")
        assert result == {}

    async def test_set_and_get(self, db):
        """Set a field and retrieve it."""
        await sessions.set_workspace_config_setting(111, "/projects/kai", "model", "opus")
        result = await sessions.get_workspace_config_settings(111, "/projects/kai")
        assert result == {"model": "opus"}

    async def test_set_multiple_fields(self, db):
        """Multiple fields for the same workspace are returned together."""
        await sessions.set_workspace_config_setting(111, "/projects/kai", "model", "opus")
        await sessions.set_workspace_config_setting(111, "/projects/kai", "timeout", "300")
        result = await sessions.get_workspace_config_settings(111, "/projects/kai")
        assert result == {"model": "opus", "timeout": "300"}

    async def test_delete_single_field(self, db):
        """Deleting one field leaves others intact."""
        await sessions.set_workspace_config_setting(111, "/projects/kai", "model", "opus")
        await sessions.set_workspace_config_setting(111, "/projects/kai", "timeout", "300")
        await sessions.delete_workspace_config_setting(111, "/projects/kai", "model")
        result = await sessions.get_workspace_config_settings(111, "/projects/kai")
        assert result == {"timeout": "300"}

    async def test_delete_all(self, db):
        """Bulk delete removes all overrides for a workspace."""
        await sessions.set_workspace_config_setting(111, "/projects/kai", "model", "opus")
        await sessions.set_workspace_config_setting(111, "/projects/kai", "timeout", "300")
        await sessions.delete_all_workspace_config(111, "/projects/kai")
        result = await sessions.get_workspace_config_settings(111, "/projects/kai")
        assert result == {}

    async def test_workspace_isolation(self, db):
        """Settings for workspace A don't leak into workspace B."""
        await sessions.set_workspace_config_setting(111, "/projects/kai", "model", "opus")
        await sessions.set_workspace_config_setting(111, "/projects/anvil", "model", "haiku")
        kai = await sessions.get_workspace_config_settings(111, "/projects/kai")
        anvil = await sessions.get_workspace_config_settings(111, "/projects/anvil")
        assert kai == {"model": "opus"}
        assert anvil == {"model": "haiku"}

    async def test_user_isolation(self, db):
        """Settings for user A don't leak into user B on the same workspace."""
        await sessions.set_workspace_config_setting(111, "/projects/kai", "model", "opus")
        await sessions.set_workspace_config_setting(222, "/projects/kai", "model", "haiku")
        user_a = await sessions.get_workspace_config_settings(111, "/projects/kai")
        user_b = await sessions.get_workspace_config_settings(222, "/projects/kai")
        assert user_a == {"model": "opus"}
        assert user_b == {"model": "haiku"}

    async def test_delete_all_preserves_other_workspaces(self, db):
        """Bulk delete for workspace A doesn't touch workspace B."""
        await sessions.set_workspace_config_setting(111, "/projects/kai", "model", "opus")
        await sessions.set_workspace_config_setting(111, "/projects/anvil", "model", "haiku")
        await sessions.delete_all_workspace_config(111, "/projects/kai")
        assert await sessions.get_workspace_config_settings(111, "/projects/kai") == {}
        assert await sessions.get_workspace_config_settings(111, "/projects/anvil") == {"model": "haiku"}

    async def test_overwrite_existing_field(self, db):
        """Setting a field that already exists overwrites it."""
        await sessions.set_workspace_config_setting(111, "/projects/kai", "model", "opus")
        await sessions.set_workspace_config_setting(111, "/projects/kai", "model", "sonnet")
        result = await sessions.get_workspace_config_settings(111, "/projects/kai")
        assert result == {"model": "sonnet"}


# ── build_workspace_config merge logic ─────────────────────────────


class TestBuildWorkspaceConfig:
    """Tests for the YAML + DB merge function."""

    async def test_neither_returns_none(self, db):
        """No YAML config and no DB overrides returns None."""
        result = await sessions.build_workspace_config(None, Path("/projects/kai"), 111)
        assert result is None

    async def test_yaml_only(self, db):
        """YAML config present, no DB overrides, returns YAML values."""
        from kai.config import WorkspaceConfig

        yaml = WorkspaceConfig(path=Path("/projects/kai"), model="opus", timeout=150)
        result = await sessions.build_workspace_config(yaml, Path("/projects/kai"), 111)
        assert result is not None
        assert result.model == "opus"
        assert result.timeout == 150

    async def test_db_only(self, db):
        """No YAML config, DB overrides present, returns DB values."""
        await sessions.set_workspace_config_setting(111, "/projects/kai", "model", "haiku")
        await sessions.set_workspace_config_setting(111, "/projects/kai", "timeout", "150")
        result = await sessions.build_workspace_config(None, Path("/projects/kai"), 111)
        assert result is not None
        assert result.model == "haiku"
        assert result.timeout == 150
        assert result.path == Path("/projects/kai")

    async def test_db_overrides_yaml(self, db):
        """DB values take precedence over YAML values."""
        from kai.config import WorkspaceConfig

        yaml = WorkspaceConfig(path=Path("/projects/kai"), model="opus", timeout=150)
        await sessions.set_workspace_config_setting(111, "/projects/kai", "model", "sonnet")
        result = await sessions.build_workspace_config(yaml, Path("/projects/kai"), 111)
        assert result is not None
        assert result.model == "sonnet"
        # Timeout from YAML is preserved (not overridden)
        assert result.timeout == 150

    async def test_partial_override(self, db):
        """YAML has model+timeout, DB overrides only model. Timeout from YAML."""
        from kai.config import WorkspaceConfig

        yaml = WorkspaceConfig(path=Path("/projects/kai"), model="opus", timeout=300)
        await sessions.set_workspace_config_setting(111, "/projects/kai", "model", "haiku")
        result = await sessions.build_workspace_config(yaml, Path("/projects/kai"), 111)
        assert result is not None
        assert result.model == "haiku"
        assert result.timeout == 300

    async def test_env_merge(self, db):
        """DB env vars merge on top of YAML env vars; DB wins on collision."""
        import json

        from kai.config import WorkspaceConfig

        yaml = WorkspaceConfig(
            path=Path("/projects/kai"),
            env={"EXISTING": "from_yaml", "SHARED": "yaml_value"},
        )
        await sessions.set_workspace_config_setting(
            111,
            "/projects/kai",
            "env",
            json.dumps({"NEW_VAR": "from_db", "SHARED": "db_wins"}),
        )
        result = await sessions.build_workspace_config(yaml, Path("/projects/kai"), 111)
        assert result is not None
        assert result.env is not None
        assert result.env["EXISTING"] == "from_yaml"
        assert result.env["NEW_VAR"] == "from_db"
        assert result.env["SHARED"] == "db_wins"

    async def test_db_prompt_replaces_yaml_file(self, db):
        """DB prompt clears system_prompt_file from YAML."""
        from kai.config import WorkspaceConfig

        yaml = WorkspaceConfig(
            path=Path("/projects/kai"),
            system_prompt_file=Path("/etc/kai/prompts/default.txt"),
        )
        await sessions.set_workspace_config_setting(111, "/projects/kai", "prompt", "Be concise.")
        result = await sessions.build_workspace_config(yaml, Path("/projects/kai"), 111)
        assert result is not None
        assert result.system_prompt == "Be concise."
        assert result.system_prompt_file is None

    async def test_env_file_preserved_from_yaml(self, db):
        """env_file from YAML is preserved even when DB has no env override."""
        from kai.config import WorkspaceConfig

        yaml = WorkspaceConfig(
            path=Path("/projects/kai"),
            env_file=Path("/etc/kai/env/extra.env"),
        )
        result = await sessions.build_workspace_config(yaml, Path("/projects/kai"), 111)
        assert result is not None
        assert result.env_file == Path("/etc/kai/env/extra.env")


# ── Per-user settings ──────────────────────────────────────────────


class TestUserSettings:
    """Tests for per-user settings CRUD in the settings table."""

    async def test_empty_returns_empty_dict(self, db):
        """No settings returns empty dict."""
        result = await sessions.get_user_settings(111)
        assert result == {}

    async def test_set_and_get(self, db):
        """Set a field and retrieve it."""
        await sessions.set_user_setting(111, "model", "opus")
        result = await sessions.get_user_settings(111)
        assert result == {"model": "opus"}

    async def test_set_multiple_fields(self, db):
        """Multiple fields are returned together."""
        await sessions.set_user_setting(111, "model", "opus")
        await sessions.set_user_setting(111, "timeout", "300")
        result = await sessions.get_user_settings(111)
        assert result == {"model": "opus", "timeout": "300"}

    async def test_delete_single(self, db):
        """Deleting one field leaves others intact."""
        await sessions.set_user_setting(111, "model", "opus")
        await sessions.set_user_setting(111, "timeout", "300")
        await sessions.delete_user_setting(111, "model")
        result = await sessions.get_user_settings(111)
        assert result == {"timeout": "300"}

    async def test_delete_all(self, db):
        """Bulk delete removes all per-user settings."""
        await sessions.set_user_setting(111, "model", "opus")
        await sessions.set_user_setting(111, "timeout", "300")
        await sessions.delete_all_user_settings(111)
        result = await sessions.get_user_settings(111)
        assert result == {}

    async def test_user_isolation(self, db):
        """User A's settings don't appear in user B's query."""
        await sessions.set_user_setting(111, "model", "opus")
        await sessions.set_user_setting(222, "model", "haiku")
        assert (await sessions.get_user_settings(111)) == {"model": "opus"}
        assert (await sessions.get_user_settings(222)) == {"model": "haiku"}

    async def test_overwrite_existing(self, db):
        """Setting a field that already exists overwrites it."""
        await sessions.set_user_setting(111, "model", "opus")
        await sessions.set_user_setting(111, "model", "sonnet")
        result = await sessions.get_user_settings(111)
        assert result == {"model": "sonnet"}

    async def test_delete_nonexistent_is_noop(self, db):
        """Deleting a field that doesn't exist is a no-op."""
        await sessions.delete_user_setting(111, "model")
        result = await sessions.get_user_settings(111)
        assert result == {}


# ── resolve_user_defaults ─────────────────────────────────────────


class TestResolveUserDefaults:
    """Tests for the per-user settings resolution function."""

    def _make_config(self, user_configs: dict | None = None, **kwargs):
        """Build a minimal Config with overridable defaults."""
        from kai.config import Config

        defaults = {
            "telegram_bot_token": "test",
            "allowed_user_ids": {111},
            "default_model": "sonnet",
            "default_timeout": 120,
        }
        defaults.update(kwargs)
        if user_configs is not None:
            defaults["user_configs"] = user_configs
        return Config(**defaults)

    async def test_no_overrides_returns_globals(self, db):
        """With no DB or YAML overrides, returns global defaults."""
        config = self._make_config()
        result = await sessions.resolve_user_defaults(111, config)
        assert result["model"] == "sonnet"
        assert result["timeout"] == 120

    async def test_db_overrides_globals(self, db):
        """DB settings override global defaults."""
        config = self._make_config()
        await sessions.set_user_setting(111, "model", "opus")
        result = await sessions.resolve_user_defaults(111, config)
        assert result["model"] == "opus"
        # Unset fields still come from globals
        assert result["timeout"] == 120

    async def test_codex_legacy_gpt56_setting_resolves_to_sol(self, db):
        """Settings displays use the same exact Codex ID as runtime dispatch."""
        config = self._make_config(
            default_backend="codex",
            default_provider="openai",
            default_model="gpt-5.5",
        )
        await sessions.set_user_setting(111, "model", "gpt-5.6")

        result = await sessions.resolve_user_defaults(111, config)

        assert result["model"] == "gpt-5.6-sol"

    async def test_yaml_overrides_globals(self, db):
        """users.yaml settings override global defaults."""
        from kai.config import UserConfig

        uc = UserConfig(
            telegram_id=111,
            name="alice",
            model="opus",
            timeout=300,
        )
        config = self._make_config(user_configs={111: uc})
        result = await sessions.resolve_user_defaults(111, config)
        assert result["model"] == "opus"
        assert result["timeout"] == 300

    async def test_db_overrides_yaml(self, db):
        """DB settings take precedence over users.yaml."""
        from kai.config import UserConfig

        uc = UserConfig(
            telegram_id=111,
            name="alice",
            model="opus",
            timeout=300,
        )
        config = self._make_config(user_configs={111: uc})
        await sessions.set_user_setting(111, "model", "haiku")
        result = await sessions.resolve_user_defaults(111, config)
        assert result["model"] == "haiku"
        # Timeout not overridden in DB, comes from YAML
        assert result["timeout"] == 300

    async def test_partial_overrides(self, db):
        """Mix of DB, YAML, and global sources."""
        from kai.config import UserConfig

        uc = UserConfig(
            telegram_id=111,
            name="alice",
            timeout=300,  # YAML only
        )
        config = self._make_config(user_configs={111: uc})
        await sessions.set_user_setting(111, "model", "opus")  # DB only
        result = await sessions.resolve_user_defaults(111, config)
        assert result["model"] == "opus"  # from DB
        assert result["timeout"] == 300  # from YAML

    # ── Empty/blank model fallthrough (finding 1) ────────────────

    async def test_empty_string_model_falls_through(self, db):
        """Empty string model in DB falls through to config default."""
        config = self._make_config()
        await sessions.set_user_setting(111, "model", "")
        result = await sessions.resolve_user_defaults(111, config)
        assert result["model"] == "sonnet"

    async def test_whitespace_model_falls_through(self, db):
        """Whitespace-only model in DB falls through to config default."""
        config = self._make_config()
        await sessions.set_user_setting(111, "model", "  ")
        result = await sessions.resolve_user_defaults(111, config)
        assert result["model"] == "sonnet"

    async def test_empty_db_model_falls_through_to_yaml(self, db):
        """Empty string model in DB falls through to yaml, not config."""
        from kai.config import UserConfig

        uc = UserConfig(telegram_id=111, name="alice", model="opus")
        config = self._make_config(user_configs={111: uc})
        await sessions.set_user_setting(111, "model", "")
        result = await sessions.resolve_user_defaults(111, config)
        assert result["model"] == "opus"


# ── Workspace history ────────────────────────────────────────────────


class TestWorkspaceHistory:
    async def test_upsert_and_get(self, db):
        await sessions.upsert_workspace_history("/path/a", 12345)
        await sessions.upsert_workspace_history("/path/b", 12345)
        history = await sessions.get_workspace_history(12345)
        paths = [h["path"] for h in history]
        assert "/path/a" in paths
        assert "/path/b" in paths

    async def test_upsert_twice_no_duplicates(self, db):
        await sessions.upsert_workspace_history("/path/a", 12345)
        await sessions.upsert_workspace_history("/path/a", 12345)
        history = await sessions.get_workspace_history(12345)
        assert len(history) == 1

    async def test_delete_workspace_history(self, db):
        await sessions.upsert_workspace_history("/path/a", 12345)
        await sessions.delete_workspace_history("/path/a", 12345)
        history = await sessions.get_workspace_history(12345)
        assert len(history) == 0

    async def test_respects_limit(self, db):
        for i in range(5):
            await sessions.upsert_workspace_history(f"/path/{i}", 12345)
        history = await sessions.get_workspace_history(12345, limit=3)
        assert len(history) == 3


# ── Allowed workspaces ──────────────────────────────────────────────


class TestAllowedWorkspaces:
    """Tests for per-user allowed workspace CRUD."""

    async def test_add_and_retrieve(self, db):
        """Add a workspace and retrieve it."""
        await sessions.add_allowed_workspace(111, "/projects/repo-a")
        result = await sessions.get_allowed_workspaces(111)
        assert len(result) == 1
        assert result[0] == Path("/projects/repo-a")

    async def test_add_duplicate_ignored(self, db):
        """INSERT OR IGNORE prevents duplicate entries."""
        await sessions.add_allowed_workspace(111, "/projects/repo-a")
        await sessions.add_allowed_workspace(111, "/projects/repo-a")
        result = await sessions.get_allowed_workspaces(111)
        assert len(result) == 1

    async def test_remove_existing(self, db):
        """Remove returns True and deletes the entry."""
        await sessions.add_allowed_workspace(111, "/projects/repo-a")
        removed = await sessions.remove_allowed_workspace(111, "/projects/repo-a")
        assert removed is True
        result = await sessions.get_allowed_workspaces(111)
        assert len(result) == 0

    async def test_remove_not_found(self, db):
        """Remove returns False when path is not in the user's list."""
        removed = await sessions.remove_allowed_workspace(111, "/nonexistent")
        assert removed is False

    async def test_get_empty_list(self, db):
        """Returns empty list for a user with no allowed workspaces."""
        result = await sessions.get_allowed_workspaces(999)
        assert result == []

    async def test_insertion_order_preserved(self, db):
        """Paths are returned in insertion order (ORDER BY rowid)."""
        await sessions.add_allowed_workspace(111, "/projects/b")
        await sessions.add_allowed_workspace(111, "/projects/a")
        await sessions.add_allowed_workspace(111, "/projects/c")
        result = await sessions.get_allowed_workspaces(111)
        assert [str(p) for p in result] == [
            "/projects/b",
            "/projects/a",
            "/projects/c",
        ]

    async def test_user_isolation(self, db):
        """User A's entries are not visible to user B."""
        await sessions.add_allowed_workspace(111, "/projects/alice")
        await sessions.add_allowed_workspace(222, "/projects/bob")
        alice = await sessions.get_allowed_workspaces(111)
        bob = await sessions.get_allowed_workspaces(222)
        assert len(alice) == 1
        assert str(alice[0]) == "/projects/alice"
        assert len(bob) == 1
        assert str(bob[0]) == "/projects/bob"


# ── resolve_workspace_access ────────────────────────────────────────


class TestResolveWorkspaceAccess:
    """Tests for per-user workspace_base and allowed_workspaces resolution."""

    def _make_config(self, user_configs: dict | None = None, **kwargs):
        """Build a minimal Config with overridable defaults."""
        from kai.config import Config

        defaults = {
            "telegram_bot_token": "test",
            "allowed_user_ids": {111},
        }
        defaults.update(kwargs)
        if user_configs is not None:
            defaults["user_configs"] = user_configs
        return Config(**defaults)

    async def test_no_user_config_falls_back_to_global(self, db, tmp_path):
        """Without users.yaml, uses global workspace_base."""
        ws_base = tmp_path / "projects"
        ws_base.mkdir()
        config = self._make_config(workspace_base=ws_base)
        base, allowed = await sessions.resolve_workspace_access(111, config)
        assert base == ws_base
        assert allowed == []

    async def test_user_workspace_base_wins(self, db, tmp_path):
        """users.yaml workspace_base overrides global."""
        from kai.config import UserConfig

        global_base = tmp_path / "global"
        global_base.mkdir()
        user_base = tmp_path / "alice"
        user_base.mkdir()

        uc = UserConfig(
            telegram_id=111,
            name="alice",
            workspace_base=user_base,
        )
        config = self._make_config(
            user_configs={111: uc},
            workspace_base=global_base,
        )
        base, _allowed = await sessions.resolve_workspace_access(111, config)
        assert base == user_base

    async def test_no_workspace_base_returns_none(self, db):
        """Returns None when neither user nor global base is set."""
        config = self._make_config()
        base, _allowed = await sessions.resolve_workspace_access(111, config)
        assert base is None

    async def test_allowed_union_db_and_global(self, db, tmp_path):
        """Effective list is the union of DB entries and global config."""
        db_path = tmp_path / "db-repo"
        db_path.mkdir()
        global_path = tmp_path / "global-repo"
        global_path.mkdir()

        await sessions.add_allowed_workspace(111, str(db_path.resolve()))
        config = self._make_config(
            allowed_workspaces=[global_path.resolve()],
        )
        _base, allowed = await sessions.resolve_workspace_access(111, config)
        assert len(allowed) == 2
        assert db_path.resolve() in allowed
        assert global_path.resolve() in allowed

    async def test_allowed_db_only(self, db, tmp_path):
        """DB entries work without any global config."""
        db_path = tmp_path / "repo"
        db_path.mkdir()
        await sessions.add_allowed_workspace(111, str(db_path.resolve()))
        config = self._make_config()
        _base, allowed = await sessions.resolve_workspace_access(111, config)
        assert len(allowed) == 1
        assert allowed[0] == db_path.resolve()

    async def test_allowed_global_only(self, db, tmp_path):
        """Global entries work when user has no DB entries."""
        global_path = tmp_path / "repo"
        global_path.mkdir()
        config = self._make_config(
            allowed_workspaces=[global_path.resolve()],
        )
        _base, allowed = await sessions.resolve_workspace_access(111, config)
        assert len(allowed) == 1
        assert allowed[0] == global_path.resolve()

    async def test_allowed_dedup(self, db, tmp_path):
        """Same path in DB and global is counted once."""
        shared = tmp_path / "repo"
        shared.mkdir()
        resolved = shared.resolve()

        await sessions.add_allowed_workspace(111, str(resolved))
        config = self._make_config(
            allowed_workspaces=[resolved],
        )
        _base, allowed = await sessions.resolve_workspace_access(111, config)
        assert len(allowed) == 1

    async def test_db_entries_appear_first(self, db, tmp_path):
        """DB entries come before global entries in the combined list."""
        db_path = tmp_path / "db-repo"
        db_path.mkdir()
        global_path = tmp_path / "global-repo"
        global_path.mkdir()

        await sessions.add_allowed_workspace(111, str(db_path.resolve()))
        config = self._make_config(
            allowed_workspaces=[global_path.resolve()],
        )
        _base, allowed = await sessions.resolve_workspace_access(111, config)
        assert allowed[0] == db_path.resolve()
        assert allowed[1] == global_path.resolve()

    # -- Per-user yaml allowed_workspaces (issue #460) -------------------

    async def test_yaml_per_user_allowed_visible(self, db, tmp_path):
        """
        A path listed only under per-user `allowed_workspaces:` in
        users.yaml is in the combined list. Pre-#460, the field
        was silently dropped by the config loader and this list
        was always empty.
        """
        from kai.config import UserConfig

        yaml_path = tmp_path / "yaml-repo"
        yaml_path.mkdir()
        uc = UserConfig(
            telegram_id=111,
            name="alice",
            allowed_workspaces=[yaml_path.resolve()],
        )
        config = self._make_config(user_configs={111: uc})
        _base, allowed = await sessions.resolve_workspace_access(111, config)
        assert yaml_path.resolve() in allowed

    async def test_yaml_per_user_unioned_with_db_and_global(self, db, tmp_path):
        """
        Effective list is the union of all three tiers: DB,
        yaml-per-user, global. Each path appears exactly once.
        """
        from kai.config import UserConfig

        db_path = tmp_path / "db-repo"
        db_path.mkdir()
        yaml_path = tmp_path / "yaml-repo"
        yaml_path.mkdir()
        global_path = tmp_path / "global-repo"
        global_path.mkdir()

        await sessions.add_allowed_workspace(111, str(db_path.resolve()))
        uc = UserConfig(
            telegram_id=111,
            name="alice",
            allowed_workspaces=[yaml_path.resolve()],
        )
        config = self._make_config(
            user_configs={111: uc},
            allowed_workspaces=[global_path.resolve()],
        )
        _base, allowed = await sessions.resolve_workspace_access(111, config)
        assert len(allowed) == 3
        assert db_path.resolve() in allowed
        assert yaml_path.resolve() in allowed
        assert global_path.resolve() in allowed

    async def test_yaml_per_user_ordering_db_yaml_global(self, db, tmp_path):
        """
        The combined list orders entries DB > yaml-per-user > global.
        The keyboard and the /workspace allowed listing both depend
        on this ordering for the source-attribution labels.
        """
        from kai.config import UserConfig

        db_path = tmp_path / "db-repo"
        db_path.mkdir()
        yaml_path = tmp_path / "yaml-repo"
        yaml_path.mkdir()
        global_path = tmp_path / "global-repo"
        global_path.mkdir()

        await sessions.add_allowed_workspace(111, str(db_path.resolve()))
        uc = UserConfig(
            telegram_id=111,
            name="alice",
            allowed_workspaces=[yaml_path.resolve()],
        )
        config = self._make_config(
            user_configs={111: uc},
            allowed_workspaces=[global_path.resolve()],
        )
        _base, allowed = await sessions.resolve_workspace_access(111, config)
        assert allowed[0] == db_path.resolve()
        assert allowed[1] == yaml_path.resolve()
        assert allowed[2] == global_path.resolve()

    async def test_yaml_per_user_dedup_with_db(self, db, tmp_path):
        """
        Same path in both the DB and the per-user yaml list collapses
        to a single entry; the DB tier wins the slot (earlier in
        the iteration order).
        """
        from kai.config import UserConfig

        shared = tmp_path / "shared-repo"
        shared.mkdir()
        resolved = shared.resolve()

        await sessions.add_allowed_workspace(111, str(resolved))
        uc = UserConfig(
            telegram_id=111,
            name="alice",
            allowed_workspaces=[resolved],
        )
        config = self._make_config(user_configs={111: uc})
        _base, allowed = await sessions.resolve_workspace_access(111, config)
        assert len(allowed) == 1
        assert allowed[0] == resolved

    async def test_yaml_per_user_dedup_with_global(self, db, tmp_path):
        """
        Same path in both per-user yaml and global config collapses
        to a single entry; the per-user tier wins the slot
        (earlier in the iteration order).
        """
        from kai.config import UserConfig

        shared = tmp_path / "shared-repo"
        shared.mkdir()
        resolved = shared.resolve()

        uc = UserConfig(
            telegram_id=111,
            name="alice",
            allowed_workspaces=[resolved],
        )
        config = self._make_config(
            user_configs={111: uc},
            allowed_workspaces=[resolved],
        )
        _base, allowed = await sessions.resolve_workspace_access(111, config)
        assert len(allowed) == 1
        assert allowed[0] == resolved


# ── Effective repos ───────────────────────────────────────────────


class TestEffectiveRepos:
    """Tests for the union/minus repo computation."""

    async def test_yaml_only(self, db):
        """With no DB overrides, returns yaml repos lowercased."""
        result = await sessions.get_effective_repos(111, ["Owner/Repo"])
        assert result == ["owner/repo"]

    async def test_db_added(self, db):
        """DB-added repos are included in the effective list."""
        await sessions.set_github_added_repos(111, ["other/added"])
        result = await sessions.get_effective_repos(111, ["owner/repo"])
        assert "other/added" in result
        assert "owner/repo" in result

    async def test_db_removed(self, db):
        """DB-removed repos are excluded from the effective list."""
        await sessions.set_github_removed_repos(111, ["owner/repo"])
        result = await sessions.get_effective_repos(111, ["Owner/Repo"])
        assert result == []

    async def test_union_minus(self, db):
        """Full union/minus: yaml + added - removed."""
        await sessions.set_github_added_repos(111, ["extra/repo"])
        await sessions.set_github_removed_repos(111, ["yaml/removed"])
        result = await sessions.get_effective_repos(111, ["yaml/kept", "yaml/removed"])
        assert "yaml/kept" in result
        assert "extra/repo" in result
        assert "yaml/removed" not in result

    async def test_add_cancels_remove(self, db):
        """Adding a repo that was previously removed cancels the removal."""
        await sessions.set_github_removed_repos(111, ["owner/repo"])
        # Simulate the cancel-out: remove from removed list
        await sessions.set_github_removed_repos(111, [])
        await sessions.set_github_added_repos(111, ["owner/repo"])
        result = await sessions.get_effective_repos(111, [])
        assert result == ["owner/repo"]

    async def test_empty_everything(self, db):
        """No yaml repos and no DB entries returns empty list."""
        result = await sessions.get_effective_repos(111, [])
        assert result == []

    async def test_case_normalization(self, db):
        """All repos are lowercased for consistent matching."""
        await sessions.set_github_added_repos(111, ["Owner/UPPER"])
        result = await sessions.get_effective_repos(111, ["YAML/Mixed"])
        assert result == ["owner/upper", "yaml/mixed"]

    async def test_corrupt_added_repos(self, db):
        """Corrupt JSON in added repos returns empty list."""
        await sessions.set_setting("github_repos_added:111", "not json")
        result = await sessions.get_effective_repos(111, ["owner/repo"])
        assert result == ["owner/repo"]

    async def test_corrupt_removed_repos(self, db):
        """Corrupt JSON in removed repos returns empty list."""
        await sessions.set_setting("github_repos_removed:111", "{}")
        result = await sessions.get_effective_repos(111, ["owner/repo"])
        assert result == ["owner/repo"]


# ── resolve_github_settings ───────────────────────────────────────


class TestResolveGitHubSettings:
    """Tests for per-user GitHub notification settings resolution."""

    def _make_config(self, user_configs: dict | None = None, **kwargs):
        """Build a minimal Config with overridable defaults."""
        from kai.config import Config

        defaults = {
            "telegram_bot_token": "test",
            "allowed_user_ids": {111},
        }
        defaults.update(kwargs)
        if user_configs is not None:
            defaults["user_configs"] = user_configs
        return Config(**defaults)

    async def test_no_config_returns_defaults(self, db):
        """Without user config or DB overrides, returns hardcoded defaults."""
        config = self._make_config()
        result = await sessions.resolve_github_settings(111, config)
        assert result["repos"] == []
        assert result["notify_chat_id"] == 111  # falls back to telegram_id
        assert result["pr_review"] is False
        assert result["issue_triage"] is False

    async def test_yaml_overrides_defaults(self, db):
        """users.yaml values override hardcoded defaults."""
        from kai.config import UserConfig

        uc = UserConfig(
            telegram_id=111,
            name="alice",
            github_repos=["alice/repo-a"],
            github_notify_chat_id=-100999,
            pr_review=True,
            issue_triage=True,
        )
        config = self._make_config(user_configs={111: uc})
        result = await sessions.resolve_github_settings(111, config)
        # get_effective_repos lowercases all repos
        assert result["repos"] == ["alice/repo-a"]
        assert result["notify_chat_id"] == -100999
        assert result["pr_review"] is True
        assert result["issue_triage"] is True

    async def test_db_overrides_yaml(self, db):
        """DB settings override users.yaml values."""
        from kai.config import UserConfig

        uc = UserConfig(
            telegram_id=111,
            name="alice",
            pr_review=True,
            issue_triage=True,
            github_notify_chat_id=-100999,
        )
        config = self._make_config(user_configs={111: uc})

        # Set DB overrides that flip the yaml values
        await sessions.set_setting("pr_review:111", "false")
        await sessions.set_setting("issue_triage:111", "false")
        await sessions.set_setting("github_notify_chat:111", "-200888")

        result = await sessions.resolve_github_settings(111, config)
        assert result["pr_review"] is False
        assert result["issue_triage"] is False
        assert result["notify_chat_id"] == -200888

    async def test_db_partial_override_falls_back(self, db):
        """DB overrides one field; other fields take their own resolution."""
        from kai.config import UserConfig

        uc = UserConfig(
            telegram_id=111,
            name="alice",
            issue_triage=True,
        )
        config = self._make_config(user_configs={111: uc})
        await sessions.set_setting("pr_review:111", "false")
        result = await sessions.resolve_github_settings(111, config)
        assert result["pr_review"] is False  # DB override
        assert result["issue_triage"] is True  # users.yaml value

    async def test_notify_fallback_chain(self, db):
        """Notification destination: DB > yaml > telegram_id."""
        from kai.config import UserConfig

        # Level 3: no config at all - falls back to telegram_id
        config = self._make_config()
        result = await sessions.resolve_github_settings(111, config)
        assert result["notify_chat_id"] == 111

        # Level 2: users.yaml set
        uc = UserConfig(
            telegram_id=111,
            name="alice",
            github_notify_chat_id=-200,
        )
        config = self._make_config(user_configs={111: uc})
        result = await sessions.resolve_github_settings(111, config)
        assert result["notify_chat_id"] == -200

        # Level 1: DB set (overrides yaml)
        await sessions.set_setting("github_notify_chat:111", "-100")
        result = await sessions.resolve_github_settings(111, config)
        assert result["notify_chat_id"] == -100

    async def test_partial_overrides(self, db):
        """Some fields from DB, some from yaml."""
        from kai.config import UserConfig

        uc = UserConfig(
            telegram_id=111,
            name="alice",
            pr_review=True,
            issue_triage=True,
        )
        config = self._make_config(user_configs={111: uc})
        # Override only pr_review in DB
        await sessions.set_setting("pr_review:111", "false")

        result = await sessions.resolve_github_settings(111, config)
        assert result["pr_review"] is False  # DB override
        assert result["issue_triage"] is True  # users.yaml value

    async def test_pr_review_unset_defaults_false(self, db):
        """yaml pr_review=None resolves to False (no global fallback)."""
        from kai.config import UserConfig

        uc = UserConfig(
            telegram_id=111,
            name="alice",
            # pr_review not set (None)
        )
        config = self._make_config(user_configs={111: uc})
        result = await sessions.resolve_github_settings(111, config)
        assert result["pr_review"] is False

    async def test_repos_from_yaml_only(self, db):
        """Repos come from users.yaml, not DB (DB repos are #220)."""
        from kai.config import UserConfig

        uc = UserConfig(
            telegram_id=111,
            name="alice",
            github_repos=["alice/repo-a", "alice/repo-b"],
        )
        config = self._make_config(user_configs={111: uc})
        result = await sessions.resolve_github_settings(111, config)
        assert result["repos"] == ["alice/repo-a", "alice/repo-b"]

    # ── Corrupt DB notify value handling ──────────────────────────

    async def test_corrupt_db_notify_falls_through_to_yaml(self, db):
        """Non-numeric DB value falls through to users.yaml entry."""
        from kai.config import UserConfig

        uc = UserConfig(
            telegram_id=111,
            name="alice",
            github_notify_chat_id=-100999,
        )
        config = self._make_config(user_configs={111: uc})
        await sessions.set_setting("github_notify_chat:111", "not-a-number")

        result = await sessions.resolve_github_settings(111, config)
        assert result["notify_chat_id"] == -100999

    async def test_corrupt_db_notify_falls_through_to_telegram_id(self, db):
        """Non-numeric DB value falls through to user's own telegram_id."""
        config = self._make_config()
        await sessions.set_setting("github_notify_chat:111", "garbage")

        result = await sessions.resolve_github_settings(111, config)
        assert result["notify_chat_id"] == 111

    async def test_corrupt_db_notify_logs_warning(self, db, caplog):
        """Corrupt DB value emits a warning with the bad value."""
        import logging

        config = self._make_config()
        await sessions.set_setting("github_notify_chat:111", "xyz")

        with caplog.at_level(logging.WARNING, logger="kai.sessions"):
            await sessions.resolve_github_settings(111, config)

        assert any("Corrupt github_notify_chat" in r.message and "xyz" in r.message for r in caplog.records)

    async def test_empty_string_db_notify_falls_through(self, db):
        """Empty string in DB fails int() and falls through to telegram_id."""
        config = self._make_config()
        await sessions.set_setting("github_notify_chat:111", "")

        result = await sessions.resolve_github_settings(111, config)
        assert result["notify_chat_id"] == 111

    # ── Case-insensitive booleans (finding 5) ────────────────────

    async def test_pr_review_case_insensitive(self, db):
        """Mixed-case 'True' in DB resolves to True, not False."""
        config = self._make_config()
        await sessions.set_setting("pr_review:111", "True")
        result = await sessions.resolve_github_settings(111, config)
        assert result["pr_review"] is True

    async def test_issue_triage_case_insensitive(self, db):
        """Uppercase 'TRUE' in DB resolves to True, not False."""
        config = self._make_config()
        await sessions.set_setting("issue_triage:111", "TRUE")
        result = await sessions.resolve_github_settings(111, config)
        assert result["issue_triage"] is True

    async def test_pr_review_false_case_insensitive(self, db):
        """Mixed-case 'False' in DB resolves to False even when users.yaml says True."""
        from kai.config import UserConfig

        uc = UserConfig(telegram_id=111, name="alice", pr_review=True)
        config = self._make_config(user_configs={111: uc})
        await sessions.set_setting("pr_review:111", "False")
        result = await sessions.resolve_github_settings(111, config)
        assert result["pr_review"] is False

    async def test_issue_triage_false_case_insensitive(self, db):
        """Uppercase 'FALSE' in DB resolves to False even when users.yaml says True."""
        from kai.config import UserConfig

        uc = UserConfig(telegram_id=111, name="alice", issue_triage=True)
        config = self._make_config(user_configs={111: uc})
        await sessions.set_setting("issue_triage:111", "FALSE")
        result = await sessions.resolve_github_settings(111, config)
        assert result["issue_triage"] is False


# ── Scheduled job type migration ────────────────────────────────────


class TestJobTypeMigration:
    async def test_legacy_agent_jobs_are_migrated_without_data_loss(self, tmp_path):
        db_path = tmp_path / "legacy_jobs.db"
        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute("""
                CREATE TABLE jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    job_type TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    schedule_type TEXT NOT NULL,
                    schedule_data TEXT NOT NULL,
                    active INTEGER DEFAULT 1,
                    auto_remove INTEGER DEFAULT 0,
                    notify_on_check INTEGER DEFAULT 0
                )
            """)
            await conn.execute(
                "INSERT INTO jobs (chat_id, name, job_type, prompt, schedule_type, schedule_data) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (1, "legacy", "claude", "check status", "interval", '{"seconds": 60}'),
            )
            await conn.commit()

        try:
            await sessions.init_db(db_path)
            job = await sessions.get_job_by_id(1)
            assert job is not None
            assert job["job_type"] == "agent"
            assert job["name"] == "legacy"
            assert job["prompt"] == "check status"
        finally:
            await sessions.close_db()


# ── Workspace history migration ─────────────────────────────────────


class TestWorkspaceHistoryMigration:
    """Verify the workspace_history DDL migration runs atomically."""

    @pytest.mark.asyncio
    async def test_migration_adds_chat_id_column(self, tmp_path):
        """Old schema (path-only PK) migrates to composite PK with chat_id."""
        db_path = tmp_path / "test_migration.db"

        # Create old-schema table directly (path as sole PK, no chat_id)
        async with aiosqlite.connect(str(db_path)) as conn:
            await conn.execute("""
                CREATE TABLE workspace_history (
                    path TEXT PRIMARY KEY,
                    last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            await conn.execute(
                "INSERT INTO workspace_history (path) VALUES (?)",
                ("/old/workspace",),
            )
            # Also create the other tables init_db expects to CREATE IF NOT EXISTS
            await conn.execute("""
                CREATE TABLE sessions (
                    chat_id INTEGER PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    model TEXT DEFAULT 'sonnet'
                )
            """)
            await conn.execute("""
                CREATE TABLE jobs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    job_type TEXT NOT NULL DEFAULT 'reminder',
                    prompt TEXT NOT NULL,
                    schedule_type TEXT NOT NULL,
                    schedule_data TEXT NOT NULL,
                    active INTEGER DEFAULT 1,
                    auto_remove INTEGER DEFAULT 0,
                    notify_on_check INTEGER DEFAULT 0
                )
            """)
            await conn.execute("""
                CREATE TABLE settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                )
            """)
            await conn.commit()

        # Run init_db which should detect the missing chat_id column
        # and perform the atomic migration
        try:
            await sessions.init_db(db_path)
            # Verify schema: chat_id column exists
            async with sessions._get_db().execute("PRAGMA table_info(workspace_history)") as cursor:
                columns = [row[1] for row in await cursor.fetchall()]
            assert "chat_id" in columns

            # Verify data preserved with default chat_id=0
            async with sessions._get_db().execute("SELECT path, chat_id FROM workspace_history") as cursor:
                rows = await cursor.fetchall()
            assert len(rows) == 1
            assert rows[0][0] == "/old/workspace"
            assert rows[0][1] == 0
        finally:
            await sessions.close_db()


# ── init_db transactional safety ────────────────────────────────────


class TestInitDbTransaction:
    """Verify init_db wraps all DDL in a single atomic transaction."""

    @pytest.mark.asyncio
    async def test_fresh_db_file_is_owner_only(self, tmp_path):
        """SQLite files can hold GitHub PATs, so the DB must be 0600."""
        db_path = tmp_path / "fresh.db"
        await sessions.init_db(db_path)
        try:
            assert stat.S_IMODE(db_path.stat().st_mode) == 0o600
            for companion in (Path(f"{db_path}-wal"), Path(f"{db_path}-shm")):
                if companion.exists():
                    assert stat.S_IMODE(companion.stat().st_mode) == 0o600
        finally:
            await sessions.close_db()

    @pytest.mark.asyncio
    async def test_existing_db_file_mode_is_repaired(self, tmp_path):
        """Startup repairs an older permissive kai.db mode."""
        db_path = tmp_path / "existing.db"
        db_path.write_bytes(b"")
        db_path.chmod(0o644)

        await sessions.init_db(db_path)
        try:
            assert stat.S_IMODE(db_path.stat().st_mode) == 0o600
        finally:
            await sessions.close_db()

    @pytest.mark.asyncio
    async def test_fresh_db_creates_all_tables(self, tmp_path):
        """A fresh database gets the core tables in one transaction."""
        db_path = tmp_path / "fresh.db"
        await sessions.init_db(db_path)
        try:
            db = sessions._get_db()
            # Check representative core tables exist.
            cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = {row[0] for row in await cursor.fetchall()}
            assert "sessions" in tables
            assert "jobs" in tables
            assert "settings" in tables
            assert "telegram_update_queue" in tables
            assert "workspace_history" in tables
        finally:
            await sessions.close_db()

    @pytest.mark.asyncio
    async def test_idempotent_on_initialized_db(self, tmp_path):
        """Running init_db twice on the same database is a no-op."""
        db_path = tmp_path / "idempotent.db"
        await sessions.init_db(db_path)
        await sessions.close_db()

        # Second init should not raise
        await sessions.init_db(db_path)
        try:
            db = sessions._get_db()
            cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = {row[0] for row in await cursor.fetchall()}
            assert "sessions" in tables
            assert "jobs" in tables
            assert "settings" in tables
            assert "telegram_update_queue" in tables
            assert "workspace_history" in tables
        finally:
            await sessions.close_db()

    @pytest.mark.asyncio
    async def test_sqlite_ddl_rollback(self, tmp_path):
        """SQLite DDL inside BEGIN/ROLLBACK is fully undone.

        This verifies the core assumption init_db relies on: that CREATE TABLE
        inside an explicit transaction is rolled back atomically. Committed
        tables survive; uncommitted ones are removed.
        """
        db_path = tmp_path / "ddl_txn.db"

        async with aiosqlite.connect(str(db_path)) as conn:
            # Committed table survives rollback of later DDL
            await conn.execute("CREATE TABLE anchor (id INTEGER PRIMARY KEY)")
            await conn.commit()

            # This table is created inside a transaction, then rolled back
            await conn.execute("BEGIN IMMEDIATE")
            await conn.execute("CREATE TABLE should_not_exist (id INTEGER PRIMARY KEY)")
            await conn.execute("ROLLBACK")

        async with aiosqlite.connect(str(db_path)) as conn:
            cursor = await conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in await cursor.fetchall()}
            assert "anchor" in tables
            assert "should_not_exist" not in tables

    @pytest.mark.asyncio
    async def test_init_failure_closes_connection(self, tmp_path):
        """A failed init_db closes and nullifies the connection."""
        db_path = tmp_path / "fail.db"

        from unittest.mock import patch

        # Force a failure inside init_db by making the commit raise
        async def failing_commit(self):
            raise RuntimeError("Simulated commit failure")

        with (
            patch.object(aiosqlite.Connection, "commit", failing_commit),
            pytest.raises(RuntimeError, match="Simulated commit failure"),
        ):
            await sessions.init_db(db_path)

        # Connection should be closed and _db should be None
        assert sessions._db is None


# ── get_all_workspace_paths ─────────────────────────────────────────


class TestGetAllWorkspacePaths:
    @pytest.fixture(autouse=True)
    async def db(self, tmp_path):
        await sessions.init_db(tmp_path / "test.db")
        yield
        await sessions.close_db()

    @pytest.mark.asyncio
    async def test_returns_paths_from_multiple_users(self):
        """Paths from different users are all returned."""
        await sessions.upsert_workspace_history("/projects/alice", 111)
        await sessions.upsert_workspace_history("/projects/bob", 222)
        paths = await sessions.get_all_workspace_paths()
        assert "/projects/alice" in paths
        assert "/projects/bob" in paths

    @pytest.mark.asyncio
    async def test_deduplicates_paths(self):
        """Same path visited by two users appears once."""
        await sessions.upsert_workspace_history("/shared/project", 111)
        await sessions.upsert_workspace_history("/shared/project", 222)
        paths = await sessions.get_all_workspace_paths()
        assert paths.count("/shared/project") == 1

    @pytest.mark.asyncio
    async def test_respects_limit(self):
        """Returns at most 'limit' paths."""
        for i in range(10):
            await sessions.upsert_workspace_history(f"/projects/{i}", 111)
        paths = await sessions.get_all_workspace_paths(limit=3)
        assert len(paths) == 3

    @pytest.mark.asyncio
    async def test_empty_when_no_history(self):
        """Returns empty list when no workspace history exists."""
        paths = await sessions.get_all_workspace_paths()
        assert paths == []

    @pytest.mark.asyncio
    async def test_most_recent_first(self):
        """Paths are ordered by most recently used."""
        # Use explicit timestamps via raw SQL to guarantee ordering.
        # CURRENT_TIMESTAMP can be identical for rapid inserts within
        # the same second, making the ordering test non-deterministic.
        db = sessions._get_db()  # test-only access for timestamp control
        await db.execute(
            "INSERT OR REPLACE INTO workspace_history (path, chat_id, last_used_at) VALUES (?, ?, ?)",
            ("/old", 111, "2026-01-01 00:00:00"),
        )
        await db.execute(
            "INSERT OR REPLACE INTO workspace_history (path, chat_id, last_used_at) VALUES (?, ?, ?)",
            ("/new", 111, "2026-01-02 00:00:00"),
        )
        await db.commit()
        paths = await sessions.get_all_workspace_paths()
        assert paths.index("/new") < paths.index("/old")
