"""Tests for sessions.py async database CRUD."""

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
        await sessions.save_session(1, "sess-abc", "sonnet", 0.5)
        result = await sessions.get_session(1)
        assert result == "sess-abc"

    async def test_save_twice_accumulates_cost(self, db):
        await sessions.save_session(1, "sess-1", "sonnet", 0.5)
        await sessions.save_session(1, "sess-1", "sonnet", 0.3)
        stats = await sessions.get_stats(1)
        assert stats["total_cost_usd"] == pytest.approx(0.8)

    async def test_clear_session(self, db):
        await sessions.save_session(1, "sess-1", "sonnet", 0.0)
        await sessions.clear_session(1)
        assert await sessions.get_session(1) is None

    async def test_get_stats(self, db):
        await sessions.save_session(1, "sess-1", "opus", 1.23)
        stats = await sessions.get_stats(1)
        assert stats["session_id"] == "sess-1"
        assert stats["model"] == "opus"
        assert stats["total_cost_usd"] == pytest.approx(1.23)
        assert "created_at" in stats
        assert "last_used_at" in stats

    async def test_get_stats_unknown(self, db):
        assert await sessions.get_stats(999) is None


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
            job_type="claude",
            prompt="analyze",
            schedule_type="daily",
            schedule_data='{"times": ["09:00"]}',
        )
        job = await sessions.get_job_by_id(job_id)
        assert job is not None
        assert job["name"] == "j1"
        assert job["job_type"] == "claude"

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
            job_type="claude",
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
            job_type="claude",
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
        await sessions.set_workspace_config_setting(111, "/projects/kai", "budget", "20.0")
        await sessions.set_workspace_config_setting(111, "/projects/kai", "timeout", "300")
        result = await sessions.get_workspace_config_settings(111, "/projects/kai")
        assert result == {"model": "opus", "budget": "20.0", "timeout": "300"}

    async def test_delete_single_field(self, db):
        """Deleting one field leaves others intact."""
        await sessions.set_workspace_config_setting(111, "/projects/kai", "model", "opus")
        await sessions.set_workspace_config_setting(111, "/projects/kai", "budget", "20.0")
        await sessions.delete_workspace_config_setting(111, "/projects/kai", "model")
        result = await sessions.get_workspace_config_settings(111, "/projects/kai")
        assert result == {"budget": "20.0"}

    async def test_delete_all(self, db):
        """Bulk delete removes all overrides for a workspace."""
        await sessions.set_workspace_config_setting(111, "/projects/kai", "model", "opus")
        await sessions.set_workspace_config_setting(111, "/projects/kai", "budget", "20.0")
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

        yaml = WorkspaceConfig(path=Path("/projects/kai"), model="opus", budget=15.0)
        result = await sessions.build_workspace_config(yaml, Path("/projects/kai"), 111)
        assert result is not None
        assert result.model == "opus"
        assert result.budget == 15.0

    async def test_db_only(self, db):
        """No YAML config, DB overrides present, returns DB values."""
        await sessions.set_workspace_config_setting(111, "/projects/kai", "model", "haiku")
        await sessions.set_workspace_config_setting(111, "/projects/kai", "budget", "5.0")
        result = await sessions.build_workspace_config(None, Path("/projects/kai"), 111)
        assert result is not None
        assert result.model == "haiku"
        assert result.budget == 5.0
        assert result.path == Path("/projects/kai")

    async def test_db_overrides_yaml(self, db):
        """DB values take precedence over YAML values."""
        from kai.config import WorkspaceConfig

        yaml = WorkspaceConfig(path=Path("/projects/kai"), model="opus", budget=15.0)
        await sessions.set_workspace_config_setting(111, "/projects/kai", "model", "sonnet")
        result = await sessions.build_workspace_config(yaml, Path("/projects/kai"), 111)
        assert result is not None
        assert result.model == "sonnet"
        # Budget from YAML is preserved (not overridden)
        assert result.budget == 15.0

    async def test_partial_override(self, db):
        """YAML has model+budget, DB overrides only model. Budget from YAML."""
        from kai.config import WorkspaceConfig

        yaml = WorkspaceConfig(path=Path("/projects/kai"), model="opus", budget=20.0, timeout=300)
        await sessions.set_workspace_config_setting(111, "/projects/kai", "model", "haiku")
        result = await sessions.build_workspace_config(yaml, Path("/projects/kai"), 111)
        assert result is not None
        assert result.model == "haiku"
        assert result.budget == 20.0
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
        await sessions.set_user_setting(111, "budget", "15.0")
        await sessions.set_user_setting(111, "timeout", "300")
        result = await sessions.get_user_settings(111)
        assert result == {"model": "opus", "budget": "15.0", "timeout": "300"}

    async def test_delete_single(self, db):
        """Deleting one field leaves others intact."""
        await sessions.set_user_setting(111, "model", "opus")
        await sessions.set_user_setting(111, "budget", "15.0")
        await sessions.delete_user_setting(111, "model")
        result = await sessions.get_user_settings(111)
        assert result == {"budget": "15.0"}

    async def test_delete_all(self, db):
        """Bulk delete removes all per-user settings."""
        await sessions.set_user_setting(111, "model", "opus")
        await sessions.set_user_setting(111, "budget", "15.0")
        await sessions.set_user_setting(111, "timeout", "300")
        await sessions.set_user_setting(111, "context_window", "200000")
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

    def _make_config(self, user_configs=None, **kwargs):
        """Build a minimal Config with overridable defaults."""
        from kai.config import Config

        defaults = {
            "telegram_bot_token": "test",
            "allowed_user_ids": {111},
            "claude_model": "sonnet",
            "claude_max_budget_usd": 10.0,
            "claude_timeout_seconds": 120,
            "claude_max_context_window": 0,
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
        assert result["budget"] == 10.0
        assert result["timeout"] == 120
        assert result["context_window"] == 0

    async def test_db_overrides_globals(self, db):
        """DB settings override global defaults."""
        config = self._make_config()
        await sessions.set_user_setting(111, "model", "opus")
        await sessions.set_user_setting(111, "budget", "25.0")
        result = await sessions.resolve_user_defaults(111, config)
        assert result["model"] == "opus"
        assert result["budget"] == 25.0
        # Unset fields still come from globals
        assert result["timeout"] == 120
        assert result["context_window"] == 0

    async def test_yaml_overrides_globals(self, db):
        """users.yaml settings override global defaults."""
        from kai.config import UserConfig

        uc = UserConfig(
            telegram_id=111,
            name="alice",
            model="opus",
            max_budget=20.0,
            timeout=300,
            context_window=200_000,
        )
        config = self._make_config(user_configs={111: uc})
        result = await sessions.resolve_user_defaults(111, config)
        assert result["model"] == "opus"
        assert result["budget"] == 20.0
        assert result["timeout"] == 300
        assert result["context_window"] == 200_000

    async def test_db_overrides_yaml(self, db):
        """DB settings take precedence over users.yaml."""
        from kai.config import UserConfig

        uc = UserConfig(
            telegram_id=111,
            name="alice",
            model="opus",
            max_budget=20.0,
            timeout=300,
        )
        config = self._make_config(user_configs={111: uc})
        await sessions.set_user_setting(111, "model", "haiku")
        await sessions.set_user_setting(111, "budget", "5.0")
        result = await sessions.resolve_user_defaults(111, config)
        assert result["model"] == "haiku"
        assert result["budget"] == 5.0
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
        assert result["budget"] == 10.0  # from global
        assert result["context_window"] == 0  # from global


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

    def _make_config(self, user_configs=None, **kwargs):
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


# ── resolve_github_settings ───────────────────────────────────────


class TestResolveGitHubSettings:
    """Tests for per-user GitHub notification settings resolution."""

    def _make_config(self, user_configs=None, **kwargs):
        """Build a minimal Config with overridable defaults."""
        from kai.config import Config

        defaults = {
            "telegram_bot_token": "test",
            "allowed_user_ids": {111},
            "pr_review_enabled": False,
            "issue_triage_enabled": False,
        }
        defaults.update(kwargs)
        if user_configs is not None:
            defaults["user_configs"] = user_configs
        return Config(**defaults)

    async def test_no_config_returns_global_defaults(self, db):
        """Without user config or DB overrides, returns global defaults."""
        config = self._make_config()
        result = await sessions.resolve_github_settings(111, config)
        assert result["repos"] == []
        assert result["notify_chat_id"] == 111  # falls back to telegram_id
        assert result["pr_review"] is False
        assert result["issue_triage"] is False

    async def test_yaml_overrides_globals(self, db):
        """users.yaml values override global defaults."""
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

    async def test_db_overrides_env(self, db):
        """DB settings override env var global defaults."""
        config = self._make_config(
            pr_review_enabled=True,
            issue_triage_enabled=True,
        )
        await sessions.set_setting("pr_review:111", "false")
        result = await sessions.resolve_github_settings(111, config)
        assert result["pr_review"] is False
        # issue_triage not in DB, falls through to env
        assert result["issue_triage"] is True

    async def test_notify_fallback_chain(self, db):
        """Notification destination: DB > yaml > env > telegram_id."""
        from kai.config import UserConfig

        # Level 4: no config at all - falls back to telegram_id
        config = self._make_config()
        result = await sessions.resolve_github_settings(111, config)
        assert result["notify_chat_id"] == 111

        # Level 3: global env var set
        config = self._make_config(github_notify_chat_id=-300)
        result = await sessions.resolve_github_settings(111, config)
        assert result["notify_chat_id"] == -300

        # Level 2: yaml set (overrides env)
        uc = UserConfig(
            telegram_id=111,
            name="alice",
            github_notify_chat_id=-200,
        )
        config = self._make_config(
            user_configs={111: uc},
            github_notify_chat_id=-300,
        )
        result = await sessions.resolve_github_settings(111, config)
        assert result["notify_chat_id"] == -200

        # Level 1: DB set (overrides yaml)
        await sessions.set_setting("github_notify_chat:111", "-100")
        result = await sessions.resolve_github_settings(111, config)
        assert result["notify_chat_id"] == -100

    async def test_partial_overrides(self, db):
        """Some fields from DB, some from yaml, some from globals."""
        from kai.config import UserConfig

        uc = UserConfig(
            telegram_id=111,
            name="alice",
            pr_review=True,
            # issue_triage omitted (None) - falls to global
        )
        config = self._make_config(
            user_configs={111: uc},
            issue_triage_enabled=True,
        )
        # Override only pr_review in DB
        await sessions.set_setting("pr_review:111", "false")

        result = await sessions.resolve_github_settings(111, config)
        assert result["pr_review"] is False  # DB override
        assert result["issue_triage"] is True  # global default

    async def test_pr_review_none_uses_global(self, db):
        """yaml pr_review=None falls through to global env var."""
        from kai.config import UserConfig

        uc = UserConfig(
            telegram_id=111,
            name="alice",
            # pr_review not set (None)
        )
        config = self._make_config(
            user_configs={111: uc},
            pr_review_enabled=True,
        )
        result = await sessions.resolve_github_settings(111, config)
        assert result["pr_review"] is True

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
                    model TEXT DEFAULT 'sonnet',
                    total_cost REAL DEFAULT 0.0
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
    async def test_fresh_db_creates_all_tables(self, tmp_path):
        """A fresh database gets all four tables in one transaction."""
        db_path = tmp_path / "fresh.db"
        await sessions.init_db(db_path)
        try:
            db = sessions._get_db()
            # Check all four tables exist
            cursor = await db.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
            tables = {row[0] for row in await cursor.fetchall()}
            assert "sessions" in tables
            assert "jobs" in tables
            assert "settings" in tables
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
