"""Tests for sessions.py async database CRUD."""

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


# ── Workspace history ────────────────────────────────────────────────


class TestWorkspaceHistory:
    async def test_upsert_and_get(self, db):
        await sessions.upsert_workspace_history("/path/a")
        await sessions.upsert_workspace_history("/path/b")
        history = await sessions.get_workspace_history()
        paths = [h["path"] for h in history]
        assert "/path/a" in paths
        assert "/path/b" in paths

    async def test_upsert_twice_no_duplicates(self, db):
        await sessions.upsert_workspace_history("/path/a")
        await sessions.upsert_workspace_history("/path/a")
        history = await sessions.get_workspace_history()
        assert len(history) == 1

    async def test_delete_workspace_history(self, db):
        await sessions.upsert_workspace_history("/path/a")
        await sessions.delete_workspace_history("/path/a")
        history = await sessions.get_workspace_history()
        assert len(history) == 0

    async def test_respects_limit(self, db):
        for i in range(5):
            await sessions.upsert_workspace_history(f"/path/{i}")
        history = await sessions.get_workspace_history(limit=3)
        assert len(history) == 3
