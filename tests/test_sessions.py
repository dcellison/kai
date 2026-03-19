"""Tests for sessions.py async database CRUD."""

import pytest

from kai import sessions

_TEST_USER_ID = 12345
_TEST_CHAT_ID = 12345


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
        await sessions.save_session(_TEST_USER_ID, _TEST_CHAT_ID, "sess-abc", "sonnet", 0.5)
        result = await sessions.get_session(_TEST_USER_ID)
        assert result == "sess-abc"

    async def test_save_twice_accumulates_cost(self, db):
        await sessions.save_session(_TEST_USER_ID, _TEST_CHAT_ID, "sess-1", "sonnet", 0.5)
        await sessions.save_session(_TEST_USER_ID, _TEST_CHAT_ID, "sess-1", "sonnet", 0.3)
        stats = await sessions.get_stats(_TEST_USER_ID)
        assert stats["total_cost_usd"] == pytest.approx(0.8)

    async def test_clear_session(self, db):
        await sessions.save_session(_TEST_USER_ID, _TEST_CHAT_ID, "sess-1", "sonnet", 0.0)
        await sessions.clear_session(_TEST_USER_ID)
        assert await sessions.get_session(_TEST_USER_ID) is None

    async def test_get_stats(self, db):
        await sessions.save_session(_TEST_USER_ID, _TEST_CHAT_ID, "sess-1", "opus", 1.23)
        stats = await sessions.get_stats(_TEST_USER_ID)
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
            user_id=_TEST_USER_ID,
            chat_id=_TEST_CHAT_ID,
            name="test",
            job_type="reminder",
            prompt="hello",
            schedule_type="once",
            schedule_data='{"run_at": "2026-12-01T00:00:00"}',
        )
        assert isinstance(job_id, int)

    async def test_get_jobs_returns_active(self, db):
        await sessions.create_job(
            user_id=_TEST_USER_ID,
            chat_id=_TEST_CHAT_ID,
            name="j1",
            job_type="reminder",
            prompt="p1",
            schedule_type="once",
            schedule_data="{}",
        )
        jobs = await sessions.get_jobs(_TEST_USER_ID)
        assert len(jobs) == 1
        assert jobs[0]["name"] == "j1"

    async def test_get_jobs_filters_by_user(self, db):
        await sessions.create_job(
            user_id=_TEST_USER_ID,
            chat_id=_TEST_CHAT_ID,
            name="j1",
            job_type="reminder",
            prompt="p",
            schedule_type="once",
            schedule_data="{}",
        )
        await sessions.create_job(
            user_id=99999,
            chat_id=99999,
            name="j2",
            job_type="reminder",
            prompt="p",
            schedule_type="once",
            schedule_data="{}",
        )
        assert len(await sessions.get_jobs(_TEST_USER_ID)) == 1
        assert len(await sessions.get_jobs(99999)) == 1

    async def test_get_job_by_id(self, db):
        job_id = await sessions.create_job(
            user_id=_TEST_USER_ID,
            chat_id=_TEST_CHAT_ID,
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
            user_id=_TEST_USER_ID,
            chat_id=_TEST_CHAT_ID,
            name="j1",
            job_type="reminder",
            prompt="p",
            schedule_type="once",
            schedule_data="{}",
        )
        await sessions.create_job(
            user_id=99999,
            chat_id=99999,
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
            user_id=_TEST_USER_ID,
            chat_id=_TEST_CHAT_ID,
            name="j1",
            job_type="reminder",
            prompt="p",
            schedule_type="once",
            schedule_data="{}",
        )
        await sessions.deactivate_job(job_id)
        assert len(await sessions.get_jobs(_TEST_USER_ID)) == 0
        assert len(await sessions.get_all_active_jobs()) == 0

    async def test_deactivate_job_wrong_user_no_op(self, db):
        """deactivate_job with wrong user_id does not deactivate another user's job."""
        job_id = await sessions.create_job(
            user_id=_TEST_USER_ID,
            chat_id=_TEST_CHAT_ID,
            name="j1",
            job_type="reminder",
            prompt="p",
            schedule_type="once",
            schedule_data="{}",
        )
        await sessions.deactivate_job(job_id, user_id=99999)
        # Job should still be active — wrong user cannot deactivate it
        jobs = await sessions.get_jobs(_TEST_USER_ID)
        assert len(jobs) == 1
        assert jobs[0]["id"] == job_id

    async def test_delete_job(self, db):
        job_id = await sessions.create_job(
            user_id=_TEST_USER_ID,
            chat_id=_TEST_CHAT_ID,
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
            user_id=_TEST_USER_ID,
            chat_id=_TEST_CHAT_ID,
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
            user_id=_TEST_USER_ID,
            chat_id=_TEST_CHAT_ID,
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
            user_id=_TEST_USER_ID,
            chat_id=_TEST_CHAT_ID,
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
            user_id=_TEST_USER_ID,
            chat_id=_TEST_CHAT_ID,
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
            user_id=_TEST_USER_ID,
            chat_id=_TEST_CHAT_ID,
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
            user_id=_TEST_USER_ID,
            chat_id=_TEST_CHAT_ID,
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
        assert await sessions.get_setting(_TEST_USER_ID, "nonexistent") is None

    async def test_set_then_get(self, db):
        await sessions.set_setting(_TEST_USER_ID, "theme", "dark")
        assert await sessions.get_setting(_TEST_USER_ID, "theme") == "dark"

    async def test_set_overwrites(self, db):
        await sessions.set_setting(_TEST_USER_ID, "theme", "dark")
        await sessions.set_setting(_TEST_USER_ID, "theme", "light")
        assert await sessions.get_setting(_TEST_USER_ID, "theme") == "light"

    async def test_delete_setting(self, db):
        await sessions.set_setting(_TEST_USER_ID, "key", "val")
        await sessions.delete_setting(_TEST_USER_ID, "key")
        assert await sessions.get_setting(_TEST_USER_ID, "key") is None


# ── Workspace history ────────────────────────────────────────────────


class TestWorkspaceHistory:
    async def test_upsert_and_get(self, db):
        await sessions.upsert_workspace_history(_TEST_USER_ID, "/path/a")
        await sessions.upsert_workspace_history(_TEST_USER_ID, "/path/b")
        history = await sessions.get_workspace_history(_TEST_USER_ID)
        paths = [h["path"] for h in history]
        assert "/path/a" in paths
        assert "/path/b" in paths

    async def test_upsert_twice_no_duplicates(self, db):
        await sessions.upsert_workspace_history(_TEST_USER_ID, "/path/a")
        await sessions.upsert_workspace_history(_TEST_USER_ID, "/path/a")
        history = await sessions.get_workspace_history(_TEST_USER_ID)
        assert len(history) == 1

    async def test_delete_workspace_history(self, db):
        await sessions.upsert_workspace_history(_TEST_USER_ID, "/path/a")
        await sessions.delete_workspace_history(_TEST_USER_ID, "/path/a")
        history = await sessions.get_workspace_history(_TEST_USER_ID)
        assert len(history) == 0

    async def test_respects_limit(self, db):
        for i in range(5):
            await sessions.upsert_workspace_history(_TEST_USER_ID, f"/path/{i}")
        history = await sessions.get_workspace_history(_TEST_USER_ID, limit=3)
        assert len(history) == 3
