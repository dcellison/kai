"""Tests for sessions.py async database CRUD."""

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
