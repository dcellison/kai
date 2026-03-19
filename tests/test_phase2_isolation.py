"""
Tests for Phase 2 per-user data isolation.

Covers:
1. Settings namespacing (workspace setting per-user)
2. History filtering (get_recent_history with chat_id filter)
3. File storage (per-user subdirectories)
4. Workspace history (per-user rows, backfill migration)
5. Crash recovery (per-user flag directory)
"""

import json
from unittest.mock import patch

import pytest

from kai import sessions
from kai.bot import _clear_responding, _save_to_workspace, _set_responding
from kai.history import get_recent_history

# ── Settings namespacing ────────────────────────────────────────────


class TestSettingsNamespacing:
    @pytest.fixture(autouse=True)
    async def db(self, tmp_path):
        await sessions.init_db(tmp_path / "test.db")
        yield
        await sessions.close_db()

    @pytest.mark.asyncio
    async def test_workspace_setting_per_user(self):
        """Workspace setting is namespaced by chat_id."""
        await sessions.set_setting("workspace:111", "/home/alice/project")
        await sessions.set_setting("workspace:222", "/home/bob/project")

        assert await sessions.get_setting("workspace:111") == "/home/alice/project"
        assert await sessions.get_setting("workspace:222") == "/home/bob/project"

    @pytest.mark.asyncio
    async def test_workspace_setting_isolated(self):
        """User A's workspace is not visible to user B."""
        await sessions.set_setting("workspace:111", "/home/alice/project")

        # User B has no workspace set
        assert await sessions.get_setting("workspace:222") is None

    @pytest.mark.asyncio
    async def test_workspace_migration(self):
        """Old global workspace key is migrated to namespaced key."""
        # Simulate the old global setting
        await sessions.set_setting("workspace", "/old/workspace")

        # Migration: rename to namespaced key
        old = await sessions.get_setting("workspace")
        assert old == "/old/workspace"
        await sessions.set_setting("workspace:111", old)
        await sessions.delete_setting("workspace")

        # Old key is gone, new key has the value
        assert await sessions.get_setting("workspace") is None
        assert await sessions.get_setting("workspace:111") == "/old/workspace"

    @pytest.mark.asyncio
    async def test_workspace_migration_idempotent(self):
        """Running migration twice produces the same result."""
        await sessions.set_setting("workspace:111", "/already/migrated")

        # Second migration attempt: old key doesn't exist
        old = await sessions.get_setting("workspace")
        assert old is None  # nothing to migrate


# ── History filtering ───────────────────────────────────────────────


class TestHistoryFiltering:
    def _write_history(self, tmp_path, records):
        """Write JSONL records to a history file."""
        history_dir = tmp_path / ".claude" / "history"
        history_dir.mkdir(parents=True, exist_ok=True)
        path = history_dir / "2026-03-19.jsonl"
        lines = [json.dumps(r) for r in records]
        path.write_text("\n".join(lines))
        return history_dir

    def test_filtered_by_chat_id(self, tmp_path):
        """Only messages from the specified chat_id are returned."""
        records = [
            {"ts": "2026-03-19T10:00:00", "dir": "user", "chat_id": 111, "text": "Hello from Alice"},
            {"ts": "2026-03-19T10:01:00", "dir": "user", "chat_id": 222, "text": "Hello from Bob"},
            {"ts": "2026-03-19T10:02:00", "dir": "assistant", "chat_id": 111, "text": "Hi Alice"},
        ]
        history_dir = self._write_history(tmp_path, records)

        with patch("kai.history._LOG_DIR", history_dir):
            result = get_recent_history(chat_id=111)

        assert "Hello from Alice" in result
        assert "Hi Alice" in result
        assert "Hello from Bob" not in result

    def test_unfiltered_returns_all(self, tmp_path):
        """chat_id=None returns all messages (backward-compatible)."""
        records = [
            {"ts": "2026-03-19T10:00:00", "dir": "user", "chat_id": 111, "text": "Alice msg"},
            {"ts": "2026-03-19T10:01:00", "dir": "user", "chat_id": 222, "text": "Bob msg"},
        ]
        history_dir = self._write_history(tmp_path, records)

        with patch("kai.history._LOG_DIR", history_dir):
            result = get_recent_history(chat_id=None)

        assert "Alice msg" in result
        assert "Bob msg" in result

    def test_pre_migration_records_included(self, tmp_path):
        """Pre-Phase-2 records (no chat_id) are included for all users."""
        records = [
            {"ts": "2026-03-19T10:00:00", "dir": "user", "text": "Old message"},
            {"ts": "2026-03-19T10:01:00", "dir": "user", "chat_id": 111, "text": "Alice msg"},
        ]
        history_dir = self._write_history(tmp_path, records)

        # User 111 sees their own message AND the pre-migration record
        with patch("kai.history._LOG_DIR", history_dir):
            result = get_recent_history(chat_id=111)
        assert "Old message" in result
        assert "Alice msg" in result

        # User 999 also sees the pre-migration record (no way to attribute it)
        with patch("kai.history._LOG_DIR", history_dir):
            result = get_recent_history(chat_id=999)
        assert "Old message" in result
        assert "Alice msg" not in result

    def test_no_messages_for_user(self, tmp_path):
        """User with no history gets empty string."""
        records = [
            {"ts": "2026-03-19T10:00:00", "dir": "user", "chat_id": 111, "text": "Alice only"},
        ]
        history_dir = self._write_history(tmp_path, records)

        with patch("kai.history._LOG_DIR", history_dir):
            result = get_recent_history(chat_id=999)

        assert result == ""


# ── File storage ────────────────────────────────────────────────────


class TestFileStorage:
    def test_save_per_user(self, tmp_path):
        """File saved with user_id goes to per-user subdirectory."""
        saved = _save_to_workspace(b"hello", "test.txt", tmp_path, user_id=123)
        assert "/files/123/" in str(saved)
        assert saved.exists()
        assert saved.read_bytes() == b"hello"

    def test_save_no_user(self, tmp_path):
        """File saved without user_id goes to shared files/ directory."""
        saved = _save_to_workspace(b"hello", "test.txt", tmp_path)
        assert "/files/" in str(saved)
        assert "/files/123/" not in str(saved)

    def test_per_user_files_isolated(self, tmp_path):
        """User A's files are not in User B's directory."""
        _save_to_workspace(b"alice data", "a.txt", tmp_path, user_id=111)
        _save_to_workspace(b"bob data", "b.txt", tmp_path, user_id=222)

        alice_files = list((tmp_path / "files" / "111").iterdir())
        bob_files = list((tmp_path / "files" / "222").iterdir())

        assert len(alice_files) == 1
        assert len(bob_files) == 1
        assert alice_files[0].read_bytes() == b"alice data"
        assert bob_files[0].read_bytes() == b"bob data"


# ── Workspace history ───────────────────────────────────────────────


class TestWorkspaceHistoryPerUser:
    @pytest.fixture(autouse=True)
    async def db(self, tmp_path):
        await sessions.init_db(tmp_path / "test.db")
        yield
        await sessions.close_db()

    @pytest.mark.asyncio
    async def test_per_user_history(self):
        """Two users visit different workspaces. Each sees only their own."""
        await sessions.upsert_workspace_history("/projects/a", 111)
        await sessions.upsert_workspace_history("/projects/b", 222)

        alice = await sessions.get_workspace_history(111)
        bob = await sessions.get_workspace_history(222)

        assert len(alice) == 1
        assert alice[0]["path"] == "/projects/a"
        assert len(bob) == 1
        assert bob[0]["path"] == "/projects/b"

    @pytest.mark.asyncio
    async def test_same_path_different_users(self):
        """Same path can appear for multiple users."""
        await sessions.upsert_workspace_history("/shared/project", 111)
        await sessions.upsert_workspace_history("/shared/project", 222)

        alice = await sessions.get_workspace_history(111)
        bob = await sessions.get_workspace_history(222)

        assert len(alice) == 1
        assert len(bob) == 1

    @pytest.mark.asyncio
    async def test_delete_per_user(self):
        """Deleting user A's entry leaves user B's entry intact."""
        await sessions.upsert_workspace_history("/projects/shared", 111)
        await sessions.upsert_workspace_history("/projects/shared", 222)

        await sessions.delete_workspace_history("/projects/shared", 111)

        alice = await sessions.get_workspace_history(111)
        bob = await sessions.get_workspace_history(222)

        assert len(alice) == 0
        assert len(bob) == 1

    @pytest.mark.asyncio
    async def test_backfill(self):
        """Rows with chat_id=0 get assigned to the default user."""
        # Simulate pre-Phase-2 rows (chat_id=0 from ALTER TABLE default)
        db = sessions._get_db()
        await db.execute(
            "INSERT INTO workspace_history (path, chat_id, last_used_at) VALUES (?, 0, CURRENT_TIMESTAMP)",
            ("/old/workspace",),
        )
        await db.commit()

        await sessions.backfill_workspace_history(111)

        # Row should now belong to user 111
        history = await sessions.get_workspace_history(111)
        assert len(history) == 1
        assert history[0]["path"] == "/old/workspace"

        # No rows with chat_id=0 remain
        history_zero = await sessions.get_workspace_history(0)
        assert len(history_zero) == 0

    @pytest.mark.asyncio
    async def test_backfill_idempotent(self):
        """Running backfill twice produces the same result."""
        db = sessions._get_db()
        await db.execute(
            "INSERT INTO workspace_history (path, chat_id, last_used_at) VALUES (?, 0, CURRENT_TIMESTAMP)",
            ("/old/ws",),
        )
        await db.commit()

        await sessions.backfill_workspace_history(111)
        await sessions.backfill_workspace_history(111)  # second run

        history = await sessions.get_workspace_history(111)
        assert len(history) == 1


# ── Crash recovery ──────────────────────────────────────────────────


class TestCrashRecoveryPerUser:
    def test_multiple_users_in_flight(self, tmp_path):
        """Two users in-flight. Both flag files exist. Clear one; the other remains."""
        with patch("kai.bot._RESPONDING_DIR", tmp_path / ".responding"):
            _set_responding(111)
            _set_responding(222)

            assert (tmp_path / ".responding" / "111").exists()
            assert (tmp_path / ".responding" / "222").exists()

            _clear_responding(111)

            assert not (tmp_path / ".responding" / "111").exists()
            assert (tmp_path / ".responding" / "222").exists()

    def test_clear_noop_if_missing(self, tmp_path):
        """Clearing a non-existent flag is a no-op."""
        responding_dir = tmp_path / ".responding"
        responding_dir.mkdir()

        with patch("kai.bot._RESPONDING_DIR", responding_dir):
            _clear_responding(999)  # should not raise
