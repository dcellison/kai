"""
Tests for memory_backup.py - snapshotting the semantic memory corpus.

Covers:
1. Snapshot content: plain files and nested directories are copied,
   SQLite databases are copied consistently via the backup API, and
   WAL/SHM sidecars are excluded
2. Owner-only permissions on every snapshot directory and file
3. The freshness skip that stops service restarts from churning
   through the retention window
4. Retention pruning of the oldest snapshots, leaving foreign
   directories in the backup root untouched
5. Failure behavior: a failed run raises and leaves no partial or
   complete snapshot behind
6. Unreadable per-principal directories are skipped with a loud
   warning instead of silently or fatally
"""

import logging
import os
import sqlite3
import stat
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kai.memory_backup import (
    MIN_SNAPSHOT_AGE,
    RETENTION_SNAPSHOTS,
    run_memory_backup,
)

# A fixed "now" for deterministic snapshot names.
NOW = datetime(2026, 8, 18, 3, 0, 0, tzinfo=UTC)


def _make_memory_tree(root: Path) -> tuple[Path, sqlite3.Connection]:
    """
    Build a miniature DATA_DIR/memory tree shaped like production.

    Returns the memory dir and an OPEN WAL-mode connection to the
    embedded database, so tests exercise the backup against a live
    store the way the daemon holds it. Callers close the connection.
    """
    memory_dir = root / "memory"
    memory_dir.mkdir()
    (memory_dir / "MEMORY.md").write_text("# core\n")
    (memory_dir / "config.json").write_text("{}")
    principal = memory_dir / "prn_abc123"
    principal.mkdir()
    (principal / "MEMORY.md").write_text("# principal\n")
    qdrant = memory_dir / "qdrant" / "collection" / "kai_memory"
    qdrant.mkdir(parents=True)
    (memory_dir / "qdrant" / "meta.json").write_text('{"collections": {}}')
    # A real SQLite database standing in for storage.sqlite, in WAL
    # mode with the connection held open so live -wal/-shm sidecars
    # exist on disk during the backup, like the production daemon.
    conn = sqlite3.connect(qdrant / "storage.sqlite")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("CREATE TABLE points (id INTEGER PRIMARY KEY, payload TEXT)")
    conn.executemany("INSERT INTO points (payload) VALUES (?)", [("a",), ("b",), ("c",)])
    conn.commit()
    return memory_dir, conn


class TestSnapshot:
    def test_snapshot_copies_tree_and_databases(self, tmp_path):
        """The snapshot contains the whole tree; SQLite content survives a live copy."""
        memory_dir, conn = _make_memory_tree(tmp_path)
        backup_root = tmp_path / "backups" / "memory"

        snapshot = run_memory_backup(memory_dir, backup_root, NOW)
        conn.close()

        assert snapshot == backup_root / "20260818_030000"
        assert (snapshot / "MEMORY.md").read_text() == "# core\n"
        assert (snapshot / "prn_abc123" / "MEMORY.md").read_text() == "# principal\n"
        assert (snapshot / "qdrant" / "meta.json").exists()

        # The backed-up database must be a valid, complete SQLite file
        # even though the source connection was open in WAL mode.
        copied = snapshot / "qdrant" / "collection" / "kai_memory" / "storage.sqlite"
        rows = sqlite3.connect(copied).execute("SELECT payload FROM points ORDER BY id").fetchall()
        assert rows == [("a",), ("b",), ("c",)]

    def test_sidecars_are_not_copied(self, tmp_path):
        """Live -wal/-shm sidecars are excluded (their content is folded into the copy)."""
        memory_dir, conn = _make_memory_tree(tmp_path)
        db_dir = memory_dir / "qdrant" / "collection" / "kai_memory"
        assert (db_dir / "storage.sqlite-wal").exists(), "test setup must produce a live WAL"

        snapshot = run_memory_backup(memory_dir, tmp_path / "backups" / "memory", NOW)
        conn.close()

        assert snapshot is not None
        copied_dir = snapshot / "qdrant" / "collection" / "kai_memory"
        assert not (copied_dir / "storage.sqlite-wal").exists()
        assert not (copied_dir / "storage.sqlite-shm").exists()

    def test_snapshot_is_owner_only(self, tmp_path):
        """Every directory is 0700 and every file 0600, including the backup root."""
        memory_dir, conn = _make_memory_tree(tmp_path)
        backup_root = tmp_path / "backups" / "memory"

        snapshot = run_memory_backup(memory_dir, backup_root, NOW)
        conn.close()

        assert snapshot is not None
        assert stat.S_IMODE(backup_root.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(backup_root.stat().st_mode) == 0o700
        for path in [snapshot, *snapshot.rglob("*")]:
            expected = 0o700 if path.is_dir() else 0o600
            assert stat.S_IMODE(path.stat().st_mode) == expected, path


class TestFreshnessSkip:
    def test_skips_when_newest_snapshot_is_fresh(self, tmp_path):
        """A snapshot younger than MIN_SNAPSHOT_AGE suppresses the run."""
        memory_dir, conn = _make_memory_tree(tmp_path)
        conn.close()
        backup_root = tmp_path / "backups" / "memory"
        backup_root.mkdir(parents=True)
        fresh = NOW - MIN_SNAPSHOT_AGE / 2
        (backup_root / fresh.strftime("%Y%m%d_%H%M%S")).mkdir()

        assert run_memory_backup(memory_dir, backup_root, NOW) is None
        assert len(list(backup_root.iterdir())) == 1

    def test_runs_when_newest_snapshot_is_stale(self, tmp_path):
        """A snapshot older than MIN_SNAPSHOT_AGE does not suppress the run."""
        memory_dir, conn = _make_memory_tree(tmp_path)
        conn.close()
        backup_root = tmp_path / "backups" / "memory"
        backup_root.mkdir(parents=True)
        stale = NOW - MIN_SNAPSHOT_AGE * 2
        (backup_root / stale.strftime("%Y%m%d_%H%M%S")).mkdir()

        assert run_memory_backup(memory_dir, backup_root, NOW) is not None


class TestRetention:
    def test_prunes_oldest_beyond_retention(self, tmp_path):
        """Only the newest RETENTION_SNAPSHOTS dated directories survive."""
        memory_dir, conn = _make_memory_tree(tmp_path)
        conn.close()
        backup_root = tmp_path / "backups" / "memory"
        backup_root.mkdir(parents=True)
        # More old snapshots than the retention window holds, all stale
        # enough not to trigger the freshness skip.
        old_names = [f"202608{day:02d}_010000" for day in range(1, 10)]
        for name in old_names:
            (backup_root / name).mkdir()
        # A foreign directory must never be a pruning candidate.
        (backup_root / "keep-me").mkdir()

        snapshot = run_memory_backup(memory_dir, backup_root, NOW)

        assert snapshot is not None
        survivors = sorted(p.name for p in backup_root.iterdir() if p.name != "keep-me")
        assert len(survivors) == RETENTION_SNAPSHOTS
        # The new snapshot is the newest; the oldest fakes are gone.
        assert survivors[-1] == "20260818_030000"
        assert old_names[0] not in survivors
        assert (backup_root / "keep-me").exists()


class TestFailure:
    def test_failed_run_raises_and_leaves_no_snapshot(self, tmp_path):
        """A copy failure propagates and removes the partial directory."""
        memory_dir = tmp_path / "memory"
        memory_dir.mkdir()
        # Garbage bytes under a database suffix make the sqlite backup
        # path fail when the backup API tries to read pages.
        (memory_dir / "broken.db").write_bytes(b"this is not a sqlite database")
        backup_root = tmp_path / "backups" / "memory"

        with pytest.raises(sqlite3.Error):
            run_memory_backup(memory_dir, backup_root, NOW)

        assert list(backup_root.iterdir()) == []

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores directory permissions")
    def test_unreadable_directory_warns_but_snapshot_succeeds(self, tmp_path, caplog):
        """Foreign-owned 0700 principal dirs are reported loudly, not silently skipped."""
        memory_dir, conn = _make_memory_tree(tmp_path)
        conn.close()
        locked = memory_dir / "prn_locked"
        locked.mkdir()
        (locked / "MEMORY.md").write_text("secret")
        os.chmod(locked, 0o000)

        try:
            with caplog.at_level(logging.WARNING, logger="kai.memory_backup"):
                snapshot = run_memory_backup(memory_dir, tmp_path / "backups" / "memory", NOW)
        finally:
            os.chmod(locked, 0o700)

        assert snapshot is not None
        assert not (snapshot / "prn_locked" / "MEMORY.md").exists()
        assert any("unreadable" in r.message and "prn_locked" in r.message for r in caplog.records)
