"""
Nightly on-disk snapshots of the semantic memory corpus.

Provides functionality to:
1. Snapshot the entire DATA_DIR/memory/ tree (embedded Qdrant vector
   store, Mem0 history DB, per-principal MEMORY.md fallback dirs) into
   a dated directory under DATA_DIR/backups/memory/
2. Copy SQLite databases through the sqlite3 backup API so a snapshot
   taken while the daemon is writing is still transactionally
   consistent (a plain file copy of a live database risks torn state)
3. Skip a run when a recent snapshot already exists, so service
   restarts don't churn through the retention window
4. Enforce a fixed retention window by pruning the oldest snapshots
5. Restrict every snapshot directory and file to owner-only permissions

The snapshot runs in-process (scheduled from main.py) because embedded
local-mode Qdrant is single-process and the daemon holds it for its
lifetime; an external job could never coordinate with the writer. The
SQLite backup API makes quiescing unnecessary: it takes a consistent
read of the source database even while other connections write.

Restore procedure (exercised end to end against a scratch DATA_DIR;
see PR for the transcript):

1. Stop the service so nothing holds the embedded Qdrant store:
   macOS:  sudo launchctl bootout system/com.syrinx.kai
   Linux:  sudo systemctl stop kai
2. Move the damaged store aside (do not delete it):
   mv "$DATA_DIR/memory" "$DATA_DIR/memory.damaged"
3. Copy the chosen snapshot back into place, preserving permissions:
   cp -Rp "$DATA_DIR/backups/memory/<YYYYMMDD_HHMMSS>" "$DATA_DIR/memory"
4. If the copy ran as root, restore ownership to the service user:
   chown -R <service_user> "$DATA_DIR/memory"
5. Start the service again:
   macOS:  sudo launchctl bootstrap system /Library/LaunchDaemons/com.syrinx.kai.plist
   Linux:  sudo systemctl start kai
6. Verify reads: the /memory dashboard should list facts, or
   `/api/memory/search` should return results.

Note the restored tree carries the snapshot's owner-only permissions,
which is tighter than the historical live-store permissions and safe
to keep: only the service user reads the store.
"""

import logging
import os
import re
import shutil
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

log = logging.getLogger(__name__)

# ── Policy constants ─────────────────────────────────────────────────

# How many dated snapshots to keep. Pruning runs after each successful
# snapshot and deletes the oldest beyond this count.
RETENTION_SNAPSHOTS = 7

# Skip a run when the newest snapshot is younger than this. The backup
# loop ticks once per day but restarts its clock on service restart;
# without this floor, a day with several deploys would write several
# snapshots and rotate real history out of the retention window.
MIN_SNAPSHOT_AGE = timedelta(hours=20)

# Snapshot directory names are UTC timestamps in this format. The
# retention scan only considers names matching this shape, so foreign
# files in the backup root are never deleted.
_SNAPSHOT_NAME_FORMAT = "%Y%m%d_%H%M%S"
_SNAPSHOT_NAME_RE = re.compile(r"^\d{8}_\d{6}$")

# SQLite databases get the backup-API treatment; their WAL/SHM/journal
# sidecars are skipped entirely because the backup API folds pending
# WAL content into the copied main database, and a plain copy of a
# live sidecar would be torn anyway.
_SQLITE_SUFFIXES = frozenset({".db", ".sqlite", ".sqlite3"})
_SQLITE_SIDECAR_ENDINGS = ("-wal", "-shm", "-journal")


# ── Snapshot ─────────────────────────────────────────────────────────


def _is_sqlite_sidecar(name: str) -> bool:
    """Check if a filename is a SQLite WAL/SHM/journal sidecar."""
    return name.endswith(_SQLITE_SIDECAR_ENDINGS)


def _backup_sqlite(source: Path, target: Path) -> None:
    """
    Copy one SQLite database via the sqlite3 backup API.

    The source is opened read-only (uri=True) so a backup can never
    mutate the live store. backup() with default arguments copies the
    whole database in one pass under a read lock, yielding a
    transactionally consistent target even if the daemon commits
    writes concurrently.
    """
    src_conn = sqlite3.connect(f"file:{source}?mode=ro", uri=True)
    try:
        dst_conn = sqlite3.connect(target)
        try:
            src_conn.backup(dst_conn)
        finally:
            dst_conn.close()
    finally:
        src_conn.close()


def _snapshot_tree(memory_dir: Path, snapshot_dir: Path) -> None:
    """
    Copy the memory tree into snapshot_dir with owner-only permissions.

    Directories are created 0700 and files chmodded 0600 as they land,
    so the snapshot never replicates the live store's historical
    group/world-readable modes.

    Per-principal MEMORY.md directories are 0700 and owned by the
    per-OS-user inner agent processes, so the service user cannot read
    them. os.walk's default is to skip unlistable directories silently;
    that would make the snapshot quietly incomplete, so unreadable
    subtrees are collected and reported in one loud warning instead.
    The run still succeeds: the SQLite corpus is the critical asset,
    and failing outright would mean no backups at all.
    """
    unreadable: list[str] = []

    def _record_walk_error(error: OSError) -> None:
        unreadable.append(str(error.filename))

    for dirpath, _dirnames, filenames in os.walk(memory_dir, onerror=_record_walk_error):
        src_dir = Path(dirpath)
        dst_dir = snapshot_dir / src_dir.relative_to(memory_dir)
        dst_dir.mkdir(exist_ok=True)
        os.chmod(dst_dir, 0o700)
        for filename in filenames:
            if _is_sqlite_sidecar(filename):
                continue
            src = src_dir / filename
            dst = dst_dir / filename
            if src.suffix in _SQLITE_SUFFIXES:
                _backup_sqlite(src, dst)
            else:
                shutil.copy2(src, dst)
            os.chmod(dst, 0o600)

    if unreadable:
        log.warning(
            "Memory backup: %d unreadable director%s not captured in the snapshot (owned by another user): %s",
            len(unreadable),
            "y" if len(unreadable) == 1 else "ies",
            ", ".join(unreadable),
        )


def _latest_snapshot_time(backup_root: Path) -> datetime | None:
    """Return the UTC timestamp of the newest snapshot, or None if there are none."""
    names = sorted(
        entry.name for entry in backup_root.iterdir() if entry.is_dir() and _SNAPSHOT_NAME_RE.match(entry.name)
    )
    if not names:
        return None
    return datetime.strptime(names[-1], _SNAPSHOT_NAME_FORMAT).replace(tzinfo=UTC)


def _prune_old_snapshots(backup_root: Path) -> None:
    """
    Delete the oldest snapshots beyond RETENTION_SNAPSHOTS.

    Snapshot names sort chronologically, so retention is a name sort.
    Only names matching the timestamp shape are candidates; anything
    else in the backup root is left alone.
    """
    snapshots = sorted(
        entry for entry in backup_root.iterdir() if entry.is_dir() and _SNAPSHOT_NAME_RE.match(entry.name)
    )
    for stale in snapshots[:-RETENTION_SNAPSHOTS]:
        shutil.rmtree(stale)
        log.info("Memory backup: pruned old snapshot %s", stale.name)


def run_memory_backup(memory_dir: Path, backup_root: Path, now: datetime) -> Path | None:
    """
    Take one snapshot of the memory tree and prune old snapshots.

    Returns the new snapshot directory, or None when the run was
    skipped because the newest existing snapshot is younger than
    MIN_SNAPSHOT_AGE. Raises on any failure; the caller is responsible
    for logging loudly (a backup that fails silently reproduces the
    no-backups problem it exists to solve).

    The snapshot is written under a dot-prefixed working name and
    renamed into place only when complete, so a crash mid-copy can
    never leave a directory that looks like a valid snapshot. A
    leftover working directory from a previous failed run is removed
    at the start of the next one (the working name is deterministic
    per timestamp, but any prior partial is stale by definition).

    Args:
        memory_dir: The live memory tree (DATA_DIR/memory).
        backup_root: Where snapshots live (DATA_DIR/backups/memory).
        now: Current UTC time; names the snapshot and drives the
            freshness skip.
    """
    backup_root.mkdir(parents=True, exist_ok=True)
    # Owner-only from the root down: the parent backups/ dir too, since
    # mkdir(parents=True) applies default modes to created parents.
    os.chmod(backup_root.parent, 0o700)
    os.chmod(backup_root, 0o700)

    latest = _latest_snapshot_time(backup_root)
    if latest is not None and now - latest < MIN_SNAPSHOT_AGE:
        log.debug("Memory backup: newest snapshot %s is fresh, skipping", latest)
        return None

    # Clear any partial left behind by a previous crashed run.
    for entry in backup_root.iterdir():
        if entry.is_dir() and entry.name.startswith(".partial-"):
            shutil.rmtree(entry)

    name = now.strftime(_SNAPSHOT_NAME_FORMAT)
    partial = backup_root / f".partial-{name}"
    snapshot = backup_root / name
    try:
        partial.mkdir()
        os.chmod(partial, 0o700)
        _snapshot_tree(memory_dir, partial)
        partial.rename(snapshot)
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise

    _prune_old_snapshots(backup_root)
    return snapshot
