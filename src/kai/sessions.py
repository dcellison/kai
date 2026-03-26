"""
SQLite database layer for sessions, jobs, settings, and workspace history.

Provides async CRUD operations for all persistent state in Kai, organized
into four tables:

1. **sessions** — Claude Code session tracking (session ID, model, cost).
   One row per chat_id, upserted on each response. Cost accumulates across
   the lifetime of a session.

2. **jobs** — Scheduled tasks (reminders, Claude jobs, conditional monitors).
   Created via the scheduling API (POST /api/schedule) or inner Claude's curl.
   Jobs have a schedule_type (once/daily/interval) and can be deactivated
   without deletion to preserve history.

3. **settings** — Generic key-value store for persistent config. Used for
   workspace path, voice mode/name preferences, and future extensibility.
   Keys are namespaced strings like "voice_mode:{chat_id}".

4. **workspace_history** — Recently used workspace paths for the /workspaces
   inline keyboard. Sorted by last_used_at for recency ordering.

All functions use a module-level aiosqlite connection initialized by init_db()
at startup. The database file is kai.db at the project root.
"""

import logging
from pathlib import Path

import aiosqlite

log = logging.getLogger(__name__)

# Module-level database connection, initialized by init_db() at startup
_db: aiosqlite.Connection | None = None


def _get_db() -> aiosqlite.Connection:
    """Return the database connection, raising if init_db() hasn't been called."""
    # RuntimeError instead of assert so this guard survives python -O
    if _db is None:
        raise RuntimeError("Database not initialized - call init_db() first")
    return _db


# ── Initialization ───────────────────────────────────────────────────


async def init_db(db_path: Path) -> None:
    """
    Open the SQLite database and create tables if they don't exist.

    Called once at startup from main.py. Uses aiosqlite.Row as the row
    factory so query results can be accessed by column name.

    All DDL (CREATE TABLE, ALTER TABLE, migrations) runs inside a single
    explicit transaction. On failure, ROLLBACK undoes everything - the
    database is either fully initialized or completely unchanged. SQLite
    supports transactional DDL (as does PostgreSQL; MySQL does not).

    Args:
        db_path: Path to the SQLite database file (created if missing).
    """
    global _db
    _db = await aiosqlite.connect(str(db_path))
    _get_db().row_factory = aiosqlite.Row
    # PRAGMAs are database configuration, not schema. They must execute
    # before any transaction begins.
    # WAL mode allows concurrent readers during writes, which prevents
    # multi-user requests from blocking each other on the database.
    # busy_timeout retries for 5 seconds on lock contention instead of
    # failing immediately with SQLITE_BUSY.
    async with _get_db().execute("PRAGMA journal_mode=WAL") as cursor:
        row = await cursor.fetchone()
        if row and row[0] != "wal":
            log.warning("Failed to enable WAL mode; journal_mode is %s", row[0])
    await _get_db().execute("PRAGMA busy_timeout=5000")

    try:
        # BEGIN IMMEDIATE acquires the write lock up front rather than on
        # the first write statement, preventing a deadlock if another
        # connection holds a read lock during our init sequence.
        await _get_db().execute("BEGIN IMMEDIATE")
        log.debug("Creating sessions table")
        await _get_db().execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                chat_id INTEGER PRIMARY KEY,
                session_id TEXT NOT NULL,
                model TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_cost_usd REAL DEFAULT 0.0
            )
        """)
        log.debug("Creating jobs table")
        await _get_db().execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                chat_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                job_type TEXT NOT NULL,
                prompt TEXT NOT NULL,
                schedule_type TEXT NOT NULL,
                schedule_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                active INTEGER DEFAULT 1,
                auto_remove INTEGER DEFAULT 0,
                notify_on_check INTEGER DEFAULT 0
            )
        """)

        # Schema evolution: add notify_on_check column to existing
        # databases that don't have it
        cursor = await _get_db().execute("PRAGMA table_info(jobs)")
        columns = [row[1] for row in await cursor.fetchall()]
        if "notify_on_check" not in columns:
            log.debug("Adding notify_on_check column to jobs table")
            await _get_db().execute("ALTER TABLE jobs ADD COLUMN notify_on_check INTEGER DEFAULT 0")

        log.debug("Creating settings table")
        await _get_db().execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        log.debug("Creating workspace_history table")
        await _get_db().execute("""
            CREATE TABLE IF NOT EXISTS workspace_history (
                path TEXT NOT NULL,
                chat_id INTEGER NOT NULL,
                last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (path, chat_id)
            )
        """)

        # Schema evolution: migrate old workspace_history tables (path-only
        # PK) to the new composite PK (path, chat_id). SQLite does not
        # support ALTER TABLE to change primary keys, so we recreate the
        # table. Existing rows get chat_id=0; main.py calls
        # backfill_workspace_history() to assign them to the admin user.
        # Individual execute() calls (not executescript) so they participate
        # in the outer transaction naturally.
        cursor = await _get_db().execute("PRAGMA table_info(workspace_history)")
        ws_columns = [row[1] for row in await cursor.fetchall()]
        if "chat_id" not in ws_columns:
            log.debug("Migrating workspace_history to composite PK (path, chat_id)")
            await _get_db().execute("""
                CREATE TABLE workspace_history_new (
                    path TEXT NOT NULL,
                    chat_id INTEGER NOT NULL DEFAULT 0,
                    last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (path, chat_id)
                )
            """)
            await _get_db().execute("""
                INSERT INTO workspace_history_new (path, last_used_at)
                    SELECT path, last_used_at FROM workspace_history
            """)
            await _get_db().execute("DROP TABLE workspace_history")
            await _get_db().execute("ALTER TABLE workspace_history_new RENAME TO workspace_history")

        await _get_db().commit()
    except Exception:
        # Roll back the entire init sequence. The database is left in its
        # pre-init state (no partial tables, no half-migrated schema).
        # Close and nullify the connection so a retry of init_db doesn't
        # silently overwrite _db with a second open connection.
        try:
            await _get_db().rollback()
        except Exception:
            pass
        if _db is not None:
            try:
                await _db.close()
            except Exception:
                pass
        _db = None
        raise


# ── Session management ───────────────────────────────────────────────


async def get_session(chat_id: int) -> str | None:
    """Get the current Claude session ID for a chat, or None if no session exists."""
    async with _get_db().execute("SELECT session_id FROM sessions WHERE chat_id = ?", (chat_id,)) as cursor:
        row = await cursor.fetchone()
        return row["session_id"] if row else None


async def save_session(chat_id: int, session_id: str, model: str, cost_usd: float) -> None:
    """
    Save or update a Claude session for a chat.

    On conflict (existing chat_id), the session_id and model are updated,
    last_used_at is refreshed, and total_cost_usd is accumulated (not replaced).

    Args:
        chat_id: Telegram chat ID.
        session_id: Claude session identifier from the stream-json response.
        model: Model name used for this session (e.g., "sonnet").
        cost_usd: Cost of this particular interaction (added to running total).
    """
    await _get_db().execute(
        """
        INSERT INTO sessions (chat_id, session_id, model, total_cost_usd)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(chat_id) DO UPDATE SET
            session_id = excluded.session_id,
            model = excluded.model,
            last_used_at = CURRENT_TIMESTAMP,
            total_cost_usd = total_cost_usd + excluded.total_cost_usd
    """,
        (chat_id, session_id, model, cost_usd),
    )
    await _get_db().commit()


async def clear_session(chat_id: int) -> None:
    """Delete the session record for a chat. Used by /new and workspace switching."""
    await _get_db().execute("DELETE FROM sessions WHERE chat_id = ?", (chat_id,))
    await _get_db().commit()


async def get_stats(chat_id: int) -> dict | None:
    """Get session statistics for the /stats command. Returns None if no session exists."""
    async with _get_db().execute(
        "SELECT session_id, model, created_at, last_used_at, total_cost_usd FROM sessions WHERE chat_id = ?",
        (chat_id,),
    ) as cursor:
        row = await cursor.fetchone()
        if not row:
            return None
        return dict(row)


# ── Job management ───────────────────────────────────────────────────


async def create_job(
    chat_id: int,
    name: str,
    job_type: str,
    prompt: str,
    schedule_type: str,
    schedule_data: str,
    auto_remove: bool = False,
    notify_on_check: bool = False,
) -> int:
    """
    Create a new scheduled job and return its integer ID.

    Args:
        chat_id: Telegram chat ID that owns this job.
        name: Human-readable job name (shown in /jobs).
        job_type: "reminder" (sends prompt as-is) or "claude" (processed by Claude).
        prompt: Message text for reminders, or Claude prompt for Claude jobs.
        schedule_type: "once", "daily", or "interval".
        schedule_data: JSON string with schedule details.
            once: {"run_at": "ISO-datetime"}
            daily: {"times": ["HH:MM", ...]} (UTC)
            interval: {"seconds": N}
        auto_remove: If True, deactivate the job when a CONDITION_MET marker is received.
        notify_on_check: If True (and auto_remove=True), forward CONDITION_NOT_MET responses
            instead of silently continuing. Useful for "heartbeat" status updates.

    Returns:
        The auto-generated integer job ID.
    """
    cursor = await _get_db().execute(
        """INSERT INTO jobs (chat_id, name, job_type, prompt, schedule_type, schedule_data, auto_remove, notify_on_check)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (chat_id, name, job_type, prompt, schedule_type, schedule_data, int(auto_remove), int(notify_on_check)),
    )
    await _get_db().commit()
    # RuntimeError instead of assert so this guard survives python -O.
    # SQLite always sets lastrowid on INSERT, but guard against None defensively.
    if cursor.lastrowid is None:
        raise RuntimeError("INSERT did not return a row ID")
    return cursor.lastrowid


async def get_jobs(chat_id: int) -> list[dict]:
    """Get all active jobs for a specific chat. Used by /jobs command."""
    async with _get_db().execute(
        "SELECT id, name, job_type, prompt, schedule_type, schedule_data, auto_remove, notify_on_check, created_at FROM jobs WHERE chat_id = ? AND active = 1",
        (chat_id,),
    ) as cursor:
        rows = await cursor.fetchall()
        # SQLite stores booleans as integers; convert back to bool
        return [
            {**dict(r), "auto_remove": bool(r["auto_remove"]), "notify_on_check": bool(r["notify_on_check"])}
            for r in rows
        ]


async def get_job_by_id(job_id: int) -> dict | None:
    """Get a single job by ID, or None if not found. Used by cron.register_job_by_id()."""
    async with _get_db().execute(
        "SELECT id, chat_id, name, job_type, prompt, schedule_type, schedule_data, auto_remove, notify_on_check FROM jobs WHERE id = ?",
        (job_id,),
    ) as cursor:
        row = await cursor.fetchone()
        if not row:
            return None
        return {**dict(row), "auto_remove": bool(row["auto_remove"]), "notify_on_check": bool(row["notify_on_check"])}


async def get_all_active_jobs() -> list[dict]:
    """Get all active jobs across all chats. Used at startup to register with APScheduler."""
    async with _get_db().execute(
        "SELECT id, chat_id, name, job_type, prompt, schedule_type, schedule_data, auto_remove, notify_on_check FROM jobs WHERE active = 1"
    ) as cursor:
        rows = await cursor.fetchall()
        return [
            {**dict(r), "auto_remove": bool(r["auto_remove"]), "notify_on_check": bool(r["notify_on_check"])}
            for r in rows
        ]


async def delete_job(job_id: int, chat_id: int | None = None) -> bool:
    """
    Permanently delete a job. Returns True if a row was deleted, False if
    not found (or not owned by chat_id when provided).
    """
    if chat_id is not None:
        cursor = await _get_db().execute(
            "DELETE FROM jobs WHERE id = ? AND chat_id = ?",
            (job_id, chat_id),
        )
    else:
        cursor = await _get_db().execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    await _get_db().commit()
    return cursor.rowcount > 0


async def deactivate_job(job_id: int, chat_id: int | None = None) -> bool:
    """
    Soft-delete a job by setting active=0. Preserves the row for history.

    When chat_id is provided, the job is only deactivated if it belongs to
    that user. This prevents cross-user job manipulation. When None, the
    job is deactivated unconditionally (backward-compatible for internal
    callers like cron.py that have already verified ownership).

    Returns True if a row was deactivated, False if not found or not
    owned by chat_id.
    """
    if chat_id is not None:
        cursor = await _get_db().execute(
            "UPDATE jobs SET active = 0 WHERE id = ? AND chat_id = ?",
            (job_id, chat_id),
        )
    else:
        cursor = await _get_db().execute("UPDATE jobs SET active = 0 WHERE id = ?", (job_id,))
    await _get_db().commit()
    return cursor.rowcount > 0


async def update_job(
    job_id: int,
    *,
    chat_id: int | None = None,
    name: str | None = None,
    prompt: str | None = None,
    schedule_type: str | None = None,
    schedule_data: str | None = None,
    auto_remove: bool | None = None,
    notify_on_check: bool | None = None,
) -> bool:
    """
    Update mutable fields on an existing active job.

    Only provided (non-None) fields are updated. The job must be active.
    Returns True if a row was updated, False if the job wasn't found or
    is inactive.

    Note: job_type and chat_id are intentionally not updatable. Changing
    a job from reminder to claude (or vice versa) is a fundamentally
    different job — delete and recreate for that.

    Args:
        job_id: Database ID of the job to update.
        name: New job name.
        prompt: New prompt text.
        schedule_type: New schedule type ("once", "daily", "interval").
        schedule_data: New schedule data (JSON string).
        auto_remove: New auto_remove flag.
        notify_on_check: New notify_on_check flag.

    Returns:
        True if the job was updated, False if not found or inactive.
    """
    # Build SET clause dynamically from provided fields. This is safe because
    # all field names are from a controlled list, not user input.
    updates = []
    values = []
    if name is not None:
        updates.append("name = ?")
        values.append(name)
    if prompt is not None:
        updates.append("prompt = ?")
        values.append(prompt)
    if schedule_type is not None:
        updates.append("schedule_type = ?")
        values.append(schedule_type)
    if schedule_data is not None:
        updates.append("schedule_data = ?")
        values.append(schedule_data)
    if auto_remove is not None:
        updates.append("auto_remove = ?")
        values.append(int(auto_remove))
    if notify_on_check is not None:
        updates.append("notify_on_check = ?")
        values.append(int(notify_on_check))

    if not updates:
        return False

    values.append(job_id)
    where = "WHERE id = ? AND active = 1"
    if chat_id is not None:
        where += " AND chat_id = ?"
        values.append(chat_id)
    sql = f"UPDATE jobs SET {', '.join(updates)} {where}"
    cursor = await _get_db().execute(sql, values)
    await _get_db().commit()
    return cursor.rowcount > 0


# ── Settings (generic key-value store) ───────────────────────────────


async def get_setting(key: str) -> str | None:
    """
    Get a setting value by key, or None if not set.

    Common keys: "workspace", "voice_mode:{chat_id}", "voice_name:{chat_id}".
    """
    async with _get_db().execute("SELECT value FROM settings WHERE key = ?", (key,)) as cursor:
        row = await cursor.fetchone()
        return row["value"] if row else None


async def set_setting(key: str, value: str) -> None:
    """Set a setting value, creating or updating as needed (upsert)."""
    await _get_db().execute(
        "INSERT INTO settings (key, value) VALUES (?, ?) ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    await _get_db().commit()


async def delete_setting(key: str) -> None:
    """Remove a setting by key. No-op if the key doesn't exist."""
    await _get_db().execute("DELETE FROM settings WHERE key = ?", (key,))
    await _get_db().commit()


# ── Workspace history ────────────────────────────────────────────────


async def upsert_workspace_history(path: str, chat_id: int) -> None:
    """Record or refresh a workspace path in the user's history."""
    await _get_db().execute(
        "INSERT OR REPLACE INTO workspace_history (path, chat_id, last_used_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
        (path, chat_id),
    )
    await _get_db().commit()


async def get_all_workspace_paths(limit: int = 100) -> list[str]:
    """
    Get distinct workspace paths across all users, most recently used first.

    Used by _resolve_local_repo() to match GitHub repos against any user's
    workspace history, since webhook routing has no user context.

    Args:
        limit: Maximum number of paths to return (default 100).

    Returns:
        List of workspace path strings (deduplicated across users).
    """
    # GROUP BY + MAX(last_used_at) instead of DISTINCT to get
    # deterministic ordering when the same path appears for multiple
    # users with different timestamps.
    async with _get_db().execute(
        "SELECT path FROM workspace_history GROUP BY path ORDER BY MAX(last_used_at) DESC LIMIT ?",
        (limit,),
    ) as cursor:
        rows = await cursor.fetchall()
        return [row[0] for row in rows]


async def get_workspace_history(chat_id: int, limit: int = 10) -> list[dict]:
    """
    Get recent workspace paths for a specific user.

    Args:
        chat_id: Telegram chat ID of the user.
        limit: Maximum number of entries to return (default 10).

    Returns:
        List of dicts with "path" and "last_used_at" keys.
    """
    async with _get_db().execute(
        "SELECT path, last_used_at FROM workspace_history WHERE chat_id = ? ORDER BY last_used_at DESC LIMIT ?",
        (chat_id, limit),
    ) as cursor:
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def delete_workspace_history(path: str, chat_id: int) -> None:
    """Remove a workspace path from a user's history."""
    await _get_db().execute(
        "DELETE FROM workspace_history WHERE path = ? AND chat_id = ?",
        (path, chat_id),
    )
    await _get_db().commit()


async def backfill_workspace_history(default_chat_id: int) -> None:
    """
    Assign unowned workspace history rows to the default user.

    Phase 2 migration: rows created before per-user workspace history
    have chat_id=0 (the ALTER TABLE default). This assigns them to the
    admin user so they appear in the right user's /workspaces list.
    Idempotent - no-op after the first run.
    """
    cursor = await _get_db().execute(
        "UPDATE workspace_history SET chat_id = ? WHERE chat_id = 0",
        (default_chat_id,),
    )
    await _get_db().commit()
    if cursor.rowcount > 0:
        log.info(
            "Migrated %d workspace history rows to user %d",
            cursor.rowcount,
            default_chat_id,
        )


# ── Lifecycle ────────────────────────────────────────────────────────


async def close_db() -> None:
    """Close the database connection. Called during shutdown from main.py."""
    global _db
    if _db:
        try:
            await _get_db().close()
        finally:
            # Clear even if close() raises so subsequent _get_db() calls
            # get a clear RuntimeError instead of using a broken connection
            _db = None
