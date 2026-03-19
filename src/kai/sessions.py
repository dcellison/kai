"""
SQLite database layer for sessions, jobs, settings, and workspace history.

Provides async CRUD operations for all persistent state in Kai, organized
into four tables:

1. **sessions** -- Claude Code session tracking (session ID, model, cost).
   One row per user_id, upserted on each response. Cost accumulates across
   the lifetime of a session.

2. **jobs** -- Scheduled tasks (reminders, Claude jobs, conditional monitors).
   Created via the scheduling API (POST /api/schedule) or inner Claude's curl.
   Jobs have a schedule_type (once/daily/interval) and can be deactivated
   without deletion to preserve history. Each job belongs to a user_id
   and targets a chat_id for message delivery.

3. **settings** -- Per-user key-value store for persistent config. Used for
   workspace path, voice mode/name preferences, and future extensibility.
   Composite primary key (user_id, key).

4. **workspace_history** -- Per-user recently used workspace paths for the
   /workspaces inline keyboard. Sorted by last_used_at for recency ordering.

All functions use a module-level aiosqlite connection initialized by init_db()
at startup. The database file is kai.db at the project root.
"""

from pathlib import Path

import aiosqlite

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

    Args:
        db_path: Path to the SQLite database file (created if missing).
    """
    global _db
    _db = await aiosqlite.connect(str(db_path))
    _get_db().row_factory = aiosqlite.Row

    # WAL mode allows concurrent readers during writes — essential for
    # multi-user operation where parallel users may write simultaneously.
    # busy_timeout makes SQLite retry for 5s on lock contention instead
    # of immediately failing with "database is locked".
    await _get_db().execute("PRAGMA journal_mode=WAL")
    await _get_db().execute("PRAGMA busy_timeout=5000")

    # Create tables with multi-user schema (user_id as primary identity)
    await _get_db().execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            user_id INTEGER PRIMARY KEY,
            chat_id INTEGER NOT NULL,
            session_id TEXT NOT NULL,
            model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_cost_usd REAL DEFAULT 0.0
        )
    """)
    await _get_db().execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
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
    await _get_db().execute("""
        CREATE TABLE IF NOT EXISTS settings (
            user_id INTEGER NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (user_id, key)
        )
    """)
    await _get_db().execute("""
        CREATE TABLE IF NOT EXISTS workspace_history (
            user_id INTEGER NOT NULL,
            path TEXT NOT NULL,
            last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (user_id, path)
        )
    """)
    await _get_db().commit()


# ── Session management ───────────────────────────────────────────────


async def get_session(user_id: int) -> str | None:
    """Get the current Claude session ID for a user, or None if no session exists."""
    async with _get_db().execute("SELECT session_id FROM sessions WHERE user_id = ?", (user_id,)) as cursor:
        row = await cursor.fetchone()
        return row["session_id"] if row else None


async def save_session(user_id: int, chat_id: int, session_id: str, model: str, cost_usd: float) -> None:
    """
    Save or update a Claude session for a user.

    On conflict (existing user_id), the session_id and model are updated,
    last_used_at is refreshed, and total_cost_usd is accumulated (not replaced).

    Args:
        user_id: Telegram user ID (primary identity).
        chat_id: Telegram chat ID (for message delivery).
        session_id: Claude session identifier from the stream-json response.
        model: Model name used for this session (e.g., "sonnet").
        cost_usd: Cost of this particular interaction (added to running total).
    """
    await _get_db().execute(
        """
        INSERT INTO sessions (user_id, chat_id, session_id, model, total_cost_usd)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET
            chat_id = excluded.chat_id,
            session_id = excluded.session_id,
            model = excluded.model,
            last_used_at = CURRENT_TIMESTAMP,
            total_cost_usd = total_cost_usd + excluded.total_cost_usd
    """,
        (user_id, chat_id, session_id, model, cost_usd),
    )
    await _get_db().commit()


async def clear_session(user_id: int) -> None:
    """Delete the session record for a user. Used by /new and workspace switching."""
    await _get_db().execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
    await _get_db().commit()


async def get_stats(user_id: int) -> dict | None:
    """Get session statistics for the /stats command. Returns None if no session exists."""
    async with _get_db().execute(
        "SELECT session_id, model, created_at, last_used_at, total_cost_usd FROM sessions WHERE user_id = ?",
        (user_id,),
    ) as cursor:
        row = await cursor.fetchone()
        if not row:
            return None
        return dict(row)


# ── Job management ───────────────────────────────────────────────────


async def create_job(
    user_id: int,
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
        user_id: Telegram user ID that owns this job.
        chat_id: Telegram chat ID for message delivery.
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
        """INSERT INTO jobs (user_id, chat_id, name, job_type, prompt, schedule_type, schedule_data, auto_remove, notify_on_check)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            user_id,
            chat_id,
            name,
            job_type,
            prompt,
            schedule_type,
            schedule_data,
            int(auto_remove),
            int(notify_on_check),
        ),
    )
    await _get_db().commit()
    # RuntimeError instead of assert so this guard survives python -O.
    # SQLite always sets lastrowid on INSERT, but guard against None defensively.
    if cursor.lastrowid is None:
        raise RuntimeError("INSERT did not return a row ID")
    return cursor.lastrowid


async def get_jobs(user_id: int) -> list[dict]:
    """Get all active jobs for a specific user. Used by /jobs command."""
    async with _get_db().execute(
        "SELECT id, name, job_type, prompt, schedule_type, schedule_data, auto_remove, notify_on_check, created_at FROM jobs WHERE user_id = ? AND active = 1",
        (user_id,),
    ) as cursor:
        rows = await cursor.fetchall()
        # SQLite stores booleans as integers; convert back to bool
        return [
            {**dict(r), "auto_remove": bool(r["auto_remove"]), "notify_on_check": bool(r["notify_on_check"])}
            for r in rows
        ]


async def get_job_by_id(job_id: int, *, user_id: int | None = None) -> dict | None:
    """Get a single job by ID, or None if not found.

    Args:
        job_id: Database ID of the job.
        user_id: If provided, only return the job if it belongs to this user.
            Internal callers (cron, startup) omit this to access any job.
    """
    if user_id is not None:
        query = "SELECT id, user_id, chat_id, name, job_type, prompt, schedule_type, schedule_data, auto_remove, notify_on_check FROM jobs WHERE id = ? AND user_id = ?"
        params = (job_id, user_id)
    else:
        query = "SELECT id, user_id, chat_id, name, job_type, prompt, schedule_type, schedule_data, auto_remove, notify_on_check FROM jobs WHERE id = ?"
        params = (job_id,)
    async with _get_db().execute(query, params) as cursor:
        row = await cursor.fetchone()
        if not row:
            return None
        return {**dict(row), "auto_remove": bool(row["auto_remove"]), "notify_on_check": bool(row["notify_on_check"])}


async def get_all_active_jobs() -> list[dict]:
    """Get all active jobs across all users. Used at startup to register with APScheduler."""
    async with _get_db().execute(
        "SELECT id, user_id, chat_id, name, job_type, prompt, schedule_type, schedule_data, auto_remove, notify_on_check FROM jobs WHERE active = 1"
    ) as cursor:
        rows = await cursor.fetchall()
        return [
            {**dict(r), "auto_remove": bool(r["auto_remove"]), "notify_on_check": bool(r["notify_on_check"])}
            for r in rows
        ]


async def delete_job(job_id: int, *, user_id: int | None = None) -> bool:
    """Permanently delete a job. Returns True if a row was deleted, False if not found.

    Args:
        job_id: Database ID of the job.
        user_id: If provided, only delete if the job belongs to this user.
            Internal callers (cron) omit this to delete any job.
    """
    if user_id is not None:
        cursor = await _get_db().execute("DELETE FROM jobs WHERE id = ? AND user_id = ?", (job_id, user_id))
    else:
        cursor = await _get_db().execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    await _get_db().commit()
    return cursor.rowcount > 0


async def deactivate_job(job_id: int, *, user_id: int | None = None) -> None:
    """Soft-delete a job by setting active=0. Preserves the row for history.

    Args:
        job_id: Database ID of the job.
        user_id: If provided, only deactivate if the job belongs to this user.
            Internal callers (cron) should pass user_id to prevent cross-user
            deactivation.
    """
    sql = "UPDATE jobs SET active = 0 WHERE id = ?"
    params: list = [job_id]
    if user_id is not None:
        sql += " AND user_id = ?"
        params.append(user_id)
    await _get_db().execute(sql, params)
    await _get_db().commit()


async def update_job(
    job_id: int,
    *,
    user_id: int | None = None,
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

    Note: job_type, user_id, and chat_id are intentionally not updatable.
    Changing a job from reminder to claude (or vice versa) is a fundamentally
    different job -- delete and recreate for that.

    Args:
        job_id: Database ID of the job to update.
        user_id: If provided, only update if the job belongs to this user.
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
    conditions = "id = ? AND active = 1"
    if user_id is not None:
        conditions += " AND user_id = ?"
        values.append(user_id)
    sql = f"UPDATE jobs SET {', '.join(updates)} WHERE {conditions}"
    cursor = await _get_db().execute(sql, values)
    await _get_db().commit()
    return cursor.rowcount > 0


# ── Settings (per-user key-value store) ───────────────────────────────


async def get_setting(user_id: int, key: str) -> str | None:
    """
    Get a setting value by user and key, or None if not set.

    Common keys: "workspace", "voice_mode", "voice_name".
    """
    async with _get_db().execute("SELECT value FROM settings WHERE user_id = ? AND key = ?", (user_id, key)) as cursor:
        row = await cursor.fetchone()
        return row["value"] if row else None


async def set_setting(user_id: int, key: str, value: str) -> None:
    """Set a setting value for a user, creating or updating as needed (upsert)."""
    await _get_db().execute(
        "INSERT INTO settings (user_id, key, value) VALUES (?, ?, ?) ON CONFLICT(user_id, key) DO UPDATE SET value = excluded.value",
        (user_id, key, value),
    )
    await _get_db().commit()


async def delete_setting(user_id: int, key: str) -> None:
    """Remove a setting by user and key. No-op if the key doesn't exist."""
    await _get_db().execute("DELETE FROM settings WHERE user_id = ? AND key = ?", (user_id, key))
    await _get_db().commit()


# ── Workspace history ────────────────────────────────────────────────


async def upsert_workspace_history(user_id: int, path: str) -> None:
    """Record or refresh a workspace path in the user's history. Used for /workspaces keyboard."""
    await _get_db().execute(
        "INSERT INTO workspace_history (user_id, path, last_used_at) VALUES (?, ?, CURRENT_TIMESTAMP) "
        "ON CONFLICT(user_id, path) DO UPDATE SET last_used_at = CURRENT_TIMESTAMP",
        (user_id, path),
    )
    await _get_db().commit()


async def get_workspace_history(user_id: int, limit: int = 10) -> list[dict]:
    """
    Get recent workspace paths for a user, ordered by most recently used first.

    Args:
        user_id: Telegram user ID.
        limit: Maximum number of entries to return (default 10).

    Returns:
        List of dicts with "path" and "last_used_at" keys.
    """
    async with _get_db().execute(
        "SELECT path, last_used_at FROM workspace_history WHERE user_id = ? ORDER BY last_used_at DESC LIMIT ?",
        (user_id, limit),
    ) as cursor:
        rows = await cursor.fetchall()
        return [dict(r) for r in rows]


async def delete_workspace_history(user_id: int, path: str) -> None:
    """Remove a workspace path from a user's history. Used when a workspace directory no longer exists."""
    await _get_db().execute("DELETE FROM workspace_history WHERE user_id = ? AND path = ?", (user_id, path))
    await _get_db().commit()


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
