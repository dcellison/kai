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

4. **workspace_history** - Recently used workspace paths for the /workspaces
   inline keyboard. Sorted by last_used_at for recency ordering.

5. **allowed_workspaces** - Per-user allowed workspace paths, managed via
   /workspace allow and /workspace deny. Unioned with global ALLOWED_WORKSPACES
   env var for the effective access list.

All functions use a module-level aiosqlite connection initialized by init_db()
at startup. The database file is kai.db at the project root.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import aiosqlite

if TYPE_CHECKING:
    from kai.config import Config, WorkspaceConfig

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
        log.debug("Creating allowed_workspaces table")
        await _get_db().execute("""
            CREATE TABLE IF NOT EXISTS allowed_workspaces (
                chat_id INTEGER NOT NULL,
                path TEXT NOT NULL,
                PRIMARY KEY (chat_id, path)
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


# ── Workspace config overrides ─────────────────────────────────────
# Per-user-per-workspace settings stored in the generic settings table.
# Keys are namespaced as ws_config:{chat_id}:{workspace_path}:{field}.
# Each user has independent overrides, so User A can set opus on a repo
# while User B uses sonnet on the same repo.


async def get_workspace_config_settings(chat_id: int, workspace_path: str) -> dict[str, str]:
    """
    Get all config overrides for a user's workspace.

    Returns a dict of field->value pairs (e.g., {"model": "opus",
    "budget": "20.0"}). Values are strings; callers parse as needed.
    Config is per-user-per-workspace: each user has independent overrides.
    """
    # Use SUBSTR for exact prefix matching instead of LIKE, which
    # treats underscores in filesystem paths as single-char wildcards.
    prefix = f"ws_config:{chat_id}:{workspace_path}:"
    prefix_len = len(prefix)
    async with _get_db().execute(
        "SELECT key, value FROM settings WHERE SUBSTR(key, 1, ?) = ?",
        (prefix_len, prefix),
    ) as cursor:
        rows = await cursor.fetchall()
        return {row["key"][prefix_len:]: row["value"] for row in rows}


async def set_workspace_config_setting(chat_id: int, workspace_path: str, field: str, value: str) -> None:
    """Set a single workspace config field for this user."""
    key = f"ws_config:{chat_id}:{workspace_path}:{field}"
    await set_setting(key, value)


async def delete_workspace_config_setting(chat_id: int, workspace_path: str, field: str) -> None:
    """Remove a single workspace config field override for this user."""
    key = f"ws_config:{chat_id}:{workspace_path}:{field}"
    await delete_setting(key)


async def delete_all_workspace_config(chat_id: int, workspace_path: str) -> None:
    """Remove all config overrides for this user's workspace."""
    # Use SUBSTR for exact prefix matching instead of LIKE, which
    # treats underscores in filesystem paths as single-char wildcards.
    prefix = f"ws_config:{chat_id}:{workspace_path}:"
    await _get_db().execute(
        "DELETE FROM settings WHERE SUBSTR(key, 1, ?) = ?",
        (len(prefix), prefix),
    )
    await _get_db().commit()


# ── Workspace config merge ─────────────────────────────────────────


async def build_workspace_config(
    yaml_config: WorkspaceConfig | None,
    workspace_path: Path,
    chat_id: int,
) -> WorkspaceConfig | None:
    """
    Build a WorkspaceConfig by layering database overrides on top of
    the YAML baseline.

    Precedence (highest to lowest):
    1. Database settings (per-user, set via /workspace config)
    2. workspaces.yaml (admin-set via file)
    3. Global defaults (from .env / Config)

    Returns None if neither YAML nor database config exists for this
    workspace (caller uses global defaults).

    The WorkspaceConfig import is deferred to avoid a circular dependency
    (config.py does not import sessions.py; this direction is safe).
    """
    from kai.config import WorkspaceConfig

    db_settings = await get_workspace_config_settings(chat_id, str(workspace_path))

    if not db_settings and yaml_config is None:
        return None

    # Start from YAML baseline or empty defaults
    model = yaml_config.model if yaml_config else None
    budget = yaml_config.budget if yaml_config else None
    timeout = yaml_config.timeout if yaml_config else None
    env = dict(yaml_config.env) if yaml_config and yaml_config.env else None
    env_file = yaml_config.env_file if yaml_config else None
    system_prompt = yaml_config.system_prompt if yaml_config else None
    system_prompt_file = yaml_config.system_prompt_file if yaml_config else None
    path = yaml_config.path if yaml_config else workspace_path

    # Layer database overrides
    if "model" in db_settings:
        model = db_settings["model"]
    if "budget" in db_settings:
        try:
            budget = float(db_settings["budget"])
        except (ValueError, TypeError):
            log.warning("Corrupt budget in DB for chat %d workspace %s", chat_id, workspace_path)
    if "timeout" in db_settings:
        try:
            timeout = int(db_settings["timeout"])
        except (ValueError, TypeError):
            log.warning("Corrupt timeout in DB for chat %d workspace %s", chat_id, workspace_path)
    if "env" in db_settings:
        # DB env vars merge on top of YAML env vars (not replace).
        # This lets admins set baseline env vars in YAML and users
        # add their own without losing the baseline.
        try:
            db_env = json.loads(db_settings["env"])
        except json.JSONDecodeError:
            log.warning("Corrupt env JSON in DB for chat %d workspace %s", chat_id, workspace_path)
            db_env = {}
        if env is None:
            env = db_env
        else:
            env.update(db_env)
    if "prompt" in db_settings:
        # DB prompt replaces YAML prompt entirely (not merged).
        system_prompt = db_settings["prompt"]
        # Clear file-based prompt since inline takes priority
        system_prompt_file = None

    return WorkspaceConfig(
        path=path,
        model=model,
        budget=budget,
        timeout=timeout,
        env=env,
        env_file=env_file,
        system_prompt=system_prompt,
        system_prompt_file=system_prompt_file,
    )


# ── Per-user settings ──────────────────────────────────────────────
# User-level defaults stored in the generic settings table. Keys are
# namespaced as {field}:{chat_id} (e.g., "model:12345"), matching the
# existing voice_mode:{chat_id} convention. These form the "user DB
# override" layer in the six-tier precedence model:
#   workspace DB > workspace YAML > user DB > users.yaml > env > hardcoded

# Canonical field names for per-user settings. Must match the storage
# keys used by set_user_setting / get_user_settings. The user-facing
# command name "context" is mapped to "context_window" in bot.py
# via _FIELD_ALIASES.
_USER_SETTING_FIELDS = {"model", "budget", "timeout", "context_window"}


async def get_user_settings(chat_id: int) -> dict[str, str]:
    """
    Get all per-user settings from the database.

    Returns a dict of field->value pairs (e.g., {"model": "opus",
    "budget": "15.0"}). Values are strings; callers parse as needed.
    Only includes fields that have been explicitly set - missing keys
    mean the user hasn't overridden that setting.
    """
    result = {}
    for field in _USER_SETTING_FIELDS:
        val = await get_setting(f"{field}:{chat_id}")
        if val is not None:
            result[field] = val
    return result


async def set_user_setting(chat_id: int, field: str, value: str) -> None:
    """Set a single per-user setting (e.g., model, budget, timeout)."""
    await set_setting(f"{field}:{chat_id}", value)


async def delete_user_setting(chat_id: int, field: str) -> None:
    """Remove a single per-user setting (reverts to default)."""
    await delete_setting(f"{field}:{chat_id}")


async def delete_all_user_settings(chat_id: int) -> None:
    """
    Remove all per-user settings (reverts everything to defaults).

    Iterates the known field set rather than using a LIKE query,
    since the key format {field}:{chat_id} has the field as a prefix
    (not a shared prefix). Four deletes vs one LIKE - negligible
    for an infrequent reset operation.
    """
    for field in _USER_SETTING_FIELDS:
        await delete_setting(f"{field}:{chat_id}")


class UserDefaults(TypedDict):
    """Resolved per-user settings with concrete types (never None)."""

    model: str
    budget: float
    timeout: int
    context_window: int


async def resolve_user_defaults(
    chat_id: int,
    config: Config,
) -> UserDefaults:
    """
    Resolve per-user settings by layering DB overrides on top of
    users.yaml and env var defaults.

    Returns a UserDefaults dict with keys: model, budget, timeout,
    context_window. All values are resolved - never None.

    Precedence (highest to lowest):
    1. Database (user-set via /settings or /model)
    2. users.yaml (admin baseline per user)
    3. Env var (global defaults from .env)
    4. Hardcoded defaults (in config.py dataclass)

    Note: this does not model workspace-config precedence (which sits
    above user defaults). Callers that need workspace-aware resolution
    should use _restore_workspace() in pool.py instead. This function
    is the canonical user-layer resolver for display, API, and webhook
    contexts where workspace overrides don't apply.
    """
    db_settings = await get_user_settings(chat_id)
    user_config = config.get_user_config(chat_id)

    # Model: DB > users.yaml > env > "sonnet". Use explicit is-not-None
    # checks (not `or`) to match budget/timeout/context_window below and
    # avoid accidentally treating "" as falsy.
    db_model = db_settings.get("model")
    yaml_model = user_config.model if user_config else None
    model = db_model if db_model is not None else yaml_model if yaml_model is not None else config.claude_model

    # Budget: DB > users.yaml max_budget (as default, not ceiling) > env.
    # max_budget in users.yaml serves double duty - it's both the admin
    # ceiling AND the baseline default. If a user has max_budget=20 and
    # hasn't set a /settings budget, they get $20. If they set /settings
    # budget 15, they get $15. They cannot set above $20.
    # Defensive try/except matches _restore_workspace and _show_settings.
    yaml_budget = user_config.max_budget if user_config and user_config.max_budget is not None else None
    try:
        budget = float(db_settings["budget"]) if "budget" in db_settings else None
    except (ValueError, TypeError):
        budget = None
    if budget is None:
        budget = yaml_budget if yaml_budget is not None else config.claude_max_budget_usd

    # Timeout: DB > users.yaml > env > 120
    yaml_timeout = user_config.timeout if user_config and user_config.timeout is not None else None
    try:
        timeout = int(db_settings["timeout"]) if "timeout" in db_settings else None
    except (ValueError, TypeError):
        timeout = None
    if timeout is None:
        timeout = yaml_timeout if yaml_timeout is not None else config.claude_timeout_seconds

    # Context window: DB > users.yaml > env > 0
    yaml_ctx = user_config.context_window if user_config and user_config.context_window is not None else None
    try:
        context_window = int(db_settings["context_window"]) if "context_window" in db_settings else None
    except (ValueError, TypeError):
        context_window = None
    if context_window is None:
        context_window = yaml_ctx if yaml_ctx is not None else config.claude_max_context_window

    return {
        "model": model,
        "budget": budget,
        "timeout": timeout,
        "context_window": context_window,
    }


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


# ── Per-user allowed workspaces ──────────────────────────────────────


async def add_allowed_workspace(chat_id: int, path: str) -> None:
    """
    Add a workspace path to the user's allowed list.

    Uses INSERT OR IGNORE so adding a duplicate is a no-op.
    Paths should be resolved to canonical form before storage.
    """
    db = _get_db()
    await db.execute(
        "INSERT OR IGNORE INTO allowed_workspaces (chat_id, path) VALUES (?, ?)",
        (chat_id, path),
    )
    await db.commit()


async def remove_allowed_workspace(chat_id: int, path: str) -> bool:
    """
    Remove a workspace path from the user's allowed list.

    Returns True if a row was deleted, False if the path was not
    in the user's list (distinguishes "removed" from "not found"
    so the caller can give appropriate feedback).
    """
    db = _get_db()
    cursor = await db.execute(
        "DELETE FROM allowed_workspaces WHERE chat_id = ? AND path = ?",
        (chat_id, path),
    )
    await db.commit()
    return cursor.rowcount > 0


async def get_allowed_workspaces(chat_id: int) -> list[Path]:
    """
    Get the user's allowed workspace paths from the database.

    Returns paths in insertion order. These are only the user-added
    paths; the global ALLOWED_WORKSPACES fallback is handled by
    resolve_workspace_access().
    """
    db = _get_db()
    cursor = await db.execute(
        "SELECT path FROM allowed_workspaces WHERE chat_id = ? ORDER BY rowid",
        (chat_id,),
    )
    rows = await cursor.fetchall()
    return [Path(row[0]) for row in rows]


async def resolve_workspace_access(chat_id: int, config: Config) -> tuple[Path | None, list[Path]]:
    """
    Resolve per-user workspace_base and allowed_workspaces.

    Returns (workspace_base, allowed_workspaces) with per-user config
    applied. The allowed list is the union of user-added DB entries and
    the global ALLOWED_WORKSPACES env var, deduplicated by resolved path.

    Precedence for workspace_base:
        1. users.yaml workspace_base (admin-set per user)
        2. Global WORKSPACE_BASE env var

    Precedence for allowed_workspaces:
        Union of DB entries (user-managed) and global config (admin-set).
        Deduplicated; DB entries appear first in the list.
    """
    user_config = config.get_user_config(chat_id)

    # workspace_base: users.yaml > env
    base = user_config.workspace_base if user_config and user_config.workspace_base else config.workspace_base

    # allowed_workspaces: union of DB + global, deduplicated.
    # DB entries first so user-added workspaces appear at the top
    # of the /workspaces keyboard and /workspace allowed list.
    db_allowed = await get_allowed_workspaces(chat_id)
    seen: set[Path] = set()
    combined: list[Path] = []

    for p in db_allowed:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            combined.append(resolved)

    for p in config.allowed_workspaces:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            combined.append(resolved)

    return base, combined


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
