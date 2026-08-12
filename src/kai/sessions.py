"""
SQLite database layer for sessions, jobs, settings, and workspace history.

Provides async CRUD operations for all persistent state in Kai, organized
into four tables:

1. **sessions** - Agent session tracking (session ID, model).
   One row per chat_id, upserted on each response.

2. **jobs** - Scheduled tasks (reminders, agent jobs, conditional monitors).
   Created via the scheduling API (POST /api/schedule) or the inner agent's curl.
   Jobs have a schedule_type (once/daily/interval) and can be deactivated
   without deletion to preserve history.

3. **settings** - Generic key-value store for persistent config. Used for
   workspace path, voice mode/name preferences, and future extensibility.
   Keys are namespaced strings like "voice_mode:{chat_id}".

4. **telegram_update_queue** - Durable inbound Telegram webhook updates.
   Webhook mode persists accepted updates before acknowledging them, then a
   worker claims and processes rows so restarts can replay unfinished work.

5. **workspace_history** - Recently used workspace paths for the /workspaces
   inline keyboard. Sorted by last_used_at for recency ordering.

6. **allowed_workspaces** - Per-user allowed workspace paths, managed via
   /workspace allow and /workspace deny. Unioned with global ALLOWED_WORKSPACES
   env var for the effective access list.

All functions use a module-level aiosqlite connection initialized by init_db()
at startup. The database file is kai.db at the project root.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import aiosqlite

from kai.job_types import JOB_TYPE_AGENT, LEGACY_JOB_TYPE_AGENT, normalize_job_type
from kai.workshop.artifacts import InboundArtifact, record_inbound_artifact
from kai.workshop.bootstrap import (
    BootstrapHuman,
    BootstrapNotificationChannel,
    BootstrapResult,
    bootstrap_default_workshop,
)
from kai.workshop.conversation_runs import (
    CompatibilityConversationRunResolution,
    resolve_canonical_conversation_run,
)
from kai.workshop.domain import MessageId
from kai.workshop.inbound import InboundMessage, record_inbound_message
from kai.workshop.outbound import (
    DeliveryObservation,
    OutboundMessage,
    OutboundStreamingFinalizationResult,
    record_delivery_observation,
    record_outbound_message,
    record_outbound_message_with_streaming_finalization,
)
from kai.workshop.schema import migrate_workshop_schema
from kai.workshop.store import AppendResult, WorkshopEventStore
from kai.workshop.streaming_preview import (
    ConfirmedTelegramStreamingPreview,
    TelegramStreamingPreviewBinding,
    bind_confirmed_telegram_streaming_preview,
)

if TYPE_CHECKING:
    from kai.config import Config, WorkspaceConfig

log = logging.getLogger(__name__)

# Module-level database connection, initialized by init_db() at startup
_db: aiosqlite.Connection | None = None
_workshop_event_lock: asyncio.Lock | None = None


class WorkshopFinalizationCommitUncertainError(RuntimeError):
    """The canonical reply transaction could not be resolved safely."""


class TelegramUpdateQueueRow(TypedDict):
    """Row shape for durable inbound Telegram webhook work."""

    id: int
    update_id: int
    payload: str
    status: str
    attempt_count: int
    last_error: str | None
    locked_at: str | None
    processed_at: str | None
    created_at: str
    updated_at: str


def _get_db() -> aiosqlite.Connection:
    """Return the database connection, raising if init_db() hasn't been called."""
    # RuntimeError instead of assert so this guard survives python -O
    if _db is None:
        raise RuntimeError("Database not initialized - call init_db() first")
    return _db


def _restrict_sqlite_file(path: Path) -> None:
    """Create/chmod a SQLite file so persisted secrets are owner-only."""
    flags = os.O_CREAT | os.O_RDWR
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    fd = os.open(path, flags, 0o600)
    try:
        os.fchmod(fd, 0o600)
    finally:
        os.close(fd)
    path.chmod(0o600)


def _restrict_sqlite_files(db_path: Path) -> None:
    """Restrict the main DB and any SQLite WAL/SHM companion files."""
    for path in (db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")):
        if path.exists():
            path.chmod(0o600)


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
    global _db, _workshop_event_lock
    _restrict_sqlite_file(db_path)
    _db = await aiosqlite.connect(str(db_path))
    _workshop_event_lock = asyncio.Lock()
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
    await _get_db().execute("PRAGMA foreign_keys=ON")
    _restrict_sqlite_files(db_path)

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
                last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

        # Durable compatibility migration: the old value named one
        # implementation even though scheduled work always routes through the
        # owning user's selected backend. Preserve every job while replacing
        # only that identifier with the backend-neutral canonical value.
        await _get_db().execute(
            "UPDATE jobs SET job_type = ? WHERE job_type = ?",
            (JOB_TYPE_AGENT, LEGACY_JOB_TYPE_AGENT),
        )

        log.debug("Creating settings table")
        await _get_db().execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        log.debug("Creating telegram_update_queue table")
        await _get_db().execute("""
            CREATE TABLE IF NOT EXISTS telegram_update_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                update_id INTEGER NOT NULL UNIQUE,
                payload TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                attempt_count INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                locked_at TIMESTAMP,
                processed_at TIMESTAMP,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CHECK (status IN ('pending', 'processing', 'done'))
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
        # User-registered memory projects (the DB layer under the
        # operator-pinned memory-projects.yaml). One root per row:
        # chat-registered projects are single-root by design; only
        # YAML entries support multi-root. workspace_root is UNIQUE
        # because the detector's longest-prefix match needs a single
        # owner per root, mirroring the YAML loader's cross-project
        # duplicate-root rule. created_by drives the unregister
        # permission check (registering user or admin).
        log.debug("Creating memory_projects table")
        await _get_db().execute("""
            CREATE TABLE IF NOT EXISTS memory_projects (
                project_id TEXT PRIMARY KEY,
                display_name TEXT NOT NULL,
                workspace_root TEXT NOT NULL UNIQUE,
                memory_enabled INTEGER NOT NULL DEFAULT 1,
                default_scope_for_new_facts TEXT,
                created_by INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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

        # Additive Workshop schema only. Canonical bootstrap records are
        # created separately after the protected user configuration has
        # loaded; no message path reads these tables yet.
        await migrate_workshop_schema(_get_db(), manage_transaction=False)
        await _get_db().commit()
        _restrict_sqlite_files(db_path)
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
        _workshop_event_lock = None
        raise


# ── Session management ───────────────────────────────────────────────


async def bootstrap_workshop_foundation(
    humans: tuple[BootstrapHuman, ...],
    *,
    notification_channels: tuple[BootstrapNotificationChannel, ...] = (),
) -> BootstrapResult:
    """Seed non-authoritative Workshop records on Kai's initialized DB."""
    if _workshop_event_lock is None:
        raise RuntimeError("Database not initialized - call init_db() first")
    async with _workshop_event_lock:
        store = WorkshopEventStore.from_initialized_connection(_get_db())
        return await bootstrap_default_workshop(
            store,
            humans,
            notification_channels=notification_channels,
        )


async def record_workshop_inbound_message(message: InboundMessage) -> AppendResult:
    """Serialize one canonical shadow write on Kai's shared SQLite connection."""
    if _workshop_event_lock is None:
        raise RuntimeError("Database not initialized - call init_db() first")
    async with _workshop_event_lock:
        store = WorkshopEventStore.from_initialized_connection(_get_db())
        return await record_inbound_message(store, message)


async def resolve_workshop_conversation_run(
    inbound_message_id: MessageId,
) -> CompatibilityConversationRunResolution:
    """Resolve one canonical inbound message for transport-neutral execution."""
    if _workshop_event_lock is None:
        raise RuntimeError("Database not initialized - call init_db() first")
    async with _workshop_event_lock:
        store = WorkshopEventStore.from_initialized_connection(_get_db())
        return await resolve_canonical_conversation_run(store, inbound_message_id)


async def record_workshop_inbound_artifact(
    artifact: InboundArtifact,
    *,
    storage_root: Path,
) -> AppendResult:
    """Serialize one canonical artifact shadow write on Kai's shared database."""
    if _workshop_event_lock is None:
        raise RuntimeError("Database not initialized - call init_db() first")
    async with _workshop_event_lock:
        store = WorkshopEventStore.from_initialized_connection(_get_db())
        return await record_inbound_artifact(store, artifact, storage_root=storage_root)


async def record_workshop_outbound_message(message: OutboundMessage) -> AppendResult:
    """Serialize one assistant-result shadow write on Kai's shared database."""
    if _workshop_event_lock is None:
        raise RuntimeError("Database not initialized - call init_db() first")
    async with _workshop_event_lock:
        store = WorkshopEventStore.from_initialized_connection(_get_db())
        return await record_outbound_message(store, message)


async def record_workshop_streaming_preview(
    preview: ConfirmedTelegramStreamingPreview,
) -> TelegramStreamingPreviewBinding:
    """Serialize one confirmed non-final Telegram preview binding."""
    if _workshop_event_lock is None:
        raise RuntimeError("Database not initialized - call init_db() first")
    async with _workshop_event_lock:
        store = WorkshopEventStore.from_initialized_connection(_get_db())
        return await bind_confirmed_telegram_streaming_preview(store, preview)


async def record_workshop_streaming_finalization(
    message: OutboundMessage,
) -> OutboundStreamingFinalizationResult:
    """Atomically record one authoritative reply and its delivery plan.

    An SQLite error may have happened while the commit result was crossing the
    driver boundary. Repeating the deterministic operation on the same locked
    connection resolves both outcomes: it either creates the rolled-back work
    or observes the already-committed message, delivery, and fragment plan.
    A second failure is reported as uncertain so the handler never falls back
    to a direct send that could duplicate committed outbox work.
    """
    if _workshop_event_lock is None:
        raise RuntimeError("Database not initialized - call init_db() first")
    async with _workshop_event_lock:
        store = WorkshopEventStore.from_initialized_connection(_get_db())
        try:
            return await record_outbound_message_with_streaming_finalization(store, message)
        except aiosqlite.Error:
            try:
                return await record_outbound_message_with_streaming_finalization(store, message)
            except Exception as resolution_error:
                raise WorkshopFinalizationCommitUncertainError(
                    "Could not determine whether the Workshop reply transaction committed"
                ) from resolution_error


async def record_workshop_delivery_observation(observation: DeliveryObservation) -> AppendResult:
    """Serialize one transport-delivery observation on Kai's shared database."""
    if _workshop_event_lock is None:
        raise RuntimeError("Database not initialized - call init_db() first")
    async with _workshop_event_lock:
        store = WorkshopEventStore.from_initialized_connection(_get_db())
        return await record_delivery_observation(store, observation)


async def get_session(chat_id: int) -> str | None:
    """Get the current agent session ID for a chat, or None if no session exists."""
    async with _get_db().execute("SELECT session_id FROM sessions WHERE chat_id = ?", (chat_id,)) as cursor:
        row = await cursor.fetchone()
        return row["session_id"] if row else None


async def save_session(chat_id: int, session_id: str, model: str) -> None:
    """
    Save or update an agent session for a chat.

    On conflict (existing chat_id), the session_id and model are updated
    and last_used_at is refreshed.

    Args:
        chat_id: Telegram chat ID.
        session_id: Agent session identifier reported by the backend.
        model: Model name used for this session (e.g., "sonnet").
    """
    await _get_db().execute(
        """
        INSERT INTO sessions (chat_id, session_id, model)
        VALUES (?, ?, ?)
        ON CONFLICT(chat_id) DO UPDATE SET
            session_id = excluded.session_id,
            model = excluded.model,
            last_used_at = CURRENT_TIMESTAMP
    """,
        (chat_id, session_id, model),
    )
    await _get_db().commit()


async def clear_session(chat_id: int) -> None:
    """Delete the session record for a chat. Used by /new and workspace switching."""
    await _get_db().execute("DELETE FROM sessions WHERE chat_id = ?", (chat_id,))
    await _get_db().commit()


async def get_stats(chat_id: int) -> dict | None:
    """Get session statistics for the /stats command. Returns None if no session exists."""
    async with _get_db().execute(
        "SELECT session_id, model, created_at, last_used_at FROM sessions WHERE chat_id = ?",
        (chat_id,),
    ) as cursor:
        row = await cursor.fetchone()
        if not row:
            return None
        return dict(row)


# ── Durable Telegram update queue ───────────────────────────────────


def _telegram_update_queue_row(row: aiosqlite.Row) -> TelegramUpdateQueueRow:
    return {
        "id": row["id"],
        "update_id": row["update_id"],
        "payload": row["payload"],
        "status": row["status"],
        "attempt_count": row["attempt_count"],
        "last_error": row["last_error"],
        "locked_at": row["locked_at"],
        "processed_at": row["processed_at"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


async def enqueue_telegram_update(update_id: int, payload: str) -> tuple[int, bool]:
    """
    Persist a Telegram webhook update before acknowledging it.

    update_id is unique per bot. Duplicate Telegram retries return the existing
    row ID and inserted=False rather than creating duplicate work.
    """
    cursor = await _get_db().execute(
        """
        INSERT OR IGNORE INTO telegram_update_queue (update_id, payload)
        VALUES (?, ?)
        """,
        (update_id, payload),
    )
    await _get_db().commit()
    if cursor.rowcount == 1:
        if cursor.lastrowid is None:
            raise RuntimeError("INSERT did not return a row ID")
        return cursor.lastrowid, True

    async with _get_db().execute(
        "SELECT id FROM telegram_update_queue WHERE update_id = ?",
        (update_id,),
    ) as existing:
        row = await existing.fetchone()
    if row is None:
        raise RuntimeError("Telegram update enqueue did not insert or find a row")
    return row["id"], False


async def claim_next_telegram_update() -> TelegramUpdateQueueRow | None:
    """
    Atomically claim the oldest pending Telegram update for processing.

    The row transitions pending -> processing and attempt_count increments.
    Returns None when no pending work exists.
    """
    db = _get_db()
    await db.execute("BEGIN IMMEDIATE")
    try:
        async with db.execute(
            """
            SELECT id FROM telegram_update_queue
            WHERE status = 'pending'
            ORDER BY id
            LIMIT 1
            """
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            await db.commit()
            return None

        row_id = row["id"]
        await db.execute(
            """
            UPDATE telegram_update_queue
            SET status = 'processing',
                attempt_count = attempt_count + 1,
                locked_at = CURRENT_TIMESTAMP,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND status = 'pending'
            """,
            (row_id,),
        )
        async with db.execute(
            "SELECT * FROM telegram_update_queue WHERE id = ?",
            (row_id,),
        ) as claimed_cursor:
            claimed = await claimed_cursor.fetchone()
        if claimed is None:
            raise RuntimeError(f"Claimed Telegram update row {row_id} disappeared")
        await db.commit()
        return _telegram_update_queue_row(claimed)
    except Exception:
        await db.rollback()
        raise


async def complete_telegram_update(row_id: int) -> bool:
    """Mark a claimed Telegram update as done."""
    cursor = await _get_db().execute(
        """
        UPDATE telegram_update_queue
        SET status = 'done',
            processed_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ? AND status = 'processing'
        """,
        (row_id,),
    )
    await _get_db().commit()
    return cursor.rowcount > 0


async def retry_telegram_update(row_id: int, error: str) -> bool:
    """Return a claimed Telegram update to pending after processing failure."""
    cursor = await _get_db().execute(
        """
        UPDATE telegram_update_queue
        SET status = 'pending',
            last_error = ?,
            locked_at = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ? AND status = 'processing'
        """,
        (error, row_id),
    )
    await _get_db().commit()
    return cursor.rowcount > 0


async def discard_telegram_update(row_id: int, error: str) -> bool:
    """Mark a claimed Telegram update done after a permanent processing failure."""
    cursor = await _get_db().execute(
        """
        UPDATE telegram_update_queue
        SET status = 'done',
            last_error = ?,
            processed_at = CURRENT_TIMESTAMP,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = ? AND status = 'processing'
        """,
        (error, row_id),
    )
    await _get_db().commit()
    return cursor.rowcount > 0


async def requeue_processing_telegram_updates() -> int:
    """
    Requeue updates left in processing by a previous process crash.

    Called during webhook startup before the worker begins claiming rows.
    """
    cursor = await _get_db().execute(
        """
        UPDATE telegram_update_queue
        SET status = 'pending',
            locked_at = NULL,
            updated_at = CURRENT_TIMESTAMP
        WHERE status = 'processing'
        """
    )
    await _get_db().commit()
    return cursor.rowcount


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
        job_type: "reminder" (sends prompt as-is) or "agent" (processed by the selected backend).
            The legacy value "claude" is accepted and stored as "agent".
        prompt: Message text for reminders, or the agent prompt for agent-type jobs.
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
    canonical_job_type = normalize_job_type(job_type)
    cursor = await _get_db().execute(
        """INSERT INTO jobs (chat_id, name, job_type, prompt, schedule_type, schedule_data, auto_remove, notify_on_check)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            chat_id,
            name,
            canonical_job_type,
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
    a job from reminder to agent processing (or vice versa) is a fundamentally
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


async def delete_settings_by_prefix(prefix: str) -> None:
    """Remove all settings whose key starts with the given prefix.

    Uses SQL LIKE with the prefix escaped for literal matching (no
    wildcard characters in the prefix itself). No-op if no keys match.
    """
    # Escape any LIKE wildcards in the prefix so callers can pass
    # arbitrary strings without accidental pattern matching.
    escaped = prefix.replace("%", "\\%").replace("_", "\\_")
    await _get_db().execute(
        "DELETE FROM settings WHERE key LIKE ? ESCAPE '\\'",
        (escaped + "%",),
    )
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
    "timeout": "300"}). Values are strings; callers parse as needed.
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
    timeout = yaml_config.timeout if yaml_config else None
    env = dict(yaml_config.env) if yaml_config and yaml_config.env else None
    env_file = yaml_config.env_file if yaml_config else None
    system_prompt = yaml_config.system_prompt if yaml_config else None
    system_prompt_file = yaml_config.system_prompt_file if yaml_config else None
    path = yaml_config.path if yaml_config else workspace_path

    # Layer database overrides
    if "model" in db_settings:
        model = db_settings["model"]
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
# keys used by set_user_setting / get_user_settings.
_USER_SETTING_FIELDS = {"model", "timeout"}


async def get_user_settings(chat_id: int) -> dict[str, str]:
    """
    Get all per-user settings from the database.

    Returns a dict of field->value pairs (e.g., {"model": "opus",
    "timeout": "300"}). Values are strings; callers parse as needed.
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
    """Set a single per-user setting (e.g., model, timeout)."""
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
    timeout: int


async def resolve_user_defaults(
    chat_id: int,
    config: Config,
) -> UserDefaults:
    """
    Resolve per-user settings by layering DB overrides on top of
    users.yaml and env var defaults.

    Returns a UserDefaults dict with keys: model, timeout.
    All values are resolved - never None.

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

    # Model aliases are backend-specific. Resolve the user's backend
    # before returning a persisted preference so command handlers that
    # inspect settings prior to subprocess creation display the same
    # canonical ID the backend will receive.
    from kai.config import canonicalize_model_for_backend, get_user_backend_and_provider

    backend, _provider = get_user_backend_and_provider(user_config, config)

    # Model: DB > users.yaml > registry/env default.
    # Strip whitespace so "" and " " don't pass through as valid model
    # names. The UI validates before storing, but direct DB manipulation
    # could insert empty strings that cause confusing runtime errors.
    # After stripping, truthiness is safe: empty string and None both
    # fall through correctly.
    raw_db_model = db_settings.get("model")
    db_model = raw_db_model.strip() if raw_db_model is not None else None
    raw_yaml_model = user_config.model if user_config else None
    yaml_model = raw_yaml_model.strip() if raw_yaml_model is not None else None
    model = db_model if db_model else yaml_model if yaml_model else config.default_model
    model = canonicalize_model_for_backend(model, backend)

    # Timeout: DB > users.yaml > env > 120
    yaml_timeout = user_config.timeout if user_config and user_config.timeout is not None else None
    try:
        timeout = int(db_settings["timeout"]) if "timeout" in db_settings else None
    except (ValueError, TypeError):
        timeout = None
    if timeout is None:
        timeout = yaml_timeout if yaml_timeout is not None else config.default_timeout

    return {
        "model": model,
        "timeout": timeout,
    }


# ── Workspace history ────────────────────────────────────────────────


async def upsert_workspace_history(path: str, chat_id: int) -> None:
    """Record or refresh a workspace path in the user's history."""
    await _get_db().execute(
        "INSERT OR REPLACE INTO workspace_history (path, chat_id, last_used_at) VALUES (?, ?, CURRENT_TIMESTAMP)",
        (path, chat_id),
    )
    await _get_db().commit()


# ── Memory project registry (DB layer) ─────────────────────────────
# User-registered memory projects. The in-memory merge with the
# operator-pinned YAML registry lives in kai.memory_projects; these
# accessors are the persistence layer only.


async def register_memory_project(
    *,
    project_id: str,
    display_name: str,
    workspace_root: str,
    created_by: int,
    memory_enabled: bool = True,
    default_scope_for_new_facts: str | None = "project",
) -> None:
    """Insert a user-registered memory project row.

    Plain INSERT, not upsert: registration collisions are a user
    error surfaced by the command handler (which checks the merged
    registry first), and the PRIMARY KEY / UNIQUE constraints are
    the backstop against a race between two concurrent registrations.
    Raises sqlite's IntegrityError on collision; the caller maps it
    to a user-facing message.
    """
    await _get_db().execute(
        "INSERT INTO memory_projects (project_id, display_name, workspace_root, memory_enabled, default_scope_for_new_facts, created_by) VALUES (?, ?, ?, ?, ?, ?)",
        (
            project_id,
            display_name,
            workspace_root,
            1 if memory_enabled else 0,
            default_scope_for_new_facts,
            created_by,
        ),
    )
    await _get_db().commit()


async def unregister_memory_project(project_id: str) -> bool:
    """Delete a user-registered project row. Returns True when a row
    was actually removed, so the handler can distinguish "gone" from
    "never existed" without a second query."""
    cursor = await _get_db().execute("DELETE FROM memory_projects WHERE project_id = ?", (project_id,))
    await _get_db().commit()
    return cursor.rowcount > 0


async def get_memory_project_rows() -> list[dict]:
    """All user-registered project rows as plain dicts.

    Consumed by kai.memory_projects at startup (cache load) and by
    the /project list handler. Returns dicts rather than dataclass
    instances so validation stays in one place (the cache loader),
    matching the YAML path where the loader validates raw dicts.
    """
    async with _get_db().execute(
        "SELECT project_id, display_name, workspace_root, memory_enabled, default_scope_for_new_facts, created_by FROM memory_projects"
    ) as cursor:
        rows = await cursor.fetchall()
        return [
            {
                "project_id": row["project_id"],
                "display_name": row["display_name"],
                "workspace_root": row["workspace_root"],
                "memory_enabled": bool(row["memory_enabled"]),
                "default_scope_for_new_facts": row["default_scope_for_new_facts"],
                "created_by": row["created_by"],
            }
            for row in rows
        ]


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
    applied. The allowed list is the union of:
      1. The per-chat DB `allowed_workspaces` table (set via
         `/workspace allow` from Telegram).
      2. The per-user `allowed_workspaces` field in users.yaml
         (admin-set baseline; see issue #460).
      3. The global `Config.allowed_workspaces` (env var
         ALLOWED_WORKSPACES + workspaces.yaml entries).
    Deduplicated by resolved path; earlier tiers win on collision.

    Precedence for workspace_base:
        1. users.yaml workspace_base (admin-set per user)
        2. Global WORKSPACE_BASE env var

    Precedence for allowed_workspaces:
        DB > yaml-per-user > global. DB first so user-added
        workspaces appear at the top of the /workspaces keyboard
        and /workspace allowed list; yaml-per-user before global
        because the user-specific tier is more relevant to the
        user than the bot-wide tier.
    """
    user_config = config.get_user_config(chat_id)

    # workspace_base: users.yaml > env
    base = user_config.workspace_base if user_config and user_config.workspace_base else config.workspace_base

    # allowed_workspaces: union of DB + yaml-per-user + global,
    # deduplicated. Resolved paths are the dedup key so the same
    # directory expressed two different ways (symlink, relative
    # form) collapses to a single entry.
    db_allowed = await get_allowed_workspaces(chat_id)
    seen: set[Path] = set()
    combined: list[Path] = []

    for p in db_allowed:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            combined.append(resolved)

    # Per-user yaml allowed_workspaces (#460). The loader already
    # resolved each path at load time, but re-resolve here for
    # symmetry with the other tiers - cheap and protects against
    # a stale Path object if the loader is ever changed to keep
    # the raw form.
    if user_config and user_config.allowed_workspaces:
        for p in user_config.allowed_workspaces:
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


# ── GitHub settings resolution ──────────────────────────────────────


async def get_github_added_repos(chat_id: int) -> list[str]:
    """Return repos the user has added via /github add, or [].

    Values are stored as a JSON array of lowercase strings in the
    settings table under the key "github_repos_added:{chat_id}".
    """
    val = await get_setting(f"github_repos_added:{chat_id}")
    if val is None:
        return []
    try:
        repos = json.loads(val)
        if not isinstance(repos, list):
            raise ValueError("expected list")
        return repos
    except (json.JSONDecodeError, ValueError):
        log.warning("Corrupt github_repos_added for chat %d: %r", chat_id, val)
        return []


async def get_github_removed_repos(chat_id: int) -> list[str]:
    """Return repos the user has removed via /github remove, or [].

    Values are stored as a JSON array of lowercase strings in the
    settings table under the key "github_repos_removed:{chat_id}".
    """
    val = await get_setting(f"github_repos_removed:{chat_id}")
    if val is None:
        return []
    try:
        repos = json.loads(val)
        if not isinstance(repos, list):
            raise ValueError("expected list")
        return repos
    except (json.JSONDecodeError, ValueError):
        log.warning("Corrupt github_repos_removed for chat %d: %r", chat_id, val)
        return []


async def set_github_added_repos(chat_id: int, repos: list[str]) -> None:
    """Persist the user's added-repos list. Stores lowercase."""
    normalized = [r.lower() for r in repos]
    await set_setting(f"github_repos_added:{chat_id}", json.dumps(normalized))


async def set_github_removed_repos(chat_id: int, repos: list[str]) -> None:
    """Persist the user's removed-repos list. Stores lowercase."""
    normalized = [r.lower() for r in repos]
    await set_setting(f"github_repos_removed:{chat_id}", json.dumps(normalized))


async def get_effective_repos(chat_id: int, yaml_repos: list[str]) -> list[str]:
    """Compute the effective repo list for a user.

    Returns the union of yaml_repos and DB-added repos, minus DB-removed
    repos. All values are lowercased for case-insensitive matching.
    This is the single source of truth for the union/minus formula;
    both resolve_github_settings() and webhook._get_subscribed_users()
    call this function.
    """
    added = await get_github_added_repos(chat_id)
    removed = await get_github_removed_repos(chat_id)
    # Lowercase everything defensively. added/removed are stored
    # lowercase by the set_ helpers, but direct DB edits or migrations
    # could introduce mixed-case values.
    return sorted(
        (set(r.lower() for r in yaml_repos) | set(r.lower() for r in added)) - set(r.lower() for r in removed)
    )


async def get_github_db_settings(chat_id: int) -> dict[str, str]:
    """Read all GitHub-related DB overrides for a user.

    Returns a dict of key->value for settings that exist. All values
    are raw strings. Note: "github_repos_added" and "github_repos_removed"
    are JSON-encoded arrays (e.g., '["owner/repo"]'), not parsed lists.
    Use get_github_added_repos()/get_github_removed_repos() for parsed
    access, or get_effective_repos() for the resolved list.

    Keys: "pr_review", "issue_triage", "github_notify_chat",
    "github_repos_added", "github_repos_removed".
    """
    result: dict[str, str] = {}
    for key in (
        "pr_review",
        "issue_triage",
        "github_notify_chat",
        "github_repos_added",
        "github_repos_removed",
    ):
        val = await get_setting(f"{key}:{chat_id}")
        if val is not None:
            result[key] = val
    return result


class GitHubSettings(TypedDict):
    """Resolved per-user GitHub notification settings."""

    repos: list[str]
    notify_chat_id: int
    pr_review: bool
    issue_triage: bool


async def resolve_github_settings(chat_id: int, config: Config) -> GitHubSettings:
    """
    Resolve per-user GitHub settings by layering DB overrides on top
    of users.yaml.

    Precedence (highest to lowest):
    1. Database (user-set via /github commands)
    2. users.yaml (admin baseline per user)
    3. Hardcoded defaults (False / chat_id)
    """
    user_config = config.get_user_config(chat_id)
    db = await get_github_db_settings(chat_id)

    # Repos: yaml baseline + DB-added - DB-removed (#220).
    yaml_repos = user_config.github_repos if user_config else []
    repos = await get_effective_repos(chat_id, yaml_repos)

    # Notification destination: DB > yaml > telegram_id.
    # Defensive try/except matches timeout above.
    # A corrupt DB value falls through to yaml/default rather than
    # aborting the entire resolution.
    notify: int | None = None
    if "github_notify_chat" in db:
        try:
            notify = int(db["github_notify_chat"])
        except (ValueError, TypeError):
            log.warning(
                "Corrupt github_notify_chat in DB for chat %d: %r (ignoring)",
                chat_id,
                db["github_notify_chat"],
            )
    if notify is None:
        if user_config and user_config.github_notify_chat_id is not None:
            notify = user_config.github_notify_chat_id
        else:
            # No per-user routing configured; deliver notifications to
            # the user's own DM (telegram_id == chat_id for private chats).
            notify = chat_id

    # PR review: DB > yaml > False
    if "pr_review" in db:
        pr_review = isinstance(db["pr_review"], str) and db["pr_review"].lower() == "true"
    elif user_config and user_config.pr_review is not None:
        pr_review = user_config.pr_review
    else:
        pr_review = False

    # Issue triage: DB > yaml > False
    if "issue_triage" in db:
        issue_triage = isinstance(db["issue_triage"], str) and db["issue_triage"].lower() == "true"
    elif user_config and user_config.issue_triage is not None:
        issue_triage = user_config.issue_triage
    else:
        issue_triage = False

    return {
        "repos": repos,
        "notify_chat_id": notify,
        "pr_review": pr_review,
        "issue_triage": issue_triage,
    }


# ── Lifecycle ────────────────────────────────────────────────────────


async def close_db() -> None:
    """Close the database connection. Called during shutdown from main.py."""
    global _db, _workshop_event_lock
    if _db:
        try:
            await _get_db().close()
        finally:
            # Clear even if close() raises so subsequent _get_db() calls
            # get a clear RuntimeError instead of using a broken connection
            _db = None
    _workshop_event_lock = None
