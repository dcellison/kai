from __future__ import annotations

from pathlib import Path

import aiosqlite

_db: aiosqlite.Connection | None = None


async def init_db(db_path: Path) -> None:
    global _db
    _db = await aiosqlite.connect(str(db_path))
    _db.row_factory = aiosqlite.Row
    await _db.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            chat_id INTEGER PRIMARY KEY,
            session_id TEXT NOT NULL,
            model TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_used_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            total_cost_usd REAL DEFAULT 0.0
        )
    """)
    await _db.execute("""
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
            auto_remove INTEGER DEFAULT 0
        )
    """)
    await _db.commit()


async def get_session(chat_id: int) -> str | None:
    async with _db.execute(
        "SELECT session_id FROM sessions WHERE chat_id = ?", (chat_id,)
    ) as cursor:
        row = await cursor.fetchone()
        return row["session_id"] if row else None


async def save_session(chat_id: int, session_id: str, model: str, cost_usd: float) -> None:
    await _db.execute("""
        INSERT INTO sessions (chat_id, session_id, model, total_cost_usd)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(chat_id) DO UPDATE SET
            session_id = excluded.session_id,
            model = excluded.model,
            last_used_at = CURRENT_TIMESTAMP,
            total_cost_usd = total_cost_usd + excluded.total_cost_usd
    """, (chat_id, session_id, model, cost_usd))
    await _db.commit()


async def clear_session(chat_id: int) -> None:
    await _db.execute("DELETE FROM sessions WHERE chat_id = ?", (chat_id,))
    await _db.commit()


async def get_stats(chat_id: int) -> dict | None:
    async with _db.execute(
        "SELECT session_id, model, created_at, last_used_at, total_cost_usd FROM sessions WHERE chat_id = ?",
        (chat_id,),
    ) as cursor:
        row = await cursor.fetchone()
        if not row:
            return None
        return dict(row)


async def create_job(
    chat_id: int, name: str, job_type: str, prompt: str,
    schedule_type: str, schedule_data: str, auto_remove: bool = False,
) -> int:
    cursor = await _db.execute(
        """INSERT INTO jobs (chat_id, name, job_type, prompt, schedule_type, schedule_data, auto_remove)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        (chat_id, name, job_type, prompt, schedule_type, schedule_data, int(auto_remove)),
    )
    await _db.commit()
    return cursor.lastrowid


async def get_jobs(chat_id: int) -> list[dict]:
    async with _db.execute(
        "SELECT id, name, job_type, prompt, schedule_type, schedule_data, auto_remove, created_at FROM jobs WHERE chat_id = ? AND active = 1",
        (chat_id,),
    ) as cursor:
        rows = await cursor.fetchall()
        return [{**dict(r), "auto_remove": bool(r["auto_remove"])} for r in rows]


async def get_all_active_jobs() -> list[dict]:
    async with _db.execute(
        "SELECT id, chat_id, name, job_type, prompt, schedule_type, schedule_data, auto_remove FROM jobs WHERE active = 1"
    ) as cursor:
        rows = await cursor.fetchall()
        return [{**dict(r), "auto_remove": bool(r["auto_remove"])} for r in rows]


async def delete_job(job_id: int) -> bool:
    cursor = await _db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
    await _db.commit()
    return cursor.rowcount > 0


async def deactivate_job(job_id: int) -> None:
    await _db.execute("UPDATE jobs SET active = 0 WHERE id = ?", (job_id,))
    await _db.commit()


async def close_db() -> None:
    global _db
    if _db:
        await _db.close()
        _db = None
