from __future__ import annotations

import aiosqlite
from pathlib import Path

_db: aiosqlite.Connection | None = None


async def init_db(db_path: Path) -> None:
    global _db
    _db = await aiosqlite.connect(str(db_path))
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
    await _db.commit()


async def get_session(chat_id: int) -> str | None:
    async with _db.execute(
        "SELECT session_id FROM sessions WHERE chat_id = ?", (chat_id,)
    ) as cursor:
        row = await cursor.fetchone()
        return row[0] if row else None


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
        return {
            "session_id": row[0],
            "model": row[1],
            "created_at": row[2],
            "last_used_at": row[3],
            "total_cost_usd": row[4],
        }


async def close_db() -> None:
    global _db
    if _db:
        await _db.close()
        _db = None
