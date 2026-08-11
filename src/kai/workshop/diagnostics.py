"""Non-secret operator diagnostics for Workshop bootstrap state."""

from __future__ import annotations

import sqlite3
from pathlib import Path

_REQUIRED_TABLES = {
    "workshops",
    "principals",
    "agents",
    "channel_bindings",
    "projection_checkpoints",
}


def _pending_status(expected_humans: int | None) -> str:
    if expected_humans is None:
        return "Workshop bootstrap: pending; configured-human count unavailable"
    return (
        "Workshop bootstrap: pending; service startup will seed "
        f"1 workshop, {expected_humans} human principal(s), "
        f"{expected_humans} Telegram direct channel binding(s), and 1 Kai agent"
    )


def _scalar(connection: sqlite3.Connection, query: str, parameters: tuple[object, ...] = ()) -> int:
    row = connection.execute(query, parameters).fetchone()
    if row is None:
        return 0
    return int(row[0])


def workshop_bootstrap_status(db_path: Path, *, expected_humans: int | None) -> str:
    """Describe canonical bootstrap state without exposing identity data."""
    if not db_path.is_file():
        return _pending_status(expected_humans)

    try:
        connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            rows = connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            tables = {str(row[0]) for row in rows}
            if not tables >= _REQUIRED_TABLES:
                return _pending_status(expected_humans)

            workshop_count = _scalar(connection, "SELECT COUNT(*) FROM workshops")
            human_count = _scalar(connection, "SELECT COUNT(*) FROM principals WHERE kind = 'human'")
            agent_count = _scalar(connection, "SELECT COUNT(*) FROM agents")
            telegram_binding_count = _scalar(
                connection,
                "SELECT COUNT(*) FROM channel_bindings WHERE transport = ?",
                ("telegram",),
            )
            projection_count = _scalar(
                connection,
                "SELECT COUNT(*) FROM projection_checkpoints WHERE name = ?",
                ("canonical_conversations",),
            )
        finally:
            connection.close()
    except (OSError, sqlite3.Error) as exc:
        return f"Workshop bootstrap: NOT VERIFIED ({type(exc).__name__})"

    expected_state_present = expected_humans is None or (
        human_count >= expected_humans and telegram_binding_count >= expected_humans
    )
    initialized = workshop_count >= 1 and agent_count >= 1 and projection_count == 1 and expected_state_present
    state = "initialized" if initialized else "pending"
    expectation = (
        "configured-human count unavailable" if expected_humans is None else f"expected humans={expected_humans}"
    )
    return (
        f"Workshop bootstrap: {state}; workshops={workshop_count}, humans={human_count}, "
        f"Telegram bindings={telegram_binding_count}, agents={agent_count}; {expectation}"
    )
