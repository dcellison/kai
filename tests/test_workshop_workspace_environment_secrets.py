"""Canonical write-only workspace environment secret authority tests."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from kai import sessions
from kai.workshop.bootstrap import BootstrapHuman
from kai.workshop.diagnostics import workshop_workspace_environment_secret_status
from tests.workshop_profiles import profile_id, profile_registry


@pytest.fixture
async def database(tmp_path: Path):
    path = tmp_path / "kai.db"
    yield path
    await sessions.close_db()


async def test_workspace_secret_audit_contains_metadata_but_never_values(database: Path) -> None:
    await sessions.init_db(database)
    await sessions.bootstrap_workshop_foundation(
        (
            BootstrapHuman(
                "Human 101",
                "admin",
                "telegram",
                "101",
                "101",
                profile_id(101),
            ),
        )
    )
    registry, _migration = await sessions.initialize_workshop_execution_state(
        profile_registry(101),
    )
    namespace = registry.namespaces[0]
    secret = "q1577-secret-value-must-not-enter-audit"

    await sessions.record_workspace_environment_secret_audit(
        namespace,
        workspace_digest="a" * 64,
        environment_key="SERVICE_TOKEN",
        operation="set",
        changed=True,
        runtime_action="restarted",
    )

    connection = sqlite3.connect(database)
    try:
        columns = {
            str(row[1])
            for row in connection.execute("PRAGMA table_info(workspace_environment_secret_audit)").fetchall()
        }
        row = connection.execute(
            "SELECT environment_key, operation, changed, runtime_action FROM workspace_environment_secret_audit"
        ).fetchone()
    finally:
        connection.close()

    assert "value" not in columns
    assert "secret" not in columns
    assert row == ("SERVICE_TOKEN", "set", 1, "restarted")
    assert secret.encode() not in database.read_bytes()
    assert workshop_workspace_environment_secret_status(database).startswith(
        "Workshop workspace environment secrets: active; workspace states=0, "
        "principal keys=0, audits=1 (set=1, remove=0), malformed=0, integrity gaps=0"
    )


def test_workspace_secret_status_is_pending_before_schema_exists(tmp_path: Path) -> None:
    assert workshop_workspace_environment_secret_status(tmp_path / "missing.db") == (
        "Workshop workspace environment secrets: pending; canonical workspace-secret schema unavailable"
    )
