"""
The installed memory qualification command.

Builds a protected install with the real-path harness, lets the query
service record what it records at startup (the legacy census and the
vector audit), and runs the command's checks against the database the way
the operator does. Each failure case changes exactly one thing the
qualification must notice.
"""

# ruff: noqa: F811 - tests take the imported `protected` fixture as a parameter by name.

from __future__ import annotations

import sqlite3
from pathlib import Path

from kai.workshop import memory_qualification
from tests.memory_fixtures import (  # noqa: F401 - pytest fixture import
    CHANNEL_ID,
    RUNTIME_ID,
    _query_service,
    protected,
)


async def _started(store, tmp_path: Path) -> None:
    """Store a fact, then record the census and vector audit the way service startup does."""
    service, authority = _query_service(store, tmp_path)
    await service.create_fact(
        authority,
        content="The operator prefers dark themes.",
        tags=(),
        scope="global",
        project_id=None,
        request_id="create-1",
    )
    await store.connection.execute(
        "INSERT INTO channel_agent_execution_settings (channel_id, agent_id, runtime_profile_id, field, value, "
        "updated_at) VALUES (?, 'agt_23000000000000000000000000000001', ?, 'model', 'gpt-5.5', '2026-01-01T00:00:00Z')",
        (CHANNEL_ID, str(RUNTIME_ID)),
    )
    await store.connection.commit()
    assert await service.refresh_legacy_census() == 1
    assert await service.refresh_vector_audits() == 1


def _failed(checks) -> list[str]:
    return [check.name for check in checks if not check.passed]


async def test_a_consistent_install_passes_and_records_defaults_once(protected, tmp_path: Path, capsys) -> None:
    store, _provider, _lifecycle = protected
    await _started(store, tmp_path)
    snapshot = tmp_path / "defaults.json"
    arguments = ["--db", str(tmp_path / "kai.db"), "--snapshot", str(snapshot)]

    first = memory_qualification.main(arguments)
    second = memory_qualification.main(arguments)

    output = capsys.readouterr().out
    assert (first, second) == (0, 0)
    assert "PASS conversational defaults: recorded 1 setting(s)" in output
    assert "PASS conversational defaults: 1 setting(s) unchanged" in output
    assert output.count("memory qualification: PASSED") == 2
    # Output names checks and counts, never memory content.
    assert "dark themes" not in output


async def test_a_changed_default_fails(protected, tmp_path: Path) -> None:
    store, _provider, _lifecycle = protected
    await _started(store, tmp_path)
    snapshot = tmp_path / "defaults.json"
    memory_qualification.qualify(tmp_path / "kai.db", snapshot)
    await store.connection.execute("UPDATE channel_agent_execution_settings SET value = 'gpt-5.6'")
    await store.connection.commit()

    checks = memory_qualification.qualify(tmp_path / "kai.db", snapshot)

    assert _failed(checks) == ["conversational defaults"]


async def test_an_audit_older_than_the_latest_projection_fails(protected, tmp_path: Path) -> None:
    store, _provider, _lifecycle = protected
    await _started(store, tmp_path)
    connection = sqlite3.connect(tmp_path / "kai.db")
    connection.execute("UPDATE memory_vector_audit SET checked_at = '2000-01-01T00:00:00+00:00'")
    connection.commit()
    connection.close()

    checks = memory_qualification.qualify(tmp_path / "kai.db", tmp_path / "defaults.json")

    assert _failed(checks) == ["vector audit"]


async def test_a_counted_owner_without_an_audit_fails(protected, tmp_path: Path) -> None:
    store, _provider, _lifecycle = protected
    await _started(store, tmp_path)
    connection = sqlite3.connect(tmp_path / "kai.db")
    connection.execute("DELETE FROM memory_vector_audit")
    connection.commit()
    connection.close()

    checks = memory_qualification.qualify(tmp_path / "kai.db", tmp_path / "defaults.json")

    # The status line also reports the missing audit as a gap.
    assert _failed(checks) == ["current truth", "vector audit"]


async def test_a_failed_projection_fails_the_status_check(protected, tmp_path: Path) -> None:
    store, _provider, _lifecycle = protected
    await _started(store, tmp_path)
    await store.connection.execute("UPDATE memory_fact_vector_operations SET status = 'failed'")
    await store.connection.commit()

    checks = memory_qualification.qualify(tmp_path / "kai.db", tmp_path / "defaults.json")

    assert "current truth" in _failed(checks)


def test_a_missing_database_is_reported(tmp_path: Path, capsys) -> None:
    code = memory_qualification.main(["--db", str(tmp_path / "absent.db"), "--snapshot", str(tmp_path / "s.json")])

    assert code == 1
    assert "no database" in capsys.readouterr().err
