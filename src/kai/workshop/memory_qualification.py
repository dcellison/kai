"""
Installed qualification of temporal memory.

Run by the operator on an installed Kai with memory enabled, while the
service is running:

    python -m kai.workshop.memory_qualification --db <DATA_DIR>/kai.db --snapshot <file>

It only reads the database (opened read-only) and never the vector store:
the service holds the store, and it records what install status needs
(the legacy census and the vector audit) at startup and after projection
retries. The qualification passes when all of these hold:

1. Every memory diagnostic line that install status prints is `active`:
   current truth, reconciliation, episode history, and extraction
   receipts. Each line folds in its own integrity, replay, projection,
   and drift gaps, so `active` means all of them are zero. Owners who
   never ran a reconciliation audit only show as "legacy review pending",
   which is not a gap.
2. The stored vector audit is complete and fresh: every owner the service
   counted has an audit, and the oldest audit is newer than the latest
   vector projection. A stale audit says nothing about the store as it is
   now; restarting the service or running "Check search index" in
   Workshop refreshes it.
3. No memory operation changed conversational defaults. The first run
   records every backend and model setting in `--snapshot`; each later
   run compares against it, so running it at the start and end of a
   qualification window proves the window changed none of them.

Exit status is 0 when every check passes and 1 otherwise. Output is one
line per check and never includes memory content.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kai.workshop.diagnostics import (
    workshop_episode_history_status,
    workshop_memory_current_truth_status,
    workshop_memory_extraction_receipt_status,
    workshop_memory_reconciliation_status,
)

# The execution settings that are conversational defaults. Workspace and
# timeout are left out: an owner may legitimately change them during a
# qualification window, and neither is something memory could set.
_DEFAULT_FIELDS = ("backend", "model")


@dataclass(frozen=True, slots=True)
class QualificationCheck:
    """One qualification check and its outcome, safe to print."""

    name: str
    passed: bool
    detail: str


def _status_checks(db_path: Path) -> list[QualificationCheck]:
    """Require every memory diagnostic line to report `active`."""
    lines = (
        ("current truth", workshop_memory_current_truth_status(db_path, memory_enabled=True)),
        ("reconciliation", workshop_memory_reconciliation_status(db_path)),
        ("episode history", workshop_episode_history_status(db_path, memory_enabled=True)),
        ("extraction receipts", workshop_memory_extraction_receipt_status(db_path)),
    )
    checks: list[QualificationCheck] = []
    for name, line in lines:
        # Each line reads "<prefix>: <state>; ...". The state is the word
        # right after the prefix.
        state = line.split(":", 1)[1].strip().split(";", 1)[0].split(" ", 1)[0] if ":" in line else ""
        checks.append(QualificationCheck(name, state == "active", line))
    return checks


def _parse_time(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _audit_check(connection: sqlite3.Connection) -> QualificationCheck:
    """Require a stored vector audit for every counted owner, newer than the latest projection."""
    tables = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    if "memory_vector_audit" not in tables or "memory_legacy_census" not in tables:
        return QualificationCheck("vector audit", False, "the stored audit or census is missing; restart the service")
    counted = {
        (str(row[0]), str(row[1]))
        for row in connection.execute("SELECT principal_id, runtime_profile_id FROM memory_legacy_census")
    }
    audited = {
        (str(row[0]), str(row[1])): _parse_time(row[2])
        for row in connection.execute("SELECT principal_id, runtime_profile_id, checked_at FROM memory_vector_audit")
    }
    missing = counted - set(audited)
    if not audited or missing:
        return QualificationCheck(
            "vector audit",
            False,
            f"{len(missing) or len(counted)} owner(s) have no stored vector audit; restart the service",
        )
    projected: list[datetime] = []
    for table in ("memory_fact_vector_operations", "memory_episode_vector_operations"):
        if table not in tables:
            continue
        row = connection.execute(f"SELECT MAX(updated_at) FROM {table} WHERE status = 'succeeded'").fetchone()
        if row is not None and row[0] is not None:
            projected.append(_parse_time(row[0]))
    oldest = min(audited.values())
    latest = max(projected) if projected else None
    if latest is not None and oldest < latest:
        return QualificationCheck(
            "vector audit",
            False,
            f"the oldest audit ({oldest.isoformat(timespec='seconds')}) predates the latest projection "
            f"({latest.isoformat(timespec='seconds')}); restart the service or run Check search index",
        )
    return QualificationCheck("vector audit", True, f"{len(audited)} owner(s) audited since the latest projection")


def _defaults(connection: sqlite3.Connection) -> list[list[str]]:
    """Every conversational backend and model setting, in a stable order."""
    tables = {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    rows: list[list[str]] = []
    if "channel_agent_execution_settings" in tables:
        placeholders = ", ".join("?" for _ in _DEFAULT_FIELDS)
        rows.extend(
            ["execution", str(row[0]), str(row[1]), str(row[2]), str(row[3]), str(row[4])]
            for row in connection.execute(
                "SELECT channel_id, agent_id, runtime_profile_id, field, value FROM channel_agent_execution_settings "
                f"WHERE field IN ({placeholders}) ORDER BY channel_id, agent_id, runtime_profile_id, field",
                _DEFAULT_FIELDS,
            )
        )
    if "settings" in tables:
        # Per-user compatibility settings are keyed "<field>:<owner>".
        rows.extend(
            ["setting", str(row[0]), str(row[1])]
            for row in connection.execute(
                "SELECT key, value FROM settings WHERE key LIKE 'model:%' OR key LIKE 'backend:%' ORDER BY key"
            )
        )
    return rows


def _defaults_check(connection: sqlite3.Connection, snapshot: Path) -> QualificationCheck:
    """Record the defaults on the first run; compare against that record afterwards."""
    current = _defaults(connection)
    if not snapshot.exists():
        snapshot.write_text(
            json.dumps({"recorded_at": datetime.now(UTC).isoformat(timespec="seconds"), "defaults": current}, indent=2)
            + "\n",
            encoding="utf-8",
        )
        return QualificationCheck(
            "conversational defaults", True, f"recorded {len(current)} setting(s); run again to compare"
        )
    recorded = json.loads(snapshot.read_text(encoding="utf-8"))
    before = recorded.get("defaults")
    if before == current:
        return QualificationCheck(
            "conversational defaults",
            True,
            f"{len(current)} setting(s) unchanged since {recorded.get('recorded_at', 'the first run')}",
        )
    changed = len({json.dumps(item) for item in before or []} ^ {json.dumps(item) for item in current})
    return QualificationCheck(
        "conversational defaults", False, f"{changed} setting row(s) differ from the recorded snapshot"
    )


def qualify(db_path: Path, snapshot: Path) -> list[QualificationCheck]:
    """Run every installed qualification check against one Kai database."""
    checks = _status_checks(db_path)
    connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        connection.execute("PRAGMA query_only=ON")
        checks.append(_audit_check(connection))
        checks.append(_defaults_check(connection, snapshot))
    finally:
        connection.close()
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m kai.workshop.memory_qualification",
        description="Qualify temporal memory on an installed Kai (read-only).",
    )
    parser.add_argument("--db", type=Path, required=True, help="the installed Kai database (DATA_DIR/kai.db)")
    parser.add_argument(
        "--snapshot",
        type=Path,
        required=True,
        help="file recording conversational defaults; written on the first run, compared on later runs",
    )
    args = parser.parse_args(argv)
    if not args.db.is_file():
        print(f"memory qualification: no database at {args.db}", file=sys.stderr)
        return 1
    checks = qualify(args.db, args.snapshot)
    for check in checks:
        print(f"{'PASS' if check.passed else 'FAIL'} {check.name}: {check.detail}")
    passed = all(check.passed for check in checks)
    print(f"memory qualification: {'PASSED' if passed else 'FAILED'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
