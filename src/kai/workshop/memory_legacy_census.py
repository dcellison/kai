"""
Per-owner count of legacy memory rows that reconciliation has not settled.

Install status reports how many legacy rows are still unclassified, but it
runs while the service holds the embedded vector store, so it cannot read
the rows itself. The service (at startup) and reconciliation apply (after
every run, from the service or the admin CLI) count them here instead,
and store one row per owner in `memory_legacy_census`. Install status reads
that row and shows how old it is.

A legacy row is a vector row with no canonical claim, revision, or episode
id. It is settled when canonical memory cites it as `legacy` evidence
(absorbed: adopted facts rewritten in place stop being legacy rows at all,
while recorded episodes, consolidation siblings, and retired duplicates
stay as cited legacy rows), or when any review of the owner's, open or
applied, rejected it. Everything else, deferred rows included, is
unclassified. Only content-free counts are stored.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kai import memory
from kai.workshop.memory_current_truth import (
    CANONICAL_CLAIM_ID_KEY,
    CANONICAL_EPISODE_ID_KEY,
    CANONICAL_REVISION_ID_KEY,
    absorbed_legacy_ids,
)


@dataclass(frozen=True, slots=True)
class LegacyCensus:
    """Content-free counts for one owner's legacy rows."""

    principal_id: str
    runtime_profile_id: str
    legacy_rows: int
    absorbed: int
    rejected: int
    unclassified: int
    counted_at: str


def _legacy_ids(rows: Iterable[memory.MemoryResult]) -> set[str]:
    return {
        row.id
        for row in rows
        if not row.metadata.get(CANONICAL_CLAIM_ID_KEY)
        and not row.metadata.get(CANONICAL_REVISION_ID_KEY)
        and not row.metadata.get(CANONICAL_EPISODE_ID_KEY)
    }


def _evidence_ids(document: object, *, collection: str, id_key: str, wanted: set[str]) -> set[str]:
    """Collect evidence memory ids of the wanted groups or candidates in a stored document."""
    parsed = json.loads(str(document))
    members: set[str] = set()
    for entry in parsed.get(collection, []):
        if isinstance(entry, dict) and entry.get(id_key) in wanted:
            members.update(str(item["memory_id"]) for item in entry.get("evidence", []))
    return members


def rejected_legacy_ids(connection: sqlite3.Connection, *, principal_id: str, runtime_profile_id: str) -> set[str]:
    """
    Return every memory id any of the owner's reviews rejected, open or applied.

    Unlike the retrieval gate, which only needs open reviews (legacy
    admission ends once a review is applied), the census counts a rejected
    row as settled for good.
    """
    rejected: set[str] = set()
    plans = connection.execute(
        "SELECT p.plan_id, p.plan_json FROM memory_reconciliation_triage_plans p "
        "JOIN memory_reconciliation_audits a ON a.audit_id = p.audit_id "
        "WHERE a.principal_id = ? AND a.runtime_profile_id = ?",
        (principal_id, runtime_profile_id),
    ).fetchall()
    for plan_id, plan_json in plans:
        groups = {
            str(row[0])
            for row in connection.execute(
                "SELECT group_id FROM memory_reconciliation_triage_groups WHERE plan_id = ? AND disposition = 'reject'",
                (plan_id,),
            ).fetchall()
        }
        if groups:
            rejected |= _evidence_ids(plan_json, collection="groups", id_key="group_id", wanted=groups)
    audits = connection.execute(
        "SELECT audit_id, audit_json FROM memory_reconciliation_audits WHERE principal_id = ? AND runtime_profile_id = ?",
        (principal_id, runtime_profile_id),
    ).fetchall()
    for audit_id, audit_json in audits:
        candidates = {
            str(row[0])
            for row in connection.execute(
                "SELECT candidate_id FROM memory_reconciliation_decisions WHERE audit_id = ? AND disposition = 'reject'",
                (audit_id,),
            ).fetchall()
        }
        if candidates:
            rejected |= _evidence_ids(audit_json, collection="candidates", id_key="candidate_id", wanted=candidates)
    return rejected


def count_legacy_census(
    db_path: Path,
    rows: Iterable[memory.MemoryResult],
    *,
    principal_id: str,
    runtime_profile_id: str,
    now: datetime | None = None,
) -> LegacyCensus:
    """
    Count one owner's legacy rows from `rows` and store the census.

    `rows` must be the owner's lifecycle-visible vector rows, read by the
    caller that holds the vector store. Absorbed and rejected counts are
    limited to ids that are actually present, so a cited row that was
    since deleted does not make the unclassified count negative.
    """
    legacy = _legacy_ids(rows)
    connection = sqlite3.connect(str(db_path), timeout=30)
    try:
        absorbed = (
            set(absorbed_legacy_ids(connection, principal_id=principal_id, runtime_profile_id=runtime_profile_id))
            & legacy
        )
        rejected = (
            rejected_legacy_ids(connection, principal_id=principal_id, runtime_profile_id=runtime_profile_id) & legacy
        )
        census = LegacyCensus(
            principal_id=principal_id,
            runtime_profile_id=runtime_profile_id,
            legacy_rows=len(legacy),
            absorbed=len(absorbed),
            rejected=len(rejected),
            unclassified=len(legacy - absorbed - rejected),
            counted_at=(now or datetime.now(UTC)).isoformat(timespec="seconds"),
        )
        connection.execute(
            "INSERT INTO memory_legacy_census ("
            "principal_id, runtime_profile_id, legacy_rows, absorbed, rejected, unclassified, counted_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (principal_id, runtime_profile_id) DO UPDATE SET legacy_rows = excluded.legacy_rows, "
            "absorbed = excluded.absorbed, rejected = excluded.rejected, unclassified = excluded.unclassified, "
            "counted_at = excluded.counted_at",
            (
                census.principal_id,
                census.runtime_profile_id,
                census.legacy_rows,
                census.absorbed,
                census.rejected,
                census.unclassified,
                census.counted_at,
            ),
        )
        connection.commit()
    finally:
        connection.close()
    return census


def refresh_legacy_census(db_path: Path, *, principal_id: str, runtime_profile_id: str) -> LegacyCensus:
    """
    Read the owner's vector rows and store a fresh census.

    Call only from a process that holds the vector store (the service, or
    the admin CLI with the service stopped). A read failure propagates;
    the previous census stays in place with its older timestamp.
    """
    rows = memory.get_all_for_lifecycle_projection(user_id=principal_id, runtime_profile_id=runtime_profile_id)
    return count_legacy_census(db_path, rows, principal_id=principal_id, runtime_profile_id=runtime_profile_id)


__all__ = ["LegacyCensus", "count_legacy_census", "refresh_legacy_census", "rejected_legacy_ids"]
