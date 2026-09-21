"""Read-only current-truth projection for canonical semantic memory."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kai.memory import MemoryResult

CANONICAL_CLAIM_ID_KEY = "canonical_memory_claim_id"
CANONICAL_REVISION_ID_KEY = "canonical_memory_revision_id"
CANONICAL_LIFECYCLE_STATE_KEY = "canonical_memory_lifecycle_state"
CANONICAL_TEMPORAL_ROLE_KEY = "canonical_memory_temporal_role"

_REQUIRED_TABLES = {
    "memory_fact_claims",
    "memory_fact_revisions",
    "memory_fact_revision_states",
    "memory_fact_lifecycle_events",
    "memory_fact_vector_operations",
}
_ADMITTED_MIGRATION_CLASSES = {"canonical", "legacy_complete"}


@dataclass(frozen=True, slots=True)
class CurrentTruthProjection:
    """Admitted rows and privacy-safe exclusion accounting."""

    rows: tuple[MemoryResult, ...]
    excluded: dict[str, int]


@dataclass(frozen=True, slots=True)
class _CanonicalRevision:
    claim_id: str
    revision_id: str
    content: str
    scope_kind: str
    scope_key: str
    state: str
    valid_from: str | None
    valid_until: str | None
    migration_classification: str
    vector_metadata: dict[str, Any]
    source_receipt_id: str | None
    source_run_id: str | None
    source_message_id: str | None
    result_message_id: str | None
    backend: str | None
    provider: str | None
    model: str | None
    prompt_version: str | None
    schema_version: str | None
    operation_revision_id: str | None
    operation_status: str | None
    operation: str | None
    memory_id: str | None


def _read_only_connection(db_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"{db_path.resolve().as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only=ON")
    return connection


def _tables(connection: sqlite3.Connection) -> set[str]:
    return {str(row[0]) for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()}


def _parse_timestamp(value: str | None) -> datetime | None:
    if value is None:
        return None
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("canonical memory timestamp is not timezone-aware")
    return parsed.astimezone(UTC)


def _load_revision(
    connection: sqlite3.Connection,
    *,
    claim_id: str,
    revision_id: str,
    principal_id: str,
    runtime_profile_id: str,
) -> _CanonicalRevision | None:
    row = connection.execute(
        "SELECT r.claim_id, r.revision_id, r.content, c.scope_kind, c.scope_key, s.state, "
        "r.valid_from, r.valid_until, r.migration_classification, r.vector_metadata_json, "
        "r.source_receipt_id, r.source_run_id, r.source_message_id, r.result_message_id, "
        "r.backend, r.provider, r.model, r.prompt_version, r.schema_version, "
        "v.revision_id AS operation_revision_id, v.status AS operation_status, "
        "v.operation, v.memory_id "
        "FROM memory_fact_claims c "
        "JOIN memory_fact_revisions r ON r.claim_id = c.claim_id "
        "JOIN memory_fact_revision_states s "
        "ON s.claim_id = r.claim_id AND s.revision_id = r.revision_id "
        "LEFT JOIN memory_fact_vector_operations v ON v.event_position = ("
        "SELECT MAX(latest.event_position) FROM memory_fact_vector_operations latest "
        "WHERE latest.claim_id = c.claim_id) "
        "WHERE c.claim_id = ? AND r.revision_id = ? "
        "AND c.owner_principal_id = ? AND c.runtime_profile_id = ?",
        (claim_id, revision_id, principal_id, runtime_profile_id),
    ).fetchone()
    if row is None:
        return None
    metadata = json.loads(str(row[9]))
    if not isinstance(metadata, dict):
        raise ValueError("canonical vector metadata is not an object")
    return _CanonicalRevision(
        claim_id=str(row[0]),
        revision_id=str(row[1]),
        content=str(row[2]),
        scope_kind=str(row[3]),
        scope_key=str(row[4]),
        state=str(row[5]),
        valid_from=str(row[6]) if row[6] is not None else None,
        valid_until=str(row[7]) if row[7] is not None else None,
        migration_classification=str(row[8]),
        vector_metadata=metadata,
        source_receipt_id=str(row[10]) if row[10] is not None else None,
        source_run_id=str(row[11]) if row[11] is not None else None,
        source_message_id=str(row[12]) if row[12] is not None else None,
        result_message_id=str(row[13]) if row[13] is not None else None,
        backend=str(row[14]) if row[14] is not None else None,
        provider=str(row[15]) if row[15] is not None else None,
        model=str(row[16]) if row[16] is not None else None,
        prompt_version=str(row[17]) if row[17] is not None else None,
        schema_version=str(row[18]) if row[18] is not None else None,
        operation_revision_id=str(row[19]) if row[19] is not None else None,
        operation_status=str(row[20]) if row[20] is not None else None,
        operation=str(row[21]) if row[21] is not None else None,
        memory_id=str(row[22]) if row[22] is not None else None,
    )


def _project_row(
    row: MemoryResult,
    revision: _CanonicalRevision,
    *,
    now: datetime,
) -> tuple[MemoryResult | None, str | None]:
    if revision.state != "active":
        return None, revision.state if revision.state in {
            "superseded",
            "retracted",
            "expired",
            "unresolved_conflict",
        } else "invalid_state"
    if revision.migration_classification not in _ADMITTED_MIGRATION_CLASSES:
        return None, "quarantined"
    try:
        valid_from = _parse_timestamp(revision.valid_from)
        valid_until = _parse_timestamp(revision.valid_until)
    except ValueError:
        return None, "invalid_validity"
    if valid_from is not None and valid_from > now:
        return None, "not_yet_valid"
    if valid_until is not None and valid_until <= now:
        return None, "expired"
    if (
        revision.operation_revision_id != revision.revision_id
        or revision.operation_status != "succeeded"
        or revision.operation not in {"upsert", "replace"}
        or revision.memory_id != row.id
    ):
        return None, "projection_not_current"

    metadata = dict(revision.vector_metadata)
    metadata.update(
        {
            CANONICAL_CLAIM_ID_KEY: revision.claim_id,
            CANONICAL_REVISION_ID_KEY: revision.revision_id,
            CANONICAL_LIFECYCLE_STATE_KEY: revision.state,
            CANONICAL_TEMPORAL_ROLE_KEY: "current_fact",
            "scope": revision.scope_kind,
            "project_id": revision.scope_key if revision.scope_kind == "project" else None,
            "valid_from": revision.valid_from,
            "valid_until": revision.valid_until,
            "migration_classification": revision.migration_classification,
            "source_receipt_id": revision.source_receipt_id,
            "source_run_id": revision.source_run_id,
            "source_message_id": revision.source_message_id,
            "result_message_id": revision.result_message_id,
            "backend": revision.backend,
            "provider": revision.provider,
            "model": revision.model,
            "prompt_version": revision.prompt_version,
            "schema_version": revision.schema_version,
        }
    )
    return replace(row, text=revision.content, memory_type="fact", metadata=metadata), None


def project_current_truth(
    rows: Iterable[MemoryResult],
    *,
    db_path: Path,
    principal_id: str,
    runtime_profile_id: str,
    now: datetime | None = None,
) -> CurrentTruthProjection:
    """Admit only exact, active, successfully projected canonical revisions.

    Any unavailable or malformed canonical authority fails closed. The caller
    may expose the returned reason counts, but this function never includes
    memory content in diagnostics.
    """
    candidates = tuple(rows)
    excluded: Counter[str] = Counter()
    admitted: list[MemoryResult] = []
    effective_now = (now or datetime.now(UTC)).astimezone(UTC)
    try:
        connection = _read_only_connection(db_path)
        try:
            if not _REQUIRED_TABLES.issubset(_tables(connection)):
                return CurrentTruthProjection((), {"authority_unavailable": len(candidates)})
            connection.execute("BEGIN")
            for row in candidates:
                claim_id = row.metadata.get(CANONICAL_CLAIM_ID_KEY)
                revision_id = row.metadata.get(CANONICAL_REVISION_ID_KEY)
                lifecycle_state = row.metadata.get(CANONICAL_LIFECYCLE_STATE_KEY)
                if claim_id is None and revision_id is None and lifecycle_state is None:
                    excluded["legacy_unclassified"] += 1
                    continue
                if not all(isinstance(value, str) and value for value in (claim_id, revision_id, lifecycle_state)):
                    excluded["malformed_lifecycle"] += 1
                    continue
                try:
                    revision = _load_revision(
                        connection,
                        claim_id=claim_id,
                        revision_id=revision_id,
                        principal_id=principal_id,
                        runtime_profile_id=runtime_profile_id,
                    )
                except (json.JSONDecodeError, TypeError, ValueError):
                    excluded["malformed_lifecycle"] += 1
                    continue
                if revision is None:
                    excluded["unknown_revision"] += 1
                    continue
                if lifecycle_state != revision.state:
                    excluded["stale_vector_state"] += 1
                    continue
                projected, reason = _project_row(row, revision, now=effective_now)
                if projected is None:
                    excluded[reason or "invalid_lifecycle"] += 1
                else:
                    admitted.append(projected)
        finally:
            connection.close()
    except (OSError, sqlite3.Error):
        return CurrentTruthProjection((), {"authority_unavailable": len(candidates)})
    return CurrentTruthProjection(tuple(admitted), dict(sorted(excluded.items())))


def current_truth_revision(db_path: Path, *, principal_id: str, runtime_profile_id: str) -> str:
    """Return a stable digest that changes with this owner's active truth."""
    try:
        connection = _read_only_connection(db_path)
        try:
            if not _REQUIRED_TABLES.issubset(_tables(connection)):
                return "unavailable"
            rows = connection.execute(
                "SELECT r.claim_id, r.revision_id, s.state, s.state_event_position, "
                "r.valid_from, r.valid_until, r.migration_classification, "
                "v.revision_id, v.status, v.operation, v.memory_id "
                "FROM memory_fact_revisions r "
                "JOIN memory_fact_claims c ON c.claim_id = r.claim_id "
                "JOIN memory_fact_revision_states s ON s.revision_id = r.revision_id "
                "LEFT JOIN memory_fact_vector_operations v ON v.event_position = ("
                "SELECT MAX(latest.event_position) FROM memory_fact_vector_operations latest "
                "WHERE latest.claim_id = r.claim_id) "
                "WHERE c.owner_principal_id = ? AND c.runtime_profile_id = ? "
                "ORDER BY r.claim_id, r.revision_id",
                (principal_id, runtime_profile_id),
            ).fetchall()
        finally:
            connection.close()
    except (OSError, sqlite3.Error):
        return "unavailable"
    now = datetime.now(UTC)
    normalized: list[tuple[object, ...]] = []
    for row in rows:
        try:
            valid_from = _parse_timestamp(str(row[4]) if row[4] is not None else None)
            valid_until = _parse_timestamp(str(row[5]) if row[5] is not None else None)
        except ValueError:
            validity = "invalid"
        else:
            validity = (
                "not_yet_valid"
                if valid_from is not None and valid_from > now
                else "expired"
                if valid_until is not None and valid_until <= now
                else "current"
            )
        normalized.append((*tuple(row), validity))
    encoded = json.dumps(normalized, separators=(",", ":"), sort_keys=False)
    return hashlib.sha256(encoded.encode()).hexdigest()
