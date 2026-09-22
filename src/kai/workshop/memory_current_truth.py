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
CANONICAL_EPISODE_ID_KEY = "canonical_memory_episode_id"
CANONICAL_ADMISSION_AUTHORITY_KEY = "canonical_memory_admission_authority"

_FACT_TABLES = {
    "memory_fact_claims",
    "memory_fact_revisions",
    "memory_fact_revision_states",
    "memory_fact_lifecycle_events",
    "memory_fact_vector_operations",
}
_EPISODE_TABLES = {
    "memory_episodes",
    "memory_episode_followups",
    "memory_episode_vector_operations",
}
_ADMITTED_MIGRATION_CLASSES = {"canonical", "legacy_complete"}


def _is_admitted(migration_classification: str, admission_authority: str) -> bool:
    return (
        migration_classification in _ADMITTED_MIGRATION_CLASSES and admission_authority == "provenance_verified"
    ) or (migration_classification != "legacy_quarantined" and admission_authority == "operator_review")


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
    admission_authority: str
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


@dataclass(frozen=True, slots=True)
class _CanonicalEpisode:
    episode_id: str
    content: str
    scope_kind: str
    scope_key: str
    stored_at: str
    migration_classification: str
    admission_authority: str
    vector_metadata: dict[str, Any]
    structured: dict[str, Any]
    provenance: dict[str, str | None]
    operation_status: str | None
    memory_id: str | None
    relationships: tuple[dict[str, str], ...]


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
        "r.valid_from, r.valid_until, r.migration_classification, r.admission_authority, "
        "r.vector_metadata_json, "
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
    metadata = json.loads(str(row[10]))
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
        admission_authority=str(row[9]),
        vector_metadata=metadata,
        source_receipt_id=str(row[11]) if row[11] is not None else None,
        source_run_id=str(row[12]) if row[12] is not None else None,
        source_message_id=str(row[13]) if row[13] is not None else None,
        result_message_id=str(row[14]) if row[14] is not None else None,
        backend=str(row[15]) if row[15] is not None else None,
        provider=str(row[16]) if row[16] is not None else None,
        model=str(row[17]) if row[17] is not None else None,
        prompt_version=str(row[18]) if row[18] is not None else None,
        schema_version=str(row[19]) if row[19] is not None else None,
        operation_revision_id=str(row[20]) if row[20] is not None else None,
        operation_status=str(row[21]) if row[21] is not None else None,
        operation=str(row[22]) if row[22] is not None else None,
        memory_id=str(row[23]) if row[23] is not None else None,
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
    if not _is_admitted(revision.migration_classification, revision.admission_authority):
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
            CANONICAL_ADMISSION_AUTHORITY_KEY: revision.admission_authority,
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


def _load_episode(
    connection: sqlite3.Connection,
    *,
    episode_id: str,
    principal_id: str,
    runtime_profile_id: str,
) -> _CanonicalEpisode | None:
    row = connection.execute(
        "SELECT e.episode_id, e.content, e.scope_kind, e.scope_key, e.stored_at, "
        "e.migration_classification, e.admission_authority, e.vector_metadata_json, "
        "e.goal, e.context, e.approach, "
        "e.outcome, e.outcome_quality, e.lessons, e.tags_json, e.actors_json, "
        "e.occurred_from, e.occurred_until, e.observed_at, e.reason, e.evidence_json, "
        "e.source_receipt_id, e.source_run_id, e.source_message_id, e.result_message_id, "
        "e.backend, e.provider, e.model, e.prompt_version, e.schema_version, "
        "v.status, v.memory_id FROM memory_episodes e "
        "LEFT JOIN memory_episode_vector_operations v ON v.episode_id = e.episode_id "
        "WHERE e.episode_id = ? AND e.owner_principal_id = ? AND e.runtime_profile_id = ?",
        (episode_id, principal_id, runtime_profile_id),
    ).fetchone()
    if row is None:
        return None
    vector_metadata = json.loads(str(row[7]))
    tags = json.loads(str(row[14]))
    actors = json.loads(str(row[15]))
    evidence = json.loads(str(row[20]))
    if not isinstance(vector_metadata, dict) or not isinstance(tags, list) or not isinstance(actors, list):
        raise ValueError("canonical episode metadata is malformed")
    if not isinstance(evidence, list):
        raise ValueError("canonical episode evidence is malformed")
    relationships = tuple(
        {
            "direction": "outgoing" if str(link[0]) == episode_id else "incoming",
            "episode_id": str(link[1]) if str(link[0]) == episode_id else str(link[0]),
            "relationship": str(link[2]),
            "reason": str(link[3]),
        }
        for link in connection.execute(
            "SELECT source_episode_id, target_episode_id, relationship, reason "
            "FROM memory_episode_followups WHERE source_episode_id = ? OR target_episode_id = ? "
            "ORDER BY created_event_position",
            (episode_id, episode_id),
        ).fetchall()
    )
    return _CanonicalEpisode(
        episode_id=str(row[0]),
        content=str(row[1]),
        scope_kind=str(row[2]),
        scope_key=str(row[3]),
        stored_at=str(row[4]),
        migration_classification=str(row[5]),
        admission_authority=str(row[6]),
        vector_metadata=vector_metadata,
        structured={
            "goal": str(row[8]),
            "context": str(row[9]),
            "approach": str(row[10]),
            "outcome": str(row[11]),
            "outcome_quality": str(row[12]),
            "lessons": str(row[13]) if row[13] is not None else None,
            "tags": tags,
            "actors": actors,
            "occurred_from": str(row[16]) if row[16] is not None else None,
            "occurred_until": str(row[17]) if row[17] is not None else None,
            "observed_at": str(row[18]) if row[18] is not None else None,
            "reason": str(row[19]),
            "evidence": evidence,
        },
        provenance={
            "source_receipt_id": str(row[21]) if row[21] is not None else None,
            "source_run_id": str(row[22]) if row[22] is not None else None,
            "source_message_id": str(row[23]) if row[23] is not None else None,
            "result_message_id": str(row[24]) if row[24] is not None else None,
            "backend": str(row[25]) if row[25] is not None else None,
            "provider": str(row[26]) if row[26] is not None else None,
            "model": str(row[27]) if row[27] is not None else None,
            "prompt_version": str(row[28]) if row[28] is not None else None,
            "schema_version": str(row[29]) if row[29] is not None else None,
        },
        operation_status=str(row[30]) if row[30] is not None else None,
        memory_id=str(row[31]) if row[31] is not None else None,
        relationships=relationships,
    )


def _project_episode(
    row: MemoryResult,
    episode: _CanonicalEpisode,
) -> tuple[MemoryResult | None, str | None]:
    if not _is_admitted(episode.migration_classification, episode.admission_authority):
        return None, "quarantined"
    if episode.operation_status != "succeeded" or episode.memory_id != row.id:
        return None, "projection_not_current"
    metadata = dict(episode.vector_metadata)
    metadata.update(episode.structured)
    metadata.update(episode.provenance)
    metadata.update(
        {
            "source": "episode",
            CANONICAL_EPISODE_ID_KEY: episode.episode_id,
            CANONICAL_TEMPORAL_ROLE_KEY: "historical_episode",
            "scope": episode.scope_kind,
            "project_id": episode.scope_key if episode.scope_kind == "project" else None,
            "stored_at": episode.stored_at,
            "migration_classification": episode.migration_classification,
            CANONICAL_ADMISSION_AUTHORITY_KEY: episode.admission_authority,
            "episode_followups": list(episode.relationships),
        }
    )
    return replace(row, text=episode.content, memory_type="episode", metadata=metadata), None


def _order_episode_chains(rows: list[MemoryResult]) -> list[MemoryResult]:
    """Order related episode occurrences newest-first without moving unrelated recall."""
    episode_indexes = {
        str(row.metadata[CANONICAL_EPISODE_ID_KEY]): index
        for index, row in enumerate(rows)
        if isinstance(row.metadata.get(CANONICAL_EPISODE_ID_KEY), str)
    }
    adjacency: dict[str, set[str]] = {episode_id: set() for episode_id in episode_indexes}
    for episode_id, index in episode_indexes.items():
        links = rows[index].metadata.get("episode_followups")
        if not isinstance(links, list):
            continue
        for link in links:
            related = link.get("episode_id") if isinstance(link, dict) else None
            if isinstance(related, str) and related in adjacency:
                adjacency[episode_id].add(related)
                adjacency[related].add(episode_id)
    ordered = list(rows)
    visited: set[str] = set()
    for root in adjacency:
        if root in visited:
            continue
        pending = [root]
        component: set[str] = set()
        while pending:
            current = pending.pop()
            if current in component:
                continue
            component.add(current)
            pending.extend(adjacency[current] - component)
        visited.update(component)
        if len(component) < 2:
            continue
        positions = sorted(episode_indexes[value] for value in component)
        chain_rows = sorted(
            (rows[episode_indexes[value]] for value in component),
            key=lambda value: str(value.metadata.get("stored_at") or ""),
            reverse=True,
        )
        for position, row in zip(positions, chain_rows, strict=True):
            ordered[position] = row
    return ordered


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
            available_tables = _tables(connection)
            if not _FACT_TABLES.issubset(available_tables):
                return CurrentTruthProjection((), {"authority_unavailable": len(candidates)})
            connection.execute("BEGIN")
            for row in candidates:
                episode_id = row.metadata.get(CANONICAL_EPISODE_ID_KEY)
                if episode_id is not None:
                    if not _EPISODE_TABLES.issubset(available_tables):
                        excluded["authority_unavailable"] += 1
                        continue
                    if not isinstance(episode_id, str) or not episode_id:
                        excluded["malformed_lifecycle"] += 1
                        continue
                    try:
                        episode = _load_episode(
                            connection,
                            episode_id=episode_id,
                            principal_id=principal_id,
                            runtime_profile_id=runtime_profile_id,
                        )
                    except (json.JSONDecodeError, TypeError, ValueError):
                        excluded["malformed_lifecycle"] += 1
                        continue
                    if episode is None:
                        excluded["unknown_episode"] += 1
                        continue
                    projected, reason = _project_episode(row, episode)
                    if projected is None:
                        excluded[reason or "invalid_lifecycle"] += 1
                    else:
                        admitted.append(projected)
                    continue
                claim_id = row.metadata.get(CANONICAL_CLAIM_ID_KEY)
                revision_id = row.metadata.get(CANONICAL_REVISION_ID_KEY)
                lifecycle_state = row.metadata.get(CANONICAL_LIFECYCLE_STATE_KEY)
                if claim_id is None and revision_id is None and lifecycle_state is None:
                    excluded["legacy_unclassified"] += 1
                    continue
                if (
                    not isinstance(claim_id, str)
                    or not claim_id
                    or not isinstance(revision_id, str)
                    or not revision_id
                    or not isinstance(lifecycle_state, str)
                    or not lifecycle_state
                ):
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
    return CurrentTruthProjection(tuple(_order_episode_chains(admitted)), dict(sorted(excluded.items())))


def current_truth_revision(db_path: Path, *, principal_id: str, runtime_profile_id: str) -> str:
    """Return a stable digest that changes with this owner's active truth."""
    try:
        connection = _read_only_connection(db_path)
        try:
            available_tables = _tables(connection)
            if not _FACT_TABLES.issubset(available_tables):
                return "unavailable"
            rows = connection.execute(
                "SELECT r.claim_id, r.revision_id, s.state, s.state_event_position, "
                "r.valid_from, r.valid_until, r.migration_classification, r.admission_authority, "
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
            episode_rows = (
                connection.execute(
                    "SELECT e.episode_id, e.stored_at, e.migration_classification, e.admission_authority, "
                    "v.status, v.memory_id, f.source_episode_id, f.target_episode_id, f.relationship, "
                    "f.created_event_position FROM memory_episodes e "
                    "LEFT JOIN memory_episode_vector_operations v ON v.episode_id = e.episode_id "
                    "LEFT JOIN memory_episode_followups f ON "
                    "f.source_episode_id = e.episode_id OR f.target_episode_id = e.episode_id "
                    "WHERE e.owner_principal_id = ? AND e.runtime_profile_id = ? "
                    "ORDER BY e.episode_id, f.created_event_position",
                    (principal_id, runtime_profile_id),
                ).fetchall()
                if _EPISODE_TABLES.issubset(available_tables)
                else []
            )
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
    encoded = json.dumps(
        {"facts": normalized, "episodes": [tuple(row) for row in episode_rows]},
        separators=(",", ":"),
        sort_keys=False,
    )
    return hashlib.sha256(encoded.encode()).hexdigest()
