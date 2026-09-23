"""
Read-only status of the canonical memory vector outbox.

Canonical facts and episodes live in SQLite; retrieval reads vector rows
that an outbox keeps in step. An operation that fails three times stays
`failed`, and for facts every later operation on the same claim queues
behind it. The affected item is then missing from recall with nothing
else to show for it, so this module answers three questions for every
surface that lists them (Workshop and the admin CLI; install status keeps
its own plain counts so it works on every schema version):

- which facts and episodes have a failed operation;
- how many fact operations are blocked behind a failure;
- which vector rows no longer match canonical state (the vector audit).

The outbox queries run on either sqlite3 (diagnostics and CLI) or
aiosqlite (the Workshop service), so the SQL lives here once and each
runner only maps rows. The vector audit needs the rows themselves, which
only a process holding the vector store can read.
"""

from __future__ import annotations

import sqlite3
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

from kai import memory
from kai.workshop.memory_current_truth import CANONICAL_CLAIM_ID_KEY, CANONICAL_EPISODE_ID_KEY

# Review lists stay bounded like the other owner review lists; totals are
# reported separately so a long tail is still visible.
MAX_FAILED_ITEMS = 200
_MAX_PREVIEW_CHARACTERS = 500


@dataclass(frozen=True, slots=True)
class ProjectionRetryResult:
    """
    Outcome of re-running failed vector operations.

    `retried` counts the failed operations reset for another attempt;
    `succeeded` and `failed` count how those same operations ended after
    the outbox drained. An operation still pending afterwards is queued
    behind another failure on its claim, so it is in neither count.
    """

    retried: int
    succeeded: int
    failed: int


@dataclass(frozen=True, slots=True)
class FailedProjection:
    """
    One fact claim or episode whose vector operation failed.

    For a fact this is the claim's earliest failed operation, because that
    is the one blocking everything after it. `revision_id` is None for
    episodes, which have a single operation each.
    """

    kind: str
    item_id: str
    principal_id: str
    runtime_profile_id: str
    operation: str
    revision_id: str | None
    attempts: int
    error_code: str | None
    updated_at: str
    preview: str


@dataclass(frozen=True, slots=True)
class ProjectionStatus:
    """Failed items (bounded), their totals, and blocked fact operations."""

    failed: tuple[FailedProjection, ...]
    failed_facts: int
    failed_episodes: int
    blocked: int
    oldest_failure_at: str | None

    @property
    def failed_total(self) -> int:
        return self.failed_facts + self.failed_episodes


@dataclass(frozen=True, slots=True)
class VectorAudit:
    """
    Vector rows that no longer match canonical state, for one owner.

    - `orphan`: rows naming a known claim or episode that are not its
      current row (left behind by a delete, a rewrite, or a duplicate add).
    - `unknown`: rows naming a claim or episode the owner does not have.
    - `duplicate`: claims or episodes with more than one row.
    - `missing`: current rows canonical state expects that the store does
      not have. Unlike the others this one is a loss, not noise: the
      current-truth gate admits a fact only through its current row, so a
      missing row is a current fact recall cannot return.

    The retrieval gate keeps orphan, unknown, and duplicate rows out of
    recall; they are wasted storage and noise in exclusion counts, not a
    leak. Memory ids are kept so an operator can inspect them; content
    never is.
    """

    orphan: tuple[str, ...]
    unknown: tuple[str, ...]
    duplicate: int
    missing: tuple[str, ...] = ()


# ── Outbox queries ───────────────────────────────────────────────────

# The earliest failed operation per claim, so each claim appears once.
_FAILED_FACTS = (
    "SELECT v.claim_id, c.owner_principal_id, c.runtime_profile_id, v.operation, v.revision_id, "
    "v.attempt_count, v.last_error_code, v.updated_at, r.content "
    "FROM memory_fact_vector_operations v "
    "JOIN memory_fact_claims c ON c.claim_id = v.claim_id "
    "JOIN memory_fact_revisions r ON r.revision_id = v.revision_id "
    "WHERE v.status = 'failed' AND v.event_position = ("
    "SELECT MIN(first.event_position) FROM memory_fact_vector_operations first "
    "WHERE first.claim_id = v.claim_id AND first.status = 'failed')"
)
_FAILED_EPISODES = (
    "SELECT v.episode_id, e.owner_principal_id, e.runtime_profile_id, 'upsert', NULL, "
    "v.attempt_count, v.last_error_code, v.updated_at, e.content "
    "FROM memory_episode_vector_operations v "
    "JOIN memory_episodes e ON e.episode_id = v.episode_id "
    "WHERE v.status = 'failed'"
)
# Pending operations that cannot run because an earlier operation on the
# same claim failed.
_BLOCKED_FACTS = (
    "SELECT COUNT(*) FROM memory_fact_vector_operations v "
    "JOIN memory_fact_claims c ON c.claim_id = v.claim_id "
    "WHERE v.status = 'pending' AND EXISTS ("
    "SELECT 1 FROM memory_fact_vector_operations prior WHERE prior.claim_id = v.claim_id "
    "AND prior.event_position < v.event_position AND prior.status = 'failed')"
)


def _owner_filter(alias: str, owner: tuple[str, str] | None) -> tuple[str, tuple[str, ...]]:
    """Return an owner-pair condition for the claim or episode alias, if any."""
    if owner is None:
        return "", ()
    return f" AND {alias}.owner_principal_id = ? AND {alias}.runtime_profile_id = ?", owner


def _statements(owner: tuple[str, str] | None) -> tuple[tuple[str, tuple[str, ...]], ...]:
    fact_filter, fact_parameters = _owner_filter("c", owner)
    episode_filter, episode_parameters = _owner_filter("e", owner)
    return (
        (_FAILED_FACTS + fact_filter, fact_parameters),
        (_FAILED_EPISODES + episode_filter, episode_parameters),
        (_BLOCKED_FACTS + fact_filter, fact_parameters),
    )


def _preview(content: object) -> str:
    text = " ".join(str(content).split())
    return text if len(text) <= _MAX_PREVIEW_CHARACTERS else text[: _MAX_PREVIEW_CHARACTERS - 1] + "…"


def _failed(kind: str, row: tuple[object, ...]) -> FailedProjection:
    return FailedProjection(
        kind=kind,
        item_id=str(row[0]),
        principal_id=str(row[1]),
        runtime_profile_id=str(row[2]),
        operation=str(row[3]),
        revision_id=str(row[4]) if row[4] is not None else None,
        attempts=int(str(row[5])),
        error_code=str(row[6]) if row[6] is not None else None,
        updated_at=str(row[7]),
        preview=_preview(row[8]),
    )


def _status(
    facts: Iterable[tuple[object, ...]],
    episodes: Iterable[tuple[object, ...]],
    blocked: int,
) -> ProjectionStatus:
    failed_facts = [_failed("fact", row) for row in facts]
    failed_episodes = [_failed("episode", row) for row in episodes]
    items = sorted([*failed_facts, *failed_episodes], key=lambda item: item.updated_at, reverse=True)
    return ProjectionStatus(
        failed=tuple(items[:MAX_FAILED_ITEMS]),
        failed_facts=len(failed_facts),
        failed_episodes=len(failed_episodes),
        blocked=blocked,
        oldest_failure_at=min((item.updated_at for item in items), default=None),
    )


def projection_status(connection: sqlite3.Connection, owner: tuple[str, str] | None = None) -> ProjectionStatus:
    """
    Read outbox status with sqlite3, for the admin CLI.

    Args:
        connection: An open connection; read-only is enough.
        owner: Optional (principal id, runtime profile id) to scope to.
    """
    (facts_sql, facts_args), (episodes_sql, episodes_args), (blocked_sql, blocked_args) = _statements(owner)
    facts = connection.execute(facts_sql, facts_args).fetchall()
    episodes = connection.execute(episodes_sql, episodes_args).fetchall()
    blocked_row = connection.execute(blocked_sql, blocked_args).fetchone()
    return _status(facts, episodes, int(blocked_row[0]) if blocked_row else 0)


async def projection_status_async(
    connection: aiosqlite.Connection,
    owner: tuple[str, str] | None = None,
) -> ProjectionStatus:
    """The same read as `projection_status`, on the service's aiosqlite connection."""
    (facts_sql, facts_args), (episodes_sql, episodes_args), (blocked_sql, blocked_args) = _statements(owner)
    async with connection.execute(facts_sql, facts_args) as cursor:
        facts = [tuple(row) for row in await cursor.fetchall()]
    async with connection.execute(episodes_sql, episodes_args) as cursor:
        episodes = [tuple(row) for row in await cursor.fetchall()]
    async with connection.execute(blocked_sql, blocked_args) as cursor:
        blocked_row = await cursor.fetchone()
    return _status(facts, episodes, int(blocked_row[0]) if blocked_row else 0)


# ── Vector audit ─────────────────────────────────────────────────────

# The row each claim or episode should have: the memory id of its latest
# succeeded operation, when that operation wrote a row. A claim whose
# latest succeeded operation was a delete should have no row at all.
_CURRENT_FACT_ROWS = (
    "SELECT c.claim_id, (SELECT v.memory_id FROM memory_fact_vector_operations v "
    "WHERE v.claim_id = c.claim_id AND v.status = 'succeeded' "
    "ORDER BY v.event_position DESC LIMIT 1), (SELECT v.operation FROM memory_fact_vector_operations v "
    "WHERE v.claim_id = c.claim_id AND v.status = 'succeeded' ORDER BY v.event_position DESC LIMIT 1) "
    "FROM memory_fact_claims c WHERE c.owner_principal_id = ? AND c.runtime_profile_id = ?"
)
_CURRENT_EPISODE_ROWS = (
    "SELECT e.episode_id, v.memory_id FROM memory_episodes e "
    "LEFT JOIN memory_episode_vector_operations v ON v.episode_id = e.episode_id AND v.status = 'succeeded' "
    "WHERE e.owner_principal_id = ? AND e.runtime_profile_id = ?"
)


def audit_vector_rows(
    rows: Iterable[memory.MemoryResult],
    *,
    current_facts: dict[str, str | None],
    current_episodes: dict[str, str | None],
) -> VectorAudit:
    """
    Compare an owner's vector rows with the rows canonical state expects.

    Rows without a canonical claim or episode id are legacy rows, which
    reconciliation owns, so they are skipped.

    Args:
        rows: Every lifecycle-visible row for the owner.
        current_facts: Claim id to its expected memory id (None: no row).
        current_episodes: Episode id to its expected memory id.
    """
    orphan: list[str] = []
    unknown: list[str] = []
    present: set[str] = set()
    per_item: Counter[tuple[str, str]] = Counter()
    for row in rows:
        present.add(row.id)
        claim_id = row.metadata.get(CANONICAL_CLAIM_ID_KEY)
        episode_id = row.metadata.get(CANONICAL_EPISODE_ID_KEY)
        if isinstance(claim_id, str) and claim_id:
            kind, item_id, expected = "fact", claim_id, current_facts
        elif isinstance(episode_id, str) and episode_id:
            kind, item_id, expected = "episode", episode_id, current_episodes
        else:
            continue
        per_item[(kind, item_id)] += 1
        if item_id not in expected:
            unknown.append(row.id)
        elif expected[item_id] != row.id:
            orphan.append(row.id)
    expected_ids = {
        memory_id for memory_id in (*current_facts.values(), *current_episodes.values()) if memory_id is not None
    }
    return VectorAudit(
        orphan=tuple(sorted(orphan)),
        unknown=tuple(sorted(unknown)),
        duplicate=sum(1 for count in per_item.values() if count > 1),
        missing=tuple(sorted(expected_ids - present)),
    )


def _expected_facts(rows: Iterable[tuple[object, ...]]) -> dict[str, str | None]:
    # Only an upsert or replace leaves a row behind; after a delete the
    # claim should have none, so any remaining row is an orphan.
    return {
        str(row[0]): (str(row[1]) if row[1] is not None and row[2] in {"upsert", "replace"} else None) for row in rows
    }


def _expected_episodes(rows: Iterable[tuple[object, ...]]) -> dict[str, str | None]:
    return {str(row[0]): (str(row[1]) if row[1] is not None else None) for row in rows}


def expected_rows(
    connection: sqlite3.Connection,
    owner: tuple[str, str],
) -> tuple[dict[str, str | None], dict[str, str | None]]:
    """Read the expected fact and episode rows for one owner with sqlite3."""
    facts = connection.execute(_CURRENT_FACT_ROWS, owner).fetchall()
    episodes = connection.execute(_CURRENT_EPISODE_ROWS, owner).fetchall()
    return _expected_facts(facts), _expected_episodes(episodes)


async def expected_rows_async(
    connection: aiosqlite.Connection,
    owner: tuple[str, str],
) -> tuple[dict[str, str | None], dict[str, str | None]]:
    """The same read as `expected_rows`, on the service's aiosqlite connection."""
    async with connection.execute(_CURRENT_FACT_ROWS, owner) as cursor:
        facts = [tuple(row) for row in await cursor.fetchall()]
    async with connection.execute(_CURRENT_EPISODE_ROWS, owner) as cursor:
        episodes = [tuple(row) for row in await cursor.fetchall()]
    return _expected_facts(facts), _expected_episodes(episodes)


def store_vector_audit(
    db_path: Path,
    audit: VectorAudit,
    *,
    principal_id: str,
    runtime_profile_id: str,
    now: datetime | None = None,
) -> None:
    """
    Record one owner's latest vector audit counts for install status.

    Only counts and the time are stored, never memory ids or content, so
    install status can report drift and its age without reading the
    vector store. The row is replaced on every audit.
    """
    connection = sqlite3.connect(str(db_path), timeout=30)
    try:
        connection.execute(
            "INSERT INTO memory_vector_audit ("
            "principal_id, runtime_profile_id, orphan_rows, unknown_rows, duplicate_items, missing_rows, checked_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT (principal_id, runtime_profile_id) DO UPDATE SET orphan_rows = excluded.orphan_rows, "
            "unknown_rows = excluded.unknown_rows, duplicate_items = excluded.duplicate_items, "
            "missing_rows = excluded.missing_rows, checked_at = excluded.checked_at",
            (
                principal_id,
                runtime_profile_id,
                len(audit.orphan),
                len(audit.unknown),
                audit.duplicate,
                len(audit.missing),
                # Full precision: qualification compares this time with the
                # latest projection's, which is recorded to the millisecond.
                (now or datetime.now(UTC)).isoformat(),
            ),
        )
        connection.commit()
    finally:
        connection.close()


def refresh_vector_audit(db_path: Path, *, principal_id: str, runtime_profile_id: str) -> VectorAudit:
    """
    Audit one owner's vector rows against canonical state and store the counts.

    Call only from a process that holds the vector store (the service, or
    the admin CLI with the service stopped). A read failure propagates and
    the previous stored audit stays with its older time.
    """
    rows = memory.get_all_for_lifecycle_projection(user_id=principal_id, runtime_profile_id=runtime_profile_id)
    connection = sqlite3.connect(str(db_path), timeout=30)
    try:
        facts, episodes = expected_rows(connection, (principal_id, runtime_profile_id))
    finally:
        connection.close()
    audit = audit_vector_rows(rows, current_facts=facts, current_episodes=episodes)
    store_vector_audit(db_path, audit, principal_id=principal_id, runtime_profile_id=runtime_profile_id)
    return audit


__all__ = [
    "MAX_FAILED_ITEMS",
    "FailedProjection",
    "ProjectionRetryResult",
    "ProjectionStatus",
    "VectorAudit",
    "audit_vector_rows",
    "expected_rows",
    "expected_rows_async",
    "projection_status",
    "projection_status_async",
    "refresh_vector_audit",
    "store_vector_audit",
]
