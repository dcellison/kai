"""Canonical Workshop human-handle authority and migration."""

from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime

from kai.workshop.domain import EventEnvelope, PrincipalId, WorkshopEventType, WorkshopId
from kai.workshop.store import WorkshopEventStore

HUMAN_HANDLE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,31}$")
_NON_HANDLE_CHARACTER_PATTERN = re.compile(r"[^a-z0-9]+")
_MIGRATION_KEY_PREFIX = "workshop-human-handle:v1"


class WorkshopHumanHandleError(ValueError):
    """A human handle cannot be normalized or assigned safely."""


@dataclass(frozen=True, slots=True)
class HumanHandleReconciliation:
    eligible: int
    assigned: int
    migrated: int
    missing: int
    invalid: int
    conflicting: int
    supported: bool = True


def normalize_human_handle(value: object) -> str:
    """Validate one explicit, case-insensitive Workshop human handle."""
    if not isinstance(value, str):
        raise WorkshopHumanHandleError("Human handle must be text")
    handle = value.strip().casefold()
    if not HUMAN_HANDLE_PATTERN.fullmatch(handle):
        raise WorkshopHumanHandleError("Human handle must be 1-32 lowercase letters, digits, or underscores")
    return handle


def derive_human_handle(display_name: object) -> str:
    """Derive a stable initial handle from a human display name."""
    if not isinstance(display_name, str) or not display_name.strip():
        raise WorkshopHumanHandleError("Human display name must be non-empty")
    ascii_name = (
        unicodedata.normalize("NFKD", display_name.strip()).encode("ascii", "ignore").decode("ascii").casefold()
    )
    handle = _NON_HANDLE_CHARACTER_PATTERN.sub("_", ascii_name).strip("_")
    if handle and not handle[0].isalpha():
        handle = f"human_{handle}"
    handle = handle[:32].rstrip("_")
    if not HUMAN_HANDLE_PATTERN.fullmatch(handle):
        raise WorkshopHumanHandleError(
            "Human display name cannot be converted safely to a canonical handle; provide one explicitly"
        )
    return handle


async def human_handle_schema_supported(store: WorkshopEventStore) -> bool:
    async with store.connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'human_handles'"
    ) as cursor:
        return await cursor.fetchone() is not None


async def reconcile_human_handles(store: WorkshopEventStore) -> HumanHandleReconciliation:
    """Assign unambiguous handles to legacy human principals with replayable facts."""
    # Keep this import local: projections validate handles through this module.
    from kai.workshop.projection import CanonicalConversationProjection

    if not await human_handle_schema_supported(store):
        return HumanHandleReconciliation(0, 0, 0, 0, 0, 0, supported=False)

    connection = store.connection
    try:
        await connection.execute("BEGIN IMMEDIATE")
        async with connection.execute(
            "SELECT wm.workshop_id, p.id, p.display_name "
            "FROM workshop_memberships wm "
            "JOIN principals p ON p.id = wm.principal_id AND p.kind = 'human' "
            "LEFT JOIN human_handles hh ON hh.workshop_id = wm.workshop_id "
            "AND hh.principal_id = p.id "
            "WHERE hh.principal_id IS NULL "
            "ORDER BY wm.workshop_id, p.id"
        ) as cursor:
            missing_rows = list(await cursor.fetchall())
        async with connection.execute(
            "SELECT workshop_id, handle FROM human_handles UNION ALL SELECT workshop_id, handle FROM agent_definitions"
        ) as cursor:
            reserved_rows = list(await cursor.fetchall())
        reserved = {(str(row[0]), str(row[1]).casefold()) for row in reserved_rows}

        invalid = 0
        candidates: dict[tuple[str, str], list[tuple[WorkshopId, PrincipalId, str]]] = defaultdict(list)
        for row in missing_rows:
            workshop_id = WorkshopId(str(row[0]))
            principal_id = PrincipalId(str(row[1]))
            try:
                handle = derive_human_handle(str(row[2]))
            except WorkshopHumanHandleError:
                invalid += 1
                continue
            candidates[(str(workshop_id), handle.casefold())].append((workshop_id, principal_id, handle))

        conflicting = 0
        migrated = 0
        for key, matches in sorted(candidates.items()):
            if len(matches) != 1 or key in reserved:
                conflicting += len(matches)
                continue
            workshop_id, principal_id, handle = matches[0]
            result = await store.append_in_transaction(
                EventEnvelope.create(
                    event_type=WorkshopEventType.PRINCIPAL_HANDLE_ASSIGNED,
                    event_version=1,
                    workshop_id=workshop_id,
                    aggregate_type="principal_handle",
                    aggregate_id=principal_id,
                    occurred_at=datetime.now(UTC),
                    idempotency_key=(f"{_MIGRATION_KEY_PREFIX}:{workshop_id}:{principal_id}"),
                    payload={"handle": handle},
                    metadata={"source": "canonical_migration"},
                )
            )
            migrated += int(result.inserted)
            reserved.add(key)
        await store.project_pending_in_transaction(CanonicalConversationProjection())
        async with connection.execute(
            "SELECT COUNT(*) FROM workshop_memberships wm "
            "JOIN principals p ON p.id = wm.principal_id AND p.kind = 'human'"
        ) as cursor:
            eligible_row = await cursor.fetchone()
        assert eligible_row is not None
        eligible = int(eligible_row[0])
        async with connection.execute(
            "SELECT COUNT(*) FROM human_handles hh "
            "JOIN workshop_memberships wm ON wm.workshop_id = hh.workshop_id "
            "AND wm.principal_id = hh.principal_id "
            "JOIN principals p ON p.id = hh.principal_id AND p.kind = 'human'"
        ) as cursor:
            assigned_row = await cursor.fetchone()
        assert assigned_row is not None
        assigned = int(assigned_row[0])
        await connection.commit()
    except Exception:
        await connection.rollback()
        raise
    return HumanHandleReconciliation(
        eligible=eligible,
        assigned=assigned,
        migrated=migrated,
        missing=max(0, eligible - assigned),
        invalid=invalid,
        conflicting=conflicting,
    )
