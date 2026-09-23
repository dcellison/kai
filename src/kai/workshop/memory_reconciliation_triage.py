"""Canonical grouped triage and exception-only review for legacy memory."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from kai import memory_reconciliation
from kai.config import ModelRole
from kai.memory_reconciliation_triage import build_triage_plan, validate_triage_plan
from kai.oneshot import OneShotError
from kai.workshop.domain import PrincipalId, RuntimeProfileId
from kai.workshop.memory_reconciliation_review import (
    MAX_OPERATION_ID,
    MAX_OPERATOR_NOTE,
    MemoryReconciliationReviewAccessDenied,
    MemoryReconciliationReviewConflict,
    MemoryReconciliationReviewError,
    MemoryReconciliationReviewNotFound,
    MemoryReconciliationReviewValidationError,
    prior_reconciliation_row_dispositions,
)
from kai.workshop.runtime_pool import WorkshopRuntimePool
from kai.workshop.store import WorkshopEventStore

log = logging.getLogger(__name__)

# Decisions stay fixed while an apply run is unfinished: the run's
# idempotency keys derive from the sealed decisions, so changing one
# would let a retry repeat writes instead of replaying them.
_APPLY_IN_PROGRESS = "An apply of this plan is unfinished; retry the apply to finish it before changing decisions"

PROMPT_VERSION = "memory_reconciliation_triage_v2"
MAX_GROUPS = 100


def build_memory_reasoner(*args: Any, **kwargs: Any) -> Any:
    """Resolve the shared reasoner factory without creating an import cycle."""
    from kai.memory_extraction import build_memory_reasoner as factory

    return factory(*args, **kwargs)


_RECOMMENDATION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "recommendations": {
            "type": "array",
            "maxItems": MAX_GROUPS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "group_id": {"type": "string"},
                    "outcome": {
                        "type": "string",
                        "enum": ["adopt", "consolidate", "obsolete", "needs_review"],
                    },
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "rationale": {"type": "string", "maxLength": 1000},
                    "related_group_ids": {
                        "type": "array",
                        "maxItems": MAX_GROUPS - 1,
                        "uniqueItems": True,
                        "items": {"type": "string"},
                    },
                },
                "required": ["group_id", "outcome", "confidence", "rationale", "related_group_ids"],
            },
        }
    },
    "required": ["recommendations"],
}


@dataclass(frozen=True, slots=True)
class TriageSummary:
    plan_id: str
    audit_id: str
    status: str
    review_version: int
    memory_count: int
    group_count: int
    resolution_counts: dict[str, int]
    disposition_counts: dict[str, int]
    deterministic_groups: int
    pending_deterministic_groups: int
    exception_groups: int
    recommended_groups: int
    applied_at: str | None


@dataclass(frozen=True, slots=True)
class TriageGroupPage:
    triage: TriageSummary
    groups: tuple[dict[str, Any], ...]
    next_offset: int | None


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _timestamp(value: object) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(UTC)


def _unchanged_prior_review_action(
    group: dict[str, Any],
    *,
    audit_boundary: object,
) -> dict[str, str] | None:
    """Recover the safe unchanged action without rewriting an immutable plan."""
    if group.get("classification") != "prior_review" or group.get("action") != {"kind": "manual_edit_required"}:
        return None
    evidence = group.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != 1:
        return None
    row = evidence[0]
    if not isinstance(row, dict) or row.get("kind") != "fact" or row.get("scope") not in {"global", "project"}:
        return None
    if row.get("scope") == "project" and not row.get("project_id"):
        return None
    gaps = row.get("migration_gaps")
    if not isinstance(gaps, list) or "scope" in gaps:
        return None
    valid_until = row.get("valid_until")
    if valid_until is not None:
        boundary = _timestamp(audit_boundary)
        validity_end = _timestamp(valid_until)
        if boundary is None or validity_end is None or validity_end <= boundary:
            return None
    return {"kind": "adopt_as_current"}


def _unchanged_fact_action(
    group: dict[str, Any],
    *,
    audit_boundary: object,
) -> dict[str, str] | None:
    """Return unchanged adoption only for one usable, current fact."""
    evidence = group.get("evidence")
    if not isinstance(evidence, list) or len(evidence) != 1:
        return None
    row = evidence[0]
    if not isinstance(row, dict) or row.get("kind") != "fact" or row.get("scope") not in {"global", "project"}:
        return None
    if row.get("scope") == "project" and not row.get("project_id"):
        return None
    gaps = row.get("migration_gaps")
    if not isinstance(gaps, list) or "scope" in gaps:
        return None
    valid_until = row.get("valid_until")
    if valid_until is not None:
        boundary = _timestamp(audit_boundary)
        validity_end = _timestamp(valid_until)
        if boundary is None or validity_end is None or validity_end <= boundary:
            return None
    return {"kind": "adopt_as_current"}


def _operator_action_is_authorized(
    group: dict[str, Any],
    action: dict[str, Any],
    *,
    audit_boundary: object,
) -> bool:
    evidence = group.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        return False
    if group.get("deterministic") is True and group.get("bulk_eligible") is True and action == group.get("action"):
        return True
    kinds = {item.get("kind") for item in evidence if isinstance(item, dict)}
    action_kind = action.get("kind")
    if action_kind == "adopt_as_current":
        return action == _unchanged_fact_action(group, audit_boundary=audit_boundary)
    if action_kind in {"adopt_corrected", "expire_all"}:
        return kinds == {"fact"}
    if action_kind == "keep_first_retract_rest":
        return kinds == {"fact"} and len(evidence) > 1
    if action_kind == "record_episode_chain":
        return kinds == {"episode"}
    return False


# Consolidations always come from at least two facts; the upper bound keeps
# one decision reviewable and one apply step bounded.
MIN_CONSOLIDATION_FACTS = 2
MAX_CONSOLIDATION_FACTS = 50
_GROUP_ID_PATTERN = re.compile(r"mtg_[0-9a-f]{32}")


def related_group_ids(
    group: dict[str, Any], recommendation: dict[str, Any], plan_groups: dict[str, dict[str, Any]]
) -> list[str]:
    """
    Return the other fact groups a recommendation says belong with this one.

    Current recommendations list them in `related_group_ids`. Earlier ones
    named them only in their rationale, so full group ids found there are
    read too. Either way, only ids of other groups in this plan whose
    evidence is all facts are returned, so a recommendation can never lead
    to a consolidation of something that is not there or not a fact.
    """
    structured = recommendation.get("related_group_ids")
    candidates = (
        [str(item) for item in structured if isinstance(item, str)]
        if isinstance(structured, list)
        else _GROUP_ID_PATTERN.findall(str(recommendation.get("rationale", "")))
    )
    related: list[str] = []
    for group_id in candidates:
        other = plan_groups.get(group_id)
        if (
            other is None
            or group_id == group["group_id"]
            or group_id in related
            or any(item.get("kind") != "fact" for item in other["evidence"])
        ):
            continue
        related.append(group_id)
    return related


def _consolidation_action(canonical: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Build the corrected adoption a consolidation applies as.

    One canonical fact, adopted from the chosen source row with the
    operator's final wording and scope, citing every selected row as
    evidence. The other rows are then absorbed: kept as evidence, never
    recalled.
    """
    if not isinstance(canonical, dict):
        raise MemoryReconciliationReviewValidationError("Invalid consolidated fact")
    content = canonical.get("content")
    source = canonical.get("source_memory_id")
    scope_kind = canonical.get("scope_kind")
    scope_key = canonical.get("scope_key") or ""
    if not isinstance(content, str) or not content.strip():
        raise MemoryReconciliationReviewValidationError("Enter the consolidated fact's wording")
    if source not in {str(item["memory_id"]) for item in rows}:
        raise MemoryReconciliationReviewValidationError("The wording source must be one of the selected facts")
    if scope_kind not in {"global", "project"} or not isinstance(scope_key, str):
        raise MemoryReconciliationReviewValidationError("Choose the consolidated fact's scope")
    return {
        "kind": "adopt_corrected",
        "source_memory_id": source,
        "replacement": {"content": content.strip(), "scope_kind": scope_kind, "scope_key": scope_key},
    }


def _public_consolidation(item: dict[str, Any]) -> dict[str, Any]:
    """Client-safe view of a staged consolidation, with the preview counts."""
    replacement = item["canonical"]["replacement"]
    return {
        "consolidation_id": item["consolidation_id"],
        "revision": item["revision"],
        "group_ids": list(item["group_ids"]),
        "content": replacement["content"],
        "scope_kind": replacement["scope_kind"],
        "scope_key": replacement.get("scope_key") or None,
        "source_memory_id": item["canonical"]["source_memory_id"],
        "operator_note": item["operator_note"],
        "updated_at": item["updated_at"],
        "selected_facts": len(item["group_ids"]),
        "duplicates_excluded": len(item["group_ids"]) - 1,
    }


def _approved_outcome_summary(
    groups: dict[str, dict[str, Any]],
    decision_rows: Iterable[Any],
) -> dict[str, int]:
    """Summarize the operator's saved actions, not the plan's earlier proposals."""
    counts = {
        "adopted": 0,
        "consolidated": 0,
        "obsolete": 0,
        "operator_admitted": 0,
    }
    for item in decision_rows:
        if str(item[1]) != "approve":
            continue
        group = groups[str(item[0])]
        action = _load_json(item[2], label="memory triage action")
        evidence_count = len(group["evidence"])
        kind = action.get("kind")
        if kind == "expire_all":
            counts["obsolete"] += evidence_count
        elif kind == "keep_first_retract_rest" or (kind == "adopt_corrected" and evidence_count > 1):
            counts["consolidated"] += evidence_count
            counts["operator_admitted"] += 1
        elif kind == "adopt_corrected":
            counts["adopted"] += evidence_count
            counts["operator_admitted"] += 1
        elif kind in {"adopt_as_current", "record_episode_chain"}:
            counts["adopted"] += evidence_count
            counts["operator_admitted"] += evidence_count
    return counts


def _load_json(value: object, *, label: str) -> dict[str, Any]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise MemoryReconciliationReviewError(f"Stored {label} is malformed") from exc
    if not isinstance(parsed, dict):
        raise MemoryReconciliationReviewError(f"Stored {label} is malformed")
    return parsed


def _parse_reasoner_payload(text: str) -> dict[str, Any]:
    try:
        envelope = json.loads(text)
    except json.JSONDecodeError as exc:
        raise MemoryReconciliationReviewError("Memory-quality model returned invalid JSON") from exc
    if not isinstance(envelope, dict) or envelope.get("is_error") is True:
        raise MemoryReconciliationReviewError("Memory-quality model returned an invalid response")
    structured = envelope.get("structured_output")
    payload = structured if isinstance(structured, dict) else envelope
    if not isinstance(payload.get("recommendations"), list):
        raise MemoryReconciliationReviewError("Memory-quality model omitted recommendations")
    return payload


class WorkshopMemoryReconciliationTriageService:
    """Partition an audit into safe batches and genuinely uncertain groups."""

    def __init__(
        self,
        store: WorkshopEventStore,
        *,
        db_path: Path,
        runtime_pool: WorkshopRuntimePool,
    ) -> None:
        self._store = store
        self._db_path = db_path
        self._runtime_pool = runtime_pool
        self._lock = asyncio.Lock()

    @staticmethod
    def _operation_id(value: str) -> str:
        if not value or len(value) > MAX_OPERATION_ID or any(character.isspace() for character in value):
            raise MemoryReconciliationReviewValidationError("Invalid memory triage operation identifier")
        return value

    async def _operation_replay(
        self,
        principal_id: PrincipalId,
        operation_id: str,
        request_sha256: str,
    ) -> dict[str, Any] | None:
        async with self._store.connection.execute(
            "SELECT request_sha256, response_json FROM memory_reconciliation_operations "
            "WHERE principal_id = ? AND client_operation_id = ?",
            (str(principal_id), operation_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        if str(row[0]) != request_sha256:
            raise MemoryReconciliationReviewConflict("Memory triage operation identity was reused")
        response = _load_json(row[1], label="memory triage operation")
        response["replayed"] = True
        return response

    async def _owned_audit(self, principal_id: PrincipalId, audit_id: str) -> dict[str, Any]:
        async with self._store.connection.execute(
            "SELECT principal_id, audit_json FROM memory_reconciliation_audits WHERE audit_id = ?",
            (audit_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise MemoryReconciliationReviewNotFound("Reconciliation audit not found")
        if str(row[0]) != str(principal_id):
            raise MemoryReconciliationReviewAccessDenied("Reconciliation review access denied")
        audit = _load_json(row[1], label="reconciliation audit")
        memory_reconciliation.validate_audit(audit)
        return audit

    async def _carried_deferrals(self, principal_id: PrincipalId, audit: dict[str, Any]) -> list[dict[str, Any]]:
        """
        Earlier applied deferrals of rows this audit covers, as prior review evidence.

        A deferral means "not now", so the row is reviewed again rather than
        suppressed, and triage turns this evidence into a `prior_review`
        group that shows the earlier decision and note. Only unchanged rows
        carry the deferral; a row that changed since is simply new again.
        Rows are grouped by the decision they came from, so each earlier
        group or candidate appears once.
        """
        prior = await asyncio.to_thread(
            prior_reconciliation_row_dispositions,
            self._db_path,
            principal_id=str(principal_id),
            runtime_profile_id=str(audit["runtime_profile_id"]),
        )
        carried: dict[str, dict[str, Any]] = {}
        for candidate in audit["candidates"]:
            for row in candidate["evidence"]:
                earlier = prior.get(str(row["memory_id"]))
                if (
                    earlier is None
                    or earlier.disposition != "defer"
                    or earlier.review_state != memory_reconciliation.row_review_state(row)
                ):
                    continue
                entry = carried.setdefault(
                    earlier.source_id,
                    {
                        "candidate_id": earlier.source_id,
                        "disposition": "defer",
                        "operator_note": earlier.operator_note,
                        "state_version": 0,
                        "memory_ids": [],
                    },
                )
                if str(row["memory_id"]) not in entry["memory_ids"]:
                    entry["memory_ids"].append(str(row["memory_id"]))
        return [{**entry, "memory_ids": sorted(entry["memory_ids"])} for _source_id, entry in sorted(carried.items())]

    async def _plan_row(self, principal_id: PrincipalId, plan_id: str) -> tuple[Any, dict[str, Any]]:
        async with self._store.connection.execute(
            "SELECT p.plan_id, p.audit_id, p.plan_json, p.status, p.review_version, p.applied_at "
            "FROM memory_reconciliation_triage_plans p "
            "JOIN memory_reconciliation_audits a ON a.audit_id = p.audit_id "
            "WHERE p.plan_id = ? AND a.principal_id = ?",
            (plan_id, str(principal_id)),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise MemoryReconciliationReviewNotFound("Memory triage plan not found")
        plan = _load_json(row[2], label="memory triage plan")
        validate_triage_plan(plan)
        return row, plan

    async def ensure(
        self,
        principal_id: PrincipalId,
        audit_id: str,
        *,
        allowed_project_ids: frozenset[str] | None = None,
    ) -> TriageSummary:
        audit = await self._owned_audit(principal_id, audit_id)
        candidates = {str(item["candidate_id"]): item for item in audit["candidates"]}
        async with self._store.connection.execute(
            "SELECT candidate_id, disposition, operator_note, state_version "
            "FROM memory_reconciliation_decisions WHERE audit_id = ? AND disposition != 'pending' "
            "ORDER BY candidate_id",
            (audit_id,),
        ) as cursor:
            prior_rows = await cursor.fetchall()
        prior_review_evidence = []
        for item in prior_rows:
            candidate = candidates.get(str(item[0]))
            if candidate is None:
                raise MemoryReconciliationReviewError("Stored raw review decision has no audit candidate")
            prior_review_evidence.append(
                {
                    "candidate_id": str(item[0]),
                    "disposition": str(item[1]),
                    "operator_note": str(item[2]),
                    "state_version": int(item[3]),
                    "memory_ids": sorted(str(row["memory_id"]) for row in candidate["evidence"]),
                }
            )
        prior_review_evidence.extend(await self._carried_deferrals(principal_id, audit))
        plan = build_triage_plan(
            audit,
            allowed_project_ids=allowed_project_ids,
            prior_review_evidence=prior_review_evidence,
        )
        async with self._lock:
            async with self._store.connection.execute(
                "SELECT plan_id, plan_sha256, policy_version, status FROM memory_reconciliation_triage_plans "
                "WHERE audit_id = ?",
                (audit_id,),
            ) as cursor:
                existing = await cursor.fetchone()
            if existing is not None and str(existing[2]) != str(plan["policy_version"]):
                if (
                    str(existing[3]) != "open"
                    or await memory_reconciliation.apply_in_progress(self._store.connection, str(existing[0]))
                    or await self._has_consolidations(str(existing[0]))
                ):
                    # An applied plan is history: show it exactly as it was
                    # applied, under the policy that produced it. A plan
                    # with an unfinished apply stays as it is too, because
                    # that run's progress is keyed to this plan's id.
                    row, stored = await self._plan_row(principal_id, str(existing[0]))
                    return await self._summary(row, stored)
                await self._replan(principal_id, str(existing[0]), str(existing[1]), str(existing[2]), plan)
            elif existing is None:
                now = _now()
                connection = self._store.connection
                try:
                    await connection.execute("BEGIN IMMEDIATE")
                    await connection.execute(
                        "INSERT INTO memory_reconciliation_triage_plans ("
                        "plan_id, audit_id, plan_sha256, policy_version, group_count, memory_count, plan_json, created_at"
                        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (
                            plan["plan_id"],
                            audit_id,
                            plan["sha256"],
                            plan["policy_version"],
                            plan["group_count"],
                            plan["memory_count"],
                            _canonical(plan),
                            now,
                        ),
                    )
                    await connection.executemany(
                        "INSERT INTO memory_reconciliation_triage_groups ("
                        "plan_id, group_id, state_sha256, classification, resolution, deterministic, "
                        "bulk_eligible, memory_count, action_json, updated_at"
                        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        [
                            (
                                plan["plan_id"],
                                group["group_id"],
                                group["state_sha256"],
                                group["classification"],
                                group["resolution"],
                                int(group["deterministic"]),
                                int(group["bulk_eligible"]),
                                len(group["evidence"]),
                                _canonical(group["action"]),
                                now,
                            )
                            for group in plan["groups"]
                        ],
                    )
                    await connection.commit()
                except Exception:
                    await connection.rollback()
                    raise
            elif str(existing[0]) != str(plan["plan_id"]) or str(existing[1]) != str(plan["sha256"]):
                raise MemoryReconciliationReviewConflict("Stored triage plan conflicts with the audit")
        row, stored = await self._plan_row(principal_id, str(plan["plan_id"]))
        return await self._summary(row, stored)

    async def _replan(
        self,
        principal_id: PrincipalId,
        old_plan_id: str,
        old_plan_sha256: str,
        old_policy_version: str,
        plan: dict[str, Any],
    ) -> None:
        """
        Replace an open plan built by an older policy with `plan`, keeping still-valid decisions.

        A plan's id derives from its audit and policy, and each audit has
        exactly one plan, so a policy change would otherwise make the stored
        plan unloadable. A group's id is a digest of its evidence,
        classification, and action, so a group the new policy left unchanged
        has the same id and state in both plans: its disposition, action,
        recommendation, note, and state version carry over as they are.
        Groups the new policy changed start pending, and the decisions they
        replace are recorded in `memory_reconciliation_operations` (the plan
        document itself must stay exactly what the policy produces, because
        every load compares it with a fresh rebuild).

        The plan row keeps its place and is updated to the new id, so the
        recommendation runs recorded against it move with it rather than
        being dropped. Foreign keys are checked at commit, after the
        children point at the new id. `review_version` increases so an open
        Workshop page saving against the old plan gets a conflict.
        Caller holds `self._lock`.
        """
        connection = self._store.connection
        async with connection.execute(
            "SELECT group_id, state_sha256, disposition, action_json, recommendation_json, operator_note, "
            "state_version FROM memory_reconciliation_triage_groups WHERE plan_id = ?",
            (old_plan_id,),
        ) as cursor:
            old_groups = {str(item[0]): item for item in await cursor.fetchall()}
        kept = {
            str(group["group_id"]): old_groups[str(group["group_id"])]
            for group in plan["groups"]
            if str(group["group_id"]) in old_groups
            and str(old_groups[str(group["group_id"])][1]) == str(group["state_sha256"])
        }
        dropped = [
            {"group_id": group_id, "disposition": str(item[2])}
            for group_id, item in sorted(old_groups.items())
            if group_id not in kept and str(item[2]) != "pending"
        ]
        now = _now()
        record = {
            "old_plan_id": old_plan_id,
            "old_plan_sha256": old_plan_sha256,
            "old_policy_version": old_policy_version,
            "new_plan_id": plan["plan_id"],
            "new_plan_sha256": plan["sha256"],
            "kept": len(kept),
            "dropped": dropped,
        }
        try:
            await connection.execute("BEGIN IMMEDIATE")
            await connection.execute("PRAGMA defer_foreign_keys = ON")
            cursor = await connection.execute(
                "UPDATE memory_reconciliation_triage_plans SET plan_id = ?, plan_sha256 = ?, policy_version = ?, "
                "group_count = ?, memory_count = ?, plan_json = ?, review_version = review_version + 1 "
                "WHERE plan_id = ? AND status = 'open'",
                (
                    plan["plan_id"],
                    plan["sha256"],
                    plan["policy_version"],
                    plan["group_count"],
                    plan["memory_count"],
                    _canonical(plan),
                    old_plan_id,
                ),
            )
            if cursor.rowcount != 1:
                raise MemoryReconciliationReviewConflict("Memory triage plan changed while it was being updated")
            await connection.execute(
                "UPDATE memory_reconciliation_triage_recommendations SET plan_id = ? WHERE plan_id = ?",
                (plan["plan_id"], old_plan_id),
            )
            await connection.execute(
                "DELETE FROM memory_reconciliation_triage_groups WHERE plan_id = ?", (old_plan_id,)
            )
            await connection.executemany(
                "INSERT INTO memory_reconciliation_triage_groups ("
                "plan_id, group_id, state_sha256, classification, resolution, deterministic, bulk_eligible, "
                "memory_count, disposition, action_json, recommendation_json, operator_note, state_version, updated_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        plan["plan_id"],
                        group["group_id"],
                        group["state_sha256"],
                        group["classification"],
                        group["resolution"],
                        int(group["deterministic"]),
                        int(group["bulk_eligible"]),
                        len(group["evidence"]),
                        str(kept[group["group_id"]][2]) if group["group_id"] in kept else "pending",
                        str(kept[group["group_id"]][3]) if group["group_id"] in kept else _canonical(group["action"]),
                        str(kept[group["group_id"]][4]) if group["group_id"] in kept else "{}",
                        str(kept[group["group_id"]][5]) if group["group_id"] in kept else "",
                        int(kept[group["group_id"]][6]) if group["group_id"] in kept else 0,
                        now,
                    )
                    for group in plan["groups"]
                ],
            )
            await connection.execute(
                "INSERT OR IGNORE INTO memory_reconciliation_operations ("
                "principal_id, client_operation_id, request_sha256, response_json, created_at"
                ") VALUES (?, ?, ?, ?, ?)",
                (
                    str(principal_id),
                    f"replan:{old_plan_id}",
                    _digest({"old": old_plan_sha256, "new": plan["sha256"]}),
                    _canonical(record),
                    now,
                ),
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise
        log.info(
            "Memory triage plan %s moved from policy %s to %s as %s: kept=%d, reopened=%d",
            old_plan_id,
            old_policy_version,
            plan["policy_version"],
            plan["plan_id"],
            len(kept),
            len(dropped),
        )

    async def latest(
        self,
        principal_id: PrincipalId,
        *,
        allowed_project_ids: frozenset[str] | None = None,
    ) -> TriageSummary | None:
        async with self._store.connection.execute(
            "SELECT audit_id FROM memory_reconciliation_audits WHERE principal_id = ? "
            "ORDER BY generated_at DESC, created_at DESC LIMIT 1",
            (str(principal_id),),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        return await self.ensure(principal_id, str(row[0]), allowed_project_ids=allowed_project_ids)

    async def _summary(self, row: Any, plan: dict[str, Any]) -> TriageSummary:
        async with self._store.connection.execute(
            "SELECT resolution, disposition, deterministic, recommendation_json, memory_count "
            "FROM memory_reconciliation_triage_groups WHERE plan_id = ?",
            (row[0],),
        ) as cursor:
            group_rows = list(await cursor.fetchall())
        resolutions: dict[str, int] = {key: 0 for key in ("adopt", "consolidate", "obsolete", "needs_review")}
        dispositions: dict[str, int] = {key: 0 for key in ("pending", "approve", "reject", "defer")}
        deterministic = pending_deterministic = recommended = 0
        for item in group_rows:
            resolutions[str(item[0])] += int(item[4])
            dispositions[str(item[1])] += 1
            deterministic += int(item[2])
            if int(item[2]) and str(item[1]) == "pending":
                pending_deterministic += 1
            if _load_json(item[3], label="memory triage recommendation"):
                recommended += 1
        return TriageSummary(
            plan_id=str(row[0]),
            audit_id=str(row[1]),
            status=str(row[3]),
            review_version=int(row[4]),
            memory_count=int(plan["memory_count"]),
            group_count=int(plan["group_count"]),
            resolution_counts=resolutions,
            disposition_counts=dispositions,
            deterministic_groups=deterministic,
            pending_deterministic_groups=pending_deterministic,
            exception_groups=len(group_rows) - deterministic,
            recommended_groups=recommended,
            applied_at=None if row[5] is None else str(row[5]),
        )

    @staticmethod
    def _client_evidence(item: dict[str, Any]) -> dict[str, Any]:
        return {
            key: item.get(key)
            for key in (
                "memory_id",
                "kind",
                "text",
                "created_at",
                "updated_at",
                "scope",
                "project_id",
                "source",
                "confidence",
                "backend",
                "provider",
                "model",
                "prompt_version",
                "schema_version",
                "valid_from",
                "valid_until",
                "asserted_at",
                "observed_at",
                "occurred_from",
                "occurred_until",
                "migration_classification",
                "migration_gaps",
            )
        }

    async def groups(
        self,
        principal_id: PrincipalId,
        plan_id: str,
        *,
        disposition: str | None = None,
        resolution: str | None = None,
        exceptions_only: bool = False,
        offset: int = 0,
        limit: int = 50,
    ) -> TriageGroupPage:
        if offset < 0 or not 1 <= limit <= 100:
            raise MemoryReconciliationReviewValidationError("Invalid memory triage page")
        row, plan = await self._plan_row(principal_id, plan_id)
        query = (
            "SELECT group_id, disposition, action_json, recommendation_json, operator_note, state_version "
            "FROM memory_reconciliation_triage_groups WHERE plan_id = ?"
        )
        parameters: list[object] = [plan_id]
        if disposition:
            query += " AND disposition = ?"
            parameters.append(disposition)
        if resolution:
            query += " AND resolution = ?"
            parameters.append(resolution)
        if exceptions_only:
            query += " AND deterministic = 0"
        async with self._store.connection.execute(query, parameters) as cursor:
            states = {
                str(item[0]): {
                    "disposition": str(item[1]),
                    "action": _load_json(item[2], label="memory triage action"),
                    "recommendation": _load_json(item[3], label="memory triage recommendation"),
                    "operator_note": str(item[4]),
                    "state_version": int(item[5]),
                }
                for item in await cursor.fetchall()
            }
        async with self._store.connection.execute(
            "SELECT group_id, consolidation_id FROM memory_reconciliation_triage_consolidation_members "
            "WHERE plan_id = ?",
            (plan_id,),
        ) as cursor:
            membership = {str(item[0]): str(item[1]) for item in await cursor.fetchall()}
        plan_groups = {str(item["group_id"]): item for item in plan["groups"]}
        visible = []
        for group in plan["groups"]:
            state = states.get(str(group["group_id"]))
            if state is None:
                continue
            unchanged_prior_review_action = _unchanged_prior_review_action(
                group,
                audit_boundary=plan["generated_at"],
            )
            visible.append(
                {
                    **{
                        key: group[key]
                        for key in (
                            "group_id",
                            "state_sha256",
                            "classification",
                            "resolution",
                            "rationale",
                            "deterministic",
                            "bulk_eligible",
                        )
                    },
                    "proposed_action": unchanged_prior_review_action or group["action"],
                    "prior_review_evidence": group["prior_review_evidence"],
                    # Names of the schema fields an incomplete episode lacks;
                    # empty for every other group.
                    "missing_fields": list(group.get("missing_fields", [])),
                    # Other fact groups the recommendation links to this one,
                    # checked against the plan, for "Consolidate with...".
                    "related_group_ids": related_group_ids(group, state["recommendation"], plan_groups),
                    "consolidation_id": membership.get(str(group["group_id"])),
                    "evidence": [self._client_evidence(item) for item in group["evidence"]],
                    "decision": state,
                }
            )
        selected = visible[offset : offset + limit]
        return TriageGroupPage(
            triage=await self._summary(row, plan),
            groups=tuple(selected),
            next_offset=offset + limit if offset + limit < len(visible) else None,
        )

    async def preview_safe_approval(
        self,
        principal_id: PrincipalId,
        plan_id: str,
        *,
        group_ids: list[str] | None,
        expected_review_version: int,
    ) -> dict[str, Any]:
        row, plan = await self._plan_row(principal_id, plan_id)
        if str(row[3]) != "open" or int(row[4]) != expected_review_version:
            raise MemoryReconciliationReviewConflict("Memory triage plan changed after it was loaded")
        selected_ids = set(group_ids or [])
        if group_ids is not None and (not selected_ids or len(selected_ids) != len(group_ids)):
            raise MemoryReconciliationReviewValidationError("Invalid memory triage group selection")
        async with self._store.connection.execute(
            "SELECT group_id, state_sha256, resolution, memory_count, action_json "
            "FROM memory_reconciliation_triage_groups "
            "WHERE plan_id = ? AND deterministic = 1 AND bulk_eligible = 1 AND disposition = 'pending'",
            (plan_id,),
        ) as cursor:
            eligible = await cursor.fetchall()
        chosen = [item for item in eligible if not selected_ids or str(item[0]) in selected_ids]
        if selected_ids != {str(item[0]) for item in chosen} and selected_ids:
            raise MemoryReconciliationReviewValidationError(
                "Bulk approval is limited to pending deterministic safe groups"
            )
        if not chosen:
            raise MemoryReconciliationReviewValidationError("No deterministic safe groups are pending")
        actions = {str(item[4]) for item in chosen}
        if len(actions) != 1:
            raise MemoryReconciliationReviewValidationError("Selected safe groups must have one homogeneous action")
        binding = {
            "plan_sha256": plan["sha256"],
            "review_version": expected_review_version,
            "groups": [{"group_id": str(item[0]), "state_sha256": str(item[1])} for item in chosen],
        }
        counts = {key: 0 for key in ("adopt", "consolidate", "obsolete")}
        for item in chosen:
            counts[str(item[2])] += int(item[3])
        return {
            "plan_id": plan_id,
            "preview_sha256": _digest(binding),
            "group_ids": [str(item[0]) for item in chosen],
            "group_count": len(chosen),
            "memory_count": sum(int(item[3]) for item in chosen),
            "resolution_counts": counts,
            "review_version": expected_review_version,
        }

    async def approve_safe(
        self,
        principal_id: PrincipalId,
        plan_id: str,
        *,
        group_ids: list[str] | None,
        expected_review_version: int,
        preview_sha256: str,
        operator_note: str,
        client_operation_id: str,
    ) -> dict[str, Any]:
        if not isinstance(operator_note, str) or len(operator_note) > MAX_OPERATOR_NOTE:
            raise MemoryReconciliationReviewValidationError("Invalid memory triage operator note")
        operation_id = self._operation_id(client_operation_id)
        request_sha256 = _digest(
            {
                "kind": "triage_safe_approve",
                "plan_id": plan_id,
                "group_ids": sorted(group_ids or []),
                "expected_review_version": expected_review_version,
                "preview_sha256": preview_sha256,
                "operator_note": operator_note,
            }
        )
        _, bound_plan = await self._plan_row(principal_id, plan_id)
        async with self._lock:
            replay = await self._operation_replay(principal_id, operation_id, request_sha256)
            if replay is not None:
                return replay
            if await memory_reconciliation.apply_in_progress(self._store.connection, plan_id):
                raise MemoryReconciliationReviewConflict(_APPLY_IN_PROGRESS)
            preview = await self.preview_safe_approval(
                principal_id,
                plan_id,
                group_ids=group_ids,
                expected_review_version=expected_review_version,
            )
            if preview_sha256 != preview["preview_sha256"]:
                raise MemoryReconciliationReviewConflict("Memory triage approval does not match its preview")
            placeholders = ",".join("?" for _ in preview["group_ids"])
            connection = self._store.connection
            try:
                await connection.execute("BEGIN IMMEDIATE")
                cursor = await connection.execute(
                    f"UPDATE memory_reconciliation_triage_groups SET disposition = 'approve', "
                    f"operator_note = ?, state_version = state_version + 1, updated_at = ? "
                    f"WHERE plan_id = ? AND group_id IN ({placeholders}) AND disposition = 'pending'",
                    (operator_note, _now(), plan_id, *preview["group_ids"]),
                )
                if cursor.rowcount != preview["group_count"]:
                    raise MemoryReconciliationReviewConflict("Memory triage groups changed after preview")
                await connection.execute(
                    "UPDATE memory_reconciliation_triage_plans SET review_version = review_version + 1 WHERE plan_id = ?",
                    (plan_id,),
                )
                response = {
                    **preview,
                    "audit_id": bound_plan["audit_id"],
                    "review_version": expected_review_version + 1,
                    "replayed": False,
                }
                await connection.execute(
                    "INSERT INTO memory_reconciliation_operations ("
                    "principal_id, client_operation_id, request_sha256, response_json, created_at"
                    ") VALUES (?, ?, ?, ?, ?)",
                    (str(principal_id), operation_id, request_sha256, _canonical(response), _now()),
                )
                await connection.commit()
            except Exception:
                await connection.rollback()
                raise
        return response

    async def decide_group(
        self,
        principal_id: PrincipalId,
        plan_id: str,
        group_id: str,
        *,
        disposition: Literal["approve", "reject", "defer"],
        action: dict[str, Any],
        operator_note: str,
        expected_state_version: int,
        allowed_project_ids: frozenset[str],
        client_operation_id: str,
    ) -> dict[str, Any]:
        operation_id = self._operation_id(client_operation_id)
        request_sha256 = _digest(
            {
                "kind": "triage_group_decision",
                "plan_id": plan_id,
                "group_id": group_id,
                "disposition": disposition,
                "action": action,
                "operator_note": operator_note,
                "expected_state_version": expected_state_version,
            }
        )
        replay = await self._operation_replay(principal_id, operation_id, request_sha256)
        if replay is not None:
            return replay
        row, plan = await self._plan_row(principal_id, plan_id)
        if str(row[3]) != "open":
            raise MemoryReconciliationReviewConflict("Memory triage plan has already been applied")
        if await memory_reconciliation.apply_in_progress(self._store.connection, plan_id):
            raise MemoryReconciliationReviewConflict(_APPLY_IN_PROGRESS)
        group = next((item for item in plan["groups"] if item["group_id"] == group_id), None)
        if group is None:
            raise MemoryReconciliationReviewNotFound("Memory triage group not found")
        async with self._store.connection.execute(
            "SELECT 1 FROM memory_reconciliation_triage_consolidation_members WHERE plan_id = ? AND group_id = ?",
            (plan_id, group_id),
        ) as cursor:
            if await cursor.fetchone() is not None:
                raise MemoryReconciliationReviewConflict(
                    "This fact is part of a consolidation; revise or cancel the consolidation instead"
                )
        if disposition == "approve":
            if not _operator_action_is_authorized(group, action, audit_boundary=plan["generated_at"]):
                raise MemoryReconciliationReviewValidationError(
                    "Memory triage approval does not match an authorized action"
                )
            synthetic = {"evidence": group["evidence"]}
            try:
                memory_reconciliation.validate_candidate_action(synthetic, action)
            except memory_reconciliation.MemoryReconciliationError as exc:
                raise MemoryReconciliationReviewValidationError(str(exc)) from exc
            project_ids = {
                str(item["project_id"])
                for item in group["evidence"]
                if item.get("scope") == "project" and item.get("project_id")
            }
            replacement = action.get("replacement")
            if (
                action.get("kind") == "adopt_corrected"
                and isinstance(replacement, dict)
                and replacement.get("scope_kind") == "project"
                and replacement.get("scope_key")
            ):
                project_ids.add(str(replacement["scope_key"]))
            if project_ids - allowed_project_ids:
                raise MemoryReconciliationReviewAccessDenied(
                    "Memory triage action references a project outside current authority"
                )
        else:
            action = dict(group["action"])
        if not isinstance(operator_note, str) or len(operator_note) > MAX_OPERATOR_NOTE:
            raise MemoryReconciliationReviewValidationError("Invalid memory triage operator note")
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            cursor = await connection.execute(
                "UPDATE memory_reconciliation_triage_groups SET disposition = ?, action_json = ?, operator_note = ?, "
                "state_version = state_version + 1, updated_at = ? WHERE plan_id = ? AND group_id = ? "
                "AND state_version = ?",
                (
                    disposition,
                    _canonical(action),
                    operator_note,
                    _now(),
                    plan_id,
                    group_id,
                    expected_state_version,
                ),
            )
            if cursor.rowcount != 1:
                raise MemoryReconciliationReviewConflict("Memory triage group changed after it was loaded")
            await connection.execute(
                "UPDATE memory_reconciliation_triage_plans SET review_version = review_version + 1 WHERE plan_id = ?",
                (plan_id,),
            )
            response = {
                "audit_id": plan["audit_id"],
                "plan_id": plan_id,
                "group_id": group_id,
                "disposition": disposition,
                "replayed": False,
            }
            await connection.execute(
                "INSERT INTO memory_reconciliation_operations ("
                "principal_id, client_operation_id, request_sha256, response_json, created_at"
                ") VALUES (?, ?, ?, ?, ?)",
                (str(principal_id), operation_id, request_sha256, _canonical(response), _now()),
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise
        return response

    async def _consolidations(self, plan_id: str) -> list[dict[str, Any]]:
        """Return the plan's staged consolidations with their members in selection order."""
        async with self._store.connection.execute(
            "SELECT consolidation_id, revision, canonical_json, selection_sha256, operator_note, updated_at "
            "FROM memory_reconciliation_triage_consolidations WHERE plan_id = ? ORDER BY created_at, consolidation_id",
            (plan_id,),
        ) as cursor:
            rows = await cursor.fetchall()
        async with self._store.connection.execute(
            "SELECT consolidation_id, group_id, prior_decision_json FROM "
            "memory_reconciliation_triage_consolidation_members WHERE plan_id = ? ORDER BY consolidation_id, position",
            (plan_id,),
        ) as cursor:
            members = await cursor.fetchall()
        by_id: dict[str, list[tuple[str, dict[str, Any]]]] = {}
        for item in members:
            by_id.setdefault(str(item[0]), []).append(
                (str(item[1]), _load_json(item[2], label="memory consolidation member"))
            )
        return [
            {
                "consolidation_id": str(row[0]),
                "revision": int(row[1]),
                "canonical": _load_json(row[2], label="memory consolidation"),
                "selection_sha256": str(row[3]),
                "operator_note": str(row[4]),
                "updated_at": str(row[5]),
                "group_ids": [group_id for group_id, _prior in by_id.get(str(row[0]), [])],
                "prior_decisions": dict(by_id.get(str(row[0]), [])),
            }
            for row in rows
        ]

    async def consolidations(self, principal_id: PrincipalId, plan_id: str) -> list[dict[str, Any]]:
        """List the plan's staged consolidations for the signed-in owner."""
        await self._plan_row(principal_id, plan_id)
        return [_public_consolidation(item) for item in await self._consolidations(plan_id)]

    async def stage_consolidation(
        self,
        principal_id: PrincipalId,
        plan_id: str,
        *,
        group_ids: list[str],
        expected_state_versions: dict[str, int],
        canonical: dict[str, Any],
        operator_note: str,
        consolidation_id: str | None,
        expected_revision: int | None,
        allowed_project_ids: frozenset[str],
        client_operation_id: str,
    ) -> dict[str, Any]:
        """
        Stage (or revise) one consolidation of several facts into one canonical fact.

        Every selected group must be a fact group of this open plan, at the
        state version the owner saw, and not part of another consolidation.
        The final wording and scope are checked exactly as an approved
        corrected adoption would be, including the schema and project
        authority. In one transaction each member's current decision is
        kept for restoring later, and every member is marked as belonging
        to the consolidation, so none of them can be decided on its own.
        Revising replaces the selection and canonical fact; facts dropped
        from it get their earlier decision back. Any failure changes nothing.
        """
        operation_id = self._operation_id(client_operation_id)
        request_sha256 = _digest(
            {
                "kind": "triage_consolidation",
                "plan_id": plan_id,
                "group_ids": group_ids,
                "expected_state_versions": expected_state_versions,
                "canonical": canonical,
                "operator_note": operator_note,
                "consolidation_id": consolidation_id,
                "expected_revision": expected_revision,
            }
        )
        async with self._lock:
            replay = await self._operation_replay(principal_id, operation_id, request_sha256)
            if replay is not None:
                return replay
            row, plan = await self._plan_row(principal_id, plan_id)
            if str(row[3]) != "open":
                raise MemoryReconciliationReviewConflict("Memory triage plan has already been applied")
            if await memory_reconciliation.apply_in_progress(self._store.connection, plan_id):
                raise MemoryReconciliationReviewConflict(_APPLY_IN_PROGRESS)
            if not isinstance(operator_note, str) or len(operator_note) > MAX_OPERATOR_NOTE:
                raise MemoryReconciliationReviewValidationError("Invalid memory triage operator note")
            if (
                not isinstance(group_ids, list)
                or not all(isinstance(group_id, str) for group_id in group_ids)
                or not MIN_CONSOLIDATION_FACTS <= len(group_ids) <= MAX_CONSOLIDATION_FACTS
                or len(set(group_ids)) != len(group_ids)
                or not isinstance(expected_state_versions, dict)
                or set(expected_state_versions) != set(group_ids)
            ):
                raise MemoryReconciliationReviewValidationError(
                    f"Select between {MIN_CONSOLIDATION_FACTS} and {MAX_CONSOLIDATION_FACTS} different facts"
                )
            plan_groups = {str(item["group_id"]): item for item in plan["groups"]}
            if any(group_id not in plan_groups for group_id in group_ids):
                raise MemoryReconciliationReviewNotFound("A selected fact is not part of this plan")
            rows = [item for group_id in group_ids for item in plan_groups[group_id]["evidence"]]
            if any(item.get("kind") != "fact" for item in rows):
                raise MemoryReconciliationReviewValidationError("Only facts can be consolidated; episodes stay history")
            action = _consolidation_action(canonical, rows)
            try:
                memory_reconciliation.validate_candidate_action({"evidence": rows}, action)
            except memory_reconciliation.MemoryReconciliationError as exc:
                raise MemoryReconciliationReviewValidationError(str(exc)) from exc
            project_ids = {
                str(item["project_id"]) for item in rows if item.get("scope") == "project" and item.get("project_id")
            }
            if action["replacement"]["scope_kind"] == "project":
                project_ids.add(str(action["replacement"]["scope_key"]))
            if project_ids - allowed_project_ids:
                raise MemoryReconciliationReviewAccessDenied(
                    "The consolidation references a project outside current authority"
                )

            existing = None
            if consolidation_id is not None:
                existing = next(
                    (
                        item
                        for item in await self._consolidations(plan_id)
                        if item["consolidation_id"] == consolidation_id
                    ),
                    None,
                )
                if existing is None:
                    raise MemoryReconciliationReviewNotFound("Memory consolidation not found")
                if existing["revision"] != expected_revision:
                    raise MemoryReconciliationReviewConflict("This consolidation changed after it was loaded")
            placeholders = ", ".join("?" for _ in group_ids)
            async with self._store.connection.execute(
                "SELECT g.group_id, g.disposition, g.action_json, g.operator_note, g.state_version, m.consolidation_id "
                "FROM memory_reconciliation_triage_groups g "
                "LEFT JOIN memory_reconciliation_triage_consolidation_members m "
                "ON m.plan_id = g.plan_id AND m.group_id = g.group_id "
                f"WHERE g.plan_id = ? AND g.group_id IN ({placeholders})",
                (plan_id, *group_ids),
            ) as cursor:
                states = {str(item[0]): item for item in await cursor.fetchall()}
            for group_id in group_ids:
                state = states[group_id]
                if int(state[4]) != int(expected_state_versions[group_id]):
                    raise MemoryReconciliationReviewConflict("A selected fact changed after it was loaded")
                if state[5] is not None and str(state[5]) != consolidation_id:
                    raise MemoryReconciliationReviewConflict("A selected fact already belongs to another consolidation")

            new_id = consolidation_id or f"mcn_{_digest({'plan': plan_id, 'operation': operation_id})[:32]}"
            revision = 1 if existing is None else int(existing["revision"]) + 1
            earlier: dict[str, dict[str, Any]] = existing["prior_decisions"] if existing is not None else {}
            dropped = [
                group_id for group_id in (existing["group_ids"] if existing else []) if group_id not in group_ids
            ]
            selection_sha256 = _digest(
                {
                    "plan_id": plan_id,
                    "members": [
                        {"group_id": group_id, "state_sha256": plan_groups[group_id]["state_sha256"]}
                        for group_id in group_ids
                    ],
                    "action": action,
                }
            )
            now = _now()
            marker = _canonical({"kind": "consolidate", "consolidation_id": new_id})
            connection = self._store.connection
            try:
                await connection.execute("BEGIN IMMEDIATE")
                for group_id in dropped:
                    await self._restore_decision(plan_id, group_id, earlier[group_id], now)
                await connection.execute(
                    "DELETE FROM memory_reconciliation_triage_consolidation_members WHERE consolidation_id = ?",
                    (new_id,),
                )
                await connection.execute(
                    "INSERT INTO memory_reconciliation_triage_consolidations ("
                    "consolidation_id, plan_id, revision, canonical_json, selection_sha256, operator_note, "
                    "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                    "ON CONFLICT (consolidation_id) DO UPDATE SET revision = excluded.revision, "
                    "canonical_json = excluded.canonical_json, selection_sha256 = excluded.selection_sha256, "
                    "operator_note = excluded.operator_note, updated_at = excluded.updated_at",
                    (new_id, plan_id, revision, _canonical(action), selection_sha256, operator_note, now, now),
                )
                for position, group_id in enumerate(group_ids):
                    state = states[group_id]
                    prior = earlier.get(group_id) or {
                        "disposition": str(state[1]),
                        "action_json": str(state[2]),
                        "operator_note": str(state[3]),
                    }
                    await connection.execute(
                        "INSERT INTO memory_reconciliation_triage_consolidation_members ("
                        "plan_id, group_id, consolidation_id, position, prior_decision_json) VALUES (?, ?, ?, ?, ?)",
                        (plan_id, group_id, new_id, position, _canonical(prior)),
                    )
                    await connection.execute(
                        "UPDATE memory_reconciliation_triage_groups SET disposition = 'approve', action_json = ?, "
                        "operator_note = ?, state_version = state_version + 1, updated_at = ? "
                        "WHERE plan_id = ? AND group_id = ?",
                        (marker, operator_note, now, plan_id, group_id),
                    )
                await connection.execute(
                    "UPDATE memory_reconciliation_triage_plans SET review_version = review_version + 1 "
                    "WHERE plan_id = ?",
                    (plan_id,),
                )
                response = {
                    "plan_id": plan_id,
                    "consolidation": _public_consolidation(
                        {
                            "consolidation_id": new_id,
                            "revision": revision,
                            "canonical": action,
                            "operator_note": operator_note,
                            "group_ids": list(group_ids),
                            "updated_at": now,
                        }
                    ),
                    "replayed": False,
                }
                await connection.execute(
                    "INSERT INTO memory_reconciliation_operations ("
                    "principal_id, client_operation_id, request_sha256, response_json, created_at"
                    ") VALUES (?, ?, ?, ?, ?)",
                    (str(principal_id), operation_id, request_sha256, _canonical(response), now),
                )
                await connection.commit()
            except Exception:
                await connection.rollback()
                raise
            return response

    async def cancel_consolidation(
        self,
        principal_id: PrincipalId,
        plan_id: str,
        consolidation_id: str,
        *,
        expected_revision: int,
        client_operation_id: str,
    ) -> dict[str, Any]:
        """Remove a staged consolidation and give every member its earlier decision back."""
        operation_id = self._operation_id(client_operation_id)
        request_sha256 = _digest(
            {
                "kind": "triage_consolidation_cancel",
                "plan_id": plan_id,
                "consolidation_id": consolidation_id,
                "expected_revision": expected_revision,
            }
        )
        async with self._lock:
            replay = await self._operation_replay(principal_id, operation_id, request_sha256)
            if replay is not None:
                return replay
            row, _plan = await self._plan_row(principal_id, plan_id)
            if str(row[3]) != "open":
                raise MemoryReconciliationReviewConflict("Memory triage plan has already been applied")
            if await memory_reconciliation.apply_in_progress(self._store.connection, plan_id):
                raise MemoryReconciliationReviewConflict(_APPLY_IN_PROGRESS)
            existing = next(
                (item for item in await self._consolidations(plan_id) if item["consolidation_id"] == consolidation_id),
                None,
            )
            if existing is None:
                raise MemoryReconciliationReviewNotFound("Memory consolidation not found")
            if existing["revision"] != expected_revision:
                raise MemoryReconciliationReviewConflict("This consolidation changed after it was loaded")
            now = _now()
            connection = self._store.connection
            try:
                await connection.execute("BEGIN IMMEDIATE")
                for group_id, prior in existing["prior_decisions"].items():
                    await self._restore_decision(plan_id, group_id, prior, now)
                await connection.execute(
                    "DELETE FROM memory_reconciliation_triage_consolidations WHERE consolidation_id = ?",
                    (consolidation_id,),
                )
                await connection.execute(
                    "UPDATE memory_reconciliation_triage_plans SET review_version = review_version + 1 "
                    "WHERE plan_id = ?",
                    (plan_id,),
                )
                response = {
                    "plan_id": plan_id,
                    "consolidation_id": consolidation_id,
                    "cancelled": True,
                    "replayed": False,
                }
                await connection.execute(
                    "INSERT INTO memory_reconciliation_operations ("
                    "principal_id, client_operation_id, request_sha256, response_json, created_at"
                    ") VALUES (?, ?, ?, ?, ?)",
                    (str(principal_id), operation_id, request_sha256, _canonical(response), now),
                )
                await connection.commit()
            except Exception:
                await connection.rollback()
                raise
            return response

    async def _with_consolidations(
        self,
        plan_id: str,
        plan: dict[str, Any],
        decision_rows: list[Any],
    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[Any]]:
        """
        Replace each consolidation's member groups with one combined candidate for apply.

        The combined candidate holds every member's evidence and carries the
        consolidation's corrected adoption as its approved decision, so apply
        records one canonical fact citing all of them. Its id is the
        consolidation id and its state digest is the selection digest, so a
        resumed apply keys its progress to exactly what was staged. Before
        that, every member must still carry this consolidation's marker and
        the selection must hash to what was staged; otherwise apply refuses.
        """
        plan_groups = {str(item["group_id"]): item for item in plan["groups"]}
        consolidations = await self._consolidations(plan_id)
        if not consolidations:
            return list(plan["groups"]), plan_groups, decision_rows
        stored = {str(item[0]): item for item in decision_rows}
        members: set[str] = set()
        combined_groups: list[dict[str, Any]] = []
        combined_rows: list[Any] = []
        for consolidation in consolidations:
            consolidation_id = consolidation["consolidation_id"]
            marker = {"kind": "consolidate", "consolidation_id": consolidation_id}
            for group_id in consolidation["group_ids"]:
                if _load_json(stored[group_id][2], label="memory triage action") != marker:
                    raise MemoryReconciliationReviewConflict("A consolidated fact's decision changed; reopen it")
            selection_sha256 = _digest(
                {
                    "plan_id": plan_id,
                    "members": [
                        {"group_id": group_id, "state_sha256": plan_groups[group_id]["state_sha256"]}
                        for group_id in consolidation["group_ids"]
                    ],
                    "action": consolidation["canonical"],
                }
            )
            if selection_sha256 != consolidation["selection_sha256"]:
                raise MemoryReconciliationReviewConflict("A consolidation no longer matches its facts; reopen it")
            members.update(consolidation["group_ids"])
            combined_groups.append(
                {
                    "group_id": consolidation_id,
                    "state_sha256": selection_sha256,
                    "classification": "consolidation",
                    "resolution": "consolidate",
                    "deterministic": False,
                    "bulk_eligible": False,
                    "rationale": f"Operator consolidation of {len(consolidation['group_ids'])} facts into one.",
                    "action": {"kind": "manual_edit_required"},
                    "prior_review_evidence": [],
                    "evidence": [
                        item for group_id in consolidation["group_ids"] for item in plan_groups[group_id]["evidence"]
                    ],
                }
            )
            combined_rows.append(
                (consolidation_id, "approve", _canonical(consolidation["canonical"]), consolidation["operator_note"])
            )
        apply_groups = [group for group in plan["groups"] if str(group["group_id"]) not in members] + combined_groups
        return (
            apply_groups,
            {str(group["group_id"]): group for group in apply_groups},
            [item for item in decision_rows if str(item[0]) not in members] + combined_rows,
        )

    async def _has_consolidations(self, plan_id: str) -> bool:
        """
        Return True when the plan has staged consolidations.

        A policy replan rebuilds groups and could change or drop members, so
        a plan with consolidations keeps its current policy until they are
        applied or cancelled.
        """
        async with self._store.connection.execute(
            "SELECT 1 FROM memory_reconciliation_triage_consolidations WHERE plan_id = ? LIMIT 1",
            (plan_id,),
        ) as cursor:
            return await cursor.fetchone() is not None

    async def _restore_decision(self, plan_id: str, group_id: str, prior: dict[str, Any], now: str) -> None:
        """Put back the decision a group had before it joined a consolidation. Caller holds a transaction."""
        await self._store.connection.execute(
            "UPDATE memory_reconciliation_triage_groups SET disposition = ?, action_json = ?, operator_note = ?, "
            "state_version = state_version + 1, updated_at = ? WHERE plan_id = ? AND group_id = ?",
            (
                str(prior["disposition"]),
                str(prior["action_json"]),
                str(prior["operator_note"]),
                now,
                plan_id,
                group_id,
            ),
        )

    async def recommend(self, principal_id: PrincipalId, plan_id: str) -> dict[str, Any]:
        row, plan = await self._plan_row(principal_id, plan_id)
        if str(row[3]) != "open":
            raise MemoryReconciliationReviewConflict("Memory triage plan has already been applied")
        async with self._store.connection.execute(
            "SELECT group_id FROM memory_reconciliation_triage_groups "
            "WHERE plan_id = ? AND deterministic = 0 AND recommendation_json = '{}' "
            "ORDER BY group_id LIMIT 100",
            (plan_id,),
        ) as cursor:
            target_rows = await cursor.fetchall()
        target_ids = {str(item[0]) for item in target_rows}
        targets = [group for group in plan["groups"] if group["group_id"] in target_ids]
        if not targets:
            return {"plan_id": plan_id, "recommended": 0, "remaining": 0}
        runtime = RuntimeProfileId(str(plan["runtime_profile_id"]))
        backend, provider = self._runtime_pool.get_backend_provider(runtime)
        model = self._runtime_pool.get_role_model(runtime, ModelRole.MEMORY_RECONCILIATION)
        profile = self._runtime_pool.runtime_profile(runtime)
        reasoner = build_memory_reasoner(backend, os_user=profile.os_user, provider=provider)
        input_groups = [
            {
                "group_id": group["group_id"],
                "classification": group["classification"],
                "evidence": [
                    {
                        "kind": item["kind"],
                        "text": item["text"],
                        "scope": item["scope"],
                        "created_at": item["created_at"],
                        "updated_at": item["updated_at"],
                        "valid_until": item["valid_until"],
                    }
                    for item in group["evidence"]
                ],
            }
            for group in targets
        ]
        prompt_input = {"policy": PROMPT_VERSION, "groups": input_groups}
        input_sha256 = _digest(prompt_input)
        recommendation_id = f"mtr_{_digest({'plan': plan_id, 'input': input_sha256})[:32]}"
        now = _now()
        await self._store.connection.execute(
            "INSERT OR IGNORE INTO memory_reconciliation_triage_recommendations ("
            "recommendation_id, plan_id, status, backend, provider, model, prompt_version, input_sha256, "
            "group_count, created_at) VALUES (?, ?, 'running', ?, ?, ?, ?, ?, ?, ?)",
            (
                recommendation_id,
                plan_id,
                backend,
                provider,
                model,
                PROMPT_VERSION,
                input_sha256,
                len(targets),
                now,
            ),
        )
        await self._store.connection.commit()
        try:
            result = await reasoner.run(
                prompt=_canonical(prompt_input),
                system_prompt=(
                    "You are reviewing incomplete legacy memory evidence. Recommend an outcome for each group. "
                    "Recommendations are advisory and must never claim missing provenance. Use needs_review when "
                    "evidence is ambiguous. Return every supplied group exactly once. If another supplied group "
                    "contains related duplicate evidence, list its exact group_id in related_group_ids; otherwise "
                    "return an empty list. Never invent or mention an unavailable group identifier."
                ),
                model=model,
                timeout=300,
                purpose="memory_reconciliation_triage",
                json_schema=_RECOMMENDATION_SCHEMA,
            )
            payload = _parse_reasoner_payload(result.text)
            expected = {group["group_id"] for group in targets}
            recommendations: dict[str, dict[str, Any]] = {}
            for raw in payload["recommendations"]:
                if not isinstance(raw, dict) or set(raw) != {
                    "group_id",
                    "outcome",
                    "confidence",
                    "rationale",
                    "related_group_ids",
                }:
                    raise MemoryReconciliationReviewError("Memory-quality recommendation is malformed")
                group_id = raw["group_id"]
                if group_id not in expected or group_id in recommendations:
                    raise MemoryReconciliationReviewError("Memory-quality recommendation references an invalid group")
                if raw["outcome"] not in {"adopt", "consolidate", "obsolete", "needs_review"}:
                    raise MemoryReconciliationReviewError("Memory-quality recommendation has an invalid outcome")
                related_group_ids = raw["related_group_ids"]
                if (
                    isinstance(raw["confidence"], bool)
                    or not isinstance(raw["confidence"], int | float)
                    or not 0 <= float(raw["confidence"]) <= 1
                    or not isinstance(raw["rationale"], str)
                    or not isinstance(related_group_ids, list)
                    or not all(isinstance(value, str) for value in related_group_ids)
                    or len(related_group_ids) != len(set(related_group_ids))
                    or group_id in related_group_ids
                    or not set(related_group_ids).issubset(expected)
                ):
                    raise MemoryReconciliationReviewError("Memory-quality recommendation is malformed")
                recommendations[str(group_id)] = {
                    **raw,
                    "backend": backend,
                    "provider": provider,
                    "model": model,
                    "prompt_version": PROMPT_VERSION,
                    "input_sha256": input_sha256,
                    "output_sha256": _digest(payload),
                }
            if set(recommendations) != expected:
                raise MemoryReconciliationReviewError("Memory-quality model did not return every unresolved group")
            connection = self._store.connection
            try:
                await connection.execute("BEGIN IMMEDIATE")
                for group_id, recommendation in recommendations.items():
                    cursor = await connection.execute(
                        "UPDATE memory_reconciliation_triage_groups SET recommendation_json = ?, updated_at = ? "
                        "WHERE plan_id = ? AND group_id = ? AND recommendation_json = '{}' AND EXISTS ("
                        "SELECT 1 FROM memory_reconciliation_triage_plans p "
                        "WHERE p.plan_id = ? AND p.status = 'open')",
                        (_canonical(recommendation), _now(), plan_id, group_id, plan_id),
                    )
                    if cursor.rowcount != 1:
                        raise MemoryReconciliationReviewConflict(
                            "Memory triage changed while recommendations were being generated"
                        )
                await connection.execute(
                    "UPDATE memory_reconciliation_triage_recommendations SET status = 'succeeded', "
                    "output_sha256 = ?, completed_at = ? WHERE recommendation_id = ?",
                    (_digest(payload), _now(), recommendation_id),
                )
                await connection.commit()
            except Exception:
                await connection.rollback()
                raise
            async with self._store.connection.execute(
                "SELECT COUNT(*) FROM memory_reconciliation_triage_groups "
                "WHERE plan_id = ? AND deterministic = 0 AND recommendation_json = '{}'",
                (plan_id,),
            ) as cursor:
                remaining_row = await cursor.fetchone()
            remaining = int(remaining_row[0]) if remaining_row is not None else 0
            return {"plan_id": plan_id, "recommended": len(recommendations), "remaining": remaining}
        except Exception as exc:
            error_code = "provider_failure" if isinstance(exc, OneShotError) else "invalid_recommendation"
            await self._store.connection.execute(
                "UPDATE memory_reconciliation_triage_recommendations SET status = 'failed', error_code = ?, "
                "completed_at = ? WHERE recommendation_id = ?",
                (error_code, _now(), recommendation_id),
            )
            await self._store.connection.commit()
            if isinstance(exc, MemoryReconciliationReviewError):
                raise
            raise MemoryReconciliationReviewError(
                "Memory-quality recommendation failed; groups remain unresolved"
            ) from exc

    async def apply(
        self,
        principal_id: PrincipalId,
        plan_id: str,
        *,
        expected_review_version: int,
        allowed_project_ids: frozenset[str],
        client_operation_id: str,
    ) -> dict[str, Any]:
        operation_id = self._operation_id(client_operation_id)
        request_sha256 = _digest(
            {
                "kind": "triage_apply",
                "plan_id": plan_id,
                "expected_review_version": expected_review_version,
            }
        )
        replay = await self._operation_replay(principal_id, operation_id, request_sha256)
        if replay is not None:
            return replay
        row, plan = await self._plan_row(principal_id, plan_id)
        if str(row[3]) == "applied":
            async with self._store.connection.execute(
                "SELECT receipt_json FROM memory_reconciliation_triage_plans WHERE plan_id = ?",
                (plan_id,),
            ) as cursor:
                receipt_row = await cursor.fetchone()
            if receipt_row is None or receipt_row[0] is None:
                raise MemoryReconciliationReviewError("Applied memory triage receipt is missing")
            stored_response = _load_json(receipt_row[0], label="memory triage receipt")
            stored_response["replayed"] = True
            return stored_response
        if int(row[4]) != expected_review_version:
            raise MemoryReconciliationReviewConflict("Memory triage plan changed after it was loaded")
        async with self._store.connection.execute(
            "SELECT group_id, disposition, action_json, operator_note FROM memory_reconciliation_triage_groups "
            "WHERE plan_id = ? ORDER BY group_id",
            (plan_id,),
        ) as cursor:
            decision_rows = await cursor.fetchall()
        if any(str(item[1]) == "pending" for item in decision_rows):
            raise MemoryReconciliationReviewValidationError("Resolve every memory triage group before applying")
        apply_groups, groups, decision_rows = await self._with_consolidations(plan_id, plan, list(decision_rows))
        async with self._store.connection.execute(
            "SELECT recommendation_id, status, backend, provider, model, prompt_version, input_sha256, "
            "output_sha256, group_count, error_code FROM memory_reconciliation_triage_recommendations "
            "WHERE plan_id = ? ORDER BY recommendation_id",
            (plan_id,),
        ) as cursor:
            recommendation_artifacts = [
                {
                    "recommendation_id": str(item[0]),
                    "status": str(item[1]),
                    "backend": str(item[2]),
                    "provider": str(item[3]),
                    "model": str(item[4]),
                    "prompt_version": str(item[5]),
                    "input_sha256": str(item[6]),
                    "output_sha256": None if item[7] is None else str(item[7]),
                    "group_count": int(item[8]),
                    "error_code": None if item[9] is None else str(item[9]),
                }
                for item in await cursor.fetchall()
            ]
        recommendation_artifacts_sha256 = _digest(recommendation_artifacts)
        candidates = [
            {
                "candidate_id": group["group_id"],
                "state_sha256": group["state_sha256"],
                "category": group["classification"],
                "uncertainty": "low" if group["deterministic"] else "high",
                "rationale": group["rationale"],
                "proposed_action": group["action"],
                "evidence": group["evidence"],
            }
            for group in apply_groups
        ]
        synthetic: dict[str, Any] = {
            "kind": memory_reconciliation.AUDIT_KIND,
            "version": memory_reconciliation.FORMAT_VERSION,
            "audit_id": plan_id,
            "principal_id": plan["principal_id"],
            "runtime_profile_id": plan["runtime_profile_id"],
            "generated_at": plan["generated_at"],
            "read_only": True,
            "corpus_sha256": plan["corpus_sha256"],
            "corpus_count": plan["memory_count"],
            "candidate_count": len(candidates),
            "suppressed_unchanged_count": 0,
            "triage_plan_sha256": plan["sha256"],
            "recommendation_artifacts_sha256": recommendation_artifacts_sha256,
            "candidates": candidates,
        }
        synthetic["sha256"] = _digest(synthetic)
        template = memory_reconciliation.build_review_template(synthetic)
        stored = {str(item[0]): item for item in decision_rows}
        for decision in template["decisions"]:
            state = stored[str(decision["candidate_id"])]
            action = _load_json(state[2], label="memory triage action")
            group = groups[str(decision["candidate_id"])]
            if state[1] == "approve":
                if not _operator_action_is_authorized(group, action, audit_boundary=plan["generated_at"]):
                    raise MemoryReconciliationReviewValidationError(
                        "Stored memory triage approval no longer matches an authorized action"
                    )
                try:
                    memory_reconciliation.validate_candidate_action({"evidence": group["evidence"]}, action)
                except memory_reconciliation.MemoryReconciliationError as exc:
                    raise MemoryReconciliationReviewValidationError(str(exc)) from exc
                project_ids = {
                    str(item["project_id"])
                    for item in group["evidence"]
                    if item.get("scope") == "project" and item.get("project_id")
                }
                replacement = action.get("replacement")
                if (
                    action.get("kind") == "adopt_corrected"
                    and isinstance(replacement, dict)
                    and replacement.get("scope_kind") == "project"
                    and replacement.get("scope_key")
                ):
                    project_ids.add(str(replacement["scope_key"]))
                if project_ids - allowed_project_ids:
                    raise MemoryReconciliationReviewAccessDenied(
                        "Memory triage action references a project outside current authority"
                    )
            decision.update(
                {
                    "disposition": str(state[1]),
                    "action": action,
                    "operator_note": str(state[3]),
                }
            )
        sealed = memory_reconciliation.seal_review(synthetic, template, reviewer=str(principal_id))
        # Apply checks only the rows it is about to change and resumes an
        # interrupted run, so unrelated writes since the audit (new
        # extracted facts, for example) no longer block it.
        try:
            receipt = await memory_reconciliation.apply_review(db_path=self._db_path, audit=synthetic, review=sealed)
        except (
            memory_reconciliation.MemoryReconciliationDrift,
            memory_reconciliation.MemoryReconciliationApplyInProgress,
        ) as exc:
            raise MemoryReconciliationReviewConflict(str(exc)) from exc
        approved_outcomes = _approved_outcome_summary(groups, decision_rows)
        summary = {
            "adopted": approved_outcomes["adopted"],
            "consolidated": approved_outcomes["consolidated"],
            "obsolete": approved_outcomes["obsolete"],
            "deferred": sum(len(groups[str(item[0])]["evidence"]) for item in decision_rows if str(item[1]) == "defer"),
            "rejected": sum(
                len(groups[str(item[0])]["evidence"]) for item in decision_rows if str(item[1]) == "reject"
            ),
            # Canonical state committed, but these items are not in recall
            # until their search projection is retried from Fact review.
            "failed": len(receipt.get("projection_failures", ())),
            # The owner's legacy rows still unclassified after this apply,
            # from the census apply refreshes (0 if that count failed).
            "still_unresolved": int((receipt.get("legacy_census") or {}).get("unclassified", 0)),
            "not_adopted": sum(
                len(groups[str(item[0])]["evidence"]) for item in decision_rows if str(item[1]) in {"reject", "defer"}
            ),
            "operator_admitted": approved_outcomes["operator_admitted"],
            "canonicalized_in_quarantine": int(receipt.get("canonicalized_in_quarantine", 0)),
        }
        response = {
            "audit_id": plan["audit_id"],
            "plan_id": plan_id,
            "receipt": receipt,
            "recommendation_artifacts_sha256": recommendation_artifacts_sha256,
            "summary": summary,
            "replayed": False,
        }
        stored_response = {key: value for key, value in response.items() if key != "replayed"}
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            await connection.execute(
                "UPDATE memory_reconciliation_triage_plans SET status = 'applied', applied_at = ?, receipt_json = ? "
                "WHERE plan_id = ?",
                (receipt["applied_at"], _canonical(stored_response), plan_id),
            )
            await connection.execute(
                "UPDATE memory_reconciliation_audits SET status = 'applied', applied_at = ? WHERE audit_id = ?",
                (receipt["applied_at"], plan["audit_id"]),
            )
            await connection.execute(
                "INSERT INTO memory_reconciliation_operations ("
                "principal_id, client_operation_id, request_sha256, response_json, created_at"
                ") VALUES (?, ?, ?, ?, ?)",
                (str(principal_id), operation_id, request_sha256, _canonical(response), _now()),
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise
        return response


__all__ = ["TriageGroupPage", "TriageSummary", "WorkshopMemoryReconciliationTriageService"]
