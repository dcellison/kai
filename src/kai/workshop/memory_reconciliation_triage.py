"""Canonical grouped triage and exception-only review for legacy memory."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from kai import memory, memory_reconciliation
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
)
from kai.workshop.runtime_pool import WorkshopRuntimePool
from kai.workshop.store import WorkshopEventStore

PROMPT_VERSION = "memory_reconciliation_triage_v1"
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
                },
                "required": ["group_id", "outcome", "confidence", "rationale"],
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
        plan = build_triage_plan(
            audit,
            allowed_project_ids=allowed_project_ids,
            prior_review_evidence=prior_review_evidence,
        )
        async with self._lock:
            async with self._store.connection.execute(
                "SELECT plan_id, plan_sha256 FROM memory_reconciliation_triage_plans WHERE audit_id = ?",
                (audit_id,),
            ) as cursor:
                existing = await cursor.fetchone()
            if existing is None:
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
        visible = []
        for group in plan["groups"]:
            state = states.get(str(group["group_id"]))
            if state is None:
                continue
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
                    "proposed_action": group["action"],
                    "prior_review_evidence": group["prior_review_evidence"],
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
        group = next((item for item in plan["groups"] if item["group_id"] == group_id), None)
        if group is None:
            raise MemoryReconciliationReviewNotFound("Memory triage group not found")
        if disposition == "approve":
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
                    "evidence is ambiguous. Return every supplied group exactly once."
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
                if not isinstance(raw, dict) or set(raw) != {"group_id", "outcome", "confidence", "rationale"}:
                    raise MemoryReconciliationReviewError("Memory-quality recommendation is malformed")
                group_id = raw["group_id"]
                if group_id not in expected or group_id in recommendations:
                    raise MemoryReconciliationReviewError("Memory-quality recommendation references an invalid group")
                if raw["outcome"] not in {"adopt", "consolidate", "obsolete", "needs_review"}:
                    raise MemoryReconciliationReviewError("Memory-quality recommendation has an invalid outcome")
                if (
                    isinstance(raw["confidence"], bool)
                    or not isinstance(raw["confidence"], int | float)
                    or not 0 <= float(raw["confidence"]) <= 1
                    or not isinstance(raw["rationale"], str)
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
        groups = {str(item["group_id"]): item for item in plan["groups"]}
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
            for group in plan["groups"]
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
                project_ids = {
                    str(item["project_id"])
                    for item in group["evidence"]
                    if item.get("scope") == "project" and item.get("project_id")
                }
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
        current = await asyncio.to_thread(
            memory.get_all_for_lifecycle_projection,
            user_id=str(principal_id),
            runtime_profile_id=str(plan["runtime_profile_id"]),
        )
        if memory_reconciliation.corpus_sha256(current) != plan["corpus_sha256"]:
            raise MemoryReconciliationReviewConflict("Memory changed after this triage plan")
        receipt = await memory_reconciliation.apply_review(db_path=self._db_path, audit=synthetic, review=sealed)
        approved_group_ids = {str(item[0]) for item in decision_rows if str(item[1]) == "approve"}
        canonicalized_in_quarantine = sum(
            len(group["evidence"])
            for group_id, group in groups.items()
            if group_id in approved_group_ids and group["action"].get("migration_classification") == "legacy_incomplete"
        )
        summary = {
            "adopted": sum(
                len(group["evidence"])
                for group_id, group in groups.items()
                if group_id in approved_group_ids and group["resolution"] == "adopt"
            ),
            "consolidated": sum(
                len(group["evidence"])
                for group_id, group in groups.items()
                if group_id in approved_group_ids and group["resolution"] == "consolidate"
            ),
            "obsolete": sum(
                len(group["evidence"])
                for group_id, group in groups.items()
                if group_id in approved_group_ids and group["resolution"] == "obsolete"
            ),
            "deferred": sum(len(groups[str(item[0])]["evidence"]) for item in decision_rows if str(item[1]) == "defer"),
            "rejected": sum(
                len(groups[str(item[0])]["evidence"]) for item in decision_rows if str(item[1]) == "reject"
            ),
            "failed": 0,
            "still_unresolved": 0,
            "not_adopted": sum(
                len(groups[str(item[0])]["evidence"]) for item in decision_rows if str(item[1]) in {"reject", "defer"}
            ),
            "canonicalized_in_quarantine": canonicalized_in_quarantine,
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
