"""Canonical, principal-scoped review state for legacy memory reconciliation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from kai import memory, memory_reconciliation
from kai.workshop.domain import PrincipalId
from kai.workshop.store import WorkshopEventStore

MAX_PAGE_SIZE = 50
MAX_BULK_TARGETS = 100
MAX_OPERATOR_NOTE = 4096
MAX_OPERATION_ID = 128


class MemoryReconciliationReviewError(RuntimeError):
    """Base error at the Workshop reconciliation-review boundary."""


class MemoryReconciliationReviewAccessDenied(MemoryReconciliationReviewError):
    """The principal does not own the requested reconciliation state."""


class MemoryReconciliationReviewNotFound(MemoryReconciliationReviewError):
    """The requested audit or candidate does not exist."""


class MemoryReconciliationReviewConflict(MemoryReconciliationReviewError):
    """The requested mutation is stale or reuses an operation identity."""


class MemoryReconciliationReviewValidationError(MemoryReconciliationReviewError):
    """The requested review operation is malformed or unsafe."""


@dataclass(frozen=True, slots=True)
class ReconciliationAuditSummary:
    audit_id: str
    runtime_profile_id: str
    generated_at: str
    corpus_count: int
    candidate_count: int
    status: str
    review_version: int
    disposition_counts: dict[str, int]
    category_counts: dict[str, int]
    kind_counts: dict[str, int]
    uncertainty_counts: dict[str, int]
    action_counts: dict[str, int]
    gap_counts: dict[str, int]
    applied_at: str | None


@dataclass(frozen=True, slots=True)
class ReconciliationCandidatePage:
    audit: ReconciliationAuditSummary
    candidates: tuple[dict[str, Any], ...]
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


def record_reconciliation_audit(db_path: Path, audit: dict[str, Any]) -> bool:
    """Persist one immutable audit and its editable decision projection.

    The semantic-memory corpus remains untouched. Repeating an audit over the
    same corpus returns the already-recorded review queue.
    """
    memory_reconciliation.validate_audit(audit)
    connection = sqlite3.connect(str(db_path))
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("BEGIN IMMEDIATE")
        existing = connection.execute(
            "SELECT principal_id, runtime_profile_id, corpus_sha256 FROM memory_reconciliation_audits "
            "WHERE audit_id = ?",
            (audit["audit_id"],),
        ).fetchone()
        if existing is not None:
            if tuple(str(value) for value in existing) != (
                str(audit["principal_id"]),
                str(audit["runtime_profile_id"]),
                str(audit["corpus_sha256"]),
            ):
                raise MemoryReconciliationReviewConflict("Audit identity conflicts with canonical review state")
            connection.rollback()
            return False
        now = _now()
        connection.execute(
            "INSERT INTO memory_reconciliation_audits ("
            "audit_id, principal_id, runtime_profile_id, audit_sha256, corpus_sha256, generated_at, "
            "candidate_count, audit_json, status, review_version, created_at"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', 0, ?)",
            (
                audit["audit_id"],
                audit["principal_id"],
                audit["runtime_profile_id"],
                audit["sha256"],
                audit["corpus_sha256"],
                audit["generated_at"],
                audit["candidate_count"],
                _canonical(audit),
                now,
            ),
        )
        connection.executemany(
            "INSERT INTO memory_reconciliation_decisions ("
            "audit_id, candidate_id, state_sha256, disposition, action_json, operator_note, "
            "state_version, updated_at) VALUES (?, ?, ?, 'pending', ?, '', 0, ?)",
            [
                (
                    audit["audit_id"],
                    candidate["candidate_id"],
                    candidate["state_sha256"],
                    _canonical(candidate["proposed_action"]),
                    now,
                )
                for candidate in audit["candidates"]
            ],
        )
        connection.commit()
        return True
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def prior_reconciliation_dispositions(
    db_path: Path,
    *,
    principal_id: str,
    runtime_profile_id: str,
) -> set[tuple[str, str]]:
    """Return receipt-backed reject/defer decisions for deterministic reruns."""
    connection = sqlite3.connect(str(db_path))
    try:
        rows = connection.execute(
            "SELECT r.receipt_json FROM memory_reconciliation_receipts r "
            "JOIN memory_reconciliation_audits a ON a.audit_id = r.audit_id "
            "WHERE a.principal_id = ? AND a.runtime_profile_id = ?",
            (principal_id, runtime_profile_id),
        ).fetchall()
    finally:
        connection.close()
    terminal: set[tuple[str, str]] = set()
    for row in rows:
        receipt = _load_json(row[0], label="reconciliation receipt")
        for decision in receipt.get("decisions", []):
            if not isinstance(decision, dict) or decision.get("disposition") not in {"reject", "defer"}:
                continue
            candidate_id = decision.get("candidate_id")
            state_sha256 = decision.get("state_sha256")
            if isinstance(candidate_id, str) and isinstance(state_sha256, str):
                terminal.add((candidate_id, state_sha256))
    return terminal


class WorkshopMemoryReconciliationReviewService:
    """Review immutable audits through principal-scoped, replay-safe state."""

    def __init__(self, store: WorkshopEventStore, *, db_path: Path) -> None:
        self._store = store
        self._db_path = db_path
        self._lock = asyncio.Lock()

    async def _owned_audit(self, principal_id: PrincipalId, audit_id: str) -> tuple[Any, dict[str, Any]]:
        if not audit_id or len(audit_id) > 128:
            raise MemoryReconciliationReviewValidationError("Invalid reconciliation audit")
        async with self._store.connection.execute(
            "SELECT audit_id, principal_id, runtime_profile_id, generated_at, candidate_count, "
            "audit_json, status, review_version, applied_at FROM memory_reconciliation_audits "
            "WHERE audit_id = ?",
            (audit_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise MemoryReconciliationReviewNotFound("Reconciliation audit not found")
        if str(row[1]) != str(principal_id):
            raise MemoryReconciliationReviewAccessDenied("Reconciliation review access denied")
        audit = _load_json(row[5], label="reconciliation audit")
        memory_reconciliation.validate_audit(audit)
        return row, audit

    async def _summary_from(self, row: Any, audit: dict[str, Any]) -> ReconciliationAuditSummary:
        async with self._store.connection.execute(
            "SELECT disposition, COUNT(*) FROM memory_reconciliation_decisions WHERE audit_id = ? GROUP BY disposition",
            (row[0],),
        ) as cursor:
            dispositions = {str(item[0]): int(item[1]) for item in await cursor.fetchall()}
        categories: dict[str, int] = {}
        kinds: dict[str, int] = {}
        uncertainties: dict[str, int] = {}
        actions: dict[str, int] = {}
        gaps: dict[str, int] = {}
        for candidate in audit["candidates"]:
            category = str(candidate["category"])
            categories[category] = categories.get(category, 0) + 1
            uncertainty = str(candidate["uncertainty"])
            uncertainties[uncertainty] = uncertainties.get(uncertainty, 0) + 1
            action = str(candidate["proposed_action"].get("kind") or "unknown")
            actions[action] = actions.get(action, 0) + 1
            for evidence in candidate["evidence"]:
                kind = str(evidence["kind"])
                kinds[kind] = kinds.get(kind, 0) + 1
                for gap in set(str(value) for value in evidence["migration_gaps"]):
                    gaps[gap] = gaps.get(gap, 0) + 1
        return ReconciliationAuditSummary(
            audit_id=str(row[0]),
            runtime_profile_id=str(row[2]),
            generated_at=str(row[3]),
            corpus_count=int(audit["corpus_count"]),
            candidate_count=int(row[4]),
            status=str(row[6]),
            review_version=int(row[7]),
            disposition_counts={key: dispositions.get(key, 0) for key in ("pending", "approve", "reject", "defer")},
            category_counts=dict(sorted(categories.items())),
            kind_counts=dict(sorted(kinds.items())),
            uncertainty_counts=dict(sorted(uncertainties.items())),
            action_counts=dict(sorted(actions.items())),
            gap_counts=dict(sorted(gaps.items())),
            applied_at=None if row[8] is None else str(row[8]),
        )

    async def latest(self, principal_id: PrincipalId) -> ReconciliationAuditSummary | None:
        async with self._store.connection.execute(
            "SELECT audit_id FROM memory_reconciliation_audits WHERE principal_id = ? "
            "ORDER BY generated_at DESC, created_at DESC LIMIT 1",
            (str(principal_id),),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        audit_row, audit = await self._owned_audit(principal_id, str(row[0]))
        return await self._summary_from(audit_row, audit)

    async def candidates(
        self,
        principal_id: PrincipalId,
        audit_id: str,
        *,
        category: str | None = None,
        kind: str | None = None,
        scope: str | None = None,
        uncertainty: str | None = None,
        disposition: str | None = None,
        action: str | None = None,
        gap: str | None = None,
        offset: int = 0,
        limit: int = 25,
    ) -> ReconciliationCandidatePage:
        if offset < 0 or not 1 <= limit <= MAX_PAGE_SIZE:
            raise MemoryReconciliationReviewValidationError("Invalid reconciliation page")
        for value in (category, kind, scope, uncertainty, disposition, action, gap):
            if value is not None and (not value or len(value) > 128):
                raise MemoryReconciliationReviewValidationError("Invalid reconciliation filter")
        row, audit = await self._owned_audit(principal_id, audit_id)
        async with self._store.connection.execute(
            "SELECT candidate_id, disposition, action_json, operator_note, state_version "
            "FROM memory_reconciliation_decisions WHERE audit_id = ?",
            (audit_id,),
        ) as cursor:
            decisions = {
                str(item[0]): {
                    "disposition": str(item[1]),
                    "action": _load_json(item[2], label="reconciliation action"),
                    "operator_note": str(item[3]),
                    "state_version": int(item[4]),
                }
                for item in await cursor.fetchall()
            }
        filtered: list[dict[str, Any]] = []
        for candidate in audit["candidates"]:
            decision = decisions.get(str(candidate["candidate_id"]))
            if decision is None:
                raise MemoryReconciliationReviewError("Reconciliation decision projection is incomplete")
            evidence = candidate["evidence"]
            if category is not None and candidate["category"] != category:
                continue
            if uncertainty is not None and candidate["uncertainty"] != uncertainty:
                continue
            if disposition is not None and decision["disposition"] != disposition:
                continue
            if action is not None and candidate["proposed_action"].get("kind") != action:
                continue
            if kind is not None and not any(item["kind"] == kind for item in evidence):
                continue
            if scope is not None and not any(item["scope"] == scope for item in evidence):
                continue
            if gap is not None and not any(gap in item["migration_gaps"] for item in evidence):
                continue
            filtered.append(self._client_candidate(candidate, decision))
        selected = filtered[offset : offset + limit]
        next_offset = offset + limit if offset + limit < len(filtered) else None
        return ReconciliationCandidatePage(
            audit=await self._summary_from(row, audit),
            candidates=tuple(selected),
            next_offset=next_offset,
        )

    @staticmethod
    def _client_candidate(candidate: dict[str, Any], decision: dict[str, Any]) -> dict[str, Any]:
        evidence = []
        for item in candidate["evidence"]:
            evidence.append(
                {
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
            )
        return {
            "candidate_id": candidate["candidate_id"],
            "state_sha256": candidate["state_sha256"],
            "category": candidate["category"],
            "uncertainty": candidate["uncertainty"],
            "rationale": candidate["rationale"],
            "proposed_action": candidate["proposed_action"],
            "evidence": evidence,
            "decision": decision,
        }

    @staticmethod
    def _required_project_ids(candidate: dict[str, Any], action: dict[str, Any]) -> frozenset[str]:
        if action.get("kind") == "adopt_corrected":
            replacement = action.get("replacement")
            if isinstance(replacement, dict) and replacement.get("scope_kind") == "project":
                scope_key = replacement.get("scope_key")
                return frozenset({scope_key}) if isinstance(scope_key, str) and scope_key else frozenset()
            return frozenset()
        return frozenset(
            str(item["project_id"])
            for item in candidate["evidence"]
            if item.get("scope") == "project" and item.get("project_id")
        )

    @classmethod
    def _validate_project_access(
        cls,
        candidate: dict[str, Any],
        action: dict[str, Any],
        allowed_project_ids: frozenset[str],
    ) -> None:
        unavailable = cls._required_project_ids(candidate, action) - allowed_project_ids
        if unavailable:
            raise MemoryReconciliationReviewAccessDenied(
                "Reconciliation action references a project outside the principal's current memory authority"
            )

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
            raise MemoryReconciliationReviewConflict("Reconciliation operation identity was reused")
        response = _load_json(row[1], label="reconciliation operation response")
        response["replayed"] = True
        return response

    @staticmethod
    def _validate_operation_id(operation_id: str) -> str:
        if (
            not isinstance(operation_id, str)
            or not operation_id
            or len(operation_id) > MAX_OPERATION_ID
            or any(character.isspace() for character in operation_id)
        ):
            raise MemoryReconciliationReviewValidationError("Invalid reconciliation operation identifier")
        return operation_id

    async def decide(
        self,
        principal_id: PrincipalId,
        audit_id: str,
        candidate_id: str,
        *,
        disposition: str,
        action: dict[str, Any],
        operator_note: str,
        expected_state_version: int,
        client_operation_id: str,
        allowed_project_ids: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        operation_id = self._validate_operation_id(client_operation_id)
        request = {
            "kind": "decision",
            "audit_id": audit_id,
            "candidate_id": candidate_id,
            "disposition": disposition,
            "action": action,
            "operator_note": operator_note,
            "expected_state_version": expected_state_version,
        }
        request_sha256 = _digest(request)
        async with self._lock:
            replay = await self._operation_replay(principal_id, operation_id, request_sha256)
            if replay is not None:
                return replay
            row, audit = await self._owned_audit(principal_id, audit_id)
            if str(row[6]) != "open":
                raise MemoryReconciliationReviewConflict("Reconciliation audit has already been applied")
            if disposition not in {"approve", "reject", "defer"}:
                raise MemoryReconciliationReviewValidationError("Invalid reconciliation disposition")
            if not isinstance(action, dict):
                raise MemoryReconciliationReviewValidationError("Invalid reconciliation action")
            if not isinstance(operator_note, str) or len(operator_note) > MAX_OPERATOR_NOTE:
                raise MemoryReconciliationReviewValidationError("Invalid reconciliation operator note")
            candidates = {str(item["candidate_id"]): item for item in audit["candidates"]}
            candidate = candidates.get(candidate_id)
            if candidate is None:
                raise MemoryReconciliationReviewNotFound("Reconciliation candidate not found")
            if disposition == "approve":
                try:
                    memory_reconciliation.validate_candidate_action(candidate, action)
                except memory_reconciliation.MemoryReconciliationError as exc:
                    raise MemoryReconciliationReviewValidationError(str(exc)) from exc
                self._validate_project_access(candidate, action, allowed_project_ids)
            else:
                action = dict(candidate["proposed_action"])
            connection = self._store.connection
            try:
                await connection.execute("BEGIN IMMEDIATE")
                cursor = await connection.execute(
                    "UPDATE memory_reconciliation_decisions SET disposition = ?, action_json = ?, "
                    "operator_note = ?, state_version = state_version + 1, updated_at = ? "
                    "WHERE audit_id = ? AND candidate_id = ? AND state_version = ?",
                    (
                        disposition,
                        _canonical(action),
                        operator_note,
                        _now(),
                        audit_id,
                        candidate_id,
                        expected_state_version,
                    ),
                )
                if cursor.rowcount != 1:
                    raise MemoryReconciliationReviewConflict("Reconciliation decision changed after it was loaded")
                await connection.execute(
                    "UPDATE memory_reconciliation_audits SET review_version = review_version + 1 WHERE audit_id = ?",
                    (audit_id,),
                )
                async with connection.execute(
                    "SELECT state_version FROM memory_reconciliation_decisions WHERE audit_id = ? AND candidate_id = ?",
                    (audit_id, candidate_id),
                ) as version_cursor:
                    version_row = await version_cursor.fetchone()
                async with connection.execute(
                    "SELECT review_version FROM memory_reconciliation_audits WHERE audit_id = ?",
                    (audit_id,),
                ) as review_cursor:
                    review_row = await review_cursor.fetchone()
                response = {
                    "audit_id": audit_id,
                    "candidate_id": candidate_id,
                    "disposition": disposition,
                    "state_version": int(version_row[0]),
                    "review_version": int(review_row[0]),
                    "replayed": False,
                }
                await connection.execute(
                    "INSERT INTO memory_reconciliation_operations ("
                    "principal_id, client_operation_id, request_sha256, response_json, created_at"
                    ") VALUES (?, ?, ?, ?, ?)",
                    (str(principal_id), operation_id, request_sha256, _canonical(response), _now()),
                )
                await connection.commit()
                return response
            except Exception:
                await connection.rollback()
                raise

    async def bulk_decide(
        self,
        principal_id: PrincipalId,
        audit_id: str,
        *,
        candidate_ids: list[str],
        disposition: Literal["reject", "defer"],
        operator_note: str,
        expected_review_version: int,
        client_operation_id: str,
    ) -> dict[str, Any]:
        operation_id = self._validate_operation_id(client_operation_id)
        if disposition not in {"reject", "defer"}:
            raise MemoryReconciliationReviewValidationError("Bulk approval is not permitted")
        if not 1 <= len(candidate_ids) <= MAX_BULK_TARGETS or len(set(candidate_ids)) != len(candidate_ids):
            raise MemoryReconciliationReviewValidationError("Invalid reconciliation candidate selection")
        if any(not isinstance(item, str) or not item or len(item) > 128 for item in candidate_ids):
            raise MemoryReconciliationReviewValidationError("Invalid reconciliation candidate selection")
        if not isinstance(operator_note, str) or len(operator_note) > MAX_OPERATOR_NOTE:
            raise MemoryReconciliationReviewValidationError("Invalid reconciliation operator note")
        request = {
            "kind": "bulk_decision",
            "audit_id": audit_id,
            "candidate_ids": sorted(candidate_ids),
            "disposition": disposition,
            "operator_note": operator_note,
            "expected_review_version": expected_review_version,
        }
        request_sha256 = _digest(request)
        async with self._lock:
            replay = await self._operation_replay(principal_id, operation_id, request_sha256)
            if replay is not None:
                return replay
            row, _audit = await self._owned_audit(principal_id, audit_id)
            if str(row[6]) != "open" or int(row[7]) != expected_review_version:
                raise MemoryReconciliationReviewConflict("Reconciliation review changed after it was loaded")
            connection = self._store.connection
            try:
                await connection.execute("BEGIN IMMEDIATE")
                placeholders = ",".join("?" for _ in candidate_ids)
                async with connection.execute(
                    f"SELECT COUNT(*) FROM memory_reconciliation_decisions "
                    f"WHERE audit_id = ? AND candidate_id IN ({placeholders})",
                    (audit_id, *candidate_ids),
                ) as cursor:
                    count_row = await cursor.fetchone()
                if int(count_row[0]) != len(candidate_ids):
                    raise MemoryReconciliationReviewNotFound("A reconciliation candidate was not found")
                await connection.execute(
                    f"UPDATE memory_reconciliation_decisions SET disposition = ?, "
                    f"state_version = state_version + 1, operator_note = ?, updated_at = ? "
                    f"WHERE audit_id = ? AND candidate_id IN ({placeholders})",
                    (disposition, operator_note, _now(), audit_id, *candidate_ids),
                )
                await connection.execute(
                    "UPDATE memory_reconciliation_audits SET review_version = review_version + 1 WHERE audit_id = ?",
                    (audit_id,),
                )
                response = {
                    "audit_id": audit_id,
                    "changed": len(candidate_ids),
                    "disposition": disposition,
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
                return response
            except Exception:
                await connection.rollback()
                raise

    async def apply(
        self,
        principal_id: PrincipalId,
        audit_id: str,
        *,
        expected_review_version: int,
        client_operation_id: str,
        allowed_project_ids: frozenset[str] = frozenset(),
    ) -> dict[str, Any]:
        operation_id = self._validate_operation_id(client_operation_id)
        request = {
            "kind": "apply",
            "audit_id": audit_id,
            "expected_review_version": expected_review_version,
        }
        request_sha256 = _digest(request)
        async with self._lock:
            replay = await self._operation_replay(principal_id, operation_id, request_sha256)
            if replay is not None:
                return replay
            row, audit = await self._owned_audit(principal_id, audit_id)
            if str(row[6]) == "applied":
                async with self._store.connection.execute(
                    "SELECT receipt_json FROM memory_reconciliation_receipts WHERE audit_id = ?",
                    (audit_id,),
                ) as cursor:
                    receipt_row = await cursor.fetchone()
                if receipt_row is None:
                    raise MemoryReconciliationReviewError("Applied reconciliation receipt is missing")
                receipt = _load_json(receipt_row[0], label="reconciliation receipt")
                return {"audit_id": audit_id, "receipt": receipt, "replayed": True}
            if int(row[7]) != expected_review_version:
                raise MemoryReconciliationReviewConflict("Reconciliation review changed after it was loaded")
            async with self._store.connection.execute(
                "SELECT candidate_id, state_sha256, disposition, action_json, operator_note "
                "FROM memory_reconciliation_decisions WHERE audit_id = ? ORDER BY candidate_id",
                (audit_id,),
            ) as cursor:
                decision_rows = await cursor.fetchall()
            template = memory_reconciliation.build_review_template(audit)
            by_candidate = {str(item[0]): item for item in decision_rows}
            for decision in template["decisions"]:
                stored = by_candidate.get(str(decision["candidate_id"]))
                if stored is None:
                    raise MemoryReconciliationReviewError("Reconciliation decision projection is incomplete")
                decision.update(
                    {
                        "state_sha256": str(stored[1]),
                        "disposition": str(stored[2]),
                        "action": _load_json(stored[3], label="reconciliation action"),
                        "operator_note": str(stored[4]),
                    }
                )
                if decision["disposition"] == "approve":
                    candidate = next(
                        item for item in audit["candidates"] if item["candidate_id"] == decision["candidate_id"]
                    )
                    self._validate_project_access(candidate, decision["action"], allowed_project_ids)
            try:
                sealed = memory_reconciliation.seal_review(audit, template, reviewer=str(principal_id))
            except memory_reconciliation.MemoryReconciliationError as exc:
                raise MemoryReconciliationReviewValidationError(str(exc)) from exc
            try:
                current = await asyncio.to_thread(
                    memory.get_all_for_lifecycle_projection,
                    user_id=str(principal_id),
                    runtime_profile_id=str(audit["runtime_profile_id"]),
                )
                if memory_reconciliation.corpus_sha256(current) != audit["corpus_sha256"]:
                    raise MemoryReconciliationReviewConflict("Memory changed after this reconciliation audit")
                receipt = await memory_reconciliation.apply_review(
                    db_path=self._db_path,
                    audit=audit,
                    review=sealed,
                )
            except MemoryReconciliationReviewError:
                raise
            except memory_reconciliation.MemoryReconciliationError as exc:
                raise MemoryReconciliationReviewValidationError(str(exc)) from exc
            except Exception as exc:
                raise MemoryReconciliationReviewError("Canonical memory reconciliation failed") from exc
            response = {"audit_id": audit_id, "receipt": receipt, "replayed": False}
            connection = self._store.connection
            try:
                await connection.execute("BEGIN IMMEDIATE")
                await connection.execute(
                    "INSERT INTO memory_reconciliation_receipts ("
                    "receipt_id, audit_id, principal_id, review_sha256, receipt_sha256, receipt_json, applied_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        receipt["receipt_id"],
                        audit_id,
                        str(principal_id),
                        sealed["sha256"],
                        receipt["sha256"],
                        _canonical(receipt),
                        receipt["applied_at"],
                    ),
                )
                await connection.execute(
                    "UPDATE memory_reconciliation_audits SET status = 'applied', applied_at = ? WHERE audit_id = ?",
                    (receipt["applied_at"], audit_id),
                )
                await connection.execute(
                    "INSERT INTO memory_reconciliation_operations ("
                    "principal_id, client_operation_id, request_sha256, response_json, created_at"
                    ") VALUES (?, ?, ?, ?, ?)",
                    (str(principal_id), operation_id, request_sha256, _canonical(response), _now()),
                )
                await connection.commit()
                return response
            except Exception:
                await connection.rollback()
                raise
