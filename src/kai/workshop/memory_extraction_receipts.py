"""Canonical, privacy-bounded receipts for semantic-memory extraction."""

from __future__ import annotations

import hashlib
import json
import secrets
from dataclasses import dataclass

import aiosqlite

from kai.workshop.domain import PrincipalId

_ROLES = frozenset({"fact_extraction", "episode_generation"})
_TERMINAL_STATUSES = frozenset({"completed", "failed"})
_MAX_CANDIDATES = 32
_MAX_DECISIONS = 32
_MAX_POLICY_FIELDS = 16
_PROCESS_CLAIM_OWNER = secrets.token_hex(16)


class MemoryExtractionReceiptError(RuntimeError):
    """Base error for canonical extraction-receipt operations."""


class MemoryExtractionReceiptAccessDenied(MemoryExtractionReceiptError):
    """The requester does not own the requested receipt."""


class MemoryExtractionReceiptConflict(MemoryExtractionReceiptError):
    """A canonical run/role identity was reused with different semantics."""


class MemoryExtractionReceiptValidationError(MemoryExtractionReceiptError):
    """A receipt contains invalid or unbounded non-content metadata."""


@dataclass(frozen=True, slots=True)
class MemoryExtractionReceiptAuthority:
    principal_id: PrincipalId


@dataclass(frozen=True, slots=True)
class MemoryExtractionReceiptSpec:
    principal_id: str
    runtime_profile_id: str
    run_id: str
    source_message_id: str
    result_message_id: str
    extraction_role: str
    backend: str
    provider: str
    model: str
    prompt_version: str
    schema_version: str
    policy_version: str


@dataclass(frozen=True, slots=True)
class MemoryExtractionStorageDecision:
    index: int
    intent: str
    outcome: str
    new_memory_id: str | None = None
    replaced_memory_id: str | None = None
    scope: str | None = None
    project_id: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryExtractionReceiptCompletion:
    status: str
    decision_outcome: str
    failure_code: str | None
    candidate_ids: tuple[str, ...]
    classifier_result: bool | None
    proposed_intents: tuple[tuple[str, str | None], ...]
    raw_count: int
    accepted_count: int
    validation_outcome: str
    storage_decisions: tuple[MemoryExtractionStorageDecision, ...]
    stored_count: int
    replaced_count: int
    skipped_count: int
    memory_scopes: tuple[tuple[str, str | None], ...]
    duration_ms: int
    policy_outcome: tuple[tuple[str, str | int], ...] = ()


@dataclass(frozen=True, slots=True)
class MemoryExtractionReceiptSnapshot:
    receipt_id: str
    principal_id: str
    runtime_profile_id: str
    run_id: str
    source_message_id: str
    result_message_id: str
    extraction_role: str
    backend: str
    provider: str
    model: str
    prompt_version: str
    schema_version: str
    policy_version: str
    status: str
    decision_outcome: str | None
    failure_code: str | None
    candidate_ids: tuple[str, ...]
    classifier_result: bool | None
    proposed_intents: tuple[dict[str, object], ...]
    validation_outcome: dict[str, object]
    storage_outcome: dict[str, object]
    memory_scopes: tuple[dict[str, object], ...]
    duration_ms: int | None
    created_at: str
    completed_at: str | None

    @property
    def stored_count(self) -> int:
        value = self.storage_outcome.get("stored_count", 0)
        return int(value) if isinstance(value, int) and not isinstance(value, bool) else 0


@dataclass(frozen=True, slots=True)
class MemoryExtractionReceiptClaim:
    receipt: MemoryExtractionReceiptSnapshot
    claimed: bool
    interrupted: bool = False


def _canonical_json(value: object) -> str:
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _receipt_id(run_id: str, extraction_role: str) -> str:
    digest = hashlib.sha256(f"{run_id}\0{extraction_role}".encode()).hexdigest()
    return f"mer_{digest[:32]}"


def _bounded(value: str, *, field: str, maximum: int) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise MemoryExtractionReceiptValidationError(f"Invalid {field}")
    return value


def _spec_payload(spec: MemoryExtractionReceiptSpec) -> dict[str, str]:
    if spec.extraction_role not in _ROLES:
        raise MemoryExtractionReceiptValidationError("Invalid extraction role")
    return {
        "principal_id": _bounded(spec.principal_id, field="principal ID", maximum=128),
        "runtime_profile_id": _bounded(spec.runtime_profile_id, field="runtime profile ID", maximum=128),
        "run_id": _bounded(spec.run_id, field="run ID", maximum=128),
        "source_message_id": _bounded(spec.source_message_id, field="source message ID", maximum=128),
        "result_message_id": _bounded(spec.result_message_id, field="result message ID", maximum=128),
        "extraction_role": spec.extraction_role,
        "backend": _bounded(spec.backend, field="backend", maximum=64),
        "provider": _bounded(spec.provider, field="provider", maximum=64),
        "model": _bounded(spec.model, field="model", maximum=256),
        "prompt_version": _bounded(spec.prompt_version, field="prompt version", maximum=64),
        "schema_version": _bounded(spec.schema_version, field="schema version", maximum=64),
        "policy_version": _bounded(spec.policy_version, field="policy version", maximum=64),
    }


def _completion_payload(completion: MemoryExtractionReceiptCompletion) -> dict[str, object]:
    if completion.status not in _TERMINAL_STATUSES:
        raise MemoryExtractionReceiptValidationError("Invalid terminal receipt status")
    _bounded(completion.decision_outcome, field="decision outcome", maximum=64)
    if completion.failure_code is not None:
        _bounded(completion.failure_code, field="failure code", maximum=64)
    if len(completion.candidate_ids) > _MAX_CANDIDATES:
        raise MemoryExtractionReceiptValidationError("Too many extraction candidates")
    if len(completion.proposed_intents) > _MAX_DECISIONS or len(completion.storage_decisions) > _MAX_DECISIONS:
        raise MemoryExtractionReceiptValidationError("Too many extraction decisions")
    if len(completion.policy_outcome) > _MAX_POLICY_FIELDS:
        raise MemoryExtractionReceiptValidationError("Too many extraction policy fields")
    for value in (
        completion.raw_count,
        completion.accepted_count,
        completion.stored_count,
        completion.replaced_count,
        completion.skipped_count,
        completion.duration_ms,
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MemoryExtractionReceiptValidationError("Invalid receipt count or duration")
    if completion.accepted_count > completion.raw_count:
        raise MemoryExtractionReceiptValidationError("Accepted count exceeds raw count")
    candidates = tuple(_bounded(value, field="candidate ID", maximum=256) for value in completion.candidate_ids)
    proposed = tuple(
        {
            "intent": _bounded(intent, field="proposed intent", maximum=64),
            **(
                {"existing_id": _bounded(existing_id, field="existing memory ID", maximum=256)}
                if existing_id is not None
                else {}
            ),
        }
        for intent, existing_id in completion.proposed_intents
    )
    decisions: list[dict[str, object]] = []
    for decision in completion.storage_decisions:
        if decision.index < 0:
            raise MemoryExtractionReceiptValidationError("Invalid storage decision index")
        encoded: dict[str, object] = {
            "index": decision.index,
            "intent": _bounded(decision.intent, field="storage intent", maximum=64),
            "outcome": _bounded(decision.outcome, field="storage outcome", maximum=64),
        }
        for key, value in (
            ("new_memory_id", decision.new_memory_id),
            ("replaced_memory_id", decision.replaced_memory_id),
            ("scope", decision.scope),
            ("project_id", decision.project_id),
        ):
            if value is not None:
                encoded[key] = _bounded(value, field=key, maximum=256)
        decisions.append(encoded)
    scopes = tuple(
        {
            "scope": _bounded(scope, field="memory scope", maximum=64),
            **({"project_id": _bounded(project_id, field="project ID", maximum=256)} if project_id is not None else {}),
        }
        for scope, project_id in completion.memory_scopes
    )
    policy: dict[str, str | int] = {}
    for key, value in completion.policy_outcome:
        bounded_key = _bounded(key, field="policy field", maximum=64)
        if bounded_key in policy:
            raise MemoryExtractionReceiptValidationError("Duplicate extraction policy field")
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise MemoryExtractionReceiptValidationError("Invalid extraction policy value")
        if isinstance(value, int):
            if value < 0:
                raise MemoryExtractionReceiptValidationError("Invalid extraction policy count")
            policy[bounded_key] = value
        else:
            policy[bounded_key] = _bounded(value, field="policy value", maximum=64)
    validation: dict[str, object] = {
        "outcome": _bounded(completion.validation_outcome, field="validation outcome", maximum=64),
        "raw_count": completion.raw_count,
        "accepted_count": completion.accepted_count,
        "rejected_count": completion.raw_count - completion.accepted_count,
    }
    if policy:
        validation["policy"] = policy
    return {
        "status": completion.status,
        "decision_outcome": completion.decision_outcome,
        "failure_code": completion.failure_code,
        "candidate_ids": candidates,
        "classifier_result": completion.classifier_result,
        "proposed_intents": proposed,
        "validation_outcome": validation,
        "storage_outcome": {
            "stored_count": completion.stored_count,
            "replaced_count": completion.replaced_count,
            "skipped_count": completion.skipped_count,
            "decisions": decisions,
        },
        "memory_scopes": scopes,
        "duration_ms": completion.duration_ms,
    }


class MemoryExtractionReceiptService:
    """Persist and inspect replay-safe receipts on one Workshop database."""

    def __init__(self, connection: aiosqlite.Connection, *, _claim_owner: str = _PROCESS_CLAIM_OWNER) -> None:
        self._connection = connection
        self._claim_owner = _bounded(_claim_owner, field="claim owner", maximum=32)
        if len(self._claim_owner) != 32:
            raise MemoryExtractionReceiptValidationError("Invalid claim owner")

    @staticmethod
    def authority_for_principal(principal_id: str | PrincipalId) -> MemoryExtractionReceiptAuthority:
        try:
            canonical = principal_id if isinstance(principal_id, PrincipalId) else PrincipalId(principal_id)
        except (TypeError, ValueError) as exc:
            raise MemoryExtractionReceiptAccessDenied("Memory extraction receipt access denied") from exc
        return MemoryExtractionReceiptAuthority(canonical)

    async def claim(self, spec: MemoryExtractionReceiptSpec) -> MemoryExtractionReceiptClaim:
        payload = _spec_payload(spec)
        await self._validate_run_binding(spec)
        receipt_id = _receipt_id(spec.run_id, spec.extraction_role)
        request_fingerprint = _fingerprint(payload)
        cursor = await self._connection.execute(
            "INSERT OR IGNORE INTO memory_extraction_receipts ("
            "receipt_id, principal_id, runtime_profile_id, run_id, source_message_id, "
            "result_message_id, extraction_role, backend, provider, model, prompt_version, "
            "schema_version, policy_version, status, request_fingerprint, claim_owner"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'running', ?, ?)",
            (
                receipt_id,
                spec.principal_id,
                spec.runtime_profile_id,
                spec.run_id,
                spec.source_message_id,
                spec.result_message_id,
                spec.extraction_role,
                spec.backend,
                spec.provider,
                spec.model,
                spec.prompt_version,
                spec.schema_version,
                spec.policy_version,
                request_fingerprint,
                self._claim_owner,
            ),
        )
        inserted = cursor.rowcount == 1
        await self._connection.commit()
        snapshot = await self._by_id(receipt_id)
        if snapshot is None:
            raise MemoryExtractionReceiptError("Claimed extraction receipt is unavailable")
        if (
            snapshot.principal_id != spec.principal_id
            or await self._request_fingerprint(receipt_id) != request_fingerprint
        ):
            raise MemoryExtractionReceiptConflict("Extraction receipt identity was reused with different semantics")
        if inserted:
            return MemoryExtractionReceiptClaim(snapshot, True)
        if snapshot.status == "running" and await self._receipt_claim_owner(receipt_id) != self._claim_owner:
            interrupted = MemoryExtractionReceiptCompletion(
                status="failed",
                decision_outcome="interrupted",
                failure_code="interrupted",
                candidate_ids=(),
                classifier_result=None,
                proposed_intents=(),
                raw_count=0,
                accepted_count=0,
                validation_outcome="not_completed",
                storage_decisions=(),
                stored_count=0,
                replaced_count=0,
                skipped_count=0,
                memory_scopes=(),
                duration_ms=0,
            )
            snapshot = await self.complete(receipt_id, spec.principal_id, interrupted)
            return MemoryExtractionReceiptClaim(snapshot, False, interrupted=True)
        return MemoryExtractionReceiptClaim(snapshot, False)

    async def complete(
        self,
        receipt_id: str,
        principal_id: str,
        completion: MemoryExtractionReceiptCompletion,
    ) -> MemoryExtractionReceiptSnapshot:
        payload = _completion_payload(completion)
        completion_fingerprint = _fingerprint(payload)
        cursor = await self._connection.execute(
            "UPDATE memory_extraction_receipts SET status = ?, decision_outcome = ?, failure_code = ?, "
            "candidate_ids_json = ?, classifier_result = ?, proposed_intents_json = ?, "
            "validation_outcome_json = ?, storage_outcome_json = ?, memory_scope_json = ?, "
            "duration_ms = ?, completion_fingerprint = ?, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), "
            "completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
            "WHERE receipt_id = ? AND principal_id = ? AND status = 'running'",
            (
                completion.status,
                completion.decision_outcome,
                completion.failure_code,
                _canonical_json(payload["candidate_ids"]),
                None if completion.classifier_result is None else int(completion.classifier_result),
                _canonical_json(payload["proposed_intents"]),
                _canonical_json(payload["validation_outcome"]),
                _canonical_json(payload["storage_outcome"]),
                _canonical_json(payload["memory_scopes"]),
                completion.duration_ms,
                completion_fingerprint,
                receipt_id,
                principal_id,
            ),
        )
        await self._connection.commit()
        snapshot = await self._by_id(receipt_id)
        if snapshot is None or snapshot.principal_id != principal_id:
            raise MemoryExtractionReceiptAccessDenied("Memory extraction receipt access denied")
        if cursor.rowcount != 1:
            existing_fingerprint = await self._completion_fingerprint(receipt_id)
            if existing_fingerprint != completion_fingerprint:
                raise MemoryExtractionReceiptConflict("Extraction receipt completion conflicts with prior result")
        return snapshot

    async def receipt(
        self,
        authority: MemoryExtractionReceiptAuthority,
        receipt_id: str,
    ) -> MemoryExtractionReceiptSnapshot:
        _bounded(receipt_id, field="receipt ID", maximum=128)
        snapshot = await self._by_id(receipt_id, principal_id=str(authority.principal_id))
        if snapshot is None:
            raise MemoryExtractionReceiptAccessDenied("Memory extraction receipt access denied")
        return snapshot

    async def _validate_run_binding(self, spec: MemoryExtractionReceiptSpec) -> None:
        async with self._connection.execute(
            "SELECT requested_by_principal_id, runtime_profile_id, inbound_message_id, result_message_id, status "
            "FROM runs WHERE id = ?",
            (spec.run_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None or tuple(str(row[index]) if row[index] is not None else None for index in range(5)) != (
            spec.principal_id,
            spec.runtime_profile_id,
            spec.source_message_id,
            spec.result_message_id,
            "completed",
        ):
            raise MemoryExtractionReceiptAccessDenied("Memory extraction receipt run binding is not authorized")

    async def _request_fingerprint(self, receipt_id: str) -> str | None:
        async with self._connection.execute(
            "SELECT request_fingerprint FROM memory_extraction_receipts WHERE receipt_id = ?",
            (receipt_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return str(row[0]) if row is not None else None

    async def _completion_fingerprint(self, receipt_id: str) -> str | None:
        async with self._connection.execute(
            "SELECT completion_fingerprint FROM memory_extraction_receipts WHERE receipt_id = ?",
            (receipt_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return str(row[0]) if row is not None and row[0] is not None else None

    async def _receipt_claim_owner(self, receipt_id: str) -> str | None:
        async with self._connection.execute(
            "SELECT claim_owner FROM memory_extraction_receipts WHERE receipt_id = ?",
            (receipt_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return str(row[0]) if row is not None else None

    async def _by_id(
        self,
        receipt_id: str,
        *,
        principal_id: str | None = None,
    ) -> MemoryExtractionReceiptSnapshot | None:
        where = "receipt_id = ?" if principal_id is None else "receipt_id = ? AND principal_id = ?"
        parameters = (receipt_id,) if principal_id is None else (receipt_id, principal_id)
        async with self._connection.execute(
            "SELECT receipt_id, principal_id, runtime_profile_id, run_id, source_message_id, "
            "result_message_id, extraction_role, backend, provider, model, prompt_version, "
            "schema_version, policy_version, status, decision_outcome, failure_code, candidate_ids_json, "
            "classifier_result, proposed_intents_json, validation_outcome_json, "
            "storage_outcome_json, memory_scope_json, duration_ms, created_at, completed_at "
            f"FROM memory_extraction_receipts WHERE {where}",
            parameters,
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        classifier = None if row[17] is None else bool(row[17])
        return MemoryExtractionReceiptSnapshot(
            receipt_id=str(row[0]),
            principal_id=str(row[1]),
            runtime_profile_id=str(row[2]),
            run_id=str(row[3]),
            source_message_id=str(row[4]),
            result_message_id=str(row[5]),
            extraction_role=str(row[6]),
            backend=str(row[7]),
            provider=str(row[8]),
            model=str(row[9]),
            prompt_version=str(row[10]),
            schema_version=str(row[11]),
            policy_version=str(row[12]),
            status=str(row[13]),
            decision_outcome=str(row[14]) if row[14] is not None else None,
            failure_code=str(row[15]) if row[15] is not None else None,
            candidate_ids=tuple(str(value) for value in json.loads(str(row[16]))),
            classifier_result=classifier,
            proposed_intents=tuple(json.loads(str(row[18]))),
            validation_outcome=dict(json.loads(str(row[19]))),
            storage_outcome=dict(json.loads(str(row[20]))),
            memory_scopes=tuple(json.loads(str(row[21]))),
            duration_ms=int(row[22]) if row[22] is not None else None,
            created_at=str(row[23]),
            completed_at=str(row[24]) if row[24] is not None else None,
        )
