"""Canonical, replay-safe authority for temporal fact mutations."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Protocol

import aiosqlite

from kai import memory
from kai.workshop.domain import (
    EventEnvelope,
    MemoryClaimId,
    MemoryRevisionId,
    PrincipalId,
    RuntimeProfileId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.memory_current_truth import (
    CANONICAL_CLAIM_ID_KEY,
    CANONICAL_LIFECYCLE_STATE_KEY,
    CANONICAL_REVISION_ID_KEY,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import IdempotencyConflictError, WorkshopEventStore
from kai.workshop.temporal_memory import resolve_memory_admission

_ADOPT_MEMORY_ID_KEY = "_canonical_adopt_memory_id"

AUTOMATIC_SUPERSESSION_CONFIDENCE = 0.9
AUTOMATIC_SUPERSESSION_EVIDENCE = 1
_MAX_VECTOR_ATTEMPTS = 3


class FactLifecycleError(RuntimeError):
    """Base error for canonical fact lifecycle mutations."""


class FactLifecycleAccessDenied(FactLifecycleError):
    """The principal/runtime pair does not own this fact authority."""


class FactLifecycleConflict(FactLifecycleError):
    """The requested mutation is stale or conflicts with a prior replay."""


class FactLifecycleProjectionFailed(FactLifecycleError):
    """Canonical truth committed, but its vector projection is pending."""


class FactMutationSource(StrEnum):
    HUMAN = "human"
    MODEL = "model"
    MIGRATION = "migration"


@dataclass(frozen=True, slots=True)
class FactLifecycleAuthority:
    workshop_id: WorkshopId
    principal_id: PrincipalId
    runtime_profile_id: RuntimeProfileId


@dataclass(frozen=True, slots=True)
class FactRevisionInput:
    content: str
    scope_kind: str
    scope_key: str
    reason: str
    evidence: tuple[dict[str, str | None], ...]
    vector_metadata: dict[str, object]
    confidence: float
    asserted_at: datetime | None = None
    observed_at: datetime | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    source_receipt_id: str | None = None
    source_run_id: str | None = None
    source_message_id: str | None = None
    result_message_id: str | None = None
    backend: str | None = None
    provider: str | None = None
    model: str | None = None
    prompt_version: str | None = None
    schema_version: str | None = None
    migration_classification: str = "canonical"
    migration_gaps: tuple[str, ...] = ()
    admission_authority: str | None = None


@dataclass(frozen=True, slots=True)
class FactMutationResult:
    claim_id: MemoryClaimId
    revision_id: MemoryRevisionId
    state: str
    event_position: int
    memory_id: str | None
    replayed: bool
    projection_status: str


class FactVectorAdapter(Protocol):
    async def get(self, authority: FactLifecycleAuthority, memory_id: str) -> memory.MemoryResult | None: ...

    async def find_revision(
        self,
        authority: FactLifecycleAuthority,
        revision_id: MemoryRevisionId,
    ) -> memory.MemoryResult | None: ...

    async def add(
        self,
        authority: FactLifecycleAuthority,
        content: str,
        metadata: dict[str, object],
    ) -> str | None: ...

    async def replace(
        self,
        authority: FactLifecycleAuthority,
        memory_id: str,
        content: str,
        metadata: dict[str, object],
    ) -> bool: ...

    async def delete(self, authority: FactLifecycleAuthority, memory_id: str) -> bool: ...


class Mem0FactVectorAdapter:
    """Project canonical facts into Mem0 without making it authoritative."""

    async def get(self, authority: FactLifecycleAuthority, memory_id: str) -> memory.MemoryResult | None:
        return await asyncio.to_thread(
            memory.get_by_id_for_lifecycle_projection,
            user_id=str(authority.principal_id),
            memory_id=memory_id,
            runtime_profile_id=str(authority.runtime_profile_id),
        )

    async def find_revision(
        self,
        authority: FactLifecycleAuthority,
        revision_id: MemoryRevisionId,
    ) -> memory.MemoryResult | None:
        rows = await asyncio.to_thread(
            memory.get_all_for_lifecycle_projection,
            user_id=str(authority.principal_id),
            runtime_profile_id=str(authority.runtime_profile_id),
        )
        return next(
            (row for row in rows if row.metadata.get(CANONICAL_REVISION_ID_KEY) == str(revision_id)),
            None,
        )

    async def add(
        self,
        authority: FactLifecycleAuthority,
        content: str,
        metadata: dict[str, object],
    ) -> str | None:
        tags = metadata.get("tags")
        return await asyncio.to_thread(
            memory.add_structured,
            content,
            user_id=str(authority.principal_id),
            memory_type="fact",
            tags=list(tags) if isinstance(tags, list) else None,
            metadata=metadata,
            runtime_profile_id=str(authority.runtime_profile_id),
        )

    async def replace(
        self,
        authority: FactLifecycleAuthority,
        memory_id: str,
        content: str,
        metadata: dict[str, object],
    ) -> bool:
        updated = await asyncio.to_thread(
            memory.update_metadata,
            user_id=str(authority.principal_id),
            memory_id=memory_id,
            data=content,
            metadata=metadata,
            runtime_profile_id=str(authority.runtime_profile_id),
        )
        if updated:
            return True
        current = await self.get(authority, memory_id)
        return (
            current is not None
            and current.text == content
            and all(current.metadata.get(key) == value for key, value in metadata.items())
        )

    async def delete(self, authority: FactLifecycleAuthority, memory_id: str) -> bool:
        current = await self.get(authority, memory_id)
        if current is None:
            return True
        return await asyncio.to_thread(
            memory.delete_by_id_for_lifecycle_projection,
            user_id=str(authority.principal_id),
            memory_id=memory_id,
            runtime_profile_id=str(authority.runtime_profile_id),
        )


def _timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Fact lifecycle timestamps must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _request_hash(value: object) -> str:
    encoded = json.dumps(value, allow_nan=False, default=str, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _semantic_request(value: dict[str, Any]) -> dict[str, Any]:
    """Exclude commit-time fields while retaining every requested semantic."""
    copied = json.loads(json.dumps(value, allow_nan=False, default=str))
    payload = copied.get("payload")
    if isinstance(payload, dict):
        revision = payload.get("revision")
        if isinstance(revision, dict):
            revision.pop("stored_at", None)
    return copied


def _claim_identity(authority: FactLifecycleAuthority, spec: FactRevisionInput, stable_key: str) -> str:
    return _request_hash(
        {
            "principal": authority.principal_id,
            "runtime": authority.runtime_profile_id,
            "scope_kind": spec.scope_kind,
            "scope_key": spec.scope_key,
            "stable_key": stable_key,
        }
    )


def _revision_payload(spec: FactRevisionInput, revision_id: MemoryRevisionId, *, supersedes: str | None) -> dict:
    now = datetime.now(UTC)
    if not 0.0 <= spec.confidence <= 1.0:
        raise ValueError("Fact confidence must be between zero and one")
    metadata = dict(spec.vector_metadata)
    metadata["confidence"] = spec.confidence
    return {
        "revision_id": str(revision_id),
        "content": spec.content.strip(),
        "asserted_at": _timestamp(spec.asserted_at),
        "observed_at": _timestamp(spec.observed_at),
        "stored_at": _timestamp(now),
        "valid_from": _timestamp(spec.valid_from),
        "valid_until": _timestamp(spec.valid_until),
        "reason": spec.reason.strip(),
        "evidence": [dict(item) for item in spec.evidence],
        "source_receipt_id": spec.source_receipt_id,
        "source_run_id": spec.source_run_id,
        "source_message_id": spec.source_message_id,
        "result_message_id": spec.result_message_id,
        "backend": spec.backend,
        "provider": spec.provider,
        "model": spec.model,
        "prompt_version": spec.prompt_version,
        "schema_version": spec.schema_version,
        "supersedes_revision_id": supersedes,
        "migration_classification": spec.migration_classification,
        "migration_gaps": list(spec.migration_gaps),
        "admission_authority": resolve_memory_admission(
            spec.migration_classification,
            spec.admission_authority,
        ).value,
        "vector_metadata": metadata,
    }


class MemoryFactLifecycleService:
    """Commit canonical fact truth first, then recoverably project it to Mem0."""

    def __init__(self, store: WorkshopEventStore, vector: FactVectorAdapter | None = None) -> None:
        self._store = store
        self._vector = vector or Mem0FactVectorAdapter()
        self._projection = CanonicalConversationProjection()
        self._lock = asyncio.Lock()

    async def authority_for(
        self,
        principal_id: PrincipalId,
        runtime_profile_id: RuntimeProfileId,
    ) -> FactLifecycleAuthority:
        async with self._store.connection.execute(
            "SELECT wm.workshop_id FROM workshop_memberships wm "
            "JOIN runtime_profile_owners rpo ON rpo.principal_id = wm.principal_id "
            "WHERE wm.principal_id = ? AND rpo.runtime_profile_id = ? ORDER BY wm.workshop_id",
            (principal_id, runtime_profile_id),
        ) as cursor:
            rows = list(await cursor.fetchall())
        if len(rows) != 1:
            raise FactLifecycleAccessDenied("Fact lifecycle owner/runtime authority is unavailable")
        return FactLifecycleAuthority(WorkshopId(str(rows[0][0])), principal_id, runtime_profile_id)

    async def create(
        self,
        authority: FactLifecycleAuthority,
        spec: FactRevisionInput,
        *,
        idempotency_key: str,
        stable_claim_key: str,
    ) -> FactMutationResult:
        claim_id = MemoryClaimId.derived(
            authority.workshop_id,
            f"fact:{authority.principal_id}:{authority.runtime_profile_id}:{stable_claim_key}",
        )
        revision_id = MemoryRevisionId.derived(authority.workshop_id, f"fact-revision:{idempotency_key}")
        payload = {
            "owner_principal_id": str(authority.principal_id),
            "runtime_profile_id": str(authority.runtime_profile_id),
            "scope_kind": spec.scope_kind,
            "scope_key": spec.scope_key,
            "claim_identity_sha256": _claim_identity(authority, spec, stable_claim_key),
            "revision": _revision_payload(spec, revision_id, supersedes=None),
        }
        return await self._mutate(
            authority,
            event_type=WorkshopEventType.MEMORY_FACT_RECORDED,
            claim_id=claim_id,
            revision_id=revision_id,
            payload=payload,
            idempotency_key=idempotency_key,
        )

    async def supersede(
        self,
        authority: FactLifecycleAuthority,
        claim_id: MemoryClaimId,
        expected_revision_id: MemoryRevisionId,
        spec: FactRevisionInput,
        *,
        idempotency_key: str,
        source: FactMutationSource,
    ) -> FactMutationResult:
        if source == FactMutationSource.MODEL and (
            spec.confidence < AUTOMATIC_SUPERSESSION_CONFIDENCE or len(spec.evidence) < AUTOMATIC_SUPERSESSION_EVIDENCE
        ):
            return await self.open_conflict(
                authority,
                claim_id,
                expected_revision_id,
                spec,
                idempotency_key=idempotency_key,
            )
        revision_id = MemoryRevisionId.derived(authority.workshop_id, f"fact-revision:{idempotency_key}")
        return await self._mutate(
            authority,
            event_type=WorkshopEventType.MEMORY_FACT_SUPERSEDED,
            claim_id=claim_id,
            revision_id=revision_id,
            payload={
                "prior_revision_id": str(expected_revision_id),
                "reason": spec.reason,
                "scope_kind": spec.scope_kind,
                "scope_key": spec.scope_key,
                "revision": _revision_payload(spec, revision_id, supersedes=str(expected_revision_id)),
            },
            idempotency_key=idempotency_key,
        )

    async def open_conflict(
        self,
        authority: FactLifecycleAuthority,
        claim_id: MemoryClaimId,
        expected_revision_id: MemoryRevisionId,
        spec: FactRevisionInput,
        *,
        idempotency_key: str,
    ) -> FactMutationResult:
        revision_id = MemoryRevisionId.derived(authority.workshop_id, f"fact-revision:{idempotency_key}")
        return await self._mutate(
            authority,
            event_type=WorkshopEventType.MEMORY_FACT_CONFLICT_OPENED,
            claim_id=claim_id,
            revision_id=revision_id,
            payload={
                "active_revision_id": str(expected_revision_id),
                "reason": spec.reason,
                "revision": _revision_payload(spec, revision_id, supersedes=None),
            },
            idempotency_key=idempotency_key,
        )

    async def retract(
        self,
        authority: FactLifecycleAuthority,
        claim_id: MemoryClaimId,
        expected_revision_id: MemoryRevisionId,
        *,
        reason: str,
        idempotency_key: str,
        expired: bool = False,
    ) -> FactMutationResult:
        return await self._mutate(
            authority,
            event_type=(WorkshopEventType.MEMORY_FACT_EXPIRED if expired else WorkshopEventType.MEMORY_FACT_RETRACTED),
            claim_id=claim_id,
            revision_id=expected_revision_id,
            payload={"revision_id": str(expected_revision_id), "reason": reason},
            idempotency_key=idempotency_key,
        )

    async def restore(
        self,
        authority: FactLifecycleAuthority,
        claim_id: MemoryClaimId,
        prior_revision_id: MemoryRevisionId,
        spec: FactRevisionInput,
        *,
        idempotency_key: str,
    ) -> FactMutationResult:
        revision_id = MemoryRevisionId.derived(authority.workshop_id, f"fact-revision:{idempotency_key}")
        return await self._mutate(
            authority,
            event_type=WorkshopEventType.MEMORY_FACT_RESTORED,
            claim_id=claim_id,
            revision_id=revision_id,
            payload={
                "prior_revision_id": str(prior_revision_id),
                "reason": spec.reason,
                "scope_kind": spec.scope_kind,
                "scope_key": spec.scope_key,
                "revision": _revision_payload(spec, revision_id, supersedes=str(prior_revision_id)),
            },
            idempotency_key=idempotency_key,
        )

    async def resolve_conflict(
        self,
        authority: FactLifecycleAuthority,
        claim_id: MemoryClaimId,
        winner_revision_id: MemoryRevisionId,
        loser_revision_ids: tuple[MemoryRevisionId, ...],
        *,
        reason: str,
        idempotency_key: str,
    ) -> FactMutationResult:
        return await self._mutate(
            authority,
            event_type=WorkshopEventType.MEMORY_FACT_CONFLICT_RESOLVED,
            claim_id=claim_id,
            revision_id=winner_revision_id,
            payload={
                "winner_revision_id": str(winner_revision_id),
                "loser_revision_ids": [str(value) for value in loser_revision_ids],
                "reason": reason,
            },
            idempotency_key=idempotency_key,
        )

    async def adopt_legacy(
        self,
        authority: FactLifecycleAuthority,
        row: memory.MemoryResult,
        *,
        idempotency_key: str,
    ) -> FactMutationResult:
        existing_claim = row.metadata.get(CANONICAL_CLAIM_ID_KEY)
        existing_revision = row.metadata.get(CANONICAL_REVISION_ID_KEY)
        if isinstance(existing_claim, str) and isinstance(existing_revision, str):
            return await self.snapshot(MemoryClaimId(existing_claim), MemoryRevisionId(existing_revision))
        resolved = memory.resolve_memory_scope(row.metadata)
        gaps = ["assertion_time", "observation_time"]
        if not row.metadata.get("source"):
            gaps.append("provenance")
        spec = FactRevisionInput(
            content=row.text,
            scope_kind=resolved.scope if resolved.scope in {"global", "project"} else "global",
            scope_key=(str(resolved.project_id) if resolved.scope == "project" and resolved.project_id else ""),
            reason="Adopted existing semantic-memory fact into canonical lifecycle authority.",
            evidence=({"kind": "legacy", "reference_id": row.id, "sha256": None},),
            vector_metadata={**row.metadata, _ADOPT_MEMORY_ID_KEY: row.id},
            confidence=float(row.metadata.get("confidence", 0.5)),
            migration_classification="legacy_incomplete",
            migration_gaps=tuple(gaps),
        )
        return await self.create(
            authority,
            spec,
            idempotency_key=idempotency_key,
            stable_claim_key=f"legacy:{row.id}",
        )

    async def apply_extracted(
        self,
        authority: FactLifecycleAuthority,
        spec: FactRevisionInput,
        *,
        idempotency_key: str,
        stable_claim_key: str,
        existing: memory.MemoryResult | None = None,
    ) -> FactMutationResult:
        """Apply one model proposal through the same lifecycle as human edits."""
        if existing is None:
            return await self.create(
                authority,
                spec,
                idempotency_key=idempotency_key,
                stable_claim_key=stable_claim_key,
            )
        adopted = await self.adopt_legacy(
            authority,
            existing,
            idempotency_key=(
                f"memory-fact-adopt:{authority.principal_id}:{authority.runtime_profile_id}:{existing.id}"
            ),
        )
        return await self.supersede(
            authority,
            adopted.claim_id,
            adopted.revision_id,
            spec,
            idempotency_key=idempotency_key,
            source=FactMutationSource.MODEL,
        )

    async def snapshot(
        self,
        claim_id: MemoryClaimId,
        revision_id: MemoryRevisionId,
        *,
        replayed: bool = True,
    ) -> FactMutationResult:
        async with self._store.connection.execute(
            "SELECT s.state, r.created_event_position, "
            "(SELECT memory_id FROM memory_fact_vector_operations v "
            " WHERE v.claim_id = r.claim_id AND v.memory_id IS NOT NULL ORDER BY v.event_position DESC LIMIT 1), "
            "(SELECT status FROM memory_fact_vector_operations v "
            " WHERE v.claim_id = r.claim_id "
            " ORDER BY v.event_position DESC LIMIT 1) "
            "FROM memory_fact_revisions r JOIN memory_fact_revision_states s ON s.revision_id = r.revision_id "
            "WHERE r.claim_id = ? AND r.revision_id = ?",
            (claim_id, revision_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise FactLifecycleConflict("Canonical fact revision is unavailable")
        return FactMutationResult(
            claim_id,
            revision_id,
            str(row[0]),
            int(row[1]),
            str(row[2]) if row[2] is not None else None,
            replayed,
            str(row[3]) if row[3] is not None else "pending",
        )

    async def recover_pending(self, *, retry_failed: bool = False) -> int:
        completed = 0
        async with self._lock:
            await self._store.connection.execute(
                "UPDATE memory_fact_vector_operations SET status = 'pending', "
                "attempt_count = CASE WHEN ? THEN 0 ELSE attempt_count END, "
                "last_error_code = CASE WHEN ? THEN NULL ELSE last_error_code END, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
                "WHERE status = 'executing' OR (? AND status = 'failed')",
                (int(retry_failed), int(retry_failed), int(retry_failed)),
            )
            await self._store.connection.commit()
            while await self._project_next():
                completed += 1
        return completed

    async def _mutate(
        self,
        authority: FactLifecycleAuthority,
        *,
        event_type: WorkshopEventType,
        claim_id: MemoryClaimId,
        revision_id: MemoryRevisionId,
        payload: dict[str, Any],
        idempotency_key: str,
    ) -> FactMutationResult:
        request_hash = _request_hash(
            _semantic_request(
                {
                    "event_type": event_type,
                    "claim_id": claim_id,
                    "revision_id": revision_id,
                    "payload": payload,
                    "authority": authority,
                }
            )
        )
        async with self._lock:
            connection = self._store.connection
            try:
                await connection.execute("BEGIN IMMEDIATE")
                existing = await self._store.event_by_idempotency_key(idempotency_key)
                replayed = existing is not None
                if existing is not None:
                    if existing.envelope.metadata.get("request_sha256") != request_hash:
                        raise FactLifecycleConflict("Fact mutation idempotency key conflicts with its prior request")
                else:
                    envelope = EventEnvelope.create(
                        event_type=event_type,
                        event_version=1,
                        workshop_id=authority.workshop_id,
                        aggregate_type="memory_fact_claim",
                        aggregate_id=claim_id,
                        actor_principal_id=authority.principal_id,
                        occurred_at=datetime.now(UTC),
                        idempotency_key=idempotency_key,
                        payload=payload,
                        metadata={"request_sha256": request_hash},
                    )
                    await self._store.append_in_transaction(envelope)
                await self._store.project_pending_in_transaction(self._projection)
                if replayed:
                    await connection.execute(
                        "UPDATE memory_fact_vector_operations SET status = 'pending', attempt_count = 0, "
                        "last_error_code = NULL, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
                        "WHERE claim_id = ? AND status = 'failed'",
                        (claim_id,),
                    )
                await connection.commit()
            except (FactLifecycleConflict, IdempotencyConflictError):
                await connection.rollback()
                raise
            except (ValueError, aiosqlite.IntegrityError) as exc:
                await connection.rollback()
                raise FactLifecycleConflict(str(exc)) from exc
            except BaseException:
                await connection.rollback()
                raise
            while await self._project_next():
                pass
            return await self.snapshot(claim_id, revision_id, replayed=replayed)

    async def _project_next(self) -> bool:
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            async with connection.execute(
                "SELECT v.event_position, v.claim_id, v.revision_id, v.prior_revision_id, v.operation, "
                "v.attempt_count, c.workshop_id, c.owner_principal_id, c.runtime_profile_id, "
                "r.content, r.vector_metadata_json "
                "FROM memory_fact_vector_operations v "
                "JOIN memory_fact_claims c ON c.claim_id = v.claim_id "
                "JOIN memory_fact_revisions r ON r.revision_id = v.revision_id "
                "WHERE v.status = 'pending' AND NOT EXISTS ("
                " SELECT 1 FROM memory_fact_vector_operations prior"
                " WHERE prior.claim_id = v.claim_id AND prior.event_position < v.event_position"
                " AND prior.status != 'succeeded'"
                ") ORDER BY v.event_position LIMIT 1"
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                await connection.commit()
                return False
            event_position = int(row[0])
            cursor = await connection.execute(
                "UPDATE memory_fact_vector_operations SET status = 'executing', attempt_count = attempt_count + 1, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
                "WHERE event_position = ? AND status = 'pending'",
                (event_position,),
            )
            if cursor.rowcount != 1:
                await connection.rollback()
                return True
            await connection.commit()
        except BaseException:
            await connection.rollback()
            raise

        authority = FactLifecycleAuthority(
            WorkshopId(str(row[6])), PrincipalId(str(row[7])), RuntimeProfileId(str(row[8]))
        )
        claim_id = MemoryClaimId(str(row[1]))
        revision_id = MemoryRevisionId(str(row[2]))
        prior_revision_id = MemoryRevisionId(str(row[3])) if row[3] is not None else None
        operation = str(row[4])
        attempt = int(row[5]) + 1
        content = str(row[9])
        metadata = dict(json.loads(str(row[10])))
        adopt_memory_id = metadata.pop(_ADOPT_MEMORY_ID_KEY, None)
        metadata.update(
            {
                CANONICAL_CLAIM_ID_KEY: str(claim_id),
                CANONICAL_REVISION_ID_KEY: str(revision_id),
                CANONICAL_LIFECYCLE_STATE_KEY: "active",
            }
        )
        memory_id: str | None = None
        try:
            already = await self._vector.find_revision(authority, revision_id)
            if operation in {"upsert", "replace"} and already is not None:
                memory_id = already.id
            else:
                memory_id = str(adopt_memory_id) if isinstance(adopt_memory_id, str) else None
                if memory_id is None and prior_revision_id is not None:
                    memory_id = await self._memory_id_for_revision(prior_revision_id)
                if memory_id is None and already is not None:
                    memory_id = already.id
                if operation in {"upsert", "replace"}:
                    if memory_id is not None:
                        current = await self._vector.get(authority, memory_id)
                        if current is None:
                            memory_id = await self._vector.add(authority, content, metadata)
                            if memory_id is None:
                                raise FactLifecycleProjectionFailed("Vector recreation failed")
                        elif not await self._vector.replace(authority, memory_id, content, metadata):
                            raise FactLifecycleProjectionFailed("Vector replacement failed")
                    else:
                        memory_id = await self._vector.add(authority, content, metadata)
                        if memory_id is None:
                            raise FactLifecycleProjectionFailed("Vector creation failed")
                elif (
                    operation == "delete"
                    and memory_id is not None
                    and not await self._vector.delete(authority, memory_id)
                ):
                    raise FactLifecycleProjectionFailed("Vector deletion failed")
            await connection.execute(
                "UPDATE memory_fact_vector_operations SET status = 'succeeded', memory_id = ?, "
                "last_error_code = NULL, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), "
                "completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE event_position = ?",
                (memory_id, event_position),
            )
            await connection.commit()
        except Exception as exc:
            await connection.execute(
                "UPDATE memory_fact_vector_operations SET status = ?, last_error_code = ?, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE event_position = ?",
                ("failed" if attempt >= _MAX_VECTOR_ATTEMPTS else "pending", type(exc).__name__[:128], event_position),
            )
            await connection.commit()
        return True

    async def _memory_id_for_revision(self, revision_id: MemoryRevisionId) -> str | None:
        async with self._store.connection.execute(
            "SELECT memory_id FROM memory_fact_vector_operations "
            "WHERE revision_id = ? AND memory_id IS NOT NULL ORDER BY event_position DESC LIMIT 1",
            (revision_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return str(row[0]) if row is not None else None


def revision_input_with_metadata(spec: FactRevisionInput, metadata: dict[str, object]) -> FactRevisionInput:
    """Return a copy used by adapters that must preserve provider metadata."""
    return replace(spec, vector_metadata=metadata)
