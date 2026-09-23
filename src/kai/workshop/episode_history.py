"""Canonical, immutable episode history with recoverable vector projection."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

import aiosqlite

from kai import memory
from kai.workshop.domain import (
    EventEnvelope,
    MemoryEpisodeId,
    PrincipalId,
    RuntimeProfileId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.memory_current_truth import CANONICAL_TEMPORAL_ROLE_KEY
from kai.workshop.memory_projection_status import ProjectionRetryResult
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import IdempotencyConflictError, WorkshopEventStore
from kai.workshop.temporal_memory import (
    EpisodeFollowupRelationship,
    resolve_memory_admission,
    validate_episode_payload,
)

log = logging.getLogger(__name__)

CANONICAL_EPISODE_ID_KEY = "canonical_memory_episode_id"
_MAX_VECTOR_ATTEMPTS = 3
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_REPEAT_SIMILARITY_THRESHOLD = 0.78


class EpisodeHistoryError(RuntimeError):
    """Base error for immutable episode-history mutations."""


class EpisodeHistoryAccessDenied(EpisodeHistoryError):
    """The principal/runtime pair does not own this history authority."""


class EpisodeHistoryConflict(EpisodeHistoryError):
    """A mutation conflicts with canonical episode history."""


class EpisodeHistoryProjectionFailed(EpisodeHistoryError):
    """Canonical history committed, but its vector projection is pending."""


@dataclass(frozen=True, slots=True)
class EpisodeHistoryAuthority:
    workshop_id: WorkshopId
    principal_id: PrincipalId
    runtime_profile_id: RuntimeProfileId


@dataclass(frozen=True, slots=True)
class EpisodeInput:
    goal: str
    context: str
    approach: str
    outcome: str
    outcome_quality: str
    lessons: str | None
    tags: tuple[str, ...]
    actors: tuple[str, ...]
    scope_kind: str
    scope_key: str
    reason: str
    evidence: tuple[dict[str, str | None], ...]
    vector_metadata: dict[str, object]
    occurred_from: datetime | None
    occurred_until: datetime | None
    observed_at: datetime | None
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
class EpisodeMutationResult:
    episode_id: MemoryEpisodeId
    event_position: int
    memory_id: str | None
    replayed: bool
    projection_status: str
    repeated_episode_id: MemoryEpisodeId | None


class EpisodeVectorAdapter(Protocol):
    async def find_episode(
        self,
        authority: EpisodeHistoryAuthority,
        episode_id: MemoryEpisodeId,
    ) -> memory.MemoryResult | None: ...

    async def add(
        self,
        authority: EpisodeHistoryAuthority,
        content: str,
        metadata: dict[str, object],
    ) -> str | None: ...


class Mem0EpisodeVectorAdapter:
    """Project immutable canonical episodes into Mem0 for semantic recall."""

    async def find_episode(
        self,
        authority: EpisodeHistoryAuthority,
        episode_id: MemoryEpisodeId,
    ) -> memory.MemoryResult | None:
        # Filtered in the vector store rather than by loading the owner's
        # whole corpus on every outbox step.
        rows = await asyncio.to_thread(
            memory.find_for_lifecycle_projection,
            user_id=str(authority.principal_id),
            runtime_profile_id=str(authority.runtime_profile_id),
            key=CANONICAL_EPISODE_ID_KEY,
            value=str(episode_id),
        )
        return rows[0] if rows else None

    async def add(
        self,
        authority: EpisodeHistoryAuthority,
        content: str,
        metadata: dict[str, object],
    ) -> str | None:
        tags = metadata.get("tags")
        return await asyncio.to_thread(
            memory.add_structured,
            content,
            user_id=str(authority.principal_id),
            memory_type="episode",
            tags=list(tags) if isinstance(tags, list) else None,
            metadata=metadata,
            runtime_profile_id=str(authority.runtime_profile_id),
        )


def _timestamp(value: datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("Episode timestamps must be timezone-aware")
    return value.astimezone(UTC).isoformat(timespec="microseconds")


def _request_hash(value: object) -> str:
    encoded = json.dumps(value, allow_nan=False, default=str, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(encoded.encode()).hexdigest()


def _semantic_payload(payload: dict[str, object]) -> dict[str, object]:
    copied = json.loads(json.dumps(payload, allow_nan=False, default=str))
    copied.pop("stored_at", None)
    return copied


def _normalized_episode_text(spec: EpisodeInput) -> str:
    return " ".join((spec.goal, spec.context, spec.approach, spec.outcome)).casefold()


def _similarity_fingerprint(spec: EpisodeInput) -> str:
    normalized = " ".join(_TOKEN_PATTERN.findall(_normalized_episode_text(spec)))
    return hashlib.sha256(normalized.encode()).hexdigest()


def _token_similarity(left: str, right: str) -> float:
    left_tokens = set(_TOKEN_PATTERN.findall(left.casefold()))
    right_tokens = set(_TOKEN_PATTERN.findall(right.casefold()))
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _episode_payload(spec: EpisodeInput) -> dict[str, object]:
    stored_at = datetime.now(UTC)
    return {
        "owner_principal_id": None,
        "runtime_profile_id": None,
        "scope_kind": spec.scope_kind,
        "scope_key": spec.scope_key,
        "content": f"{spec.goal}\n\n{spec.context}",
        "occurred_from": _timestamp(spec.occurred_from),
        "occurred_until": _timestamp(spec.occurred_until),
        "observed_at": _timestamp(spec.observed_at),
        "stored_at": _timestamp(stored_at),
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
        "migration_classification": spec.migration_classification,
        "migration_gaps": list(spec.migration_gaps),
        "admission_authority": resolve_memory_admission(
            spec.migration_classification,
            spec.admission_authority,
        ).value,
        "goal": spec.goal.strip(),
        "context": spec.context.strip(),
        "approach": spec.approach.strip(),
        "outcome": spec.outcome.strip(),
        "outcome_quality": spec.outcome_quality,
        "lessons": spec.lessons.strip() if spec.lessons else None,
        "tags": list(spec.tags),
        "actors": list(spec.actors),
        "vector_metadata": dict(spec.vector_metadata),
        "similarity_fingerprint": _similarity_fingerprint(spec),
    }


def validate_episode_input(
    spec: EpisodeInput,
    *,
    principal_id: PrincipalId,
    runtime_profile_id: RuntimeProfileId,
) -> None:
    """
    Raise ValueError when recording `spec` would be rejected, without writing.

    Builds exactly the payload `record` appends and runs the projection's
    own pure checks on it, so callers deciding in advance (reconciliation
    triage, apply preflight) use the rules that will actually apply. Only
    the database-backed ownership check is left to `record`.
    """
    try:
        payload = _episode_payload(spec)
    except (AttributeError, TypeError) as exc:
        raise ValueError("Episode input has an invalid shape") from exc
    payload["owner_principal_id"] = str(principal_id)
    payload["runtime_profile_id"] = str(runtime_profile_id)
    validate_episode_payload(payload)


class MemoryEpisodeHistoryService:
    """Commit episode history first, then recoverably project it to Mem0."""

    def __init__(self, store: WorkshopEventStore, vector: EpisodeVectorAdapter | None = None) -> None:
        self._store = store
        self._vector = vector or Mem0EpisodeVectorAdapter()
        self._projection = CanonicalConversationProjection()
        self._lock = asyncio.Lock()

    async def authority_for(
        self,
        principal_id: PrincipalId,
        runtime_profile_id: RuntimeProfileId,
    ) -> EpisodeHistoryAuthority:
        async with self._store.connection.execute(
            "SELECT wm.workshop_id FROM workshop_memberships wm "
            "JOIN runtime_profile_owners rpo ON rpo.principal_id = wm.principal_id "
            "WHERE wm.principal_id = ? AND rpo.runtime_profile_id = ? ORDER BY wm.workshop_id",
            (principal_id, runtime_profile_id),
        ) as cursor:
            rows = list(await cursor.fetchall())
        if len(rows) != 1:
            raise EpisodeHistoryAccessDenied("Episode-history owner/runtime authority is unavailable")
        return EpisodeHistoryAuthority(WorkshopId(str(rows[0][0])), principal_id, runtime_profile_id)

    async def record(
        self,
        authority: EpisodeHistoryAuthority,
        spec: EpisodeInput,
        *,
        idempotency_key: str,
    ) -> EpisodeMutationResult:
        episode_id = MemoryEpisodeId.derived(authority.workshop_id, f"episode:{idempotency_key}")
        payload = _episode_payload(spec)
        payload["owner_principal_id"] = str(authority.principal_id)
        payload["runtime_profile_id"] = str(authority.runtime_profile_id)
        request_hash = _request_hash({"episode_id": episode_id, "payload": _semantic_payload(payload)})
        repeated_episode_id: MemoryEpisodeId | None = None
        async with self._lock:
            connection = self._store.connection
            try:
                await connection.execute("BEGIN IMMEDIATE")
                existing = await self._store.event_by_idempotency_key(idempotency_key)
                replayed = existing is not None
                if existing is not None:
                    if existing.envelope.metadata.get("request_sha256") != request_hash:
                        raise EpisodeHistoryConflict(
                            "Episode mutation idempotency key conflicts with its prior request"
                        )
                else:
                    repeated_episode_id = await self._find_near_duplicate(authority, spec)
                    await self._store.append_in_transaction(
                        EventEnvelope.create(
                            event_type=WorkshopEventType.MEMORY_EPISODE_RECORDED,
                            event_version=2,
                            workshop_id=authority.workshop_id,
                            aggregate_type="memory_episode",
                            aggregate_id=episode_id,
                            actor_principal_id=authority.principal_id,
                            occurred_at=datetime.now(UTC),
                            idempotency_key=idempotency_key,
                            payload=payload,
                            metadata={"request_sha256": request_hash},
                        )
                    )
                    if repeated_episode_id is not None:
                        await self._store.append_in_transaction(
                            EventEnvelope.create(
                                event_type=WorkshopEventType.MEMORY_EPISODE_FOLLOWUP_RECORDED,
                                event_version=2,
                                workshop_id=authority.workshop_id,
                                aggregate_type="memory_episode",
                                aggregate_id=repeated_episode_id,
                                actor_principal_id=authority.principal_id,
                                occurred_at=datetime.now(UTC),
                                idempotency_key=f"{idempotency_key}:repeated",
                                payload={
                                    "target_episode_id": str(episode_id),
                                    "relationship": EpisodeFollowupRelationship.REPEATED.value,
                                    "reason": "A distinct later occurrence closely resembles this episode.",
                                },
                                metadata={"request_sha256": request_hash},
                            )
                        )
                await self._store.project_pending_in_transaction(self._projection)
                if replayed:
                    await connection.execute(
                        "UPDATE memory_episode_vector_operations SET status = 'pending', attempt_count = 0, "
                        "last_error_code = NULL, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
                        "WHERE episode_id = ? AND status = 'failed'",
                        (episode_id,),
                    )
                await connection.commit()
            except (EpisodeHistoryConflict, IdempotencyConflictError):
                await connection.rollback()
                raise
            except (ValueError, aiosqlite.IntegrityError) as exc:
                await connection.rollback()
                raise EpisodeHistoryConflict(str(exc)) from exc
            except BaseException:
                await connection.rollback()
                raise
            while await self._project_next():
                pass
            result = await self.snapshot(episode_id, replayed=replayed)
            if repeated_episode_id is None:
                repeated_episode_id = await self._repeated_parent(episode_id)
            return EpisodeMutationResult(
                result.episode_id,
                result.event_position,
                result.memory_id,
                result.replayed,
                result.projection_status,
                repeated_episode_id,
            )

    async def followup(
        self,
        authority: EpisodeHistoryAuthority,
        source_episode_id: MemoryEpisodeId,
        target_episode_id: MemoryEpisodeId,
        relationship: EpisodeFollowupRelationship,
        *,
        reason: str,
        idempotency_key: str,
    ) -> None:
        request_hash = _request_hash(
            {
                "source": source_episode_id,
                "target": target_episode_id,
                "relationship": relationship,
                "reason": reason,
            }
        )
        async with self._lock:
            connection = self._store.connection
            try:
                await connection.execute("BEGIN IMMEDIATE")
                existing = await self._store.event_by_idempotency_key(idempotency_key)
                if existing is not None:
                    if existing.envelope.metadata.get("request_sha256") != request_hash:
                        raise EpisodeHistoryConflict(
                            "Episode follow-up idempotency key conflicts with its prior request"
                        )
                else:
                    await self._store.append_in_transaction(
                        EventEnvelope.create(
                            event_type=WorkshopEventType.MEMORY_EPISODE_FOLLOWUP_RECORDED,
                            event_version=2,
                            workshop_id=authority.workshop_id,
                            aggregate_type="memory_episode",
                            aggregate_id=source_episode_id,
                            actor_principal_id=authority.principal_id,
                            occurred_at=datetime.now(UTC),
                            idempotency_key=idempotency_key,
                            payload={
                                "target_episode_id": str(target_episode_id),
                                "relationship": relationship.value,
                                "reason": reason,
                            },
                            metadata={"request_sha256": request_hash},
                        )
                    )
                await self._store.project_pending_in_transaction(self._projection)
                await connection.commit()
            except (IdempotencyConflictError, ValueError, aiosqlite.IntegrityError) as exc:
                await connection.rollback()
                raise EpisodeHistoryConflict(str(exc)) from exc
            except BaseException:
                await connection.rollback()
                raise

    async def snapshot(
        self,
        episode_id: MemoryEpisodeId,
        *,
        replayed: bool = True,
    ) -> EpisodeMutationResult:
        async with self._store.connection.execute(
            "SELECT e.created_event_position, v.memory_id, v.status "
            "FROM memory_episodes e LEFT JOIN memory_episode_vector_operations v "
            "ON v.episode_id = e.episode_id WHERE e.episode_id = ?",
            (episode_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise EpisodeHistoryConflict("Canonical episode is unavailable")
        return EpisodeMutationResult(
            episode_id=episode_id,
            event_position=int(row[0]),
            memory_id=str(row[1]) if row[1] is not None else None,
            replayed=replayed,
            projection_status=str(row[2]) if row[2] is not None else "pending",
            repeated_episode_id=await self._repeated_parent(episode_id),
        )

    async def recover_pending(self, *, retry_failed: bool = False) -> int:
        completed = 0
        async with self._lock:
            await self._store.connection.execute(
                "UPDATE memory_episode_vector_operations SET status = 'pending', "
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

    async def retry_failed(
        self,
        *,
        principal_id: PrincipalId | None = None,
        runtime_profile_id: RuntimeProfileId | None = None,
        episode_ids: tuple[MemoryEpisodeId, ...] | None = None,
    ) -> ProjectionRetryResult:
        """
        Re-run failed episode vector operations, then drain the outbox.

        Episodes have one operation each and never block one another, so
        this only resets the matching failed rows (keeping their identity)
        and projects them again. Filters work as in the fact lifecycle
        service; an empty `episode_ids` retries nothing.
        """
        if episode_ids is not None and not episode_ids:
            return ProjectionRetryResult(0, 0, 0)
        conditions = ["v.status = 'failed'"]
        parameters: list[str] = []
        if principal_id is not None:
            conditions.append("e.owner_principal_id = ?")
            parameters.append(str(principal_id))
        if runtime_profile_id is not None:
            conditions.append("e.runtime_profile_id = ?")
            parameters.append(str(runtime_profile_id))
        if episode_ids is not None:
            conditions.append(f"v.episode_id IN ({', '.join('?' for _ in episode_ids)})")
            parameters.extend(str(episode_id) for episode_id in episode_ids)
        async with self._lock:
            connection = self._store.connection
            async with connection.execute(
                "SELECT v.event_position FROM memory_episode_vector_operations v "
                "JOIN memory_episodes e ON e.episode_id = v.episode_id WHERE " + " AND ".join(conditions),
                parameters,
            ) as cursor:
                positions = [int(row[0]) for row in await cursor.fetchall()]
            if not positions:
                return ProjectionRetryResult(0, 0, 0)
            marks = ", ".join("?" for _ in positions)
            await connection.execute(
                "UPDATE memory_episode_vector_operations SET status = 'pending', attempt_count = 0, "
                "last_error_code = NULL, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') "
                f"WHERE status = 'failed' AND event_position IN ({marks})",
                positions,
            )
            await connection.commit()
            while await self._project_next():
                pass
            async with connection.execute(
                "SELECT status, COUNT(*) FROM memory_episode_vector_operations "
                f"WHERE event_position IN ({marks}) GROUP BY status",
                positions,
            ) as cursor:
                outcomes = {str(row[0]): int(row[1]) for row in await cursor.fetchall()}
        return ProjectionRetryResult(len(positions), outcomes.get("succeeded", 0), outcomes.get("failed", 0))

    async def _find_near_duplicate(
        self,
        authority: EpisodeHistoryAuthority,
        spec: EpisodeInput,
    ) -> MemoryEpisodeId | None:
        async with self._store.connection.execute(
            "SELECT episode_id, goal, context, approach, outcome, similarity_fingerprint "
            "FROM memory_episodes WHERE workshop_id = ? AND owner_principal_id = ? "
            "AND runtime_profile_id = ? AND scope_kind = ? AND scope_key = ? "
            "AND goal IS NOT NULL ORDER BY stored_at DESC LIMIT 100",
            (
                authority.workshop_id,
                authority.principal_id,
                authority.runtime_profile_id,
                spec.scope_kind,
                spec.scope_key,
            ),
        ) as cursor:
            rows = list(await cursor.fetchall())
        fingerprint = _similarity_fingerprint(spec)
        candidate = _normalized_episode_text(spec)
        for row in rows:
            if str(row[5]) == fingerprint:
                return MemoryEpisodeId(str(row[0]))
            prior = " ".join(str(value or "") for value in row[1:5])
            if _token_similarity(candidate, prior) >= _REPEAT_SIMILARITY_THRESHOLD:
                return MemoryEpisodeId(str(row[0]))
        return None

    async def _repeated_parent(self, episode_id: MemoryEpisodeId) -> MemoryEpisodeId | None:
        async with self._store.connection.execute(
            "SELECT source_episode_id FROM memory_episode_followups "
            "WHERE target_episode_id = ? AND relationship = 'repeated' "
            "ORDER BY created_event_position DESC LIMIT 1",
            (episode_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return MemoryEpisodeId(str(row[0])) if row is not None else None

    async def _project_next(self) -> bool:
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            async with connection.execute(
                "SELECT v.event_position, v.episode_id, v.attempt_count, e.workshop_id, "
                "e.owner_principal_id, e.runtime_profile_id, e.content, e.goal, e.context, e.approach, "
                "e.outcome, e.outcome_quality, e.lessons, e.tags_json, e.actors_json, "
                "e.scope_kind, e.scope_key, e.occurred_from, e.occurred_until, e.observed_at, "
                "e.stored_at, e.reason, e.evidence_json, e.source_receipt_id, e.source_run_id, "
                "e.source_message_id, e.result_message_id, e.backend, e.provider, e.model, "
                "e.prompt_version, e.schema_version, e.migration_classification, "
                "e.vector_metadata_json FROM memory_episode_vector_operations v "
                "JOIN memory_episodes e ON e.episode_id = v.episode_id "
                "WHERE v.status = 'pending' ORDER BY v.event_position LIMIT 1"
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                await connection.commit()
                return False
            event_position = int(row[0])
            cursor = await connection.execute(
                "UPDATE memory_episode_vector_operations SET status = 'executing', "
                "attempt_count = attempt_count + 1, "
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

        authority = EpisodeHistoryAuthority(
            WorkshopId(str(row[3])), PrincipalId(str(row[4])), RuntimeProfileId(str(row[5]))
        )
        episode_id = MemoryEpisodeId(str(row[1]))
        attempt = int(row[2]) + 1
        metadata = dict(json.loads(str(row[33])))
        metadata.update(
            {
                "source": "episode",
                "goal": str(row[7]),
                "context": str(row[8]),
                "approach": str(row[9]),
                "outcome": str(row[10]),
                "outcome_quality": str(row[11]),
                "lessons": str(row[12]) if row[12] is not None else None,
                "tags": list(json.loads(str(row[13]))),
                "actors": list(json.loads(str(row[14]))),
                "scope": str(row[15]),
                "project_id": str(row[16]) if str(row[15]) == "project" else None,
                "occurred_from": str(row[17]) if row[17] is not None else None,
                "occurred_until": str(row[18]) if row[18] is not None else None,
                "observed_at": str(row[19]) if row[19] is not None else None,
                "stored_at": str(row[20]),
                "reason": str(row[21]),
                "evidence": list(json.loads(str(row[22]))),
                "source_receipt_id": str(row[23]) if row[23] is not None else None,
                "source_run_id": str(row[24]) if row[24] is not None else None,
                "source_message_id": str(row[25]) if row[25] is not None else None,
                "result_message_id": str(row[26]) if row[26] is not None else None,
                "backend": str(row[27]) if row[27] is not None else None,
                "provider": str(row[28]) if row[28] is not None else None,
                "model": str(row[29]) if row[29] is not None else None,
                "prompt_version": str(row[30]) if row[30] is not None else None,
                "schema_version": str(row[31]) if row[31] is not None else None,
                "migration_classification": str(row[32]),
                CANONICAL_EPISODE_ID_KEY: str(episode_id),
                CANONICAL_TEMPORAL_ROLE_KEY: "historical_episode",
            }
        )
        try:
            existing = await self._vector.find_episode(authority, episode_id)
            memory_id = (
                existing.id if existing is not None else await self._vector.add(authority, str(row[6]), metadata)
            )
            if memory_id is None:
                raise EpisodeHistoryProjectionFailed("Episode vector creation failed")
            await connection.execute(
                "UPDATE memory_episode_vector_operations SET status = 'succeeded', memory_id = ?, "
                "last_error_code = NULL, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now'), "
                "completed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE event_position = ?",
                (memory_id, event_position),
            )
            await connection.commit()
        except Exception as exc:
            exhausted = attempt >= _MAX_VECTOR_ATTEMPTS
            await connection.execute(
                "UPDATE memory_episode_vector_operations SET status = ?, last_error_code = ?, "
                "updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE event_position = ?",
                ("failed" if exhausted else "pending", type(exc).__name__[:128], event_position),
            )
            await connection.commit()
            if exhausted:
                # The episode stays out of recall until someone retries.
                # Only identifiers and the error class are logged.
                log.warning(
                    "Episode vector projection failed for %s after %s attempts: %s",
                    episode_id,
                    attempt,
                    type(exc).__name__,
                )
        return True
