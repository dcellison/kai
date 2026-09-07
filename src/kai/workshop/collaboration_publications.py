"""Attempt-scoped, canonical agent-authored Workshop publications."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from kai.backend import TraceEntry
from kai.workshop.artifacts import (
    MAX_ARTIFACT_BYTES,
    ArtifactCollaborationAttribution,
    StagedArtifact,
    WorkshopArtifactService,
    record_published_artifact_in_transaction,
)
from kai.workshop.collaboration_authority import (
    CollaborationAuthorization,
    CollaborationBaseIdentity,
    CollaborationOperation,
)
from kai.workshop.delivery_policy import WorkshopDeliveryBindingPolicy
from kai.workshop.domain import (
    ArtifactId,
    EventEnvelope,
    EventId,
    MessageId,
    MessageMention,
    WorkshopEventType,
)
from kai.workshop.human_notifications import append_human_notifications_in_transaction
from kai.workshop.inbound import resolve_message_mentions
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.run_execution_authority import RunExecutionClaim, StaleRunExecutionAuthorityError
from kai.workshop.run_traces import WorkshopRunTraceStore
from kai.workshop.store import WorkshopEventStore

_IDEMPOTENCY_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_MAX_BODY_LENGTH = 10_000
_MAX_CAPTION_LENGTH = 2_000
_MUTATION_TIMEOUT_SECONDS = 5.0


class CollaborationPublicationError(RuntimeError):
    """An attempt-scoped publication could not be completed."""


class CollaborationPublicationValidationError(CollaborationPublicationError):
    """The publication request is malformed."""


class CollaborationPublicationDenied(CollaborationPublicationError):
    """The exact attempt can no longer publish to its granted context."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class CollaborationPublicationResult:
    message_id: MessageId
    artifact_id: ArtifactId | None
    event_position: int
    replayed: bool


class _ExecutionService(Protocol):
    async def authorize_collaboration(
        self,
        proof: str,
        operation: CollaborationOperation,
        *,
        base_identity: CollaborationBaseIdentity,
        idempotency_key: str,
        request_hash: str,
        occurred_at: datetime,
    ) -> CollaborationAuthorization: ...


def _normalize_text(value: object, *, field: str, limit: int) -> str:
    if not isinstance(value, str):
        raise CollaborationPublicationValidationError(f"{field} must be a string")
    normalized = value.strip()
    if not normalized:
        raise CollaborationPublicationValidationError(f"{field} must not be empty")
    if len(normalized) > limit:
        raise CollaborationPublicationValidationError(f"{field} must be at most {limit} characters")
    return normalized


def _normalize_key(value: object) -> str:
    if not isinstance(value, str) or not _IDEMPOTENCY_KEY_PATTERN.fullmatch(value):
        raise CollaborationPublicationValidationError("idempotency_key must be a bounded opaque identifier")
    return value


def _hash(payload: dict[str, object]) -> str:
    return hashlib.sha256(json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def _timestamp(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CollaborationPublicationError("Stored collaboration lease is invalid")
    return parsed.astimezone(UTC)


def _mention_payload(mentions: tuple[MessageMention, ...]) -> list[dict[str, object]]:
    return [
        {
            "principal_id": mention.principal_id,
            "kind": mention.kind,
            "start": mention.start,
            "length": mention.length,
        }
        for mention in mentions
    ]


class WorkshopCollaborationPublicationService:
    """Publish bounded progress, thread replies, and artifacts from one exact attempt."""

    def __init__(
        self,
        store: WorkshopEventStore,
        execution: _ExecutionService,
        artifacts: WorkshopArtifactService,
        *,
        data_dir: Path,
        delivery_policy: WorkshopDeliveryBindingPolicy,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store
        self._execution = execution
        self._artifacts = artifacts
        self._artifact_storage_root = data_dir.resolve() / "files"
        self._delivery_policy = delivery_policy
        self._traces = WorkshopRunTraceStore(store)
        self._clock = clock or (lambda: datetime.now(UTC))

    async def publish_message(
        self,
        base_identity: CollaborationBaseIdentity,
        *,
        proof: str,
        kind: object,
        body: object,
        idempotency_key: object,
    ) -> CollaborationPublicationResult:
        if kind == "progress":
            operation = CollaborationOperation.PROGRESS_PUBLISH
        elif kind == "thread_reply":
            operation = CollaborationOperation.THREAD_REPLY
        else:
            raise CollaborationPublicationValidationError("kind must be progress or thread_reply")
        text = _normalize_text(body, field="body", limit=_MAX_BODY_LENGTH)
        key = _normalize_key(idempotency_key)
        request_hash = _hash({"body": text, "kind": str(kind)})
        now = self._clock()
        async with asyncio.timeout(_MUTATION_TIMEOUT_SECONDS):
            authorization = await self._execution.authorize_collaboration(
                proof,
                operation,
                base_identity=base_identity,
                idempotency_key=key,
                request_hash=request_hash,
                occurred_at=now,
            )
            return await self._record(
                authorization,
                body=text,
                idempotency_key=key,
                request_hash=request_hash,
                occurred_at=now,
                staged=None,
            )

    async def publish_artifact(
        self,
        base_identity: CollaborationBaseIdentity,
        *,
        proof: str,
        path: Path,
        caption: object,
        idempotency_key: object,
    ) -> CollaborationPublicationResult:
        if not isinstance(path, Path) or not path.is_absolute() or not path.is_file():
            raise CollaborationPublicationValidationError("path must identify an existing absolute file")
        normalized_caption = ""
        if caption is not None and caption != "":
            normalized_caption = _normalize_text(caption, field="caption", limit=_MAX_CAPTION_LENGTH)
        key = _normalize_key(idempotency_key)
        byte_size, digest = self._file_identity(path)
        request_hash = _hash(
            {
                "byte_size": byte_size,
                "caption": normalized_caption,
                "content_sha256": digest,
                "filename": path.name,
            }
        )
        now = self._clock()
        async with asyncio.timeout(_MUTATION_TIMEOUT_SECONDS):
            authorization = await self._execution.authorize_collaboration(
                proof,
                CollaborationOperation.ARTIFACT_PUBLISH,
                base_identity=base_identity,
                idempotency_key=key,
                request_hash=request_hash,
                occurred_at=now,
            )
            staged = await self._stage(authorization, path, key, now)
            try:
                return await self._record(
                    authorization,
                    body=normalized_caption or f"File: {path.name}",
                    idempotency_key=key,
                    request_hash=request_hash,
                    occurred_at=now,
                    staged=staged,
                )
            except Exception:
                staged.discard()
                raise

    @staticmethod
    def _file_identity(path: Path) -> tuple[int, str]:
        digest = hashlib.sha256()
        size = 0
        try:
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_ARTIFACT_BYTES:
                        raise CollaborationPublicationValidationError(
                            f"artifact must be at most {MAX_ARTIFACT_BYTES} bytes"
                        )
                    digest.update(chunk)
        except OSError as exc:
            raise CollaborationPublicationValidationError("artifact content is unavailable") from exc
        if size == 0:
            raise CollaborationPublicationValidationError("artifact content must not be empty")
        return size, digest.hexdigest()

    async def _stage(
        self,
        authorization: CollaborationAuthorization,
        path: Path,
        key: str,
        occurred_at: datetime,
    ) -> StagedArtifact:
        async def chunks() -> AsyncIterator[bytes]:
            with path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    yield chunk

        grant = authorization.grant
        return await self._artifacts.stage_upload(
            principal_id=grant.sponsor_principal_id,
            runtime_profile_id=grant.runtime_profile_id,
            filename=path.name,
            claimed_media_type=None,
            chunks=chunks(),
            source_transport="workshop_collaboration",
            source_unique_id=f"{grant.grant_id}:{key}",
            occurred_at=occurred_at,
            original_filename=path.name,
        )

    async def _record(
        self,
        authorization: CollaborationAuthorization,
        *,
        body: str,
        idempotency_key: str,
        request_hash: str,
        occurred_at: datetime,
        staged: StagedArtifact | None,
    ) -> CollaborationPublicationResult:
        grant = authorization.grant
        operation = authorization.operation
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            projection = CanonicalConversationProjection()
            await self._store.project_pending_in_transaction(projection)
            replay = await self._receipt(grant.grant_id, operation, idempotency_key)
            if replay is not None:
                if str(replay[0]) != request_hash:
                    raise CollaborationPublicationDenied(
                        "idempotency_conflict",
                        "Collaboration publication idempotency key was reused with different content",
                    )
                if staged is not None:
                    staged.discard()
                await connection.commit()
                if str(replay[1]) == "denied":
                    raise CollaborationPublicationDenied(
                        str(replay[2]),
                        f"Collaboration publication was denied: {replay[2]}",
                    )
                return CollaborationPublicationResult(
                    MessageId(str(replay[3])),
                    ArtifactId(str(replay[4])) if replay[4] is not None else None,
                    int(str(replay[5])),
                    True,
                )
            try:
                lease_version = await self._require_active(authorization, occurred_at)
            except CollaborationPublicationDenied as exc:
                await self._append_receipt(
                    authorization,
                    idempotency_key=idempotency_key,
                    request_hash=request_hash,
                    outcome="denied",
                    denial_code=exc.code,
                    message_id=None,
                    artifact_id=None,
                    occurred_at=occurred_at,
                )
                await self._store.project_pending_in_transaction(projection)
                await connection.commit()
                raise
            message_id = MessageId.derived(
                grant.grant_id,
                f"publication:{operation.value}:{idempotency_key}",
            )
            thread_root = grant.thread_root_id
            mentions = await resolve_message_mentions(self._store, grant.channel_id, body)
            event = EventEnvelope.create(
                event_id=EventId.derived(message_id, "created"),
                event_type=WorkshopEventType.MESSAGE_CREATED,
                event_version=3,
                workshop_id=grant.workshop_id,
                aggregate_type="message",
                aggregate_id=message_id,
                actor_principal_id=grant.agent_principal_id,
                occurred_at=occurred_at,
                idempotency_key=(
                    f"workshop-collaboration-publication:v1:{grant.grant_id}:"
                    f"{operation.value}:{idempotency_key}:message"
                ),
                payload={
                    "channel_id": grant.channel_id,
                    "author_principal_id": grant.agent_principal_id,
                    "body": body,
                    "mentions": _mention_payload(mentions),
                    "reply_to_message_id": thread_root,
                    "thread_root_id": thread_root,
                    "collaboration_operation": operation.value,
                    "collaboration_grant_id": grant.grant_id,
                    "agent_definition_revision_id": grant.agent_definition_revision_id,
                    "run_id": grant.run_id,
                    "run_attempt_id": grant.attempt_id,
                },
                metadata={"source": "workshop_collaboration_publication"},
            )
            inserted = await self._store.append_in_transaction(event)
            if not inserted.inserted:
                raise CollaborationPublicationError("Publication message unexpectedly already exists")
            await append_human_notifications_in_transaction(
                self._store,
                inserted.event,
                delivery_policy=self._delivery_policy,
            )
            await self._store.project_pending_in_transaction(projection)
            artifact_id: ArtifactId | None = None
            if staged is not None:
                artifact = await record_published_artifact_in_transaction(
                    self._store,
                    staged.for_message(message_id),
                    storage_root=self._artifact_storage_root,
                    collaboration=ArtifactCollaborationAttribution(
                        grant.grant_id,
                        grant.agent_definition_revision_id,
                        grant.run_id,
                        grant.attempt_id,
                    ),
                )
                if not artifact.inserted:
                    raise CollaborationPublicationError("Publication artifact unexpectedly already exists")
                artifact_id = ArtifactId(str(artifact.event.envelope.aggregate_id))
            await self._append_receipt(
                authorization,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                outcome="succeeded",
                denial_code=None,
                message_id=message_id,
                artifact_id=artifact_id,
                occurred_at=occurred_at,
            )
            await self._store.project_pending_in_transaction(projection)
            await self._append_trace(
                authorization,
                idempotency_key=idempotency_key,
                message_id=message_id,
                artifact_id=artifact_id,
                lease_version=lease_version,
                occurred_at=occurred_at,
            )
            await connection.commit()
            return CollaborationPublicationResult(message_id, artifact_id, inserted.event.position, False)
        except Exception:
            await connection.rollback()
            raise

    async def _receipt(
        self,
        grant_id: object,
        operation: CollaborationOperation,
        key: str,
    ) -> tuple[object, ...] | None:
        async with self._store.connection.execute(
            "SELECT receipt.request_hash, receipt.outcome, receipt.denial_code, "
            "receipt.message_id, receipt.artifact_id, message.created_event_position "
            "FROM collaboration_publication_receipts receipt "
            "LEFT JOIN messages message ON message.id = receipt.message_id "
            "WHERE grant_id = ? AND operation = ? AND idempotency_key = ?",
            (grant_id, operation.value, key),
        ) as cursor:
            row = await cursor.fetchone()
        return tuple(row) if row is not None else None

    async def _require_active(
        self,
        authorization: CollaborationAuthorization,
        now: datetime,
    ) -> int:
        grant = authorization.grant
        async with self._store.connection.execute(
            "SELECT ra.lease_version, ra.status, ra.lease_expires_at, r.status, "
            "r.cancellation_requested_at, d.lifecycle_state, d.active_revision_id, c.archived_at, "
            "g.revoked_at, g.thread_root_id, c.kind, "
            "EXISTS(SELECT 1 FROM channel_agents ca WHERE ca.channel_id = g.channel_id "
            "AND ca.agent_id = g.agent_id AND ca.sponsor_principal_id = g.sponsor_principal_id "
            "AND ca.sponsored_runtime_profile_id = g.runtime_profile_id AND ca.detached_at IS NULL), "
            "EXISTS(SELECT 1 FROM principal_agent_enablements pae "
            "WHERE pae.principal_id = g.sponsor_principal_id AND pae.agent_id = g.agent_id "
            "AND pae.runtime_profile_id = g.runtime_profile_id AND pae.lifecycle_state = 'enabled') "
            "FROM collaboration_grants g JOIN run_attempts ra ON ra.id = g.attempt_id "
            "JOIN runs r ON r.id = g.run_id JOIN agent_definition_revisions revision "
            "ON revision.id = g.agent_definition_revision_id JOIN agent_definitions d "
            "ON d.id = revision.agent_definition_id JOIN channels c ON c.id = g.channel_id "
            "WHERE g.id = ? AND g.attempt_id = ? AND g.run_id = ? "
            "AND g.agent_principal_id = ? AND g.agent_definition_revision_id = ?",
            (
                grant.grant_id,
                grant.attempt_id,
                grant.run_id,
                grant.agent_principal_id,
                grant.agent_definition_revision_id,
            ),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise CollaborationPublicationDenied("authority_unavailable", "Publication authority is unavailable")
        if (
            str(row[1]) != "started"
            or str(row[3]) != "started"
            or row[4] is not None
            or now >= _timestamp(row[2])
            or str(row[5]) != "active"
            or str(row[6]) != str(grant.agent_definition_revision_id)
            or row[7] is not None
            or row[8] is not None
            or not bool(row[11])
            or not bool(row[12])
        ):
            raise CollaborationPublicationDenied("attempt_not_active", "The collaboration attempt is no longer active")
        thread_root = str(row[9]) if row[9] is not None else None
        if authorization.operation == CollaborationOperation.PROGRESS_PUBLISH and thread_root is not None:
            raise CollaborationPublicationDenied("invalid_context", "Progress publication requires a top-level context")
        if authorization.operation == CollaborationOperation.THREAD_REPLY and (
            thread_root is None or str(row[10]) != "group"
        ):
            raise CollaborationPublicationDenied("invalid_context", "Thread reply requires the granted group thread")
        return int(row[0])

    async def _append_receipt(
        self,
        authorization: CollaborationAuthorization,
        *,
        idempotency_key: str,
        request_hash: str,
        outcome: str,
        denial_code: str | None,
        message_id: MessageId | None,
        artifact_id: ArtifactId | None,
        occurred_at: datetime,
    ) -> None:
        grant = authorization.grant
        event = EventEnvelope.create(
            event_id=EventId.derived(
                grant.grant_id,
                f"publication:{authorization.operation.value}:{idempotency_key}:recorded",
            ),
            event_type=WorkshopEventType.COLLABORATION_PUBLICATION_RECORDED,
            event_version=1,
            workshop_id=grant.workshop_id,
            aggregate_type="collaboration_grant",
            aggregate_id=grant.grant_id,
            actor_principal_id=grant.agent_principal_id,
            occurred_at=occurred_at,
            idempotency_key=(
                f"workshop-collaboration-publication:v1:{grant.grant_id}:"
                f"{authorization.operation.value}:{idempotency_key}:recorded"
            ),
            payload={
                "operation": authorization.operation.value,
                "idempotency_key": idempotency_key,
                "request_hash": request_hash,
                "outcome": outcome,
                "denial_code": denial_code,
                "message_id": message_id,
                "artifact_id": artifact_id,
            },
            metadata={"source": "workshop_collaboration_publication"},
        )
        result = await self._store.append_in_transaction(event)
        if not result.inserted:
            raise CollaborationPublicationError("Publication receipt unexpectedly already exists")

    async def _append_trace(
        self,
        authorization: CollaborationAuthorization,
        *,
        idempotency_key: str,
        message_id: MessageId,
        artifact_id: ArtifactId | None,
        lease_version: int,
        occurred_at: datetime,
    ) -> None:
        grant = authorization.grant
        detail = json.dumps(
            {
                "artifact_id": str(artifact_id) if artifact_id is not None else None,
                "channel_id": str(grant.channel_id),
                "grant_id": str(grant.grant_id),
                "message_id": str(message_id),
                "operation": authorization.operation.value,
                "thread_root_id": str(grant.thread_root_id) if grant.thread_root_id is not None else None,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        try:
            await self._traces.append_in_transaction(
                RunExecutionClaim(
                    grant.attempt_id,
                    grant.run_id,
                    grant.execution_owner_id,
                    grant.fence_token,
                    lease_version,
                ),
                TraceEntry(
                    kind="tool_result",
                    tool_use_id=f"publication:{idempotency_key}",
                    summary=f"Published {authorization.operation.value.replace('_', ' ')}",
                    detail=detail,
                    is_error=False,
                ),
                occurred_at=occurred_at,
            )
        except StaleRunExecutionAuthorityError as exc:
            raise CollaborationPublicationDenied(
                "attempt_not_active",
                "The collaboration attempt ended before publication was committed",
            ) from exc
