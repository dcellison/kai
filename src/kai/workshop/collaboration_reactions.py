"""Attempt-scoped, canonical agent-authored Workshop reactions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from kai.backend import TraceEntry
from kai.workshop.collaboration_authority import (
    CollaborationAuthorization,
    CollaborationBaseIdentity,
    CollaborationOperation,
)
from kai.workshop.domain import EventEnvelope, EventId, MessageId, WorkshopEventType
from kai.workshop.message_reactions import validate_message_reaction
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.run_execution_authority import (
    RunExecutionClaim,
    StaleRunExecutionAuthorityError,
)
from kai.workshop.run_traces import WorkshopRunTraceStore
from kai.workshop.store import WorkshopEventStore

_IDEMPOTENCY_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_MUTATION_TIMEOUT_SECONDS = 2.0


class CollaborationReactionError(RuntimeError):
    """A collaboration reaction could not be completed."""


class CollaborationReactionValidationError(CollaborationReactionError):
    """The reaction request is malformed."""


class CollaborationReactionDenied(CollaborationReactionError):
    """The target is outside the exact attempt's visible context."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class CollaborationReactionResult:
    message_id: MessageId
    reaction: str
    active: bool
    changed: bool
    event_position: int | None
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


def _normalize_message_id(value: object) -> MessageId:
    if not isinstance(value, str):
        raise CollaborationReactionValidationError("message_id must be a canonical message identifier")
    try:
        return MessageId(value)
    except (TypeError, ValueError) as exc:
        raise CollaborationReactionValidationError("message_id must be a canonical message identifier") from exc


def _normalize_active(value: object) -> bool:
    if not isinstance(value, bool):
        raise CollaborationReactionValidationError("active must be a boolean")
    return value


def _normalize_idempotency_key(value: object) -> str:
    if not isinstance(value, str) or not _IDEMPOTENCY_KEY_PATTERN.fullmatch(value):
        raise CollaborationReactionValidationError("idempotency_key must be a bounded opaque identifier")
    return value


def _request_hash(message_id: MessageId, reaction: str, active: bool) -> str:
    payload = json.dumps(
        {"active": active, "message_id": str(message_id), "reaction": reaction},
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _timestamp(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise CollaborationReactionError("Stored collaboration lease is invalid")
    return parsed.astimezone(UTC)


class WorkshopCollaborationReactionService:
    """Set one agent reaction under an exact active attempt grant."""

    def __init__(
        self,
        store: WorkshopEventStore,
        execution: _ExecutionService,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._store = store
        self._execution = execution
        self._traces = WorkshopRunTraceStore(store)
        self._clock = clock or (lambda: datetime.now(UTC))

    async def react(
        self,
        base_identity: CollaborationBaseIdentity,
        *,
        proof: str,
        message_id: object,
        reaction: object,
        active: object,
        idempotency_key: object,
    ) -> CollaborationReactionResult:
        target = _normalize_message_id(message_id)
        try:
            normalized_reaction = validate_message_reaction(reaction)
        except ValueError as exc:
            raise CollaborationReactionValidationError(str(exc)) from exc
        desired = _normalize_active(active)
        key = _normalize_idempotency_key(idempotency_key)
        fingerprint = _request_hash(target, normalized_reaction, desired)
        now = self._clock()
        async with asyncio.timeout(_MUTATION_TIMEOUT_SECONDS):
            authorization = await self._execution.authorize_collaboration(
                proof,
                CollaborationOperation.REACTION,
                base_identity=base_identity,
                idempotency_key=key,
                request_hash=fingerprint,
                occurred_at=now,
            )
            return await self._mutate(
                authorization,
                message_id=target,
                reaction=normalized_reaction,
                active=desired,
                idempotency_key=key,
                request_hash=fingerprint,
                occurred_at=now,
            )

    async def _mutate(
        self,
        authorization: CollaborationAuthorization,
        *,
        message_id: MessageId,
        reaction: str,
        active: bool,
        idempotency_key: str,
        request_hash: str,
        occurred_at: datetime,
    ) -> CollaborationReactionResult:
        grant = authorization.grant
        now = occurred_at
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            projection = CanonicalConversationProjection()
            await self._store.project_pending_in_transaction(projection)
            replay = await self._receipt(grant.grant_id, idempotency_key)
            if replay is not None:
                if replay[0] != request_hash:
                    raise CollaborationReactionDenied(
                        "idempotency_conflict",
                        "Collaboration reaction idempotency key was reused with different content",
                    )
                await connection.commit()
                if str(replay[1]) == "denied":
                    raise CollaborationReactionDenied(
                        str(replay[2]),
                        f"Collaboration reaction was denied: {replay[2]}",
                    )
                return CollaborationReactionResult(
                    message_id,
                    reaction,
                    bool(replay[3]),
                    bool(replay[4]),
                    int(str(replay[5])) if replay[5] is not None else None,
                    True,
                )

            try:
                target = await self._active_target(authorization, message_id, now)
            except CollaborationReactionDenied as exc:
                receipt_event = self._receipt_event(
                    authorization,
                    message_id=message_id,
                    reaction=reaction,
                    active=active,
                    idempotency_key=idempotency_key,
                    request_hash=request_hash,
                    outcome="denied",
                    denial_code=exc.code,
                    changed=None,
                    mutation_position=None,
                    occurred_at=now,
                )
                inserted_receipt = await self._store.append_in_transaction(receipt_event)
                if not inserted_receipt.inserted:
                    raise CollaborationReactionError("Reaction denial receipt unexpectedly already exists") from exc
                await self._store.project_pending_in_transaction(projection)
                try:
                    await self._append_trace(
                        authorization,
                        idempotency_key=idempotency_key,
                        message_id=message_id,
                        reaction=reaction,
                        active=active,
                        changed=None,
                        denial_code=exc.code,
                        lease_version=None,
                        occurred_at=now,
                    )
                except StaleRunExecutionAuthorityError:
                    # The receipt remains the durable denial trace when the
                    # attempt loses execution authority during this race.
                    pass
                await connection.commit()
                raise
            async with connection.execute(
                "SELECT 1 FROM message_reactions WHERE message_id = ? AND principal_id = ? AND reaction = ?",
                (message_id, grant.agent_principal_id, reaction),
            ) as cursor:
                current_active = await cursor.fetchone() is not None
            changed = current_active != active
            mutation_position: int | None = None
            if changed:
                mutation_event = EventEnvelope.create(
                    event_id=EventId.derived(grant.grant_id, f"reaction:{idempotency_key}:mutation"),
                    event_type=(
                        WorkshopEventType.MESSAGE_REACTION_ADDED
                        if active
                        else WorkshopEventType.MESSAGE_REACTION_REMOVED
                    ),
                    event_version=2,
                    workshop_id=grant.workshop_id,
                    aggregate_type="message",
                    aggregate_id=message_id,
                    actor_principal_id=grant.agent_principal_id,
                    occurred_at=now,
                    idempotency_key=(f"workshop-collaboration-reaction:v1:{grant.grant_id}:{idempotency_key}:mutation"),
                    payload={
                        "channel_id": grant.channel_id,
                        "principal_id": grant.agent_principal_id,
                        "reaction": reaction,
                        "collaboration_grant_id": grant.grant_id,
                        "agent_definition_revision_id": grant.agent_definition_revision_id,
                        "run_id": grant.run_id,
                        "run_attempt_id": grant.attempt_id,
                    },
                    metadata={"source": "workshop_collaboration_reaction"},
                )
                inserted = await self._store.append_in_transaction(mutation_event)
                if not inserted.inserted:
                    raise CollaborationReactionError("Reaction mutation event unexpectedly already exists")
                mutation_position = inserted.event.position

            receipt_event = self._receipt_event(
                authorization,
                message_id=message_id,
                reaction=reaction,
                active=active,
                idempotency_key=idempotency_key,
                request_hash=request_hash,
                outcome="succeeded",
                denial_code=None,
                changed=changed,
                mutation_position=mutation_position,
                occurred_at=now,
            )
            inserted_receipt = await self._store.append_in_transaction(receipt_event)
            if not inserted_receipt.inserted:
                raise CollaborationReactionError("Reaction receipt event unexpectedly already exists")
            await self._store.project_pending_in_transaction(projection)
            await self._append_trace(
                authorization,
                idempotency_key=idempotency_key,
                message_id=message_id,
                reaction=reaction,
                active=active,
                changed=changed,
                denial_code=None,
                lease_version=int(str(target[0])),
                occurred_at=now,
            )
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise

        return CollaborationReactionResult(
            message_id,
            reaction,
            active,
            changed,
            mutation_position,
            False,
        )

    async def _receipt(self, grant_id: object, idempotency_key: str) -> tuple[object, ...] | None:
        async with self._store.connection.execute(
            "SELECT request_hash, outcome, denial_code, active, changed, mutation_event_position "
            "FROM collaboration_reaction_receipts WHERE grant_id = ? AND idempotency_key = ?",
            (grant_id, idempotency_key),
        ) as cursor:
            row = await cursor.fetchone()
        return tuple(row) if row is not None else None

    def _receipt_event(
        self,
        authorization: CollaborationAuthorization,
        *,
        message_id: MessageId,
        reaction: str,
        active: bool,
        idempotency_key: str,
        request_hash: str,
        outcome: str,
        denial_code: str | None,
        changed: bool | None,
        mutation_position: int | None,
        occurred_at: datetime,
    ) -> EventEnvelope:
        grant = authorization.grant
        return EventEnvelope.create(
            event_id=EventId.derived(grant.grant_id, f"reaction:{idempotency_key}:recorded"),
            event_type=WorkshopEventType.COLLABORATION_REACTION_RECORDED,
            event_version=1,
            workshop_id=grant.workshop_id,
            aggregate_type="collaboration_grant",
            aggregate_id=grant.grant_id,
            actor_principal_id=grant.agent_principal_id,
            occurred_at=occurred_at,
            idempotency_key=f"workshop-collaboration-reaction:v1:{grant.grant_id}:{idempotency_key}:recorded",
            payload={
                "idempotency_key": idempotency_key,
                "request_hash": request_hash,
                "message_id": message_id,
                "reaction": reaction,
                "active": active,
                "outcome": outcome,
                "denial_code": denial_code,
                "changed": changed,
                "mutation_event_position": mutation_position,
            },
            metadata={"source": "workshop_collaboration_reaction"},
        )

    async def _append_trace(
        self,
        authorization: CollaborationAuthorization,
        *,
        idempotency_key: str,
        message_id: MessageId,
        reaction: str,
        active: bool,
        changed: bool | None,
        denial_code: str | None,
        lease_version: int | None,
        occurred_at: datetime,
    ) -> None:
        grant = authorization.grant
        if lease_version is None:
            async with self._store.connection.execute(
                "SELECT lease_version FROM run_attempts WHERE id = ?",
                (grant.attempt_id,),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                return
            lease_version = int(row[0])
        detail = json.dumps(
            {
                "active": active,
                "changed": changed,
                "channel_id": str(grant.channel_id),
                "denial_code": denial_code,
                "grant_id": str(grant.grant_id),
                "message_id": str(message_id),
                "reaction": reaction,
                "thread_root_id": str(grant.thread_root_id) if grant.thread_root_id is not None else None,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
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
                tool_use_id=f"reaction:{idempotency_key}",
                summary=(
                    f"Denied {reaction} reaction: {denial_code}"
                    if denial_code is not None
                    else f"{'Added' if active else 'Removed'} {reaction} reaction"
                ),
                detail=detail,
                is_error=denial_code is not None,
            ),
            occurred_at=occurred_at,
        )

    async def _active_target(
        self,
        authorization: CollaborationAuthorization,
        message_id: MessageId,
        now: datetime,
    ) -> tuple[object, ...]:
        grant = authorization.grant
        async with self._store.connection.execute(
            "SELECT ra.lease_version, ra.status, ra.lease_expires_at, r.status, "
            "r.cancellation_requested_at, d.lifecycle_state, d.active_revision_id, c.archived_at, "
            "c.kind, g.revoked_at, g.issued_event_position, g.thread_root_id, "
            "m.created_event_position, m.thread_root_id, "
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
            "JOIN messages m ON m.id = ? AND m.channel_id = g.channel_id "
            "WHERE g.id = ? AND g.attempt_id = ? AND g.run_id = ? "
            "AND g.agent_principal_id = ? AND g.agent_definition_revision_id = ?",
            (
                message_id,
                grant.grant_id,
                grant.attempt_id,
                grant.run_id,
                grant.agent_principal_id,
                grant.agent_definition_revision_id,
            ),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise CollaborationReactionDenied("target_not_visible", "Reaction target is outside the granted context")
        if (
            str(row[1]) != "started"
            or str(row[3]) != "started"
            or row[4] is not None
            or now >= _timestamp(row[2])
            or str(row[5]) != "active"
            or str(row[6]) != str(grant.agent_definition_revision_id)
            or row[7] is not None
            or row[9] is not None
            or not bool(row[14])
            or not bool(row[15])
        ):
            raise CollaborationReactionDenied("attempt_not_active", "The collaboration attempt is no longer active")
        if int(row[12]) > int(row[10]):
            raise CollaborationReactionDenied(
                "target_not_visible", "Reaction target is newer than the attempt snapshot"
            )
        granted_thread = MessageId(str(row[11])) if row[11] is not None else None
        target_thread = MessageId(str(row[13])) if row[13] is not None else None
        if granted_thread is not None:
            visible = message_id == granted_thread or target_thread == granted_thread
        else:
            visible = str(row[8]) == "direct" or target_thread is None
        if not visible:
            raise CollaborationReactionDenied("target_not_visible", "Reaction target is outside the granted thread")
        return tuple(row)
