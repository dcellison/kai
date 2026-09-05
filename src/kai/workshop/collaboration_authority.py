"""Attempt-scoped authority for structured Workshop agent collaboration."""

from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from kai.workshop.agent_definitions import (
    COLLABORATION_OPERATIONS,
    AgentDefinitionRevision,
    load_agent_definition_revision,
    validate_collaboration_operations,
)
from kai.workshop.domain import (
    AgentDefinitionRevisionId,
    AgentId,
    ChannelId,
    CollaborationGrantId,
    EventEnvelope,
    EventId,
    MessageId,
    PrincipalId,
    RunAttemptId,
    RunExecutionOwnerId,
    RunId,
    RuntimeProfileId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.run_execution_authority import RunExecutionClaim
from kai.workshop.store import WorkshopEventStore

_REVOCATION_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_PROOF_REDACTION = "[redacted collaboration proof]"


class CollaborationOperation(StrEnum):
    """Versioned operation groups kept separate from runtime capabilities."""

    CONTEXT_READ = "context_read"
    REACTION = "reaction"
    PROGRESS_PUBLISH = "progress_publish"
    THREAD_REPLY = "thread_reply"
    ARTIFACT_PUBLISH = "artifact_publish"
    AGENT_DELEGATION = "agent_delegation"


if {item.value for item in CollaborationOperation} != COLLABORATION_OPERATIONS:
    raise RuntimeError("Collaboration operation vocabulary drifted from agent-definition validation")


class CollaborationAuthorityError(RuntimeError):
    """Base error for collaboration-grant operations."""


class CollaborationProofError(CollaborationAuthorityError):
    """A transient invocation proof is absent, unknown, or no longer usable."""


class CollaborationDenied(CollaborationAuthorityError):
    """A known invocation does not have the requested effective authority."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class CollaborationGrantConflict(CollaborationAuthorityError):
    """Durable state conflicts with the proposed attempt grant."""


@dataclass(frozen=True, slots=True)
class CollaborationOwnerPolicy:
    """Owner-controlled allowance captured immutably at attempt acceptance."""

    version: int
    allowed_operations: frozenset[CollaborationOperation]

    def __post_init__(self) -> None:
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 0:
            raise ValueError("owner policy version must be a non-negative integer")
        _validate_operation_set(self.allowed_operations, field_name="owner allowed_operations")


@dataclass(frozen=True, slots=True)
class CollaborationHostPolicy:
    """Server-owned maxima that no agent revision or owner can exceed."""

    version: int = 1
    allowed_operations: frozenset[CollaborationOperation] = frozenset(CollaborationOperation)
    quotas: Mapping[CollaborationOperation, int] = field(
        default_factory=lambda: {
            CollaborationOperation.CONTEXT_READ: 200,
            CollaborationOperation.REACTION: 64,
            CollaborationOperation.PROGRESS_PUBLISH: 32,
            CollaborationOperation.THREAD_REPLY: 32,
            CollaborationOperation.ARTIFACT_PUBLISH: 16,
            CollaborationOperation.AGENT_DELEGATION: 12,
        }
    )

    def __post_init__(self) -> None:
        if not isinstance(self.version, int) or isinstance(self.version, bool) or self.version < 1:
            raise ValueError("host policy version must be a positive integer")
        _validate_operation_set(self.allowed_operations, field_name="host allowed_operations")
        if set(self.quotas) != set(self.allowed_operations):
            raise ValueError("host quotas must cover exactly the host-allowed operations")
        if any(not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 for limit in self.quotas.values()):
            raise ValueError("host quotas must be positive integers")


@dataclass(frozen=True, slots=True)
class CollaborationGrantSnapshot:
    grant_id: CollaborationGrantId
    attempt_id: RunAttemptId
    run_id: RunId
    execution_owner_id: RunExecutionOwnerId
    fence_token: int
    workshop_id: WorkshopId
    requested_by_principal_id: PrincipalId
    agent_principal_id: PrincipalId
    agent_id: AgentId
    agent_definition_revision_id: AgentDefinitionRevisionId
    sponsor_principal_id: PrincipalId
    runtime_profile_id: RuntimeProfileId
    channel_id: ChannelId
    thread_root_id: MessageId | None
    requested_operations: frozenset[CollaborationOperation]
    owner_allowed_operations: frozenset[CollaborationOperation]
    host_allowed_operations: frozenset[CollaborationOperation]
    effective_operations: frozenset[CollaborationOperation]
    owner_policy_version: int
    host_policy_version: int
    quotas: Mapping[CollaborationOperation, int]
    proof_fingerprint: str
    issued_at: datetime
    initial_lease_expires_at: datetime
    revoked_at: datetime | None
    revocation_code: str | None
    state_version: int


@dataclass(frozen=True, slots=True)
class CollaborationInvocation:
    """Transient proof held only for one live attempt and never persisted."""

    grant_id: CollaborationGrantId
    attempt_id: RunAttemptId
    token: str = field(repr=False)

    def redact(self, text: str) -> str:
        return text.replace(self.token, _PROOF_REDACTION)


@dataclass(frozen=True, slots=True)
class CollaborationAuthorization:
    """Server-derived context returned after a proof and live-state check."""

    grant: CollaborationGrantSnapshot
    operation: CollaborationOperation


type OwnerPolicyResolver = Callable[[AgentDefinitionRevision], CollaborationOwnerPolicy]


def _default_owner_policy(revision: AgentDefinitionRevision) -> CollaborationOwnerPolicy:
    """Preserve existing delegation only; every new operation starts denied."""
    allowed = (
        frozenset({CollaborationOperation.AGENT_DELEGATION})
        if CollaborationOperation.AGENT_DELEGATION.value in revision.collaboration_operations
        else frozenset()
    )
    return CollaborationOwnerPolicy(version=0, allowed_operations=allowed)


def _validate_operation_set(value: object, *, field_name: str) -> None:
    if not isinstance(value, frozenset) or any(not isinstance(item, CollaborationOperation) for item in value):
        raise TypeError(f"{field_name} must be a frozenset of CollaborationOperation values")


def _timestamp(value: datetime, *, field_name: str) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _parse_timestamp(value: object) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def _operations_json(operations: frozenset[CollaborationOperation]) -> list[str]:
    return sorted(item.value for item in operations)


def _decode_operations(value: object) -> frozenset[CollaborationOperation]:
    normalized = validate_collaboration_operations(json.loads(str(value)))
    return frozenset(CollaborationOperation(item) for item in normalized)


class WorkshopCollaborationAuthority:
    """Issue durable grants and authenticate transient exact-attempt proofs."""

    def __init__(
        self,
        store: WorkshopEventStore,
        *,
        host_policy: CollaborationHostPolicy | None = None,
        owner_policy_resolver: OwnerPolicyResolver | None = None,
        token_factory: Callable[[], str] | None = None,
    ) -> None:
        self._store = store
        self._host_policy = host_policy or CollaborationHostPolicy()
        self._owner_policy_resolver = owner_policy_resolver or _default_owner_policy
        self._token_factory = token_factory or (lambda: secrets.token_urlsafe(32))
        self._invocations_by_token: dict[str, CollaborationInvocation] = {}
        self._invocations_by_attempt: dict[RunAttemptId, CollaborationInvocation] = {}

    async def available(self) -> bool:
        """Return whether the current database has crossed the authority schema boundary."""
        async with self._store.connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'collaboration_grants'"
        ) as cursor:
            return await cursor.fetchone() is not None

    async def issue(
        self,
        claim: RunExecutionClaim,
        *,
        occurred_at: datetime,
    ) -> tuple[CollaborationGrantSnapshot, CollaborationInvocation]:
        """Snapshot effective policy for one exact started attempt and mint its proof."""
        if not isinstance(claim, RunExecutionClaim):
            raise ValueError("claim must be a RunExecutionClaim")
        now = _timestamp(occurred_at, field_name="occurred_at")
        existing_invocation = self._invocations_by_attempt.get(claim.attempt_id)
        if existing_invocation is not None:
            return await self.snapshot(existing_invocation.grant_id), existing_invocation

        connection = self._store.connection
        token = self._new_token()
        fingerprint = hashlib.sha256(token.encode()).hexdigest()
        try:
            await connection.execute("BEGIN IMMEDIATE")
            projection = CanonicalConversationProjection()
            await self._store.project_pending_in_transaction(projection)
            async with connection.execute(
                "SELECT ra.run_id, ra.owner_id, ra.fence_token, ra.status, ra.lease_expires_at, "
                "r.workshop_id, r.requested_by_principal_id, r.agent_id, "
                "r.agent_definition_revision_id, r.sponsor_principal_id, r.runtime_profile_id, "
                "r.channel_id, r.status, r.cancellation_requested_at, m.thread_root_id, a.principal_id "
                "FROM run_attempts ra JOIN runs r ON r.id = ra.run_id "
                "JOIN messages m ON m.id = r.inbound_message_id "
                "JOIN agents a ON a.id = r.agent_id WHERE ra.id = ?",
                (claim.attempt_id,),
            ) as cursor:
                row = await cursor.fetchone()
            if (
                row is None
                or RunId(str(row[0])) != claim.run_id
                or RunExecutionOwnerId(str(row[1])) != claim.owner_id
                or int(row[2]) != claim.fence_token
                or str(row[3]) != "started"
                or str(row[12]) != "started"
                or row[13] is not None
                or now >= _parse_timestamp(row[4])
            ):
                raise CollaborationGrantConflict("Collaboration requires the exact active started attempt")
            revision_id = AgentDefinitionRevisionId(str(row[8]))
            revision = await load_agent_definition_revision(self._store, revision_id)
            if revision is None or revision.agent_id != AgentId(str(row[7])):
                raise CollaborationGrantConflict("Collaboration revision is unavailable")
            requested = frozenset(CollaborationOperation(item) for item in revision.collaboration_operations)
            owner_policy = self._owner_policy_resolver(revision)
            if not isinstance(owner_policy, CollaborationOwnerPolicy):
                raise TypeError("owner_policy_resolver must return CollaborationOwnerPolicy")
            host_allowed = self._host_policy.allowed_operations
            effective = requested & owner_policy.allowed_operations & host_allowed
            grant_id = CollaborationGrantId.derived(claim.attempt_id, "collaboration-grant")
            key = f"workshop-collaboration-grant:v1:{grant_id}:issued"
            prior = await self._store.event_by_idempotency_key(key)
            if prior is not None:
                await connection.rollback()
                raise CollaborationGrantConflict("Durable collaboration grant exists without a live invocation proof")
            workshop_id = WorkshopId(str(row[5]))
            requested_by = PrincipalId(str(row[6]))
            agent_id = AgentId(str(row[7]))
            sponsor = PrincipalId(str(row[9]))
            runtime_profile = RuntimeProfileId(str(row[10]))
            channel_id = ChannelId(str(row[11]))
            thread_root = MessageId(str(row[14])) if row[14] is not None else None
            agent_principal = PrincipalId(str(row[15]))
            expiry = _parse_timestamp(row[4])
            event = EventEnvelope.create(
                event_id=EventId.derived(grant_id, "issued"),
                event_type=WorkshopEventType.COLLABORATION_GRANT_ISSUED,
                event_version=1,
                workshop_id=workshop_id,
                aggregate_type="collaboration_grant",
                aggregate_id=grant_id,
                actor_principal_id=agent_principal,
                occurred_at=now,
                idempotency_key=key,
                payload={
                    "attempt_id": claim.attempt_id,
                    "run_id": claim.run_id,
                    "execution_owner_id": claim.owner_id,
                    "fence_token": claim.fence_token,
                    "requested_by_principal_id": requested_by,
                    "agent_principal_id": agent_principal,
                    "agent_id": agent_id,
                    "agent_definition_revision_id": revision_id,
                    "sponsor_principal_id": sponsor,
                    "runtime_profile_id": runtime_profile,
                    "channel_id": channel_id,
                    "thread_root_id": thread_root,
                    "requested_operations": _operations_json(requested),
                    "owner_allowed_operations": _operations_json(owner_policy.allowed_operations),
                    "host_allowed_operations": _operations_json(host_allowed),
                    "effective_operations": _operations_json(effective),
                    "owner_policy_version": owner_policy.version,
                    "host_policy_version": self._host_policy.version,
                    "quotas": {
                        operation.value: self._host_policy.quotas[operation]
                        for operation in sorted(effective, key=lambda item: item.value)
                    },
                    "proof_fingerprint": fingerprint,
                    "initial_lease_expires_at": expiry.isoformat(),
                },
                metadata={"source": "workshop_collaboration_authority"},
            )
            await self._store.append_in_transaction(event)
            await self._store.project_pending_in_transaction(projection)
            snapshot = await self._snapshot(grant_id)
            await connection.commit()
        except Exception:
            await connection.rollback()
            raise
        invocation = CollaborationInvocation(grant_id, claim.attempt_id, token)
        self._invocations_by_token[token] = invocation
        self._invocations_by_attempt[claim.attempt_id] = invocation
        return snapshot, invocation

    async def authenticate(
        self,
        token: str,
        operation: CollaborationOperation,
        *,
        occurred_at: datetime,
    ) -> CollaborationAuthorization:
        """Resolve an untrusted token through the exact live grant and attempt."""
        if not isinstance(operation, CollaborationOperation):
            raise ValueError("operation must be a CollaborationOperation")
        invocation = self._match_token(token)
        if invocation is None:
            raise CollaborationProofError("Invalid collaboration proof")
        now = _timestamp(occurred_at, field_name="occurred_at")
        grant = await self._live_grant(invocation.grant_id, occurred_at=now)
        fingerprint = hashlib.sha256(token.encode()).hexdigest()
        if not hmac.compare_digest(fingerprint, grant.proof_fingerprint):
            raise CollaborationProofError("Invalid collaboration proof")
        if operation not in grant.effective_operations:
            raise CollaborationDenied(
                "operation_not_granted",
                f"The active attempt is not authorized for {operation.value}",
            )
        return CollaborationAuthorization(grant, operation)

    async def revoke(
        self,
        invocation: CollaborationInvocation,
        *,
        revocation_code: str,
        occurred_at: datetime,
    ) -> tuple[CollaborationGrantSnapshot, bool]:
        """Fence one live proof first, then record durable revocation."""
        if not isinstance(invocation, CollaborationInvocation):
            raise ValueError("invocation must be a CollaborationInvocation")
        if not _REVOCATION_CODE_PATTERN.fullmatch(revocation_code):
            raise ValueError("revocation_code must be a lowercase bounded identifier")
        now = _timestamp(occurred_at, field_name="occurred_at")
        self._drop_invocation(invocation)
        return await self._revoke_grant(
            invocation.grant_id,
            revocation_code=revocation_code,
            occurred_at=now,
        )

    async def reconcile_unbound(self, *, occurred_at: datetime) -> int:
        """Durably revoke grants whose transient proof did not survive this host."""
        now = _timestamp(occurred_at, field_name="occurred_at")
        async with self._store.connection.execute(
            "SELECT id FROM collaboration_grants WHERE revoked_at IS NULL ORDER BY issued_event_position"
        ) as cursor:
            grant_ids = [CollaborationGrantId(str(row[0])) for row in await cursor.fetchall()]
        live_grants = {invocation.grant_id for invocation in self._invocations_by_attempt.values()}
        reconciled = 0
        for grant_id in grant_ids:
            if grant_id in live_grants:
                continue
            _snapshot, changed = await self._revoke_grant(
                grant_id,
                revocation_code="host_restart",
                occurred_at=now,
            )
            reconciled += int(changed)
        return reconciled

    async def _revoke_grant(
        self,
        grant_id: CollaborationGrantId,
        *,
        revocation_code: str,
        occurred_at: datetime,
    ) -> tuple[CollaborationGrantSnapshot, bool]:
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            projection = CanonicalConversationProjection()
            await self._store.project_pending_in_transaction(projection)
            current = await self._snapshot(grant_id)
            if current.revoked_at is not None:
                await connection.commit()
                return current, False
            key = f"workshop-collaboration-grant:v1:{grant_id}:revoked"
            event = EventEnvelope.create(
                event_id=EventId.derived(grant_id, "revoked"),
                event_type=WorkshopEventType.COLLABORATION_GRANT_REVOKED,
                event_version=1,
                workshop_id=current.workshop_id,
                aggregate_type="collaboration_grant",
                aggregate_id=grant_id,
                actor_principal_id=current.agent_principal_id,
                occurred_at=occurred_at,
                idempotency_key=key,
                payload={
                    "revocation_code": revocation_code,
                    "expected_state_version": current.state_version,
                },
                metadata={"source": "workshop_collaboration_authority"},
            )
            await self._store.append_in_transaction(event)
            await self._store.project_pending_in_transaction(projection)
            updated = await self._snapshot(grant_id)
            await connection.commit()
            return updated, True
        except Exception:
            await connection.rollback()
            raise

    async def snapshot(self, grant_id: CollaborationGrantId) -> CollaborationGrantSnapshot:
        if not isinstance(grant_id, CollaborationGrantId):
            raise ValueError("grant_id must be a CollaborationGrantId")
        snapshot = await self._snapshot(grant_id)
        return snapshot

    def _new_token(self) -> str:
        for _ in range(100):
            token = self._token_factory()
            if not isinstance(token, str) or len(token) < 32:
                raise ValueError("collaboration token factory must return at least 32 characters")
            if token not in self._invocations_by_token:
                return token
        raise RuntimeError("Could not allocate a unique collaboration proof")

    def _match_token(self, token: str) -> CollaborationInvocation | None:
        if not isinstance(token, str) or not token:
            return None
        matched = None
        for expected, invocation in self._invocations_by_token.items():
            if hmac.compare_digest(token, expected):
                matched = invocation
        return matched

    def _drop_invocation(self, invocation: CollaborationInvocation) -> None:
        current = self._invocations_by_attempt.get(invocation.attempt_id)
        if current == invocation:
            self._invocations_by_attempt.pop(invocation.attempt_id, None)
        stored = self._invocations_by_token.get(invocation.token)
        if stored == invocation:
            self._invocations_by_token.pop(invocation.token, None)

    async def _live_grant(
        self,
        grant_id: CollaborationGrantId,
        *,
        occurred_at: datetime,
    ) -> CollaborationGrantSnapshot:
        snapshot = await self._snapshot(grant_id)
        async with self._store.connection.execute(
            "SELECT ra.status, ra.lease_expires_at, r.status, r.cancellation_requested_at, "
            "d.lifecycle_state, c.archived_at, "
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
            "WHERE g.id = ?",
            (grant_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if snapshot.revoked_at is not None:
            raise CollaborationDenied("grant_revoked", "The collaboration grant was revoked")
        if row is None:
            raise CollaborationProofError("Collaboration grant is unavailable")
        if str(row[0]) != "started" or str(row[2]) != "started" or row[3] is not None:
            raise CollaborationDenied("attempt_not_active", "The collaboration attempt is no longer active")
        if occurred_at >= _parse_timestamp(row[1]):
            raise CollaborationDenied("grant_expired", "The collaboration grant has expired")
        if str(row[4]) != "active" or row[5] is not None:
            raise CollaborationDenied("agent_unavailable", "The collaboration agent or channel is unavailable")
        if not bool(row[6]) or not bool(row[7]):
            raise CollaborationDenied("authority_detached", "The collaboration runtime authority is detached")
        return snapshot

    async def _snapshot(self, grant_id: CollaborationGrantId) -> CollaborationGrantSnapshot:
        async with self._store.connection.execute(
            "SELECT id, attempt_id, run_id, execution_owner_id, fence_token, workshop_id, "
            "requested_by_principal_id, agent_principal_id, agent_id, "
            "agent_definition_revision_id, sponsor_principal_id, runtime_profile_id, "
            "channel_id, thread_root_id, requested_operations_json, "
            "owner_allowed_operations_json, host_allowed_operations_json, "
            "effective_operations_json, owner_policy_version, host_policy_version, quotas_json, "
            "proof_fingerprint, issued_at, initial_lease_expires_at, revoked_at, "
            "revocation_code, state_version FROM collaboration_grants WHERE id = ?",
            (grant_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise CollaborationGrantConflict("Collaboration grant was not found")
        quotas_payload = json.loads(str(row[20]))
        if not isinstance(quotas_payload, dict):
            raise CollaborationGrantConflict("Stored collaboration quotas are invalid")
        quotas = {CollaborationOperation(str(key)): int(value) for key, value in quotas_payload.items()}
        return CollaborationGrantSnapshot(
            grant_id=CollaborationGrantId(str(row[0])),
            attempt_id=RunAttemptId(str(row[1])),
            run_id=RunId(str(row[2])),
            execution_owner_id=RunExecutionOwnerId(str(row[3])),
            fence_token=int(row[4]),
            workshop_id=WorkshopId(str(row[5])),
            requested_by_principal_id=PrincipalId(str(row[6])),
            agent_principal_id=PrincipalId(str(row[7])),
            agent_id=AgentId(str(row[8])),
            agent_definition_revision_id=AgentDefinitionRevisionId(str(row[9])),
            sponsor_principal_id=PrincipalId(str(row[10])),
            runtime_profile_id=RuntimeProfileId(str(row[11])),
            channel_id=ChannelId(str(row[12])),
            thread_root_id=MessageId(str(row[13])) if row[13] is not None else None,
            requested_operations=_decode_operations(row[14]),
            owner_allowed_operations=_decode_operations(row[15]),
            host_allowed_operations=_decode_operations(row[16]),
            effective_operations=_decode_operations(row[17]),
            owner_policy_version=int(row[18]),
            host_policy_version=int(row[19]),
            quotas=quotas,
            proof_fingerprint=str(row[21]),
            issued_at=_parse_timestamp(row[22]),
            initial_lease_expires_at=_parse_timestamp(row[23]),
            revoked_at=_parse_timestamp(row[24]) if row[24] is not None else None,
            revocation_code=str(row[25]) if row[25] is not None else None,
            state_version=int(row[26]),
        )
