"""Canonical owner controls and activity for Workshop collaboration tools."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from kai.workshop.agent_definitions import (
    validate_collaboration_operations,
)
from kai.workshop.collaboration_authority import (
    CollaborationHostPolicy,
    CollaborationOperation,
)
from kai.workshop.domain import (
    AgentDefinitionId,
    EventEnvelope,
    EventId,
    PrincipalId,
    RunId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.private_text_execution import WorkshopPrivateTextExecutionService
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.standing_participation import end_active_standings_in_transaction
from kai.workshop.store import IdempotencyConflictError, WorkshopEventStore

_OPERATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class WorkshopCollaborationPolicyError(RuntimeError):
    """Base error for owner collaboration policy."""


class WorkshopCollaborationPolicyAccessDenied(WorkshopCollaborationPolicyError):
    """The principal cannot inspect or mutate this definition."""


class WorkshopCollaborationPolicyValidationError(WorkshopCollaborationPolicyError):
    """The requested policy is malformed."""


class WorkshopCollaborationPolicyConflict(WorkshopCollaborationPolicyError):
    """The policy changed or an idempotency key conflicted."""


class WorkshopCollaborationPolicyStorageError(WorkshopCollaborationPolicyError):
    """Canonical policy state could not be read or persisted."""


@dataclass(frozen=True, slots=True)
class CollaborationOperationState:
    operation: str
    requested: bool
    owner_allowed: bool | None
    host_allowed: bool
    effective_for_new_attempt: bool
    unavailable_reason: str | None
    quota: int | None


@dataclass(frozen=True, slots=True)
class CollaborationPolicySnapshot:
    definition_id: AgentDefinitionId
    owner_principal_id: PrincipalId
    can_manage: bool
    policy_version: int
    active_revision_id: str | None
    active_grants: int | None
    operations: tuple[CollaborationOperationState, ...]


@dataclass(frozen=True, slots=True)
class CollaborationPolicyMutation:
    snapshot: CollaborationPolicySnapshot
    changed: bool
    replayed: bool


@dataclass(frozen=True, slots=True)
class CollaborationActivityEntry:
    event_position: int
    kind: str
    operation: str | None
    outcome: str
    detail: str | None
    occurred_at: str


class WorkshopCollaborationPolicyService:
    """Resolve owner policy and fence active grants through the live authority."""

    def __init__(
        self,
        store: WorkshopEventStore,
        private_execution: WorkshopPrivateTextExecutionService,
    ) -> None:
        self._store = store
        self._private_execution = private_execution
        self._host_policy: CollaborationHostPolicy = private_execution.collaboration_authority.host_policy

    def validate_initial_allowed_operations(
        self,
        requested_operations: object,
        allowed_operations: object,
    ) -> tuple[str, ...]:
        """Validate creation-time owner policy without requiring a definition."""
        try:
            requested = validate_collaboration_operations(requested_operations)
            allowed = validate_collaboration_operations(allowed_operations)
        except ValueError as exc:
            raise WorkshopCollaborationPolicyValidationError(str(exc)) from exc
        if not set(allowed).issubset(requested):
            raise WorkshopCollaborationPolicyValidationError(
                "Owner policy cannot allow operations not requested by Revision 1"
            )
        if any(CollaborationOperation(item) not in self._host_policy.effective_allowed_operations for item in allowed):
            raise WorkshopCollaborationPolicyValidationError("Owner policy cannot exceed host policy")
        return allowed

    async def inspect(
        self,
        principal_id: PrincipalId,
        definition_id: AgentDefinitionId,
    ) -> CollaborationPolicySnapshot:
        workshop_id, owner_id, lifecycle_state, active_revision_id = await self._definition_access(
            principal_id,
            definition_id,
        )
        del workshop_id
        requested: tuple[str, ...] = ()
        if active_revision_id is not None:
            async with self._store.connection.execute(
                "SELECT collaboration_operations_json FROM agent_definition_revisions WHERE id = ?",
                (active_revision_id,),
            ) as cursor:
                revision_row = await cursor.fetchone()
            if revision_row is None:
                raise WorkshopCollaborationPolicyStorageError("Active agent revision is unavailable")
            requested = validate_collaboration_operations(json.loads(str(revision_row[0])))
        async with self._store.connection.execute(
            "SELECT allowed_operations_json, policy_version FROM "
            "agent_collaboration_owner_policies WHERE agent_definition_id = ?",
            (definition_id,),
        ) as cursor:
            policy_row = await cursor.fetchone()
        if policy_row is None:
            allowed = ("agent_delegation",) if "agent_delegation" in requested else ()
            policy_version = 0
        else:
            allowed = validate_collaboration_operations(json.loads(str(policy_row[0])))
            policy_version = int(policy_row[1])
        async with self._store.connection.execute(
            "SELECT COUNT(*) FROM collaboration_grants g JOIN agent_definition_revisions r "
            "ON r.id = g.agent_definition_revision_id WHERE r.agent_definition_id = ? "
            "AND g.revoked_at IS NULL",
            (definition_id,),
        ) as cursor:
            active_row = await cursor.fetchone()
        assert active_row is not None
        can_manage = principal_id == owner_id
        runtime_available = await self._owner_runtime_available(definition_id, owner_id)
        attachment_available = await self._active_attachment_available(definition_id)
        operation_states: list[CollaborationOperationState] = []
        for operation in CollaborationOperation:
            is_requested = operation.value in requested
            is_allowed = operation.value in allowed
            host_allowed = operation in self._host_policy.effective_allowed_operations
            unavailable: str | None = None
            if lifecycle_state != "active":
                unavailable = "Agent is not active"
            elif active_revision_id is None:
                unavailable = "No active revision"
            elif not runtime_available:
                unavailable = "Owner runtime is unavailable"
            elif not attachment_available:
                unavailable = "Agent is not attached to an active conversation"
            elif not host_allowed:
                unavailable = "Host policy does not permit this operation"
            elif not is_requested:
                unavailable = "Active revision does not request this operation"
            elif not is_allowed:
                unavailable = "Owner policy does not allow this operation"
            if not can_manage and unavailable in {
                "Owner runtime is unavailable",
                "Agent is not attached to an active conversation",
                "Owner policy does not allow this operation",
            }:
                unavailable = "Unavailable under current agent policy"
            operation_states.append(
                CollaborationOperationState(
                    operation=operation.value,
                    requested=is_requested,
                    owner_allowed=(is_allowed if can_manage else None),
                    host_allowed=host_allowed,
                    effective_for_new_attempt=unavailable is None,
                    unavailable_reason=unavailable,
                    quota=(self._host_policy.quotas[operation] if host_allowed else None),
                )
            )
        return CollaborationPolicySnapshot(
            definition_id=definition_id,
            owner_principal_id=owner_id,
            can_manage=can_manage,
            policy_version=policy_version,
            active_revision_id=active_revision_id,
            active_grants=(int(active_row[0]) if can_manage else None),
            operations=tuple(operation_states),
        )

    async def set_allowed(
        self,
        principal_id: PrincipalId,
        definition_id: AgentDefinitionId,
        *,
        allowed_operations: object,
        expected_policy_version: object,
        client_operation_id: object,
    ) -> CollaborationPolicyMutation:
        try:
            allowed = validate_collaboration_operations(allowed_operations)
        except ValueError as exc:
            raise WorkshopCollaborationPolicyValidationError(str(exc)) from exc
        if any(CollaborationOperation(item) not in self._host_policy.effective_allowed_operations for item in allowed):
            raise WorkshopCollaborationPolicyValidationError("Owner policy cannot exceed host policy")
        if (
            not isinstance(expected_policy_version, int)
            or isinstance(expected_policy_version, bool)
            or expected_policy_version < 0
        ):
            raise WorkshopCollaborationPolicyValidationError("expected_policy_version must be non-negative")
        key = self._operation_id(client_operation_id)
        workshop_id, owner_id, _lifecycle, _revision = await self._definition_access(principal_id, definition_id)
        if principal_id != owner_id:
            raise WorkshopCollaborationPolicyAccessDenied("Only the agent owner may change collaboration policy")
        request_hash = self._request_hash(
            "set_allowed",
            {
                "definition_id": str(definition_id),
                "expected_policy_version": expected_policy_version,
                "allowed_operations": list(allowed),
            },
        )
        idempotency_key = f"workshop-client:collaboration-policy:{workshop_id}:{principal_id}:{key}"
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            existing = await self._store.event_by_idempotency_key(idempotency_key)
            if existing is not None:
                if existing.envelope.metadata.get("request_hash") != request_hash:
                    raise WorkshopCollaborationPolicyConflict("Operation identity was reused with different content")
                await connection.commit()
                return CollaborationPolicyMutation(
                    await self.inspect(principal_id, definition_id),
                    changed=False,
                    replayed=True,
                )
            async with connection.execute(
                "SELECT policy_version, allowed_operations_json FROM "
                "agent_collaboration_owner_policies WHERE agent_definition_id = ?",
                (definition_id,),
            ) as cursor:
                current = await cursor.fetchone()
            current_version = int(current[0]) if current is not None else 0
            if current_version != expected_policy_version:
                raise WorkshopCollaborationPolicyConflict("Collaboration policy changed; refresh and retry")
            current_allowed = (
                validate_collaboration_operations(json.loads(str(current[1]))) if current is not None else None
            )
            event = EventEnvelope.create(
                event_id=EventId.derived(definition_id, f"collaboration-policy:{key}"),
                event_type=WorkshopEventType.AGENT_DEFINITION_COLLABORATION_POLICY_SET,
                event_version=1,
                workshop_id=workshop_id,
                aggregate_type="agent_definition",
                aggregate_id=definition_id,
                actor_principal_id=principal_id,
                occurred_at=datetime.now(UTC),
                idempotency_key=idempotency_key,
                payload={
                    "allowed_operations": list(allowed),
                    "expected_policy_version": expected_policy_version,
                    "policy_version": expected_policy_version + 1,
                },
                metadata={"source": "workshop_client", "request_hash": request_hash},
            )
            policy_event = await self._store.append_in_transaction(event)
            await self._store.project_pending_in_transaction(CanonicalConversationProjection())
            if "standing_participation" not in allowed:
                await end_active_standings_in_transaction(
                    self._store,
                    definition_id=definition_id,
                    reason="owner_policy_revoked",
                    occurred_at=event.occurred_at,
                    cause_event_id=policy_event.event.envelope.event_id,
                    actor_principal_id=principal_id,
                )
            await connection.commit()
        except WorkshopCollaborationPolicyError:
            await connection.rollback()
            raise
        except IdempotencyConflictError as exc:
            await connection.rollback()
            raise WorkshopCollaborationPolicyConflict("Operation identity conflicted") from exc
        except Exception as exc:
            await connection.rollback()
            raise WorkshopCollaborationPolicyStorageError("Collaboration policy could not be persisted") from exc
        return CollaborationPolicyMutation(
            await self.inspect(principal_id, definition_id),
            changed=current_allowed != allowed,
            replayed=False,
        )

    async def revoke_active(
        self,
        principal_id: PrincipalId,
        definition_id: AgentDefinitionId,
    ) -> tuple[CollaborationPolicySnapshot, int]:
        _workshop_id, owner_id, _lifecycle, _revision = await self._definition_access(
            principal_id,
            definition_id,
        )
        if principal_id != owner_id:
            raise WorkshopCollaborationPolicyAccessDenied("Only the agent owner may revoke collaboration grants")
        revoked = await self._private_execution.revoke_collaboration_for_definition(
            definition_id,
            occurred_at=datetime.now(UTC),
        )
        return await self.inspect(principal_id, definition_id), revoked

    async def activity(
        self,
        principal_id: PrincipalId,
        run_id: RunId,
    ) -> tuple[CollaborationActivityEntry, ...]:
        async with self._store.connection.execute(
            "SELECT r.channel_id FROM runs r JOIN channel_memberships cm ON cm.channel_id = r.channel_id "
            "WHERE r.id = ? AND cm.principal_id = ?",
            (run_id, principal_id),
        ) as cursor:
            if await cursor.fetchone() is None:
                raise WorkshopCollaborationPolicyAccessDenied("Run is unavailable")
        entries: list[CollaborationActivityEntry] = []
        async with self._store.connection.execute(
            "SELECT d.decided_event_position, d.operation, d.decision, d.denial_code, "
            "d.quota_ordinal, d.decided_at FROM collaboration_operation_decisions d "
            "JOIN collaboration_grants g ON g.id = d.grant_id WHERE g.run_id = ?",
            (run_id,),
        ) as cursor:
            for row in await cursor.fetchall():
                detail = (
                    str(row[3]) if row[3] is not None else (f"quota use {int(row[4])}" if row[4] is not None else None)
                )
                entries.append(
                    CollaborationActivityEntry(int(row[0]), "operation", str(row[1]), str(row[2]), detail, str(row[5]))
                )
        async with self._store.connection.execute(
            "SELECT revoked_event_position, revocation_code, revoked_at FROM collaboration_grants "
            "WHERE run_id = ? AND revoked_event_position IS NOT NULL",
            (run_id,),
        ) as cursor:
            for row in await cursor.fetchall():
                entries.append(
                    CollaborationActivityEntry(int(row[0]), "revocation", None, "revoked", str(row[1]), str(row[2]))
                )
        return tuple(sorted(entries, key=lambda item: item.event_position))

    async def _definition_access(
        self,
        principal_id: PrincipalId,
        definition_id: AgentDefinitionId,
    ) -> tuple[WorkshopId, PrincipalId, str, str | None]:
        async with self._store.connection.execute(
            "SELECT d.workshop_id, d.owner_principal_id, d.lifecycle_state, d.active_revision_id "
            "FROM agent_definitions d JOIN workshop_memberships wm ON wm.workshop_id = d.workshop_id "
            "WHERE d.id = ? AND wm.principal_id = ? AND (d.lifecycle_state = 'active' "
            "OR d.owner_principal_id = ?)",
            (definition_id, principal_id, principal_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None or row[1] is None:
            raise WorkshopCollaborationPolicyAccessDenied("Agent collaboration policy is unavailable")
        return WorkshopId(str(row[0])), PrincipalId(str(row[1])), str(row[2]), (str(row[3]) if row[3] else None)

    async def _owner_runtime_available(
        self,
        definition_id: AgentDefinitionId,
        owner_id: PrincipalId,
    ) -> bool:
        async with self._store.connection.execute(
            "SELECT EXISTS(SELECT 1 FROM principal_agent_enablements pae "
            "JOIN agent_definitions d ON d.agent_id = pae.agent_id "
            "WHERE d.id = ? AND pae.principal_id = ? AND pae.lifecycle_state = 'enabled')",
            (definition_id, owner_id),
        ) as cursor:
            row = await cursor.fetchone()
        return bool(row and row[0])

    async def _active_attachment_available(self, definition_id: AgentDefinitionId) -> bool:
        async with self._store.connection.execute(
            "SELECT EXISTS(SELECT 1 FROM channel_agents ca JOIN agent_definitions d "
            "ON d.agent_id = ca.agent_id WHERE d.id = ? AND ca.detached_at IS NULL)",
            (definition_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return bool(row and row[0])

    @staticmethod
    def _operation_id(value: object) -> str:
        if not isinstance(value, str) or not _OPERATION_ID_PATTERN.fullmatch(value):
            raise WorkshopCollaborationPolicyValidationError("client_operation_id is invalid")
        return value

    @staticmethod
    def _request_hash(kind: str, payload: dict[str, object]) -> str:
        encoded = json.dumps(
            {"kind": kind, "payload": payload},
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(encoded.encode()).hexdigest()
