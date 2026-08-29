"""Admin-authorized lifecycle operations for Workshop agent definitions."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from kai.workshop.agent_definitions import (
    MAX_AGENT_DESCRIPTION,
    MAX_AGENT_DISPLAY_NAME,
    MAX_AGENT_INSTRUCTIONS,
    MAX_AGENT_PURPOSE,
    normalize_agent_handle,
    validate_agent_capabilities,
    validate_agent_presentation,
    validate_agent_text,
)
from kai.workshop.domain import (
    AgentDefinitionId,
    AgentDefinitionRevisionId,
    AgentId,
    EventEnvelope,
    PrincipalId,
    WorkshopEventType,
    WorkshopId,
    WorkshopMembershipId,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import IdempotencyConflictError, WorkshopEventStore

_IDEMPOTENCY_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class WorkshopAgentLifecycleError(RuntimeError):
    """An agent-definition lifecycle operation could not be completed."""


class WorkshopAgentLifecycleAccessDenied(WorkshopAgentLifecycleError):
    """The principal may not access the requested agent definition."""


class WorkshopAgentLifecycleValidationError(WorkshopAgentLifecycleError):
    """A lifecycle request is malformed."""


class WorkshopAgentLifecycleConflict(WorkshopAgentLifecycleError):
    """A lifecycle request conflicts with current canonical state."""


class WorkshopAgentLifecycleStorageError(WorkshopAgentLifecycleError):
    """A lifecycle request could not be persisted."""


@dataclass(frozen=True, slots=True)
class AgentRevisionSnapshot:
    revision_id: AgentDefinitionRevisionId
    revision_number: int
    purpose: str
    instructions: str
    capabilities: tuple[str, ...]
    created_at: str
    created_by_principal_id: PrincipalId | None
    event_position: int


@dataclass(frozen=True, slots=True)
class AgentDefinitionSnapshot:
    definition_id: AgentDefinitionId
    workshop_id: WorkshopId
    agent_id: AgentId
    handle: str
    display_name: str
    description: str
    presentation: dict[str, object]
    lifecycle_state: str
    active_revision_id: AgentDefinitionRevisionId | None
    state_version: int
    created_at: str
    created_by_principal_id: PrincipalId | None
    revisions: tuple[AgentRevisionSnapshot, ...]


@dataclass(frozen=True, slots=True)
class AgentLifecycleAuthority:
    principal_id: PrincipalId
    workshop_id: WorkshopId
    role: str


def _normalize_idempotency_key(value: object) -> str:
    if not isinstance(value, str) or not _IDEMPOTENCY_KEY_PATTERN.fullmatch(value):
        raise WorkshopAgentLifecycleValidationError(
            "idempotency_key must be 1-128 letters, digits, dots, underscores, colons, or hyphens"
        )
    return value


def _normalize_expected_version(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise WorkshopAgentLifecycleValidationError("expected_version must be a positive integer")
    return value


def _operation_hash(kind: str, payload: dict[str, object]) -> str:
    encoded = json.dumps(
        {"kind": kind, "payload": payload},
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


class WorkshopAgentLifecycleService:
    """Manage versioned agent definitions from canonical Workshop authority."""

    def __init__(self, store: WorkshopEventStore) -> None:
        self._store = store

    async def authority_for(self, principal_id: PrincipalId) -> AgentLifecycleAuthority:
        workshop_id, role = await self._authority(principal_id)
        return AgentLifecycleAuthority(principal_id, workshop_id, role)

    async def list_visible(self, principal_id: PrincipalId) -> tuple[AgentDefinitionSnapshot, ...]:
        workshop_id, role = await self._authority(principal_id)
        lifecycle_clause = "" if role == "admin" else "AND d.lifecycle_state = 'active'"
        async with self._store.connection.execute(
            f"SELECT d.id FROM agent_definitions d WHERE d.workshop_id = ? {lifecycle_clause} ORDER BY d.handle",
            (workshop_id,),
        ) as cursor:
            rows = list(await cursor.fetchall())
        snapshots: list[AgentDefinitionSnapshot] = []
        for row in rows:
            snapshots.append(await self._snapshot(AgentDefinitionId(str(row[0])), workshop_id=workshop_id))
        return tuple(snapshots)

    async def get_visible(
        self,
        principal_id: PrincipalId,
        definition_id: AgentDefinitionId,
    ) -> AgentDefinitionSnapshot:
        workshop_id, role = await self._authority(principal_id)
        snapshot = await self._snapshot_or_denied(definition_id, workshop_id)
        if role != "admin" and snapshot.lifecycle_state != "active":
            raise WorkshopAgentLifecycleAccessDenied("Access denied")
        return snapshot

    async def create_draft(
        self,
        principal_id: PrincipalId,
        *,
        idempotency_key: object,
        handle: object,
        display_name: object,
        description: object,
        presentation: object,
        purpose: object,
        instructions: object,
        capabilities: object,
    ) -> AgentDefinitionSnapshot:
        workshop_id = await self._admin_workshop(principal_id)
        key = _normalize_idempotency_key(idempotency_key)
        try:
            normalized_handle = normalize_agent_handle(handle)
            normalized_display_name = validate_agent_text(
                display_name, field="display_name", maximum=MAX_AGENT_DISPLAY_NAME
            )
            normalized_description = validate_agent_text(
                description,
                field="description",
                maximum=MAX_AGENT_DESCRIPTION,
                allow_empty=True,
            )
            presentation_json = validate_agent_presentation(presentation)
            normalized_purpose = validate_agent_text(purpose, field="purpose", maximum=MAX_AGENT_PURPOSE)
            normalized_instructions = validate_agent_text(
                instructions, field="instructions", maximum=MAX_AGENT_INSTRUCTIONS
            )
            normalized_capabilities = validate_agent_capabilities(capabilities)
        except ValueError as exc:
            raise WorkshopAgentLifecycleValidationError(str(exc)) from exc
        operation_payload: dict[str, object] = {
            "handle": normalized_handle,
            "display_name": normalized_display_name,
            "description": normalized_description,
            "presentation": json.loads(presentation_json),
            "purpose": normalized_purpose,
            "instructions": normalized_instructions,
            "capabilities": list(normalized_capabilities),
        }
        fingerprint = _operation_hash("create", operation_payload)
        operation_key = self._operation_key(workshop_id, principal_id, key)
        definition_id = AgentDefinitionId.derived(workshop_id, f"agent-definition-operation:{principal_id}:{key}")
        agent_id = AgentId.derived(definition_id, "agent")
        agent_principal_id = PrincipalId.derived(agent_id, "principal")
        revision_id = AgentDefinitionRevisionId.derived(definition_id, "revision:1")
        now = datetime.now(UTC)
        metadata = {
            "source": "workshop_client",
            "operation": "create",
            "operation_hash": fingerprint,
            "definition_id": str(definition_id),
        }
        events = (
            EventEnvelope.create(
                event_type=WorkshopEventType.PRINCIPAL_CREATED,
                event_version=1,
                workshop_id=workshop_id,
                aggregate_type="principal",
                aggregate_id=agent_principal_id,
                actor_principal_id=principal_id,
                occurred_at=now,
                idempotency_key=operation_key,
                payload={"kind": "agent", "display_name": normalized_display_name},
                metadata=metadata,
            ),
            EventEnvelope.create(
                event_type=WorkshopEventType.WORKSHOP_MEMBER_ADDED,
                event_version=1,
                workshop_id=workshop_id,
                aggregate_type="workshop_membership",
                aggregate_id=WorkshopMembershipId.derived(agent_id, "membership"),
                actor_principal_id=principal_id,
                occurred_at=now,
                idempotency_key=f"{operation_key}:membership",
                payload={"principal_id": agent_principal_id, "role": "agent"},
                metadata=metadata,
            ),
            EventEnvelope.create(
                event_type=WorkshopEventType.AGENT_CREATED,
                event_version=1,
                workshop_id=workshop_id,
                aggregate_type="agent",
                aggregate_id=agent_id,
                actor_principal_id=principal_id,
                occurred_at=now,
                idempotency_key=f"{operation_key}:agent",
                payload={"principal_id": agent_principal_id, "name": normalized_display_name},
                metadata=metadata,
            ),
            EventEnvelope.create(
                event_type=WorkshopEventType.AGENT_DEFINITION_CREATED,
                event_version=1,
                workshop_id=workshop_id,
                aggregate_type="agent_definition",
                aggregate_id=definition_id,
                actor_principal_id=principal_id,
                occurred_at=now,
                idempotency_key=f"{operation_key}:definition",
                payload={
                    "agent_id": agent_id,
                    "handle": normalized_handle,
                    "display_name": normalized_display_name,
                    "description": normalized_description,
                    "presentation": json.loads(presentation_json),
                    "lifecycle_state": "draft",
                },
                metadata=metadata,
            ),
            EventEnvelope.create(
                event_type=WorkshopEventType.AGENT_DEFINITION_REVISION_ADDED,
                event_version=1,
                workshop_id=workshop_id,
                aggregate_type="agent_definition_revision",
                aggregate_id=revision_id,
                actor_principal_id=principal_id,
                occurred_at=now,
                idempotency_key=f"{operation_key}:revision",
                payload={
                    "definition_id": definition_id,
                    "revision_number": 1,
                    "purpose": normalized_purpose,
                    "instructions": normalized_instructions,
                    "capabilities": list(normalized_capabilities),
                },
                metadata=metadata,
            ),
        )
        return await self._mutate(
            principal_id,
            workshop_id,
            definition_id,
            operation_key,
            fingerprint,
            events,
            exclusive_handle=normalized_handle,
        )

    async def add_revision(
        self,
        principal_id: PrincipalId,
        definition_id: AgentDefinitionId,
        *,
        idempotency_key: object,
        expected_version: object,
        purpose: object,
        instructions: object,
        capabilities: object,
    ) -> AgentDefinitionSnapshot:
        workshop_id = await self._admin_workshop(principal_id)
        key = _normalize_idempotency_key(idempotency_key)
        expected = _normalize_expected_version(expected_version)
        try:
            normalized_purpose = validate_agent_text(purpose, field="purpose", maximum=MAX_AGENT_PURPOSE)
            normalized_instructions = validate_agent_text(
                instructions, field="instructions", maximum=MAX_AGENT_INSTRUCTIONS
            )
            normalized_capabilities = validate_agent_capabilities(capabilities)
        except ValueError as exc:
            raise WorkshopAgentLifecycleValidationError(str(exc)) from exc
        operation_payload: dict[str, object] = {
            "definition_id": str(definition_id),
            "expected_version": expected,
            "purpose": normalized_purpose,
            "instructions": normalized_instructions,
            "capabilities": list(normalized_capabilities),
        }
        fingerprint = _operation_hash("revise", operation_payload)
        operation_key = self._operation_key(workshop_id, principal_id, key)
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            replay = await self._replayed(operation_key, fingerprint, definition_id, workshop_id)
            if replay is not None:
                await connection.commit()
                return replay
            current = await self._snapshot_or_denied(definition_id, workshop_id)
            self._require_version(current, expected)
            if current.lifecycle_state == "archived":
                raise WorkshopAgentLifecycleConflict("Archived agent definitions cannot be revised")
            revision_number = len(current.revisions) + 1
            revision_id = AgentDefinitionRevisionId.derived(definition_id, f"revision:{revision_number}")
            event = EventEnvelope.create(
                event_type=WorkshopEventType.AGENT_DEFINITION_REVISION_ADDED,
                event_version=1,
                workshop_id=workshop_id,
                aggregate_type="agent_definition_revision",
                aggregate_id=revision_id,
                actor_principal_id=principal_id,
                occurred_at=datetime.now(UTC),
                idempotency_key=operation_key,
                payload={
                    "definition_id": definition_id,
                    "revision_number": revision_number,
                    "purpose": normalized_purpose,
                    "instructions": normalized_instructions,
                    "capabilities": list(normalized_capabilities),
                },
                metadata=self._metadata("revise", fingerprint, definition_id),
            )
            await self._store.append_in_transaction(event)
            await self._store.project_pending_in_transaction(CanonicalConversationProjection())
            result = await self._snapshot(definition_id, workshop_id=workshop_id)
            await connection.commit()
            return result
        except WorkshopAgentLifecycleError:
            await connection.rollback()
            raise
        except IdempotencyConflictError as exc:
            await connection.rollback()
            raise WorkshopAgentLifecycleConflict("Idempotency key conflicts with another request") from exc
        except Exception as exc:
            await connection.rollback()
            raise WorkshopAgentLifecycleStorageError("Agent revision could not be persisted") from exc

    async def activate_revision(
        self,
        principal_id: PrincipalId,
        definition_id: AgentDefinitionId,
        *,
        revision_id: AgentDefinitionRevisionId,
        idempotency_key: object,
        expected_version: object,
    ) -> AgentDefinitionSnapshot:
        return await self._transition(
            principal_id,
            definition_id,
            operation="activate",
            event_type=WorkshopEventType.AGENT_DEFINITION_REVISION_ACTIVATED,
            payload={"revision_id": revision_id},
            idempotency_key=idempotency_key,
            expected_version=expected_version,
        )

    async def archive(
        self,
        principal_id: PrincipalId,
        definition_id: AgentDefinitionId,
        *,
        idempotency_key: object,
        expected_version: object,
    ) -> AgentDefinitionSnapshot:
        return await self._transition(
            principal_id,
            definition_id,
            operation="archive",
            event_type=WorkshopEventType.AGENT_DEFINITION_ARCHIVED,
            payload={},
            idempotency_key=idempotency_key,
            expected_version=expected_version,
        )

    async def _transition(
        self,
        principal_id: PrincipalId,
        definition_id: AgentDefinitionId,
        *,
        operation: str,
        event_type: WorkshopEventType,
        payload: dict[str, object],
        idempotency_key: object,
        expected_version: object,
    ) -> AgentDefinitionSnapshot:
        workshop_id = await self._admin_workshop(principal_id)
        key = _normalize_idempotency_key(idempotency_key)
        expected = _normalize_expected_version(expected_version)
        operation_payload = {
            "definition_id": str(definition_id),
            "expected_version": expected,
            **{key: str(value) for key, value in payload.items()},
        }
        fingerprint = _operation_hash(operation, operation_payload)
        operation_key = self._operation_key(workshop_id, principal_id, key)
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            replay = await self._replayed(operation_key, fingerprint, definition_id, workshop_id)
            if replay is not None:
                await connection.commit()
                return replay
            current = await self._snapshot_or_denied(definition_id, workshop_id)
            self._require_version(current, expected)
            if current.lifecycle_state == "archived":
                raise WorkshopAgentLifecycleConflict("Archived agent definitions cannot be changed")
            if operation == "activate":
                revision_id = payload["revision_id"]
                if revision_id not in {revision.revision_id for revision in current.revisions}:
                    raise WorkshopAgentLifecycleValidationError("Revision is not part of this agent definition")
            event = EventEnvelope.create(
                event_type=event_type,
                event_version=1,
                workshop_id=workshop_id,
                aggregate_type="agent_definition",
                aggregate_id=definition_id,
                actor_principal_id=principal_id,
                occurred_at=datetime.now(UTC),
                idempotency_key=operation_key,
                payload=payload,
                metadata=self._metadata(operation, fingerprint, definition_id),
            )
            await self._store.append_in_transaction(event)
            await self._store.project_pending_in_transaction(CanonicalConversationProjection())
            result = await self._snapshot(definition_id, workshop_id=workshop_id)
            await connection.commit()
            return result
        except WorkshopAgentLifecycleError:
            await connection.rollback()
            raise
        except IdempotencyConflictError as exc:
            await connection.rollback()
            raise WorkshopAgentLifecycleConflict("Idempotency key conflicts with another request") from exc
        except Exception as exc:
            await connection.rollback()
            raise WorkshopAgentLifecycleStorageError("Agent lifecycle change could not be persisted") from exc

    async def _mutate(
        self,
        principal_id: PrincipalId,
        workshop_id: WorkshopId,
        definition_id: AgentDefinitionId,
        operation_key: str,
        fingerprint: str,
        events: tuple[EventEnvelope, ...],
        exclusive_handle: str | None = None,
    ) -> AgentDefinitionSnapshot:
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            replay = await self._replayed(operation_key, fingerprint, definition_id, workshop_id)
            if replay is not None:
                await connection.commit()
                return replay
            if exclusive_handle is not None:
                async with connection.execute(
                    "SELECT 1 FROM agent_definitions WHERE workshop_id = ? AND handle = ? COLLATE NOCASE LIMIT 1",
                    (workshop_id, exclusive_handle),
                ) as cursor:
                    if await cursor.fetchone() is not None:
                        raise WorkshopAgentLifecycleConflict("Agent handle is already in use")
            for event in events:
                await self._store.append_in_transaction(event)
            await self._store.project_pending_in_transaction(CanonicalConversationProjection())
            result = await self._snapshot(definition_id, workshop_id=workshop_id)
            await connection.commit()
            return result
        except WorkshopAgentLifecycleError:
            await connection.rollback()
            raise
        except IdempotencyConflictError as exc:
            await connection.rollback()
            raise WorkshopAgentLifecycleConflict("Idempotency key conflicts with another request") from exc
        except Exception as exc:
            await connection.rollback()
            raise WorkshopAgentLifecycleStorageError("Agent definition could not be persisted") from exc

    async def _authority(self, principal_id: PrincipalId) -> tuple[WorkshopId, str]:
        if not isinstance(principal_id, PrincipalId):
            raise WorkshopAgentLifecycleAccessDenied("Access denied")
        async with self._store.connection.execute(
            "SELECT wm.workshop_id, wm.role FROM workshop_memberships wm "
            "JOIN principals p ON p.id = wm.principal_id AND p.kind = 'human' "
            "WHERE wm.principal_id = ? ORDER BY wm.workshop_id",
            (principal_id,),
        ) as cursor:
            rows = list(await cursor.fetchall())
        if len(rows) != 1:
            raise WorkshopAgentLifecycleAccessDenied("Access denied")
        return WorkshopId(str(rows[0][0])), str(rows[0][1])

    async def _admin_workshop(self, principal_id: PrincipalId) -> WorkshopId:
        workshop_id, role = await self._authority(principal_id)
        if role != "admin":
            raise WorkshopAgentLifecycleAccessDenied("Administrator access required")
        return workshop_id

    async def _snapshot_or_denied(
        self,
        definition_id: AgentDefinitionId,
        workshop_id: WorkshopId,
    ) -> AgentDefinitionSnapshot:
        try:
            return await self._snapshot(definition_id, workshop_id=workshop_id)
        except WorkshopAgentLifecycleAccessDenied:
            raise

    async def _snapshot(
        self,
        definition_id: AgentDefinitionId,
        *,
        workshop_id: WorkshopId,
    ) -> AgentDefinitionSnapshot:
        async with self._store.connection.execute(
            "SELECT d.id, d.workshop_id, d.agent_id, d.handle, d.display_name, d.description, "
            "presentation_json, lifecycle_state, active_revision_id, created_at, "
            "created_event_position, e.actor_principal_id FROM agent_definitions d "
            "JOIN event_log e ON e.position = d.created_event_position "
            "WHERE d.id = ? AND d.workshop_id = ?",
            (definition_id, workshop_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise WorkshopAgentLifecycleAccessDenied("Access denied")
        async with self._store.connection.execute(
            "SELECT r.id, r.revision_number, r.purpose, r.instructions, r.capabilities_json, "
            "r.created_at, e.actor_principal_id, r.created_event_position "
            "FROM agent_definition_revisions r JOIN event_log e "
            "ON e.position = r.created_event_position WHERE r.agent_definition_id = ? "
            "ORDER BY r.revision_number",
            (definition_id,),
        ) as cursor:
            revision_rows = list(await cursor.fetchall())
        revisions = tuple(
            AgentRevisionSnapshot(
                revision_id=AgentDefinitionRevisionId(str(revision[0])),
                revision_number=int(revision[1]),
                purpose=str(revision[2]),
                instructions=str(revision[3]),
                capabilities=validate_agent_capabilities(json.loads(str(revision[4]))),
                created_at=str(revision[5]),
                created_by_principal_id=(PrincipalId(str(revision[6])) if revision[6] is not None else None),
                event_position=int(revision[7]),
            )
            for revision in revision_rows
        )
        async with self._store.connection.execute(
            "SELECT COALESCE(MAX(position), ?) FROM event_log WHERE workshop_id = ? "
            "AND ((aggregate_type = 'agent_definition' AND aggregate_id = ?) "
            "OR (aggregate_type = 'agent_definition_revision' "
            "AND json_extract(payload_json, '$.definition_id') = ?))",
            (int(row[10]), workshop_id, definition_id, definition_id),
        ) as cursor:
            version_row = await cursor.fetchone()
        assert version_row is not None
        active_revision = row[8]
        return AgentDefinitionSnapshot(
            definition_id=AgentDefinitionId(str(row[0])),
            workshop_id=WorkshopId(str(row[1])),
            agent_id=AgentId(str(row[2])),
            handle=str(row[3]),
            display_name=str(row[4]),
            description=str(row[5]),
            presentation=dict(json.loads(str(row[6]))),
            lifecycle_state=str(row[7]),
            active_revision_id=(AgentDefinitionRevisionId(str(active_revision)) if active_revision else None),
            state_version=int(version_row[0]),
            created_at=str(row[9]),
            created_by_principal_id=(PrincipalId(str(row[11])) if row[11] is not None else None),
            revisions=revisions,
        )

    async def _replayed(
        self,
        operation_key: str,
        fingerprint: str,
        definition_id: AgentDefinitionId,
        workshop_id: WorkshopId,
    ) -> AgentDefinitionSnapshot | None:
        existing = await self._store.event_by_idempotency_key(operation_key)
        if existing is None:
            return None
        metadata = existing.envelope.metadata
        if metadata.get("operation_hash") != fingerprint or metadata.get("definition_id") != str(definition_id):
            raise WorkshopAgentLifecycleConflict("Idempotency key conflicts with another request")
        return await self._snapshot(definition_id, workshop_id=workshop_id)

    @staticmethod
    def _require_version(snapshot: AgentDefinitionSnapshot, expected: int) -> None:
        if snapshot.state_version != expected:
            raise WorkshopAgentLifecycleConflict("Agent definition changed; refresh and retry")

    @staticmethod
    def _metadata(operation: str, fingerprint: str, definition_id: AgentDefinitionId) -> dict[str, object]:
        return {
            "source": "workshop_client",
            "operation": operation,
            "operation_hash": fingerprint,
            "definition_id": str(definition_id),
        }

    @staticmethod
    def _operation_key(workshop_id: WorkshopId, principal_id: PrincipalId, key: str) -> str:
        return f"workshop-client:agent-lifecycle:{workshop_id}:{principal_id}:{key}"
