"""Principal-owned activation of reusable Workshop agent definitions."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime

from kai.workshop.domain import (
    AgentDefinitionId,
    AgentEnablementId,
    AgentId,
    ChannelAgentId,
    ChannelId,
    ChannelMembershipId,
    EventEnvelope,
    PrincipalId,
    RuntimeAssignmentId,
    RuntimeProfileId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.execution_state import (
    WorkshopExecutionStateNamespace,
    WorkshopExecutionStateRegistry,
)
from kai.workshop.internal_api_contexts import (
    WorkshopInternalAPIContextRegistry,
    WorkshopInternalAPIExecutionContext,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.runtime_pool import WorkshopRuntimePool
from kai.workshop.runtime_profiles import WorkshopRuntimeProfileRegistry
from kai.workshop.store import IdempotencyConflictError, WorkshopEventStore

_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class WorkshopAgentEnablementError(RuntimeError):
    """A principal-agent enablement operation failed."""


class WorkshopAgentEnablementAccessDenied(WorkshopAgentEnablementError):
    """The acting principal does not own the requested authority."""


class WorkshopAgentEnablementConflict(WorkshopAgentEnablementError):
    """The requested transition conflicts with canonical state."""


class WorkshopAgentEnablementValidationError(WorkshopAgentEnablementError):
    """The requested transition is malformed."""


class WorkshopAgentEnablementStorageError(WorkshopAgentEnablementError):
    """The requested transition could not be persisted."""


@dataclass(frozen=True, slots=True)
class EligibleAgentRuntime:
    runtime_profile_id: RuntimeProfileId
    display_name: str
    backend_options: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class PrincipalAgentEnablement:
    enablement_id: AgentEnablementId | None
    definition_id: AgentDefinitionId
    agent_id: AgentId
    handle: str
    display_name: str
    lifecycle_state: str
    direct_channel_id: ChannelId | None
    runtime_profile_id: RuntimeProfileId | None
    state_version: int | None
    eligible_runtimes: tuple[EligibleAgentRuntime, ...]
    owner_principal_id: PrincipalId | None
    owner_runtime_profile_id: RuntimeProfileId | None
    can_manage: bool
    conversation_started: bool = False


class WorkshopAgentEnablementService:
    """Enable an active agent in an isolated direct lane owned by one human."""

    def __init__(
        self,
        store: WorkshopEventStore,
        runtime_profiles: WorkshopRuntimeProfileRegistry,
        execution_state: WorkshopExecutionStateRegistry,
        internal_api_contexts: WorkshopInternalAPIContextRegistry,
        runtime_pool: WorkshopRuntimePool,
    ) -> None:
        self._store = store
        self._runtime_profiles = runtime_profiles
        self._execution_state = execution_state
        self._internal_api_contexts = internal_api_contexts
        self._runtime_pool = runtime_pool

    async def list_for_principal(
        self,
        principal_id: PrincipalId,
    ) -> tuple[PrincipalAgentEnablement, ...]:
        workshop_id = await self._workshop_for(principal_id)
        eligible = await self._eligible_runtimes(principal_id)
        async with self._store.connection.execute(
            "SELECT d.id FROM agent_definitions d WHERE d.workshop_id = ? "
            "AND d.lifecycle_state = 'active' ORDER BY d.handle",
            (workshop_id,),
        ) as cursor:
            rows = list(await cursor.fetchall())
        return tuple(
            [
                await self._snapshot(
                    principal_id,
                    AgentDefinitionId(str(row[0])),
                    eligible=eligible,
                )
                for row in rows
            ]
        )

    async def inspect(
        self,
        principal_id: PrincipalId,
        definition_id: AgentDefinitionId,
    ) -> PrincipalAgentEnablement:
        await self._workshop_for(principal_id)
        return await self._snapshot(principal_id, definition_id)

    async def enable(
        self,
        principal_id: PrincipalId,
        definition_id: AgentDefinitionId,
        runtime_profile_id: RuntimeProfileId,
        *,
        idempotency_key: object,
        expected_version: object | None = None,
    ) -> PrincipalAgentEnablement:
        key = self._key(idempotency_key)
        workshop_id = await self._workshop_for(principal_id)
        eligible = await self._eligible_runtimes(principal_id)
        if runtime_profile_id not in {item.runtime_profile_id for item in eligible}:
            raise WorkshopAgentEnablementAccessDenied("Runtime profile is not owned by this principal")
        fingerprint = self._fingerprint(
            "enable",
            definition_id,
            runtime_profile_id=runtime_profile_id,
            expected_version=expected_version,
        )
        operation_key = self._operation_key(workshop_id, principal_id, key)
        connection = self._store.connection
        prior_context: WorkshopInternalAPIExecutionContext | None = None
        new_context: WorkshopInternalAPIExecutionContext | None = None
        created = False
        try:
            await connection.execute("BEGIN IMMEDIATE")
            replay = await self._replayed(operation_key, fingerprint, principal_id, definition_id)
            if replay is not None:
                await connection.commit()
                return replay
            definition = await self._active_definition(workshop_id, definition_id)
            owner_principal_id = definition[4]
            current = await self._snapshot(principal_id, definition_id, eligible=eligible)
            self._check_version(current, expected_version)
            now = datetime.now(UTC)
            metadata = {
                "source": "workshop_client",
                "operation": "enable",
                "operation_hash": fingerprint,
                "definition_id": str(definition_id),
            }
            if current.enablement_id is None:
                created = True
                enablement_id = AgentEnablementId.derived(definition_id, f"principal:{principal_id}")
                channel_id = ChannelId.derived(enablement_id, "direct-channel")
                events = self._creation_events(
                    workshop_id,
                    principal_id,
                    definition_id,
                    definition,
                    enablement_id,
                    channel_id,
                    runtime_profile_id,
                    now,
                    operation_key,
                    metadata,
                )
            else:
                enablement_id = current.enablement_id
                assert current.direct_channel_id is not None
                channel_id = current.direct_channel_id
                if current.lifecycle_state == "enabled" and current.runtime_profile_id == runtime_profile_id:
                    raise WorkshopAgentEnablementConflict("Agent is already enabled with this runtime")
                events: list[EventEnvelope] = []
                if current.runtime_profile_id != runtime_profile_id:
                    assert current.runtime_profile_id is not None
                    assignment_id = RuntimeAssignmentId.derived(channel_id, f"runtime-profile:{definition[0]}")
                    events.extend(
                        (
                            EventEnvelope.create(
                                event_type=WorkshopEventType.RUNTIME_PROFILE_REASSIGNED,
                                event_version=1,
                                workshop_id=workshop_id,
                                aggregate_type="runtime_assignment",
                                aggregate_id=assignment_id,
                                actor_principal_id=principal_id,
                                occurred_at=now,
                                idempotency_key=f"{operation_key}:assignment",
                                payload={
                                    "channel_id": channel_id,
                                    "agent_id": definition[0],
                                    "runtime_profile_id": runtime_profile_id,
                                },
                                metadata=metadata,
                            ),
                            EventEnvelope.create(
                                event_type=WorkshopEventType.PRINCIPAL_AGENT_RUNTIME_CHANGED,
                                event_version=1,
                                workshop_id=workshop_id,
                                aggregate_type="agent_enablement",
                                aggregate_id=enablement_id,
                                actor_principal_id=principal_id,
                                occurred_at=now,
                                idempotency_key=f"{operation_key}:runtime",
                                payload={"runtime_profile_id": runtime_profile_id},
                                metadata=metadata,
                            ),
                        )
                    )
                    prior_context = WorkshopInternalAPIExecutionContext(
                        principal_id,
                        channel_id,
                        definition[0],
                        current.runtime_profile_id,
                    )
                events.append(
                    self._enablement_event(
                        workshop_id,
                        principal_id,
                        definition_id,
                        definition[0],
                        enablement_id,
                        channel_id,
                        runtime_profile_id,
                        now,
                        operation_key,
                        metadata,
                    )
                )
            for event in events:
                await self._store.append_in_transaction(event)
            await self._store.project_pending_in_transaction(CanonicalConversationProjection())
            if principal_id == owner_principal_id:
                authority_event = EventEnvelope.create(
                    event_type=WorkshopEventType.AGENT_DEFINITION_AUTHORITY_ASSIGNED,
                    event_version=2,
                    workshop_id=workshop_id,
                    aggregate_type="agent_definition",
                    aggregate_id=definition_id,
                    actor_principal_id=principal_id,
                    occurred_at=now,
                    idempotency_key=(f"{operation_key}:authority:{runtime_profile_id}:{channel_id}"),
                    payload={
                        "owner_principal_id": principal_id,
                        "runtime_profile_id": runtime_profile_id,
                        "owner_direct_channel_id": channel_id,
                    },
                    metadata=metadata,
                )
                await self._store.append_in_transaction(authority_event)
                await self._store.project_pending_in_transaction(CanonicalConversationProjection())
            result = await self._snapshot(principal_id, definition_id, eligible=eligible)
            await connection.commit()
            new_context = WorkshopInternalAPIExecutionContext(
                principal_id,
                channel_id,
                definition[0],
                runtime_profile_id,
            )
        except WorkshopAgentEnablementError:
            await connection.rollback()
            raise
        except IdempotencyConflictError as exc:
            await connection.rollback()
            raise WorkshopAgentEnablementConflict("Idempotency key conflicts with another request") from exc
        except Exception as exc:
            await connection.rollback()
            raise WorkshopAgentEnablementStorageError("Agent enablement could not be persisted") from exc

        assert new_context is not None
        namespace = WorkshopExecutionStateNamespace(
            new_context.principal_id,
            new_context.channel_id,
            new_context.agent_id,
            new_context.runtime_profile_id,
            None,
        )
        if prior_context is not None:
            prior_namespace = self._execution_state.maybe_for_principal_channel(
                principal_id,
                channel_id,
            )
            assert prior_namespace is not None
            await self._runtime_pool.rebind_canonical_lane(prior_context, new_context)
            self._execution_state.replace_lane(prior_namespace, namespace)
            self._internal_api_contexts.replace_context(prior_context, new_context)
        elif created:
            self._execution_state.register_lane(namespace)
            self._internal_api_contexts.register_context(new_context)
            self._runtime_pool.register_canonical_lane(new_context)
        else:
            self._execution_state.register_lane(namespace)
            self._internal_api_contexts.register_context(new_context)
            self._runtime_pool.register_canonical_lane(new_context)
        return result

    async def start_conversation(
        self,
        principal_id: PrincipalId,
        definition_id: AgentDefinitionId,
        *,
        idempotency_key: object,
        expected_version: object,
    ) -> PrincipalAgentEnablement:
        """Durably expose one already-enabled direct conversation."""
        key = self._key(idempotency_key)
        workshop_id = await self._workshop_for(principal_id)
        fingerprint = self._fingerprint(
            "start_conversation",
            definition_id,
            expected_version=expected_version,
        )
        operation_key = self._operation_key(workshop_id, principal_id, key)
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            replay = await self._replayed(operation_key, fingerprint, principal_id, definition_id)
            if replay is not None:
                await connection.commit()
                return replay
            current = await self._snapshot(principal_id, definition_id)
            self._check_version(current, expected_version)
            if current.enablement_id is None or current.lifecycle_state != "enabled":
                raise WorkshopAgentEnablementConflict("Agent is not enabled")
            if current.conversation_started:
                await connection.commit()
                return current
            event = EventEnvelope.create(
                event_type=WorkshopEventType.PRINCIPAL_AGENT_CONVERSATION_STARTED,
                event_version=1,
                workshop_id=workshop_id,
                aggregate_type="agent_enablement",
                aggregate_id=current.enablement_id,
                actor_principal_id=principal_id,
                occurred_at=datetime.now(UTC),
                idempotency_key=operation_key,
                payload={},
                metadata={
                    "source": "workshop_client",
                    "operation": "start_conversation",
                    "operation_hash": fingerprint,
                    "definition_id": str(definition_id),
                },
            )
            await self._store.append_in_transaction(event)
            await self._store.project_pending_in_transaction(CanonicalConversationProjection())
            result = await self._snapshot(principal_id, definition_id)
            await connection.commit()
            return result
        except WorkshopAgentEnablementError:
            await connection.rollback()
            raise
        except IdempotencyConflictError as exc:
            await connection.rollback()
            raise WorkshopAgentEnablementConflict("Idempotency key conflicts with another request") from exc
        except Exception as exc:
            await connection.rollback()
            raise WorkshopAgentEnablementStorageError("Agent conversation could not be started") from exc

    async def suspend_archived_definition(
        self,
        principal_id: PrincipalId,
        definition_id: AgentDefinitionId,
    ) -> None:
        """Revoke every runtime credential retired by one canonical archival."""
        workshop_id = await self._workshop_for(principal_id)
        async with self._store.connection.execute(
            "SELECT owner_principal_id, lifecycle_state FROM agent_definitions WHERE id = ? AND workshop_id = ?",
            (definition_id, workshop_id),
        ) as cursor:
            definition = await cursor.fetchone()
        if (
            definition is None
            or definition[0] is None
            or PrincipalId(str(definition[0])) != principal_id
            or str(definition[1]) != "archived"
        ):
            raise WorkshopAgentEnablementAccessDenied("Access denied")
        async with self._store.connection.execute(
            "SELECT principal_id, direct_channel_id, agent_id, runtime_profile_id "
            "FROM principal_agent_enablements WHERE agent_definition_id = ? "
            "AND lifecycle_state = 'disabled'",
            (definition_id,),
        ) as cursor:
            rows = list(await cursor.fetchall())
        for row in rows:
            context = WorkshopInternalAPIExecutionContext(
                PrincipalId(str(row[0])),
                ChannelId(str(row[1])),
                AgentId(str(row[2])),
                RuntimeProfileId(str(row[3])),
            )
            await self._runtime_pool.suspend_canonical_lane(context)
            namespace = self._execution_state.maybe_for_principal_channel(
                context.principal_id,
                context.channel_id,
            )
            if namespace is not None:
                self._execution_state.unregister_lane(namespace)
            self._internal_api_contexts.unregister_context(context)

    async def disable(
        self,
        principal_id: PrincipalId,
        definition_id: AgentDefinitionId,
        *,
        idempotency_key: object,
        expected_version: object,
    ) -> PrincipalAgentEnablement:
        key = self._key(idempotency_key)
        workshop_id = await self._workshop_for(principal_id)
        fingerprint = self._fingerprint("disable", definition_id, expected_version=expected_version)
        operation_key = self._operation_key(workshop_id, principal_id, key)
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            replay = await self._replayed(operation_key, fingerprint, principal_id, definition_id)
            if replay is not None:
                await connection.commit()
                return replay
            current = await self._snapshot(principal_id, definition_id)
            self._check_version(current, expected_version)
            if current.enablement_id is None or current.lifecycle_state != "enabled":
                raise WorkshopAgentEnablementConflict("Agent is not enabled")
            if current.can_manage:
                raise WorkshopAgentEnablementConflict(
                    "The owner runtime cannot be disabled while the agent definition is active"
                )
            event = EventEnvelope.create(
                event_type=WorkshopEventType.PRINCIPAL_AGENT_DISABLED,
                event_version=1,
                workshop_id=workshop_id,
                aggregate_type="agent_enablement",
                aggregate_id=current.enablement_id,
                actor_principal_id=principal_id,
                occurred_at=datetime.now(UTC),
                idempotency_key=operation_key,
                payload={},
                metadata={
                    "source": "workshop_client",
                    "operation": "disable",
                    "operation_hash": fingerprint,
                    "definition_id": str(definition_id),
                },
            )
            await self._store.append_in_transaction(event)
            await self._store.project_pending_in_transaction(CanonicalConversationProjection())
            result = await self._snapshot(principal_id, definition_id)
            await connection.commit()
        except WorkshopAgentEnablementError:
            await connection.rollback()
            raise
        except IdempotencyConflictError as exc:
            await connection.rollback()
            raise WorkshopAgentEnablementConflict("Idempotency key conflicts with another request") from exc
        except Exception as exc:
            await connection.rollback()
            raise WorkshopAgentEnablementStorageError("Agent disablement could not be persisted") from exc
        assert current.direct_channel_id is not None
        assert current.runtime_profile_id is not None
        context = WorkshopInternalAPIExecutionContext(
            principal_id,
            current.direct_channel_id,
            current.agent_id,
            current.runtime_profile_id,
        )
        await self._runtime_pool.suspend_canonical_lane(context)
        namespace = self._execution_state.maybe_for_principal_channel(
            principal_id,
            current.direct_channel_id,
        )
        if namespace is not None:
            self._execution_state.unregister_lane(namespace)
        self._internal_api_contexts.unregister_context(context)
        return result

    async def _workshop_for(self, principal_id: PrincipalId) -> WorkshopId:
        if not isinstance(principal_id, PrincipalId):
            raise WorkshopAgentEnablementAccessDenied("Access denied")
        async with self._store.connection.execute(
            "SELECT wm.workshop_id FROM workshop_memberships wm JOIN principals p "
            "ON p.id = wm.principal_id AND p.kind = 'human' WHERE wm.principal_id = ?",
            (principal_id,),
        ) as cursor:
            rows = list(await cursor.fetchall())
        if len(rows) != 1:
            raise WorkshopAgentEnablementAccessDenied("Access denied")
        return WorkshopId(str(rows[0][0]))

    async def _eligible_runtimes(
        self,
        principal_id: PrincipalId,
    ) -> tuple[EligibleAgentRuntime, ...]:
        async with self._store.connection.execute(
            "SELECT ra.runtime_profile_id FROM channel_agent_runtime_assignments ra "
            "JOIN channels c ON c.id = ra.channel_id AND c.kind = 'direct' "
            "JOIN channel_memberships cm ON cm.channel_id = c.id AND cm.role = 'owner' "
            "JOIN principals p ON p.id = cm.principal_id AND p.kind = 'human' "
            "GROUP BY ra.runtime_profile_id HAVING COUNT(DISTINCT cm.principal_id) = 1 "
            "AND MIN(cm.principal_id) = ? ORDER BY ra.runtime_profile_id",
            (principal_id,),
        ) as cursor:
            rows = list(await cursor.fetchall())
        result: list[EligibleAgentRuntime] = []
        for row in rows:
            try:
                profile = self._runtime_profiles.resolve(RuntimeProfileId(str(row[0])))
            except Exception:
                continue
            result.append(
                EligibleAgentRuntime(
                    profile.profile_id,
                    profile.display_name,
                    tuple(option.option_id for option in profile.backend_options),
                )
            )
        return tuple(result)

    async def _active_definition(
        self,
        workshop_id: WorkshopId,
        definition_id: AgentDefinitionId,
    ) -> tuple[
        AgentId,
        PrincipalId,
        str,
        str,
        PrincipalId,
        RuntimeProfileId | None,
        ChannelId | None,
    ]:
        async with self._store.connection.execute(
            "SELECT d.agent_id, a.principal_id, d.handle, d.display_name, "
            "d.owner_principal_id, d.owner_runtime_profile_id, d.owner_direct_channel_id "
            "FROM agent_definitions d JOIN agents a ON a.id = d.agent_id "
            "WHERE d.id = ? AND d.workshop_id = ? AND d.lifecycle_state = 'active' "
            "AND d.active_revision_id IS NOT NULL",
            (definition_id, workshop_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None or row[4] is None:
            raise WorkshopAgentEnablementAccessDenied("Active agent definition is unavailable")
        return (
            AgentId(str(row[0])),
            PrincipalId(str(row[1])),
            str(row[2]),
            str(row[3]),
            PrincipalId(str(row[4])),
            RuntimeProfileId(str(row[5])) if row[5] is not None else None,
            ChannelId(str(row[6])) if row[6] is not None else None,
        )

    async def _snapshot(
        self,
        principal_id: PrincipalId,
        definition_id: AgentDefinitionId,
        *,
        eligible: tuple[EligibleAgentRuntime, ...] | None = None,
    ) -> PrincipalAgentEnablement:
        workshop_id = await self._workshop_for(principal_id)
        async with self._store.connection.execute(
            "SELECT d.agent_id, d.handle, d.display_name, d.lifecycle_state, e.id, "
            "e.direct_channel_id, e.runtime_profile_id, e.lifecycle_state, e.last_event_position, "
            "d.owner_principal_id, d.owner_runtime_profile_id, "
            "e.conversation_started_at "
            "FROM agent_definitions d LEFT JOIN principal_agent_enablements e "
            "ON e.agent_definition_id = d.id AND e.principal_id = ? "
            "WHERE d.id = ? AND d.workshop_id = ?",
            (principal_id, definition_id, workshop_id),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None or str(row[3]) != "active":
            raise WorkshopAgentEnablementAccessDenied("Active agent definition is unavailable")
        owner_principal_id = PrincipalId(str(row[9])) if row[9] is not None else None
        can_manage = owner_principal_id == principal_id
        return PrincipalAgentEnablement(
            AgentEnablementId(str(row[4])) if row[4] is not None else None,
            definition_id,
            AgentId(str(row[0])),
            str(row[1]),
            str(row[2]),
            str(row[7]) if row[7] is not None else "available",
            ChannelId(str(row[5])) if row[5] is not None else None,
            RuntimeProfileId(str(row[6])) if row[6] is not None else None,
            int(row[8]) if row[8] is not None else None,
            eligible if eligible is not None else await self._eligible_runtimes(principal_id),
            owner_principal_id,
            RuntimeProfileId(str(row[10])) if row[10] is not None else None,
            can_manage,
            row[11] is not None,
        )

    def _creation_events(
        self,
        workshop_id: WorkshopId,
        principal_id: PrincipalId,
        definition_id: AgentDefinitionId,
        definition: tuple[
            AgentId,
            PrincipalId,
            str,
            str,
            PrincipalId,
            RuntimeProfileId | None,
            ChannelId | None,
        ],
        enablement_id: AgentEnablementId,
        channel_id: ChannelId,
        runtime_profile_id: RuntimeProfileId,
        now: datetime,
        operation_key: str,
        metadata: dict[str, str],
    ) -> list[EventEnvelope]:
        agent_id, agent_principal_id, _handle, display_name, _owner, _owner_runtime, _owner_channel = definition
        assignment_id = RuntimeAssignmentId.derived(channel_id, f"runtime-profile:{agent_id}")
        return [
            EventEnvelope.create(
                event_type=WorkshopEventType.CHANNEL_CREATED,
                event_version=1,
                workshop_id=workshop_id,
                aggregate_type="channel",
                aggregate_id=channel_id,
                actor_principal_id=principal_id,
                occurred_at=now,
                idempotency_key=f"{operation_key}:channel",
                payload={"kind": "direct", "name": display_name},
                metadata=metadata,
            ),
            EventEnvelope.create(
                event_type=WorkshopEventType.CHANNEL_MEMBER_ADDED,
                event_version=1,
                workshop_id=workshop_id,
                aggregate_type="channel_membership",
                aggregate_id=ChannelMembershipId.derived(channel_id, f"human:{principal_id}"),
                actor_principal_id=principal_id,
                occurred_at=now,
                idempotency_key=f"{operation_key}:human",
                payload={"channel_id": channel_id, "principal_id": principal_id, "role": "owner"},
                metadata=metadata,
            ),
            EventEnvelope.create(
                event_type=WorkshopEventType.CHANNEL_MEMBER_ADDED,
                event_version=1,
                workshop_id=workshop_id,
                aggregate_type="channel_membership",
                aggregate_id=ChannelMembershipId.derived(channel_id, f"agent:{agent_principal_id}"),
                actor_principal_id=principal_id,
                occurred_at=now,
                idempotency_key=f"{operation_key}:agent-member",
                payload={
                    "channel_id": channel_id,
                    "principal_id": agent_principal_id,
                    "role": "participant",
                },
                metadata=metadata,
            ),
            EventEnvelope.create(
                event_type=WorkshopEventType.CHANNEL_AGENT_ATTACHED,
                event_version=1,
                workshop_id=workshop_id,
                aggregate_type="channel_agent",
                aggregate_id=ChannelAgentId.derived(channel_id, f"agent:{agent_id}"),
                actor_principal_id=principal_id,
                occurred_at=now,
                idempotency_key=f"{operation_key}:attachment",
                payload={"channel_id": channel_id, "agent_id": agent_id},
                metadata=metadata,
            ),
            EventEnvelope.create(
                event_type=WorkshopEventType.RUNTIME_PROFILE_ASSIGNED,
                event_version=1,
                workshop_id=workshop_id,
                aggregate_type="runtime_assignment",
                aggregate_id=assignment_id,
                actor_principal_id=principal_id,
                occurred_at=now,
                idempotency_key=f"{operation_key}:assignment",
                payload={
                    "channel_id": channel_id,
                    "agent_id": agent_id,
                    "runtime_profile_id": runtime_profile_id,
                },
                metadata=metadata,
            ),
            self._enablement_event(
                workshop_id,
                principal_id,
                definition_id,
                agent_id,
                enablement_id,
                channel_id,
                runtime_profile_id,
                now,
                operation_key,
                metadata,
            ),
        ]

    @staticmethod
    def _enablement_event(
        workshop_id: WorkshopId,
        principal_id: PrincipalId,
        definition_id: AgentDefinitionId,
        agent_id: AgentId,
        enablement_id: AgentEnablementId,
        channel_id: ChannelId,
        runtime_profile_id: RuntimeProfileId,
        now: datetime,
        operation_key: str,
        metadata: dict[str, str],
    ) -> EventEnvelope:
        return EventEnvelope.create(
            event_type=WorkshopEventType.PRINCIPAL_AGENT_ENABLED,
            event_version=1,
            workshop_id=workshop_id,
            aggregate_type="agent_enablement",
            aggregate_id=enablement_id,
            actor_principal_id=principal_id,
            occurred_at=now,
            idempotency_key=operation_key,
            payload={
                "principal_id": principal_id,
                "agent_definition_id": definition_id,
                "agent_id": agent_id,
                "direct_channel_id": channel_id,
                "runtime_profile_id": runtime_profile_id,
            },
            metadata=metadata,
        )

    async def _replayed(
        self,
        operation_key: str,
        fingerprint: str,
        principal_id: PrincipalId,
        definition_id: AgentDefinitionId,
    ) -> PrincipalAgentEnablement | None:
        event = await self._store.event_by_idempotency_key(operation_key)
        if event is None:
            return None
        if event.envelope.metadata.get("operation_hash") != fingerprint or event.envelope.metadata.get(
            "definition_id"
        ) != str(definition_id):
            raise WorkshopAgentEnablementConflict("Idempotency key conflicts with another request")
        return await self._snapshot(principal_id, definition_id)

    @staticmethod
    def _check_version(snapshot: PrincipalAgentEnablement, expected: object | None) -> None:
        if snapshot.enablement_id is None:
            if expected is not None:
                raise WorkshopAgentEnablementConflict("Agent enablement version changed")
            return
        if not isinstance(expected, int) or isinstance(expected, bool) or expected != snapshot.state_version:
            raise WorkshopAgentEnablementConflict("Agent enablement version changed")

    @staticmethod
    def _key(value: object) -> str:
        if not isinstance(value, str) or not _KEY_PATTERN.fullmatch(value):
            raise WorkshopAgentEnablementValidationError("idempotency_key is invalid")
        return value

    @staticmethod
    def _operation_key(workshop_id: WorkshopId, principal_id: PrincipalId, key: str) -> str:
        digest = hashlib.sha256(f"{workshop_id}:{principal_id}:{key}".encode()).hexdigest()
        return f"agent-enablement:{digest}"

    @staticmethod
    def _fingerprint(
        operation: str,
        definition_id: AgentDefinitionId,
        **payload: object,
    ) -> str:
        encoded = json.dumps(
            {"operation": operation, "definition_id": str(definition_id), **payload},
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        return hashlib.sha256(encoded.encode()).hexdigest()
