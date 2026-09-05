"""Bounded, visible, canonical agent-to-agent delegation."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from kai.workshop.agent_definitions import normalize_agent_handle
from kai.workshop.collaboration_authority import (
    CollaborationAuthorization,
    CollaborationBaseIdentity,
    CollaborationDenied,
    CollaborationOperation,
    CollaborationProofError,
)
from kai.workshop.domain import (
    AgentDefinitionRevisionId,
    AgentDelegationId,
    AgentId,
    ChannelId,
    EventEnvelope,
    EventId,
    MessageId,
    PrincipalId,
    RunId,
    RuntimeProfileId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.execution_coordinator import (
    CanonicalCancellationDisposition,
    CanonicalExecutionDisposition,
    CanonicalExecutionResult,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.run_lifecycle import DurableRun, RunStatus, WorkshopRunLifecycle
from kai.workshop.store import WorkshopEventStore

if TYPE_CHECKING:
    from kai.workshop.private_text_execution import WorkshopPrivateTextExecutionService

log = logging.getLogger(__name__)

_IDEMPOTENCY_KEY_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_RECONCILE_INTERVAL_SECONDS = 2.0
_MAX_TASK_CHARACTERS = 6_000
_MAX_CONTEXT_SUMMARY_CHARACTERS = 4_000
_MAX_CONTEXT_REFERENCES = 12


class AgentDelegationError(RuntimeError):
    """A structured agent delegation could not be accepted or completed."""


class AgentDelegationDenied(AgentDelegationError):
    """Server-owned delegation policy denied a request."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class AgentDelegationConflict(AgentDelegationError):
    """An idempotent delegation identity was reused with different facts."""


@dataclass(frozen=True, slots=True)
class AgentDelegationPolicy:
    """Server-owned hard limits; callers cannot override these values."""

    max_depth: int = 3
    max_fan_out: int = 3
    max_turns_per_pair: int = 4
    max_total_runs: int = 12
    max_elapsed: timedelta = timedelta(minutes=10)

    def __post_init__(self) -> None:
        for name in ("max_depth", "max_fan_out", "max_turns_per_pair", "max_total_runs"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_elapsed <= timedelta(0):
            raise ValueError("max_elapsed must be positive")


@dataclass(frozen=True, slots=True)
class AgentDelegationContext:
    summary: str = ""
    message_ids: tuple[MessageId, ...] = ()

    def as_json(self) -> dict[str, object]:
        return {
            "summary": self.summary,
            "message_ids": [str(message_id) for message_id in self.message_ids],
        }


@dataclass(frozen=True, slots=True)
class AgentDelegationSnapshot:
    delegation_id: AgentDelegationId
    workshop_id: WorkshopId
    channel_id: ChannelId
    thread_root_id: MessageId | None
    root_run_id: RunId
    parent_run_id: RunId
    parent_delegation_id: AgentDelegationId | None
    child_run_id: RunId
    requesting_principal_id: PrincipalId
    caller_agent_id: AgentId
    target_agent_id: AgentId
    target_handle: str
    target_sponsor_principal_id: PrincipalId
    target_runtime_profile_id: RuntimeProfileId
    request_message_id: MessageId
    response_message_id: MessageId | None
    task: str
    context: AgentDelegationContext
    request_hash: str
    depth: int
    status: str
    outcome_code: str | None
    created_at: datetime
    started_at: datetime | None
    terminal_at: datetime | None


@dataclass(frozen=True, slots=True)
class AgentDelegationResult:
    delegation: AgentDelegationSnapshot
    response: str | None


@dataclass(frozen=True, slots=True)
class _ParentAuthority:
    run: DurableRun
    workshop_id: WorkshopId
    caller_principal_id: PrincipalId
    caller_revision_id: AgentDefinitionRevisionId
    thread_root_id: MessageId | None
    root_run_id: RunId
    parent_delegation_id: AgentDelegationId | None
    depth: int


@dataclass(frozen=True, slots=True)
class _TargetAuthority:
    agent_id: AgentId
    principal_id: PrincipalId
    revision_id: AgentDefinitionRevisionId
    sponsor_principal_id: PrincipalId
    runtime_profile_id: RuntimeProfileId
    handle: str


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

    async def execute(self, run_id: RunId) -> CanonicalExecutionResult: ...

    async def run_state(self, run_id: RunId) -> DurableRun: ...

    async def request_run_cancellation(
        self,
        run_id: RunId,
    ) -> CanonicalCancellationDisposition: ...


def _timestamp(value: object) -> datetime:
    return datetime.fromisoformat(str(value).replace("Z", "+00:00")).astimezone(UTC)


def _optional_timestamp(value: object) -> datetime | None:
    return None if value is None else _timestamp(value)


def _normalize_task(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AgentDelegationDenied("invalid_task", "Delegation task must contain text")
    task = value.strip()
    if len(task) > _MAX_TASK_CHARACTERS:
        raise AgentDelegationDenied(
            "task_too_large",
            f"Delegation task must be at most {_MAX_TASK_CHARACTERS} characters",
        )
    return task


def _normalize_context(value: object) -> AgentDelegationContext:
    if value is None:
        return AgentDelegationContext()
    if not isinstance(value, dict) or set(value) - {"summary", "message_ids"}:
        raise AgentDelegationDenied(
            "invalid_context",
            "Delegation context accepts only summary and message_ids",
        )
    raw_summary = value.get("summary", "")
    if not isinstance(raw_summary, str):
        raise AgentDelegationDenied("invalid_context", "Delegation context summary must be text")
    summary = raw_summary.strip()
    if len(summary) > _MAX_CONTEXT_SUMMARY_CHARACTERS:
        raise AgentDelegationDenied(
            "context_too_large",
            f"Delegation context summary must be at most {_MAX_CONTEXT_SUMMARY_CHARACTERS} characters",
        )
    raw_ids = value.get("message_ids", [])
    if not isinstance(raw_ids, list) or len(raw_ids) > _MAX_CONTEXT_REFERENCES:
        raise AgentDelegationDenied(
            "invalid_context",
            f"Delegation context may reference at most {_MAX_CONTEXT_REFERENCES} messages",
        )
    try:
        message_ids = tuple(MessageId(item) for item in raw_ids)
    except (TypeError, ValueError) as exc:
        raise AgentDelegationDenied(
            "invalid_context",
            "Delegation context message_ids must be canonical message IDs",
        ) from exc
    if len(set(message_ids)) != len(message_ids):
        raise AgentDelegationDenied("invalid_context", "Delegation context message_ids must be unique")
    return AgentDelegationContext(summary, message_ids)


def _normalize_idempotency_key(value: object) -> str:
    if not isinstance(value, str) or not _IDEMPOTENCY_KEY_PATTERN.fullmatch(value):
        raise AgentDelegationDenied(
            "invalid_idempotency_key",
            "idempotency_key must be a bounded opaque identifier",
        )
    return value


def _request_hash(target_handle: str, task: str, context: AgentDelegationContext) -> str:
    encoded = json.dumps(
        {
            "target_handle": target_handle,
            "task": task,
            "context": context.as_json(),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


class WorkshopAgentDelegationService:
    """Accept, execute, recover, and settle explicit delegation trees."""

    def __init__(
        self,
        store: WorkshopEventStore,
        execution: WorkshopPrivateTextExecutionService,
        *,
        policy: AgentDelegationPolicy | None = None,
        owns_store: bool = False,
    ) -> None:
        self._store = store
        self._execution: _ExecutionService = execution
        self._policy = policy or AgentDelegationPolicy()
        self._owns_store = owns_store
        self._lock = asyncio.Lock()
        self._task_lock = asyncio.Lock()
        self._tasks: dict[AgentDelegationId, asyncio.Task[None]] = {}
        self._stop_event = asyncio.Event()
        self._reconcile_task: asyncio.Task[None] | None = None
        self._closed = False

    @classmethod
    async def open_and_start(
        cls,
        database_path: Path,
        execution: WorkshopPrivateTextExecutionService,
        *,
        policy: AgentDelegationPolicy | None = None,
    ) -> WorkshopAgentDelegationService:
        """Open an isolated database connection and start reconciliation."""
        store = await WorkshopEventStore.open(database_path)
        service = cls(store, execution, policy=policy, owns_store=True)
        try:
            await service.start()
        except BaseException:
            await store.close()
            raise
        return service

    @property
    def ready(self) -> bool:
        task = self._reconcile_task
        return not self._closed and task is not None and not task.done()

    async def start(self) -> None:
        if self._closed or self._reconcile_task is not None:
            raise RuntimeError("Workshop delegation service cannot be started")
        await self._reconcile()
        self._reconcile_task = asyncio.create_task(
            self._reconcile_loop(),
            name="kai-workshop-agent-delegation-reconciliation",
        )

    async def delegate(
        self,
        base_identity: CollaborationBaseIdentity,
        *,
        proof: str,
        target_handle: object,
        task: object,
        context: object = None,
        idempotency_key: object,
    ) -> AgentDelegationResult:
        if not self.ready:
            raise AgentDelegationError("Workshop delegation service is unavailable")
        try:
            handle = normalize_agent_handle(target_handle)
        except ValueError as exc:
            raise AgentDelegationDenied("invalid_target", str(exc)) from exc
        normalized_task = _normalize_task(task)
        normalized_context = _normalize_context(context)
        key = _normalize_idempotency_key(idempotency_key)
        fingerprint = _request_hash(handle, normalized_task, normalized_context)
        try:
            authorization = await self._execution.authorize_collaboration(
                proof,
                CollaborationOperation.AGENT_DELEGATION,
                base_identity=base_identity,
                idempotency_key=key,
                request_hash=fingerprint,
                occurred_at=datetime.now(UTC),
            )
        except CollaborationProofError as exc:
            raise AgentDelegationDenied("invalid_proof", str(exc)) from exc
        except CollaborationDenied as exc:
            raise AgentDelegationDenied(exc.code, str(exc)) from exc
        async with self._lock:
            delegation = await self._accept(
                authorization,
                target_handle=handle,
                task=normalized_task,
                context=normalized_context,
                idempotency_key=key,
            )
        await self._schedule(delegation.delegation_id)
        return await self._wait_for_terminal(delegation.delegation_id)

    async def snapshot(self, delegation_id: AgentDelegationId) -> AgentDelegationSnapshot:
        async with self._store.connection.execute(
            "SELECT d.id, d.workshop_id, d.channel_id, d.thread_root_id, d.root_run_id, "
            "d.parent_run_id, d.parent_delegation_id, d.child_run_id, "
            "d.requesting_principal_id, d.caller_agent_id, d.target_agent_id, ad.handle, "
            "d.target_sponsor_principal_id, d.target_runtime_profile_id, d.request_message_id, "
            "d.response_message_id, d.task, d.context_json, d.request_hash, d.depth, d.status, "
            "d.outcome_code, d.created_at, d.started_at, d.terminal_at "
            "FROM agent_delegations d JOIN agent_definitions ad ON ad.agent_id = d.target_agent_id "
            "WHERE d.id = ?",
            (delegation_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise AgentDelegationError("Delegation was not found")
        raw_context = json.loads(str(row[17]))
        context = _normalize_context(raw_context)
        request_hash = str(row[18])
        if not _SHA256_PATTERN.fullmatch(request_hash):
            raise AgentDelegationError("Delegation request hash is malformed")
        return AgentDelegationSnapshot(
            delegation_id=AgentDelegationId(str(row[0])),
            workshop_id=WorkshopId(str(row[1])),
            channel_id=ChannelId(str(row[2])),
            thread_root_id=MessageId(str(row[3])) if row[3] is not None else None,
            root_run_id=RunId(str(row[4])),
            parent_run_id=RunId(str(row[5])),
            parent_delegation_id=AgentDelegationId(str(row[6])) if row[6] is not None else None,
            child_run_id=RunId(str(row[7])),
            requesting_principal_id=PrincipalId(str(row[8])),
            caller_agent_id=AgentId(str(row[9])),
            target_agent_id=AgentId(str(row[10])),
            target_handle=str(row[11]),
            target_sponsor_principal_id=PrincipalId(str(row[12])),
            target_runtime_profile_id=RuntimeProfileId(str(row[13])),
            request_message_id=MessageId(str(row[14])),
            response_message_id=MessageId(str(row[15])) if row[15] is not None else None,
            task=str(row[16]),
            context=context,
            request_hash=request_hash,
            depth=int(row[19]),
            status=str(row[20]),
            outcome_code=str(row[21]) if row[21] is not None else None,
            created_at=_timestamp(row[22]),
            started_at=_optional_timestamp(row[23]),
            terminal_at=_optional_timestamp(row[24]),
        )

    async def stop(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        reconcile = self._reconcile_task
        if reconcile is not None:
            await asyncio.shield(reconcile)
        async with self._store.connection.execute(
            "SELECT child_run_id FROM agent_delegations WHERE status IN ('requested', 'executing')"
        ) as cursor:
            active_runs = tuple(RunId(str(row[0])) for row in await cursor.fetchall())
        for run_id in active_runs:
            try:
                await self._execution.request_run_cancellation(run_id)
            except Exception:
                log.exception("Could not cancel delegated run %s during shutdown", run_id)
        async with self._task_lock:
            tasks = tuple(self._tasks.values())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._reconcile_task = None
        if self._owns_store:
            await self._store.close()

    async def wait(self) -> None:
        task = self._reconcile_task
        if task is None:
            raise RuntimeError("Workshop delegation service was not started")
        await asyncio.shield(task)

    async def _accept(
        self,
        authorization: CollaborationAuthorization,
        *,
        target_handle: str,
        task: str,
        context: AgentDelegationContext,
        idempotency_key: str,
    ) -> AgentDelegationSnapshot:
        if not isinstance(authorization, CollaborationAuthorization):
            raise AgentDelegationDenied("invalid_authority", "Delegation authority is unavailable")
        now = datetime.now(UTC)
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            projection = CanonicalConversationProjection()
            await self._store.project_pending_in_transaction(projection)
            parent = await self._active_parent(authorization)
            delegation_id = AgentDelegationId.derived(
                parent.workshop_id,
                f"agent-delegation:{parent.run.run_id}:{idempotency_key}",
            )
            fingerprint = _request_hash(target_handle, task, context)
            existing = await self._store.event_by_idempotency_key(
                f"workshop-agent-delegation:v1:{delegation_id}:requested"
            )
            if existing is not None:
                await self._store.project_pending_in_transaction(projection)
                replay = await self.snapshot(delegation_id)
                if replay.request_hash != fingerprint:
                    raise AgentDelegationConflict("Delegation idempotency key was reused with different content")
                await connection.commit()
                return replay
            target = await self._target(parent, target_handle)
            await self._validate_policy(parent, target, fingerprint, context, now)
            message_id = MessageId.derived(delegation_id, "request-message")
            child_run_id = RunId.derived(delegation_id, "child-run")
            body = self._request_body(parent, target, task, context)
            mention_start = body.index(f"@{target.handle}")
            message_event = EventEnvelope.create(
                event_id=EventId.derived(delegation_id, "request-message"),
                event_type=WorkshopEventType.MESSAGE_CREATED,
                event_version=1,
                workshop_id=parent.workshop_id,
                aggregate_type="message",
                aggregate_id=message_id,
                actor_principal_id=parent.caller_principal_id,
                occurred_at=now,
                idempotency_key=f"workshop-agent-delegation:v1:{delegation_id}:message",
                payload={
                    "channel_id": parent.run.channel_id,
                    "author_principal_id": parent.caller_principal_id,
                    "reply_to_message_id": parent.run.inbound_message_id,
                    "body": body,
                    "mentions": [
                        {
                            "principal_id": target.principal_id,
                            "kind": "agent",
                            "start": mention_start,
                            "length": len(target.handle) + 1,
                        }
                    ],
                    **({"thread_root_id": parent.thread_root_id} if parent.thread_root_id is not None else {}),
                },
                metadata={
                    "source": "agent_delegation",
                    "delegation_id": str(delegation_id),
                    "parent_run_id": str(parent.run.run_id),
                },
            )
            run_event = EventEnvelope.create(
                event_id=EventId.derived(child_run_id, "accepted"),
                event_type=WorkshopEventType.RUN_ACCEPTED,
                event_version=4,
                workshop_id=parent.workshop_id,
                aggregate_type="run",
                aggregate_id=child_run_id,
                actor_principal_id=parent.caller_principal_id,
                occurred_at=now,
                idempotency_key=f"workshop-run:v1:{child_run_id}:accepted",
                payload={
                    "inbound_message_id": message_id,
                    "channel_id": parent.run.channel_id,
                    "requested_by_principal_id": parent.run.requested_by_principal_id,
                    "agent_id": target.agent_id,
                    "agent_definition_revision_id": target.revision_id,
                    "runtime_profile_id": target.runtime_profile_id,
                    "sponsor_principal_id": target.sponsor_principal_id,
                    "parent_run_id": parent.run.run_id,
                    "delegation_id": delegation_id,
                },
                metadata={"source": "agent_delegation", "delegation_id": str(delegation_id)},
            )
            delegation_event = EventEnvelope.create(
                event_id=EventId.derived(delegation_id, "requested"),
                event_type=WorkshopEventType.AGENT_DELEGATION_REQUESTED,
                event_version=1,
                workshop_id=parent.workshop_id,
                aggregate_type="agent_delegation",
                aggregate_id=delegation_id,
                actor_principal_id=parent.caller_principal_id,
                occurred_at=now,
                idempotency_key=f"workshop-agent-delegation:v1:{delegation_id}:requested",
                payload={
                    "channel_id": parent.run.channel_id,
                    "thread_root_id": parent.thread_root_id,
                    "root_run_id": parent.root_run_id,
                    "parent_run_id": parent.run.run_id,
                    "parent_delegation_id": parent.parent_delegation_id,
                    "child_run_id": child_run_id,
                    "requesting_principal_id": parent.run.requested_by_principal_id,
                    "caller_agent_id": parent.run.agent_id,
                    "target_agent_id": target.agent_id,
                    "caller_sponsor_principal_id": authorization.grant.sponsor_principal_id,
                    "caller_runtime_profile_id": authorization.grant.runtime_profile_id,
                    "target_sponsor_principal_id": target.sponsor_principal_id,
                    "target_runtime_profile_id": target.runtime_profile_id,
                    "caller_definition_revision_id": parent.caller_revision_id,
                    "target_definition_revision_id": target.revision_id,
                    "request_message_id": message_id,
                    "task": task,
                    "context": context.as_json(),
                    "request_hash": fingerprint,
                    "depth": parent.depth,
                },
                metadata={"source": "internal_api", "protocol": "bounded_delegation_v1"},
            )
            await self._store.append_in_transaction(message_event)
            await self._store.append_in_transaction(run_event)
            await self._store.append_in_transaction(delegation_event)
            await self._store.project_pending_in_transaction(projection)
            accepted = await self.snapshot(delegation_id)
            await connection.commit()
            return accepted
        except Exception:
            await connection.rollback()
            raise

    async def _active_parent(self, authorization: CollaborationAuthorization) -> _ParentAuthority:
        grant = authorization.grant
        async with self._store.connection.execute(
            "SELECT kind FROM channels WHERE id = ? AND archived_at IS NULL",
            (grant.channel_id,),
        ) as cursor:
            channel_row = await cursor.fetchone()
        if channel_row is None or str(channel_row[0]) != "group":
            raise AgentDelegationDenied(
                "shared_channel_required",
                "Agent delegation is available only in a shared group channel",
            )
        run = await WorkshopRunLifecycle(self._store).state(grant.run_id)
        if (
            run.status != RunStatus.STARTED
            or run.cancellation_requested_at is not None
            or run.channel_id != grant.channel_id
            or run.agent_id != grant.agent_id
            or run.runtime_profile_id != grant.runtime_profile_id
            or run.sponsor_principal_id != grant.sponsor_principal_id
            or run.agent_definition_revision_id != grant.agent_definition_revision_id
        ):
            raise AgentDelegationDenied(
                "no_active_attempt",
                "Delegation requires its exact active caller attempt",
            )
        assert run.agent_definition_revision_id is not None
        root_run_id = run.run_id
        parent_delegation_id = run.delegation_id
        depth = 1
        if parent_delegation_id is not None:
            async with self._store.connection.execute(
                "SELECT root_run_id, depth FROM agent_delegations WHERE id = ?",
                (parent_delegation_id,),
            ) as cursor:
                parent_row = await cursor.fetchone()
            if parent_row is None:
                raise AgentDelegationDenied(
                    "invalid_lineage",
                    "Caller delegation lineage is unavailable",
                )
            root_run_id = RunId(str(parent_row[0]))
            depth = int(parent_row[1]) + 1
        return _ParentAuthority(
            run=run,
            workshop_id=run.workshop_id,
            caller_principal_id=grant.agent_principal_id,
            caller_revision_id=run.agent_definition_revision_id,
            thread_root_id=grant.thread_root_id,
            root_run_id=root_run_id,
            parent_delegation_id=parent_delegation_id,
            depth=depth,
        )

    async def _target(
        self,
        parent: _ParentAuthority,
        target_handle: str,
    ) -> _TargetAuthority:
        async with self._store.connection.execute(
            "SELECT a.id, a.principal_id, d.active_revision_id, ca.sponsor_principal_id, "
            "ca.sponsored_runtime_profile_id, d.handle "
            "FROM agent_definitions d JOIN agents a ON a.id = d.agent_id "
            "JOIN channel_agents ca ON ca.agent_id = a.id AND ca.channel_id = ? "
            "AND ca.detached_at IS NULL "
            "JOIN channel_memberships cm ON cm.channel_id = ca.channel_id "
            "AND cm.principal_id = a.principal_id "
            "JOIN channel_agent_runtime_assignments ra ON ra.channel_id = ca.channel_id "
            "AND ra.agent_id = ca.agent_id AND ra.runtime_profile_id = ca.sponsored_runtime_profile_id "
            "JOIN principal_agent_enablements enabled ON enabled.principal_id = ca.sponsor_principal_id "
            "AND enabled.agent_id = ca.agent_id "
            "AND enabled.runtime_profile_id = ca.sponsored_runtime_profile_id "
            "AND enabled.lifecycle_state = 'enabled' "
            "WHERE d.workshop_id = ? AND d.handle = ? COLLATE NOCASE "
            "AND d.lifecycle_state = 'active' AND d.active_revision_id IS NOT NULL",
            (parent.run.channel_id, parent.workshop_id, target_handle),
        ) as cursor:
            rows = list(await cursor.fetchall())
        if len(rows) != 1 or rows[0][3] is None or rows[0][4] is None:
            raise AgentDelegationDenied(
                "target_unavailable",
                "The requested target is not active, attached, sponsored, and runnable",
            )
        row = rows[0]
        return _TargetAuthority(
            agent_id=AgentId(str(row[0])),
            principal_id=PrincipalId(str(row[1])),
            revision_id=AgentDefinitionRevisionId(str(row[2])),
            sponsor_principal_id=PrincipalId(str(row[3])),
            runtime_profile_id=RuntimeProfileId(str(row[4])),
            handle=str(row[5]),
        )

    async def _validate_policy(
        self,
        parent: _ParentAuthority,
        target: _TargetAuthority,
        fingerprint: str,
        context: AgentDelegationContext,
        now: datetime,
    ) -> None:
        if parent.depth > self._policy.max_depth:
            raise AgentDelegationDenied("max_depth", "Delegation maximum depth was reached")
        async with self._store.connection.execute(
            "WITH RECURSIVE ancestry(id, agent_id, parent_run_id) AS ("
            "SELECT id, agent_id, parent_run_id FROM runs WHERE id = ? "
            "UNION ALL SELECT parent.id, parent.agent_id, parent.parent_run_id "
            "FROM runs parent JOIN ancestry child ON parent.id = child.parent_run_id) "
            "SELECT agent_id FROM ancestry",
            (parent.run.run_id,),
        ) as cursor:
            ancestor_agents = {AgentId(str(row[0])) for row in await cursor.fetchall()}
        if target.agent_id in ancestor_agents:
            raise AgentDelegationDenied("cycle", "Delegation cycles are not permitted")
        async with self._store.connection.execute(
            "SELECT accepted_at FROM runs WHERE id = ?",
            (parent.root_run_id,),
        ) as cursor:
            root_row = await cursor.fetchone()
        if root_row is None or now - _timestamp(root_row[0]) > self._policy.max_elapsed:
            raise AgentDelegationDenied("elapsed_budget", "Delegation elapsed-time budget was exhausted")
        async with self._store.connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT target_agent_id) FROM agent_delegations WHERE parent_run_id = ?",
            (parent.run.run_id,),
        ) as cursor:
            parent_counts = await cursor.fetchone()
        assert parent_counts is not None
        existing_targets = int(parent_counts[1])
        async with self._store.connection.execute(
            "SELECT 1 FROM agent_delegations WHERE parent_run_id = ? AND target_agent_id = ? LIMIT 1",
            (parent.run.run_id, target.agent_id),
        ) as cursor:
            target_seen = await cursor.fetchone() is not None
        if existing_targets + (0 if target_seen else 1) > self._policy.max_fan_out:
            raise AgentDelegationDenied("max_fan_out", "Delegation fan-out budget was exhausted")
        async with self._store.connection.execute(
            "SELECT COUNT(*) FROM agent_delegations WHERE root_run_id = ?",
            (parent.root_run_id,),
        ) as cursor:
            total_row = await cursor.fetchone()
        if total_row is None or int(total_row[0]) >= self._policy.max_total_runs:
            raise AgentDelegationDenied("execution_budget", "Delegation execution budget was exhausted")
        async with self._store.connection.execute(
            "SELECT COUNT(*) FROM agent_delegations WHERE root_run_id = ? "
            "AND caller_agent_id = ? AND target_agent_id = ?",
            (parent.root_run_id, parent.run.agent_id, target.agent_id),
        ) as cursor:
            pair_row = await cursor.fetchone()
        if pair_row is None or int(pair_row[0]) >= self._policy.max_turns_per_pair:
            raise AgentDelegationDenied("max_turns", "Delegation dialogue turn budget was exhausted")
        async with self._store.connection.execute(
            "SELECT 1 FROM agent_delegations WHERE root_run_id = ? AND request_hash = ? LIMIT 1",
            (parent.root_run_id, fingerprint),
        ) as cursor:
            if await cursor.fetchone() is not None:
                raise AgentDelegationDenied("repeated_handoff", "Repeated delegation hand-offs are not permitted")
        if context.message_ids:
            placeholders = ",".join("?" for _ in context.message_ids)
            async with self._store.connection.execute(
                f"SELECT id FROM messages WHERE channel_id = ? AND id IN ({placeholders})",
                (parent.run.channel_id, *context.message_ids),
            ) as cursor:
                found = {MessageId(str(row[0])) for row in await cursor.fetchall()}
            if found != set(context.message_ids):
                raise AgentDelegationDenied(
                    "context_out_of_scope",
                    "Delegation context may reference only messages in the shared channel",
                )

    @staticmethod
    def _request_body(
        parent: _ParentAuthority,
        target: _TargetAuthority,
        task: str,
        context: AgentDelegationContext,
    ) -> str:
        lines = [
            f"Delegation request from the current agent to @{target.handle}.",
            "",
            "Task:",
            task,
        ]
        if context.summary:
            lines.extend(("", "Bounded shared context:", context.summary))
        if context.message_ids:
            lines.extend(
                (
                    "",
                    "Canonical shared-channel references:",
                    ", ".join(str(message_id) for message_id in context.message_ids),
                )
            )
        lines.extend(
            (
                "",
                f"Parent run: {parent.run.run_id}",
                "Return the requested result to the calling agent. Do not expose credentials or private memory.",
            )
        )
        return "\n".join(lines)

    async def _schedule(self, delegation_id: AgentDelegationId) -> None:
        async with self._task_lock:
            if self._closed or delegation_id in self._tasks:
                return
            task = asyncio.create_task(
                self._execute(delegation_id),
                name=f"kai-workshop-agent-delegation:{delegation_id}",
            )
            self._tasks[delegation_id] = task

    async def _execute(self, delegation_id: AgentDelegationId) -> None:
        task = asyncio.current_task()
        try:
            delegation = await self.snapshot(delegation_id)
            child = await self._execution.run_state(delegation.child_run_id)
            if child.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
                await self._settle(delegation, child)
                return
            await self._mark_started(delegation)
            watchdog = asyncio.create_task(
                self._cancel_at_deadline(delegation),
                name=f"kai-workshop-agent-delegation-deadline:{delegation_id}",
            )
            try:
                result = await self._execution.execute(delegation.child_run_id)
            finally:
                watchdog.cancel()
                await asyncio.gather(watchdog, return_exceptions=True)
            if result.disposition != CanonicalExecutionDisposition.PREPARATION_DEFERRED:
                await self._settle(delegation, result.run)
        except Exception:
            log.exception("Workshop delegated run failed for %s", delegation_id)
        finally:
            async with self._task_lock:
                if self._tasks.get(delegation_id) is task:
                    del self._tasks[delegation_id]

    async def _mark_started(self, delegation: AgentDelegationSnapshot) -> None:
        if delegation.status != "requested":
            return
        await self._append_transition(
            delegation,
            WorkshopEventType.AGENT_DELEGATION_STARTED,
            {"child_run_id": delegation.child_run_id},
            "started",
        )

    async def _settle(self, delegation: AgentDelegationSnapshot, run: DurableRun) -> None:
        current = await self.snapshot(delegation.delegation_id)
        if current.status in {"completed", "failed", "cancelled"}:
            return
        event_type = {
            RunStatus.COMPLETED: WorkshopEventType.AGENT_DELEGATION_COMPLETED,
            RunStatus.FAILED: WorkshopEventType.AGENT_DELEGATION_FAILED,
            RunStatus.CANCELLED: WorkshopEventType.AGENT_DELEGATION_CANCELLED,
        }.get(run.status)
        if event_type is None:
            return
        outcome_code = run.terminal_code or run.status.value
        await self._append_transition(
            current,
            event_type,
            {
                "child_run_id": current.child_run_id,
                "outcome_code": outcome_code,
                "response_message_id": run.result_message_id,
            },
            run.status.value,
        )

    async def _append_transition(
        self,
        delegation: AgentDelegationSnapshot,
        event_type: WorkshopEventType,
        payload: dict[str, object],
        stable_status: str,
    ) -> None:
        async with self._store.connection.execute(
            "SELECT a.principal_id FROM agents a WHERE a.id = ?",
            (delegation.target_agent_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise AgentDelegationError("Delegation target principal is unavailable")
        actor = PrincipalId(str(row[0]))
        envelope = EventEnvelope.create(
            event_id=EventId.derived(delegation.delegation_id, stable_status),
            event_type=event_type,
            event_version=1,
            workshop_id=delegation.workshop_id,
            aggregate_type="agent_delegation",
            aggregate_id=delegation.delegation_id,
            actor_principal_id=actor,
            occurred_at=datetime.now(UTC),
            idempotency_key=(f"workshop-agent-delegation:v1:{delegation.delegation_id}:{stable_status}"),
            payload=payload,
            metadata={"source": "agent_delegation_worker"},
        )
        async with self._lock:
            connection = self._store.connection
            try:
                await connection.execute("BEGIN IMMEDIATE")
                await self._store.append_in_transaction(envelope)
                await self._store.project_pending_in_transaction(CanonicalConversationProjection())
                await connection.commit()
            except Exception:
                await connection.rollback()
                raise

    async def _cancel_at_deadline(self, delegation: AgentDelegationSnapshot) -> None:
        async with self._store.connection.execute(
            "SELECT accepted_at FROM runs WHERE id = ?",
            (delegation.root_run_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return
        remaining = self._policy.max_elapsed.total_seconds() - (datetime.now(UTC) - _timestamp(row[0])).total_seconds()
        if remaining > 0:
            await asyncio.sleep(remaining)
        await self._execution.request_run_cancellation(delegation.child_run_id)

    async def _wait_for_terminal(self, delegation_id: AgentDelegationId) -> AgentDelegationResult:
        while True:
            delegation = await self.snapshot(delegation_id)
            if delegation.status in {"completed", "failed", "cancelled"}:
                response = None
                if delegation.response_message_id is not None:
                    async with self._store.connection.execute(
                        "SELECT body FROM messages WHERE id = ?",
                        (delegation.response_message_id,),
                    ) as cursor:
                        row = await cursor.fetchone()
                    response = None if row is None else str(row[0])
                return AgentDelegationResult(delegation, response)
            async with self._task_lock:
                running = self._tasks.get(delegation_id)
            if running is None:
                await self._schedule(delegation_id)
            await asyncio.sleep(0.2)

    async def _reconcile(self) -> None:
        async with self._store.connection.execute(
            "SELECT id, child_run_id FROM agent_delegations "
            "WHERE status IN ('requested', 'executing') ORDER BY created_event_position"
        ) as cursor:
            rows = list(await cursor.fetchall())
        for row in rows:
            delegation_id = AgentDelegationId(str(row[0]))
            run = await self._execution.run_state(RunId(str(row[1])))
            if run.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
                await self._settle(await self.snapshot(delegation_id), run)
            elif run.status == RunStatus.ACCEPTED:
                await self._schedule(delegation_id)

    async def _reconcile_loop(self) -> None:
        while True:
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=_RECONCILE_INTERVAL_SECONDS,
                )
                return
            except TimeoutError:
                pass
            try:
                await self._reconcile()
            except Exception:
                log.exception("Workshop agent delegation reconciliation failed")
