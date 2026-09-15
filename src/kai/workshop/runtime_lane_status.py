"""Canonical, adapter-neutral status for one executable Workshop lane."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime

from kai.workshop.authorization import CanonicalChannelAuthorizer
from kai.workshop.domain import AgentId, ChannelId, PrincipalId, RunId, RuntimeProfileId
from kai.workshop.internal_api_contexts import WorkshopInternalAPIExecutionContext
from kai.workshop.run_lifecycle import DurableRun, load_durable_run
from kai.workshop.runtime_pool import WorkshopRuntimePool
from kai.workshop.runtime_sessions import (
    CanonicalRuntimeSession,
    ProviderSessionResetConflictError,
    ProviderSessionResetResult,
    load_provider_session_reset_operation,
    load_provider_session_reset_state,
    load_runtime_session,
    reset_provider_session,
)
from kai.workshop.settings_workspaces import (
    SettingsWorkspaceSnapshot,
    WorkshopSettingsWorkspaceAccessDenied,
    WorkshopSettingsWorkspaceService,
)
from kai.workshop.store import WorkshopEventStore


class WorkshopRuntimeLaneStatusError(RuntimeError):
    """A runtime-lane status request could not be resolved safely."""


class WorkshopRuntimeLaneStatusAccessDenied(WorkshopRuntimeLaneStatusError):
    """The requester cannot inspect the requested channel."""


class WorkshopRuntimeLaneStatusUnavailable(WorkshopRuntimeLaneStatusError):
    """The requested channel-agent lane has no canonical runtime authority."""


class WorkshopRuntimeLaneStatusAmbiguous(WorkshopRuntimeLaneStatusError):
    """The request identifies more than one runtime lane."""


class WorkshopRuntimeLaneStatusBusy(WorkshopRuntimeLaneStatusError):
    """The requested runtime lane has an accepted or executing run."""


class WorkshopRuntimeLaneStatusReplayConflict(WorkshopRuntimeLaneStatusError):
    """A client operation ID belongs to another runtime lane."""


@dataclass(frozen=True, slots=True)
class RuntimeLaneStatusAuthority:
    requester_principal_id: PrincipalId
    channel_id: ChannelId
    channel_kind: str
    agent_id: AgentId
    agent_name: str
    agent_handle: str
    sponsor_principal_id: PrincipalId
    sponsor_display_name: str
    runtime_profile_id: RuntimeProfileId
    owns_agent: bool
    operator: bool
    private_context: bool = True
    settings_channel_id: ChannelId | None = None
    workspace_runtime_profile_id: RuntimeProfileId | None = None


@dataclass(frozen=True, slots=True)
class RuntimeLaneRunStatus:
    run_id: str
    status: str
    accepted_at: datetime
    started_at: datetime | None
    terminal_at: datetime | None
    terminal_code: str | None


@dataclass(frozen=True, slots=True)
class RuntimeLaneDiagnostics:
    runtime_profile_id: RuntimeProfileId
    provider_session_revision: str | None
    provider_session_present: bool
    last_run_id: str | None
    context_through_event_position: int | None


@dataclass(frozen=True, slots=True)
class RuntimeLaneStatusSnapshot:
    authority: RuntimeLaneStatusAuthority
    backend: str
    provider: str
    model_value: str
    model_source: str
    timeout_seconds: int
    timeout_source: str
    workspace_mode: str
    workspace_label: str | None
    workspace: str | None
    workspace_revision: str | None
    workspaces: tuple[tuple[str, str, bool, bool], ...]
    process_state: str
    provider_session_state: str
    continuity_state: str
    session_created_at: datetime | None
    session_updated_at: datetime | None
    active_run: RuntimeLaneRunStatus | None
    last_run: RuntimeLaneRunStatus | None
    diagnostics: RuntimeLaneDiagnostics | None
    fresh_session_generation: int | None = None
    fresh_session_revision: str | None = None


class WorkshopRuntimeLaneStatusService:
    """Resolve runtime status once for every client adapter."""

    def __init__(
        self,
        store: WorkshopEventStore,
        settings: WorkshopSettingsWorkspaceService,
        runtime_pool: WorkshopRuntimePool,
    ) -> None:
        self._store = store
        self._settings = settings
        self._runtime_pool = runtime_pool
        self._reset_locks: dict[tuple[ChannelId, AgentId], asyncio.Lock] = {}

    async def authority_for_principal_channel(
        self,
        principal_id: PrincipalId,
        channel_id: ChannelId,
        *,
        agent_id: AgentId | None = None,
        agent_handle: str | None = None,
    ) -> RuntimeLaneStatusAuthority:
        if not await CanonicalChannelAuthorizer(self._store).can_read_channel(principal_id, channel_id):
            raise WorkshopRuntimeLaneStatusAccessDenied("Access denied")
        if agent_id is not None and agent_handle is not None:
            raise WorkshopRuntimeLaneStatusError("Choose an agent ID or handle, not both")
        clauses = ["ca.channel_id = ?", "ca.detached_at IS NULL", "ad.lifecycle_state = 'active'"]
        values: list[object] = [channel_id]
        if agent_id is not None:
            clauses.append("ca.agent_id = ?")
            values.append(agent_id)
        if agent_handle is not None:
            normalized_handle = agent_handle.removeprefix("@").strip().lower()
            if not normalized_handle:
                raise WorkshopRuntimeLaneStatusError("Agent handle is required")
            clauses.append("lower(ad.handle) = ?")
            values.append(normalized_handle)
        async with self._store.connection.execute(
            "SELECT c.kind, a.id, a.name, ad.handle, ad.owner_principal_id, "
            "ad.owner_runtime_profile_id, owner.display_name, wm.role, "
            "ad.owner_direct_channel_id, ra.runtime_profile_id "
            "FROM channel_agents ca "
            "JOIN channels c ON c.id = ca.channel_id "
            "JOIN agents a ON a.id = ca.agent_id AND a.workshop_id = c.workshop_id "
            "JOIN agent_definitions ad ON ad.agent_id = a.id "
            "JOIN principals owner ON owner.id = ad.owner_principal_id AND owner.kind = 'human' "
            "JOIN workshop_memberships wm ON wm.workshop_id = c.workshop_id "
            "AND wm.principal_id = ? "
            "LEFT JOIN channel_agent_runtime_assignments ra "
            "ON ra.channel_id = ca.channel_id AND ra.agent_id = ca.agent_id "
            f"WHERE {' AND '.join(clauses)} ORDER BY a.id",
            (principal_id, *values),
        ) as cursor:
            rows = list(await cursor.fetchall())
        if not rows:
            raise WorkshopRuntimeLaneStatusUnavailable("This channel has no matching active agent runtime")
        if len(rows) != 1:
            raise WorkshopRuntimeLaneStatusAmbiguous("Choose one agent to inspect in this channel")
        row = rows[0]
        if row[4] is None or row[5] is None:
            raise WorkshopRuntimeLaneStatusUnavailable("This agent has no owner-sponsored runtime")
        try:
            sponsor_principal_id = PrincipalId(str(row[4]))
            runtime_profile_id = RuntimeProfileId(str(row[5]))
            resolved_agent_id = AgentId(str(row[1]))
        except (TypeError, ValueError) as exc:
            raise WorkshopRuntimeLaneStatusUnavailable("This agent has invalid runtime authority") from exc
        return RuntimeLaneStatusAuthority(
            requester_principal_id=principal_id,
            channel_id=channel_id,
            channel_kind=str(row[0]),
            agent_id=resolved_agent_id,
            agent_name=str(row[2]),
            agent_handle=str(row[3]),
            sponsor_principal_id=sponsor_principal_id,
            sponsor_display_name=str(row[6]),
            runtime_profile_id=runtime_profile_id,
            owns_agent=principal_id == sponsor_principal_id,
            operator=str(row[7]) == "admin",
            private_context=str(row[0]) == "direct",
            settings_channel_id=(ChannelId(str(row[8])) if row[8] is not None else None),
            workspace_runtime_profile_id=(
                RuntimeProfileId(str(row[9])) if str(row[0]) == "direct" and row[9] is not None else None
            ),
        )

    @staticmethod
    def _runtime_authority(authority: RuntimeLaneStatusAuthority) -> WorkshopInternalAPIExecutionContext:
        return WorkshopInternalAPIExecutionContext(
            principal_id=authority.requester_principal_id,
            channel_id=authority.channel_id,
            agent_id=authority.agent_id,
            runtime_profile_id=authority.runtime_profile_id,
            private_context=authority.private_context,
            sponsor_principal_id=authority.sponsor_principal_id,
            settings_channel_id=authority.settings_channel_id,
            workspace_runtime_profile_id=authority.workspace_runtime_profile_id,
        )

    async def reset_provider_session(
        self,
        authority: RuntimeLaneStatusAuthority,
        client_operation_id: str,
    ) -> ProviderSessionResetResult:
        """Start a fresh provider session without clearing canonical user data."""
        if not isinstance(client_operation_id, str) or not 1 <= len(client_operation_id) <= 128:
            raise ValueError("client_operation_id must contain between 1 and 128 characters")
        lock = self._reset_locks.setdefault((authority.channel_id, authority.agent_id), asyncio.Lock())
        async with lock:
            replay = await load_provider_session_reset_operation(
                self._store,
                authority.requester_principal_id,
                client_operation_id,
            )
            if replay is not None:
                if (
                    replay.channel_id != authority.channel_id
                    or replay.agent_id != authority.agent_id
                    or replay.runtime_profile_id != authority.runtime_profile_id
                ):
                    raise WorkshopRuntimeLaneStatusReplayConflict(
                        "The fresh-session operation ID already belongs to another runtime lane"
                    )
                return replay
            active_run = await self._load_run(authority, active=True)
            if active_run is not None:
                raise WorkshopRuntimeLaneStatusBusy(
                    "A fresh provider session cannot start while this agent has an active run"
                )
            result: ProviderSessionResetResult | None = None

            async def commit_reset(live_process_stopped: bool) -> None:
                nonlocal result
                try:
                    result = await reset_provider_session(
                        self._store,
                        requester_principal_id=authority.requester_principal_id,
                        channel_id=authority.channel_id,
                        agent_id=authority.agent_id,
                        runtime_profile_id=authority.runtime_profile_id,
                        client_operation_id=client_operation_id,
                        live_process_stopped=live_process_stopped,
                        occurred_at=datetime.now(UTC),
                    )
                except ProviderSessionResetConflictError as exc:
                    raise WorkshopRuntimeLaneStatusReplayConflict(str(exc)) from exc

            accepted, _ = await self._runtime_pool.reset_provider_session(
                self._runtime_authority(authority),
                commit_reset=commit_reset,
            )
            if not accepted:
                raise WorkshopRuntimeLaneStatusBusy(
                    "A fresh provider session cannot start while this agent has an active run"
                )
            if result is None:
                raise WorkshopRuntimeLaneStatusError("Fresh provider-session reset was not recorded")
            return result

    async def authority_for_transport_binding(
        self,
        *,
        transport: str,
        sender_subject: str,
        channel_subject: str,
        agent_handle: str | None = None,
    ) -> RuntimeLaneStatusAuthority:
        async with self._store.connection.execute(
            "SELECT e.principal_id, c.id FROM external_identities e "
            "JOIN workshop_memberships wm ON wm.principal_id = e.principal_id "
            "JOIN channel_bindings b ON b.transport = e.provider "
            "JOIN channels c ON c.id = b.channel_id AND c.workshop_id = wm.workshop_id "
            "WHERE e.provider = ? AND e.external_subject = ? "
            "AND b.transport = ? AND b.external_channel_id = ? AND c.archived_at IS NULL",
            (transport, sender_subject, transport, channel_subject),
        ) as cursor:
            rows = list(await cursor.fetchall())
        if len(rows) != 1:
            raise WorkshopRuntimeLaneStatusAccessDenied("Adapter identity does not resolve uniquely")
        return await self.authority_for_principal_channel(
            PrincipalId(str(rows[0][0])),
            ChannelId(str(rows[0][1])),
            agent_handle=agent_handle,
        )

    async def inspect(self, authority: RuntimeLaneStatusAuthority) -> RuntimeLaneStatusSnapshot:
        try:
            runtime_authority = self._settings.authority_for_principal_profile(
                authority.sponsor_principal_id,
                authority.runtime_profile_id,
            )
            runtime = await self._settings.inspect(runtime_authority)
        except WorkshopSettingsWorkspaceAccessDenied as exc:
            raise WorkshopRuntimeLaneStatusUnavailable("Runtime authority changed") from exc
        model_value = runtime.model.value
        timeout_seconds = runtime.timeout_seconds.value
        if (
            not isinstance(model_value, str)
            or not isinstance(timeout_seconds, int)
            or isinstance(timeout_seconds, bool)
        ):
            raise WorkshopRuntimeLaneStatusUnavailable("Runtime settings are invalid")

        owner_workspace: SettingsWorkspaceSnapshot | None = None
        if authority.owns_agent and authority.channel_kind == "direct":
            try:
                channel_authority = self._settings.authority_for_principal_channel(
                    authority.requester_principal_id,
                    authority.channel_id,
                )
                owner_workspace = await self._settings.inspect(channel_authority)
            except WorkshopSettingsWorkspaceAccessDenied:
                owner_workspace = None

        session = await load_runtime_session(self._store, authority.channel_id, authority.agent_id)
        reset_state = await load_provider_session_reset_state(
            self._store,
            authority.channel_id,
            authority.agent_id,
        )
        active_run = await self._load_run(authority, active=True)
        last_run = await self._load_run(authority, active=False)
        provider_state, continuity_state = self._session_states(
            session,
            authority,
            backend=runtime.backend,
            provider=runtime.provider,
            model=model_value,
            workspace=(owner_workspace.workspace if owner_workspace is not None else None),
        )
        workspace_label = self._workspace_label(owner_workspace)
        diagnostics = (
            RuntimeLaneDiagnostics(
                runtime_profile_id=authority.runtime_profile_id,
                provider_session_revision=(session.retained_context_revision if session is not None else None),
                provider_session_present=(session is not None and session.provider_session_id is not None),
                last_run_id=(str(session.last_run_id) if session is not None else None),
                context_through_event_position=(
                    session.context_through_event_position if session is not None else None
                ),
            )
            if authority.operator
            else None
        )
        return RuntimeLaneStatusSnapshot(
            authority=authority,
            backend=runtime.backend,
            provider=runtime.provider,
            model_value=model_value,
            model_source=runtime.model.source,
            timeout_seconds=timeout_seconds,
            timeout_source=runtime.timeout_seconds.source,
            workspace_mode="owner" if owner_workspace is not None else "neutral",
            workspace_label=workspace_label,
            workspace=(owner_workspace.workspace if owner_workspace is not None else None),
            workspace_revision=(owner_workspace.revision if owner_workspace is not None else None),
            workspaces=(
                tuple((item.path, item.name, item.current, item.home) for item in owner_workspace.workspaces)
                if owner_workspace is not None
                else ()
            ),
            process_state=("alive" if self._runtime_pool.is_alive(authority.runtime_profile_id) else "stopped")
            if authority.owns_agent or authority.operator
            else "hidden",
            provider_session_state=provider_state,
            continuity_state=continuity_state,
            session_created_at=(session.created_at if session is not None else None),
            session_updated_at=(session.updated_at if session is not None else None),
            active_run=self._run_status(active_run),
            last_run=self._run_status(last_run),
            diagnostics=diagnostics,
            fresh_session_generation=(reset_state.generation if reset_state is not None else None),
            fresh_session_revision=(reset_state.revision if reset_state is not None else None),
        )

    async def _load_run(
        self,
        authority: RuntimeLaneStatusAuthority,
        *,
        active: bool,
    ) -> DurableRun | None:
        status_clause = (
            "AND status IN ('accepted', 'started') "
            if active
            else "AND status IN ('completed', 'failed', 'cancelled') "
        )
        async with self._store.connection.execute(
            "SELECT id FROM runs WHERE channel_id = ? AND agent_id = ? "
            f"{status_clause}ORDER BY accepted_at DESC, id DESC LIMIT 1",
            (authority.channel_id, authority.agent_id),
        ) as cursor:
            row = await cursor.fetchone()
        return None if row is None else await load_durable_run(self._store, RunId(str(row[0])))

    @staticmethod
    def _session_states(
        session: CanonicalRuntimeSession | None,
        authority: RuntimeLaneStatusAuthority,
        *,
        backend: str,
        provider: str,
        model: str,
        workspace: str | None,
    ) -> tuple[str, str]:
        if session is None:
            return "not_started", "not_started"
        if session.runtime_profile_id != authority.runtime_profile_id:
            return "stale", "stale"
        if (
            session.selection.backend != backend
            or session.selection.provider != provider
            or session.selection.model != model
            or (workspace is not None and session.workspace != workspace)
        ):
            return "refresh_pending", "refresh_pending"
        if session.retained_context_revision == "0" * 64:
            return "refresh_pending", "refresh_pending"
        if session.provider_session_id is None:
            return "stateless", "active"
        return "active", "active"

    @staticmethod
    def _workspace_label(snapshot: SettingsWorkspaceSnapshot | None) -> str | None:
        if snapshot is None:
            return None
        return next((item.name for item in snapshot.workspaces if item.current), "Current workspace")

    @staticmethod
    def _run_status(run: DurableRun | None) -> RuntimeLaneRunStatus | None:
        if run is None:
            return None
        return RuntimeLaneRunStatus(
            run_id=str(run.run_id),
            status=run.status.value,
            accepted_at=run.accepted_at,
            started_at=run.started_at,
            terminal_at=run.terminal_at,
            terminal_code=run.terminal_code,
        )
