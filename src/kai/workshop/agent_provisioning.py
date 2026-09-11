"""Replay-safe orchestration for provisioning principal-owned Workshop agents."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from kai.config import canonicalize_model_for_backend
from kai.workshop.agent_creation_options import (
    AgentCreationBackendOption,
    WorkshopAgentCreationOptionsService,
)
from kai.workshop.agent_definitions import (
    MAX_AGENT_DESCRIPTION,
    MAX_AGENT_DISPLAY_NAME,
    MAX_AGENT_INSTRUCTIONS,
    MAX_AGENT_PURPOSE,
    normalize_agent_handle,
    validate_agent_capabilities,
    validate_agent_presentation,
    validate_agent_text,
    validate_collaboration_operations,
)
from kai.workshop.agent_enablement import (
    WorkshopAgentEnablementAccessDenied,
    WorkshopAgentEnablementConflict,
    WorkshopAgentEnablementError,
    WorkshopAgentEnablementService,
)
from kai.workshop.agent_lifecycle import (
    WorkshopAgentLifecycleAccessDenied,
    WorkshopAgentLifecycleConflict,
    WorkshopAgentLifecycleError,
    WorkshopAgentLifecycleService,
)
from kai.workshop.collaboration_policy import (
    WorkshopCollaborationPolicyAccessDenied,
    WorkshopCollaborationPolicyConflict,
    WorkshopCollaborationPolicyError,
    WorkshopCollaborationPolicyService,
    WorkshopCollaborationPolicyValidationError,
)
from kai.workshop.domain import (
    AgentDefinitionId,
    AgentDefinitionRevisionId,
    AgentEnablementId,
    AgentId,
    AgentProvisioningId,
    ChannelId,
    PrincipalId,
    RuntimeProfileId,
    WorkshopId,
)
from kai.workshop.settings_workspaces import (
    WorkshopSettingsWorkspaceAccessDenied,
    WorkshopSettingsWorkspaceBusy,
    WorkshopSettingsWorkspaceError,
    WorkshopSettingsWorkspaceService,
    WorkshopSettingsWorkspaceValidationError,
)
from kai.workshop.store import WorkshopEventStore

_OPERATION_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_BACKEND_OPTION_PATTERN = re.compile(r"^[a-z][a-z0-9_-]*:[a-z][a-z0-9_-]*$")
_STAGES = (
    "definition_created",
    "revision_activated",
    "enablement_created",
    "runtime_registered",
    "backend_selected",
    "model_selected",
    "workspace_selected",
    "timeout_selected",
    "collaboration_policy_set",
    "ready",
)


class WorkshopAgentProvisioningError(RuntimeError):
    """A ready-agent provisioning operation could not be completed."""


class WorkshopAgentProvisioningAccessDenied(WorkshopAgentProvisioningError):
    """The request references authority not owned by the principal."""


class WorkshopAgentProvisioningValidationError(WorkshopAgentProvisioningError):
    """The provisioning request is malformed or currently unavailable."""


class WorkshopAgentProvisioningConflict(WorkshopAgentProvisioningError):
    """An operation identity or canonical stage conflicts with prior state."""


class WorkshopAgentProvisioningStorageError(WorkshopAgentProvisioningError):
    """Provisioning coordination state could not be persisted."""


@dataclass(frozen=True, slots=True)
class AgentProvisioningBlocker:
    code: str
    detail: str


@dataclass(frozen=True, slots=True)
class AgentProvisioningResult:
    operation_id: AgentProvisioningId
    client_operation_id: str
    status: str
    replayed: bool
    definition_id: AgentDefinitionId | None
    revision_id: AgentDefinitionRevisionId | None
    agent_id: AgentId | None
    enablement_id: AgentEnablementId | None
    direct_channel_id: ChannelId | None
    runtime_profile_id: RuntimeProfileId
    completed_stages: tuple[str, ...]
    next_stage: str | None
    blockers: tuple[AgentProvisioningBlocker, ...]


@dataclass(frozen=True, slots=True)
class _NormalizedProvisioningRequest:
    client_operation_id: str
    handle: str
    display_name: str
    description: str
    presentation: dict[str, object]
    purpose: str
    instructions: str
    capabilities: tuple[str, ...]
    collaboration_operations: tuple[str, ...]
    runtime_profile_id: RuntimeProfileId
    backend_option_id: str
    model: str
    workspace: str
    timeout_seconds: int
    allowed_collaboration_operations: tuple[str, ...]

    def payload(self) -> dict[str, object]:
        return {
            "client_operation_id": self.client_operation_id,
            "definition": {
                "handle": self.handle,
                "display_name": self.display_name,
                "description": self.description,
                "presentation": self.presentation,
            },
            "revision": {
                "purpose": self.purpose,
                "instructions": self.instructions,
                "capabilities": list(self.capabilities),
                "collaboration_operations": list(self.collaboration_operations),
            },
            "runtime": {
                "runtime_profile_id": str(self.runtime_profile_id),
                "backend_option_id": self.backend_option_id,
                "model": self.model,
                "workspace": self.workspace,
                "timeout_seconds": self.timeout_seconds,
            },
            "collaboration_policy": {
                "allowed_operations": list(self.allowed_collaboration_operations),
            },
        }


@dataclass(frozen=True, slots=True)
class _ProvisioningOperation:
    operation_id: AgentProvisioningId
    workshop_id: WorkshopId
    principal_id: PrincipalId
    client_operation_id: str
    request_hash: str
    status: str
    next_stage: str | None
    failure_code: str | None
    failure_detail: str | None
    definition_id: AgentDefinitionId | None
    revision_id: AgentDefinitionRevisionId | None
    definition_initial_version: int | None
    agent_id: AgentId | None
    enablement_id: AgentEnablementId | None
    direct_channel_id: ChannelId | None
    runtime_profile_id: RuntimeProfileId


class WorkshopAgentProvisioningService:
    """Compose existing canonical authorities into one recoverable operation."""

    def __init__(
        self,
        store: WorkshopEventStore,
        creation_options: WorkshopAgentCreationOptionsService,
        lifecycle: WorkshopAgentLifecycleService,
        enablement: WorkshopAgentEnablementService,
        settings: WorkshopSettingsWorkspaceService,
        collaboration_policy: WorkshopCollaborationPolicyService,
    ) -> None:
        self._store = store
        self._creation_options = creation_options
        self._lifecycle = lifecycle
        self._enablement = enablement
        self._settings = settings
        self._collaboration_policy = collaboration_policy
        self._locks: dict[tuple[PrincipalId, str], asyncio.Lock] = {}

    async def provision(
        self,
        principal_id: PrincipalId,
        *,
        client_operation_id: object,
        definition: object,
        revision: object,
        runtime: object,
        collaboration_policy: object,
    ) -> AgentProvisioningResult:
        request = self._normalize(
            client_operation_id=client_operation_id,
            definition=definition,
            revision=revision,
            runtime=runtime,
            collaboration_policy=collaboration_policy,
        )
        authority = await self._lifecycle.authority_for(principal_id)
        request_json = json.dumps(request.payload(), sort_keys=True, separators=(",", ":"))
        request_hash = hashlib.sha256(request_json.encode()).hexdigest()
        lock = self._locks.setdefault((principal_id, request.client_operation_id), asyncio.Lock())
        async with lock:
            existing = await self._operation_for_identity(
                authority.workshop_id,
                principal_id,
                request.client_operation_id,
            )
            if existing is not None:
                if existing.request_hash != request_hash:
                    raise WorkshopAgentProvisioningConflict(
                        "Operation identity was reused with different provisioning input"
                    )
                if existing.status == "ready":
                    return await self._result(existing.operation_id, replayed=True)

            await self._validate_authority(principal_id, request)
            operation = existing or await self._create_operation(
                authority.workshop_id,
                principal_id,
                request,
                request_json,
                request_hash,
            )
            await self._mark_running(operation.operation_id)
            return await self._resume(operation.operation_id, principal_id, request)

    async def _resume(
        self,
        operation_id: AgentProvisioningId,
        principal_id: PrincipalId,
        request: _NormalizedProvisioningRequest,
    ) -> AgentProvisioningResult:
        completed = set(await self._completed_stages(operation_id))
        current_stage = next((stage for stage in _STAGES if stage not in completed), None)
        if current_stage is None:
            await self._record_stage(operation_id, "ready", {})
            return await self._result(operation_id, replayed=True)
        try:
            operation = await self._require_operation(operation_id)
            if "definition_created" not in completed:
                draft = await self._lifecycle.create_draft(
                    principal_id,
                    idempotency_key=self._stage_key(operation_id, "definition"),
                    handle=request.handle,
                    display_name=request.display_name,
                    description=request.description,
                    presentation=request.presentation,
                    purpose=request.purpose,
                    instructions=request.instructions,
                    capabilities=list(request.capabilities),
                    collaboration_operations=list(request.collaboration_operations),
                )
                revision = draft.revisions[0]
                await self._record_stage(
                    operation_id,
                    "definition_created",
                    {
                        "definition_id": str(draft.definition_id),
                        "revision_id": str(revision.revision_id),
                        "agent_id": str(draft.agent_id),
                        "definition_initial_version": draft.state_version,
                    },
                )
                await self._after_stage("definition_created")
                completed.add("definition_created")
                operation = await self._require_operation(operation_id)

            definition_id, revision_id = self._definition_ids(operation)
            if "revision_activated" not in completed:
                if operation.definition_initial_version is None:
                    raise WorkshopAgentProvisioningStorageError("Provisioning definition receipt is incomplete")
                await self._lifecycle.activate_revision(
                    principal_id,
                    definition_id,
                    revision_id=revision_id,
                    idempotency_key=self._stage_key(operation_id, "activation"),
                    expected_version=operation.definition_initial_version,
                )
                await self._record_stage(operation_id, "revision_activated", {})
                await self._after_stage("revision_activated")
                completed.add("revision_activated")

            if "enablement_created" not in completed:
                enabled = await self._enablement.enable(
                    principal_id,
                    definition_id,
                    request.runtime_profile_id,
                    idempotency_key=self._stage_key(operation_id, "enablement"),
                )
                if enabled.enablement_id is None or enabled.direct_channel_id is None:
                    raise WorkshopAgentProvisioningStorageError(
                        "Provisioning enablement did not produce a canonical runtime lane"
                    )
                await self._record_stage(
                    operation_id,
                    "enablement_created",
                    {
                        "enablement_id": str(enabled.enablement_id),
                        "direct_channel_id": str(enabled.direct_channel_id),
                    },
                )
                await self._after_stage("enablement_created")
                completed.add("enablement_created")
                operation = await self._require_operation(operation_id)

            if "runtime_registered" not in completed:
                await self._enablement.ensure_runtime_registered(principal_id, definition_id)
                await self._record_stage(operation_id, "runtime_registered", {})
                await self._after_stage("runtime_registered")
                completed.add("runtime_registered")
            else:
                # Registration is process-local. Reconcile it on every retry,
                # including one handled by a newly started Kai process.
                await self._enablement.ensure_runtime_registered(principal_id, definition_id)

            operation = await self._require_operation(operation_id)
            if operation.direct_channel_id is None:
                raise WorkshopAgentProvisioningStorageError("Provisioning runtime receipt is incomplete")
            settings_authority = self._settings.authority_for_principal_channel(
                principal_id,
                operation.direct_channel_id,
            )

            if "backend_selected" not in completed:
                await self._settings.set_backend(settings_authority, request.backend_option_id)
                await self._record_stage(operation_id, "backend_selected", {})
                await self._after_stage("backend_selected")
                completed.add("backend_selected")

            if "model_selected" not in completed:
                await self._settings.set_model(settings_authority, request.model)
                await self._record_stage(operation_id, "model_selected", {})
                await self._after_stage("model_selected")
                completed.add("model_selected")

            if "workspace_selected" not in completed:
                await self._settings.switch_workspace(settings_authority, request.workspace)
                await self._record_stage(operation_id, "workspace_selected", {})
                await self._after_stage("workspace_selected")
                completed.add("workspace_selected")

            if "timeout_selected" not in completed:
                await self._settings.set_timeout(settings_authority, request.timeout_seconds)
                await self._record_stage(operation_id, "timeout_selected", {})
                await self._after_stage("timeout_selected")
                completed.add("timeout_selected")

            if "collaboration_policy_set" not in completed:
                await self._collaboration_policy.set_allowed(
                    principal_id,
                    definition_id,
                    allowed_operations=list(request.allowed_collaboration_operations),
                    expected_policy_version=0,
                    client_operation_id=self._stage_key(operation_id, "collaboration"),
                )
                await self._record_stage(operation_id, "collaboration_policy_set", {})
                await self._after_stage("collaboration_policy_set")

            await self._record_stage(operation_id, "ready", {})
            await self._after_stage("ready")
            return await self._result(operation_id, replayed=False)
        except WorkshopAgentProvisioningError:
            raise
        except BaseException as exc:
            # Process cancellation/crash must leave the last durable receipt
            # untouched so a new coordinator can resume the exact next stage.
            if not isinstance(exc, Exception):
                raise
            code, detail = self._failure(exc)
            completed = set(await self._completed_stages(operation_id))
            next_stage = next((stage for stage in _STAGES if stage not in completed), None)
            status = (
                "draft"
                if "definition_created" in completed and "revision_activated" not in completed
                else "needs_attention"
            )
            await self._mark_failure(operation_id, status, next_stage, code, detail)
            return await self._result(operation_id, replayed=False)

    async def _validate_authority(
        self,
        principal_id: PrincipalId,
        request: _NormalizedProvisioningRequest,
    ) -> None:
        options = await self._creation_options.inspect(principal_id)
        runtime = next(
            (item for item in options.runtimes if item.runtime_profile_id == request.runtime_profile_id),
            None,
        )
        if runtime is None:
            raise WorkshopAgentProvisioningAccessDenied("Runtime profile is not authorized for this principal")
        backend = next(
            (item for item in runtime.backends if item.option_id == request.backend_option_id),
            None,
        )
        if backend is None:
            raise WorkshopAgentProvisioningAccessDenied("Backend option is not authorized for this runtime profile")
        if backend.blockers:
            raise WorkshopAgentProvisioningValidationError("The selected backend is not currently usable")
        workspace = next(
            (item for item in runtime.workspaces if item.path == request.workspace),
            None,
        )
        if workspace is None:
            raise WorkshopAgentProvisioningAccessDenied("Workspace is not authorized for this runtime profile")
        if not workspace.available:
            raise WorkshopAgentProvisioningValidationError("The selected workspace is unavailable")
        if not runtime.minimum_timeout_seconds <= request.timeout_seconds <= runtime.maximum_timeout_seconds:
            raise WorkshopAgentProvisioningValidationError("Timeout is outside this runtime profile's policy bounds")
        self._validate_model(backend, request.model)
        try:
            self._collaboration_policy.validate_initial_allowed_operations(
                request.collaboration_operations,
                request.allowed_collaboration_operations,
            )
        except WorkshopCollaborationPolicyValidationError as exc:
            raise WorkshopAgentProvisioningValidationError(str(exc)) from exc

    @staticmethod
    def _validate_model(backend: AgentCreationBackendOption, model: str) -> None:
        if model == backend.default_model:
            return
        if not any(item.model_id == model and item.selectable for item in backend.models):
            raise WorkshopAgentProvisioningValidationError("Model is not selectable for the authorized backend")

    def _normalize(
        self,
        *,
        client_operation_id: object,
        definition: object,
        revision: object,
        runtime: object,
        collaboration_policy: object,
    ) -> _NormalizedProvisioningRequest:
        if not isinstance(client_operation_id, str) or not _OPERATION_ID_PATTERN.fullmatch(client_operation_id):
            raise WorkshopAgentProvisioningValidationError(
                "client_operation_id must be 1-128 letters, digits, dots, underscores, colons, or hyphens"
            )
        definition_map = self._exact_object(
            definition,
            "definition",
            {"handle", "display_name", "description", "presentation"},
        )
        revision_map = self._exact_object(
            revision,
            "revision",
            {
                "purpose",
                "instructions",
                "capabilities",
                "collaboration_operations",
            },
        )
        runtime_map = self._exact_object(
            runtime,
            "runtime",
            {
                "runtime_profile_id",
                "backend_option_id",
                "model",
                "workspace",
                "timeout_seconds",
            },
        )
        policy_map = self._exact_object(
            collaboration_policy,
            "collaboration_policy",
            {"allowed_operations"},
        )
        try:
            handle = normalize_agent_handle(definition_map["handle"])
            display_name = validate_agent_text(
                definition_map["display_name"],
                field="display_name",
                maximum=MAX_AGENT_DISPLAY_NAME,
            )
            description = validate_agent_text(
                definition_map["description"],
                field="description",
                maximum=MAX_AGENT_DESCRIPTION,
                allow_empty=True,
            )
            presentation = json.loads(validate_agent_presentation(definition_map["presentation"]))
            purpose = validate_agent_text(
                revision_map["purpose"],
                field="purpose",
                maximum=MAX_AGENT_PURPOSE,
            )
            instructions = validate_agent_text(
                revision_map["instructions"],
                field="instructions",
                maximum=MAX_AGENT_INSTRUCTIONS,
            )
            capabilities = validate_agent_capabilities(revision_map["capabilities"])
            requested_operations = validate_collaboration_operations(revision_map["collaboration_operations"])
            runtime_profile_id = RuntimeProfileId(str(runtime_map["runtime_profile_id"]))
        except (TypeError, ValueError) as exc:
            raise WorkshopAgentProvisioningValidationError(str(exc)) from exc
        backend_option_id = runtime_map["backend_option_id"]
        if not isinstance(backend_option_id, str) or not _BACKEND_OPTION_PATTERN.fullmatch(backend_option_id):
            raise WorkshopAgentProvisioningValidationError("backend_option_id is invalid")
        backend = backend_option_id.partition(":")[0]
        raw_model = runtime_map["model"]
        if not isinstance(raw_model, str):
            raise WorkshopAgentProvisioningValidationError("model must be a string")
        model = canonicalize_model_for_backend(raw_model.strip(), backend)
        if not model:
            raise WorkshopAgentProvisioningValidationError("model must be non-empty")
        raw_workspace = runtime_map["workspace"]
        if not isinstance(raw_workspace, str) or not Path(raw_workspace).is_absolute():
            raise WorkshopAgentProvisioningValidationError("workspace must be an absolute path")
        workspace = str(Path(raw_workspace).expanduser().resolve())
        timeout = runtime_map["timeout_seconds"]
        if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout < 1:
            raise WorkshopAgentProvisioningValidationError("timeout_seconds must be a positive integer")
        try:
            allowed = validate_collaboration_operations(policy_map["allowed_operations"])
        except ValueError as exc:
            raise WorkshopAgentProvisioningValidationError(str(exc)) from exc
        return _NormalizedProvisioningRequest(
            client_operation_id,
            handle,
            display_name,
            description,
            presentation,
            purpose,
            instructions,
            capabilities,
            requested_operations,
            runtime_profile_id,
            backend_option_id,
            model,
            workspace,
            timeout,
            allowed,
        )

    @staticmethod
    def _exact_object(value: object, field: str, keys: set[str]) -> dict[str, object]:
        if not isinstance(value, dict) or set(value) != keys or any(not isinstance(key, str) for key in value):
            raise WorkshopAgentProvisioningValidationError(f"{field} must contain exactly: {', '.join(sorted(keys))}")
        return value

    async def _create_operation(
        self,
        workshop_id: WorkshopId,
        principal_id: PrincipalId,
        request: _NormalizedProvisioningRequest,
        request_json: str,
        request_hash: str,
    ) -> _ProvisioningOperation:
        operation_id = AgentProvisioningId.derived(
            principal_id,
            f"{workshop_id}:{request.client_operation_id}",
        )
        now = datetime.now(UTC).isoformat()
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            await connection.execute(
                "INSERT INTO agent_provisioning_operations ("
                "id, workshop_id, principal_id, client_operation_id, request_hash, request_json, "
                "status, next_stage, runtime_profile_id, backend_option_id, model, workspace, "
                "timeout_seconds, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, "
                "'provisioning', 'definition_created', ?, ?, ?, ?, ?, ?, ?)",
                (
                    operation_id,
                    workshop_id,
                    principal_id,
                    request.client_operation_id,
                    request_hash,
                    request_json,
                    request.runtime_profile_id,
                    request.backend_option_id,
                    request.model,
                    request.workspace,
                    request.timeout_seconds,
                    now,
                    now,
                ),
            )
            await connection.commit()
        except Exception as exc:
            await connection.rollback()
            existing = await self._operation_for_identity(
                workshop_id,
                principal_id,
                request.client_operation_id,
            )
            if existing is not None:
                if existing.request_hash != request_hash:
                    raise WorkshopAgentProvisioningConflict(
                        "Operation identity was reused with different provisioning input"
                    ) from exc
                return existing
            raise WorkshopAgentProvisioningStorageError("Provisioning operation could not be persisted") from exc
        return await self._require_operation(operation_id)

    async def _operation_for_identity(
        self,
        workshop_id: WorkshopId,
        principal_id: PrincipalId,
        client_operation_id: str,
    ) -> _ProvisioningOperation | None:
        async with self._store.connection.execute(
            "SELECT * FROM agent_provisioning_operations WHERE workshop_id = ? "
            "AND principal_id = ? AND client_operation_id = ?",
            (workshop_id, principal_id, client_operation_id),
        ) as cursor:
            row = await cursor.fetchone()
        return self._operation(row) if row is not None else None

    async def _require_operation(
        self,
        operation_id: AgentProvisioningId,
    ) -> _ProvisioningOperation:
        async with self._store.connection.execute(
            "SELECT * FROM agent_provisioning_operations WHERE id = ?",
            (operation_id,),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise WorkshopAgentProvisioningStorageError("Provisioning operation is unavailable")
        return self._operation(row)

    @staticmethod
    def _operation(row) -> _ProvisioningOperation:
        return _ProvisioningOperation(
            AgentProvisioningId(str(row["id"])),
            WorkshopId(str(row["workshop_id"])),
            PrincipalId(str(row["principal_id"])),
            str(row["client_operation_id"]),
            str(row["request_hash"]),
            str(row["status"]),
            str(row["next_stage"]) if row["next_stage"] is not None else None,
            str(row["failure_code"]) if row["failure_code"] is not None else None,
            str(row["failure_detail"]) if row["failure_detail"] is not None else None,
            AgentDefinitionId(str(row["definition_id"])) if row["definition_id"] is not None else None,
            AgentDefinitionRevisionId(str(row["revision_id"])) if row["revision_id"] is not None else None,
            int(row["definition_initial_version"]) if row["definition_initial_version"] is not None else None,
            AgentId(str(row["agent_id"])) if row["agent_id"] is not None else None,
            AgentEnablementId(str(row["enablement_id"])) if row["enablement_id"] is not None else None,
            ChannelId(str(row["direct_channel_id"])) if row["direct_channel_id"] is not None else None,
            RuntimeProfileId(str(row["runtime_profile_id"])),
        )

    async def _record_stage(
        self,
        operation_id: AgentProvisioningId,
        stage: str,
        details: dict[str, object],
    ) -> None:
        if stage not in _STAGES:
            raise WorkshopAgentProvisioningStorageError("Unknown provisioning stage")
        now = datetime.now(UTC).isoformat()
        next_stage = _STAGES[_STAGES.index(stage) + 1] if stage != "ready" else None
        updates: dict[str, object] = {}
        for key in (
            "definition_id",
            "revision_id",
            "definition_initial_version",
            "agent_id",
            "enablement_id",
            "direct_channel_id",
        ):
            if key in details:
                updates[key] = details[key]
        set_clause = [
            "status = ?",
            "next_stage = ?",
            "failure_code = NULL",
            "failure_detail = NULL",
            "updated_at = ?",
        ]
        parameters: list[object] = [
            "ready" if stage == "ready" else "provisioning",
            next_stage,
            now,
        ]
        for key, value in updates.items():
            set_clause.append(f"{key} = ?")
            parameters.append(value)
        parameters.append(operation_id)
        connection = self._store.connection
        try:
            await connection.execute("BEGIN IMMEDIATE")
            async with connection.execute(
                "SELECT stage, details_json FROM agent_provisioning_stage_receipts WHERE operation_id = ?",
                (operation_id,),
            ) as cursor:
                receipt_rows = list(await cursor.fetchall())
            completed = {str(row[0]) for row in receipt_rows}
            prior = set(_STAGES[: _STAGES.index(stage)])
            if not prior.issubset(completed):
                raise WorkshopAgentProvisioningStorageError("Provisioning stage receipts are out of order")
            existing_details = next(
                (str(row[1]) for row in receipt_rows if str(row[0]) == stage),
                None,
            )
            encoded_details = json.dumps(details, sort_keys=True, separators=(",", ":"))
            if existing_details is not None and existing_details != encoded_details:
                raise WorkshopAgentProvisioningConflict("Provisioning stage receipt conflicts with canonical state")
            await connection.execute(
                "INSERT OR IGNORE INTO agent_provisioning_stage_receipts "
                "(operation_id, stage, details_json, completed_at) VALUES (?, ?, ?, ?)",
                (
                    operation_id,
                    stage,
                    encoded_details,
                    now,
                ),
            )
            await connection.execute(
                f"UPDATE agent_provisioning_operations SET {', '.join(set_clause)} WHERE id = ?",
                tuple(parameters),
            )
            await connection.commit()
        except WorkshopAgentProvisioningError:
            await connection.rollback()
            raise
        except Exception as exc:
            await connection.rollback()
            raise WorkshopAgentProvisioningStorageError("Provisioning stage receipt could not be persisted") from exc

    async def _completed_stages(self, operation_id: AgentProvisioningId) -> tuple[str, ...]:
        async with self._store.connection.execute(
            "SELECT stage FROM agent_provisioning_stage_receipts WHERE operation_id = ?",
            (operation_id,),
        ) as cursor:
            found = {str(row[0]) for row in await cursor.fetchall()}
        return tuple(stage for stage in _STAGES if stage in found)

    async def _mark_running(self, operation_id: AgentProvisioningId) -> None:
        operation = await self._require_operation(operation_id)
        await self._update_status(
            operation_id,
            "provisioning",
            operation.next_stage,
            None,
            None,
        )

    async def _mark_failure(
        self,
        operation_id: AgentProvisioningId,
        status: str,
        next_stage: str | None,
        code: str,
        detail: str,
    ) -> None:
        await self._update_status(operation_id, status, next_stage, code, detail)

    async def _update_status(
        self,
        operation_id: AgentProvisioningId,
        status: str,
        next_stage: str | None,
        failure_code: str | None,
        failure_detail: str | None,
    ) -> None:
        try:
            await self._store.connection.execute(
                "UPDATE agent_provisioning_operations SET status = ?, next_stage = ?, "
                "failure_code = ?, failure_detail = ?, updated_at = ? WHERE id = ?",
                (
                    status,
                    next_stage,
                    failure_code,
                    failure_detail,
                    datetime.now(UTC).isoformat(),
                    operation_id,
                ),
            )
            await self._store.connection.commit()
        except Exception as exc:
            await self._store.connection.rollback()
            raise WorkshopAgentProvisioningStorageError("Provisioning status could not be persisted") from exc

    async def _result(
        self,
        operation_id: AgentProvisioningId,
        *,
        replayed: bool,
    ) -> AgentProvisioningResult:
        operation = await self._require_operation(operation_id)
        blockers = (
            (
                AgentProvisioningBlocker(
                    operation.failure_code,
                    operation.failure_detail,
                ),
            )
            if operation.failure_code is not None and operation.failure_detail is not None
            else ()
        )
        return AgentProvisioningResult(
            operation.operation_id,
            operation.client_operation_id,
            operation.status,
            replayed,
            operation.definition_id,
            operation.revision_id,
            operation.agent_id,
            operation.enablement_id,
            operation.direct_channel_id,
            operation.runtime_profile_id,
            await self._completed_stages(operation_id),
            operation.next_stage,
            blockers,
        )

    @staticmethod
    def _definition_ids(
        operation: _ProvisioningOperation,
    ) -> tuple[AgentDefinitionId, AgentDefinitionRevisionId]:
        if operation.definition_id is None or operation.revision_id is None:
            raise WorkshopAgentProvisioningStorageError("Provisioning definition receipt is incomplete")
        return operation.definition_id, operation.revision_id

    @staticmethod
    def _stage_key(operation_id: AgentProvisioningId, stage: str) -> str:
        return f"provision:{operation_id}:{stage}"

    @staticmethod
    def _failure(exc: Exception) -> tuple[str, str]:
        if isinstance(
            exc,
            (
                WorkshopAgentEnablementAccessDenied,
                WorkshopAgentLifecycleAccessDenied,
                WorkshopCollaborationPolicyAccessDenied,
                WorkshopSettingsWorkspaceAccessDenied,
            ),
        ):
            return (
                "authority_changed",
                "Provisioning authority changed. Review the requested runtime choices and retry.",
            )
        if isinstance(
            exc,
            (
                WorkshopSettingsWorkspaceValidationError,
                WorkshopCollaborationPolicyValidationError,
            ),
        ):
            return (
                "selection_invalidated",
                "A selected runtime value is no longer available. Review the choices and retry.",
            )
        if isinstance(exc, WorkshopSettingsWorkspaceBusy):
            return (
                "runtime_busy",
                "The selected runtime is busy. Retry this provisioning operation when it is idle.",
            )
        if isinstance(
            exc,
            (
                WorkshopAgentEnablementConflict,
                WorkshopAgentLifecycleConflict,
                WorkshopCollaborationPolicyConflict,
            ),
        ):
            return (
                "canonical_conflict",
                "Canonical agent state changed during provisioning. Review the agent and retry.",
            )
        if isinstance(
            exc,
            (
                WorkshopAgentEnablementError,
                WorkshopAgentLifecycleError,
                WorkshopCollaborationPolicyError,
                WorkshopSettingsWorkspaceError,
            ),
        ):
            return (
                "stage_failed",
                "A canonical provisioning stage failed. Retry this operation to resume it.",
            )
        return (
            "stage_failed",
            "A provisioning stage failed. Retry this operation to resume it.",
        )

    async def _after_stage(self, stage: str) -> None:
        """Test seam for simulating process interruption after a receipt."""
        del stage
