"""Authenticated HTTP contracts for Workshop enrollment and conversation access."""

from __future__ import annotations

import asyncio
import ipaddress
import json
import math
import re
import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Protocol
from urllib.parse import quote

from aiohttp import BodyPartReader, web

from kai.workshop.artifacts import (
    MAX_ARTIFACT_BYTES,
    ArtifactAccessDeniedError,
    ArtifactStorageBoundaryError,
    ArtifactSummary,
    ArtifactTooLargeError,
    StagedArtifact,
    StoredArtifact,
    WorkshopArtifactService,
)
from kai.workshop.authorization import CanonicalChannelAuthorizer
from kai.workshop.client_commands import (
    ClientCommandExecutorUnavailableError,
    ClientCommandSubmission,
)
from kai.workshop.client_events import (
    ClientChannelEventBatch,
    ClientRunLifecycleEvent,
    ClientTimelineMessageEvent,
    read_client_channel_events,
)
from kai.workshop.client_sessions import EnrollmentGrantUnavailableError, WorkshopClientEnrollmentManager
from kai.workshop.conversation_commands import ConversationCommandAcceptanceError
from kai.workshop.domain import ArtifactId, ChannelId, PrincipalId, RunId
from kai.workshop.execution_coordinator import CanonicalCancellationDisposition
from kai.workshop.inbound import ClientInboundMessage, InboundBindingNotFoundError
from kai.workshop.memory_queries import (
    DEFAULT_PAGE_SIZE,
    MemoryMutationBatch,
    MemoryQueryAuthority,
    MemoryQueryFilters,
    MemoryRecordDetail,
    MemoryRecordSummary,
    MemorySourceContext,
    MemorySourceMessage,
    WorkshopMemoryAccessDenied,
    WorkshopMemoryNotFound,
    WorkshopMemoryQueryError,
    WorkshopMemoryQueryService,
    WorkshopMemoryResponseTooLarge,
    WorkshopMemoryValidationError,
)
from kai.workshop.run_lifecycle import DurableRun, RunNotFoundError, RunStatus
from kai.workshop.run_previews import RunPreview, WorkshopRunPreviewRegistry
from kai.workshop.settings_workspaces import (
    SettingsWorkspaceAuthority,
    SettingsWorkspaceSnapshot,
    WorkshopSettingsWorkspaceAccessDenied,
    WorkshopSettingsWorkspaceService,
    WorkshopSettingsWorkspaceValidationError,
    WorkspaceConfigSnapshot,
)
from kai.workshop.store import IdempotencyConflictError, WorkshopEventStore
from kai.workshop.timeline import (
    TimelineAccessDeniedError,
    TimelineCursorError,
    TimelineMessage,
    TimelineResumeError,
    read_channel_timeline,
)

_TIMELINE_PATH = "/v1/channels/{channel_id}/timeline"
_TIMELINE_EVENTS_PATH = "/v1/channels/{channel_id}/events"
_CLIENT_NAVIGATION_PATH = "/v1/client/navigation"
_ENROLLMENT_REDEMPTION_PATH = "/v1/client/enrollment/redeem"
_COMMAND_SUBMISSION_PATH = "/v1/channels/{channel_id}/commands"
_ARTIFACT_CONTENT_PATH = "/v1/channels/{channel_id}/artifacts/{artifact_id}/content"
_ARTIFACT_DOWNLOAD_PATH = "/v1/channels/{channel_id}/artifacts/{artifact_id}/download"
_RUN_STATE_PATH = "/v1/channels/{channel_id}/runs/{run_id}"
_RUN_TRACE_PATH = "/v1/channels/{channel_id}/runs/{run_id}/trace"
_RUN_CANCELLATION_PATH = "/v1/channels/{channel_id}/runs/{run_id}/cancel"
_RUNTIME_SETTINGS_PATH = "/v1/channels/{channel_id}/settings"
_ACTIVE_WORKSPACE_PATH = "/v1/channels/{channel_id}/workspace"
_WORKSPACE_CONFIG_PATH = "/v1/channels/{channel_id}/workspace-config"
_MEMORY_STATS_PATH = "/v1/memory/stats"
_MEMORY_RECORDS_PATH = "/v1/memory/records"
_MEMORY_SEARCH_PATH = "/v1/memory/search"
_MEMORY_DETAIL_PATH = "/v1/memory/records/{memory_id}"
_MEMORY_SOURCE_PATH = "/v1/memory/records/{memory_id}/source"
_MEMORY_SCOPE_PATH = "/v1/memory/records/{memory_id}/scope"
_MEMORY_BULK_SCOPE_PATH = "/v1/memory/actions/scope"
_MEMORY_BULK_DELETE_PATH = "/v1/memory/actions/delete"
_ALLOWED_TIMELINE_QUERY_PARAMETERS = frozenset({"cursor", "limit", "tail"})
_ALLOWED_EVENT_QUERY_PARAMETERS = frozenset({"after_position"})
_ALLOWED_MEMORY_FILTERS = frozenset({"kind", "source", "memory_type", "tag", "scope", "project_id"})
_ALLOWED_MEMORY_LIST_PARAMETERS = _ALLOWED_MEMORY_FILTERS | {"cursor", "limit", "order"}
_ALLOWED_MEMORY_SEARCH_PARAMETERS = _ALLOWED_MEMORY_FILTERS | {"q", "limit"}
_ENROLLMENT_REQUEST_FIELDS = frozenset({"enrollment_token", "device_display_name"})
_COMMAND_REQUEST_FIELDS = frozenset({"client_message_id", "body"})
_SETTINGS_REQUEST_FIELDS = frozenset({"model", "timeout_seconds", "reset"})
_WORKSPACE_REQUEST_FIELDS = frozenset({"path"})
_WORKSPACE_CONFIG_REQUEST_FIELDS = frozenset({"field", "value", "path"})
_WORKSPACE_CONFIG_RESET_FIELDS = frozenset({"reset", "path"})
_DECIMAL_INTEGER = re.compile(r"^[0-9]+$")
_CLIENT_MESSAGE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_EVENT_BATCH_SIZE = 100
_SSE_RETRY_MILLISECONDS = 2000
# Trace entries served per /trace response; the client pages with
# after_seq until has_more is false.
_TRACE_PAGE_SIZE = 200


class WorkshopClientAuthenticator(Protocol):
    """Resolve a human Workshop principal from a client request."""

    async def authenticate(self, request: web.Request) -> PrincipalId | None: ...

    async def authenticate_token(self, token: str) -> PrincipalId | None: ...


class WorkshopClientCommandSubmitter(Protocol):
    async def submit(
        self,
        message: ClientInboundMessage,
        *,
        artifact: StagedArtifact | None = None,
    ) -> ClientCommandSubmission: ...

    async def state(self, run_id: RunId) -> DurableRun: ...

    async def cancel(self, run_id: RunId) -> CanonicalCancellationDisposition: ...


def _serialize_settings_workspace(
    snapshot: SettingsWorkspaceSnapshot,
) -> dict[str, object]:
    return {
        "version": 1,
        "principal_id": str(snapshot.principal_id),
        "channel_id": str(snapshot.channel_id),
        "runtime_profile_id": str(snapshot.runtime_profile_id),
        "backend": snapshot.backend,
        "provider": snapshot.provider,
        "model": {
            "value": snapshot.model.value,
            "source": snapshot.model.source,
        },
        "timeout_seconds": {
            "value": snapshot.timeout_seconds.value,
            "source": snapshot.timeout_seconds.source,
        },
        "workspace": snapshot.workspace,
        "model_options": (
            [
                {
                    "model_id": option.model_id,
                    "display_name": option.display_name,
                }
                for option in snapshot.model_options
            ]
            if snapshot.model_options is not None
            else None
        ),
        "workspaces": [
            {
                "path": option.path,
                "name": option.name,
                "current": option.current,
                "home": option.home,
            }
            for option in snapshot.workspaces
        ],
    }


def _serialize_workspace_config(
    snapshot: WorkspaceConfigSnapshot,
) -> dict[str, object]:
    return {
        "version": 1,
        "workspace": snapshot.workspace,
        "model": {
            "value": snapshot.model.value,
            "source": snapshot.model.source,
        },
        "timeout_seconds": {
            "value": snapshot.timeout_seconds.value,
            "source": snapshot.timeout_seconds.source,
        },
        "environment_keys": list(snapshot.environment_keys),
        "prompt": snapshot.prompt,
        "has_prompt": snapshot.has_prompt,
        "prompt_source": snapshot.prompt_source,
        "override_fields": list(snapshot.override_fields),
    }


def _serialize_memory_record(record: MemoryRecordSummary) -> dict[str, object]:
    return {
        "memory_id": record.memory_id,
        "kind": record.kind,
        "source": record.source,
        "memory_type": record.memory_type,
        "preview": record.preview,
        "tags": list(record.tags),
        "speaker": record.speaker,
        "confidence": record.confidence,
        "created_at": record.created_at,
        "updated_at": record.updated_at,
        "scope": {
            "scope": record.scope.scope,
            "project_id": record.scope.project_id,
            "scope_confidence": record.scope.scope_confidence,
            "scope_source": record.scope.scope_source,
            "legacy_defaulted": record.scope.legacy_defaulted,
            "invalid_defaulted": record.scope.invalid_defaulted,
            "retrievable": record.scope.retrievable,
            "exclusion_reason": record.scope.exclusion_reason,
        },
    }


def _serialize_memory_detail(detail: MemoryRecordDetail) -> dict[str, object]:
    return {
        **_serialize_memory_record(detail.record),
        "content": detail.content,
        "compact_recall": detail.compact_recall,
        "confirmation_quote": detail.confirmation_quote,
        "prompt_version": detail.prompt_version,
        "episode": detail.episode,
    }


def _serialize_memory_source(context: MemorySourceContext) -> dict[str, object]:
    def serialize_message(message: MemorySourceMessage | None) -> dict[str, object] | None:
        if message is None:
            return None
        return {
            "message_id": str(message.message_id),
            "channel_id": str(message.channel_id),
            "author_principal_id": str(message.author_principal_id),
            "author_kind": message.author_kind,
            "author_display_name": message.author_display_name,
            "body": message.body,
            "created_at": message.created_at,
        }

    return {
        "status": context.status,
        "reason": context.reason,
        "run_id": str(context.run_id) if context.run_id is not None else None,
        "source": serialize_message(context.source),
        "result": serialize_message(context.result),
    }


def _serialize_memory_mutation(batch: MemoryMutationBatch) -> dict[str, object]:
    def serialize_scope(scope) -> dict[str, object] | None:
        if scope is None:
            return None
        return {
            "scope": scope.scope,
            "project_id": scope.project_id,
            "scope_confidence": scope.scope_confidence,
            "scope_source": scope.scope_source,
            "legacy_defaulted": scope.legacy_defaulted,
            "invalid_defaulted": scope.invalid_defaulted,
            "retrievable": scope.retrievable,
            "exclusion_reason": scope.exclusion_reason,
        }

    return {
        "version": 1,
        "operation": batch.operation,
        "results": [
            {
                "memory_id": result.memory_id,
                "outcome": result.outcome,
                "prior_scope": serialize_scope(result.prior_scope),
                "new_scope": serialize_scope(result.new_scope),
            }
            for result in batch.results
        ],
    }


def _memory_filters(request: web.Request) -> MemoryQueryFilters:
    values: dict[str, str | None] = {}
    for field in _ALLOWED_MEMORY_FILTERS:
        values[field] = _single_query_value(request, field)
    return MemoryQueryFilters(**values)


def _memory_limit(request: web.Request, *, default: int) -> int:
    raw = _single_query_value(request, "limit")
    if raw is None:
        return default
    if not _DECIMAL_INTEGER.fullmatch(raw):
        raise WorkshopMemoryValidationError("Invalid memory limit")
    return int(raw)


async def _memory_authority(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopMemoryQueryService,
) -> tuple[MemoryQueryAuthority | None, web.Response | None]:
    principal_id = await authenticator.authenticate(request)
    if not isinstance(principal_id, PrincipalId):
        response = _error_response(
            status=401,
            code="authentication_required",
            message="Authentication required",
        )
        response.headers["WWW-Authenticate"] = "Bearer"
        return None, response
    try:
        return service.authority_for_principal(principal_id), None
    except WorkshopMemoryAccessDenied:
        return None, _error_response(
            status=403,
            code="access_denied",
            message="Access denied",
        )


def _memory_error_response(exc: Exception) -> web.Response:
    if isinstance(exc, WorkshopMemoryNotFound):
        return _error_response(status=404, code="memory_not_found", message="Memory not found")
    if isinstance(exc, WorkshopMemoryResponseTooLarge):
        return _error_response(status=413, code="response_too_large", message=str(exc))
    if isinstance(exc, WorkshopMemoryAccessDenied):
        return _error_response(status=403, code="access_denied", message="Access denied")
    if isinstance(exc, WorkshopMemoryValidationError):
        return _error_response(status=400, code="invalid_memory_query", message=str(exc))
    return _error_response(
        status=400,
        code="invalid_memory_query",
        message="Invalid memory query",
    )


async def _handle_memory_stats(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopMemoryQueryService,
) -> web.Response:
    authority, error = await _memory_authority(request, authenticator=authenticator, service=service)
    if error is not None:
        return error
    if request.query or request.can_read_body:
        return _error_response(status=400, code="invalid_request", message="Invalid memory request")
    assert authority is not None
    stats = await service.stats(authority)
    return _json_response(
        {
            "version": 1,
            "stats": {
                "total": stats.total,
                "facts": stats.facts,
                "episodes": stats.episodes,
                "by_source": stats.by_source,
                "by_type": stats.by_type,
                "by_scope": stats.by_scope,
                "allowed_projects": [
                    {
                        "project_id": project.project_id,
                        "display_name": project.display_name,
                    }
                    for project in stats.allowed_projects
                ],
            },
        },
        status=200,
    )


async def _handle_memory_records(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopMemoryQueryService,
) -> web.Response:
    authority, error = await _memory_authority(request, authenticator=authenticator, service=service)
    if error is not None:
        return error
    if not set(request.query).issubset(_ALLOWED_MEMORY_LIST_PARAMETERS) or request.can_read_body:
        return _error_response(status=400, code="invalid_request", message="Invalid memory request")
    assert authority is not None
    try:
        page = await service.list_records(
            authority,
            filters=_memory_filters(request),
            limit=_memory_limit(request, default=DEFAULT_PAGE_SIZE),
            cursor=_single_query_value(request, "cursor"),
            order=_single_query_value(request, "order") or "newest",
        )
    except (ValueError, WorkshopMemoryValidationError) as exc:
        return _memory_error_response(exc)
    return _json_response(
        {
            "version": 1,
            "records": [_serialize_memory_record(record) for record in page.records],
            "next_cursor": page.next_cursor,
        },
        status=200,
    )


async def _handle_memory_search(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopMemoryQueryService,
) -> web.Response:
    authority, error = await _memory_authority(request, authenticator=authenticator, service=service)
    if error is not None:
        return error
    if not set(request.query).issubset(_ALLOWED_MEMORY_SEARCH_PARAMETERS) or request.can_read_body:
        return _error_response(status=400, code="invalid_request", message="Invalid memory request")
    assert authority is not None
    try:
        query = _single_query_value(request, "q")
        if query is None:
            raise WorkshopMemoryValidationError("Memory search query is required")
        snapshot = await service.search(
            authority,
            query,
            filters=_memory_filters(request),
            limit=_memory_limit(request, default=10),
        )
    except (ValueError, WorkshopMemoryQueryError) as exc:
        return _memory_error_response(exc)
    return _json_response(
        {
            "version": 1,
            "active_project_id": snapshot.active_project_id,
            "reason": snapshot.reason,
            "hits": [
                {
                    "record": _serialize_memory_record(hit.record),
                    "raw_score": hit.raw_score,
                    "adjusted_score": hit.adjusted_score,
                    "compact_recall": hit.compact_recall,
                }
                for hit in snapshot.hits
            ],
        },
        status=200,
    )


async def _handle_memory_detail(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopMemoryQueryService,
    source: bool,
) -> web.Response:
    authority, error = await _memory_authority(request, authenticator=authenticator, service=service)
    if error is not None:
        return error
    if request.query or request.can_read_body:
        return _error_response(status=400, code="invalid_request", message="Invalid memory request")
    assert authority is not None
    try:
        memory_id = request.match_info["memory_id"]
        if source:
            context = await service.source_context(authority, memory_id)
            payload = {"version": 1, "source_context": _serialize_memory_source(context)}
        else:
            detail = await service.detail(authority, memory_id)
            payload = {"version": 1, "record": _serialize_memory_detail(detail)}
    except (KeyError, WorkshopMemoryQueryError) as exc:
        return _memory_error_response(exc)
    return _json_response(payload, status=200)


async def _memory_mutation_payload(
    request: web.Request,
    *,
    allowed_fields: frozenset[str],
) -> dict[str, object]:
    if request.content_type != "application/json":
        raise WorkshopMemoryValidationError("Content-Type must be application/json")
    try:
        payload = await request.json()
    except (UnicodeDecodeError, ValueError) as exc:
        raise WorkshopMemoryValidationError("Invalid memory mutation request") from exc
    if not isinstance(payload, dict) or set(payload) not in (
        allowed_fields,
        allowed_fields - {"project_id"},
    ):
        raise WorkshopMemoryValidationError("Invalid memory mutation request")
    return payload


async def _handle_memory_scope_mutation(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopMemoryQueryService,
    bulk: bool,
) -> web.Response:
    authority, error = await _memory_authority(request, authenticator=authenticator, service=service)
    if error is not None:
        return error
    if request.query:
        return _error_response(status=400, code="invalid_request", message="Invalid memory mutation request")
    assert authority is not None
    try:
        payload = await _memory_mutation_payload(
            request,
            allowed_fields=(
                frozenset({"memory_ids", "scope", "project_id"}) if bulk else frozenset({"scope", "project_id"})
            ),
        )
        memory_ids = payload.get("memory_ids") if bulk else [request.match_info["memory_id"]]
        if not isinstance(memory_ids, list) or not all(isinstance(item, str) for item in memory_ids):
            raise WorkshopMemoryValidationError("Memory identifiers must be a list")
        scope = payload.get("scope")
        project_id = payload.get("project_id")
        if not isinstance(scope, str) or (project_id is not None and not isinstance(project_id, str)):
            raise WorkshopMemoryValidationError("Invalid memory scope")
        batch = await service.move_scope(
            authority,
            memory_ids,
            scope=scope,
            project_id=project_id,
        )
    except (KeyError, WorkshopMemoryQueryError) as exc:
        return _memory_error_response(exc)
    return _json_response(_serialize_memory_mutation(batch), status=200)


async def _handle_memory_delete_mutation(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopMemoryQueryService,
    bulk: bool,
) -> web.Response:
    authority, error = await _memory_authority(request, authenticator=authenticator, service=service)
    if error is not None:
        return error
    if request.query:
        return _error_response(status=400, code="invalid_request", message="Invalid memory mutation request")
    assert authority is not None
    try:
        if bulk:
            payload = await _memory_mutation_payload(
                request,
                allowed_fields=frozenset({"memory_ids"}),
            )
            memory_ids = payload.get("memory_ids")
            if not isinstance(memory_ids, list) or not all(isinstance(item, str) for item in memory_ids):
                raise WorkshopMemoryValidationError("Memory identifiers must be a list")
        else:
            if request.can_read_body:
                raise WorkshopMemoryValidationError("Delete requests cannot include a body")
            memory_ids = [request.match_info["memory_id"]]
        batch = await service.delete(authority, memory_ids)
    except (KeyError, WorkshopMemoryQueryError) as exc:
        return _memory_error_response(exc)
    return _json_response(_serialize_memory_mutation(batch), status=200)


async def _authenticate_settings_authority(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopSettingsWorkspaceService,
) -> tuple[SettingsWorkspaceAuthority | None, web.Response | None]:
    principal_id = await authenticator.authenticate(request)
    if not isinstance(principal_id, PrincipalId):
        return None, _error_response(
            status=401,
            code="authentication_required",
            message="Authentication required",
        )
    try:
        channel_id = ChannelId(request.match_info["channel_id"])
        authority = service.authority_for_principal_channel(
            principal_id,
            channel_id,
        )
    except (KeyError, TypeError, ValueError):
        return None, _error_response(
            status=400,
            code="invalid_request",
            message="Invalid channel request",
        )
    except WorkshopSettingsWorkspaceAccessDenied:
        return None, _error_response(
            status=403,
            code="access_denied",
            message="Access denied",
        )
    return authority, None


async def _handle_runtime_settings(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopSettingsWorkspaceService,
) -> web.Response:
    authority, error = await _authenticate_settings_authority(
        request,
        authenticator=authenticator,
        service=service,
    )
    if error is not None:
        if error.status == 401:
            error.headers["WWW-Authenticate"] = "Bearer"
        return error
    assert authority is not None
    if request.query or request.can_read_body:
        return _error_response(
            status=400,
            code="invalid_request",
            message="Invalid settings request",
        )
    return _json_response(
        _serialize_settings_workspace(await service.inspect(authority)),
        status=200,
    )


async def _handle_runtime_settings_update(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopSettingsWorkspaceService,
) -> web.Response:
    authority, error = await _authenticate_settings_authority(
        request,
        authenticator=authenticator,
        service=service,
    )
    if error is not None:
        if error.status == 401:
            error.headers["WWW-Authenticate"] = "Bearer"
        return error
    assert authority is not None
    if request.query:
        return _error_response(status=400, code="invalid_request", message="Invalid settings request")
    try:
        payload = await request.json()
    except (UnicodeDecodeError, ValueError):
        return _error_response(status=400, code="invalid_request", message="Invalid JSON request")
    if not isinstance(payload, dict) or not payload or not set(payload).issubset(_SETTINGS_REQUEST_FIELDS):
        return _error_response(status=400, code="invalid_request", message="Invalid settings request")
    operations = sum(field in payload for field in _SETTINGS_REQUEST_FIELDS)
    if operations != 1:
        return _error_response(
            status=400,
            code="invalid_request",
            message="Change exactly one setting at a time",
        )
    try:
        if "model" in payload:
            if not isinstance(payload["model"], str):
                raise WorkshopSettingsWorkspaceValidationError("Model must be text")
            snapshot = await service.set_model(authority, payload["model"])
        elif "timeout_seconds" in payload:
            snapshot = await service.set_timeout(
                authority,
                payload["timeout_seconds"],
            )
        else:
            reset = payload["reset"]
            if reset not in {"model", "timeout", "all"}:
                raise WorkshopSettingsWorkspaceValidationError("Reset must be model, timeout, or all")
            snapshot = await service.reset_settings(
                authority,
                None if reset == "all" else reset,
            )
    except WorkshopSettingsWorkspaceValidationError as exc:
        return _error_response(
            status=400,
            code="invalid_setting",
            message=str(exc),
        )
    return _json_response(_serialize_settings_workspace(snapshot), status=200)


async def _handle_active_workspace_update(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopSettingsWorkspaceService,
) -> web.Response:
    authority, error = await _authenticate_settings_authority(
        request,
        authenticator=authenticator,
        service=service,
    )
    if error is not None:
        if error.status == 401:
            error.headers["WWW-Authenticate"] = "Bearer"
        return error
    assert authority is not None
    if request.query:
        return _error_response(status=400, code="invalid_request", message="Invalid workspace request")
    try:
        payload = await request.json()
    except (UnicodeDecodeError, ValueError):
        return _error_response(status=400, code="invalid_request", message="Invalid JSON request")
    if (
        not isinstance(payload, dict)
        or set(payload) != _WORKSPACE_REQUEST_FIELDS
        or not isinstance(payload.get("path"), str)
    ):
        return _error_response(status=400, code="invalid_request", message="Invalid workspace request")
    try:
        snapshot = await service.switch_workspace(authority, payload["path"])
    except WorkshopSettingsWorkspaceAccessDenied:
        return _error_response(status=403, code="access_denied", message="Access denied")
    except WorkshopSettingsWorkspaceValidationError as exc:
        return _error_response(status=400, code="invalid_workspace", message=str(exc))
    return _json_response(_serialize_settings_workspace(snapshot), status=200)


async def _handle_workspace_config(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopSettingsWorkspaceService,
) -> web.Response:
    authority, error = await _authenticate_settings_authority(
        request,
        authenticator=authenticator,
        service=service,
    )
    if error is not None:
        if error.status == 401:
            error.headers["WWW-Authenticate"] = "Bearer"
        return error
    assert authority is not None
    if request.query or request.can_read_body:
        return _error_response(
            status=400,
            code="invalid_request",
            message="Invalid workspace config request",
        )
    try:
        snapshot = await service.workspace_config(authority)
    except WorkshopSettingsWorkspaceAccessDenied:
        return _error_response(status=403, code="access_denied", message="Access denied")
    return _json_response(_serialize_workspace_config(snapshot), status=200)


async def _handle_workspace_config_update(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopSettingsWorkspaceService,
) -> web.Response:
    authority, error = await _authenticate_settings_authority(
        request,
        authenticator=authenticator,
        service=service,
    )
    if error is not None:
        if error.status == 401:
            error.headers["WWW-Authenticate"] = "Bearer"
        return error
    assert authority is not None
    if request.query:
        return _error_response(status=400, code="invalid_request", message="Invalid workspace config request")
    try:
        payload = await request.json()
    except (UnicodeDecodeError, ValueError):
        return _error_response(status=400, code="invalid_request", message="Invalid JSON request")
    if not isinstance(payload, dict):
        return _error_response(status=400, code="invalid_request", message="Invalid workspace config request")
    keys = set(payload)
    path = payload.get("path")
    if path is not None and not isinstance(path, str):
        return _error_response(status=400, code="invalid_request", message="Invalid workspace config request")
    try:
        if "reset" in payload:
            if not keys.issubset(_WORKSPACE_CONFIG_RESET_FIELDS):
                raise WorkshopSettingsWorkspaceValidationError("Invalid workspace config reset")
            reset = payload["reset"]
            if reset not in {"model", "timeout", "env", "prompt", "all"}:
                raise WorkshopSettingsWorkspaceValidationError("Invalid workspace config reset")
            snapshot = await service.reset_workspace_config(
                authority,
                field=None if reset == "all" else reset,
                workspace_path=path,
            )
        else:
            if (
                not keys.issubset(_WORKSPACE_CONFIG_REQUEST_FIELDS)
                or "field" not in payload
                or "value" not in payload
                or not isinstance(payload["field"], str)
                or not isinstance(payload["value"], str)
            ):
                raise WorkshopSettingsWorkspaceValidationError("Invalid workspace config change")
            snapshot = await service.set_workspace_config(
                authority,
                field=payload["field"],
                value=payload["value"],
                workspace_path=path,
            )
    except WorkshopSettingsWorkspaceAccessDenied:
        return _error_response(status=403, code="access_denied", message="Access denied")
    except WorkshopSettingsWorkspaceValidationError as exc:
        return _error_response(status=400, code="invalid_setting", message=str(exc))
    return _json_response(_serialize_workspace_config(snapshot), status=200)


class WorkshopEnrollmentRateLimiter:
    """Bound enrollment attempts by source and across the whole process.

    Cloudflare Tunnel connects to Kai over loopback, so ``request.remote`` is
    normally the tunnel process. ``CF-Connecting-IP`` is used only as a
    rate-limit partition when it contains one valid address; it never grants
    identity or authorization. A global ceiling still applies when a local
    caller spoofs or rotates that advisory header.
    """

    def __init__(
        self,
        *,
        per_source_limit: int = 10,
        global_limit: int = 120,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if per_source_limit < 1 or global_limit < per_source_limit or window_seconds <= 0:
            raise ValueError("Enrollment rate-limit bounds are invalid")
        self._per_source_limit = per_source_limit
        self._global_limit = global_limit
        self._window_seconds = window_seconds
        self._clock = clock
        self._global_attempts: deque[float] = deque()
        self._source_attempts: dict[str, deque[float]] = {}

    @staticmethod
    def _source(request: web.Request) -> str:
        forwarded = request.headers.getall("CF-Connecting-IP", [])
        if len(forwarded) == 1:
            try:
                return str(ipaddress.ip_address(forwarded[0].strip()))
            except ValueError:
                pass
        remote = request.remote
        if remote:
            try:
                return str(ipaddress.ip_address(remote))
            except ValueError:
                return "peer:unknown"
        return "peer:unknown"

    def check(self, request: web.Request) -> int | None:
        """Record one attempt, or return whole seconds until retry is allowed."""
        now = self._clock()
        cutoff = now - self._window_seconds
        while self._global_attempts and self._global_attempts[0] <= cutoff:
            self._global_attempts.popleft()

        for prior_source, prior_attempts in list(self._source_attempts.items()):
            while prior_attempts and prior_attempts[0] <= cutoff:
                prior_attempts.popleft()
            if not prior_attempts:
                del self._source_attempts[prior_source]

        source = self._source(request)
        attempts = self._source_attempts.setdefault(source, deque())

        blocked_until: float | None = None
        if len(self._global_attempts) >= self._global_limit:
            blocked_until = self._global_attempts[0] + self._window_seconds
        if len(attempts) >= self._per_source_limit:
            source_until = attempts[0] + self._window_seconds
            blocked_until = max(blocked_until or source_until, source_until)
        if blocked_until is not None:
            return max(1, math.ceil(blocked_until - now))

        self._global_attempts.append(now)
        attempts.append(now)
        return None


class WorkshopEventStreamLimiter:
    """Bound concurrent long-lived streams per principal and process."""

    def __init__(self, *, per_principal_limit: int = 4, global_limit: int = 32) -> None:
        if per_principal_limit < 1 or global_limit < per_principal_limit:
            raise ValueError("Event-stream concurrency bounds are invalid")
        self._per_principal_limit = per_principal_limit
        self._global_limit = global_limit
        self._active_total = 0
        self._active_by_principal: dict[PrincipalId, int] = {}

    def acquire(self, principal_id: PrincipalId) -> bool:
        if not isinstance(principal_id, PrincipalId):
            return False
        active_for_principal = self._active_by_principal.get(principal_id, 0)
        if self._active_total >= self._global_limit or active_for_principal >= self._per_principal_limit:
            return False
        self._active_total += 1
        self._active_by_principal[principal_id] = active_for_principal + 1
        return True

    def release(self, principal_id: PrincipalId) -> None:
        active_for_principal = self._active_by_principal.get(principal_id, 0)
        if active_for_principal < 1 or self._active_total < 1:
            raise RuntimeError("Event-stream capacity was released without an active claim")
        if active_for_principal == 1:
            del self._active_by_principal[principal_id]
        else:
            self._active_by_principal[principal_id] = active_for_principal - 1
        self._active_total -= 1


def _apply_client_security_headers(response: web.StreamResponse) -> None:
    response.headers["Cache-Control"] = "private, no-store"
    response.headers["Content-Security-Policy"] = "default-src 'none'"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["X-Content-Type-Options"] = "nosniff"


def _json_response(payload: dict[str, object], *, status: int) -> web.Response:
    response = web.json_response(payload, status=status)
    _apply_client_security_headers(response)
    return response


def _error_response(*, status: int, code: str, message: str) -> web.Response:
    return _json_response({"error": {"code": code, "message": message}}, status=status)


def _single_query_value(request: web.Request, name: str) -> str | None:
    values = request.query.getall(name, [])
    if len(values) > 1:
        raise ValueError(f"Duplicate {name} parameter")
    return values[0] if values else None


def _parse_timeline_request(request: web.Request) -> tuple[ChannelId, str | None, int, bool]:
    if not set(request.query).issubset(_ALLOWED_TIMELINE_QUERY_PARAMETERS):
        raise ValueError("Unsupported query parameter")

    channel_id = ChannelId(request.match_info["channel_id"])
    cursor = _single_query_value(request, "cursor")
    limit_value = _single_query_value(request, "limit")
    if limit_value is None:
        limit = 50
    elif not _DECIMAL_INTEGER.fullmatch(limit_value):
        raise ValueError("Invalid limit")
    else:
        limit = int(limit_value)
    tail_value = _single_query_value(request, "tail")
    if tail_value is None:
        tail = False
    elif tail_value != "1":
        raise ValueError("Invalid tail flag")
    else:
        tail = True
    # A cursor already encodes its direction; combining the two would make
    # the request ambiguous, so it is rejected rather than resolved.
    if tail and cursor is not None:
        raise ValueError("tail requests must not carry a cursor")
    return channel_id, cursor, limit, tail


def _parse_event_stream_request(request: web.Request) -> tuple[ChannelId, int | None]:
    if not set(request.query).issubset(_ALLOWED_EVENT_QUERY_PARAMETERS):
        raise ValueError("Unsupported query parameter")

    channel_id = ChannelId(request.match_info["channel_id"])
    last_event_ids = request.headers.getall("Last-Event-ID", [])
    if len(last_event_ids) > 1:
        raise ValueError("Duplicate Last-Event-ID header")
    value = last_event_ids[0] if last_event_ids else _single_query_value(request, "after_position")
    if value is None:
        return channel_id, None
    if not _DECIMAL_INTEGER.fullmatch(value):
        raise ValueError("Invalid timeline resume position")
    return channel_id, int(value)


def _format_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _serialize_message(message: TimelineMessage) -> dict[str, object]:
    return {
        "message_id": str(message.message_id),
        "channel_id": str(message.channel_id),
        "author_principal_id": str(message.author_principal_id),
        "author_kind": message.author_kind,
        "author_display_name": message.author_display_name,
        "reply_to_message_id": (str(message.reply_to_message_id) if message.reply_to_message_id is not None else None),
        "body": message.body,
        "event_position": message.event_position,
        "created_at": _format_timestamp(message.created_at),
        "artifacts": [_serialize_artifact(artifact) for artifact in message.artifacts],
    }


def _serialize_artifact(artifact: ArtifactSummary) -> dict[str, object]:
    return {
        "artifact_id": str(artifact.artifact_id),
        "kind": artifact.kind,
        "media_type": artifact.media_type,
        "byte_size": artifact.byte_size,
        "content_sha256": artifact.content_sha256,
        "original_filename": artifact.original_filename,
        "created_at": _format_timestamp(artifact.created_at),
    }


def _serialize_run(run: DurableRun) -> dict[str, object]:
    return {
        "run_id": str(run.run_id),
        "channel_id": str(run.channel_id),
        "status": run.status.value,
        "accepted_at": _format_timestamp(run.accepted_at),
        "started_at": _format_timestamp(run.started_at) if run.started_at is not None else None,
        "terminal_at": _format_timestamp(run.terminal_at) if run.terminal_at is not None else None,
        "terminal_code": run.terminal_code,
        "cancellation_requested_at": (
            _format_timestamp(run.cancellation_requested_at) if run.cancellation_requested_at is not None else None
        ),
        "result_message_id": str(run.result_message_id) if run.result_message_id is not None else None,
    }


async def _handle_client_navigation(
    request: web.Request,
    *,
    store: WorkshopEventStore,
    authenticator: WorkshopClientAuthenticator,
) -> web.Response:
    """Return only the Workshops and channels explicitly visible to a human."""
    principal_id = await authenticator.authenticate(request)
    if not isinstance(principal_id, PrincipalId):
        response = _error_response(
            status=401,
            code="authentication_required",
            message="Authentication required",
        )
        response.headers["WWW-Authenticate"] = "Bearer"
        return response
    if request.query or request.can_read_body:
        return _error_response(
            status=400,
            code="invalid_request",
            message="Invalid navigation request",
        )

    async with store.connection.execute(
        "SELECT display_name FROM principals WHERE id = ? AND kind = 'human'",
        (principal_id,),
    ) as cursor:
        principal_row = await cursor.fetchone()
    if principal_row is None:
        return _error_response(status=403, code="access_denied", message="Access denied")

    async with store.connection.execute(
        "SELECT w.id, w.name, wm.role FROM workshop_memberships wm "
        "JOIN workshops w ON w.id = wm.workshop_id "
        "WHERE wm.principal_id = ? ORDER BY lower(w.name), w.id",
        (principal_id,),
    ) as cursor:
        workshop_rows = list(await cursor.fetchall())
    async with store.connection.execute(
        "SELECT c.workshop_id, c.id, c.kind, c.name, cm.role, a.id, a.name, "
        "CASE WHEN cara.id IS NULL THEN 0 ELSE 1 END "
        "FROM channel_memberships cm "
        "JOIN channels c ON c.id = cm.channel_id "
        "JOIN workshop_memberships wm ON wm.workshop_id = c.workshop_id "
        "AND wm.principal_id = cm.principal_id "
        "LEFT JOIN channel_agents ca ON ca.channel_id = c.id "
        "LEFT JOIN agents a ON a.id = ca.agent_id AND a.workshop_id = c.workshop_id "
        "LEFT JOIN channel_agent_runtime_assignments cara "
        "ON cara.channel_id = c.id AND cara.agent_id = a.id "
        "WHERE cm.principal_id = ? "
        "ORDER BY c.workshop_id, "
        "CASE c.kind WHEN 'direct' THEN 0 WHEN 'group' THEN 1 "
        "WHEN 'notification' THEN 2 ELSE 3 END, lower(coalesce(c.name, '')), c.id, a.name, a.id",
        (principal_id,),
    ) as cursor:
        channel_rows = list(await cursor.fetchall())
    async with store.connection.execute(
        "SELECT c.workshop_id, c.id, peer.id, peer.kind, peer.display_name "
        "FROM channel_memberships own_cm "
        "JOIN channels c ON c.id = own_cm.channel_id "
        "JOIN workshop_memberships wm ON wm.workshop_id = c.workshop_id "
        "AND wm.principal_id = own_cm.principal_id "
        "JOIN channel_memberships peer_cm ON peer_cm.channel_id = c.id "
        "AND peer_cm.principal_id != own_cm.principal_id "
        "JOIN principals peer ON peer.id = peer_cm.principal_id "
        "WHERE own_cm.principal_id = ? "
        "ORDER BY c.workshop_id, c.id, lower(peer.display_name), peer.id",
        (principal_id,),
    ) as cursor:
        participant_rows = list(await cursor.fetchall())

    channels_by_workshop: dict[str, dict[str, dict[str, object]]] = {}
    for row in channel_rows:
        workshop_id = str(row[0])
        channel_id = str(row[1])
        workshop_channels = channels_by_workshop.setdefault(workshop_id, {})
        channel = workshop_channels.setdefault(
            channel_id,
            {
                "channel_id": channel_id,
                "name": str(row[3]) if row[3] is not None else None,
                "kind": str(row[2]),
                "role": str(row[4]),
                "agents": [],
                "participants": [],
                "_runtime_assignments": [],
            },
        )
        if row[5] is not None:
            agents = channel["agents"]
            assignments = channel["_runtime_assignments"]
            if not isinstance(agents, list) or not isinstance(assignments, list):
                raise RuntimeError("Workshop navigation channel assembly failed")
            agents.append({"agent_id": str(row[5]), "name": str(row[6])})
            assignments.append(bool(row[7]))

    for row in participant_rows:
        workshop_channels = channels_by_workshop.get(str(row[0]))
        channel = workshop_channels.get(str(row[1])) if workshop_channels is not None else None
        if channel is None:
            raise RuntimeError("Workshop navigation participant assembly failed")
        participants = channel["participants"]
        if not isinstance(participants, list):
            raise RuntimeError("Workshop navigation participant assembly failed")
        participants.append(
            {
                "principal_id": str(row[2]),
                "kind": str(row[3]),
                "display_name": str(row[4]),
            }
        )

    workshops: list[dict[str, object]] = []
    for row in workshop_rows:
        workshop_id = str(row[0])
        visible_channels: list[dict[str, object]] = []
        for channel in channels_by_workshop.get(workshop_id, {}).values():
            assignments = channel.pop("_runtime_assignments")
            agents = channel["agents"]
            if not isinstance(assignments, list) or not isinstance(agents, list):
                raise RuntimeError("Workshop navigation capability assembly failed")
            channel["can_submit_commands"] = (
                channel["kind"] in {"direct", "group"} and len(agents) == 1 and assignments == [True]
            )
            visible_channels.append(channel)
        workshops.append(
            {
                "workshop_id": workshop_id,
                "name": str(row[1]),
                "role": str(row[2]),
                "channels": visible_channels,
            }
        )

    return _json_response(
        {
            "version": 1,
            "principal": {
                "principal_id": str(principal_id),
                "display_name": str(principal_row[0]),
            },
            "workshops": workshops,
        },
        status=200,
    )


async def _handle_channel_timeline(
    request: web.Request,
    *,
    store: WorkshopEventStore,
    authenticator: WorkshopClientAuthenticator,
) -> web.Response:
    principal_id = await authenticator.authenticate(request)
    if not isinstance(principal_id, PrincipalId):
        response = _error_response(
            status=401,
            code="authentication_required",
            message="Authentication required",
        )
        response.headers["WWW-Authenticate"] = "Bearer"
        return response

    try:
        channel_id, cursor, limit, tail = _parse_timeline_request(request)
        page = await read_channel_timeline(
            store,
            principal_id=principal_id,
            channel_id=channel_id,
            authorizer=CanonicalChannelAuthorizer(store),
            cursor=cursor,
            limit=limit,
            tail=tail,
        )
    except TimelineAccessDeniedError:
        return _error_response(status=403, code="access_denied", message="Access denied")
    except (TimelineCursorError, TypeError, ValueError):
        return _error_response(
            status=400,
            code="invalid_request",
            message="Invalid timeline request",
        )

    return _json_response(
        {
            "version": 1,
            "channel_id": str(channel_id),
            "messages": [_serialize_message(message) for message in page.messages],
            "next_cursor": page.next_cursor,
            "previous_cursor": page.previous_cursor,
            "through_position": page.through_position,
        },
        status=200,
    )


def _serialize_timeline_event(message: TimelineMessage) -> bytes:
    payload = json.dumps(
        {
            "version": 1,
            "channel_id": str(message.channel_id),
            "message": _serialize_message(message),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (f"id: {message.event_position}\nevent: timeline.message.created\ndata: {payload}\n\n").encode()


def _serialize_run_lifecycle_event(activity: ClientRunLifecycleEvent) -> bytes:
    payload = json.dumps(
        {
            "version": 1,
            "channel_id": str(activity.run.channel_id),
            "event_position": activity.event_position,
            "transition": activity.transition.value,
            "occurred_at": _format_timestamp(activity.occurred_at),
            "run": _serialize_run(activity.run),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (f"id: {activity.event_position}\nevent: run.lifecycle.changed\ndata: {payload}\n\n").encode()


def _serialize_run_preview_event(preview: RunPreview) -> bytes:
    """Render one ephemeral preview event.

    Deliberately carries no SSE `id:` line: the client's Last-Event-ID
    resume cursor must remain a durable store position, and previews are
    advisory display state that never participates in resume.
    """
    payload = json.dumps(
        {
            "version": 1,
            "channel_id": str(preview.channel_id),
            "run_id": str(preview.run_id),
            "sequence": preview.sequence,
            "text": preview.text,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (f"event: run.preview.updated\ndata: {payload}\n\n").encode()


def _serialize_run_trace_event(run_id: str, channel_id: str, seq: int) -> bytes:
    """Render one trace doorbell event.

    Deliberately carries no SSE `id:` line, for exactly the reason the
    preview event omits it: the client's Last-Event-ID resume cursor
    must remain a durable store position, and a missed doorbell costs
    nothing because the trace endpoint is the source of truth.
    """
    payload = json.dumps(
        {
            "version": 1,
            "channel_id": channel_id,
            "run_id": run_id,
            "seq": seq,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return (f"event: run.trace.updated\ndata: {payload}\n\n").encode()


async def _latest_channel_trace(store: WorkshopEventStore, channel_id: ChannelId) -> tuple[str, int] | None:
    """Return (run_id, seq) of the channel's appending run's trace tip.

    Two indexed lookups (the channel's started run, then that run's
    MAX(seq) via the primary key) rather than a reverse scan of
    run_traces, whose cost would grow with every other channel's trace
    history. The coordinator appends strictly between start and
    settlement, so only a started run can be appending; a queued
    accepted run never masks the executing one. Rows landing just
    before settlement can miss a final doorbell, which the terminal
    run-lifecycle event covers with the card's final refresh.
    """
    async with store.connection.execute(
        "SELECT id FROM runs WHERE channel_id = ? AND status = 'started' ORDER BY accepted_at DESC, id DESC LIMIT 1",
        (str(channel_id),),
    ) as cursor:
        run_row = await cursor.fetchone()
    if run_row is None:
        return None
    run_id = str(run_row[0])
    async with store.connection.execute(
        "SELECT MAX(seq) FROM run_traces WHERE run_id = ?",
        (run_id,),
    ) as cursor:
        seq_row = await cursor.fetchone()
    if seq_row is None or seq_row[0] is None:
        return None
    return run_id, int(seq_row[0])


async def _authorized_update_batch(
    request: web.Request,
    *,
    store: WorkshopEventStore,
    authenticator: WorkshopClientAuthenticator,
    request_lock: asyncio.Lock,
    expected_principal_id: PrincipalId | None,
    channel_id: ChannelId,
    after_position: int | None,
    reauthenticate: bool,
) -> tuple[PrincipalId | None, ClientChannelEventBatch | None]:
    async with request_lock:
        principal_id = expected_principal_id
        if reauthenticate:
            principal_id = await authenticator.authenticate(request)
            if not isinstance(principal_id, PrincipalId) or principal_id != expected_principal_id:
                return None, None
        if not isinstance(principal_id, PrincipalId):
            return None, None
        try:
            batch = await read_client_channel_events(
                store,
                principal_id=principal_id,
                channel_id=channel_id,
                authorizer=CanonicalChannelAuthorizer(store),
                after_position=after_position,
                limit=_EVENT_BATCH_SIZE,
            )
        except TimelineAccessDeniedError:
            return principal_id, None
    return principal_id, batch


async def _handle_channel_event_stream(
    request: web.Request,
    *,
    store: WorkshopEventStore,
    authenticator: WorkshopClientAuthenticator,
    request_lock: asyncio.Lock,
    poll_interval: float,
    heartbeat_interval: float,
    authentication_recheck_interval: float,
    stream_limiter: WorkshopEventStreamLimiter,
    shutdown_event: asyncio.Event,
    run_previews: WorkshopRunPreviewRegistry | None = None,
) -> web.StreamResponse:
    try:
        async with request_lock:
            principal_id = await authenticator.authenticate(request)
            if not isinstance(principal_id, PrincipalId):
                response = _error_response(
                    status=401,
                    code="authentication_required",
                    message="Authentication required",
                )
                response.headers["WWW-Authenticate"] = "Bearer"
                return response
            channel_id, after_position = _parse_event_stream_request(request)
            initial_batch = await read_client_channel_events(
                store,
                principal_id=principal_id,
                channel_id=channel_id,
                authorizer=CanonicalChannelAuthorizer(store),
                after_position=after_position,
                limit=_EVENT_BATCH_SIZE,
            )
    except TimelineAccessDeniedError:
        return _error_response(status=403, code="access_denied", message="Access denied")
    except TimelineResumeError:
        return _error_response(
            status=409,
            code="resynchronization_required",
            message="Timeline resynchronization required",
        )
    except (TypeError, ValueError):
        return _error_response(
            status=400,
            code="invalid_request",
            message="Invalid event-stream request",
        )

    if not stream_limiter.acquire(principal_id):
        response = _error_response(
            status=429,
            code="stream_capacity_exceeded",
            message="Too many active event streams",
        )
        response.headers["Retry-After"] = "5"
        return response

    response = web.StreamResponse(status=200)
    try:
        response.content_type = "text/event-stream"
        response.charset = "utf-8"
        _apply_client_security_headers(response)
        response.headers["X-Accel-Buffering"] = "no"
        await response.prepare(request)

        position = initial_batch.next_position
        batch = initial_batch
        last_heartbeat = time.monotonic()
        last_authentication_check = last_heartbeat
        # (run_id, sequence) of the preview most recently written to this
        # connection, so an unchanged preview is not re-sent every poll.
        last_preview_sent: tuple[str, int] | None = None
        # (run_id, seq) of the trace doorbell most recently written, so
        # the signal fires at most once per poll interval per run and
        # only when the durable trace actually advanced.
        last_trace_sent: tuple[str, int] | None = None
        await response.write(f": connected\nretry: {_SSE_RETRY_MILLISECONDS}\n\n".encode())
        while True:
            if run_previews is not None:
                preview = run_previews.channel_preview(channel_id)
                if preview is not None:
                    preview_key = (str(preview.run_id), preview.sequence)
                    if preview_key != last_preview_sent:
                        await response.write(_serialize_run_preview_event(preview))
                        last_preview_sent = preview_key
                        last_heartbeat = time.monotonic()
            async with request_lock:
                trace_key = await _latest_channel_trace(store, channel_id)
            if trace_key is not None and trace_key != last_trace_sent:
                await response.write(_serialize_run_trace_event(trace_key[0], str(channel_id), trace_key[1]))
                last_trace_sent = trace_key
                last_heartbeat = time.monotonic()
            if batch.events:
                for event in batch.events:
                    if isinstance(event, ClientTimelineMessageEvent):
                        await response.write(_serialize_timeline_event(event.message))
                    else:
                        await response.write(_serialize_run_lifecycle_event(event))
                position = batch.next_position
                batch = ClientChannelEventBatch((), position)
                last_heartbeat = time.monotonic()
                continue

            now = time.monotonic()
            if now - last_heartbeat >= heartbeat_interval:
                await response.write(b": keep-alive\n\n")
                last_heartbeat = now
            try:
                await asyncio.wait_for(shutdown_event.wait(), timeout=poll_interval)
                break
            except TimeoutError:
                pass

            reauthenticate = time.monotonic() - last_authentication_check >= authentication_recheck_interval
            _, next_batch = await _authorized_update_batch(
                request,
                store=store,
                authenticator=authenticator,
                request_lock=request_lock,
                expected_principal_id=principal_id,
                channel_id=channel_id,
                after_position=position,
                reauthenticate=reauthenticate,
            )
            if reauthenticate:
                last_authentication_check = time.monotonic()
            if next_batch is None:
                break
            batch = next_batch
    except (BrokenPipeError, ConnectionResetError):
        pass
    finally:
        stream_limiter.release(principal_id)
    return response


async def _handle_enrollment_redemption(
    request: web.Request,
    *,
    enrollment_manager: WorkshopClientEnrollmentManager,
    rate_limiter: WorkshopEnrollmentRateLimiter,
    request_lock: asyncio.Lock,
) -> web.Response:
    """Exchange one opaque grant without accepting a client identity claim."""
    retry_after = rate_limiter.check(request)
    if retry_after is not None:
        response = _error_response(
            status=429,
            code="rate_limited",
            message="Too many enrollment attempts",
        )
        response.headers["Retry-After"] = str(retry_after)
        return response
    if request.content_type != "application/json":
        return _error_response(
            status=415,
            code="unsupported_media_type",
            message="Content-Type must be application/json",
        )

    try:
        payload = await request.json()
    except (UnicodeDecodeError, ValueError):
        payload = None
    if not isinstance(payload, dict) or set(payload) != _ENROLLMENT_REQUEST_FIELDS:
        return _error_response(
            status=400,
            code="invalid_request",
            message="Invalid enrollment request",
        )

    enrollment_token = payload["enrollment_token"]
    device_display_name = payload["device_display_name"]
    if not isinstance(enrollment_token, str) or not isinstance(device_display_name, str):
        return _error_response(
            status=400,
            code="invalid_request",
            message="Invalid enrollment request",
        )

    try:
        async with request_lock:
            redeemed = await enrollment_manager.redeem_grant(
                enrollment_token,
                device_display_name,
            )
    except EnrollmentGrantUnavailableError:
        # Malformed, unknown, expired, revoked, and reused grants deliberately
        # share one response so this endpoint is not a grant-enumeration oracle.
        return _error_response(
            status=401,
            code="enrollment_unavailable",
            message="Enrollment unavailable",
        )
    except (TypeError, ValueError):
        return _error_response(
            status=400,
            code="invalid_request",
            message="Invalid enrollment request",
        )

    return _json_response(
        {
            "version": 1,
            "device": {
                "device_id": str(redeemed.device.device_id),
                "display_name": redeemed.device.display_name,
            },
            "session": {
                "session_id": str(redeemed.session.session_id),
                "token": redeemed.session.token,
                "expires_at": _format_timestamp(redeemed.session.expires_at),
            },
        },
        status=201,
    )


async def _handle_command_submission(
    request: web.Request,
    *,
    store: WorkshopEventStore,
    authenticator: WorkshopClientAuthenticator,
    submitter: WorkshopClientCommandSubmitter,
    request_lock: asyncio.Lock,
    artifact_service: WorkshopArtifactService | None = None,
) -> web.Response:
    """Authenticate, authorize, and durably enqueue one canonical command."""
    async with request_lock:
        principal_id = await authenticator.authenticate(request)
    if not isinstance(principal_id, PrincipalId):
        response = _error_response(
            status=401,
            code="authentication_required",
            message="Authentication required",
        )
        response.headers["WWW-Authenticate"] = "Bearer"
        return response
    try:
        channel_id = ChannelId(request.match_info["channel_id"])
    except (TypeError, ValueError):
        return _error_response(status=400, code="invalid_request", message="Invalid command request")

    async with request_lock:
        authorized = await CanonicalChannelAuthorizer(store).can_submit_command(principal_id, channel_id)
    if not authorized:
        return _error_response(status=403, code="access_denied", message="Access denied")

    artifact: StagedArtifact | None = None
    if request.content_type == "application/json":
        try:
            payload = await request.json()
        except (UnicodeDecodeError, ValueError):
            payload = None
        if not isinstance(payload, dict) or set(payload) != _COMMAND_REQUEST_FIELDS:
            return _error_response(status=400, code="invalid_request", message="Invalid command request")
        client_message_id = payload["client_message_id"]
        body = payload["body"]
    elif request.content_type == "multipart/form-data" and artifact_service is not None:
        try:
            reader = await request.multipart()
            first = await reader.next()
            if not isinstance(first, BodyPartReader) or first.name != "client_message_id":
                raise ValueError("invalid multipart fields")
            client_message_id = await first.text()
            second = await reader.next()
            if not isinstance(second, BodyPartReader) or second.name != "body":
                raise ValueError("invalid multipart fields")
            body = await second.text()
            if not _CLIENT_MESSAGE_ID_PATTERN.fullmatch(client_message_id) or len(body) > 50_000:
                raise ValueError("invalid multipart command metadata")
            file_field = await reader.next()
            if not isinstance(file_field, BodyPartReader) or file_field.name != "file":
                raise ValueError("invalid multipart fields")

            async def chunks():
                while chunk := await file_field.read_chunk(size=64 * 1024):
                    yield bytes(chunk)

            occurred_at = datetime.now(UTC)
            artifact = await artifact_service.stage_client_upload(
                principal_id=principal_id,
                channel_id=channel_id,
                client_message_id=client_message_id,
                filename=file_field.filename,
                claimed_media_type=file_field.headers.get("Content-Type"),
                chunks=chunks(),
                occurred_at=occurred_at,
            )
            if await reader.next() is not None:
                artifact.discard()
                raise ValueError("invalid multipart fields")
            if not body.strip():
                body = f"[Attached file: {artifact.original_filename or 'upload'}]"
        except ArtifactTooLargeError:
            return _error_response(
                status=413,
                code="artifact_too_large",
                message=f"Attachments must be no larger than {MAX_ARTIFACT_BYTES // (1024 * 1024)} MB",
            )
        except ArtifactAccessDeniedError:
            return _error_response(status=403, code="access_denied", message="Access denied")
        except (OSError, TypeError, ValueError):
            return _error_response(
                status=400,
                code="invalid_request",
                message="Invalid artifact command request",
            )
    else:
        return _error_response(
            status=415,
            code="unsupported_media_type",
            message="Content-Type must be application/json or multipart/form-data",
        )
    if not isinstance(client_message_id, str) or not isinstance(body, str):
        return _error_response(status=400, code="invalid_request", message="Invalid command request")

    try:
        command = ClientInboundMessage(
            principal_id=principal_id,
            channel_id=channel_id,
            client_message_id=client_message_id,
            body=body,
            occurred_at=datetime.now(UTC),
            artifact_source_unique_id=(artifact.source_unique_id if artifact is not None else None),
        )
    except (TypeError, ValueError):
        if artifact is not None:
            artifact.discard()
        return _error_response(status=400, code="invalid_request", message="Invalid command request")

    try:
        result = (
            await submitter.submit(command) if artifact is None else await submitter.submit(command, artifact=artifact)
        )
    except IdempotencyConflictError:
        if artifact is not None:
            artifact.discard()
        return _error_response(
            status=409,
            code="idempotency_conflict",
            message="Command identity conflicts with an existing request",
        )
    except (ConversationCommandAcceptanceError, InboundBindingNotFoundError):
        if artifact is not None:
            artifact.discard()
        return _error_response(
            status=409,
            code="command_state_conflict",
            message="Command could not be accepted in the current channel state",
        )
    except ClientCommandExecutorUnavailableError:
        response = _error_response(
            status=503,
            code="execution_unavailable",
            message="Kai cannot accept Workshop commands right now",
        )
        response.headers["Retry-After"] = "2"
        return response

    terminal = result.run.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}
    return _json_response(
        {
            "version": 2,
            "message_id": str(result.acceptance.command.message.event.envelope.aggregate_id),
            "run_id": str(result.acceptance.run.run_id),
            "acceptance": result.acceptance.command.disposition.value,
            "run": _serialize_run(result.run),
        },
        status=200 if terminal else 202,
    )


def _artifact_file_response(
    artifact: StoredArtifact,
    artifact_id: ArtifactId,
    *,
    disposition: str,
) -> web.FileResponse:
    filename = artifact.summary.original_filename or f"{artifact_id}.bin"
    safe_ascii = re.sub(r"[^A-Za-z0-9._-]", "_", filename) or "artifact.bin"
    response = web.FileResponse(artifact.storage_path)
    response.content_type = artifact.summary.media_type
    response.headers["Content-Disposition"] = (
        f"{disposition}; filename=\"{safe_ascii}\"; filename*=UTF-8''{quote(filename)}"
    )
    _apply_client_security_headers(response)
    return response


async def _handle_artifact_content(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    artifact_service: WorkshopArtifactService,
    request_lock: asyncio.Lock,
) -> web.StreamResponse:
    async with request_lock:
        principal_id = await authenticator.authenticate(request)
    if not isinstance(principal_id, PrincipalId):
        response = _error_response(
            status=401,
            code="authentication_required",
            message="Authentication required",
        )
        response.headers["WWW-Authenticate"] = "Bearer"
        return response
    if request.query or request.can_read_body:
        return _error_response(status=400, code="invalid_request", message="Invalid artifact request")
    try:
        channel_id = ChannelId(request.match_info["channel_id"])
        artifact_id = ArtifactId(request.match_info["artifact_id"])
    except (TypeError, ValueError):
        return _error_response(status=403, code="access_denied", message="Access denied")
    try:
        async with request_lock:
            artifact = await artifact_service.authorized_artifact(
                principal_id,
                channel_id,
                artifact_id,
            )
    except (ArtifactAccessDeniedError, ArtifactStorageBoundaryError):
        return _error_response(status=403, code="access_denied", message="Access denied")

    disposition = (
        "inline"
        if (
            artifact.summary.media_type in {"image/gif", "image/jpeg", "image/png", "image/webp"}
            or artifact.summary.media_type.startswith("audio/")
        )
        else "attachment"
    )
    return _artifact_file_response(artifact, artifact_id, disposition=disposition)


async def _handle_artifact_download(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    artifact_service: WorkshopArtifactService,
    request_lock: asyncio.Lock,
) -> web.StreamResponse:
    if (
        request.query
        or request.content_type != "application/x-www-form-urlencoded"
        or request.content_length is None
        or request.content_length > 512
    ):
        return _error_response(status=400, code="invalid_request", message="Invalid artifact download request")
    try:
        form = await request.post()
    except ValueError:
        return _error_response(status=400, code="invalid_request", message="Invalid artifact download request")
    if set(form) != {"session_token"} or len(form.getall("session_token", [])) != 1:
        return _error_response(status=400, code="invalid_request", message="Invalid artifact download request")
    token = form["session_token"]
    if not isinstance(token, str) or not token:
        response = _error_response(status=401, code="authentication_required", message="Authentication required")
        response.headers["WWW-Authenticate"] = "Bearer"
        return response
    async with request_lock:
        principal_id = await authenticator.authenticate_token(token)
    if not isinstance(principal_id, PrincipalId):
        response = _error_response(status=401, code="authentication_required", message="Authentication required")
        response.headers["WWW-Authenticate"] = "Bearer"
        return response
    try:
        channel_id = ChannelId(request.match_info["channel_id"])
        artifact_id = ArtifactId(request.match_info["artifact_id"])
    except (TypeError, ValueError):
        return _error_response(status=403, code="access_denied", message="Access denied")
    try:
        async with request_lock:
            artifact = await artifact_service.authorized_artifact(
                principal_id,
                channel_id,
                artifact_id,
            )
    except (ArtifactAccessDeniedError, ArtifactStorageBoundaryError):
        return _error_response(status=403, code="access_denied", message="Access denied")
    return _artifact_file_response(artifact, artifact_id, disposition="attachment")


async def _authorized_run(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    submitter: WorkshopClientCommandSubmitter,
    request_lock: asyncio.Lock,
) -> tuple[PrincipalId, ChannelId, DurableRun] | web.Response:
    async with request_lock:
        principal_id = await authenticator.authenticate(request)
    if not isinstance(principal_id, PrincipalId):
        response = _error_response(
            status=401,
            code="authentication_required",
            message="Authentication required",
        )
        response.headers["WWW-Authenticate"] = "Bearer"
        return response
    try:
        channel_id = ChannelId(request.match_info["channel_id"])
        run_id = RunId(request.match_info["run_id"])
    except (TypeError, ValueError):
        return _error_response(status=400, code="invalid_request", message="Invalid run request")
    try:
        run = await submitter.state(run_id)
    except RunNotFoundError:
        return _error_response(status=403, code="access_denied", message="Access denied")
    if run.channel_id != channel_id or run.requested_by_principal_id != principal_id:
        return _error_response(status=403, code="access_denied", message="Access denied")
    return principal_id, channel_id, run


async def _handle_run_state(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    submitter: WorkshopClientCommandSubmitter,
    request_lock: asyncio.Lock,
) -> web.Response:
    if request.query:
        return _error_response(status=400, code="invalid_request", message="Invalid run request")
    authorized = await _authorized_run(
        request,
        authenticator=authenticator,
        submitter=submitter,
        request_lock=request_lock,
    )
    if isinstance(authorized, web.Response):
        return authorized
    return _json_response({"version": 1, "run": _serialize_run(authorized[2])}, status=200)


async def _handle_run_trace(
    request: web.Request,
    *,
    store: WorkshopEventStore,
    authenticator: WorkshopClientAuthenticator,
    submitter: WorkshopClientCommandSubmitter,
    request_lock: asyncio.Lock,
) -> web.Response:
    """Serve one page of a run's durable trace rows, as stored.

    Rows are served raw: the kind vocabulary is the table's (tool_call,
    tool_result, and the synthetic truncated marker), which deliberately
    never grows the shared TraceEntry emitter type.
    """
    if not set(request.query) <= {"after_seq"}:
        return _error_response(status=400, code="invalid_request", message="Invalid trace request")
    after_seq = 0
    try:
        raw_after_seq = _single_query_value(request, "after_seq")
    except ValueError:
        return _error_response(status=400, code="invalid_request", message="Invalid trace request")
    if raw_after_seq is not None:
        if not _DECIMAL_INTEGER.fullmatch(raw_after_seq):
            return _error_response(status=400, code="invalid_request", message="Invalid trace request")
        after_seq = int(raw_after_seq)
    authorized = await _authorized_run(
        request,
        authenticator=authenticator,
        submitter=submitter,
        request_lock=request_lock,
    )
    if isinstance(authorized, web.Response):
        return authorized
    _, channel_id, run = authorized
    async with (
        request_lock,
        store.connection.execute(
            "SELECT seq, kind, tool_name, tool_use_id, summary, detail, is_diff, is_error, created_at "
            "FROM run_traces WHERE run_id = ? AND seq > ? ORDER BY seq LIMIT ?",
            (run.run_id, after_seq, _TRACE_PAGE_SIZE + 1),
        ) as cursor,
    ):
        rows = list(await cursor.fetchall())
    has_more = len(rows) > _TRACE_PAGE_SIZE
    entries = [
        {
            "seq": int(row[0]),
            "kind": str(row[1]),
            "tool_name": None if row[2] is None else str(row[2]),
            "tool_use_id": None if row[3] is None else str(row[3]),
            "summary": str(row[4]),
            "detail": str(row[5]),
            "is_diff": bool(row[6]),
            "is_error": bool(row[7]),
            "created_at": str(row[8]),
        }
        for row in rows[:_TRACE_PAGE_SIZE]
    ]
    return _json_response(
        {
            "version": 1,
            "channel_id": str(channel_id),
            "run_id": str(run.run_id),
            "entries": entries,
            "has_more": has_more,
        },
        status=200,
    )


async def _handle_run_cancellation(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    submitter: WorkshopClientCommandSubmitter,
    request_lock: asyncio.Lock,
) -> web.Response:
    if request.query or request.can_read_body:
        return _error_response(status=400, code="invalid_request", message="Invalid cancellation request")
    authorized = await _authorized_run(
        request,
        authenticator=authenticator,
        submitter=submitter,
        request_lock=request_lock,
    )
    if isinstance(authorized, web.Response):
        return authorized
    _, _, run = authorized
    disposition = await submitter.cancel(run.run_id)
    current = await submitter.state(run.run_id)
    return _json_response(
        {
            "version": 1,
            "cancellation": disposition.value,
            "run": _serialize_run(current),
        },
        status=(
            202
            if disposition == CanonicalCancellationDisposition.REQUESTED
            and current.status not in {RunStatus.CANCELLED, RunStatus.COMPLETED, RunStatus.FAILED}
            else 200
        ),
    )


def register_workshop_read_routes(
    app: web.Application,
    *,
    store: WorkshopEventStore,
    authenticator: WorkshopClientAuthenticator,
    request_lock: asyncio.Lock,
    event_poll_interval: float = 1.0,
    event_heartbeat_interval: float = 15.0,
    event_authentication_recheck_interval: float = 15.0,
    event_stream_limiter: WorkshopEventStreamLimiter | None = None,
    run_previews: WorkshopRunPreviewRegistry | None = None,
    artifact_service: WorkshopArtifactService | None = None,
    settings_workspaces: WorkshopSettingsWorkspaceService | None = None,
    memory_queries: WorkshopMemoryQueryService | None = None,
) -> None:
    """Register authenticated Workshop client routes on an application."""
    if event_poll_interval <= 0 or event_heartbeat_interval <= 0 or event_authentication_recheck_interval <= 0:
        raise ValueError("Event-stream intervals must be positive")
    stream_limiter = event_stream_limiter or WorkshopEventStreamLimiter()
    shutdown_event = asyncio.Event()

    async def stop_event_streams(_app: web.Application) -> None:
        """Release persistent SSE requests before aiohttp drains handlers."""
        shutdown_event.set()

    app.on_shutdown.append(stop_event_streams)

    async def handle_channel_timeline(request: web.Request) -> web.Response:
        async with request_lock:
            return await _handle_channel_timeline(
                request,
                store=store,
                authenticator=authenticator,
            )

    async def handle_client_navigation(request: web.Request) -> web.Response:
        async with request_lock:
            return await _handle_client_navigation(
                request,
                store=store,
                authenticator=authenticator,
            )

    async def handle_channel_event_stream(request: web.Request) -> web.StreamResponse:
        return await _handle_channel_event_stream(
            request,
            store=store,
            authenticator=authenticator,
            request_lock=request_lock,
            poll_interval=event_poll_interval,
            heartbeat_interval=event_heartbeat_interval,
            authentication_recheck_interval=event_authentication_recheck_interval,
            stream_limiter=stream_limiter,
            shutdown_event=shutdown_event,
            run_previews=run_previews,
        )

    app.router.add_get(_CLIENT_NAVIGATION_PATH, handle_client_navigation)
    app.router.add_get(_TIMELINE_PATH, handle_channel_timeline)
    app.router.add_get(_TIMELINE_EVENTS_PATH, handle_channel_event_stream)
    if artifact_service is not None:

        async def handle_artifact_content(request: web.Request) -> web.StreamResponse:
            return await _handle_artifact_content(
                request,
                authenticator=authenticator,
                artifact_service=artifact_service,
                request_lock=request_lock,
            )

        async def handle_artifact_download(request: web.Request) -> web.StreamResponse:
            return await _handle_artifact_download(
                request,
                authenticator=authenticator,
                artifact_service=artifact_service,
                request_lock=request_lock,
            )

        app.router.add_get(_ARTIFACT_CONTENT_PATH, handle_artifact_content)
        app.router.add_post(_ARTIFACT_DOWNLOAD_PATH, handle_artifact_download)
    if settings_workspaces is not None:

        async def handle_runtime_settings(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_runtime_settings(
                    request,
                    authenticator=authenticator,
                    service=settings_workspaces,
                )

        async def handle_runtime_settings_update(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_runtime_settings_update(
                    request,
                    authenticator=authenticator,
                    service=settings_workspaces,
                )

        async def handle_active_workspace_update(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_active_workspace_update(
                    request,
                    authenticator=authenticator,
                    service=settings_workspaces,
                )

        async def handle_workspace_config(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_workspace_config(
                    request,
                    authenticator=authenticator,
                    service=settings_workspaces,
                )

        async def handle_workspace_config_update(
            request: web.Request,
        ) -> web.Response:
            async with request_lock:
                return await _handle_workspace_config_update(
                    request,
                    authenticator=authenticator,
                    service=settings_workspaces,
                )

        app.router.add_get(_RUNTIME_SETTINGS_PATH, handle_runtime_settings)
        app.router.add_patch(
            _RUNTIME_SETTINGS_PATH,
            handle_runtime_settings_update,
        )
        app.router.add_post(_ACTIVE_WORKSPACE_PATH, handle_active_workspace_update)
        app.router.add_get(_WORKSPACE_CONFIG_PATH, handle_workspace_config)
        app.router.add_patch(
            _WORKSPACE_CONFIG_PATH,
            handle_workspace_config_update,
        )
    if memory_queries is not None:

        async def handle_memory_stats(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_memory_stats(
                    request,
                    authenticator=authenticator,
                    service=memory_queries,
                )

        async def handle_memory_records(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_memory_records(
                    request,
                    authenticator=authenticator,
                    service=memory_queries,
                )

        async def handle_memory_search(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_memory_search(
                    request,
                    authenticator=authenticator,
                    service=memory_queries,
                )

        async def handle_memory_detail(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_memory_detail(
                    request,
                    authenticator=authenticator,
                    service=memory_queries,
                    source=False,
                )

        async def handle_memory_source(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_memory_detail(
                    request,
                    authenticator=authenticator,
                    service=memory_queries,
                    source=True,
                )

        async def handle_memory_scope_mutation(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_memory_scope_mutation(
                    request,
                    authenticator=authenticator,
                    service=memory_queries,
                    bulk=False,
                )

        async def handle_memory_bulk_scope_mutation(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_memory_scope_mutation(
                    request,
                    authenticator=authenticator,
                    service=memory_queries,
                    bulk=True,
                )

        async def handle_memory_delete_mutation(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_memory_delete_mutation(
                    request,
                    authenticator=authenticator,
                    service=memory_queries,
                    bulk=False,
                )

        async def handle_memory_bulk_delete_mutation(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_memory_delete_mutation(
                    request,
                    authenticator=authenticator,
                    service=memory_queries,
                    bulk=True,
                )

        app.router.add_get(_MEMORY_STATS_PATH, handle_memory_stats)
        app.router.add_get(_MEMORY_RECORDS_PATH, handle_memory_records)
        app.router.add_get(_MEMORY_SEARCH_PATH, handle_memory_search)
        app.router.add_get(_MEMORY_DETAIL_PATH, handle_memory_detail)
        app.router.add_get(_MEMORY_SOURCE_PATH, handle_memory_source)
        app.router.add_patch(_MEMORY_SCOPE_PATH, handle_memory_scope_mutation)
        app.router.add_delete(_MEMORY_DETAIL_PATH, handle_memory_delete_mutation)
        app.router.add_post(_MEMORY_BULK_SCOPE_PATH, handle_memory_bulk_scope_mutation)
        app.router.add_post(_MEMORY_BULK_DELETE_PATH, handle_memory_bulk_delete_mutation)


def register_workshop_enrollment_routes(
    app: web.Application,
    *,
    enrollment_manager: WorkshopClientEnrollmentManager,
    rate_limiter: WorkshopEnrollmentRateLimiter,
    request_lock: asyncio.Lock,
) -> None:
    """Register grant redemption; grant issuance remains operator-only."""

    async def handle_enrollment_redemption(request: web.Request) -> web.Response:
        return await _handle_enrollment_redemption(
            request,
            enrollment_manager=enrollment_manager,
            rate_limiter=rate_limiter,
            request_lock=request_lock,
        )

    app.router.add_post(_ENROLLMENT_REDEMPTION_PATH, handle_enrollment_redemption)


def register_workshop_command_routes(
    app: web.Application,
    *,
    store: WorkshopEventStore,
    authenticator: WorkshopClientAuthenticator,
    submitter: WorkshopClientCommandSubmitter,
    request_lock: asyncio.Lock,
    artifact_service: WorkshopArtifactService | None = None,
) -> None:
    """Register the authenticated command boundary on a supplied application."""

    async def handle_command_submission(request: web.Request) -> web.Response:
        return await _handle_command_submission(
            request,
            store=store,
            authenticator=authenticator,
            submitter=submitter,
            request_lock=request_lock,
            artifact_service=artifact_service,
        )

    async def handle_run_state(request: web.Request) -> web.Response:
        return await _handle_run_state(
            request,
            authenticator=authenticator,
            submitter=submitter,
            request_lock=request_lock,
        )

    async def handle_run_trace(request: web.Request) -> web.Response:
        return await _handle_run_trace(
            request,
            store=store,
            authenticator=authenticator,
            submitter=submitter,
            request_lock=request_lock,
        )

    async def handle_run_cancellation(request: web.Request) -> web.Response:
        return await _handle_run_cancellation(
            request,
            authenticator=authenticator,
            submitter=submitter,
            request_lock=request_lock,
        )

    app.router.add_post(_COMMAND_SUBMISSION_PATH, handle_command_submission)
    app.router.add_get(_RUN_STATE_PATH, handle_run_state)
    app.router.add_get(_RUN_TRACE_PATH, handle_run_trace)
    app.router.add_post(_RUN_CANCELLATION_PATH, handle_run_cancellation)
