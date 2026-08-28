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

from kai.workshop.appearance_preferences import (
    AppearancePreferenceAuthority,
    AppearancePreferenceSnapshot,
    WorkshopAppearancePreferenceAccessDenied,
    WorkshopAppearancePreferenceConflict,
    WorkshopAppearancePreferenceError,
    WorkshopAppearancePreferenceService,
    WorkshopAppearancePreferenceStorageError,
    WorkshopAppearancePreferenceValidationError,
)
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
from kai.workshop.channel_lifecycle import (
    CreatedWorkshopChannel,
    WorkshopChannelLifecycleAccessDenied,
    WorkshopChannelLifecycleError,
    WorkshopChannelLifecycleService,
    WorkshopChannelLifecycleStorageError,
    WorkshopChannelLifecycleValidationError,
)
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
from kai.workshop.client_preferences import (
    ClientPreferenceAuthority,
    ClientPreferenceSnapshot,
    WorkshopClientPreferenceAccessDenied,
    WorkshopClientPreferenceConflict,
    WorkshopClientPreferenceError,
    WorkshopClientPreferenceService,
    WorkshopClientPreferenceStorageError,
    WorkshopClientPreferenceValidationError,
)
from kai.workshop.client_sessions import EnrollmentGrantUnavailableError, WorkshopClientEnrollmentManager
from kai.workshop.conversation_commands import ConversationCommandAcceptanceError
from kai.workshop.domain import ArtifactId, ChannelId, PrincipalId, RunId
from kai.workshop.execution_coordinator import CanonicalCancellationDisposition
from kai.workshop.github_settings import (
    GitHubSettingsAuthority,
    GitHubSettingsSnapshot,
    WorkshopGitHubSettingsAccessDenied,
    WorkshopGitHubSettingsConflict,
    WorkshopGitHubSettingsError,
    WorkshopGitHubSettingsService,
    WorkshopGitHubSettingsStorageError,
    WorkshopGitHubSettingsValidationError,
)
from kai.workshop.inbound import ClientInboundMessage, InboundBindingNotFoundError
from kai.workshop.memory_queries import (
    DEFAULT_PAGE_SIZE,
    MemoryCreationSnapshot,
    MemoryEditSnapshot,
    MemoryEpisodeEdit,
    MemoryFactEdit,
    MemoryMutationBatch,
    MemoryQueryAuthority,
    MemoryQueryFilters,
    MemoryRecordDetail,
    MemoryRecordSummary,
    MemorySourceContext,
    MemorySourceMessage,
    WorkshopMemoryAccessDenied,
    WorkshopMemoryConflict,
    WorkshopMemoryMutationFailed,
    WorkshopMemoryNotFound,
    WorkshopMemoryQueryError,
    WorkshopMemoryQueryService,
    WorkshopMemoryResponseTooLarge,
    WorkshopMemoryValidationError,
)
from kai.workshop.model_catalogue import (
    ModelCatalogueAccessDenied,
    ModelCatalogueError,
    ModelCatalogueSnapshot,
)
from kai.workshop.notification_preferences import (
    NotificationPreferenceAuthority,
    NotificationPreferenceSnapshot,
    WorkshopNotificationPreferenceAccessDenied,
    WorkshopNotificationPreferenceConflict,
    WorkshopNotificationPreferenceError,
    WorkshopNotificationPreferenceService,
    WorkshopNotificationPreferenceStorageError,
    WorkshopNotificationPreferenceValidationError,
)
from kai.workshop.preferences import (
    MAX_PREFERENCE_BYTES,
    PreferenceAuthority,
    PreferenceDocument,
    PreferenceRevisionHistory,
    WorkshopPreferenceAccessDenied,
    WorkshopPreferenceConflict,
    WorkshopPreferenceError,
    WorkshopPreferenceRevisionNotFound,
    WorkshopPreferenceService,
    WorkshopPreferenceStorageError,
    WorkshopPreferenceValidationError,
)
from kai.workshop.run_lifecycle import DurableRun, RunNotFoundError, RunStatus
from kai.workshop.run_previews import RunPreview, WorkshopRunPreviewRegistry
from kai.workshop.settings_workspaces import (
    EditableCapability,
    SettingsMutationOutcome,
    SettingsWorkspaceAuthority,
    SettingsWorkspaceSnapshot,
    WorkshopSettingsWorkspaceAccessDenied,
    WorkshopSettingsWorkspaceBusy,
    WorkshopSettingsWorkspaceConflict,
    WorkshopSettingsWorkspaceError,
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
_CHANNEL_CREATION_PATH = "/v1/channels"
_ENROLLMENT_REDEMPTION_PATH = "/v1/client/enrollment/redeem"
_COMMAND_SUBMISSION_PATH = "/v1/channels/{channel_id}/commands"
_ARTIFACT_CONTENT_PATH = "/v1/channels/{channel_id}/artifacts/{artifact_id}/content"
_ARTIFACT_DOWNLOAD_PATH = "/v1/channels/{channel_id}/artifacts/{artifact_id}/download"
_RUN_STATE_PATH = "/v1/channels/{channel_id}/runs/{run_id}"
_RUN_TRACE_PATH = "/v1/channels/{channel_id}/runs/{run_id}/trace"
_RUN_CANCELLATION_PATH = "/v1/channels/{channel_id}/runs/{run_id}/cancel"
_RUNTIME_SETTINGS_PATH = "/v1/channels/{channel_id}/settings"
_MODEL_CATALOGUE_PATH = "/v1/channels/{channel_id}/models"
_MODEL_CATALOGUE_ADMIN_REFRESH_PATH = "/v1/settings/model-catalogue/refresh-all"
_ACTIVE_WORKSPACE_PATH = "/v1/channels/{channel_id}/workspace"
_WORKSPACE_CONFIG_PATH = "/v1/channels/{channel_id}/workspace-config"
_PREFERENCES_PATH = "/v1/preferences"
_PREFERENCE_REVISIONS_PATH = "/v1/preferences/revisions"
_PREFERENCE_RESTORE_PATH = "/v1/preferences/revisions/{preference_revision}/restore"
_GITHUB_SETTINGS_PATH = "/v1/settings/github"
_NOTIFICATION_PREFERENCES_PATH = "/v1/settings/notifications"
_CLIENT_PREFERENCES_PATH = "/v1/settings/clients"
_APPEARANCE_PREFERENCES_PATH = "/v1/settings/appearance"
_MAX_PREFERENCE_UPDATE_BODY_BYTES = MAX_PREFERENCE_BYTES * 6 + 1024
_MAX_PREFERENCE_RESTORE_BODY_BYTES = 512
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
_SETTINGS_OPERATION_FIELDS = frozenset({"backend", "model", "timeout_seconds", "reset"})
_SETTINGS_REQUEST_FIELDS = _SETTINGS_OPERATION_FIELDS | {"revision"}
_MODEL_CATALOGUE_REFRESH_FIELDS = frozenset({"option_id"})
_MODEL_CATALOGUE_OPERATOR_FIELDS = frozenset({"option_id", "model_id", "display_label"})
_WORKSPACE_REQUEST_FIELDS = frozenset({"path", "revision"})
_WORKSPACE_CONFIG_REQUEST_FIELDS = frozenset({"field", "value", "path", "revision"})
_WORKSPACE_CONFIG_RESET_FIELDS = frozenset({"reset", "path", "revision"})
_PREFERENCE_UPDATE_FIELDS = frozenset({"content", "revision"})
_PREFERENCE_RESTORE_FIELDS = frozenset({"revision"})
_GITHUB_SETTINGS_REQUEST_FIELDS = frozenset({"revision", "repository", "reset_repositories", "toggle", "token"})
_GITHUB_REPOSITORY_FIELDS = frozenset({"name", "subscribed"})
_GITHUB_TOGGLE_FIELDS = frozenset({"field", "enabled"})
_MAX_GITHUB_SETTINGS_BODY_BYTES = 10_240
_NOTIFICATION_PREFERENCE_REQUEST_FIELDS = frozenset({"revision", "integration_class", "destination_choice_id", "reset"})
_MAX_NOTIFICATION_PREFERENCE_BODY_BYTES = 2_048
_CLIENT_PREFERENCE_REQUEST_FIELDS = frozenset({"revision", "binding_choice_id", "mode", "voice"})
_MAX_CLIENT_PREFERENCE_BODY_BYTES = 2_048
_APPEARANCE_PREFERENCE_REQUEST_FIELDS = frozenset({"revision", "theme_id"})
_MAX_APPEARANCE_PREFERENCE_BODY_BYTES = 1_024
_CHANNEL_CREATION_REQUEST_FIELDS = frozenset({"name", "agent_ids", "origin_channel_id"})
_MAX_CHANNEL_CREATION_BODY_BYTES = 8_192
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
        "backend_option_id": snapshot.backend_option_id,
        "backend": snapshot.backend,
        "provider": snapshot.provider,
        "backend_options": [
            {
                "option_id": option.option_id,
                "backend": option.backend,
                "provider": option.provider,
                "current": option.current,
            }
            for option in snapshot.backend_options
        ],
        "model": {
            "value": snapshot.model.value,
            "source": snapshot.model.source,
            "default_value": snapshot.model.default_value,
        },
        "timeout_seconds": {
            "value": snapshot.timeout_seconds.value,
            "source": snapshot.timeout_seconds.source,
            "default_value": snapshot.timeout_seconds.default_value,
        },
        "revision": snapshot.revision,
        "capabilities": [_serialize_editable_capability(item) for item in snapshot.capabilities],
        "mutation": _serialize_settings_mutation(snapshot.mutation),
        "workspace": snapshot.workspace,
        "model_options": (
            [
                {
                    "model_id": option.model_id,
                    "display_name": option.display_name,
                    "status": option.status,
                    "selectable": option.selectable,
                    "retained": option.retained,
                }
                for option in snapshot.model_options
            ]
            if snapshot.model_options is not None
            else None
        ),
        "model_catalogue": (
            {
                "status": snapshot.model_catalogue.status,
                "stale": snapshot.model_catalogue.stale,
                "last_known_good": snapshot.model_catalogue.last_known_good,
                "last_attempt_at": snapshot.model_catalogue.last_attempt_at,
                "last_successful_refresh_at": snapshot.model_catalogue.last_successful_refresh_at,
                "error_code": snapshot.model_catalogue.error_code,
                "error_detail": snapshot.model_catalogue.error_detail,
            }
            if snapshot.model_catalogue is not None
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


def _serialize_editable_capability(capability: EditableCapability) -> dict[str, object]:
    return {
        "field": capability.field,
        "scope": capability.scope,
        "value_type": capability.value_type,
        "resettable": capability.resettable,
        "choices": list(capability.choices) if capability.choices is not None else None,
        "minimum": capability.minimum,
        "maximum": capability.maximum,
    }


def _serialize_settings_mutation(
    mutation: SettingsMutationOutcome | None,
) -> dict[str, object] | None:
    if mutation is None:
        return None
    return {
        "operation": mutation.operation,
        "changed": mutation.changed,
        "runtime_action": mutation.runtime_action,
        "provider_session_invalidated": mutation.provider_session_invalidated,
    }


def _serialize_model_catalogue(snapshot: ModelCatalogueSnapshot) -> dict[str, object]:
    refresh = snapshot.refresh
    return {
        "version": 1,
        "principal_id": str(snapshot.principal_id),
        "runtime_profile_id": str(snapshot.runtime_profile_id),
        "option_id": snapshot.option_id,
        "stale": snapshot.stale,
        "last_known_good": snapshot.last_known_good,
        "refresh": (
            {
                "status": refresh.status.value,
                "generation": refresh.generation,
                "last_attempt_at": refresh.last_attempt_at.isoformat(),
                "last_successful_refresh_at": (
                    refresh.last_successful_refresh_at.isoformat()
                    if refresh.last_successful_refresh_at is not None
                    else None
                ),
                "expires_at": refresh.expires_at.isoformat() if refresh.expires_at is not None else None,
                "error_code": refresh.error_code,
                "error_detail": refresh.error_detail,
            }
            if refresh is not None
            else None
        ),
        "models": [
            {
                "model_id": entry.model_id,
                "display_name": entry.display_label,
                "status": entry.status.value,
                "selectable": entry.selectable,
                "retained": entry.retained,
                "sources": [provenance.source for provenance in entry.provenances],
            }
            for entry in snapshot.entries
        ],
    }


def _serialize_preference_document(document: PreferenceDocument) -> dict[str, object]:
    return {
        "version": 1,
        "document": {
            "content": document.content,
            "revision": document.revision,
            "updated_at": document.updated_at,
            "size_bytes": document.size_bytes,
            "max_bytes": document.max_bytes,
            "editable": document.editable,
        },
    }


def _serialize_github_settings(snapshot: GitHubSettingsSnapshot) -> dict[str, object]:
    return {
        "version": 1,
        "github_login": snapshot.github_login,
        "repositories_resettable": snapshot.repositories_resettable,
        "repositories": [
            {
                "repository": item.repository,
                "source": item.source,
                "automation_authorized": item.automation_authorized,
            }
            for item in snapshot.repositories
        ],
        "pr_review": {
            "enabled": snapshot.pr_review.enabled,
            "source": snapshot.pr_review.source,
            "resettable": snapshot.pr_review.resettable,
        },
        "issue_triage": {
            "enabled": snapshot.issue_triage.enabled,
            "source": snapshot.issue_triage.source,
            "resettable": snapshot.issue_triage.resettable,
        },
        "token_stored": snapshot.token_stored,
        "revision": snapshot.revision,
        "mutation": (
            {
                "operation": snapshot.mutation.operation,
                "changed": snapshot.mutation.changed,
            }
            if snapshot.mutation is not None
            else None
        ),
    }


def _serialize_notification_preferences(
    snapshot: NotificationPreferenceSnapshot,
) -> dict[str, object]:
    return {
        "version": 1,
        "destinations": [
            {
                "choice_id": item.choice_id,
                "display_name": item.display_name,
                "kind": item.kind,
                "supported_classes": list(item.supported_classes),
            }
            for item in snapshot.destinations
        ],
        "preferences": [
            {
                "integration_class": item.integration_class,
                "display_name": item.display_name,
                "destination_choice_id": item.destination_choice_id,
                "destination_name": item.destination_name,
                "destination_kind": item.destination_kind,
                "source": item.source,
                "editable": item.editable,
                "resettable": item.resettable,
            }
            for item in snapshot.preferences
        ],
        "revision": snapshot.revision,
        "mutation": (
            {
                "operation": snapshot.mutation.operation,
                "changed": snapshot.mutation.changed,
            }
            if snapshot.mutation is not None
            else None
        ),
    }


def _serialize_client_preferences(snapshot: ClientPreferenceSnapshot) -> dict[str, object]:
    return {
        "version": 1,
        "voice_output": {
            "available": snapshot.available,
            "unavailable_reason": snapshot.unavailable_reason,
            "modes": ["off", "text_and_voice", "voice_only"],
            "voices": [{"value": item.value, "display_name": item.display_name} for item in snapshot.voices],
            "bindings": [
                {
                    "choice_id": item.choice_id,
                    "client_name": item.client_name,
                    "mode": item.mode,
                    "voice": item.voice,
                    "voice_name": item.voice_name,
                    "editable": item.editable,
                }
                for item in snapshot.bindings
            ],
        },
        "revision": snapshot.revision,
        "mutation": (
            {
                "operation": snapshot.mutation.operation,
                "changed": snapshot.mutation.changed,
            }
            if snapshot.mutation is not None
            else None
        ),
    }


def _serialize_appearance_preferences(
    snapshot: AppearancePreferenceSnapshot,
) -> dict[str, object]:
    return {
        "version": 1,
        "theme_id": snapshot.theme_id,
        "themes": [
            {
                "theme_id": item.theme_id,
                "display_name": item.display_name,
                "color_scheme": item.color_scheme,
            }
            for item in snapshot.themes
        ],
        "revision": snapshot.revision,
        "mutation": (
            {
                "operation": snapshot.mutation.operation,
                "changed": snapshot.mutation.changed,
            }
            if snapshot.mutation is not None
            else None
        ),
    }


def _serialize_created_channel(
    channel: CreatedWorkshopChannel,
) -> dict[str, object]:
    return {
        "version": 1,
        "channel": {
            "channel_id": str(channel.channel_id),
            "workshop_id": str(channel.workshop_id),
            "name": channel.name,
            "kind": "group",
            "visibility": channel.visibility,
            "origin_channel_id": (str(channel.origin_channel_id) if channel.origin_channel_id is not None else None),
            "role": "owner",
            "agent_ids": [str(agent_id) for agent_id in channel.agent_ids],
        },
    }


def _serialize_preference_history(history: PreferenceRevisionHistory) -> dict[str, object]:
    return {
        "version": 1,
        "limit": history.limit,
        "revisions": [
            {
                "revision": item.revision,
                "updated_at": item.updated_at,
                "size_bytes": item.size_bytes,
            }
            for item in history.revisions
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
            "default_value": snapshot.model.default_value,
        },
        "timeout_seconds": {
            "value": snapshot.timeout_seconds.value,
            "source": snapshot.timeout_seconds.source,
            "default_value": snapshot.timeout_seconds.default_value,
        },
        "revision": snapshot.revision,
        "capabilities": [_serialize_editable_capability(item) for item in snapshot.capabilities],
        "mutation": _serialize_settings_mutation(snapshot.mutation),
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
        "revision": record.revision,
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


def _serialize_memory_edit(snapshot: MemoryEditSnapshot) -> dict[str, object]:
    return {
        "version": 1,
        "record": _serialize_memory_detail(snapshot.record),
        "changed_fields": list(snapshot.changed_fields),
        "idempotent_replay": snapshot.idempotent_replay,
    }


def _serialize_memory_creation(snapshot: MemoryCreationSnapshot) -> dict[str, object]:
    return {
        "version": 1,
        "record": _serialize_memory_detail(snapshot.record),
        "created": snapshot.created,
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
    if isinstance(exc, WorkshopMemoryConflict):
        return _json_response(
            {
                "error": {
                    "code": "memory_revision_conflict",
                    "message": str(exc),
                    "current_revision": exc.current_revision,
                }
            },
            status=409,
        )
    if isinstance(exc, WorkshopMemoryMutationFailed):
        return _error_response(status=503, code="memory_mutation_failed", message=str(exc))
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


async def _memory_json_object(request: web.Request) -> dict[str, object]:
    if request.content_type != "application/json":
        raise WorkshopMemoryValidationError("Content-Type must be application/json")
    try:
        payload = await request.json()
    except (UnicodeDecodeError, ValueError) as exc:
        raise WorkshopMemoryValidationError("Invalid memory request") from exc
    if not isinstance(payload, dict):
        raise WorkshopMemoryValidationError("Invalid memory request")
    return payload


def _memory_string_list(value: object, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise WorkshopMemoryValidationError(f"{field} must be a list of strings")
    return tuple(value)


async def _handle_memory_create(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopMemoryQueryService,
) -> web.Response:
    authority, error = await _memory_authority(request, authenticator=authenticator, service=service)
    if error is not None:
        return error
    if request.query:
        return _error_response(status=400, code="invalid_request", message="Invalid memory creation request")
    assert authority is not None
    try:
        payload = await _memory_json_object(request)
        expected = frozenset({"kind", "content", "tags", "scope", "request_id"})
        if set(payload) not in (expected, expected | {"project_id"}) or payload.get("kind") != "fact":
            raise WorkshopMemoryValidationError("Invalid memory creation request")
        content = payload.get("content")
        scope = payload.get("scope")
        request_id = payload.get("request_id")
        project_id = payload.get("project_id")
        if (
            not isinstance(content, str)
            or not isinstance(scope, str)
            or not isinstance(request_id, str)
            or (project_id is not None and not isinstance(project_id, str))
        ):
            raise WorkshopMemoryValidationError("Invalid memory creation request")
        snapshot = await service.create_fact(
            authority,
            content=content,
            tags=_memory_string_list(payload.get("tags"), field="Tags"),
            scope=scope,
            project_id=project_id,
            request_id=request_id,
        )
    except WorkshopMemoryQueryError as exc:
        return _memory_error_response(exc)
    return _json_response(
        _serialize_memory_creation(snapshot),
        status=201 if snapshot.created else 200,
    )


async def _handle_memory_edit(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopMemoryQueryService,
) -> web.Response:
    authority, error = await _memory_authority(request, authenticator=authenticator, service=service)
    if error is not None:
        return error
    if request.query:
        return _error_response(status=400, code="invalid_request", message="Invalid memory edit request")
    assert authority is not None
    try:
        payload = await _memory_json_object(request)
        kind = payload.get("kind")
        revision = payload.get("revision")
        request_id = payload.get("request_id")
        if not isinstance(revision, str) or not isinstance(request_id, str):
            raise WorkshopMemoryValidationError("Invalid memory edit request")
        if kind == "fact":
            if set(payload) != {"kind", "revision", "request_id", "content", "tags"}:
                raise WorkshopMemoryValidationError("Invalid fact edit request")
            content = payload.get("content")
            if not isinstance(content, str):
                raise WorkshopMemoryValidationError("Content must be text")
            edit: MemoryFactEdit | MemoryEpisodeEdit = MemoryFactEdit(
                content,
                _memory_string_list(payload.get("tags"), field="Tags"),
            )
        elif kind == "episode":
            if set(payload) != {"kind", "revision", "request_id", "episode"}:
                raise WorkshopMemoryValidationError("Invalid episode edit request")
            episode = payload.get("episode")
            required = {
                "goal",
                "context",
                "approach",
                "outcome",
                "outcome_quality",
                "tags",
                "actors",
            }
            if not isinstance(episode, dict) or set(episode) not in (required, required | {"lessons"}):
                raise WorkshopMemoryValidationError("Invalid episode edit request")
            text_fields = ("goal", "context", "approach", "outcome", "outcome_quality")
            if any(not isinstance(episode.get(field), str) for field in text_fields):
                raise WorkshopMemoryValidationError("Episode fields must be text")
            lessons = episode.get("lessons")
            if lessons is not None and not isinstance(lessons, str):
                raise WorkshopMemoryValidationError("Lessons must be text")
            edit = MemoryEpisodeEdit(
                goal=episode["goal"],
                context=episode["context"],
                approach=episode["approach"],
                outcome=episode["outcome"],
                outcome_quality=episode["outcome_quality"],
                lessons=lessons,
                tags=_memory_string_list(episode.get("tags"), field="Tags"),
                actors=_memory_string_list(episode.get("actors"), field="Actors"),
            )
        else:
            raise WorkshopMemoryValidationError("Invalid memory kind")
        snapshot = await service.edit(
            authority,
            request.match_info["memory_id"],
            revision=revision,
            request_id=request_id,
            edit=edit,
        )
    except (KeyError, WorkshopMemoryQueryError) as exc:
        return _memory_error_response(exc)
    return _json_response(_serialize_memory_edit(snapshot), status=200)


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


async def _authenticate_preference_authority(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopPreferenceService,
) -> tuple[PreferenceAuthority | None, web.Response | None]:
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
    except WorkshopPreferenceAccessDenied:
        return None, _error_response(
            status=403,
            code="access_denied",
            message="Access denied",
        )


def _preference_error_response(exc: WorkshopPreferenceError) -> web.Response:
    if isinstance(exc, WorkshopPreferenceConflict):
        return _json_response(
            {
                "error": {
                    "code": "revision_conflict",
                    "message": str(exc),
                    "current_revision": exc.current_revision,
                }
            },
            status=409,
        )
    if isinstance(exc, WorkshopPreferenceRevisionNotFound):
        return _error_response(status=404, code="not_found", message="Preference revision was not found")
    if isinstance(exc, WorkshopPreferenceValidationError):
        return _error_response(status=400, code="invalid_request", message=str(exc))
    if isinstance(exc, WorkshopPreferenceAccessDenied):
        return _error_response(status=403, code="access_denied", message="Access denied")
    if isinstance(exc, WorkshopPreferenceStorageError):
        return _error_response(
            status=503,
            code="preferences_unavailable",
            message="Preferences are temporarily unavailable",
        )
    return _error_response(
        status=503, code="preferences_unavailable", message="Preferences are temporarily unavailable"
    )


async def _preference_json_object(
    request: web.Request,
    *,
    max_bytes: int,
) -> dict[str, object]:
    if request.content_type != "application/json":
        raise WorkshopPreferenceValidationError("Content-Type must be application/json")
    if request.content_length is not None and request.content_length > max_bytes:
        raise WorkshopPreferenceValidationError("Preference request is too large")
    raw = await request.content.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise WorkshopPreferenceValidationError("Preference request is too large")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise WorkshopPreferenceValidationError("Invalid JSON request") from exc
    if not isinstance(payload, dict):
        raise WorkshopPreferenceValidationError("Invalid preference request")
    return payload


async def _handle_preference_document(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopPreferenceService,
) -> web.Response:
    authority, error = await _authenticate_preference_authority(
        request,
        authenticator=authenticator,
        service=service,
    )
    if error is not None:
        return error
    assert authority is not None
    if request.query or request.can_read_body:
        return _error_response(status=400, code="invalid_request", message="Invalid preference request")
    try:
        document = await service.read(authority)
    except WorkshopPreferenceError as exc:
        return _preference_error_response(exc)
    return _json_response(_serialize_preference_document(document), status=200)


async def _handle_preference_update(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopPreferenceService,
) -> web.Response:
    authority, error = await _authenticate_preference_authority(
        request,
        authenticator=authenticator,
        service=service,
    )
    if error is not None:
        return error
    assert authority is not None
    if request.query:
        return _error_response(status=400, code="invalid_request", message="Invalid preference request")
    try:
        payload = await _preference_json_object(
            request,
            max_bytes=_MAX_PREFERENCE_UPDATE_BODY_BYTES,
        )
    except WorkshopPreferenceValidationError as exc:
        return _preference_error_response(exc)
    if set(payload) != _PREFERENCE_UPDATE_FIELDS:
        return _error_response(status=400, code="invalid_request", message="Invalid preference request")
    content = payload.get("content")
    revision = payload.get("revision")
    if not isinstance(content, str) or not isinstance(revision, str):
        return _error_response(status=400, code="invalid_request", message="Invalid preference request")
    try:
        document = await service.save(
            authority,
            expected_revision=revision,
            content=content,
        )
    except WorkshopPreferenceError as exc:
        return _preference_error_response(exc)
    return _json_response(_serialize_preference_document(document), status=200)


async def _handle_preference_history(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopPreferenceService,
) -> web.Response:
    authority, error = await _authenticate_preference_authority(
        request,
        authenticator=authenticator,
        service=service,
    )
    if error is not None:
        return error
    assert authority is not None
    if request.query or request.can_read_body:
        return _error_response(status=400, code="invalid_request", message="Invalid preference history request")
    try:
        history = await service.history(authority)
    except WorkshopPreferenceError as exc:
        return _preference_error_response(exc)
    return _json_response(_serialize_preference_history(history), status=200)


async def _handle_preference_restore(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopPreferenceService,
) -> web.Response:
    authority, error = await _authenticate_preference_authority(
        request,
        authenticator=authenticator,
        service=service,
    )
    if error is not None:
        return error
    assert authority is not None
    if request.query:
        return _error_response(status=400, code="invalid_request", message="Invalid preference restore request")
    try:
        payload = await _preference_json_object(
            request,
            max_bytes=_MAX_PREFERENCE_RESTORE_BODY_BYTES,
        )
    except WorkshopPreferenceValidationError as exc:
        return _preference_error_response(exc)
    if set(payload) != _PREFERENCE_RESTORE_FIELDS:
        return _error_response(status=400, code="invalid_request", message="Invalid preference restore request")
    revision = payload.get("revision")
    if not isinstance(revision, str):
        return _error_response(status=400, code="invalid_request", message="Invalid preference restore request")
    try:
        document = await service.restore(
            authority,
            target_revision=request.match_info["preference_revision"],
            expected_revision=revision,
        )
    except (KeyError, WorkshopPreferenceError) as exc:
        if isinstance(exc, KeyError):
            return _error_response(status=400, code="invalid_request", message="Invalid preference restore request")
        return _preference_error_response(exc)
    return _json_response(_serialize_preference_document(document), status=200)


async def _authenticate_github_settings_authority(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopGitHubSettingsService,
) -> tuple[GitHubSettingsAuthority | None, web.Response | None]:
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
    except WorkshopGitHubSettingsAccessDenied:
        return None, _error_response(status=403, code="access_denied", message="Access denied")


def _github_settings_error_response(exc: WorkshopGitHubSettingsError) -> web.Response:
    if isinstance(exc, WorkshopGitHubSettingsConflict):
        return _error_response(status=409, code="settings_conflict", message=str(exc))
    if isinstance(exc, WorkshopGitHubSettingsValidationError):
        return _error_response(status=400, code="invalid_setting", message=str(exc))
    if isinstance(exc, WorkshopGitHubSettingsAccessDenied):
        return _error_response(status=403, code="access_denied", message="Access denied")
    if isinstance(exc, WorkshopGitHubSettingsStorageError):
        return _error_response(
            status=503,
            code="github_settings_unavailable",
            message="GitHub settings are temporarily unavailable",
        )
    return _error_response(
        status=503,
        code="github_settings_unavailable",
        message="GitHub settings are temporarily unavailable",
    )


async def _github_settings_json_object(request: web.Request) -> dict[str, object]:
    if request.content_type != "application/json":
        raise WorkshopGitHubSettingsValidationError("Content-Type must be application/json")
    if request.content_length is not None and request.content_length > _MAX_GITHUB_SETTINGS_BODY_BYTES:
        raise WorkshopGitHubSettingsValidationError("GitHub settings request is too large")
    raw = await request.content.read(_MAX_GITHUB_SETTINGS_BODY_BYTES + 1)
    if len(raw) > _MAX_GITHUB_SETTINGS_BODY_BYTES:
        raise WorkshopGitHubSettingsValidationError("GitHub settings request is too large")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise WorkshopGitHubSettingsValidationError("Invalid JSON request") from exc
    if not isinstance(payload, dict):
        raise WorkshopGitHubSettingsValidationError("Invalid GitHub settings request")
    return payload


async def _handle_github_settings(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopGitHubSettingsService,
) -> web.Response:
    authority, error = await _authenticate_github_settings_authority(
        request,
        authenticator=authenticator,
        service=service,
    )
    if error is not None:
        return error
    assert authority is not None
    if request.query or request.can_read_body:
        return _error_response(status=400, code="invalid_request", message="Invalid GitHub settings request")
    try:
        snapshot = await service.inspect(authority)
    except WorkshopGitHubSettingsError as exc:
        return _github_settings_error_response(exc)
    return _json_response(_serialize_github_settings(snapshot), status=200)


async def _handle_github_settings_update(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopGitHubSettingsService,
) -> web.Response:
    authority, error = await _authenticate_github_settings_authority(
        request,
        authenticator=authenticator,
        service=service,
    )
    if error is not None:
        return error
    assert authority is not None
    if request.query:
        return _error_response(status=400, code="invalid_request", message="Invalid GitHub settings request")
    try:
        payload = await _github_settings_json_object(request)
        if not set(payload).issubset(_GITHUB_SETTINGS_REQUEST_FIELDS):
            raise WorkshopGitHubSettingsValidationError("Invalid GitHub settings request")
        revision = payload.get("revision")
        if not isinstance(revision, str):
            raise WorkshopGitHubSettingsValidationError("GitHub settings revision is required")
        operations = sum(field in payload for field in ("repository", "reset_repositories", "toggle", "token"))
        if operations != 1 or len(payload) != 2:
            raise WorkshopGitHubSettingsValidationError("Change exactly one GitHub setting at a time")
        if "repository" in payload:
            repository = payload["repository"]
            if not isinstance(repository, dict) or set(repository) != _GITHUB_REPOSITORY_FIELDS:
                raise WorkshopGitHubSettingsValidationError("Invalid repository subscription request")
            name = repository.get("name")
            subscribed = repository.get("subscribed")
            if not isinstance(name, str) or not isinstance(subscribed, bool):
                raise WorkshopGitHubSettingsValidationError("Invalid repository subscription request")
            snapshot = await service.set_repository_subscription(
                authority,
                name,
                subscribed=subscribed,
                expected_revision=revision,
            )
        elif "reset_repositories" in payload:
            if payload["reset_repositories"] is not True:
                raise WorkshopGitHubSettingsValidationError("Invalid repository reset request")
            snapshot = await service.reset_repository_subscriptions(
                authority,
                expected_revision=revision,
            )
        elif "toggle" in payload:
            toggle = payload["toggle"]
            if not isinstance(toggle, dict) or set(toggle) != _GITHUB_TOGGLE_FIELDS:
                raise WorkshopGitHubSettingsValidationError("Invalid GitHub automation request")
            field = toggle.get("field")
            enabled = toggle.get("enabled")
            if not isinstance(field, str) or (enabled is not None and not isinstance(enabled, bool)):
                raise WorkshopGitHubSettingsValidationError("Invalid GitHub automation request")
            snapshot = await service.set_toggle(
                authority,
                field,
                enabled,
                expected_revision=revision,
            )
        else:
            token = payload["token"]
            if token is not None and not isinstance(token, str):
                raise WorkshopGitHubSettingsValidationError("GitHub token must be text or null")
            snapshot = await service.set_token(
                authority,
                token,
                expected_revision=revision,
            )
    except WorkshopGitHubSettingsError as exc:
        return _github_settings_error_response(exc)
    return _json_response(_serialize_github_settings(snapshot), status=200)


async def _authenticate_notification_preference_authority(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopNotificationPreferenceService,
) -> tuple[NotificationPreferenceAuthority | None, web.Response | None]:
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
    except WorkshopNotificationPreferenceAccessDenied:
        return None, _error_response(status=403, code="access_denied", message="Access denied")


def _notification_preference_error_response(
    exc: WorkshopNotificationPreferenceError,
) -> web.Response:
    if isinstance(exc, WorkshopNotificationPreferenceConflict):
        return _error_response(status=409, code="settings_conflict", message=str(exc))
    if isinstance(exc, WorkshopNotificationPreferenceValidationError):
        return _error_response(status=400, code="invalid_setting", message=str(exc))
    if isinstance(exc, WorkshopNotificationPreferenceAccessDenied):
        return _error_response(status=403, code="access_denied", message="Access denied")
    if isinstance(exc, WorkshopNotificationPreferenceStorageError):
        return _error_response(
            status=503,
            code="notification_preferences_unavailable",
            message="Notification preferences are temporarily unavailable",
        )
    return _error_response(
        status=503,
        code="notification_preferences_unavailable",
        message="Notification preferences are temporarily unavailable",
    )


async def _notification_preference_json_object(
    request: web.Request,
) -> dict[str, object]:
    if request.content_type != "application/json":
        raise WorkshopNotificationPreferenceValidationError("Content-Type must be application/json")
    if request.content_length is not None and request.content_length > _MAX_NOTIFICATION_PREFERENCE_BODY_BYTES:
        raise WorkshopNotificationPreferenceValidationError("Notification preference request is too large")
    raw = await request.content.read(_MAX_NOTIFICATION_PREFERENCE_BODY_BYTES + 1)
    if len(raw) > _MAX_NOTIFICATION_PREFERENCE_BODY_BYTES:
        raise WorkshopNotificationPreferenceValidationError("Notification preference request is too large")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise WorkshopNotificationPreferenceValidationError("Invalid JSON request") from exc
    if not isinstance(payload, dict):
        raise WorkshopNotificationPreferenceValidationError("Invalid notification preference request")
    return payload


async def _handle_notification_preferences(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopNotificationPreferenceService,
) -> web.Response:
    authority, error = await _authenticate_notification_preference_authority(
        request,
        authenticator=authenticator,
        service=service,
    )
    if error is not None:
        return error
    assert authority is not None
    if request.query or request.can_read_body:
        return _error_response(
            status=400,
            code="invalid_request",
            message="Invalid notification preference request",
        )
    try:
        snapshot = await service.inspect(authority)
    except WorkshopNotificationPreferenceError as exc:
        return _notification_preference_error_response(exc)
    return _json_response(_serialize_notification_preferences(snapshot), status=200)


async def _handle_notification_preference_update(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopNotificationPreferenceService,
) -> web.Response:
    authority, error = await _authenticate_notification_preference_authority(
        request,
        authenticator=authenticator,
        service=service,
    )
    if error is not None:
        return error
    assert authority is not None
    if request.query:
        return _error_response(
            status=400,
            code="invalid_request",
            message="Invalid notification preference request",
        )
    try:
        payload = await _notification_preference_json_object(request)
        if not set(payload).issubset(_NOTIFICATION_PREFERENCE_REQUEST_FIELDS):
            raise WorkshopNotificationPreferenceValidationError("Invalid notification preference request")
        revision = payload.get("revision")
        integration_class = payload.get("integration_class")
        if not isinstance(revision, str) or not isinstance(integration_class, str):
            raise WorkshopNotificationPreferenceValidationError(
                "Notification preference revision and integration class are required"
            )
        has_destination = "destination_choice_id" in payload
        has_reset = "reset" in payload
        if has_destination == has_reset:
            raise WorkshopNotificationPreferenceValidationError("Change exactly one notification preference at a time")
        if has_destination:
            if set(payload) != {"revision", "integration_class", "destination_choice_id"}:
                raise WorkshopNotificationPreferenceValidationError("Invalid notification destination request")
            choice_id = payload["destination_choice_id"]
            if not isinstance(choice_id, str):
                raise WorkshopNotificationPreferenceValidationError("Invalid notification destination request")
            snapshot = await service.select(
                authority,
                integration_class,
                choice_id,
                expected_revision=revision,
            )
        else:
            if set(payload) != {"revision", "integration_class", "reset"} or payload["reset"] is not True:
                raise WorkshopNotificationPreferenceValidationError("Invalid notification preference reset request")
            snapshot = await service.reset(
                authority,
                integration_class,
                expected_revision=revision,
            )
    except WorkshopNotificationPreferenceError as exc:
        return _notification_preference_error_response(exc)
    return _json_response(_serialize_notification_preferences(snapshot), status=200)


async def _authenticate_client_preference_authority(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopClientPreferenceService,
) -> tuple[ClientPreferenceAuthority | None, web.Response | None]:
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
    except WorkshopClientPreferenceAccessDenied:
        return None, _error_response(status=403, code="access_denied", message="Access denied")


def _client_preference_error_response(exc: WorkshopClientPreferenceError) -> web.Response:
    if isinstance(exc, WorkshopClientPreferenceConflict):
        return _error_response(status=409, code="settings_conflict", message=str(exc))
    if isinstance(exc, WorkshopClientPreferenceValidationError):
        return _error_response(status=400, code="invalid_setting", message=str(exc))
    if isinstance(exc, WorkshopClientPreferenceAccessDenied):
        return _error_response(status=403, code="access_denied", message="Access denied")
    if isinstance(exc, WorkshopClientPreferenceStorageError):
        return _error_response(
            status=503,
            code="client_preferences_unavailable",
            message="Client preferences are temporarily unavailable",
        )
    return _error_response(
        status=503,
        code="client_preferences_unavailable",
        message="Client preferences are temporarily unavailable",
    )


async def _client_preference_json_object(request: web.Request) -> dict[str, object]:
    if request.content_type != "application/json":
        raise WorkshopClientPreferenceValidationError("Content-Type must be application/json")
    if request.content_length is not None and request.content_length > _MAX_CLIENT_PREFERENCE_BODY_BYTES:
        raise WorkshopClientPreferenceValidationError("Client preference request is too large")
    raw = await request.content.read(_MAX_CLIENT_PREFERENCE_BODY_BYTES + 1)
    if len(raw) > _MAX_CLIENT_PREFERENCE_BODY_BYTES:
        raise WorkshopClientPreferenceValidationError("Client preference request is too large")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as exc:
        raise WorkshopClientPreferenceValidationError("Invalid JSON request") from exc
    if not isinstance(payload, dict):
        raise WorkshopClientPreferenceValidationError("Invalid client preference request")
    return payload


async def _handle_client_preferences(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopClientPreferenceService,
) -> web.Response:
    authority, error = await _authenticate_client_preference_authority(
        request,
        authenticator=authenticator,
        service=service,
    )
    if error is not None:
        return error
    assert authority is not None
    if request.query or request.can_read_body:
        return _error_response(
            status=400,
            code="invalid_request",
            message="Invalid client preference request",
        )
    try:
        snapshot = await service.inspect(authority)
    except WorkshopClientPreferenceError as exc:
        return _client_preference_error_response(exc)
    return _json_response(_serialize_client_preferences(snapshot), status=200)


async def _handle_client_preference_update(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopClientPreferenceService,
) -> web.Response:
    authority, error = await _authenticate_client_preference_authority(
        request,
        authenticator=authenticator,
        service=service,
    )
    if error is not None:
        return error
    assert authority is not None
    if request.query:
        return _error_response(
            status=400,
            code="invalid_request",
            message="Invalid client preference request",
        )
    try:
        payload = await _client_preference_json_object(request)
        if not set(payload).issubset(_CLIENT_PREFERENCE_REQUEST_FIELDS):
            raise WorkshopClientPreferenceValidationError("Invalid client preference request")
        revision = payload.get("revision")
        choice_id = payload.get("binding_choice_id")
        if not isinstance(revision, str) or not isinstance(choice_id, str):
            raise WorkshopClientPreferenceValidationError("Client preference revision and binding choice are required")
        has_mode = "mode" in payload
        has_voice = "voice" in payload
        if has_mode == has_voice:
            raise WorkshopClientPreferenceValidationError("Change exactly one client preference at a time")
        if has_mode:
            if set(payload) != {"revision", "binding_choice_id", "mode"}:
                raise WorkshopClientPreferenceValidationError("Invalid client voice mode request")
            mode = payload["mode"]
            if not isinstance(mode, str):
                raise WorkshopClientPreferenceValidationError("Invalid client voice mode request")
            snapshot = await service.set_choice_mode(
                authority,
                choice_id,
                mode,
                expected_revision=revision,
            )
        else:
            if set(payload) != {"revision", "binding_choice_id", "voice"}:
                raise WorkshopClientPreferenceValidationError("Invalid client voice request")
            voice = payload["voice"]
            if not isinstance(voice, str):
                raise WorkshopClientPreferenceValidationError("Invalid client voice request")
            snapshot = await service.set_choice_voice(
                authority,
                choice_id,
                voice,
                expected_revision=revision,
            )
    except WorkshopClientPreferenceError as exc:
        return _client_preference_error_response(exc)
    return _json_response(_serialize_client_preferences(snapshot), status=200)


async def _authenticate_appearance_preference_authority(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopAppearancePreferenceService,
) -> tuple[AppearancePreferenceAuthority | None, web.Response | None]:
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
    except WorkshopAppearancePreferenceAccessDenied:
        return None, _error_response(status=403, code="access_denied", message="Access denied")


def _appearance_preference_error_response(
    exc: WorkshopAppearancePreferenceError,
) -> web.Response:
    if isinstance(exc, WorkshopAppearancePreferenceConflict):
        return _error_response(status=409, code="settings_conflict", message=str(exc))
    if isinstance(exc, WorkshopAppearancePreferenceValidationError):
        return _error_response(status=400, code="invalid_setting", message=str(exc))
    if isinstance(exc, WorkshopAppearancePreferenceAccessDenied):
        return _error_response(status=403, code="access_denied", message="Access denied")
    if isinstance(exc, WorkshopAppearancePreferenceStorageError):
        return _error_response(
            status=503,
            code="appearance_preferences_unavailable",
            message="Appearance preferences are temporarily unavailable",
        )
    return _error_response(
        status=503,
        code="appearance_preferences_unavailable",
        message="Appearance preferences are temporarily unavailable",
    )


async def _handle_appearance_preferences(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopAppearancePreferenceService,
) -> web.Response:
    authority, error = await _authenticate_appearance_preference_authority(
        request,
        authenticator=authenticator,
        service=service,
    )
    if error is not None:
        return error
    assert authority is not None
    if request.query or request.can_read_body:
        return _error_response(
            status=400,
            code="invalid_request",
            message="Invalid appearance preference request",
        )
    try:
        snapshot = await service.inspect(authority)
    except WorkshopAppearancePreferenceError as exc:
        return _appearance_preference_error_response(exc)
    return _json_response(_serialize_appearance_preferences(snapshot), status=200)


async def _handle_appearance_preference_update(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopAppearancePreferenceService,
) -> web.Response:
    authority, error = await _authenticate_appearance_preference_authority(
        request,
        authenticator=authenticator,
        service=service,
    )
    if error is not None:
        return error
    assert authority is not None
    if request.query:
        return _error_response(
            status=400,
            code="invalid_request",
            message="Invalid appearance preference request",
        )
    try:
        if request.content_type != "application/json":
            raise WorkshopAppearancePreferenceValidationError("Content-Type must be application/json")
        if request.content_length is not None and request.content_length > _MAX_APPEARANCE_PREFERENCE_BODY_BYTES:
            raise WorkshopAppearancePreferenceValidationError("Appearance preference request is too large")
        raw = await request.content.read(_MAX_APPEARANCE_PREFERENCE_BODY_BYTES + 1)
        if len(raw) > _MAX_APPEARANCE_PREFERENCE_BODY_BYTES:
            raise WorkshopAppearancePreferenceValidationError("Appearance preference request is too large")
        try:
            payload = json.loads(raw)
        except (UnicodeDecodeError, ValueError) as exc:
            raise WorkshopAppearancePreferenceValidationError("Invalid JSON request") from exc
        if not isinstance(payload, dict) or set(payload) != _APPEARANCE_PREFERENCE_REQUEST_FIELDS:
            raise WorkshopAppearancePreferenceValidationError("Invalid appearance preference request")
        revision = payload.get("revision")
        theme_id = payload.get("theme_id")
        if not isinstance(revision, str) or not isinstance(theme_id, str):
            raise WorkshopAppearancePreferenceValidationError("Appearance preference revision and theme are required")
        snapshot = await service.set_theme(
            authority,
            theme_id,
            expected_revision=revision,
        )
    except WorkshopAppearancePreferenceError as exc:
        return _appearance_preference_error_response(exc)
    return _json_response(_serialize_appearance_preferences(snapshot), status=200)


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
    revision = payload.get("revision")
    if not isinstance(revision, str):
        return _error_response(status=400, code="invalid_request", message="Settings revision is required")
    operations = sum(field in payload for field in _SETTINGS_OPERATION_FIELDS)
    if operations != 1:
        return _error_response(
            status=400,
            code="invalid_request",
            message="Change exactly one setting at a time",
        )
    try:
        if "backend" in payload:
            if not isinstance(payload["backend"], str):
                raise WorkshopSettingsWorkspaceValidationError("Backend must be text")
            snapshot = await service.set_backend(
                authority,
                payload["backend"],
                expected_revision=revision,
            )
        elif "model" in payload:
            if not isinstance(payload["model"], str):
                raise WorkshopSettingsWorkspaceValidationError("Model must be text")
            snapshot = await service.set_model(
                authority,
                payload["model"],
                expected_revision=revision,
                clear_workspace_override=False,
            )
        elif "timeout_seconds" in payload:
            snapshot = await service.set_timeout(
                authority,
                payload["timeout_seconds"],
                expected_revision=revision,
            )
        else:
            reset = payload["reset"]
            if reset not in {"model", "timeout", "all"}:
                raise WorkshopSettingsWorkspaceValidationError("Reset must be model, timeout, or all")
            snapshot = await service.reset_settings(
                authority,
                None if reset == "all" else reset,
                expected_revision=revision,
            )
    except WorkshopSettingsWorkspaceBusy as exc:
        return _error_response(status=409, code="runtime_busy", message=str(exc))
    except WorkshopSettingsWorkspaceConflict as exc:
        return _error_response(status=409, code="settings_conflict", message=str(exc))
    except WorkshopSettingsWorkspaceValidationError as exc:
        return _error_response(
            status=400,
            code="invalid_setting",
            message=str(exc),
        )
    return _json_response(_serialize_settings_workspace(snapshot), status=200)


async def _handle_model_catalogue(
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
        return error
    assert authority is not None
    if request.can_read_body or any(key != "option_id" for key in request.query):
        return _error_response(status=400, code="invalid_request", message="Invalid model catalogue request")
    option_id = request.query.get("option_id")
    try:
        snapshot = await service.inspect_model_catalogue(authority, option_id)
    except (WorkshopSettingsWorkspaceAccessDenied, ModelCatalogueAccessDenied):
        return _error_response(status=403, code="access_denied", message="Access denied")
    except (WorkshopSettingsWorkspaceError, ModelCatalogueError):
        return _error_response(
            status=503,
            code="model_catalogue_unavailable",
            message="Model catalogue is temporarily unavailable",
        )
    return _json_response(_serialize_model_catalogue(snapshot), status=200)


async def _handle_model_catalogue_refresh(
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
        return error
    assert authority is not None
    if request.query:
        return _error_response(status=400, code="invalid_request", message="Invalid model refresh request")
    try:
        payload = await request.json()
    except (UnicodeDecodeError, ValueError):
        return _error_response(status=400, code="invalid_request", message="Invalid JSON request")
    if not isinstance(payload, dict) or not set(payload).issubset(_MODEL_CATALOGUE_REFRESH_FIELDS):
        return _error_response(status=400, code="invalid_request", message="Invalid model refresh request")
    option_id = payload.get("option_id")
    if option_id is not None and not isinstance(option_id, str):
        return _error_response(status=400, code="invalid_request", message="Invalid backend option")
    try:
        await service.refresh_model_catalogue(authority, option_id)
        snapshot = await service.inspect_model_catalogue(authority, option_id)
    except (WorkshopSettingsWorkspaceAccessDenied, ModelCatalogueAccessDenied):
        return _error_response(status=403, code="access_denied", message="Access denied")
    except (WorkshopSettingsWorkspaceError, ModelCatalogueError):
        return _error_response(
            status=503,
            code="model_catalogue_unavailable",
            message="Model catalogue refresh is temporarily unavailable",
        )
    return _json_response(_serialize_model_catalogue(snapshot), status=200)


async def _principal_is_workshop_admin(
    store: WorkshopEventStore,
    principal_id: PrincipalId,
) -> bool:
    async with store.connection.execute(
        "SELECT 1 FROM workshop_memberships WHERE principal_id = ? AND role = 'admin' LIMIT 1",
        (principal_id,),
    ) as cursor:
        return await cursor.fetchone() is not None


async def _handle_model_catalogue_operator_entry(
    request: web.Request,
    *,
    store: WorkshopEventStore,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopSettingsWorkspaceService,
    deactivate: bool,
) -> web.Response:
    authority, error = await _authenticate_settings_authority(
        request,
        authenticator=authenticator,
        service=service,
    )
    if error is not None:
        return error
    assert authority is not None
    if not await _principal_is_workshop_admin(store, authority.principal_id):
        return _error_response(status=403, code="access_denied", message="Administrator access required")
    if request.query:
        return _error_response(status=400, code="invalid_request", message="Invalid operator model request")
    try:
        payload = await request.json()
    except (UnicodeDecodeError, ValueError):
        return _error_response(status=400, code="invalid_request", message="Invalid JSON request")
    expected_fields = {"option_id", "model_id"} if deactivate else _MODEL_CATALOGUE_OPERATOR_FIELDS
    if (
        not isinstance(payload, dict)
        or set(payload) != set(expected_fields)
        or any(not isinstance(payload.get(field), str) for field in expected_fields)
    ):
        return _error_response(status=400, code="invalid_request", message="Invalid operator model request")
    try:
        if deactivate:
            snapshot = await service.deactivate_operator_model(
                authority,
                payload["option_id"],
                model_id=payload["model_id"],
            )
        else:
            snapshot = await service.upsert_operator_model(
                authority,
                payload["option_id"],
                model_id=payload["model_id"],
                display_label=payload["display_label"],
            )
    except (WorkshopSettingsWorkspaceAccessDenied, ModelCatalogueAccessDenied):
        return _error_response(status=403, code="access_denied", message="Access denied")
    except ModelCatalogueError as exc:
        return _error_response(status=400, code="invalid_model_entry", message=str(exc))
    return _json_response(_serialize_model_catalogue(snapshot), status=200)


async def _handle_model_catalogue_refresh_all(
    request: web.Request,
    *,
    store: WorkshopEventStore,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopSettingsWorkspaceService,
) -> web.Response:
    principal_id = await authenticator.authenticate(request)
    if not isinstance(principal_id, PrincipalId):
        return _error_response(status=401, code="authentication_required", message="Authentication required")
    if request.query or request.can_read_body:
        return _error_response(status=400, code="invalid_request", message="Invalid model refresh request")
    if not await _principal_is_workshop_admin(store, principal_id):
        return _error_response(status=403, code="access_denied", message="Administrator access required")
    try:
        results = await service.refresh_all_model_catalogues_as_operator()
    except ModelCatalogueError:
        return _error_response(
            status=503,
            code="model_catalogue_unavailable",
            message="Model catalogue refresh is temporarily unavailable",
        )
    counts: dict[str, int] = {}
    for result in results:
        counts[result.status.value] = counts.get(result.status.value, 0) + 1
    return _json_response(
        {
            "version": 1,
            "contexts": len(results),
            "statuses": counts,
            "selection_changed": False,
        },
        status=200,
    )


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
        or not isinstance(payload.get("revision"), str)
    ):
        return _error_response(status=400, code="invalid_request", message="Invalid workspace request")
    try:
        snapshot = await service.switch_workspace(
            authority,
            payload["path"],
            expected_revision=payload["revision"],
        )
    except WorkshopSettingsWorkspaceConflict as exc:
        return _error_response(status=409, code="settings_conflict", message=str(exc))
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
    revision = payload.get("revision")
    if path is not None and not isinstance(path, str):
        return _error_response(status=400, code="invalid_request", message="Invalid workspace config request")
    if not isinstance(revision, str):
        return _error_response(status=400, code="invalid_request", message="Workspace config revision is required")
    try:
        if "reset" in payload:
            if not keys.issubset(_WORKSPACE_CONFIG_RESET_FIELDS):
                raise WorkshopSettingsWorkspaceValidationError("Invalid workspace config reset")
            reset = payload["reset"]
            if reset not in {"model", "timeout", "prompt", "all"}:
                raise WorkshopSettingsWorkspaceValidationError("Invalid workspace config reset")
            snapshot = await service.reset_self_service_workspace_config(
                authority,
                field=None if reset == "all" else reset,
                workspace_path=path,
                expected_revision=revision,
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
            if payload["field"] not in {"model", "timeout", "prompt"}:
                raise WorkshopSettingsWorkspaceValidationError("Unsupported self-service workspace setting")
            snapshot = await service.set_self_service_workspace_config(
                authority,
                field=payload["field"],
                value=payload["value"],
                workspace_path=path,
                expected_revision=revision,
            )
    except WorkshopSettingsWorkspaceConflict as exc:
        return _error_response(status=409, code="settings_conflict", message=str(exc))
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


async def _handle_channel_creation(
    request: web.Request,
    *,
    authenticator: WorkshopClientAuthenticator,
    service: WorkshopChannelLifecycleService,
) -> web.Response:
    """Create one private canonical group channel for the authenticated human."""
    principal_id = await authenticator.authenticate(request)
    if not isinstance(principal_id, PrincipalId):
        response = _error_response(
            status=401,
            code="authentication_required",
            message="Authentication required",
        )
        response.headers["WWW-Authenticate"] = "Bearer"
        return response
    if request.query or request.content_type != "application/json":
        return _error_response(
            status=400,
            code="invalid_request",
            message="Invalid channel creation request",
        )
    if request.content_length is not None and request.content_length > _MAX_CHANNEL_CREATION_BODY_BYTES:
        return _error_response(
            status=400,
            code="invalid_request",
            message="Channel creation request is too large",
        )
    raw = await request.content.read(_MAX_CHANNEL_CREATION_BODY_BYTES + 1)
    if len(raw) > _MAX_CHANNEL_CREATION_BODY_BYTES:
        return _error_response(
            status=400,
            code="invalid_request",
            message="Channel creation request is too large",
        )
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, ValueError):
        payload = None
    if (
        not isinstance(payload, dict)
        or not {"name", "agent_ids"}.issubset(payload)
        or not set(payload).issubset(_CHANNEL_CREATION_REQUEST_FIELDS)
    ):
        return _error_response(
            status=400,
            code="invalid_request",
            message="Invalid channel creation request",
        )
    try:
        created = await service.create_group(
            principal_id,
            name=payload["name"],
            agent_ids=payload["agent_ids"],
            origin_channel_id=payload.get("origin_channel_id"),
        )
    except WorkshopChannelLifecycleAccessDenied:
        return _error_response(
            status=403,
            code="access_denied",
            message="Access denied",
        )
    except WorkshopChannelLifecycleValidationError as exc:
        return _error_response(
            status=400,
            code="invalid_request",
            message=str(exc),
        )
    except WorkshopChannelLifecycleStorageError:
        return _error_response(
            status=503,
            code="channel_creation_unavailable",
            message="Channel creation is temporarily unavailable",
        )
    except WorkshopChannelLifecycleError:
        return _error_response(
            status=409,
            code="channel_creation_conflict",
            message="Channel creation conflicted with current state",
        )
    return _json_response(_serialize_created_channel(created), status=201)


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
    preference_documents: WorkshopPreferenceService | None = None,
    github_settings: WorkshopGitHubSettingsService | None = None,
    notification_preferences: WorkshopNotificationPreferenceService | None = None,
    client_preferences: WorkshopClientPreferenceService | None = None,
    appearance_preferences: WorkshopAppearancePreferenceService | None = None,
) -> None:
    """Register authenticated Workshop client routes on an application."""
    if event_poll_interval <= 0 or event_heartbeat_interval <= 0 or event_authentication_recheck_interval <= 0:
        raise ValueError("Event-stream intervals must be positive")
    stream_limiter = event_stream_limiter or WorkshopEventStreamLimiter()
    channel_lifecycle = WorkshopChannelLifecycleService(store)
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

    async def handle_channel_creation(request: web.Request) -> web.Response:
        async with request_lock:
            return await _handle_channel_creation(
                request,
                authenticator=authenticator,
                service=channel_lifecycle,
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
    app.router.add_post(_CHANNEL_CREATION_PATH, handle_channel_creation)
    app.router.add_get(_TIMELINE_PATH, handle_channel_timeline)
    app.router.add_get(_TIMELINE_EVENTS_PATH, handle_channel_event_stream)
    if preference_documents is not None:

        async def handle_preference_document(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_preference_document(
                    request,
                    authenticator=authenticator,
                    service=preference_documents,
                )

        async def handle_preference_update(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_preference_update(
                    request,
                    authenticator=authenticator,
                    service=preference_documents,
                )

        async def handle_preference_history(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_preference_history(
                    request,
                    authenticator=authenticator,
                    service=preference_documents,
                )

        async def handle_preference_restore(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_preference_restore(
                    request,
                    authenticator=authenticator,
                    service=preference_documents,
                )

        app.router.add_get(_PREFERENCES_PATH, handle_preference_document)
        app.router.add_put(_PREFERENCES_PATH, handle_preference_update)
        app.router.add_get(_PREFERENCE_REVISIONS_PATH, handle_preference_history)
        app.router.add_post(_PREFERENCE_RESTORE_PATH, handle_preference_restore)
    if github_settings is not None:

        async def handle_github_settings(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_github_settings(
                    request,
                    authenticator=authenticator,
                    service=github_settings,
                )

        async def handle_github_settings_update(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_github_settings_update(
                    request,
                    authenticator=authenticator,
                    service=github_settings,
                )

        app.router.add_get(_GITHUB_SETTINGS_PATH, handle_github_settings)
        app.router.add_patch(_GITHUB_SETTINGS_PATH, handle_github_settings_update)
    if notification_preferences is not None:

        async def handle_notification_preferences(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_notification_preferences(
                    request,
                    authenticator=authenticator,
                    service=notification_preferences,
                )

        async def handle_notification_preference_update(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_notification_preference_update(
                    request,
                    authenticator=authenticator,
                    service=notification_preferences,
                )

        app.router.add_get(
            _NOTIFICATION_PREFERENCES_PATH,
            handle_notification_preferences,
        )
        app.router.add_patch(
            _NOTIFICATION_PREFERENCES_PATH,
            handle_notification_preference_update,
        )
    if client_preferences is not None:

        async def handle_client_preferences(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_client_preferences(
                    request,
                    authenticator=authenticator,
                    service=client_preferences,
                )

        async def handle_client_preference_update(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_client_preference_update(
                    request,
                    authenticator=authenticator,
                    service=client_preferences,
                )

        app.router.add_get(_CLIENT_PREFERENCES_PATH, handle_client_preferences)
        app.router.add_patch(_CLIENT_PREFERENCES_PATH, handle_client_preference_update)
    if appearance_preferences is not None:

        async def handle_appearance_preferences(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_appearance_preferences(
                    request,
                    authenticator=authenticator,
                    service=appearance_preferences,
                )

        async def handle_appearance_preference_update(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_appearance_preference_update(
                    request,
                    authenticator=authenticator,
                    service=appearance_preferences,
                )

        app.router.add_get(_APPEARANCE_PREFERENCES_PATH, handle_appearance_preferences)
        app.router.add_patch(
            _APPEARANCE_PREFERENCES_PATH,
            handle_appearance_preference_update,
        )
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

        async def handle_model_catalogue(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_model_catalogue(
                    request,
                    authenticator=authenticator,
                    service=settings_workspaces,
                )

        async def handle_model_catalogue_refresh(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_model_catalogue_refresh(
                    request,
                    authenticator=authenticator,
                    service=settings_workspaces,
                )

        async def handle_model_catalogue_operator_upsert(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_model_catalogue_operator_entry(
                    request,
                    store=store,
                    authenticator=authenticator,
                    service=settings_workspaces,
                    deactivate=False,
                )

        async def handle_model_catalogue_operator_deactivate(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_model_catalogue_operator_entry(
                    request,
                    store=store,
                    authenticator=authenticator,
                    service=settings_workspaces,
                    deactivate=True,
                )

        async def handle_model_catalogue_refresh_all(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_model_catalogue_refresh_all(
                    request,
                    store=store,
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
        app.router.add_get(_MODEL_CATALOGUE_PATH, handle_model_catalogue)
        app.router.add_post(_MODEL_CATALOGUE_PATH, handle_model_catalogue_refresh)
        app.router.add_put(_MODEL_CATALOGUE_PATH, handle_model_catalogue_operator_upsert)
        app.router.add_delete(_MODEL_CATALOGUE_PATH, handle_model_catalogue_operator_deactivate)
        app.router.add_post(
            _MODEL_CATALOGUE_ADMIN_REFRESH_PATH,
            handle_model_catalogue_refresh_all,
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

        async def handle_memory_create(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_memory_create(
                    request,
                    authenticator=authenticator,
                    service=memory_queries,
                )

        async def handle_memory_edit(request: web.Request) -> web.Response:
            async with request_lock:
                return await _handle_memory_edit(
                    request,
                    authenticator=authenticator,
                    service=memory_queries,
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
        app.router.add_post(_MEMORY_RECORDS_PATH, handle_memory_create)
        app.router.add_get(_MEMORY_SEARCH_PATH, handle_memory_search)
        app.router.add_get(_MEMORY_DETAIL_PATH, handle_memory_detail)
        app.router.add_patch(_MEMORY_DETAIL_PATH, handle_memory_edit)
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
