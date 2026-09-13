"""Typed internal-API contracts shared by authorization and prompt rendering."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from kai.internal_api_scopes import InternalAPIScope


class InternalAPIOperation(StrEnum):
    """Stable operation names used by server routes and capability guidance."""

    JOB_CREATE = "job_create"
    JOB_LIST = "job_list"
    JOB_GET = "job_get"
    JOB_UPDATE = "job_update"
    JOB_DELETE = "job_delete"
    SERVICE_CALL = "service_call"
    MESSAGE_SEND = "message_send"
    FILE_SEND = "file_send"
    MEMORY_ADD = "memory_add"
    MEMORY_SEARCH = "memory_search"
    MEMORY_STATS = "memory_stats"
    MEMORY_DELETE_ALL = "memory_delete_all"
    AGENT_DELEGATE = "agent_delegate"
    CONTEXT_READ = "context_read"
    REACTION_SET = "reaction_set"
    COLLABORATION_MESSAGE = "collaboration_message"
    COLLABORATION_ARTIFACT = "collaboration_artifact"


@dataclass(frozen=True, slots=True)
class InternalAPIContract:
    """One authenticated HTTP operation and its documented JSON fields."""

    operation: InternalAPIOperation
    method: str
    path: str
    scope: InternalAPIScope
    required_fields: tuple[str, ...] = ()
    optional_fields: tuple[str, ...] = ()
    summary: str = ""
    collaboration_operation: str | None = None
    collaboration_summaries: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if self.method not in {"GET", "POST", "PATCH", "DELETE"}:
            raise ValueError("Internal API contract method is invalid")
        if not self.path.startswith("/api/"):
            raise ValueError("Internal API contract path must be internal")
        fields = (*self.required_fields, *self.optional_fields)
        if len(fields) != len(set(fields)) or any(not field for field in fields):
            raise ValueError("Internal API contract fields must be unique identifiers")
        collaboration_operations = (
            frozenset(self.collaboration_operation.split("|"))
            if self.collaboration_operation is not None
            else frozenset()
        )
        if self.collaboration_summaries and (
            not collaboration_operations
            or frozenset(operation for operation, _summary in self.collaboration_summaries) != collaboration_operations
            or any(not summary for _operation, summary in self.collaboration_summaries)
        ):
            raise ValueError("Collaboration summaries must cover each operation exactly once")

    def missing_required(self, payload: Mapping[str, object]) -> tuple[str, ...]:
        """Return required fields absent from an already-decoded JSON object."""
        return tuple(field for field in self.required_fields if field not in payload or payload[field] is None)

    @property
    def accepted_fields(self) -> frozenset[str]:
        """Return the complete JSON field set accepted by this operation."""
        return frozenset((*self.required_fields, *self.optional_fields))

    def render(self, webhook_port: int, *, collaboration_operations: frozenset[str] | None = None) -> str:
        """Render the compact operation line delivered to an authorized agent."""
        fields: list[str] = []
        if self.required_fields:
            fields.append(f"required JSON: {', '.join(self.required_fields)}")
        if self.optional_fields:
            fields.append(f"optional JSON: {', '.join(self.optional_fields)}")
        suffix = f"; {'; '.join(fields)}" if fields else ""
        summary = self.summary
        if self.collaboration_summaries and collaboration_operations is not None:
            summary = " ".join(
                operation_summary
                for operation, operation_summary in self.collaboration_summaries
                if operation in collaboration_operations
            )
        return (
            f"- {self.operation.value}: {self.method} http://localhost:{webhook_port}{self.path}{suffix}. {summary}"
        ).strip()


def _contract(
    operation: InternalAPIOperation,
    method: str,
    path: str,
    scope: InternalAPIScope,
    *,
    required: tuple[str, ...] = (),
    optional: tuple[str, ...] = (),
    summary: str,
    collaboration_operation: str | None = None,
    collaboration_summaries: tuple[tuple[str, str], ...] = (),
) -> InternalAPIContract:
    return InternalAPIContract(
        operation,
        method,
        path,
        scope,
        required,
        optional,
        summary,
        collaboration_operation,
        collaboration_summaries,
    )


INTERNAL_API_CONTRACTS: Mapping[InternalAPIOperation, InternalAPIContract] = {
    contract.operation: contract
    for contract in (
        _contract(
            InternalAPIOperation.JOB_CREATE,
            "POST",
            "/api/schedule",
            InternalAPIScope.JOBS_WRITE,
            required=("name", "prompt", "schedule_type", "schedule_data"),
            optional=("job_type", "auto_remove", "notify_on_check"),
            summary="Times in schedule_data are UTC; schedule_type is once, daily, or interval.",
        ),
        _contract(
            InternalAPIOperation.JOB_LIST,
            "GET",
            "/api/jobs",
            InternalAPIScope.JOBS_READ,
            summary="List active jobs in this credential-bound lane.",
        ),
        _contract(
            InternalAPIOperation.JOB_GET,
            "GET",
            "/api/jobs/{id}",
            InternalAPIScope.JOBS_READ,
            summary="Read one active job by its returned numeric ID.",
        ),
        _contract(
            InternalAPIOperation.JOB_UPDATE,
            "PATCH",
            "/api/jobs/{id}",
            InternalAPIScope.JOBS_WRITE,
            optional=("name", "prompt", "schedule_type", "schedule_data", "auto_remove", "notify_on_check"),
            summary="Update only the supplied fields.",
        ),
        _contract(
            InternalAPIOperation.JOB_DELETE,
            "DELETE",
            "/api/jobs/{id}",
            InternalAPIScope.JOBS_WRITE,
            summary="Delete one active job by its returned numeric ID.",
        ),
        _contract(
            InternalAPIOperation.SERVICE_CALL,
            "POST",
            "/api/services/{name}",
            InternalAPIScope.SERVICES_CALL,
            optional=("body", "params", "path_suffix"),
            summary="Only service names listed below are authorized.",
        ),
        _contract(
            InternalAPIOperation.MESSAGE_SEND,
            "POST",
            "/api/send-message",
            InternalAPIScope.MESSAGES_SEND,
            required=("text",),
            optional=("idempotency_key",),
            summary="Record a proactive message canonically before optional adapter delivery.",
        ),
        _contract(
            InternalAPIOperation.FILE_SEND,
            "POST",
            "/api/send-file",
            InternalAPIScope.FILES_SEND,
            required=("path",),
            optional=("caption", "idempotency_key"),
            summary="Publish an allowed workspace, incoming, or private-outbox file canonically.",
        ),
        _contract(
            InternalAPIOperation.MEMORY_ADD,
            "POST",
            "/api/memory/add",
            InternalAPIScope.MEMORY_ADD,
            required=("content",),
            optional=("memory_type", "tags", "metadata"),
            summary="Store one explicit fact with server-owned principal and project provenance.",
        ),
        _contract(
            InternalAPIOperation.MEMORY_SEARCH,
            "POST",
            "/api/memory/search",
            InternalAPIScope.MEMORY_READ,
            required=("query",),
            optional=("limit",),
            summary="Search only memory visible to the credential-bound principal and workspace.",
        ),
        _contract(
            InternalAPIOperation.MEMORY_STATS,
            "GET",
            "/api/memory/stats",
            InternalAPIScope.MEMORY_READ,
            summary="Return memory statistics; null confidence values mean no extracted facts.",
        ),
        _contract(
            InternalAPIOperation.MEMORY_DELETE_ALL,
            "DELETE",
            "/api/memory/all",
            InternalAPIScope.MEMORY_DELETE_ALL,
            required=("confirm",),
            summary="Destructive operator-only memory deletion.",
        ),
        _contract(
            InternalAPIOperation.AGENT_DELEGATE,
            "POST",
            "/api/agent-delegations",
            InternalAPIScope.COLLABORATION_INVOKE,
            required=("target_handle", "task", "idempotency_key"),
            optional=("context",),
            summary="Delegate once to an attached agent and await its bounded terminal result.",
            collaboration_operation="agent_delegation",
        ),
        _contract(
            InternalAPIOperation.CONTEXT_READ,
            "POST",
            "/api/collaboration/context",
            InternalAPIScope.COLLABORATION_INVOKE,
            required=("idempotency_key",),
            optional=("cursor", "limit"),
            summary="Read the active attempt's bounded immutable conversation snapshot.",
            collaboration_operation="context_read",
        ),
        _contract(
            InternalAPIOperation.REACTION_SET,
            "POST",
            "/api/collaboration/reactions",
            InternalAPIScope.COLLABORATION_INVOKE,
            required=("message_id", "reaction", "active", "idempotency_key"),
            summary="Set participation metadata; reactions never wake or delegate.",
            collaboration_operation="reaction",
        ),
        _contract(
            InternalAPIOperation.COLLABORATION_MESSAGE,
            "POST",
            "/api/collaboration/messages",
            InternalAPIScope.COLLABORATION_INVOKE,
            required=("kind", "body", "idempotency_key"),
            summary="Publish progress or a current-thread reply without replacing the terminal response.",
            collaboration_operation="progress_publish|thread_reply",
            collaboration_summaries=(
                (
                    "progress_publish",
                    "Set kind to 'progress' and publish progress without replacing the terminal response.",
                ),
                (
                    "thread_reply",
                    "Set kind to 'thread_reply' and publish a current-thread reply without replacing the terminal response.",
                ),
            ),
        ),
        _contract(
            InternalAPIOperation.COLLABORATION_ARTIFACT,
            "POST",
            "/api/collaboration/artifacts",
            InternalAPIScope.COLLABORATION_INVOKE,
            required=("path", "caption", "idempotency_key"),
            summary="Publish one allowed artifact into the active attempt's exact context.",
            collaboration_operation="artifact_publish",
        ),
    )
}


def contract(operation: InternalAPIOperation) -> InternalAPIContract:
    """Return the single authoritative contract for an operation."""
    return INTERNAL_API_CONTRACTS[operation]


def persistent_contracts(
    scopes: Iterable[InternalAPIScope], *, memory_enabled: bool
) -> tuple[InternalAPIContract, ...]:
    """Return promptable non-collaboration contracts for an exact base credential."""
    allowed = frozenset(scopes)
    return tuple(
        item
        for item in INTERNAL_API_CONTRACTS.values()
        if item.scope in allowed
        and item.scope is not InternalAPIScope.COLLABORATION_INVOKE
        and item.scope is not InternalAPIScope.MEMORY_DELETE_ALL
        and (memory_enabled or item.scope not in {InternalAPIScope.MEMORY_ADD, InternalAPIScope.MEMORY_READ})
    )


def collaboration_contracts(operations: Iterable[str]) -> tuple[InternalAPIContract, ...]:
    """Return only contracts enabled by this exact attempt's operation set."""
    allowed = frozenset(operations)
    selected: list[InternalAPIContract] = []
    for item in INTERNAL_API_CONTRACTS.values():
        if item.collaboration_operation is None:
            continue
        alternatives = frozenset(item.collaboration_operation.split("|"))
        if allowed & alternatives:
            selected.append(item)
    return tuple(selected)
