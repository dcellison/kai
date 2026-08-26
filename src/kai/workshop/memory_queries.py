"""Canonical, principal-scoped query and management service for memory."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from kai import memory
from kai.config import Config
from kai.workshop.authorization import CanonicalChannelAuthorizer
from kai.workshop.domain import AgentId, ChannelId, MessageId, PrincipalId, RunId
from kai.workshop.execution_state import (
    WorkshopExecutionStateNamespace,
    WorkshopExecutionStateRegistry,
)
from kai.workshop.runtime_pool import WorkshopRuntimePool
from kai.workshop.store import WorkshopEventStore
from kai.workspace_utils import is_workspace_allowed

DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 100
MAX_SEARCH_LIMIT = 50
MAX_QUERY_CHARACTERS = 2_000
MAX_CURSOR_CHARACTERS = 2_048
MAX_PREVIEW_CHARACTERS = 500
MAX_CONTENT_CHARACTERS = 100_000
MAX_COMPACT_RECALL_CHARACTERS = 120_000
MAX_SOURCE_BODY_CHARACTERS = 50_000
MAX_MUTATION_TARGETS = 50
MAX_MEMORY_TAGS = 32
MAX_MEMORY_TAG_CHARACTERS = 128
MAX_EPISODE_FIELD_CHARACTERS = 20_000
MAX_REQUEST_ID_CHARACTERS = 128
MEMORY_MANAGEMENT_AUDIT_EVENT = "workshop.memory.mutation"
MEMORY_CONTENT_AUDIT_EVENT = "workshop.memory.content_mutation"
_CURSOR_VERSION = 1
_REVISION_VERSION = 1
_VALID_KINDS = frozenset({"fact", "episode"})
_VALID_SCOPES = frozenset({"global", "project", "task"})
_VALID_ORDERS = frozenset({"newest", "oldest"})
_VALID_MUTATION_SCOPES = frozenset({memory.SCOPE_GLOBAL, memory.SCOPE_PROJECT})
_VALID_OUTCOME_QUALITIES = frozenset({"success", "partial", "failure"})

log = logging.getLogger(__name__)


class WorkshopMemoryQueryError(RuntimeError):
    """Base error for the Workshop memory read boundary."""


class WorkshopMemoryAccessDenied(WorkshopMemoryQueryError):
    """A canonical principal has no memory-query authority."""


class WorkshopMemoryValidationError(WorkshopMemoryQueryError):
    """A bounded memory query is malformed."""


class WorkshopMemoryCursorError(WorkshopMemoryValidationError):
    """A memory-page cursor is malformed or belongs to another query."""


class WorkshopMemoryNotFound(WorkshopMemoryQueryError):
    """A visible memory does not exist for the authenticated principal."""


class WorkshopMemoryResponseTooLarge(WorkshopMemoryQueryError):
    """A stored record exceeds the bounded client response contract."""


class WorkshopMemoryConflict(WorkshopMemoryQueryError):
    """An optimistic memory revision no longer matches the stored row."""

    def __init__(self, current_revision: str) -> None:
        super().__init__("Memory changed since it was opened")
        self.current_revision = current_revision


class WorkshopMemoryMutationFailed(WorkshopMemoryQueryError):
    """A provider mutation failed without producing a verified result."""


@dataclass(frozen=True, slots=True)
class MemoryQueryAuthority:
    principal_id: PrincipalId
    search_namespace: WorkshopExecutionStateNamespace | None


@dataclass(frozen=True, slots=True)
class MemoryQueryFilters:
    kind: str | None = None
    source: str | None = None
    memory_type: str | None = None
    tag: str | None = None
    scope: str | None = None
    project_id: str | None = None


EMPTY_MEMORY_FILTERS = MemoryQueryFilters()


@dataclass(frozen=True, slots=True)
class MemoryScopeSnapshot:
    scope: str
    project_id: str | None
    scope_confidence: float
    scope_source: str
    legacy_defaulted: bool
    invalid_defaulted: bool
    retrievable: bool
    exclusion_reason: str | None


@dataclass(frozen=True, slots=True)
class MemoryRecordSummary:
    memory_id: str
    kind: str
    source: str
    memory_type: str
    preview: str
    tags: tuple[str, ...]
    speaker: str
    confidence: float
    created_at: str
    updated_at: str
    revision: str
    scope: MemoryScopeSnapshot


@dataclass(frozen=True, slots=True)
class MemoryRecordDetail:
    record: MemoryRecordSummary
    content: str
    compact_recall: str
    confirmation_quote: str | None
    prompt_version: str | None
    episode: dict[str, object] | None


@dataclass(frozen=True, slots=True)
class MemoryFactEdit:
    content: str
    tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MemoryEpisodeEdit:
    goal: str
    context: str
    approach: str
    outcome: str
    outcome_quality: str
    lessons: str | None
    tags: tuple[str, ...]
    actors: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MemoryEditSnapshot:
    record: MemoryRecordDetail
    changed_fields: tuple[str, ...]
    idempotent_replay: bool


@dataclass(frozen=True, slots=True)
class MemoryCreationSnapshot:
    record: MemoryRecordDetail
    created: bool


@dataclass(frozen=True, slots=True)
class MemoryRecordPage:
    records: tuple[MemoryRecordSummary, ...]
    next_cursor: str | None


@dataclass(frozen=True, slots=True)
class MemorySearchHit:
    record: MemoryRecordSummary
    raw_score: float
    adjusted_score: float
    compact_recall: str


@dataclass(frozen=True, slots=True)
class MemorySearchSnapshot:
    hits: tuple[MemorySearchHit, ...]
    active_project_id: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class MemoryProjectOption:
    project_id: str
    display_name: str


@dataclass(frozen=True, slots=True)
class MemoryStatsSnapshot:
    total: int
    facts: int
    episodes: int
    by_source: dict[str, int]
    by_type: dict[str, int]
    by_scope: dict[str, int]
    allowed_projects: tuple[MemoryProjectOption, ...]


@dataclass(frozen=True, slots=True)
class MemoryMutationResult:
    memory_id: str
    outcome: Literal["succeeded", "not_found", "stale", "failed"]
    prior_scope: MemoryScopeSnapshot | None
    new_scope: MemoryScopeSnapshot | None


@dataclass(frozen=True, slots=True)
class MemoryMutationBatch:
    operation: Literal["move_scope", "delete"]
    results: tuple[MemoryMutationResult, ...]


@dataclass(frozen=True, slots=True)
class MemorySourceMessage:
    message_id: MessageId
    channel_id: ChannelId
    author_principal_id: PrincipalId
    author_kind: str
    author_display_name: str
    body: str
    created_at: str


@dataclass(frozen=True, slots=True)
class MemorySourceContext:
    status: Literal["available", "unavailable"]
    reason: str | None
    run_id: RunId | None
    source: MemorySourceMessage | None
    result: MemorySourceMessage | None


def _bounded_text(value: object, *, maximum: int) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    return value[:maximum]


def _record_kind(result: memory.MemoryResult) -> str:
    return "episode" if result.metadata.get("source") == "episode" else "fact"


def _source(result: memory.MemoryResult) -> str:
    source = result.metadata.get("source")
    return str(source) if isinstance(source, str) and source else "legacy"


def _tags(result: memory.MemoryResult) -> tuple[str, ...]:
    raw = result.metadata.get("tags")
    if not isinstance(raw, list):
        return ()
    return tuple(tag[:128] for tag in raw[:32] if isinstance(tag, str) and tag)


def _memory_revision(result: memory.MemoryResult) -> str:
    """Return an opaque digest covering every client-visible mutable field."""
    payload = json.dumps(
        {
            "version": _REVISION_VERSION,
            "memory_id": result.id,
            "content": result.text,
            "metadata": result.metadata,
            "created_at": result.created_at,
            "updated_at": result.updated_at,
        },
        default=str,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()
    return f"mr{_REVISION_VERSION}_{digest}"


def _sort_key(result: memory.MemoryResult) -> tuple[str, str]:
    return (result.updated_at or result.created_at, result.id)


def _compact_recall(
    result: memory.MemoryResult,
    *,
    resolved_scope: memory.ResolvedMemoryScope | None = None,
    speaker: str | None = None,
    confidence: float | None = None,
) -> str:
    rendered = memory.format_memory_result_for_recall(
        result,
        resolved_scope=resolved_scope,
        speaker=speaker,
        confidence=confidence,
    )
    if len(rendered) > MAX_COMPACT_RECALL_CHARACTERS:
        raise WorkshopMemoryResponseTooLarge("Memory recall representation is too large")
    return rendered


def _filter_fingerprint(filters: MemoryQueryFilters, *, order: str) -> str:
    encoded = json.dumps(
        {
            "kind": filters.kind,
            "source": filters.source,
            "memory_type": filters.memory_type,
            "tag": filters.tag,
            "scope": filters.scope,
            "project_id": filters.project_id,
            "order": order,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _encode_cursor(
    filters: MemoryQueryFilters,
    result: memory.MemoryResult,
    *,
    order: str,
) -> str:
    payload = {
        "v": _CURSOR_VERSION,
        "f": _filter_fingerprint(filters, order=order),
        "t": _sort_key(result)[0],
        "i": result.id,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()


def _decode_cursor(
    cursor: str,
    filters: MemoryQueryFilters,
    *,
    order: str,
) -> tuple[str, str]:
    if not cursor or len(cursor) > MAX_CURSOR_CHARACTERS:
        raise WorkshopMemoryCursorError("Invalid memory cursor")
    try:
        raw = base64.b64decode(
            cursor + "=" * (-len(cursor) % 4),
            altchars=b"-_",
            validate=True,
        )
        payload = json.loads(raw)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as exc:
        raise WorkshopMemoryCursorError("Invalid memory cursor") from exc
    if (
        not isinstance(payload, dict)
        or set(payload) != {"v", "f", "t", "i"}
        or payload.get("v") != _CURSOR_VERSION
        or payload.get("f") != _filter_fingerprint(filters, order=order)
        or not isinstance(payload.get("t"), str)
        or not isinstance(payload.get("i"), str)
        or not payload["i"]
    ):
        raise WorkshopMemoryCursorError("Invalid memory cursor")
    return payload["t"], payload["i"]


class WorkshopMemoryQueryService:
    """Query and mutate semantic memory through canonical authority only."""

    def __init__(
        self,
        config: Config,
        store: WorkshopEventStore,
        runtime_pool: WorkshopRuntimePool,
        execution_state: WorkshopExecutionStateRegistry,
    ) -> None:
        self._config = config
        self._store = store
        self._runtime_pool = runtime_pool
        self._channel_authorizer = CanonicalChannelAuthorizer(store)
        by_principal: dict[PrincipalId, list[WorkshopExecutionStateNamespace]] = {}
        for namespace in execution_state.namespaces:
            by_principal.setdefault(namespace.principal_id, []).append(namespace)
        self._namespaces = {principal_id: tuple(namespaces) for principal_id, namespaces in by_principal.items()}
        self._mutation_locks: dict[PrincipalId, asyncio.Lock] = {}

    def authority_for_principal(
        self,
        principal_id: str | PrincipalId,
    ) -> MemoryQueryAuthority:
        try:
            canonical = principal_id if isinstance(principal_id, PrincipalId) else PrincipalId(principal_id)
        except (TypeError, ValueError) as exc:
            raise WorkshopMemoryAccessDenied("Memory access denied") from exc
        namespaces = self._namespaces.get(canonical)
        if not namespaces:
            raise WorkshopMemoryAccessDenied("Memory access denied")
        return MemoryQueryAuthority(
            principal_id=canonical,
            search_namespace=namespaces[0] if len(namespaces) == 1 else None,
        )

    @staticmethod
    def validate_filters(filters: MemoryQueryFilters) -> None:
        for value in (
            filters.kind,
            filters.source,
            filters.memory_type,
            filters.tag,
            filters.scope,
            filters.project_id,
        ):
            if value is not None and (not value or len(value) > 128):
                raise WorkshopMemoryValidationError("Invalid memory filter")
        if filters.kind is not None and filters.kind not in _VALID_KINDS:
            raise WorkshopMemoryValidationError("Invalid memory kind")
        if filters.scope is not None and filters.scope not in _VALID_SCOPES:
            raise WorkshopMemoryValidationError("Invalid memory scope")

    async def _all_visible(
        self,
        authority: MemoryQueryAuthority,
    ) -> list[memory.MemoryResult]:
        rows = await asyncio.to_thread(
            memory.get_all,
            user_id=str(authority.principal_id),
            limit=None,
        )
        return [row for row in rows if row.metadata.get("source") in memory.USER_VISIBLE_SOURCES]

    async def _allowed_project_id(
        self,
        authority: MemoryQueryAuthority,
    ) -> str | None:
        namespace = authority.search_namespace
        if namespace is None:
            return None
        workspace = await self._runtime_pool.get_effective_workspace(namespace.runtime_profile_id)
        from kai.memory_projects import detect_active_memory_project, merged_registry

        active = detect_active_memory_project(
            Path(workspace),
            merged_registry(self._config.memory_projects),
        )
        return active.project_id if active is not None and active.memory_enabled else None

    async def allowed_projects(
        self,
        authority: MemoryQueryAuthority,
    ) -> tuple[MemoryProjectOption, ...]:
        """Return memory-enabled projects reachable by an owned runtime."""
        from kai.memory_projects import detect_active_memory_project, merged_registry

        registry = merged_registry(self._config.memory_projects)
        if not registry:
            return ()
        authorized: dict[str, MemoryProjectOption] = {}
        for namespace in self._namespaces.get(authority.principal_id, ()):
            profile_id = namespace.runtime_profile_id
            home = self._runtime_pool.get_home_workspace(profile_id).expanduser().resolve()
            current = (await self._runtime_pool.get_effective_workspace(profile_id)).expanduser().resolve()
            base, allowed = await self._runtime_pool.resolve_workspace_access(profile_id)
            candidates = (home, current, *allowed)
            for candidate in candidates:
                active = detect_active_memory_project(candidate, registry)
                if active is not None and active.memory_enabled:
                    authorized[active.project_id] = MemoryProjectOption(
                        active.project_id,
                        active.display_name,
                    )
            for project in registry.values():
                if not project.memory_enabled:
                    continue
                if any(
                    root == home or root.is_relative_to(home) or is_workspace_allowed(root, base, allowed)
                    for root in project.workspace_roots
                ):
                    authorized[project.project_id] = MemoryProjectOption(
                        project.project_id,
                        project.display_name,
                    )
        return tuple(sorted(authorized.values(), key=lambda item: (item.display_name.casefold(), item.project_id)))

    @staticmethod
    def _scope_snapshot(
        result: memory.MemoryResult,
        *,
        allowed_project_id: str | None,
    ) -> MemoryScopeSnapshot:
        resolved = memory.resolve_memory_scope(result.metadata)
        reason = memory.memory_scope_admission_reason(
            resolved,
            allowed_project_id=allowed_project_id,
        )
        return MemoryScopeSnapshot(
            scope=resolved.scope,
            project_id=resolved.project_id,
            scope_confidence=float(resolved.scope_confidence),
            scope_source=resolved.scope_source,
            legacy_defaulted=resolved.legacy_defaulted,
            invalid_defaulted=resolved.invalid_defaulted,
            retrievable=reason is None,
            exclusion_reason=reason,
        )

    @classmethod
    def _summary(
        cls,
        result: memory.MemoryResult,
        *,
        allowed_project_id: str | None,
    ) -> MemoryRecordSummary:
        speaker, confidence = memory.read_time_memory_speaker(result.metadata)
        return MemoryRecordSummary(
            memory_id=result.id,
            kind=_record_kind(result),
            source=_source(result),
            memory_type=result.memory_type,
            preview=result.text[:MAX_PREVIEW_CHARACTERS],
            tags=_tags(result),
            speaker=speaker,
            confidence=float(confidence),
            created_at=result.created_at,
            updated_at=result.updated_at,
            revision=_memory_revision(result),
            scope=cls._scope_snapshot(
                result,
                allowed_project_id=allowed_project_id,
            ),
        )

    @staticmethod
    def _matches(
        result: memory.MemoryResult,
        filters: MemoryQueryFilters,
    ) -> bool:
        resolved = memory.resolve_memory_scope(result.metadata)
        return all(
            (
                filters.kind is None or _record_kind(result) == filters.kind,
                filters.source is None or _source(result) == filters.source,
                filters.memory_type is None or result.memory_type == filters.memory_type,
                filters.tag is None or filters.tag in _tags(result),
                filters.scope is None or resolved.scope == filters.scope,
                filters.project_id is None or resolved.project_id == filters.project_id,
            )
        )

    async def list_records(
        self,
        authority: MemoryQueryAuthority,
        *,
        filters: MemoryQueryFilters = EMPTY_MEMORY_FILTERS,
        limit: int = DEFAULT_PAGE_SIZE,
        cursor: str | None = None,
        order: str = "newest",
    ) -> MemoryRecordPage:
        self.validate_filters(filters)
        if order not in _VALID_ORDERS:
            raise WorkshopMemoryValidationError("Invalid memory record order")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_PAGE_SIZE:
            raise WorkshopMemoryValidationError(f"Memory page size must be between 1 and {MAX_PAGE_SIZE}")
        anchor = _decode_cursor(cursor, filters, order=order) if cursor is not None else None
        rows = [row for row in await self._all_visible(authority) if self._matches(row, filters)]
        reverse = order == "newest"
        rows.sort(key=_sort_key, reverse=reverse)
        if anchor is not None:
            rows = [row for row in rows if (_sort_key(row) < anchor if reverse else _sort_key(row) > anchor)]
        selected = rows[: limit + 1]
        has_more = len(selected) > limit
        selected = selected[:limit]
        allowed_project_id = await self._allowed_project_id(authority)
        return MemoryRecordPage(
            records=tuple(self._summary(row, allowed_project_id=allowed_project_id) for row in selected),
            next_cursor=(_encode_cursor(filters, selected[-1], order=order) if has_more and selected else None),
        )

    async def stats(
        self,
        authority: MemoryQueryAuthority,
    ) -> MemoryStatsSnapshot:
        rows = await self._all_visible(authority)
        by_source: dict[str, int] = {}
        by_type: dict[str, int] = {}
        by_scope: dict[str, int] = {}
        facts = 0
        episodes = 0
        for row in rows:
            source = _source(row)
            by_source[source] = by_source.get(source, 0) + 1
            by_type[row.memory_type] = by_type.get(row.memory_type, 0) + 1
            resolved = memory.resolve_memory_scope(row.metadata)
            scope_key = (
                f"project:{resolved.project_id}"
                if resolved.scope == "project" and resolved.project_id
                else resolved.scope
            )
            if resolved.legacy_defaulted:
                scope_key = "global_legacy"
            elif resolved.invalid_defaulted:
                scope_key = "invalid"
            by_scope[scope_key] = by_scope.get(scope_key, 0) + 1
            if _record_kind(row) == "episode":
                episodes += 1
            else:
                facts += 1
        return MemoryStatsSnapshot(
            total=len(rows),
            facts=facts,
            episodes=episodes,
            by_source=dict(sorted(by_source.items())),
            by_type=dict(sorted(by_type.items())),
            by_scope=dict(sorted(by_scope.items())),
            allowed_projects=await self.allowed_projects(authority),
        )

    @staticmethod
    def _validate_memory_ids(memory_ids: Sequence[str]) -> tuple[str, ...]:
        if isinstance(memory_ids, (str, bytes)):
            raise WorkshopMemoryValidationError("Memory identifiers must be a list")
        checked = tuple(memory_ids)
        if not 1 <= len(checked) <= MAX_MUTATION_TARGETS:
            raise WorkshopMemoryValidationError(
                f"A memory mutation must contain between 1 and {MAX_MUTATION_TARGETS} identifiers"
            )
        if any(not isinstance(memory_id, str) or not memory_id or len(memory_id) > 256 for memory_id in checked):
            raise WorkshopMemoryValidationError("Invalid memory identifier")
        if len(set(checked)) != len(checked):
            raise WorkshopMemoryValidationError("Memory identifiers must be unique")
        return checked

    async def _scope_metadata(
        self,
        authority: MemoryQueryAuthority,
        *,
        scope: str,
        project_id: str | None,
    ) -> dict[str, object]:
        if scope not in _VALID_MUTATION_SCOPES:
            raise WorkshopMemoryValidationError("Memory scope must be global or project")
        if scope == memory.SCOPE_GLOBAL:
            if project_id is not None:
                raise WorkshopMemoryValidationError("Global memory scope cannot include a project")
        else:
            if not isinstance(project_id, str) or not project_id or len(project_id) > 128:
                raise WorkshopMemoryValidationError("Project memory scope requires a project")
            allowed = {item.project_id for item in await self.allowed_projects(authority)}
            if project_id not in allowed:
                raise WorkshopMemoryAccessDenied("Memory project access denied")
        try:
            return memory.build_scope_metadata(
                scope=scope,
                project_id=project_id,
                scope_confidence=1.0,
                scope_source=memory.SCOPE_SOURCE_OPERATOR,
            )
        except ValueError as exc:
            raise WorkshopMemoryValidationError("Invalid memory scope") from exc

    @staticmethod
    def _validate_request_id(request_id: str) -> str:
        if (
            not isinstance(request_id, str)
            or not request_id.strip()
            or len(request_id) > MAX_REQUEST_ID_CHARACTERS
            or any(character.isspace() for character in request_id)
        ):
            raise WorkshopMemoryValidationError("Invalid memory mutation request identifier")
        return request_id

    @staticmethod
    def _validate_revision(revision: str) -> str:
        if not isinstance(revision, str) or not revision.startswith(f"mr{_REVISION_VERSION}_") or len(revision) > 128:
            raise WorkshopMemoryValidationError("Invalid memory revision")
        return revision

    @staticmethod
    def _validate_text(value: str, *, field: str, maximum: int) -> str:
        if not isinstance(value, str) or not value.strip():
            raise WorkshopMemoryValidationError(f"{field} is required")
        cleaned = value.strip()
        if len(cleaned) > maximum:
            raise WorkshopMemoryValidationError(f"{field} is too long")
        return cleaned

    @staticmethod
    def _validate_tags(values: Sequence[str], *, field: str = "Tags") -> tuple[str, ...]:
        if isinstance(values, (str, bytes)):
            raise WorkshopMemoryValidationError(f"{field} must be a list")
        checked = tuple(values)
        if len(checked) > MAX_MEMORY_TAGS:
            raise WorkshopMemoryValidationError(f"{field} must contain at most {MAX_MEMORY_TAGS} values")
        cleaned: list[str] = []
        for value in checked:
            if not isinstance(value, str) or not value.strip() or len(value.strip()) > MAX_MEMORY_TAG_CHARACTERS:
                raise WorkshopMemoryValidationError(f"Invalid {field.lower()} value")
            normalized = value.strip()
            if normalized in cleaned:
                raise WorkshopMemoryValidationError(f"{field} must not contain duplicates")
            cleaned.append(normalized)
        return tuple(cleaned)

    @staticmethod
    def _namespace_for_mutation(authority: MemoryQueryAuthority) -> WorkshopExecutionStateNamespace:
        namespace = authority.search_namespace
        if namespace is None:
            raise WorkshopMemoryAccessDenied("Memory mutation requires one unambiguous runtime profile")
        return namespace

    def _audit_content_mutation(
        self,
        authority: MemoryQueryAuthority,
        *,
        operation: Literal["create", "edit"],
        memory_id: str | None,
        changed_fields: Sequence[str],
        outcome: Literal["succeeded", "idempotent", "conflict", "failed"],
    ) -> None:
        log.info(
            "%s %s",
            MEMORY_CONTENT_AUDIT_EVENT,
            json.dumps(
                {
                    "actor_principal_id": str(authority.principal_id),
                    "operation": operation,
                    "memory_id": memory_id,
                    "changed_fields": sorted(changed_fields),
                    "timestamp": datetime.now(UTC).isoformat(),
                    "outcome": outcome,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    @staticmethod
    def _fact_matches(result: memory.MemoryResult, *, content: str, tags: Sequence[str]) -> bool:
        return (
            _record_kind(result) == "fact"
            and result.text == content
            and tuple(result.metadata.get("tags") or ()) == tuple(tags)
        )

    @staticmethod
    def _episode_matches(result: memory.MemoryResult, edit: MemoryEpisodeEdit) -> bool:
        metadata = result.metadata
        return (
            _record_kind(result) == "episode"
            and result.text == f"{edit.goal}\n\n{edit.context}"
            and metadata.get("goal") == edit.goal
            and metadata.get("context") == edit.context
            and metadata.get("approach") == edit.approach
            and metadata.get("outcome") == edit.outcome
            and metadata.get("outcome_quality") == edit.outcome_quality
            and metadata.get("lessons") == edit.lessons
            and tuple(metadata.get("tags") or ()) == edit.tags
            and tuple(metadata.get("actors") or ()) == edit.actors
        )

    async def create_fact(
        self,
        authority: MemoryQueryAuthority,
        *,
        content: str,
        tags: Sequence[str],
        scope: str,
        project_id: str | None,
        request_id: str,
    ) -> MemoryCreationSnapshot:
        namespace = self._namespace_for_mutation(authority)
        cleaned_content = self._validate_text(content, field="Content", maximum=MAX_CONTENT_CHARACTERS)
        cleaned_tags = self._validate_tags(tags)
        checked_request_id = self._validate_request_id(request_id)
        scope_metadata = await self._scope_metadata(authority, scope=scope, project_id=project_id)
        lock = self._mutation_locks.setdefault(authority.principal_id, asyncio.Lock())
        async with lock:
            for existing in await self._all_visible(authority):
                if existing.metadata.get("operator_creation_request_id") != checked_request_id:
                    continue
                if (
                    existing.metadata.get("source") == "explicit"
                    and self._fact_matches(existing, content=cleaned_content, tags=cleaned_tags)
                    and all(existing.metadata.get(key) == value for key, value in scope_metadata.items())
                ):
                    self._audit_content_mutation(
                        authority,
                        operation="create",
                        memory_id=existing.id,
                        changed_fields=("content", "tags", "scope"),
                        outcome="idempotent",
                    )
                    return MemoryCreationSnapshot(await self.detail(authority, existing.id), False)
                self._audit_content_mutation(
                    authority,
                    operation="create",
                    memory_id=existing.id,
                    changed_fields=("content", "tags", "scope"),
                    outcome="conflict",
                )
                raise WorkshopMemoryConflict(_memory_revision(existing))

            now = datetime.now(UTC).isoformat()
            metadata: dict[str, object] = {
                "source": "explicit",
                "speaker": "user",
                "confidence": 1.0,
                "operator_created_at": now,
                "operator_created_by_principal_id": str(authority.principal_id),
                "operator_creation_request_id": checked_request_id,
                **scope_metadata,
            }
            memory_id = await asyncio.to_thread(
                memory.add_structured,
                cleaned_content,
                user_id=str(authority.principal_id),
                memory_type="fact",
                tags=list(cleaned_tags),
                metadata=metadata,
                runtime_profile_id=str(namespace.runtime_profile_id),
            )
            if not isinstance(memory_id, str) or not memory_id:
                self._audit_content_mutation(
                    authority,
                    operation="create",
                    memory_id=None,
                    changed_fields=("content", "tags", "scope"),
                    outcome="failed",
                )
                raise WorkshopMemoryMutationFailed("Memory creation failed")
            stored = await asyncio.to_thread(
                memory.get_by_id,
                user_id=str(authority.principal_id),
                memory_id=memory_id,
                runtime_profile_id=str(namespace.runtime_profile_id),
            )
            if (
                stored is None
                or not self._fact_matches(stored, content=cleaned_content, tags=cleaned_tags)
                or any(stored.metadata.get(key) != value for key, value in scope_metadata.items())
                or stored.metadata.get("operator_creation_request_id") != checked_request_id
            ):
                self._audit_content_mutation(
                    authority,
                    operation="create",
                    memory_id=memory_id,
                    changed_fields=("content", "tags", "scope"),
                    outcome="failed",
                )
                raise WorkshopMemoryMutationFailed("Memory creation could not be verified")
            self._audit_content_mutation(
                authority,
                operation="create",
                memory_id=memory_id,
                changed_fields=("content", "tags", "scope"),
                outcome="succeeded",
            )
            return MemoryCreationSnapshot(await self.detail(authority, memory_id), True)

    async def edit(
        self,
        authority: MemoryQueryAuthority,
        memory_id: str,
        *,
        revision: str,
        request_id: str,
        edit: MemoryFactEdit | MemoryEpisodeEdit,
    ) -> MemoryEditSnapshot:
        if not isinstance(memory_id, str) or not memory_id or len(memory_id) > 256:
            raise WorkshopMemoryValidationError("Invalid memory identifier")
        checked_revision = self._validate_revision(revision)
        checked_request_id = self._validate_request_id(request_id)
        namespace = self._namespace_for_mutation(authority)
        if isinstance(edit, MemoryFactEdit):
            normalized: MemoryFactEdit | MemoryEpisodeEdit = MemoryFactEdit(
                self._validate_text(edit.content, field="Content", maximum=MAX_CONTENT_CHARACTERS),
                self._validate_tags(edit.tags),
            )
        else:
            normalized = MemoryEpisodeEdit(
                goal=self._validate_text(edit.goal, field="Goal", maximum=MAX_EPISODE_FIELD_CHARACTERS),
                context=self._validate_text(edit.context, field="Context", maximum=MAX_EPISODE_FIELD_CHARACTERS),
                approach=self._validate_text(edit.approach, field="Approach", maximum=MAX_EPISODE_FIELD_CHARACTERS),
                outcome=self._validate_text(edit.outcome, field="Outcome", maximum=MAX_EPISODE_FIELD_CHARACTERS),
                outcome_quality=edit.outcome_quality,
                lessons=(
                    self._validate_text(edit.lessons, field="Lessons", maximum=MAX_EPISODE_FIELD_CHARACTERS)
                    if edit.lessons is not None and edit.lessons.strip()
                    else None
                ),
                tags=self._validate_tags(edit.tags),
                actors=self._validate_tags(edit.actors, field="Actors"),
            )
            if normalized.outcome_quality not in _VALID_OUTCOME_QUALITIES:
                raise WorkshopMemoryValidationError("Outcome quality must be success, partial, or failure")

        lock = self._mutation_locks.setdefault(authority.principal_id, asyncio.Lock())
        async with lock:
            existing = await asyncio.to_thread(
                memory.get_by_id,
                user_id=str(authority.principal_id),
                memory_id=memory_id,
                runtime_profile_id=str(namespace.runtime_profile_id),
            )
            if existing is None:
                raise WorkshopMemoryNotFound("Memory not found")
            if isinstance(normalized, MemoryFactEdit) and _record_kind(existing) != "fact":
                raise WorkshopMemoryValidationError("Memory kind does not match the edit request")
            if isinstance(normalized, MemoryEpisodeEdit) and _record_kind(existing) != "episode":
                raise WorkshopMemoryValidationError("Memory kind does not match the edit request")

            matches = (
                self._fact_matches(existing, content=normalized.content, tags=normalized.tags)
                if isinstance(normalized, MemoryFactEdit)
                else self._episode_matches(existing, normalized)
            )
            if _memory_revision(existing) != checked_revision:
                if existing.metadata.get("operator_edit_request_id") == checked_request_id and matches:
                    self._audit_content_mutation(
                        authority,
                        operation="edit",
                        memory_id=memory_id,
                        changed_fields=(),
                        outcome="idempotent",
                    )
                    return MemoryEditSnapshot(await self.detail(authority, memory_id), (), True)
                self._audit_content_mutation(
                    authority,
                    operation="edit",
                    memory_id=memory_id,
                    changed_fields=(),
                    outcome="conflict",
                )
                raise WorkshopMemoryConflict(_memory_revision(existing))

            merged = dict(existing.metadata)
            changed_fields: list[str] = []
            if isinstance(normalized, MemoryFactEdit):
                data = normalized.content
                if existing.text != data:
                    changed_fields.append("content")
                if tuple(existing.metadata.get("tags") or ()) != normalized.tags:
                    changed_fields.append("tags")
                merged["tags"] = list(normalized.tags)
            else:
                data = f"{normalized.goal}\n\n{normalized.context}"
                episode_values: dict[str, object] = {
                    "goal": normalized.goal,
                    "context": normalized.context,
                    "approach": normalized.approach,
                    "outcome": normalized.outcome,
                    "outcome_quality": normalized.outcome_quality,
                    "tags": list(normalized.tags),
                    "actors": list(normalized.actors),
                }
                for field, value in episode_values.items():
                    current = existing.metadata.get(field)
                    if field in {"tags", "actors"}:
                        current = list(current or ())
                    if current != value:
                        changed_fields.append(field)
                    merged[field] = value
                if normalized.lessons is None:
                    if "lessons" in merged:
                        changed_fields.append("lessons")
                        merged.pop("lessons", None)
                else:
                    if merged.get("lessons") != normalized.lessons:
                        changed_fields.append("lessons")
                    merged["lessons"] = normalized.lessons

            if not changed_fields:
                self._audit_content_mutation(
                    authority,
                    operation="edit",
                    memory_id=memory_id,
                    changed_fields=(),
                    outcome="idempotent",
                )
                return MemoryEditSnapshot(await self.detail(authority, memory_id), (), True)

            now = datetime.now(UTC).isoformat()
            edit_count = existing.metadata.get("operator_edit_count")
            merged.update(
                {
                    "operator_edited_at": now,
                    "operator_edited_by_principal_id": str(authority.principal_id),
                    "operator_edited_fields": sorted(changed_fields),
                    "operator_edit_count": (edit_count if isinstance(edit_count, int) and edit_count >= 0 else 0) + 1,
                    "operator_edit_request_id": checked_request_id,
                }
            )
            updated = await asyncio.to_thread(
                memory.update_metadata,
                user_id=str(authority.principal_id),
                memory_id=memory_id,
                data=data,
                metadata=merged,
                runtime_profile_id=str(namespace.runtime_profile_id),
            )
            current = await asyncio.to_thread(
                memory.get_by_id,
                user_id=str(authority.principal_id),
                memory_id=memory_id,
                runtime_profile_id=str(namespace.runtime_profile_id),
            )
            current_matches = current is not None and (
                self._fact_matches(current, content=normalized.content, tags=normalized.tags)
                if isinstance(normalized, MemoryFactEdit)
                else self._episode_matches(current, normalized)
            )
            if (
                current_matches
                and current is not None
                and current.metadata.get("operator_edit_request_id") == checked_request_id
            ):
                self._audit_content_mutation(
                    authority,
                    operation="edit",
                    memory_id=memory_id,
                    changed_fields=changed_fields,
                    outcome="succeeded",
                )
                return MemoryEditSnapshot(
                    await self.detail(authority, memory_id),
                    tuple(sorted(changed_fields)),
                    not updated,
                )
            if current is not None and _memory_revision(current) != checked_revision:
                self._audit_content_mutation(
                    authority,
                    operation="edit",
                    memory_id=memory_id,
                    changed_fields=changed_fields,
                    outcome="conflict",
                )
                raise WorkshopMemoryConflict(_memory_revision(current))
            self._audit_content_mutation(
                authority,
                operation="edit",
                memory_id=memory_id,
                changed_fields=changed_fields,
                outcome="failed",
            )
            raise WorkshopMemoryMutationFailed("Memory edit failed; the original revision remains current")

    def _audit_mutation(
        self,
        authority: MemoryQueryAuthority,
        *,
        operation: str,
        result: MemoryMutationResult,
    ) -> None:
        def serialized_scope(scope: MemoryScopeSnapshot | None) -> dict[str, object] | None:
            if scope is None:
                return None
            return {
                "scope": scope.scope,
                "project_id": scope.project_id,
                "scope_source": scope.scope_source,
            }

        log.info(
            "%s %s",
            MEMORY_MANAGEMENT_AUDIT_EVENT,
            json.dumps(
                {
                    "actor_principal_id": str(authority.principal_id),
                    "operation": operation,
                    "memory_id": result.memory_id,
                    "prior_scope": serialized_scope(result.prior_scope),
                    "new_scope": serialized_scope(result.new_scope),
                    "timestamp": datetime.now(UTC).isoformat(),
                    "outcome": result.outcome,
                },
                sort_keys=True,
                separators=(",", ":"),
            ),
        )

    async def move_scope(
        self,
        authority: MemoryQueryAuthority,
        memory_ids: Sequence[str],
        *,
        scope: str,
        project_id: str | None = None,
    ) -> MemoryMutationBatch:
        checked = self._validate_memory_ids(memory_ids)
        scope_metadata = await self._scope_metadata(
            authority,
            scope=scope,
            project_id=project_id,
        )
        allowed_project_id = await self._allowed_project_id(authority)
        results: list[MemoryMutationResult] = []
        lock = self._mutation_locks.setdefault(authority.principal_id, asyncio.Lock())
        async with lock:
            for memory_id in checked:
                record = await asyncio.to_thread(
                    memory.get_by_id,
                    user_id=str(authority.principal_id),
                    memory_id=memory_id,
                )
                if record is None:
                    result = MemoryMutationResult(memory_id, "not_found", None, None)
                else:
                    prior = self._scope_snapshot(record, allowed_project_id=allowed_project_id)
                    merged = dict(record.metadata)
                    merged.update(scope_metadata)
                    mutation_raised = False
                    try:
                        updated = await asyncio.to_thread(
                            memory.update_metadata,
                            user_id=str(authority.principal_id),
                            memory_id=memory_id,
                            data=record.text,
                            metadata=merged,
                        )
                    except Exception:
                        log.exception("Workshop memory scope mutation failed for %s", memory_id)
                        updated = False
                        mutation_raised = True
                    if updated:
                        new_scope = self._scope_snapshot(
                            memory.MemoryResult(
                                id=record.id,
                                text=record.text,
                                score=record.score,
                                memory_type=record.memory_type,
                                metadata=merged,
                                created_at=record.created_at,
                                updated_at=record.updated_at,
                            ),
                            allowed_project_id=allowed_project_id,
                        )
                        result = MemoryMutationResult(memory_id, "succeeded", prior, new_scope)
                    else:
                        current = (
                            record
                            if mutation_raised
                            else await asyncio.to_thread(
                                memory.get_by_id,
                                user_id=str(authority.principal_id),
                                memory_id=memory_id,
                            )
                        )
                        result = MemoryMutationResult(
                            memory_id,
                            "failed" if mutation_raised else "stale" if current is None else "failed",
                            prior,
                            self._scope_snapshot(current, allowed_project_id=allowed_project_id)
                            if current is not None
                            else None,
                        )
                results.append(result)
                self._audit_mutation(authority, operation="move_scope", result=result)
        return MemoryMutationBatch("move_scope", tuple(results))

    async def delete(
        self,
        authority: MemoryQueryAuthority,
        memory_ids: Sequence[str],
    ) -> MemoryMutationBatch:
        checked = self._validate_memory_ids(memory_ids)
        allowed_project_id = await self._allowed_project_id(authority)
        results: list[MemoryMutationResult] = []
        lock = self._mutation_locks.setdefault(authority.principal_id, asyncio.Lock())
        async with lock:
            for memory_id in checked:
                record = await asyncio.to_thread(
                    memory.get_by_id,
                    user_id=str(authority.principal_id),
                    memory_id=memory_id,
                )
                if record is None:
                    result = MemoryMutationResult(memory_id, "not_found", None, None)
                else:
                    prior = self._scope_snapshot(record, allowed_project_id=allowed_project_id)
                    mutation_raised = False
                    try:
                        deleted = await asyncio.to_thread(
                            memory.delete_by_id,
                            user_id=str(authority.principal_id),
                            memory_id=memory_id,
                        )
                    except Exception:
                        log.exception("Workshop memory deletion failed for %s", memory_id)
                        deleted = False
                        mutation_raised = True
                    if deleted:
                        result = MemoryMutationResult(memory_id, "succeeded", prior, None)
                    else:
                        current = (
                            record
                            if mutation_raised
                            else await asyncio.to_thread(
                                memory.get_by_id,
                                user_id=str(authority.principal_id),
                                memory_id=memory_id,
                            )
                        )
                        result = MemoryMutationResult(
                            memory_id,
                            "failed" if mutation_raised else "stale" if current is None else "failed",
                            prior,
                            self._scope_snapshot(current, allowed_project_id=allowed_project_id)
                            if current is not None
                            else None,
                        )
                results.append(result)
                self._audit_mutation(authority, operation="delete", result=result)
        return MemoryMutationBatch("delete", tuple(results))

    async def detail(
        self,
        authority: MemoryQueryAuthority,
        memory_id: str,
    ) -> MemoryRecordDetail:
        if not memory_id or len(memory_id) > 256:
            raise WorkshopMemoryValidationError("Invalid memory identifier")
        result = await asyncio.to_thread(
            memory.get_by_id,
            user_id=str(authority.principal_id),
            memory_id=memory_id,
        )
        if result is None:
            raise WorkshopMemoryNotFound("Memory not found")
        if len(result.text) > MAX_CONTENT_CHARACTERS:
            raise WorkshopMemoryResponseTooLarge("Memory content is too large")
        allowed_project_id = await self._allowed_project_id(authority)
        resolved = memory.resolve_memory_scope(result.metadata)
        speaker, confidence = memory.read_time_memory_speaker(result.metadata)
        episode = None
        if _record_kind(result) == "episode":
            episode = {}
            for key in ("goal", "context", "approach", "outcome", "lessons", "outcome_quality"):
                value = _bounded_text(result.metadata.get(key), maximum=MAX_EPISODE_FIELD_CHARACTERS)
                if value is not None:
                    episode[key] = value
            for key in ("tags", "actors"):
                value = result.metadata.get(key)
                if isinstance(value, list):
                    episode[key] = [
                        item[:MAX_MEMORY_TAG_CHARACTERS]
                        for item in value[:MAX_MEMORY_TAGS]
                        if isinstance(item, str) and item
                    ]
        return MemoryRecordDetail(
            record=self._summary(
                result,
                allowed_project_id=allowed_project_id,
            ),
            content=result.text,
            compact_recall=_compact_recall(
                result,
                resolved_scope=resolved,
                speaker=speaker,
                confidence=confidence,
            ),
            confirmation_quote=_bounded_text(
                result.metadata.get("confirmation_quote"),
                maximum=10_000,
            ),
            prompt_version=_bounded_text(
                result.metadata.get("prompt_version"),
                maximum=256,
            ),
            episode=episode,
        )

    async def search(
        self,
        authority: MemoryQueryAuthority,
        query: str,
        *,
        filters: MemoryQueryFilters = EMPTY_MEMORY_FILTERS,
        limit: int = 10,
    ) -> MemorySearchSnapshot:
        self.validate_filters(filters)
        if not isinstance(query, str) or not query.strip() or len(query) > MAX_QUERY_CHARACTERS:
            raise WorkshopMemoryValidationError("Invalid memory search query")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_SEARCH_LIMIT:
            raise WorkshopMemoryValidationError(f"Memory search limit must be between 1 and {MAX_SEARCH_LIMIT}")
        namespace = authority.search_namespace
        if namespace is None:
            raise WorkshopMemoryAccessDenied("Memory search requires one unambiguous runtime profile")
        workspace = await self._runtime_pool.get_effective_workspace(namespace.runtime_profile_id)
        scoped = await memory.retrieve_scoped_memories(
            memory.ScopedRetrievalContext(
                chat_id=str(authority.principal_id),
                message=query,
                workspace=Path(workspace),
            ),
            limit=limit,
        )
        hits: list[MemorySearchHit] = []
        for hit in scoped.hits:
            result = hit.result
            if result.metadata.get("source") not in memory.USER_VISIBLE_SOURCES:
                continue
            if not self._matches(result, filters):
                continue
            hits.append(
                MemorySearchHit(
                    record=self._summary(
                        result,
                        allowed_project_id=scoped.debug.allowed_project_id,
                    ),
                    raw_score=float(result.score),
                    adjusted_score=float(hit.adjusted_score),
                    compact_recall=_compact_recall(
                        result,
                        resolved_scope=hit.resolved_scope,
                        speaker=hit.speaker,
                        confidence=hit.confidence,
                    ),
                )
            )
            if len(hits) >= limit:
                break
        return MemorySearchSnapshot(
            hits=tuple(hits),
            active_project_id=scoped.debug.active_project_id,
            reason=scoped.debug.reason,
        )

    async def source_context(
        self,
        authority: MemoryQueryAuthority,
        memory_id: str,
    ) -> MemorySourceContext:
        if not memory_id or len(memory_id) > 256:
            raise WorkshopMemoryValidationError("Invalid memory identifier")
        result = await asyncio.to_thread(
            memory.get_by_id,
            user_id=str(authority.principal_id),
            memory_id=memory_id,
        )
        if result is None:
            raise WorkshopMemoryNotFound("Memory not found")
        provenance = memory.read_transcript_provenance(result.metadata)
        if provenance.malformed:
            return MemorySourceContext("unavailable", "invalid_provenance", None, None, None)
        if not provenance.canonical_present:
            return MemorySourceContext("unavailable", "legacy_source", None, None, None)
        if provenance.principal_id != str(authority.principal_id):
            return MemorySourceContext("unavailable", "source_not_authorized", None, None, None)
        try:
            channel_id = ChannelId(provenance.channel_id or "")
            run_id = RunId(provenance.run_id or "")
            source_id = MessageId(provenance.source_message_id or "")
            result_id = MessageId(provenance.result_message_id or "")
            provenance_agent_id = AgentId(provenance.agent_id or "")
        except (TypeError, ValueError):
            return MemorySourceContext("unavailable", "invalid_provenance", None, None, None)
        if not await self._channel_authorizer.can_read_channel(
            authority.principal_id,
            channel_id,
        ):
            return MemorySourceContext("unavailable", "source_not_authorized", None, None, None)
        async with self._store.connection.execute(
            "SELECT r.requested_by_principal_id, r.agent_id, a.principal_id "
            "FROM runs r JOIN agents a ON a.id = r.agent_id "
            "WHERE r.id = ? AND r.channel_id = ? "
            "AND r.inbound_message_id = ? AND r.result_message_id = ? LIMIT 1",
            (run_id, channel_id, source_id, result_id),
        ) as cursor:
            run_row = await cursor.fetchone()
            if run_row is None:
                return MemorySourceContext("unavailable", "canonical_source_missing", run_id, None, None)
        if str(run_row[0]) != str(authority.principal_id) or str(run_row[1]) != str(provenance_agent_id):
            return MemorySourceContext("unavailable", "source_not_authorized", run_id, None, None)
        agent_principal_id = PrincipalId(str(run_row[2]))
        messages: list[MemorySourceMessage] = []
        for message_id in (source_id, result_id):
            async with self._store.connection.execute(
                "SELECT m.id, m.channel_id, m.author_principal_id, p.kind, "
                "p.display_name, m.body, m.created_at FROM messages m "
                "JOIN principals p ON p.id = m.author_principal_id "
                "WHERE m.id = ? AND m.channel_id = ?",
                (message_id, channel_id),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None:
                return MemorySourceContext("unavailable", "canonical_source_missing", run_id, None, None)
            body = str(row[5])
            if len(body) > MAX_SOURCE_BODY_CHARACTERS:
                raise WorkshopMemoryResponseTooLarge("Memory source context is too large")
            messages.append(
                MemorySourceMessage(
                    message_id=MessageId(str(row[0])),
                    channel_id=ChannelId(str(row[1])),
                    author_principal_id=PrincipalId(str(row[2])),
                    author_kind=str(row[3]),
                    author_display_name=str(row[4]),
                    body=body,
                    created_at=str(row[6]),
                )
            )
        if messages[0].author_principal_id != authority.principal_id:
            return MemorySourceContext("unavailable", "source_not_authorized", run_id, None, None)
        if messages[1].author_principal_id != agent_principal_id or messages[1].author_kind != "agent":
            return MemorySourceContext("unavailable", "source_not_authorized", run_id, None, None)
        return MemorySourceContext(
            status="available",
            reason=None,
            run_id=run_id,
            source=messages[0],
            result=messages[1],
        )
