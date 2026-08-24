"""Canonical, principal-scoped read service for Workshop semantic memory."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
from dataclasses import dataclass
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

DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 100
MAX_SEARCH_LIMIT = 50
MAX_QUERY_CHARACTERS = 2_000
MAX_CURSOR_CHARACTERS = 2_048
MAX_PREVIEW_CHARACTERS = 500
MAX_CONTENT_CHARACTERS = 100_000
MAX_COMPACT_RECALL_CHARACTERS = 120_000
MAX_SOURCE_BODY_CHARACTERS = 50_000
_CURSOR_VERSION = 1
_VALID_KINDS = frozenset({"fact", "episode"})
_VALID_SCOPES = frozenset({"global", "project", "task"})
_VALID_ORDERS = frozenset({"newest", "oldest"})


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
    scope: MemoryScopeSnapshot


@dataclass(frozen=True, slots=True)
class MemoryRecordDetail:
    record: MemoryRecordSummary
    content: str
    compact_recall: str
    confirmation_quote: str | None
    prompt_version: str | None
    episode: dict[str, str] | None


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
class MemoryStatsSnapshot:
    total: int
    facts: int
    episodes: int
    by_source: dict[str, int]
    by_type: dict[str, int]
    by_scope: dict[str, int]


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
    """Read semantic memory through canonical Workshop authority only."""

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
        )

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
            episode = {
                key: value
                for key in ("goal", "approach", "outcome", "lessons", "actors", "outcome_quality")
                if (value := _bounded_text(result.metadata.get(key), maximum=10_000)) is not None
            }
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
