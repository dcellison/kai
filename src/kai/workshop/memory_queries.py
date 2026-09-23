"""Canonical, principal-scoped query and management service for memory."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from kai import memory
from kai.config import Config
from kai.workshop.authorization import CanonicalChannelAuthorizer
from kai.workshop.domain import (
    AgentId,
    ChannelId,
    MemoryClaimId,
    MemoryEpisodeId,
    MemoryRevisionId,
    MessageId,
    PrincipalId,
    RunId,
    RuntimeProfileId,
)
from kai.workshop.episode_history import MemoryEpisodeHistoryService
from kai.workshop.execution_state import (
    WorkshopExecutionStateNamespace,
    WorkshopExecutionStateRegistry,
)
from kai.workshop.fact_lifecycle import (
    FactLifecycleConflict,
    FactMutationSource,
    FactRevisionInput,
    MemoryFactLifecycleService,
)
from kai.workshop.memory_current_truth import (
    CANONICAL_CLAIM_ID_KEY,
    CANONICAL_EPISODE_ID_KEY,
    CANONICAL_REVISION_ID_KEY,
    CANONICAL_TEMPORAL_ROLE_KEY,
    LEGACY_UNRECONCILED_ROLE,
)
from kai.workshop.memory_extraction_receipts import (
    MemoryExtractionReceiptAccessDenied,
    MemoryExtractionReceiptService,
    MemoryExtractionReceiptSnapshot,
)
from kai.workshop.memory_legacy_census import refresh_legacy_census
from kai.workshop.memory_projection_status import (
    ProjectionRetryResult,
    ProjectionStatus,
    VectorAudit,
    audit_vector_rows,
    expected_rows_async,
    projection_status_async,
)
from kai.workshop.memory_reconciliation_review import WorkshopMemoryReconciliationReviewService
from kai.workshop.memory_reconciliation_triage import WorkshopMemoryReconciliationTriageService
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
# Owner review lists (conflicts, forgotten facts) are bounded so one read
# stays cheap; a principal with more open items sees the newest first and
# the total count.
MAX_REVIEW_ITEMS = 200
MAX_REVIEW_NOTE_CHARACTERS = 500
MAX_MEMORY_TAG_CHARACTERS = 128
MAX_EPISODE_FIELD_CHARACTERS = 20_000
MAX_REQUEST_ID_CHARACTERS = 128
MEMORY_MANAGEMENT_AUDIT_EVENT = "workshop.memory.mutation"
MEMORY_CONTENT_AUDIT_EVENT = "workshop.memory.content_mutation"
_CURSOR_VERSION = 1
_REVISION_VERSION = 1
_VALID_KINDS = frozenset({"fact", "episode"})
_VALID_SCOPES = frozenset({"global", "project", "task"})
_VALID_LIFECYCLE_FILTERS = frozenset({"active", "historical", "legacy"})
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


class WorkshopMemoryConflictChanged(WorkshopMemoryQueryError):
    """
    The set of competing revisions changed since the owner opened it.

    Resolution settles every unresolved revision of a claim at once, so it
    must never settle a set the owner did not see.
    """

    def __init__(self) -> None:
        super().__init__("This conflict changed since you opened it")


class WorkshopMemoryAwaitingReconciliation(WorkshopMemoryQueryError):
    """
    The memory is an unreconciled legacy row and cannot be changed yet.

    Legacy rows are readable while temporary legacy admission is active,
    but editing, moving, or forgetting one outside legacy reconciliation
    would bypass the reconciliation plan that is responsible for it.
    """

    def __init__(self) -> None:
        super().__init__("This memory is waiting for legacy reconciliation and can't be changed yet")


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
    lifecycle: str | None = None


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
    source_reference: MemorySourceReference | None = None
    extraction_provenance: str = "not_applicable"
    extraction_receipt: MemoryExtractionReceiptSnapshot | None = None
    lifecycle: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MemorySourceReference:
    state: Literal["canonical", "legacy", "explicit", "invalid"]
    source_user_ts: str | None
    source_assistant_ts: str | None
    source_date: str | None
    source_date_end: str | None


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
class MemoryRevisionPreview:
    """One revision in an owner review list, bounded for display."""

    revision_id: str
    preview: str
    stored_at: str


@dataclass(frozen=True, slots=True)
class MemoryConflictSummary:
    """A claim with unresolved competing revisions."""

    claim_id: str
    scope_kind: str
    scope_key: str | None
    opened_at: str
    revisions: tuple[MemoryRevisionPreview, ...]


@dataclass(frozen=True, slots=True)
class MemoryForgottenSummary:
    """A claim with no current truth whose latest revision was retracted or expired."""

    claim_id: str
    scope_kind: str
    scope_key: str | None
    state: str
    revision_id: str
    changed_at: str
    reason: str
    preview: str


@dataclass(frozen=True, slots=True)
class MemoryReviewList[T]:
    """A bounded review list plus the unbounded total."""

    items: tuple[T, ...]
    total: int


@dataclass(frozen=True, slots=True)
class MemoryLifecycleOutcome:
    """Result of an owner resolution or restore."""

    claim_id: str
    active_revision_id: str
    replayed: bool
    # Vector memory id of the claim's current row, so the client can open
    # the kept or restored fact in the explorer. None when no projection
    # has succeeded yet (the caller then has nothing to open).
    memory_id: str | None = None


@dataclass(frozen=True, slots=True)
class MemoryStatsSnapshot:
    total: int
    facts: int
    episodes: int
    by_source: dict[str, int]
    by_type: dict[str, int]
    by_scope: dict[str, int]
    allowed_projects: tuple[MemoryProjectOption, ...]
    by_tag: dict[str, int] = field(default_factory=dict)
    confidence_min: float | None = None
    confidence_median: float | None = None
    confidence_max: float | None = None
    confidence_below_0_7: int = 0
    confidence_below_0_6: int = 0
    confirmation_quote_count: int = 0
    # Claims waiting for their owner to settle competing revisions. These
    # have no vector row, so the counts above never include them.
    unresolved_conflicts: int = 0
    # Facts and episodes whose search projection failed; together with
    # conflicts they drive the Fact review badge.
    projection_failures: int = 0
    by_prompt_version: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class MemoryMutationResult:
    memory_id: str
    outcome: Literal["succeeded", "not_found", "stale", "failed", "awaiting_reconciliation"]
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


def _stored_confidence(value: object) -> float | None:
    """Read the confidence the lifecycle stores inside revision vector metadata."""
    try:
        metadata = json.loads(str(value))
    except (TypeError, ValueError):
        return None
    confidence = metadata.get("confidence") if isinstance(metadata, dict) else None
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        return None
    return float(confidence)


def _parse_stored_time(value: object) -> datetime | None:
    """Parse a stored aware timestamp; None when absent or unreadable."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return parsed if parsed.tzinfo is not None else None


def _review_preview(content: str) -> str:
    """Bound review-list previews the same way record previews are bounded."""
    text = " ".join(content.split())
    return text if len(text) <= MAX_PREVIEW_CHARACTERS else text[: MAX_PREVIEW_CHARACTERS - 1] + "…"


def _stored_json_list(value: object) -> list[object]:
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


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
            "lifecycle": filters.lifecycle,
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
        self._fact_lifecycle = MemoryFactLifecycleService(store)
        self._episode_history = MemoryEpisodeHistoryService(store)
        self.reconciliation = WorkshopMemoryReconciliationReviewService(
            store,
            db_path=Path(config.session_db_path),
        )
        self.reconciliation_triage = WorkshopMemoryReconciliationTriageService(
            store,
            db_path=Path(config.session_db_path),
            runtime_pool=runtime_pool,
        )

    async def recover_fact_projections(self) -> int:
        """
        Recover canonical fact and episode projections at service startup.

        Operations interrupted by a prior process resume first. Failed
        operations then get one more round of attempts, because a restart
        often clears whatever failed them (an embedder that had not
        loaded, a store another process held). A failure that persists
        costs one bounded round per start and stays visible in Fact
        review and install status.

        Nothing runs while semantic memory is disabled: every projection
        read would fail, turning pending work into failures for no reason.
        """
        if not memory.is_enabled():
            return 0
        facts = await self._fact_lifecycle.recover_pending()
        episodes = await self._episode_history.recover_pending()
        retried_facts = await self._fact_lifecycle.retry_failed()
        retried_episodes = await self._episode_history.retry_failed()
        retried = retried_facts.retried + retried_episodes.retried
        if retried:
            log.info(
                "Retried %d failed memory vector projection(s) at startup: succeeded=%d, failed=%d",
                retried,
                retried_facts.succeeded + retried_episodes.succeeded,
                retried_facts.failed + retried_episodes.failed,
            )
        return facts + episodes

    async def refresh_legacy_census(self) -> int:
        """
        Recount every owner's unclassified legacy rows at service startup.

        Install status reads this census because it cannot open the vector
        store the service holds. A failure for one owner is logged and the
        others still count; their previous census stays with its older time.
        Returns the number of owners counted.
        """
        if not memory.is_enabled():
            return 0
        counted = 0
        owners = sorted(
            {
                (str(namespace.principal_id), str(namespace.runtime_profile_id))
                for namespaces in self._namespaces.values()
                for namespace in namespaces
            }
        )
        for principal_id, runtime_profile_id in owners:
            try:
                await asyncio.to_thread(
                    refresh_legacy_census,
                    Path(self._config.session_db_path),
                    principal_id=principal_id,
                    runtime_profile_id=runtime_profile_id,
                )
            except Exception:
                log.warning("Legacy memory census failed for %s/%s", principal_id, runtime_profile_id, exc_info=True)
                continue
            counted += 1
        return counted

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

    async def authority_for_transport_binding(
        self,
        *,
        transport: str,
        external_subject: str,
        external_channel_id: str,
    ) -> MemoryQueryAuthority:
        """Resolve an adapter identity through a canonical channel binding."""
        async with self._store.connection.execute(
            "SELECT ei.principal_id FROM external_identities ei "
            "JOIN channel_bindings cb ON cb.transport = ei.provider "
            "AND cb.external_channel_id = ? "
            "JOIN channels c ON c.id = cb.channel_id AND c.archived_at IS NULL "
            "JOIN channel_memberships cm ON cm.channel_id = cb.channel_id "
            "AND cm.principal_id = ei.principal_id "
            "WHERE ei.provider = ? AND ei.external_subject = ?",
            (external_channel_id, transport, external_subject),
        ) as cursor:
            rows = tuple(await cursor.fetchall())
        if len(rows) != 1:
            raise WorkshopMemoryAccessDenied("Memory access denied")
        return self.authority_for_principal(PrincipalId(str(rows[0][0])))

    @staticmethod
    def validate_filters(filters: MemoryQueryFilters) -> None:
        for value in (
            filters.kind,
            filters.source,
            filters.memory_type,
            filters.tag,
            filters.scope,
            filters.project_id,
            filters.lifecycle,
        ):
            if value is not None and (not value or len(value) > 128):
                raise WorkshopMemoryValidationError("Invalid memory filter")
        if filters.kind is not None and filters.kind not in _VALID_KINDS:
            raise WorkshopMemoryValidationError("Invalid memory kind")
        if filters.scope is not None and filters.scope not in _VALID_SCOPES:
            raise WorkshopMemoryValidationError("Invalid memory scope")
        if filters.lifecycle is not None and filters.lifecycle not in _VALID_LIFECYCLE_FILTERS:
            raise WorkshopMemoryValidationError("Invalid memory lifecycle filter")

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
            merged_registry(
                self._config.memory_projects,
                principal_id=str(authority.principal_id),
            ),
        )
        return active.project_id if active is not None and active.memory_enabled else None

    async def allowed_projects(
        self,
        authority: MemoryQueryAuthority,
    ) -> tuple[MemoryProjectOption, ...]:
        """Return memory-enabled projects reachable by an owned runtime."""
        from kai.memory_projects import detect_active_memory_project, merged_registry

        registry = merged_registry(
            self._config.memory_projects,
            principal_id=str(authority.principal_id),
        )
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
        *,
        active_claim_ids: frozenset[str] = frozenset(),
        historical_claim_ids: frozenset[str] = frozenset(),
        episode_ids: frozenset[str] = frozenset(),
    ) -> bool:
        resolved = memory.resolve_memory_scope(result.metadata)
        claim_id = result.metadata.get(CANONICAL_CLAIM_ID_KEY)
        episode_id = result.metadata.get(CANONICAL_EPISODE_ID_KEY)
        lifecycle_matches = (
            filters.lifecycle is None
            or (filters.lifecycle == "active" and isinstance(claim_id, str) and claim_id in active_claim_ids)
            or (
                filters.lifecycle == "historical"
                and (
                    (isinstance(claim_id, str) and claim_id in historical_claim_ids)
                    or (isinstance(episode_id, str) and episode_id in episode_ids)
                )
            )
            or (filters.lifecycle == "legacy" and not isinstance(claim_id, str) and not isinstance(episode_id, str))
        )
        return all(
            (
                filters.kind is None or _record_kind(result) == filters.kind,
                filters.source is None or _source(result) == filters.source,
                filters.memory_type is None or result.memory_type == filters.memory_type,
                filters.tag is None or filters.tag in _tags(result),
                filters.scope is None or resolved.scope == filters.scope,
                filters.project_id is None or resolved.project_id == filters.project_id,
                lifecycle_matches,
            )
        )

    async def _lifecycle_filter_sets(
        self,
        authority: MemoryQueryAuthority,
        lifecycle: str | None,
    ) -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
        if lifecycle is None or lifecycle == "legacy":
            return frozenset(), frozenset(), frozenset()
        async with self._store.connection.execute(
            "SELECT c.claim_id, "
            "MAX(CASE WHEN s.state = 'active' THEN 1 ELSE 0 END), COUNT(r.revision_id) "
            "FROM memory_fact_claims c "
            "JOIN memory_fact_revisions r ON r.claim_id = c.claim_id "
            "JOIN memory_fact_revision_states s ON s.claim_id = r.claim_id AND s.revision_id = r.revision_id "
            "WHERE c.owner_principal_id = ? GROUP BY c.claim_id",
            (str(authority.principal_id),),
        ) as cursor:
            fact_rows = await cursor.fetchall()
        active = frozenset(str(row[0]) for row in fact_rows if int(row[1]) == 1)
        historical = frozenset(str(row[0]) for row in fact_rows if int(row[2]) > 1)
        async with self._store.connection.execute(
            "SELECT episode_id FROM memory_episodes WHERE owner_principal_id = ?",
            (str(authority.principal_id),),
        ) as cursor:
            episodes = frozenset(str(row[0]) for row in await cursor.fetchall())
        return active, historical, episodes

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
        lifecycle_sets = await self._lifecycle_filter_sets(authority, filters.lifecycle)
        rows = [
            row
            for row in await self._all_visible(authority)
            if self._matches(
                row,
                filters,
                active_claim_ids=lifecycle_sets[0],
                historical_claim_ids=lifecycle_sets[1],
                episode_ids=lifecycle_sets[2],
            )
        ]
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
        by_tag: dict[str, int] = {}
        by_prompt_version: dict[str, int] = {}
        confidences: list[float] = []
        confirmation_quote_count = 0
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
            if source == "extracted":
                for tag in _tags(row):
                    by_tag[tag] = by_tag.get(tag, 0) + 1
                prompt_version = str(row.metadata.get("prompt_version") or "")
                by_prompt_version[prompt_version] = by_prompt_version.get(prompt_version, 0) + 1
                confidence = row.metadata.get("confidence")
                if isinstance(confidence, int | float):
                    confidences.append(float(confidence))
                if row.metadata.get("confirmation_quote"):
                    confirmation_quote_count += 1
        sorted_confidences = sorted(confidences)
        return MemoryStatsSnapshot(
            total=len(rows),
            facts=facts,
            episodes=episodes,
            by_source=dict(sorted(by_source.items())),
            by_type=dict(sorted(by_type.items())),
            by_scope=dict(sorted(by_scope.items())),
            allowed_projects=await self.allowed_projects(authority),
            by_tag=dict(sorted(by_tag.items())),
            confidence_min=(sorted_confidences[0] if sorted_confidences else None),
            confidence_median=(sorted_confidences[(len(sorted_confidences) - 1) // 2] if sorted_confidences else None),
            confidence_max=(sorted_confidences[-1] if sorted_confidences else None),
            confidence_below_0_7=sum(1 for value in confidences if value < 0.7),
            confidence_below_0_6=sum(1 for value in confidences if value < 0.6),
            confirmation_quote_count=confirmation_quote_count,
            unresolved_conflicts=await self.unresolved_conflict_count(authority),
            projection_failures=await self.projection_failure_count(authority),
            by_prompt_version=dict(sorted(by_prompt_version.items())),
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

    @classmethod
    def _validate_expected_revisions(
        cls,
        memory_ids: tuple[str, ...],
        expected_revisions: Mapping[str, str] | None,
    ) -> dict[str, str]:
        if expected_revisions is None:
            return {}
        if not isinstance(expected_revisions, Mapping) or set(expected_revisions) != set(memory_ids):
            raise WorkshopMemoryValidationError("Expected memory revisions must match mutation targets")
        return {memory_id: cls._validate_revision(revision) for memory_id, revision in expected_revisions.items()}

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

            now_value = datetime.now(UTC)
            now = now_value.isoformat()
            metadata: dict[str, object] = {
                "source": "explicit",
                "speaker": "user",
                "confidence": 1.0,
                "operator_created_at": now,
                "operator_created_by_principal_id": str(authority.principal_id),
                "operator_creation_request_id": checked_request_id,
                **scope_metadata,
            }
            lifecycle_authority = await self._fact_lifecycle.authority_for(
                authority.principal_id,
                namespace.runtime_profile_id,
            )
            mutation = await self._fact_lifecycle.create(
                lifecycle_authority,
                FactRevisionInput(
                    content=cleaned_content,
                    scope_kind=scope,
                    scope_key=project_id or "",
                    reason="Explicit fact created by its owning principal.",
                    evidence=({"kind": "operator", "reference_id": checked_request_id, "sha256": None},),
                    vector_metadata={**metadata, "tags": list(cleaned_tags)},
                    confidence=1.0,
                    asserted_at=now_value,
                    observed_at=now_value,
                    valid_from=now_value,
                ),
                idempotency_key=(
                    f"workshop-memory:create:{authority.principal_id}:{namespace.runtime_profile_id}:"
                    f"{checked_request_id}"
                ),
                stable_claim_key=f"operator:{checked_request_id}",
            )
            memory_id = mutation.memory_id
            if not isinstance(memory_id, str) or not memory_id or mutation.projection_status != "succeeded":
                self._audit_content_mutation(
                    authority,
                    operation="create",
                    memory_id=None,
                    changed_fields=("content", "tags", "scope"),
                    outcome="failed",
                )
                raise WorkshopMemoryMutationFailed("Memory creation is canonical but its search projection is pending")
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
                if await self._awaiting_reconciliation(authority, memory_id):
                    raise WorkshopMemoryAwaitingReconciliation()
                raise WorkshopMemoryNotFound("Memory not found")
            if existing.metadata.get("canonical_memory_episode_id"):
                raise WorkshopMemoryValidationError(
                    "Canonical episodes are immutable; record a linked follow-up instead"
                )
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
            lifecycle_authority = await self._fact_lifecycle.authority_for(
                authority.principal_id,
                namespace.runtime_profile_id,
            )
            if isinstance(normalized, MemoryFactEdit):
                adopted = await self._fact_lifecycle.adopt_legacy(
                    lifecycle_authority,
                    existing,
                    idempotency_key=(
                        f"workshop-memory:adopt:{authority.principal_id}:{namespace.runtime_profile_id}:{memory_id}"
                    ),
                )
                if adopted.projection_status != "succeeded" or adopted.memory_id is None:
                    raise WorkshopMemoryMutationFailed(
                        "Memory adoption is canonical but its search projection is pending"
                    )
                resolved_scope = memory.resolve_memory_scope(existing.metadata)
                mutation = await self._fact_lifecycle.supersede(
                    lifecycle_authority,
                    adopted.claim_id,
                    adopted.revision_id,
                    FactRevisionInput(
                        content=data,
                        scope_kind=(
                            resolved_scope.scope if resolved_scope.scope in {"global", "project"} else "global"
                        ),
                        scope_key=(
                            str(resolved_scope.project_id)
                            if resolved_scope.scope == "project" and resolved_scope.project_id
                            else ""
                        ),
                        reason="Explicit fact edit by its owning principal.",
                        evidence=({"kind": "operator", "reference_id": checked_request_id, "sha256": None},),
                        vector_metadata=merged,
                        confidence=1.0,
                        asserted_at=datetime.now(UTC),
                        observed_at=datetime.now(UTC),
                        valid_from=datetime.now(UTC),
                    ),
                    idempotency_key=(
                        f"workshop-memory:edit:{authority.principal_id}:{namespace.runtime_profile_id}:"
                        f"{checked_request_id}"
                    ),
                    source=FactMutationSource.HUMAN,
                )
                updated = mutation.projection_status == "succeeded"
                if not updated:
                    raise WorkshopMemoryMutationFailed("Memory edit is canonical but its search projection is pending")
            else:
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
        expected_revisions: Mapping[str, str] | None = None,
    ) -> MemoryMutationBatch:
        checked = self._validate_memory_ids(memory_ids)
        revisions = self._validate_expected_revisions(checked, expected_revisions)
        scope_metadata = await self._scope_metadata(
            authority,
            scope=scope,
            project_id=project_id,
        )
        namespace = self._namespace_for_mutation(authority)
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
                    outcome: Literal["not_found", "awaiting_reconciliation"] = (
                        "awaiting_reconciliation"
                        if await self._awaiting_reconciliation(authority, memory_id)
                        else "not_found"
                    )
                    result = MemoryMutationResult(memory_id, outcome, None, None)
                else:
                    prior = self._scope_snapshot(record, allowed_project_id=allowed_project_id)
                    if record.metadata.get("canonical_memory_episode_id"):
                        result = MemoryMutationResult(memory_id, "failed", prior, prior)
                    elif revisions and _memory_revision(record) != revisions[memory_id]:
                        result = MemoryMutationResult(memory_id, "stale", prior, prior)
                    else:
                        merged = dict(record.metadata)
                        merged.update(scope_metadata)
                        mutation_raised = False
                        try:
                            if _record_kind(record) == "fact":
                                lifecycle_authority = await self._fact_lifecycle.authority_for(
                                    authority.principal_id,
                                    namespace.runtime_profile_id,
                                )
                                adopted = await self._fact_lifecycle.adopt_legacy(
                                    lifecycle_authority,
                                    record,
                                    idempotency_key=(
                                        f"workshop-memory:adopt:{authority.principal_id}:"
                                        f"{namespace.runtime_profile_id}:{memory_id}"
                                    ),
                                )
                                resolved_scope = memory.resolve_memory_scope(merged)
                                now_value = datetime.now(UTC)
                                mutation = await self._fact_lifecycle.supersede(
                                    lifecycle_authority,
                                    adopted.claim_id,
                                    adopted.revision_id,
                                    FactRevisionInput(
                                        content=record.text,
                                        scope_kind=(
                                            resolved_scope.scope
                                            if resolved_scope.scope in {"global", "project"}
                                            else "global"
                                        ),
                                        scope_key=(
                                            str(resolved_scope.project_id)
                                            if resolved_scope.scope == "project" and resolved_scope.project_id
                                            else ""
                                        ),
                                        reason="Explicit fact scope change by its owning principal.",
                                        evidence=(
                                            {
                                                "kind": "operator",
                                                "reference_id": _memory_revision(record),
                                                "sha256": None,
                                            },
                                        ),
                                        vector_metadata=merged,
                                        confidence=float(record.metadata.get("confidence", 1.0)),
                                        asserted_at=now_value,
                                        observed_at=now_value,
                                        valid_from=now_value,
                                    ),
                                    idempotency_key=(
                                        f"workshop-memory:scope:{authority.principal_id}:"
                                        f"{namespace.runtime_profile_id}:{memory_id}:"
                                        f"{_memory_revision(record)}:{resolved_scope.scope}:"
                                        f"{resolved_scope.project_id or ''}"
                                    ),
                                    source=FactMutationSource.HUMAN,
                                )
                                updated = mutation.projection_status == "succeeded"
                            else:
                                updated = await asyncio.to_thread(
                                    memory.update_metadata,
                                    user_id=str(authority.principal_id),
                                    memory_id=memory_id,
                                    data=record.text,
                                    metadata=merged,
                                )
                        except FactLifecycleConflict:
                            updated = False
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
        *,
        expected_revisions: Mapping[str, str] | None = None,
    ) -> MemoryMutationBatch:
        checked = self._validate_memory_ids(memory_ids)
        revisions = self._validate_expected_revisions(checked, expected_revisions)
        namespace = self._namespace_for_mutation(authority)
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
                    outcome: Literal["not_found", "awaiting_reconciliation"] = (
                        "awaiting_reconciliation"
                        if await self._awaiting_reconciliation(authority, memory_id)
                        else "not_found"
                    )
                    result = MemoryMutationResult(memory_id, outcome, None, None)
                else:
                    prior = self._scope_snapshot(record, allowed_project_id=allowed_project_id)
                    if record.metadata.get("canonical_memory_episode_id"):
                        result = MemoryMutationResult(memory_id, "failed", prior, prior)
                    elif revisions and _memory_revision(record) != revisions[memory_id]:
                        result = MemoryMutationResult(memory_id, "stale", prior, prior)
                    else:
                        mutation_raised = False
                        try:
                            if _record_kind(record) == "fact":
                                lifecycle_authority = await self._fact_lifecycle.authority_for(
                                    authority.principal_id,
                                    namespace.runtime_profile_id,
                                )
                                adopted = await self._fact_lifecycle.adopt_legacy(
                                    lifecycle_authority,
                                    record,
                                    idempotency_key=(
                                        f"workshop-memory:adopt:{authority.principal_id}:"
                                        f"{namespace.runtime_profile_id}:{memory_id}"
                                    ),
                                )
                                mutation = await self._fact_lifecycle.retract(
                                    lifecycle_authority,
                                    adopted.claim_id,
                                    adopted.revision_id,
                                    reason="Explicit fact deletion by its owning principal.",
                                    idempotency_key=(
                                        f"workshop-memory:delete:{authority.principal_id}:"
                                        f"{namespace.runtime_profile_id}:{memory_id}:{adopted.revision_id}"
                                    ),
                                )
                                deleted = mutation.projection_status == "succeeded"
                            else:
                                deleted = await asyncio.to_thread(
                                    memory.delete_by_id,
                                    user_id=str(authority.principal_id),
                                    memory_id=memory_id,
                                )
                        except FactLifecycleConflict:
                            deleted = False
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

    async def _fact_lifecycle_detail(
        self,
        authority: MemoryQueryAuthority,
        claim_id: str,
        projected_revision_id: str | None,
    ) -> dict[str, object] | None:
        async with self._store.connection.execute(
            "SELECT c.runtime_profile_id, c.scope_kind, c.scope_key, c.created_at, "
            "r.revision_id, r.content, r.asserted_at, r.observed_at, r.stored_at, "
            "r.valid_from, r.valid_until, r.reason, r.evidence_json, r.backend, r.provider, "
            "r.model, r.prompt_version, r.schema_version, r.supersedes_revision_id, "
            "r.migration_classification, r.migration_gaps_json, s.state, s.state_reason, s.updated_at, "
            "r.vector_metadata_json, r.admission_authority "
            "FROM memory_fact_claims c "
            "JOIN runtime_profile_owners rpo ON rpo.runtime_profile_id = c.runtime_profile_id "
            "AND rpo.principal_id = c.owner_principal_id "
            "JOIN memory_fact_revisions r ON r.claim_id = c.claim_id "
            "JOIN memory_fact_revision_states s ON s.claim_id = r.claim_id AND s.revision_id = r.revision_id "
            "WHERE c.claim_id = ? AND c.owner_principal_id = ? "
            "ORDER BY r.created_event_position DESC",
            (claim_id, str(authority.principal_id)),
        ) as cursor:
            rows = list(await cursor.fetchall())
        if not rows:
            return None
        async with self._store.connection.execute(
            "SELECT revision_id, transition, previous_state, new_state, reason, occurred_at, event_position "
            "FROM memory_fact_lifecycle_events WHERE claim_id = ? ORDER BY event_position DESC",
            (claim_id,),
        ) as cursor:
            events = list(await cursor.fetchall())
        return {
            "authority": "canonical",
            "kind": "fact",
            "identity": claim_id,
            "runtimeProfileId": str(rows[0][0]),
            "scope": {"kind": str(rows[0][1]), "key": str(rows[0][2]) or None},
            "createdAt": str(rows[0][3]),
            "currentRevisionId": projected_revision_id,
            "currentState": next(
                (str(row[21]) for row in rows if str(row[4]) == projected_revision_id),
                str(rows[0][21]),
            ),
            "revisions": [
                {
                    "revisionId": str(row[4]),
                    "content": str(row[5]),
                    "assertedAt": None if row[6] is None else str(row[6]),
                    "observedAt": None if row[7] is None else str(row[7]),
                    "storedAt": str(row[8]),
                    "validFrom": None if row[9] is None else str(row[9]),
                    "validUntil": None if row[10] is None else str(row[10]),
                    "reason": str(row[11]),
                    "evidence": _stored_json_list(row[12]),
                    "backend": None if row[13] is None else str(row[13]),
                    "provider": None if row[14] is None else str(row[14]),
                    "model": None if row[15] is None else str(row[15]),
                    "promptVersion": None if row[16] is None else str(row[16]),
                    "schemaVersion": None if row[17] is None else str(row[17]),
                    "supersedesRevisionId": None if row[18] is None else str(row[18]),
                    "migrationClassification": str(row[19]),
                    "migrationGaps": _stored_json_list(row[20]),
                    "state": str(row[21]),
                    "stateReason": str(row[22]),
                    "stateUpdatedAt": str(row[23]),
                    "confidence": _stored_confidence(row[24]),
                    "admissionAuthority": str(row[25]),
                }
                for row in rows
            ],
            "events": [
                {
                    "revisionId": str(row[0]),
                    "transition": str(row[1]),
                    "previousState": None if row[2] is None else str(row[2]),
                    "newState": str(row[3]),
                    "reason": str(row[4]),
                    "occurredAt": str(row[5]),
                    "eventPosition": int(row[6]),
                }
                for row in events
            ],
            "followups": [],
        }

    async def _episode_lifecycle_detail(
        self,
        authority: MemoryQueryAuthority,
        episode_id: str,
    ) -> dict[str, object] | None:
        async with self._store.connection.execute(
            "SELECT e.runtime_profile_id, e.scope_kind, e.scope_key, e.content, e.occurred_from, "
            "e.occurred_until, e.observed_at, e.stored_at, e.reason, e.evidence_json, e.backend, "
            "e.provider, e.model, e.prompt_version, e.schema_version, e.migration_classification, "
            "e.migration_gaps_json "
            "FROM memory_episodes e "
            "JOIN runtime_profile_owners rpo ON rpo.runtime_profile_id = e.runtime_profile_id "
            "AND rpo.principal_id = e.owner_principal_id "
            "WHERE e.episode_id = ? AND e.owner_principal_id = ?",
            (episode_id, str(authority.principal_id)),
        ) as cursor:
            row = await cursor.fetchone()
        if row is None:
            return None
        async with self._store.connection.execute(
            "SELECT f.source_episode_id, f.target_episode_id, f.relationship, f.reason, f.created_at "
            "FROM memory_episode_followups f "
            "WHERE f.owner_principal_id = ? AND (f.source_episode_id = ? OR f.target_episode_id = ?) "
            "ORDER BY f.created_event_position",
            (str(authority.principal_id), episode_id, episode_id),
        ) as cursor:
            followups = list(await cursor.fetchall())
        return {
            "authority": "canonical",
            "kind": "episode",
            "identity": episode_id,
            "runtimeProfileId": str(row[0]),
            "scope": {"kind": str(row[1]), "key": str(row[2]) or None},
            "createdAt": str(row[7]),
            "currentRevisionId": None,
            "currentState": "historical",
            "revisions": [
                {
                    "revisionId": episode_id,
                    "content": str(row[3]),
                    "occurredFrom": None if row[4] is None else str(row[4]),
                    "occurredUntil": None if row[5] is None else str(row[5]),
                    "observedAt": None if row[6] is None else str(row[6]),
                    "storedAt": str(row[7]),
                    "reason": str(row[8]),
                    "evidence": _stored_json_list(row[9]),
                    "backend": None if row[10] is None else str(row[10]),
                    "provider": None if row[11] is None else str(row[11]),
                    "model": None if row[12] is None else str(row[12]),
                    "promptVersion": None if row[13] is None else str(row[13]),
                    "schemaVersion": None if row[14] is None else str(row[14]),
                    "migrationClassification": str(row[15]),
                    "migrationGaps": _stored_json_list(row[16]),
                    "state": "historical",
                }
            ],
            "events": [],
            "followups": [
                {
                    "sourceEpisodeId": str(item[0]),
                    "targetEpisodeId": str(item[1]),
                    "relationship": str(item[2]),
                    "reason": str(item[3]),
                    "createdAt": str(item[4]),
                }
                for item in followups
            ],
        }

    async def _lifecycle_detail(
        self,
        authority: MemoryQueryAuthority,
        result: memory.MemoryResult,
    ) -> dict[str, object]:
        claim_id = result.metadata.get(CANONICAL_CLAIM_ID_KEY)
        revision_id = result.metadata.get(CANONICAL_REVISION_ID_KEY)
        if isinstance(claim_id, str) and claim_id:
            canonical = await self._fact_lifecycle_detail(
                authority,
                claim_id,
                revision_id if isinstance(revision_id, str) else None,
            )
            if canonical is not None:
                return canonical
        episode_id = result.metadata.get(CANONICAL_EPISODE_ID_KEY)
        if isinstance(episode_id, str) and episode_id:
            canonical = await self._episode_lifecycle_detail(authority, episode_id)
            if canonical is not None:
                return canonical
        migration_gaps = result.metadata.get("migration_gaps")
        return {
            "authority": "legacy",
            "kind": _record_kind(result),
            "identity": result.id,
            "runtimeProfileId": None,
            "scope": {
                "kind": memory.resolve_memory_scope(result.metadata).scope,
                "key": memory.resolve_memory_scope(result.metadata).project_id,
            },
            "createdAt": result.created_at,
            "currentRevisionId": None,
            "currentState": "requires_review",
            "revisions": [],
            "events": [],
            "followups": [],
            "migrationClassification": str(result.metadata.get("migration_classification") or "unclassified"),
            "migrationGaps": (
                [str(value) for value in migration_gaps if isinstance(value, str)]
                if isinstance(migration_gaps, list)
                else ["canonical provenance"]
            ),
        }

    # ── Owner review: conflicts and forgotten facts ─────────────────
    #
    # Conflicted, retracted, and expired claims have no vector row (their
    # vectors are deleted by the lifecycle), so no vector-backed listing can
    # show them. These reads go to the canonical fact tables directly and
    # are scoped to the owner pair every fact mutation uses.

    def _owner_pair(self, authority: MemoryQueryAuthority) -> tuple[str, str]:
        namespace = self._namespace_for_mutation(authority)
        return str(authority.principal_id), str(namespace.runtime_profile_id)

    @staticmethod
    def _validate_claim_id(claim_id: str) -> str:
        if not isinstance(claim_id, str) or not claim_id or len(claim_id) > 128 or claim_id != claim_id.strip():
            raise WorkshopMemoryValidationError("Invalid memory claim identifier")
        return claim_id

    @staticmethod
    def _validate_note(note: str) -> str:
        if not isinstance(note, str) or len(note) > MAX_REVIEW_NOTE_CHARACTERS:
            raise WorkshopMemoryValidationError("Notes must be at most 500 characters")
        return note.strip()

    async def _unresolved_revisions(self, claim_id: str) -> list[tuple[str, str, str]]:
        """Return (revision_id, content, stored_at) of unresolved revisions, oldest first."""
        async with self._store.connection.execute(
            "SELECT r.revision_id, r.content, r.stored_at FROM memory_fact_revisions r "
            "JOIN memory_fact_revision_states s ON s.claim_id = r.claim_id AND s.revision_id = r.revision_id "
            "WHERE r.claim_id = ? AND s.state = 'unresolved_conflict' ORDER BY r.created_event_position",
            (claim_id,),
        ) as cursor:
            return [(str(row[0]), str(row[1]), str(row[2])) for row in await cursor.fetchall()]

    async def _owns_claim(self, owner: tuple[str, str], claim_id: str) -> bool:
        async with self._store.connection.execute(
            "SELECT 1 FROM memory_fact_claims WHERE claim_id = ? AND owner_principal_id = ? AND runtime_profile_id = ?",
            (claim_id, *owner),
        ) as cursor:
            return await cursor.fetchone() is not None

    async def list_conflicts(self, authority: MemoryQueryAuthority) -> MemoryReviewList[MemoryConflictSummary]:
        """List the owner's claims with unresolved competing revisions, newest conflict first."""
        owner = self._owner_pair(authority)
        where = (
            "FROM memory_fact_claims c WHERE c.owner_principal_id = ? AND c.runtime_profile_id = ? "
            "AND EXISTS (SELECT 1 FROM memory_fact_revision_states s "
            "WHERE s.claim_id = c.claim_id AND s.state = 'unresolved_conflict')"
        )
        async with self._store.connection.execute(f"SELECT COUNT(*) {where}", owner) as cursor:
            total_row = await cursor.fetchone()
        async with self._store.connection.execute(
            "SELECT c.claim_id, c.scope_kind, c.scope_key, "
            "(SELECT MAX(e.occurred_at) FROM memory_fact_lifecycle_events e "
            "WHERE e.claim_id = c.claim_id AND e.transition = 'conflict_opened') AS opened_at "
            f"{where} ORDER BY opened_at DESC, c.claim_id LIMIT ?",
            (*owner, MAX_REVIEW_ITEMS),
        ) as cursor:
            claims = list(await cursor.fetchall())
        items: list[MemoryConflictSummary] = []
        for claim_id, scope_kind, scope_key, opened_at in claims:
            revisions = await self._unresolved_revisions(str(claim_id))
            items.append(
                MemoryConflictSummary(
                    claim_id=str(claim_id),
                    scope_kind=str(scope_kind),
                    scope_key=str(scope_key) or None,
                    opened_at=str(opened_at or ""),
                    revisions=tuple(
                        MemoryRevisionPreview(revision_id, _review_preview(content), stored_at)
                        for revision_id, content, stored_at in revisions
                    ),
                )
            )
        return MemoryReviewList(tuple(items), int(total_row[0]) if total_row is not None else 0)

    async def conflict_detail(self, authority: MemoryQueryAuthority, claim_id: str) -> dict[str, object]:
        """Return full lifecycle detail for one of the owner's conflicted claims."""
        checked = self._validate_claim_id(claim_id)
        owner = self._owner_pair(authority)
        if not await self._owns_claim(owner, checked) or not await self._unresolved_revisions(checked):
            raise WorkshopMemoryNotFound("Memory not found")
        detail = await self._fact_lifecycle_detail(authority, checked, None)
        if detail is None:
            raise WorkshopMemoryNotFound("Memory not found")
        return detail

    async def resolve_conflict(
        self,
        authority: MemoryQueryAuthority,
        claim_id: str,
        *,
        keep_revision_id: str,
        expected_revision_ids: Sequence[str],
        note: str,
        client_operation_id: str,
    ) -> MemoryLifecycleOutcome:
        """
        Keep one competing revision and supersede the rest.

        `expected_revision_ids` is the set the owner was shown. A new request
        whose set no longer matches the claim's unresolved revisions is
        refused, so a stale screen can never settle revisions the owner did
        not see. Replays of the same operation return the original outcome
        without re-checking, because the claim has legitimately moved on.

        Raises:
            WorkshopMemoryValidationError: Malformed input, or the operation
                id was already used for a different request.
            WorkshopMemoryNotFound: The claim is not the owner's.
            WorkshopMemoryConflictChanged: The unresolved set changed.
            WorkshopMemoryMutationFailed: Resolved canonically, but the kept
                revision's search projection did not complete.
        """
        checked_claim = self._validate_claim_id(claim_id)
        operation_id = self._validate_request_id(client_operation_id)
        checked_note = self._validate_note(note)
        expected = [str(value) for value in expected_revision_ids] if isinstance(expected_revision_ids, list) else []
        if (
            not expected
            or len(expected) > 32
            or len(set(expected)) != len(expected)
            or not all(value and len(value) <= 128 for value in expected)
            or keep_revision_id not in expected
        ):
            raise WorkshopMemoryValidationError("Choose one of the competing versions")
        losers = tuple(MemoryRevisionId(value) for value in sorted(set(expected) - {keep_revision_id}))
        reason = "Owner resolved the conflict" + (f": {checked_note}" if checked_note else "")
        owner = self._owner_pair(authority)
        idempotency_key = f"memory-conflict-resolve:{authority.principal_id}:{operation_id}"
        lock = self._mutation_locks.setdefault(authority.principal_id, asyncio.Lock())
        async with lock:
            if not await self._owns_claim(owner, checked_claim):
                raise WorkshopMemoryNotFound("Memory not found")
            prior = await self._store.event_by_idempotency_key(idempotency_key)
            if prior is not None:
                payload = prior.envelope.payload
                if (
                    str(prior.envelope.aggregate_id) != checked_claim
                    or payload.get("winner_revision_id") != keep_revision_id
                    or sorted(payload.get("loser_revision_ids") or []) != [str(value) for value in losers]
                    or payload.get("reason") != reason
                ):
                    raise WorkshopMemoryValidationError("This operation was already used for a different request")
                return MemoryLifecycleOutcome(
                    checked_claim, keep_revision_id, True, await self._claim_memory_id(checked_claim)
                )
            current = {
                revision_id for revision_id, _content, _stored in await self._unresolved_revisions(checked_claim)
            }
            if current != set(expected):
                raise WorkshopMemoryConflictChanged()
            lifecycle_authority = await self._fact_lifecycle.authority_for(
                authority.principal_id,
                RuntimeProfileId(owner[1]),
            )
            try:
                mutation = await self._fact_lifecycle.resolve_conflict(
                    lifecycle_authority,
                    MemoryClaimId(checked_claim),
                    MemoryRevisionId(keep_revision_id),
                    losers,
                    reason=reason,
                    idempotency_key=idempotency_key,
                )
            except FactLifecycleConflict as exc:
                # The unresolved set changed between the check above and the
                # lifecycle commit; projection refused the stale settlement.
                raise WorkshopMemoryConflictChanged() from exc
        if mutation.projection_status != "succeeded":
            raise WorkshopMemoryMutationFailed("The conflict is resolved, but its search projection is pending")
        return MemoryLifecycleOutcome(
            checked_claim,
            str(mutation.revision_id),
            mutation.replayed,
            await self._claim_memory_id(checked_claim),
        )

    async def list_forgotten(self, authority: MemoryQueryAuthority) -> MemoryReviewList[MemoryForgottenSummary]:
        """List the owner's claims with no current truth whose latest revision was retracted or expired."""
        owner = self._owner_pair(authority)
        latest = (
            "FROM memory_fact_claims c "
            "JOIN memory_fact_revisions r ON r.claim_id = c.claim_id AND r.created_event_position = ("
            "SELECT MAX(latest.created_event_position) FROM memory_fact_revisions latest "
            "WHERE latest.claim_id = c.claim_id) "
            "JOIN memory_fact_revision_states s ON s.claim_id = r.claim_id AND s.revision_id = r.revision_id "
            "WHERE c.owner_principal_id = ? AND c.runtime_profile_id = ? AND s.state IN ('retracted', 'expired') "
            "AND NOT EXISTS (SELECT 1 FROM memory_fact_revision_states current "
            "WHERE current.claim_id = c.claim_id AND current.state IN ('active', 'unresolved_conflict'))"
        )
        async with self._store.connection.execute(f"SELECT COUNT(*) {latest}", owner) as cursor:
            total_row = await cursor.fetchone()
        async with self._store.connection.execute(
            "SELECT c.claim_id, c.scope_kind, c.scope_key, s.state, r.revision_id, s.updated_at, "
            f"s.state_reason, r.content {latest} ORDER BY s.updated_at DESC, c.claim_id LIMIT ?",
            (*owner, MAX_REVIEW_ITEMS),
        ) as cursor:
            rows = list(await cursor.fetchall())
        items = tuple(
            MemoryForgottenSummary(
                claim_id=str(row[0]),
                scope_kind=str(row[1]),
                scope_key=str(row[2]) or None,
                state=str(row[3]),
                revision_id=str(row[4]),
                changed_at=str(row[5]),
                reason=str(row[6]),
                preview=_review_preview(str(row[7])),
            )
            for row in rows
        )
        return MemoryReviewList(items, int(total_row[0]) if total_row is not None else 0)

    async def restore_fact(
        self,
        authority: MemoryQueryAuthority,
        claim_id: str,
        *,
        revision_id: str,
        note: str,
        client_operation_id: str,
    ) -> MemoryLifecycleOutcome:
        """
        Restore a forgotten or expired fact to current truth.

        The new revision copies the prior revision's content, scope, and
        vector metadata (tags, source, speaker, confidence), becomes valid
        from now, and has no end date: restoring an expired fact with its
        old end date would expire it again immediately.

        Raises:
            WorkshopMemoryValidationError: Malformed input, or the operation
                id was already used for a different request.
            WorkshopMemoryNotFound: The claim is not the owner's, or the
                revision is not its latest retracted or expired revision.
            WorkshopMemoryConflictChanged: The claim gained current truth.
            WorkshopMemoryMutationFailed: Restored canonically, but the
                search projection did not complete.
        """
        checked_claim = self._validate_claim_id(claim_id)
        operation_id = self._validate_request_id(client_operation_id)
        checked_note = self._validate_note(note)
        if not isinstance(revision_id, str) or not revision_id or len(revision_id) > 128:
            raise WorkshopMemoryValidationError("Invalid memory revision")
        reason = "Owner restored the fact" + (f": {checked_note}" if checked_note else "")
        owner = self._owner_pair(authority)
        idempotency_key = f"memory-fact-restore:{authority.principal_id}:{operation_id}"
        lock = self._mutation_locks.setdefault(authority.principal_id, asyncio.Lock())
        async with lock:
            if not await self._owns_claim(owner, checked_claim):
                raise WorkshopMemoryNotFound("Memory not found")
            prior_event = await self._store.event_by_idempotency_key(idempotency_key)
            if prior_event is not None:
                payload = prior_event.envelope.payload
                if (
                    str(prior_event.envelope.aggregate_id) != checked_claim
                    or payload.get("prior_revision_id") != revision_id
                    or payload.get("reason") != reason
                ):
                    raise WorkshopMemoryValidationError("This operation was already used for a different request")
                restored = payload.get("revision")
                restored_id = restored.get("revision_id") if isinstance(restored, dict) else None
                return MemoryLifecycleOutcome(
                    checked_claim, str(restored_id), True, await self._claim_memory_id(checked_claim)
                )
            async with self._store.connection.execute(
                "SELECT r.content, r.asserted_at, r.observed_at, r.migration_classification, r.migration_gaps_json, "
                "r.vector_metadata_json, c.scope_kind, c.scope_key, s.state, "
                "(SELECT MAX(latest.created_event_position) FROM memory_fact_revisions latest "
                "WHERE latest.claim_id = c.claim_id) = r.created_event_position, "
                "(SELECT COUNT(*) FROM memory_fact_revision_states current "
                "WHERE current.claim_id = c.claim_id AND current.state IN ('active', 'unresolved_conflict')), "
                "r.admission_authority "
                "FROM memory_fact_revisions r JOIN memory_fact_claims c ON c.claim_id = r.claim_id "
                "JOIN memory_fact_revision_states s ON s.claim_id = r.claim_id AND s.revision_id = r.revision_id "
                "WHERE r.claim_id = ? AND r.revision_id = ?",
                (checked_claim, revision_id),
            ) as cursor:
                row = await cursor.fetchone()
            if row is None or not bool(row[9]) or str(row[8]) not in {"retracted", "expired"}:
                raise WorkshopMemoryNotFound("Memory not found")
            if int(row[10]) != 0:
                raise WorkshopMemoryConflictChanged()
            metadata = json.loads(str(row[5])) if row[5] else {}
            metadata = metadata if isinstance(metadata, dict) else {}
            confidence = metadata.get("confidence")
            now = datetime.now(UTC)
            spec = FactRevisionInput(
                content=str(row[0]),
                scope_kind=str(row[6]),
                scope_key=str(row[7] or ""),
                reason=reason,
                evidence=({"kind": "operator", "reference_id": operation_id, "sha256": None},),
                vector_metadata=metadata,
                confidence=(
                    float(confidence)
                    if isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
                    else 1.0
                ),
                asserted_at=_parse_stored_time(row[1]),
                observed_at=_parse_stored_time(row[2]),
                valid_from=now,
                valid_until=None,
                migration_classification=str(row[3]),
                migration_gaps=tuple(str(value) for value in _stored_json_list(row[4])),
                # Keep the prior admission: a fact the owner or operator had
                # admitted must not come back quarantined just because the
                # restore is a new revision.
                admission_authority=str(row[11]),
            )
            lifecycle_authority = await self._fact_lifecycle.authority_for(
                authority.principal_id,
                RuntimeProfileId(owner[1]),
            )
            try:
                mutation = await self._fact_lifecycle.restore(
                    lifecycle_authority,
                    MemoryClaimId(checked_claim),
                    MemoryRevisionId(revision_id),
                    spec,
                    idempotency_key=idempotency_key,
                )
            except FactLifecycleConflict as exc:
                raise WorkshopMemoryConflictChanged() from exc
        if mutation.projection_status != "succeeded":
            raise WorkshopMemoryMutationFailed("The fact is restored, but its search projection is pending")
        return MemoryLifecycleOutcome(
            checked_claim,
            str(mutation.revision_id),
            mutation.replayed,
            await self._claim_memory_id(checked_claim),
        )

    async def _claim_memory_id(self, claim_id: str) -> str | None:
        """
        Return the vector memory id of the claim's newest succeeded projection.

        The explorer addresses facts by vector memory id, not claim id. A
        resolution recreates the row through an upsert and a restore
        replaces it, so the newest succeeded operation that names a memory
        id is the row the owner can open next. Delete operations are
        excluded because any id they carry names a row that is gone.
        """
        async with self._store.connection.execute(
            "SELECT memory_id FROM memory_fact_vector_operations "
            "WHERE claim_id = ? AND status = 'succeeded' AND operation != 'delete' AND memory_id IS NOT NULL "
            "ORDER BY event_position DESC LIMIT 1",
            (claim_id,),
        ) as cursor:
            row = await cursor.fetchone()
        return str(row[0]) if row is not None else None

    async def projection_review(self, authority: MemoryQueryAuthority) -> ProjectionStatus:
        """List the owner's failed projections and count the operations blocked behind them."""
        return await projection_status_async(self._store.connection, self._owner_pair(authority))

    async def projection_failure_count(self, authority: MemoryQueryAuthority) -> int:
        """Count the owner's failed facts and episodes; zero when no runtime profile is unambiguous."""
        if authority.search_namespace is None:
            return 0
        return (await self.projection_review(authority)).failed_total

    async def retry_projections(
        self,
        authority: MemoryQueryAuthority,
        *,
        claim_ids: tuple[str, ...] | None,
        episode_ids: tuple[str, ...] | None,
    ) -> ProjectionRetryResult:
        """
        Retry the owner's failed fact and episode projections.

        With neither list given, every failed item the owner has is
        retried. With either list given, only the named items are, and an
        omitted list retries none of its kind. Retrying only resets rows
        already marked failed, so repeating a request is harmless and the
        route needs no operation id. Retries are scoped to the owner pair,
        so an id the owner does not have simply matches nothing.
        """
        _, runtime_profile_id = self._owner_pair(authority)
        for item_id in (*(claim_ids or ()), *(episode_ids or ())):
            self._validate_claim_id(item_id)
        everything = claim_ids is None and episode_ids is None
        facts = await self._fact_lifecycle.retry_failed(
            principal_id=authority.principal_id,
            runtime_profile_id=RuntimeProfileId(runtime_profile_id),
            claim_ids=None if everything else tuple(MemoryClaimId(item) for item in claim_ids or ()),
        )
        episodes = await self._episode_history.retry_failed(
            principal_id=authority.principal_id,
            runtime_profile_id=RuntimeProfileId(runtime_profile_id),
            episode_ids=None if everything else tuple(MemoryEpisodeId(item) for item in episode_ids or ()),
        )
        return ProjectionRetryResult(
            facts.retried + episodes.retried,
            facts.succeeded + episodes.succeeded,
            facts.failed + episodes.failed,
        )

    async def audit_projections(self, authority: MemoryQueryAuthority) -> VectorAudit:
        """
        Compare the owner's vector rows with canonical state.

        This reads the owner's whole vector corpus, so the Workshop runs it
        only when the owner asks. A store that cannot answer is a failed
        mutation-class error, not an empty audit.
        """
        owner = self._owner_pair(authority)
        try:
            rows = await asyncio.to_thread(
                memory.get_all_for_lifecycle_projection,
                user_id=owner[0],
                runtime_profile_id=owner[1],
            )
        except Exception as exc:
            raise WorkshopMemoryMutationFailed("The search index could not be read") from exc
        facts, episodes = await expected_rows_async(self._store.connection, owner)
        return audit_vector_rows(rows, current_facts=facts, current_episodes=episodes)

    async def unresolved_conflict_count(self, authority: MemoryQueryAuthority) -> int:
        """Count the owner's conflicted claims; zero when no runtime profile is unambiguous."""
        if authority.search_namespace is None:
            return 0
        owner = self._owner_pair(authority)
        async with self._store.connection.execute(
            "SELECT COUNT(*) FROM memory_fact_claims c WHERE c.owner_principal_id = ? AND c.runtime_profile_id = ? "
            "AND EXISTS (SELECT 1 FROM memory_fact_revision_states s "
            "WHERE s.claim_id = c.claim_id AND s.state = 'unresolved_conflict')",
            owner,
        ) as cursor:
            row = await cursor.fetchone()
        return int(row[0]) if row is not None else 0

    async def _awaiting_reconciliation(self, authority: MemoryQueryAuthority, memory_id: str) -> bool:
        """
        Return True when a strict lookup missed only because the row is legacy.

        Mutation paths read their target strictly, so an unreconciled
        legacy row looks absent to them. Re-reading with legacy admission
        tells that case apart from a truly missing row, so the caller can
        say the memory is waiting for reconciliation instead of reporting
        it as not found.
        """
        admitted = await asyncio.to_thread(
            memory.get_by_id,
            user_id=str(authority.principal_id),
            memory_id=memory_id,
            admit_legacy=True,
        )
        return admitted is not None and admitted.metadata.get(CANONICAL_TEMPORAL_ROLE_KEY) == LEGACY_UNRECONCILED_ROLE

    async def detail(
        self,
        authority: MemoryQueryAuthority,
        memory_id: str,
    ) -> MemoryRecordDetail:
        if not memory_id or len(memory_id) > 256:
            raise WorkshopMemoryValidationError("Invalid memory identifier")
        # Read-only view: admit legacy rows the list already shows, so a
        # listed memory never opens as "not found".
        result = await asyncio.to_thread(
            memory.get_by_id,
            user_id=str(authority.principal_id),
            memory_id=memory_id,
            admit_legacy=True,
        )
        if result is None:
            raise WorkshopMemoryNotFound("Memory not found")
        if len(result.text) > MAX_CONTENT_CHARACTERS:
            raise WorkshopMemoryResponseTooLarge("Memory content is too large")
        allowed_project_id = await self._allowed_project_id(authority)
        resolved = memory.resolve_memory_scope(result.metadata)
        speaker, confidence = memory.read_time_memory_speaker(result.metadata)
        provenance = memory.read_transcript_provenance(result.metadata)
        if provenance.malformed:
            source_reference = MemorySourceReference("invalid", None, None, None, None)
        elif provenance.canonical_present:
            source_reference = MemorySourceReference(
                "canonical",
                provenance.user_ts,
                provenance.assistant_ts,
                provenance.date,
                provenance.date_end,
            )
        elif result.metadata.get("source") == "explicit":
            source_reference = MemorySourceReference("explicit", None, None, None, None)
        else:
            source_reference = MemorySourceReference(
                "legacy",
                provenance.user_ts,
                provenance.assistant_ts,
                provenance.date,
                provenance.date_end,
            )
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
        source = str(result.metadata.get("source") or "")
        receipt_id = result.metadata.get(memory.EXTRACTION_RECEIPT_ID_KEY)
        extraction_provenance = "not_applicable"
        extraction_receipt = None
        if source in {"extracted", "episode"}:
            if receipt_id is None:
                extraction_provenance = "legacy"
            elif not isinstance(receipt_id, str) or not receipt_id or len(receipt_id) > 128:
                extraction_provenance = "invalid"
            else:
                try:
                    extraction_receipt = await MemoryExtractionReceiptService(self._store.connection).receipt(
                        MemoryExtractionReceiptService.authority_for_principal(authority.principal_id),
                        receipt_id,
                    )
                except MemoryExtractionReceiptAccessDenied:
                    extraction_provenance = "invalid"
                else:
                    run_id = result.metadata.get(memory.WORKSHOP_RUN_ID_KEY)
                    source_message_id = result.metadata.get(memory.WORKSHOP_SOURCE_MESSAGE_ID_KEY)
                    result_message_id = result.metadata.get(memory.WORKSHOP_RESULT_MESSAGE_ID_KEY)
                    if (
                        run_id != extraction_receipt.run_id
                        or source_message_id != extraction_receipt.source_message_id
                        or result_message_id != extraction_receipt.result_message_id
                    ):
                        extraction_receipt = None
                        extraction_provenance = "invalid"
                    else:
                        extraction_provenance = "canonical"
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
            source_reference=source_reference,
            extraction_provenance=extraction_provenance,
            extraction_receipt=extraction_receipt,
            lifecycle=await self._lifecycle_detail(authority, result),
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
        lifecycle_sets = await self._lifecycle_filter_sets(authority, filters.lifecycle)
        hits: list[MemorySearchHit] = []
        for hit in scoped.hits:
            result = hit.result
            if result.metadata.get("source") not in memory.USER_VISIBLE_SOURCES:
                continue
            if not self._matches(
                result,
                filters,
                active_claim_ids=lifecycle_sets[0],
                historical_claim_ids=lifecycle_sets[1],
                episode_ids=lifecycle_sets[2],
            ):
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
        # Read-only view: admit legacy rows the list already shows, so a
        # listed memory never opens as "not found".
        result = await asyncio.to_thread(
            memory.get_by_id,
            user_id=str(authority.principal_id),
            memory_id=memory_id,
            admit_legacy=True,
        )
        if result is None:
            raise WorkshopMemoryNotFound("Memory not found")
        provenance = memory.read_transcript_provenance(result.metadata)
        if provenance.malformed:
            return MemorySourceContext("unavailable", "invalid_provenance", None, None, None)
        if not provenance.canonical_present:
            if result.metadata.get("source") == "explicit":
                return MemorySourceContext("unavailable", "explicit_creation", None, None, None)
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
