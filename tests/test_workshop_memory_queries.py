"""Canonical Workshop semantic-memory query service tests."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from kai import memory
from kai.config import Config, MemoryProjectConfig
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.conversation_commands import WorkshopConversationCommandService
from kai.workshop.domain import (
    AgentId,
    ChannelId,
    EventEnvelope,
    EventId,
    MessageId,
    PrincipalId,
    WorkshopEventType,
)
from kai.workshop.execution_state import (
    WorkshopExecutionStateNamespace,
    WorkshopExecutionStateRegistry,
)
from kai.workshop.inbound import InboundMessage
from kai.workshop.memory_queries import (
    MAX_MUTATION_TARGETS,
    MEMORY_CONTENT_AUDIT_EVENT,
    MEMORY_MANAGEMENT_AUDIT_EVENT,
    MemoryEpisodeEdit,
    MemoryFactEdit,
    MemoryProjectOption,
    MemoryQueryFilters,
    WorkshopMemoryAccessDenied,
    WorkshopMemoryConflict,
    WorkshopMemoryCursorError,
    WorkshopMemoryMutationFailed,
    WorkshopMemoryNotFound,
    WorkshopMemoryQueryService,
    WorkshopMemoryResponseTooLarge,
    WorkshopMemoryValidationError,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id


class _RuntimePool:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    async def get_effective_workspace(self, _profile_id) -> Path:
        return self.workspace


def _result(
    memory_id: str,
    text: str,
    *,
    source: str = "extracted",
    created_at: str = "2026-08-24T10:00:00Z",
    score: float = 0.8,
    scope: str = "global",
    project_id: str | None = None,
    tags: list[str] | None = None,
) -> memory.MemoryResult:
    metadata: dict[str, object] = {
        "source": source,
        "type": "episode" if source == "episode" else "fact",
        "scope": scope,
        "scope_source": "extraction_default",
        "scope_confidence": 1.0,
        "tags": tags or [],
        "speaker": "episode_summary" if source == "episode" else "user",
        "confidence": 1.0,
    }
    if project_id is not None:
        metadata["project_id"] = project_id
    return memory.MemoryResult(
        id=memory_id,
        text=text,
        score=score,
        memory_type=str(metadata["type"]),
        metadata=metadata,
        created_at=created_at,
        updated_at=created_at,
    )


def _service(tmp_path: Path):
    principal_id = PrincipalId("prn_" + "1" * 32)
    namespace = WorkshopExecutionStateNamespace(
        principal_id=principal_id,
        channel_id=ChannelId("chn_" + "2" * 32),
        agent_id=AgentId("agt_" + "3" * 32),
        runtime_profile_id=profile_id(101),
        legacy_runtime_key=101,
    )
    runtime_pool = _RuntimePool(tmp_path)
    service = WorkshopMemoryQueryService(
        Config(
            telegram_bot_token="unused",
            allowed_user_ids=set(),
            default_backend="codex",
            default_model="gpt-5.6-sol",
        ),
        SimpleNamespace(),  # type: ignore[arg-type]
        runtime_pool,  # type: ignore[arg-type]
        WorkshopExecutionStateRegistry((namespace,)),
    )
    return service, service.authority_for_principal(principal_id), principal_id


def test_authority_is_canonical_and_fails_closed(tmp_path: Path) -> None:
    service, authority, principal_id = _service(tmp_path)

    assert authority.principal_id == principal_id
    with pytest.raises(WorkshopMemoryAccessDenied):
        service.authority_for_principal(PrincipalId("prn_" + "9" * 32))
    with pytest.raises(WorkshopMemoryAccessDenied):
        service.authority_for_principal("101")


async def test_management_projects_follow_canonical_workspace_authority(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, authority, _ = _service(tmp_path)
    allowed_root = (tmp_path / "allowed" / "kai").resolve()
    blocked_root = (tmp_path / "blocked" / "other").resolve()
    disabled_root = (tmp_path / "allowed" / "disabled").resolve()
    service._config = replace(
        service._config,
        memory_projects={
            "kai": MemoryProjectConfig("kai", "Kai", (allowed_root,), True),
            "other": MemoryProjectConfig("other", "Other", (blocked_root,), True),
            "disabled": MemoryProjectConfig("disabled", "Disabled", (disabled_root,), False),
        },
    )
    runtime_pool = service._runtime_pool
    monkeypatch.setattr(runtime_pool, "get_home_workspace", lambda _profile_id: tmp_path / "home", raising=False)

    async def workspace(_profile_id):
        return allowed_root

    async def access(_profile_id):
        return None, [allowed_root, disabled_root]

    monkeypatch.setattr(runtime_pool, "get_effective_workspace", workspace)
    monkeypatch.setattr(runtime_pool, "resolve_workspace_access", access, raising=False)

    assert await service.allowed_projects(authority) == (MemoryProjectOption("kai", "Kai"),)


async def test_list_is_visible_filtered_deterministic_and_cursor_bound(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, authority, principal_id = _service(tmp_path)
    rows = [
        _result("new", "Newest", created_at="2026-08-24T12:00:00Z", tags=["kai"]),
        _result("old", "Older", created_at="2026-08-24T11:00:00Z", tags=["kai"]),
        _result("hidden", "Hidden", source="legacy-internal"),
    ]
    calls: list[tuple[str, int | None]] = []
    offloaded: list[object] = []
    real_to_thread = asyncio.to_thread

    async def observed_to_thread(function, /, *args, **kwargs):
        offloaded.append(function)
        return await real_to_thread(function, *args, **kwargs)

    def get_all(*, user_id: str, limit: int | None):
        calls.append((user_id, limit))
        return rows

    monkeypatch.setattr(memory, "get_all", get_all)
    monkeypatch.setattr(asyncio, "to_thread", observed_to_thread)

    first = await service.list_records(
        authority,
        filters=MemoryQueryFilters(tag="kai"),
        limit=1,
    )
    assert [record.memory_id for record in first.records] == ["new"]
    assert first.next_cursor is not None
    second = await service.list_records(
        authority,
        filters=MemoryQueryFilters(tag="kai"),
        limit=1,
        cursor=first.next_cursor,
    )
    assert [record.memory_id for record in second.records] == ["old"]
    assert second.next_cursor is None
    assert calls == [(str(principal_id), None), (str(principal_id), None)]
    assert offloaded == [get_all, get_all]

    oldest_first = await service.list_records(
        authority,
        filters=MemoryQueryFilters(tag="kai"),
        order="oldest",
    )
    assert [record.memory_id for record in oldest_first.records] == ["old", "new"]

    with pytest.raises(WorkshopMemoryCursorError):
        await service.list_records(
            authority,
            filters=MemoryQueryFilters(source="episode"),
            cursor=first.next_cursor,
        )
    with pytest.raises(WorkshopMemoryCursorError):
        await service.list_records(
            authority,
            filters=MemoryQueryFilters(tag="kai"),
            cursor=first.next_cursor,
            order="oldest",
        )
    with pytest.raises(WorkshopMemoryValidationError):
        await service.list_records(authority, limit=101)
    with pytest.raises(WorkshopMemoryValidationError):
        await service.list_records(authority, order="relevance")


async def test_stats_and_detail_expose_only_bounded_stable_fields(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, authority, principal_id = _service(tmp_path)
    fact = _result("fact-1", "Daniel prefers concise output.", tags=["preference"])
    episode = _result("episode-1", "Deploy succeeded.", source="episode")
    episode.metadata.update(
        {
            "goal": "Deploy Kai",
            "outcome": "Succeeded",
            "private_unknown_field": "/secret/path",
        }
    )
    monkeypatch.setattr(
        memory,
        "get_all",
        lambda **_kwargs: [fact, episode, _result("hidden", "no", source="hidden")],
    )
    monkeypatch.setattr(
        memory,
        "get_by_id",
        lambda *, user_id, memory_id: episode if user_id == str(principal_id) and memory_id == episode.id else None,
    )

    stats = await service.stats(authority)
    assert (stats.total, stats.facts, stats.episodes) == (2, 1, 1)
    assert stats.by_source == {"episode": 1, "extracted": 1}

    detail = await service.detail(authority, episode.id)
    assert detail.content == "Deploy succeeded."
    assert detail.episode == {
        "goal": "Deploy Kai",
        "outcome": "Succeeded",
        "tags": [],
    }
    assert "private_unknown_field" not in detail.compact_recall
    assert '"record_type":"memory"' in detail.compact_recall
    with pytest.raises(WorkshopMemoryNotFound):
        await service.detail(authority, "someone-elses-memory")
    with pytest.raises(WorkshopMemoryValidationError):
        await service.detail(authority, "x" * 257)

    oversized = _result("oversized", "x" * 100_001)
    monkeypatch.setattr(memory, "get_by_id", lambda **_kwargs: oversized)
    with pytest.raises(WorkshopMemoryResponseTooLarge):
        await service.detail(authority, oversized.id)

    oversized_episode = _result("oversized-episode", "short", source="episode")
    oversized_episode.metadata["goal"] = "x" * 120_001
    monkeypatch.setattr(memory, "get_by_id", lambda **_kwargs: oversized_episode)
    with pytest.raises(WorkshopMemoryResponseTooLarge):
        await service.detail(authority, oversized_episode.id)


async def test_scope_interpretation_matches_production_admission(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, authority, _ = _service(tmp_path)
    global_row = _result("global", "Global")
    matching = _result(
        "matching",
        "Matching",
        scope="project",
        project_id="kai",
    )
    foreign = _result(
        "foreign",
        "Foreign",
        scope="project",
        project_id="other",
    )
    legacy = _result("legacy", "Legacy")
    legacy.metadata.pop("scope")
    legacy.metadata.pop("scope_source")
    invalid = _result("invalid", "Invalid")
    invalid.metadata["scope"] = "nonsense"
    monkeypatch.setattr(
        memory,
        "get_all",
        lambda **_kwargs: [global_row, matching, foreign, legacy, invalid],
    )

    async def allowed_project(_authority):
        return "kai"

    monkeypatch.setattr(service, "_allowed_project_id", allowed_project)
    page = await service.list_records(authority)
    scopes = {record.memory_id: record.scope for record in page.records}
    assert scopes["global"].retrievable is True
    assert scopes["matching"].retrievable is True
    assert scopes["foreign"].exclusion_reason == "project_id_mismatch"
    assert scopes["legacy"].exclusion_reason == "legacy_scope_quarantined"
    assert scopes["invalid"].exclusion_reason == "invalid_scope_quarantined"

    project_only = await service.list_records(
        authority,
        filters=MemoryQueryFilters(
            memory_type="fact",
            scope="project",
            project_id="kai",
        ),
    )
    assert [record.memory_id for record in project_only.records] == ["matching"]


async def test_scope_management_is_bounded_authorized_partial_and_audited(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    service, authority, principal_id = _service(tmp_path)
    secret_text = "Daniel's private memory content"
    rows = {
        "ok": _result("ok", secret_text),
        "stale": _result("stale", "vanishes"),
        "failed": _result("failed", "remains"),
    }
    updates: list[tuple[str, dict[str, object]]] = []

    async def projects(_authority):
        return (MemoryProjectOption("kai", "Kai"),)

    async def active_project(_authority):
        return "kai"

    def get_by_id(*, user_id: str, memory_id: str):
        assert user_id == str(principal_id)
        return rows.get(memory_id)

    def update_metadata(*, user_id: str, memory_id: str, data: str, metadata):
        assert user_id == str(principal_id)
        assert data == rows[memory_id].text
        updates.append((memory_id, metadata))
        if memory_id == "stale":
            rows.pop(memory_id)
            return False
        return memory_id == "ok"

    monkeypatch.setattr(service, "allowed_projects", projects)
    monkeypatch.setattr(service, "_allowed_project_id", active_project)
    monkeypatch.setattr(memory, "get_by_id", get_by_id)
    monkeypatch.setattr(memory, "update_metadata", update_metadata)
    caplog.set_level("INFO", logger="kai.workshop.memory_queries")

    batch = await service.move_scope(
        authority,
        ["ok", "failed", "stale", "missing"],
        scope="project",
        project_id="kai",
    )

    assert [result.outcome for result in batch.results] == [
        "succeeded",
        "failed",
        "stale",
        "not_found",
    ]
    assert updates[0][1]["source"] == "extracted"
    assert updates[0][1]["scope"] == "project"
    assert updates[0][1]["project_id"] == "kai"
    assert updates[0][1]["scope_source"] == memory.SCOPE_SOURCE_OPERATOR
    audit = [record.message for record in caplog.records if record.message.startswith(MEMORY_MANAGEMENT_AUDIT_EVENT)]
    assert len(audit) == 4
    assert all(str(principal_id) in line for line in audit)
    assert secret_text not in "\n".join(audit)
    assert '"outcome":"stale"' in "\n".join(audit)

    with pytest.raises(WorkshopMemoryAccessDenied):
        await service.move_scope(authority, ["ok"], scope="project", project_id="foreign")
    with pytest.raises(WorkshopMemoryValidationError):
        await service.move_scope(authority, ["ok"], scope="global", project_id="kai")
    with pytest.raises(WorkshopMemoryValidationError):
        await service.move_scope(authority, ["ok"] * 2, scope="global")
    with pytest.raises(WorkshopMemoryValidationError):
        await service.move_scope(
            authority,
            [f"memory-{index}" for index in range(MAX_MUTATION_TARGETS + 1)],
            scope="global",
        )


async def test_delete_management_restricts_sources_and_reports_stale_targets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, authority, principal_id = _service(tmp_path)
    rows = {
        "ok": _result("ok", "delete me", source="episode"),
        "stale": _result("stale", "already deleting"),
    }

    def get_by_id(*, user_id: str, memory_id: str):
        assert user_id == str(principal_id)
        return rows.get(memory_id)

    def delete_by_id(*, user_id: str, memory_id: str):
        assert user_id == str(principal_id)
        rows.pop(memory_id, None)
        return memory_id == "ok"

    monkeypatch.setattr(memory, "get_by_id", get_by_id)
    monkeypatch.setattr(memory, "delete_by_id", delete_by_id)

    batch = await service.delete(authority, ["ok", "missing", "stale"])

    assert [result.outcome for result in batch.results] == ["succeeded", "not_found", "stale"]
    assert all(memory_id not in rows for memory_id in ("ok", "stale"))


async def test_explicit_fact_creation_is_scoped_idempotent_and_content_free_in_audit(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    service, authority, principal_id = _service(tmp_path)
    rows: dict[str, memory.MemoryResult] = {}
    secret = "Q1055 private explicit fact"

    monkeypatch.setattr(memory, "get_all", lambda **_kwargs: list(rows.values()))
    monkeypatch.setattr(memory, "get_by_id", lambda **kwargs: rows.get(kwargs["memory_id"]))

    def add_structured(content: str, **kwargs):
        metadata = dict(kwargs["metadata"])
        metadata.update({"type": kwargs["memory_type"], "tags": kwargs["tags"]})
        rows["created"] = _result("created", content, source="explicit")
        rows["created"].metadata.clear()
        rows["created"].metadata.update(metadata)
        return "created"

    monkeypatch.setattr(memory, "add_structured", add_structured)
    caplog.set_level("INFO", logger="kai.workshop.memory_queries")

    created = await service.create_fact(
        authority,
        content=secret,
        tags=["qualification"],
        scope="global",
        project_id=None,
        request_id="create-request-1",
    )
    replay = await service.create_fact(
        authority,
        content=secret,
        tags=["qualification"],
        scope="global",
        project_id=None,
        request_id="create-request-1",
    )

    assert created.created is True
    assert replay.created is False
    assert created.record.record.source == "explicit"
    assert rows["created"].metadata["operator_created_by_principal_id"] == str(principal_id)
    assert rows["created"].metadata["scope_source"] == memory.SCOPE_SOURCE_OPERATOR
    audit = "\n".join(
        record.message for record in caplog.records if record.message.startswith(MEMORY_CONTENT_AUDIT_EVENT)
    )
    assert '"outcome":"succeeded"' in audit
    assert '"outcome":"idempotent"' in audit
    assert secret not in audit


async def test_fact_edit_preserves_provenance_reindexes_and_replays_safely(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, authority, principal_id = _service(tmp_path)
    original = _result("fact", "Old searchable wording", tags=["old"])
    original.metadata["confirmation_quote"] = "immutable source quote"
    rows = {original.id: original}
    writes: list[tuple[str, dict[str, object]]] = []

    monkeypatch.setattr(memory, "get_by_id", lambda **kwargs: rows.get(kwargs["memory_id"]))

    def update_metadata(*, memory_id: str, data: str, metadata, **_kwargs):
        writes.append((data, metadata))
        previous = rows[memory_id]
        rows[memory_id] = memory.MemoryResult(
            id=previous.id,
            text=data,
            score=previous.score,
            memory_type=previous.memory_type,
            metadata=dict(metadata),
            created_at=previous.created_at,
            updated_at="2026-08-26T12:00:00Z",
        )
        return True

    monkeypatch.setattr(memory, "update_metadata", update_metadata)
    revision = (await service.detail(authority, original.id)).record.revision
    edit = MemoryFactEdit("New searchable wording", ("new",))

    saved = await service.edit(
        authority,
        original.id,
        revision=revision,
        request_id="edit-request-1",
        edit=edit,
    )
    replay = await service.edit(
        authority,
        original.id,
        revision=revision,
        request_id="edit-request-1",
        edit=edit,
    )

    assert saved.changed_fields == ("content", "tags")
    assert saved.idempotent_replay is False
    assert replay.idempotent_replay is True
    assert len(writes) == 1
    assert writes[0][0] == "New searchable wording"
    assert writes[0][1]["source"] == "extracted"
    assert writes[0][1]["confirmation_quote"] == "immutable source quote"
    assert writes[0][1]["operator_edited_by_principal_id"] == str(principal_id)
    assert saved.record.record.revision != revision


async def test_episode_edit_rebuilds_canonical_search_text_and_schema(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, authority, _ = _service(tmp_path)
    row = _result("episode", "Old goal\n\nOld context", source="episode")
    row.metadata.update(
        {
            "goal": "Old goal",
            "context": "Old context",
            "approach": "Old approach",
            "outcome": "Old outcome",
            "outcome_quality": "partial",
            "tags": ["old"],
            "actors": ["Daniel"],
        }
    )
    rows = {row.id: row}
    monkeypatch.setattr(memory, "get_by_id", lambda **kwargs: rows.get(kwargs["memory_id"]))

    def update_metadata(*, memory_id: str, data: str, metadata, **_kwargs):
        previous = rows[memory_id]
        rows[memory_id] = memory.MemoryResult(
            previous.id,
            data,
            previous.score,
            previous.memory_type,
            dict(metadata),
            previous.created_at,
            "2026-08-26T12:00:00Z",
        )
        return True

    monkeypatch.setattr(memory, "update_metadata", update_metadata)
    revision = (await service.detail(authority, row.id)).record.revision
    edit = MemoryEpisodeEdit(
        goal="New semantic goal",
        context="New semantic context",
        approach="New approach",
        outcome="New outcome",
        outcome_quality="success",
        lessons="Keep the canonical representation aligned.",
        tags=("new",),
        actors=("Daniel", "Kai"),
    )

    saved = await service.edit(
        authority,
        row.id,
        revision=revision,
        request_id="episode-edit-1",
        edit=edit,
    )

    assert rows[row.id].text == "New semantic goal\n\nNew semantic context"
    assert saved.record.episode == {
        "goal": "New semantic goal",
        "context": "New semantic context",
        "approach": "New approach",
        "outcome": "New outcome",
        "lessons": "Keep the canonical representation aligned.",
        "outcome_quality": "success",
        "tags": ["new"],
        "actors": ["Daniel", "Kai"],
    }


async def test_edit_rejects_stale_revision_and_reports_unchanged_provider_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, authority, _ = _service(tmp_path)
    row = _result("fact", "Original")
    rows = {row.id: row}
    monkeypatch.setattr(memory, "get_by_id", lambda **kwargs: rows.get(kwargs["memory_id"]))
    monkeypatch.setattr(memory, "update_metadata", lambda **_kwargs: False)
    revision = (await service.detail(authority, row.id)).record.revision

    with pytest.raises(WorkshopMemoryMutationFailed):
        await service.edit(
            authority,
            row.id,
            revision=revision,
            request_id="failed-edit",
            edit=MemoryFactEdit("Changed", ()),
        )

    rows[row.id] = memory.MemoryResult(
        row.id,
        "External correction",
        row.score,
        row.memory_type,
        dict(row.metadata),
        row.created_at,
        "2026-08-26T12:00:00Z",
    )
    with pytest.raises(WorkshopMemoryConflict):
        await service.edit(
            authority,
            row.id,
            revision=revision,
            request_id="stale-edit",
            edit=MemoryFactEdit("Changed", ()),
        )


async def test_edit_recovers_when_provider_reports_failure_after_committing(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, authority, _ = _service(tmp_path)
    row = _result("fact", "Original", tags=["old"])
    rows = {row.id: row}
    monkeypatch.setattr(memory, "get_by_id", lambda **kwargs: rows.get(kwargs["memory_id"]))

    def update_then_report_failure(*, memory_id: str, data: str, metadata, **_kwargs):
        previous = rows[memory_id]
        rows[memory_id] = memory.MemoryResult(
            previous.id,
            data,
            previous.score,
            previous.memory_type,
            dict(metadata),
            previous.created_at,
            "2026-08-26T12:00:00Z",
        )
        return False

    monkeypatch.setattr(memory, "update_metadata", update_then_report_failure)
    revision = (await service.detail(authority, row.id)).record.revision

    saved = await service.edit(
        authority,
        row.id,
        revision=revision,
        request_id="commit-before-failure",
        edit=MemoryFactEdit("Committed correction", ("new",)),
    )

    assert saved.idempotent_replay is True
    assert saved.record.content == "Committed correction"
    assert saved.record.record.tags == ("new",)


async def test_content_mutations_reject_invalid_typed_fields(
    tmp_path: Path,
) -> None:
    service, authority, _ = _service(tmp_path)

    with pytest.raises(WorkshopMemoryValidationError, match="at most 32"):
        await service.create_fact(
            authority,
            content="Bounded fact",
            tags=[f"tag-{index}" for index in range(33)],
            scope="global",
            project_id=None,
            request_id="too-many-tags",
        )

    with pytest.raises(WorkshopMemoryValidationError, match="Outcome quality"):
        await service.edit(
            authority,
            "episode",
            revision="mr1_valid",
            request_id="bad-outcome-quality",
            edit=MemoryEpisodeEdit(
                goal="Goal",
                context="Context",
                approach="Approach",
                outcome="Outcome",
                outcome_quality="unknown",
                lessons=None,
                tags=(),
                actors=(),
            ),
        )


async def test_search_uses_live_scoped_retrieval_and_visible_sources(
    tmp_path: Path,
    monkeypatch,
) -> None:
    service, authority, principal_id = _service(tmp_path)
    visible = _result("visible", "Remember this", score=0.9)
    hidden = _result("hidden", "Do not expose", source="internal", score=0.99)
    resolved = memory.resolve_memory_scope(visible.metadata)
    contexts: list[memory.ScopedRetrievalContext] = []

    async def retrieve(context, *, limit):
        contexts.append(context)
        return memory.ScopedRetrievalResult(
            hits=[
                memory.ScopedMemoryHit(hidden, resolved, "user", 1.0, 1.0, 0.99),
                memory.ScopedMemoryHit(visible, resolved, "user", 1.0, 1.0, 0.9),
            ],
            debug=SimpleNamespace(
                active_project_id=None,
                allowed_project_id=None,
                reason="ok",
            ),
        )

    monkeypatch.setattr(memory, "retrieve_scoped_memories", retrieve)

    result = await service.search(authority, "remember", limit=5)
    assert [hit.record.memory_id for hit in result.hits] == ["visible"]
    assert result.hits[0].compact_recall == memory.format_memory_result_for_recall(visible)
    assert contexts[0].chat_id == str(principal_id)
    assert contexts[0].workspace == tmp_path
    with pytest.raises(WorkshopMemoryValidationError):
        await service.search(authority, "")
    with pytest.raises(WorkshopMemoryValidationError):
        await service.search(authority, "x" * 2_001)


async def test_source_context_requires_canonical_lineage_and_channel_membership(
    tmp_path: Path,
    monkeypatch,
) -> None:
    store = await WorkshopEventStore.open(tmp_path / "kai.db")
    try:
        await bootstrap_default_workshop(
            store,
            (
                BootstrapHuman(
                    "Alice",
                    "admin",
                    "telegram",
                    "101",
                    "101",
                    profile_id(101),
                ),
            ),
        )
        async with store.connection.execute(
            "SELECT e.principal_id, b.channel_id FROM external_identities e "
            "JOIN channel_bindings b ON b.transport = e.provider "
            "AND b.external_channel_id = e.external_subject "
            "WHERE e.external_subject = '101'",
        ) as cursor:
            identity = await cursor.fetchone()
        assert identity is not None
        principal_id = PrincipalId(str(identity[0]))
        channel_id = ChannelId(str(identity[1]))
        async with store.connection.execute(
            "SELECT a.id, a.principal_id FROM channel_agents ca "
            "JOIN agents a ON a.id = ca.agent_id WHERE ca.channel_id = ?",
            (channel_id,),
        ) as cursor:
            agent_row = await cursor.fetchone()
        assert agent_row is not None
        agent_id = AgentId(str(agent_row[0]))
        agent_principal_id = PrincipalId(str(agent_row[1]))
        namespace = WorkshopExecutionStateNamespace(
            principal_id=principal_id,
            channel_id=channel_id,
            agent_id=agent_id,
            runtime_profile_id=profile_id(101),
            legacy_runtime_key=101,
        )
        service = WorkshopMemoryQueryService(
            Config(
                telegram_bot_token="unused",
                allowed_user_ids=set(),
                default_backend="codex",
                default_model="gpt-5.6-sol",
            ),
            store,
            _RuntimePool(tmp_path),  # type: ignore[arg-type]
            WorkshopExecutionStateRegistry((namespace,)),
        )
        authority = service.authority_for_principal(principal_id)
        now = datetime(2026, 8, 24, 12, 0, tzinfo=UTC)
        accepted = await WorkshopConversationCommandService(store).accept(
            InboundMessage(
                "telegram",
                "update-1",
                "message-1",
                "101",
                "101",
                "What is the marker?",
                now,
            )
        )
        run = accepted.run
        result_message_id = MessageId.new()
        result_event = EventEnvelope.create(
            event_id=EventId.new(),
            event_type=WorkshopEventType.MESSAGE_CREATED,
            event_version=1,
            workshop_id=run.workshop_id,
            aggregate_type="message",
            aggregate_id=result_message_id,
            actor_principal_id=agent_principal_id,
            occurred_at=now,
            idempotency_key="memory-source-result",
            payload={
                "channel_id": channel_id,
                "author_principal_id": agent_principal_id,
                "reply_to_message_id": run.inbound_message_id,
                "body": "The marker is BLUE-LANTERN.",
            },
            metadata={"source": "agent"},
        )
        appended = await store.append(result_event)
        await store.project_pending(CanonicalConversationProjection())
        await store.connection.execute(
            "UPDATE runs SET status = 'completed', started_at = ?, terminal_at = ?, "
            "result_message_id = ?, last_event_position = ? WHERE id = ?",
            (now.isoformat(), now.isoformat(), result_message_id, appended.event.position, run.run_id),
        )
        await store.connection.commit()
        row = _result("canonical", "The marker is BLUE-LANTERN.")
        row.metadata.update(
            {
                memory.WORKSHOP_PRINCIPAL_ID_KEY: str(principal_id),
                memory.WORKSHOP_CHANNEL_ID_KEY: str(channel_id),
                memory.WORKSHOP_AGENT_ID_KEY: str(agent_id),
                memory.WORKSHOP_RUNTIME_PROFILE_ID_KEY: str(profile_id(101)),
                memory.WORKSHOP_RUN_ID_KEY: str(run.run_id),
                memory.WORKSHOP_SOURCE_MESSAGE_ID_KEY: str(run.inbound_message_id),
                memory.WORKSHOP_RESULT_MESSAGE_ID_KEY: str(result_message_id),
            }
        )
        monkeypatch.setattr(memory, "get_by_id", lambda **_kwargs: row)

        context = await service.source_context(authority, row.id)
        assert context.status == "available"
        assert context.source is not None and context.source.body == "What is the marker?"
        assert context.result is not None and context.result.body == "The marker is BLUE-LANTERN."

        row.metadata[memory.WORKSHOP_PRINCIPAL_ID_KEY] = "prn_" + "9" * 32
        denied = await service.source_context(authority, row.id)
        assert (denied.status, denied.reason) == ("unavailable", "source_not_authorized")

        row.metadata[memory.WORKSHOP_PRINCIPAL_ID_KEY] = str(principal_id)
        row.metadata[memory.WORKSHOP_RUN_ID_KEY] = "run_" + "9" * 32
        missing = await service.source_context(authority, row.id)
        assert (missing.status, missing.reason) == (
            "unavailable",
            "canonical_source_missing",
        )

        row.metadata.clear()
        row.metadata.update({"source": "extracted"})
        legacy = await service.source_context(authority, row.id)
        assert (legacy.status, legacy.reason) == ("unavailable", "legacy_source")
    finally:
        await store.close()
