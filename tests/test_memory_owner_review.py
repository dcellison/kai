"""
Owner review of canonical facts: conflict resolution and fact restore.

Runs the Workshop memory query service against the real Workshop schema,
the real fact lifecycle and vector adapter, and the real protected
current-truth gate, reusing the storage-only Mem0 stand-in and harness
from the canonical write-path tests. Conflicts are opened the way
production opens them: a low-confidence extraction update of a stored
fact.
"""

# ruff: noqa: F811 - tests take the imported `protected` fixture as a parameter by name.

from __future__ import annotations

from pathlib import Path

import pytest

from kai import memory
from kai.config import Config
from kai.workshop.domain import AgentId, ChannelId
from kai.workshop.execution_state import WorkshopExecutionStateNamespace, WorkshopExecutionStateRegistry
from kai.workshop.memory_queries import (
    WorkshopMemoryConflictChanged,
    WorkshopMemoryMutationFailed,
    WorkshopMemoryNotFound,
    WorkshopMemoryQueryService,
    WorkshopMemoryValidationError,
)
from tests.test_memory_canonical_write_path import (  # noqa: F401 - pytest fixture import
    PRINCIPAL_ID,
    RUNTIME_ID,
    _new,
    _recall,
    _store_fact,
    protected,
)


class _RuntimePool:
    """Runtime pool stand-in: the owner review paths only need a workspace."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    async def get_effective_workspace(self, _profile_id) -> Path:
        return self.workspace


def _query_service(store, tmp_path: Path) -> tuple[WorkshopMemoryQueryService, object]:
    namespace = WorkshopExecutionStateNamespace(
        principal_id=PRINCIPAL_ID,
        channel_id=ChannelId("chn_23000000000000000000000000000001"),
        agent_id=AgentId("agt_23000000000000000000000000000001"),
        runtime_profile_id=RUNTIME_ID,
        legacy_runtime_key=1,
    )
    service = WorkshopMemoryQueryService(
        # The service writes stored audits through `session_db_path`, which
        # in production is the store's own database; the fixture's store
        # lives at tmp_path / "kai.db".
        Config(
            telegram_bot_token="token",
            allowed_user_ids={1},
            memory_enabled=True,
            session_db_path=tmp_path / "kai.db",
        ),
        store,
        _RuntimePool(tmp_path),  # type: ignore[arg-type]
        WorkshopExecutionStateRegistry((namespace,)),
    )
    return service, service.authority_for_principal(PRINCIPAL_ID)


async def _conflict(store, provider) -> str:
    """Store a fact, then open a conflict with a low-confidence update; return its claim id."""
    await _store_fact(store, _new("The operator prefers dark themes."), run=1)
    (memory_id,) = provider.rows
    decisions = await _store_fact(
        store,
        {**_new("The operator prefers light themes.", confidence=0.6), "intent": "update_of", "existing_id": memory_id},
        run=2,
    )
    assert [decision.outcome for decision in decisions] == ["conflict_opened"]
    async with store.connection.execute("SELECT claim_id FROM memory_fact_claims") as cursor:
        (claim_id,) = [str(row[0]) for row in await cursor.fetchall()]
    return claim_id


# ── Conflicts ────────────────────────────────────────────────────────


async def test_conflict_is_listed_with_both_sides_and_full_detail(protected, tmp_path: Path) -> None:
    store, provider, _lifecycle = protected
    claim_id = await _conflict(store, provider)
    service, authority = _query_service(store, tmp_path)

    listing = await service.list_conflicts(authority)
    detail = await service.conflict_detail(authority, claim_id)
    stats_count = await service.unresolved_conflict_count(authority)

    assert listing.total == 1 and stats_count == 1
    (conflict,) = listing.items
    assert conflict.claim_id == claim_id
    assert [revision.preview for revision in conflict.revisions] == [
        "The operator prefers dark themes.",
        "The operator prefers light themes.",
    ]
    unresolved = [revision for revision in detail["revisions"] if revision["state"] == "unresolved_conflict"]
    assert {revision["confidence"] for revision in unresolved} == {0.95, 0.6}
    assert {revision["admissionAuthority"] for revision in unresolved} == {"provenance_verified"}
    assert _recall() == []


@pytest.mark.parametrize(("keep_index", "expected"), [(1, "light"), (0, "dark")])
async def test_keeping_either_side_restores_exactly_that_fact_to_recall(
    protected, tmp_path: Path, keep_index: int, expected: str
) -> None:
    store, provider, _lifecycle = protected
    claim_id = await _conflict(store, provider)
    service, authority = _query_service(store, tmp_path)
    revisions = [revision.revision_id for revision in (await service.list_conflicts(authority)).items[0].revisions]

    outcome = await service.resolve_conflict(
        authority,
        claim_id,
        keep_revision_id=revisions[keep_index],
        expected_revision_ids=revisions,
        note="",
        client_operation_id="resolve-1",
    )

    assert outcome.active_revision_id == revisions[keep_index]
    assert _recall() == [f"The operator prefers {expected} themes."]
    assert outcome.memory_id in provider.rows
    assert (await service.list_conflicts(authority)).total == 0
    detail = await service._fact_lifecycle_detail(authority, claim_id, None)
    assert detail is not None
    states = {revision["revisionId"]: revision["state"] for revision in detail["revisions"]}
    assert states[revisions[keep_index]] == "active"
    assert states[revisions[1 - keep_index]] == "superseded"


async def test_stale_or_invalid_selection_changes_nothing(protected, tmp_path: Path) -> None:
    store, provider, _lifecycle = protected
    claim_id = await _conflict(store, provider)
    service, authority = _query_service(store, tmp_path)
    revisions = [revision.revision_id for revision in (await service.list_conflicts(authority)).items[0].revisions]
    async with store.connection.execute("SELECT COUNT(*) FROM event_log") as cursor:
        (before,) = await cursor.fetchone()

    with pytest.raises(WorkshopMemoryConflictChanged):
        await service.resolve_conflict(
            authority,
            claim_id,
            keep_revision_id=revisions[0],
            expected_revision_ids=revisions[:1],
            note="",
            client_operation_id="stale",
        )
    with pytest.raises(WorkshopMemoryValidationError):
        await service.resolve_conflict(
            authority,
            claim_id,
            keep_revision_id="not-a-member",
            expected_revision_ids=revisions,
            note="",
            client_operation_id="invalid",
        )
    with pytest.raises(WorkshopMemoryNotFound):
        await service.resolve_conflict(
            authority,
            "mcl_" + "f" * 32,
            keep_revision_id=revisions[0],
            expected_revision_ids=revisions,
            note="",
            client_operation_id="foreign",
        )

    assert (await service.list_conflicts(authority)).total == 1
    async with store.connection.execute("SELECT COUNT(*) FROM event_log") as cursor:
        (after,) = await cursor.fetchone()
    assert after == before


async def test_resolution_replays_and_refuses_a_reused_operation(protected, tmp_path: Path) -> None:
    store, provider, _lifecycle = protected
    claim_id = await _conflict(store, provider)
    service, authority = _query_service(store, tmp_path)
    revisions = [revision.revision_id for revision in (await service.list_conflicts(authority)).items[0].revisions]
    request = {
        "keep_revision_id": revisions[1],
        "expected_revision_ids": revisions,
        "note": "The newer statement is right.",
        "client_operation_id": "resolve-once",
    }

    first = await service.resolve_conflict(authority, claim_id, **request)
    async with store.connection.execute("SELECT COUNT(*) FROM event_log") as cursor:
        (events_after_first,) = await cursor.fetchone()
    replay = await service.resolve_conflict(authority, claim_id, **request)
    async with store.connection.execute("SELECT COUNT(*) FROM event_log") as cursor:
        (events_after_replay,) = await cursor.fetchone()

    assert first.replayed is False
    assert replay.replayed is True and replay.active_revision_id == first.active_revision_id
    assert events_after_replay == events_after_first
    with pytest.raises(WorkshopMemoryValidationError):
        await service.resolve_conflict(authority, claim_id, **{**request, "keep_revision_id": revisions[0]})


async def test_resolution_reports_a_failed_projection(protected, tmp_path: Path, monkeypatch) -> None:
    store, provider, _lifecycle = protected
    claim_id = await _conflict(store, provider)
    service, authority = _query_service(store, tmp_path)
    revisions = [revision.revision_id for revision in (await service.list_conflicts(authority)).items[0].revisions]

    def refuse(*_args, **_kwargs):
        raise RuntimeError("vector store unavailable")

    monkeypatch.setattr(provider, "add", refuse)

    with pytest.raises(WorkshopMemoryMutationFailed):
        await service.resolve_conflict(
            authority,
            claim_id,
            keep_revision_id=revisions[1],
            expected_revision_ids=revisions,
            note="",
            client_operation_id="resolve-failed",
        )


# ── Forgotten facts ──────────────────────────────────────────────────


async def test_forgotten_fact_restores_with_metadata_and_no_end_date(protected, tmp_path: Path) -> None:
    store, provider, _lifecycle = protected
    await _store_fact(store, _new("The operator prefers dark themes."), run=1)
    (memory_id,) = provider.rows
    service, authority = _query_service(store, tmp_path)
    deleted = await service.delete(authority, [memory_id])
    assert [result.outcome for result in deleted.results] == ["succeeded"]
    assert _recall() == []

    listing = await service.list_forgotten(authority)
    (fact,) = listing.items
    assert fact.state == "retracted"
    assert fact.preview == "The operator prefers dark themes."

    outcome = await service.restore_fact(
        authority,
        fact.claim_id,
        revision_id=fact.revision_id,
        note="Still true.",
        client_operation_id="restore-1",
    )

    assert _recall() == ["The operator prefers dark themes."]
    assert outcome.memory_id in provider.rows
    assert (await service.list_forgotten(authority)).total == 0
    (row,) = provider.rows.values()
    assert row["metadata"]["tags"] == ["preference"]
    detail = await service._fact_lifecycle_detail(authority, fact.claim_id, None)
    assert detail is not None
    restored = next(
        revision for revision in detail["revisions"] if revision["revisionId"] == outcome.active_revision_id
    )
    assert restored["validUntil"] is None
    assert restored["supersedesRevisionId"] == fact.revision_id
    assert restored["admissionAuthority"] == "provenance_verified"


async def test_expired_fact_restores_and_current_claims_are_not_restorable(protected, tmp_path: Path) -> None:
    store, provider, lifecycle = protected
    await _store_fact(store, _new("The operator prefers dark themes."), run=1)
    service, authority = _query_service(store, tmp_path)
    (memory_id,) = provider.rows
    current = memory.get_by_id(user_id=str(PRINCIPAL_ID), memory_id=memory_id, runtime_profile_id=str(RUNTIME_ID))
    assert current is not None
    lifecycle_authority = await lifecycle.authority_for(PRINCIPAL_ID, RUNTIME_ID)
    snapshot = await lifecycle.adopt_legacy(lifecycle_authority, current, idempotency_key="snapshot")

    with pytest.raises(WorkshopMemoryNotFound):
        await service.restore_fact(
            authority,
            str(snapshot.claim_id),
            revision_id=str(snapshot.revision_id),
            note="",
            client_operation_id="restore-active",
        )

    await lifecycle.retract(
        lifecycle_authority,
        snapshot.claim_id,
        snapshot.revision_id,
        reason="Validity ended.",
        idempotency_key="expire",
        expired=True,
    )
    (fact,) = (await service.list_forgotten(authority)).items
    assert fact.state == "expired"

    await service.restore_fact(
        authority,
        fact.claim_id,
        revision_id=fact.revision_id,
        note="",
        client_operation_id="restore-expired",
    )
    assert _recall() == ["The operator prefers dark themes."]


# ── Diagnostics and HTTP mapping ─────────────────────────────────────


async def test_install_status_counts_open_conflicts_without_marking_gaps(protected, tmp_path: Path) -> None:
    from kai.workshop.diagnostics import workshop_memory_current_truth_status

    store, provider, _lifecycle = protected
    await _conflict(store, provider)
    database = Path(memory._config.session_db_path)  # type: ignore[union-attr]

    status = workshop_memory_current_truth_status(database, memory_enabled=True)

    assert "unresolved conflicts=1 (oldest=0d)" in status


def test_conflict_changed_maps_to_its_own_conflict_response() -> None:
    import json

    from kai.workshop.client_api import _memory_error_response

    response = _memory_error_response(WorkshopMemoryConflictChanged())
    body = json.loads(response.text or "{}")

    assert response.status == 409
    assert body["error"]["code"] == "memory_conflict_changed"
