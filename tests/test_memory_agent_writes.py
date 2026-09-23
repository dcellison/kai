"""
Agent memory writes and the stored vector audit on a protected install.

The internal memory API lets an agent save a fact or forget everything.
These tests run those paths through the Workshop memory query service,
the real fact lifecycle and vector adapter, and the real protected
current-truth gate, with only Mem0's storage replaced (the harness from
the canonical write-path tests). Recall assertions go through
`memory.search`, so a save that reached the store but not current truth
shows up as missing, exactly as it would for the agent.
"""

# ruff: noqa: F811 - tests take the imported `protected` fixture as a parameter by name.

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from kai import memory, memory_reconciliation
from kai.workshop.diagnostics import workshop_memory_current_truth_status
from kai.workshop.memory_queries import WorkshopMemoryMutationFailed
from kai.workshop.memory_reconciliation_review import record_reconciliation_audit
from tests.test_memory_canonical_write_path import (  # noqa: F401 - pytest fixture import
    NOW,
    PRINCIPAL_ID,
    RUNTIME_ID,
    _new,
    _recall,
    _store_fact,
    protected,
)
from tests.test_memory_owner_review import _conflict, _query_service


def _agent_metadata(**overrides: object) -> dict[str, object]:
    """The metadata the internal API handler builds for a global explicit save."""
    return {
        "speaker": "assistant",
        "confidence": 0.9,
        "source": "explicit",
        **memory.build_scope_metadata(scope="global", project_id=None, scope_source="extraction_default"),
        **overrides,
    }


def _legacy_row(content: str) -> str:
    """
    Write a row the way the internal API used to: owner-stamped, no claim.

    The protected gate treats such a row as legacy memory, which is the
    failure the lifecycle path fixes; here it stands for legacy rows an
    owner still has when they forget everything.
    """
    memory_id = memory.add_structured(
        content,
        user_id=str(PRINCIPAL_ID),
        memory_type="fact",
        metadata={"source": "explicit", "speaker": "assistant", "confidence": 0.9},
        runtime_profile_id=str(RUNTIME_ID),
    )
    assert memory_id is not None
    return memory_id


# ── Saving ───────────────────────────────────────────────────────────


async def test_agent_save_is_current_truth_and_recalled(protected, tmp_path: Path) -> None:
    store, provider, _lifecycle = protected
    service, authority = _query_service(store, tmp_path)

    memory_id = await service.record_agent_fact(
        principal_id=PRINCIPAL_ID,
        runtime_profile_id=RUNTIME_ID,
        content="The operator drinks Earl Grey.",
        tags=["tea"],
        vector_metadata=_agent_metadata(),
    )

    assert _recall() == ["The operator drinks Earl Grey."]
    assert provider.rows[memory_id]["metadata"]["tags"] == ["tea"]
    assert provider.rows[memory_id]["metadata"]["source"] == "explicit"
    # History names where the fact came from.
    detail = await service.detail(authority, memory_id)
    assert detail.lifecycle is not None
    (revision,) = detail.lifecycle["revisions"]
    assert revision["state"] == "active"
    assert revision["reason"] == "Fact saved deliberately by the owner's agent."


async def test_the_old_direct_write_is_not_recalled(protected, tmp_path: Path) -> None:
    # The defect the lifecycle path fixes: once the owner's legacy memory
    # is reconciled, a claimless row is never admitted by the gate.
    store, _provider, _lifecycle = protected
    audit = memory_reconciliation.build_audit(
        principal_id=str(PRINCIPAL_ID), runtime_profile_id=str(RUNTIME_ID), rows=[], now=NOW
    )
    record_reconciliation_audit(tmp_path / "kai.db", audit)
    await store.connection.execute(
        "UPDATE memory_reconciliation_audits SET status = 'applied', applied_at = ?", (NOW.isoformat(),)
    )
    await store.connection.commit()

    _legacy_row("The operator drinks Earl Grey.")

    assert _recall() == []


async def test_agent_save_in_a_project_keeps_its_scope(protected, tmp_path: Path) -> None:
    store, _provider, _lifecycle = protected
    service, _authority = _query_service(store, tmp_path)

    await service.record_agent_fact(
        principal_id=PRINCIPAL_ID,
        runtime_profile_id=RUNTIME_ID,
        content="The build uses make check.",
        tags=None,
        vector_metadata=_agent_metadata(
            **memory.build_scope_metadata(scope="project", project_id="kai", scope_source="extraction_default")
        ),
    )

    async with store.connection.execute("SELECT scope_kind, scope_key FROM memory_fact_claims") as cursor:
        assert [tuple(row) for row in await cursor.fetchall()] == [("project", "kai")]


async def test_agent_save_with_a_failed_projection_is_reported(protected, tmp_path: Path, monkeypatch) -> None:
    store, provider, _lifecycle = protected
    service, _authority = _query_service(store, tmp_path)

    def refuse(*_args, **_kwargs):
        raise RuntimeError("vector store unavailable")

    monkeypatch.setattr(provider, "add", refuse)

    with pytest.raises(WorkshopMemoryMutationFailed):
        await service.record_agent_fact(
            principal_id=PRINCIPAL_ID,
            runtime_profile_id=RUNTIME_ID,
            content="The operator drinks Earl Grey.",
            tags=None,
            vector_metadata=_agent_metadata(),
        )

    # The claim is canonical and waits for the retry; it is not lost.
    async with store.connection.execute(
        "SELECT status FROM memory_fact_vector_operations ORDER BY event_position"
    ) as cursor:
        assert [str(row[0]) for row in await cursor.fetchall()] == ["failed"]


# ── Forgetting everything ────────────────────────────────────────────


async def test_forget_all_retracts_facts_deletes_legacy_rows_and_keeps_conflicts(protected, tmp_path: Path) -> None:
    store, provider, _lifecycle = protected
    claim_in_conflict = await _conflict(store, provider)
    await _store_fact(store, _new("The operator uses a Mac mini."), run=3)
    legacy_id = _legacy_row("An old legacy fact.")
    service, authority = _query_service(store, tmp_path)

    result = await service.forget_all_for_agent(principal_id=PRINCIPAL_ID, runtime_profile_id=RUNTIME_ID)

    assert (result.retracted, result.projection_failed, result.legacy_deleted) == (1, 0, 1)
    assert (result.kept_conflicts, result.kept_episodes) == (1, 0)
    assert _recall() == []
    assert legacy_id not in provider.rows
    # Every forgotten fact stays in history and can be restored.
    forgotten = await service.list_forgotten(authority)
    assert [item.preview for item in forgotten.items] == ["The operator uses a Mac mini."]
    conflicts = await service.list_conflicts(authority)
    assert [item.claim_id for item in conflicts.items] == [claim_in_conflict]


async def test_forget_all_is_safe_to_repeat(protected, tmp_path: Path) -> None:
    store, _provider, _lifecycle = protected
    await _store_fact(store, _new("The operator uses a Mac mini."), run=1)
    service, _authority = _query_service(store, tmp_path)

    await service.forget_all_for_agent(principal_id=PRINCIPAL_ID, runtime_profile_id=RUNTIME_ID)
    again = await service.forget_all_for_agent(principal_id=PRINCIPAL_ID, runtime_profile_id=RUNTIME_ID)

    assert (again.retracted, again.legacy_deleted) == (0, 0)


# ── Stored vector audit ──────────────────────────────────────────────


def _stored_audit(database: Path) -> list[tuple[int, int, int, int]]:
    connection = sqlite3.connect(database)
    try:
        return [
            (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
            for row in connection.execute(
                "SELECT orphan_rows, unknown_rows, duplicate_items, missing_rows FROM memory_vector_audit"
            )
        ]
    finally:
        connection.close()


async def test_startup_audit_is_stored_and_install_status_reads_it(protected, tmp_path: Path) -> None:
    store, provider, _lifecycle = protected
    await _store_fact(store, _new("The operator prefers dark themes."), run=1)
    service, _authority = _query_service(store, tmp_path)
    database = tmp_path / "kai.db"
    assert "vector drift=not checked" in workshop_memory_current_truth_status(database, memory_enabled=True)

    assert await service.refresh_vector_audits() == 1

    assert _stored_audit(database) == [(0, 0, 0, 0)]
    assert "vector drift=0 (missing=0, checked 0m ago)" in workshop_memory_current_truth_status(
        database, memory_enabled=True
    )

    # A current fact whose row vanished is a loss recall cannot return.
    provider.rows.clear()
    await service.refresh_vector_audits()

    assert _stored_audit(database) == [(0, 0, 0, 1)]
    status = workshop_memory_current_truth_status(database, memory_enabled=True)
    assert "vector drift=1 (missing=1," in status
    assert status.startswith("Workshop memory current truth: INCOMPLETE;")


async def test_projection_retry_refreshes_the_stored_audit(protected, tmp_path: Path, monkeypatch) -> None:
    store, provider, _lifecycle = protected
    original_add = provider.add

    def refuse(*_args, **_kwargs):
        raise RuntimeError("vector store unavailable")

    monkeypatch.setattr(provider, "add", refuse)
    await _store_fact(store, _new("The operator prefers dark themes."), run=1)
    monkeypatch.setattr(provider, "add", original_add)
    service, authority = _query_service(store, tmp_path)

    result = await service.retry_projections(authority, claim_ids=None, episode_ids=None)

    assert result.succeeded == 1
    assert _stored_audit(tmp_path / "kai.db") == [(0, 0, 0, 0)]
