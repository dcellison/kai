"""
Failed canonical memory vector projections: surfacing, retry, and read errors.

Runs the real fact lifecycle and episode history services, the real Mem0
vector adapters, and the real protected current-truth gate against the
real Workshop schema, reusing the storage-only Mem0 stand-in and harness
from the canonical write-path tests. Failures are injected into the
stand-in, the only fake in the path, so each test sees exactly what
production would: what the outbox records, what recall returns, what the
owner and operator surfaces report, and what a retry repairs.
"""

# ruff: noqa: F811 - tests take the imported `protected` fixture as a parameter by name.

from __future__ import annotations

import argparse
import asyncio
import inspect
from pathlib import Path

import pytest

from kai import memory, memory_admin
from kai.workshop.diagnostics import workshop_memory_current_truth_status
from kai.workshop.domain import MemoryClaimId, MemoryRevisionId
from kai.workshop.episode_history import Mem0EpisodeVectorAdapter, MemoryEpisodeHistoryService
from kai.workshop.memory_current_truth import CANONICAL_CLAIM_ID_KEY
from kai.workshop.memory_projection_status import projection_status_async
from kai.workshop.memory_queries import WorkshopMemoryMutationFailed
from tests.test_memory_canonical_write_path import (  # noqa: F401 - pytest fixture import
    PRINCIPAL_ID,
    RUNTIME_ID,
    _new,
    _recall,
    _store_fact,
    protected,
)
from tests.test_memory_owner_review import _query_service
from tests.test_workshop_episode_history import _spec as _episode_spec

OWNER = (str(PRINCIPAL_ID), str(RUNTIME_ID))


def _fail(*_args, **_kwargs):
    raise RuntimeError("vector store unavailable")


def _status_line() -> str:
    return workshop_memory_current_truth_status(
        Path(memory._config.session_db_path),  # type: ignore[union-attr]
        memory_enabled=True,
    )


async def _only_claim(store) -> MemoryClaimId:
    async with store.connection.execute("SELECT claim_id FROM memory_fact_claims") as cursor:
        (claim_id,) = [str(row[0]) for row in await cursor.fetchall()]
    return MemoryClaimId(claim_id)


def _fail_projection_reads(provider) -> None:
    """
    Make only the projection worker's row reads fail.

    Extraction reads the same row through the ordinary gated lookup first,
    to decide that an update applies; failing that read would stop the
    update before it reached the worker. The worker is the path under test.
    """
    original_get = provider.get

    def get(*, memory_id: str):
        if any(frame.function == "get_by_id_for_lifecycle_projection" for frame in inspect.stack()):
            raise RuntimeError("vector store unavailable")
        return original_get(memory_id=memory_id)

    provider.get = get
    provider.restore_get = original_get


async def _operation_statuses(store) -> list[tuple[str, str]]:
    async with store.connection.execute(
        "SELECT operation, status FROM memory_fact_vector_operations ORDER BY event_position"
    ) as cursor:
        return [(str(row[0]), str(row[1])) for row in await cursor.fetchall()]


# ── Failure, blocking, and retry ─────────────────────────────────────


async def test_failed_upsert_is_reported_everywhere_and_retry_restores_recall(
    protected, tmp_path: Path, monkeypatch
) -> None:
    store, provider, _lifecycle = protected
    original_add = provider.add
    monkeypatch.setattr(provider, "add", _fail)

    decisions = await _store_fact(store, _new("The operator prefers dark themes."), run=1)

    # Extraction already records a failed projection as a storage failure.
    assert [decision.outcome for decision in decisions] == ["storage_failed"]
    assert _recall() == []
    status = await projection_status_async(store.connection, OWNER)
    (item,) = status.failed
    # `add_structured` reports a provider failure as "no id", which the
    # worker records as a failed projection.
    assert (item.kind, item.operation, item.attempts, item.error_code) == (
        "fact",
        "upsert",
        3,
        "FactLifecycleProjectionFailed",
    )
    assert item.preview == "The operator prefers dark themes."
    line = _status_line()
    assert "INCOMPLETE" in line and "projection failed=1 (facts=1, episodes=0), blocked=0" in line
    service, authority = _query_service(store, tmp_path)
    assert (await service.stats(authority)).projection_failures == 1

    monkeypatch.setattr(provider, "add", original_add)
    result = await service.retry_projections(authority, claim_ids=(item.item_id,), episode_ids=None)

    assert (result.retried, result.succeeded, result.failed) == (1, 1, 0)
    assert _recall() == ["The operator prefers dark themes."]
    assert (await projection_status_async(store.connection, OWNER)).failed_total == 0
    assert "projection failed=0" in _status_line()


async def test_a_failure_blocks_later_changes_until_a_retry_runs_them_in_order(protected, tmp_path: Path) -> None:
    store, provider, lifecycle = protected
    await _store_fact(store, _new("The operator prefers dark themes."), run=1)
    claim_id = await _only_claim(store)
    revision_id = await lifecycle_current_revision(store, claim_id)
    authority = await lifecycle.authority_for(PRINCIPAL_ID, RUNTIME_ID)
    original_get = provider.get
    provider.get = _fail  # the retraction's delete cannot read its row

    retracted = await lifecycle.retract(
        authority, claim_id, revision_id, reason="No longer true.", idempotency_key="retract-1"
    )
    assert retracted.projection_status == "failed"
    provider.get = original_get
    service, owner = _query_service(store, tmp_path)

    # The restore commits, but its operation cannot run while the earlier
    # delete is failed, and the owner is told so rather than shown success.
    with pytest.raises(WorkshopMemoryMutationFailed):
        await service.restore_fact(
            owner, str(claim_id), revision_id=str(revision_id), note="", client_operation_id="r-1"
        )
    assert (await projection_status_async(store.connection, OWNER)).blocked == 1
    assert "blocked=1" in _status_line()

    result = await lifecycle.retry_failed(principal_id=PRINCIPAL_ID, runtime_profile_id=RUNTIME_ID)

    assert (result.retried, result.succeeded, result.failed) == (1, 1, 0)
    assert [status for _operation, status in await _operation_statuses(store)] == ["succeeded"] * 3
    assert _recall() == ["The operator prefers dark themes."]
    assert len(provider.rows) == 1


async def lifecycle_current_revision(store, claim_id: str) -> MemoryRevisionId:
    async with store.connection.execute(
        "SELECT revision_id FROM memory_fact_revision_states WHERE claim_id = ? AND state = 'active'",
        (claim_id,),
    ) as cursor:
        (row,) = await cursor.fetchall()
    return MemoryRevisionId(str(row[0]))


# ── Read errors are failures, never absence ──────────────────────────


async def test_delete_with_a_failed_read_fails_instead_of_succeeding(protected) -> None:
    store, provider, lifecycle = protected
    await _store_fact(store, _new("The operator prefers dark themes."), run=1)
    claim_id = await _only_claim(store)
    revision_id = await lifecycle_current_revision(store, claim_id)
    authority = await lifecycle.authority_for(PRINCIPAL_ID, RUNTIME_ID)
    original_get = provider.get
    provider.get = _fail

    retracted = await lifecycle.retract(
        authority, claim_id, revision_id, reason="No longer true.", idempotency_key="retract-1"
    )

    assert retracted.projection_status == "failed"
    assert len(provider.rows) == 1  # the row was not deleted, and nothing claims it was
    provider.get = original_get
    await lifecycle.retry_failed()
    assert provider.rows == {}


async def test_rewrite_with_a_failed_read_never_adds_a_duplicate(protected) -> None:
    store, provider, lifecycle = protected
    await _store_fact(store, _new("The operator prefers dark themes."), run=1)
    (memory_id,) = provider.rows
    _fail_projection_reads(provider)

    decisions = await _store_fact(
        store,
        {**_new("The operator prefers light themes."), "intent": "update_of", "existing_id": memory_id},
        run=2,
    )

    # The update itself read the row through the gate before the failure,
    # so the supersede was decided; only its rewrite failed to project.
    assert [decision.outcome for decision in decisions] == ["storage_failed"]
    assert list(provider.rows) == [memory_id]
    provider.get = provider.restore_get
    result = await lifecycle.retry_failed()
    assert result.succeeded == 1
    assert list(provider.rows) == [memory_id]
    assert _recall() == ["The operator prefers light themes."]


async def test_uninitialized_memory_fails_the_operation(protected, monkeypatch) -> None:
    store, _provider, _lifecycle = protected
    monkeypatch.setattr(memory, "_memory", None)

    with pytest.raises(memory.LifecycleProjectionReadError):
        memory.get_by_id_for_lifecycle_projection(
            user_id=str(PRINCIPAL_ID), memory_id="vec-1", runtime_profile_id=str(RUNTIME_ID)
        )
    with pytest.raises(memory.LifecycleProjectionReadError):
        memory.get_all_for_lifecycle_projection(user_id=str(PRINCIPAL_ID), runtime_profile_id=str(RUNTIME_ID))
    with pytest.raises(memory.LifecycleProjectionReadError):
        memory.delete_by_id_for_lifecycle_projection(
            user_id=str(PRINCIPAL_ID), memory_id="vec-1", runtime_profile_id=str(RUNTIME_ID)
        )
    del store


# ── Episodes ─────────────────────────────────────────────────────────


async def test_failed_episode_is_listed_and_retries_to_success(protected, tmp_path: Path, monkeypatch) -> None:
    store, provider, _lifecycle = protected
    episodes = MemoryEpisodeHistoryService(store, Mem0EpisodeVectorAdapter())
    authority = await episodes.authority_for(PRINCIPAL_ID, RUNTIME_ID)
    original_add = provider.add
    monkeypatch.setattr(provider, "add", _fail)

    recorded = await episodes.record(authority, _episode_spec(), idempotency_key="episode-1")

    assert recorded.projection_status == "failed"
    (item,) = (await projection_status_async(store.connection, OWNER)).failed
    assert (item.kind, item.item_id) == ("episode", str(recorded.episode_id))
    assert "projection failed=1 (facts=0, episodes=1)" in _status_line()

    monkeypatch.setattr(provider, "add", original_add)
    service, owner = _query_service(store, tmp_path)
    result = await service.retry_projections(owner, claim_ids=None, episode_ids=(str(recorded.episode_id),))
    assert (result.retried, result.succeeded) == (1, 1)
    assert len(provider.rows) == 1


# ── Vector audit ─────────────────────────────────────────────────────


async def test_audit_reports_orphan_unknown_and_duplicate_rows(protected, tmp_path: Path) -> None:
    store, provider, _lifecycle = protected
    await _store_fact(store, _new("The operator prefers dark themes."), run=1)
    ((memory_id, row),) = provider.rows.items()
    claim_id = row["metadata"][CANONICAL_CLAIM_ID_KEY]
    provider.add("Stale copy", user_id=row["user_id"], infer=False, metadata=dict(row["metadata"]))
    provider.add(
        "Unknown claim",
        user_id=row["user_id"],
        infer=False,
        metadata={**row["metadata"], CANONICAL_CLAIM_ID_KEY: "mcl_" + "0" * 32},
    )
    service, authority = _query_service(store, tmp_path)

    audit = await service.audit_projections(authority)

    assert audit.orphan == ("vec-2",)
    assert audit.unknown == ("vec-3",)
    assert audit.duplicate == 1
    # The gate never recalls either extra row.
    assert _recall() == ["The operator prefers dark themes."]
    assert memory_id == "vec-1" and claim_id


# ── Operator CLI ─────────────────────────────────────────────────────


async def test_cli_reports_and_retries_failures(protected, monkeypatch, capsys) -> None:
    store, provider, _lifecycle = protected
    original_add = provider.add
    monkeypatch.setattr(provider, "add", _fail)
    await _store_fact(store, _new("The operator prefers dark themes."), run=1)
    monkeypatch.setattr(provider, "add", original_add)
    config = memory._config
    monkeypatch.setattr("kai.config.load_config", lambda: config)
    monkeypatch.setattr(memory_admin, "_initialize_memory", lambda _config=None: config)
    monkeypatch.setattr(memory, "close_memory", lambda: None)

    status_code = await asyncio.to_thread(
        memory_admin._cmd_projections, argparse.Namespace(projections_command="status", principal=None)
    )
    status_output = capsys.readouterr().out
    retry_code = await asyncio.to_thread(
        memory_admin._cmd_projections, argparse.Namespace(projections_command="retry", principal=None)
    )
    retry_output = capsys.readouterr().out

    assert status_code == 0
    assert "outbox; failed=1, blocked=0" in status_output
    assert (
        "fact mcl_" in status_output
        and "upsert failed after 3 attempt(s) (FactLifecycleProjectionFailed)" in status_output
    )
    assert "vector audit" in status_output and "orphan=0, unknown=0, duplicate=0" in status_output
    assert retry_code == 0
    assert "retried=1, succeeded=1, failed=0" in retry_output
    assert _recall() == ["The operator prefers dark themes."]


def test_cli_parses_projection_commands() -> None:
    parser = memory_admin._build_parser()
    assert parser.parse_args(["projections", "status"]).projections_command == "status"
    assert parser.parse_args(["projections", "retry", "--principal", "prn_x"]).principal == "prn_x"


# ── Payload filter against a real local vector store ─────────────────


def test_payload_filter_matches_one_revision_in_local_qdrant(tmp_path: Path) -> None:
    """
    Mem0's local Qdrant filter must match a flattened metadata key.

    The projection worker's lookup depends on this: it asks the store for
    the rows whose payload carries one canonical revision id, rather than
    listing the owner's whole corpus.
    """
    # Mem0 is the optional `memory` extra; like the other real-store tests,
    # this runs only where it is installed.
    qdrant = pytest.importorskip("mem0.vector_stores.qdrant", reason="mem0ai not installed")
    Qdrant = qdrant.Qdrant

    store = Qdrant(collection_name="lookup", embedding_model_dims=4, path=str(tmp_path / "qdrant"), on_disk=True)
    store.insert(
        vectors=[[0.1, 0.2, 0.3, 0.4]] * 3,
        payloads=[
            {"user_id": "owner", "data": "a", "canonical_memory_revision_id": "mrv_a"},
            {"user_id": "owner", "data": "b", "canonical_memory_revision_id": "mrv_b"},
            {"user_id": "other", "data": "c", "canonical_memory_revision_id": "mrv_a"},
        ],
        ids=[
            "00000000-0000-0000-0000-000000000001",
            "00000000-0000-0000-0000-000000000002",
            "00000000-0000-0000-0000-000000000003",
        ],
    )

    listed = store.list(filters={"user_id": "owner", "canonical_memory_revision_id": "mrv_a"}, top_k=100)
    rows = listed[0] if listed and isinstance(listed[0], list) else listed

    assert [row.payload["data"] for row in rows] == ["a"]
    store.client.close()


async def test_single_row_reads_log_their_exclusions(protected, monkeypatch, caplog) -> None:
    store, provider, _lifecycle = protected
    await _store_fact(store, _new("The operator prefers dark themes."), run=1)
    ((_memory_id, row),) = provider.rows.items()
    provider.add("Stale copy", user_id=row["user_id"], infer=False, metadata=dict(row["metadata"]))
    monkeypatch.setattr(memory, "_current_truth_log_times", {})
    caplog.set_level("INFO", logger="kai.memory")

    assert memory.get_by_id(user_id=str(PRINCIPAL_ID), memory_id="vec-2", runtime_profile_id=str(RUNTIME_ID)) is None

    (line,) = [
        record.getMessage() for record in caplog.records if "Memory current truth get_by_id" in record.getMessage()
    ]
    assert "projection_not_current" in line and "Stale copy" not in line
