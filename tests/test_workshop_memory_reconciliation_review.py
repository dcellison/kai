"""Principal-scoped Workshop review state for legacy memory reconciliation."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from kai import memory, memory_reconciliation
from kai.memory import MemoryResult
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.diagnostics import workshop_memory_reconciliation_status
from kai.workshop.domain import PrincipalId
from kai.workshop.memory_reconciliation_review import (
    MemoryReconciliationReviewAccessDenied,
    MemoryReconciliationReviewConflict,
    MemoryReconciliationReviewValidationError,
    WorkshopMemoryReconciliationReviewService,
    prior_reconciliation_dispositions,
    record_reconciliation_audit,
)
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id

NOW = datetime(2026, 9, 21, 18, 0, tzinfo=UTC)


@pytest.mark.asyncio
async def test_schema_91_upgrades_to_canonical_reconciliation_triage(tmp_path: Path, monkeypatch) -> None:
    from kai.workshop import schema

    path = tmp_path / "upgrade.db"
    with monkeypatch.context() as migration_context:
        migration_context.setattr(schema, "WORKSHOP_SCHEMA_VERSION", 91)
        migration_context.setattr(schema, "_MIGRATIONS", schema._MIGRATIONS[:91])
        old = await WorkshopEventStore.open(path)
        await old.close()

    upgraded = await WorkshopEventStore.open(path)
    try:
        assert await upgraded.schema_version() == schema.WORKSHOP_SCHEMA_VERSION
        assert {
            "memory_reconciliation_audits",
            "memory_reconciliation_decisions",
            "memory_reconciliation_receipts",
            "memory_reconciliation_operations",
            "memory_reconciliation_triage_plans",
            "memory_reconciliation_triage_groups",
            "memory_reconciliation_triage_recommendations",
        }.issubset(await upgraded.schema_tables())
    finally:
        await upgraded.close()


def _row(memory_id: str, text: str, **metadata: object) -> MemoryResult:
    return MemoryResult(
        memory_id,
        text,
        0.0,
        "fact",
        {"source": "extracted", "scope": "global", "confidence": 0.8, **metadata},
        NOW.isoformat(),
        NOW.isoformat(),
    )


async def _open_store(path: Path) -> tuple[WorkshopEventStore, PrincipalId, PrincipalId]:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman("Alice", "admin", "telegram", "101", "101", profile_id(101)),
            BootstrapHuman("Bob", "member", "telegram", "202", "202", profile_id(202)),
        ),
    )
    principals: dict[str, PrincipalId] = {}
    async with store.connection.execute(
        "SELECT external_subject, principal_id FROM external_identities "
        "WHERE provider = 'telegram' AND external_subject IN ('101', '202')"
    ) as cursor:
        for row in await cursor.fetchall():
            principals[str(row[0])] = PrincipalId(str(row[1]))
    return store, principals["101"], principals["202"]


@pytest.mark.asyncio
async def test_review_decisions_are_principal_scoped_replay_safe_and_durable(tmp_path: Path):
    db_path = tmp_path / "kai.db"
    store, alice, bob = await _open_store(db_path)
    rows = [_row("mem-1", "Alice uses a legacy preference")]
    audit = memory_reconciliation.build_audit(
        principal_id=str(alice),
        runtime_profile_id=str(profile_id(101)),
        rows=rows,
        now=NOW,
    )
    assert record_reconciliation_audit(db_path, audit) is True
    assert record_reconciliation_audit(db_path, audit) is False
    service = WorkshopMemoryReconciliationReviewService(store, db_path=db_path)

    summary = await service.latest(alice)
    assert summary is not None
    assert summary.candidate_count == audit["candidate_count"]
    page = await service.candidates(alice, audit["audit_id"])
    candidate = page.candidates[0]
    with pytest.raises(MemoryReconciliationReviewAccessDenied):
        await service.candidates(bob, audit["audit_id"])

    result = await service.decide(
        alice,
        audit["audit_id"],
        candidate["candidate_id"],
        disposition="defer",
        action=candidate["proposed_action"],
        operator_note="Needs later review.",
        expected_state_version=0,
        client_operation_id="review-one",
    )
    replay = await service.decide(
        alice,
        audit["audit_id"],
        candidate["candidate_id"],
        disposition="defer",
        action=candidate["proposed_action"],
        operator_note="Needs later review.",
        expected_state_version=0,
        client_operation_id="review-one",
    )
    assert result["replayed"] is False
    assert replay["replayed"] is True
    with pytest.raises(MemoryReconciliationReviewConflict):
        await service.decide(
            alice,
            audit["audit_id"],
            candidate["candidate_id"],
            disposition="reject",
            action=candidate["proposed_action"],
            operator_note="Different request.",
            expected_state_version=1,
            client_operation_id="review-one",
        )

    await store.close()
    reopened = await WorkshopEventStore.open(db_path)
    try:
        durable = await WorkshopMemoryReconciliationReviewService(reopened, db_path=db_path).candidates(
            alice,
            audit["audit_id"],
            disposition="defer",
        )
        assert [item["candidate_id"] for item in durable.candidates] == [candidate["candidate_id"]]
    finally:
        await reopened.close()
    assert workshop_memory_reconciliation_status(db_path).startswith(
        "Workshop memory reconciliation: active; audits=1 (open=1, applied=0)"
    )


@pytest.mark.asyncio
async def test_bulk_review_forbids_approval_and_apply_is_receipt_backed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    db_path = tmp_path / "kai.db"
    store, alice, _bob = await _open_store(db_path)
    rows = [_row("mem-1", "A legacy fact")]
    audit = memory_reconciliation.build_audit(
        principal_id=str(alice),
        runtime_profile_id=str(profile_id(101)),
        rows=rows,
        now=NOW,
    )
    record_reconciliation_audit(db_path, audit)
    service = WorkshopMemoryReconciliationReviewService(store, db_path=db_path)
    page = await service.candidates(alice, audit["audit_id"], limit=50)
    candidate_ids = [item["candidate_id"] for item in page.candidates]

    with pytest.raises(MemoryReconciliationReviewValidationError, match="Bulk approval"):
        await service.bulk_decide(
            alice,
            audit["audit_id"],
            candidate_ids=candidate_ids,
            disposition="approve",  # type: ignore[arg-type]
            operator_note="Unsafe.",
            expected_review_version=0,
            client_operation_id="bulk-approve",
        )
    reviewed = await service.bulk_decide(
        alice,
        audit["audit_id"],
        candidate_ids=candidate_ids,
        disposition="reject",
        operator_note="Legacy evidence is not trustworthy enough to adopt.",
        expected_review_version=0,
        client_operation_id="bulk-reject",
    )
    monkeypatch.setattr(memory, "get_all_for_lifecycle_projection", lambda **_kwargs: rows)
    applied = await service.apply(
        alice,
        audit["audit_id"],
        expected_review_version=reviewed["review_version"],
        client_operation_id="apply-reviewed",
    )
    replay = await service.apply(
        alice,
        audit["audit_id"],
        expected_review_version=reviewed["review_version"],
        client_operation_id="apply-reviewed",
    )

    assert applied["receipt"]["applied"] == []
    assert replay["replayed"] is True
    assert (candidate_ids[0], page.candidates[0]["state_sha256"]) in prior_reconciliation_dispositions(
        db_path,
        principal_id=str(alice),
        runtime_profile_id=str(profile_id(101)),
    )
    await store.close()


@pytest.mark.asyncio
async def test_review_approval_rejects_project_outside_current_authority(tmp_path: Path):
    db_path = tmp_path / "kai.db"
    store, alice, _bob = await _open_store(db_path)
    audit = memory_reconciliation.build_audit(
        principal_id=str(alice),
        runtime_profile_id=str(profile_id(101)),
        rows=[_row("mem-project", "Project memory", scope="project", project_id="private")],
        now=NOW,
    )
    record_reconciliation_audit(db_path, audit)
    service = WorkshopMemoryReconciliationReviewService(store, db_path=db_path)
    page = await service.candidates(alice, audit["audit_id"])
    candidate = page.candidates[0]

    with pytest.raises(MemoryReconciliationReviewAccessDenied, match="outside"):
        await service.decide(
            alice,
            audit["audit_id"],
            candidate["candidate_id"],
            disposition="approve",
            action=candidate["proposed_action"],
            operator_note="Adopt project memory.",
            expected_state_version=0,
            client_operation_id="project-denied",
            allowed_project_ids=frozenset(),
        )
    await store.close()
