"""
Resumable, drift-scoped reconciliation apply, carried decisions, and the legacy census.

Runs `apply_review` and the Workshop reconciliation services against the
real Workshop schema and the real fact lifecycle and episode history
services, with the storage-only Mem0 stand-in holding the legacy rows the
audit saw. Failures are injected at the one step each test is about, so
each test checks what production would see: which writes happened, what
the run and progress rows say, and what a retry does.
"""

from __future__ import annotations

import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from kai import memory, memory_reconciliation
from kai.memory import MemoryResult
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.domain import AgentId, ChannelId, PrincipalId
from kai.workshop.execution_state import WorkshopExecutionStateNamespace, WorkshopExecutionStateRegistry
from kai.workshop.fact_lifecycle import MemoryFactLifecycleService
from kai.workshop.memory_legacy_census import count_legacy_census
from kai.workshop.memory_reconciliation_review import (
    MemoryReconciliationReviewConflict,
    WorkshopMemoryReconciliationReviewService,
    prior_reconciliation_row_dispositions,
    record_reconciliation_audit,
)
from kai.workshop.memory_reconciliation_triage import WorkshopMemoryReconciliationTriageService
from kai.workshop.runtime_pool import WorkshopRuntimePool
from kai.workshop.store import WorkshopEventStore
from tests.test_memory_legacy_episode_triage import _approve_safe_groups, _episode, _stored
from tests.test_memory_reconciliation_triage import NOW, _row
from tests.workshop_profiles import profile_id


class _Owner:
    """A bootstrapped owner with memory authority, a stamped corpus, and its audit."""

    def __init__(self, tmp_path: Path, rows: list[MemoryResult]) -> None:
        self.tmp_path = tmp_path
        self.db_path = tmp_path / "kai.db"
        self.rows = rows

    async def start(self, monkeypatch) -> _Owner:
        self.store = await WorkshopEventStore.open(self.db_path)
        await bootstrap_default_workshop(
            self.store,
            (BootstrapHuman("Alice", "admin", "telegram", "101", "101", profile_id(101)),),
        )
        async with self.store.connection.execute(
            "SELECT principal_id FROM external_identities WHERE provider = 'telegram' AND external_subject = '101'"
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        self.principal = PrincipalId(str(row[0]))
        self.runtime = str(profile_id(101))
        # Production legacy rows carry their owner's principal stamp, which
        # every owner-verified read requires; the audit sees the same rows.
        self.rows = [
            replace(item, metadata={**item.metadata, memory.WORKSHOP_PRINCIPAL_ID_KEY: str(self.principal)})
            for item in self.rows
        ]
        self.provider = _stored(self.principal, self.rows)
        monkeypatch.setattr(memory, "_memory", self.provider)
        memory.configure_memory_authority(
            WorkshopExecutionStateRegistry(
                (
                    WorkshopExecutionStateNamespace(
                        principal_id=self.principal,
                        channel_id=ChannelId("chn_" + "1" * 32),
                        agent_id=AgentId("agt_" + "1" * 32),
                        runtime_profile_id=profile_id(101),
                        legacy_runtime_key=101,
                    ),
                )
            )
        )
        self.audit = memory_reconciliation.build_audit(
            principal_id=str(self.principal), runtime_profile_id=self.runtime, rows=self.rows, now=NOW
        )
        record_reconciliation_audit(self.db_path, self.audit)
        return self

    async def close(self) -> None:
        memory.configure_memory_authority(None)
        await self.store.close()

    def sealed_all_approved(self) -> dict[str, Any]:
        review = memory_reconciliation.build_review_template(self.audit)
        for decision in review["decisions"]:
            decision["disposition"] = "approve"
        return memory_reconciliation.seal_review(self.audit, review, reviewer="Owner")

    def triage(self) -> WorkshopMemoryReconciliationTriageService:
        return WorkshopMemoryReconciliationTriageService(
            self.store, db_path=self.db_path, runtime_pool=cast(WorkshopRuntimePool, object())
        )


def _five_facts() -> list[MemoryResult]:
    # Legacy facts without source receipts each become their own audit
    # candidate, so the apply has five independent steps.
    return [_row(f"fact-{index}", f"Alice keeps note number {index}") for index in range(1, 6)]


def _count(db_path: Path, query: str, *args: object) -> int:
    connection = sqlite3.connect(db_path)
    try:
        return int(connection.execute(query, args).fetchone()[0])
    finally:
        connection.close()


def _created_facts(db_path: Path) -> int:
    return _count(db_path, "SELECT COUNT(*) FROM event_log WHERE event_type = 'memory_fact.recorded'")


# ── Resume ───────────────────────────────────────────────────────────


async def test_a_failed_apply_resumes_without_repeating_finished_steps(tmp_path: Path, monkeypatch) -> None:
    owner = await _Owner(tmp_path, _five_facts()).start(monkeypatch)
    try:
        sealed = owner.sealed_all_approved()
        original_create = MemoryFactLifecycleService.create
        calls = {"count": 0}

        async def failing_third(self, *args, **kwargs):
            calls["count"] += 1
            if calls["count"] == 3:
                raise RuntimeError("store unavailable")
            return await original_create(self, *args, **kwargs)

        monkeypatch.setattr(MemoryFactLifecycleService, "create", failing_third)
        with pytest.raises(RuntimeError, match="store unavailable"):
            await memory_reconciliation.apply_review(db_path=owner.db_path, audit=owner.audit, review=sealed)

        audit_id = owner.audit["audit_id"]
        assert _count(owner.db_path, "SELECT COUNT(*) FROM memory_reconciliation_apply_progress") == 2
        connection = sqlite3.connect(owner.db_path)
        attempts, last_error, completed = connection.execute(
            "SELECT attempts, last_error, completed_at FROM memory_reconciliation_apply_runs WHERE audit_id = ?",
            (audit_id,),
        ).fetchone()
        connection.close()
        assert (attempts, last_error, completed) == (1, "RuntimeError", None)
        assert _created_facts(owner.db_path) == 2

        # The two finished facts were rewritten in place; the retry does not
        # read them for drift and does not write them again.
        monkeypatch.setattr(MemoryFactLifecycleService, "create", original_create)
        receipt = await memory_reconciliation.apply_review(db_path=owner.db_path, audit=owner.audit, review=sealed)

        assert _created_facts(owner.db_path) == 5
        assert [entry["candidate_id"] for entry in receipt["applied"]] == [
            decision["candidate_id"] for decision in sealed["decisions"]
        ]
        assert (
            _count(
                owner.db_path,
                "SELECT COUNT(*) FROM memory_reconciliation_apply_runs WHERE completed_at IS NOT NULL AND attempts = 2",
            )
            == 1
        )
    finally:
        await owner.close()


async def test_a_crash_before_progress_is_recorded_replays_instead_of_repeating(tmp_path: Path, monkeypatch) -> None:
    owner = await _Owner(tmp_path, _five_facts()[:2]).start(monkeypatch)
    try:
        sealed = owner.sealed_all_approved()
        original = memory_reconciliation._record_apply_progress

        async def crash_once(*_args, **_kwargs):
            raise RuntimeError("process stopped")

        monkeypatch.setattr(memory_reconciliation, "_record_apply_progress", crash_once)
        with pytest.raises(RuntimeError):
            await memory_reconciliation.apply_review(db_path=owner.db_path, audit=owner.audit, review=sealed)
        assert _created_facts(owner.db_path) == 1

        monkeypatch.setattr(memory_reconciliation, "_record_apply_progress", original)
        await memory_reconciliation.apply_review(db_path=owner.db_path, audit=owner.audit, review=sealed)

        # The first fact's writes replayed under the same keys.
        assert _created_facts(owner.db_path) == 2
    finally:
        await owner.close()


async def test_an_unfinished_run_of_different_decisions_is_refused(tmp_path: Path, monkeypatch) -> None:
    owner = await _Owner(tmp_path, _five_facts()[:2]).start(monkeypatch)
    try:
        sealed = owner.sealed_all_approved()
        monkeypatch.setattr(
            MemoryFactLifecycleService, "create", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("down"))
        )
        with pytest.raises(RuntimeError):
            await memory_reconciliation.apply_review(db_path=owner.db_path, audit=owner.audit, review=sealed)
        review = memory_reconciliation.build_review_template(owner.audit)
        review["decisions"][0]["disposition"] = "approve"
        review["decisions"][1]["disposition"] = "reject"
        different = memory_reconciliation.seal_review(owner.audit, review, reviewer="Owner")

        with pytest.raises(memory_reconciliation.MemoryReconciliationApplyInProgress):
            await memory_reconciliation.apply_review(db_path=owner.db_path, audit=owner.audit, review=different)
    finally:
        await owner.close()


async def test_decisions_are_frozen_while_an_apply_is_unfinished(tmp_path: Path, monkeypatch) -> None:
    owner = await _Owner(tmp_path, _five_facts()[:2]).start(monkeypatch)
    try:
        triage = owner.triage()
        summary = await triage.latest(owner.principal)
        assert summary is not None
        await owner.store.connection.execute(
            "INSERT INTO memory_reconciliation_apply_runs ("
            "audit_id, principal_id, runtime_profile_id, review_sha256, receipt_id, started_at"
            ") VALUES (?, ?, ?, ?, ?, ?)",
            (summary.plan_id, str(owner.principal), owner.runtime, "a" * 64, "mrr_x", NOW.isoformat()),
        )
        raw_audit_id = owner.audit["audit_id"]
        await owner.store.connection.execute(
            "INSERT INTO memory_reconciliation_apply_runs ("
            "audit_id, principal_id, runtime_profile_id, review_sha256, receipt_id, started_at"
            ") VALUES (?, ?, ?, ?, ?, ?)",
            (raw_audit_id, str(owner.principal), owner.runtime, "a" * 64, "mrr_y", NOW.isoformat()),
        )
        await owner.store.connection.commit()
        page = await triage.groups(owner.principal, summary.plan_id)
        group = page.groups[0]

        with pytest.raises(MemoryReconciliationReviewConflict, match="unfinished"):
            await triage.decide_group(
                owner.principal,
                summary.plan_id,
                group["group_id"],
                disposition="reject",
                action={"kind": "manual_edit_required"},
                operator_note="",
                expected_state_version=0,
                allowed_project_ids=frozenset(),
                client_operation_id="decide-frozen",
            )
        preview = await triage.preview_safe_approval(
            owner.principal, summary.plan_id, group_ids=None, expected_review_version=summary.review_version
        )
        with pytest.raises(MemoryReconciliationReviewConflict, match="unfinished"):
            await triage.approve_safe(
                owner.principal,
                summary.plan_id,
                group_ids=None,
                expected_review_version=summary.review_version,
                preview_sha256=preview["preview_sha256"],
                operator_note="",
                client_operation_id="approve-frozen",
            )
        raw = WorkshopMemoryReconciliationReviewService(owner.store, db_path=owner.db_path)
        candidate = owner.audit["candidates"][0]
        with pytest.raises(MemoryReconciliationReviewConflict, match="unfinished"):
            await raw.decide(
                owner.principal,
                raw_audit_id,
                candidate["candidate_id"],
                disposition="reject",
                action=candidate["proposed_action"],
                operator_note="",
                expected_state_version=0,
                client_operation_id="raw-frozen",
            )
    finally:
        await owner.close()


# ── Drift on covered rows only ───────────────────────────────────────


async def test_unrelated_writes_do_not_block_apply(tmp_path: Path, monkeypatch) -> None:
    owner = await _Owner(tmp_path, _five_facts()[:2]).start(monkeypatch)
    try:
        # A fact written after the audit, the way extraction writes one.
        owner.provider.add(
            "A new canonical fact",
            user_id=str(owner.principal),
            infer=False,
            metadata={"source": "extracted", memory.WORKSHOP_PRINCIPAL_ID_KEY: str(owner.principal)},
        )

        receipt = await memory_reconciliation.apply_review(
            db_path=owner.db_path, audit=owner.audit, review=owner.sealed_all_approved()
        )

        assert len(receipt["applied"]) == 2
    finally:
        await owner.close()


@pytest.mark.parametrize("change", ["edited", "deleted"])
async def test_a_reviewed_row_that_changed_blocks_apply(tmp_path: Path, monkeypatch, change: str) -> None:
    owner = await _Owner(tmp_path, _five_facts()[:2]).start(monkeypatch)
    try:
        if change == "edited":
            owner.provider.rows["fact-2"]["memory"] = "Alice keeps a different note"
        else:
            del owner.provider.rows["fact-2"]

        with pytest.raises(memory_reconciliation.MemoryReconciliationDrift):
            await memory_reconciliation.apply_review(
                db_path=owner.db_path, audit=owner.audit, review=owner.sealed_all_approved()
            )

        assert _created_facts(owner.db_path) == 0
    finally:
        await owner.close()


async def test_a_read_error_fails_the_apply_and_a_retry_finishes_it(tmp_path: Path, monkeypatch) -> None:
    owner = await _Owner(tmp_path, _five_facts()[:2]).start(monkeypatch)
    try:
        sealed = owner.sealed_all_approved()
        original_get = owner.provider.get

        def unreadable(**_kwargs):
            raise RuntimeError("store locked")

        owner.provider.get = unreadable
        with pytest.raises(memory.LifecycleProjectionReadError):
            await memory_reconciliation.apply_review(db_path=owner.db_path, audit=owner.audit, review=sealed)
        owner.provider.get = original_get

        receipt = await memory_reconciliation.apply_review(db_path=owner.db_path, audit=owner.audit, review=sealed)

        assert len(receipt["applied"]) == 2
    finally:
        await owner.close()


# ── Carried decisions ────────────────────────────────────────────────


async def _applied_plan(owner: _Owner, *, reject: set[str], defer: set[str]) -> None:
    """Apply a triage plan in which the named rows' groups were rejected or deferred."""
    triage = owner.triage()
    summary = await triage.latest(owner.principal)
    assert summary is not None
    page = await triage.groups(owner.principal, summary.plan_id)
    for group in page.groups:
        ids = {item["memory_id"] for item in group["evidence"]}
        disposition = "reject" if ids & reject else "defer" if ids & defer else None
        if disposition is not None:
            await triage.decide_group(
                owner.principal,
                summary.plan_id,
                group["group_id"],
                disposition=disposition,
                action={"kind": "manual_edit_required"},
                operator_note=f"{disposition} for now",
                expected_state_version=group["decision"]["state_version"],
                allowed_project_ids=frozenset(),
                client_operation_id=f"decide-{group['group_id']}",
            )
    summary = await triage.latest(owner.principal)
    assert summary is not None
    version = await _approve_safe_groups(triage, owner.principal, summary.plan_id, summary.review_version)
    await triage.apply(
        owner.principal,
        summary.plan_id,
        expected_review_version=version,
        allowed_project_ids=frozenset(),
        client_operation_id="apply-carry",
    )


async def test_rejects_stay_out_of_the_next_audit_until_the_row_changes(tmp_path: Path, monkeypatch) -> None:
    rows = [
        _episode("ep-reject", "Deployed the fix", complete=False),
        _episode("ep-keep", "Repaired the deploy script", complete=True),
    ]
    owner = await _Owner(tmp_path, rows).start(monkeypatch)
    try:
        await _applied_plan(owner, reject={"ep-reject"}, defer=set())
        prior = prior_reconciliation_row_dispositions(
            owner.db_path, principal_id=str(owner.principal), runtime_profile_id=owner.runtime
        )
        assert prior["ep-reject"].disposition == "reject"
        rejected = {key: value.review_state for key, value in prior.items() if value.disposition == "reject"}
        unchanged = [row for row in owner.rows if row.id == "ep-reject"]

        next_audit = memory_reconciliation.build_audit(
            principal_id=str(owner.principal),
            runtime_profile_id=owner.runtime,
            rows=unchanged,
            prior_rejected_rows=rejected,
            now=NOW,
        )
        edited = memory_reconciliation.build_audit(
            principal_id=str(owner.principal),
            runtime_profile_id=owner.runtime,
            rows=[replace(unchanged[0], text="Deployed the fix, then rolled it back")],
            prior_rejected_rows=rejected,
            now=NOW,
        )

        assert next_audit["suppressed_rejected_rows"] == 1 and next_audit["candidates"] == []
        assert edited["suppressed_rejected_rows"] == 0 and edited["candidates"]
    finally:
        await owner.close()


async def test_a_deferral_comes_back_with_the_earlier_note(tmp_path: Path, monkeypatch) -> None:
    rows = [
        _episode("ep-defer", "Deployed the fix", complete=False),
        _episode("ep-keep", "Repaired the deploy script", complete=True),
    ]
    owner = await _Owner(tmp_path, rows).start(monkeypatch)
    try:
        await _applied_plan(owner, reject=set(), defer={"ep-defer"})
        later_rows = [row for row in owner.rows if row.id == "ep-defer"]
        later = memory_reconciliation.build_audit(
            principal_id=str(owner.principal),
            runtime_profile_id=owner.runtime,
            rows=later_rows,
            now=NOW.replace(day=NOW.day + 1),
        )
        record_reconciliation_audit(owner.db_path, later)

        summary = await owner.triage().latest(owner.principal)
        assert summary is not None and summary.plan_id != ""
        page = await owner.triage().groups(owner.principal, summary.plan_id)

        (group,) = page.groups
        assert group["classification"] == "prior_review"
        (earlier,) = group["prior_review_evidence"]
        assert (earlier["disposition"], earlier["operator_note"]) == ("defer", "defer for now")
    finally:
        await owner.close()


# ── Summary counts and the census ────────────────────────────────────


async def test_apply_reports_census_and_quarantine_counts(tmp_path: Path, monkeypatch) -> None:
    rows = [
        _episode("ep-reject", "Deployed the fix", complete=False),
        _episode("ep-keep", "Repaired the deploy script", complete=True),
        _row("fact-1", "Alice prefers direct answers"),
    ]
    owner = await _Owner(tmp_path, rows).start(monkeypatch)
    try:
        await _applied_plan(owner, reject={"ep-reject"}, defer=set())
        connection = sqlite3.connect(owner.db_path)
        receipt_json = connection.execute("SELECT receipt_json FROM memory_reconciliation_triage_plans").fetchone()[0]
        census = connection.execute(
            "SELECT legacy_rows, absorbed, rejected, unclassified FROM memory_legacy_census"
        ).fetchone()
        connection.close()
        import json

        summary = json.loads(receipt_json)["summary"]

        # The fact was adopted in place (no longer a legacy row); both
        # episode rows stay legacy, one absorbed by its canonical episode
        # and one rejected, so nothing is left unclassified.
        assert census == (2, 1, 1, 0)
        assert summary["still_unresolved"] == 0
        assert summary["canonicalized_in_quarantine"] == 0
    finally:
        await owner.close()


def test_census_counts_only_present_legacy_rows(tmp_path: Path) -> None:
    import asyncio

    async def prepare() -> None:
        store = await WorkshopEventStore.open(tmp_path / "kai.db")
        await store.close()

    asyncio.run(prepare())
    rows = [
        _row("legacy-1", "Unsettled"),
        replace(_row("canonical-1", "Current"), metadata={"canonical_memory_claim_id": "mcl_1"}),
    ]

    census = count_legacy_census(tmp_path / "kai.db", rows, principal_id="prn_x", runtime_profile_id="rtp_x")

    assert (census.legacy_rows, census.absorbed, census.rejected, census.unclassified) == (1, 0, 0, 1)
    assert _count(tmp_path / "kai.db", "SELECT unclassified FROM memory_legacy_census") == 1
