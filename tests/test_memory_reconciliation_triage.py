"""Grouped, exception-only legacy-memory triage."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from kai import memory_reconciliation
from kai.memory import MemoryResult
from kai.memory_reconciliation_triage import build_triage_plan, validate_triage_plan
from kai.oneshot import OneShotResult
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.domain import PrincipalId
from kai.workshop.memory_reconciliation_review import (
    MemoryReconciliationReviewAccessDenied,
    MemoryReconciliationReviewValidationError,
    record_reconciliation_audit,
)
from kai.workshop.memory_reconciliation_triage import (
    WorkshopMemoryReconciliationTriageService,
    _approved_outcome_summary,
)
from kai.workshop.runtime_pool import WorkshopRuntimePool
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id

NOW = datetime(2026, 9, 22, 12, 0, tzinfo=UTC)


def _row(memory_id: str, text: str, **metadata: object) -> MemoryResult:
    created_at = str(metadata.pop("created_at", NOW.isoformat()))
    updated_at = str(metadata.pop("updated_at", created_at))
    return MemoryResult(
        memory_id,
        text,
        0.0,
        "fact",
        {"source": "extracted", "scope": "global", "confidence": 0.8, **metadata},
        created_at,
        updated_at,
    )


def _audit(rows: list[MemoryResult]) -> dict[str, Any]:
    return memory_reconciliation.build_audit(
        principal_id="prn_test",
        runtime_profile_id="rtp_test",
        rows=rows,
        now=NOW,
    )


def test_applied_summary_uses_saved_exception_actions_instead_of_plan_proposals() -> None:
    groups = {
        "adopt": {"evidence": [{"memory_id": "a"}], "resolution": "needs_review"},
        "consolidate": {
            "evidence": [{"memory_id": "b"}, {"memory_id": "c"}],
            "resolution": "needs_review",
        },
        "obsolete": {"evidence": [{"memory_id": "d"}], "resolution": "needs_review"},
        "episode": {"evidence": [{"memory_id": "e"}], "resolution": "needs_review"},
    }
    decision_rows = [
        ("adopt", "approve", json.dumps({"kind": "adopt_as_current"})),
        ("consolidate", "approve", json.dumps({"kind": "adopt_corrected"})),
        ("obsolete", "approve", json.dumps({"kind": "expire_all"})),
        ("episode", "approve", json.dumps({"kind": "record_episode_chain"})),
    ]

    assert _approved_outcome_summary(groups, decision_rows) == {
        "adopted": 2,
        "consolidated": 2,
        "obsolete": 1,
        "operator_admitted": 3,
    }


def test_triage_partitions_large_legacy_corpus_into_safe_batches() -> None:
    rows = [_row(f"mem-{index}", f"Daniel has stable preference number {index}") for index in range(437)]
    plan = build_triage_plan(_audit(rows))
    validate_triage_plan(plan)

    assert plan["memory_count"] == 437
    assert plan["group_count"] == 1
    assert plan["groups"][0]["resolution"] == "adopt"
    assert plan["groups"][0]["bulk_eligible"] is True
    assert len(plan["groups"][0]["evidence"]) == 437


def test_triage_separates_duplicates_expiry_and_uncertain_conflicts() -> None:
    rows = [
        _row("duplicate-1", "Daniel prefers concise answers"),
        _row("duplicate-2", "Daniel prefers concise answers"),
        _row(
            "expired",
            "Daniel is temporarily testing a backend",
            valid_until=(NOW - timedelta(days=1)).isoformat(),
        ),
        _row("positive", "Daniel does use Telegram alerts"),
        _row("negative", "Daniel does not use Telegram alerts"),
    ]
    plan = build_triage_plan(_audit(rows))
    assigned = [item["memory_id"] for group in plan["groups"] for item in group["evidence"]]

    assert sorted(assigned) == sorted(row.id for row in rows)
    assert len(assigned) == len(set(assigned))
    by_resolution = {group["resolution"]: group for group in plan["groups"]}
    assert by_resolution["consolidate"]["deterministic"] is True
    assert by_resolution["obsolete"]["deterministic"] is True
    assert by_resolution["needs_review"]["bulk_eligible"] is False


def test_triage_does_not_auto_consolidate_mixed_validity_or_aged_current_claims() -> None:
    rows = [
        _row(
            "expired",
            "Daniel currently uses this backend",
            created_at=(NOW - timedelta(days=200)).isoformat(),
            valid_until=(NOW - timedelta(days=1)).isoformat(),
        ),
        _row(
            "unbounded",
            "Daniel currently uses this backend",
            created_at=(NOW - timedelta(days=200)).isoformat(),
        ),
    ]

    plan = build_triage_plan(_audit(rows))

    assert plan["group_count"] == 1
    assert plan["groups"][0]["classification"] == "validity_conflict"
    assert plan["groups"][0]["resolution"] == "needs_review"


def test_triage_never_bulk_approves_uncertain_scope() -> None:
    plan = build_triage_plan(_audit([_row("missing-scope", "A fact with unknown scope", scope="")]))

    assert plan["group_count"] == 1
    assert plan["groups"][0]["classification"] == "uncertain_scope"
    assert plan["groups"][0]["bulk_eligible"] is False


def test_triage_never_bulk_approves_a_project_outside_current_authority() -> None:
    plan = build_triage_plan(
        _audit([_row("project-fact", "A project fact", scope="project", project_id="project-private")]),
        allowed_project_ids=frozenset(),
    )

    assert plan["groups"][0]["classification"] == "missing_project_authority"
    assert plan["groups"][0]["bulk_eligible"] is False


@pytest.mark.asyncio
async def test_safe_group_approval_is_preview_bound_replay_safe_and_preserves_raw_decisions(tmp_path: Path) -> None:
    db_path = tmp_path / "kai.db"
    store = await WorkshopEventStore.open(db_path)
    await bootstrap_default_workshop(
        store,
        (BootstrapHuman("Alice", "admin", "telegram", "101", "101", profile_id(101)),),
    )
    async with store.connection.execute(
        "SELECT principal_id FROM external_identities WHERE provider = 'telegram' AND external_subject = '101'"
    ) as cursor:
        principal_row = await cursor.fetchone()
    assert principal_row is not None
    principal = PrincipalId(str(principal_row[0]))
    audit = memory_reconciliation.build_audit(
        principal_id=str(principal),
        runtime_profile_id=str(profile_id(101)),
        rows=[_row("mem-1", "Alice prefers exact summaries"), _row("mem-2", "Alice uses dark mode")],
        now=NOW,
    )
    record_reconciliation_audit(db_path, audit)
    raw_candidate = audit["candidates"][0]
    await store.connection.execute(
        "UPDATE memory_reconciliation_decisions SET disposition = 'defer', state_version = 1 "
        "WHERE audit_id = ? AND candidate_id = ?",
        (audit["audit_id"], raw_candidate["candidate_id"]),
    )
    await store.connection.commit()
    service = WorkshopMemoryReconciliationTriageService(
        store,
        db_path=db_path,
        runtime_pool=cast(WorkshopRuntimePool, object()),
    )

    summary = await service.latest(principal)
    assert summary is not None
    assert summary.group_count == 2
    assert summary.memory_count == 2
    assert summary.exception_groups == 1
    assert summary.pending_deterministic_groups == 1
    exception_page = await service.groups(principal, summary.plan_id, exceptions_only=True)
    assert exception_page.groups[0]["classification"] == "prior_review"
    assert exception_page.groups[0]["proposed_action"] == {"kind": "adopt_as_current"}
    assert exception_page.groups[0]["prior_review_evidence"] == [
        {
            "candidate_id": raw_candidate["candidate_id"],
            "disposition": "defer",
            "memory_ids": [raw_candidate["evidence"][0]["memory_id"]],
            "operator_note": "",
            "state_version": 1,
        }
    ]
    preview = await service.preview_safe_approval(
        principal,
        summary.plan_id,
        group_ids=None,
        expected_review_version=0,
    )
    approved = await service.approve_safe(
        principal,
        summary.plan_id,
        group_ids=None,
        expected_review_version=0,
        preview_sha256=preview["preview_sha256"],
        operator_note="Reviewed grouped evidence.",
        client_operation_id="safe-approve-1",
    )
    replay = await service.approve_safe(
        principal,
        summary.plan_id,
        group_ids=None,
        expected_review_version=0,
        preview_sha256=preview["preview_sha256"],
        operator_note="Reviewed grouped evidence.",
        client_operation_id="safe-approve-1",
    )

    assert approved["replayed"] is False
    assert replay["replayed"] is True
    grouped = await service.decide_group(
        principal,
        summary.plan_id,
        exception_page.groups[0]["group_id"],
        disposition="approve",
        action={"kind": "adopt_as_current"},
        operator_note="The unchanged fact remains current.",
        expected_state_version=0,
        allowed_project_ids=frozenset(),
        client_operation_id="reverse-raw-defer-1",
    )
    assert grouped["disposition"] == "approve"
    grouped_page = await service.groups(principal, summary.plan_id, exceptions_only=True)
    assert grouped_page.groups[0]["decision"]["action"] == {"kind": "adopt_as_current"}
    assert grouped_page.groups[0]["decision"]["disposition"] == "approve"
    async with store.connection.execute(
        "SELECT disposition FROM memory_reconciliation_decisions WHERE audit_id = ? AND candidate_id = ?",
        (audit["audit_id"], raw_candidate["candidate_id"]),
    ) as cursor:
        raw_state = await cursor.fetchone()
    assert raw_state is not None and raw_state[0] == "defer"
    await store.close()


@pytest.mark.asyncio
async def test_group_decision_rejects_unchanged_adoption_for_ambiguous_facts(tmp_path: Path) -> None:
    db_path = tmp_path / "kai.db"
    store = await WorkshopEventStore.open(db_path)
    await bootstrap_default_workshop(
        store,
        (BootstrapHuman("Alice", "admin", "telegram", "101", "101", profile_id(101)),),
    )
    async with store.connection.execute(
        "SELECT principal_id FROM external_identities WHERE provider = 'telegram' AND external_subject = '101'"
    ) as cursor:
        principal_row = await cursor.fetchone()
    assert principal_row is not None
    principal = PrincipalId(str(principal_row[0]))
    audit = memory_reconciliation.build_audit(
        principal_id=str(principal),
        runtime_profile_id=str(profile_id(101)),
        rows=[
            _row("positive", "Alice does use Telegram alerts"),
            _row("negative", "Alice does not use Telegram alerts"),
        ],
        now=NOW,
    )
    record_reconciliation_audit(db_path, audit)
    service = WorkshopMemoryReconciliationTriageService(
        store,
        db_path=db_path,
        runtime_pool=cast(WorkshopRuntimePool, object()),
    )
    summary = await service.latest(principal)
    assert summary is not None
    page = await service.groups(principal, summary.plan_id, exceptions_only=True)

    with pytest.raises(MemoryReconciliationReviewValidationError, match="authorized action"):
        await service.decide_group(
            principal,
            summary.plan_id,
            page.groups[0]["group_id"],
            disposition="approve",
            action={"kind": "adopt_as_current"},
            operator_note="",
            expected_state_version=0,
            allowed_project_ids=frozenset(),
            client_operation_id="unsafe-unchanged-adoption",
        )
    project_action = {
        "kind": "adopt_corrected",
        "source_memory_id": "positive",
        "replacement": {
            "content": "Alice uses Telegram alerts.",
            "scope_kind": "project",
            "scope_key": "private-project",
            "confidence": 0.8,
        },
    }
    with pytest.raises(MemoryReconciliationReviewAccessDenied, match="outside current authority"):
        await service.decide_group(
            principal,
            summary.plan_id,
            page.groups[0]["group_id"],
            disposition="approve",
            action=project_action,
            operator_note="",
            expected_state_version=0,
            allowed_project_ids=frozenset(),
            client_operation_id="unauthorized-corrected-scope",
        )
    corrected = await service.decide_group(
        principal,
        summary.plan_id,
        page.groups[0]["group_id"],
        disposition="approve",
        action={
            "kind": "adopt_corrected",
            "source_memory_id": "positive",
            "replacement": {
                "content": "Alice uses Telegram alerts.",
                "scope_kind": "global",
                "scope_key": "",
                "confidence": 0.8,
            },
        },
        operator_note="Resolved the contradiction explicitly.",
        expected_state_version=0,
        allowed_project_ids=frozenset(),
        client_operation_id="resolve-contradiction",
    )
    assert corrected["disposition"] == "approve"
    await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("action_kind", ["adopt_as_current", "expire_all"])
async def test_group_decision_supports_complete_single_fact_outcomes(tmp_path: Path, action_kind: str) -> None:
    db_path = tmp_path / "kai.db"
    store = await WorkshopEventStore.open(db_path)
    await bootstrap_default_workshop(
        store,
        (BootstrapHuman("Alice", "admin", "telegram", "101", "101", profile_id(101)),),
    )
    async with store.connection.execute(
        "SELECT principal_id FROM external_identities WHERE provider = 'telegram' AND external_subject = '101'"
    ) as cursor:
        principal_row = await cursor.fetchone()
    assert principal_row is not None
    principal = PrincipalId(str(principal_row[0]))
    audit = memory_reconciliation.build_audit(
        principal_id=str(principal),
        runtime_profile_id=str(profile_id(101)),
        rows=[
            _row(
                "time-sensitive",
                "Alice currently prefers direct answers",
                created_at=(NOW - timedelta(days=200)).isoformat(),
            )
        ],
        now=NOW,
    )
    record_reconciliation_audit(db_path, audit)
    service = WorkshopMemoryReconciliationTriageService(
        store,
        db_path=db_path,
        runtime_pool=cast(WorkshopRuntimePool, object()),
    )
    summary = await service.latest(principal)
    assert summary is not None
    page = await service.groups(principal, summary.plan_id, exceptions_only=True)

    result = await service.decide_group(
        principal,
        summary.plan_id,
        page.groups[0]["group_id"],
        disposition="approve",
        action={"kind": action_kind},
        operator_note="Explicit operator outcome.",
        expected_state_version=0,
        allowed_project_ids=frozenset(),
        client_operation_id=f"single-fact-{action_kind}",
    )

    assert result["disposition"] == "approve"
    decided = await service.groups(principal, summary.plan_id, exceptions_only=True)
    assert decided.groups[0]["decision"]["action"] == {"kind": action_kind}
    await store.close()


class _FakeRuntimePool:
    def get_backend_provider(self, _runtime: object) -> tuple[str, str]:
        return "codex", "openai"

    def get_role_model(self, _runtime: object, _role: object) -> str:
        return "gpt-5.5"

    def runtime_profile(self, _runtime: object) -> object:
        return SimpleNamespace(os_user="alice")


class _FakeReasoner:
    async def run(self, **kwargs: object) -> OneShotResult:
        prompt = json.loads(str(kwargs["prompt"]))
        recommendations = [
            {
                "group_id": group["group_id"],
                "outcome": "needs_review",
                "confidence": 0.72,
                "rationale": "The two statements conflict and need a human decision.",
                "related_group_ids": [],
            }
            for group in prompt["groups"]
        ]
        return OneShotResult(
            text=json.dumps({"is_error": False, "structured_output": {"recommendations": recommendations}}),
            backend="codex",
            model="gpt-5.5",
        )


@pytest.mark.asyncio
async def test_model_recommendations_are_advisory_provenanced_and_do_not_decide(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "kai.db"
    store = await WorkshopEventStore.open(db_path)
    await bootstrap_default_workshop(
        store,
        (BootstrapHuman("Alice", "admin", "telegram", "101", "101", profile_id(101)),),
    )
    async with store.connection.execute(
        "SELECT principal_id FROM external_identities WHERE provider = 'telegram' AND external_subject = '101'"
    ) as cursor:
        principal_row = await cursor.fetchone()
    assert principal_row is not None
    principal = PrincipalId(str(principal_row[0]))
    audit = memory_reconciliation.build_audit(
        principal_id=str(principal),
        runtime_profile_id=str(profile_id(101)),
        rows=[
            _row("positive", "Alice does use Telegram alerts"),
            _row("negative", "Alice does not use Telegram alerts"),
        ],
        now=NOW,
    )
    record_reconciliation_audit(db_path, audit)
    monkeypatch.setattr(
        "kai.workshop.memory_reconciliation_triage.build_memory_reasoner", lambda *_a, **_k: _FakeReasoner()
    )
    service = WorkshopMemoryReconciliationTriageService(
        store,
        db_path=db_path,
        runtime_pool=cast(WorkshopRuntimePool, _FakeRuntimePool()),
    )
    summary = await service.latest(principal)
    assert summary is not None and summary.exception_groups == 1

    result = await service.recommend(principal, summary.plan_id)
    page = await service.groups(principal, summary.plan_id, exceptions_only=True)

    assert result["recommended"] == 1
    assert page.groups[0]["decision"]["disposition"] == "pending"
    assert page.groups[0]["decision"]["recommendation"] == {
        "backend": "codex",
        "confidence": 0.72,
        "group_id": page.groups[0]["group_id"],
        "input_sha256": page.groups[0]["decision"]["recommendation"]["input_sha256"],
        "model": "gpt-5.5",
        "outcome": "needs_review",
        "output_sha256": page.groups[0]["decision"]["recommendation"]["output_sha256"],
        "prompt_version": "memory_reconciliation_triage_v2",
        "provider": "openai",
        "rationale": "The two statements conflict and need a human decision.",
        "related_group_ids": [],
    }
    await store.close()


@pytest.mark.asyncio
async def test_model_recommendations_advance_past_the_first_hundred_exception_groups(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "kai.db"
    store = await WorkshopEventStore.open(db_path)
    await bootstrap_default_workshop(
        store,
        (BootstrapHuman("Alice", "admin", "telegram", "101", "101", profile_id(101)),),
    )
    async with store.connection.execute(
        "SELECT principal_id FROM external_identities WHERE provider = 'telegram' AND external_subject = '101'"
    ) as cursor:
        principal_row = await cursor.fetchone()
    assert principal_row is not None
    principal = PrincipalId(str(principal_row[0]))
    rows = []
    for index in range(101):
        text = f"Alice does use service group {index} marker {index}"
        rows.extend(
            (
                _row(f"positive-{index}", text),
                _row(f"negative-{index}", text.replace("does use", "does not use")),
            )
        )
    audit = memory_reconciliation.build_audit(
        principal_id=str(principal),
        runtime_profile_id=str(profile_id(101)),
        rows=rows,
        now=NOW,
    )
    record_reconciliation_audit(db_path, audit)
    monkeypatch.setattr(
        "kai.workshop.memory_reconciliation_triage.build_memory_reasoner", lambda *_a, **_k: _FakeReasoner()
    )
    service = WorkshopMemoryReconciliationTriageService(
        store,
        db_path=db_path,
        runtime_pool=cast(WorkshopRuntimePool, _FakeRuntimePool()),
    )
    summary = await service.latest(principal)
    assert summary is not None and summary.exception_groups == 101

    first = await service.recommend(principal, summary.plan_id)
    second = await service.recommend(principal, summary.plan_id)

    assert first == {"plan_id": summary.plan_id, "recommended": 100, "remaining": 1}
    assert second == {"plan_id": summary.plan_id, "recommended": 1, "remaining": 0}
    await store.close()


@pytest.mark.asyncio
async def test_grouped_apply_emits_aggregate_summary_and_is_replay_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "kai.db"
    store = await WorkshopEventStore.open(db_path)
    await bootstrap_default_workshop(
        store,
        (BootstrapHuman("Alice", "admin", "telegram", "101", "101", profile_id(101)),),
    )
    async with store.connection.execute(
        "SELECT principal_id FROM external_identities WHERE provider = 'telegram' AND external_subject = '101'"
    ) as cursor:
        principal_row = await cursor.fetchone()
    assert principal_row is not None
    principal = PrincipalId(str(principal_row[0]))
    rows = [_row("mem-1", "Alice prefers direct answers"), _row("mem-2", "Alice uses dark mode")]
    audit = memory_reconciliation.build_audit(
        principal_id=str(principal),
        runtime_profile_id=str(profile_id(101)),
        rows=rows,
        now=NOW,
    )
    record_reconciliation_audit(db_path, audit)
    service = WorkshopMemoryReconciliationTriageService(
        store,
        db_path=db_path,
        runtime_pool=cast(WorkshopRuntimePool, object()),
    )
    summary = await service.latest(principal)
    assert summary is not None
    preview = await service.preview_safe_approval(
        principal,
        summary.plan_id,
        group_ids=None,
        expected_review_version=0,
    )
    approved = await service.approve_safe(
        principal,
        summary.plan_id,
        group_ids=None,
        expected_review_version=0,
        preview_sha256=preview["preview_sha256"],
        operator_note="Safe adoption batch.",
        client_operation_id="approve-safe-apply",
    )

    async def fake_apply_review(**kwargs: object) -> dict[str, object]:
        review = kwargs["review"]
        assert isinstance(review, dict)
        return {
            "receipt_id": "mrr_grouped",
            "applied_at": NOW.isoformat(),
            "sha256": "d" * 64,
            "decisions": review["decisions"],
            "applied": [],
        }

    monkeypatch.setattr(
        "kai.workshop.memory_reconciliation_triage.memory_reconciliation.apply_review", fake_apply_review
    )
    applied = await service.apply(
        principal,
        summary.plan_id,
        expected_review_version=approved["review_version"],
        allowed_project_ids=frozenset(),
        client_operation_id="apply-grouped",
    )
    replay = await service.apply(
        principal,
        summary.plan_id,
        expected_review_version=approved["review_version"],
        allowed_project_ids=frozenset(),
        client_operation_id="apply-grouped",
    )

    assert applied["summary"] == {
        "adopted": 2,
        "consolidated": 0,
        "obsolete": 0,
        "deferred": 0,
        "rejected": 0,
        "failed": 0,
        "still_unresolved": 0,
        "not_adopted": 0,
        "operator_admitted": 2,
        "canonicalized_in_quarantine": 0,
    }
    assert replay["replayed"] is True
    applied_replay = await service.apply(
        principal,
        summary.plan_id,
        expected_review_version=approved["review_version"],
        allowed_project_ids=frozenset(),
        client_operation_id="apply-grouped-after-completion",
    )
    assert applied_replay["summary"] == applied["summary"]
    assert applied_replay["replayed"] is True
    await store.close()
