"""
Legacy episode reconciliation stays within the canonical episode schema.

A legacy episode saved as plain text has no outcome quality and no
actors, and canonical episodes require both. These tests cover every
place that has to notice: the triage policy (such an episode is never
offered for bulk approval), decision saving and apply (approving one is
refused before anything is written), and moving a plan made by the older
policy forward without losing the operator's other decisions.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest

from kai import memory, memory_reconciliation
from kai import memory_reconciliation_triage as triage_policy
from kai.memory import MemoryResult
from kai.memory_reconciliation_triage import build_triage_plan
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.domain import AgentId, ChannelId, PrincipalId
from kai.workshop.execution_state import WorkshopExecutionStateNamespace, WorkshopExecutionStateRegistry
from kai.workshop.memory_reconciliation_review import (
    MemoryReconciliationReviewValidationError,
    record_reconciliation_audit,
)
from kai.workshop.memory_reconciliation_triage import WorkshopMemoryReconciliationTriageService
from kai.workshop.runtime_pool import WorkshopRuntimePool
from kai.workshop.store import WorkshopEventStore
from tests.memory_fixtures import FakeMem0
from tests.test_memory_reconciliation_triage import NOW, _row
from tests.workshop_profiles import profile_id

_COMPLETE_EPISODE = {
    "source": "episode",
    "scope": "global",
    "goal": "Repair the deploy script",
    "context": "The deploy failed on a missing path.",
    "approach": "Traced the path and fixed the default.",
    "outcome": "The deploy succeeded.",
    "outcome_quality": "success",
    "actors": ["Owner", "Kai"],
    "tags": ["deploy"],
}


def _episode(memory_id: str, text: str, *, complete: bool) -> MemoryResult:
    metadata = dict(_COMPLETE_EPISODE) if complete else {"source": "episode", "scope": "global"}
    return MemoryResult(memory_id, text, 0.0, "episode", metadata, NOW.isoformat(), NOW.isoformat())


def _plan(rows: list[MemoryResult]) -> dict[str, Any]:
    audit = memory_reconciliation.build_audit(
        principal_id="prn_test", runtime_profile_id="rtp_test", rows=rows, now=NOW
    )
    return build_triage_plan(audit)


def _by_memory(plan: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["memory_id"]): group for group in plan["groups"] for item in group["evidence"]}


# ── Schema checks without writing ────────────────────────────────────


def test_plain_text_episode_fails_the_schema_and_a_complete_one_passes() -> None:
    plan = _plan(
        [_episode("ep-plain", "Deployed the fix", complete=False), _episode("ep-full", "Repaired", complete=True)]
    )
    groups = _by_memory(plan)

    plain = memory_reconciliation.action_schema_problem(
        groups["ep-plain"]["evidence"], {"kind": "record_episode_chain"}
    )
    full = memory_reconciliation.action_schema_problem(groups["ep-full"]["evidence"], {"kind": "record_episode_chain"})

    assert plain is not None and "outcome_quality" in plain
    assert full is None


# ── Triage policy ────────────────────────────────────────────────────


def test_incomplete_episode_goes_to_review_and_complete_one_stays_bulk_eligible() -> None:
    plan = _plan(
        [_episode("ep-plain", "Deployed the fix", complete=False), _episode("ep-full", "Repaired", complete=True)]
    )
    groups = _by_memory(plan)

    incomplete = groups["ep-plain"]
    assert incomplete["classification"] == "incomplete_episode"
    assert (incomplete["deterministic"], incomplete["bulk_eligible"]) == (False, False)
    assert incomplete["action"] == {"kind": "manual_edit_required"}
    assert incomplete["missing_fields"] == ["actors", "outcome_quality"]
    complete = groups["ep-full"]
    assert complete["classification"] == "legacy_episode_history"
    assert (complete["deterministic"], complete["bulk_eligible"]) == (True, True)
    assert "missing_fields" not in complete


def test_a_failing_fact_is_split_out_of_its_bulk_batch() -> None:
    plan = _plan([_row("fact-long", "x " * 9000), _row("fact-ok", "Alice prefers direct answers")])
    groups = _by_memory(plan)

    # The failing fact leaves its per-scope batch; the rest of the batch
    # stays bulk-eligible.
    assert groups["fact-long"]["deterministic"] is False
    assert "rejected by canonical memory" in groups["fact-long"]["rationale"]
    assert groups["fact-ok"]["deterministic"] is True
    assert [item["memory_id"] for item in groups["fact-ok"]["evidence"]] == ["fact-ok"]


def test_every_deterministic_group_passes_the_schema() -> None:
    rows = [
        _episode("ep-plain", "Deployed the fix", complete=False),
        _episode("ep-full", "Repaired", complete=True),
        _row("fact-1", "Alice prefers direct answers"),
        _row("fact-2", "Alice prefers direct answers"),
        _row("fact-3", "Alice uses dark mode", valid_until="2026-01-01T00:00:00+00:00"),
        _row("fact-long", "x " * 9000),
    ]

    plan = _plan(rows)

    deterministic = [group for group in plan["groups"] if group["deterministic"]]
    assert deterministic
    assert all(
        memory_reconciliation.action_schema_problem(group["evidence"], group["action"]) is None
        for group in deterministic
    )


# ── Service: decisions, apply preflight, and moving a plan forward ───


async def _service(tmp_path: Path, rows: list[MemoryResult], *, stamp_owner: bool = False):
    db_path = tmp_path / "kai.db"
    store = await WorkshopEventStore.open(db_path)
    await bootstrap_default_workshop(
        store,
        (BootstrapHuman("Alice", "admin", "telegram", "101", "101", profile_id(101)),),
    )
    async with store.connection.execute(
        "SELECT principal_id FROM external_identities WHERE provider = 'telegram' AND external_subject = '101'"
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    principal = PrincipalId(str(row[0]))
    if stamp_owner:
        # Production legacy rows carry their owner's principal stamp, which
        # the owner-verified reads require; the audit sees the same rows.
        rows[:] = [
            replace(item, metadata={**item.metadata, memory.WORKSHOP_PRINCIPAL_ID_KEY: str(principal)}) for item in rows
        ]
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
    return store, service, principal


_ROWS = [
    _episode("ep-plain", "Deployed the fix", complete=False),
    _episode("ep-full", "Repaired the deploy script", complete=True),
    _row("fact-1", "Alice prefers direct answers"),
]


async def _group_rows(store, plan_id: str) -> dict[str, tuple[str, str]]:
    async with store.connection.execute(
        "SELECT group_id, classification, disposition FROM memory_reconciliation_triage_groups WHERE plan_id = ?",
        (plan_id,),
    ) as cursor:
        return {str(row[0]): (str(row[1]), str(row[2])) for row in await cursor.fetchall()}


async def test_approving_an_incomplete_episode_is_refused_and_reject_saves(tmp_path: Path) -> None:
    store, service, principal = await _service(tmp_path, _ROWS)
    try:
        summary = await service.latest(principal)
        assert summary is not None
        page = await service.groups(principal, summary.plan_id, exceptions_only=True)
        (incomplete,) = [group for group in page.groups if group["classification"] == "incomplete_episode"]
        assert incomplete["missing_fields"] == ["actors", "outcome_quality"]

        with pytest.raises(MemoryReconciliationReviewValidationError, match="rejected by canonical memory"):
            await service.decide_group(
                principal,
                summary.plan_id,
                incomplete["group_id"],
                disposition="approve",
                action={"kind": "record_episode_chain"},
                operator_note="",
                expected_state_version=0,
                allowed_project_ids=frozenset(),
                client_operation_id="decide-approve",
            )
        await service.decide_group(
            principal,
            summary.plan_id,
            incomplete["group_id"],
            disposition="reject",
            action={"kind": "manual_edit_required"},
            operator_note="Not worth keeping.",
            expected_state_version=0,
            allowed_project_ids=frozenset(),
            client_operation_id="decide-reject",
        )
        assert (await _group_rows(store, summary.plan_id))[incomplete["group_id"]][1] == "reject"
    finally:
        await store.close()


async def _approve_safe_groups(service, principal, plan_id: str, review_version: int) -> int:
    """Bulk-approve every pending safe group, one group at a time (each batch is one action kind)."""
    page = await service.groups(principal, plan_id)
    for group in page.groups:
        if not group["bulk_eligible"] or group["decision"]["disposition"] != "pending":
            continue
        preview = await service.preview_safe_approval(
            principal, plan_id, group_ids=[group["group_id"]], expected_review_version=review_version
        )
        approved = await service.approve_safe(
            principal,
            plan_id,
            group_ids=[group["group_id"]],
            expected_review_version=review_version,
            preview_sha256=preview["preview_sha256"],
            operator_note="Safe batch.",
            client_operation_id=f"approve-{group['group_id']}",
        )
        review_version = approved["review_version"]
    return review_version


async def _v1_plan_with_everything_approved(store, service, principal, monkeypatch) -> str:
    """Store a plan the way the older policy made it, then bulk-approve it, as in the installed plan."""
    with monkeypatch.context() as older:
        older.setattr(triage_policy, "POLICY_VERSION", "legacy_triage_v1")
        older.setattr(triage_policy, "_schema_checked_groups", lambda group: [group])
        summary = await service.latest(principal)
        assert summary is not None
        await _approve_safe_groups(service, principal, summary.plan_id, summary.review_version)
    return summary.plan_id


async def test_an_open_older_plan_moves_forward_keeping_unchanged_decisions(tmp_path: Path, monkeypatch) -> None:
    store, service, principal = await _service(tmp_path, _ROWS)
    try:
        old_plan_id = await _v1_plan_with_everything_approved(store, service, principal, monkeypatch)
        before = await _group_rows(store, old_plan_id)
        assert {disposition for _classification, disposition in before.values()} == {"approve"}

        moved = await service.latest(principal)
        again = await service.latest(principal)

        assert moved is not None and again is not None
        assert moved.plan_id != old_plan_id and again.plan_id == moved.plan_id
        assert again.review_version == moved.review_version  # a second load changes nothing
        after = await _group_rows(store, moved.plan_id)
        kept = {group_id: state for group_id, state in after.items() if group_id in before}
        assert len(kept) == len(before) - 1 and all(state[1] == "approve" for state in kept.values())
        (reopened,) = [state for group_id, state in after.items() if group_id not in before]
        assert reopened == ("incomplete_episode", "pending")
        async with store.connection.execute(
            "SELECT response_json FROM memory_reconciliation_operations WHERE client_operation_id = ?",
            (f"replan:{old_plan_id}",),
        ) as cursor:
            (record_json,) = await cursor.fetchone()
        record = json.loads(record_json)
        assert record["kept"] == len(kept)
        assert [item["disposition"] for item in record["dropped"]] == ["approve"]
        async with store.connection.execute("PRAGMA foreign_key_check") as cursor:
            assert await cursor.fetchall() == []
        # The replan record names the plan it produced, so install status
        # accounts for it instead of reporting a replay gap.
        from kai.workshop.diagnostics import workshop_memory_reconciliation_status

        assert "replay gaps=0;" in workshop_memory_reconciliation_status(tmp_path / "kai.db")
    finally:
        await store.close()


async def test_an_applied_older_plan_is_left_as_history(tmp_path: Path, monkeypatch) -> None:
    store, service, principal = await _service(tmp_path, _ROWS)
    try:
        old_plan_id = await _v1_plan_with_everything_approved(store, service, principal, monkeypatch)
        await store.connection.execute(
            "UPDATE memory_reconciliation_triage_plans SET status = 'applied' WHERE plan_id = ?", (old_plan_id,)
        )
        await store.connection.commit()

        summary = await service.latest(principal)

        assert summary is not None and summary.plan_id == old_plan_id
        assert {state[0] for state in (await _group_rows(store, old_plan_id)).values()} >= {"legacy_episode_history"}
    finally:
        await store.close()


async def test_apply_refuses_a_stored_approval_that_would_fail_the_schema(tmp_path: Path, monkeypatch) -> None:
    store, service, principal = await _service(tmp_path, _ROWS)
    try:
        summary = await service.latest(principal)
        assert summary is not None
        page = await service.groups(principal, summary.plan_id, exceptions_only=True)
        (incomplete,) = [group for group in page.groups if group["classification"] == "incomplete_episode"]
        # An approval stored before the check existed, written directly.
        await store.connection.execute(
            "UPDATE memory_reconciliation_triage_groups SET disposition = 'approve', action_json = ? "
            "WHERE group_id = ?",
            (json.dumps({"kind": "record_episode_chain"}), incomplete["group_id"]),
        )
        await store.connection.execute(
            "UPDATE memory_reconciliation_triage_groups SET disposition = 'approve' WHERE disposition = 'pending'"
        )
        await store.connection.commit()
        monkeypatch.setattr(memory, "get_all_for_lifecycle_projection", lambda **_kwargs: _ROWS)
        async with store.connection.execute("SELECT COUNT(*) FROM event_log") as cursor:
            (events_before,) = await cursor.fetchone()

        with pytest.raises(MemoryReconciliationReviewValidationError, match="rejected by canonical memory"):
            await service.apply(
                principal,
                summary.plan_id,
                expected_review_version=summary.review_version,
                allowed_project_ids=frozenset(),
                client_operation_id="apply-bad",
            )

        async with store.connection.execute("SELECT COUNT(*) FROM event_log") as cursor:
            assert (await cursor.fetchone())[0] == events_before
    finally:
        await store.close()


def _stored(principal: PrincipalId, rows: list[MemoryResult]) -> FakeMem0:
    """A vector store holding the legacy rows exactly as the audit saw them."""
    provider = FakeMem0()
    for row in rows:
        provider.rows[row.id] = {
            "id": row.id,
            "memory": row.text,
            "metadata": dict(row.metadata),
            "user_id": str(principal),
            "created_at": row.created_at,
            "updated_at": row.updated_at,
        }
    return provider


async def test_rejecting_incomplete_episodes_lets_every_approved_episode_apply(tmp_path: Path, monkeypatch) -> None:
    rows = list(_ROWS)
    store, service, principal = await _service(tmp_path, rows, stamp_owner=True)
    monkeypatch.setattr(memory, "_memory", _stored(principal, rows))
    memory.configure_memory_authority(
        WorkshopExecutionStateRegistry(
            (
                WorkshopExecutionStateNamespace(
                    principal_id=principal,
                    channel_id=ChannelId("chn_" + "1" * 32),
                    agent_id=AgentId("agt_" + "1" * 32),
                    runtime_profile_id=profile_id(101),
                    legacy_runtime_key=101,
                ),
            )
        )
    )
    try:
        summary = await service.latest(principal)
        assert summary is not None
        page = await service.groups(principal, summary.plan_id, exceptions_only=True)
        (incomplete,) = [group for group in page.groups if group["classification"] == "incomplete_episode"]
        await service.decide_group(
            principal,
            summary.plan_id,
            incomplete["group_id"],
            disposition="reject",
            action={"kind": "manual_edit_required"},
            operator_note="",
            expected_state_version=0,
            allowed_project_ids=frozenset(),
            client_operation_id="decide-reject-before-apply",
        )
        review_version = await _approve_safe_groups(service, principal, summary.plan_id, 1)
        applied = await service.apply(
            principal,
            summary.plan_id,
            expected_review_version=review_version,
            allowed_project_ids=frozenset(),
            client_operation_id="apply-good",
        )

        assert applied["summary"]["rejected"] == 1
        async with store.connection.execute("SELECT COUNT(*) FROM memory_episodes") as cursor:
            assert (await cursor.fetchone())[0] == 1  # the complete episode; the rejected one is not recorded
    finally:
        memory.configure_memory_authority(None)
        await store.close()
