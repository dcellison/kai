"""
Consolidating several legacy facts into one canonical fact through triage.

Uses the real Workshop schema, the real triage service, and the real
resumable apply with the storage-only Mem0 stand-in, like the resume
tests. Facts with no recorded scope each become their own uncertain-scope
group, the case an operator consolidates from exception review.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from kai.memory import MemoryResult
from kai.workshop.memory_reconciliation_review import (
    MemoryReconciliationReviewAccessDenied,
    MemoryReconciliationReviewConflict,
    MemoryReconciliationReviewValidationError,
)
from kai.workshop.memory_reconciliation_triage import related_group_ids
from tests.test_memory_legacy_episode_triage import _episode
from tests.test_memory_reconciliation_resume import _count, _Owner
from tests.test_memory_reconciliation_triage import _row


def _unscoped(memory_id: str, text: str) -> MemoryResult:
    """A legacy fact saved without a scope, which triage sends to exception review on its own."""
    row = _row(memory_id, text)
    metadata = dict(row.metadata)
    metadata.pop("scope")
    return replace(row, metadata=metadata)


_FACTS = [
    _unscoped("fact-1", "Alice prefers the dark theme"),
    _unscoped("fact-2", "Alice likes dark mode in every app"),
    _unscoped("fact-3", "Alice uses a dark color scheme"),
    _unscoped("fact-4", "Alice's standup is at nine"),
]


async def _groups(owner: _Owner) -> tuple[str, dict[str, dict[str, Any]]]:
    """Return the plan id and each fact's group, keyed by memory id."""
    triage = owner.triage()
    summary = await triage.latest(owner.principal)
    assert summary is not None
    page = await triage.groups(owner.principal, summary.plan_id)
    return summary.plan_id, {group["evidence"][0]["memory_id"]: group for group in page.groups}


def _selection(groups: dict[str, dict[str, Any]], memory_ids: list[str]) -> tuple[list[str], dict[str, int]]:
    ids = [groups[memory_id]["group_id"] for memory_id in memory_ids]
    return ids, {
        groups[memory_id]["group_id"]: groups[memory_id]["decision"]["state_version"] for memory_id in memory_ids
    }


_CANONICAL = {"content": "Alice prefers dark themes everywhere", "source_memory_id": "fact-1", "scope_kind": "global"}


async def _stage(owner: _Owner, plan_id: str, groups, memory_ids, *, operation: str = "stage-1", **overrides):
    ids, versions = _selection(groups, memory_ids)
    return await owner.triage().stage_consolidation(
        owner.principal,
        plan_id,
        group_ids=overrides.pop("group_ids", ids),
        expected_state_versions=overrides.pop("expected_state_versions", versions),
        canonical=overrides.pop("canonical", _CANONICAL),
        operator_note=overrides.pop("operator_note", "Same preference."),
        consolidation_id=overrides.pop("consolidation_id", None),
        expected_revision=overrides.pop("expected_revision", None),
        allowed_project_ids=overrides.pop("allowed_project_ids", frozenset()),
        client_operation_id=operation,
    )


async def test_three_facts_stage_as_one_consolidation_in_one_step(tmp_path: Path, monkeypatch) -> None:
    owner = await _Owner(tmp_path, list(_FACTS)).start(monkeypatch)
    try:
        plan_id, groups = await _groups(owner)

        staged = await _stage(owner, plan_id, groups, ["fact-1", "fact-2", "fact-3"])
        replay = await _stage(owner, plan_id, groups, ["fact-1", "fact-2", "fact-3"])

        consolidation = staged["consolidation"]
        assert (consolidation["selected_facts"], consolidation["duplicates_excluded"]) == (3, 2)
        assert consolidation["content"] == "Alice prefers dark themes everywhere"
        assert replay["replayed"] is True and replay["consolidation"] == consolidation
        _plan, after = await _groups(owner)
        members = [after[memory_id] for memory_id in ("fact-1", "fact-2", "fact-3")]
        assert all(group["consolidation_id"] == consolidation["consolidation_id"] for group in members)
        assert all(group["decision"]["disposition"] == "approve" for group in members)
        assert after["fact-4"]["consolidation_id"] is None
        listed = await owner.triage().consolidations(owner.principal, plan_id)
        assert [item["consolidation_id"] for item in listed] == [consolidation["consolidation_id"]]

        # A member cannot be decided on its own while it is consolidated.
        with pytest.raises(MemoryReconciliationReviewConflict, match="part of a consolidation"):
            await owner.triage().decide_group(
                owner.principal,
                plan_id,
                members[1]["group_id"],
                disposition="reject",
                action={"kind": "manual_edit_required"},
                operator_note="",
                expected_state_version=members[1]["decision"]["state_version"],
                allowed_project_ids=frozenset(),
                client_operation_id="decide-member",
            )
    finally:
        await owner.close()


@pytest.mark.parametrize(
    ("change", "error"),
    [
        ("stale", MemoryReconciliationReviewConflict),
        ("one fact", MemoryReconciliationReviewValidationError),
        ("foreign source", MemoryReconciliationReviewValidationError),
        ("project outside authority", MemoryReconciliationReviewAccessDenied),
    ],
)
async def test_an_invalid_selection_changes_nothing(tmp_path: Path, monkeypatch, change: str, error) -> None:
    owner = await _Owner(tmp_path, list(_FACTS)).start(monkeypatch)
    try:
        plan_id, groups = await _groups(owner)
        ids, versions = _selection(groups, ["fact-1", "fact-2"])
        overrides: dict[str, Any] = {
            "stale": {"expected_state_versions": {group_id: version + 5 for group_id, version in versions.items()}},
            "one fact": {"group_ids": ids[:1], "expected_state_versions": {ids[0]: versions[ids[0]]}},
            "foreign source": {"canonical": {**_CANONICAL, "source_memory_id": "fact-4"}},
            "project outside authority": {"canonical": {**_CANONICAL, "scope_kind": "project", "scope_key": "secret"}},
        }[change]
        before = _count(
            owner.db_path, "SELECT COALESCE(SUM(state_version), 0) FROM memory_reconciliation_triage_groups"
        )

        with pytest.raises(error):
            await _stage(owner, plan_id, groups, ["fact-1", "fact-2"], **overrides)

        assert _count(owner.db_path, "SELECT COUNT(*) FROM memory_reconciliation_triage_consolidations") == 0
        assert (
            _count(owner.db_path, "SELECT COALESCE(SUM(state_version), 0) FROM memory_reconciliation_triage_groups")
            == before
        )
    finally:
        await owner.close()


async def test_a_fact_cannot_join_two_consolidations_and_episodes_cannot_join(tmp_path: Path, monkeypatch) -> None:
    rows = [*_FACTS, _episode("ep-1", "Deployed the fix", complete=True)]
    owner = await _Owner(tmp_path, rows).start(monkeypatch)
    try:
        plan_id, groups = await _groups(owner)
        await _stage(owner, plan_id, groups, ["fact-1", "fact-2"])
        _plan, groups = await _groups(owner)

        with pytest.raises(MemoryReconciliationReviewConflict, match="another consolidation"):
            await _stage(
                owner,
                plan_id,
                groups,
                ["fact-2", "fact-3"],
                operation="stage-2",
                canonical={**_CANONICAL, "source_memory_id": "fact-2"},
            )
        with pytest.raises(MemoryReconciliationReviewValidationError, match="Only facts"):
            await _stage(
                owner,
                plan_id,
                groups,
                ["fact-3", "ep-1"],
                operation="stage-3",
                canonical={**_CANONICAL, "source_memory_id": "fact-3"},
            )
    finally:
        await owner.close()


async def test_revising_and_cancelling_restore_earlier_decisions(tmp_path: Path, monkeypatch) -> None:
    owner = await _Owner(tmp_path, list(_FACTS)).start(monkeypatch)
    try:
        triage = owner.triage()
        plan_id, groups = await _groups(owner)
        await triage.decide_group(
            owner.principal,
            plan_id,
            groups["fact-3"]["group_id"],
            disposition="defer",
            action={"kind": "manual_edit_required"},
            operator_note="Check later.",
            expected_state_version=groups["fact-3"]["decision"]["state_version"],
            allowed_project_ids=frozenset(),
            client_operation_id="defer-3",
        )
        _plan, groups = await _groups(owner)
        staged = (await _stage(owner, plan_id, groups, ["fact-1", "fact-2", "fact-3"]))["consolidation"]
        _plan, groups = await _groups(owner)

        revised = await _stage(
            owner,
            plan_id,
            groups,
            ["fact-1", "fact-2"],
            operation="revise-1",
            consolidation_id=staged["consolidation_id"],
            expected_revision=staged["revision"],
        )
        _plan, after_revise = await _groups(owner)

        assert revised["consolidation"]["revision"] == 2
        assert after_revise["fact-3"]["consolidation_id"] is None
        assert (
            after_revise["fact-3"]["decision"]["disposition"],
            after_revise["fact-3"]["decision"]["operator_note"],
        ) == (
            "defer",
            "Check later.",
        )
        with pytest.raises(MemoryReconciliationReviewConflict, match="changed after it was loaded"):
            await triage.cancel_consolidation(
                owner.principal,
                plan_id,
                staged["consolidation_id"],
                expected_revision=1,
                client_operation_id="cancel-stale",
            )

        await triage.cancel_consolidation(
            owner.principal, plan_id, staged["consolidation_id"], expected_revision=2, client_operation_id="cancel-1"
        )
        _plan, after_cancel = await _groups(owner)

        assert all(after_cancel[memory_id]["consolidation_id"] is None for memory_id in ("fact-1", "fact-2"))
        assert all(
            after_cancel[memory_id]["decision"]["disposition"] == "pending" for memory_id in ("fact-1", "fact-2")
        )
        assert _count(owner.db_path, "SELECT COUNT(*) FROM memory_reconciliation_triage_consolidations") == 0
    finally:
        await owner.close()


async def test_applying_a_consolidation_creates_one_current_fact_citing_all(tmp_path: Path, monkeypatch) -> None:
    owner = await _Owner(tmp_path, list(_FACTS)).start(monkeypatch)
    try:
        triage = owner.triage()
        plan_id, groups = await _groups(owner)
        await _stage(owner, plan_id, groups, ["fact-1", "fact-2", "fact-3"])
        _plan, groups = await _groups(owner)
        await triage.decide_group(
            owner.principal,
            plan_id,
            groups["fact-4"]["group_id"],
            disposition="reject",
            action={"kind": "manual_edit_required"},
            operator_note="",
            expected_state_version=groups["fact-4"]["decision"]["state_version"],
            allowed_project_ids=frozenset(),
            client_operation_id="reject-4",
        )
        summary = await triage.latest(owner.principal)
        assert summary is not None

        applied = await triage.apply(
            owner.principal,
            plan_id,
            expected_review_version=summary.review_version,
            allowed_project_ids=frozenset(),
            client_operation_id="apply-consolidation",
        )

        assert applied["summary"]["consolidated"] == 3 and applied["summary"]["operator_admitted"] == 1
        connection = sqlite3.connect(owner.db_path)
        revisions = connection.execute("SELECT content, evidence_json FROM memory_fact_revisions").fetchall()
        census = connection.execute(
            "SELECT legacy_rows, absorbed, rejected, unclassified FROM memory_legacy_census"
        ).fetchone()
        connection.close()
        ((content, evidence_json),) = revisions
        cited = {item["reference_id"] for item in json.loads(evidence_json) if item["kind"] == "legacy"}
        assert content == "Alice prefers dark themes everywhere"
        assert cited == {"fact-1", "fact-2", "fact-3"}
        # fact-1 was rewritten in place as the canonical fact; fact-2 and
        # fact-3 stay as absorbed evidence and fact-4 is rejected.
        assert census == (3, 2, 1, 0)
    finally:
        await owner.close()


def test_related_facts_come_from_the_recommendation_and_must_be_other_fact_groups() -> None:
    groups = {
        "mtg_" + "a" * 32: {"group_id": "mtg_" + "a" * 32, "evidence": [{"kind": "fact"}]},
        "mtg_" + "b" * 32: {"group_id": "mtg_" + "b" * 32, "evidence": [{"kind": "fact"}]},
        "mtg_" + "c" * 32: {"group_id": "mtg_" + "c" * 32, "evidence": [{"kind": "episode"}]},
    }
    current = groups["mtg_" + "a" * 32]
    prose = {"rationale": f"Duplicates mtg_{'b' * 32} and mtg_{'c' * 32}, also mtg_{'a' * 32} and mtg_{'d' * 32}."}
    structured = {"related_group_ids": ["mtg_" + "b" * 32, "mtg_" + "d" * 32], "rationale": "unrelated"}

    assert related_group_ids(current, prose, groups) == ["mtg_" + "b" * 32]
    assert related_group_ids(current, structured, groups) == ["mtg_" + "b" * 32]
    assert related_group_ids(current, {}, groups) == []


async def test_consolidation_operations_are_not_replay_gaps(tmp_path: Path, monkeypatch) -> None:
    # Consolidation saves and cancels carry no audit id by design; install
    # status accounts for them through their plan and consolidation.
    from kai.workshop.diagnostics import workshop_memory_reconciliation_status

    owner = await _Owner(tmp_path, list(_FACTS)).start(monkeypatch)
    try:
        plan_id, groups = await _groups(owner)
        kept = (await _stage(owner, plan_id, groups, ["fact-1", "fact-2"], operation="stage-kept"))["consolidation"]
        _plan, groups = await _groups(owner)
        cancelled = (
            await _stage(
                owner,
                plan_id,
                groups,
                ["fact-3", "fact-4"],
                operation="stage-cancelled",
                canonical={**_CANONICAL, "source_memory_id": "fact-3"},
            )
        )["consolidation"]
        await owner.triage().cancel_consolidation(
            owner.principal, plan_id, cancelled["consolidation_id"], expected_revision=1, client_operation_id="cancel"
        )

        assert kept["consolidation_id"] != cancelled["consolidation_id"]
        assert "replay gaps=0;" in workshop_memory_reconciliation_status(owner.db_path)

        # An operation nothing accounts for is still a gap.
        connection = sqlite3.connect(owner.db_path)
        connection.execute(
            "INSERT INTO memory_reconciliation_operations VALUES (?, 'stray', ?, '{}', '2026-01-01T00:00:00Z')",
            (str(owner.principal), "0" * 64),
        )
        connection.commit()
        connection.close()
        assert "replay gaps=1;" in workshop_memory_reconciliation_status(owner.db_path)
    finally:
        await owner.close()
