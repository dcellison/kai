from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from kai import memory_admin
from kai import memory_reconciliation as reconciliation
from kai.memory import MemoryResult

PRINCIPAL = "prn_30000000000000000000000000000001"
RUNTIME = "rtp_30000000000000000000000000000001"
NOW = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)


def _row(memory_id: str, text: str, **metadata) -> MemoryResult:
    values = {"source": "extracted", "scope": "global", "confidence": 0.8, **metadata}
    created = values.pop("created_at", NOW.isoformat())
    return MemoryResult(memory_id, text, 0.0, str(values.get("type", "fact")), values, created, created)


def test_read_only_audit_groups_required_candidate_classes_deterministically():
    rows = [
        _row(
            "mem_1",
            "Daniel currently uses Codex",
            source_receipt_id="receipt-a",
            created_at=(NOW - timedelta(days=200)).isoformat(),
        ),
        _row("mem_2", "Daniel currently uses Codex", source_receipt_id="receipt-a"),
        _row("mem_3", "Daniel does not use Codex", source_receipt_id="receipt-b"),
        _row("mem_4", "Daniel does use Codex", source_receipt_id="receipt-b"),
        _row("mem_5", "Built the feature", type="episode", source="episode", source_run_id="run-a"),
        _row("mem_6", "Fixed the feature", type="episode", source="episode", source_run_id="run-a"),
    ]
    first = reconciliation.build_audit(principal_id=PRINCIPAL, runtime_profile_id=RUNTIME, rows=rows, now=NOW)
    second = reconciliation.build_audit(principal_id=PRINCIPAL, runtime_profile_id=RUNTIME, rows=rows, now=NOW)

    assert first == second
    assert first["read_only"] is True
    assert {candidate["category"] for candidate in first["candidates"]} >= {
        "likely_duplicate",
        "fragmented_claims",
        "possible_contradiction",
        "stale_current_state",
        "malformed_provenance",
        "related_episodes",
    }
    evidence = first["candidates"][0]["evidence"][0]
    assert {"text", "scope", "source", "confidence", "model", "prompt_version"} <= set(evidence)


def test_review_requires_every_decision_and_rejects_overlapping_approvals():
    audit = reconciliation.build_audit(
        principal_id=PRINCIPAL,
        runtime_profile_id=RUNTIME,
        rows=[_row("mem_1", "Same"), _row("mem_2", "Same")],
        now=NOW,
    )
    review = reconciliation.build_review_template(audit)
    with pytest.raises(reconciliation.MemoryReconciliationError, match="approved, rejected, or deferred"):
        reconciliation.seal_review(audit, review, reviewer="Daniel")

    for decision in review["decisions"]:
        decision["disposition"] = "approve"
    with pytest.raises(reconciliation.MemoryReconciliationError, match="overlap"):
        reconciliation.seal_review(audit, review, reviewer="Daniel")


def test_rejected_and_deferred_candidates_are_suppressed_until_evidence_changes(tmp_path: Path):
    rows = [_row("mem_1", "A legacy fact")]
    audit = reconciliation.build_audit(principal_id=PRINCIPAL, runtime_profile_id=RUNTIME, rows=rows, now=NOW)
    review = reconciliation.build_review_template(audit)
    for decision in review["decisions"]:
        decision["disposition"] = "defer"
    sealed = reconciliation.seal_review(audit, review, reviewer="Daniel")
    receipt = {
        "kind": reconciliation.RECEIPT_KIND,
        "version": 1,
        "decisions": sealed["decisions"],
    }
    receipt["sha256"] = reconciliation._document_digest(receipt)
    reconciliation.write_receipt(tmp_path / "receipt-test.json", receipt)

    repeated = reconciliation.build_audit(
        principal_id=PRINCIPAL,
        runtime_profile_id=RUNTIME,
        rows=rows,
        prior_decisions=reconciliation.prior_dispositions(tmp_path),
        now=NOW,
    )
    assert repeated["candidate_count"] == 0
    changed = reconciliation.build_audit(
        principal_id=PRINCIPAL,
        runtime_profile_id=RUNTIME,
        rows=[_row("mem_1", "Changed legacy fact")],
        prior_decisions=reconciliation.prior_dispositions(tmp_path),
        now=NOW,
    )
    assert changed["candidate_count"] > 0


@pytest.mark.asyncio
async def test_apply_routes_approval_through_fact_lifecycle_and_returns_receipt(tmp_path: Path, monkeypatch):
    audit = reconciliation.build_audit(
        principal_id=PRINCIPAL,
        runtime_profile_id=RUNTIME,
        rows=[_row("mem_1", "Expired", valid_until=(NOW - timedelta(days=1)).isoformat())],
        now=NOW,
    )
    audit["candidates"] = [item for item in audit["candidates"] if item["category"] == "stale_current_state"]
    audit["candidate_count"] = 1
    audit["sha256"] = reconciliation._document_digest(audit)
    review = reconciliation.build_review_template(audit)
    review["decisions"][0]["disposition"] = "approve"
    sealed = reconciliation.seal_review(audit, review, reviewer="Daniel")
    calls: list[tuple[str, str]] = []

    class FakeStore:
        async def close(self):
            calls.append(("store", "close"))

    class FakeFactService:
        def __init__(self, store):
            pass

        async def authority_for(self, principal, runtime):
            return "authority"

        async def create(self, authority, spec, *, idempotency_key, stable_claim_key):
            calls.append(("create", idempotency_key))
            return SimpleNamespace(claim_id="mcl_test", revision_id="mrv_test")

        async def retract(self, authority, claim_id, revision_id, *, reason, idempotency_key, expired):
            calls.append(("expire", idempotency_key))

    class FakeEpisodeService:
        def __init__(self, store):
            pass

        async def authority_for(self, principal, runtime):
            return "episode-authority"

    async def fake_open(path):
        return FakeStore()

    monkeypatch.setattr(reconciliation.WorkshopEventStore, "open", fake_open)
    monkeypatch.setattr(reconciliation, "MemoryFactLifecycleService", FakeFactService)
    monkeypatch.setattr(reconciliation, "MemoryEpisodeHistoryService", FakeEpisodeService)

    receipt = await reconciliation.apply_review(db_path=tmp_path / "kai.db", audit=audit, review=sealed)
    assert [call[0] for call in calls] == ["create", "expire", "store"]
    assert receipt["kind"] == reconciliation.RECEIPT_KIND
    assert receipt["applied"][0]["candidate_id"] == audit["candidates"][0]["candidate_id"]


def test_cli_exposes_complete_reconciliation_workflow():
    parser = memory_admin._build_parser()
    assert parser.parse_args(["reconciliation", "audit", PRINCIPAL, RUNTIME]).reconciliation_command == "audit"
    assert (
        parser.parse_args(["reconciliation", "review-template", "audit.json"]).reconciliation_command
        == "review-template"
    )
    assert (
        parser.parse_args(
            ["reconciliation", "seal-review", "audit.json", "review.json", "--reviewer", "Daniel"]
        ).reconciliation_command
        == "seal-review"
    )
    assert (
        parser.parse_args(["reconciliation", "apply", "audit.json", "sealed.json", "--yes"]).reconciliation_command
        == "apply"
    )


def test_markdown_report_is_private_operator_facing():
    audit = reconciliation.build_audit(
        principal_id=PRINCIPAL,
        runtime_profile_id=RUNTIME,
        rows=[_row("mem_1", "Sensitive fact")],
        now=NOW,
    )
    report = reconciliation.render_audit_markdown(audit)
    assert "read only; no memory was changed" in report
    assert "Sensitive fact" in report
    assert "Model provenance" in report
