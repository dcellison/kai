from __future__ import annotations

import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from kai import memory_admin
from kai import memory_reconciliation as reconciliation
from kai.memory import MemoryResult
from kai.workshop import memory_reconciliation_review

PRINCIPAL = "prn_30000000000000000000000000000001"
RUNTIME = "rtp_30000000000000000000000000000001"
NOW = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _isolate_workshop_review_projection(monkeypatch: pytest.MonkeyPatch):
    """The CLI unit database intentionally contains only ownership authority."""
    monkeypatch.setattr(
        memory_reconciliation_review,
        "prior_reconciliation_dispositions",
        lambda *_args, **_kwargs: set(),
    )
    monkeypatch.setattr(
        memory_reconciliation_review,
        "record_reconciliation_audit",
        lambda *_args, **_kwargs: True,
    )


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


def test_review_accepts_a_bounded_operator_corrected_fact():
    audit = reconciliation.build_audit(
        principal_id=PRINCIPAL,
        runtime_profile_id=RUNTIME,
        rows=[_row("mem_1", "Daniel uses an obsolete model")],
        now=NOW,
    )
    review = reconciliation.build_review_template(audit)
    for decision in review["decisions"]:
        decision["disposition"] = "reject"
    decision = review["decisions"][0]
    candidate = next(item for item in audit["candidates"] if item["candidate_id"] == decision["candidate_id"])
    decision["disposition"] = "approve"
    decision["action"] = {
        "kind": "adopt_corrected",
        "source_memory_id": candidate["evidence"][0]["memory_id"],
        "replacement": {
            "content": "Daniel currently uses the configured runtime model.",
            "scope_kind": "global",
            "scope_key": "",
            "confidence": 0.95,
            "valid_from": NOW.isoformat(),
            "valid_until": None,
        },
    }

    sealed = reconciliation.seal_review(audit, review, reviewer="Daniel")

    assert sealed["decisions"][0]["action"]["kind"] == "adopt_corrected"


def test_corrected_fact_rejects_invalid_scope_and_validity():
    audit = reconciliation.build_audit(
        principal_id=PRINCIPAL,
        runtime_profile_id=RUNTIME,
        rows=[_row("mem_1", "A legacy fact")],
        now=NOW,
    )
    candidate = audit["candidates"][0]
    action = {
        "kind": "adopt_corrected",
        "replacement": {
            "content": "A corrected fact",
            "scope_kind": "project",
            "scope_key": "",
            "confidence": 0.8,
        },
    }

    with pytest.raises(reconciliation.MemoryReconciliationError, match="requires a project"):
        reconciliation.validate_candidate_action(candidate, action)


def test_legacy_fact_adoption_preserves_incomplete_provenance() -> None:
    audit = reconciliation.build_audit(
        principal_id=PRINCIPAL,
        runtime_profile_id=RUNTIME,
        rows=[_row("mem_1", "A legacy fact without complete extraction provenance")],
        now=NOW,
    )
    evidence = audit["candidates"][0]["evidence"][0]

    spec = reconciliation._fact_spec(evidence, receipt_id="mrr_test", reason="Legacy adoption")

    assert evidence["migration_gaps"]
    assert spec.migration_classification == "legacy_incomplete"
    assert set(spec.migration_gaps) == set(evidence["migration_gaps"])
    assert spec.admission_authority == "operator_review"

    corrected = reconciliation._fact_spec(
        evidence,
        receipt_id="mrr_test",
        reason="Operator correction",
        action={
            "kind": "adopt_corrected",
            "replacement": {"content": "A corrected current fact"},
        },
    )
    assert corrected.migration_classification == "legacy_complete"
    assert corrected.migration_gaps == ()
    assert corrected.admission_authority == "operator_review"


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

    real_open = reconciliation.WorkshopEventStore.open
    real_store = await real_open(tmp_path / "kai.db")

    class FakeStore:
        """The real store's connection (apply records its run there) with the close observed."""

        connection = real_store.connection

        async def close(self):
            calls.append(("store", "close"))

    class FakeFactService:
        def __init__(self, store):
            pass

        async def authority_for(self, principal, runtime):
            return "authority"

        async def create(self, authority, spec, *, idempotency_key, stable_claim_key):
            calls.append(("create", idempotency_key))
            return SimpleNamespace(claim_id="mcl_test", revision_id="mrv_test", projection_status="succeeded")

        async def retract(self, authority, claim_id, revision_id, *, reason, idempotency_key, expired):
            calls.append(("expire", idempotency_key))
            # The expiry's vector delete failed: canonical state committed,
            # but the receipt must say the item did not reach search.
            return SimpleNamespace(claim_id=claim_id, revision_id="mrv_expired", projection_status="failed")

    class FakeEpisodeService:
        def __init__(self, store):
            pass

        async def authority_for(self, principal, runtime):
            return "episode-authority"

    async def fake_open(path):
        return FakeStore()

    monkeypatch.setattr(reconciliation.WorkshopEventStore, "open", fake_open)

    async def evidence_unchanged(*_args, **_kwargs) -> None:
        return None

    # The row-level drift check is covered with a real store elsewhere.
    monkeypatch.setattr(reconciliation, "_verify_apply_evidence", evidence_unchanged)
    monkeypatch.setattr(reconciliation, "MemoryFactLifecycleService", FakeFactService)
    monkeypatch.setattr(reconciliation, "MemoryEpisodeHistoryService", FakeEpisodeService)

    receipt = await reconciliation.apply_review(db_path=tmp_path / "kai.db", audit=audit, review=sealed)
    assert [call[0] for call in calls] == ["create", "expire", "store"]
    assert receipt["kind"] == reconciliation.RECEIPT_KIND
    assert receipt["applied"][0]["candidate_id"] == audit["candidates"][0]["candidate_id"]
    assert receipt["applied"][0]["projection"] == ["failed"]
    await real_store.close()
    assert receipt["projection_failures"] == [
        {
            "candidate_id": audit["candidates"][0]["candidate_id"],
            "kind": "fact",
            "id": "mrv_expired",
            "status": "failed",
        }
    ]


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


def _reconciliation_cli_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE principals (id TEXT PRIMARY KEY, kind TEXT NOT NULL);
        CREATE TABLE runtime_profile_owners (
            runtime_profile_id TEXT PRIMARY KEY,
            principal_id TEXT NOT NULL
        );
        INSERT INTO principals VALUES ('prn_30000000000000000000000000000001', 'human');
        INSERT INTO runtime_profile_owners VALUES (
            'rtp_30000000000000000000000000000001',
            'prn_30000000000000000000000000000001'
        );
        """
    )
    connection.commit()
    connection.close()


@pytest.mark.parametrize("lookup_fails", [False, True])
def test_reconciliation_audit_closes_offline_memory_on_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lookup_fails: bool
):
    db_path = tmp_path / "kai.db"
    _reconciliation_cli_database(db_path)
    config = SimpleNamespace(session_db_path=db_path, protected_install=False)
    monkeypatch.setattr("kai.config.load_config", lambda: config)
    monkeypatch.setattr(memory_admin, "_initialize_memory", lambda loaded: loaded)
    close_calls: list[bool] = []
    monkeypatch.setattr("kai.memory.close_memory", lambda: close_calls.append(True))
    if lookup_fails:
        monkeypatch.setattr(
            "kai.memory.get_all_for_lifecycle_projection",
            lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("lookup failed")),
        )
    else:
        monkeypatch.setattr("kai.memory.get_all_for_lifecycle_projection", lambda **_kwargs: [_row("mem_1", "Fact")])
    args = memory_admin._build_parser().parse_args(
        ["reconciliation", "audit", PRINCIPAL, RUNTIME, "--out-dir", str(tmp_path / "reports")]
    )

    assert memory_admin._cmd_reconciliation(args) == (1 if lookup_fails else 0)
    assert close_calls == [True]


@pytest.mark.parametrize("apply_fails", [False, True])
def test_reconciliation_apply_closes_offline_memory_on_success_and_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, apply_fails: bool
):
    db_path = tmp_path / "kai.db"
    sqlite3.connect(db_path).close()
    rows = [_row("mem_1", "Same"), _row("mem_2", "Same")]
    audit = reconciliation.build_audit(
        principal_id=PRINCIPAL,
        runtime_profile_id=RUNTIME,
        rows=rows,
        now=NOW,
    )
    review = reconciliation.build_review_template(audit)
    for decision in review["decisions"]:
        decision["disposition"] = "reject"
    sealed = reconciliation.seal_review(audit, review, reviewer="Daniel")
    audit_path = tmp_path / "audit.json"
    review_path = tmp_path / "review.json"
    reconciliation.write_audit(audit_path, audit)
    reconciliation.write_sealed_review(review_path, sealed)
    config = SimpleNamespace(session_db_path=db_path, protected_install=False)
    monkeypatch.setattr("kai.config.load_config", lambda: config)
    monkeypatch.setattr(memory_admin, "_initialize_memory", lambda loaded: loaded)
    monkeypatch.setattr("kai.memory.get_all_for_lifecycle_projection", lambda **_kwargs: rows)
    close_calls: list[bool] = []
    monkeypatch.setattr("kai.memory.close_memory", lambda: close_calls.append(True))

    async def fake_apply_review(**_kwargs):
        if apply_fails:
            raise RuntimeError("apply failed")
        return {"kind": reconciliation.RECEIPT_KIND, "version": 1, "sha256": "receipt"}

    monkeypatch.setattr(reconciliation, "apply_review", fake_apply_review)
    args = memory_admin._build_parser().parse_args(
        [
            "reconciliation",
            "apply",
            str(audit_path),
            str(review_path),
            "--out-dir",
            str(tmp_path / "receipts"),
            "--yes",
        ]
    )

    assert memory_admin._cmd_reconciliation(args) == (1 if apply_fails else 0)
    assert close_calls == [True]


def test_default_reconciliation_artifacts_keep_private_modes_and_transfer_to_runtime_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    class FakeRegistry:
        @staticmethod
        def load(_config):
            return FakeRegistry()

        def resolve(self, runtime_profile_id):
            assert runtime_profile_id == RUNTIME
            return SimpleNamespace(os_user="daniel")

    monkeypatch.setattr("kai.workshop.runtime_profiles.WorkshopRuntimeProfileRegistry", FakeRegistry)
    monkeypatch.setattr(memory_admin.os, "geteuid", lambda: 0)
    monkeypatch.setattr(memory_admin.pwd, "getpwnam", lambda name: SimpleNamespace(pw_uid=501, pw_gid=20))
    owner = memory_admin._reconciliation_artifact_owner(SimpleNamespace(protected_install=True), RUNTIME)
    assert owner == (501, 20)

    chowns: list[tuple[Path, int, int]] = []
    monkeypatch.setattr(
        memory_admin.os,
        "chown",
        lambda path, uid, gid, **_kwargs: chowns.append((Path(path), uid, gid)),
    )
    report_dir = tmp_path / "reports"
    memory_admin._prepare_reconciliation_directory(report_dir, owner)
    report = report_dir / "audit.md"
    report.write_text("private", encoding="utf-8")
    os.chmod(report, 0o400)
    memory_admin._claim_reconciliation_artifact(report, owner)

    assert chowns == [(report_dir, 501, 20), (report, 501, 20)]
    assert report_dir.stat().st_mode & 0o777 == 0o700
    assert report.stat().st_mode & 0o777 == 0o400


def test_explicit_reconciliation_output_keeps_caller_ownership(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    db_path = tmp_path / "kai.db"
    _reconciliation_cli_database(db_path)
    config = SimpleNamespace(session_db_path=db_path, protected_install=True)
    monkeypatch.setattr("kai.config.load_config", lambda: config)
    monkeypatch.setattr(memory_admin, "_initialize_memory", lambda loaded: loaded)
    monkeypatch.setattr("kai.memory.get_all_for_lifecycle_projection", lambda **_kwargs: [_row("mem_1", "Fact")])
    monkeypatch.setattr("kai.memory.close_memory", lambda: None)
    monkeypatch.setattr(
        memory_admin,
        "_reconciliation_artifact_owner",
        lambda *_args: pytest.fail("explicit output must not resolve or change principal ownership"),
    )
    out_dir = tmp_path / "explicit"
    args = memory_admin._build_parser().parse_args(
        ["reconciliation", "audit", PRINCIPAL, RUNTIME, "--out-dir", str(out_dir)]
    )

    assert memory_admin._cmd_reconciliation(args) == 0
    artifacts = list(out_dir.iterdir())
    assert len(artifacts) == 2
    assert all(path.stat().st_uid == os.geteuid() for path in artifacts)
    assert all(path.stat().st_mode & 0o777 == 0o400 for path in artifacts)


def test_default_reconciliation_audit_claims_directory_and_artifacts_for_principal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    db_path = tmp_path / "kai.db"
    _reconciliation_cli_database(db_path)
    config = SimpleNamespace(session_db_path=db_path, protected_install=True)
    monkeypatch.setattr("kai.config.load_config", lambda: config)
    monkeypatch.setattr(memory_admin, "_initialize_memory", lambda loaded: loaded)
    monkeypatch.setattr("kai.memory.get_all_for_lifecycle_projection", lambda **_kwargs: [_row("mem_1", "Fact")])
    monkeypatch.setattr("kai.memory.close_memory", lambda: None)
    out_dir = tmp_path / "canonical" / "memory-reconciliation"
    monkeypatch.setattr(memory_admin, "_default_human_report_directory", lambda *_args: out_dir)
    monkeypatch.setattr(memory_admin, "_reconciliation_artifact_owner", lambda *_args: (501, 20))
    chowns: list[tuple[Path, int, int]] = []
    monkeypatch.setattr(
        memory_admin.os,
        "chown",
        lambda path, uid, gid, **_kwargs: chowns.append((Path(path), uid, gid)),
    )
    args = memory_admin._build_parser().parse_args(["reconciliation", "audit", PRINCIPAL, RUNTIME])

    assert memory_admin._cmd_reconciliation(args) == 0
    artifacts = sorted(out_dir.iterdir())
    assert len(artifacts) == 2
    assert chowns == [(out_dir, 501, 20)] + [(artifact, 501, 20) for artifact in artifacts]
    assert out_dir.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o400 for path in artifacts)
