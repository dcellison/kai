"""Contracts for canonical, privacy-bounded memory extraction receipts."""

from __future__ import annotations

import json
import sqlite3
from dataclasses import replace
from pathlib import Path

import pytest

from kai.memory_extraction import ExtractionResult, _fact_receipt_completion
from kai.workshop.diagnostics import workshop_memory_extraction_receipt_status
from kai.workshop.memory_extraction_receipts import (
    MemoryExtractionReceiptAccessDenied,
    MemoryExtractionReceiptCompletion,
    MemoryExtractionReceiptConflict,
    MemoryExtractionReceiptService,
    MemoryExtractionReceiptSpec,
    MemoryExtractionStorageDecision,
)
from kai.workshop.memory_quality_diagnostics import (
    build_memory_quality_diagnostic,
    load_memory_route_provenance,
    workshop_memory_extraction_quality_status,
)
from kai.workshop.store import WorkshopEventStore

_WORKSHOP_ID = "wsp_11111111111111111111111111111111"
_PRINCIPAL_ID = "prn_22222222222222222222222222222222"
_OTHER_PRINCIPAL_ID = "prn_33333333333333333333333333333333"
_AGENT_PRINCIPAL_ID = "prn_44444444444444444444444444444444"
_CHANNEL_ID = "chn_55555555555555555555555555555555"
_AGENT_ID = "agt_66666666666666666666666666666666"
_RUN_ID = "run_77777777777777777777777777777777"
_SOURCE_ID = "msg_88888888888888888888888888888888"
_RESULT_ID = "msg_99999999999999999999999999999999"
_RUNTIME_ID = "rtp_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
_NOW = "2026-09-20T12:00:00.000000Z"


async def _seed_completed_run(path: Path) -> WorkshopEventStore:
    store = await WorkshopEventStore.open(path)
    connection = store.connection
    await connection.execute(
        "INSERT INTO workshops (id, name, created_at) VALUES (?, 'Kai', ?)",
        (_WORKSHOP_ID, _NOW),
    )
    await connection.executemany(
        "INSERT INTO principals (id, kind, display_name, created_at) VALUES (?, ?, ?, ?)",
        (
            (_PRINCIPAL_ID, "human", "Daniel", _NOW),
            (_OTHER_PRINCIPAL_ID, "human", "Scott", _NOW),
            (_AGENT_PRINCIPAL_ID, "agent", "Kai", _NOW),
        ),
    )
    await connection.execute(
        "INSERT INTO channels (id, workshop_id, kind, name, created_at) VALUES (?, ?, 'direct', 'Kai', ?)",
        (_CHANNEL_ID, _WORKSHOP_ID, _NOW),
    )
    await connection.execute(
        "INSERT INTO agents (id, workshop_id, principal_id, name, created_at) VALUES (?, ?, ?, 'Kai', ?)",
        (_AGENT_ID, _WORKSHOP_ID, _AGENT_PRINCIPAL_ID, _NOW),
    )
    for index, aggregate_id in enumerate((_SOURCE_ID, _RESULT_ID, _RUN_ID), start=1):
        await connection.execute(
            "INSERT INTO event_log (position, event_id, envelope_version, event_type, event_version, "
            "workshop_id, aggregate_type, aggregate_id, actor_principal_id, occurred_at, "
            "payload_json, metadata_json, content_hash) "
            "VALUES (?, ?, 1, 'test.created', 1, ?, 'test', ?, ?, ?, '{}', '{}', ?)",
            (
                index,
                f"evt_{index:032x}",
                _WORKSHOP_ID,
                aggregate_id,
                _PRINCIPAL_ID,
                _NOW,
                f"{index:064x}",
            ),
        )
    await connection.execute(
        "INSERT INTO messages (id, channel_id, author_principal_id, body, created_event_position, created_at) "
        "VALUES (?, ?, ?, ?, 1, ?)",
        (_SOURCE_ID, _CHANNEL_ID, _PRINCIPAL_ID, "SECRET source message", _NOW),
    )
    await connection.execute(
        "INSERT INTO messages (id, channel_id, author_principal_id, body, created_event_position, created_at) "
        "VALUES (?, ?, ?, ?, 2, ?)",
        (_RESULT_ID, _CHANNEL_ID, _AGENT_PRINCIPAL_ID, "SECRET result message", _NOW),
    )
    await connection.execute(
        "INSERT INTO runs (id, workshop_id, channel_id, requested_by_principal_id, agent_id, "
        "inbound_message_id, status, accepted_at, started_at, terminal_at, last_event_position, "
        "result_message_id, runtime_profile_id) "
        "VALUES (?, ?, ?, ?, ?, ?, 'completed', ?, ?, ?, 3, ?, ?)",
        (
            _RUN_ID,
            _WORKSHOP_ID,
            _CHANNEL_ID,
            _PRINCIPAL_ID,
            _AGENT_ID,
            _SOURCE_ID,
            _NOW,
            _NOW,
            _NOW,
            _RESULT_ID,
            _RUNTIME_ID,
        ),
    )
    await connection.commit()
    return store


def _spec(role: str = "fact_extraction") -> MemoryExtractionReceiptSpec:
    return MemoryExtractionReceiptSpec(
        principal_id=_PRINCIPAL_ID,
        runtime_profile_id=_RUNTIME_ID,
        run_id=_RUN_ID,
        source_message_id=_SOURCE_ID,
        result_message_id=_RESULT_ID,
        extraction_role=role,
        backend="codex",
        provider="openai",
        model="gpt-5.6-sol",
        prompt_version="13" if role == "fact_extraction" else "2",
        schema_version="1",
        policy_version="1",
    )


def _zero_memory_completion() -> MemoryExtractionReceiptCompletion:
    return MemoryExtractionReceiptCompletion(
        status="completed",
        decision_outcome="zero_memory",
        failure_code=None,
        candidate_ids=("memory-1",),
        classifier_result=False,
        proposed_intents=(),
        raw_count=0,
        accepted_count=0,
        validation_outcome="empty",
        storage_decisions=(),
        stored_count=0,
        replaced_count=0,
        skipped_count=0,
        memory_scopes=(),
        duration_ms=125,
    )


class TestMemoryExtractionReceiptAuthority:
    async def test_zero_memory_receipt_is_replay_safe_and_content_free(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        store = await _seed_completed_run(path)
        service = MemoryExtractionReceiptService(store.connection)
        try:
            claim = await service.claim(_spec())
            assert claim.claimed is True
            completed = await service.complete(
                claim.receipt.receipt_id,
                _PRINCIPAL_ID,
                _zero_memory_completion(),
            )
            replay_completion = await service.complete(
                claim.receipt.receipt_id,
                _PRINCIPAL_ID,
                _zero_memory_completion(),
            )
            replay_claim = await service.claim(_spec())

            assert completed == replay_completion
            assert replay_claim.claimed is False
            assert replay_claim.receipt.decision_outcome == "zero_memory"
            assert replay_claim.receipt.model == "gpt-5.6-sol"
            assert replay_claim.receipt.candidate_ids == ("memory-1",)
            async with store.connection.execute(
                "SELECT COUNT(*), GROUP_CONCAT("
                "receipt_id || principal_id || runtime_profile_id || run_id || source_message_id || "
                "result_message_id || backend || provider || model || prompt_version || schema_version || "
                "policy_version || "
                "candidate_ids_json || proposed_intents_json || validation_outcome_json || "
                "storage_outcome_json || memory_scope_json, '') "
                "FROM memory_extraction_receipts"
            ) as cursor:
                row = await cursor.fetchone()
            assert row is not None and int(row[0]) == 1
            persisted = str(row[1])
            assert "SECRET source message" not in persisted
            assert "SECRET result message" not in persisted
            assert "/Users/" not in persisted
            assert "api_key" not in persisted.casefold()
        finally:
            await store.close()

        assert workshop_memory_extraction_receipt_status(path) == (
            "Workshop memory extraction receipts: active; receipts=1 (fact=1, episode=0, "
            "completed=1, failed=0, running=0), zero-memory=1; "
            "policy=(admitted=0, suppressed=0, unclassified=1, batch<=1), "
            "facts=(raw=0, accepted=0, stored=0), empty-pass rate=0.000, "
            "memories/admitted=0.000, fragmentation rate=0.000; "
            "integrity gaps=0, replay gaps=0; "
            "authority=canonical/privacy-bounded, legacy provenance=read-time classified"
        )

    async def test_records_bounded_storage_decisions_and_exact_provenance(self, tmp_path: Path):
        store = await _seed_completed_run(tmp_path / "kai.db")
        service = MemoryExtractionReceiptService(store.connection)
        try:
            claim = await service.claim(_spec())
            completion = replace(
                _zero_memory_completion(),
                decision_outcome="replaced",
                classifier_result=True,
                proposed_intents=(("update_of", "memory-old"),),
                raw_count=1,
                accepted_count=1,
                validation_outcome="accepted",
                storage_decisions=(
                    MemoryExtractionStorageDecision(
                        index=0,
                        intent="update_of",
                        outcome="replaced",
                        new_memory_id="memory-new",
                        replaced_memory_id="memory-old",
                        scope="project",
                        project_id="project-kai",
                    ),
                ),
                stored_count=1,
                replaced_count=1,
                memory_scopes=(("project", "project-kai"),),
            )
            receipt = await service.complete(claim.receipt.receipt_id, _PRINCIPAL_ID, completion)

            assert receipt.backend == "codex"
            assert receipt.provider == "openai"
            assert receipt.model == "gpt-5.6-sol"
            assert receipt.prompt_version == "13"
            assert receipt.schema_version == "1"
            assert receipt.policy_version == "1"
            assert receipt.classifier_result is True
            assert receipt.proposed_intents == ({"existing_id": "memory-old", "intent": "update_of"},)
            assert receipt.storage_outcome["decisions"][0]["outcome"] == "replaced"
            assert receipt.memory_scopes == ({"project_id": "project-kai", "scope": "project"},)
        finally:
            await store.close()

    async def test_quality_diagnostics_distinguish_configured_and_actual_routes(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        store = await _seed_completed_run(path)
        service = MemoryExtractionReceiptService(store.connection)
        try:
            await store.connection.execute(
                "INSERT INTO runtime_profile_owners (runtime_profile_id, principal_id) VALUES (?, ?)",
                (_RUNTIME_ID, _PRINCIPAL_ID),
            )
            claim = await service.claim(_spec())
            completion = replace(
                _zero_memory_completion(),
                decision_outcome="replaced",
                classifier_result=True,
                raw_count=1,
                accepted_count=1,
                validation_outcome="accepted",
                storage_decisions=(
                    MemoryExtractionStorageDecision(
                        index=0,
                        intent="update_of",
                        outcome="replaced",
                        new_memory_id="memory-new",
                        replaced_memory_id="memory-old",
                        scope="principal",
                        project_id=None,
                    ),
                ),
                stored_count=1,
                replaced_count=1,
                memory_scopes=(("principal", None),),
            )
            await service.complete(claim.receipt.receipt_id, _PRINCIPAL_ID, completion)
            await store.connection.commit()
        finally:
            await store.close()

        policy = tmp_path / "runtime-profiles.yaml"
        policy.write_text(
            f"""version: 1
runtime_profiles:
  {_RUNTIME_ID}:
    display_name: Daniel
    backend: codex
    provider: openai
    model: gpt-5.6-sol
    timeout_seconds: 600
    models:
      memory_extraction: gpt-5.6-sol
      memory_episode: gpt-5.6-sol
    allowed_services: []
    allowed_workspaces: []
"""
        )
        snapshot = build_memory_quality_diagnostic(path, runtime_policy_path=policy)
        status = workshop_memory_extraction_quality_status(path, runtime_policy_path=policy)

        assert snapshot.state == "active"
        assert snapshot.configured_roles == (
            "episode:codex/openai/gpt-5.6-solx1",
            "fact:codex/openai/gpt-5.6-solx1",
        )
        assert snapshot.actual_roles == ("fact:codex/openai/gpt-5.6-solx1",)
        assert snapshot.prompt_versions == ("fact:v13x1",)
        assert snapshot.replaced == 1
        assert snapshot.episode_attempts == 0
        assert "alerts=none" in status
        assert "SECRET source message" not in status
        assert "SECRET result message" not in status
        assert _PRINCIPAL_ID not in status
        assert _RUNTIME_ID not in status

        connection = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            provenance = load_memory_route_provenance(connection, _PRINCIPAL_ID, limit=100)
        finally:
            connection.close()
        assert len(provenance) == 1
        assert provenance[0].memory_id == "memory-new"
        assert provenance[0].model == "gpt-5.6-sol"
        assert provenance[0].prompt_version == "13"

    async def test_quality_diagnostics_alert_on_route_drift_and_malformed_artifact(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        store = await _seed_completed_run(path)
        service = MemoryExtractionReceiptService(store.connection)
        try:
            claim = await service.claim(_spec())
            await service.complete(claim.receipt.receipt_id, _PRINCIPAL_ID, _zero_memory_completion())
        finally:
            await store.close()

        policy = tmp_path / "runtime-profiles.yaml"
        policy.write_text(
            f"""version: 1
runtime_profiles:
  {_RUNTIME_ID}:
    display_name: Daniel
    backend: codex
    provider: openai
    model: gpt-5.5
    timeout_seconds: 600
    models:
      memory_extraction: gpt-5.5
    allowed_services: []
    allowed_workspaces: []
"""
        )
        artifact = tmp_path / "home" / "opaque" / "docs" / "memory-quality" / "bad-score.json"
        artifact.parent.mkdir(parents=True)
        artifact.write_text("not json")

        snapshot = build_memory_quality_diagnostic(
            path,
            runtime_policy_path=policy,
            quality_root=tmp_path / "home",
        )

        assert snapshot.state == "DEGRADED"
        assert snapshot.alerts == ("model_drift:1", "malformed_quality_artifacts:1")

    async def test_interrupted_claim_fails_closed_without_duplicate(self, tmp_path: Path):
        store = await _seed_completed_run(tmp_path / "kai.db")
        service = MemoryExtractionReceiptService(store.connection, _claim_owner="a" * 32)
        try:
            first = await service.claim(_spec("episode_generation"))
            concurrent = await service.claim(_spec("episode_generation"))
            replay = await MemoryExtractionReceiptService(
                store.connection,
                _claim_owner="b" * 32,
            ).claim(_spec("episode_generation"))
            assert first.claimed is True
            assert concurrent.claimed is False
            assert concurrent.interrupted is False
            assert concurrent.receipt.status == "running"
            assert replay.claimed is False
            assert replay.interrupted is True
            assert replay.receipt.status == "failed"
            assert replay.receipt.failure_code == "interrupted"
            async with store.connection.execute(
                "SELECT COUNT(*) FROM memory_extraction_receipts WHERE run_id = ? AND extraction_role = ?",
                (_RUN_ID, "episode_generation"),
            ) as cursor:
                assert int((await cursor.fetchone())[0]) == 1
        finally:
            await store.close()

    async def test_cross_principal_reads_and_mismatched_run_bindings_fail_closed(self, tmp_path: Path):
        store = await _seed_completed_run(tmp_path / "kai.db")
        service = MemoryExtractionReceiptService(store.connection)
        try:
            claim = await service.claim(_spec())
            await service.complete(claim.receipt.receipt_id, _PRINCIPAL_ID, _zero_memory_completion())
            other = service.authority_for_principal(_OTHER_PRINCIPAL_ID)
            with pytest.raises(MemoryExtractionReceiptAccessDenied):
                await service.receipt(other, claim.receipt.receipt_id)
            with pytest.raises(MemoryExtractionReceiptAccessDenied):
                await service.claim(replace(_spec("episode_generation"), principal_id=_OTHER_PRINCIPAL_ID))
            with pytest.raises(MemoryExtractionReceiptConflict):
                await service.claim(replace(_spec(), model="different-model"))
            with pytest.raises(MemoryExtractionReceiptConflict):
                await service.complete(
                    claim.receipt.receipt_id,
                    _PRINCIPAL_ID,
                    replace(_zero_memory_completion(), decision_outcome="episode_only"),
                )
        finally:
            await store.close()

    async def test_receipt_json_is_structured_and_bounded(self, tmp_path: Path):
        store = await _seed_completed_run(tmp_path / "kai.db")
        service = MemoryExtractionReceiptService(store.connection)
        try:
            claim = await service.claim(_spec())
            await service.complete(claim.receipt.receipt_id, _PRINCIPAL_ID, _zero_memory_completion())
            async with store.connection.execute(
                "SELECT candidate_ids_json, proposed_intents_json, validation_outcome_json, "
                "storage_outcome_json, memory_scope_json FROM memory_extraction_receipts"
            ) as cursor:
                row = await cursor.fetchone()
            assert row is not None
            assert [json.loads(str(value)) for value in row] == [
                ["memory-1"],
                [],
                {"accepted_count": 0, "outcome": "empty", "raw_count": 0, "rejected_count": 0},
                {"decisions": [], "replaced_count": 0, "skipped_count": 0, "stored_count": 0},
                [],
            ]
        finally:
            await store.close()

    async def test_receipt_records_bounded_policy_decisions(self, tmp_path: Path):
        store = await _seed_completed_run(tmp_path / "kai.db")
        service = MemoryExtractionReceiptService(store.connection)
        try:
            claim = await service.claim(_spec())
            completion = replace(
                _zero_memory_completion(),
                decision_outcome="admission_suppressed",
                validation_outcome="not_attempted",
                policy_outcome=(
                    ("admission", "suppressed"),
                    ("admission_reason", "routine_workflow_ack"),
                    ("cadence", "suppressed"),
                    ("batch_size", 1),
                    ("batch_limit", 1),
                ),
            )

            receipt = await service.complete(claim.receipt.receipt_id, _PRINCIPAL_ID, completion)

            assert receipt.validation_outcome["policy"] == {
                "admission": "suppressed",
                "admission_reason": "routine_workflow_ack",
                "cadence": "suppressed",
                "batch_size": 1,
                "batch_limit": 1,
            }
        finally:
            await store.close()


@pytest.mark.parametrize(
    ("result", "decisions", "stored", "replaced", "skipped", "expected"),
    (
        (ExtractionResult([], False), [], 0, 0, 0, ("completed", "zero_memory", None)),
        (ExtractionResult([], False, outcome="timeout"), [], 0, 0, 0, ("failed", "timeout", "timeout")),
        (
            ExtractionResult([], False, outcome="validation_rejection", raw_fact_count=1),
            [],
            0,
            0,
            0,
            ("completed", "validation_rejected", None),
        ),
        (
            ExtractionResult([{"intent": "skip_redundant", "existing_id": "old"}], False, raw_fact_count=1),
            [MemoryExtractionStorageDecision(0, "skip_redundant", "duplicate_skipped")],
            0,
            0,
            1,
            ("completed", "duplicate_skipped", None),
        ),
        (
            ExtractionResult([{"intent": "update_of", "existing_id": "old"}], False, raw_fact_count=1),
            [MemoryExtractionStorageDecision(0, "update_of", "replaced", new_memory_id="new")],
            1,
            1,
            0,
            ("completed", "replaced", None),
        ),
        (
            ExtractionResult([{"intent": "new"}], False, raw_fact_count=1),
            [MemoryExtractionStorageDecision(0, "new", "storage_failed")],
            0,
            0,
            0,
            ("failed", "storage_failure", "storage_failure"),
        ),
    ),
)
def test_fact_completion_preserves_distinct_decision_outcomes(
    result,
    decisions,
    stored,
    replaced,
    skipped,
    expected,
):
    completion = _fact_receipt_completion(
        result=result,
        candidate_ids={"candidate"},
        decisions=decisions,
        stored=stored,
        replaced=replaced,
        skipped=skipped,
        duration_ms=10,
    )
    assert (completion.status, completion.decision_outcome, completion.failure_code) == expected
