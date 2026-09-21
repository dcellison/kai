from __future__ import annotations

import json
import os
import sqlite3
from types import SimpleNamespace

import pytest

from kai import memory_admin
from kai.memory import MemoryResult
from kai.memory_quality_corpus import (
    MemoryQualityCorpusError,
    ProductionReceipt,
    build_review_template,
    build_snapshot,
    load_production_receipts,
    resolve_human_principal,
    score_review,
    seal_review,
    validate_snapshot,
    write_review_template,
    write_sealed_review,
    write_snapshot,
)


def _receipt(*, role: str = "fact_extraction", suffix: str = "1") -> ProductionReceipt:
    decision = {
        "index": 0,
        "intent": "new" if role == "fact_extraction" else "store_episode",
        "outcome": "stored",
        "new_memory_id": f"mem_{suffix}",
        "scope": "global",
    }
    return ProductionReceipt(
        receipt_id=f"mer_{suffix}",
        principal_id="prn_owner",
        runtime_profile_id="rtp_owner",
        run_id=f"run_{suffix}",
        source_message_id=f"msg_source_{suffix}",
        result_message_id=f"msg_result_{suffix}",
        extraction_role=role,
        backend="codex",
        provider="openai",
        model="gpt-test",
        prompt_version="prompt-v1",
        schema_version="schema-v1",
        policy_version="policy-v1",
        status="completed",
        decision_outcome="stored",
        classifier_result=role == "episode_generation",
        proposed_intents=(({"intent": decision["intent"]}),),
        validation_outcome={"outcome": "accepted", "raw_count": 1, "accepted_count": 1},
        storage_outcome={
            "stored_count": 1,
            "replaced_count": 0,
            "skipped_count": 0,
            "decisions": [decision],
        },
        memory_scopes=({"scope": "global"},),
        candidate_ids=(),
        created_at="2026-09-21T00:00:00Z",
        completed_at="2026-09-21T00:00:01Z",
        channel_id="chn_general",
        channel_kind="group",
        channel_name="General",
        agent_handle="kai",
        agent_display_name="Kai",
        source_body="The operator changed a preference.",
        result_body="Understood.",
    )


def _memory(owner: str, runtime_profile_id: str, memory_id: str) -> MemoryResult:
    assert owner == "prn_owner"
    assert runtime_profile_id == "rtp_owner"
    return MemoryResult(
        id=memory_id,
        text=f"remembered {memory_id}",
        score=0.0,
        memory_type="fact",
        metadata={"source": "extracted", "scope": "global"},
        created_at="2026-09-21T00:00:02Z",
        updated_at="2026-09-21T00:00:02Z",
    )


def _snapshot():
    return build_snapshot(
        principal_id="prn_owner",
        receipts=[_receipt(), _receipt(role="episode_generation", suffix="2")],
        memory_lookup=_memory,
        seed=1706,
    )


def test_snapshot_is_hash_sealed_and_detects_private_input_drift():
    snapshot = _snapshot()
    validate_snapshot(snapshot)

    cases = snapshot["cases"]
    assert isinstance(cases, list)
    conversation = cases[0]["conversation"]
    assert isinstance(conversation, dict)
    conversation["user"] = "tampered"

    with pytest.raises(MemoryQualityCorpusError, match="digest"):
        validate_snapshot(snapshot)


def test_review_seal_and_score_cover_each_quality_dimension():
    snapshot = _snapshot()
    review = build_review_template(snapshot)
    decisions = review["decisions"]
    assert isinstance(decisions, list)

    fact = decisions[0]
    episode = decisions[1]
    assert isinstance(fact, dict) and isinstance(episode, dict)
    fact.update(
        {
            "review_status": "complete",
            "scenario_tags": ["changed_value", "global_scope"],
            "case_labels": [],
            "expected_fact_count": 1,
            "episode_expected": False,
            "update_expected": True,
            "update_detected": True,
        }
    )
    fact_output = fact["outputs"][0]
    fact_output.update(
        {
            "verdict": "useful",
            "labels": ["useful"],
            "scope_correct": True,
            "consolidation_correct": True,
        }
    )
    episode.update(
        {
            "review_status": "complete",
            "scenario_tags": ["completed_workflow"],
            "case_labels": [],
            "expected_fact_count": 0,
            "episode_expected": True,
            "update_expected": False,
            "update_detected": False,
        }
    )
    episode_output = episode["outputs"][0]
    episode_output.update(
        {
            "verdict": "not_useful",
            "labels": ["fragmented"],
            "scope_correct": False,
            "consolidation_correct": None,
        }
    )

    sealed = seal_review(snapshot, review, reviewer="Daniel")
    report = score_review(snapshot, sealed)

    assert report["metrics"] == {
        "precision": 1.0,
        "useful_memory_rate": 0.5,
        "duplication_rate": 0.0,
        "fragmentation_rate": 0.5,
        "update_detection_rate": 1.0,
        "episode_classifier_accuracy": 1.0,
        "episode_useful_rate": 0.0,
        "scope_accuracy": 0.5,
        "consolidation_accuracy": 1.0,
    }
    assert report["qualification_ready"] is False


def test_seal_refuses_pending_or_unknown_labels():
    snapshot = _snapshot()
    review = build_review_template(snapshot)
    with pytest.raises(MemoryQualityCorpusError, match="complete"):
        seal_review(snapshot, review, reviewer="Daniel")

    decisions = review["decisions"]
    assert isinstance(decisions, list)
    for decision in decisions:
        decision["review_status"] = "complete"
        decision["scenario_tags"] = []
        decision["case_labels"] = []
        for output in decision["outputs"]:
            output["verdict"] = "not_applicable"
            output["labels"] = []
    decisions[0]["case_labels"] = ["invented"]
    with pytest.raises(MemoryQualityCorpusError, match="case labels"):
        seal_review(snapshot, review, reviewer="Daniel")


def test_seal_refuses_review_output_drift_from_snapshot():
    snapshot = _snapshot()
    review = build_review_template(snapshot)
    decisions = review["decisions"]
    assert isinstance(decisions, list)
    for decision in decisions:
        decision["review_status"] = "complete"
        for output in decision["outputs"]:
            output["verdict"] = "not_applicable"
    decisions[0]["outputs"][0]["memory_id"] = "mem_substituted"

    with pytest.raises(MemoryQualityCorpusError, match="immutable snapshot"):
        seal_review(snapshot, review, reviewer="Daniel")


def test_private_artifacts_refuse_overwrite_and_use_owner_only_modes(tmp_path):
    snapshot = _snapshot()
    snapshot_path = tmp_path / "private" / "snapshot.json"
    write_snapshot(snapshot_path, snapshot)
    assert os.stat(snapshot_path).st_mode & 0o777 == 0o400
    assert os.stat(snapshot_path.parent).st_mode & 0o777 == 0o700
    with pytest.raises(MemoryQualityCorpusError, match="overwrite"):
        write_snapshot(snapshot_path, snapshot)

    review = build_review_template(snapshot)
    review_path = tmp_path / "private" / "review.json"
    write_review_template(review_path, review)
    assert os.stat(review_path).st_mode & 0o777 == 0o600

    for decision in review["decisions"]:
        decision["review_status"] = "complete"
        for output in decision["outputs"]:
            output["verdict"] = "not_applicable"
    sealed = seal_review(snapshot, review, reviewer="operator")
    sealed_path = tmp_path / "private" / "sealed.json"
    write_sealed_review(sealed_path, sealed, snapshot)
    assert os.stat(sealed_path).st_mode & 0o777 == 0o400


def test_private_artifacts_cannot_be_written_inside_source_tree():
    from kai.config import PROJECT_ROOT

    with pytest.raises(MemoryQualityCorpusError, match="source tree"):
        write_snapshot(PROJECT_ROOT / "forbidden-memory-corpus.json", _snapshot())


def _create_receipt_database(path):
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE principals (id TEXT, kind TEXT, display_name TEXT);
        CREATE TABLE external_identities (principal_id TEXT, external_subject TEXT);
        CREATE TABLE runtime_profile_owners (runtime_profile_id TEXT, principal_id TEXT);
        CREATE TABLE memory_extraction_receipts (
            receipt_id TEXT, principal_id TEXT, runtime_profile_id TEXT, run_id TEXT,
            source_message_id TEXT, result_message_id TEXT, extraction_role TEXT,
            backend TEXT, provider TEXT, model TEXT, prompt_version TEXT,
            schema_version TEXT, policy_version TEXT, status TEXT, decision_outcome TEXT,
            classifier_result INTEGER, proposed_intents_json TEXT,
            validation_outcome_json TEXT, storage_outcome_json TEXT,
            memory_scope_json TEXT, candidate_ids_json TEXT,
            created_at TEXT, completed_at TEXT
        );
        CREATE TABLE runs (
            id TEXT, requested_by_principal_id TEXT, channel_id TEXT, agent_id TEXT,
            inbound_message_id TEXT, result_message_id TEXT, status TEXT
        );
        CREATE TABLE channels (id TEXT, kind TEXT, name TEXT);
        CREATE TABLE agents (id TEXT, principal_id TEXT);
        CREATE TABLE agent_definitions (agent_id TEXT, handle TEXT);
        CREATE TABLE messages (id TEXT, body TEXT, created_event_position INTEGER);
        """
    )
    connection.execute("INSERT INTO principals VALUES ('prn_owner', 'human', 'Daniel')")
    connection.execute("INSERT INTO principals VALUES ('prn_agent', 'agent', 'Kai')")
    connection.execute("INSERT INTO external_identities VALUES ('prn_owner', '42')")
    connection.execute("INSERT INTO runtime_profile_owners VALUES ('rtp_owner', 'prn_owner')")
    connection.execute("INSERT INTO channels VALUES ('chn_general', 'group', 'General')")
    connection.execute("INSERT INTO agents VALUES ('agt_kai', 'prn_agent')")
    connection.execute("INSERT INTO agent_definitions VALUES ('agt_kai', 'kai')")
    connection.execute("INSERT INTO messages VALUES ('msg_source', 'Hello', 2)")
    connection.execute("INSERT INTO messages VALUES ('msg_result', 'Hi', 3)")
    connection.execute(
        "INSERT INTO runs VALUES "
        "('run_1', 'prn_owner', 'chn_general', 'agt_kai', 'msg_source', 'msg_result', 'completed')"
    )
    receipt = _receipt()
    connection.execute(
        "INSERT INTO memory_extraction_receipts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            receipt.receipt_id,
            receipt.principal_id,
            receipt.runtime_profile_id,
            "run_1",
            "msg_source",
            "msg_result",
            receipt.extraction_role,
            receipt.backend,
            receipt.provider,
            receipt.model,
            receipt.prompt_version,
            receipt.schema_version,
            receipt.policy_version,
            receipt.status,
            receipt.decision_outcome,
            0,
            json.dumps(receipt.proposed_intents),
            json.dumps(receipt.validation_outcome),
            json.dumps(receipt.storage_outcome),
            json.dumps(receipt.memory_scopes),
            json.dumps(receipt.candidate_ids),
            receipt.created_at,
            receipt.completed_at,
        ),
    )
    connection.commit()
    return connection


def test_receipt_sampling_is_principal_authorized_and_resolves_external_subject(tmp_path):
    connection = _create_receipt_database(tmp_path / "kai.db")
    try:
        assert resolve_human_principal(connection, "42") == "prn_owner"
        receipts = load_production_receipts(connection, "prn_owner", limit=10, seed=1706)
    finally:
        connection.close()

    assert len(receipts) == 1
    assert receipts[0].source_body == "Hello"
    assert receipts[0].result_body == "Hi"


def test_receipt_sampling_excludes_mismatched_runtime_owner(tmp_path):
    connection = _create_receipt_database(tmp_path / "kai.db")
    connection.execute("UPDATE runtime_profile_owners SET principal_id = 'prn_someone_else'")
    connection.commit()
    try:
        assert load_production_receipts(connection, "prn_owner", limit=10, seed=1706) == []
    finally:
        connection.close()


def test_quality_corpus_parser_exposes_all_four_stages():
    parser = memory_admin._build_parser()
    assert parser.parse_args(["quality-corpus", "sample", "prn_owner"]).quality_command == "sample"
    assert (
        parser.parse_args(["quality-corpus", "review-template", "snapshot.json"]).quality_command == "review-template"
    )
    assert (
        parser.parse_args(
            ["quality-corpus", "seal-review", "snapshot.json", "review.json", "--reviewer", "Daniel"]
        ).quality_command
        == "seal-review"
    )
    assert parser.parse_args(["quality-corpus", "score", "snapshot.json", "sealed.json"]).quality_command == "score"


def test_empty_receipt_ledger_fails_before_memory_initialization(tmp_path, monkeypatch, capsys):
    connection = _create_receipt_database(tmp_path / "kai.db")
    connection.execute("DELETE FROM memory_extraction_receipts")
    connection.commit()
    connection.close()
    monkeypatch.setattr("kai.config.load_config", lambda: SimpleNamespace(session_db_path=str(tmp_path / "kai.db")))

    def unexpected_initialization(_config=None):
        pytest.fail("empty-ledger sampling must not initialize semantic memory")

    monkeypatch.setattr(memory_admin, "_initialize_memory", unexpected_initialization)
    args = memory_admin._build_parser().parse_args(
        ["quality-corpus", "sample", "prn_owner", "--out-dir", str(tmp_path / "output")]
    )

    assert memory_admin._cmd_quality_corpus(args) == 1
    assert "No post-cutover terminal production extraction receipts" in capsys.readouterr().err


@pytest.mark.parametrize("snapshot_fails", [False, True])
def test_sample_closes_offline_memory_on_success_and_failure(tmp_path, monkeypatch, snapshot_fails):
    connection = _create_receipt_database(tmp_path / "kai.db")
    connection.close()
    config = SimpleNamespace(session_db_path=str(tmp_path / "kai.db"))
    monkeypatch.setattr("kai.config.load_config", lambda: config)
    monkeypatch.setattr(memory_admin, "_initialize_memory", lambda loaded: loaded)

    close_calls = []
    monkeypatch.setattr("kai.memory.close_memory", lambda: close_calls.append(True))
    monkeypatch.setattr(
        "kai.memory.get_by_id",
        lambda *, user_id, runtime_profile_id, memory_id: _memory(user_id, runtime_profile_id, memory_id),
    )
    if snapshot_fails:
        monkeypatch.setattr(
            "kai.memory_quality_corpus.build_snapshot",
            lambda **_kwargs: (_ for _ in ()).throw(MemoryQualityCorpusError("snapshot failed")),
        )
    args = memory_admin._build_parser().parse_args(
        ["quality-corpus", "sample", "prn_owner", "--out-dir", str(tmp_path / "output")]
    )

    expected = 1 if snapshot_fails else 0
    assert memory_admin._cmd_quality_corpus(args) == expected
    assert close_calls == [True]
