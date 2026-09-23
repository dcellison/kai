"""Canonical current-truth admission for semantic-memory reads."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from kai.memory import MemoryResult
from kai.workshop.memory_current_truth import (
    CANONICAL_ADMISSION_AUTHORITY_KEY,
    CANONICAL_CLAIM_ID_KEY,
    CANONICAL_LIFECYCLE_STATE_KEY,
    CANONICAL_REVISION_ID_KEY,
    current_truth_revision,
    project_current_truth,
)

PRINCIPAL = "prn_20000000000000000000000000000001"
RUNTIME = "rtp_20000000000000000000000000000001"
CLAIM = "mcl_20000000000000000000000000000001"
REVISION = "mrv_20000000000000000000000000000001"
NOW = datetime(2026, 9, 21, 14, 0, tzinfo=UTC)


def _database(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE memory_fact_claims (
            claim_id TEXT PRIMARY KEY,
            owner_principal_id TEXT NOT NULL,
            runtime_profile_id TEXT NOT NULL,
            scope_kind TEXT NOT NULL,
            scope_key TEXT NOT NULL
        );
        CREATE TABLE memory_fact_revisions (
            revision_id TEXT PRIMARY KEY,
            claim_id TEXT NOT NULL,
            content TEXT NOT NULL,
            valid_from TEXT,
            valid_until TEXT,
            migration_classification TEXT NOT NULL,
            admission_authority TEXT NOT NULL,
            vector_metadata_json TEXT NOT NULL,
            source_receipt_id TEXT,
            source_run_id TEXT,
            source_message_id TEXT,
            result_message_id TEXT,
            backend TEXT,
            provider TEXT,
            model TEXT,
            prompt_version TEXT,
            schema_version TEXT
        );
        CREATE TABLE memory_fact_revision_states (
            revision_id TEXT PRIMARY KEY,
            claim_id TEXT NOT NULL,
            state TEXT NOT NULL,
            state_event_position INTEGER NOT NULL
        );
        CREATE TABLE memory_fact_lifecycle_events (event_position INTEGER PRIMARY KEY);
        CREATE TABLE memory_fact_vector_operations (
            event_position INTEGER PRIMARY KEY,
            claim_id TEXT NOT NULL,
            revision_id TEXT NOT NULL,
            operation TEXT NOT NULL,
            status TEXT NOT NULL,
            memory_id TEXT
        );
        CREATE TABLE workshop_memory_authority_migrations (total_count INTEGER NOT NULL);
        INSERT INTO workshop_memory_authority_migrations VALUES (0);
        CREATE TABLE memory_legacy_census (
            principal_id TEXT NOT NULL,
            runtime_profile_id TEXT NOT NULL,
            legacy_rows INTEGER NOT NULL,
            absorbed INTEGER NOT NULL,
            rejected INTEGER NOT NULL,
            unclassified INTEGER NOT NULL,
            counted_at TEXT NOT NULL,
            PRIMARY KEY (principal_id, runtime_profile_id)
        );
        -- The service counts legacy rows at startup; here nothing is unclassified.
        INSERT INTO memory_legacy_census VALUES ('prn_x', 'rtp_x', 0, 0, 0, 0, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'));
        CREATE TABLE memory_reconciliation_audits (
            audit_id TEXT PRIMARY KEY,
            principal_id TEXT NOT NULL,
            runtime_profile_id TEXT NOT NULL,
            status TEXT NOT NULL
        );
        CREATE TABLE memory_vector_audit (
            principal_id TEXT NOT NULL,
            runtime_profile_id TEXT NOT NULL,
            orphan_rows INTEGER NOT NULL,
            unknown_rows INTEGER NOT NULL,
            duplicate_items INTEGER NOT NULL,
            missing_rows INTEGER NOT NULL,
            checked_at TEXT NOT NULL,
            PRIMARY KEY (principal_id, runtime_profile_id)
        );
        -- The service audits vector rows at startup; here the store matches.
        INSERT INTO memory_vector_audit VALUES ('prn_x', 'rtp_x', 0, 0, 0, 0, strftime('%Y-%m-%dT%H:%M:%SZ', 'now'));
        """
    )
    return connection


def _insert_revision(
    connection: sqlite3.Connection,
    *,
    state: str = "active",
    status: str = "succeeded",
    classification: str = "canonical",
    admission_authority: str | None = None,
    valid_from: datetime | None = None,
    valid_until: datetime | None = None,
) -> None:
    connection.execute(
        "INSERT INTO memory_fact_claims VALUES (?, ?, ?, 'project', 'kai')",
        (CLAIM, PRINCIPAL, RUNTIME),
    )
    connection.execute(
        "INSERT INTO memory_fact_revisions VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            REVISION,
            CLAIM,
            "Canonical current content",
            valid_from.isoformat() if valid_from is not None else None,
            valid_until.isoformat() if valid_until is not None else None,
            classification,
            admission_authority
            or ("provenance_verified" if classification in {"canonical", "legacy_complete"} else "quarantined"),
            '{"confidence":0.95,"scope":"global","scope_source":"operator","source":"explicit"}',
            "receipt-1",
            "run-1",
            "message-1",
            "message-2",
            "codex",
            "openai",
            "gpt-5.6-sol",
            "v1",
            "v1",
        ),
    )
    connection.execute(
        "INSERT INTO memory_fact_revision_states VALUES (?, ?, ?, 7)",
        (REVISION, CLAIM, state),
    )
    connection.execute("INSERT INTO memory_fact_lifecycle_events VALUES (7)")
    connection.execute(
        "INSERT INTO memory_fact_vector_operations VALUES (7, ?, ?, 'upsert', ?, 'memory-1')",
        (CLAIM, REVISION, status),
    )
    connection.commit()


def _row(metadata: dict[str, object] | None = None) -> MemoryResult:
    return MemoryResult(
        id="memory-1",
        text="Stale vector content",
        score=0.91,
        memory_type="fact",
        metadata=metadata
        or {
            CANONICAL_CLAIM_ID_KEY: CLAIM,
            CANONICAL_REVISION_ID_KEY: REVISION,
            CANONICAL_LIFECYCLE_STATE_KEY: "active",
            "scope": "global",
            "confidence": 0.1,
            "source": "explicit",
        },
        created_at=NOW.isoformat(),
    )


def test_projection_uses_canonical_content_scope_provenance_confidence_and_validity(tmp_path: Path) -> None:
    database = tmp_path / "kai.db"
    connection = _database(database)
    _insert_revision(connection, valid_from=NOW - timedelta(days=1), valid_until=NOW + timedelta(days=1))
    connection.close()

    result = project_current_truth(
        (_row(),),
        db_path=database,
        principal_id=PRINCIPAL,
        runtime_profile_id=RUNTIME,
        now=NOW,
    )

    assert result.excluded == {}
    assert len(result.rows) == 1
    admitted = result.rows[0]
    assert admitted.text == "Canonical current content"
    assert admitted.metadata["scope"] == "project"
    assert admitted.metadata["project_id"] == "kai"
    assert admitted.metadata["confidence"] == 0.95
    assert admitted.metadata["source_receipt_id"] == "receipt-1"
    assert admitted.metadata["valid_until"] == (NOW + timedelta(days=1)).isoformat()
    assert admitted.metadata["canonical_memory_temporal_role"] == "current_fact"


@pytest.mark.parametrize("state", ["superseded", "retracted", "expired", "unresolved_conflict"])
def test_projection_excludes_every_noncurrent_lifecycle_state(tmp_path: Path, state: str) -> None:
    database = tmp_path / "kai.db"
    connection = _database(database)
    _insert_revision(connection, state=state)
    connection.close()

    result = project_current_truth(
        (_row(),),
        db_path=database,
        principal_id=PRINCIPAL,
        runtime_profile_id=RUNTIME,
        now=NOW,
    )

    assert result.rows == ()
    assert sum(result.excluded.values()) == 1


@pytest.mark.parametrize(
    ("classification", "valid_from", "valid_until"),
    [
        ("legacy_incomplete", None, None),
        ("legacy_quarantined", None, None),
        ("canonical", NOW + timedelta(minutes=1), None),
        ("canonical", None, NOW),
    ],
)
def test_projection_excludes_quarantine_and_invalid_time_windows(
    tmp_path: Path,
    classification: str,
    valid_from: datetime | None,
    valid_until: datetime | None,
) -> None:
    database = tmp_path / "kai.db"
    connection = _database(database)
    _insert_revision(
        connection,
        classification=classification,
        valid_from=valid_from,
        valid_until=valid_until,
    )
    connection.close()
    result = project_current_truth(
        (_row(),),
        db_path=database,
        principal_id=PRINCIPAL,
        runtime_profile_id=RUNTIME,
        now=NOW,
    )
    assert result.rows == ()


def test_operator_review_admits_incomplete_fact_without_rewriting_provenance(tmp_path: Path) -> None:
    from kai.workshop.diagnostics import workshop_memory_current_truth_status

    database = tmp_path / "kai.db"
    connection = _database(database)
    _insert_revision(
        connection,
        classification="legacy_incomplete",
        admission_authority="operator_review",
    )
    connection.close()

    result = project_current_truth(
        (_row(),),
        db_path=database,
        principal_id=PRINCIPAL,
        runtime_profile_id=RUNTIME,
        now=NOW,
    )

    assert result.excluded == {}
    assert len(result.rows) == 1
    admitted = result.rows[0]
    assert admitted.metadata["migration_classification"] == "legacy_incomplete"
    assert admitted.metadata[CANONICAL_ADMISSION_AUTHORITY_KEY] == "operator_review"
    status = workshop_memory_current_truth_status(database, memory_enabled=True)
    assert status.startswith("Workshop memory current truth: active;")
    assert "current=1" in status
    assert "quarantined=0" in status
    assert "operator admitted=1" in status


def test_projection_fails_closed_for_legacy_malformed_and_unavailable_authority(tmp_path: Path) -> None:
    legacy = _row({"source": "extracted"})
    malformed = _row({CANONICAL_CLAIM_ID_KEY: CLAIM})
    result = project_current_truth(
        (legacy, malformed),
        db_path=tmp_path / "missing.db",
        principal_id=PRINCIPAL,
        runtime_profile_id=RUNTIME,
        now=NOW,
    )
    assert result.rows == ()
    assert result.excluded == {"authority_unavailable": 2}

    database = tmp_path / "kai.db"
    connection = _database(database)
    connection.close()
    result = project_current_truth(
        (legacy, malformed),
        db_path=database,
        principal_id=PRINCIPAL,
        runtime_profile_id=RUNTIME,
        now=NOW,
    )
    assert result.excluded == {"legacy_unclassified": 1, "malformed_lifecycle": 1}


def test_projection_withholds_pending_vector_and_revision_is_restart_stable(tmp_path: Path) -> None:
    database = tmp_path / "kai.db"
    connection = _database(database)
    _insert_revision(connection, status="failed")
    connection.close()
    first_revision = current_truth_revision(database, principal_id=PRINCIPAL, runtime_profile_id=RUNTIME)
    second_revision = current_truth_revision(database, principal_id=PRINCIPAL, runtime_profile_id=RUNTIME)
    result = project_current_truth(
        (_row(),),
        db_path=database,
        principal_id=PRINCIPAL,
        runtime_profile_id=RUNTIME,
        now=NOW,
    )
    assert first_revision == second_revision
    assert len(first_revision) == 64
    assert result.rows == ()
    assert result.excluded == {"projection_not_current": 1}

    connection = sqlite3.connect(database)
    connection.execute("UPDATE memory_fact_vector_operations SET status = 'succeeded'")
    connection.commit()
    connection.close()
    recovered_revision = current_truth_revision(
        database,
        principal_id=PRINCIPAL,
        runtime_profile_id=RUNTIME,
    )
    assert recovered_revision != first_revision

    connection = sqlite3.connect(database)
    connection.execute("UPDATE memory_fact_revision_states SET state = 'retracted', state_event_position = 8")
    connection.execute("INSERT INTO memory_fact_lifecycle_events VALUES (8)")
    connection.commit()
    connection.close()
    changed_revision = current_truth_revision(
        database,
        principal_id=PRINCIPAL,
        runtime_profile_id=RUNTIME,
    )
    assert changed_revision != recovered_revision


def test_protected_search_and_get_all_share_current_truth_projection(tmp_path: Path) -> None:
    import kai.memory as memory
    from kai.config import Config, DeploymentMode
    from kai.workshop.domain import AgentId, ChannelId, PrincipalId, RuntimeProfileId
    from kai.workshop.execution_state import WorkshopExecutionStateNamespace, WorkshopExecutionStateRegistry

    database = tmp_path / "kai.db"
    connection = _database(database)
    _insert_revision(connection)
    connection.close()
    namespace = WorkshopExecutionStateNamespace(
        principal_id=PrincipalId(PRINCIPAL),
        channel_id=ChannelId("chn_20000000000000000000000000000001"),
        agent_id=AgentId("agt_20000000000000000000000000000001"),
        runtime_profile_id=RuntimeProfileId(RUNTIME),
        legacy_runtime_key=1,
    )
    metadata = {
        **memory._owner_metadata(namespace),
        CANONICAL_CLAIM_ID_KEY: CLAIM,
        CANONICAL_REVISION_ID_KEY: REVISION,
        CANONICAL_LIFECYCLE_STATE_KEY: "active",
        "source": "explicit",
    }
    raw = {
        "id": "memory-1",
        "memory": "Stale vector content",
        "score": 0.9,
        "metadata": metadata,
        "created_at": NOW.isoformat(),
        "user_id": PRINCIPAL,
    }
    provider = MagicMock()
    provider.search.return_value = {"results": [raw]}
    provider.get_all.return_value = {"results": [raw]}
    prior_memory, prior_config = memory._memory, memory._config
    memory._memory = provider
    memory._config = Config(
        telegram_bot_token="token",
        allowed_user_ids={1},
        memory_enabled=True,
        deployment_mode=DeploymentMode.PROTECTED,
        session_db_path=database,
    )
    memory.configure_memory_authority(WorkshopExecutionStateRegistry((namespace,)))
    try:
        searched = memory.search("current content", user_id=PRINCIPAL, runtime_profile_id=RUNTIME)
        listed = memory.get_all(user_id=PRINCIPAL, runtime_profile_id=RUNTIME)
        assert [row.text for row in searched] == ["Canonical current content"]
        assert [row.text for row in listed] == ["Canonical current content"]
    finally:
        memory.configure_memory_authority(None)
        memory._memory, memory._config = prior_memory, prior_config


def test_episode_rendering_explicitly_marks_history() -> None:
    from kai.memory import format_memory_result_for_recall

    rendered = format_memory_result_for_recall(
        MemoryResult(
            "episode-1",
            "A prior deployment",
            0.8,
            "episode",
            {"source": "episode", "goal": "Deploy Kai", "outcome": "Succeeded"},
            NOW.isoformat(),
        )
    )
    assert '"temporal_role":"historical_episode"' in rendered
    assert '"current_truth":false' in rendered


def test_status_surfaces_quarantine_projection_and_legacy_gaps(tmp_path: Path) -> None:
    from kai.workshop.diagnostics import workshop_memory_current_truth_status

    database = tmp_path / "kai.db"
    connection = _database(database)
    _insert_revision(connection)
    connection.close()
    active = workshop_memory_current_truth_status(database, memory_enabled=True)
    assert active.startswith("Workshop memory current truth: active;")
    assert "current=1" in active
    assert "legacy unclassified=0 (counted 0m ago)" in active

    connection = sqlite3.connect(database)
    connection.execute(
        "UPDATE memory_fact_revisions SET migration_classification = 'legacy_quarantined', "
        "admission_authority = 'quarantined'"
    )
    # The latest census found two legacy rows nothing has settled, for an
    # owner who applied a reconciliation audit, so they are gaps.
    connection.execute("UPDATE memory_legacy_census SET legacy_rows = 2, unclassified = 2")
    connection.execute("INSERT INTO memory_reconciliation_audits VALUES ('mra_x', 'prn_x', 'rtp_x', 'applied')")
    connection.commit()
    connection.close()
    incomplete = workshop_memory_current_truth_status(database, memory_enabled=True)
    assert incomplete.startswith("Workshop memory current truth: INCOMPLETE;")
    assert "quarantined=1" in incomplete
    assert "legacy unclassified=2 (counted 0m ago)" in incomplete

    connection = sqlite3.connect(database)
    connection.execute("DELETE FROM memory_legacy_census")
    connection.commit()
    connection.close()
    uncounted = workshop_memory_current_truth_status(database, memory_enabled=True)
    assert "legacy unclassified=not counted" in uncounted and "INCOMPLETE" in uncounted


def test_status_reports_owners_awaiting_review_without_marking_gaps(tmp_path: Path) -> None:
    # An owner who never applied a reconciliation audit keeps temporary
    # legacy admission, so their unclassified rows wait on their own review
    # and are not integrity gaps.
    from kai.workshop.diagnostics import workshop_memory_current_truth_status

    database = tmp_path / "kai.db"
    connection = _database(database)
    _insert_revision(connection)
    connection.execute("UPDATE memory_legacy_census SET legacy_rows = 7, unclassified = 7")
    connection.commit()
    connection.close()

    status = workshop_memory_current_truth_status(database, memory_enabled=True)

    assert status.startswith("Workshop memory current truth: active;")
    assert "legacy unclassified=0 (counted 0m ago)" in status
    assert "legacy review pending=1 owner(s) (7 rows)" in status


def test_status_counts_revisions_per_lifecycle_state(tmp_path: Path) -> None:
    from kai.workshop.diagnostics import workshop_memory_current_truth_status

    database = tmp_path / "kai.db"
    connection = _database(database)
    _insert_revision(connection)
    connection.execute(
        "INSERT INTO memory_fact_revision_states VALUES ('mrv_old', 'mcl_x', 'superseded', 1), "
        "('mrv_gone', 'mcl_y', 'retracted', 2), ('mrv_split', 'mcl_z', 'unresolved_conflict', 3)"
    )
    connection.commit()
    connection.close()

    status = workshop_memory_current_truth_status(database, memory_enabled=True)

    assert "states=(active=1, superseded=1, retracted=1, expired=0, conflicted=1)" in status


def test_status_without_a_vector_audit_is_incomplete(tmp_path: Path) -> None:
    from kai.workshop.diagnostics import workshop_memory_current_truth_status

    database = tmp_path / "kai.db"
    connection = _database(database)
    _insert_revision(connection)
    connection.execute("DELETE FROM memory_vector_audit")
    connection.commit()
    connection.close()

    status = workshop_memory_current_truth_status(database, memory_enabled=True)

    assert status.startswith("Workshop memory current truth: INCOMPLETE;")
    assert "vector drift=not checked" in status
