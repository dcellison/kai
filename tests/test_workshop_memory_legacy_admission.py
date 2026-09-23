"""
Temporary legacy admission in protected current-truth retrieval.

Covers:
1. The admission rules for rows without canonical lifecycle keys: opt-in,
   owner switch-off, user-visible source, absorption by canonical evidence,
   operator rejection, and explicit validity end.
2. Fail-closed handling of unreadable reconciliation state, and the
   rejected-id cache's invalidation by review version.
3. The memory call sites: search and full listings admit legacy rows,
   exact-id reads and metadata updates stay strict.
4. Recall labeling, extraction's `legacy_target_unreconciled` outcome,
   receipt aggregation, install diagnostics, and log rate limiting.

Canonical fact revisions and episodes are created through the real
lifecycle services on the real Workshop schema, so evidence and
reconciliation queries run against the columns production uses.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from kai.memory import MemoryResult
from kai.workshop.domain import (
    EventEnvelope,
    PrincipalId,
    RuntimeProfileId,
    WorkshopEventType,
    WorkshopId,
    WorkshopMembershipId,
)
from kai.workshop.episode_history import CANONICAL_EPISODE_ID_KEY, EpisodeInput, MemoryEpisodeHistoryService
from kai.workshop.fact_lifecycle import CANONICAL_REVISION_ID_KEY, FactRevisionInput, MemoryFactLifecycleService
from kai.workshop.memory_current_truth import (
    CANONICAL_TEMPORAL_ROLE_KEY,
    LEGACY_UNRECONCILED_ROLE,
    project_current_truth,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import WorkshopEventStore

NOW = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
WORKSHOP_ID = WorkshopId("wsp_22000000000000000000000000000001")
PRINCIPAL_ID = PrincipalId("prn_22000000000000000000000000000001")
RUNTIME_ID = RuntimeProfileId("rtp_22000000000000000000000000000001")
OTHER_RUNTIME_ID = RuntimeProfileId("rtp_22000000000000000000000000000002")
AUDIT_ID = "mra_legacy_admission"
PLAN_ID = "mrp_legacy_admission"
SHA = "a" * 64


# ── Fakes and fixtures ───────────────────────────────────────────────


class FakeFactVector:
    """In-memory fact vector store matching the lifecycle adapter protocol."""

    def __init__(self) -> None:
        self.rows: dict[str, MemoryResult] = {}

    async def get(self, authority, memory_id: str) -> MemoryResult | None:
        return self.rows.get(memory_id)

    async def find_revision(self, authority, revision_id) -> MemoryResult | None:
        return next(
            (row for row in self.rows.values() if row.metadata.get(CANONICAL_REVISION_ID_KEY) == str(revision_id)),
            None,
        )

    async def add(self, authority, content: str, metadata: dict[str, object]) -> str | None:
        memory_id = f"fact-vector-{len(self.rows) + 1}"
        self.rows[memory_id] = MemoryResult(memory_id, content, 0.0, "fact", dict(metadata), NOW.isoformat())
        return memory_id

    async def replace(self, authority, memory_id: str, content: str, metadata: dict[str, object]) -> bool:
        self.rows[memory_id] = MemoryResult(memory_id, content, 0.0, "fact", dict(metadata), NOW.isoformat())
        return True

    async def delete(self, authority, memory_id: str) -> bool:
        self.rows.pop(memory_id, None)
        return True


class FakeEpisodeVector:
    """In-memory episode vector store matching the history adapter protocol."""

    def __init__(self) -> None:
        self.rows: dict[str, MemoryResult] = {}

    async def find_episode(self, authority, episode_id) -> MemoryResult | None:
        return next(
            (row for row in self.rows.values() if row.metadata.get(CANONICAL_EPISODE_ID_KEY) == str(episode_id)),
            None,
        )

    async def add(self, authority, content: str, metadata: dict[str, object]) -> str | None:
        memory_id = f"episode-vector-{len(self.rows) + 1}"
        self.rows[memory_id] = MemoryResult(memory_id, content, 0.0, "episode", dict(metadata), NOW.isoformat())
        return memory_id


def _event(event_type: WorkshopEventType, aggregate_type: str, aggregate_id, payload: dict) -> EventEnvelope:
    return EventEnvelope.create(
        event_type=event_type,
        event_version=1,
        workshop_id=WORKSHOP_ID,
        aggregate_type=aggregate_type,
        aggregate_id=aggregate_id,
        occurred_at=NOW,
        payload=payload,
    )


async def _store(path: Path) -> WorkshopEventStore:
    """Open a migrated Workshop store with one owner and its runtime profile."""
    store = await WorkshopEventStore.open(path)
    await store.append(_event(WorkshopEventType.WORKSHOP_CREATED, "workshop", WORKSHOP_ID, {"name": "Legacy"}))
    await store.append(
        _event(
            WorkshopEventType.PRINCIPAL_CREATED,
            "principal",
            PRINCIPAL_ID,
            {"kind": "human", "display_name": "Operator"},
        )
    )
    await store.append(
        _event(
            WorkshopEventType.WORKSHOP_MEMBER_ADDED,
            "workshop_membership",
            WorkshopMembershipId("wmb_22000000000000000000000000000001"),
            {"principal_id": str(PRINCIPAL_ID), "role": "owner"},
        )
    )
    await store.project_pending(CanonicalConversationProjection())
    await store.connection.execute(
        "INSERT INTO runtime_profile_owners (runtime_profile_id, principal_id) VALUES (?, ?)",
        (RUNTIME_ID, PRINCIPAL_ID),
    )
    await store.connection.commit()
    return store


def _legacy(memory_id: str = "legacy-1", **metadata: object) -> MemoryResult:
    """A legacy vector row: no canonical lifecycle keys."""
    values: dict[str, object] = {"source": "extracted", "scope": "global"}
    values.update(metadata)
    return MemoryResult(memory_id, f"Legacy memory {memory_id}", 0.9, "fact", values, NOW.isoformat())


def _fact_spec(legacy_ids: tuple[str, ...]) -> FactRevisionInput:
    return FactRevisionInput(
        content="Canonical reconciled fact",
        scope_kind="global",
        scope_key="",
        reason="Operator-reviewed reconciliation of existing semantic memory.",
        evidence=tuple({"kind": "legacy", "reference_id": value, "sha256": SHA} for value in legacy_ids)
        + ({"kind": "operator", "reference_id": "receipt-1", "sha256": None},),
        vector_metadata={"source": "explicit", "speaker": "user"},
        confidence=1.0,
        asserted_at=NOW,
        observed_at=NOW,
        valid_from=NOW - timedelta(days=1),
    )


def _episode_spec(legacy_id: str) -> EpisodeInput:
    return EpisodeInput(
        goal="Deploy the memory service",
        context="A legacy episode being reconciled.",
        approach="Recorded from the legacy row.",
        outcome="It shipped.",
        outcome_quality="success",
        lessons=None,
        tags=("memory",),
        actors=("Operator",),
        scope_kind="global",
        scope_key="",
        reason="Operator-reviewed reconciliation of existing episode history.",
        evidence=(
            {"kind": "legacy", "reference_id": legacy_id, "sha256": SHA},
            {"kind": "operator", "reference_id": "receipt-1", "sha256": None},
        ),
        vector_metadata={"source": "episode"},
        occurred_from=NOW - timedelta(days=2),
        occurred_until=NOW - timedelta(days=2),
        observed_at=NOW,
    )


async def _open_audit(
    store: WorkshopEventStore,
    *,
    runtime_profile_id: str = str(RUNTIME_ID),
    audit_id: str = AUDIT_ID,
    status: str = "open",
    candidates: list[dict[str, object]] | None = None,
) -> None:
    """Insert one reconciliation audit row for the owner."""
    document = {"candidates": candidates or []}
    await store.connection.execute(
        "INSERT INTO memory_reconciliation_audits (audit_id, principal_id, runtime_profile_id, audit_sha256, "
        "corpus_sha256, generated_at, candidate_count, audit_json, status, review_version, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)",
        (
            audit_id,
            str(PRINCIPAL_ID),
            runtime_profile_id,
            SHA,
            ("b" if audit_id == AUDIT_ID else "c") * 64,
            NOW.isoformat(),
            len(document["candidates"]),
            json.dumps(document),
            status,
            NOW.isoformat(),
        ),
    )
    await store.connection.commit()


async def _audit_decision(store: WorkshopEventStore, candidate_id: str, disposition: str) -> None:
    await store.connection.execute(
        "INSERT INTO memory_reconciliation_decisions (audit_id, candidate_id, state_sha256, disposition, "
        "action_json, updated_at) VALUES (?, ?, ?, ?, '{}', ?)",
        (AUDIT_ID, candidate_id, SHA, disposition, NOW.isoformat()),
    )
    await store.connection.commit()


async def _open_plan(store: WorkshopEventStore, groups: dict[str, tuple[str, list[str]]], *, plan_json=None) -> None:
    """
    Insert an open triage plan with the given groups.

    Args:
        store: Migrated store with an open audit already inserted.
        groups: group_id -> (disposition, member memory ids).
        plan_json: Optional raw plan JSON, to simulate malformed documents.
    """
    document = plan_json or json.dumps(
        {
            "groups": [
                {"group_id": group_id, "evidence": [{"memory_id": member} for member in members]}
                for group_id, (_disposition, members) in groups.items()
            ]
        }
    )
    await store.connection.execute(
        "INSERT INTO memory_reconciliation_triage_plans (plan_id, audit_id, plan_sha256, policy_version, "
        "group_count, memory_count, plan_json, status, review_version, created_at) "
        "VALUES (?, ?, ?, 'v1', ?, ?, ?, 'open', 0, ?)",
        (PLAN_ID, AUDIT_ID, SHA, len(groups), sum(len(m) for _d, m in groups.values()), document, NOW.isoformat()),
    )
    for group_id, (disposition, members) in groups.items():
        await store.connection.execute(
            "INSERT INTO memory_reconciliation_triage_groups (plan_id, group_id, state_sha256, classification, "
            "resolution, deterministic, bulk_eligible, memory_count, disposition, action_json, updated_at) "
            "VALUES (?, ?, ?, 'uncertain_scope', 'needs_review', 0, 0, ?, ?, '{}', ?)",
            (PLAN_ID, group_id, SHA, len(members), disposition, NOW.isoformat()),
        )
    await store.connection.commit()


def _project(path: Path, rows, *, admit_legacy: bool = True, runtime_profile_id: str = str(RUNTIME_ID)):
    return project_current_truth(
        rows,
        db_path=path,
        principal_id=str(PRINCIPAL_ID),
        runtime_profile_id=runtime_profile_id,
        now=NOW,
        admit_legacy=admit_legacy,
    )


# ── Admission rules ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_legacy_rows_stay_excluded_without_opt_in(tmp_path: Path) -> None:
    path = tmp_path / "kai.db"
    store = await _store(path)
    await store.close()

    result = _project(path, (_legacy(),), admit_legacy=False)

    assert result.rows == ()
    assert result.excluded == {"legacy_unclassified": 1}
    assert result.admitted == {}


@pytest.mark.asyncio
async def test_opted_in_legacy_row_is_admitted_and_labeled_without_any_audit(tmp_path: Path) -> None:
    path = tmp_path / "kai.db"
    store = await _store(path)
    await store.close()

    result = _project(path, (_legacy(),))

    assert [row.id for row in result.rows] == ["legacy-1"]
    assert result.rows[0].metadata[CANONICAL_TEMPORAL_ROLE_KEY] == LEGACY_UNRECONCILED_ROLE
    assert result.admitted == {"legacy_admitted": 1}
    assert result.excluded == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("source", ["", None, "exchange"])
async def test_legacy_row_without_user_visible_source_is_excluded(tmp_path: Path, source) -> None:
    path = tmp_path / "kai.db"
    store = await _store(path)
    await store.close()

    result = _project(path, (_legacy(source=source),))

    assert result.rows == ()
    assert result.excluded == {"legacy_invalid_source": 1}


@pytest.mark.asyncio
@pytest.mark.parametrize("retract", [False, True])
async def test_legacy_row_cited_by_any_fact_revision_is_absorbed(tmp_path: Path, retract: bool) -> None:
    path = tmp_path / "kai.db"
    store = await _store(path)
    service = MemoryFactLifecycleService(store, FakeFactVector())
    authority = await service.authority_for(PRINCIPAL_ID, RUNTIME_ID)
    created = await service.create(
        authority,
        _fact_spec(("legacy-1", "legacy-2")),
        idempotency_key="reconcile:consolidate",
        stable_claim_key="reconciled:group-1",
    )
    if retract:
        await service.retract(
            authority,
            created.claim_id,
            created.revision_id,
            reason="Retired during reconciliation.",
            idempotency_key="reconcile:retire",
        )
    await store.close()

    result = _project(path, (_legacy("legacy-1"), _legacy("legacy-2"), _legacy("legacy-3")))

    assert [row.id for row in result.rows] == ["legacy-3"]
    assert result.excluded == {"legacy_absorbed": 2}


@pytest.mark.asyncio
async def test_reconciled_episode_hides_its_legacy_source_without_duplicating(tmp_path: Path) -> None:
    path = tmp_path / "kai.db"
    store = await _store(path)
    vector = FakeEpisodeVector()
    service = MemoryEpisodeHistoryService(store, vector)
    authority = await service.authority_for(PRINCIPAL_ID, RUNTIME_ID)
    await service.record(authority, _episode_spec("legacy-episode"), idempotency_key="reconcile:episode")
    await store.close()
    canonical_row = next(iter(vector.rows.values()))

    result = _project(path, (_legacy("legacy-episode", source="episode"), canonical_row))

    assert [row.id for row in result.rows] == [canonical_row.id]
    assert result.excluded == {"legacy_absorbed": 1}


@pytest.mark.asyncio
async def test_rejected_members_are_excluded_while_deferred_and_pending_are_admitted(tmp_path: Path) -> None:
    path = tmp_path / "kai.db"
    store = await _store(path)
    await _open_audit(
        store,
        candidates=[
            {"candidate_id": "cand-reject", "evidence": [{"memory_id": "legacy-audit-rejected"}]},
            {"candidate_id": "cand-defer", "evidence": [{"memory_id": "legacy-audit-deferred"}]},
        ],
    )
    await _audit_decision(store, "cand-reject", "reject")
    await _audit_decision(store, "cand-defer", "defer")
    await _open_plan(
        store,
        {
            "group-reject": ("reject", ["legacy-group-rejected"]),
            "group-pending": ("pending", ["legacy-group-pending"]),
        },
    )
    await store.close()

    result = _project(
        path,
        (
            _legacy("legacy-audit-rejected"),
            _legacy("legacy-audit-deferred"),
            _legacy("legacy-group-rejected"),
            _legacy("legacy-group-pending"),
        ),
    )

    assert sorted(row.id for row in result.rows) == ["legacy-audit-deferred", "legacy-group-pending"]
    assert result.excluded == {"legacy_rejected": 2}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("valid_until", "admitted"),
    [
        ((NOW - timedelta(minutes=1)).isoformat(), False),
        (NOW.isoformat(), False),
        ((NOW + timedelta(days=1)).isoformat(), True),
        ("2026-09-01T00:00:00", True),
        ("not a timestamp", True),
    ],
)
async def test_only_an_explicit_aware_past_validity_end_excludes(tmp_path: Path, valid_until: str, admitted: bool):
    path = tmp_path / "kai.db"
    store = await _store(path)
    await store.close()

    result = _project(path, (_legacy(valid_until=valid_until),))

    assert bool(result.rows) is admitted
    if not admitted:
        assert result.excluded == {"legacy_expired": 1}


@pytest.mark.asyncio
async def test_applied_audit_switches_admission_off_only_for_that_owner(tmp_path: Path) -> None:
    path = tmp_path / "kai.db"
    store = await _store(path)
    await _open_audit(store, status="applied")
    await store.close()

    same_owner = _project(path, (_legacy(),))
    other_runtime = _project(path, (_legacy(),), runtime_profile_id=str(OTHER_RUNTIME_ID))

    assert same_owner.rows == ()
    assert same_owner.excluded == {"legacy_unclassified": 1}
    assert [row.id for row in other_runtime.rows] == ["legacy-1"]


@pytest.mark.asyncio
async def test_later_open_audit_does_not_reenable_admission(tmp_path: Path) -> None:
    path = tmp_path / "kai.db"
    store = await _store(path)
    await _open_audit(store, status="applied")
    await _open_audit(store, audit_id="mra_later", status="open")
    await store.close()

    assert _project(path, (_legacy(),)).rows == ()


# ── Fail-closed handling and caching ─────────────────────────────────


@pytest.mark.asyncio
async def test_malformed_plan_disables_legacy_admission_but_not_canonical_rows(tmp_path: Path) -> None:
    path = tmp_path / "kai.db"
    store = await _store(path)
    vector = FakeFactVector()
    service = MemoryFactLifecycleService(store, vector)
    authority = await service.authority_for(PRINCIPAL_ID, RUNTIME_ID)
    await service.create(authority, _fact_spec(()), idempotency_key="canonical", stable_claim_key="canonical")
    await _open_audit(store)
    await _open_plan(store, {"group-reject": ("reject", ["legacy-2"])}, plan_json='{"groups": "not a list"}')
    await store.close()
    canonical_row = next(iter(vector.rows.values()))

    result = _project(path, (_legacy("legacy-1"), canonical_row))

    assert [row.id for row in result.rows] == [canonical_row.id]
    assert result.excluded == {"legacy_unclassified": 1}


@pytest.mark.asyncio
async def test_rejected_cache_follows_the_plan_review_version(tmp_path: Path) -> None:
    import sqlite3

    path = tmp_path / "kai.db"
    store = await _store(path)
    await _open_audit(store)
    await _open_plan(store, {"group-1": ("reject", ["legacy-1"])})
    await store.close()

    assert _project(path, (_legacy(),)).excluded == {"legacy_rejected": 1}

    # An operator decision changes the disposition and bumps the plan's
    # review version in the same transaction, as the triage service does.
    connection = sqlite3.connect(path)
    connection.execute("UPDATE memory_reconciliation_triage_groups SET disposition = 'pending'")
    connection.execute("UPDATE memory_reconciliation_triage_plans SET review_version = review_version + 1")
    connection.commit()
    connection.close()

    assert [row.id for row in _project(path, (_legacy(),)).rows] == ["legacy-1"]


# ── Memory call sites ────────────────────────────────────────────────


def _configured_memory(monkeypatch, database: Path, raw: dict[str, object]):
    """
    Point the memory module at a mocked provider in protected mode.

    Returns the memory module and the namespace for the configured owner.
    Uses monkeypatch so module globals are restored after each test.
    """
    import kai.memory as memory
    from kai.config import Config, DeploymentMode
    from kai.workshop.domain import AgentId, ChannelId
    from kai.workshop.execution_state import WorkshopExecutionStateNamespace, WorkshopExecutionStateRegistry

    namespace = WorkshopExecutionStateNamespace(
        principal_id=PRINCIPAL_ID,
        channel_id=ChannelId("chn_22000000000000000000000000000001"),
        agent_id=AgentId("agt_22000000000000000000000000000001"),
        runtime_profile_id=RUNTIME_ID,
        legacy_runtime_key=1,
    )
    raw = {**raw, "metadata": {**memory._owner_metadata(namespace), **raw["metadata"]}}  # type: ignore[dict-item]
    provider = MagicMock()
    provider.search.return_value = {"results": [raw]}
    provider.get_all.return_value = {"results": [raw]}
    provider.get.return_value = raw
    monkeypatch.setattr(memory, "_memory", provider)
    monkeypatch.setattr(
        memory,
        "_config",
        Config(
            telegram_bot_token="token",
            allowed_user_ids={1},
            memory_enabled=True,
            deployment_mode=DeploymentMode.PROTECTED,
            session_db_path=database,
        ),
    )
    memory.configure_memory_authority(WorkshopExecutionStateRegistry((namespace,)))
    return memory, namespace


@pytest.mark.asyncio
async def test_read_surfaces_admit_legacy_while_exact_reads_stay_strict_by_default(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "kai.db"
    store = await _store(path)
    await store.close()
    raw = {
        "id": "legacy-1",
        "memory": "A legacy preference",
        "score": 0.9,
        "metadata": {"source": "extracted", "scope": "global"},
        "created_at": NOW.isoformat(),
        "user_id": str(PRINCIPAL_ID),
    }
    memory, _namespace = _configured_memory(monkeypatch, path, raw)
    try:
        searched = memory.search("preference", user_id=str(PRINCIPAL_ID), runtime_profile_id=str(RUNTIME_ID))
        listed = memory.get_all(user_id=str(PRINCIPAL_ID), runtime_profile_id=str(RUNTIME_ID))
        exact = memory.get_by_id(user_id=str(PRINCIPAL_ID), memory_id="legacy-1", runtime_profile_id=str(RUNTIME_ID))
        read_only = memory.get_by_id(
            user_id=str(PRINCIPAL_ID),
            memory_id="legacy-1",
            runtime_profile_id=str(RUNTIME_ID),
            admit_legacy=True,
        )
        updated = memory.update_metadata(
            user_id=str(PRINCIPAL_ID),
            memory_id="legacy-1",
            data="changed",
            metadata={"scope": "global"},
            runtime_profile_id=str(RUNTIME_ID),
        )

        assert [row.id for row in searched] == ["legacy-1"]
        assert [row.id for row in listed] == ["legacy-1"]
        assert searched[0].metadata[CANONICAL_TEMPORAL_ROLE_KEY] == LEGACY_UNRECONCILED_ROLE
        assert exact is None
        assert read_only is not None
        assert read_only.metadata[CANONICAL_TEMPORAL_ROLE_KEY] == LEGACY_UNRECONCILED_ROLE
        assert updated is False
    finally:
        memory.configure_memory_authority(None)


def test_recall_marks_admitted_legacy_rows_as_not_current_truth() -> None:
    from kai.memory import format_memory_result_for_recall

    rendered = format_memory_result_for_recall(
        _legacy(**{CANONICAL_TEMPORAL_ROLE_KEY: LEGACY_UNRECONCILED_ROLE}),
    )

    assert '"temporal_role":"legacy_unreconciled"' in rendered
    assert '"current_truth":false' in rendered


def test_count_logging_is_rate_limited_per_owner(monkeypatch, caplog) -> None:
    import kai.memory as memory
    from kai.workshop.memory_current_truth import CurrentTruthProjection

    clock = [100.0]
    monkeypatch.setattr(memory, "_current_truth_log_clock", lambda: clock[0])
    monkeypatch.setattr(memory, "_current_truth_log_times", {})
    namespace = SimpleNamespace(principal_id=PRINCIPAL_ID, runtime_profile_id=RUNTIME_ID)
    projection = CurrentTruthProjection((), {"legacy_absorbed": 1}, {"legacy_admitted": 2})

    with caplog.at_level(logging.INFO, logger="kai.memory"):
        memory._log_current_truth_counts("search", namespace, projection)
        clock[0] += 30
        memory._log_current_truth_counts("search", namespace, projection)
        clock[0] += 31
        memory._log_current_truth_counts("search", namespace, projection)
        memory._log_current_truth_counts("search", namespace, CurrentTruthProjection((), {}))

    lines = [record.getMessage() for record in caplog.records if "Memory current truth" in record.getMessage()]
    assert len(lines) == 2
    assert "legacy_admitted" in lines[0]
    assert "Legacy memory" not in "".join(lines)


# ── Extraction and receipts ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_of_admitted_legacy_candidate_records_outcome_without_mutation(monkeypatch) -> None:
    from kai import memory_extraction, sessions
    from kai.config import Config

    monkeypatch.setattr(memory_extraction.memory, "get_by_id", lambda **_kwargs: None)

    async def apply(*_args, **_kwargs):
        pytest.fail("an unreconciled legacy target must not reach lifecycle authority")

    monkeypatch.setattr(sessions, "apply_canonical_extracted_fact", apply)
    decisions: list = []
    fact = {
        "content": "The corrected preference",
        "intent": "update_of",
        "existing_id": "legacy-1",
        "speaker": "user",
        "confidence": 0.98,
        "scope_hint": "global",
    }
    await memory_extraction._store_canonical_facts(
        [fact, {**fact, "existing_id": "gone-1"}],
        user_id="prn_" + "1" * 32,
        session_id="session-1",
        config=Config(telegram_bot_token="token", allowed_user_ids={1}),
        active_project=None,
        user_log=None,
        assistant_log=None,
        canonical_provenance={
            memory_extraction.memory.WORKSHOP_RUN_ID_KEY: "run_" + "2" * 32,
            memory_extraction.memory.WORKSHOP_SOURCE_MESSAGE_ID_KEY: "msg_" + "3" * 32,
            memory_extraction.memory.WORKSHOP_RESULT_MESSAGE_ID_KEY: "msg_" + "4" * 32,
        },
        runtime_profile_id="rtp_" + "5" * 32,
        receipt_id="mxr_" + "6" * 32,
        backend="codex",
        provider="openai",
        model="gpt-5.6-sol",
        receipt_decisions=decisions,
        legacy_candidate_ids=frozenset({"legacy-1"}),
    )

    assert [decision.outcome for decision in decisions] == ["legacy_target_unreconciled", "stale_existing_fact"]


def test_receipt_reports_legacy_target_when_it_is_the_only_outcome() -> None:
    from kai.memory_extraction import ExtractionResult, _fact_receipt_completion
    from kai.workshop.memory_extraction_receipts import MemoryExtractionStorageDecision

    only = _fact_receipt_completion(
        result=ExtractionResult([{"intent": "update_of"}], False, raw_fact_count=1),
        candidate_ids={"legacy-1"},
        decisions=[MemoryExtractionStorageDecision(0, "update_of", "legacy_target_unreconciled")],
        stored=0,
        replaced=0,
        skipped=0,
        duration_ms=10,
    )
    mixed = _fact_receipt_completion(
        result=ExtractionResult([{"intent": "update_of"}, {"intent": "new"}], False, raw_fact_count=2),
        candidate_ids={"legacy-1"},
        decisions=[
            MemoryExtractionStorageDecision(0, "update_of", "legacy_target_unreconciled"),
            MemoryExtractionStorageDecision(1, "new", "stored"),
        ],
        stored=1,
        replaced=0,
        skipped=0,
        duration_ms=10,
    )

    assert (only.status, only.decision_outcome, only.failure_code) == (
        "completed",
        "legacy_target_unreconciled",
        None,
    )
    assert mixed.decision_outcome == "mixed"


# ── Install diagnostics ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_diagnostics_count_owners_with_active_and_inactive_legacy_admission(tmp_path: Path) -> None:
    from kai.workshop.diagnostics import workshop_memory_current_truth_status

    path = tmp_path / "kai.db"
    store = await _store(path)
    await _open_audit(store, runtime_profile_id=str(OTHER_RUNTIME_ID), status="applied")
    await store.close()

    status = workshop_memory_current_truth_status(path, memory_enabled=True)

    assert "legacy admission active=1 (no applied audit), inactive=1 (audit applied)" in status
