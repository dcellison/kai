"""
Real-path canonical fact writes on a protected install.

These tests run extraction storage and lifecycle projection through the
real layers that earlier tests replaced with stubs:

1. `memory_extraction._store_canonical_facts` builds real lifecycle input,
   so lifecycle evidence validation runs.
2. `MemoryFactLifecycleService` commits canonical events on the real
   Workshop schema and projects them through the real
   `Mem0FactVectorAdapter`.
3. The adapter calls the real `kai.memory` functions, including the
   protected current-truth gate on reads.

Only Mem0's storage is replaced, by a small in-memory provider with the
same call surface (`add`, `get`, `update`, `delete`, `get_all`,
`search`). Retrieval assertions go through `memory.search`, so a fact
that the canonical record considers active but whose vector projection
failed shows up as missing, exactly as it would in recall.
"""

from __future__ import annotations

import itertools
from datetime import UTC, datetime
from pathlib import Path

import pytest

from kai import memory, memory_extraction, sessions
from kai.config import Config, DeploymentMode
from kai.workshop.domain import (
    AgentId,
    ChannelId,
    EventEnvelope,
    PrincipalId,
    RuntimeProfileId,
    WorkshopEventType,
    WorkshopId,
    WorkshopMembershipId,
)
from kai.workshop.execution_state import WorkshopExecutionStateNamespace, WorkshopExecutionStateRegistry
from kai.workshop.fact_lifecycle import (
    CANONICAL_CLAIM_ID_KEY,
    FactMutationSource,
    Mem0FactVectorAdapter,
    MemoryFactLifecycleService,
    revision_input_with_metadata,
)
from kai.workshop.memory_extraction_receipts import MemoryExtractionReceiptService, MemoryExtractionReceiptSpec
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import WorkshopEventStore

NOW = datetime(2026, 9, 23, 12, 0, tzinfo=UTC)
# Retrieval evaluates validity against the real clock, so revisions that
# must be current start in the past.
PAST = datetime(2026, 1, 1, tzinfo=UTC)
WORKSHOP_ID = WorkshopId("wsp_23000000000000000000000000000001")
PRINCIPAL_ID = PrincipalId("prn_23000000000000000000000000000001")
RUNTIME_ID = RuntimeProfileId("rtp_23000000000000000000000000000001")
AGENT_PRINCIPAL_ID = "prn_23000000000000000000000000000002"
CHANNEL_ID = "chn_23000000000000000000000000000001"
AGENT_ID = "agt_23000000000000000000000000000001"


# ── In-memory Mem0 stand-in ──────────────────────────────────────────


class FakeMem0:
    """
    Minimal Mem0 provider: stores rows by id, as Mem0 does.

    `update` replaces text and metadata wholesale and keeps the owner,
    matching Mem0's documented update semantics. `search` scores an exact
    text match as 1.0 and everything else low, so paraphrase checks only
    fire for true duplicates.
    """

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self._ids = itertools.count(1)

    def add(self, content: str, *, user_id: str, infer: bool, metadata: dict) -> dict:
        memory_id = f"vec-{next(self._ids)}"
        self.rows[memory_id] = {
            "id": memory_id,
            "memory": content,
            "metadata": dict(metadata),
            "user_id": user_id,
            "created_at": NOW.isoformat(),
            "updated_at": NOW.isoformat(),
        }
        return {"results": [{"id": memory_id}]}

    def get(self, *, memory_id: str) -> dict | None:
        row = self.rows.get(memory_id)
        return dict(row) if row is not None else None

    def update(self, *, memory_id: str, data: str, metadata: dict) -> None:
        row = self.rows[memory_id]
        row["memory"] = data
        row["metadata"] = dict(metadata)

    def delete(self, *, memory_id: str) -> None:
        self.rows.pop(memory_id, None)

    def get_all(self, *, filters: dict, top_k: int) -> dict:
        # Mem0 flattens metadata into the vector payload, so any filter key
        # other than the owner matches a metadata field.
        payload_filters = {key: value for key, value in filters.items() if key != "user_id"}
        owned = [
            dict(row)
            for row in self.rows.values()
            if row["user_id"] == filters["user_id"]
            and all(row["metadata"].get(key) == value for key, value in payload_filters.items())
        ]
        return {"results": owned[:top_k]}

    def search(self, query: str, *, filters: dict, top_k: int) -> dict:
        owned = [
            {**row, "score": 1.0 if row["memory"] == query else 0.1}
            for row in self.rows.values()
            if row["user_id"] == filters["user_id"]
        ]
        return {"results": sorted(owned, key=lambda row: -row["score"])[:top_k]}


# ── Fixtures ─────────────────────────────────────────────────────────


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


@pytest.fixture
async def protected(tmp_path: Path, monkeypatch):
    """
    Yield a migrated store, the fake provider, and a lifecycle service.

    The memory module runs in protected mode against the store's database,
    with the store's owner registered as canonical memory authority.
    Extraction's lifecycle bridge is pointed at the same store, because
    the production bridge needs the session database and its locks.
    """
    path = tmp_path / "kai.db"
    store = await WorkshopEventStore.open(path)
    await store.append(_event(WorkshopEventType.WORKSHOP_CREATED, "workshop", WORKSHOP_ID, {"name": "Writes"}))
    await store.append(
        _event(WorkshopEventType.PRINCIPAL_CREATED, "principal", PRINCIPAL_ID, {"kind": "human", "display_name": "Op"})
    )
    await store.append(
        _event(
            WorkshopEventType.WORKSHOP_MEMBER_ADDED,
            "workshop_membership",
            WorkshopMembershipId("wmb_23000000000000000000000000000001"),
            {"principal_id": str(PRINCIPAL_ID), "role": "owner"},
        )
    )
    await store.project_pending(CanonicalConversationProjection())
    await store.connection.execute(
        "INSERT INTO runtime_profile_owners (runtime_profile_id, principal_id) VALUES (?, ?)",
        (RUNTIME_ID, PRINCIPAL_ID),
    )
    await store.connection.commit()

    provider = FakeMem0()
    namespace = WorkshopExecutionStateNamespace(
        principal_id=PRINCIPAL_ID,
        channel_id=ChannelId("chn_23000000000000000000000000000001"),
        agent_id=AgentId("agt_23000000000000000000000000000001"),
        runtime_profile_id=RUNTIME_ID,
        legacy_runtime_key=1,
    )
    monkeypatch.setattr(memory, "_memory", provider)
    monkeypatch.setattr(
        memory,
        "_config",
        Config(
            telegram_bot_token="token",
            allowed_user_ids={1},
            memory_enabled=True,
            deployment_mode=DeploymentMode.PROTECTED,
            session_db_path=path,
        ),
    )
    memory.configure_memory_authority(WorkshopExecutionStateRegistry((namespace,)))
    service = MemoryFactLifecycleService(store, Mem0FactVectorAdapter())

    async def apply_canonical_extracted_fact(principal_id, runtime_profile_id, spec, **kwargs):
        authority = await service.authority_for(principal_id, runtime_profile_id)
        return await service.apply_extracted(authority, spec, **kwargs)

    monkeypatch.setattr(sessions, "apply_canonical_extracted_fact", apply_canonical_extracted_fact)
    try:
        yield store, provider, service
    finally:
        memory.configure_memory_authority(None)
        await store.close()


async def _seed_run(store: WorkshopEventStore, number: int) -> dict[str, str]:
    """
    Seed one completed run with its source and result messages, and claim
    its fact-extraction receipt.

    Canonical fact revisions reference the receipt, run, and messages by
    foreign key, exactly as production extraction does, so every stored
    fact in these tests is bound to real provenance rows.
    """
    connection = store.connection
    stamp = NOW.isoformat()
    run_id = f"run_{number:032x}"
    source_id = f"msg_{2 * number:032x}"
    result_id = f"msg_{2 * number + 1:032x}"
    if number == 1:
        await connection.execute(
            "INSERT INTO principals (id, kind, display_name, created_at) VALUES (?, 'agent', 'Kai', ?)",
            (AGENT_PRINCIPAL_ID, stamp),
        )
        await connection.execute(
            "INSERT INTO channels (id, workshop_id, kind, name, created_at) VALUES (?, ?, 'direct', 'Kai', ?)",
            (CHANNEL_ID, str(WORKSHOP_ID), stamp),
        )
        await connection.execute(
            "INSERT INTO agents (id, workshop_id, principal_id, name, created_at) VALUES (?, ?, ?, 'Kai', ?)",
            (AGENT_ID, str(WORKSHOP_ID), AGENT_PRINCIPAL_ID, stamp),
        )
        await connection.commit()
    # Messages and runs each need a distinct, integrity-checked event
    # position. Appending placeholder principal-creation events provides
    # them without inventing conversation events the projection would
    # interpret.
    positions: list[int] = []
    for offset in range(3):
        appended = await store.append(
            _event(
                WorkshopEventType.PRINCIPAL_CREATED,
                "principal",
                PrincipalId(f"prn_{0x24 << 120 | number << 8 | offset:032x}"),
                {"kind": "human", "display_name": f"Placeholder {number}.{offset}"},
            )
        )
        positions.append(appended.event.position)
    await connection.execute(
        "INSERT INTO messages (id, channel_id, author_principal_id, body, created_event_position, created_at) "
        "VALUES (?, ?, ?, 'source', ?, ?)",
        (source_id, CHANNEL_ID, str(PRINCIPAL_ID), positions[0], stamp),
    )
    await connection.execute(
        "INSERT INTO messages (id, channel_id, author_principal_id, body, created_event_position, created_at) "
        "VALUES (?, ?, ?, 'result', ?, ?)",
        (result_id, CHANNEL_ID, AGENT_PRINCIPAL_ID, positions[1], stamp),
    )
    await connection.execute(
        "INSERT INTO runs (id, workshop_id, channel_id, requested_by_principal_id, agent_id, "
        "inbound_message_id, status, accepted_at, started_at, terminal_at, last_event_position, "
        "result_message_id, runtime_profile_id) "
        "VALUES (?, ?, ?, ?, ?, ?, 'completed', ?, ?, ?, ?, ?, ?)",
        (
            run_id,
            str(WORKSHOP_ID),
            CHANNEL_ID,
            str(PRINCIPAL_ID),
            AGENT_ID,
            source_id,
            stamp,
            stamp,
            stamp,
            positions[2],
            result_id,
            str(RUNTIME_ID),
        ),
    )
    await connection.commit()
    claim = await MemoryExtractionReceiptService(connection).claim(
        MemoryExtractionReceiptSpec(
            principal_id=str(PRINCIPAL_ID),
            runtime_profile_id=str(RUNTIME_ID),
            run_id=run_id,
            source_message_id=source_id,
            result_message_id=result_id,
            extraction_role="fact_extraction",
            backend="codex",
            provider="openai",
            model="gpt-5.6-sol",
            prompt_version="13",
            schema_version="1",
            policy_version="1",
        )
    )
    return {"run": run_id, "source": source_id, "result": result_id, "receipt": claim.receipt.receipt_id}


async def _store_fact(store: WorkshopEventStore, fact: dict, *, run: int) -> list:
    """Run one fact through production extraction storage; return decisions."""
    ids = await _seed_run(store, run)
    decisions: list = []
    await memory_extraction._store_canonical_facts(
        [fact],
        user_id=str(PRINCIPAL_ID),
        session_id="session-1",
        config=Config(telegram_bot_token="token", allowed_user_ids={1}, memory_enabled=True),
        active_project=None,
        user_log=None,
        assistant_log=None,
        canonical_provenance={
            memory.WORKSHOP_RUN_ID_KEY: ids["run"],
            memory.WORKSHOP_SOURCE_MESSAGE_ID_KEY: ids["source"],
            memory.WORKSHOP_RESULT_MESSAGE_ID_KEY: ids["result"],
        },
        runtime_profile_id=str(RUNTIME_ID),
        receipt_id=ids["receipt"],
        backend="codex",
        provider="openai",
        model="gpt-5.6-sol",
        receipt_decisions=decisions,
    )
    return decisions


def _recall() -> list[str]:
    """Texts visible to protected recall for the owner."""
    return sorted(
        row.text for row in memory.search("anything", user_id=str(PRINCIPAL_ID), runtime_profile_id=str(RUNTIME_ID))
    )


def _new(content: str, *, confidence: float = 0.95) -> dict:
    return {
        "content": content,
        "intent": "new",
        "speaker": "user",
        "confidence": confidence,
        "tags": ["preference"],
        "scope_hint": "global",
    }


# ── Extraction through lifecycle validation ──────────────────────────


async def test_new_extracted_fact_is_stored_canonically_and_recalled(protected) -> None:
    store, provider, _service = protected

    decisions = await _store_fact(store, _new("The operator prefers dark themes."), run=1)

    assert [decision.outcome for decision in decisions] == ["stored"]
    assert _recall() == ["The operator prefers dark themes."]
    (row,) = provider.rows.values()
    assert row["metadata"][CANONICAL_CLAIM_ID_KEY]


async def test_confident_update_supersedes_in_place_and_recall_follows(protected) -> None:
    store, provider, _service = protected
    await _store_fact(store, _new("The operator prefers dark themes."), run=1)
    (original_id,) = provider.rows

    decisions = await _store_fact(
        store,
        {
            **_new("The operator prefers light themes.", confidence=0.97),
            "intent": "update_of",
            "existing_id": original_id,
        },
        run=2,
    )

    assert [decision.outcome for decision in decisions] == ["replaced"]
    assert _recall() == ["The operator prefers light themes."]
    assert list(provider.rows) == [original_id]


async def test_low_confidence_update_opens_a_conflict_and_says_so(protected) -> None:
    store, provider, _service = protected
    await _store_fact(store, _new("The operator prefers dark themes."), run=1)
    (original_id,) = provider.rows

    decisions = await _store_fact(
        store,
        {
            **_new("The operator prefers light themes.", confidence=0.6),
            "intent": "update_of",
            "existing_id": original_id,
        },
        run=2,
    )

    assert [decision.outcome for decision in decisions] == ["conflict_opened"]
    # An unresolved conflict is excluded from ordinary retrieval until
    # it is resolved; neither side is presented as current truth.
    assert _recall() == []


async def test_exact_duplicate_is_skipped_without_a_second_claim(protected) -> None:
    store, provider, _service = protected
    await _store_fact(store, _new("The operator prefers dark themes."), run=1)

    decisions = await _store_fact(store, _new("The operator prefers dark themes."), run=2)

    assert [decision.outcome for decision in decisions] == ["duplicate_skipped"]
    assert len(provider.rows) == 1


async def test_lifecycle_validation_failure_is_recorded_as_storage_failure(protected) -> None:
    store, provider, _service = protected

    # Canonical revision content is bounded at 16,384 characters.
    decisions = await _store_fact(store, _new("x" * 17_000), run=1)

    assert [decision.outcome for decision in decisions] == ["storage_failed"]
    assert provider.rows == {}


# ── Lifecycle projection through the protected gate ──────────────────


async def test_legacy_adoption_rewrites_the_hidden_row_in_place(protected) -> None:
    _store, provider, service = protected
    storage_user = memory.canonical_memory_user_id(str(PRINCIPAL_ID))
    provider.add(
        "The operator deploys on Fridays.",
        user_id=storage_user,
        infer=False,
        metadata=memory._metadata_for_owner(
            {"source": "extracted", "type": "fact", "scope": "global"},
            memory._canonical_memory_owner(str(PRINCIPAL_ID), runtime_profile_id=str(RUNTIME_ID))[1],
        ),
    )
    (legacy_id,) = provider.rows
    legacy = memory.get_by_id_for_lifecycle_projection(
        user_id=str(PRINCIPAL_ID), memory_id=legacy_id, runtime_profile_id=str(RUNTIME_ID)
    )
    assert legacy is not None
    # Strict exact-id reads hide the unadopted legacy row.
    assert memory.get_by_id(user_id=str(PRINCIPAL_ID), memory_id=legacy_id, runtime_profile_id=str(RUNTIME_ID)) is None

    authority = await service.authority_for(PRINCIPAL_ID, RUNTIME_ID)
    adopted = await service.adopt_legacy(authority, legacy, idempotency_key="adopt:legacy")

    assert adopted.projection_status == "succeeded"
    assert list(provider.rows) == [legacy_id]
    assert provider.rows[legacy_id]["metadata"][CANONICAL_CLAIM_ID_KEY] == str(adopted.claim_id)
    # Adoption without reviewed provenance records the row as
    # legacy_incomplete, which current truth admits only after operator
    # review; the projection itself is what this path must get right.
    assert adopted.projection_status == "succeeded"


async def test_human_supersede_retract_and_restore_all_project(protected) -> None:
    store, provider, service = protected
    await _store_fact(store, _new("The operator prefers dark themes."), run=1)
    (memory_id,) = provider.rows
    authority = await service.authority_for(PRINCIPAL_ID, RUNTIME_ID)
    current = memory.get_by_id(user_id=str(PRINCIPAL_ID), memory_id=memory_id, runtime_profile_id=str(RUNTIME_ID))
    assert current is not None
    first = await service.adopt_legacy(authority, current, idempotency_key="snapshot")
    spec = memory_extraction.FactRevisionInput(
        content="The operator prefers high-contrast themes.",
        scope_kind="global",
        scope_key="",
        reason="The operator corrected the preference.",
        evidence=({"kind": "operator", "reference_id": "request-1", "sha256": None},),
        vector_metadata={"source": "explicit", "speaker": "user", "type": "fact"},
        confidence=1.0,
        asserted_at=PAST,
        observed_at=PAST,
        valid_from=PAST,
    )

    superseded = await service.supersede(
        authority,
        first.claim_id,
        first.revision_id,
        revision_input_with_metadata(spec, dict(spec.vector_metadata)),
        idempotency_key="human:supersede",
        source=FactMutationSource.HUMAN,
    )
    assert superseded.projection_status == "succeeded"
    assert _recall() == ["The operator prefers high-contrast themes."]

    retracted = await service.retract(
        authority,
        superseded.claim_id,
        superseded.revision_id,
        reason="Withdrawn by the operator.",
        idempotency_key="human:retract",
    )
    assert retracted.projection_status == "succeeded"
    assert _recall() == []

    restored = await service.restore(
        authority,
        retracted.claim_id,
        retracted.revision_id,
        revision_input_with_metadata(spec, dict(spec.vector_metadata)),
        idempotency_key="human:restore",
    )
    assert restored.projection_status == "succeeded"
    assert _recall() == ["The operator prefers high-contrast themes."]


async def test_projection_update_refuses_rows_owned_by_someone_else(protected) -> None:
    _store, provider, _service = protected
    provider.add("Foreign row", user_id="someone-else", infer=False, metadata={"source": "extracted"})
    (foreign_id,) = provider.rows

    # A foreign owner under a recorded memory id means something is wrong,
    # so the projection read refuses instead of overwriting the row.
    with pytest.raises(memory.LifecycleProjectionReadError):
        memory.update_for_lifecycle_projection(
            user_id=str(PRINCIPAL_ID),
            memory_id=foreign_id,
            data="Overwritten",
            metadata={"source": "extracted"},
            runtime_profile_id=str(RUNTIME_ID),
        )

    assert provider.rows[foreign_id]["memory"] == "Foreign row"


def test_receipt_reports_a_conflict_when_it_is_the_only_outcome() -> None:
    from kai.memory_extraction import ExtractionResult, _fact_receipt_completion
    from kai.workshop.memory_extraction_receipts import MemoryExtractionStorageDecision

    completion = _fact_receipt_completion(
        result=ExtractionResult([{"intent": "update_of"}], False, raw_fact_count=1),
        candidate_ids={"vec-1"},
        decisions=[MemoryExtractionStorageDecision(0, "update_of", "conflict_opened")],
        stored=0,
        replaced=0,
        skipped=0,
        duration_ms=10,
    )

    assert (completion.status, completion.decision_outcome, completion.failure_code) == (
        "completed",
        "conflict_opened",
        None,
    )
