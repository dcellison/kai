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

# ruff: noqa: F811 - tests take the imported `protected` fixture as a parameter by name.

from __future__ import annotations

import pytest

from kai import memory, memory_extraction
from kai.workshop.fact_lifecycle import (
    CANONICAL_CLAIM_ID_KEY,
    FactMutationSource,
    revision_input_with_metadata,
)
from tests.memory_fixtures import (  # noqa: F401 - pytest fixture import
    AGENT_ID,
    AGENT_PRINCIPAL_ID,
    CHANNEL_ID,
    NOW,
    PAST,
    PRINCIPAL_ID,
    RUNTIME_ID,
    WORKSHOP_ID,
    FakeMem0,
    _event,
    _new,
    _recall,
    _seed_run,
    _store_fact,
    protected,
)

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
