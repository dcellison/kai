"""Canonical temporal-memory schema, vocabulary, and replay contracts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kai.workshop.domain import (
    EventEnvelope,
    MemoryClaimId,
    MemoryEpisodeId,
    MemoryRevisionId,
    PrincipalId,
    RuntimeProfileId,
    WorkshopEventType,
    WorkshopId,
    WorkshopMembershipId,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import WorkshopEventStore
from kai.workshop.temporal_memory import (
    FactLifecycleState,
    FactLifecycleTransition,
    LegacyMemoryClassification,
    LegacyTemporalGap,
    classify_legacy_temporal_metadata,
    validate_fact_transition,
)

NOW = datetime(2026, 9, 21, 12, 0, tzinfo=UTC)
WORKSHOP_ID = WorkshopId("wsp_10000000000000000000000000000001")
PRINCIPAL_ID = PrincipalId("prn_10000000000000000000000000000001")
RUNTIME_ID = RuntimeProfileId("rtp_10000000000000000000000000000001")
CLAIM_ID = MemoryClaimId("mcl_10000000000000000000000000000001")
REVISION_ONE = MemoryRevisionId("mrv_10000000000000000000000000000001")
REVISION_TWO = MemoryRevisionId("mrv_10000000000000000000000000000002")
EPISODE_ONE = MemoryEpisodeId("mep_10000000000000000000000000000001")
EPISODE_TWO = MemoryEpisodeId("mep_10000000000000000000000000000002")


def _event(
    event_type: WorkshopEventType,
    *,
    aggregate_type: str,
    aggregate_id,
    payload: dict,
    offset: int,
    actor: PrincipalId | None = PRINCIPAL_ID,
) -> EventEnvelope:
    return EventEnvelope.create(
        event_type=event_type,
        event_version=1,
        workshop_id=WORKSHOP_ID,
        aggregate_type=aggregate_type,
        aggregate_id=aggregate_id,
        actor_principal_id=actor,
        occurred_at=NOW + timedelta(seconds=offset),
        idempotency_key=f"temporal-memory:{offset}:{event_type.value}",
        payload=payload,
    )


def _revision(revision_id: MemoryRevisionId, content: str, *, supersedes: MemoryRevisionId | None = None) -> dict:
    return {
        "revision_id": str(revision_id),
        "content": content,
        "asserted_at": NOW.isoformat(),
        "observed_at": NOW.isoformat(),
        "stored_at": NOW.isoformat(),
        "valid_from": NOW.isoformat(),
        "valid_until": None,
        "reason": "Canonical assertion from an accepted message.",
        "evidence": [
            {
                "kind": "message",
                "reference_id": "msg_10000000000000000000000000000001",
                "sha256": "a" * 64,
            }
        ],
        "source_receipt_id": None,
        "source_run_id": None,
        "source_message_id": None,
        "result_message_id": None,
        "backend": "codex",
        "provider": "openai",
        "model": "gpt-5.6-luna",
        "prompt_version": "fact_extraction_v1",
        "schema_version": "fact_schema_v1",
        "supersedes_revision_id": None if supersedes is None else str(supersedes),
        "migration_classification": "canonical",
        "migration_gaps": [],
    }


def _episode_payload(content: str, *, scope_key: str = "") -> dict:
    return {
        "owner_principal_id": str(PRINCIPAL_ID),
        "runtime_profile_id": str(RUNTIME_ID),
        "scope_kind": "global" if not scope_key else "project",
        "scope_key": scope_key,
        "content": content,
        "occurred_from": NOW.isoformat(),
        "occurred_until": (NOW + timedelta(minutes=2)).isoformat(),
        "observed_at": (NOW + timedelta(minutes=2)).isoformat(),
        "stored_at": (NOW + timedelta(minutes=3)).isoformat(),
        "reason": "Canonical episode generated from a bounded conversation.",
        "evidence": [
            {
                "kind": "run",
                "reference_id": "run_10000000000000000000000000000001",
                "sha256": None,
            }
        ],
        "source_receipt_id": None,
        "source_run_id": None,
        "source_message_id": None,
        "result_message_id": None,
        "backend": "codex",
        "provider": "openai",
        "model": "gpt-5.6-luna",
        "prompt_version": "episode_generation_v1",
        "schema_version": "episode_schema_v1",
        "migration_classification": "canonical",
        "migration_gaps": [],
    }


async def _ready_store(path: Path) -> tuple[WorkshopEventStore, CanonicalConversationProjection]:
    store = await WorkshopEventStore.open(path)
    projection = CanonicalConversationProjection()
    await store.append(
        _event(
            WorkshopEventType.WORKSHOP_CREATED,
            aggregate_type="workshop",
            aggregate_id=WORKSHOP_ID,
            payload={"name": "Temporal qualification"},
            offset=0,
            actor=None,
        )
    )
    await store.append(
        _event(
            WorkshopEventType.PRINCIPAL_CREATED,
            aggregate_type="principal",
            aggregate_id=PRINCIPAL_ID,
            payload={"kind": "human", "display_name": "Daniel"},
            offset=1,
            actor=None,
        )
    )
    await store.append(
        _event(
            WorkshopEventType.WORKSHOP_MEMBER_ADDED,
            aggregate_type="workshop_membership",
            aggregate_id=WorkshopMembershipId("wmb_10000000000000000000000000000001"),
            payload={"principal_id": str(PRINCIPAL_ID), "role": "owner"},
            offset=2,
            actor=None,
        )
    )
    await store.project_pending(projection)
    await store.connection.execute(
        "INSERT INTO runtime_profile_owners (runtime_profile_id, principal_id) VALUES (?, ?)",
        (RUNTIME_ID, PRINCIPAL_ID),
    )
    await store.connection.commit()
    return store, projection


async def _rows(store: WorkshopEventStore, table: str) -> list[tuple]:
    async with store.connection.execute(f"SELECT * FROM {table} ORDER BY rowid") as cursor:
        return [tuple(row) for row in await cursor.fetchall()]


def test_legacy_classification_names_missing_temporal_authority_without_guessing() -> None:
    fact = classify_legacy_temporal_metadata(
        {"source": "migration", "scope": "global", "operator_principal_id": str(PRINCIPAL_ID)},
        memory_kind="fact",
    )
    episode = classify_legacy_temporal_metadata(
        {"source": "episode", "scope": "global", "model": "legacy-model", "evidence": ["archive"]},
        memory_kind="episode",
    )

    assert fact.classification == LegacyMemoryClassification.LEGACY_INCOMPLETE
    assert fact.gaps == (
        LegacyTemporalGap.ASSERTION_TIME,
        LegacyTemporalGap.OBSERVATION_TIME,
        LegacyTemporalGap.EVIDENCE,
    )
    assert episode.classification == LegacyMemoryClassification.LEGACY_INCOMPLETE
    assert episode.gaps == (LegacyTemporalGap.OCCURRENCE_TIME,)


def test_invalid_fact_transition_fails_closed() -> None:
    assert (
        validate_fact_transition(FactLifecycleState.ACTIVE, FactLifecycleTransition.SUPERSEDED)
        == FactLifecycleState.SUPERSEDED
    )
    with pytest.raises(ValueError, match="Invalid memory fact transition"):
        validate_fact_transition(FactLifecycleState.RETRACTED, FactLifecycleTransition.SUPERSEDED)


@pytest.mark.asyncio
async def test_fact_revisions_preserve_history_and_rebuild_exactly(tmp_path: Path) -> None:
    store, projection = await _ready_store(tmp_path / "kai.db")
    try:
        await store.append(
            _event(
                WorkshopEventType.MEMORY_FACT_RECORDED,
                aggregate_type="memory_fact_claim",
                aggregate_id=CLAIM_ID,
                payload={
                    "owner_principal_id": str(PRINCIPAL_ID),
                    "runtime_profile_id": str(RUNTIME_ID),
                    "scope_kind": "global",
                    "scope_key": "",
                    "claim_identity_sha256": "b" * 64,
                    "revision": _revision(REVISION_ONE, "Daniel prefers the dark theme."),
                },
                offset=3,
            )
        )
        await store.append(
            _event(
                WorkshopEventType.MEMORY_FACT_SUPERSEDED,
                aggregate_type="memory_fact_claim",
                aggregate_id=CLAIM_ID,
                payload={
                    "prior_revision_id": str(REVISION_ONE),
                    "reason": "Daniel selected a light theme instead.",
                    "revision": _revision(
                        REVISION_TWO,
                        "Daniel prefers the light theme.",
                        supersedes=REVISION_ONE,
                    ),
                },
                offset=4,
            )
        )
        await store.project_pending(projection)

        tables = (
            "memory_fact_claims",
            "memory_fact_revisions",
            "memory_fact_revision_states",
            "memory_fact_lifecycle_events",
        )
        before = {table: await _rows(store, table) for table in tables}
        assert [row[2] for row in before["memory_fact_revision_states"]] == ["superseded", "active"]
        assert before["memory_fact_revisions"][1][19] == str(REVISION_ONE)

        await store.rebuild_projection(projection)

        assert {table: await _rows(store, table) for table in tables} == before
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_episode_followup_rejects_cross_scope_link(tmp_path: Path) -> None:
    store, projection = await _ready_store(tmp_path / "kai.db")
    try:
        await store.append(
            _event(
                WorkshopEventType.MEMORY_EPISODE_RECORDED,
                aggregate_type="memory_episode",
                aggregate_id=EPISODE_ONE,
                payload=_episode_payload("Initial incident."),
                offset=3,
            )
        )
        await store.append(
            _event(
                WorkshopEventType.MEMORY_EPISODE_RECORDED,
                aggregate_type="memory_episode",
                aggregate_id=EPISODE_TWO,
                payload=_episode_payload("Project incident.", scope_key="project:kai"),
                offset=4,
            )
        )
        await store.append(
            _event(
                WorkshopEventType.MEMORY_EPISODE_FOLLOWUP_RECORDED,
                aggregate_type="memory_episode",
                aggregate_id=EPISODE_ONE,
                payload={
                    "target_episode_id": str(EPISODE_TWO),
                    "relationship": "revisited",
                    "reason": "This invalid link crosses scopes.",
                },
                offset=5,
            )
        )

        with pytest.raises(ValueError, match="cannot cross owner, runtime, or scope"):
            await store.project_pending(projection)
        assert await _rows(store, "memory_episodes") == []
        assert await _rows(store, "memory_episode_followups") == []
    finally:
        await store.close()
