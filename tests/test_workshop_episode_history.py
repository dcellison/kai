"""Immutable canonical episode history and recoverable search projection."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

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
from kai.workshop.episode_history import (
    CANONICAL_EPISODE_ID_KEY,
    EpisodeInput,
    MemoryEpisodeHistoryService,
)
from kai.workshop.memory_current_truth import CANONICAL_ADMISSION_AUTHORITY_KEY, project_current_truth
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import WorkshopEventStore

NOW = datetime(2026, 9, 21, 13, 0, tzinfo=UTC)
WORKSHOP_ID = WorkshopId("wsp_21000000000000000000000000000001")
PRINCIPAL_ID = PrincipalId("prn_21000000000000000000000000000001")
RUNTIME_ID = RuntimeProfileId("rtp_21000000000000000000000000000001")


@pytest.mark.asyncio
async def test_schema_89_upgrades_to_immutable_episode_history(tmp_path: Path, monkeypatch) -> None:
    from kai.workshop import schema

    path = tmp_path / "upgrade.db"
    with monkeypatch.context() as migration_context:
        migration_context.setattr(schema, "WORKSHOP_SCHEMA_VERSION", 89)
        migration_context.setattr(schema, "_MIGRATIONS", schema._MIGRATIONS[:89])
        old = await WorkshopEventStore.open(path)
        await old.close()

    upgraded = await WorkshopEventStore.open(path)
    try:
        assert await upgraded.schema_version() == schema.WORKSHOP_SCHEMA_VERSION
        async with upgraded.connection.execute("PRAGMA table_info(memory_episodes)") as cursor:
            columns = {str(row[1]) for row in await cursor.fetchall()}
        assert {"goal", "outcome", "vector_metadata_json", "similarity_fingerprint"}.issubset(columns)
        async with upgraded.connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'memory_episode_vector_operations'"
        ) as cursor:
            assert await cursor.fetchone() is not None
    finally:
        await upgraded.close()


class FakeEpisodeVector:
    def __init__(self) -> None:
        self.rows: dict[str, MemoryResult] = {}
        self.fail = False
        self.add_calls = 0

    async def find_episode(self, authority, episode_id) -> MemoryResult | None:
        return next(
            (row for row in self.rows.values() if row.metadata.get(CANONICAL_EPISODE_ID_KEY) == str(episode_id)),
            None,
        )

    async def add(self, authority, content: str, metadata: dict[str, object]) -> str | None:
        self.add_calls += 1
        if self.fail:
            return None
        memory_id = f"episode-vector-{len(self.rows) + 1}"
        self.rows[memory_id] = MemoryResult(
            memory_id,
            content,
            0.0,
            "episode",
            dict(metadata),
            NOW.isoformat(),
            NOW.isoformat(),
        )
        return memory_id


def _foundation_event(event_type: WorkshopEventType, aggregate_type: str, aggregate_id, payload: dict):
    return EventEnvelope.create(
        event_type=event_type,
        event_version=1,
        workshop_id=WORKSHOP_ID,
        aggregate_type=aggregate_type,
        aggregate_id=aggregate_id,
        occurred_at=NOW,
        payload=payload,
    )


async def _service(path: Path, vector: FakeEpisodeVector):
    store = await WorkshopEventStore.open(path)
    await store.append(
        _foundation_event(WorkshopEventType.WORKSHOP_CREATED, "workshop", WORKSHOP_ID, {"name": "Episodes"})
    )
    await store.append(
        _foundation_event(
            WorkshopEventType.PRINCIPAL_CREATED,
            "principal",
            PRINCIPAL_ID,
            {"kind": "human", "display_name": "Daniel"},
        )
    )
    await store.append(
        _foundation_event(
            WorkshopEventType.WORKSHOP_MEMBER_ADDED,
            "workshop_membership",
            WorkshopMembershipId("wmb_21000000000000000000000000000001"),
            {"principal_id": str(PRINCIPAL_ID), "role": "owner"},
        )
    )
    await store.project_pending(CanonicalConversationProjection())
    await store.connection.execute(
        "INSERT INTO runtime_profile_owners (runtime_profile_id, principal_id) VALUES (?, ?)",
        (RUNTIME_ID, PRINCIPAL_ID),
    )
    await store.connection.commit()
    service = MemoryEpisodeHistoryService(store, vector)
    return store, service, await service.authority_for(PRINCIPAL_ID, RUNTIME_ID)


def _spec(*, outcome: str = "The first attempt succeeded.") -> EpisodeInput:
    return EpisodeInput(
        goal="Repair the production memory lifecycle",
        context="An installation exposed stale temporal-memory state.",
        approach="Inspected canonical rows and applied a replay-safe correction.",
        outcome=outcome,
        outcome_quality="success",
        lessons="Canonical state must commit before vector projection.",
        tags=("memory", "lifecycle"),
        actors=("Daniel", "Kai"),
        scope_kind="project",
        scope_key="kai",
        reason="Extracted from a completed canonical run.",
        evidence=({"kind": "run", "reference_id": "run_qualified", "sha256": None},),
        vector_metadata={"source": "episode", "session_id": "session-1"},
        occurred_from=NOW,
        occurred_until=NOW,
        observed_at=NOW,
        backend="codex",
        provider="openai",
        model="gpt-5.6-luna",
        prompt_version="2",
        schema_version="1",
    )


@pytest.mark.asyncio
async def test_record_and_replay_preserve_one_immutable_episode_and_vector(tmp_path: Path) -> None:
    vector = FakeEpisodeVector()
    store, service, authority = await _service(tmp_path / "kai.db", vector)
    try:
        first = await service.record(authority, _spec(), idempotency_key="episode:run-1")
        replay = await service.record(authority, _spec(), idempotency_key="episode:run-1")

        assert first.projection_status == "succeeded"
        assert replay.replayed is True
        assert replay.episode_id == first.episode_id
        assert vector.add_calls == 1
        assert len(vector.rows) == 1
        projected = next(iter(vector.rows.values()))
        assert projected.metadata[CANONICAL_EPISODE_ID_KEY] == str(first.episode_id)
        assert projected.metadata["canonical_memory_temporal_role"] == "historical_episode"
        assert projected.metadata["backend"] == "codex"
        async with store.connection.execute(
            "SELECT goal, outcome, model FROM memory_episodes WHERE episode_id = ?",
            (first.episode_id,),
        ) as cursor:
            assert tuple(await cursor.fetchone()) == (
                "Repair the production memory lifecycle",
                "The first attempt succeeded.",
                "gpt-5.6-luna",
            )
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_repeated_occurrence_remains_distinct_and_is_linked(tmp_path: Path) -> None:
    vector = FakeEpisodeVector()
    store, service, authority = await _service(tmp_path / "kai.db", vector)
    try:
        first = await service.record(authority, _spec(), idempotency_key="episode:run-1")
        second = await service.record(authority, _spec(), idempotency_key="episode:run-2")

        assert second.episode_id != first.episode_id
        assert second.repeated_episode_id == first.episode_id
        assert len(vector.rows) == 2
        async with store.connection.execute(
            "SELECT source_episode_id, target_episode_id, relationship FROM memory_episode_followups"
        ) as cursor:
            assert tuple(await cursor.fetchone()) == (
                str(first.episode_id),
                str(second.episode_id),
                "repeated",
            )
        projected = project_current_truth(
            vector.rows.values(),
            db_path=tmp_path / "kai.db",
            principal_id=str(PRINCIPAL_ID),
            runtime_profile_id=str(RUNTIME_ID),
            now=NOW,
        )
        assert len(projected.rows) == 2
        assert [row.id for row in projected.rows] == [second.memory_id, first.memory_id]
        newer = next(row for row in projected.rows if row.id == second.memory_id)
        assert newer.metadata["canonical_memory_temporal_role"] == "historical_episode"
        assert newer.metadata["episode_followups"] == [
            {
                "direction": "incoming",
                "episode_id": str(first.episode_id),
                "relationship": "repeated",
                "reason": "A distinct later occurrence closely resembles this episode.",
            }
        ]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_operator_review_admits_incomplete_episode_as_history(tmp_path: Path) -> None:
    vector = FakeEpisodeVector()
    database = tmp_path / "kai.db"
    store, service, authority = await _service(database, vector)
    try:
        recorded = await service.record(
            authority,
            replace(
                _spec(),
                migration_classification="legacy_incomplete",
                migration_gaps=("assertion_time", "provenance"),
                admission_authority="operator_review",
            ),
            idempotency_key="episode:operator-reviewed-legacy",
        )

        projected = project_current_truth(
            vector.rows.values(),
            db_path=database,
            principal_id=str(PRINCIPAL_ID),
            runtime_profile_id=str(RUNTIME_ID),
            now=NOW,
        )

        assert projected.excluded == {}
        assert len(projected.rows) == 1
        admitted = projected.rows[0]
        assert admitted.id == recorded.memory_id
        assert admitted.metadata["migration_classification"] == "legacy_incomplete"
        assert admitted.metadata[CANONICAL_ADMISSION_AUTHORITY_KEY] == "operator_review"
        assert admitted.metadata["canonical_memory_temporal_role"] == "historical_episode"
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_failed_vector_projection_recovers_without_duplicate_episode(tmp_path: Path) -> None:
    vector = FakeEpisodeVector()
    vector.fail = True
    store, service, authority = await _service(tmp_path / "kai.db", vector)
    try:
        first = await service.record(authority, _spec(), idempotency_key="episode:run-1")
        assert first.projection_status == "failed"
        vector.fail = False

        assert await service.recover_pending(retry_failed=True) == 1
        recovered = await service.snapshot(first.episode_id)
        assert recovered.projection_status == "succeeded"
        async with store.connection.execute("SELECT COUNT(*) FROM memory_episodes") as cursor:
            assert int((await cursor.fetchone())[0]) == 1
        assert len(vector.rows) == 1
    finally:
        await store.close()
