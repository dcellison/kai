"""Atomic canonical fact lifecycle and recoverable vector projection."""

from __future__ import annotations

import asyncio
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
from kai.workshop.fact_lifecycle import (
    CANONICAL_CLAIM_ID_KEY,
    CANONICAL_REVISION_ID_KEY,
    FactLifecycleConflict,
    FactMutationSource,
    FactRevisionInput,
    MemoryFactLifecycleService,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import WorkshopEventStore

NOW = datetime(2026, 9, 21, 13, 0, tzinfo=UTC)
WORKSHOP_ID = WorkshopId("wsp_20000000000000000000000000000001")
PRINCIPAL_ID = PrincipalId("prn_20000000000000000000000000000001")
RUNTIME_ID = RuntimeProfileId("rtp_20000000000000000000000000000001")


class FakeVector:
    def __init__(self) -> None:
        self.rows: dict[str, MemoryResult] = {}
        self.fail = False
        self.add_calls = 0
        self.replace_calls = 0
        self.delete_calls = 0

    async def get(self, authority, memory_id: str) -> MemoryResult | None:
        return self.rows.get(memory_id)

    async def find_revision(self, authority, revision_id) -> MemoryResult | None:
        return next(
            (row for row in self.rows.values() if row.metadata.get(CANONICAL_REVISION_ID_KEY) == str(revision_id)),
            None,
        )

    async def add(self, authority, content: str, metadata: dict[str, object]) -> str | None:
        self.add_calls += 1
        if self.fail:
            return None
        memory_id = f"vector-{len(self.rows) + 1}"
        self.rows[memory_id] = MemoryResult(
            memory_id,
            content,
            0.0,
            "fact",
            dict(metadata),
            NOW.isoformat(),
            NOW.isoformat(),
        )
        return memory_id

    async def replace(self, authority, memory_id: str, content: str, metadata: dict[str, object]) -> bool:
        self.replace_calls += 1
        if self.fail or memory_id not in self.rows:
            return False
        prior = self.rows[memory_id]
        self.rows[memory_id] = MemoryResult(
            memory_id,
            content,
            0.0,
            "fact",
            dict(metadata),
            prior.created_at,
            NOW.isoformat(),
        )
        return True

    async def delete(self, authority, memory_id: str) -> bool:
        self.delete_calls += 1
        if self.fail:
            return False
        self.rows.pop(memory_id, None)
        return True


def _foundation_event(event_type: WorkshopEventType, aggregate_type: str, aggregate_id, payload: dict) -> EventEnvelope:
    return EventEnvelope.create(
        event_type=event_type,
        event_version=1,
        workshop_id=WORKSHOP_ID,
        aggregate_type=aggregate_type,
        aggregate_id=aggregate_id,
        occurred_at=NOW,
        payload=payload,
    )


async def _service(path: Path, vector: FakeVector) -> tuple[WorkshopEventStore, MemoryFactLifecycleService, object]:
    store = await WorkshopEventStore.open(path)
    await store.append(
        _foundation_event(WorkshopEventType.WORKSHOP_CREATED, "workshop", WORKSHOP_ID, {"name": "Facts"})
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
            WorkshopMembershipId("wmb_20000000000000000000000000000001"),
            {"principal_id": str(PRINCIPAL_ID), "role": "owner"},
        )
    )
    await store.project_pending(CanonicalConversationProjection())
    await store.connection.execute(
        "INSERT INTO runtime_profile_owners (runtime_profile_id, principal_id) VALUES (?, ?)",
        (RUNTIME_ID, PRINCIPAL_ID),
    )
    await store.connection.commit()
    service = MemoryFactLifecycleService(store, vector)
    return store, service, await service.authority_for(PRINCIPAL_ID, RUNTIME_ID)


def _spec(
    content: str,
    *,
    confidence: float = 1.0,
    scope_kind: str = "global",
    scope_key: str = "",
) -> FactRevisionInput:
    return FactRevisionInput(
        content=content,
        scope_kind=scope_kind,
        scope_key=scope_key,
        reason="The operator changed the current preference.",
        evidence=({"kind": "operator", "reference_id": "request-1", "sha256": None},),
        vector_metadata={"source": "explicit", "speaker": "user", "tags": ["preference"]},
        confidence=confidence,
        asserted_at=NOW,
        observed_at=NOW,
        valid_from=NOW,
    )


@pytest.mark.asyncio
async def test_create_and_replay_write_one_canonical_revision_and_one_vector(tmp_path: Path) -> None:
    vector = FakeVector()
    store, service, authority = await _service(tmp_path / "kai.db", vector)
    try:
        first = await service.create(
            authority,
            _spec("Daniel prefers Atom One Dark."),
            idempotency_key="fact:create:request-1",
            stable_claim_key="operator:request-1",
        )
        replay = await service.create(
            authority,
            _spec("Daniel prefers Atom One Dark."),
            idempotency_key="fact:create:request-1",
            stable_claim_key="operator:request-1",
        )

        assert first.state == "active"
        assert first.projection_status == "succeeded"
        assert replay.replayed is True
        assert replay.revision_id == first.revision_id
        assert vector.add_calls == 1
        assert len(vector.rows) == 1
        assert next(iter(vector.rows.values())).metadata[CANONICAL_CLAIM_ID_KEY] == str(first.claim_id)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_supersession_is_revision_checked_and_updates_one_vector_in_place(tmp_path: Path) -> None:
    vector = FakeVector()
    store, service, authority = await _service(tmp_path / "kai.db", vector)
    try:
        first = await service.create(
            authority,
            _spec("Daniel prefers a dark theme."),
            idempotency_key="fact:create:dark",
            stable_claim_key="theme-preference",
        )
        second = await service.supersede(
            authority,
            first.claim_id,
            first.revision_id,
            _spec("Daniel prefers a light theme.", scope_kind="project", scope_key="kai"),
            idempotency_key="fact:supersede:light",
            source=FactMutationSource.HUMAN,
        )

        assert second.state == "active"
        assert vector.add_calls == 1
        assert vector.replace_calls == 1
        assert len(vector.rows) == 1
        assert next(iter(vector.rows.values())).text == "Daniel prefers a light theme."
        async with store.connection.execute(
            "SELECT scope_kind, scope_key FROM memory_fact_claims WHERE claim_id = ?",
            (first.claim_id,),
        ) as cursor:
            assert tuple(await cursor.fetchone()) == ("project", "kai")
        async with store.connection.execute(
            "SELECT revision_id, state FROM memory_fact_revision_states WHERE claim_id = ? ORDER BY rowid",
            (first.claim_id,),
        ) as cursor:
            assert [(str(row[0]), str(row[1])) for row in await cursor.fetchall()] == [
                (str(first.revision_id), "superseded"),
                (str(second.revision_id), "active"),
            ]

        with pytest.raises(FactLifecycleConflict):
            await service.supersede(
                authority,
                first.claim_id,
                first.revision_id,
                _spec("A stale third preference."),
                idempotency_key="fact:supersede:stale",
                source=FactMutationSource.HUMAN,
            )
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_concurrent_supersession_allows_exactly_one_current_revision(tmp_path: Path) -> None:
    vector = FakeVector()
    database = tmp_path / "kai.db"
    store, service, authority = await _service(database, vector)
    other_store = await WorkshopEventStore.open(database)
    other_service = MemoryFactLifecycleService(other_store, vector)
    other_authority = await other_service.authority_for(PRINCIPAL_ID, RUNTIME_ID)
    try:
        first = await service.create(
            authority,
            _spec("Daniel uses the old value."),
            idempotency_key="fact:create:concurrent",
            stable_claim_key="concurrent",
        )
        outcomes = await asyncio.gather(
            other_service.supersede(
                other_authority,
                first.claim_id,
                first.revision_id,
                _spec("Daniel uses value A."),
                idempotency_key="fact:supersede:a",
                source=FactMutationSource.HUMAN,
            ),
            service.supersede(
                authority,
                first.claim_id,
                first.revision_id,
                _spec("Daniel uses value B."),
                idempotency_key="fact:supersede:b",
                source=FactMutationSource.HUMAN,
            ),
            return_exceptions=True,
        )
        assert sum(not isinstance(outcome, Exception) for outcome in outcomes) == 1
        assert sum(isinstance(outcome, FactLifecycleConflict) for outcome in outcomes) == 1
        async with store.connection.execute(
            "SELECT COUNT(*) FROM memory_fact_revision_states WHERE claim_id = ? AND state = 'active'",
            (first.claim_id,),
        ) as cursor:
            assert int((await cursor.fetchone())[0]) == 1
        assert len(vector.rows) == 1
    finally:
        await other_store.close()
        await store.close()


@pytest.mark.asyncio
async def test_ambiguous_model_contradiction_opens_conflict_and_withholds_vector_truth(tmp_path: Path) -> None:
    vector = FakeVector()
    store, service, authority = await _service(tmp_path / "kai.db", vector)
    try:
        first = await service.create(
            authority,
            _spec("Daniel's preferred path is /srv/old."),
            idempotency_key="fact:create:path",
            stable_claim_key="preferred-path",
        )
        conflict = await service.supersede(
            authority,
            first.claim_id,
            first.revision_id,
            _spec("Daniel's preferred path is /srv/new.", confidence=0.7),
            idempotency_key="fact:model:path-conflict",
            source=FactMutationSource.MODEL,
        )

        assert conflict.state == "unresolved_conflict"
        assert vector.rows == {}
        assert vector.delete_calls == 1
        async with store.connection.execute(
            "SELECT state, COUNT(*) FROM memory_fact_revision_states WHERE claim_id = ? GROUP BY state",
            (first.claim_id,),
        ) as cursor:
            assert [tuple(row) for row in await cursor.fetchall()] == [("unresolved_conflict", 2)]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_conflict_resolution_restores_only_the_selected_truth(tmp_path: Path) -> None:
    vector = FakeVector()
    store, service, authority = await _service(tmp_path / "kai.db", vector)
    try:
        first = await service.create(
            authority,
            _spec("Daniel's preferred path is /srv/old."),
            idempotency_key="fact:create:resolve",
            stable_claim_key="resolved-path",
        )
        conflict = await service.supersede(
            authority,
            first.claim_id,
            first.revision_id,
            _spec("Daniel's preferred path is /srv/new.", confidence=0.7),
            idempotency_key="fact:model:resolve",
            source=FactMutationSource.MODEL,
        )
        resolved = await service.resolve_conflict(
            authority,
            first.claim_id,
            conflict.revision_id,
            (first.revision_id,),
            reason="Daniel selected the corrected path.",
            idempotency_key="fact:resolve:new",
        )

        assert resolved.state == "active"
        assert len(vector.rows) == 1
        assert next(iter(vector.rows.values())).text == "Daniel's preferred path is /srv/new."
        async with store.connection.execute(
            "SELECT state, COUNT(*) FROM memory_fact_revision_states WHERE claim_id = ? GROUP BY state",
            (first.claim_id,),
        ) as cursor:
            assert [tuple(row) for row in await cursor.fetchall()] == [("active", 1), ("superseded", 1)]
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_failed_vector_projection_recovers_from_canonical_revision_without_duplicate(tmp_path: Path) -> None:
    vector = FakeVector()
    vector.fail = True
    store, service, authority = await _service(tmp_path / "kai.db", vector)
    try:
        created = await service.create(
            authority,
            _spec("Daniel uses the Kai workspace."),
            idempotency_key="fact:create:workspace",
            stable_claim_key="current-workspace",
        )
        assert created.state == "active"
        assert created.projection_status == "failed"
        assert vector.rows == {}

        vector.fail = False
        assert await service.recover_pending(retry_failed=True) == 1
        recovered = await service.snapshot(created.claim_id, created.revision_id)
        assert recovered.projection_status == "succeeded"
        assert len(vector.rows) == 1
        assert next(iter(vector.rows.values())).metadata[CANONICAL_REVISION_ID_KEY] == str(created.revision_id)
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_retract_then_restore_creates_new_revision_without_rewriting_history(tmp_path: Path) -> None:
    vector = FakeVector()
    store, service, authority = await _service(tmp_path / "kai.db", vector)
    try:
        created = await service.create(
            authority,
            _spec("Daniel prefers concise answers."),
            idempotency_key="fact:create:concise",
            stable_claim_key="answer-length",
        )
        retracted = await service.retract(
            authority,
            created.claim_id,
            created.revision_id,
            reason="Daniel withdrew this preference.",
            idempotency_key="fact:retract:concise",
        )
        assert retracted.state == "retracted"
        assert vector.rows == {}

        restored = await service.restore(
            authority,
            created.claim_id,
            created.revision_id,
            _spec("Daniel prefers concise answers."),
            idempotency_key="fact:restore:concise",
        )
        assert restored.state == "active"
        assert restored.revision_id != created.revision_id
        assert len(vector.rows) == 1
        async with store.connection.execute(
            "SELECT COUNT(*) FROM memory_fact_revisions WHERE claim_id = ?",
            (created.claim_id,),
        ) as cursor:
            assert int((await cursor.fetchone())[0]) == 2
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_restore_rejects_superseded_revision_while_successor_is_active(tmp_path: Path) -> None:
    vector = FakeVector()
    store, service, authority = await _service(tmp_path / "kai.db", vector)
    try:
        first = await service.create(
            authority,
            _spec("Daniel uses the old setting."),
            idempotency_key="fact:create:restore-guard",
            stable_claim_key="restore-guard",
        )
        await service.supersede(
            authority,
            first.claim_id,
            first.revision_id,
            _spec("Daniel uses the new setting."),
            idempotency_key="fact:supersede:restore-guard",
            source=FactMutationSource.HUMAN,
        )
        with pytest.raises(FactLifecycleConflict):
            await service.restore(
                authority,
                first.claim_id,
                first.revision_id,
                _spec("Daniel uses the old setting again."),
                idempotency_key="fact:restore:invalid",
            )
        assert len(vector.rows) == 1
        assert next(iter(vector.rows.values())).text == "Daniel uses the new setting."
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_expiry_retires_current_truth_without_creating_a_replacement(tmp_path: Path) -> None:
    vector = FakeVector()
    store, service, authority = await _service(tmp_path / "kai.db", vector)
    try:
        first = await service.create(
            authority,
            _spec("Daniel is temporarily using a staging endpoint."),
            idempotency_key="fact:create:temporary",
            stable_claim_key="temporary-endpoint",
        )
        expired = await service.retract(
            authority,
            first.claim_id,
            first.revision_id,
            reason="The stated validity period ended.",
            idempotency_key="fact:expire:temporary",
            expired=True,
        )
        assert expired.state == "expired"
        assert vector.rows == {}
        async with store.connection.execute(
            "SELECT COUNT(*) FROM memory_fact_revisions WHERE claim_id = ?",
            (first.claim_id,),
        ) as cursor:
            assert int((await cursor.fetchone())[0]) == 1
    finally:
        await store.close()
