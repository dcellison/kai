"""
Shared harness for real-path memory tests on a protected install.

Memory tests that exercise production behaviour run extraction storage,
the fact and episode lifecycles, the vector adapters, and the protected
current-truth gate for real, replacing only Mem0's storage with
`FakeMem0`. This module holds that one harness so every such test file
builds the same install: a migrated Workshop store with one owning
principal and runtime profile registered as canonical memory authority,
completed runs with real provenance rows, and the Workshop memory query
service over the same store.
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
    Mem0FactVectorAdapter,
    MemoryFactLifecycleService,
)
from kai.workshop.memory_extraction_receipts import MemoryExtractionReceiptService, MemoryExtractionReceiptSpec
from kai.workshop.memory_queries import WorkshopMemoryQueryService
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


async def _seed_run(
    store: WorkshopEventStore,
    number: int,
    *,
    principal_id: PrincipalId = PRINCIPAL_ID,
    runtime_id: RuntimeProfileId = RUNTIME_ID,
) -> dict[str, str]:
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
        (source_id, CHANNEL_ID, str(principal_id), positions[0], stamp),
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
            str(principal_id),
            AGENT_ID,
            source_id,
            stamp,
            stamp,
            stamp,
            positions[2],
            result_id,
            str(runtime_id),
        ),
    )
    await connection.commit()
    claim = await MemoryExtractionReceiptService(connection).claim(
        MemoryExtractionReceiptSpec(
            principal_id=str(principal_id),
            runtime_profile_id=str(runtime_id),
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


async def _store_fact(
    store: WorkshopEventStore,
    fact: dict,
    *,
    run: int,
    principal_id: PrincipalId = PRINCIPAL_ID,
    runtime_id: RuntimeProfileId = RUNTIME_ID,
    active_project: object | None = None,
) -> list:
    """Run one fact through production extraction storage for one owner; return decisions."""
    ids = await _seed_run(store, run, principal_id=principal_id, runtime_id=runtime_id)
    decisions: list = []
    await memory_extraction._store_canonical_facts(
        [fact],
        user_id=str(principal_id),
        session_id="session-1",
        config=Config(telegram_bot_token="token", allowed_user_ids={1}, memory_enabled=True),
        active_project=active_project,  # type: ignore[arg-type]
        user_log=None,
        assistant_log=None,
        canonical_provenance={
            memory.WORKSHOP_RUN_ID_KEY: ids["run"],
            memory.WORKSHOP_SOURCE_MESSAGE_ID_KEY: ids["source"],
            memory.WORKSHOP_RESULT_MESSAGE_ID_KEY: ids["result"],
        },
        runtime_profile_id=str(runtime_id),
        receipt_id=ids["receipt"],
        backend="codex",
        provider="openai",
        model="gpt-5.6-sol",
        receipt_decisions=decisions,
    )
    return decisions


def _recall(principal_id: PrincipalId = PRINCIPAL_ID, runtime_id: RuntimeProfileId = RUNTIME_ID) -> list[str]:
    """Texts visible to protected recall for one owner."""
    return sorted(
        row.text for row in memory.search("anything", user_id=str(principal_id), runtime_profile_id=str(runtime_id))
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


# ── Workshop memory query service ─────────────────────────────────────


class _RuntimePool:
    """Runtime pool stand-in: the owner review paths only need a workspace."""

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace

    async def get_effective_workspace(self, _profile_id) -> Path:
        return self.workspace


def _namespace(
    principal_id: PrincipalId = PRINCIPAL_ID, runtime_id: RuntimeProfileId = RUNTIME_ID, number: int = 1
) -> WorkshopExecutionStateNamespace:
    """The canonical execution namespace of one owner's direct lane."""
    return WorkshopExecutionStateNamespace(
        principal_id=principal_id,
        channel_id=ChannelId(f"chn_{0x23 << 120 | number:032x}"),
        agent_id=AgentId(f"agt_{0x23 << 120 | number:032x}"),
        runtime_profile_id=runtime_id,
        legacy_runtime_key=number,
    )


async def _add_owner(store: WorkshopEventStore, number: int) -> tuple[PrincipalId, RuntimeProfileId]:
    """
    Add another human workshop member with their own runtime profile.

    The new owner becomes canonical memory authority alongside the
    fixture's owner, so recall, history, and lifecycle writes treat the
    two as separate namespaces, as a multi-user install does.
    """
    principal_id = PrincipalId(f"prn_{0x23 << 120 | 0x10 + number:032x}")
    runtime_id = RuntimeProfileId(f"rtp_{0x23 << 120 | 0x10 + number:032x}")
    await store.append(
        _event(WorkshopEventType.PRINCIPAL_CREATED, "principal", principal_id, {"kind": "human", "display_name": "Two"})
    )
    await store.append(
        _event(
            WorkshopEventType.WORKSHOP_MEMBER_ADDED,
            "workshop_membership",
            WorkshopMembershipId(f"wmb_{0x23 << 120 | 0x10 + number:032x}"),
            {"principal_id": str(principal_id), "role": "member"},
        )
    )
    await store.project_pending(CanonicalConversationProjection())
    await store.connection.execute(
        "INSERT INTO runtime_profile_owners (runtime_profile_id, principal_id) VALUES (?, ?)",
        (runtime_id, principal_id),
    )
    await store.connection.commit()
    return principal_id, runtime_id


def _register_owners(*owners: tuple[PrincipalId, RuntimeProfileId]) -> WorkshopExecutionStateRegistry:
    """Register every listed owner (and the fixture's) as canonical memory authority."""
    namespaces = [_namespace()] + [
        _namespace(principal_id, runtime_id, number) for number, (principal_id, runtime_id) in enumerate(owners, 2)
    ]
    registry = WorkshopExecutionStateRegistry(tuple(namespaces))
    memory.configure_memory_authority(registry)
    return registry


def _query_service(
    store,
    tmp_path: Path,
    *,
    registry: WorkshopExecutionStateRegistry | None = None,
    principal_id: PrincipalId = PRINCIPAL_ID,
) -> tuple[WorkshopMemoryQueryService, object]:
    """The Workshop memory query service over the fixture store, with one owner's authority."""
    service = WorkshopMemoryQueryService(
        # The service writes stored audits through `session_db_path`, which
        # in production is the store's own database; the fixture's store
        # lives at tmp_path / "kai.db".
        Config(
            telegram_bot_token="token",
            allowed_user_ids={1},
            memory_enabled=True,
            session_db_path=tmp_path / "kai.db",
        ),
        store,
        _RuntimePool(tmp_path),  # type: ignore[arg-type]
        registry or WorkshopExecutionStateRegistry((_namespace(),)),
    )
    return service, service.authority_for_principal(principal_id)
