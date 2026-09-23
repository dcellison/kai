"""
Qualification matrix for temporal memory on a protected install.

Each test drives one lifecycle path through the production layers:
extraction storage, the fact and episode lifecycles, the Mem0 vector
adapters, the protected current-truth gate, and the Workshop memory query
service. Only Mem0's storage is replaced (`FakeMem0` in
`tests.memory_fixtures`); nothing between extraction and recall is
stubbed. Tests that leave facts behind end by proving the same three
things:

1. recall returns exactly the expected current facts;
2. authorized history still shows every revision, whatever its state;
3. canonical state and the vector store agree, with no integrity,
   projection, or drift gaps.

The matrix covers every fact transition, episode history, non-active
exclusion, scopes, several principals and runtime profiles, the Telegram
and Workshop adapters, restart recovery, projection rebuild, replay, and
provider failure. It also checks that no memory operation changes an
owner's conversational backend or model settings. Agent saves and
forget-all through the internal API are qualified in
`tests.test_memory_agent_writes`.
"""

# ruff: noqa: F811 - tests take the imported `protected` fixture as a parameter by name.

from __future__ import annotations

import dataclasses
import sqlite3
from pathlib import Path

import pytest

from kai import memory, memory_extraction
from kai.config import Config, MemoryProjectConfig
from kai.memory_projects import detect_active_memory_project
from kai.workshop.diagnostics import workshop_memory_current_truth_status
from kai.workshop.domain import MemoryClaimId, MemoryRevisionId, RuntimeProfileId
from kai.workshop.episode_history import (
    EpisodeHistoryConflict,
    Mem0EpisodeVectorAdapter,
    MemoryEpisodeHistoryService,
)
from kai.workshop.execution_state import WorkshopExecutionStateRegistry
from kai.workshop.fact_lifecycle import FactLifecycleAccessDenied, FactRevisionInput
from kai.workshop.memory_current_truth import CANONICAL_CLAIM_ID_KEY
from kai.workshop.memory_queries import (
    MemoryFactEdit,
    WorkshopMemoryAccessDenied,
    WorkshopMemoryNotFound,
    WorkshopMemoryQueryService,
)
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.temporal_memory import EpisodeFollowupRelationship
from tests.memory_fixtures import (  # noqa: F401 - pytest fixture import
    CHANNEL_ID,
    PAST,
    PRINCIPAL_ID,
    RUNTIME_ID,
    _add_owner,
    _namespace,
    _new,
    _query_service,
    _recall,
    _register_owners,
    _seed_run,
    _store_fact,
    protected,
)
from tests.test_workshop_episode_history import _spec as _episode_spec

# ── Shared assertions ────────────────────────────────────────────────


async def _assert_consistent(
    service: WorkshopMemoryQueryService,
    authority,
    tmp_path: Path,
    expected: list[str],
    *,
    principal_id=PRINCIPAL_ID,
    runtime_id=RUNTIME_ID,
) -> None:
    """
    Recall shows exactly `expected`, and canonical state and the store agree.

    The vector audit is the Workshop's own "Check search index" read, so a
    duplicate, orphan, unknown, or missing row fails here exactly as the
    owner would see it. The install status line is the operator's view of
    the same state.
    """
    assert _recall(principal_id, runtime_id) == sorted(expected)
    audit = await service.audit_projections(authority)
    assert (audit.orphan, audit.unknown, audit.duplicate, audit.missing) == ((), (), 0, ())
    status = workshop_memory_current_truth_status(tmp_path / "kai.db", memory_enabled=True)
    assert "projection gaps=0, integrity gaps=0" in status
    assert "projection failed=0 (facts=0, episodes=0), blocked=0" in status


async def _history(service: WorkshopMemoryQueryService, authority, claim_id: str) -> list[tuple[str, str]]:
    """Every revision of one claim, newest first, as (content, state) through the owner's authority."""
    detail = await service._fact_lifecycle_detail(authority, claim_id, None)
    assert detail is not None, "the owner can always read their own history"
    revisions = detail["revisions"]
    assert isinstance(revisions, list)
    return [(str(item["content"]), str(item["state"])) for item in revisions]


async def _claim_ids(store) -> list[str]:
    async with store.connection.execute("SELECT claim_id FROM memory_fact_claims ORDER BY created_at") as cursor:
        return [str(row[0]) for row in await cursor.fetchall()]


async def _active_revision(store, claim_id: str) -> str:
    async with store.connection.execute(
        "SELECT revision_id FROM memory_fact_revision_states WHERE claim_id = ? AND state = 'active'", (claim_id,)
    ) as cursor:
        (row,) = await cursor.fetchall()
    return str(row[0])


async def _event_count(store) -> int:
    async with store.connection.execute("SELECT COUNT(*) FROM event_log") as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return int(row[0])


def _update(content: str, existing_id: str, *, confidence: float) -> dict:
    """An extraction result that restates an existing fact."""
    return {**_new(content, confidence=confidence), "intent": "update_of", "existing_id": existing_id}


# ── Fact transitions ─────────────────────────────────────────────────


async def test_create_then_a_repeated_statement_is_not_a_second_fact(protected, tmp_path: Path) -> None:
    store, provider, _lifecycle = protected
    service, authority = _query_service(store, tmp_path)

    await _store_fact(store, _new("The operator prefers dark themes."), run=1)
    repeated = await _store_fact(store, _new("The operator prefers dark themes."), run=2)

    assert [decision.outcome for decision in repeated] == ["duplicate_skipped"]
    assert len(await _claim_ids(store)) == 1 and len(provider.rows) == 1
    await _assert_consistent(service, authority, tmp_path, ["The operator prefers dark themes."])


@pytest.mark.parametrize("source", ["human", "model"])
async def test_a_refinement_supersedes_and_keeps_the_earlier_wording(protected, tmp_path: Path, source: str) -> None:
    store, provider, _lifecycle = protected
    service, authority = _query_service(store, tmp_path)
    await _store_fact(store, _new("The operator prefers dark themes."), run=1)
    (memory_id,) = provider.rows

    if source == "human":
        current = await service.detail(authority, memory_id)
        await service.edit(
            authority,
            memory_id,
            revision=current.record.revision,
            request_id="refine-1",
            edit=MemoryFactEdit("The operator prefers dark themes in every editor.", ("preference",)),
        )
    else:
        await _store_fact(
            store, _update("The operator prefers dark themes in every editor.", memory_id, confidence=0.9), run=2
        )

    (claim_id,) = await _claim_ids(store)
    assert await _history(service, authority, claim_id) == [
        ("The operator prefers dark themes in every editor.", "active"),
        ("The operator prefers dark themes.", "superseded"),
    ]
    await _assert_consistent(service, authority, tmp_path, ["The operator prefers dark themes in every editor."])


@pytest.mark.parametrize("keep", ["earlier", "later"])
async def test_an_unsure_contradiction_conflicts_until_the_owner_keeps_one(
    protected, tmp_path: Path, keep: str
) -> None:
    store, provider, _lifecycle = protected
    service, authority = _query_service(store, tmp_path)
    await _store_fact(store, _new("The operator prefers dark themes."), run=1)
    (memory_id,) = provider.rows

    await _store_fact(store, _update("The operator prefers light themes.", memory_id, confidence=0.6), run=2)

    # A conflict is out of recall until the owner decides, but both sides
    # stay in history.
    (claim_id,) = await _claim_ids(store)
    assert [state for _content, state in await _history(service, authority, claim_id)] == [
        "unresolved_conflict",
        "unresolved_conflict",
    ]
    await _assert_consistent(service, authority, tmp_path, [])

    (conflict,) = (await service.list_conflicts(authority)).items
    revisions = [revision.revision_id for revision in conflict.revisions]
    kept = revisions[0] if keep == "earlier" else revisions[1]
    await service.resolve_conflict(
        authority,
        claim_id,
        keep_revision_id=kept,
        expected_revision_ids=revisions,
        note="",
        client_operation_id=f"resolve-{keep}",
    )

    wording = "The operator prefers dark themes." if keep == "earlier" else "The operator prefers light themes."
    await _assert_consistent(service, authority, tmp_path, [wording])
    assert [state for _content, state in await _history(service, authority, claim_id)].count("active") == 1


@pytest.mark.parametrize("ending", ["retracted", "expired"])
async def test_a_forgotten_or_expired_fact_leaves_recall_and_can_be_restored(
    protected, tmp_path: Path, ending: str
) -> None:
    store, provider, lifecycle = protected
    service, authority = _query_service(store, tmp_path)
    await _store_fact(store, _new("The operator prefers dark themes."), run=1)
    (memory_id,) = provider.rows
    (claim_id,) = await _claim_ids(store)

    if ending == "retracted":
        (result,) = (await service.delete(authority, [memory_id])).results
        assert result.outcome == "succeeded"
    else:
        # Expiry is recorded by reconciliation's "obsolete" decision.
        await lifecycle.retract(
            await lifecycle.authority_for(PRINCIPAL_ID, RUNTIME_ID),
            MemoryClaimId(claim_id),
            MemoryRevisionId(await _active_revision(store, claim_id)),
            reason="No longer true.",
            idempotency_key="expire-1",
            expired=True,
        )

    await _assert_consistent(service, authority, tmp_path, [])
    (forgotten,) = (await service.list_forgotten(authority)).items
    assert forgotten.state == ending

    await service.restore_fact(
        authority, claim_id, revision_id=forgotten.revision_id, note="Still true.", client_operation_id="restore-1"
    )

    await _assert_consistent(service, authority, tmp_path, ["The operator prefers dark themes."])
    assert [state for _content, state in await _history(service, authority, claim_id)] == ["active", ending]


async def test_restore_is_refused_while_the_claim_is_current(protected, tmp_path: Path) -> None:
    store, provider, _lifecycle = protected
    service, authority = _query_service(store, tmp_path)
    await _store_fact(store, _new("The operator prefers dark themes."), run=1)
    (memory_id,) = provider.rows
    current = await service.detail(authority, memory_id)
    await service.edit(
        authority,
        memory_id,
        revision=current.record.revision,
        request_id="refine-1",
        edit=MemoryFactEdit("The operator prefers dark themes everywhere.", ("preference",)),
    )
    (claim_id,) = await _claim_ids(store)
    superseded = next(
        revision["revisionId"]
        for revision in (await service._fact_lifecycle_detail(authority, claim_id, None))["revisions"]  # type: ignore[index, union-attr]
        if revision["state"] == "superseded"
    )

    # Only the latest revision of a forgotten claim is restorable; an
    # older revision of a current claim is not a restore target at all.
    with pytest.raises(WorkshopMemoryNotFound):
        await service.restore_fact(
            authority, claim_id, revision_id=str(superseded), note="", client_operation_id="restore-refused"
        )

    await _assert_consistent(service, authority, tmp_path, ["The operator prefers dark themes everywhere."])


# ── Episodes ─────────────────────────────────────────────────────────


async def test_episodes_record_link_repeats_and_follow_ups_and_stay_immutable(protected, tmp_path: Path) -> None:
    store, _provider, _lifecycle = protected
    service, authority = _query_service(store, tmp_path)
    episodes = MemoryEpisodeHistoryService(store, Mem0EpisodeVectorAdapter())
    owner = await episodes.authority_for(PRINCIPAL_ID, RUNTIME_ID)

    first = await episodes.record(owner, _episode_spec(), idempotency_key="episode-1")

    async def recorded(episode_id) -> tuple:
        async with store.connection.execute(
            "SELECT goal, context, approach, outcome, outcome_quality, lessons FROM memory_episodes WHERE episode_id = ?",
            (str(episode_id),),
        ) as cursor:
            row = await cursor.fetchone()
        assert row is not None
        return tuple(row)

    original = await recorded(first.episode_id)
    repeat = await episodes.record(owner, _episode_spec(), idempotency_key="episode-2")
    later = await episodes.record(owner, _episode_spec(outcome="A later attempt failed."), idempotency_key="episode-3")
    for number, relationship in enumerate(EpisodeFollowupRelationship):
        if relationship is EpisodeFollowupRelationship.REPEATED:
            continue
        await episodes.followup(
            owner,
            later.episode_id,
            first.episode_id,
            relationship,
            reason=f"Follow-up {relationship.value}.",
            idempotency_key=f"followup-{number}",
        )

    async with store.connection.execute(
        "SELECT relationship FROM memory_episode_followups ORDER BY relationship"
    ) as cursor:
        relationships = sorted(str(row[0]) for row in await cursor.fetchall())
    # The near-duplicate check links every closely matching record as a
    # repeat, so `repeated` can appear more than once.
    assert set(relationships) == {item.value for item in EpisodeFollowupRelationship}
    assert repeat.episode_id != first.episode_id
    # Later evidence links to the first account; it never rewrites it.
    assert await recorded(first.episode_id) == original
    # Every episode is recalled as the account it was recorded with.
    assert len(_recall()) == 3
    audit = await service.audit_projections(authority)
    assert (audit.orphan, audit.unknown, audit.duplicate, audit.missing) == ((), (), 0, ())


async def test_follow_ups_cannot_cross_owners(protected, tmp_path: Path) -> None:
    store, _provider, _lifecycle = protected
    other_principal, other_runtime = await _add_owner(store, 2)
    _register_owners((other_principal, other_runtime))
    episodes = MemoryEpisodeHistoryService(store, Mem0EpisodeVectorAdapter())
    mine = await episodes.record(
        await episodes.authority_for(PRINCIPAL_ID, RUNTIME_ID), _episode_spec(), idempotency_key="mine"
    )
    theirs_authority = await episodes.authority_for(other_principal, other_runtime)
    theirs = await episodes.record(theirs_authority, _episode_spec(), idempotency_key="theirs")

    with pytest.raises(EpisodeHistoryConflict, match="cannot cross owner"):
        await episodes.followup(
            theirs_authority,
            theirs.episode_id,
            mine.episode_id,
            EpisodeFollowupRelationship.REVISITED,
            reason="Crossing owners.",
            idempotency_key="cross",
        )


# ── Non-active states ────────────────────────────────────────────────


@pytest.mark.parametrize("state", ["superseded", "retracted", "expired", "unresolved_conflict"])
async def test_every_non_active_state_is_out_of_recall_and_in_history(protected, tmp_path: Path, state: str) -> None:
    store, provider, lifecycle = protected
    service, authority = _query_service(store, tmp_path)
    await _store_fact(store, _new("The operator prefers dark themes."), run=1)
    (memory_id,) = provider.rows
    (claim_id,) = await _claim_ids(store)
    owner = await lifecycle.authority_for(PRINCIPAL_ID, RUNTIME_ID)
    active = MemoryRevisionId(await _active_revision(store, claim_id))

    if state == "superseded":
        await _store_fact(store, _update("The operator prefers light themes.", memory_id, confidence=0.95), run=2)
        expected = ["The operator prefers light themes."]
    elif state == "unresolved_conflict":
        await _store_fact(store, _update("The operator prefers light themes.", memory_id, confidence=0.5), run=2)
        expected = []
    else:
        await lifecycle.retract(
            owner,
            MemoryClaimId(claim_id),
            active,
            reason="Qualification.",
            idempotency_key=f"end-{state}",
            expired=state == "expired",
        )
        expected = []

    assert state in [revision_state for _content, revision_state in await _history(service, authority, claim_id)]
    await _assert_consistent(service, authority, tmp_path, expected)


# ── Scopes ───────────────────────────────────────────────────────────


async def test_project_facts_are_recalled_only_inside_their_project(protected, tmp_path: Path, monkeypatch) -> None:
    store, _provider, _lifecycle = protected
    service, authority = _query_service(store, tmp_path)
    kai_root = tmp_path / "kai"
    other_root = tmp_path / "other"
    kai_root.mkdir()
    other_root.mkdir()
    project = MemoryProjectConfig(
        project_id="kai",
        display_name="Kai",
        workspace_roots=(kai_root.resolve(),),
        memory_enabled=True,
        default_scope_for_new_facts=None,
    )
    # The fake store scores inexact queries low, so the relevance floor is
    # lowered to let scope, the thing under test, decide admission.
    monkeypatch.setattr(
        memory,
        "_config",
        dataclasses.replace(memory._config, memory_projects={"kai": project}, memory_search_floor=0.0),
    )
    # Extraction receives the project detected from the run's workspace.
    active = detect_active_memory_project(kai_root, {"kai": project})
    assert active is not None
    await _store_fact(
        store, {**_new("The build runs make check."), "scope_hint": "project"}, run=1, active_project=active
    )
    await _store_fact(store, _new("The operator prefers dark themes."), run=2)

    async def scoped(workspace: Path) -> list[str]:
        result = await memory.retrieve_scoped_memories(
            memory.ScopedRetrievalContext(
                chat_id=str(PRINCIPAL_ID),
                message="anything",
                workspace=workspace,
                runtime_profile_id=str(RUNTIME_ID),
            )
        )
        return sorted(hit.result.text for hit in result.hits)

    assert await scoped(kai_root) == ["The build runs make check.", "The operator prefers dark themes."]
    assert await scoped(other_root) == ["The operator prefers dark themes."]
    async with store.connection.execute(
        "SELECT scope_kind, scope_key FROM memory_fact_claims ORDER BY scope_kind"
    ) as cursor:
        assert [tuple(row) for row in await cursor.fetchall()] == [("global", ""), ("project", "kai")]
    await _assert_consistent(
        service, authority, tmp_path, ["The build runs make check.", "The operator prefers dark themes."]
    )


# ── Principals and runtime profiles ──────────────────────────────────


async def test_two_owners_never_see_or_change_each_others_memory(protected, tmp_path: Path) -> None:
    store, provider, _lifecycle = protected
    other_principal, other_runtime = await _add_owner(store, 2)
    registry = _register_owners((other_principal, other_runtime))
    mine, my_authority = _query_service(store, tmp_path, registry=registry)
    theirs, their_authority = _query_service(store, tmp_path, registry=registry, principal_id=other_principal)

    await _store_fact(store, _new("The operator prefers dark themes."), run=1)
    await _store_fact(
        store,
        _new("The second owner prefers light themes."),
        run=2,
        principal_id=other_principal,
        runtime_id=other_runtime,
    )
    await _store_fact(store, _new("The operator uses a Mac mini."), run=3)
    their_row = next(
        memory_id
        for memory_id, row in provider.rows.items()
        if row["memory"] == "The second owner prefers light themes."
    )
    their_claim = next(
        str(row["metadata"][CANONICAL_CLAIM_ID_KEY])
        for row in provider.rows.values()
        if row["memory"] == "The second owner prefers light themes."
    )

    # A foreign row is simply not there for the other owner.
    (attempt,) = (await mine.delete(my_authority, [their_row])).results
    assert attempt.outcome == "not_found"
    assert await mine._fact_lifecycle_detail(my_authority, their_claim, None) is None

    await _assert_consistent(
        mine, my_authority, tmp_path, ["The operator prefers dark themes.", "The operator uses a Mac mini."]
    )
    await _assert_consistent(
        theirs,
        their_authority,
        tmp_path,
        ["The second owner prefers light themes."],
        principal_id=other_principal,
        runtime_id=other_runtime,
    )


async def test_an_owner_with_two_runtime_profiles_cannot_mutate_ambiguously(protected, tmp_path: Path) -> None:
    # A principal with two runtime profiles has two memory namespaces, and
    # a mutation that does not name one must fail closed rather than pick.
    store, _provider, _lifecycle = protected
    second = RuntimeProfileId("rtp_23000000000000000000000000000099")
    await store.connection.execute(
        "INSERT INTO runtime_profile_owners (runtime_profile_id, principal_id) VALUES (?, ?)", (second, PRINCIPAL_ID)
    )
    await store.connection.commit()
    registry = WorkshopExecutionStateRegistry((_namespace(), _namespace(PRINCIPAL_ID, second, 9)))
    service, authority = _query_service(store, tmp_path, registry=registry)

    with pytest.raises(WorkshopMemoryAccessDenied):
        await service.create_fact(
            authority, content="Ambiguous.", tags=(), scope="global", project_id=None, request_id="ambiguous-1"
        )


# ── Telegram and Workshop adapters ───────────────────────────────────


async def test_telegram_and_workshop_read_and_change_the_same_memory(protected, tmp_path: Path) -> None:
    store, provider, _lifecycle = protected
    await _store_fact(store, _new("The operator prefers dark themes."), run=1)
    await _store_fact(store, _new("The operator uses a Mac mini."), run=2)
    connection = store.connection
    await connection.execute(
        "INSERT INTO external_identities (id, principal_id, provider, external_subject, created_at) "
        "VALUES ('eid_1', ?, 'telegram', '101', '2026-01-01T00:00:00Z')",
        (str(PRINCIPAL_ID),),
    )
    await connection.execute(
        "INSERT INTO channel_bindings (id, channel_id, transport, external_channel_id, created_at) "
        "VALUES ('cbd_1', ?, 'telegram', '101', '2026-01-01T00:00:00Z')",
        (CHANNEL_ID,),
    )
    await connection.execute(
        "INSERT INTO channel_memberships (id, channel_id, principal_id, role, created_at) "
        "VALUES ('cmb_1', ?, ?, 'participant', '2026-01-01T00:00:00Z')",
        (CHANNEL_ID, str(PRINCIPAL_ID)),
    )
    await connection.commit()
    service, workshop = _query_service(store, tmp_path)

    telegram = await service.authority_for_transport_binding(
        transport="telegram", external_subject="101", external_channel_id="101"
    )

    assert telegram.principal_id == workshop.principal_id
    workshop_page = await service.list_records(workshop)
    telegram_page = await service.list_records(telegram)
    assert [record.memory_id for record in telegram_page.records] == [
        record.memory_id for record in workshop_page.records
    ]
    mac_mini = next(
        memory_id for memory_id, row in provider.rows.items() if row["memory"] == "The operator uses a Mac mini."
    )
    (result,) = (await service.delete(telegram, [mac_mini])).results
    assert result.outcome == "succeeded"
    assert [item.preview for item in (await service.list_forgotten(workshop)).items] == [
        "The operator uses a Mac mini."
    ]
    await _assert_consistent(service, workshop, tmp_path, ["The operator prefers dark themes."])


# ── Restart ──────────────────────────────────────────────────────────


async def test_a_projection_interrupted_mid_write_recovers_without_a_duplicate(protected, tmp_path: Path) -> None:
    # The service stopped after the vector row was written but before the
    # outbox recorded it: the operation is still executing with no row id.
    store, provider, _lifecycle = protected
    service, authority = _query_service(store, tmp_path)
    await _store_fact(store, _new("The operator prefers dark themes."), run=1)
    await store.connection.execute(
        "UPDATE memory_fact_vector_operations SET status = 'executing', memory_id = NULL, completed_at = NULL"
    )
    await store.connection.commit()

    await service.recover_fact_projections()

    assert len(provider.rows) == 1
    await _assert_consistent(service, authority, tmp_path, ["The operator prefers dark themes."])


async def test_a_failed_projection_is_retried_at_startup(protected, tmp_path: Path, monkeypatch) -> None:
    store, provider, _lifecycle = protected
    service, authority = _query_service(store, tmp_path)
    original_add = provider.add

    def refuse(*_args, **_kwargs):
        raise RuntimeError("vector store unavailable")

    monkeypatch.setattr(provider, "add", refuse)
    await _store_fact(store, _new("The operator prefers dark themes."), run=1)
    status = workshop_memory_current_truth_status(tmp_path / "kai.db", memory_enabled=True)
    assert "projection failed=1 (facts=1, episodes=0)" in status
    assert _recall() == []
    monkeypatch.setattr(provider, "add", original_add)

    await service.recover_fact_projections()

    assert len(provider.rows) == 1
    await _assert_consistent(service, authority, tmp_path, ["The operator prefers dark themes."])


# ── Rebuild ──────────────────────────────────────────────────────────


async def test_a_full_projection_rebuild_leaves_recall_and_the_store_unchanged(protected, tmp_path: Path) -> None:
    # Facts are written through event-backed paths only: a rebuild replays
    # events, and the harness seeds extraction runs directly in SQL, which
    # a rebuild would not restore.
    store, provider, _lifecycle = protected
    service, authority = _query_service(store, tmp_path)
    for number, content in enumerate(
        ("The operator prefers dark themes.", "The operator uses a Mac mini.", "The operator drinks Earl Grey."), 1
    ):
        await service.create_fact(
            authority, content=content, tags=(), scope="global", project_id=None, request_id=f"create-{number}"
        )
    rows = dict(provider.rows)
    themes = next(memory_id for memory_id, row in rows.items() if row["memory"] == "The operator prefers dark themes.")
    tea = next(memory_id for memory_id, row in rows.items() if row["memory"] == "The operator drinks Earl Grey.")
    current = await service.detail(authority, themes)
    await service.edit(
        authority,
        themes,
        revision=current.record.revision,
        request_id="refine-1",
        edit=MemoryFactEdit("The operator prefers light themes.", ()),
    )
    (retracted,) = (await service.delete(authority, [tea])).results
    assert retracted.outcome == "succeeded"
    episodes = MemoryEpisodeHistoryService(store, Mem0EpisodeVectorAdapter())
    await episodes.record(
        await episodes.authority_for(PRINCIPAL_ID, RUNTIME_ID), _episode_spec(), idempotency_key="episode-1"
    )

    async def outbox() -> list[tuple]:
        async with store.connection.execute(
            "SELECT event_position, status, memory_id FROM memory_fact_vector_operations "
            "UNION ALL SELECT event_position, status, memory_id FROM memory_episode_vector_operations "
            "ORDER BY event_position"
        ) as cursor:
            return [tuple(row) for row in await cursor.fetchall()]

    before_recall, before_rows, before_outbox = _recall(), dict(provider.rows), await outbox()

    await store.rebuild_projection(CanonicalConversationProjection())

    assert await outbox() == before_outbox
    assert _recall() == before_recall
    # Recovery after a rebuild has nothing left to project again.
    await service.recover_fact_projections()
    assert provider.rows.keys() == before_rows.keys()
    assert len(before_recall) == 3
    await _assert_consistent(service, authority, tmp_path, before_recall)


# ── Replay ───────────────────────────────────────────────────────────


async def _extract_again(store, fact: dict, ids: dict[str, str]) -> list:
    """Run extraction storage for a run, with its receipt and provenance fixed by `ids`."""
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


async def test_storing_the_same_extraction_output_again_writes_nothing_new(protected, tmp_path: Path) -> None:
    store, provider, _lifecycle = protected
    service, authority = _query_service(store, tmp_path)
    ids = await _seed_run(store, 1)
    fact = _new("The operator prefers dark themes.")
    await _extract_again(store, fact, ids)
    events = await _event_count(store)

    await _extract_again(store, fact, ids)

    assert await _event_count(store) == events
    assert len(provider.rows) == 1 and len(await _claim_ids(store)) == 1
    await _assert_consistent(service, authority, tmp_path, ["The operator prefers dark themes."])


async def test_replay_after_a_crash_between_event_and_projection_completes_once(
    protected, tmp_path: Path, monkeypatch
) -> None:
    # The event committed but its projection failed, as if the service
    # stopped mid-write. Extraction never stores a receipt's facts twice
    # (a second pass finds the receipt and does not extract again), so the
    # replay that matters is the identical lifecycle request: it must
    # reuse the committed event and finish the projection exactly once.
    store, provider, lifecycle = protected
    service, authority = _query_service(store, tmp_path)
    owner = await lifecycle.authority_for(PRINCIPAL_ID, RUNTIME_ID)
    spec = FactRevisionInput(
        content="The operator prefers dark themes.",
        scope_kind="global",
        scope_key="",
        reason="Qualification replay.",
        evidence=({"kind": "operator", "reference_id": "replay-1", "sha256": None},),
        vector_metadata={"source": "explicit", "speaker": "user", "confidence": 1.0, "scope": "global"},
        confidence=1.0,
        asserted_at=PAST,
        observed_at=PAST,
        valid_from=PAST,
    )
    original_add = provider.add

    def refuse(*_args, **_kwargs):
        raise RuntimeError("vector store unavailable")

    monkeypatch.setattr(provider, "add", refuse)
    first = await lifecycle.create(owner, spec, idempotency_key="replay-1", stable_claim_key="replay-1")
    assert first.projection_status == "failed"
    events = await _event_count(store)
    monkeypatch.setattr(provider, "add", original_add)

    replay = await lifecycle.create(owner, spec, idempotency_key="replay-1", stable_claim_key="replay-1")

    assert replay.replayed and replay.projection_status == "succeeded"
    assert await _event_count(store) == events
    assert len(provider.rows) == 1 and len(await _claim_ids(store)) == 1
    await _assert_consistent(service, authority, tmp_path, ["The operator prefers dark themes."])


# ── Provider failure ─────────────────────────────────────────────────


async def test_a_failing_vector_read_admits_nothing_stale(protected, tmp_path: Path, monkeypatch) -> None:
    store, provider, _lifecycle = protected
    await _store_fact(store, _new("The operator prefers dark themes."), run=1)

    def unreadable(*_args, **_kwargs):
        raise RuntimeError("vector store unavailable")

    monkeypatch.setattr(provider, "search", unreadable)

    assert _recall() == []


# ── Conversational defaults ──────────────────────────────────────────


async def test_no_memory_operation_changes_backend_or_model_settings(protected, tmp_path: Path) -> None:
    store, provider, lifecycle = protected
    service, authority = _query_service(store, tmp_path)
    await _store_fact(store, _new("The operator prefers dark themes."), run=1)
    for field, value in (("backend", "codex"), ("model", "gpt-5.5"), ("timeout", "600")):
        await store.connection.execute(
            "INSERT INTO channel_agent_execution_settings (channel_id, agent_id, runtime_profile_id, field, value, "
            "updated_at) VALUES (?, 'agt_23000000000000000000000000000001', ?, ?, ?, '2026-01-01T00:00:00Z')",
            (CHANNEL_ID, str(RUNTIME_ID), field, value),
        )
    await store.connection.commit()

    def settings() -> list[tuple]:
        connection = sqlite3.connect(tmp_path / "kai.db")
        try:
            return [
                tuple(row)
                for table in ("channel_agent_execution_settings", "runtime_profile_owners")
                for row in connection.execute(f"SELECT * FROM {table} ORDER BY 1, 2, 3")
            ]
        finally:
            connection.close()

    before = settings()
    (memory_id,) = provider.rows
    await _store_fact(store, _update("The operator prefers light themes.", memory_id, confidence=0.5), run=2)
    (conflict,) = (await service.list_conflicts(authority)).items
    revisions = [revision.revision_id for revision in conflict.revisions]
    await service.resolve_conflict(
        authority,
        conflict.claim_id,
        keep_revision_id=revisions[1],
        expected_revision_ids=revisions,
        note="",
        client_operation_id="resolve-1",
    )
    (current,) = provider.rows
    await service.delete(authority, [current])
    (forgotten,) = (await service.list_forgotten(authority)).items
    await service.restore_fact(
        authority, forgotten.claim_id, revision_id=forgotten.revision_id, note="", client_operation_id="restore-1"
    )
    await service.record_agent_fact(
        principal_id=PRINCIPAL_ID,
        runtime_profile_id=RUNTIME_ID,
        content="The operator drinks Earl Grey.",
        tags=None,
        vector_metadata={
            "source": "explicit",
            **memory.build_scope_metadata(scope="global", project_id=None, scope_source="extraction_default"),
        },
    )
    await service.forget_all_for_agent(principal_id=PRINCIPAL_ID, runtime_profile_id=RUNTIME_ID)
    with pytest.raises(FactLifecycleAccessDenied):
        await lifecycle.authority_for(PRINCIPAL_ID, RuntimeProfileId_unknown())

    assert settings() == before


def RuntimeProfileId_unknown():
    from kai.workshop.domain import RuntimeProfileId

    return RuntimeProfileId("rtp_23000000000000000000000000000077")
