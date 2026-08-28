"""Canonical semantic-memory ownership, migration, and diagnostics."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from kai import memory, sessions
from kai.memory_extraction import _store_facts
from kai.workshop.bootstrap import BootstrapHuman
from kai.workshop.diagnostics import workshop_memory_authority_status
from kai.workshop.domain import AgentId, ChannelId, PrincipalId
from kai.workshop.execution_state import (
    WorkshopExecutionStateNamespace,
    WorkshopExecutionStateRegistry,
)
from kai.workshop.memory_authority import (
    WorkshopMemoryAuthorityError,
    memory_authority_registry_from_database,
)
from kai.workshop.store import WorkshopEventStore
from tests.workshop_profiles import profile_id, profile_registry


class _FakeVectorStore:
    def __init__(self, memory_store: _FakeMem0) -> None:
        self._memory_store = memory_store

    def update(self, *, vector_id: str, payload: dict[str, object]) -> None:
        row = self._memory_store.rows[vector_id]
        for key, value in payload.items():
            if key == "user_id":
                row["user_id"] = str(value)
            else:
                row.setdefault("metadata", {})[key] = value


class _FakeMem0:
    def __init__(self, rows: list[dict] | None = None) -> None:
        self.rows = {str(row["id"]): dict(row) for row in rows or []}
        self.next_id = 1
        self.vector_store = _FakeVectorStore(self)

    @staticmethod
    def _result(row: dict) -> dict:
        return {
            "id": row["id"],
            "memory": row["memory"],
            "metadata": dict(row.get("metadata", {})),
            "user_id": row["user_id"],
            "score": row.get("score", 0.9),
            "created_at": row.get("created_at", "2026-08-15T00:00:00Z"),
            "updated_at": row.get("updated_at", "2026-08-15T00:00:00Z"),
        }

    def get_all(self, *, filters: dict, top_k: int) -> dict:
        del top_k
        return {"results": [self._result(row) for row in self.rows.values() if row["user_id"] == filters["user_id"]]}

    def search(self, query: str, *, filters: dict, top_k: int) -> dict:
        del query
        return {"results": self.get_all(filters=filters, top_k=top_k)["results"][:top_k]}

    def add(self, content: str, *, user_id: str, infer: bool, metadata: dict) -> dict:
        assert infer is False
        memory_id = f"mem-{self.next_id}"
        self.next_id += 1
        self.rows[memory_id] = {
            "id": memory_id,
            "memory": content,
            "user_id": user_id,
            "metadata": dict(metadata),
        }
        return {"results": [{"id": memory_id, "memory": content}]}

    def update(self, *, memory_id: str, data: str, metadata: dict) -> dict:
        row = self.rows[memory_id]
        updated = dict(metadata)
        row["user_id"] = str(updated.pop("user_id", row["user_id"]))
        row["memory"] = data
        row["metadata"] = updated
        return {"message": "updated"}

    def get(self, memory_id: str) -> dict | None:
        row = self.rows.get(memory_id)
        return self._result(row) if row is not None else None

    def delete(self, memory_id: str) -> None:
        self.rows.pop(memory_id, None)

    def delete_all(self, *, user_id: str) -> None:
        self.rows = {key: row for key, row in self.rows.items() if row["user_id"] != user_id}


def _namespace(runtime_config_id: int = 101) -> WorkshopExecutionStateNamespace:
    return WorkshopExecutionStateNamespace(
        principal_id=PrincipalId.new(),
        channel_id=ChannelId.new(),
        agent_id=AgentId.new(),
        runtime_profile_id=profile_id(runtime_config_id),
        legacy_runtime_key=runtime_config_id,
    )


@pytest.fixture(autouse=True)
def _reset_memory_authority():
    memory._memory = None
    memory._config = None
    memory.configure_memory_authority(None)
    yield
    memory._memory = None
    memory._config = None
    memory.configure_memory_authority(None)


class TestCanonicalMemoryNamespace:
    def test_execution_registry_keeps_exact_profiles_when_principal_is_shared(self):
        principal_id = PrincipalId.new()
        first = WorkshopExecutionStateNamespace(
            principal_id=principal_id,
            channel_id=ChannelId.new(),
            agent_id=AgentId.new(),
            runtime_profile_id=profile_id(101),
            legacy_runtime_key=101,
        )
        second = WorkshopExecutionStateNamespace(
            principal_id=principal_id,
            channel_id=ChannelId.new(),
            agent_id=AgentId.new(),
            runtime_profile_id=profile_id(202),
            legacy_runtime_key=202,
        )

        registry = WorkshopExecutionStateRegistry((first, second))

        assert registry.maybe_for_runtime_config_id(101) == first
        assert registry.maybe_for_runtime_config_id(202) == second
        assert registry.maybe_for_principal_id(str(principal_id)) is None

    def test_rekeys_in_place_stamps_owner_and_is_idempotent(self):
        namespace = _namespace()
        fake = _FakeMem0(
            [
                {
                    "id": "legacy-id",
                    "memory": "Legacy fact",
                    "user_id": "101",
                    "metadata": {"source": "extracted", "scope": "global"},
                },
                {
                    "id": "partial-id",
                    "memory": "Already moved fact",
                    "user_id": str(namespace.principal_id),
                    "metadata": {"source": "episode", "scope": "global"},
                },
            ]
        )
        memory._memory = fake

        first = memory.migrate_memory_namespace(namespace)
        second = memory.migrate_memory_namespace(namespace)

        assert first == memory.CanonicalMemoryMigrationResult(moved=1, stamped=1, total=2)
        assert second == memory.CanonicalMemoryMigrationResult(moved=0, stamped=0, total=2)
        assert set(fake.rows) == {"legacy-id", "partial-id"}
        assert {row["memory"] for row in fake.rows.values()} == {"Legacy fact", "Already moved fact"}
        assert {row["user_id"] for row in fake.rows.values()} == {str(namespace.principal_id)}
        for row in fake.rows.values():
            assert row["metadata"][memory.WORKSHOP_PRINCIPAL_ID_KEY] == str(namespace.principal_id)
            assert row["metadata"][memory.WORKSHOP_CHANNEL_ID_KEY] == str(namespace.channel_id)
            assert row["metadata"][memory.WORKSHOP_AGENT_ID_KEY] == str(namespace.agent_id)
            assert row["metadata"][memory.WORKSHOP_RUNTIME_PROFILE_ID_KEY] == str(namespace.runtime_profile_id)

    def test_protected_reads_and_writes_use_only_the_canonical_owner(self):
        namespace = _namespace()
        registry = WorkshopExecutionStateRegistry((namespace,))
        fake = _FakeMem0()
        memory._memory = fake
        memory._config = SimpleNamespace(memory_search_limit=10)
        memory.configure_memory_authority(registry)

        memory_id = memory.add_structured(
            "Canonical fact",
            user_id="101",
            memory_type="fact",
            metadata={
                memory.WORKSHOP_RUN_ID_KEY: "run-1",
                memory.WORKSHOP_SOURCE_MESSAGE_ID_KEY: "message-in",
                memory.WORKSHOP_RESULT_MESSAGE_ID_KEY: "message-out",
            },
        )

        assert memory_id is not None
        row = fake.rows[memory_id]
        assert row["user_id"] == str(namespace.principal_id)
        assert row["metadata"][memory.WORKSHOP_PRINCIPAL_ID_KEY] == str(namespace.principal_id)
        assert row["metadata"][memory.WORKSHOP_RUN_ID_KEY] == "run-1"
        assert [result.id for result in memory.get_all(user_id="101")] == [memory_id]
        assert [result.id for result in memory.get_all(user_id=str(namespace.principal_id))] == [memory_id]

        row["metadata"][memory.WORKSHOP_PRINCIPAL_ID_KEY] = str(PrincipalId.new())
        assert memory.get_all(user_id="101") == []
        assert memory.search("Canonical", user_id="101") == []
        assert memory.get_by_id(user_id="101", memory_id=memory_id) is None

    def test_conflicting_owner_metadata_fails_closed(self):
        """Fail-closed means the misowned row is never stored. The
        refusal surfaces as the documented never-raise failure shape
        (None plus a warning) rather than an escaping
        CanonicalMemoryAuthorityError; the audit flagged the old
        raise as a contract violation, and callers of add_structured
        are written against the None contract."""
        namespace = _namespace()
        fake = _FakeMem0()
        memory._memory = fake
        memory._config = SimpleNamespace(memory_search_limit=10)
        memory.configure_memory_authority(WorkshopExecutionStateRegistry((namespace,)))

        result = memory.add_structured(
            "Misowned fact",
            user_id="101",
            memory_type="fact",
            metadata={memory.WORKSHOP_PRINCIPAL_ID_KEY: str(PrincipalId.new())},
        )

        assert result is None
        assert memory.get_all(user_id="101") == []

    def test_partial_external_store_move_resumes_without_duplication(self):
        namespace = _namespace()
        fake = _FakeMem0(
            [
                {"id": "one", "memory": "One", "user_id": "101", "metadata": {}},
                {"id": "two", "memory": "Two", "user_id": "101", "metadata": {}},
            ]
        )
        memory._memory = fake
        original_update = fake.vector_store.update
        calls = 0

        def interrupted_update(**kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError("simulated interruption")
            return original_update(**kwargs)

        fake.vector_store.update = interrupted_update  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="simulated interruption"):
            memory.migrate_memory_namespace(namespace)

        fake.vector_store.update = original_update  # type: ignore[method-assign]
        resumed = memory.migrate_memory_namespace(namespace)

        assert resumed == memory.CanonicalMemoryMigrationResult(moved=1, stamped=0, total=2)
        assert set(fake.rows) == {"one", "two"}
        assert {row["user_id"] for row in fake.rows.values()} == {str(namespace.principal_id)}

    def test_shared_principal_keeps_profile_provenance_disjoint_during_migration(self):
        principal_id = PrincipalId.new()
        first = WorkshopExecutionStateNamespace(
            principal_id=principal_id,
            channel_id=ChannelId.new(),
            agent_id=AgentId.new(),
            runtime_profile_id=profile_id(101),
            legacy_runtime_key=101,
        )
        second = WorkshopExecutionStateNamespace(
            principal_id=principal_id,
            channel_id=ChannelId.new(),
            agent_id=AgentId.new(),
            runtime_profile_id=profile_id(202),
            legacy_runtime_key=202,
        )
        fake = _FakeMem0(
            [
                {"id": "first", "memory": "First", "user_id": "101", "metadata": {}},
                {"id": "second", "memory": "Second", "user_id": "202", "metadata": {}},
            ]
        )
        memory._memory = fake

        first_result = memory.migrate_memory_namespace(first, sibling_namespaces=(second,))
        second_result = memory.migrate_memory_namespace(second, sibling_namespaces=(first,))

        assert first_result == memory.CanonicalMemoryMigrationResult(moved=1, stamped=0, total=1)
        assert second_result == memory.CanonicalMemoryMigrationResult(moved=1, stamped=0, total=1)
        assert {row["user_id"] for row in fake.rows.values()} == {str(principal_id)}
        assert fake.rows["first"]["metadata"][memory.WORKSHOP_RUNTIME_PROFILE_ID_KEY] == str(first.runtime_profile_id)
        assert fake.rows["second"]["metadata"][memory.WORKSHOP_RUNTIME_PROFILE_ID_KEY] == str(second.runtime_profile_id)

    def test_shared_principal_rejects_ambiguous_unowned_canonical_rows(self):
        principal_id = PrincipalId.new()
        first = WorkshopExecutionStateNamespace(
            principal_id=principal_id,
            channel_id=ChannelId.new(),
            agent_id=AgentId.new(),
            runtime_profile_id=profile_id(101),
            legacy_runtime_key=101,
        )
        second = WorkshopExecutionStateNamespace(
            principal_id=principal_id,
            channel_id=ChannelId.new(),
            agent_id=AgentId.new(),
            runtime_profile_id=profile_id(202),
            legacy_runtime_key=202,
        )
        memory._memory = _FakeMem0(
            [{"id": "ambiguous", "memory": "Unknown lane", "user_id": str(principal_id), "metadata": {}}]
        )

        with pytest.raises(memory.CanonicalMemoryAuthorityError, match="restore the memory and database backups"):
            memory.migrate_memory_namespace(first, sibling_namespaces=(second,))

    def test_exact_runtime_authority_stamps_shared_principal_provenance(self):
        principal_id = PrincipalId.new()
        first = WorkshopExecutionStateNamespace(
            principal_id=principal_id,
            channel_id=ChannelId.new(),
            agent_id=AgentId.new(),
            runtime_profile_id=profile_id(101),
            legacy_runtime_key=101,
        )
        second = WorkshopExecutionStateNamespace(
            principal_id=principal_id,
            channel_id=ChannelId.new(),
            agent_id=AgentId.new(),
            runtime_profile_id=profile_id(202),
            legacy_runtime_key=202,
        )
        fake = _FakeMem0(
            [
                {"id": "first", "memory": "First", "user_id": "101", "metadata": {}},
                {"id": "second", "memory": "Second", "user_id": "202", "metadata": {}},
            ]
        )
        memory._memory = fake
        memory._config = SimpleNamespace(memory_search_limit=10)
        registry = WorkshopExecutionStateRegistry((first, second))
        memory.configure_memory_authority(registry)
        memory.migrate_memory_namespace(first, sibling_namespaces=(second,))
        memory.migrate_memory_namespace(second, sibling_namespaces=(first,))

        first_hits = memory.search(
            "fact",
            user_id=str(principal_id),
            runtime_profile_id=str(first.runtime_profile_id),
        )
        second_hits = memory.search(
            "fact",
            user_id=str(principal_id),
            runtime_profile_id=str(second.runtime_profile_id),
        )
        added = memory.add_structured(
            "First runtime only",
            user_id=str(principal_id),
            runtime_profile_id=str(first.runtime_profile_id),
        )

        # Memory ownership is intentionally per-principal: both runtime
        # profiles recall the human's full corpus. Exact runtime authority is
        # still required to stamp and validate the provenance of new writes.
        assert [row.id for row in first_hits] == ["first", "second"]
        assert [row.id for row in second_hits] == ["first", "second"]
        assert added is not None
        assert fake.rows[added]["metadata"][memory.WORKSHOP_RUNTIME_PROFILE_ID_KEY] == str(first.runtime_profile_id)

    def test_exact_runtime_authority_rejects_wrong_principal(self):
        namespace = _namespace(101)
        memory._memory = _FakeMem0()
        memory._config = SimpleNamespace(memory_search_limit=10)
        memory.configure_memory_authority(WorkshopExecutionStateRegistry((namespace,)))

        with pytest.raises(memory.CanonicalMemoryAuthorityError, match="does not belong"):
            memory.search(
                "fact",
                user_id=str(PrincipalId.new()),
                runtime_profile_id=str(namespace.runtime_profile_id),
            )

    def test_two_protected_humans_have_disjoint_memory_namespaces(self):
        first = _namespace(101)
        second = _namespace(202)
        fake = _FakeMem0()
        memory._memory = fake
        memory._config = SimpleNamespace(memory_search_limit=10)
        memory.configure_memory_authority(WorkshopExecutionStateRegistry((first, second)))

        first_id = memory.add_structured("First", user_id="101", memory_type="fact")
        second_id = memory.add_structured("Second", user_id="202", memory_type="fact")

        assert [row.id for row in memory.get_all(user_id="101")] == [first_id]
        assert [row.id for row in memory.get_all(user_id="202")] == [second_id]

    def test_extracted_fact_carries_canonical_run_and_message_provenance(self, monkeypatch):
        captured: dict[str, object] = {}
        monkeypatch.setattr("kai.memory_extraction._paraphrase_neighbor", lambda *args, **kwargs: None)
        monkeypatch.setattr(
            "kai.memory_extraction.memory.add_structured",
            lambda content, **kwargs: captured.update(kwargs) or "memory-id",
        )
        provenance = {
            memory.WORKSHOP_RUN_ID_KEY: "run-1",
            memory.WORKSHOP_SOURCE_MESSAGE_ID_KEY: "message-in",
            memory.WORKSHOP_RESULT_MESSAGE_ID_KEY: "message-out",
        }

        stored = _store_facts(
            [
                {
                    "content": "Canonical provenance",
                    "intent": "new",
                    "speaker": "user",
                    "confidence": 1.0,
                    "tags": ["architecture"],
                }
            ],
            user_id="101",
            session_id="session-1",
            config=SimpleNamespace(memory_duplicate_threshold=0.95),
            canonical_provenance=provenance,
        )

        assert stored == (1, 0, 0)
        metadata = captured["metadata"]
        assert isinstance(metadata, dict)
        assert {key: metadata[key] for key in provenance} == provenance


class TestCanonicalMemoryAuthorityMigration:
    async def test_receipt_is_durable_and_changed_ownership_fails_closed(self, tmp_path: Path, monkeypatch):
        database = tmp_path / "kai.db"
        await sessions.init_db(database)
        try:
            await sessions.bootstrap_workshop_foundation(
                (BootstrapHuman("Human 101", "admin", "telegram", "101", "101", profile_id(101)),)
            )
            registry, _ = await sessions.initialize_workshop_execution_state(profile_registry(101))
            fake = _FakeMem0(
                [{"id": "stable", "memory": "Fact", "user_id": "101", "metadata": {"source": "extracted"}}]
            )
            memory._memory = fake
            memory._config = SimpleNamespace(memory_search_limit=10)
            memory.configure_memory_authority(registry)

            first = await sessions.initialize_workshop_memory_authority(registry)
            second = await sessions.initialize_workshop_memory_authority(registry)

            assert first.newly_migrated == 1
            assert first.moved == 1
            assert second.newly_migrated == 0
            assert workshop_memory_authority_status(database, memory_enabled=True).startswith(
                "Workshop memory authority: active;"
            )
            offline_registry = memory_authority_registry_from_database(database)
            assert offline_registry is not None
            assert offline_registry.maybe_for_runtime_config_id(101) == registry.maybe_for_runtime_config_id(101)
            initialized: list[object] = []
            monkeypatch.setattr(memory, "init_memory", initialized.append)
            memory.configure_memory_authority(None)
            offline_config = SimpleNamespace(session_db_path=database)
            memory.init_offline_memory(offline_config)  # type: ignore[arg-type]
            assert initialized == [offline_config]
            assert memory.canonical_memory_user_id("101") == str(registry.namespaces[0].principal_id)

            await sessions._get_db().execute(
                "UPDATE workshop_memory_authority_migrations SET channel_id = ?",
                (ChannelId.new(),),
            )
            await sessions._get_db().commit()
            with pytest.raises(WorkshopMemoryAuthorityError, match="conflicts"):
                await sessions.initialize_workshop_memory_authority(registry)
        finally:
            await sessions.close_db()

    async def test_adding_empty_profile_for_existing_principal_does_not_break_restart(self, tmp_path: Path):
        database = tmp_path / "kai.db"
        await sessions.init_db(database)
        try:
            principal_id = PrincipalId.new()
            first = WorkshopExecutionStateNamespace(
                principal_id=principal_id,
                channel_id=ChannelId.new(),
                agent_id=AgentId.new(),
                runtime_profile_id=profile_id(101),
                legacy_runtime_key=101,
            )
            second = WorkshopExecutionStateNamespace(
                principal_id=principal_id,
                channel_id=ChannelId.new(),
                agent_id=AgentId.new(),
                runtime_profile_id=profile_id(202),
                legacy_runtime_key=202,
            )
            fake = _FakeMem0([{"id": "first", "memory": "First", "user_id": "101", "metadata": {}}])
            memory._memory = fake

            initial = await sessions.initialize_workshop_memory_authority(WorkshopExecutionStateRegistry((first,)))
            restarted = await sessions.initialize_workshop_memory_authority(
                WorkshopExecutionStateRegistry((first, second))
            )

            assert initial.newly_migrated == 1
            assert restarted.newly_migrated == 1
            assert restarted.moved == 0
            assert restarted.total == 0
        finally:
            await sessions.close_db()

    async def test_version_twenty_one_database_upgrades_additively(self, tmp_path: Path, monkeypatch):
        from kai.workshop import schema

        database = tmp_path / "preexisting.db"
        with monkeypatch.context() as migration_context:
            migration_context.setattr(schema, "WORKSHOP_SCHEMA_VERSION", 21)
            migration_context.setattr(schema, "_MIGRATIONS", schema._MIGRATIONS[:21])
            old_store = await WorkshopEventStore.open(database)
            await old_store.close()

        upgraded = await WorkshopEventStore.open(database)
        try:
            assert await upgraded.schema_version() == 42
            assert "workshop_memory_authority_migrations" in await upgraded.schema_tables()
        finally:
            await upgraded.close()

    def test_disabled_policy_does_not_report_a_migration_backlog(self, tmp_path: Path):
        assert workshop_memory_authority_status(tmp_path / "missing.db", memory_enabled=False) == (
            "Workshop memory authority: disabled by policy"
        )
