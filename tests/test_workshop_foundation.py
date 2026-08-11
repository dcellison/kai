"""Contract tests for the production-unused Kai Workshop foundation."""

from __future__ import annotations

import asyncio
import stat
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite
import pytest

from kai.workshop.domain import (
    AgentId,
    ChannelId,
    ChannelMembershipId,
    DeliveryId,
    EventEnvelope,
    EventId,
    MessageId,
    PrincipalId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.store import EventIntegrityError, IdempotencyConflictError, Projection, WorkshopEventStore


def _message_event(
    *,
    idempotency_key: str | None = "telegram:update:100:message:200",
    text: str = "hello",
) -> EventEnvelope:
    return EventEnvelope.create(
        event_type="message.created",
        event_version=1,
        workshop_id=WorkshopId("wsp_00000000000000000000000000000001"),
        aggregate_type="message",
        aggregate_id=MessageId("msg_00000000000000000000000000000001"),
        actor_principal_id=PrincipalId("prn_00000000000000000000000000000001"),
        occurred_at=datetime(2026, 8, 11, 12, 0, tzinfo=UTC),
        idempotency_key=idempotency_key,
        payload={
            "channel_id": "chn_00000000000000000000000000000001",
            "text": text,
        },
        metadata={"transport": "telegram"},
    )


class TestWorkshopIdentifiers:
    def test_generated_identifiers_are_typed_unique_and_opaque(self):
        first = WorkshopId.new()
        second = WorkshopId.new()

        assert isinstance(first, WorkshopId)
        assert first.startswith("wsp_")
        assert len(first) == 36
        assert first != second

    @pytest.mark.parametrize(
        ("identifier_type", "value"),
        [
            (WorkshopId, "prn_00000000000000000000000000000001"),
            (PrincipalId, "principal-1"),
            (ChannelId, "chn_not-hex"),
            (ChannelMembershipId, "cag_00000000000000000000000000000001"),
            (AgentId, ""),
            (MessageId, "msg_0000000000000000000000000000000"),
            (DeliveryId, "msg_00000000000000000000000000000001"),
            (EventId, "evt_000000000000000000000000000000011"),
        ],
    )
    def test_identifiers_reject_wrong_kind_or_malformed_values(self, identifier_type, value):
        with pytest.raises(ValueError):
            identifier_type(value)


class TestEventEnvelope:
    def test_initial_event_vocabulary_is_collaboration_only(self):
        assert {event_type.value for event_type in WorkshopEventType} == {
            "workshop.created",
            "workshop.member_added",
            "principal.created",
            "external_identity.bound",
            "channel.created",
            "channel.member_added",
            "transport.channel_bound",
            "agent.created",
            "channel.agent_attached",
            "message.created",
            "delivery.succeeded",
            "delivery.failed",
        }

    def test_create_builds_a_versioned_transport_independent_envelope(self):
        event = _message_event()

        assert event.envelope_version == 1
        assert event.event_type == "message.created"
        assert event.event_version == 1
        assert isinstance(event.event_id, EventId)
        assert isinstance(event.workshop_id, WorkshopId)
        assert isinstance(event.aggregate_id, MessageId)
        assert event.payload["text"] == "hello"
        assert event.metadata == {"transport": "telegram"}

    @pytest.mark.parametrize(
        ("changes", "match"),
        [
            ({"event_type": "Message Created"}, "event_type"),
            ({"event_version": 0}, "event_version"),
            ({"aggregate_type": ""}, "aggregate_type"),
            ({"idempotency_key": ""}, "idempotency_key"),
            ({"occurred_at": datetime(2026, 8, 11, 12, 0)}, "timezone-aware"),
            ({"payload": {"bad": object()}}, "JSON"),
        ],
    )
    def test_invalid_envelopes_fail_before_persistence(self, changes, match):
        values = {
            "event_type": "message.created",
            "event_version": 1,
            "workshop_id": WorkshopId.new(),
            "aggregate_type": "message",
            "aggregate_id": MessageId.new(),
            "occurred_at": datetime.now(UTC),
            "idempotency_key": "test:event:1",
            "payload": {"text": "hello"},
        }
        values.update(changes)

        with pytest.raises(ValueError, match=match):
            EventEnvelope.create(**values)


@pytest.fixture
async def workshop_store(tmp_path: Path):
    store = await WorkshopEventStore.open(tmp_path / "workshop.db")
    yield store
    await store.close()


class TestWorkshopSchema:
    async def test_initial_schema_contains_domain_event_and_projection_records(self, workshop_store):
        expected = {
            "workshop_schema_migrations",
            "workshops",
            "principals",
            "external_identities",
            "workshop_memberships",
            "channels",
            "channel_memberships",
            "channel_bindings",
            "agents",
            "channel_agents",
            "messages",
            "deliveries",
            "event_log",
            "projection_checkpoints",
        }

        assert expected <= await workshop_store.schema_tables()
        assert await workshop_store.schema_version() == 3

    async def test_schema_migration_is_idempotent(self, tmp_path: Path):
        path = tmp_path / "workshop.db"
        first = await WorkshopEventStore.open(path)
        await first.close()

        second = await WorkshopEventStore.open(path)
        assert await second.schema_version() == 3
        async with second.connection.execute(
            "SELECT COUNT(*) FROM workshop_schema_migrations WHERE version = 1"
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 1
        await second.close()

    async def test_version_one_database_upgrades_additively(self, tmp_path: Path, monkeypatch):
        from kai.workshop import schema

        path = tmp_path / "workshop.db"
        with monkeypatch.context() as migration_context:
            migration_context.setattr(schema, "WORKSHOP_SCHEMA_VERSION", 1)
            migration_context.setattr(schema, "_MIGRATIONS", (schema._INITIAL_SCHEMA,))
            version_one = await WorkshopEventStore.open(path)
            assert await version_one.schema_version() == 1
            await version_one.close()

        upgraded = await WorkshopEventStore.open(path)
        try:
            assert await upgraded.schema_version() == 3
            assert {"deliveries", "channel_memberships"} <= await upgraded.schema_tables()
            async with upgraded.connection.execute(
                "SELECT version FROM workshop_schema_migrations ORDER BY version"
            ) as cursor:
                assert [row[0] for row in await cursor.fetchall()] == [1, 2, 3]
        finally:
            await upgraded.close()

    async def test_version_two_database_upgrades_without_replacing_existing_records(self, tmp_path: Path, monkeypatch):
        from kai.workshop import schema

        path = tmp_path / "workshop.db"
        with monkeypatch.context() as migration_context:
            migration_context.setattr(schema, "WORKSHOP_SCHEMA_VERSION", 2)
            migration_context.setattr(schema, "_MIGRATIONS", (schema._INITIAL_SCHEMA, schema._DELIVERY_SCHEMA))
            version_two = await WorkshopEventStore.open(path)
            workshop_id = WorkshopId.new()
            await version_two.connection.execute(
                "INSERT INTO workshops (id, name, created_at) VALUES (?, ?, ?)",
                (workshop_id, "Existing", "2026-08-11T12:00:00Z"),
            )
            await version_two.connection.commit()
            await version_two.close()

        upgraded = await WorkshopEventStore.open(path)
        try:
            assert await upgraded.schema_version() == 3
            assert "channel_memberships" in await upgraded.schema_tables()
            async with upgraded.connection.execute("SELECT name FROM workshops WHERE id = ?", (workshop_id,)) as cursor:
                assert (await cursor.fetchone())[0] == "Existing"
        finally:
            await upgraded.close()

    async def test_schema_is_additive_to_an_existing_kai_database(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        connection = await aiosqlite.connect(path)
        await connection.execute("CREATE TABLE sessions (chat_id INTEGER PRIMARY KEY, session_id TEXT NOT NULL)")
        await connection.execute("INSERT INTO sessions (chat_id, session_id) VALUES (42, 'existing-session')")
        await connection.commit()
        await connection.close()

        store = await WorkshopEventStore.open(path)
        async with store.connection.execute("SELECT session_id FROM sessions WHERE chat_id = 42") as cursor:
            row = await cursor.fetchone()
        assert row[0] == "existing-session"
        await store.close()

    async def test_failed_initial_migration_leaves_no_partial_workshop_schema(self, tmp_path: Path):
        path = tmp_path / "incompatible.db"
        connection = await aiosqlite.connect(path)
        await connection.execute("CREATE TABLE principals (incompatible TEXT)")
        await connection.commit()
        await connection.close()

        with pytest.raises(aiosqlite.OperationalError):
            await WorkshopEventStore.open(path)

        connection = await aiosqlite.connect(path)
        async with connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
        ) as cursor:
            tables = [row[0] for row in await cursor.fetchall()]
        await connection.close()
        assert tables == ["principals"]

    async def test_new_database_is_owner_only(self, tmp_path: Path):
        path = tmp_path / "workshop.db"
        store = await WorkshopEventStore.open(path)
        await store.close()

        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    async def test_domain_foreign_keys_are_enforced(self, workshop_store):
        with pytest.raises(aiosqlite.IntegrityError):
            await workshop_store.connection.execute(
                "INSERT INTO channels (id, workshop_id, kind, created_at) VALUES (?, ?, ?, ?)",
                (ChannelId.new(), WorkshopId.new(), "direct", "2026-08-11T12:00:00Z"),
            )


class TestWorkshopEventStore:
    async def test_append_assigns_monotonic_positions_and_replays_in_order(self, workshop_store):
        first = await workshop_store.append(_message_event(idempotency_key="telegram:message:1", text="one"))
        second = await workshop_store.append(_message_event(idempotency_key="telegram:message:2", text="two"))

        assert first.inserted is True
        assert first.event.position == 1
        assert second.event.position == 2
        assert [event.envelope.payload["text"] for event in await workshop_store.read_events()] == ["one", "two"]

    async def test_idempotent_retry_returns_original_event_without_duplicate(self, workshop_store):
        first = await workshop_store.append(_message_event())
        retry = await workshop_store.append(_message_event())

        assert first.inserted is True
        assert retry.inserted is False
        assert retry.event == first.event
        assert len(await workshop_store.read_events()) == 1

    async def test_idempotency_is_atomic_across_connections(self, tmp_path: Path):
        path = tmp_path / "shared.db"
        first_store = await WorkshopEventStore.open(path)
        second_store = await WorkshopEventStore.open(path)

        first, second = await asyncio.gather(
            first_store.append(_message_event()),
            second_store.append(_message_event()),
        )

        assert sorted((first.inserted, second.inserted)) == [False, True]
        assert first.event == second.event
        assert len(await first_store.read_events()) == 1
        await first_store.close()
        await second_store.close()

    async def test_idempotency_key_reuse_with_different_content_fails_closed(self, workshop_store):
        await workshop_store.append(_message_event(text="original"))

        with pytest.raises(IdempotencyConflictError, match="telegram:update:100:message:200"):
            await workshop_store.append(_message_event(text="changed"))

        assert len(await workshop_store.read_events()) == 1

    async def test_idempotency_key_reuse_with_different_occurrence_time_fails_closed(self, workshop_store):
        original = _message_event()
        await workshop_store.append(original)
        changed = EventEnvelope.create(
            event_type=original.event_type,
            event_version=original.event_version,
            workshop_id=original.workshop_id,
            aggregate_type=original.aggregate_type,
            aggregate_id=original.aggregate_id,
            actor_principal_id=original.actor_principal_id,
            occurred_at=datetime(2026, 8, 11, 12, 1, tzinfo=UTC),
            idempotency_key=original.idempotency_key,
            payload=original.payload,
            metadata=original.metadata,
        )

        with pytest.raises(IdempotencyConflictError, match="telegram:update:100:message:200"):
            await workshop_store.append(changed)

    async def test_duplicate_event_id_with_different_content_fails_closed(self, workshop_store):
        original = _message_event(idempotency_key=None, text="original")
        await workshop_store.append(original)
        changed = EventEnvelope.create(
            event_id=original.event_id,
            event_type=original.event_type,
            event_version=original.event_version,
            workshop_id=original.workshop_id,
            aggregate_type=original.aggregate_type,
            aggregate_id=original.aggregate_id,
            actor_principal_id=original.actor_principal_id,
            occurred_at=original.occurred_at,
            payload={"text": "changed"},
        )

        with pytest.raises(IdempotencyConflictError, match=str(original.event_id)):
            await workshop_store.append(changed)

    async def test_event_id_and_idempotency_key_cannot_resolve_to_different_events(self, workshop_store):
        first = _message_event(idempotency_key="telegram:message:1", text="one")
        second = _message_event(idempotency_key="telegram:message:2", text="two")
        await workshop_store.append(first)
        await workshop_store.append(second)
        crossed = EventEnvelope.create(
            event_id=first.event_id,
            event_type=first.event_type,
            event_version=first.event_version,
            workshop_id=first.workshop_id,
            aggregate_type=first.aggregate_type,
            aggregate_id=first.aggregate_id,
            actor_principal_id=first.actor_principal_id,
            occurred_at=first.occurred_at,
            idempotency_key=second.idempotency_key,
            payload=first.payload,
        )

        with pytest.raises(IdempotencyConflictError, match="different existing events"):
            await workshop_store.append(crossed)

    async def test_replay_detects_stored_content_tampering(self, workshop_store):
        result = await workshop_store.append(_message_event())
        await workshop_store.connection.execute(
            "UPDATE event_log SET payload_json = ? WHERE position = ?",
            ('{"text":"tampered"}', result.event.position),
        )
        await workshop_store.connection.commit()

        with pytest.raises(EventIntegrityError, match=str(result.event.envelope.event_id)):
            await workshop_store.read_events()


class MessageCountProjection(Projection):
    name = "test_message_counts"
    version = 1

    async def reset(self, connection: aiosqlite.Connection) -> None:
        await connection.execute(
            "CREATE TABLE IF NOT EXISTS test_message_counts "
            "(channel_id TEXT PRIMARY KEY, message_count INTEGER NOT NULL)"
        )
        await connection.execute("DELETE FROM test_message_counts")

    async def apply(self, connection: aiosqlite.Connection, event) -> None:
        if event.envelope.event_type != "message.created":
            return
        await connection.execute(
            "INSERT INTO test_message_counts (channel_id, message_count) VALUES (?, 1) "
            "ON CONFLICT(channel_id) DO UPDATE SET message_count = message_count + 1",
            (event.envelope.payload["channel_id"],),
        )


class TestProjectionReplay:
    async def test_rebuild_is_deterministic_and_records_checkpoint(self, workshop_store):
        await workshop_store.append(_message_event(idempotency_key="telegram:message:1", text="one"))
        await workshop_store.append(_message_event(idempotency_key="telegram:message:2", text="two"))
        projection = MessageCountProjection()

        first_checkpoint = await workshop_store.rebuild_projection(projection)
        second_checkpoint = await workshop_store.rebuild_projection(projection)

        assert first_checkpoint == second_checkpoint
        assert first_checkpoint.name == projection.name
        assert first_checkpoint.version == projection.version
        assert first_checkpoint.last_position == 2
        async with workshop_store.connection.execute(
            "SELECT message_count FROM test_message_counts WHERE channel_id = ?",
            ("chn_00000000000000000000000000000001",),
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 2

    async def test_projection_failure_rolls_back_rows_and_checkpoint(self, workshop_store):
        await workshop_store.append(_message_event())
        await workshop_store.connection.execute(
            "CREATE TABLE test_message_counts (channel_id TEXT PRIMARY KEY, message_count INTEGER NOT NULL)"
        )
        await workshop_store.connection.execute(
            "INSERT INTO test_message_counts (channel_id, message_count) VALUES ('preexisting', 7)"
        )
        await workshop_store.connection.commit()

        class FailingProjection(MessageCountProjection):
            name = "test_failing_projection"

            async def apply(self, connection: aiosqlite.Connection, event) -> None:
                await super().apply(connection, event)
                raise RuntimeError("projection failed")

        with pytest.raises(RuntimeError, match="projection failed"):
            await workshop_store.rebuild_projection(FailingProjection())

        async with workshop_store.connection.execute(
            "SELECT name FROM projection_checkpoints WHERE name = ?",
            (FailingProjection.name,),
        ) as cursor:
            assert await cursor.fetchone() is None
        async with workshop_store.connection.execute("SELECT COUNT(*) FROM test_message_counts") as cursor:
            row = await cursor.fetchone()
        assert row[0] == 1
        async with workshop_store.connection.execute(
            "SELECT message_count FROM test_message_counts WHERE channel_id = 'preexisting'"
        ) as cursor:
            row = await cursor.fetchone()
        assert row[0] == 7
