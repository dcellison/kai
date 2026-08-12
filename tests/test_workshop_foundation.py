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
    ArtifactId,
    ChannelId,
    ChannelMembershipId,
    DeliveryAttemptId,
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
            (ArtifactId, "msg_00000000000000000000000000000001"),
            (DeliveryId, "msg_00000000000000000000000000000001"),
            (DeliveryAttemptId, "dlv_00000000000000000000000000000001"),
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
            "artifact.created",
            "delivery.requested",
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
            "workshop_client_devices",
            "workshop_client_enrollment_grants",
            "workshop_client_sessions",
            "channel_bindings",
            "agents",
            "channel_agents",
            "messages",
            "artifacts",
            "deliveries",
            "delivery_outbox",
            "delivery_attempts",
            "delivery_fragments",
            "event_log",
            "projection_checkpoints",
        }

        assert expected <= await workshop_store.schema_tables()
        assert await workshop_store.schema_version() == 11
        async with workshop_store.connection.execute("PRAGMA index_list(delivery_outbox)") as cursor:
            indexes = {str(row[1]) for row in await cursor.fetchall()}
        assert "delivery_outbox_binding_order_idx" in indexes
        assert "delivery_outbox_purpose_due_idx" in indexes
        assert "delivery_outbox_purpose_binding_order_idx" in indexes

    async def test_schema_migration_is_idempotent(self, tmp_path: Path):
        path = tmp_path / "workshop.db"
        first = await WorkshopEventStore.open(path)
        await first.close()

        second = await WorkshopEventStore.open(path)
        assert await second.schema_version() == 11
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
            assert await upgraded.schema_version() == 11
            assert {
                "deliveries",
                "channel_memberships",
                "workshop_client_devices",
                "workshop_client_enrollment_grants",
                "workshop_client_sessions",
                "artifacts",
                "delivery_outbox",
                "delivery_attempts",
                "delivery_fragments",
            } <= await upgraded.schema_tables()
            async with upgraded.connection.execute(
                "SELECT version FROM workshop_schema_migrations ORDER BY version"
            ) as cursor:
                assert [row[0] for row in await cursor.fetchall()] == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
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
            assert await upgraded.schema_version() == 11
            assert {
                "channel_memberships",
                "workshop_client_devices",
                "workshop_client_enrollment_grants",
                "workshop_client_sessions",
                "artifacts",
                "delivery_outbox",
                "delivery_attempts",
            } <= await upgraded.schema_tables()
            async with upgraded.connection.execute("SELECT name FROM workshops WHERE id = ?", (workshop_id,)) as cursor:
                assert (await cursor.fetchone())[0] == "Existing"
        finally:
            await upgraded.close()

    async def test_version_three_database_adds_client_auth_without_replacing_principals(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        from kai.workshop import schema

        path = tmp_path / "workshop.db"
        with monkeypatch.context() as migration_context:
            migration_context.setattr(schema, "WORKSHOP_SCHEMA_VERSION", 3)
            migration_context.setattr(
                schema,
                "_MIGRATIONS",
                (schema._INITIAL_SCHEMA, schema._DELIVERY_SCHEMA, schema._CHANNEL_MEMBERSHIP_SCHEMA),
            )
            version_three = await WorkshopEventStore.open(path)
            principal_id = PrincipalId.new()
            await version_three.connection.execute(
                "INSERT INTO principals (id, kind, display_name, created_at) VALUES (?, 'human', ?, ?)",
                (principal_id, "Existing human", "2026-08-11T12:00:00Z"),
            )
            await version_three.connection.commit()
            await version_three.close()

        upgraded = await WorkshopEventStore.open(path)
        try:
            assert await upgraded.schema_version() == 11
            assert {
                "workshop_client_devices",
                "workshop_client_enrollment_grants",
                "workshop_client_sessions",
                "artifacts",
                "delivery_outbox",
                "delivery_attempts",
            } <= await upgraded.schema_tables()
            async with upgraded.connection.execute(
                "SELECT display_name FROM principals WHERE id = ?",
                (principal_id,),
            ) as cursor:
                assert (await cursor.fetchone())[0] == "Existing human"
        finally:
            await upgraded.close()

    async def test_version_four_database_adds_enrollment_without_replacing_sessions(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        from kai.workshop import schema

        path = tmp_path / "workshop.db"
        with monkeypatch.context() as migration_context:
            migration_context.setattr(schema, "WORKSHOP_SCHEMA_VERSION", 4)
            migration_context.setattr(
                schema,
                "_MIGRATIONS",
                (
                    schema._INITIAL_SCHEMA,
                    schema._DELIVERY_SCHEMA,
                    schema._CHANNEL_MEMBERSHIP_SCHEMA,
                    schema._CLIENT_SESSION_SCHEMA,
                ),
            )
            version_four = await WorkshopEventStore.open(path)
            principal_id = PrincipalId.new()
            await version_four.connection.execute(
                "INSERT INTO principals (id, kind, display_name, created_at) VALUES (?, 'human', ?, ?)",
                (principal_id, "Existing human", "2026-08-11T12:00:00Z"),
            )
            await version_four.connection.execute(
                "INSERT INTO workshop_client_devices (id, principal_id, display_name, created_at) "
                "VALUES ('dev_00000000000000000000000000000001', ?, 'Existing device', ?)",
                (principal_id, "2026-08-11T12:00:00Z"),
            )
            await version_four.connection.commit()
            await version_four.close()

        upgraded = await WorkshopEventStore.open(path)
        try:
            assert await upgraded.schema_version() == 11
            assert "workshop_client_enrollment_grants" in await upgraded.schema_tables()
            assert "artifacts" in await upgraded.schema_tables()
            assert "delivery_outbox" in await upgraded.schema_tables()
            async with upgraded.connection.execute(
                "SELECT display_name FROM workshop_client_devices WHERE principal_id = ?",
                (principal_id,),
            ) as cursor:
                assert (await cursor.fetchone())[0] == "Existing device"
        finally:
            await upgraded.close()

    async def test_version_five_database_adds_artifacts_without_replacing_principals(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        from kai.workshop import schema

        path = tmp_path / "workshop.db"
        with monkeypatch.context() as migration_context:
            migration_context.setattr(schema, "WORKSHOP_SCHEMA_VERSION", 5)
            migration_context.setattr(
                schema,
                "_MIGRATIONS",
                (
                    schema._INITIAL_SCHEMA,
                    schema._DELIVERY_SCHEMA,
                    schema._CHANNEL_MEMBERSHIP_SCHEMA,
                    schema._CLIENT_SESSION_SCHEMA,
                    schema._CLIENT_ENROLLMENT_SCHEMA,
                ),
            )
            version_five = await WorkshopEventStore.open(path)
            principal_id = PrincipalId.new()
            await version_five.connection.execute(
                "INSERT INTO principals (id, kind, display_name, created_at) VALUES (?, 'human', ?, ?)",
                (principal_id, "Existing human", "2026-08-11T12:00:00Z"),
            )
            await version_five.connection.commit()
            await version_five.close()

        upgraded = await WorkshopEventStore.open(path)
        try:
            assert await upgraded.schema_version() == 11
            assert "artifacts" in await upgraded.schema_tables()
            assert "delivery_outbox" in await upgraded.schema_tables()
            async with upgraded.connection.execute(
                "SELECT display_name FROM principals WHERE id = ?",
                (principal_id,),
            ) as cursor:
                assert (await cursor.fetchone())[0] == "Existing human"
        finally:
            await upgraded.close()

    async def test_version_six_database_adds_outbox_without_replacing_artifacts(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        from kai.workshop import schema

        path = tmp_path / "workshop.db"
        with monkeypatch.context() as migration_context:
            migration_context.setattr(schema, "WORKSHOP_SCHEMA_VERSION", 6)
            migration_context.setattr(
                schema,
                "_MIGRATIONS",
                (
                    schema._INITIAL_SCHEMA,
                    schema._DELIVERY_SCHEMA,
                    schema._CHANNEL_MEMBERSHIP_SCHEMA,
                    schema._CLIENT_SESSION_SCHEMA,
                    schema._CLIENT_ENROLLMENT_SCHEMA,
                    schema._ARTIFACT_SCHEMA,
                ),
            )
            version_six = await WorkshopEventStore.open(path)
            principal_id = PrincipalId.new()
            await version_six.connection.execute(
                "INSERT INTO principals (id, kind, display_name, created_at) VALUES (?, 'human', ?, ?)",
                (principal_id, "Existing human", "2026-08-11T12:00:00Z"),
            )
            await version_six.connection.commit()
            await version_six.close()

        upgraded = await WorkshopEventStore.open(path)
        try:
            assert await upgraded.schema_version() == 11
            assert {"delivery_outbox", "delivery_attempts"} <= await upgraded.schema_tables()
            async with upgraded.connection.execute(
                "SELECT display_name FROM principals WHERE id = ?",
                (principal_id,),
            ) as cursor:
                assert (await cursor.fetchone())[0] == "Existing human"
        finally:
            await upgraded.close()

    async def test_version_seven_database_adds_delivery_fragments_without_replacing_outbox(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        from kai.workshop import schema

        path = tmp_path / "workshop.db"
        with monkeypatch.context() as migration_context:
            migration_context.setattr(schema, "WORKSHOP_SCHEMA_VERSION", 7)
            migration_context.setattr(
                schema,
                "_MIGRATIONS",
                (
                    schema._INITIAL_SCHEMA,
                    schema._DELIVERY_SCHEMA,
                    schema._CHANNEL_MEMBERSHIP_SCHEMA,
                    schema._CLIENT_SESSION_SCHEMA,
                    schema._CLIENT_ENROLLMENT_SCHEMA,
                    schema._ARTIFACT_SCHEMA,
                    schema._DELIVERY_OUTBOX_SCHEMA,
                ),
            )
            version_seven = await WorkshopEventStore.open(path)
            principal_id = PrincipalId.new()
            await version_seven.connection.execute(
                "INSERT INTO principals (id, kind, display_name, created_at) VALUES (?, 'human', ?, ?)",
                (principal_id, "Existing human", "2026-08-12T12:00:00Z"),
            )
            await version_seven.connection.commit()
            await version_seven.close()

        upgraded = await WorkshopEventStore.open(path)
        try:
            assert await upgraded.schema_version() == 11
            assert {"delivery_outbox", "delivery_attempts", "delivery_fragments"} <= await upgraded.schema_tables()
            async with upgraded.connection.execute(
                "SELECT display_name FROM principals WHERE id = ?",
                (principal_id,),
            ) as cursor:
                assert (await cursor.fetchone())[0] == "Existing human"
        finally:
            await upgraded.close()

    async def test_version_eight_database_migrates_legacy_delivery_identity_additively(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        from kai.workshop import schema

        path = tmp_path / "workshop.db"
        workshop_id = WorkshopId.new()
        principal_id = PrincipalId.new()
        channel_id = ChannelId.new()
        message_id = MessageId.new()
        delivery_id = DeliveryId.new()
        timestamp = "2026-08-12T12:00:00Z"
        with monkeypatch.context() as migration_context:
            migration_context.setattr(schema, "WORKSHOP_SCHEMA_VERSION", 8)
            migration_context.setattr(
                schema,
                "_MIGRATIONS",
                (
                    schema._INITIAL_SCHEMA,
                    schema._DELIVERY_SCHEMA,
                    schema._CHANNEL_MEMBERSHIP_SCHEMA,
                    schema._CLIENT_SESSION_SCHEMA,
                    schema._CLIENT_ENROLLMENT_SCHEMA,
                    schema._ARTIFACT_SCHEMA,
                    schema._DELIVERY_OUTBOX_SCHEMA,
                    schema._DELIVERY_FRAGMENT_SCHEMA,
                ),
            )
            version_eight = await WorkshopEventStore.open(path)
            await version_eight.connection.execute(
                "INSERT INTO workshops (id, name, created_at) VALUES (?, 'Existing', ?)",
                (workshop_id, timestamp),
            )
            await version_eight.connection.execute(
                "INSERT INTO principals (id, kind, display_name, created_at) VALUES (?, 'agent', 'Kai', ?)",
                (principal_id, timestamp),
            )
            await version_eight.connection.execute(
                "INSERT INTO channels (id, workshop_id, kind, created_at) VALUES (?, ?, 'direct', ?)",
                (channel_id, workshop_id, timestamp),
            )
            await version_eight.connection.commit()
            message_event = await version_eight.append(
                EventEnvelope.create(
                    event_type=WorkshopEventType.MESSAGE_CREATED,
                    event_version=1,
                    workshop_id=workshop_id,
                    aggregate_type="message",
                    aggregate_id=message_id,
                    actor_principal_id=principal_id,
                    occurred_at=datetime(2026, 8, 12, 12, 0, tzinfo=UTC),
                    payload={
                        "channel_id": channel_id,
                        "author_principal_id": principal_id,
                        "body": "Existing message",
                    },
                )
            )
            await version_eight.connection.execute(
                "INSERT INTO messages "
                "(id, channel_id, author_principal_id, body, created_event_position, created_at) "
                "VALUES (?, ?, ?, 'Existing message', ?, ?)",
                (message_id, channel_id, principal_id, message_event.event.position, timestamp),
            )
            await version_eight.connection.commit()
            delivery_event = await version_eight.append(
                EventEnvelope.create(
                    event_type=WorkshopEventType.DELIVERY_SUCCEEDED,
                    event_version=1,
                    workshop_id=workshop_id,
                    aggregate_type="delivery",
                    aggregate_id=delivery_id,
                    occurred_at=datetime(2026, 8, 12, 12, 0, tzinfo=UTC),
                    payload={
                        "message_id": message_id,
                        "channel_id": channel_id,
                        "transport": "telegram",
                        "mode": "text",
                    },
                )
            )
            await version_eight.connection.execute(
                "INSERT INTO deliveries "
                "(id, message_id, channel_id, transport, mode, status, created_at, updated_at, "
                "last_event_position) VALUES (?, ?, ?, 'telegram', 'text', 'succeeded', ?, ?, ?)",
                (delivery_id, message_id, channel_id, timestamp, timestamp, delivery_event.event.position),
            )
            await version_eight.connection.commit()
            await version_eight.close()

        upgraded = await WorkshopEventStore.open(path)
        try:
            assert await upgraded.schema_version() == 11
            async with upgraded.connection.execute(
                "SELECT message_id, channel_binding_id, transport, mode, status FROM deliveries WHERE id = ?",
                (delivery_id,),
            ) as cursor:
                assert tuple(await cursor.fetchone()) == (message_id, None, "telegram", "text", "succeeded")
        finally:
            await upgraded.close()

    async def test_version_ten_outbox_rows_migrate_as_qualification_work(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        from kai.workshop import schema
        from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
        from kai.workshop.inbound import InboundMessage, record_inbound_message
        from kai.workshop.outbound import OutboundMessage, record_outbound_message

        path = tmp_path / "workshop.db"
        timestamp = "2026-08-12T12:00:00.000000Z"
        with monkeypatch.context() as migration_context:
            migration_context.setattr(schema, "WORKSHOP_SCHEMA_VERSION", 10)
            migration_context.setattr(schema, "_MIGRATIONS", schema._MIGRATIONS[:10])
            version_ten = await WorkshopEventStore.open(path)
            await bootstrap_default_workshop(
                version_ten,
                (
                    BootstrapHuman(
                        display_name="Existing human",
                        role="admin",
                        transport="telegram",
                        external_subject="101",
                        external_channel_id="101",
                    ),
                ),
            )
            inbound = await record_inbound_message(
                version_ten,
                InboundMessage(
                    transport="telegram",
                    update_id="9001",
                    message_id="42",
                    sender_subject="101",
                    channel_subject="101",
                    body="Existing inbound",
                    occurred_at=datetime(2026, 8, 12, 12, 0, tzinfo=UTC),
                ),
            )
            outbound = await record_outbound_message(
                version_ten,
                OutboundMessage(
                    in_reply_to_message_id=MessageId(str(inbound.event.envelope.aggregate_id)),
                    body="Existing reply",
                    occurred_at=datetime(2026, 8, 12, 12, 0, tzinfo=UTC),
                ),
            )
            message_id = MessageId(str(outbound.event.envelope.aggregate_id))
            async with version_ten.connection.execute(
                "SELECT c.workshop_id, m.channel_id, cb.id, m.author_principal_id "
                "FROM messages m JOIN channels c ON c.id = m.channel_id "
                "JOIN channel_bindings cb ON cb.channel_id = m.channel_id "
                "WHERE m.id = ? AND cb.transport = 'telegram'",
                (message_id,),
            ) as cursor:
                row = await cursor.fetchone()
            assert row is not None
            workshop_id = WorkshopId(str(row[0]))
            channel_id = ChannelId(str(row[1]))
            binding_id = str(row[2])
            delivery_id = DeliveryId.derived(workshop_id, f"delivery:{message_id}:{binding_id}:text")
            requested = await version_ten.append(
                EventEnvelope.create(
                    event_type=WorkshopEventType.DELIVERY_REQUESTED,
                    event_version=1,
                    workshop_id=workshop_id,
                    aggregate_type="delivery",
                    aggregate_id=delivery_id,
                    actor_principal_id=PrincipalId(str(row[3])),
                    occurred_at=datetime(2026, 8, 12, 12, 0, tzinfo=UTC),
                    payload={
                        "message_id": message_id,
                        "channel_id": channel_id,
                        "channel_binding_id": binding_id,
                        "transport": "telegram",
                        "mode": "text",
                        "max_attempts": 3,
                    },
                )
            )
            await version_ten.connection.execute(
                "INSERT INTO delivery_outbox "
                "(id, workshop_id, channel_id, channel_binding_id, message_id, transport, mode, "
                "status, max_attempts, attempt_count, available_at, requested_event_position, "
                "created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, 'telegram', 'text', 'pending', 3, 0, ?, ?, ?, ?)",
                (
                    delivery_id,
                    workshop_id,
                    channel_id,
                    binding_id,
                    message_id,
                    timestamp,
                    requested.event.position,
                    timestamp,
                    timestamp,
                ),
            )
            await version_ten.connection.commit()
            await version_ten.close()

        upgraded = await WorkshopEventStore.open(path)
        try:
            assert await upgraded.schema_version() == 11
            async with upgraded.connection.execute(
                "SELECT purpose, status, attempt_count FROM delivery_outbox WHERE id = ?",
                (delivery_id,),
            ) as cursor:
                assert tuple(await cursor.fetchone()) == ("qualification", "pending", 0)
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
    async def test_transactional_append_requires_and_preserves_caller_transaction(self, workshop_store):
        with pytest.raises(RuntimeError, match="active transaction"):
            await workshop_store.append_in_transaction(_message_event())

        await workshop_store.connection.execute("BEGIN IMMEDIATE")
        result = await workshop_store.append_in_transaction(_message_event())
        assert result.inserted is True
        await workshop_store.connection.rollback()
        assert await workshop_store.read_events() == []

    async def test_transactional_duplicate_does_not_rollback_caller_work(self, workshop_store):
        original = await workshop_store.append(_message_event())
        await workshop_store.connection.execute("CREATE TABLE transaction_probe (value TEXT NOT NULL)")
        await workshop_store.connection.commit()

        await workshop_store.connection.execute("BEGIN IMMEDIATE")
        await workshop_store.connection.execute("INSERT INTO transaction_probe (value) VALUES ('kept')")
        duplicate = await workshop_store.append_in_transaction(_message_event())
        await workshop_store.connection.commit()

        assert duplicate.inserted is False
        assert duplicate.event == original.event
        async with workshop_store.connection.execute("SELECT value FROM transaction_probe") as cursor:
            assert (await cursor.fetchone())[0] == "kept"

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
    async def test_transactional_projection_requires_and_preserves_caller_transaction(self, workshop_store):
        projection = MessageCountProjection()
        with pytest.raises(RuntimeError, match="active transaction"):
            await workshop_store.project_pending_in_transaction(projection)

        await workshop_store.append(_message_event())
        await workshop_store.connection.execute("BEGIN IMMEDIATE")
        checkpoint = await workshop_store.project_pending_in_transaction(projection)
        assert workshop_store.connection.in_transaction is True
        assert checkpoint.last_position == 1
        async with workshop_store.connection.execute("SELECT SUM(message_count) FROM test_message_counts") as cursor:
            assert (await cursor.fetchone())[0] == 1
        await workshop_store.connection.rollback()

        async with workshop_store.connection.execute(
            "SELECT COUNT(*) FROM projection_checkpoints WHERE name = ?",
            (projection.name,),
        ) as cursor:
            assert (await cursor.fetchone())[0] == 0

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
