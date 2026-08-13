"""Tests for deterministic, non-authoritative Workshop bootstrap records."""

from __future__ import annotations

from pathlib import Path

import pytest

from kai import sessions
from kai.workshop.bootstrap import BootstrapHuman, BootstrapNotificationChannel, bootstrap_default_workshop
from kai.workshop.diagnostics import workshop_bootstrap_status
from kai.workshop.store import WorkshopEventStore


@pytest.fixture
async def store(tmp_path: Path):
    event_store = await WorkshopEventStore.open(tmp_path / "kai.db")
    yield event_store
    await event_store.close()


def _human(
    telegram_id: int,
    name: str,
    *,
    role: str = "member",
) -> BootstrapHuman:
    return BootstrapHuman(
        display_name=name,
        role=role,
        transport="telegram",
        external_subject=str(telegram_id),
        external_channel_id=str(telegram_id),
    )


class TestBootstrapInput:
    @pytest.mark.parametrize(
        ("changes", "match"),
        [
            ({"display_name": ""}, "display_name"),
            ({"role": "owner"}, "role"),
            ({"transport": "Telegram"}, "transport"),
            ({"external_subject": ""}, "external_subject"),
            ({"external_channel_id": ""}, "external_channel_id"),
            ({"runtime_profile_id": ""}, "runtime_profile_id"),
        ],
    )
    def test_invalid_human_fails_before_database_changes(self, changes, match):
        values = {
            "display_name": "Daniel",
            "role": "admin",
            "transport": "telegram",
            "external_subject": "123",
            "external_channel_id": "123",
        }
        values.update(changes)

        with pytest.raises(ValueError, match=match):
            BootstrapHuman(**values)

    @pytest.mark.parametrize(
        ("changes", "match"),
        [
            ({"transport": "Telegram"}, "transport"),
            ({"external_channel_id": ""}, "external_channel_id"),
            ({"member_external_subjects": ()}, "member_external_subjects"),
            ({"member_external_subjects": ("101", "101")}, "unique"),
            ({"member_external_subjects": ("",)}, "non-empty"),
        ],
    )
    def test_invalid_notification_channel_fails_before_database_changes(self, changes, match):
        values = {
            "transport": "telegram",
            "external_channel_id": "-100123",
            "member_external_subjects": ("101",),
        }
        values.update(changes)

        with pytest.raises(ValueError, match=match):
            BootstrapNotificationChannel(**values)


class TestDefaultWorkshopBootstrap:
    async def test_creates_default_workshop_agent_humans_and_direct_channels(self, store):
        result = await bootstrap_default_workshop(
            store,
            [_human(202, "Second"), _human(101, "Admin", role="admin")],
        )

        assert result.created_events == 20
        assert result.existing_events == 0
        assert result.human_count == 2
        assert result.channel_count == 2

        connection = store.connection
        async with connection.execute("SELECT id, name FROM workshops") as cursor:
            workshops = await cursor.fetchall()
        async with connection.execute(
            "SELECT id, kind, display_name FROM principals ORDER BY kind, display_name"
        ) as cursor:
            principals = await cursor.fetchall()
        async with connection.execute("SELECT role FROM workshop_memberships ORDER BY role") as cursor:
            roles = [row[0] for row in await cursor.fetchall()]
        async with connection.execute("SELECT name FROM agents") as cursor:
            agents = await cursor.fetchall()
        async with connection.execute("SELECT id, kind FROM channels ORDER BY id") as cursor:
            channels = await cursor.fetchall()
        async with connection.execute(
            "SELECT transport, external_channel_id FROM channel_bindings ORDER BY external_channel_id"
        ) as cursor:
            bindings = await cursor.fetchall()
        async with connection.execute("SELECT COUNT(*) FROM channel_agents") as cursor:
            channel_agent_count = (await cursor.fetchone())[0]
        async with connection.execute(
            "SELECT p.kind, cm.role, COUNT(*) FROM channel_memberships cm "
            "JOIN principals p ON p.id = cm.principal_id GROUP BY p.kind, cm.role ORDER BY p.kind"
        ) as cursor:
            channel_memberships = [tuple(row) for row in await cursor.fetchall()]

        assert len(workshops) == 1
        assert workshops[0][1] == "Kai Workshop"
        assert [(row[1], row[2]) for row in principals] == [
            ("agent", "Kai"),
            ("human", "Admin"),
            ("human", "Second"),
        ]
        assert roles == ["admin", "agent", "member"]
        assert [row[0] for row in agents] == ["Kai"]
        assert len(channels) == 2
        assert all(row[1] == "direct" for row in channels)
        assert [(row[0], row[1]) for row in bindings] == [("telegram", "101"), ("telegram", "202")]
        assert channel_agent_count == 2
        assert channel_memberships == [("agent", "participant", 2), ("human", "owner", 2)]

    async def test_creates_one_outbound_only_notification_channel_for_shared_members(self, store):
        humans = [_human(202, "Second"), _human(101, "Admin", role="admin")]
        notification = BootstrapNotificationChannel(
            transport="telegram",
            external_channel_id="-100123",
            member_external_subjects=("202", "101"),
        )

        first = await bootstrap_default_workshop(
            store,
            humans,
            notification_channels=(notification,),
        )
        second = await bootstrap_default_workshop(
            store,
            list(reversed(humans)),
            notification_channels=(notification,),
        )

        assert first.created_events == 26
        assert first.channel_count == 3
        assert second.created_events == 0
        assert second.existing_events == 26
        async with store.connection.execute(
            "SELECT c.kind, c.name, cb.transport, cb.external_channel_id "
            "FROM channels c JOIN channel_bindings cb ON cb.channel_id = c.id "
            "WHERE c.kind = 'notification'"
        ) as cursor:
            row = await cursor.fetchone()
        assert tuple(row) == ("notification", "Notifications", "telegram", "-100123")
        async with store.connection.execute(
            "SELECT p.kind, cm.role FROM channel_memberships cm "
            "JOIN channels c ON c.id = cm.channel_id "
            "JOIN principals p ON p.id = cm.principal_id "
            "WHERE c.kind = 'notification' ORDER BY p.kind, p.display_name"
        ) as cursor:
            memberships = [tuple(row) for row in await cursor.fetchall()]
        assert memberships == [
            ("agent", "participant"),
            ("human", "participant"),
            ("human", "participant"),
        ]

    async def test_notification_channel_requires_configured_members_and_unique_destination(self, store):
        notification = BootstrapNotificationChannel("telegram", "-100123", ("999",))
        with pytest.raises(ValueError, match="configured external identity"):
            await bootstrap_default_workshop(
                store,
                [_human(101, "Admin")],
                notification_channels=(notification,),
            )
        assert await store.read_events() == []

        collision = BootstrapNotificationChannel("telegram", "101", ("101",))
        with pytest.raises(ValueError, match="Duplicate bootstrap external channel"):
            await bootstrap_default_workshop(
                store,
                [_human(101, "Admin")],
                notification_channels=(collision,),
            )
        assert await store.read_events() == []

    async def test_human_principal_and_conversation_have_distinct_ids(self, store):
        await bootstrap_default_workshop(store, [_human(101, "Admin", role="admin")])

        async with store.connection.execute(
            "SELECT p.id, c.id FROM principals p "
            "JOIN external_identities e ON e.principal_id = p.id "
            "JOIN channel_bindings b ON b.external_channel_id = e.external_subject "
            "JOIN channels c ON c.id = b.channel_id "
            "WHERE p.kind = 'human'"
        ) as cursor:
            row = await cursor.fetchone()

        assert row[0].startswith("prn_")
        assert row[1].startswith("chn_")
        assert row[0] != row[1]

    async def test_rerun_is_idempotent_and_preserves_stable_ids(self, store):
        first = await bootstrap_default_workshop(store, [_human(101, "Admin", role="admin")])
        first_events = await store.read_events()

        second = await bootstrap_default_workshop(store, [_human(101, "Admin", role="admin")])
        second_events = await store.read_events()

        assert first.created_events == 12
        assert second.created_events == 0
        assert second.existing_events == 12
        assert second.workshop_id == first.workshop_id
        assert second.agent_id == first.agent_id
        assert second_events == first_events

    async def test_rerun_adds_one_runtime_assignment_without_replacing_transport_identity(
        self,
        store,
    ):
        original = _human(101, "Admin", role="admin")
        first = await bootstrap_default_workshop(store, [original])
        upgraded = await bootstrap_default_workshop(
            store,
            [
                BootstrapHuman(
                    display_name=original.display_name,
                    role=original.role,
                    transport=original.transport,
                    external_subject=original.external_subject,
                    external_channel_id=original.external_channel_id,
                    runtime_profile_id="101",
                )
            ],
        )

        assert first.created_events == 12
        assert upgraded.created_events == 1
        assert upgraded.existing_events == 12
        async with store.connection.execute(
            "SELECT provider, external_subject FROM external_identities ORDER BY provider"
        ) as cursor:
            assert [tuple(row) for row in await cursor.fetchall()] == [("telegram", "101")]
        async with store.connection.execute(
            "SELECT runtime_profile_id FROM channel_agent_runtime_assignments"
        ) as cursor:
            assert [str(row[0]) for row in await cursor.fetchall()] == ["101"]

    async def test_input_order_does_not_change_event_or_projection_order(self, tmp_path: Path):
        first_store = await WorkshopEventStore.open(tmp_path / "first.db")
        second_store = await WorkshopEventStore.open(tmp_path / "second.db")
        users = [_human(202, "Second"), _human(101, "Admin", role="admin")]

        await bootstrap_default_workshop(first_store, users)
        await bootstrap_default_workshop(second_store, list(reversed(users)))

        first_types = [event.envelope.event_type for event in await first_store.read_events()]
        second_types = [event.envelope.event_type for event in await second_store.read_events()]
        async with first_store.connection.execute(
            "SELECT external_channel_id FROM channel_bindings ORDER BY rowid"
        ) as cursor:
            first_channels = [row[0] for row in await cursor.fetchall()]
        async with second_store.connection.execute(
            "SELECT external_channel_id FROM channel_bindings ORDER BY rowid"
        ) as cursor:
            second_channels = [row[0] for row in await cursor.fetchall()]

        assert first_types == second_types
        assert first_channels == second_channels == ["101", "202"]
        await first_store.close()
        await second_store.close()

    async def test_new_configured_human_can_be_added_after_initial_bootstrap(self, store):
        await bootstrap_default_workshop(store, [_human(101, "Admin", role="admin")])

        result = await bootstrap_default_workshop(
            store,
            [_human(101, "Admin", role="admin"), _human(202, "Second")],
        )

        assert result.created_events == 8
        assert result.existing_events == 12
        assert result.human_count == 2
        assert result.channel_count == 2
        assert len(await store.read_events()) == 20

    async def test_existing_bootstrap_receives_missing_channel_memberships(self, store):
        await bootstrap_default_workshop(
            store,
            [_human(101, "Admin", role="admin"), _human(202, "Second")],
        )
        await store.connection.execute("DELETE FROM channel_memberships")
        await store.connection.execute("DELETE FROM event_log WHERE event_type = 'channel.member_added'")
        await store.connection.commit()

        result = await bootstrap_default_workshop(
            store,
            [_human(101, "Admin", role="admin"), _human(202, "Second")],
        )

        assert result.created_events == 4
        assert result.existing_events == 16
        async with store.connection.execute("SELECT COUNT(*) FROM channel_memberships") as cursor:
            assert (await cursor.fetchone())[0] == 4

    async def test_duplicate_external_identity_or_channel_is_rejected(self, store):
        with pytest.raises(ValueError, match="Duplicate bootstrap external identity"):
            await bootstrap_default_workshop(
                store,
                [_human(101, "First"), _human(101, "Duplicate")],
            )

        assert await store.read_events() == []

    async def test_existing_bootstrap_metadata_is_not_silently_rewritten(self, store):
        await bootstrap_default_workshop(store, [_human(101, "Original", role="member")])

        result = await bootstrap_default_workshop(store, [_human(101, "Changed", role="admin")])

        assert result.created_events == 0
        async with store.connection.execute(
            "SELECT p.display_name, m.role FROM principals p "
            "JOIN external_identities e ON e.principal_id = p.id "
            "JOIN workshop_memberships m ON m.principal_id = p.id "
            "WHERE e.provider = 'telegram' AND e.external_subject = '101'"
        ) as cursor:
            row = await cursor.fetchone()
        assert tuple(row) == ("Original", "member")


class TestKaiDatabaseIntegration:
    async def test_sessions_initialization_applies_workshop_schema_atomically(self, tmp_path: Path):
        await sessions.init_db(tmp_path / "kai.db")
        try:
            tables = await WorkshopEventStore.from_initialized_connection(sessions._get_db()).schema_tables()
            assert {"sessions", "event_log", "workshops", "projection_checkpoints"} <= tables
        finally:
            await sessions.close_db()

    async def test_sessions_bootstrap_wrapper_uses_the_shared_database(self, tmp_path: Path):
        await sessions.init_db(tmp_path / "kai.db")
        try:
            result = await sessions.bootstrap_workshop_foundation((_human(101, "Admin", role="admin"),))
            assert result.human_count == 1
            async with sessions._get_db().execute("SELECT COUNT(*) FROM workshops") as cursor:
                assert (await cursor.fetchone())[0] == 1
        finally:
            await sessions.close_db()


class TestWorkshopBootstrapStatus:
    def test_missing_database_reports_pending_without_identity_data(self, tmp_path: Path):
        status = workshop_bootstrap_status(tmp_path / "kai.db", expected_humans=2)

        assert status.startswith("Workshop bootstrap: pending")
        assert "2 human principal(s)" in status

    async def test_initialized_database_reports_counts_only(self, tmp_path: Path):
        db_path = tmp_path / "kai.db"
        event_store = await WorkshopEventStore.open(db_path)
        try:
            await bootstrap_default_workshop(
                event_store,
                [
                    BootstrapHuman(
                        "Secret Name",
                        "admin",
                        "telegram",
                        "101",
                        "101",
                        runtime_profile_id="101",
                    )
                ],
            )
        finally:
            await event_store.close()

        status = workshop_bootstrap_status(db_path, expected_humans=1)

        assert status == (
            "Workshop bootstrap: initialized; workshops=1, humans=1, Telegram bindings=1, "
            "channel memberships=2, agents=1, runtime assignments=1; expected humans=1"
        )
        assert "Secret Name" not in status
        assert "101" not in status

    async def test_schema_without_bootstrap_reports_pending(self, tmp_path: Path):
        db_path = tmp_path / "kai.db"
        event_store = await WorkshopEventStore.open(db_path)
        await event_store.close()

        status = workshop_bootstrap_status(db_path, expected_humans=1)

        assert status.startswith("Workshop bootstrap: pending;")
        assert "humans=0" in status

    def test_invalid_database_reports_not_verified(self, tmp_path: Path):
        db_path = tmp_path / "kai.db"
        db_path.write_text("not a SQLite database")

        status = workshop_bootstrap_status(db_path, expected_humans=1)

        assert status.startswith("Workshop bootstrap: NOT VERIFIED (DatabaseError)")
