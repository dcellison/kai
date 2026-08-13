"""Contracts for the production-unused durable Workshop run lifecycle."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.domain import (
    AgentId,
    ChannelAgentId,
    EventEnvelope,
    MessageId,
    PrincipalId,
    RunId,
    WorkshopEventType,
)
from kai.workshop.inbound import InboundMessage, record_inbound_message
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.run_lifecycle import (
    RunStatus,
    WorkshopRunLifecycle,
)
from kai.workshop.store import WorkshopEventStore

_NOW = datetime(2026, 8, 12, 16, 0, tzinfo=UTC)


async def _open_with_inbound(path: Path) -> tuple[WorkshopEventStore, MessageId]:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman(
                display_name="Workshop Human",
                role="admin",
                transport="desktop",
                external_subject="human-1",
                external_channel_id="human-1",
            ),
        ),
    )
    inbound = await record_inbound_message(
        store,
        InboundMessage(
            transport="desktop",
            update_id="command-1",
            message_id="message-1",
            sender_subject="human-1",
            channel_subject="human-1",
            body="Perform one durable unit of work",
            occurred_at=_NOW,
        ),
    )
    message_id = inbound.event.envelope.aggregate_id
    assert isinstance(message_id, MessageId)
    return store, message_id


class TestDurableRunAcceptance:
    async def test_rejects_acceptance_before_inbound_message(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            lifecycle = WorkshopRunLifecycle(store)
            with pytest.raises(ValueError, match="before its inbound message"):
                await lifecycle.accept(inbound_id, occurred_at=_NOW - timedelta(seconds=1))

            async with store.connection.execute("SELECT COUNT(*) FROM runs") as cursor:
                assert (await cursor.fetchone())[0] == 0
            async with store.connection.execute(
                "SELECT COUNT(*) FROM event_log WHERE event_type = 'run.accepted'"
            ) as cursor:
                assert (await cursor.fetchone())[0] == 0
        finally:
            await store.close()

    async def test_accepts_canonical_message_without_telegram_identity(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            lifecycle = WorkshopRunLifecycle(store)

            accepted = await lifecycle.accept(inbound_id, occurred_at=_NOW + timedelta(seconds=1))

            assert accepted.changed is True
            assert isinstance(accepted.run.run_id, RunId)
            assert accepted.run.inbound_message_id == inbound_id
            assert accepted.run.status == RunStatus.ACCEPTED
            assert accepted.run.started_at is None
            assert accepted.run.terminal_at is None
            assert accepted.event.envelope.event_type == WorkshopEventType.RUN_ACCEPTED
            assert accepted.event.envelope.actor_principal_id == accepted.run.requested_by_principal_id
            assert accepted.event.envelope.payload == {
                "agent_id": accepted.run.agent_id,
                "channel_id": accepted.run.channel_id,
                "inbound_message_id": inbound_id,
                "requested_by_principal_id": accepted.run.requested_by_principal_id,
            }
            async with store.connection.execute(
                "SELECT COUNT(*) FROM external_identities WHERE provider = 'telegram'"
            ) as cursor:
                assert (await cursor.fetchone())[0] == 0
        finally:
            await store.close()

    async def test_accept_retry_returns_original_event_and_current_state(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            lifecycle = WorkshopRunLifecycle(store)
            first = await lifecycle.accept(inbound_id, occurred_at=_NOW + timedelta(seconds=1))

            retry = await lifecycle.accept(inbound_id, occurred_at=_NOW + timedelta(hours=1))

            assert retry.changed is False
            assert retry.event.position == first.event.position
            assert retry.run.status == RunStatus.ACCEPTED
            async with store.connection.execute(
                "SELECT COUNT(*) FROM event_log WHERE event_type = 'run.accepted'"
            ) as cursor:
                assert (await cursor.fetchone())[0] == 1
        finally:
            await store.close()

    async def test_accept_retry_uses_durable_fact_after_attachment_policy_changes(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            lifecycle = WorkshopRunLifecycle(store)
            first = await lifecycle.accept(inbound_id, occurred_at=_NOW + timedelta(seconds=1))
            second_principal = PrincipalId.new()
            second_agent = AgentId.new()
            await store.connection.execute(
                "INSERT INTO principals (id, kind, display_name, created_at) VALUES (?, 'agent', 'Second agent', ?)",
                (second_principal, _NOW.isoformat()),
            )
            await store.connection.execute(
                "INSERT INTO agents (id, workshop_id, principal_id, name, created_at) "
                "VALUES (?, ?, ?, 'Second agent', ?)",
                (second_agent, first.run.workshop_id, second_principal, _NOW.isoformat()),
            )
            await store.connection.execute(
                "INSERT INTO channel_agents (id, channel_id, agent_id, created_at) VALUES (?, ?, ?, ?)",
                (ChannelAgentId.new(), first.run.channel_id, second_agent, _NOW.isoformat()),
            )
            await store.connection.commit()

            retry = await lifecycle.accept(inbound_id, occurred_at=_NOW + timedelta(hours=1))

            assert retry.changed is False
            assert retry.run.agent_id == first.run.agent_id
            assert retry.event.position == first.event.position
        finally:
            await store.close()


class TestDurableRunReplay:
    async def test_canonical_rebuild_restores_lifecycle_exactly(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            lifecycle = WorkshopRunLifecycle(store)
            before = (await lifecycle.accept(inbound_id, occurred_at=_NOW + timedelta(seconds=1))).run
            await store.connection.execute("DELETE FROM runs")
            await store.connection.commit()

            checkpoint = await store.rebuild_projection(CanonicalConversationProjection())
            after = await lifecycle.state(before.run_id)

            assert checkpoint.version == 8
            assert after == before
        finally:
            await store.close()

    async def test_projection_rejects_completion_without_start(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            lifecycle = WorkshopRunLifecycle(store)
            accepted = await lifecycle.accept(inbound_id, occurred_at=_NOW + timedelta(seconds=1))
            async with store.connection.execute(
                "SELECT principal_id FROM agents WHERE id = ?",
                (accepted.run.agent_id,),
            ) as cursor:
                agent_principal_id = PrincipalId(str((await cursor.fetchone())[0]))
            await store.append(
                EventEnvelope.create(
                    event_type=WorkshopEventType.RUN_COMPLETED,
                    event_version=1,
                    workshop_id=accepted.run.workshop_id,
                    aggregate_type="run",
                    aggregate_id=accepted.run.run_id,
                    actor_principal_id=agent_principal_id,
                    occurred_at=_NOW + timedelta(seconds=2),
                    payload={},
                )
            )

            with pytest.raises(ValueError, match="complete only from started"):
                await store.project_pending(CanonicalConversationProjection())

            assert (await lifecycle.state(accepted.run.run_id)).status == RunStatus.ACCEPTED
        finally:
            await store.close()


class TestDurableRunMigration:
    async def test_version_fourteen_database_adds_empty_run_projection(self, tmp_path: Path, monkeypatch):
        from kai.workshop import schema

        path = tmp_path / "kai.db"
        with monkeypatch.context() as migration_context:
            migration_context.setattr(schema, "WORKSHOP_SCHEMA_VERSION", 14)
            migration_context.setattr(schema, "_MIGRATIONS", schema._MIGRATIONS[:14])
            version_fourteen = await WorkshopEventStore.open(path)
            await version_fourteen.connection.execute(
                "INSERT INTO workshops (id, name, created_at) "
                "VALUES ('wsp_00000000000000000000000000000001', 'Existing', ?)",
                (_NOW.isoformat(),),
            )
            await version_fourteen.connection.commit()
            await version_fourteen.close()

        upgraded = await WorkshopEventStore.open(path)
        try:
            assert await upgraded.schema_version() == 18
            assert "runs" in await upgraded.schema_tables()
            assert "run_attempts" in await upgraded.schema_tables()
            async with upgraded.connection.execute("SELECT name FROM workshops") as cursor:
                assert (await cursor.fetchone())[0] == "Existing"
            async with upgraded.connection.execute("SELECT COUNT(*) FROM runs") as cursor:
                assert (await cursor.fetchone())[0] == 0
        finally:
            await upgraded.close()
