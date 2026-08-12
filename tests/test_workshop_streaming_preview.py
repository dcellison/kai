"""Contracts for durable, production-unused Telegram streaming previews."""

from __future__ import annotations

import asyncio
from dataclasses import fields
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiosqlite
import pytest

from kai.workshop.bootstrap import (
    BootstrapHuman,
    BootstrapNotificationChannel,
    bootstrap_default_workshop,
)
from kai.workshop.domain import (
    ChannelBindingId,
    EventEnvelope,
    MessageId,
    PrincipalId,
    WorkshopEventType,
    WorkshopId,
)
from kai.workshop.inbound import InboundMessage, record_inbound_message
from kai.workshop.outbound import OutboundMessage, record_outbound_message
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.store import WorkshopEventStore
from kai.workshop.streaming_preview import (
    ConfirmedTelegramStreamingPreview,
    StreamingPreviewBindingError,
    StreamingPreviewConflictError,
    StreamingPreviewTargetError,
    bind_confirmed_telegram_streaming_preview,
)

_NOW = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)


async def _open_with_inbound(path: Path) -> tuple[WorkshopEventStore, MessageId]:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman("First human", "admin", "telegram", "101", "101"),
            BootstrapHuman("Second human", "member", "telegram", "202", "202"),
        ),
        notification_channels=(BootstrapNotificationChannel("telegram", "-1001", ("101", "202")),),
    )
    inbound = await _record_inbound(store, update_id="9001", message_id="42")
    return store, inbound


async def _record_inbound(
    store: WorkshopEventStore,
    *,
    update_id: str,
    message_id: str,
) -> MessageId:
    result = await record_inbound_message(
        store,
        InboundMessage(
            transport="telegram",
            update_id=update_id,
            message_id=message_id,
            sender_subject="101",
            channel_subject="101",
            body="Hello",
            occurred_at=_NOW,
        ),
    )
    assert isinstance(result.event.envelope.aggregate_id, MessageId)
    return result.event.envelope.aggregate_id


def _preview(
    inbound_message_id: MessageId,
    *,
    external_message_id: int = 7001,
    confirmed_at: datetime = _NOW + timedelta(seconds=1),
) -> ConfirmedTelegramStreamingPreview:
    return ConfirmedTelegramStreamingPreview(
        inbound_message_id=inbound_message_id,
        external_message_id=external_message_id,
        confirmed_at=confirmed_at,
    )


class TestConfirmedTelegramStreamingPreviewInput:
    def test_accepts_only_canonical_inbound_id_external_message_id_and_time(self):
        assert [field.name for field in fields(ConfirmedTelegramStreamingPreview)] == [
            "inbound_message_id",
            "external_message_id",
            "confirmed_at",
        ]

    @pytest.mark.parametrize("external_message_id", [True, False, 0, -1, 1 << 63, "7001"])
    def test_rejects_invalid_external_message_ids(self, external_message_id):
        with pytest.raises(ValueError, match="positive signed 64-bit"):
            _preview(MessageId.new(), external_message_id=external_message_id)

    def test_rejects_naive_confirmation_time(self):
        with pytest.raises(ValueError, match="timezone-aware"):
            _preview(MessageId.new(), confirmed_at=datetime(2026, 8, 12, 12, 0))


class TestTelegramStreamingPreviewBinding:
    async def test_resolves_direct_channel_and_binding_without_destination_input(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            result = await bind_confirmed_telegram_streaming_preview(store, _preview(inbound_id))

            assert result.inserted is True
            assert result.inbound_message_id == inbound_id
            assert isinstance(result.workshop_id, WorkshopId)
            assert isinstance(result.channel_binding_id, ChannelBindingId)
            assert result.external_message_id == 7001
            assert result.state == "confirmed_non_final"
            async with store.connection.execute(
                "SELECT c.kind, cb.transport, cb.external_channel_id "
                "FROM channels c JOIN channel_bindings cb ON cb.channel_id = c.id "
                "WHERE c.id = ? AND cb.id = ?",
                (result.channel_id, result.channel_binding_id),
            ) as cursor:
                assert tuple(await cursor.fetchone()) == ("direct", "telegram", "101")
        finally:
            await store.close()

    async def test_retry_is_idempotent_across_restart_and_observation_time(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        store, inbound_id = await _open_with_inbound(path)
        first = await bind_confirmed_telegram_streaming_preview(store, _preview(inbound_id))
        await store.close()

        reopened = await WorkshopEventStore.open(path)
        try:
            retry = await bind_confirmed_telegram_streaming_preview(
                reopened,
                _preview(inbound_id, confirmed_at=_NOW + timedelta(minutes=5)),
            )
            assert retry.inserted is False
            assert retry.confirmed_at == first.confirmed_at
            assert retry.channel_binding_id == first.channel_binding_id
            assert retry.external_message_id == first.external_message_id
            async with reopened.connection.execute("SELECT COUNT(*) FROM telegram_streaming_previews") as cursor:
                assert (await cursor.fetchone())[0] == 1
        finally:
            await reopened.close()

    async def test_concurrent_retry_creates_one_binding(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        first_store, inbound_id = await _open_with_inbound(path)
        second_store = await WorkshopEventStore.open(path)
        try:
            first, second = await asyncio.gather(
                bind_confirmed_telegram_streaming_preview(first_store, _preview(inbound_id)),
                bind_confirmed_telegram_streaming_preview(second_store, _preview(inbound_id)),
            )
            assert sorted((first.inserted, second.inserted)) == [False, True]
            assert first.channel_binding_id == second.channel_binding_id
            assert first.external_message_id == second.external_message_id
        finally:
            await first_store.close()
            await second_store.close()

    async def test_changed_preview_id_for_same_inbound_fails_closed(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            await bind_confirmed_telegram_streaming_preview(store, _preview(inbound_id))
            with pytest.raises(StreamingPreviewConflictError, match="different"):
                await bind_confirmed_telegram_streaming_preview(
                    store,
                    _preview(inbound_id, external_message_id=7002),
                )
        finally:
            await store.close()

    async def test_same_external_preview_cannot_cross_inbound_messages(self, tmp_path: Path):
        store, first_inbound = await _open_with_inbound(tmp_path / "kai.db")
        try:
            second_inbound = await _record_inbound(store, update_id="9002", message_id="43")
            await bind_confirmed_telegram_streaming_preview(store, _preview(first_inbound))
            with pytest.raises(StreamingPreviewConflictError, match="different canonical inbound"):
                await bind_confirmed_telegram_streaming_preview(store, _preview(second_inbound))
        finally:
            await store.close()

    async def test_missing_and_assistant_messages_are_not_eligible_targets(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            with pytest.raises(StreamingPreviewTargetError):
                await bind_confirmed_telegram_streaming_preview(store, _preview(MessageId.new()))

            outbound = await record_outbound_message(
                store,
                OutboundMessage(inbound_id, "Assistant reply", _NOW + timedelta(seconds=2)),
            )
            assistant_id = MessageId(str(outbound.event.envelope.aggregate_id))
            with pytest.raises(StreamingPreviewTargetError):
                await bind_confirmed_telegram_streaming_preview(store, _preview(assistant_id))
        finally:
            await store.close()

    async def test_non_telegram_direct_message_is_not_a_preview_target(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            async with store.connection.execute(
                "SELECT c.workshop_id, m.channel_id, m.author_principal_id "
                "FROM messages m JOIN channels c ON c.id = m.channel_id WHERE m.id = ?",
                (inbound_id,),
            ) as cursor:
                row = await cursor.fetchone()
            assert row is not None
            message_id = MessageId.new()
            await store.append(
                EventEnvelope.create(
                    event_type=WorkshopEventType.MESSAGE_CREATED,
                    event_version=1,
                    workshop_id=WorkshopId(str(row[0])),
                    aggregate_type="message",
                    aggregate_id=message_id,
                    actor_principal_id=PrincipalId(str(row[2])),
                    occurred_at=_NOW,
                    payload={
                        "channel_id": str(row[1]),
                        "author_principal_id": str(row[2]),
                        "body": "Canonical, but not Telegram inbound",
                    },
                    metadata={"source": "workshop_client"},
                )
            )
            await store.project_pending(CanonicalConversationProjection())

            with pytest.raises(StreamingPreviewTargetError, match="Telegram inbound"):
                await bind_confirmed_telegram_streaming_preview(store, _preview(message_id))
        finally:
            await store.close()

    async def test_notification_channel_message_is_not_a_direct_preview_target(self, tmp_path: Path):
        store, _ = await _open_with_inbound(tmp_path / "kai.db")
        try:
            async with store.connection.execute(
                "SELECT c.workshop_id, c.id, p.id FROM channels c "
                "JOIN channel_memberships cm ON cm.channel_id = c.id "
                "JOIN principals p ON p.id = cm.principal_id AND p.kind = 'human' "
                "WHERE c.kind = 'notification' ORDER BY p.id LIMIT 1"
            ) as cursor:
                row = await cursor.fetchone()
            assert row is not None
            message_id = MessageId.new()
            await store.append(
                EventEnvelope.create(
                    event_type=WorkshopEventType.MESSAGE_CREATED,
                    event_version=1,
                    workshop_id=WorkshopId(str(row[0])),
                    aggregate_type="message",
                    aggregate_id=message_id,
                    actor_principal_id=PrincipalId(str(row[2])),
                    occurred_at=_NOW,
                    payload={
                        "channel_id": str(row[1]),
                        "author_principal_id": str(row[2]),
                        "body": "Not an inbound direct message",
                    },
                )
            )
            await store.project_pending(CanonicalConversationProjection())

            with pytest.raises(StreamingPreviewTargetError, match="direct channel"):
                await bind_confirmed_telegram_streaming_preview(store, _preview(message_id))
        finally:
            await store.close()

    async def test_missing_or_ambiguous_telegram_binding_fails_without_persisting(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            async with store.connection.execute(
                "SELECT channel_id FROM messages WHERE id = ?", (inbound_id,)
            ) as cursor:
                channel_id = str((await cursor.fetchone())[0])
            await store.connection.execute(
                "DELETE FROM channel_bindings WHERE channel_id = ? AND transport = 'telegram'",
                (channel_id,),
            )
            await store.connection.commit()
            with pytest.raises(StreamingPreviewBindingError):
                await bind_confirmed_telegram_streaming_preview(store, _preview(inbound_id))

            await store.rebuild_projection(CanonicalConversationProjection())
            async with store.connection.execute(
                "SELECT c.workshop_id FROM channels c WHERE c.id = ?", (channel_id,)
            ) as cursor:
                workshop_id = WorkshopId(str((await cursor.fetchone())[0]))
            await store.append(
                EventEnvelope.create(
                    event_type=WorkshopEventType.TRANSPORT_CHANNEL_BOUND,
                    event_version=1,
                    workshop_id=workshop_id,
                    aggregate_type="channel_binding",
                    aggregate_id=ChannelBindingId.new(),
                    occurred_at=_NOW,
                    payload={
                        "channel_id": channel_id,
                        "transport": "telegram",
                        "external_channel_id": "303",
                    },
                )
            )
            await store.project_pending(CanonicalConversationProjection())
            with pytest.raises(StreamingPreviewBindingError):
                await bind_confirmed_telegram_streaming_preview(store, _preview(inbound_id))

            async with store.connection.execute("SELECT COUNT(*) FROM telegram_streaming_previews") as cursor:
                assert (await cursor.fetchone())[0] == 0
        finally:
            await store.close()

    async def test_projection_rebuild_preserves_confirmed_external_effect(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            first = await bind_confirmed_telegram_streaming_preview(store, _preview(inbound_id))

            await store.rebuild_projection(CanonicalConversationProjection())

            retry = await bind_confirmed_telegram_streaming_preview(store, _preview(inbound_id))
            assert retry.inserted is False
            assert retry.channel_binding_id == first.channel_binding_id
            assert retry.external_message_id == first.external_message_id
        finally:
            await store.close()

    async def test_insert_failure_rolls_back_without_partial_binding(self, tmp_path: Path):
        store, inbound_id = await _open_with_inbound(tmp_path / "kai.db")
        try:
            await store.connection.execute(
                "CREATE TRIGGER reject_preview BEFORE INSERT ON telegram_streaming_previews "
                "BEGIN SELECT RAISE(ABORT, 'test preview rejection'); END"
            )
            await store.connection.commit()

            with pytest.raises(aiosqlite.IntegrityError, match="test preview rejection"):
                await bind_confirmed_telegram_streaming_preview(store, _preview(inbound_id))
            async with store.connection.execute("SELECT COUNT(*) FROM telegram_streaming_previews") as cursor:
                assert (await cursor.fetchone())[0] == 0
        finally:
            await store.close()


class TestTelegramStreamingPreviewMigration:
    async def test_version_eleven_database_upgrades_additively(self, tmp_path: Path, monkeypatch):
        from kai.workshop import schema

        path = tmp_path / "kai.db"
        with monkeypatch.context() as migration_context:
            migration_context.setattr(schema, "WORKSHOP_SCHEMA_VERSION", 11)
            migration_context.setattr(schema, "_MIGRATIONS", schema._MIGRATIONS[:11])
            version_eleven = await WorkshopEventStore.open(path)
            assert "telegram_streaming_previews" not in await version_eleven.schema_tables()
            await version_eleven.close()

        upgraded = await WorkshopEventStore.open(path)
        try:
            assert await upgraded.schema_version() == 12
            assert "telegram_streaming_previews" in await upgraded.schema_tables()
            async with upgraded.connection.execute(
                "SELECT name FROM workshop_schema_migrations WHERE version = 12"
            ) as cursor:
                assert (await cursor.fetchone())[0] == "durable_telegram_streaming_preview_bindings"
        finally:
            await upgraded.close()
