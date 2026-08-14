"""Integrated shared-writer/dedicated-worker contract for the first cutover."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

from kai import sessions
from kai.workshop.bootstrap import BootstrapHuman
from kai.workshop.delivery_authority import WorkshopConversationDeliveryAuthority
from kai.workshop.domain import MessageId
from kai.workshop.inbound import InboundMessage
from kai.workshop.outbound import OutboundMessage
from kai.workshop.store import WorkshopEventStore
from kai.workshop.streaming_preview import ConfirmedTelegramStreamingPreview
from kai.workshop.telegram_delivery_runtime import WorkshopTelegramConversationDeliveryService

_NOW = datetime(2026, 1, 1, 18, 0, tzinfo=UTC)


async def test_shared_finalizer_and_dedicated_worker_finalize_one_preview_without_send(tmp_path: Path):
    database = tmp_path / "kai.db"
    await sessions.init_db(database)
    service: WorkshopTelegramConversationDeliveryService | None = None
    try:
        await sessions.bootstrap_workshop_foundation((BootstrapHuman("Operator", "admin", "telegram", "101", "101"),))
        inbound = await sessions.record_workshop_inbound_message(
            InboundMessage(
                transport="telegram",
                update_id="9001",
                message_id="41",
                sender_subject="101",
                channel_subject="101",
                body="Hello",
                occurred_at=_NOW,
            )
        )
        inbound_id = inbound.event.envelope.aggregate_id
        assert isinstance(inbound_id, MessageId)

        bot = AsyncMock()
        bot.edit_message_text.return_value = SimpleNamespace(message_id=7001)
        authority_store = await WorkshopEventStore.open(database)
        try:
            epoch = (await WorkshopConversationDeliveryAuthority(authority_store).activate()).epoch
        finally:
            await authority_store.close()
        service = await WorkshopTelegramConversationDeliveryService.open_and_start(
            database,
            bot,
            authority_epoch_id=epoch.epoch_id,
        )
        await sessions.record_workshop_streaming_preview(
            ConfirmedTelegramStreamingPreview(
                inbound_message_id=inbound_id,
                external_message_id=7001,
                confirmed_at=_NOW + timedelta(seconds=1),
            )
        )
        finalization = await sessions.record_workshop_streaming_finalization(
            OutboundMessage(
                in_reply_to_message_id=inbound_id,
                body="Durable final answer",
                occurred_at=_NOW + timedelta(seconds=2),
            )
        )

        for _ in range(200):
            async with sessions._get_db().execute(
                "SELECT status FROM delivery_outbox WHERE id = ?",
                (finalization.delivery.delivery.delivery_id,),
            ) as cursor:
                row = await cursor.fetchone()
            if row is not None and row[0] == "succeeded":
                break
            await asyncio.sleep(0.01)
        else:
            async with sessions._get_db().execute(
                "SELECT status, last_error_code FROM delivery_outbox WHERE id = ?",
                (finalization.delivery.delivery.delivery_id,),
            ) as cursor:
                unresolved = tuple(await cursor.fetchone())
            raise AssertionError(
                "Dedicated Workshop worker did not settle the cutover delivery: "
                f"state={unresolved}, runtime_failure={service.runtime.failure!r}"
            )

        bot.edit_message_text.assert_awaited_once()
        assert bot.edit_message_text.await_args.kwargs["chat_id"] == 101
        assert bot.edit_message_text.await_args.kwargs["message_id"] == 7001
        assert bot.edit_message_text.await_args.kwargs["text"] == "Durable final answer"
        bot.send_message.assert_not_awaited()
    finally:
        if service is not None:
            await service.stop()
        await sessions.close_db()
