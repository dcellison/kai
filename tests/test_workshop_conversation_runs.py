"""Contracts for transport-neutral canonical Workshop conversation runs."""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from kai.backend import AgentResponse, StreamEvent
from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.conversation_runs import (
    CanonicalConversationRunTarget,
    CompatibilityConversationRunResolution,
    ConversationRunUnavailableError,
    WorkshopConversationRunService,
    resolve_canonical_conversation_run,
)
from kai.workshop.domain import AgentId, ChannelId, MessageId, PrincipalId, WorkshopId
from kai.workshop.inbound import InboundMessage, record_inbound_message
from kai.workshop.store import WorkshopEventStore

_NOW = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)


async def _canonical_inbound(store: WorkshopEventStore, telegram_id: int = 101) -> MessageId:
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman(
                display_name="Authorized Human",
                role="admin",
                transport="telegram",
                external_subject=str(telegram_id),
                external_channel_id=str(telegram_id),
                runtime_profile_id=str(telegram_id),
            ),
        ),
    )
    result = await record_inbound_message(
        store,
        InboundMessage(
            transport="telegram",
            update_id="9001",
            message_id="42",
            sender_subject=str(telegram_id),
            channel_subject=str(telegram_id),
            body="Run this canonical message",
            occurred_at=_NOW,
        ),
    )
    aggregate_id = result.event.envelope.aggregate_id
    assert isinstance(aggregate_id, MessageId)
    return aggregate_id


async def _events() -> AsyncIterator[StreamEvent]:
    yield StreamEvent(
        text_so_far="Done.",
        done=True,
        response=AgentResponse(text="Done.", success=True),
    )


class TestCanonicalRunResolution:
    async def test_resolves_human_message_channel_and_attached_agent(self, tmp_path: Path):
        store = await WorkshopEventStore.open(tmp_path / "kai.db")
        try:
            inbound_id = await _canonical_inbound(store, telegram_id=101)

            resolution = await resolve_canonical_conversation_run(store, inbound_id)
            target = resolution.target

            assert target.inbound_message_id == inbound_id
            assert isinstance(target.workshop_id, WorkshopId)
            assert isinstance(target.channel_id, ChannelId)
            assert isinstance(target.requested_by_principal_id, PrincipalId)
            assert isinstance(target.agent_id, AgentId)
            assert resolution._legacy_pool_key == 101
        finally:
            await store.close()

    async def test_rejects_unknown_message(self, tmp_path: Path):
        store = await WorkshopEventStore.open(tmp_path / "kai.db")
        try:
            await _canonical_inbound(store)

            with pytest.raises(ConversationRunUnavailableError, match="one human channel member"):
                await resolve_canonical_conversation_run(store, MessageId.new())
        finally:
            await store.close()

    async def test_runtime_resolution_does_not_require_telegram_identity_or_binding(
        self,
        tmp_path: Path,
    ):
        store = await WorkshopEventStore.open(tmp_path / "kai.db")
        try:
            await bootstrap_default_workshop(
                store,
                (
                    BootstrapHuman(
                        display_name="Workshop-only Human",
                        role="admin",
                        transport="desktop",
                        external_subject="desktop-human",
                        external_channel_id="desktop-human",
                        runtime_profile_id="202",
                    ),
                ),
            )
            inbound = await record_inbound_message(
                store,
                InboundMessage(
                    transport="desktop",
                    update_id="desktop-command-1",
                    message_id="desktop-message-1",
                    sender_subject="desktop-human",
                    channel_subject="desktop-human",
                    body="Run without Telegram",
                    occurred_at=_NOW,
                ),
            )
            inbound_id = MessageId(str(inbound.event.envelope.aggregate_id))

            resolution = await resolve_canonical_conversation_run(store, inbound_id)

            assert resolution._legacy_pool_key == 202
            async with store.connection.execute(
                "SELECT COUNT(*) FROM external_identities WHERE provider = 'telegram'"
            ) as cursor:
                assert int((await cursor.fetchone())[0]) == 0
            async with store.connection.execute(
                "SELECT COUNT(*) FROM channel_bindings WHERE transport = 'telegram'"
            ) as cursor:
                assert int((await cursor.fetchone())[0]) == 0
        finally:
            await store.close()

    async def test_rejects_missing_runtime_assignment(self, tmp_path: Path):
        store = await WorkshopEventStore.open(tmp_path / "kai.db")
        try:
            inbound_id = await _canonical_inbound(store)
            await store.connection.execute("DELETE FROM channel_agent_runtime_assignments")
            await store.connection.commit()

            with pytest.raises(ConversationRunUnavailableError, match="explicit runtime profile assignment"):
                await resolve_canonical_conversation_run(store, inbound_id)
        finally:
            await store.close()

    async def test_rejects_multiple_attached_agents(self, tmp_path: Path):
        store = await WorkshopEventStore.open(tmp_path / "kai.db")
        try:
            inbound_id = await _canonical_inbound(store)
            target = (await resolve_canonical_conversation_run(store, inbound_id)).target
            second_principal = PrincipalId.new()
            second_agent = AgentId.new()
            await store.connection.execute(
                "INSERT INTO principals (id, kind, display_name, created_at) VALUES (?, 'agent', 'Other', ?)",
                (second_principal, _NOW.isoformat()),
            )
            await store.connection.execute(
                "INSERT INTO agents (id, workshop_id, principal_id, name, created_at) VALUES (?, ?, ?, 'Other', ?)",
                (second_agent, target.workshop_id, second_principal, _NOW.isoformat()),
            )
            await store.connection.execute(
                "INSERT INTO channel_agents (id, channel_id, agent_id, created_at) VALUES (?, ?, ?, ?)",
                ("cag_00000000000000000000000000000002", target.channel_id, second_agent, _NOW.isoformat()),
            )
            await store.connection.commit()

            with pytest.raises(ConversationRunUnavailableError, match="one attached agent"):
                await resolve_canonical_conversation_run(store, inbound_id)
        finally:
            await store.close()


class TestWorkshopConversationRunService:
    async def test_hides_compatibility_key_behind_canonical_request(self):
        inbound_id = MessageId.new()
        target = CompatibilityConversationRunResolution(
            target=CanonicalConversationRunTarget(
                inbound_message_id=inbound_id,
                workshop_id=WorkshopId.new(),
                channel_id=ChannelId.new(),
                requested_by_principal_id=PrincipalId.new(),
                agent_id=AgentId.new(),
            ),
            _legacy_pool_key=101,
        )
        resolver = AsyncMock(return_value=target)
        pool = MagicMock()
        pool.get_model.return_value = "gpt-5.6-sol"
        pool.send.return_value = _events()
        pool.get_effective_workspace = AsyncMock(return_value=Path("/srv/project"))
        service = WorkshopConversationRunService(pool, resolver)

        prepared = await service.prepare(inbound_id)
        events = [event async for event in prepared.stream("hello")]
        workspace = await prepared.effective_workspace()

        resolver.assert_awaited_once_with(inbound_id)
        pool.get_model.assert_called_once_with(101)
        pool.send.assert_called_once_with("hello", chat_id=101)
        pool.get_effective_workspace.assert_awaited_once_with(101)
        assert prepared.target.inbound_message_id == inbound_id
        assert prepared.model == "gpt-5.6-sol"
        assert events[0].response is not None and events[0].response.text == "Done."
        assert workspace == Path("/srv/project")

    async def test_rejects_resolver_substitution(self):
        requested_id = MessageId.new()
        target = CompatibilityConversationRunResolution(
            target=CanonicalConversationRunTarget(
                inbound_message_id=MessageId.new(),
                workshop_id=WorkshopId.new(),
                channel_id=ChannelId.new(),
                requested_by_principal_id=PrincipalId.new(),
                agent_id=AgentId.new(),
            ),
            _legacy_pool_key=101,
        )
        pool = MagicMock()
        service = WorkshopConversationRunService(pool, AsyncMock(return_value=target))

        with pytest.raises(ConversationRunUnavailableError, match="different inbound message"):
            await service.prepare(requested_id)

        pool.get_model.assert_not_called()
