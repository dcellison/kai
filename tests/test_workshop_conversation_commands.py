"""Contracts for production-unused atomic Workshop command acceptance."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from kai.workshop.bootstrap import BootstrapHuman, bootstrap_default_workshop
from kai.workshop.conversation_commands import (
    ConversationCommandDisposition,
    ConversationCommandStateConflictError,
    WorkshopConversationCommandService,
)
from kai.workshop.delivery_outbox import CONVERSATION_REPLY_PURPOSE, SEND_FRAGMENTS_CONTRACT
from kai.workshop.domain import AgentId, ChannelAgentId, ChannelId, PrincipalId, RunExecutionOwnerId
from kai.workshop.inbound import ClientInboundMessage, InboundMessage, record_inbound_message
from kai.workshop.projection import CanonicalConversationProjection
from kai.workshop.run_execution_authority import (
    RunExecutionSelection,
    WorkshopRunExecutionAuthority,
)
from kai.workshop.run_lifecycle import RunStatus, WorkshopRunLifecycle
from kai.workshop.store import IdempotencyConflictError, WorkshopEventStore

_NOW = datetime(2026, 8, 12, 20, 0, tzinfo=UTC)


def _message(*, body: str = "Perform one atomic unit of work") -> InboundMessage:
    return InboundMessage(
        transport="desktop",
        update_id="command-1",
        message_id="message-1",
        sender_subject="human-1",
        channel_subject="human-1",
        body=body,
        occurred_at=_NOW,
    )


async def _open_store(path: Path) -> WorkshopEventStore:
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
    return store


async def _open_client_store(
    path: Path,
) -> tuple[WorkshopEventStore, PrincipalId, ChannelId]:
    store = await WorkshopEventStore.open(path)
    await bootstrap_default_workshop(
        store,
        (
            BootstrapHuman(
                display_name="Workshop Human",
                role="admin",
                transport="telegram",
                external_subject="101",
                external_channel_id="101",
            ),
        ),
    )
    async with store.connection.execute(
        "SELECT e.principal_id, cb.channel_id FROM external_identities e "
        "JOIN channel_bindings cb ON cb.transport = e.provider "
        "AND cb.external_channel_id = e.external_subject "
        "WHERE e.provider = 'telegram' AND e.external_subject = '101'"
    ) as cursor:
        row = await cursor.fetchone()
    assert row is not None
    return store, PrincipalId(str(row[0])), ChannelId(str(row[1]))


def _client_message(
    principal_id: PrincipalId,
    channel_id: ChannelId,
    *,
    body: str = "Run this from Workshop",
    occurred_at: datetime = _NOW,
) -> ClientInboundMessage:
    return ClientInboundMessage(
        principal_id=principal_id,
        channel_id=channel_id,
        client_message_id="browser-command-1",
        body=body,
        occurred_at=occurred_at,
    )


def _authority(store: WorkshopEventStore) -> WorkshopRunExecutionAuthority:
    return WorkshopRunExecutionAuthority(
        store,
        selection_resolver=lambda _run: RunExecutionSelection(
            backend="codex",
            provider=None,
            model="gpt-5.6-sol",
        ),
        registered_backend_ids=frozenset({"codex"}),
    )


class TestAtomicConversationCommandAcceptance:
    async def test_accepts_message_and_run_in_one_transaction(self, tmp_path: Path):
        store = await _open_store(tmp_path / "kai.db")
        try:
            accepted = await WorkshopConversationCommandService(store).accept(_message())

            assert accepted.message.inserted is True
            assert accepted.lifecycle.changed is True
            assert accepted.disposition == ConversationCommandDisposition.NEWLY_ACCEPTED
            assert accepted.run.status == RunStatus.ACCEPTED
            assert accepted.run.inbound_message_id == accepted.message.event.envelope.aggregate_id
            assert accepted.message.event.position + 1 == accepted.lifecycle.event.position
            async with store.connection.execute(
                "SELECT event_type FROM event_log WHERE position IN (?, ?) ORDER BY position",
                (accepted.message.event.position, accepted.lifecycle.event.position),
            ) as cursor:
                assert [str(row[0]) for row in await cursor.fetchall()] == [
                    "message.created",
                    "run.accepted",
                ]
        finally:
            await store.close()

    async def test_exact_retry_returns_ready_without_new_facts(self, tmp_path: Path):
        store = await _open_store(tmp_path / "kai.db")
        try:
            service = WorkshopConversationCommandService(store)
            first = await service.accept(_message())
            retry = await service.accept(_message())

            assert retry.message.inserted is False
            assert retry.lifecycle.changed is False
            assert retry.disposition == ConversationCommandDisposition.READY_REPLAY
            assert retry.message.event == first.message.event
            assert retry.lifecycle.event == first.lifecycle.event
            async with store.connection.execute(
                "SELECT COUNT(*) FROM event_log WHERE event_type IN ('message.created', 'run.accepted')"
            ) as cursor:
                assert int((await cursor.fetchone())[0]) == 2
        finally:
            await store.close()

    async def test_concurrent_duplicate_commands_serialize_to_one_acceptance(self, tmp_path: Path):
        path = tmp_path / "kai.db"
        first_store = await _open_store(path)
        second_store = await WorkshopEventStore.open(path)
        try:
            first, second = await asyncio.gather(
                WorkshopConversationCommandService(first_store).accept(_message()),
                WorkshopConversationCommandService(second_store).accept(_message()),
            )

            assert {first.disposition, second.disposition} == {
                ConversationCommandDisposition.NEWLY_ACCEPTED,
                ConversationCommandDisposition.READY_REPLAY,
            }
            async with first_store.connection.execute(
                "SELECT COUNT(*) FROM event_log WHERE event_type IN ('message.created', 'run.accepted')"
            ) as cursor:
                assert int((await cursor.fetchone())[0]) == 2
        finally:
            await second_store.close()
            await first_store.close()

    async def test_changed_content_under_same_transport_identity_fails_closed(self, tmp_path: Path):
        store = await _open_store(tmp_path / "kai.db")
        try:
            service = WorkshopConversationCommandService(store)
            first = await service.accept(_message(body="Original command"))

            with pytest.raises(IdempotencyConflictError):
                await service.accept(_message(body="Changed command"))

            assert (await WorkshopRunLifecycle(store).state(first.run.run_id)).status == RunStatus.ACCEPTED
            async with store.connection.execute("SELECT body FROM messages") as cursor:
                assert str((await cursor.fetchone())[0]) == "Original command"
        finally:
            await store.close()

    async def test_ambiguous_agent_rolls_back_message_and_acceptance(self, tmp_path: Path):
        store = await _open_store(tmp_path / "kai.db")
        try:
            async with store.connection.execute("SELECT id FROM workshops") as cursor:
                workshop_id = str((await cursor.fetchone())[0])
            async with store.connection.execute("SELECT id FROM channels WHERE kind = 'direct'") as cursor:
                channel_id = str((await cursor.fetchone())[0])
            principal_id = PrincipalId.new()
            agent_id = AgentId.new()
            await store.connection.execute(
                "INSERT INTO principals (id, kind, display_name, created_at) VALUES (?, 'agent', 'Second agent', ?)",
                (principal_id, _NOW.isoformat()),
            )
            await store.connection.execute(
                "INSERT INTO agents (id, workshop_id, principal_id, name, created_at) "
                "VALUES (?, ?, ?, 'Second agent', ?)",
                (agent_id, workshop_id, principal_id, _NOW.isoformat()),
            )
            await store.connection.execute(
                "INSERT INTO channel_agents (id, channel_id, agent_id, created_at) VALUES (?, ?, ?, ?)",
                (ChannelAgentId.new(), channel_id, agent_id, _NOW.isoformat()),
            )
            await store.connection.commit()

            with pytest.raises(LookupError, match="one human channel member and one attached agent"):
                await WorkshopConversationCommandService(store).accept(_message())

            async with store.connection.execute(
                "SELECT COUNT(*) FROM event_log WHERE event_type IN ('message.created', 'run.accepted')"
            ) as cursor:
                assert int((await cursor.fetchone())[0]) == 0
            async with store.connection.execute("SELECT COUNT(*) FROM messages") as cursor:
                assert int((await cursor.fetchone())[0]) == 0
            async with store.connection.execute("SELECT COUNT(*) FROM runs") as cursor:
                assert int((await cursor.fetchone())[0]) == 0
        finally:
            await store.close()

    async def test_preexisting_message_without_run_is_rejected_and_not_repaired(self, tmp_path: Path):
        store = await _open_store(tmp_path / "kai.db")
        try:
            await record_inbound_message(store, _message())

            with pytest.raises(ConversationCommandStateConflictError, match="did not share one prior state"):
                await WorkshopConversationCommandService(store).accept(_message())

            async with store.connection.execute("SELECT COUNT(*) FROM runs") as cursor:
                assert int((await cursor.fetchone())[0]) == 0
            async with store.connection.execute(
                "SELECT COUNT(*) FROM event_log WHERE event_type = 'run.accepted'"
            ) as cursor:
                assert int((await cursor.fetchone())[0]) == 0
        finally:
            await store.close()


class TestConversationCommandReplayDisposition:
    async def test_active_and_cancellation_pending_replays_never_look_ready(self, tmp_path: Path):
        store = await _open_store(tmp_path / "kai.db")
        try:
            service = WorkshopConversationCommandService(store)
            accepted = await service.accept(_message())
            authority = _authority(store)
            grant = await authority.grant(
                accepted.run.run_id,
                owner_id=RunExecutionOwnerId.new(),
                occurred_at=_NOW + timedelta(seconds=1),
                lease_expires_at=_NOW + timedelta(seconds=61),
            )

            active = await service.accept(_message())
            assert active.disposition == ConversationCommandDisposition.ACTIVE_REPLAY

            await authority.request_cancellation(
                accepted.run.run_id,
                cancellation_code="requested_by_human",
                occurred_at=_NOW + timedelta(seconds=2),
            )
            cancelling = await service.accept(_message())
            assert cancelling.disposition == ConversationCommandDisposition.CANCELLATION_PENDING_REPLAY
            assert cancelling.run.cancellation_requested_at is not None
            assert grant.attempt.run_id == cancelling.run.run_id
        finally:
            await store.close()

    async def test_terminal_replay_suppresses_ready_disposition(self, tmp_path: Path):
        store = await _open_store(tmp_path / "kai.db")
        try:
            service = WorkshopConversationCommandService(store)
            accepted = await service.accept(_message())
            authority = _authority(store)
            grant = await authority.grant(
                accepted.run.run_id,
                owner_id=RunExecutionOwnerId.new(),
                occurred_at=_NOW + timedelta(seconds=1),
                lease_expires_at=_NOW + timedelta(seconds=61),
            )
            started = await authority.start(grant.claim, occurred_at=_NOW + timedelta(seconds=2))
            await authority.fail(
                started.claim,
                failure_code="backend_unavailable",
                occurred_at=_NOW + timedelta(seconds=3),
            )

            replay = await service.accept(_message())

            assert replay.disposition == ConversationCommandDisposition.TERMINAL_REPLAY
            assert replay.run.status == RunStatus.FAILED
        finally:
            await store.close()


class TestConversationCommandReplay:
    async def test_projection_rebuild_restores_atomic_command_exactly(self, tmp_path: Path):
        store = await _open_store(tmp_path / "kai.db")
        try:
            service = WorkshopConversationCommandService(store)
            before = await service.accept(_message())
            await store.connection.execute("DELETE FROM runs")
            await store.connection.execute("DELETE FROM messages")
            await store.connection.commit()

            checkpoint = await store.rebuild_projection(CanonicalConversationProjection())
            after = await WorkshopRunLifecycle(store).state(before.run.run_id)

            assert checkpoint.version == 6
            assert after == before.run
            async with store.connection.execute("SELECT body FROM messages") as cursor:
                assert str((await cursor.fetchone())[0]) == _message().body
        finally:
            await store.close()

    def test_service_is_registered_only_through_private_text_runtime_owner(self):
        source_root = Path(__file__).parents[1] / "src" / "kai"
        owner_source = (source_root / "workshop" / "private_text_execution.py").read_text(encoding="utf-8")
        assert "WorkshopConversationCommandService" in owner_source
        assert "WorkshopConversationCommandService" not in (source_root / "main.py").read_text(encoding="utf-8")


class TestAtomicClientConversationCommandAcceptance:
    async def test_accepts_message_run_and_telegram_echo_in_one_transaction(
        self,
        tmp_path: Path,
    ):
        store, principal_id, channel_id = await _open_client_store(tmp_path / "kai.db")
        try:
            accepted = await WorkshopConversationCommandService(store).accept_client(
                _client_message(principal_id, channel_id)
            )

            assert accepted.command.disposition == ConversationCommandDisposition.NEWLY_ACCEPTED
            assert accepted.command.message.inserted is True
            assert accepted.command.lifecycle.changed is True
            assert accepted.delivery.inserted is True
            assert accepted.compatibility_chat_id == 101
            assert accepted.delivery.delivery.message_id == accepted.command.message.event.envelope.aggregate_id
            assert accepted.delivery.delivery.channel_id == channel_id
            assert accepted.delivery.delivery.mode == "workshop_client_text"
            assert accepted.delivery.delivery.purpose == CONVERSATION_REPLY_PURPOSE
            assert accepted.delivery.delivery.execution_contract == SEND_FRAGMENTS_CONTRACT
            assert accepted.delivery.delivery.authority_epoch_id is None
            async with store.connection.execute(
                "SELECT event_type FROM event_log WHERE position BETWEEN ? AND ? ORDER BY position",
                (
                    accepted.command.message.event.position,
                    accepted.delivery.delivery.requested_event_position,
                ),
            ) as cursor:
                assert [str(row[0]) for row in await cursor.fetchall()] == [
                    "message.created",
                    "run.accepted",
                    "delivery.requested",
                ]
        finally:
            await store.close()

    async def test_exact_retry_reuses_all_three_facts_and_changed_body_conflicts(
        self,
        tmp_path: Path,
    ):
        store, principal_id, channel_id = await _open_client_store(tmp_path / "kai.db")
        service = WorkshopConversationCommandService(store)
        message = _client_message(principal_id, channel_id)
        try:
            first = await service.accept_client(message)
            retry = await service.accept_client(
                _client_message(
                    principal_id,
                    channel_id,
                    occurred_at=_NOW + timedelta(minutes=5),
                )
            )

            assert retry.command.disposition == ConversationCommandDisposition.READY_REPLAY
            assert retry.command.message.inserted is False
            assert retry.command.lifecycle.changed is False
            assert retry.delivery.inserted is False
            assert retry.command.message.event == first.command.message.event
            assert retry.command.lifecycle.event == first.command.lifecycle.event
            assert retry.delivery.delivery == first.delivery.delivery

            with pytest.raises(IdempotencyConflictError):
                await service.accept_client(_client_message(principal_id, channel_id, body="Changed command"))
            async with store.connection.execute(
                "SELECT COUNT(*) FROM event_log WHERE event_type IN "
                "('message.created', 'run.accepted', 'delivery.requested')"
            ) as cursor:
                assert int((await cursor.fetchone())[0]) == 3
        finally:
            await store.close()
